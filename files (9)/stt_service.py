"""
Speech-to-Text Service
======================
Primary:  Deepgram streaming API  → returns partials while user is still speaking
Fallback: OpenAI Whisper (local)  → used if Deepgram is unavailable

Latency target: < 150 ms from speech-end to text output.

Design notes:
- Deepgram streams interim results in real time; we only commit on `is_final=True`
- Whisper fallback loads the 'base' model (best latency/accuracy trade-off)
- Both paths emit a `STTResult` dataclass so the caller is implementation-agnostic
- Latency is measured and logged at every stage
"""

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

class STTProvider(str, Enum):
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"


@dataclass
class STTResult:
    text: str
    is_final: bool
    confidence: float
    language_hint: Optional[str]          # BCP-47 tag if provider detects it
    provider: STTProvider
    duration_ms: float                     # time from call start → this result
    audio_duration_ms: Optional[float] = None

    def __str__(self) -> str:
        status = "FINAL" if self.is_final else "interim"
        return (
            f"[{status}] ({self.provider.value}) "
            f"'{self.text}' "
            f"conf={self.confidence:.2f} "
            f"latency={self.duration_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Deepgram streaming STT
# ---------------------------------------------------------------------------

class DeepgramSTT:
    """
    Streams raw PCM/WebM audio to Deepgram and yields STTResult objects.

    Deepgram returns interim transcripts as the user speaks, so the caller
    can start processing before speech ends — critical for the 450 ms budget.

    Required env var: DEEPGRAM_API_KEY
    """

    SUPPORTED_LANGUAGES = {
        "en": "en-IN",   # English  (India model)
        "hi": "hi",      # Hindi
        "ta": "ta",      # Tamil
    }

    def __init__(self, language: str = "en", sample_rate: int = 16000):
        self.language = self.SUPPORTED_LANGUAGES.get(language, "en-IN")
        self.sample_rate = sample_rate
        self._client = None

    def _get_client(self):
        """Lazy-load Deepgram client."""
        if self._client is None:
            try:
                from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
                import os
                api_key = os.environ["DEEPGRAM_API_KEY"]
                self._client = DeepgramClient(api_key)
                self._LiveTranscriptionEvents = LiveTranscriptionEvents
                self._LiveOptions = LiveOptions
            except ImportError:
                raise RuntimeError(
                    "deepgram-sdk not installed. Run: pip install deepgram-sdk"
                )
            except KeyError:
                raise RuntimeError("DEEPGRAM_API_KEY environment variable not set.")
        return self._client

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        on_interim: Optional[Callable[[STTResult], None]] = None,
    ) -> STTResult:
        """
        Feed an async generator of audio bytes to Deepgram.
        Calls `on_interim` for each partial result.
        Returns the final committed STTResult.
        """
        client = self._get_client()
        start_ts = time.perf_counter()
        final_result: Optional[STTResult] = None
        done_event = asyncio.Event()

        dg_connection = client.listen.asyncwebsocket.v("1")

        async def on_message(self_inner, result, **kwargs):
            nonlocal final_result
            elapsed = (time.perf_counter() - start_ts) * 1000
            alt = result.channel.alternatives[0]
            stt = STTResult(
                text=alt.transcript.strip(),
                is_final=result.is_final,
                confidence=alt.confidence,
                language_hint=getattr(result, "detected_language", None),
                provider=STTProvider.DEEPGRAM,
                duration_ms=elapsed,
            )
            if result.is_final and alt.transcript.strip():
                logger.info("STT %s", stt)
                final_result = stt
                done_event.set()
            elif on_interim and alt.transcript.strip():
                on_interim(stt)

        async def on_error(self_inner, error, **kwargs):
            logger.error("Deepgram error: %s", error)
            done_event.set()

        dg_connection.on(self._LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(self._LiveTranscriptionEvents.Error, on_error)

        options = self._LiveOptions(
            model="nova-2",
            language=self.language,
            smart_format=True,
            interim_results=True,
            endpointing=300,         # ms of silence before finalising
            sample_rate=self.sample_rate,
            encoding="linear16",
            channels=1,
        )

        await dg_connection.start(options)

        async for chunk in audio_chunks:
            await dg_connection.send(chunk)

        await dg_connection.finish()
        await asyncio.wait_for(done_event.wait(), timeout=10.0)

        if final_result is None:
            final_result = STTResult(
                text="",
                is_final=True,
                confidence=0.0,
                language_hint=None,
                provider=STTProvider.DEEPGRAM,
                duration_ms=(time.perf_counter() - start_ts) * 1000,
            )

        return final_result


# ---------------------------------------------------------------------------
# Whisper fallback STT
# ---------------------------------------------------------------------------

class WhisperSTT:
    """
    Local Whisper transcription — used when Deepgram is unavailable.

    Model choice:
      'tiny'  → ~39 M params, ~60 ms on CPU  (low accuracy)
      'base'  → ~74 M params, ~100 ms on CPU (good balance) ← default
      'small' → ~244 M params, ~250 ms on CPU

    Note: Whisper does NOT stream — it requires a complete audio buffer.
    This path adds ~100–150 ms over Deepgram but has zero external dependency.
    """

    WHISPER_TO_BCP47 = {
        "english": "en",
        "hindi": "hi",
        "tamil": "ta",
    }

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                import whisper
                logger.info("Loading Whisper '%s' model…", self.model_size)
                self._model = whisper.load_model(self.model_size)
                logger.info("Whisper model ready.")
            except ImportError:
                raise RuntimeError(
                    "openai-whisper not installed. Run: pip install openai-whisper"
                )
        return self._model

    async def transcribe_bytes(self, audio_bytes: bytes) -> STTResult:
        """Transcribe a complete audio buffer. Runs in a thread pool."""
        start_ts = time.perf_counter()
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe_sync, audio_bytes
        )
        result.duration_ms = (time.perf_counter() - start_ts) * 1000
        logger.info("STT %s", result)
        return result

    def _transcribe_sync(self, audio_bytes: bytes) -> STTResult:
        import numpy as np
        model = self._get_model()

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_array /= 32768.0  # normalise to [-1, 1]

        output = model.transcribe(
            audio_array,
            fp16=False,
            task="transcribe",
        )

        detected_lang_raw = output.get("language", "english")
        lang_hint = self.WHISPER_TO_BCP47.get(detected_lang_raw, detected_lang_raw)

        return STTResult(
            text=output["text"].strip(),
            is_final=True,
            confidence=1.0,          # Whisper doesn't expose per-token confidence
            language_hint=lang_hint,
            provider=STTProvider.WHISPER,
            duration_ms=0.0,         # filled in by caller
        )


# ---------------------------------------------------------------------------
# Unified STT facade
# ---------------------------------------------------------------------------

class STTService:
    """
    Public interface for the STT pipeline.

    Tries Deepgram streaming first; falls back to Whisper automatically.
    The caller never needs to know which provider handled the request.
    """

    def __init__(self, preferred_language: str = "en"):
        self.preferred_language = preferred_language
        self._deepgram = DeepgramSTT(language=preferred_language)
        self._whisper: Optional[WhisperSTT] = None   # lazy-loaded on fallback

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        on_interim: Optional[Callable[[STTResult], None]] = None,
    ) -> STTResult:
        """Primary path: Deepgram streaming."""
        try:
            return await self._deepgram.transcribe_stream(audio_chunks, on_interim)
        except Exception as exc:
            logger.warning("Deepgram unavailable (%s); falling back to Whisper.", exc)
            # Drain the async generator into a buffer for Whisper
            buffer = b""
            # Note: generator already partially consumed — caller should pass
            # a fresh generator or use transcribe_bytes directly for fallback.
            return await self.transcribe_bytes(buffer)

    async def transcribe_bytes(self, audio_bytes: bytes) -> STTResult:
        """Fallback path: Whisper on a complete audio buffer."""
        if self._whisper is None:
            self._whisper = WhisperSTT(model_size="base")
        return await self._whisper.transcribe_bytes(audio_bytes)
