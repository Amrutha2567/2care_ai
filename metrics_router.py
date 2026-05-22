"""
tts_streamer.py — ElevenLabs TTS with sentence-boundary streaming.

Strategy:
- Buffer LLM tokens until a sentence boundary (. ? ! । ?)
- Send complete sentences to ElevenLabs streaming endpoint
- Stream audio bytes back to the telephony layer immediately
- This gives natural speech cadence while minimising latency

Why sentence-boundary buffering?
- Word-by-word: choppy, unnatural prosody
- Full response: 300-500ms additional latency for complete LLM output
- Sentence-by-sentence: ~15ms buffer overhead, natural speech, starts fast
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator, Callable, Optional

import httpx
import structlog

log = structlog.get_logger()

# Voice IDs per language (ElevenLabs)
VOICE_IDS = {
    "en": "21m00Tcm4TlvDq8ikWAM",   # Rachel — professional, warm
    "hi": "EXAVITQu4vr4xnSDxMaL",   # Bella — clear, Hindi-accent friendly
    "ta": "pNInz6obpgDQGcFmaJgB",   # Adam — Tamil cadence
}

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Sentence boundary detection — covers English, Hindi (।), Tamil (?)
SENTENCE_END_RE = re.compile(r"[.!?।?]+\s*$")

# Minimum chars before we consider flushing on boundary
MIN_FLUSH_LENGTH = 20


class TTSStreamer:
    """
    Streams LLM text to ElevenLabs TTS, flushing at sentence boundaries.
    
    Usage:
        streamer = TTSStreamer(api_key=..., language="hi")
        
        async for audio_chunk in streamer.stream("नमस्ते, क्या मैं आपकी मदद कर सकता हूँ?"):
            # Send chunk to telephony layer
            await send_audio(audio_chunk)
    """

    def __init__(
        self,
        api_key: str,
        language: str = "en",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "pcm_16000",  # 16kHz PCM for Twilio
        voice_stability: float = 0.5,
        voice_similarity_boost: float = 0.75,
    ):
        self.api_key = api_key
        self.language = language
        self.model_id = model_id
        self.output_format = output_format
        self.voice_id = VOICE_IDS.get(language, VOICE_IDS["en"])
        self.settings = {
            "stability": voice_stability,
            "similarity_boost": voice_similarity_boost,
            "style": 0.0,
            "use_speaker_boost": True,
        }

    def set_language(self, language: str):
        """Switch language (and voice) mid-session."""
        if language in VOICE_IDS:
            self.language = language
            self.voice_id = VOICE_IDS[language]

    async def synthesise_streaming(
        self, text: str, on_first_chunk: Optional[Callable] = None
    ) -> AsyncIterator[bytes]:
        """
        Stream audio for a complete text string.
        Yields PCM audio chunks as they arrive from ElevenLabs.
        """
        url = (
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{self.voice_id}/stream"
            f"?output_format={self.output_format}&optimize_streaming_latency=3"
        )
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": self.settings,
        }

        first_chunk = True
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        log.error(
                            "tts.api_error",
                            status=resp.status_code,
                            body=error_body[:200],
                        )
                        return

                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if chunk:
                            if first_chunk and on_first_chunk:
                                on_first_chunk()
                                first_chunk = False
                            yield chunk

            except httpx.TimeoutException:
                log.error("tts.timeout", text_preview=text[:50])
            except Exception as e:
                log.error("tts.unexpected_error", error=str(e))


class SentenceBufferedTTSPipeline:
    """
    Sits between the LLM token stream and TTS.
    
    Receives tokens via feed_token(), buffers to sentence boundary,
    then fires TTS synthesis for each complete sentence.
    
    Audio chunks are yielded via the async generator interface.
    """

    def __init__(self, tts: TTSStreamer):
        self.tts = tts
        self._buffer = ""
        self._audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._synthesis_task: Optional[asyncio.Task] = None
        self._pending_sentences: asyncio.Queue[str] = asyncio.Queue()
        self._done = False
        self._first_chunk_time: Optional[float] = None

    def feed_token(self, token: str):
        """
        Called by the LLM stream on each text token.
        Thread-safe via asyncio queue.
        """
        self._buffer += token

        # Check for sentence boundary
        if len(self._buffer) >= MIN_FLUSH_LENGTH and SENTENCE_END_RE.search(self._buffer):
            sentence = self._buffer.strip()
            self._buffer = ""
            if sentence:
                self._pending_sentences.put_nowait(sentence)

    def flush(self):
        """Flush any remaining buffer at end of LLM response."""
        remaining = self._buffer.strip()
        if remaining:
            self._pending_sentences.put_nowait(remaining)
            self._buffer = ""
        self._pending_sentences.put_nowait(None)  # Sentinel

    async def run(self, trace=None) -> AsyncIterator[bytes]:
        """
        Main pipeline loop. Yields audio chunks.
        Starts consuming sentences as soon as they arrive.
        """
        while True:
            sentence = await self._pending_sentences.get()
            if sentence is None:
                break  # Done

            log.debug("tts.synthesising_sentence", preview=sentence[:60])
            first_chunk_for_sentence = True

            async for chunk in self.tts.synthesise_streaming(
                text=sentence,
                on_first_chunk=lambda: self._record_first_chunk(trace),
            ):
                yield chunk

    def _record_first_chunk(self, trace):
        if trace and not self._first_chunk_time:
            self._first_chunk_time = time.perf_counter() * 1000
            trace.record("tts_first_chunk")
            log.debug("tts.first_chunk_received")


class BargeinDetector:
    """
    Voice Activity Detection for barge-in (interrupt) handling.
    
    When a patient starts speaking while the agent is talking,
    we detect it via energy threshold on incoming audio and signal
    the TTS pipeline to stop.
    
    This is a simplified energy-based VAD. For production,
    use Silero VAD or WebRTC VAD for better accuracy.
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        min_speech_frames: int = 3,  # consecutive frames above threshold
        frame_size: int = 160,       # 10ms at 16kHz
    ):
        self.threshold = energy_threshold
        self.min_frames = min_speech_frames
        self.frame_size = frame_size
        self._consecutive_frames = 0
        self._is_speaking = False
        self._on_barge_in: Optional[Callable] = None

    def set_barge_in_callback(self, callback: Callable):
        self._on_barge_in = callback

    def process_audio_frame(self, pcm_bytes: bytes) -> bool:
        """
        Process a 10ms audio frame.
        Returns True if barge-in detected.
        """
        import struct
        import math

        if len(pcm_bytes) < 2:
            return False

        # Calculate RMS energy
        num_samples = len(pcm_bytes) // 2
        samples = struct.unpack(f"{num_samples}h", pcm_bytes[:num_samples * 2])
        rms = math.sqrt(sum(s * s for s in samples) / num_samples) / 32768.0

        if rms > self.threshold:
            self._consecutive_frames += 1
        else:
            self._consecutive_frames = 0

        if self._consecutive_frames >= self.min_frames and not self._is_speaking:
            self._is_speaking = True
            log.info("barge_in.detected", rms=f"{rms:.4f}")
            if self._on_barge_in:
                self._on_barge_in()
            return True

        return False

    def reset(self):
        self._consecutive_frames = 0
        self._is_speaking = False
