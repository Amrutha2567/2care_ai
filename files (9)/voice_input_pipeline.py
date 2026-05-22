"""
Voice Input Pipeline
====================
Combines STT → Language Detection into a single call.

This is the entry point for the WebSocket handler. It:
  1. Accepts raw audio bytes or an async audio stream
  2. Runs STT (Deepgram streaming or Whisper fallback)
  3. Runs language detection on the result
  4. Emits a VoiceInputResult with the full latency breakdown
  5. Logs structured metrics at every stage

Latency budget allocation (out of 450 ms total):
  STT              → 100–150 ms  (Deepgram streaming)
  Language detect  → 0–5 ms     (script heuristic or STT hint)
  ─────────────────────────────
  This module      → ~150 ms
  Remaining budget → ~300 ms    (LLM + TTS)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Optional

from .stt_service import STTResult, STTService
from ..language_detection.detector import DetectedLanguage, Language, LanguageDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class VoiceInputResult:
    """Everything downstream needs from one voice turn."""
    text: str
    language: Language
    language_confidence: float
    stt_confidence: float

    # Latency breakdown (milliseconds)
    stt_ms: float
    lang_ms: float
    total_ms: float

    # Raw sub-results for debugging
    stt_result: STTResult = field(repr=False)
    lang_result: DetectedLanguage = field(repr=False)

    def log_metrics(self) -> None:
        logger.info(
            "voice_input | text=%r | lang=%s(%.2f) | "
            "stt_ms=%.0f | lang_ms=%.1f | total_ms=%.0f",
            self.text,
            self.language.value,
            self.language_confidence,
            self.stt_ms,
            self.lang_ms,
            self.total_ms,
        )

    @property
    def within_budget(self) -> bool:
        """True if STT+LangDetect stayed within their share of the 450 ms budget."""
        return self.total_ms < 160


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VoiceInputPipeline:
    """
    Orchestrates STT + language detection for one voice turn.

    Args:
        default_language: BCP-47 code used to initialise the STT provider.
                          Should come from the patient's persistent profile.
    """

    def __init__(self, default_language: str = "en"):
        self._stt = STTService(preferred_language=default_language)
        self._lang = LanguageDetector(confidence_threshold=0.65)

    async def process_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        on_interim_text: Optional[Callable[[str], None]] = None,
    ) -> VoiceInputResult:
        """
        Primary path: streaming audio from WebSocket.

        `on_interim_text` is called with partial transcripts as they arrive,
        allowing downstream components to begin warm-up before speech ends.
        """
        pipeline_start = time.perf_counter()

        # --- STT ---------------------------------------------------------
        def _on_interim(stt_result: STTResult) -> None:
            if on_interim_text and stt_result.text:
                on_interim_text(stt_result.text)

        stt_result = await self._stt.transcribe_stream(audio_chunks, _on_interim)
        stt_end = time.perf_counter()

        # --- Language detection ------------------------------------------
        lang_result = await self._lang.detect(
            stt_result.text,
            stt_language_hint=stt_result.language_hint,
        )
        lang_end = time.perf_counter()

        # --- Build result ------------------------------------------------
        result = VoiceInputResult(
            text=stt_result.text,
            language=lang_result.language,
            language_confidence=lang_result.confidence,
            stt_confidence=stt_result.confidence,
            stt_ms=(stt_end - pipeline_start) * 1000,
            lang_ms=(lang_end - stt_end) * 1000,
            total_ms=(lang_end - pipeline_start) * 1000,
            stt_result=stt_result,
            lang_result=lang_result,
        )

        result.log_metrics()

        if not result.within_budget:
            logger.warning(
                "voice_input exceeded latency budget: %.0f ms (limit 160 ms)",
                result.total_ms,
            )

        return result

    async def process_bytes(self, audio_bytes: bytes) -> VoiceInputResult:
        """
        Fallback path: complete audio buffer (e.g. from file upload or test).
        Uses Whisper directly — no streaming.
        """
        pipeline_start = time.perf_counter()

        stt_result = await self._stt.transcribe_bytes(audio_bytes)
        stt_end = time.perf_counter()

        lang_result = await self._lang.detect(
            stt_result.text,
            stt_language_hint=stt_result.language_hint,
        )
        lang_end = time.perf_counter()

        result = VoiceInputResult(
            text=stt_result.text,
            language=lang_result.language,
            language_confidence=lang_result.confidence,
            stt_confidence=stt_result.confidence,
            stt_ms=(stt_end - pipeline_start) * 1000,
            lang_ms=(lang_end - stt_end) * 1000,
            total_ms=(lang_end - pipeline_start) * 1000,
            stt_result=stt_result,
            lang_result=lang_result,
        )

        result.log_metrics()
        return result
