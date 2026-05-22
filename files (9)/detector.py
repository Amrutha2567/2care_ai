"""
Language Detection Service
===========================
Detects language from text with confidence scoring.

Strategy (in order of preference):
  1. Use the language_hint already returned by the STT provider
     (Deepgram/Whisper both detect language natively — zero extra latency)
  2. Script-based heuristics for Hindi (Devanagari) and Tamil script
     (deterministic, ~0 ms, very reliable for native script input)
  3. LLM-based detection for romanised / ambiguous text
     (reliable but adds ~50 ms — only triggered when needed)

Supported languages:
  en  English
  hi  Hindi
  ta  Tamil

All paths return a `DetectedLanguage` dataclass so the rest of the pipeline
is implementation-agnostic.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TAMIL = "ta"
    UNKNOWN = "unknown"


LANGUAGE_NAMES = {
    Language.ENGLISH: "English",
    Language.HINDI: "Hindi",
    Language.TAMIL: "Tamil",
    Language.UNKNOWN: "Unknown",
}


@dataclass
class DetectedLanguage:
    language: Language
    confidence: float            # 0.0 – 1.0
    method: str                  # "stt_hint" | "script" | "llm" | "default"
    duration_ms: float

    @property
    def name(self) -> str:
        return LANGUAGE_NAMES[self.language]

    def __str__(self) -> str:
        return (
            f"[lang] {self.name} "
            f"conf={self.confidence:.2f} "
            f"via={self.method} "
            f"latency={self.duration_ms:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Script-based heuristic detector
# ---------------------------------------------------------------------------

# Unicode ranges
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")   # Hindi / Sanskrit
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")         # Tamil

# Common Hindi words in romanised form
_HINDI_ROMANISED = {
    "mujhe", "aap", "kya", "hai", "hain", "nahi", "nahin",
    "doctor", "kal", "aaj", "milna", "chahiye", "appointment",
    "please", "theek", "shukriya", "dhanyavad",
}

# Common Tamil words in romanised form
_TAMIL_ROMANISED = {
    "naan", "avan", "aval", "illai", "enna", "eppo", "yaar",
    "paarkka", "venum", "maruthuvar", "naalai", "indha",
}


def _script_detect(text: str) -> Optional[DetectedLanguage]:
    """
    Return a DetectedLanguage if the text contains native script characters.
    Confidence is proportional to the fraction of script characters.
    """
    total = len(text.replace(" ", "")) or 1

    deva_count = len(_DEVANAGARI_RE.findall(text))
    if deva_count / total > 0.15:
        return DetectedLanguage(
            language=Language.HINDI,
            confidence=min(0.5 + deva_count / total, 0.99),
            method="script",
            duration_ms=0.0,
        )

    tamil_count = len(_TAMIL_RE.findall(text))
    if tamil_count / total > 0.15:
        return DetectedLanguage(
            language=Language.TAMIL,
            confidence=min(0.5 + tamil_count / total, 0.99),
            method="script",
            duration_ms=0.0,
        )

    return None


def _romanised_detect(text: str) -> Optional[DetectedLanguage]:
    """
    Light heuristic for romanised Hindi / Tamil based on keyword matching.
    Low confidence — only used when script detection fails.
    """
    tokens = set(text.lower().split())

    hindi_hits = len(tokens & _HINDI_ROMANISED)
    tamil_hits = len(tokens & _TAMIL_ROMANISED)

    if hindi_hits > tamil_hits and hindi_hits >= 1:
        return DetectedLanguage(
            language=Language.HINDI,
            confidence=0.55 + 0.05 * hindi_hits,
            method="romanised_keyword",
            duration_ms=0.0,
        )

    if tamil_hits > hindi_hits and tamil_hits >= 1:
        return DetectedLanguage(
            language=Language.TAMIL,
            confidence=0.55 + 0.05 * tamil_hits,
            method="romanised_keyword",
            duration_ms=0.0,
        )

    return None


# ---------------------------------------------------------------------------
# LLM-based detector (fallback for ambiguous / short text)
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a language detection assistant.
Given a text snippet, identify whether it is in English, Hindi, or Tamil.
Respond ONLY with a JSON object — no preamble, no markdown.
Format: {"language": "<en|hi|ta>", "confidence": <0.0-1.0>}
If the text is mixed or unclear, use the dominant language."""


async def _llm_detect(text: str) -> DetectedLanguage:
    """
    Use the LLM to detect language for ambiguous romanised text.
    Only called when heuristics are not confident enough.
    """
    import json
    import os
    import httpx

    start_ts = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={
                    "model": "gpt-4o-mini",     # fast + cheap for this micro-task
                    "max_tokens": 30,
                    "messages": [
                        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Text: {text[:300]}"},
                    ],
                },
            )
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        lang_code = parsed.get("language", "en")
        confidence = float(parsed.get("confidence", 0.7))
        language = Language(lang_code) if lang_code in Language._value2member_map_ else Language.ENGLISH
    except Exception as exc:
        logger.warning("LLM language detection failed (%s); defaulting to English.", exc)
        language = Language.ENGLISH
        confidence = 0.5

    elapsed = (time.perf_counter() - start_ts) * 1000
    return DetectedLanguage(
        language=language,
        confidence=confidence,
        method="llm",
        duration_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

_BCP47_TO_LANGUAGE = {
    "en": Language.ENGLISH,
    "en-IN": Language.ENGLISH,
    "en-US": Language.ENGLISH,
    "hi": Language.HINDI,
    "hi-IN": Language.HINDI,
    "ta": Language.TAMIL,
    "ta-IN": Language.TAMIL,
}


class LanguageDetector:
    """
    Unified language detection with a tiered strategy.

    Usage:
        detector = LanguageDetector()

        # After STT, pass the hint from the provider:
        lang = await detector.detect(text, stt_language_hint="hi")

        # Without a hint (pure text):
        lang = await detector.detect(text)
    """

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold

    async def detect(
        self,
        text: str,
        stt_language_hint: Optional[str] = None,
    ) -> DetectedLanguage:
        """
        Detect language using the tiered strategy.
        Returns a DetectedLanguage with the best available confidence.
        """
        start_ts = time.perf_counter()

        # --- Tier 1: STT provider hint (fastest, no added latency) ----------
        if stt_language_hint:
            language = _BCP47_TO_LANGUAGE.get(stt_language_hint)
            if language:
                result = DetectedLanguage(
                    language=language,
                    confidence=0.95,
                    method="stt_hint",
                    duration_ms=(time.perf_counter() - start_ts) * 1000,
                )
                logger.info("LangDetect %s", result)
                return result

        if not text or not text.strip():
            return DetectedLanguage(
                language=Language.ENGLISH,
                confidence=0.5,
                method="default",
                duration_ms=0.0,
            )

        # --- Tier 2: Script heuristics (0 ms) --------------------------------
        result = _script_detect(text)
        if result and result.confidence >= self.confidence_threshold:
            result.duration_ms = (time.perf_counter() - start_ts) * 1000
            logger.info("LangDetect %s", result)
            return result

        # --- Tier 3: Romanised keyword heuristics (~0 ms) --------------------
        result = _romanised_detect(text)
        if result and result.confidence >= self.confidence_threshold:
            result.duration_ms = (time.perf_counter() - start_ts) * 1000
            logger.info("LangDetect %s", result)
            return result

        # --- Tier 4: LLM fallback (~50 ms) -----------------------------------
        result = await _llm_detect(text)
        logger.info("LangDetect %s", result)
        return result

    def detect_sync(self, text: str, stt_language_hint: Optional[str] = None) -> DetectedLanguage:
        """Synchronous wrapper for use outside async contexts."""
        return asyncio.get_event_loop().run_until_complete(
            self.detect(text, stt_language_hint)
        )
