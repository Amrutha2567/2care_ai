"""
lang_detector.py — Two-stage language detection for Indian multilingual voice.

Stage 1: Deepgram metadata (primary — already computed during STT, free)
Stage 2: fasttext lid.176 (fallback — in-process, < 3ms)

Handles: English, Hindi (hi), Tamil (ta)
Also handles: Hinglish and Tanglish (code-switching) by dominant language detection.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()

SUPPORTED_LANGUAGES = {"en", "hi", "ta"}
DEFAULT_LANGUAGE = "en"

# fasttext language code remapping
# Deepgram and fasttext may use different codes
LANG_REMAP = {
    "en-IN": "en",
    "en-US": "en",
    "en-GB": "en",
    "hi-IN": "hi",
    "ta-IN": "ta",
    "tam": "ta",
    "hin": "hi",
    "eng": "en",
}

# Hindi Unicode range (Devanagari)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Tamil Unicode range
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")


class LanguageDetector:
    """
    Lightweight language detector for voice transcripts.
    
    Priorities:
    1. Unicode script detection (most reliable for Hindi/Tamil)
    2. Deepgram-reported language (if confidence >= 0.85)
    3. fasttext classification (fallback)
    4. Session preference (if all else fails)
    """

    def __init__(self, fasttext_model_path: Optional[str] = None):
        self._ft_model = None
        self._ft_model_path = fasttext_model_path
        self._load_fasttext()

    def _load_fasttext(self):
        """Load fasttext model lazily."""
        if not self._ft_model_path:
            log.warning("lang_detector.no_fasttext_model", 
                       msg="Using Unicode heuristics only. Download lid.176.bin for better accuracy.")
            return
        try:
            import fasttext
            self._ft_model = fasttext.load_model(self._ft_model_path)
            log.info("lang_detector.fasttext_loaded")
        except Exception as e:
            log.error("lang_detector.fasttext_load_failed", error=str(e))

    def detect(
        self,
        text: str,
        deepgram_lang: Optional[str] = None,
        deepgram_confidence: float = 0.0,
        session_lang: Optional[str] = None,
    ) -> tuple[str, float]:
        """
        Detect language of transcript text.
        
        Returns:
            (language_code, confidence) where language_code in {"en", "hi", "ta"}
        """
        start = time.perf_counter()
        text = text.strip()

        if not text:
            return (session_lang or DEFAULT_LANGUAGE, 0.5)

        # Stage 1: Unicode script detection (most reliable for Hindi/Tamil)
        unicode_lang = self._detect_by_unicode(text)
        if unicode_lang:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.debug("lang_detect.unicode", lang=unicode_lang, elapsed_ms=f"{elapsed_ms:.1f}")
            return (unicode_lang, 0.99)

        # Stage 2: Deepgram metadata
        if deepgram_lang and deepgram_confidence >= 0.85:
            normalised = self._normalise_lang_code(deepgram_lang)
            if normalised in SUPPORTED_LANGUAGES:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log.debug("lang_detect.deepgram", lang=normalised, 
                         confidence=deepgram_confidence, elapsed_ms=f"{elapsed_ms:.1f}")
                return (normalised, deepgram_confidence)

        # Stage 3: fasttext
        if self._ft_model:
            ft_lang, ft_conf = self._detect_by_fasttext(text)
            if ft_conf >= 0.7 and ft_lang in SUPPORTED_LANGUAGES:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log.debug("lang_detect.fasttext", lang=ft_lang, 
                         confidence=ft_conf, elapsed_ms=f"{elapsed_ms:.1f}")
                return (ft_lang, ft_conf)

        # Stage 4: Session preference or default
        lang = session_lang or DEFAULT_LANGUAGE
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.debug("lang_detect.fallback", lang=lang, elapsed_ms=f"{elapsed_ms:.1f}")
        return (lang, 0.5)

    def _detect_by_unicode(self, text: str) -> Optional[str]:
        """
        Fast Unicode-based detection.
        If > 10% of chars are Devanagari → Hindi.
        If > 10% of chars are Tamil script → Tamil.
        """
        if len(text) == 0:
            return None
        
        devanagari_count = len(DEVANAGARI_RE.findall(text))
        tamil_count = len(TAMIL_RE.findall(text))
        total = len(text)

        if devanagari_count / total > 0.1:
            return "hi"
        if tamil_count / total > 0.1:
            return "ta"
        return None

    def _detect_by_fasttext(self, text: str) -> tuple[str, float]:
        """Run fasttext classification."""
        try:
            # fasttext expects single line
            clean = text.replace("\n", " ").strip()
            predictions = self._ft_model.predict(clean, k=3)
            labels, scores = predictions
            
            for label, score in zip(labels, scores):
                lang_code = label.replace("__label__", "")
                normalised = self._normalise_lang_code(lang_code)
                if normalised in SUPPORTED_LANGUAGES:
                    return (normalised, float(score))
            
            return (DEFAULT_LANGUAGE, 0.5)
        except Exception as e:
            log.warning("lang_detect.fasttext_error", error=str(e))
            return (DEFAULT_LANGUAGE, 0.5)

    @staticmethod
    def _normalise_lang_code(code: str) -> str:
        """Normalise various language code formats to our standard codes."""
        code = code.lower().strip()
        return LANG_REMAP.get(code, code[:2])  # Take first 2 chars as fallback


# ── Standalone helper ──────────────────────────────────────────────────────

_detector_singleton: Optional[LanguageDetector] = None

def get_detector(fasttext_model_path: Optional[str] = None) -> LanguageDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = LanguageDetector(fasttext_model_path)
    return _detector_singleton


def detect_language(
    text: str,
    deepgram_lang: Optional[str] = None,
    deepgram_confidence: float = 0.0,
    session_lang: Optional[str] = None,
    fasttext_model_path: Optional[str] = None,
) -> tuple[str, float]:
    """Convenience function for one-off detection."""
    detector = get_detector(fasttext_model_path)
    return detector.detect(
        text=text,
        deepgram_lang=deepgram_lang,
        deepgram_confidence=deepgram_confidence,
        session_lang=session_lang,
    )
