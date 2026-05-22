"""
Tests: STT + Language Detection
=================================
Run with:  pytest tests/test_stt_language.py -v

These tests cover:
  - Script-based language detection (no external calls)
  - Romanised keyword detection
  - STT hint passthrough
  - VoiceInputResult latency fields are populated
  - Fallback to English on empty/unknown input
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.language_detection.detector import (
    DetectedLanguage,
    Language,
    LanguageDetector,
    _script_detect,
    _romanised_detect,
)
from services.speech_to_text.stt_service import STTResult, STTProvider


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

class TestScriptDetect:

    def test_devanagari_detected_as_hindi(self):
        text = "मुझे कल डॉक्टर से मिलना है"
        result = _script_detect(text)
        assert result is not None
        assert result.language == Language.HINDI
        assert result.confidence > 0.7
        assert result.method == "script"

    def test_tamil_script_detected(self):
        text = "நாளை மருத்துவரை பார்க்க வேண்டும்"
        result = _script_detect(text)
        assert result is not None
        assert result.language == Language.TAMIL
        assert result.confidence > 0.7

    def test_latin_script_returns_none(self):
        text = "Book appointment with cardiologist tomorrow"
        result = _script_detect(text)
        assert result is None

    def test_mixed_script_detected(self):
        # Mostly Devanagari
        text = "Doctor से appointment चाहिए"
        result = _script_detect(text)
        assert result is not None
        assert result.language == Language.HINDI


# ---------------------------------------------------------------------------
# Romanised keyword detection
# ---------------------------------------------------------------------------

class TestRomanisedDetect:

    def test_hindi_romanised_keywords(self):
        text = "mujhe kal doctor se milna hai"
        result = _romanised_detect(text)
        assert result is not None
        assert result.language == Language.HINDI
        assert result.confidence >= 0.55

    def test_tamil_romanised_keywords(self):
        text = "naan naalai maruthuvar paarkka venum"
        result = _romanised_detect(text)
        assert result is not None
        assert result.language == Language.TAMIL

    def test_pure_english_returns_none(self):
        text = "Book an appointment with the cardiologist"
        result = _romanised_detect(text)
        assert result is None


# ---------------------------------------------------------------------------
# LanguageDetector (async)
# ---------------------------------------------------------------------------

class TestLanguageDetector:

    @pytest.fixture
    def detector(self):
        return LanguageDetector(confidence_threshold=0.65)

    @pytest.mark.asyncio
    async def test_stt_hint_takes_priority(self, detector):
        result = await detector.detect("hello doctor", stt_language_hint="hi")
        assert result.language == Language.HINDI
        assert result.method == "stt_hint"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_devanagari_via_script(self, detector):
        result = await detector.detect("मुझे डॉक्टर से मिलना है")
        assert result.language == Language.HINDI
        assert result.method == "script"

    @pytest.mark.asyncio
    async def test_tamil_via_script(self, detector):
        result = await detector.detect("நாளை மருத்துவரை பார்க்க வேண்டும்")
        assert result.language == Language.TAMIL

    @pytest.mark.asyncio
    async def test_english_default_on_empty(self, detector):
        result = await detector.detect("")
        assert result.language == Language.ENGLISH
        assert result.method == "default"

    @pytest.mark.asyncio
    async def test_english_stt_hint(self, detector):
        result = await detector.detect("book appointment tomorrow", stt_language_hint="en-IN")
        assert result.language == Language.ENGLISH

    @pytest.mark.asyncio
    async def test_duration_ms_populated(self, detector):
        result = await detector.detect("hello", stt_language_hint="en")
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# VoiceInputPipeline integration (mocked STT)
# ---------------------------------------------------------------------------

class TestVoiceInputPipeline:

    def _make_stt_result(self, text: str, lang_hint: str = "en") -> STTResult:
        return STTResult(
            text=text,
            is_final=True,
            confidence=0.92,
            language_hint=lang_hint,
            provider=STTProvider.DEEPGRAM,
            duration_ms=134.0,
        )

    @pytest.mark.asyncio
    async def test_pipeline_populates_latency_fields(self):
        from services.voice_input_pipeline import VoiceInputPipeline

        pipeline = VoiceInputPipeline(default_language="en")

        mock_stt_result = self._make_stt_result(
            "Book appointment with cardiologist tomorrow", "en"
        )

        with patch.object(
            pipeline._stt, "transcribe_bytes", return_value=mock_stt_result
        ):
            result = await pipeline.process_bytes(b"\x00" * 100)

        assert result.text == "Book appointment with cardiologist tomorrow"
        assert result.language == Language.ENGLISH
        assert result.stt_ms >= 0
        assert result.lang_ms >= 0
        assert result.total_ms >= 0

    @pytest.mark.asyncio
    async def test_pipeline_hindi_audio(self):
        from services.voice_input_pipeline import VoiceInputPipeline

        pipeline = VoiceInputPipeline(default_language="hi")
        mock_stt_result = self._make_stt_result("मुझे कल डॉक्टर से मिलना है", "hi")

        with patch.object(
            pipeline._stt, "transcribe_bytes", return_value=mock_stt_result
        ):
            result = await pipeline.process_bytes(b"\x00" * 100)

        assert result.language == Language.HINDI
        assert result.language_confidence >= 0.65

    @pytest.mark.asyncio
    async def test_pipeline_interim_callback(self):
        from services.voice_input_pipeline import VoiceInputPipeline

        pipeline = VoiceInputPipeline()
        received_interims = []

        async def fake_stream():
            for chunk in [b"\x00" * 100]:
                yield chunk

        mock_stt_result = self._make_stt_result("book appointment")

        with patch.object(
            pipeline._stt, "transcribe_stream", return_value=mock_stt_result
        ):
            result = await pipeline.process_stream(
                fake_stream(),
                on_interim_text=received_interims.append,
            )

        assert result.text == "book appointment"
