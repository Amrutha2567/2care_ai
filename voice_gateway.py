"""
voice_gateway.py — FastAPI WebSocket handler for real-time voice pipeline.

This is the main orchestrator. For each call:
1. Establishes Deepgram STT stream
2. Detects language from transcripts
3. Loads patient context from memory
4. Runs Claude agent on each turn
5. Streams TTS audio back to telephony

Latency instrumentation at every stage.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Optional

import anthropic
import structlog
from deepgram import (
    DeepgramClient, DeepgramClientOptions,
    LiveTranscriptionEvents, LiveOptions
)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Response, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from agent.agent_core import VoiceAgent, LatencyTrace, build_system_prompt
from audio.lang_detector import detect_language
from audio.tts_streamer import TTSStreamer, SentenceBufferedTTSPipeline, BargeinDetector
from memory.memory_manager import MemoryManager, SessionState
from scheduling.scheduling_service import SchedulingService
from .dependencies import get_db, get_redis, get_memory_manager, get_scheduling_service, get_agent

log = structlog.get_logger()
router = APIRouter(prefix="/voice", tags=["voice"])


# ── Twilio Inbound Webhook ─────────────────────────────────────────────────

@router.post("/inbound")
async def handle_inbound_call(request: Request):
    """
    Twilio calls this when a patient calls in.
    Responds with TwiML to connect to our WebSocket media stream.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", str(uuid.uuid4()))
    caller_number = form_data.get("From", "unknown")

    log.info("inbound_call.received", call_sid=call_sid, caller=caller_number)

    public_url = os.getenv("PUBLIC_BASE_URL", "https://your-domain.ngrok.io")
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}/api/voice/media-stream">
            <Parameter name="callSid" value="{call_sid}"/>
            <Parameter name="callerNumber" value="{caller_number}"/>
        </Stream>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


# ── Twilio Media Stream WebSocket ──────────────────────────────────────────

@router.websocket("/media-stream")
async def media_stream(
    websocket: WebSocket,
    memory: MemoryManager = Depends(get_memory_manager),
    scheduling: SchedulingService = Depends(get_scheduling_service),
    agent: VoiceAgent = Depends(get_agent),
):
    """
    Twilio Media Streams WebSocket handler.
    Receives mu-law audio, sends back synthesised speech.
    
    Pipeline per turn:
    STT → language detect → memory read → Claude agent → TTS → audio out
    """
    await websocket.accept()
    
    call_sid = None
    caller_number = "unknown"
    session: Optional[SessionState] = None
    deepgram_connection = None
    tts_streamer: Optional[TTSStreamer] = None
    barge_in_detector = BargeinDetector()
    
    # Track if we're currently speaking (to handle barge-in)
    is_agent_speaking = asyncio.Event()
    barge_in_triggered = asyncio.Event()
    
    # Queue for transcript→agent processing
    transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
    
    # Latency tracking
    call_start_time = time.perf_counter() * 1000
    current_trace: Optional[LatencyTrace] = None
    utterance_start_time: Optional[float] = None

    async def on_barge_in():
        """Called when patient speaks while agent is talking."""
        if is_agent_speaking.is_set():
            log.info("barge_in.triggered", call_sid=call_sid)
            barge_in_triggered.set()
            # Signal TTS pipeline to stop (implementation: close current synthesis)

    barge_in_detector.set_barge_in_callback(lambda: asyncio.ensure_future(on_barge_in()))

    async def handle_transcript(transcript_data: dict):
        """
        Called when Deepgram returns a final transcript.
        Runs the full agent pipeline and streams audio back.
        """
        nonlocal current_trace, utterance_start_time

        transcript = transcript_data.get("text", "").strip()
        if not transcript:
            return

        is_agent_speaking.set()
        barge_in_triggered.clear()

        # Build latency trace
        trace = LatencyTrace(
            call_sid=call_sid,
            turn=session.turn_count if session else 0,
        )
        trace.record("utterance_end")
        current_trace = trace

        stt_end = time.perf_counter() * 1000

        # Language detection
        lang, lang_conf = detect_language(
            text=transcript,
            deepgram_lang=transcript_data.get("detected_language"),
            deepgram_confidence=transcript_data.get("language_confidence", 0.0),
            session_lang=session.language if session else None,
        )
        trace.record("lang_detect")
        lang_detect_ms = (time.perf_counter() * 1000) - stt_end

        if session and lang != session.language and lang_conf > 0.85:
            log.info("language.switched", from_lang=session.language, to_lang=lang)
            session.language = lang
            if tts_streamer:
                tts_streamer.set_language(lang)

        log.info(
            "turn.processing",
            call_sid=call_sid,
            transcript=transcript[:80],
            language=lang,
            lang_confidence=f"{lang_conf:.2f}",
        )

        # Load patient context (uses Redis cache)
        t_redis = time.perf_counter()
        patient_context = None
        if session and session.patient_id:
            async with get_db() as db:
                patient_context = await memory.get_patient_context(session.patient_id, db)
        trace.record("redis_read")
        redis_read_ms = (time.perf_counter() - t_redis) * 1000

        # TTS pipeline
        tts_pipeline = SentenceBufferedTTSPipeline(tts_streamer)

        def on_token(token: str):
            """Called for each LLM token — feeds TTS pipeline."""
            if barge_in_triggered.is_set():
                return  # Drop tokens after barge-in
            tts_pipeline.feed_token(token)

        # Run agent (streaming)
        agent_task = asyncio.create_task(
            agent.process_turn(
                transcript=transcript,
                session=session,
                patient_context=patient_context,
                on_token=on_token,
                trace=trace,
            )
        )

        # Flush TTS pipeline when agent is done
        async def flush_when_done():
            await agent_task
            tts_pipeline.flush()

        flush_task = asyncio.create_task(flush_when_done())

        # Stream audio back to Twilio
        async for audio_chunk in tts_pipeline.run(trace=trace):
            if barge_in_triggered.is_set():
                log.info("barge_in.audio_truncated", call_sid=call_sid)
                break

            # Encode as base64 for Twilio media stream
            import base64
            audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": audio_b64},
            })

        await agent_task  # Ensure agent finished
        is_agent_speaking.clear()

        # Compute and log latency
        total_ms = (time.perf_counter() * 1000) - stt_end
        log.info(
            "turn.latency",
            call_sid=call_sid,
            stt_to_first_audio_ms=f"{total_ms:.0f}",
            lang_detect_ms=f"{lang_detect_ms:.0f}",
            redis_read_ms=f"{redis_read_ms:.0f}",
            target_met=total_ms < 450,
        )

        # Append trace to session
        if session:
            session.latency_traces.append({
                "turn": session.turn_count,
                "total_ms": total_ms,
                "lang_detect_ms": lang_detect_ms,
                "redis_read_ms": redis_read_ms,
                "timestamps": trace.timestamps,
            })
            await memory.update_session(session)

    # ── Deepgram STT Setup ─────────────────────────────────────────────────

    async def setup_deepgram(dg_client: DeepgramClient):
        live_options = LiveOptions(
            model="nova-2",
            language="multi",        # Multilingual detection
            smart_format=True,
            interim_results=True,
            utterance_end_ms=1000,   # 1s silence = end of utterance
            vad_events=True,
            endpointing=300,          # ms
            encoding="mulaw",
            sample_rate=8000,         # Twilio sends 8kHz mu-law
            channels=1,
        )

        dg_conn = dg_client.listen.asyncwebsocket.v("1")

        async def on_message(self, result, **kwargs):
            """Handle Deepgram transcript events."""
            try:
                sentence = result.channel.alternatives[0].transcript
                if not sentence or not result.is_final:
                    return

                detected_lang = getattr(result.channel.alternatives[0], "languages", [None])[0]
                lang_conf = getattr(result.channel.alternatives[0], "confidence", 0.0)

                await transcript_queue.put({
                    "text": sentence,
                    "detected_language": detected_lang,
                    "language_confidence": lang_conf,
                    "is_final": result.is_final,
                    "speech_final": result.speech_final,
                })
            except Exception as e:
                log.error("deepgram.message_error", error=str(e))

        async def on_error(self, error, **kwargs):
            log.error("deepgram.error", error=str(error))

        dg_conn.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_conn.on(LiveTranscriptionEvents.Error, on_error)

        await dg_conn.start(live_options)
        return dg_conn

    # ── Main WebSocket Loop ────────────────────────────────────────────────

    stream_sid = None

    try:
        dg_api_key = os.getenv("DEEPGRAM_API_KEY")
        dg_client = DeepgramClient(dg_api_key)
        deepgram_connection = await setup_deepgram(dg_client)

        # Consumer task — processes transcripts in order
        async def transcript_consumer():
            while True:
                data = await transcript_queue.get()
                if data is None:
                    break
                await handle_transcript(data)

        consumer_task = asyncio.create_task(transcript_consumer())

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                log.warning("websocket.timeout", call_sid=call_sid)
                break

            data = json.loads(message)
            event = data.get("event")

            if event == "connected":
                log.info("twilio.connected")

            elif event == "start":
                stream_sid = data["streamSid"]
                call_sid = data["start"]["callSid"]
                caller_number = data["start"]["customParameters"].get("callerNumber", "unknown")

                log.info("call.started", call_sid=call_sid, caller=caller_number)

                # Initialise TTS
                tts_streamer = TTSStreamer(
                    api_key=os.getenv("ELEVENLABS_API_KEY"),
                    language="en",  # Will be updated after first transcript
                )

                # Create or load session
                async with get_db() as db:
                    patient_ctx = await memory.get_patient_by_phone(caller_number, db)

                patient_id = patient_ctx.patient_id if patient_ctx else None
                preferred_lang = patient_ctx.preferred_language if patient_ctx else "en"

                if tts_streamer:
                    tts_streamer.set_language(preferred_lang)

                session = await memory.create_session(
                    call_sid=call_sid,
                    patient_id=patient_id,
                    patient_name=patient_ctx.name if patient_ctx else None,
                    language=preferred_lang,
                )

                log.info(
                    "session.initialised",
                    call_sid=call_sid,
                    patient_id=patient_id,
                    language=preferred_lang,
                )

                # Send greeting
                greeting_by_lang = {
                    "en": "Hello! Welcome to the clinic. I'm here to help you with your appointments. How can I help you today?",
                    "hi": "नमस्ते! क्लिनिक में आपका स्वागत है। मैं आपके अपॉइंटमेंट में मदद करने के लिए यहाँ हूँ। आज आपकी कैसे मदद कर सकता हूँ?",
                    "ta": "வணக்கம்! கிளினிக்கிற்கு வரவேற்கிறோம். உங்கள் அப்பாயிண்ட்மென்ட்களில் உதவ நான் இங்கே இருக்கிறேன். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
                }
                greeting = greeting_by_lang.get(preferred_lang, greeting_by_lang["en"])

                # Synthesise and send greeting
                import base64
                async for chunk in tts_streamer.synthesise_streaming(greeting):
                    audio_b64 = base64.b64encode(chunk).decode("utf-8")
                    await websocket.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": audio_b64},
                    })

            elif event == "media":
                # Audio from patient — send to Deepgram + barge-in detector
                payload = data["media"]["payload"]
                import base64
                audio_bytes = base64.b64decode(payload)

                # Barge-in detection (when agent is speaking)
                if is_agent_speaking.is_set():
                    barge_in_detector.process_audio_frame(audio_bytes)

                # Forward to Deepgram
                await deepgram_connection.send(audio_bytes)

            elif event == "stop":
                log.info("call.ended", call_sid=call_sid)
                break

    except WebSocketDisconnect:
        log.info("websocket.disconnected", call_sid=call_sid)
    except Exception as e:
        log.error("voice_gateway.error", error=str(e), call_sid=call_sid)
    finally:
        # Cleanup
        transcript_queue.put_nowait(None)
        
        if deepgram_connection:
            try:
                await deepgram_connection.finish()
            except Exception:
                pass

        # Finalise session and write to PostgreSQL
        if session:
            try:
                summary = await agent.generate_session_summary(session)
                outcome = session.entities_extracted.get("last_action", "no_action")
                last_appt_id = session.entities_extracted.get("last_appointment_id")

                async with get_db() as db:
                    await memory.finalise_session(
                        state=session,
                        db=db,
                        summary_text=summary,
                        outcome=outcome,
                        appointment_id=last_appt_id,
                    )
                log.info("call.session_finalised", call_sid=call_sid)
            except Exception as e:
                log.error("session_finalise.error", error=str(e))


# ── Direct WebSocket (for testing without Twilio) ─────────────────────────

@router.websocket("/direct-test")
async def direct_test_ws(
    websocket: WebSocket,
    memory: MemoryManager = Depends(get_memory_manager),
    agent: VoiceAgent = Depends(get_agent),
):
    """
    Simplified WebSocket for local testing — accepts JSON text messages,
    returns agent text responses without audio processing.
    
    Message format: {"text": "Book me an appointment", "phone": "+919876543210"}
    """
    await websocket.accept()
    call_sid = f"test_{uuid.uuid4().hex[:8]}"
    session = None

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") == "init":
                phone = data.get("phone", "+910000000000")
                async with get_db() as db:
                    patient_ctx = await memory.get_patient_by_phone(phone, db)

                session = await memory.create_session(
                    call_sid=call_sid,
                    patient_id=patient_ctx.patient_id if patient_ctx else None,
                    patient_name=patient_ctx.name if patient_ctx else None,
                    language=patient_ctx.preferred_language if patient_ctx else "en",
                )
                await websocket.send_json({
                    "type": "session_started",
                    "call_sid": call_sid,
                    "patient": patient_ctx.name if patient_ctx else "Unknown",
                    "language": session.language,
                })
                continue

            if not session:
                await websocket.send_json({"error": "Send {type: init, phone: ...} first"})
                continue

            transcript = data.get("text", "").strip()
            if not transcript:
                continue

            lang, _ = detect_language(text=transcript, session_lang=session.language)
            session.language = lang

            tokens_collected = []

            def collect_token(token: str):
                tokens_collected.append(token)

            async with get_db() as db:
                patient_ctx = await memory.get_patient_context(session.patient_id, db) if session.patient_id else None

            full_response = await agent.process_turn(
                transcript=transcript,
                session=session,
                patient_context=patient_ctx,
                on_token=collect_token,
            )
            await memory.update_session(session)

            await websocket.send_json({
                "type": "response",
                "text": full_response,
                "language": session.language,
                "turn": session.turn_count,
            })

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            try:
                async with get_db() as db:
                    summary = await agent.generate_session_summary(session)
                    await memory.finalise_session(
                        state=session, db=db,
                        summary_text=summary,
                        outcome=session.entities_extracted.get("last_action", "no_action"),
                    )
            except Exception as e:
                log.error("test_session_finalise.error", error=str(e))
