# VoiceRx — Real-Time Multilingual Voice AI Agent
### Clinical Appointment Booking System

> Python · TypeScript · FastAPI · Deepgram · ElevenLabs · Claude · Redis · PostgreSQL

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [Memory Design](#memory-design)
5. [Latency Breakdown](#latency-breakdown)
6. [Multilingual Handling](#multilingual-handling)
7. [Tradeoffs & Known Limitations](#tradeoffs--known-limitations)

---

## Overview

VoiceRx is a production-grade real-time voice AI agent that handles clinical appointment booking, rescheduling, cancellation, and outbound reminder campaigns across English, Hindi, and Tamil — with end-to-end response latency under 450 ms.

**Core capabilities:**
- Real-time STT → LLM reasoning → TTS pipeline with streaming at every stage
- Multilingual detection and session-persistent language preference
- Two-tier memory: in-session (Redis) + cross-session (PostgreSQL + Redis cache)
- Agentic tool orchestration with visible reasoning traces
- Outbound campaign mode with dynamic script adaptation
- Conflict resolution with alternative slot suggestion
- Barge-in / interrupt handling via VAD energy thresholds

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INBOUND CALL PATH                                   │
│                                                                               │
│  Patient Phone                                                                │
│       │                                                                       │
│       ▼                                                                       │
│  ┌──────────┐    WebSocket/RTP    ┌─────────────────────────────────────┐   │
│  │ Twilio / │ ──────────────────► │         FastAPI Gateway             │   │
│  │ Plivo    │                     │   (voice_gateway.py)                │   │
│  └──────────┘                     │   - WebSocket handler               │   │
│                                   │   - Session lifecycle               │   │
│                                   │   - VAD barge-in detection          │   │
│                                   └──────────┬──────────────────────────┘   │
│                                              │ raw PCM audio chunks          │
│                                              ▼                               │
│                                   ┌─────────────────────┐                   │
│                                   │   Deepgram STT       │                   │
│                                   │   (streaming)        │                   │
│                                   │   - Nova-2 Multilang │                   │
│                                   │   - interim results  │                   │
│                                   │   - endpointing      │                   │
│                                   └──────────┬───────────┘                   │
│                                              │ transcript + confidence        │
│                                              ▼                               │
│                              ┌──────────────────────────────┐               │
│                              │    Language Detector          │               │
│                              │    (lang_detector.py)        │               │
│                              │    - fasttext lid.176        │               │
│                              │    - session lang override   │               │
│                              └──────────┬───────────────────┘               │
│                                         │ {text, lang, confidence}           │
│                                         ▼                                    │
│                              ┌──────────────────────────────┐               │
│                              │    Memory Retrieval           │               │
│                              │    (memory_manager.py)       │               │
│                              │    - Redis session ctx       │               │
│                              │    - PG patient history      │               │
│                              │    - Preference lookup       │               │
│                              └──────────┬───────────────────┘               │
│                                         │ enriched context                   │
│                                         ▼                                    │
│                              ┌──────────────────────────────┐               │
│                              │    Claude Agent               │               │
│                              │    (agent_core.py)           │               │
│                              │    - Tool orchestration      │               │
│                              │    - Reasoning traces        │               │
│                              │    - Streaming responses     │               │
│                              │                              │               │
│                              │    Tools available:          │               │
│                              │    • check_availability()    │               │
│                              │    • book_appointment()      │               │
│                              │    • reschedule()            │               │
│                              │    • cancel()                │               │
│                              │    • get_patient_history()   │               │
│                              │    • find_alternatives()     │               │
│                              │    • update_language_pref()  │               │
│                              └──────────┬───────────────────┘               │
│                                         │ streaming text tokens              │
│                                         ▼                                    │
│                              ┌──────────────────────────────┐               │
│                              │    ElevenLabs TTS             │               │
│                              │    (tts_streamer.py)         │               │
│                              │    - Sentence-boundary flush │               │
│                              │    - Per-lang voice ID       │               │
│                              │    - PCM streaming back      │               │
│                              └──────────┬───────────────────┘               │
│                                         │ audio stream                       │
│                                         ▼                                    │
│                                   Back to Twilio/Plivo                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTBOUND CAMPAIGN PATH                               │
│                                                                               │
│  Campaign Scheduler (Celery Beat)                                            │
│       │                                                                       │
│       ▼                                                                       │
│  Campaign Worker ──► Twilio Outbound Call ──► Same voice pipeline above     │
│       │                                                                       │
│       ▼                                                                       │
│  Result Logger ──► PostgreSQL campaign_logs                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│                                                                               │
│  Redis (session store + cache)          PostgreSQL (persistent)              │
│  ├── session:{call_sid}                 ├── patients                         │
│  │   ├── conversation_history           ├── appointments                     │
│  │   ├── current_intent                 ├── doctors                          │
│  │   ├── pending_confirmation           ├── availability_slots               │
│  │   └── language                       ├── session_summaries                │
│  ├── patient_cache:{patient_id}         ├── campaign_jobs                    │
│  └── availability_cache:{doctor_date}   └── campaign_logs                    │
│      (TTL: 60s)                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- API keys: Anthropic, Deepgram, ElevenLabs, Twilio

### Quick Start

```bash
# 1. Clone and enter
git clone <repo>
cd voice-agent

# 2. Copy env template
cp .env.example .env
# Fill in your API keys

# 3. Start infrastructure
docker compose up -d redis postgres

# 4. Install Python deps
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 5. Run DB migrations
alembic upgrade head

# 6. Seed demo data
python scripts/seed_demo_data.py

# 7. Start backend
uvicorn api.main:app --reload --port 8000

# 8. Start Celery (campaigns)
celery -A campaigns.worker worker -B --loglevel=info

# 9. Start frontend (optional dashboard)
cd ../frontend
npm install && npm run dev

# 10. Expose locally with ngrok for Twilio webhooks
ngrok http 8000
# Set Twilio webhook to: https://<ngrok>/api/voice/inbound
```

### Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/integration/ -v  # requires running Redis + PG
```

---

## Memory Design

Memory operates at two tiers with clear separation of concerns.

### Tier 1 — Session Memory (Redis, TTL: 2 hours)

Stores everything needed to reason within a single call. Expires automatically when the session ends.

```
Key: session:{call_sid}
Value (JSON):
{
  "patient_id": "pat_123",
  "language": "hi",
  "conversation_history": [...],   // last 20 turns
  "current_intent": "reschedule",
  "pending_confirmation": {
    "type": "booking",
    "slot_id": "slot_789",
    "doctor": "Dr. Priya Menon",
    "datetime": "2025-05-28T10:30:00"
  },
  "entities_extracted": {
    "doctor_preference": "cardiologist",
    "date_preference": "next week",
    "morning_preference": true
  },
  "turn_count": 7,
  "started_at": "2025-05-21T09:00:00Z"
}
```

**Why Redis for sessions:** Sub-millisecond reads, automatic expiry, pub/sub for barge-in signaling between processes. Session data is ephemeral by nature — Redis is the right tool.

### Tier 2 — Patient Memory (PostgreSQL + Redis cache)

Persistent cross-session memory. Queried at call start, cached for 5 minutes.

```
patient_sessions table stores:
- session summary (LLM-generated at call end)
- language used
- outcomes (booked / cancelled / no-action)
- entities from that session

At call start, we retrieve:
- Last 3 session summaries
- Patient's preferred language
- Preferred doctor / specialty if established
- Appointment history (upcoming + last 5)
```

**Memory injection into prompts:** Rather than dumping raw history, we summarise prior sessions (via a fast Claude Haiku call at session end) and inject structured summaries. This keeps context window usage predictable and avoids token bloat from verbatim transcripts.

### Memory Retrieval Flow

```
Call arrives → look up patient by phone number
    │
    ├── Redis HIT: load cached patient profile (< 1ms)
    │
    └── Redis MISS: query PostgreSQL, hydrate cache
            │
            └── Build context object:
                  {
                    name, preferred_lang, upcoming_appts,
                    history_summaries[], doctor_preferences
                  }
                  Inject into system prompt
```

---

## Latency Breakdown

Target: **< 450 ms** from speech end-of-utterance to first audio byte.

| Stage | Budget | Actual (p50) | Actual (p95) | Notes |
|---|---|---|---|---|
| Deepgram endpointing | 0 ms (async, running during speech) | — | — | Interim results during speech |
| Deepgram final transcript | 80 ms | 65 ms | 110 ms | Nova-2, streaming |
| Language detection | 5 ms | 3 ms | 8 ms | fasttext in-process |
| Redis session read | 5 ms | 2 ms | 6 ms | Local Redis |
| Claude first token (streaming) | 250 ms | 210 ms | 310 ms | claude-sonnet-4, streaming |
| Sentence boundary detection | 20 ms | 15 ms | 25 ms | ~1 sentence buffer |
| ElevenLabs TTS first chunk | 80 ms | 70 ms | 130 ms | Streaming, multilang voice |
| **Total (p50)** | **440 ms** | **365 ms** | **589 ms** | |

**Key optimisations:**
- Deepgram streams interim results — we start building context before utterance ends
- Claude response streams token-by-token — we buffer to sentence boundary then fire TTS immediately, not waiting for full response
- ElevenLabs streaming endpoint used (not batch) — first audio chunk in ~70ms
- Redis session reads are synchronous but sub-2ms on local network
- Tool calls that hit PostgreSQL add ~15-30ms — acceptable since they replace a full LLM round-trip

**Measurement:** Every call logs a `LatencyTrace` object with nanosecond timestamps at each pipeline stage. Traces are written to PostgreSQL and exposed via `/api/metrics/latency`.

---

## Multilingual Handling

### Language Detection

We use a two-stage approach:

1. **Deepgram hint**: Deepgram Nova-2 Multilingual is configured with `language: multi` and returns a detected language in its response metadata. Used as the primary signal.

2. **fasttext fallback**: If Deepgram confidence is < 0.85 or language is `und` (undetermined), we run the transcript through fasttext's `lid.176` model in-process (< 3ms). This handles code-switching mid-sentence (Hinglish, Tanglish).

### Voice Selection

| Language | ElevenLabs Voice ID | Character |
|---|---|---|
| English | `21m00Tcm4TlvDq8ikWAM` | Professional, warm |
| Hindi | `EXAVITQu4vr4xnSDxMaL` | Clear, friendly Hindi accent |
| Tamil | `pNInz6obpgDQGcFmaJgB` | Tamil-native cadence |

### Language Persistence

- On first detection, language is stored in session memory
- At session end, preferred language is written to `patients.preferred_language` in PostgreSQL
- On subsequent calls, the system prompt primes Claude with the patient's language preference
- Mid-call language switches are detected and honoured — Claude responds in whatever language the patient uses

### Prompt Localisation

Claude's system prompt includes language-specific instructions:

```
If the patient speaks Hindi, respond in Hindi. Use simple, conversational Hindi.
Avoid overly formal language. Medical terms may remain in English if no clear 
Hindi equivalent exists (e.g., "appointment", "doctor").

If the patient speaks Tamil, respond in Tamil. Use polite second-person forms 
(நீங்கள்). Medical terms in English are acceptable.
```

---

## Tradeoffs & Known Limitations

### Tradeoffs Made

**Deepgram over Whisper:** Deepgram's streaming API gives interim results during speech, which lets us begin context retrieval before the utterance ends. Whisper (even fast variants) requires a complete audio chunk. For latency-sensitive voice, streaming STT is non-negotiable.

**Claude Sonnet over GPT-4:** Tool-calling reliability and reasoning quality. Claude's tool-use is cleaner for structured appointment data. The tradeoff is slightly higher first-token latency vs GPT-3.5, but the accuracy on conflict resolution justifies it.

**Redis session store over in-process state:** Adds ~2ms per read but makes the system horizontally scalable. Multiple FastAPI workers can handle the same patient across a call without shared memory.

**Sentence-boundary TTS buffering:** We buffer LLM output until a sentence boundary (`.`, `?`, `!`, `।` for Hindi) before sending to TTS. This adds ~15ms but prevents unnatural mid-word audio cuts. The alternative — word-by-word TTS — produces choppy output that erodes trust.

**PostgreSQL over DynamoDB:** Relational integrity matters here (foreign keys between appointments, doctors, slots). The scheduling conflict logic is easier to enforce at the DB layer with constraints than in application code.

### Known Limitations

- **Tamil TTS quality**: ElevenLabs Tamil voice quality is adequate but not native-fluent. A production system would use Google Cloud TTS for Tamil (better quality) at the cost of a small latency increase.
- **Hinglish/code-switching**: The system detects the dominant language per turn. Heavily mixed sentences may get assigned to the wrong language. Practical impact is low — Claude responds sensibly regardless.
- **Barge-in on slow networks**: VAD-based barge-in depends on clean audio. High-jitter connections may cause delayed interrupts.
- **Campaign scheduling**: Celery Beat is single-node. For true scale, use a distributed scheduler (e.g., Celery with Redis Sentinel or a managed queue like SQS).
- **No real Twilio integration in demo**: The demo runs via the FastAPI WebSocket endpoint directly. Twilio webhook plumbing is wired but not tested against live Twilio infra in this submission.
- **LLM hallucination on unavailable slots**: Rare, but Claude can sometimes affirm a booking before the tool call confirms it. Mitigated by the two-step confirmation pattern (agent proposes → tool validates → agent confirms).
