"""
main.py — FastAPI application entry point.

Sets up:
- Database connection pool
- Redis connection
- Dependency injection for memory, scheduling, agent
- All routers (voice, campaigns, metrics, health)
- Structured logging
- Prometheus metrics
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import anthropic
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from agent.agent_core import VoiceAgent
from memory.memory_manager import MemoryManager
from scheduling.scheduling_service import SchedulingService
from .voice_gateway import router as voice_router
from .campaigns_router import router as campaigns_router
from .metrics_router import router as metrics_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("APP_ENV") == "development"
        else structlog.processors.JSONRenderer(),
    ],
)

log = structlog.get_logger()

# ── Global singletons ──────────────────────────────────────────────────────

_engine = None
_session_factory = None
_redis_client = None
_memory_manager = None
_scheduling_service = None
_anthropic_client = None
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all resources at startup, clean up at shutdown."""
    global _engine, _session_factory, _redis_client
    global _memory_manager, _scheduling_service, _anthropic_client, _agent

    log.info("voicerx.starting")

    # Database
    _engine = create_async_engine(
        os.getenv("DATABASE_URL", "postgresql+asyncpg://voicerx:voicerx@localhost:5432/voicerx"),
        pool_size=int(os.getenv("DATABASE_POOL_SIZE", 20)),
        max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", 40)),
        echo=os.getenv("APP_ENV") == "development",
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)

    # Redis
    _redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        encoding="utf-8",
        decode_responses=True,
    )
    await _redis_client.ping()

    # Services
    _memory_manager = MemoryManager(
        redis_client=_redis_client,
        session_ttl=int(os.getenv("REDIS_SESSION_TTL", 7200)),
        patient_cache_ttl=int(os.getenv("REDIS_PATIENT_CACHE_TTL", 300)),
        availability_cache_ttl=int(os.getenv("REDIS_AVAILABILITY_CACHE_TTL", 60)),
    )

    _scheduling_service = SchedulingService(memory=_memory_manager)

    _anthropic_client = anthropic.AsyncAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    _agent = VoiceAgent(
        anthropic_client=_anthropic_client,
        scheduling_service=_scheduling_service,
        memory_manager=_memory_manager,
        db_session_factory=_session_factory,
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    )

    log.info("voicerx.ready")
    yield

    # Shutdown
    log.info("voicerx.shutting_down")
    await _redis_client.aclose()
    await _engine.dispose()
    log.info("voicerx.stopped")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VoiceRx",
    description="Real-Time Multilingual Voice AI Agent for Clinical Appointment Booking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routers
app.include_router(voice_router, prefix="/api")
app.include_router(campaigns_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VoiceRx"}


@app.get("/api/debug/session/{call_sid}")
async def debug_session(call_sid: str):
    """Dev-only: inspect a live session."""
    if os.getenv("APP_ENV") != "development":
        return {"error": "Not available in production"}
    state = await _memory_manager.get_session(call_sid)
    if not state:
        return {"error": "Session not found"}
    return {
        "call_sid": state.call_sid,
        "language": state.language,
        "turn_count": state.turn_count,
        "current_intent": state.current_intent,
        "pending_confirmation": state.pending_confirmation,
        "history_turns": len(state.conversation_history),
        "latency_traces": state.latency_traces[-3:],  # Last 3 turns
    }
