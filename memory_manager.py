"""
memory_manager.py — Two-tier memory system for VoiceRx.

Tier 1: Redis (in-session, TTL-based)
Tier 2: PostgreSQL + Redis cache (cross-session, patient history)

Design principle: the LLM sees structured summaries, not raw transcripts.
This keeps context window usage bounded and prompts predictable.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as aioredis
import structlog
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from models import Patient, SessionSummary, Appointment, AppointmentStatus, Language

log = structlog.get_logger()

# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    role: str          # "user" | "assistant" | "tool_result"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None


@dataclass
class PendingConfirmation:
    confirmation_type: str    # "booking" | "cancellation" | "reschedule"
    slot_id: Optional[str] = None
    appointment_id: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_id: Optional[str] = None
    datetime_str: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class SessionState:
    """Complete in-session state stored in Redis."""
    call_sid: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    language: str = "en"
    conversation_history: list[dict] = field(default_factory=list)
    current_intent: Optional[str] = None
    pending_confirmation: Optional[dict] = None
    entities_extracted: dict = field(default_factory=dict)
    turn_count: int = 0
    is_outbound: bool = False
    campaign_job_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    latency_traces: list[dict] = field(default_factory=list)

    def add_turn(self, role: str, content: str, **kwargs):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            **kwargs
        })
        # Keep last 20 turns in session to bound context
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        if role == "user":
            self.turn_count += 1

    def to_claude_messages(self) -> list[dict]:
        """Convert history to Claude API message format."""
        messages = []
        for turn in self.conversation_history:
            role = turn["role"]
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": turn["content"]})
        return messages


@dataclass
class PatientContext:
    """Cross-session patient context loaded at call start."""
    patient_id: str
    name: str
    phone_number: str
    preferred_language: str
    upcoming_appointments: list[dict]
    recent_session_summaries: list[str]   # LLM-generated summaries
    preferred_doctor_name: Optional[str]
    preferred_specialty: Optional[str]
    notes: Optional[str]

    def to_prompt_fragment(self) -> str:
        """Render as a structured string for injection into the system prompt."""
        lines = [
            f"Patient: {self.name}",
            f"Preferred language: {self.preferred_language}",
        ]
        if self.preferred_doctor_name:
            lines.append(f"Usual doctor: {self.preferred_doctor_name}")
        if self.preferred_specialty:
            lines.append(f"Preferred specialty: {self.preferred_specialty}")
        if self.upcoming_appointments:
            lines.append("\nUpcoming appointments:")
            for appt in self.upcoming_appointments:
                lines.append(
                    f"  - {appt['datetime']} with {appt['doctor_name']} "
                    f"(Ref: {appt['confirmation_code']}) [{appt['status']}]"
                )
        if self.recent_session_summaries:
            lines.append("\nPast interaction summaries:")
            for i, summary in enumerate(self.recent_session_summaries, 1):
                lines.append(f"  {i}. {summary}")
        if self.notes:
            lines.append(f"\nClinical notes: {self.notes}")
        return "\n".join(lines)


# ── Memory Manager ─────────────────────────────────────────────────────────

class MemoryManager:
    """
    Central memory interface. All reads/writes go through this class.

    Session memory: Redis (no DB round-trip on every turn).
    Patient context: PostgreSQL with Redis cache (5-minute TTL).
    """

    SESSION_KEY = "session:{call_sid}"
    PATIENT_CACHE_KEY = "patient_cache:{patient_id}"
    AVAILABILITY_CACHE_KEY = "avail:{doctor_id}:{date}"

    def __init__(
        self,
        redis_client: aioredis.Redis,
        session_ttl: int = 7200,
        patient_cache_ttl: int = 300,
        availability_cache_ttl: int = 60,
    ):
        self.redis = redis_client
        self.session_ttl = session_ttl
        self.patient_cache_ttl = patient_cache_ttl
        self.availability_cache_ttl = availability_cache_ttl

    # ── Session (Tier 1) ───────────────────────────────────────────────────

    async def create_session(self, call_sid: str, **kwargs) -> SessionState:
        state = SessionState(call_sid=call_sid, **kwargs)
        await self._save_session(state)
        log.info("session.created", call_sid=call_sid)
        return state

    async def get_session(self, call_sid: str) -> Optional[SessionState]:
        key = self.SESSION_KEY.format(call_sid=call_sid)
        data = await self.redis.get(key)
        if data is None:
            return None
        raw = json.loads(data)
        state = SessionState(**{k: v for k, v in raw.items() if k in SessionState.__dataclass_fields__})
        return state

    async def update_session(self, state: SessionState) -> None:
        await self._save_session(state)

    async def delete_session(self, call_sid: str) -> None:
        key = self.SESSION_KEY.format(call_sid=call_sid)
        await self.redis.delete(key)
        log.info("session.deleted", call_sid=call_sid)

    async def _save_session(self, state: SessionState) -> None:
        key = self.SESSION_KEY.format(call_sid=state.call_sid)
        await self.redis.setex(key, self.session_ttl, json.dumps(asdict(state)))

    # ── Patient Context (Tier 2) ───────────────────────────────────────────

    async def get_patient_by_phone(
        self, phone_number: str, db: AsyncSession
    ) -> Optional[PatientContext]:
        """
        Look up patient by phone. Tries Redis cache first, falls back to PostgreSQL.
        """
        # Normalise phone (strip spaces, leading zeroes for Indian numbers)
        phone_number = self._normalise_phone(phone_number)

        # Query PostgreSQL for patient ID by phone
        result = await db.execute(
            select(Patient).where(Patient.phone_number == phone_number)
        )
        patient = result.scalar_one_or_none()
        if patient is None:
            return None

        return await self.get_patient_context(str(patient.id), db)

    async def get_patient_context(
        self, patient_id: str, db: AsyncSession
    ) -> Optional[PatientContext]:
        """Load full patient context with Redis caching."""
        cache_key = self.PATIENT_CACHE_KEY.format(patient_id=patient_id)

        # Try cache
        cached = await self.redis.get(cache_key)
        if cached:
            raw = json.loads(cached)
            log.debug("patient_context.cache_hit", patient_id=patient_id)
            return PatientContext(**raw)

        # Load from PostgreSQL
        context = await self._load_patient_from_db(patient_id, db)
        if context:
            await self.redis.setex(
                cache_key, self.patient_cache_ttl, json.dumps(asdict(context))
            )
        return context

    async def _load_patient_from_db(
        self, patient_id: str, db: AsyncSession
    ) -> Optional[PatientContext]:
        """Full patient context load from PostgreSQL."""
        result = await db.execute(
            select(Patient).where(Patient.id == patient_id)
        )
        patient = result.scalar_one_or_none()
        if not patient:
            return None

        # Upcoming appointments
        appt_result = await db.execute(
            select(Appointment)
            .where(
                Appointment.patient_id == patient_id,
                Appointment.status.in_([
                    AppointmentStatus.SCHEDULED,
                    AppointmentStatus.CONFIRMED
                ])
            )
            .order_by(Appointment.created_at)
            .limit(5)
        )
        upcoming_appts = []
        for appt in appt_result.scalars():
            upcoming_appts.append({
                "appointment_id": str(appt.id),
                "datetime": appt.slot.start_time.isoformat() if appt.slot else "unknown",
                "doctor_name": appt.doctor.name if appt.doctor else "unknown",
                "status": appt.status.value,
                "confirmation_code": appt.confirmation_code,
                "reason": appt.reason,
            })

        # Recent session summaries (last 3)
        summary_result = await db.execute(
            select(SessionSummary)
            .where(SessionSummary.patient_id == patient_id)
            .order_by(desc(SessionSummary.created_at))
            .limit(3)
        )
        summaries = [s.summary_text for s in summary_result.scalars() if s.summary_text]

        preferred_doctor_name = None
        preferred_specialty = None
        if patient.preferred_doctor:
            preferred_doctor_name = patient.preferred_doctor.name
            preferred_specialty = patient.preferred_doctor.specialty

        return PatientContext(
            patient_id=str(patient.id),
            name=patient.name,
            phone_number=patient.phone_number,
            preferred_language=patient.preferred_language.value,
            upcoming_appointments=upcoming_appts,
            recent_session_summaries=summaries,
            preferred_doctor_name=preferred_doctor_name,
            preferred_specialty=preferred_specialty,
            notes=patient.notes,
        )

    async def invalidate_patient_cache(self, patient_id: str) -> None:
        """Call after any write that modifies patient state."""
        cache_key = self.PATIENT_CACHE_KEY.format(patient_id=patient_id)
        await self.redis.delete(cache_key)
        log.debug("patient_cache.invalidated", patient_id=patient_id)

    # ── Availability Cache ─────────────────────────────────────────────────

    async def cache_availability(
        self, doctor_id: str, date_str: str, slots: list[dict]
    ) -> None:
        key = self.AVAILABILITY_CACHE_KEY.format(doctor_id=doctor_id, date=date_str)
        await self.redis.setex(key, self.availability_cache_ttl, json.dumps(slots))

    async def get_cached_availability(
        self, doctor_id: str, date_str: str
    ) -> Optional[list[dict]]:
        key = self.AVAILABILITY_CACHE_KEY.format(doctor_id=doctor_id, date=date_str)
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def invalidate_availability_cache(self, doctor_id: str, date_str: str) -> None:
        key = self.AVAILABILITY_CACHE_KEY.format(doctor_id=doctor_id, date=date_str)
        await self.redis.delete(key)

    # ── Session Finalisation ───────────────────────────────────────────────

    async def finalise_session(
        self,
        state: SessionState,
        db: AsyncSession,
        summary_text: str,
        outcome: str,
        appointment_id: Optional[str] = None,
    ) -> None:
        """
        Write session summary to PostgreSQL for cross-session memory.
        Called at call end.
        """
        if not state.patient_id:
            return  # Unknown caller, nothing to persist

        summary = SessionSummary(
            patient_id=state.patient_id,
            call_sid=state.call_sid,
            language_used=Language(state.language),
            duration_seconds=int(time.time() - state.started_at),
            turn_count=state.turn_count,
            summary_text=summary_text,
            outcome=outcome,
            appointment_id=appointment_id,
            entities_extracted=state.entities_extracted,
            latency_traces=state.latency_traces,
        )
        db.add(summary)

        # Update patient's preferred language
        result = await db.execute(select(Patient).where(Patient.id == state.patient_id))
        patient = result.scalar_one_or_none()
        if patient:
            patient.preferred_language = Language(state.language)

        await db.commit()
        await self.invalidate_patient_cache(state.patient_id)
        await self.delete_session(state.call_sid)
        log.info(
            "session.finalised",
            call_sid=state.call_sid,
            outcome=outcome,
            turns=state.turn_count,
        )

    @staticmethod
    def _normalise_phone(phone: str) -> str:
        """Normalise Indian phone numbers to E.164-ish format."""
        phone = phone.strip().replace(" ", "").replace("-", "")
        if phone.startswith("0"):
            phone = "+91" + phone[1:]
        if not phone.startswith("+"):
            phone = "+91" + phone
        return phone
