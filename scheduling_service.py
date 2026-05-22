"""
scheduling_service.py — Appointment lifecycle and conflict management.

All DB writes acquire row-level locks to prevent double-booking under concurrent calls.
Tool functions here are called directly by the Claude agent.
"""
from __future__ import annotations

import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    Appointment, AppointmentStatus, AvailabilitySlot, Doctor, Patient
)
from memory.memory_manager import MemoryManager

log = structlog.get_logger()


def _generate_confirmation_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


class ConflictError(Exception):
    """Raised when a requested slot is not available."""
    pass


class NotFoundError(Exception):
    pass


class SchedulingService:
    """
    Stateless service layer. All methods are async and accept a DB session.
    Returns plain dicts suitable for direct JSON serialisation and LLM injection.
    """

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    # ── Availability ───────────────────────────────────────────────────────

    async def check_availability(
        self,
        db: AsyncSession,
        doctor_id: Optional[str] = None,
        specialty: Optional[str] = None,
        date_str: Optional[str] = None,        # "YYYY-MM-DD"
        days_ahead: int = 7,
    ) -> dict:
        """
        Find available slots. Returns grouped slots by doctor.
        Uses Redis cache (60s TTL) to reduce DB load.
        """
        # Parse date
        if date_str:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                target_date = datetime.now(timezone.utc).date()
        else:
            target_date = datetime.now(timezone.utc).date()

        start_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=days_ahead)

        # Try cache if specific doctor+date
        if doctor_id and date_str:
            cached = await self.memory.get_cached_availability(doctor_id, date_str)
            if cached is not None:
                log.debug("availability.cache_hit", doctor_id=doctor_id, date=date_str)
                return {"slots": cached, "cached": True}

        # Build query
        q = (
            select(AvailabilitySlot, Doctor)
            .join(Doctor, AvailabilitySlot.doctor_id == Doctor.id)
            .where(
                and_(
                    AvailabilitySlot.is_booked == False,
                    AvailabilitySlot.is_blocked == False,
                    AvailabilitySlot.start_time >= start_dt,
                    AvailabilitySlot.start_time < end_dt,
                    AvailabilitySlot.start_time > func.now(),  # No past slots
                    Doctor.is_active == True,
                )
            )
            .order_by(AvailabilitySlot.start_time)
            .limit(50)
        )

        if doctor_id:
            q = q.where(AvailabilitySlot.doctor_id == uuid.UUID(doctor_id))
        if specialty:
            q = q.where(Doctor.specialty.ilike(f"%{specialty}%"))

        result = await db.execute(q)
        rows = result.all()

        slots = []
        for slot, doctor in rows:
            slots.append({
                "slot_id": str(slot.id),
                "doctor_id": str(doctor.id),
                "doctor_name": doctor.name,
                "specialty": doctor.specialty,
                "start_time": slot.start_time.isoformat(),
                "end_time": slot.end_time.isoformat(),
                "date": slot.start_time.strftime("%A, %d %B %Y"),
                "time": slot.start_time.strftime("%I:%M %p"),
                "duration_minutes": doctor.consultation_duration_minutes,
            })

        # Cache if specific doctor+date
        if doctor_id and date_str:
            await self.memory.cache_availability(doctor_id, date_str, slots)

        return {"slots": slots, "total": len(slots)}

    # ── Booking ────────────────────────────────────────────────────────────

    async def book_appointment(
        self,
        db: AsyncSession,
        patient_id: str,
        slot_id: str,
        reason: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Book an appointment. Uses SELECT FOR UPDATE to prevent double-booking.
        Raises ConflictError if slot is taken.
        """
        # Lock the slot row
        slot_result = await db.execute(
            select(AvailabilitySlot)
            .where(AvailabilitySlot.id == uuid.UUID(slot_id))
            .with_for_update()
        )
        slot = slot_result.scalar_one_or_none()

        if slot is None:
            raise NotFoundError(f"Slot {slot_id} not found")

        if slot.is_booked or slot.is_blocked:
            raise ConflictError(
                f"Slot at {slot.start_time.strftime('%I:%M %p, %d %B')} is no longer available"
            )

        if slot.start_time <= datetime.now(timezone.utc):
            raise ConflictError("Cannot book an appointment in the past")

        # Check patient has no overlapping appointment
        overlap_result = await db.execute(
            select(Appointment)
            .join(AvailabilitySlot, Appointment.slot_id == AvailabilitySlot.id)
            .where(
                and_(
                    Appointment.patient_id == uuid.UUID(patient_id),
                    Appointment.status.in_([
                        AppointmentStatus.SCHEDULED,
                        AppointmentStatus.CONFIRMED
                    ]),
                    AvailabilitySlot.start_time == slot.start_time,
                )
            )
        )
        if overlap_result.scalar_one_or_none():
            raise ConflictError("You already have an appointment at this time")

        # Fetch doctor
        doctor_result = await db.execute(select(Doctor).where(Doctor.id == slot.doctor_id))
        doctor = doctor_result.scalar_one()

        # Create appointment
        confirmation_code = _generate_confirmation_code()
        appointment = Appointment(
            patient_id=uuid.UUID(patient_id),
            doctor_id=slot.doctor_id,
            slot_id=slot.id,
            status=AppointmentStatus.SCHEDULED,
            reason=reason,
            confirmation_code=confirmation_code,
            booked_via="voice_agent",
            session_id=session_id,
        )
        slot.is_booked = True
        db.add(appointment)
        await db.commit()
        await db.refresh(appointment)

        # Invalidate caches
        await self.memory.invalidate_availability_cache(
            str(slot.doctor_id), slot.start_time.strftime("%Y-%m-%d")
        )
        await self.memory.invalidate_patient_cache(patient_id)

        log.info(
            "appointment.booked",
            appointment_id=str(appointment.id),
            patient_id=patient_id,
            doctor=doctor.name,
            slot=slot.start_time.isoformat(),
        )

        return {
            "success": True,
            "appointment_id": str(appointment.id),
            "confirmation_code": confirmation_code,
            "doctor_name": doctor.name,
            "specialty": doctor.specialty,
            "datetime": slot.start_time.isoformat(),
            "date": slot.start_time.strftime("%A, %d %B %Y"),
            "time": slot.start_time.strftime("%I:%M %p"),
            "duration_minutes": doctor.consultation_duration_minutes,
            "message": (
                f"Appointment confirmed with {doctor.name} on "
                f"{slot.start_time.strftime('%A, %d %B at %I:%M %p')}. "
                f"Confirmation code: {confirmation_code}"
            ),
        }

    # ── Reschedule ─────────────────────────────────────────────────────────

    async def reschedule_appointment(
        self,
        db: AsyncSession,
        appointment_id: str,
        new_slot_id: str,
        patient_id: str,
    ) -> dict:
        """
        Reschedule an existing appointment to a new slot.
        Frees the old slot and books the new one atomically.
        """
        # Fetch and validate current appointment
        appt_result = await db.execute(
            select(Appointment)
            .where(
                and_(
                    Appointment.id == uuid.UUID(appointment_id),
                    Appointment.patient_id == uuid.UUID(patient_id),
                )
            )
            .with_for_update()
        )
        appointment = appt_result.scalar_one_or_none()
        if not appointment:
            raise NotFoundError("Appointment not found or does not belong to this patient")
        if appointment.status == AppointmentStatus.CANCELLED:
            raise ConflictError("Cannot reschedule a cancelled appointment")

        # Lock new slot
        new_slot_result = await db.execute(
            select(AvailabilitySlot)
            .where(AvailabilitySlot.id == uuid.UUID(new_slot_id))
            .with_for_update()
        )
        new_slot = new_slot_result.scalar_one_or_none()
        if not new_slot:
            raise NotFoundError("New slot not found")
        if new_slot.is_booked or new_slot.is_blocked:
            raise ConflictError("New slot is no longer available")
        if new_slot.start_time <= datetime.now(timezone.utc):
            raise ConflictError("Cannot reschedule to a time in the past")

        # Free old slot
        old_slot_result = await db.execute(
            select(AvailabilitySlot)
            .where(AvailabilitySlot.id == appointment.slot_id)
            .with_for_update()
        )
        old_slot = old_slot_result.scalar_one()
        old_slot.is_booked = False

        # Reassign appointment
        appointment.slot_id = new_slot.id
        appointment.doctor_id = new_slot.doctor_id
        appointment.status = AppointmentStatus.SCHEDULED
        new_slot.is_booked = True

        await db.commit()

        doctor_result = await db.execute(select(Doctor).where(Doctor.id == new_slot.doctor_id))
        doctor = doctor_result.scalar_one()

        await self.memory.invalidate_patient_cache(patient_id)

        log.info(
            "appointment.rescheduled",
            appointment_id=appointment_id,
            new_slot=new_slot.start_time.isoformat(),
        )

        return {
            "success": True,
            "appointment_id": appointment_id,
            "confirmation_code": appointment.confirmation_code,
            "doctor_name": doctor.name,
            "new_datetime": new_slot.start_time.isoformat(),
            "date": new_slot.start_time.strftime("%A, %d %B %Y"),
            "time": new_slot.start_time.strftime("%I:%M %p"),
            "message": (
                f"Your appointment has been rescheduled to "
                f"{new_slot.start_time.strftime('%A, %d %B at %I:%M %p')} with {doctor.name}. "
                f"Confirmation code remains: {appointment.confirmation_code}"
            ),
        }

    # ── Cancellation ───────────────────────────────────────────────────────

    async def cancel_appointment(
        self,
        db: AsyncSession,
        appointment_id: str,
        patient_id: str,
        reason: Optional[str] = None,
    ) -> dict:
        """Cancel an appointment and free the slot."""
        appt_result = await db.execute(
            select(Appointment)
            .where(
                and_(
                    Appointment.id == uuid.UUID(appointment_id),
                    Appointment.patient_id == uuid.UUID(patient_id),
                )
            )
            .with_for_update()
        )
        appointment = appt_result.scalar_one_or_none()
        if not appointment:
            raise NotFoundError("Appointment not found")
        if appointment.status == AppointmentStatus.CANCELLED:
            raise ConflictError("Appointment is already cancelled")

        # Free slot
        slot_result = await db.execute(
            select(AvailabilitySlot)
            .where(AvailabilitySlot.id == appointment.slot_id)
            .with_for_update()
        )
        slot = slot_result.scalar_one()
        slot.is_booked = False

        appointment.status = AppointmentStatus.CANCELLED
        appointment.cancelled_at = datetime.now(timezone.utc)
        appointment.cancellation_reason = reason

        await db.commit()

        doctor_result = await db.execute(select(Doctor).where(Doctor.id == appointment.doctor_id))
        doctor = doctor_result.scalar_one()

        await self.memory.invalidate_patient_cache(patient_id)
        await self.memory.invalidate_availability_cache(
            str(appointment.doctor_id), slot.start_time.strftime("%Y-%m-%d")
        )

        log.info("appointment.cancelled", appointment_id=appointment_id)

        return {
            "success": True,
            "appointment_id": appointment_id,
            "doctor_name": doctor.name,
            "cancelled_datetime": slot.start_time.isoformat(),
            "message": (
                f"Your appointment with {doctor.name} on "
                f"{slot.start_time.strftime('%A, %d %B at %I:%M %p')} has been cancelled."
            ),
        }

    # ── Find Alternatives ─────────────────────────────────────────────────

    async def find_alternatives(
        self,
        db: AsyncSession,
        doctor_id: Optional[str] = None,
        specialty: Optional[str] = None,
        preferred_date_str: Optional[str] = None,
        count: int = 3,
    ) -> dict:
        """
        Suggest alternative slots when the requested one is unavailable.
        Expands the search window progressively (1 day → 3 days → 7 days).
        """
        for days in [1, 3, 7, 14]:
            result = await self.check_availability(
                db=db,
                doctor_id=doctor_id,
                specialty=specialty,
                date_str=preferred_date_str,
                days_ahead=days,
            )
            if result["slots"]:
                alternatives = result["slots"][:count]
                return {
                    "alternatives": alternatives,
                    "search_window_days": days,
                    "message": (
                        f"I found {len(alternatives)} available slot(s) within "
                        f"the next {days} day(s)."
                    ),
                }

        return {
            "alternatives": [],
            "message": "No availability found in the next 14 days. Please try a different doctor or specialty.",
        }

    # ── Patient History ────────────────────────────────────────────────────

    async def get_patient_appointments(
        self,
        db: AsyncSession,
        patient_id: str,
        include_past: bool = False,
        limit: int = 10,
    ) -> dict:
        """Retrieve a patient's appointment history."""
        q = (
            select(Appointment)
            .where(Appointment.patient_id == uuid.UUID(patient_id))
            .order_by(Appointment.created_at.desc())
            .limit(limit)
        )
        if not include_past:
            q = q.where(
                Appointment.status.in_([
                    AppointmentStatus.SCHEDULED,
                    AppointmentStatus.CONFIRMED
                ])
            )

        result = await db.execute(q)
        appointments = []
        for appt in result.scalars():
            slot_r = await db.execute(
                select(AvailabilitySlot).where(AvailabilitySlot.id == appt.slot_id)
            )
            slot = slot_r.scalar_one_or_none()
            doctor_r = await db.execute(
                select(Doctor).where(Doctor.id == appt.doctor_id)
            )
            doctor = doctor_r.scalar_one_or_none()

            appointments.append({
                "appointment_id": str(appt.id),
                "confirmation_code": appt.confirmation_code,
                "doctor_name": doctor.name if doctor else "Unknown",
                "specialty": doctor.specialty if doctor else "Unknown",
                "datetime": slot.start_time.isoformat() if slot else "Unknown",
                "date": slot.start_time.strftime("%A, %d %B %Y") if slot else "Unknown",
                "time": slot.start_time.strftime("%I:%M %p") if slot else "Unknown",
                "status": appt.status.value,
                "reason": appt.reason,
            })

        return {"appointments": appointments, "total": len(appointments)}
