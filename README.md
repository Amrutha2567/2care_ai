"""
Database models for VoiceRx appointment booking system.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, ForeignKey,
    Integer, String, Text, UniqueConstraint, Index, Float, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Language(str, enum.Enum):
    EN = "en"
    HI = "hi"
    TA = "ta"


class AppointmentStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


class CampaignStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CampaignOutcome(str, enum.Enum):
    CONFIRMED = "confirmed"
    RESCHEDULED = "rescheduled"
    CANCELLED = "cancelled"
    NO_ANSWER = "no_answer"
    REJECTED = "rejected"
    VOICEMAIL = "voicemail"


# ── Doctors ────────────────────────────────────────────────────────────────

class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    specialty = Column(String(100), nullable=False)
    qualification = Column(String(200))
    languages_spoken = Column(JSON, default=list)  # ["en", "hi", "ta"]
    consultation_duration_minutes = Column(Integer, default=30)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    slots = relationship("AvailabilitySlot", back_populates="doctor", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="doctor")

    def __repr__(self):
        return f"<Doctor {self.name} ({self.specialty})>"


# ── Availability ───────────────────────────────────────────────────────────

class AvailabilitySlot(Base):
    __tablename__ = "availability_slots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    is_booked = Column(Boolean, default=False)
    is_blocked = Column(Boolean, default=False)  # doctor unavailable (holiday etc.)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    doctor = relationship("Doctor", back_populates="slots")
    appointment = relationship("Appointment", back_populates="slot", uselist=False)

    __table_args__ = (
        UniqueConstraint("doctor_id", "start_time", name="uq_doctor_slot"),
        Index("ix_slots_doctor_start", "doctor_id", "start_time"),
        Index("ix_slots_available", "doctor_id", "is_booked", "is_blocked", "start_time"),
    )


# ── Patients ───────────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    phone_number = Column(String(20), unique=True, nullable=False)
    preferred_language = Column(Enum(Language), default=Language.EN)
    preferred_doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    email = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)  # clinical notes, chronic conditions (non-sensitive)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    appointments = relationship("Appointment", back_populates="patient")
    sessions = relationship("SessionSummary", back_populates="patient")
    preferred_doctor = relationship("Doctor", foreign_keys=[preferred_doctor_id])

    __table_args__ = (
        Index("ix_patients_phone", "phone_number"),
    )


# ── Appointments ───────────────────────────────────────────────────────────

class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)
    slot_id = Column(UUID(as_uuid=True), ForeignKey("availability_slots.id"), nullable=False, unique=True)
    status = Column(Enum(AppointmentStatus), default=AppointmentStatus.SCHEDULED)
    reason = Column(Text, nullable=True)
    confirmation_code = Column(String(8), unique=True)
    booked_via = Column(String(50), default="voice_agent")  # voice_agent | web | campaign
    session_id = Column(String(200), nullable=True)  # call_sid that created this
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    cancellation_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")
    slot = relationship("AvailabilitySlot", back_populates="appointment")

    __table_args__ = (
        Index("ix_appointments_patient", "patient_id", "status"),
        Index("ix_appointments_doctor_date", "doctor_id", "status"),
    )


# ── Session Summaries ──────────────────────────────────────────────────────

class SessionSummary(Base):
    __tablename__ = "session_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    call_sid = Column(String(200), unique=True)
    language_used = Column(Enum(Language), default=Language.EN)
    duration_seconds = Column(Integer, nullable=True)
    turn_count = Column(Integer, default=0)
    summary_text = Column(Text)          # LLM-generated summary for future context
    outcome = Column(String(50))          # booked | cancelled | rescheduled | inquiry | no_action
    appointment_id = Column(UUID(as_uuid=True), ForeignKey("appointments.id"), nullable=True)
    entities_extracted = Column(JSON, default=dict)  # doctor prefs, date prefs etc.
    latency_traces = Column(JSON, default=list)       # array of LatencyTrace dicts
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="sessions")

    __table_args__ = (
        Index("ix_sessions_patient", "patient_id", "created_at"),
    )


# ── Campaigns ─────────────────────────────────────────────────────────────

class CampaignJob(Base):
    __tablename__ = "campaign_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    campaign_type = Column(String(50), nullable=False)  # reminder | follow_up | recall
    scheduled_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(Enum(CampaignStatus), default=CampaignStatus.PENDING)
    target_patient_ids = Column(JSON, default=list)
    script_template = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    logs = relationship("CampaignLog", back_populates="job", cascade="all, delete-orphan")


class CampaignLog(Base):
    __tablename__ = "campaign_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("campaign_jobs.id", ondelete="CASCADE"), nullable=False)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    call_sid = Column(String(200), nullable=True)
    outcome = Column(Enum(CampaignOutcome), nullable=True)
    outcome_details = Column(Text, nullable=True)
    call_duration_seconds = Column(Integer, nullable=True)
    attempted_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("CampaignJob", back_populates="logs")
    patient = relationship("Patient")
