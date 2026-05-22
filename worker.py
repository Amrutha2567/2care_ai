"""
campaigns_router.py — REST API for campaign management.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, text

from api.dependencies import get_db
from campaigns.worker import make_outbound_call, build_reminder_script

log = structlog.get_logger()
router = APIRouter(prefix="/campaigns", tags=["campaigns"])


class CreateCampaignRequest(BaseModel):
    name: str
    campaign_type: str  # reminder | follow_up | recall
    patient_ids: List[str]
    scheduled_at: Optional[datetime] = None  # None = immediate


class ManualCallRequest(BaseModel):
    patient_phone: str
    patient_name: str
    language: str = "en"
    script: str


@router.post("/")
async def create_campaign(req: CreateCampaignRequest):
    """Create and schedule a campaign."""
    job_id = str(uuid.uuid4())
    
    # Enqueue calls
    for patient_id in req.patient_ids:
        make_outbound_call.apply_async(
            kwargs={
                "patient_id": patient_id,
                "patient_phone": "LOOKUP",  # Worker resolves from DB
                "patient_name": "LOOKUP",
                "language": "en",
                "campaign_type": req.campaign_type,
                "campaign_script": f"Campaign: {req.name}",
                "campaign_job_id": job_id,
            },
            eta=req.scheduled_at,
        )

    log.info("campaign.created", job_id=job_id, patient_count=len(req.patient_ids))
    return {"job_id": job_id, "queued": len(req.patient_ids)}


@router.post("/manual-call")
async def manual_outbound_call(req: ManualCallRequest):
    """Trigger a single manual outbound call for testing."""
    task = make_outbound_call.apply_async(
        kwargs={
            "patient_id": "manual",
            "patient_phone": req.patient_phone,
            "patient_name": req.patient_name,
            "language": req.language,
            "campaign_type": "manual",
            "campaign_script": req.script,
            "campaign_job_id": str(uuid.uuid4()),
        }
    )
    return {"task_id": task.id, "status": "queued"}


@router.get("/jobs")
async def list_campaign_jobs():
    """List recent campaign jobs."""
    async with get_db() as db:
        result = await db.execute(
            text("""
                SELECT id, name, campaign_type, status, scheduled_at, created_at,
                       (SELECT COUNT(*) FROM campaign_logs WHERE job_id = campaign_jobs.id) as call_count
                FROM campaign_jobs
                ORDER BY created_at DESC
                LIMIT 20
            """)
        )
        rows = result.fetchall()

    return {
        "jobs": [
            {
                "id": str(r[0]),
                "name": r[1],
                "type": r[2],
                "status": r[3],
                "scheduled_at": r[4].isoformat() if r[4] else None,
                "created_at": r[5].isoformat(),
                "call_count": r[6],
            }
            for r in rows
        ]
    }
