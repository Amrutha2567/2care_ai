"""
metrics_router.py — Latency and performance metrics endpoints.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_db

log = structlog.get_logger()
router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/latency")
async def get_latency_metrics():
    """
    Return p50/p95/p99 latency breakdown across recent calls.
    Reads from session_summaries.latency_traces JSON column.
    """
    async with get_db() as db:
        result = await db.execute(
            text("""
                SELECT 
                    jsonb_array_elements(latency_traces::jsonb) as trace
                FROM session_summaries
                WHERE created_at > NOW() - INTERVAL '24 hours'
                  AND latency_traces IS NOT NULL
                  AND latency_traces != '[]'
                LIMIT 1000
            """)
        )
        rows = result.fetchall()

    totals = []
    for row in rows:
        trace = row[0]
        if isinstance(trace, dict) and "total_ms" in trace:
            totals.append(trace["total_ms"])

    if not totals:
        return {"message": "No latency data in last 24 hours", "samples": 0}

    totals.sort()
    n = len(totals)

    def percentile(data, p):
        idx = int(p / 100 * len(data))
        return data[min(idx, len(data) - 1)]

    return {
        "samples": n,
        "window": "24h",
        "latency_ms": {
            "p50": round(percentile(totals, 50), 1),
            "p75": round(percentile(totals, 75), 1),
            "p95": round(percentile(totals, 95), 1),
            "p99": round(percentile(totals, 99), 1),
            "min": round(min(totals), 1),
            "max": round(max(totals), 1),
            "mean": round(sum(totals) / n, 1),
        },
        "target_450ms_met_pct": round(
            sum(1 for t in totals if t < 450) / n * 100, 1
        ),
    }


@router.get("/calls")
async def get_call_metrics():
    """Summary of recent call activity."""
    async with get_db() as db:
        result = await db.execute(
            text("""
                SELECT 
                    COUNT(*) as total_calls,
                    AVG(duration_seconds) as avg_duration_s,
                    COUNT(CASE WHEN outcome = 'booked' THEN 1 END) as booked,
                    COUNT(CASE WHEN outcome = 'cancelled' THEN 1 END) as cancelled,
                    COUNT(CASE WHEN outcome = 'rescheduled' THEN 1 END) as rescheduled,
                    COUNT(CASE WHEN language_used = 'en' THEN 1 END) as english_calls,
                    COUNT(CASE WHEN language_used = 'hi' THEN 1 END) as hindi_calls,
                    COUNT(CASE WHEN language_used = 'ta' THEN 1 END) as tamil_calls
                FROM session_summaries
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
        )
        row = result.fetchone()

    return {
        "window": "24h",
        "total_calls": row[0] or 0,
        "avg_duration_seconds": round(row[1] or 0, 1),
        "outcomes": {
            "booked": row[2] or 0,
            "cancelled": row[3] or 0,
            "rescheduled": row[4] or 0,
        },
        "languages": {
            "en": row[5] or 0,
            "hi": row[6] or 0,
            "ta": row[7] or 0,
        },
    }
