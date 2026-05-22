"""
dependencies.py — FastAPI dependency injection.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

# Import globals from main — resolved at request time
def get_memory_manager():
    from api.main import _memory_manager
    return _memory_manager


def get_scheduling_service():
    from api.main import _scheduling_service
    return _scheduling_service


def get_agent():
    from api.main import _agent
    return _agent


@asynccontextmanager
async def get_db() -> AsyncIterator[AsyncSession]:
    from api.main import _session_factory
    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_redis():
    from api.main import _redis_client
    return _redis_client
