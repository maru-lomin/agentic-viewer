"""Timezone constants and utilities for KST (Korea Standard Time, UTC+9)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

# Korea Standard Time (UTC+9)
KST = timezone(timedelta(hours=9), name="KST")


def kst_now() -> datetime:
    """Return the current datetime in KST."""
    return datetime.now(KST)


def kst_now_iso() -> str:
    """Return the current KST timestamp formatted as ISO 8601 string."""
    return kst_now().isoformat()


def to_kst(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a datetime to KST.

    If naive, assumes KST. If tz-aware, converts to KST.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=KST)
    return dt.astimezone(KST)
