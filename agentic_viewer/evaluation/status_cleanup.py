"""Clean up stale agentic-eval status files on the shared runs volume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

from agentic_viewer.timezone import kst_now_iso


def mark_running_eval_status_cancelled(
    run_dir: Path,
    reason: str = "cancelled by user",
    key: Optional[str] = None,
) -> int:
    """Mark ``06_agentic_eval/*.status.json`` entries still ``running`` as cancelled."""
    out_dir = run_dir / "06_agentic_eval"
    if not out_dir.is_dir():
        return 0
    now = kst_now_iso()
    n = 0
    for path in out_dir.glob("*.status.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("status") != "running":
            continue
        if key is not None and str(data.get("key") or "").strip() != key.strip():
            continue
        data["status"] = "cancelled"
        data["finished_at"] = now
        data["error"] = reason
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            n += 1
        except OSError:
            # Docker-owned status files may be unwritable from the viewer process.
            continue
    return n


def cleanup_all_running_eval_statuses(
    runs_root: Path,
    run_ids: Optional[Sequence[str]] = None,
    reason: str = "cancelled (server restarted or interrupted)",
) -> Dict[str, int]:
    """Clean up orphaned running status files across runs."""
    cleaned: Dict[str, int] = {}
    if not runs_root.is_dir():
        return cleaned

    if run_ids:
        targets = [runs_root / rid for rid in run_ids]
    else:
        targets = [p for p in runs_root.iterdir() if p.is_dir()]

    for run_dir in targets:
        if not run_dir.is_dir():
            continue
        count = mark_running_eval_status_cancelled(run_dir, reason=reason)
        if count > 0:
            cleaned[run_dir.name] = count
    return cleaned
