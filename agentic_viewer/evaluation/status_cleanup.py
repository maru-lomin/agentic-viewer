"""Clean up stale agentic-eval status files on the shared runs volume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def mark_running_eval_status_cancelled(run_dir: Path) -> int:
    """Mark ``06_agentic_eval/*.status.json`` entries still ``running`` as cancelled."""
    out_dir = run_dir / "06_agentic_eval"
    if not out_dir.is_dir():
        return 0
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for path in out_dir.glob("*.status.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("status") != "running":
            continue
        data["status"] = "cancelled"
        data["finished_at"] = now
        data["error"] = "cancelled by user"
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            n += 1
        except OSError:
            # Docker-owned status files may be unwritable from the viewer process.
            continue
    return n
