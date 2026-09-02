"""Resolve agentic-eval trace directories under 06_agentic_eval/."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_viewer.evaluation.summary import read_agentic_evals


def safe_key_filename(key: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(key or "").strip())[:120].strip("_")
    return safe or "key"


def _remap_trace_path(run_dir: Path, trace_path: str) -> Optional[Path]:
    """Map docker ``/workspace/outputs/runs/...`` paths to the host run tree."""
    raw = Path(str(trace_path).strip())
    if raw.is_dir() and (raw / "03_agent").is_dir():
        return raw.resolve()
    parts = raw.parts
    if "06_agentic_eval" in parts:
        idx = parts.index("06_agentic_eval")
        candidate = (run_dir / Path(*parts[idx:])).resolve()
        if candidate.is_dir() and (candidate / "03_agent").is_dir():
            return candidate
    return None


def resolve_agentic_eval_trace_dir(run_dir: Path, key: str) -> Path:
    """
    Return ``06_agentic_eval/<trace>/`` for a logical schema key.

    Raises ``KeyError`` when no trace artifacts exist.
    """
    run_dir = run_dir.resolve()
    key = str(key or "").strip()
    if not key:
        raise KeyError("key is required")

    by_key = read_agentic_evals(run_dir)
    row = by_key.get(key)
    if isinstance(row, dict):
        trace_dir = row.get("trace_dir")
        if trace_dir:
            remapped = _remap_trace_path(run_dir, str(trace_dir))
            if remapped is not None:
                return remapped

    out_dir = run_dir / "06_agentic_eval"
    if not out_dir.is_dir():
        raise KeyError(f"agentic eval not found for key={key!r}")

    candidate = out_dir / safe_key_filename(key)
    if (candidate / "03_agent").is_dir():
        return candidate.resolve()

    for child in sorted(out_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "03_agent").is_dir():
            continue
        req = child / "00_request.json"
        if req.is_file():
            try:
                data = json.loads(req.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict) and str(data.get("key") or "") == key:
                return child.resolve()

    raise KeyError(f"agentic eval trace not found for key={key!r}")


def list_agentic_eval_keys(run_dir: Path) -> List[Dict[str, Any]]:
    """Keys with eval traces or status under ``06_agentic_eval/``."""
    by_key = read_agentic_evals(run_dir)
    rows: List[Dict[str, Any]] = []
    for key, payload in sorted(by_key.items(), key=lambda kv: kv[0]):
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "key": key,
                "status": payload.get("status") or "pending",
                "is_correct_answer": payload.get("is_correct_answer"),
                "is_valid_gold": payload.get("is_valid_gold"),
                "reason_summary": payload.get("reason_summary") or payload.get("reason"),
                "error": payload.get("error"),
                "trace_dir": payload.get("trace_dir"),
            }
        )
    return rows
