"""Read live agentic-eval progress from per-key timeline.jsonl."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_key_filename(key: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(key or "").strip())[:120].strip("_")
    return safe or "key"

_SEARCH_LABEL_RE = re.compile(r"_n\d+_s(\d+)(?:_s(\d+))?$")


def _parse_search_label(label: str) -> Tuple[Optional[int], Optional[int]]:
    match = _SEARCH_LABEL_RE.search(str(label or ""))
    if not match:
        return None, None
    session = int(match.group(1))
    turn = int(match.group(2)) if match.group(2) else None
    return session, turn


def _tail_jsonl(path: Path, *, max_lines: int = 500) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _count_search_sessions(events: List[Dict[str, Any]]) -> int:
    count = 0
    for event in events:
        if (
            event.get("agent") == "search"
            and event.get("event") == "llm_request"
            and event.get("step") == 1
        ):
            count += 1
    return count


def _read_status_started_at(run_dir: Path, key: str) -> Optional[str]:
    status_path = run_dir / "06_agentic_eval" / f"{_safe_key_filename(key)}.status.json"
    if not status_path.is_file():
        return None
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    started = data.get("started_at")
    return str(started) if started else None


def read_eval_live_progress(run_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    """
    Summarize the latest EvalMaster / SearchAgent position for a running key.

    Returns None when no timeline exists yet.
    """
    run_dir = run_dir.resolve()
    key = str(key or "").strip()
    if not key:
        return None

    timeline_path = run_dir / "06_agentic_eval" / _safe_key_filename(key) / "timeline.jsonl"
    if not timeline_path.is_file():
        return None

    events = _tail_jsonl(timeline_path)
    if not events:
        return None

    last = events[-1]
    last_event = str(last.get("event") or "")
    last_agent = str(last.get("agent") or "")
    last_name = str(last.get("name") or "")

    master_turn: Optional[int] = None
    for event in reversed(events):
        if event.get("agent") != "eval":
            continue
        if event.get("event") not in {"llm_request", "llm_response", "step"}:
            continue
        step = event.get("step")
        if isinstance(step, int):
            master_turn = step
            break

    search_session: Optional[int] = None
    search_turn: Optional[int] = None
    for event in reversed(events):
        label = str(event.get("label") or "")
        session, turn = _parse_search_label(label)
        if session is not None:
            search_session = session
            if turn is not None:
                search_turn = turn
            break
        if event.get("agent") == "search":
            step = event.get("step")
            if isinstance(step, int):
                search_turn = step
            break

    if search_session is None:
        search_session = _count_search_sessions(events) or None

    activity = last_event or None
    active_agent = last_agent or None
    if last_event == "llm_request":
        activity = "waiting_llm"
        active_agent = last_agent or active_agent
    elif last_event == "tool" and last_name:
        activity = f"tool:{last_name}"
        if label := str(last.get("label") or ""):
            if "_search_" in label:
                active_agent = "search"

    elapsed_s: Optional[float] = None
    if isinstance(last.get("t"), (int, float)):
        elapsed_s = round(float(last["t"]), 1)

    started_at = _read_status_started_at(run_dir, key)
    wall_elapsed_s: Optional[float] = None
    if started_at:
        try:
            started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            wall_elapsed_s = round(
                (datetime.now(timezone.utc) - started_dt).total_seconds(), 1
            )
        except ValueError:
            wall_elapsed_s = None

    return {
        "master_turn": master_turn,
        "search_session": search_session,
        "search_turn": search_turn,
        "active_agent": active_agent or None,
        "activity": activity,
        "elapsed_s": elapsed_s,
        "wall_elapsed_s": wall_elapsed_s,
        "last_event": last_event or None,
        "timeline_mtime": timeline_path.stat().st_mtime,
    }


def format_live_progress(progress: Optional[Dict[str, Any]]) -> str:
    """Human-readable one-line summary for the viewer UI."""
    if not progress:
        return ""
    parts: List[str] = []
    if progress.get("master_turn") is not None:
        parts.append(f"EvalMaster turn {progress['master_turn']}")
    session = progress.get("search_session")
    turn = progress.get("search_turn")
    if session is not None:
        if turn is not None:
            parts.append(f"Search session {session} turn {turn}")
        else:
            parts.append(f"Search session {session}")
    activity = str(progress.get("activity") or "")
    if activity == "waiting_llm":
        agent = progress.get("active_agent")
        if agent == "search":
            parts.append("waiting Search LLM")
        elif agent == "eval":
            parts.append("waiting EvalMaster LLM")
        else:
            parts.append("waiting LLM")
    elif activity.startswith("tool:"):
        parts.append(activity.replace("tool:", "tool ", 1))
    return " · ".join(parts)
