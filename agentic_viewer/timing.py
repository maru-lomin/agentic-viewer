"""Build agent / session / turn timing reports from timeline.jsonl."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_timeline(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "timeline.jsonl"
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _agent_key(ev: Dict[str, Any]) -> Tuple[str, str, int]:
    agent = str(ev.get("agent") or "master")
    key = str(ev.get("key") or "")
    step = int(ev.get("step") or 0)
    return agent, key, step


def _round_secs(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def _pct(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)


def _pair_llm_events(
    events: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pair llm_request → llm_response; also return unpaired in-flight requests."""
    pending: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    pairs: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("event") == "llm_request":
            pending[_agent_key(ev)] = ev
            continue
        if ev.get("event") != "llm_response":
            continue
        k = _agent_key(ev)
        req = pending.pop(k, None)
        if req is None:
            continue
        start = float(req.get("t") or 0)
        end = float(ev.get("t") or 0)
        pairs.append(
            {
                "agent": k[0],
                "key": k[1],
                "step": k[2],
                "start_t": start,
                "end_t": end,
                "llm_seconds": _round_secs(end - start),
                "prompt_est_tokens": req.get("prompt_est_tokens"),
                "input_tokens": ev.get("input_tokens"),
                "output_tokens": ev.get("output_tokens"),
            }
        )
    open_reqs: List[Dict[str, Any]] = []
    for k, req in pending.items():
        open_reqs.append(
            {
                "agent": k[0],
                "key": k[1],
                "step": k[2],
                "start_t": float(req.get("t") or 0),
                "prompt_est_tokens": req.get("prompt_est_tokens"),
                "input_budget": req.get("input_budget"),
            }
        )
    return pairs, open_reqs


def _safe_key_prefix(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(key or ""))[:40]


def _stage_timings(events: List[Dict[str, Any]], total: float) -> List[Dict[str, Any]]:
    bounds: Dict[str, List[float]] = {}
    done_at: Dict[str, float] = {}
    request_t = 0.0
    for ev in events:
        stage = str(ev.get("stage") or "")
        t = float(ev.get("t") or 0)
        if stage == "request" and request_t <= 0:
            request_t = t
        if stage not in {"parse", "chunk", "agent", "result"}:
            continue
        if stage not in bounds:
            bounds[stage] = [t, t]
        else:
            bounds[stage][0] = min(bounds[stage][0], t)
            bounds[stage][1] = max(bounds[stage][1], t)
        if ev.get("event") == "done" or (
            stage == "result" and ev.get("event") in {"done", "error"}
        ):
            done_at[stage] = t

    # Legacy traces often emit only parse/done (and chunk/done). Anchor those
    # stages to the preceding boundary so Timing still shows wall time.
    if "parse" in bounds and "parse" in done_at:
        only_done = bounds["parse"][0] == bounds["parse"][1] == done_at["parse"]
        if only_done:
            bounds["parse"][0] = request_t
    if "chunk" in bounds and "chunk" in done_at and "parse" in done_at:
        only_done = bounds["chunk"][0] == bounds["chunk"][1] == done_at["chunk"]
        if only_done:
            bounds["chunk"][0] = done_at["parse"]

    rows: List[Dict[str, Any]] = []
    for stage in ("parse", "chunk", "agent", "result"):
        if stage not in bounds:
            continue
        start, end = bounds[stage]
        finished = stage in done_at
        if not finished:
            end = max(end, total)
        seconds = max(0.0, end - start)
        rows.append(
            {
                "stage": stage,
                "status": "complete" if finished else "running",
                "start_t": _round_secs(start),
                "end_t": _round_secs(end),
                "seconds": _round_secs(seconds),
                "pct": _pct(seconds, total) if finished else None,
            }
        )
    return rows


def _pipeline_progress(
    run_dir: Path, events: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Current parse/chunk stage progress for the Timing banner."""
    parse_progress = None
    parse_path = run_dir / "01_parse" / "progress.json"
    if parse_path.is_file():
        try:
            parse_progress = json.loads(parse_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parse_progress = None

    chunk_progress = None
    chunk_path = run_dir / "02_chunk" / "progress.json"
    if chunk_path.is_file():
        try:
            chunk_progress = json.loads(chunk_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            chunk_progress = None

    has_parse_done = any(
        e.get("stage") == "parse" and e.get("event") == "done" for e in events
    )
    has_chunk_done = any(
        e.get("stage") == "chunk" and e.get("event") == "done" for e in events
    )
    has_parse_start = any(
        e.get("stage") == "parse" and e.get("event") in {"start", "page"}
        for e in events
    )
    has_chunk_start = any(
        e.get("stage") == "chunk" and e.get("event") == "start" for e in events
    )

    if parse_progress and str(parse_progress.get("status") or "") == "running":
        page = int(parse_progress.get("page") or 0)
        total_pages = parse_progress.get("total_pages")
        return {
            "stage": "parse",
            "status": "running",
            "page": page,
            "total_pages": total_pages,
            "seconds": parse_progress.get("seconds"),
            "label": (
                f"Parsing page {page}/{total_pages}"
                if total_pages
                else (f"Parsing page {page}" if page else "Parsing…")
            ),
        }
    if (
        has_parse_start
        and not has_parse_done
        and (not parse_progress or str(parse_progress.get("status")) != "done")
    ):
        last_page = 0
        total_pages = None
        for e in events:
            if e.get("stage") != "parse":
                continue
            if e.get("event") == "page":
                last_page = int(e.get("page") or last_page)
                if e.get("total_pages") is not None:
                    total_pages = e.get("total_pages")
            elif e.get("event") == "start" and e.get("total_pages") is not None:
                total_pages = e.get("total_pages")
        return {
            "stage": "parse",
            "status": "running",
            "page": last_page,
            "total_pages": total_pages,
            "seconds": events[-1].get("t") if events else None,
            "label": (
                f"Parsing page {last_page}/{total_pages}"
                if total_pages
                else (f"Parsing page {last_page}" if last_page else "Parsing…")
            ),
        }

    if chunk_progress and str(chunk_progress.get("status") or "") == "running":
        return {
            "stage": "chunk",
            "status": "running",
            "strategy": chunk_progress.get("strategy"),
            "seconds": chunk_progress.get("seconds"),
            "label": f"Chunking ({chunk_progress.get('strategy') or '…'})",
        }
    if has_chunk_start and not has_chunk_done:
        strategy = None
        for e in events:
            if e.get("stage") == "chunk" and e.get("event") == "start":
                strategy = e.get("strategy")
        return {
            "stage": "chunk",
            "status": "running",
            "strategy": strategy,
            "seconds": events[-1].get("t") if events else None,
            "label": f"Chunking ({strategy or '…'})",
        }
    return None


def _master_turn_timings(
    events: List[Dict[str, Any]],
    llm_pairs: List[Dict[str, Any]],
    total: float,
    search_calls: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    master_requests = [
        ev
        for ev in events
        if ev.get("stage") == "agent"
        and ev.get("event") == "llm_request"
        and str(ev.get("agent") or "master") == "master"
    ]
    if not master_requests:
        return []

    llm_by_step = {
        int(p["step"]): p
        for p in llm_pairs
        if p.get("agent") == "master"
    }

    # Async search_pages jobs are attributed to the master step that started
    # them. Include their wall/model time so Master turn bars match the
    # SearchAgent calls section (request→next-request alone under-counts).
    search_by_step: Dict[int, List[Dict[str, Any]]] = {}
    for sc in search_calls or []:
        search_by_step.setdefault(int(sc.get("master_step") or 0), []).append(sc)

    rows: List[Dict[str, Any]] = []
    for i, req in enumerate(master_requests):
        step = int(req.get("step") or 0)
        start = float(req.get("t") or 0)
        if i + 1 < len(master_requests):
            end = float(master_requests[i + 1].get("t") or 0)
        else:
            end = total
        span = max(0.0, end - start)
        pair = llm_by_step.get(step) or {}
        llm = float(pair.get("llm_seconds") or 0)

        searches = search_by_step.get(step) or []
        # Parallel SearchAgents → wall = max; model time still sums.
        search_wall = max(
            (float(s.get("wall_seconds") or 0) for s in searches), default=0.0
        )
        search_llm = sum(float(s.get("llm_seconds") or 0) for s in searches)
        wall = max(span, search_wall)
        rows.append(
            {
                "step": step,
                "wall_seconds": _round_secs(wall),
                "span_seconds": _round_secs(span),
                "llm_seconds": _round_secs(llm),
                "search_wall_seconds": _round_secs(search_wall) if searches else None,
                "search_llm_seconds": _round_secs(search_llm) if searches else None,
                "n_search_calls": len(searches),
                "tool_seconds": _round_secs(max(0.0, wall - llm)),
                "pct": _pct(wall, total),
                "prompt_est_tokens": req.get("prompt_est_tokens"),
                "input_tokens": pair.get("input_tokens"),
                "output_tokens": pair.get("output_tokens"),
            }
        )
    return rows


def _search_call_timings(
    events: List[Dict[str, Any]],
    llm_pairs: List[Dict[str, Any]],
    total: float,
) -> List[Dict[str, Any]]:
    """One row per search_pages completion (SearchAgent invocation)."""
    sp_events = [
        ev for ev in events if ev.get("stage") == "agent" and ev.get("event") == "search_pages"
    ]
    if not sp_events:
        return []

    rows: List[Dict[str, Any]] = []
    # Per-key lower bound so parallel SearchAgents (different keys) do not
    # steal each other's LLM windows. Same-key follow-ups still stay sequential.
    prev_end_by_key: Dict[str, float] = {}
    for spe in sp_events:
        end_t = float(spe.get("t") or 0)
        key = str(spe.get("key") or "")
        master_step = int(spe.get("step") or 0)
        t_lo = float(prev_end_by_key.get(key, -1.0))

        window_pairs = [
            p
            for p in llm_pairs
            if p.get("agent") == "search"
            and p.get("key") == key
            and t_lo < float(p.get("start_t") or 0) <= end_t
        ]
        window_pairs.sort(key=lambda p: float(p.get("start_t") or 0))

        start_t = end_t
        if window_pairs:
            start_t = min(float(p["start_t"]) for p in window_pairs)

        # Labels for turns inside this search_pages window.
        label_by_step: Dict[int, str] = {}
        for ev in events:
            if ev.get("stage") != "agent" or ev.get("event") != "step":
                continue
            t = float(ev.get("t") or 0)
            if t <= t_lo or t > end_t:
                continue
            label = str(ev.get("label") or "")
            if not _is_search_step_label(label):
                continue
            # Prefer labels that belong to this key when parallel agents interleave.
            parsed = _parse_timing_search_label(label)
            if parsed is not None:
                _label_key_prefix, _sess, _turn, _mstep = parsed
                # Soft filter: if master_step encoded, require match when present.
                if _mstep and _mstep != master_step:
                    continue
            label_by_step[int(ev.get("step") or 0)] = label

        llm_total = sum(float(p.get("llm_seconds") or 0) for p in window_pairs)
        wall = max(0.0, end_t - start_t) if window_pairs else 0.0

        turns: List[Dict[str, Any]] = []
        for p in window_pairs:
            turn_step = int(p.get("step") or 0)
            label = label_by_step.get(turn_step, "")
            session_index = 1
            search_turn = turn_step
            parsed = _parse_timing_search_label(label) if label else None
            if parsed is not None:
                _prefix, session_index, search_turn, _mstep = parsed
            else:
                m = re.match(r"^search_(.+)_s(\d+)_s(\d+)$", label)
                if m:
                    session_index = int(m.group(2))
                    search_turn = int(m.group(3))
                else:
                    m = re.match(r"^search_(.+)_s(\d+)$", label)
                    if m:
                        search_turn = int(m.group(2))

            turns.append(
                {
                    "step": turn_step,
                    "search_session": session_index,
                    "search_turn": search_turn,
                    "label": label or None,
                    "llm_seconds": p.get("llm_seconds"),
                    "prompt_est_tokens": p.get("prompt_est_tokens"),
                    "input_tokens": p.get("input_tokens"),
                    "output_tokens": p.get("output_tokens"),
                }
            )

        sessions: Dict[int, List[Dict[str, Any]]] = {}
        for turn in turns:
            sess = int(turn.get("search_session") or 1)
            sessions.setdefault(sess, []).append(turn)

        session_rows = [
            {
                "session_index": sess,
                "llm_seconds": _round_secs(
                    sum(float(t.get("llm_seconds") or 0) for t in sess_turns)
                ),
                "n_turns": len(sess_turns),
                "turns": sess_turns,
            }
            for sess, sess_turns in sorted(sessions.items())
        ]

        rows.append(
            {
                "master_step": master_step,
                "key": key,
                "status": "complete",
                "wall_seconds": _round_secs(wall),
                "llm_seconds": _round_secs(llm_total),
                "overhead_seconds": _round_secs(max(0.0, wall - llm_total)),
                "pct": _pct(wall, total),
                "n_turns": len(turns),
                "sessions": session_rows,
                "turns": turns,
            }
        )
        prev_end_by_key[key] = end_t
    return rows


def _latest_search_session_for_key(
    events: List[Dict[str, Any]],
    *,
    key: str,
    t_lo: float,
    t_hi: float,
) -> Tuple[int, int, int]:
    """
    Best-effort (session_index, last_completed_turn, master_step) for a key.

    Session comes from the newest search label in the window (step/tool/message).
    Turn comes from the newest completed ``_sN_sM`` step label.
    """
    want = _safe_key_prefix(key)
    session_index = 1
    last_turn = 0
    master_step = 0
    best_t = -1.0
    for ev in events:
        if ev.get("stage") != "agent":
            continue
        event = ev.get("event")
        if event not in {"step", "tool", "message"}:
            continue
        t = float(ev.get("t") or 0)
        if t <= t_lo or t > t_hi:
            continue
        raw = str(ev.get("label") or ev.get("kind") or "")
        if not _is_search_step_label(raw):
            continue
        label = re.sub(r"_(?:assistant(?:_tool_calls)?|tool|user_nudge|user)$", "", raw)

        full = re.match(
            r"^m(\d+)(?:_t\d+_k\d+)?_search_(.+)_s(\d+)_s(\d+)$", label
        )
        sess_only = None
        if full:
            prefix = full.group(2)
            sess = int(full.group(3))
            turn = int(full.group(4))
            mstep = int(full.group(1))
        else:
            sess_only = re.match(
                r"^m(\d+)(?:_t\d+_k\d+)?_search_(.+)_s(\d+)$", label
            )
            if not sess_only:
                legacy = _parse_timing_search_label(label)
                if legacy is None:
                    continue
                prefix, sess, turn, mstep = legacy
            else:
                prefix = sess_only.group(2)
                sess = int(sess_only.group(3))
                turn = 0
                mstep = int(sess_only.group(1))

        if want and prefix and not (
            prefix == want or want.startswith(prefix) or prefix.startswith(want)
        ):
            continue
        if t >= best_t:
            best_t = t
            session_index = sess
            if mstep:
                master_step = mstep
        if event == "step" and full is not None:
            last_turn = max(last_turn, turn)
    return session_index, last_turn, master_step


def _build_search_call_row(
    *,
    master_step: int,
    key: str,
    status: str,
    window_pairs: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    t_lo: float,
    end_t: float,
    total: float,
    open_req: Optional[Dict[str, Any]] = None,
    current_session: Optional[int] = None,
    current_turn: Optional[int] = None,
) -> Dict[str, Any]:
    label_by_step: Dict[int, str] = {}
    for ev in events:
        if ev.get("stage") != "agent" or ev.get("event") != "step":
            continue
        t = float(ev.get("t") or 0)
        if t <= t_lo or t > end_t:
            continue
        label = str(ev.get("label") or "")
        if not _is_search_step_label(label):
            continue
        parsed = _parse_timing_search_label(label)
        if parsed is not None:
            _label_key_prefix, _sess, _turn, _mstep = parsed
            if _mstep and master_step and _mstep != master_step:
                continue
        label_by_step[int(ev.get("step") or 0)] = label

    turns: List[Dict[str, Any]] = []
    for p in window_pairs:
        turn_step = int(p.get("step") or 0)
        label = label_by_step.get(turn_step, "")
        session_index = 1
        search_turn = turn_step
        parsed = _parse_timing_search_label(label) if label else None
        if parsed is not None:
            _prefix, session_index, search_turn, _mstep = parsed
        else:
            m = re.match(r"^search_(.+)_s(\d+)_s(\d+)$", label)
            if m:
                session_index = int(m.group(2))
                search_turn = int(m.group(3))
            else:
                m = re.match(r"^search_(.+)_s(\d+)$", label)
                if m:
                    search_turn = int(m.group(2))
        turns.append(
            {
                "step": turn_step,
                "search_session": session_index,
                "search_turn": search_turn,
                "label": label or None,
                "status": "complete",
                "llm_seconds": p.get("llm_seconds"),
                "prompt_est_tokens": p.get("prompt_est_tokens"),
                "input_tokens": p.get("input_tokens"),
                "output_tokens": p.get("output_tokens"),
            }
        )

    if open_req is not None:
        turn_step = int(open_req.get("step") or 0)
        sess = int(current_session or 1)
        turns.append(
            {
                "step": turn_step,
                "search_session": sess,
                "search_turn": int(current_turn or turn_step),
                "label": None,
                "status": "running",
                "llm_seconds": _round_secs(
                    max(0.0, end_t - float(open_req.get("start_t") or end_t))
                ),
                "prompt_est_tokens": open_req.get("prompt_est_tokens"),
                "input_tokens": None,
                "output_tokens": None,
            }
        )

    sessions: Dict[int, List[Dict[str, Any]]] = {}
    for turn in turns:
        sess = int(turn.get("search_session") or 1)
        sessions.setdefault(sess, []).append(turn)

    session_rows = [
        {
            "session_index": sess,
            "llm_seconds": _round_secs(
                sum(float(t.get("llm_seconds") or 0) for t in sess_turns)
            ),
            "n_turns": len(sess_turns),
            "turns": sess_turns,
        }
        for sess, sess_turns in sorted(sessions.items())
    ]

    start_t = end_t
    if window_pairs:
        start_t = min(float(p["start_t"]) for p in window_pairs)
    elif open_req is not None:
        start_t = float(open_req.get("start_t") or end_t)
    llm_total = sum(float(p.get("llm_seconds") or 0) for p in window_pairs)
    if open_req is not None and turns and turns[-1].get("status") == "running":
        llm_total += float(turns[-1].get("llm_seconds") or 0)
    wall = max(0.0, end_t - start_t) if (window_pairs or open_req) else 0.0

    row: Dict[str, Any] = {
        "master_step": master_step,
        "key": key,
        "status": status,
        "wall_seconds": _round_secs(wall),
        "llm_seconds": _round_secs(llm_total),
        "overhead_seconds": _round_secs(max(0.0, wall - llm_total)),
        "pct": _pct(wall, total) if status == "complete" else None,
        "n_turns": len(turns),
        "sessions": session_rows,
        "turns": turns,
    }
    if status == "running":
        row["current_session"] = int(current_session or 1)
        row["current_turn"] = int(current_turn or (open_req or {}).get("step") or 0)
        if open_req is not None:
            row["phase"] = "llm"
        elif window_pairs:
            row["phase"] = "tools"
        else:
            row["phase"] = "starting"
    return row


def _in_progress_search_calls(
    events: List[Dict[str, Any]],
    llm_pairs: List[Dict[str, Any]],
    open_reqs: List[Dict[str, Any]],
    total: float,
) -> List[Dict[str, Any]]:
    """SearchAgent jobs that have started but not yet emitted search_pages."""
    last_complete_t: Dict[str, float] = {}
    for ev in events:
        if ev.get("stage") == "agent" and ev.get("event") == "search_pages":
            key = str(ev.get("key") or "")
            last_complete_t[key] = max(
                last_complete_t.get(key, -1.0), float(ev.get("t") or 0)
            )

    # Keys kicked off via search_pages_start after last completion.
    started: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        if ev.get("stage") != "agent" or ev.get("event") not in {
            "search_pages_start",
            "search_pages_enqueue",
        }:
            continue
        t = float(ev.get("t") or 0)
        master_step = int(ev.get("step") or 0)
        for key in ev.get("keys_started") or []:
            key_s = str(key)
            if t > last_complete_t.get(key_s, -1.0):
                started[key_s] = {"master_step": master_step, "start_t": t}

    # Any search LLM activity after last completion also counts as in-progress.
    active_keys: set = set(started.keys())
    for p in llm_pairs:
        if p.get("agent") != "search":
            continue
        key = str(p.get("key") or "")
        if float(p.get("start_t") or 0) > last_complete_t.get(key, -1.0):
            active_keys.add(key)
    for req in open_reqs:
        if req.get("agent") != "search":
            continue
        key = str(req.get("key") or "")
        if float(req.get("start_t") or 0) > last_complete_t.get(key, -1.0):
            active_keys.add(key)

    if not active_keys:
        return []

    now_t = total
    if events:
        now_t = max(total, float(events[-1].get("t") or 0))

    open_by_key: Dict[str, Dict[str, Any]] = {}
    for req in open_reqs:
        if req.get("agent") != "search":
            continue
        key = str(req.get("key") or "")
        if key not in active_keys:
            continue
        prev = open_by_key.get(key)
        if prev is None or float(req.get("start_t") or 0) >= float(
            prev.get("start_t") or 0
        ):
            open_by_key[key] = req

    rows: List[Dict[str, Any]] = []
    for key in sorted(active_keys):
        t_lo = float(last_complete_t.get(key, -1.0))
        window_pairs = [
            p
            for p in llm_pairs
            if p.get("agent") == "search"
            and str(p.get("key") or "") == key
            and float(p.get("start_t") or 0) > t_lo
        ]
        window_pairs.sort(key=lambda p: float(p.get("start_t") or 0))
        open_req = open_by_key.get(key)

        session_index, last_turn, label_master = _latest_search_session_for_key(
            events, key=key, t_lo=t_lo, t_hi=now_t
        )
        master_step = int((started.get(key) or {}).get("master_step") or 0)
        if not master_step:
            master_step = label_master

        if open_req is not None:
            current_turn = int(open_req.get("step") or 0)
            current_session = session_index
        elif window_pairs:
            current_turn = int(window_pairs[-1].get("step") or last_turn or 0)
            current_session = session_index
        else:
            current_turn = 0
            current_session = session_index

        rows.append(
            _build_search_call_row(
                master_step=master_step,
                key=key,
                status="running",
                window_pairs=window_pairs,
                events=events,
                t_lo=t_lo,
                end_t=now_t,
                total=total,
                open_req=open_req,
                current_session=current_session,
                current_turn=current_turn,
            )
        )
    return rows


def _is_search_step_label(label: str) -> bool:
    if not label:
        return False
    return bool(re.search(r"(?:^|_)search_", label))


def _parse_timing_search_label(
    label: str,
) -> Optional[Tuple[str, int, int, int]]:
    """
    Parse search step labels.

    Returns (key_prefix, session_index, turn, master_step) or None.
    master_step is 0 when the legacy search_* form has no master index.
    """
    m = re.match(r"^m(\d+)(?:_t\d+_k\d+)?_search_(.+)_s(\d+)_s(\d+)$", label)
    if m:
        return m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(1))
    m = re.match(r"^search_(.+)_s(\d+)_s(\d+)$", label)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), 0
    m = re.match(r"^(?:m\d+(?:_t\d+_k\d+)?)?_?search_(.+)_s(\d+)$", label)
    if m:
        return m.group(1), 1, int(m.group(2)), 0
    return None


def attach_timing_to_tree(
    tree: Dict[str, Any], timing: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge timing rows into hierarchy tree nodes."""
    master_by_step = {
        int(row["step"]): row for row in timing.get("master_turns") or []
    }

    search_calls = [
        row
        for row in (timing.get("search_calls") or [])
        if str(row.get("status") or "complete") != "running"
    ]
    search_queues: Dict[int, List[Dict[str, Any]]] = {}
    for row in search_calls:
        search_queues.setdefault(int(row.get("master_step") or 0), []).append(row)
    # Mutable global pool so collect_search_results on a later master turn can still
    # claim SearchAgent timings that were dumped under the start step.
    unmatched_search_calls: List[Dict[str, Any]] = list(search_calls)

    for mt in tree.get("master_turns") or []:
        step = int(mt.get("step") or 0)
        mt_timing = master_by_step.get(step)
        if mt_timing:
            mt["timing"] = mt_timing

        call_queue = list(search_queues.get(step) or [])
        for tool in mt.get("tools") or []:
            tname = tool.get("name")
            if tname not in {
                "search_pages",
                "collect_search_results",
                "await_searches",
                "search_pages_start",
            }:
                continue
            # Prefer same-step queue; fall back to unmatched global for async collect.
            active_queue = (
                unmatched_search_calls
                if tname in {"collect_search_results", "await_searches"}
                else call_queue
            )
            if not active_queue and tname not in {
                "search_pages",
                "search_pages_start",
            }:
                continue
            args_key = (tool.get("arguments") or {}).get("key")
            batch_keys: List[str] = []
            if isinstance(args_key, list):
                batch_keys = [str(k) for k in args_key if str(k).strip()]
            elif isinstance((tool.get("result") or {}).get("results"), list):
                batch_keys = [
                    str(x.get("key") or "")
                    for x in (tool.get("result") or {}).get("results")
                    if isinstance(x, dict) and x.get("key")
                ]
            elif isinstance((tool.get("result") or {}).get("completed"), list):
                batch_keys = [
                    str(x.get("key") or "")
                    for x in (tool.get("result") or {}).get("completed")
                    if isinstance(x, dict) and x.get("key")
                ]
            elif isinstance((tool.get("result") or {}).get("started"), list):
                batch_keys = [
                    str(x.get("key") or "")
                    for x in (tool.get("result") or {}).get("started")
                    if isinstance(x, dict) and x.get("key")
                ]

            for child in tool.get("children") or []:
                if child.get("type") != "search_agent":
                    continue

                if child.get("batch") and batch_keys:
                    # One timing row per key → attach onto session_index 1..N.
                    by_key: Dict[str, Dict[str, Any]] = {}
                    wall_sum = 0.0
                    llm_sum = 0.0
                    for bk in batch_keys:
                        matched = None
                        for idx, cand in enumerate(active_queue):
                            if str(cand.get("key") or "") == bk:
                                matched = active_queue.pop(idx)
                                break
                        if matched is None:
                            continue
                        by_key[bk] = matched
                        wall_sum = max(
                            wall_sum, float(matched.get("wall_seconds") or 0)
                        )
                        llm_sum += float(matched.get("llm_seconds") or 0)
                    child["timing"] = {
                        "wall_seconds": _round_secs(wall_sum),
                        "llm_seconds": _round_secs(llm_sum),
                        "overhead_seconds": _round_secs(
                            max(0.0, wall_sum - llm_sum)
                        ),
                        "pct": None,
                    }
                    for sess in child.get("sessions") or []:
                        st = by_key.get(str(sess.get("key") or ""))
                        if not st:
                            continue
                        sess["timing"] = {
                            "wall_seconds": st.get("wall_seconds"),
                            "llm_seconds": st.get("llm_seconds"),
                            "n_turns": st.get("n_turns"),
                        }
                        # Flattened turns: match by order within the key's LLM pairs.
                        src_turns = st.get("turns") or []
                        for ti, turn in enumerate(sess.get("turns") or []):
                            if ti < len(src_turns):
                                tt = src_turns[ti]
                                turn["timing"] = {
                                    "llm_seconds": tt.get("llm_seconds"),
                                    "input_tokens": tt.get("input_tokens"),
                                    "output_tokens": tt.get("output_tokens"),
                                    "prompt_est_tokens": tt.get("prompt_est_tokens"),
                                }
                    continue

                # Single-key search_pages (legacy / one key).
                tool_key = str(args_key or "") if not isinstance(args_key, list) else ""
                if not tool_key and isinstance((tool.get("result") or {}), dict):
                    one = (tool.get("result") or {}).get("key")
                    if one:
                        tool_key = str(one)
                if not tool_key:
                    completed = (tool.get("result") or {}).get("completed") or []
                    if (
                        isinstance(completed, list)
                        and completed
                        and isinstance(completed[0], dict)
                    ):
                        tool_key = str(completed[0].get("key") or "")
                call_timing = None
                if tool_key:
                    for idx, cand in enumerate(active_queue):
                        if str(cand.get("key") or "") == tool_key:
                            call_timing = active_queue.pop(idx)
                            break
                if call_timing is None and active_queue and tname == "search_pages":
                    call_timing = active_queue.pop(0)
                if call_timing is None:
                    continue
                child["timing"] = {
                    "wall_seconds": call_timing.get("wall_seconds"),
                    "llm_seconds": call_timing.get("llm_seconds"),
                    "overhead_seconds": call_timing.get("overhead_seconds"),
                    "pct": call_timing.get("pct"),
                }
                sessions_by_index = {
                    int(s.get("session_index") or 1): s
                    for s in call_timing.get("sessions") or []
                }
                for sess in child.get("sessions") or []:
                    sess_idx = int(sess.get("session_index") or 1)
                    st = sessions_by_index.get(sess_idx)
                    if not st:
                        continue
                    sess["timing"] = {
                        "llm_seconds": st.get("llm_seconds"),
                        "n_turns": st.get("n_turns"),
                    }
                    turns_by_key = {
                        (
                            int(t.get("search_turn") or t.get("step") or 0),
                            int(t.get("step") or 0),
                        ): t
                        for t in st.get("turns") or []
                    }
                    for turn in sess.get("turns") or []:
                        tk = (
                            int(turn.get("search_turn") or turn.get("step") or 0),
                            int(turn.get("step") or 0),
                        )
                        tt = turns_by_key.get(tk)
                        if tt:
                            turn["timing"] = {
                                "llm_seconds": tt.get("llm_seconds"),
                                "input_tokens": tt.get("input_tokens"),
                                "output_tokens": tt.get("output_tokens"),
                                "prompt_est_tokens": tt.get("prompt_est_tokens"),
                            }

    tree["timing"] = timing
    return tree


def build_timing_report(run_dir: Path) -> Dict[str, Any]:
    events = _read_timeline(run_dir)
    total = float(events[-1].get("t") or 0) if events else 0.0
    llm_pairs, open_reqs = _pair_llm_events(events)

    completed = _search_call_timings(events, llm_pairs, total)
    running = _in_progress_search_calls(events, llm_pairs, open_reqs, total)
    # Running first so the Timing UI surfaces live work above completed history.
    search_calls = list(running) + list(completed)

    master_llm = sum(float(p.get("llm_seconds") or 0) for p in llm_pairs if p.get("agent") == "master")
    search_llm = sum(float(p.get("llm_seconds") or 0) for p in llm_pairs if p.get("agent") == "search")

    return {
        "total_seconds": _round_secs(total),
        "stages": _stage_timings(events, total),
        "pipeline_progress": _pipeline_progress(run_dir, events),
        "master_turns": _master_turn_timings(
            events, llm_pairs, total, search_calls=search_calls
        ),
        "search_calls": search_calls,
        "active_searches": running,
        "summary": {
            "master_llm_seconds": _round_secs(master_llm),
            "search_llm_seconds": _round_secs(search_llm),
            "master_turns": len(
                [e for e in events if e.get("event") == "llm_request" and str(e.get("agent") or "master") == "master"]
            ),
            "search_llm_calls": len([p for p in llm_pairs if p.get("agent") == "search"]),
            "search_page_calls": len(
                [e for e in events if e.get("event") == "search_pages"]
            ),
            "active_searches": len(running),
        },
    }
