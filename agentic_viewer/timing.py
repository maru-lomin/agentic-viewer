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


def _safe_key_batch_fragment(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(key or ""))[:24]


def _strip_batch_n_suffix(prefix: str) -> str:
    return re.sub(r"_n\d+$", "", str(prefix or ""))


def _key_prefix_matches(label_prefix: str, key: str) -> bool:
    """True when a parsed label key_prefix belongs to ``key``."""
    want = _safe_key_prefix(key)
    prefix = _strip_batch_n_suffix(label_prefix)
    if not want:
        return not prefix
    if not prefix:
        return False
    if prefix == want or want.startswith(prefix) or prefix.startswith(want):
        return True
    want24 = _safe_key_batch_fragment(key)
    if not want24:
        return False
    padded = f"_{prefix}_"
    return (
        f"_{want24}_" in padded
        or prefix.startswith(want24 + "_")
        or prefix.endswith("_" + want24)
        or prefix == want24
    )


def _collect_search_step_labels(
    events: List[Dict[str, Any]],
    *,
    key: str = "",
    t_lo: float,
    t_hi: float,
    master_step: int = 0,
    tool_index: Optional[int] = None,
    keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Chronological step-dump labels for a search call in ``(t_lo, t_hi]``.

    Prefer ``master_step`` + ``tool_index`` (shared multi-key batch). Fall back to
    matching any of ``keys`` / ``key`` against the label prefix.
    """
    key_list = [str(k) for k in (keys or []) if str(k).strip()]
    if key and key not in key_list:
        key_list.append(key)

    rows: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("stage") != "agent" or ev.get("event") != "step":
            continue
        t = float(ev.get("t") or 0)
        if t <= t_lo or t > t_hi:
            continue
        label = str(ev.get("label") or "")
        if not _is_search_step_label(label):
            continue
        parsed_mt = _parse_label_master_tool(label)
        if parsed_mt is not None:
            mstep, tidx, label_prefix, session_index, search_turn = parsed_mt
            if master_step and mstep and mstep != master_step:
                continue
            if tool_index is not None and tidx != int(tool_index):
                continue
            if tool_index is None and key_list:
                if not any(_key_prefix_matches(label_prefix, k) for k in key_list):
                    continue
        else:
            parsed = _parse_timing_search_label(label)
            if parsed is None:
                continue
            label_prefix, session_index, search_turn, mstep = parsed
            if mstep and master_step and mstep != master_step:
                continue
            if key_list and not any(
                _key_prefix_matches(label_prefix, k) for k in key_list
            ):
                continue
        rows.append(
            {
                "t": t,
                "step": int(ev.get("step") or 0),
                "search_session": int(session_index),
                "search_turn": int(search_turn),
                "label": label,
            }
        )
    rows.sort(key=lambda r: (float(r["t"]), int(r["step"]), int(r["search_session"])))
    return rows


def _parse_label_master_tool(
    label: str,
) -> Optional[Tuple[int, int, str, int, int]]:
    """Parse ``m{step}_t{tool}_search_{prefix}_s{sess}_s{turn}``."""
    m = re.match(
        r"^m(\d+)_t(\d+)(?:_k\d+)?_search_(.+)_s(\d+)_s(\d+)$",
        str(label or ""),
    )
    if not m:
        return None
    return (
        int(m.group(1)),
        int(m.group(2)),
        m.group(3),
        int(m.group(4)),
        int(m.group(5)),
    )


def _annotate_turns_from_labels(
    window_pairs: List[Dict[str, Any]],
    step_labels: List[Dict[str, Any]],
    *,
    status: str = "complete",
) -> List[Dict[str, Any]]:
    """
    Attach session/turn/label to each LLM pair.

    Matches each pair to the earliest unused step dump with the same ``step``
    whose timestamp is at/after the LLM request start. This survives handoff
    sessions that reset the per-session step counter to 1.
    """
    unused = list(step_labels)
    turns: List[Dict[str, Any]] = []
    for p in window_pairs:
        turn_step = int(p.get("step") or 0)
        start_t = float(p.get("start_t") or 0)
        matched: Optional[Dict[str, Any]] = None
        for i, se in enumerate(unused):
            if int(se.get("step") or 0) != turn_step:
                continue
            # Step dumps are written at/after the LLM response.
            if float(se.get("t") or 0) < start_t - 0.05:
                continue
            matched = unused.pop(i)
            break

        if matched is not None:
            session_index = int(matched.get("search_session") or 1)
            search_turn = int(matched.get("search_turn") or turn_step)
            label = matched.get("label") or None
        else:
            session_index = 1
            search_turn = turn_step
            label = None

        turns.append(
            {
                "step": turn_step,
                "search_session": session_index,
                "search_turn": search_turn,
                "label": label,
                "status": status,
                "llm_seconds": p.get("llm_seconds"),
                "prompt_est_tokens": p.get("prompt_est_tokens"),
                "input_tokens": p.get("input_tokens"),
                "output_tokens": p.get("output_tokens"),
                "start_t": p.get("start_t"),
            }
        )
    return turns


def _group_turns_into_sessions(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group turns by session; sort sessions and turns numerically."""
    ordered = sorted(
        turns,
        key=lambda t: (
            int(t.get("search_session") or 1),
            int(t.get("search_turn") or 0),
            float(t.get("start_t") or 0),
        ),
    )
    sessions: Dict[int, List[Dict[str, Any]]] = {}
    for turn in ordered:
        sess = int(turn.get("search_session") or 1)
        sessions.setdefault(sess, []).append(turn)

    return [
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
    """
    One row per SearchAgent invocation (shared multi-key session).

    Groups ``search_pages_enqueue`` batches with their per-key ``search_pages``
    completions. Wall/model/turns are session-scoped; key outcomes hang off the
    row without duplicating time.
    """
    enqueues = [
        ev
        for ev in events
        if ev.get("stage") == "agent" and ev.get("event") == "search_pages_enqueue"
    ]
    if not enqueues:
        return _search_call_timings_legacy_per_key(events, llm_pairs, total)

    completions = [
        ev
        for ev in events
        if ev.get("stage") == "agent" and ev.get("event") == "search_pages"
    ]
    comps_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for ev in completions:
        comps_by_key.setdefault(str(ev.get("key") or ""), []).append(ev)

    used_comp_ids: set[int] = set()
    rows: List[Dict[str, Any]] = []
    for enq in enqueues:
        master_step = int(enq.get("step") or 0)
        tool_index = int(enq.get("tool_index") or 0)
        keys = [str(k) for k in (enq.get("keys_accepted") or []) if str(k).strip()]
        if not keys:
            continue
        t_enq = float(enq.get("t") or 0)

        matched_evs: List[Optional[Dict[str, Any]]] = []
        for key in keys:
            matched = None
            for ev in comps_by_key.get(key) or []:
                if id(ev) in used_comp_ids:
                    continue
                if float(ev.get("t") or 0) <= t_enq:
                    continue
                matched = ev
                break
            matched_evs.append(matched)

        # Incomplete batches belong in the running section.
        if any(ev is None for ev in matched_evs):
            continue

        key_outcomes: List[Dict[str, Any]] = []
        end_t = t_enq
        for key, matched in zip(keys, matched_evs):
            assert matched is not None
            used_comp_ids.add(id(matched))
            end_t = max(end_t, float(matched.get("t") or 0))
            key_outcomes.append(
                {
                    "key": key,
                    "status": str(matched.get("status") or "complete"),
                    "n_pages": int(matched.get("n_pages") or 0),
                }
            )

        rows.append(
            _build_batch_search_call_row(
                events=events,
                llm_pairs=llm_pairs,
                master_step=master_step,
                tool_index=tool_index,
                keys=key_outcomes,
                t_lo=t_enq,
                end_t=end_t,
                total=total,
                status="complete",
            )
        )

    # Legacy orphan completions not covered by any enqueue batch.
    for ev in completions:
        if id(ev) in used_comp_ids:
            continue
        key = str(ev.get("key") or "")
        if not key:
            continue
        end_t = float(ev.get("t") or 0)
        rows.append(
            _build_batch_search_call_row(
                events=events,
                llm_pairs=llm_pairs,
                master_step=int(ev.get("step") or 0),
                tool_index=None,
                keys=[
                    {
                        "key": key,
                        "status": str(ev.get("status") or "complete"),
                        "n_pages": int(ev.get("n_pages") or 0),
                    }
                ],
                t_lo=-1.0,
                end_t=end_t,
                total=total,
                status="complete",
            )
        )
    return rows


def _search_call_timings_legacy_per_key(
    events: List[Dict[str, Any]],
    llm_pairs: List[Dict[str, Any]],
    total: float,
) -> List[Dict[str, Any]]:
    """Pre-batch traces: one row per search_pages completion."""
    sp_events = [
        ev
        for ev in events
        if ev.get("stage") == "agent" and ev.get("event") == "search_pages"
    ]
    if not sp_events:
        return []

    rows: List[Dict[str, Any]] = []
    prev_end_by_key: Dict[str, float] = {}
    for spe in sp_events:
        end_t = float(spe.get("t") or 0)
        key = str(spe.get("key") or "")
        master_step = int(spe.get("step") or 0)
        t_lo = float(prev_end_by_key.get(key, -1.0))
        rows.append(
            _build_batch_search_call_row(
                events=events,
                llm_pairs=llm_pairs,
                master_step=master_step,
                tool_index=None,
                keys=[
                    {
                        "key": key,
                        "status": str(spe.get("status") or "complete"),
                        "n_pages": int(spe.get("n_pages") or 0),
                    }
                ],
                t_lo=t_lo,
                end_t=end_t,
                total=total,
                status="complete",
            )
        )
        prev_end_by_key[key] = end_t
    return rows


def _build_batch_search_call_row(
    *,
    events: List[Dict[str, Any]],
    llm_pairs: List[Dict[str, Any]],
    master_step: int,
    tool_index: Optional[int],
    keys: List[Dict[str, Any]],
    t_lo: float,
    end_t: float,
    total: float,
    status: str,
    open_req: Optional[Dict[str, Any]] = None,
    current_session: Optional[int] = None,
    current_turn: Optional[int] = None,
) -> Dict[str, Any]:
    """Shared-session timing row with per-key outcome list."""
    key_names = [str(k.get("key") or "") for k in keys if k.get("key")]
    key_set = set(key_names)
    primary = key_names[0] if key_names else ""

    window_pairs = [
        p
        for p in llm_pairs
        if p.get("agent") == "search"
        and str(p.get("key") or "") in key_set
        and t_lo < float(p.get("start_t") or 0) <= end_t
    ]
    window_pairs.sort(key=lambda p: float(p.get("start_t") or 0))

    step_labels = _collect_search_step_labels(
        events,
        key=primary,
        keys=key_names,
        t_lo=t_lo,
        t_hi=end_t,
        master_step=master_step,
        tool_index=tool_index,
    )
    turns = _annotate_turns_from_labels(
        window_pairs,
        step_labels,
        status="complete" if status == "complete" else status,
    )

    running_llm = 0.0
    if open_req is not None:
        turn_step = int(open_req.get("step") or 0)
        sess = int(current_session or 1)
        search_turn = int(current_turn or turn_step)
        running_llm = float(
            _round_secs(max(0.0, end_t - float(open_req.get("start_t") or end_t)))
            or 0.0
        )
        turns.append(
            {
                "step": turn_step,
                "search_session": sess,
                "search_turn": search_turn,
                "label": None,
                "status": "running",
                "llm_seconds": running_llm,
                "prompt_est_tokens": open_req.get("prompt_est_tokens"),
                "input_tokens": None,
                "output_tokens": None,
                "start_t": open_req.get("start_t"),
            }
        )

    session_rows = _group_turns_into_sessions(turns)
    turns = [t for s in session_rows for t in (s.get("turns") or [])]

    start_t = end_t
    if window_pairs:
        start_t = min(float(p["start_t"]) for p in window_pairs)
    elif open_req is not None:
        start_t = float(open_req.get("start_t") or end_t)

    llm_total = sum(float(p.get("llm_seconds") or 0) for p in window_pairs) + running_llm
    wall = max(0.0, end_t - start_t) if (window_pairs or open_req) else 0.0

    n_keys = len(key_names)
    label = primary if n_keys <= 1 else f"{n_keys} keys (shared)"
    row: Dict[str, Any] = {
        "master_step": master_step,
        "tool_index": tool_index,
        "key": primary,
        "keys": keys,
        "n_keys": n_keys,
        "label": label,
        "shared": n_keys > 1,
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
        row["current_turn"] = int(
            current_turn or (open_req or {}).get("step") or 0
        )
        if open_req is not None:
            row["phase"] = "llm"
        elif window_pairs:
            row["phase"] = "tools"
        else:
            row["phase"] = "starting"
    return row


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
            r"^m(\d+)(?:_t\d+(?:_k\d+)?)?_search_(.+)_s(\d+)_s(\d+)$", label
        )
        sess_only = None
        if full:
            prefix = full.group(2)
            sess = int(full.group(3))
            turn = int(full.group(4))
            mstep = int(full.group(1))
        else:
            sess_only = re.match(
                r"^m(\d+)(?:_t\d+(?:_k\d+)?)?_search_(.+)_s(\d+)$", label
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

        if not _key_prefix_matches(prefix, key):
            continue
        if t >= best_t:
            best_t = t
            session_index = sess
            if mstep:
                master_step = mstep
        if event == "step" and full is not None:
            last_turn = max(last_turn, turn)
    return session_index, last_turn, master_step


def _in_progress_search_calls(
    events: List[Dict[str, Any]],
    llm_pairs: List[Dict[str, Any]],
    open_reqs: List[Dict[str, Any]],
    total: float,
) -> List[Dict[str, Any]]:
    """SearchAgent batches that have started but not finished every key."""
    enqueues = [
        ev
        for ev in events
        if ev.get("stage") == "agent" and ev.get("event") == "search_pages_enqueue"
    ]
    completions = [
        ev
        for ev in events
        if ev.get("stage") == "agent" and ev.get("event") == "search_pages"
    ]
    comps_by_key: Dict[str, List[Dict[str, Any]]] = {}
    for ev in completions:
        comps_by_key.setdefault(str(ev.get("key") or ""), []).append(ev)

    now_t = total
    if events:
        now_t = max(total, float(events[-1].get("t") or 0))

    open_by_key: Dict[str, Dict[str, Any]] = {}
    for req in open_reqs:
        if req.get("agent") != "search":
            continue
        key = str(req.get("key") or "")
        prev = open_by_key.get(key)
        if prev is None or float(req.get("start_t") or 0) >= float(
            prev.get("start_t") or 0
        ):
            open_by_key[key] = req

    rows: List[Dict[str, Any]] = []
    if enqueues:
        for enq in enqueues:
            master_step = int(enq.get("step") or 0)
            tool_index = int(enq.get("tool_index") or 0)
            keys = [str(k) for k in (enq.get("keys_accepted") or []) if str(k).strip()]
            if not keys:
                continue
            t_enq = float(enq.get("t") or 0)
            key_outcomes: List[Dict[str, Any]] = []
            pending = False
            for key in keys:
                hit = None
                for ev in comps_by_key.get(key) or []:
                    if float(ev.get("t") or 0) <= t_enq:
                        continue
                    hit = ev
                    break
                if hit is None:
                    pending = True
                    key_outcomes.append(
                        {"key": key, "status": "pending", "n_pages": 0}
                    )
                else:
                    key_outcomes.append(
                        {
                            "key": key,
                            "status": str(hit.get("status") or "complete"),
                            "n_pages": int(hit.get("n_pages") or 0),
                        }
                    )
            if not pending:
                continue

            primary = keys[0]
            open_req = open_by_key.get(primary)
            session_index, last_turn, _ = _latest_search_session_for_key(
                events, key=primary, t_lo=t_enq, t_hi=now_t
            )
            if open_req is not None:
                current_turn = int(open_req.get("step") or 0)
                current_session = session_index
            else:
                current_turn = last_turn
                current_session = session_index

            rows.append(
                _build_batch_search_call_row(
                    events=events,
                    llm_pairs=llm_pairs,
                    master_step=master_step,
                    tool_index=tool_index,
                    keys=key_outcomes,
                    t_lo=t_enq,
                    end_t=now_t,
                    total=total,
                    status="running",
                    open_req=open_req,
                    current_session=current_session,
                    current_turn=current_turn,
                )
            )
        return rows

    # Legacy: no enqueue events — fall back to per-key running detection.
    last_complete_t: Dict[str, float] = {}
    for ev in completions:
        key = str(ev.get("key") or "")
        last_complete_t[key] = max(
            last_complete_t.get(key, -1.0), float(ev.get("t") or 0)
        )
    active_keys: set = set()
    for ev in events:
        if ev.get("stage") != "agent" or ev.get("event") not in {
            "search_pages_start",
            "search_pages_enqueue",
        }:
            continue
        t = float(ev.get("t") or 0)
        for key in ev.get("keys_started") or ev.get("keys_accepted") or []:
            key_s = str(key)
            if t > last_complete_t.get(key_s, -1.0):
                active_keys.add(key_s)
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

    for key in sorted(active_keys):
        t_lo = float(last_complete_t.get(key, -1.0))
        open_req = open_by_key.get(key)
        session_index, last_turn, label_master = _latest_search_session_for_key(
            events, key=key, t_lo=t_lo, t_hi=now_t
        )
        master_step = label_master
        current_turn = int(
            (open_req or {}).get("step") or last_turn or 0
        )
        rows.append(
            _build_batch_search_call_row(
                events=events,
                llm_pairs=llm_pairs,
                master_step=master_step,
                tool_index=None,
                keys=[{"key": key, "status": "pending", "n_pages": 0}],
                t_lo=t_lo,
                end_t=now_t,
                total=total,
                status="running",
                open_req=open_req,
                current_session=session_index,
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
    m = re.match(
        r"^m(\d+)(?:_t\d+(?:_k\d+)?)?_search_(.+)_s(\d+)_s(\d+)$", label
    )
    if m:
        return m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(1))
    m = re.match(r"^search_(.+)_s(\d+)_s(\d+)$", label)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), 0
    m = re.match(
        r"^(?:m\d+(?:_t\d+(?:_k\d+)?)?)?_?search_(.+)_s(\d+)$", label
    )
    if m:
        return m.group(1), 1, int(m.group(2)), 0
    return None


def _normalize_trace_label(label: Any) -> str:
    return str(label or "").strip()


def _timing_turn_payload(tt: Dict[str, Any]) -> Dict[str, Any]:
    llm = tt.get("llm_seconds")
    return {
        "llm_seconds": llm,
        "wall_seconds": llm,
        "input_tokens": tt.get("input_tokens"),
        "output_tokens": tt.get("output_tokens"),
        "prompt_est_tokens": tt.get("prompt_est_tokens"),
    }


def _merge_timing_turn_rows(matched_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe timing turns across matched search call rows (async collect)."""
    merged: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, str]] = set()
    for row in matched_rows:
        for tt in row.get("turns") or []:
            if not isinstance(tt, dict):
                continue
            ident = (
                int(tt.get("search_session") or 1),
                int(tt.get("search_turn") or tt.get("step") or 0),
                _normalize_trace_label(tt.get("label")),
            )
            if ident in seen:
                continue
            seen.add(ident)
            merged.append(dict(tt))
    merged.sort(
        key=lambda t: (
            int(t.get("search_session") or 1),
            int(t.get("search_turn") or t.get("step") or 0),
            float(t.get("start_t") or 0),
        )
    )
    return merged


def _lookup_timing_turn(
    turn: Dict[str, Any],
    src_turns: List[Dict[str, Any]],
    *,
    used: set[int],
) -> Optional[Dict[str, Any]]:
    """
    Match a hierarchy turn to a timing row by label, then session/turn, then step.
    """
    label = _normalize_trace_label(turn.get("label"))
    filename = _normalize_trace_label(turn.get("filename"))
    sess = int(
        turn.get("search_session")
        or turn.get("handoff_session_index")
        or 1
    )
    search_turn = int(turn.get("search_turn") or turn.get("step") or 0)
    step = int(turn.get("step") or 0)

    if label:
        for i, tt in enumerate(src_turns):
            if i in used:
                continue
            if _normalize_trace_label(tt.get("label")) == label:
                used.add(i)
                return tt
        for i, tt in enumerate(src_turns):
            if i in used:
                continue
            tl = _normalize_trace_label(tt.get("label"))
            if tl and tl in filename:
                used.add(i)
                return tt

    for i, tt in enumerate(src_turns):
        if i in used:
            continue
        tt_sess = int(tt.get("search_session") or 1)
        tt_turn = int(tt.get("search_turn") or tt.get("step") or 0)
        if tt_sess == sess and tt_turn == search_turn:
            used.add(i)
            return tt

    for i, tt in enumerate(src_turns):
        if i in used:
            continue
        if int(tt.get("search_session") or 1) == sess and int(tt.get("step") or 0) == step:
            used.add(i)
            return tt

    return None


def _attach_session_turn_timings(
    sess: Dict[str, Any],
    src_turns: List[Dict[str, Any]],
    *,
    shared: bool = False,
) -> None:
    """Attach per-turn LLM timings using label / search_turn keys."""
    used: set[int] = set()
    matched_llm = 0.0
    n_matched = 0
    for turn in sess.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        tt = _lookup_timing_turn(turn, src_turns, used=used)
        if tt is None:
            continue
        turn["timing"] = _timing_turn_payload(tt)
        matched_llm += float(tt.get("llm_seconds") or 0)
        n_matched += 1

    sess_timing = dict(sess.get("timing") or {})
    sess_timing.update(
        {
            "n_turns": len(sess.get("turns") or []),
            "n_turns_timed": n_matched,
            "shared": shared,
        }
    )
    if n_matched:
        sess_timing["llm_seconds"] = _round_secs(matched_llm)
    sess["timing"] = sess_timing


def _batch_keys_from_tool(tool: Dict[str, Any]) -> List[str]:
    args = tool.get("arguments") or {}
    raw = args.get("key")
    if raw is None:
        raw = args.get("keys")
    if isinstance(raw, list):
        return [str(k) for k in raw if str(k).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    result = tool.get("result") or {}
    for field in ("results", "completed", "started", "accepted"):
        rows = result.get(field)
        if not isinstance(rows, list):
            continue
        keys = [
            str(x.get("key") or x)
            for x in rows
            if (isinstance(x, dict) and x.get("key")) or str(x).strip()
        ]
        if keys:
            return keys
    return []


def _cand_row_keys(cand: Dict[str, Any]) -> set[str]:
    keys = {
        str(k.get("key") or k)
        for k in (cand.get("keys") or [])
        if (isinstance(k, dict) and k.get("key")) or str(k).strip()
    }
    if not keys and cand.get("key"):
        keys = {str(cand.get("key"))}
    return keys


def _partition_search_call_rows(
    active_queue: List[Dict[str, Any]],
    *,
    batch_set: set[str],
    master_step: int,
    tool_index: Optional[int],
    require_tool_index: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matched: List[Dict[str, Any]] = []
    remain: List[Dict[str, Any]] = []
    for cand in active_queue:
        if batch_set and not (_cand_row_keys(cand) & batch_set):
            remain.append(cand)
            continue
        if master_step and int(cand.get("master_step") or 0) not in (0, master_step):
            remain.append(cand)
            continue
        if require_tool_index and tool_index is not None:
            cand_idx = cand.get("tool_index")
            if cand_idx is not None and int(cand_idx) != int(tool_index):
                remain.append(cand)
                continue
        matched.append(cand)
    return matched, remain


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
        for ti, tool in enumerate(mt.get("tools") or []):
            tname = tool.get("name")
            if tname not in {
                "search_pages",
                "collect_search_results",
                "await_searches",
                "search_pages_start",
            }:
                continue
            tool_index = tool.get("tool_index")
            if tool_index is None:
                tool_index = ti
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
            batch_keys = _batch_keys_from_tool(tool)
            require_tool_index = tname in {"search_pages", "search_pages_start"}

            for child in tool.get("children") or []:
                if child.get("type") != "search_agent":
                    continue

                if child.get("batch") and batch_keys:
                    batch_set = {str(k) for k in batch_keys if str(k).strip()}
                    matched_rows, remain = _partition_search_call_rows(
                        active_queue,
                        batch_set=batch_set,
                        master_step=step,
                        tool_index=int(tool_index),
                        require_tool_index=require_tool_index,
                    )
                    active_queue[:] = remain
                    if not matched_rows:
                        continue
                    wall = max(
                        float(m.get("wall_seconds") or 0) for m in matched_rows
                    )
                    llm = sum(float(m.get("llm_seconds") or 0) for m in matched_rows)
                    child["timing"] = {
                        "wall_seconds": _round_secs(wall),
                        "llm_seconds": _round_secs(llm),
                        "overhead_seconds": _round_secs(max(0.0, wall - llm)),
                        "pct": matched_rows[0].get("pct")
                        if len(matched_rows) == 1
                        else None,
                        "n_keys": sum(int(m.get("n_keys") or 0) for m in matched_rows),
                        "shared": any(m.get("shared") for m in matched_rows)
                        or len(matched_rows) > 1,
                    }
                    src_turns = _merge_timing_turn_rows(matched_rows)
                    is_shared = bool(child["timing"].get("shared"))
                    for sess in child.get("sessions") or []:
                        _attach_session_turn_timings(
                            sess,
                            src_turns,
                            shared=is_shared,
                        )
                    continue

                # Single-key search_pages (legacy / one key).
                tool_key = batch_keys[0] if len(batch_keys) == 1 else ""
                call_timing = None
                if tool_key:
                    matched_rows, remain = _partition_search_call_rows(
                        active_queue,
                        batch_set={tool_key},
                        master_step=step if require_tool_index else 0,
                        tool_index=int(tool_index),
                        require_tool_index=require_tool_index,
                    )
                    if matched_rows:
                        call_timing = matched_rows[0]
                        if require_tool_index:
                            active_queue[:] = remain
                        else:
                            # Global collect pool: drop only the claimed row.
                            try:
                                unmatched_search_calls.remove(call_timing)
                            except ValueError:
                                pass
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
                src_turns = list(call_timing.get("turns") or [])
                sessions_by_index = {
                    int(s.get("session_index") or 1): s
                    for s in call_timing.get("sessions") or []
                }
                for sess in child.get("sessions") or []:
                    sess_idx = int(sess.get("session_index") or 1)
                    st = sessions_by_index.get(sess_idx)
                    session_turns = (
                        list(st.get("turns") or []) if st else src_turns
                    )
                    _attach_session_turn_timings(sess, session_turns, shared=False)

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
