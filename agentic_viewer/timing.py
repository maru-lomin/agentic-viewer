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


def _pair_llm_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair llm_request → llm_response events in timeline order."""
    pending: Dict[Tuple[str, str, int], float] = {}
    pairs: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("event") == "llm_request":
            pending[_agent_key(ev)] = float(ev.get("t") or 0)
            continue
        if ev.get("event") != "llm_response":
            continue
        k = _agent_key(ev)
        start = pending.pop(k, None)
        if start is None:
            continue
        end = float(ev.get("t") or 0)
        pairs.append(
            {
                "agent": k[0],
                "key": k[1],
                "step": k[2],
                "start_t": start,
                "end_t": end,
                "llm_seconds": _round_secs(end - start),
                "prompt_est_tokens": ev.get("prompt_est_tokens"),
                "input_tokens": ev.get("input_tokens"),
                "output_tokens": ev.get("output_tokens"),
            }
        )
    return pairs


def _stage_timings(events: List[Dict[str, Any]], total: float) -> List[Dict[str, Any]]:
    bounds: Dict[str, List[float]] = {}
    for ev in events:
        stage = str(ev.get("stage") or "")
        if not stage:
            continue
        t = float(ev.get("t") or 0)
        if stage not in bounds:
            bounds[stage] = [t, t]
        else:
            bounds[stage][0] = min(bounds[stage][0], t)
            bounds[stage][1] = max(bounds[stage][1], t)

    rows: List[Dict[str, Any]] = []
    for stage in ("parse", "chunk", "agent", "result"):
        if stage not in bounds:
            continue
        start, end = bounds[stage]
        seconds = max(0.0, end - start)
        rows.append(
            {
                "stage": stage,
                "start_t": _round_secs(start),
                "end_t": _round_secs(end),
                "seconds": _round_secs(seconds),
                "pct": _pct(seconds, total),
            }
        )
    return rows


def _master_turn_timings(
    events: List[Dict[str, Any]],
    llm_pairs: List[Dict[str, Any]],
    total: float,
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

    rows: List[Dict[str, Any]] = []
    for i, req in enumerate(master_requests):
        step = int(req.get("step") or 0)
        start = float(req.get("t") or 0)
        if i + 1 < len(master_requests):
            end = float(master_requests[i + 1].get("t") or 0)
        else:
            end = total
        wall = max(0.0, end - start)
        pair = llm_by_step.get(step) or {}
        llm = float(pair.get("llm_seconds") or 0)
        rows.append(
            {
                "step": step,
                "wall_seconds": _round_secs(wall),
                "llm_seconds": _round_secs(llm),
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

    search_calls = timing.get("search_calls") or []
    search_queues: Dict[int, List[Dict[str, Any]]] = {}
    for row in search_calls:
        search_queues.setdefault(int(row.get("master_step") or 0), []).append(row)

    for mt in tree.get("master_turns") or []:
        step = int(mt.get("step") or 0)
        mt_timing = master_by_step.get(step)
        if mt_timing:
            mt["timing"] = mt_timing

        call_queue = list(search_queues.get(step) or [])
        for tool in mt.get("tools") or []:
            if tool.get("name") != "search_pages" or not call_queue:
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
                        for idx, cand in enumerate(call_queue):
                            if str(cand.get("key") or "") == bk:
                                matched = call_queue.pop(idx)
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
                call_timing = None
                if tool_key:
                    for idx, cand in enumerate(call_queue):
                        if str(cand.get("key") or "") == tool_key:
                            call_timing = call_queue.pop(idx)
                            break
                if call_timing is None and call_queue:
                    call_timing = call_queue.pop(0)
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
    llm_pairs = _pair_llm_events(events)

    master_llm = sum(float(p.get("llm_seconds") or 0) for p in llm_pairs if p.get("agent") == "master")
    search_llm = sum(float(p.get("llm_seconds") or 0) for p in llm_pairs if p.get("agent") == "search")

    return {
        "total_seconds": _round_secs(total),
        "stages": _stage_timings(events, total),
        "master_turns": _master_turn_timings(events, llm_pairs, total),
        "search_calls": _search_call_timings(events, llm_pairs, total),
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
        },
    }
