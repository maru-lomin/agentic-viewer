"""Build hierarchical Master → SearchAgent tree from run trace dumps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(key or ""))[:40]


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_search_label(label: str) -> Tuple[str, int, int, int]:
    """
    Parse search step label into (key_prefix, session, turn, master_step).

    Supports:
      m{master}_t{tool}_k{keyidx}_search_{safe_key}_s{session}_s{turn}
      m{master}_search_{safe_key}_s{session}_s{turn}
      search_{safe_key}_s{session}_s{turn}
      search_{safe_key}_s{turn}

    master_step is 0 when the label has no mNNN prefix (legacy runs).
    """
    label = str(label or "")
    m = re.match(
        r"^m(\d+)(?:_t\d+_k\d+)?_search_(.+)_s(\d+)_s(\d+)$", label
    )
    if m:
        return m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(1))
    m = re.match(r"^search_(.+)_s(\d+)_s(\d+)$", label)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), 0
    m = re.match(r"^search_(.+)_s(\d+)$", label)
    if m:
        return m.group(1), 1, int(m.group(2)), 0
    return label.replace("search_", "", 1), 1, 0, 0


def _extract_submit_output(tool_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return submit_pages / no_relevant_pages output when this turn finalized search."""
    for tr in tool_results:
        name = tr.get("name")
        if name not in {"submit_pages", "no_relevant_pages"}:
            continue
        preview = tr.get("result_preview")
        parsed: Optional[Dict[str, Any]] = None
        if isinstance(preview, dict):
            parsed = dict(preview)
        elif isinstance(preview, str):
            try:
                loaded = json.loads(preview)
                if isinstance(loaded, dict):
                    parsed = loaded
            except json.JSONDecodeError:
                pass
        if parsed is None:
            args = tr.get("arguments")
            if isinstance(args, dict) and (
                args.get("pages") is not None or name == "no_relevant_pages"
            ):
                parsed = {
                    "pages": args.get("pages") or [],
                    "page_reasons": args.get("page_reasons") or {},
                    "reason": args.get("reason") or "",
                }
        if parsed is not None:
            parsed.setdefault("tool", name)
            return parsed
    return None


def _compact_step(step: Dict[str, Any], *, filename: str) -> Dict[str, Any]:
    assistant = step.get("assistant") or {}
    tool_calls = assistant.get("tool_calls") or []
    tool_results = step.get("tool_results") or []
    compact_tool_results = [
        {
            "name": tr.get("name"),
            "arguments": tr.get("arguments"),
            "result_preview": tr.get("result_preview"),
        }
        for tr in tool_results
        if isinstance(tr, dict)
    ]
    return {
        "filename": filename,
        "step": step.get("step"),
        "label": step.get("label"),
        "prompt_est_tokens": step.get("prompt_est_tokens"),
        "input_tokens": step.get("input_tokens"),
        "output_tokens": step.get("output_tokens"),
        "input_budget": step.get("input_budget"),
        "max_tokens": step.get("max_tokens"),
        "error": step.get("error"),
        "n_tool_calls": len(tool_calls),
        "assistant_content": assistant.get("content") or "",
        "tool_calls": [
            {
                "name": (tc.get("function") or {}).get("name"),
                "arguments": (tc.get("function") or {}).get("arguments"),
            }
            for tc in tool_calls
            if isinstance(tc, dict)
        ],
        "tool_results": compact_tool_results,
        "submit_output": _extract_submit_output(compact_tool_results),
        # Keep first user message for prior_context reconstruction (legacy runs).
        "first_user_content": _first_user_content(step.get("request_messages") or []),
    }


def _first_user_content(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else ""
    return ""


def _extract_prior_from_user(content: str) -> Optional[Dict[str, Any]]:
    """Parse prior_context JSON embedded in SearchAgent user prompt."""
    if not content or "Prior search session progress" not in content:
        return None
    marker = "Prior search session progress"
    idx = content.find(marker)
    brace = content.find("{", idx)
    if brace < 0:
        return None
    # Find matching JSON object.
    depth = 0
    end = None
    for i, ch in enumerate(content[brace:], start=brace):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        obj = json.loads(content[brace:end])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _load_steps(
    agent_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Return (master_steps, search_steps_by_prefix)."""
    master: List[Dict[str, Any]] = []
    search_by_prefix: Dict[str, List[Dict[str, Any]]] = {}

    if not agent_dir.is_dir():
        return master, search_by_prefix

    for path in sorted(agent_dir.glob("step_*.json")):
        data = _read_json(path) or {}
        name = path.name
        label = data.get("label")
        compact = _compact_step(data, filename=name)

        if label:
            key_prefix, session, turn, master_step = _parse_search_label(str(label))
            compact["search_key_prefix"] = key_prefix
            compact["search_session"] = session
            compact["search_turn"] = turn
            compact["master_step"] = master_step
            search_by_prefix.setdefault(key_prefix, []).append(compact)
        elif re.fullmatch(r"step_\d+\.json", name):
            master.append(compact)

    master.sort(key=lambda s: int(s.get("step") or 0))
    for rows in search_by_prefix.values():
        rows.sort(
            key=lambda s: (
                int(s.get("master_step") or 0),
                int(s.get("search_session") or 1),
                int(s.get("search_turn") or 0),
            )
        )
    return master, search_by_prefix


def _label_index(
    search_by_prefix: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rows in search_by_prefix.values():
        for row in rows:
            label = row.get("label")
            if label:
                out[str(label)] = row
    return out


def _normalize_page_reasons(result: Dict[str, Any]) -> Dict[str, str]:
    """Accept page_reasons dict or legacy reasons list from search_pages dumps."""
    pr = result.get("page_reasons")
    if isinstance(pr, dict) and pr:
        return {str(k): str(v) for k, v in pr.items()}
    reasons = result.get("reasons")
    if isinstance(reasons, dict):
        return {str(k): str(v) for k, v in reasons.items()}
    if isinstance(reasons, list):
        out: Dict[str, str] = {}
        for item in reasons:
            if not isinstance(item, dict):
                continue
            page = item.get("page")
            text = item.get("reason") or item.get("text") or ""
            if page is not None:
                out[str(page)] = str(text)
        return out
    return {}


def _load_timeline_search_links(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Map each search_pages completion to SearchAgent step labels.

    Uses a per-key time window so parallel SearchAgents (different keys) that
    finish out of tool-call order still keep their own step labels.
    """
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

    step_events: List[Dict[str, Any]] = []
    search_pages_events: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("stage") != "agent":
            continue
        event = row.get("event")
        label = str(row.get("label") or "")
        if event == "step" and (
            label.startswith("search_") or re.search(r"(?:^|_)search_", label)
        ):
            step_events.append(row)
        elif event == "search_pages":
            search_pages_events.append(row)

    prev_end_by_key: Dict[str, float] = {}
    links: List[Dict[str, Any]] = []
    for sp in search_pages_events:
        key = str(sp.get("key") or "")
        t_lo = float(prev_end_by_key.get(key, -1.0))
        t_hi = float(sp.get("t") or 0)
        master_step = int(sp.get("step") or 0)
        labels = []
        want_prefix = _safe_key(key)
        for e in step_events:
            t = float(e.get("t") or 0)
            if not (t_lo < t <= t_hi and e.get("label")):
                continue
            label = str(e.get("label"))
            parsed = _parse_search_label(label)
            # parsed = (key_prefix, session, turn, master_step)
            if parsed[3] and parsed[3] != master_step:
                continue
            # Must belong to this key — parallel keys share the same time window.
            if want_prefix and parsed[0] != want_prefix:
                continue
            labels.append(label)
        links.append(
            {
                "master_step": master_step,
                "key": key,
                "labels": labels,
            }
        )
        prev_end_by_key[key] = t_hi
    return links


def _consume_queue_key(
    queues: Dict[Any, List[Any]], key: Any
) -> Optional[List[Any]]:
    bucket = queues.get(key) or []
    if not bucket:
        return None
    item = bucket.pop(0)
    if not bucket:
        queues.pop(key, None)
    else:
        queues[key] = bucket
    return item


def _link_search_steps_to_calls(
    search_page_calls: List[Dict[str, Any]],
    search_by_prefix: Dict[str, List[Dict[str, Any]]],
    timeline_links: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Attach search step rows to each search_pages call.

    Prefer timeline-based linking (robust for repeated keys). Fall back to per-prefix
    + master_step queues when timeline data is missing.
    """
    by_label = _label_index(search_by_prefix)
    used_labels: set[str] = set()

    timeline_queues: Dict[tuple[int, str], List[List[str]]] = {}
    for link in timeline_links:
        k = (int(link.get("master_step") or 0), str(link.get("key") or ""))
        timeline_queues.setdefault(k, []).append(list(link.get("labels") or []))

    prefix_queues: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for prefix, rows in search_by_prefix.items():
        for row in rows:
            mstep = int(row.get("master_step") or 0)
            prefix_queues.setdefault((prefix, mstep), []).append(row)

    for call in search_page_calls:
        prefix = call["key_prefix"]
        master_step = int(call["master_step"] or 0)
        labels = _consume_queue_key(
            timeline_queues, (master_step, call["key"])
        )

        consumed: List[Dict[str, Any]] = []
        if labels:
            for label in labels:
                row = by_label.get(label)
                if row is not None and label not in used_labels:
                    consumed.append(row)
                    used_labels.add(label)

        if not consumed:
            n_steps = int(call["result"].get("n_search_steps") or 0)
            queue = [
                row
                for row in (prefix_queues.get((prefix, master_step)) or [])
                if str(row.get("label") or "") not in used_labels
            ]
            if n_steps <= 0 and queue:
                n_steps = len(queue)
            if n_steps > 0 and queue:
                consumed = queue[:n_steps]
                for row in consumed:
                    label = str(row.get("label") or "")
                    if label:
                        used_labels.add(label)
            elif queue and not n_steps:
                consumed = queue[:]
                for row in consumed:
                    label = str(row.get("label") or "")
                    if label:
                        used_labels.add(label)

        if consumed:
            remaining = [
                row
                for row in (prefix_queues.get((prefix, master_step)) or [])
                if str(row.get("label") or "") not in used_labels
            ]
            prefix_queues[(prefix, master_step)] = remaining

        call["search_steps"] = consumed

    # Return unassigned rows keyed by prefix (legacy shape for debug panel).
    unassigned_by_prefix: Dict[str, List[Dict[str, Any]]] = {}
    for (prefix, _mstep), rows in prefix_queues.items():
        if rows:
            unassigned_by_prefix.setdefault(prefix, []).extend(rows)
    return unassigned_by_prefix


def _load_priors_from_conversation(
    agent_dir: Path,
) -> Dict[Tuple[int, str, int], Dict[str, Any]]:
    """
    Map (master_step, key_prefix, session_index) → prior_context received by that session.

    Parsed from conversation.jsonl user messages (less truncated than step dumps).
    Kind examples:
      m021_search_Distance_between_GSU_Transformers_s1_user
      search_Distance_between_GSU_Transformers_s2_user  (legacy; master_step=0)
    """
    path = agent_dir / "conversation.jsonl"
    out: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = str(row.get("kind") or "")
        if not kind.endswith("_user"):
            continue
        master_step = 0
        prefix = ""
        sess = 0
        m = re.match(r"^m(\d+)(?:_t\d+_k\d+)?_search_(.+)_s(\d+)_user$", kind)
        if m:
            master_step, prefix, sess = int(m.group(1)), m.group(2), int(m.group(3))
        else:
            m = re.match(r"^search_(.+)_s(\d+)_user$", kind)
            if not m:
                continue
            prefix, sess = m.group(1), int(m.group(2))
        content = row.get("content")
        if not isinstance(content, str):
            continue
        prior = _extract_prior_from_user(content)
        if prior:
            out[(master_step, prefix, sess)] = prior
    return out


def _group_search_sessions(
    steps: List[Dict[str, Any]],
    *,
    meta_by_session: Optional[Dict[int, Dict[str, Any]]] = None,
    final_handoff_summary: str = "",
    final_prior_context: Optional[Dict[str, Any]] = None,
    conversation_priors: Optional[Dict[Tuple[int, str, int], Dict[str, Any]]] = None,
    key_prefix: str = "",
    master_step: int = 0,
) -> List[Dict[str, Any]]:
    sessions: Dict[int, List[Dict[str, Any]]] = {}
    for row in steps:
        sess = int(row.get("search_session") or 1)
        sessions.setdefault(sess, []).append(row)

    meta_by_session = meta_by_session or {}
    conversation_priors = conversation_priors or {}
    out: List[Dict[str, Any]] = []
    session_ids = sorted(sessions)
    for sess in session_ids:
        turns = sessions[sess]
        meta = meta_by_session.get(sess) or {}

        # What this session received.
        prior_in = meta.get("prior_context_in")
        if prior_in is None:
            prior_in = conversation_priors.get((master_step, key_prefix, sess))
        if prior_in is None and turns:
            prior_in = _extract_prior_from_user(
                turns[0].get("first_user_content") or ""
            )
        for t in turns:
            t.pop("first_user_content", None)

        prior_out = meta.get("prior_context_out")
        handoff_summary = str(meta.get("handoff_summary") or "")
        status = meta.get("status")
        pages = meta.get("pages")
        page_reasons = meta.get("page_reasons")
        if page_reasons is None:
            page_reasons = meta.get("reasons")

        # Legacy: final blobs only on last session.
        if sess == session_ids[-1]:
            if not handoff_summary and final_handoff_summary:
                handoff_summary = final_handoff_summary
            if prior_out is None and final_prior_context:
                prior_out = final_prior_context

        if not status:
            if handoff_summary or (sess < session_ids[-1]):
                status = "handoff"
            else:
                status = "complete" if pages else "unknown"

        out.append(
            {
                "session_index": sess,
                "n_turns": len(turns),
                "turns": turns,
                "error": next((t.get("error") for t in turns if t.get("error")), None),
                "status": status,
                "pages": pages if pages is not None else [],
                "page_reasons": page_reasons if page_reasons is not None else {},
                "reasons": page_reasons if page_reasons is not None else {},
                "prior_context_in": prior_in,
                "prior_context_out": prior_out,
                "handoff_summary": handoff_summary,
            }
        )

        # Session N out ≈ Session N+1 in (legacy reconstruction).
        if len(out) >= 2:
            prev = out[-2]
            cur = out[-1]
            if prev.get("prior_context_out") is None and cur.get("prior_context_in"):
                prev["prior_context_out"] = cur["prior_context_in"]
                if not prev.get("handoff_summary"):
                    prev["handoff_summary"] = (
                        "(reconstructed from next session's prior_context_in) "
                        "Full handoff_summary text was not stored for this session."
                    )

    return out


def _load_tool_dumps(tools_dir: Path) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    """
    Index 03_agent/tools/*.json by (step, tool_index, name).

    Prefer unlabeled Master dumps over SearchAgent-prefixed dumps when both exist
    for the same key (legacy runs may have collisions; new runs namespace search).
    """
    out: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    if not tools_dir.is_dir():
        return out

    # Filename patterns:
    #   step_{step}_{idx}_{name}.json                         (master / legacy)
    #   {label}_step_{step}_{idx}_{name}.json                 (search, new)
    pat = re.compile(
        r"^(?:(?P<label>.+)_)?step_(?P<step>\d+)_(?P<idx>\d+)_(?P<name>.+)\.json$"
    )
    rows: List[Tuple[bool, Tuple[int, int, str], Dict[str, Any]]] = []
    for path in sorted(tools_dir.glob("*.json")):
        data = _read_json(path) or {}
        m = pat.match(path.name)
        label = str(data.get("label") or "")
        name = str(data.get("name") or "")
        step = int(data.get("step") or 0)
        tool_index = int(data.get("tool_index") or 0)
        if m:
            if not name:
                name = m.group("name")
            if not step:
                step = int(m.group("step"))
            if data.get("tool_index") is None:
                tool_index = int(m.group("idx"))
            if not label and m.group("label"):
                label = m.group("label")
        if not name:
            continue
        is_master = not label
        rows.append(
            (
                is_master,
                (step, tool_index, name),
                {
                    "arguments": data.get("arguments"),
                    "result": data.get("result"),
                    "extra_files": data.get("extra_files") or {},
                    "filename": path.name,
                    "label": label or None,
                },
            )
        )

    # Search dumps first, then master dumps overwrite same keys.
    for is_master, key, payload in sorted(rows, key=lambda r: (r[0], r[1])):
        out[key] = payload
    return out


def _parse_result_preview(preview: Any) -> Any:
    """Normalize truncated string previews back toward JSON when possible."""
    if preview is None or isinstance(preview, (dict, list)):
        return preview
    if not isinstance(preview, str):
        return preview
    text = preview.strip()
    # Truncation markers from master `_preview` ("...") or sanitize ("...<N more chars>")
    text = re.sub(r"\.\.\.(?:<\d+ more chars>)?$", "", text).rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return preview


def _result_matches_preview(result: Any, preview: Any) -> bool:
    """True when a tools/*.json dump plausibly belongs to this master tool_result."""
    if result is None or preview is None:
        return False
    if isinstance(preview, (dict, list)):
        try:
            return json.dumps(preview, sort_keys=True, ensure_ascii=False, default=str)[
                :200
            ] == json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)[
                :200
            ]
        except TypeError:
            return False
    if not isinstance(preview, str):
        return False
    head = preview.strip()
    if not head:
        return False
    # Compare against a compact JSON rendering of the dump.
    try:
        dumped = json.dumps(result, ensure_ascii=False, default=str)
    except TypeError:
        return False
    # Preview may be truncated; require a shared prefix (ignore trailing "...").
    head_cmp = re.sub(r"\.\.\.(?:<\d+ more chars>)?$", "", head).rstrip()
    n = min(len(head_cmp), len(dumped), 120)
    if n < 20:
        return head_cmp in dumped
    return dumped[:n] == head_cmp[:n]


def _load_master_tool_results_from_conversation(
    agent_dir: Path,
) -> Dict[Tuple[int, str], List[Any]]:
    """
    Map (turn, tool_name) → list of parsed tool contents from Master conversation.

    SearchAgent tool rows use kind like search_*_tool and are ignored.
    """
    path = agent_dir / "conversation.jsonl"
    out: Dict[Tuple[int, str], List[Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") != "tool":
            continue
        name = str(row.get("name") or "")
        if not name:
            continue
        turn = int(row.get("turn") or 0)
        content = row.get("content")
        parsed: Any = content
        if isinstance(content, str):
            parsed = _parse_result_preview(content)
        out.setdefault((turn, name), []).append(parsed)
    return out


def _recover_schema_items(payload: Any) -> Optional[Dict[str, Any]]:
    """Best-effort recover load_kv_schema items from truncated JSON/text."""
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return {
            "items": payload["items"],
            "count": payload.get("count", len(payload["items"])),
        }
    if not isinstance(payload, str):
        return None
    items: List[Dict[str, str]] = []
    for m in re.finditer(
        r'\{\s*"key"\s*:\s*("(?:\\.|[^"])*")\s*,\s*"description"\s*:\s*("(?:\\.|[^"])*")\s*\}',
        payload,
    ):
        try:
            items.append(
                {
                    "key": json.loads(m.group(1)),
                    "description": json.loads(m.group(2)),
                }
            )
        except json.JSONDecodeError:
            continue
    if not items:
        return None
    return {"items": items, "count": len(items)}


def _resolve_tool_result(
    *,
    name: str,
    arguments: Any,
    preview: Any,
    dump: Optional[Dict[str, Any]],
    conversation_queue: Optional[List[Any]],
) -> Tuple[Any, Optional[str], Dict[str, str]]:
    """Pick the best full result for a Master tool node."""
    filename = dump.get("filename") if dump else None
    extra_files = (dump.get("extra_files") if dump else None) or {}

    dump_result = dump.get("result") if dump else None
    dump_ok = dump_result is not None and (
        preview is None
        or name == "extract_kv_vlm"  # master-only; dumps are authoritative
        or _result_matches_preview(dump_result, preview)
    )
    if dump_ok:
        return dump_result, filename, extra_files

    # Dump is missing or belongs to another agent (legacy path collision).
    filename = filename if dump_ok else None
    extra_files = extra_files if dump_ok else {}

    candidates: List[Any] = []
    if conversation_queue:
        while conversation_queue:
            candidates.append(conversation_queue.pop(0))
    if preview is not None:
        candidates.append(preview)
    if dump_result is not None:
        candidates.append(dump_result)

    for cand in candidates:
        if name == "load_kv_schema":
            recovered = _recover_schema_items(cand)
            if recovered:
                return recovered, filename, extra_files
        parsed = _parse_result_preview(cand)
        if isinstance(parsed, (dict, list)):
            return parsed, filename, extra_files

    if candidates:
        return candidates[0], filename, extra_files
    return None, filename, extra_files


def _flatten_sessions_as_turns(
    search_sessions: List[Dict[str, Any]],
    *,
    display_session_index: int,
) -> List[Dict[str, Any]]:
    """
    Merge handoff sessions for one key into a single turn list with unique
    search_turn indices (1..N). Used when a multi-key search_pages call maps
    each key to one display session.
    """
    turns: List[Dict[str, Any]] = []
    n = 0
    for sess in search_sessions or []:
        for turn in sess.get("turns") or []:
            n += 1
            row = dict(turn)
            row["search_turn"] = n
            row["search_session"] = display_session_index
            row["handoff_session_index"] = sess.get("session_index")
            turns.append(row)
    return turns


def _build_batch_search_agent(
    *,
    batch_items: List[Dict[str, Any]],
    step_call_queue: List[Dict[str, Any]],
    search_by_prefix: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    master_step: int = 0,
    default_status: str = "unknown",
) -> Dict[str, Any]:
    """
    One SearchAgent node with session_index 1..N = one key each (parallel).

    Consumes matching per-key calls from step_call_queue by key. If a per-key
    dump is missing, rebuild turns from search step files for that key_prefix.
    """
    search_by_prefix = search_by_prefix or {}
    sessions: List[Dict[str, Any]] = []
    for i, item in enumerate(batch_items):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "")
        call = None
        for j, cand in enumerate(step_call_queue):
            if str(cand.get("key") or "") == key:
                call = step_call_queue.pop(j)
                break

        display_idx = i + 1
        turns: List[Dict[str, Any]] = []
        if call and (call.get("search_sessions") or []):
            turns = _flatten_sessions_as_turns(
                call.get("search_sessions") or [],
                display_session_index=display_idx,
            )
        elif key:
            # Recover when the per-key dump was overwritten / missing.
            prefix = _safe_key(key)
            rows = [
                r
                for r in (search_by_prefix.get(prefix) or [])
                if int(r.get("master_step") or 0) == master_step
            ]
            if not rows:
                # Async jobs may still be writing; take any matching prefix.
                rows = list(search_by_prefix.get(prefix) or [])
            if rows:
                recovered = _group_search_sessions(
                    rows,
                    key_prefix=prefix,
                    master_step=master_step,
                )
                turns = _flatten_sessions_as_turns(
                    recovered, display_session_index=display_idx
                )

        pages = item.get("pages")
        if pages is None and call:
            pages = (call.get("output") or {}).get("pages")
        page_reasons = item.get("page_reasons")
        if page_reasons is None and call:
            page_reasons = (call.get("output") or {}).get("page_reasons")
        status = item.get("status")
        if not status and call:
            status = (call.get("output") or {}).get("status") or (
                call.get("result") or {}
            ).get("status")
        if not status:
            status = "complete" if turns else default_status

        sessions.append(
            {
                "session_index": display_idx,
                "key": key,
                "n_turns": len(turns),
                "turns": turns,
                "status": status or default_status,
                "pages": pages if pages is not None else [],
                "page_reasons": page_reasons if isinstance(page_reasons, dict) else {},
                "reasons": page_reasons if isinstance(page_reasons, dict) else {},
                "reason": item.get("reason")
                or ((call or {}).get("output") or {}).get("reason")
                or "",
                "prior_context_in": None,
                "prior_context_out": None,
                "handoff_summary": "",
                "error": next(
                    (t.get("error") for t in turns if t.get("error")), None
                ),
                "filename": (call or {}).get("filename"),
            }
        )

    return {
        "type": "search_agent",
        "key": f"{len(sessions)} keys (parallel)",
        "batch": True,
        "output": {
            "status": "complete",
            "pages": [],
            "page_reasons": {},
            "n_keys": len(sessions),
        },
        "result": {
            "status": "complete",
            "n_search_sessions": len(sessions),
            "n_keys": len(sessions),
            "n_search_steps": sum(s.get("n_turns") or 0 for s in sessions),
        },
        "sessions": sessions,
    }


def build_agent_tree(run_dir: Path) -> Dict[str, Any]:
    """
    Build Master → tools → SearchAgent sessions tree.

    Links search steps to master search_pages via tools/step_{N}_{M}_search_pages.json
    and key prefix matching. Attaches full tool dump results for other Master tools
    (load_kv_schema, extract_kv_vlm, …) so the viewer can show outcomes, not just args.
    """
    agent_dir = run_dir / "03_agent"
    tools_dir = agent_dir / "tools"
    master_steps, search_by_prefix = _load_steps(agent_dir)
    conversation_priors = _load_priors_from_conversation(agent_dir)
    tool_dumps = _load_tool_dumps(tools_dir)
    conversation_tools = _load_master_tool_results_from_conversation(agent_dir)
    # Mutable queues so repeated same-name tools in one turn consume in order.
    conversation_queues: Dict[Tuple[int, str], List[Any]] = {
        k: list(v) for k, v in conversation_tools.items()
    }

    search_page_calls: List[Dict[str, Any]] = []
    for (step, tool_index, name), dump in tool_dumps.items():
        if name != "search_pages":
            continue
        # Prefer unlabeled master dumps (search_pages is master-only).
        if dump.get("label"):
            continue
        args = dump.get("arguments") or {}
        result = dump.get("result") or {}
        if not isinstance(result, dict):
            result = {}
        # Batched multi-key summary dump — per-key dumps are saved separately.
        if isinstance(result.get("results"), list) and "key" not in result:
            continue
        if isinstance(args.get("key"), list):
            continue
        key = str(args.get("key") or result.get("key") or "")
        prefix = _safe_key(key)
        search_page_calls.append(
            {
                "master_step": step,
                "tool_index": tool_index,
                "key": key,
                "key_prefix": prefix,
                "arguments": args,
                "result": result,
                "filename": dump.get("filename"),
            }
        )
    search_page_calls.sort(
        key=lambda r: (r["master_step"], r["tool_index"])
    )

    timeline_links = _load_timeline_search_links(run_dir)
    search_queues = _link_search_steps_to_calls(
        search_page_calls,
        search_by_prefix,
        timeline_links,
    )

    calls_by_master: Dict[int, List[Dict[str, Any]]] = {}
    for call in search_page_calls:
        prefix = call["key_prefix"]
        consumed = call.get("search_steps") or []
        n_sessions = int(call["result"].get("n_search_sessions") or 0)
        page_reasons = _normalize_page_reasons(call["result"])

        meta_rows = call["result"].get("sessions") or []
        meta_by_session: Dict[int, Dict[str, Any]] = {}
        for row in meta_rows:
            if isinstance(row, dict) and row.get("session_index") is not None:
                meta_by_session[int(row["session_index"])] = row

        call["search_sessions"] = _group_search_sessions(
            consumed,
            meta_by_session=meta_by_session,
            final_handoff_summary=str(
                call["result"].get("handoff_summary") or ""
            ),
            final_prior_context=call["result"].get("prior_context"),
            conversation_priors=conversation_priors,
            key_prefix=prefix,
            master_step=int(call["master_step"] or 0),
        )
        if not call["search_sessions"] and n_sessions:
            call["note"] = f"{n_sessions} session(s) recorded in result metadata"
        call["output"] = {
            "pages": call["result"].get("pages") or [],
            "page_reasons": page_reasons,
            "status": call["result"].get("status"),
            "reason": call["result"].get("reason") or "",
        }
        calls_by_master.setdefault(call["master_step"], []).append(call)

    call_queues: Dict[int, List[Dict[str, Any]]] = {
        k: list(v) for k, v in calls_by_master.items()
    }

    tree: List[Dict[str, Any]] = []
    for ms in master_steps:
        mstep = int(ms.get("step") or 0)
        step_call_queue = call_queues.get(mstep) or []
        node: Dict[str, Any] = {
            "type": "master_turn",
            "step": mstep,
            "prompt_est_tokens": ms.get("prompt_est_tokens"),
            "input_tokens": ms.get("input_tokens"),
            "output_tokens": ms.get("output_tokens"),
            "max_tokens": ms.get("max_tokens"),
            "error": ms.get("error"),
            "assistant": ms.get("tool_calls"),
            "tools": [],
        }

        for ti, tr in enumerate(ms.get("tool_results") or []):
            tname = tr.get("name") or "?"
            dump = tool_dumps.get((mstep, ti, tname))
            # Ignore SearchAgent-prefixed dumps when resolving Master tools.
            if dump and dump.get("label"):
                dump = None
            preview = tr.get("result_preview")
            conv_q = conversation_queues.get((mstep, tname))
            result, filename, extra_files = _resolve_tool_result(
                name=tname,
                arguments=tr.get("arguments"),
                preview=preview,
                dump=dump,
                conversation_queue=conv_q,
            )
            tool_node: Dict[str, Any] = {
                "type": "tool",
                "name": tname,
                "arguments": tr.get("arguments"),
                "result_preview": preview,
                "result": result,
                "filename": filename,
                "extra_files": extra_files,
                "children": [],
            }
            if dump and dump.get("arguments") is not None and not tool_node["arguments"]:
                tool_node["arguments"] = dump.get("arguments")

            # Normalize key/keys so viewers can always read a single field.
            args = tool_node.get("arguments")
            if isinstance(args, dict):
                if args.get("key") is None and args.get("keys") is not None:
                    args = dict(args)
                    args["key"] = args.get("keys")
                    tool_node["arguments"] = args

            if tname == "search_pages" and isinstance(result, dict) and "accepted" in result:
                accepted = result.get("accepted") or []
                if not isinstance(accepted, list):
                    accepted = []
                # Prefer explicit accepted list; fall back to normalized key(s).
                if not accepted:
                    raw = (tool_node.get("arguments") or {}).get("key")
                    if isinstance(raw, list):
                        accepted = [str(k) for k in raw if str(k).strip()]
                    elif raw:
                        accepted = [str(raw)]
                tool_node["search_output"] = {
                    "status": "accepted",
                    "n_keys": len(accepted),
                    "accepted": accepted,
                    "skipped": result.get("skipped") or [],
                }
                # Link whatever per-key dumps / step turns already exist for these
                # keys (async jobs finish and dump under the start step).
                if accepted:
                    batch_items = [{"key": str(k)} for k in accepted]
                    cross_queue: List[Dict[str, Any]] = []
                    for q in call_queues.values():
                        cross_queue.extend(q)
                    child = _build_batch_search_agent(
                        batch_items=batch_items,
                        step_call_queue=cross_queue,
                        search_by_prefix=search_by_prefix,
                        master_step=mstep,
                        default_status="running",
                    )
                    # Reflect enqueue status when nothing finished yet.
                    n_done = sum(
                        1
                        for s in (child.get("sessions") or [])
                        if str(s.get("status") or "")
                        not in {"", "unknown", "running", "accepted", "queued"}
                    )
                    child["key"] = (
                        accepted[0]
                        if len(accepted) == 1
                        else f"{len(accepted)} keys (accepted)"
                    )
                    child["output"] = {
                        **(child.get("output") or {}),
                        "status": "complete" if n_done == len(accepted) else "accepted",
                        "accepted": accepted,
                        "n_keys": len(accepted),
                    }
                    child["result"] = {
                        **(child.get("result") or {}),
                        "status": child["output"]["status"],
                    }
                    remaining_ids = {id(c) for c in cross_queue}
                    for q in call_queues.values():
                        q[:] = [c for c in q if id(c) in remaining_ids]
                    tool_node["children"].append(child)
            elif tname in {"search_pages", "collect_search_results", "await_searches"}:
                batch_items = None
                if isinstance(result, dict) and isinstance(result.get("results"), list):
                    batch_items = [
                        x for x in result["results"] if isinstance(x, dict)
                    ]
                if (
                    batch_items is None
                    and isinstance(result, dict)
                    and isinstance(result.get("completed"), list)
                ):
                    batch_items = [
                        x for x in result["completed"] if isinstance(x, dict)
                    ]
                args_key = (tool_node.get("arguments") or {}).get("key")
                if args_key is None:
                    args_key = (tool_node.get("arguments") or {}).get("keys")
                if (
                    batch_items is None
                    and isinstance(args_key, list)
                    and len(args_key) >= 1
                ):
                    # Arguments list keys but result missing — still try queue.
                    batch_items = [{"key": str(k)} for k in args_key]
                elif (
                    batch_items is None
                    and isinstance(args_key, str)
                    and args_key.strip()
                ):
                    batch_items = [{"key": args_key}]

                if batch_items and len(batch_items) >= 1 and tname in {
                    "collect_search_results",
                    "await_searches",
                    "search_pages",
                }:
                    # Async jobs dump under the start step; match by key across steps.
                    # Works for 1..N completed keys (single-key used to miss the tree
                    # when the dump was already claimed by the enqueue tool).
                    if tname in {"collect_search_results", "await_searches"} or len(
                        batch_items
                    ) > 1:
                        cross_queue: List[Dict[str, Any]] = []
                        for q in call_queues.values():
                            cross_queue.extend(q)
                        tool_node["children"].append(
                            _build_batch_search_agent(
                                batch_items=batch_items,
                                step_call_queue=cross_queue,
                                search_by_prefix=search_by_prefix,
                                master_step=mstep,
                            )
                        )
                        remaining_ids = {id(c) for c in cross_queue}
                        for q in call_queues.values():
                            q[:] = [c for c in q if id(c) in remaining_ids]
                        tool_node["search_output"] = {
                            "status": "complete",
                            "n_keys": len(batch_items),
                            "results": batch_items,
                        }
                    elif step_call_queue and tname == "search_pages":
                        call = step_call_queue.pop(0)
                        output = call.get("output") or {}
                        tool_node["search_output"] = output
                        tool_node["result"] = call.get("result") or tool_node.get(
                            "result"
                        )
                        tool_node["filename"] = call.get("filename") or tool_node.get(
                            "filename"
                        )
                        tool_node["children"].append(
                            {
                                "type": "search_agent",
                                "key": call["key"],
                                "key_prefix": call["key_prefix"],
                                "output": output,
                                "result": {
                                    "pages": output.get("pages"),
                                    "page_reasons": output.get("page_reasons"),
                                    "reasons": output.get("page_reasons"),
                                    "status": output.get("status")
                                    or call["result"].get("status"),
                                    "reason": output.get("reason")
                                    or call["result"].get("reason"),
                                    "n_search_steps": len(
                                        call.get("search_steps") or []
                                    )
                                    or call["result"].get("n_search_steps"),
                                    "n_search_sessions": call["result"].get(
                                        "n_search_sessions"
                                    ),
                                },
                                "sessions": call.get("search_sessions") or [],
                                "filename": call.get("filename"),
                                "note": call.get("note"),
                            }
                        )
                elif step_call_queue and tname == "search_pages":
                    call = step_call_queue.pop(0)
                    output = call.get("output") or {}
                    tool_node["search_output"] = output
                    tool_node["result"] = call.get("result") or tool_node.get("result")
                    tool_node["filename"] = call.get("filename") or tool_node.get(
                        "filename"
                    )
                    tool_node["children"].append(
                        {
                            "type": "search_agent",
                            "key": call["key"],
                            "key_prefix": call["key_prefix"],
                            "output": output,
                            "result": {
                                "pages": output.get("pages"),
                                "page_reasons": output.get("page_reasons"),
                                "reasons": output.get("page_reasons"),
                                "status": output.get("status")
                                or call["result"].get("status"),
                                "reason": output.get("reason")
                                or call["result"].get("reason"),
                                "n_search_steps": len(call.get("search_steps") or [])
                                or call["result"].get("n_search_steps"),
                                "n_search_sessions": call["result"].get(
                                    "n_search_sessions"
                                ),
                            },
                            "sessions": call.get("search_sessions") or [],
                            "filename": call.get("filename"),
                            "note": call.get("note"),
                        }
                    )
            node["tools"].append(tool_node)


        tree.append(node)

    unassigned = {
        prefix: rows
        for prefix, rows in search_queues.items()
        if rows
    }

    result_doc = _read_json(run_dir / "04_result.json") or {}
    error_doc = _read_json(run_dir / "04_error.json") or {}
    kv_results = result_doc.get("kv_results")
    if not isinstance(kv_results, list):
        kv_results = []

    return {
        "master_turns": tree,
        "n_master_turns": len(tree),
        "n_search_page_calls": len(search_page_calls),
        "unassigned_search_steps": unassigned,
        "output": {
            "kv_results": kv_results,
            "n_kv": len(kv_results),
            "error": error_doc.get("error") or result_doc.get("error"),
        },
    }
