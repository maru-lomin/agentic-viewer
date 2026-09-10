"""Load or compute baseline KV eval (05_eval.json) from run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_viewer.eval.evaluate_kv import (
    build_pred_only_report,
    build_report,
    infer_document_name_from_pred,
    load_json,
)
from agentic_viewer.eval.paths import answer_sheet_path
from agentic_viewer.pdf_source import infer_run_document


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def eval_cache_has_reason_split(report: Dict[str, Any]) -> bool:
    """True when cached eval separates VLM evidence vs SearchAgent reasons."""
    per_key = report.get("per_key")
    if not isinstance(per_key, list) or not per_key:
        return False
    first = per_key[0]
    if not isinstance(first, dict):
        return False
    return "search_reasons" in first


def eval_cache_has_searched_pages(report: Dict[str, Any]) -> bool:
    """True when cached eval includes SearchAgent inspected pages and bm25_queries."""
    per_key = report.get("per_key")
    if not isinstance(per_key, list) or not per_key:
        return False
    first = per_key[0]
    if not isinstance(first, dict):
        return False
    sp = first.get("search_pages")
    return isinstance(sp, dict) and "inspected" in sp and "bm25_queries" in sp


def extract_searched_pages(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Extract all pages inspected and queried by SearchAgent for each key in run_dir."""
    out: Dict[str, Dict[str, Any]] = {}
    pred_path = run_dir / "04_result.json"
    sat: list = []
    if pred_path.is_file():
        try:
            pred = json.loads(pred_path.read_text(encoding="utf-8"))
            if isinstance(pred, dict):
                sat = pred.get("search_agent_traces") or []
        except Exception:
            sat = []

    if sat:
        for entry in sat:
            if not isinstance(entry, dict):
                continue
            k = entry.get("key")
            if not k:
                continue
            k = str(k).strip()
            prior = entry.get("prior_context_out") or {}
            inspected = prior.get("pages_inspected") or []
            bm25_hits = prior.get("bm25_hits") or []
            cand_rows = prior.get("candidate_pages") or []

            if k not in out:
                out[k] = {
                    "inspected": set(),
                    "bm25": set(),
                    "candidates": set(),
                    "queries": {},
                }
            for p in inspected:
                try:
                    out[k]["inspected"].add(int(p))
                except (TypeError, ValueError):
                    pass
            for h in bm25_hits:
                if isinstance(h, dict):
                    p = h.get("page")
                    if p is not None:
                        try:
                            out[k]["bm25"].add(int(p))
                        except (TypeError, ValueError):
                            pass
                    for p2 in h.get("pages") or []:
                        try:
                            out[k]["bm25"].add(int(p2))
                        except (TypeError, ValueError):
                            pass
                    q = str(h.get("query") or "").strip()
                    cid = str(h.get("chunk_id") or "").strip()
                    if q and cid:
                        if q not in out[k]["queries"]:
                            out[k]["queries"][q] = {}
                        if cid not in out[k]["queries"][q]:
                            score = h.get("score")
                            out[k]["queries"][q][cid] = {
                                "chunk_id": cid,
                                "page": int(p) if p is not None else None,
                                "score": round(float(score), 4) if score is not None else None,
                            }
            for c in cand_rows:
                if isinstance(c, dict):
                    p = c.get("page")
                    if p is not None:
                        try:
                            out[k]["candidates"].add(int(p))
                        except (TypeError, ValueError):
                            pass
    # Enrich from 03_agent/tools if present (restores any BM25 hits truncated in prior_context,
    # or serves as fallback when search_agent_traces is absent).
    tools_dir = run_dir / "03_agent" / "tools"
    if tools_dir.is_dir():
            session_keys: Dict[str, set] = {}
            session_inspected: Dict[str, set] = {}
            session_bm25: Dict[str, set] = {}
            session_queries: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for f in tools_dir.glob("*.json"):
                parts = f.stem.split("_step_")
                s_lbl = parts[0]
                if s_lbl not in session_inspected:
                    session_inspected[s_lbl] = set()
                    session_bm25[s_lbl] = set()
                    session_keys[s_lbl] = set()
                    session_queries[s_lbl] = {}
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                name = d.get("name")
                args = d.get("arguments") or {}
                res = d.get("result") or {}
                if name in ("get_page_text", "get_page_image"):
                    p = args.get("page")
                    if p is not None:
                        try:
                            session_inspected[s_lbl].add(int(p))
                        except (TypeError, ValueError):
                            pass
                elif name == "bm25_search":
                    q = str(args.get("query") or "").strip()
                    hits = res.get("hits") or []
                    for h in hits:
                        if isinstance(h, dict):
                            p = h.get("page")
                            if p is not None:
                                try:
                                    session_bm25[s_lbl].add(int(p))
                                except (TypeError, ValueError):
                                    pass
                            cid = str(h.get("chunk_id") or "").strip()
                            if q and cid:
                                if q not in session_queries[s_lbl]:
                                    session_queries[s_lbl][q] = {}
                                if cid not in session_queries[s_lbl][q]:
                                    score = h.get("score")
                                    session_queries[s_lbl][q][cid] = {
                                        "chunk_id": cid,
                                        "page": int(p) if p is not None else None,
                                        "score": round(float(score), 4) if score is not None else None,
                                    }
                elif name in ("submit_pages", "no_relevant_pages"):
                    k_arg = args.get("key")
                    if k_arg:
                        session_keys[s_lbl].add(str(k_arg).strip())
            for s_lbl, keys in session_keys.items():
                for k in keys:
                    if k not in out:
                        out[k] = {"inspected": set(), "bm25": set(), "candidates": set(), "queries": {}}
                    out[k]["inspected"].update(session_inspected.get(s_lbl, set()))
                    out[k]["bm25"].update(session_bm25.get(s_lbl, set()))
                    s_q = session_queries.get(s_lbl) or {}
                    for q_str, c_map in s_q.items():
                        if q_str not in out[k]["queries"]:
                            out[k]["queries"][q_str] = {}
                        out[k]["queries"][q_str].update(c_map)

    final_out = {}
    for k, v in out.items():
        q_list = []
        for q_str, c_map in (v.get("queries") or {}).items():
            q_list.append({
                "query": q_str,
                "hits": list(c_map.values()),
            })
        final_out[k] = {
            "inspected": sorted(list(v["inspected"])),
            "bm25": sorted(list(v["bm25"])),
            "candidates": sorted(list(v["candidates"])),
            "bm25_queries": q_list,
        }
    return final_out


def enrich_eval_with_searched_pages(
    report: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Enrich per_key rows in eval report with pages inspected and queried by SearchAgent."""
    per_key = report.get("per_key")
    if not isinstance(per_key, list):
        return report

    searched_map = extract_searched_pages(run_dir)
    for row in per_key:
        if not isinstance(row, dict):
            continue
        k = row.get("key")
        if not k:
            continue
        info = searched_map.get(str(k).strip()) or {}
        sp = row.setdefault("search_pages", {})
        if not isinstance(sp, dict):
            continue
        pred_pages = sp.get("pred") or []
        pred_set = set(pred_pages)
        inspected = info.get("inspected") or []
        other_inspected = [p for p in inspected if p not in pred_set]

        sp["inspected"] = inspected
        sp["other_inspected"] = other_inspected
        sp["bm25"] = info.get("bm25") or []
        sp["candidates"] = info.get("candidates") or []
        sp["bm25_queries"] = info.get("bm25_queries") or []

    return report


def load_or_compute_run_eval(
    run_dir: Path,
    *,
    run_id: Optional[str] = None,
    refresh: bool = False,
    write_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Return baseline eval report for a run directory.

    Uses cached ``05_eval.json`` when valid; otherwise scores ``04_result.json``
    against the answer sheet (same logic as the Inference Eval tab).
    """
    cache_path = run_dir / "05_eval.json"
    if cache_path.is_file() and not refresh:
        cached = _read_json(cache_path)
        if (
            isinstance(cached, dict)
            and cached.get("overall")
            and eval_cache_has_reason_split(cached)
        ):
            if cached.get("has_gt") is False:
                fresh = load_or_compute_run_eval(
                    run_dir,
                    run_id=run_id,
                    refresh=True,
                    write_cache=write_cache,
                )
                if isinstance(fresh, dict) and fresh.get("has_gt") is not False:
                    return fresh
            if not eval_cache_has_searched_pages(cached):
                enrich_eval_with_searched_pages(cached, run_dir)
                if write_cache:
                    try:
                        cache_path.write_text(
                            json.dumps(cached, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
            return cached

    pred_path = run_dir / "04_result.json"
    pred = _read_json(pred_path)
    if not isinstance(pred, dict):
        return None

    ans_path = answer_sheet_path()
    answer_sheet: Optional[Dict[str, Any]] = None
    if ans_path.is_file():
        loaded = load_json(ans_path)
        if isinstance(loaded, dict):
            answer_sheet = loaded

    report: Optional[Dict[str, Any]] = None
    if answer_sheet is not None:
        try:
            report = build_report(
                pred,
                answer_sheet,
                pred_path=str(pred_path),
                answer_sheet_path=str(ans_path),
            )
        except (KeyError, ValueError):
            report = None

    if report is None:
        doc_name = infer_run_document(run_dir, result=pred) or infer_document_name_from_pred(
            pred
        )
        if not doc_name:
            return None
        report = build_pred_only_report(
            pred,
            doc_name,
            pred_path=str(pred_path),
            answer_sheet_path=str(ans_path) if ans_path.is_file() else None,
        )

    if run_id:
        report["run_id"] = run_id
    enrich_eval_with_searched_pages(report, run_dir)
    if write_cache:
        try:
            cache_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            report["cache_write_error"] = str(cache_path)
    return report
