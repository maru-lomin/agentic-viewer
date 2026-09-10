"""Manage and persist tracked wrong cases for analysis and improvement tracking."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
from agentic_viewer.evaluation.summary import read_agentic_evals
from agentic_viewer.pdf_source import infer_run_document
from agentic_viewer.timezone import KST, kst_now, kst_now_iso, to_kst


def default_wrong_cases_path() -> Path:
    env = os.environ.get("AGENTIC_WRONG_CASES") or os.environ.get("AGENTIC_WRONG_CASES_PATH")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    # store.py is at repo_root / agentic-viewer / agentic_viewer / wrong_cases / store.py
    repo_root = here.parents[3] if len(here.parents) > 3 and (here.parents[3] / "outputs").is_dir() else here.parents[2]
    # Primary location: outputs/wrong_cases.json
    primary = repo_root / "outputs" / "wrong_cases.json"
    if primary.is_file():
        return primary.resolve()
    # Secondary check: dataset/wrong_cases.json
    secondary = repo_root / "dataset" / "wrong_cases.json"
    if secondary.is_file():
        return secondary.resolve()
    # Default to outputs/wrong_cases.json
    primary.parent.mkdir(parents=True, exist_ok=True)
    return primary.resolve()


def wrong_cases_path() -> Path:
    return default_wrong_cases_path()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def make_case_id(run_id: str, key: str) -> str:
    """Generate a deterministic, safe case identifier."""
    r = str(run_id or "").strip()
    k = str(key or "").strip()
    digest = hashlib.sha1(f"{r}:{k}".encode("utf-8")).hexdigest()[:10]
    # Simple clean prefix for readability
    clean_k = "".join(c if c.isalnum() else "_" for c in k)[:24].strip("_")
    return f"wc-{clean_k}-{digest}" if clean_k else f"wc-{digest}"


def load_wrong_cases() -> Dict[str, Any]:
    path = wrong_cases_path()
    data = _read_json(path)
    if not isinstance(data, dict):
        return {"version": 1, "cases": []}
    if not isinstance(data.get("cases"), list):
        data["cases"] = []
    return data


def save_wrong_cases(data: Dict[str, Any]) -> Path:
    path = wrong_cases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            stamp = kst_now().strftime("%Y%m%dT%H%M%S")
            backup = path.with_name(f"{path.stem}.bak.{stamp}{path.suffix}")
            shutil.copy2(path, backup)
        except OSError:
            pass
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _parse_ts(val: Any) -> Optional[datetime]:
    if not val:
        return None
    try:
        return to_kst(datetime.fromisoformat(str(val).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def _find_key_in_eval(eval_report: Optional[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    if not eval_report or not isinstance(eval_report.get("per_key"), list):
        return None
    for row in eval_report["per_key"]:
        if isinstance(row, dict) and str(row.get("key")) == key:
            return row
    return None


def _load_run_eval_safe(run_dir: Path, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    cached = _read_json(run_dir / "05_eval.json")
    if isinstance(cached, dict) and isinstance(cached.get("per_key"), list):
        return cached
    return load_or_compute_run_eval(run_dir, run_id=run_id)


def extract_snapshot_for_key(
    run_dir: Path,
    key: str,
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract evaluation details, VLM reasons, and agentic eval for snapshot."""
    eval_report = _load_run_eval_safe(run_dir, run_id=run_id)
    key_eval = _find_key_in_eval(eval_report, key) or {}
    val = key_eval.get("value") or {}
    sp = key_eval.get("search_pages") or {}
    et = key_eval.get("evidence_text") or {}
    sr = key_eval.get("search_reasons") or {}
    pc = key_eval.get("page_chunk_id") or {}

    agentic_by_key = read_agentic_evals(run_dir)
    ae = agentic_by_key.get(key)

    agentic_summary = None
    if isinstance(ae, dict):
        agentic_summary = {
            "status": ae.get("status"),
            "is_correct_answer": ae.get("is_correct_answer"),
            "is_valid_gold": ae.get("is_valid_gold"),
            "reason_summary": ae.get("reason_summary") or ae.get("reason"),
            "reason_detail": ae.get("reason_detail") or ae.get("text"),
            "seconds": ae.get("seconds"),
            "n_steps": ae.get("n_steps"),
            "finished_at": ae.get("finished_at"),
        }

    return {
        "pred": val.get("pred"),
        "gold": val.get("gold"),
        "exact_match": val.get("exact_match"),
        "pred_pages": sp.get("pred") if isinstance(sp.get("pred"), list) else [],
        "gold_pages": sp.get("gold") if isinstance(sp.get("gold"), list) else [],
        "page_f1": sp.get("f1"),
        "token_f1": et.get("token_f1"),
        "vlm_reason": et.get("pred") or "",
        "search_reasons": sr.get("pred") or key_eval.get("reason") or "",
        "gold_evidences": et.get("gold") or "",
        "pred_chunks": pc.get("pred_map") if isinstance(pc.get("pred_map"), dict) else {},
        "agentic_eval": agentic_summary,
    }


def add_or_update_wrong_case(
    runs_root: Path,
    run_id: str,
    key: str,
    *,
    note: str = "",
    status: str = "open",
    tags: Optional[List[str]] = None,
    user_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record or update a wrong case entry."""
    run_id_clean = str(run_id or "").strip()
    key_clean = str(key or "").strip()
    if not run_id_clean:
        raise ValueError("run_id is required")
    if not key_clean:
        raise ValueError("key is required")

    run_dir = (runs_root / run_id_clean).resolve()
    meta = _read_json(run_dir / "meta.json") or {}
    document = meta.get("source_filename") or infer_run_document(run_dir) or ""
    dataset_id = meta.get("dataset_id")
    dataset_name = meta.get("dataset_name")

    snapshot = extract_snapshot_for_key(run_dir, key_clean, run_id=run_id_clean)
    if user_snapshot and isinstance(user_snapshot, dict):
        snapshot.update(user_snapshot)

    case_id = make_case_id(run_id_clean, key_clean)
    now_iso = kst_now_iso()

    data = load_wrong_cases()
    cases: List[Dict[str, Any]] = data.get("cases", [])

    existing_idx = next(
        (i for i, c in enumerate(cases) if c.get("id") == case_id or (c.get("run_id") == run_id_clean and c.get("key") == key_clean)),
        None,
    )

    clean_status = status if status in {"open", "investigating", "resolved"} else "open"
    clean_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]

    if existing_idx is not None:
        case = dict(cases[existing_idx])
        case["id"] = case_id
        case["run_id"] = run_id_clean
        case["key"] = key_clean
        case["document"] = document or case.get("document", "")
        if dataset_id:
            case["dataset_id"] = dataset_id
        if dataset_name:
            case["dataset_name"] = dataset_name
        if note:
            case["note"] = note
        if status:
            case["status"] = clean_status
        if tags is not None:
            case["tags"] = clean_tags
        case["snapshot"] = snapshot
        case["updated_at"] = now_iso
        cases[existing_idx] = case
    else:
        case = {
            "id": case_id,
            "run_id": run_id_clean,
            "key": key_clean,
            "document": document,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "status": clean_status,
            "note": note,
            "tags": clean_tags,
            "snapshot": snapshot,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        cases.insert(0, case)

    data["cases"] = cases
    save_wrong_cases(data)
    return case


def batch_add_wrong_cases(
    runs_root: Path,
    run_id: str,
    keys: List[str],
    *,
    note: str = "",
    status: str = "open",
) -> List[Dict[str, Any]]:
    added: List[Dict[str, Any]] = []
    for k in keys:
        k_clean = str(k or "").strip()
        if not k_clean:
            continue
        case = add_or_update_wrong_case(
            runs_root,
            run_id,
            k_clean,
            note=note,
            status=status,
        )
        added.append(case)
    return added


def delete_wrong_case(case_id: str) -> bool:
    data = load_wrong_cases()
    cases: List[Dict[str, Any]] = data.get("cases", [])
    initial_len = len(cases)
    cases = [c for c in cases if c.get("id") != case_id]
    if len(cases) == initial_len:
        return False
    data["cases"] = cases
    save_wrong_cases(data)
    return True


def update_wrong_case_status(
    case_id: str,
    *,
    status: Optional[str] = None,
    note: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    data = load_wrong_cases()
    cases: List[Dict[str, Any]] = data.get("cases", [])
    target: Optional[Dict[str, Any]] = None
    for case in cases:
        if case.get("id") == case_id:
            target = case
            break
    if not target:
        return None

    if status and status in {"open", "investigating", "resolved"}:
        target["status"] = status
    if note is not None:
        target["note"] = str(note)
    if tags is not None:
        target["tags"] = [str(t).strip() for t in tags if str(t).strip()]
    target["updated_at"] = kst_now_iso()

    save_wrong_cases(data)
    return target


def list_wrong_cases(
    *,
    run_id: Optional[str] = None,
    document: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    data = load_wrong_cases()
    all_cases: List[Dict[str, Any]] = data.get("cases", [])

    counts = {
        "total": len(all_cases),
        "open": sum(1 for c in all_cases if c.get("status") == "open"),
        "investigating": sum(1 for c in all_cases if c.get("status") == "investigating"),
        "resolved": sum(1 for c in all_cases if c.get("status") == "resolved"),
    }

    filtered = all_cases
    if run_id:
        filtered = [c for c in filtered if c.get("run_id") == run_id]
    if document:
        doc_lower = document.lower()
        filtered = [c for c in filtered if doc_lower in (c.get("document") or "").lower()]
    if status and status != "all":
        filtered = [c for c in filtered if c.get("status") == status]
    if search:
        s_lower = search.lower()
        filtered = [
            c for c in filtered
            if s_lower in (c.get("key") or "").lower()
            or s_lower in (c.get("document") or "").lower()
            or s_lower in (c.get("note") or "").lower()
            or s_lower in (c.get("run_id") or "").lower()
        ]

    return {
        "cases": filtered,
        "total": len(filtered),
        "counts": counts,
        "path": str(wrong_cases_path()),
    }


def find_all_document_runs(runs_root: Path, document: str) -> List[Dict[str, Any]]:
    """Scan runs_root for runs belonging to the specified document, sorted newest first."""
    if not document or not runs_root.is_dir():
        return []

    doc_norm = document.strip().lower()
    matched: List[Dict[str, Any]] = []

    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        meta = _read_json(meta_file) or {}
        run_doc = meta.get("source_filename") or infer_run_document(child) or ""
        if run_doc.strip().lower() != doc_norm:
            continue
        started_at = meta.get("started_at")
        matched.append({
            "run_id": child.name,
            "run_dir": child,
            "started_at": started_at,
            "dt": _parse_ts(started_at) or datetime.fromtimestamp(child.stat().st_mtime, tz=KST),
            "status": meta.get("status", "unknown"),
        })

    matched.sort(key=lambda x: x["dt"], reverse=True)
    return matched


def get_wrong_case_detail(runs_root: Path, case_id: str) -> Optional[Dict[str, Any]]:
    """Get single wrong case with run history and latest progress comparison."""
    data = load_wrong_cases()
    cases: List[Dict[str, Any]] = data.get("cases", [])
    case = next((c for c in cases if c.get("id") == case_id), None)
    if not case:
        return None

    document = case.get("document") or ""
    key = case.get("key") or ""
    initial_run_id = case.get("run_id")
    initial_em = (case.get("snapshot") or {}).get("exact_match")

    doc_runs = find_all_document_runs(runs_root, document)

    history: List[Dict[str, Any]] = []
    latest_run_info: Optional[Dict[str, Any]] = None

    for r in doc_runs:
        rid = r["run_id"]
        rdir = r["run_dir"]
        ev = _load_run_eval_safe(rdir, run_id=rid)
        key_eval = _find_key_in_eval(ev, key)
        val = (key_eval or {}).get("value") or {}
        sp = (key_eval or {}).get("search_pages") or {}
        et = (key_eval or {}).get("evidence_text") or {}
        aes = read_agentic_evals(rdir).get(key)

        entry = {
            "run_id": rid,
            "started_at": r["started_at"],
            "status": r["status"],
            "is_initial": (rid == initial_run_id),
            "found_key": key_eval is not None,
            "pred": val.get("pred"),
            "gold": val.get("gold"),
            "exact_match": val.get("exact_match"),
            "pred_pages": sp.get("pred") if isinstance(sp.get("pred"), list) else [],
            "gold_pages": sp.get("gold") if isinstance(sp.get("gold"), list) else [],
            "vlm_reason": et.get("pred") or "",
            "search_reasons": (key_eval or {}).get("search_reasons", {}).get("pred") or (key_eval or {}).get("reason") or "",
            "agentic_verdict": aes.get("is_correct_answer") if isinstance(aes, dict) else None,
            "agentic_summary": (aes.get("reason_summary") or aes.get("reason")) if isinstance(aes, dict) else None,
            "agentic_detail": (aes.get("reason_detail") or aes.get("text")) if isinstance(aes, dict) else None,
        }
        history.append(entry)

        if latest_run_info is None and key_eval is not None:
            improved = False
            if initial_em is False and val.get("exact_match") is True:
                improved = True
            elif initial_em is False and val.get("exact_match") is False:
                # Check if agentic verdict improved
                initial_av = (case.get("snapshot") or {}).get("agentic_eval", {}).get("is_correct_answer")
                curr_av = entry.get("agentic_verdict")
                if initial_av == "incorrect" and curr_av == "correct":
                    improved = True

            latest_run_info = {
                "run_id": rid,
                "started_at": r["started_at"],
                "is_same_as_initial": (rid == initial_run_id),
                "exact_match": val.get("exact_match"),
                "pred": val.get("pred"),
                "gold": val.get("gold"),
                "pred_pages": entry["pred_pages"],
                "gold_pages": entry["gold_pages"],
                "vlm_reason": entry["vlm_reason"],
                "search_reasons": entry["search_reasons"],
                "agentic_verdict": entry["agentic_verdict"],
                "agentic_summary": entry["agentic_summary"],
                "improved": improved,
            }

    return {
        "case": case,
        "latest_run": latest_run_info,
        "history": history,
        "total_document_runs": len(doc_runs),
    }
