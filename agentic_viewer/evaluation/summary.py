"""Build cross-run evaluation summaries from baseline eval + agentic eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agentic_viewer.evaluation.baseline import load_or_compute_run_eval


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_agentic_evals(run_dir: Path) -> Dict[str, Any]:
    """Load per-key agentic eval payloads from ``06_agentic_eval/``."""
    out_dir = run_dir / "06_agentic_eval"
    by_key: Dict[str, Any] = {}
    if not out_dir.is_dir():
        return by_key
    for path in sorted(out_dir.glob("*.json")):
        if path.name.endswith(".status.json"):
            continue
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        key = data.get("key")
        if not key:
            continue
        by_key[str(key)] = data
    for path in out_dir.glob("*.status.json"):
        status = _read_json(path)
        if not isinstance(status, dict):
            continue
        key = status.get("key")
        if not key:
            continue
        key = str(key)
        if key not in by_key and status.get("status") == "running":
            by_key[key] = status
    return by_key


def _agentic_cell(by_key: Dict[str, Any], key: str) -> Dict[str, Any]:
    ae = by_key.get(key)
    if not ae:
        return {"status": "pending"}
    status = str(ae.get("status") or "pending")
    if status == "done" or ae.get("is_correct_answer"):
        verdict = str(ae.get("is_correct_answer") or "").lower()
        return {
            "status": "done",
            "is_correct_answer": verdict or None,
            "reason_summary": ae.get("reason_summary") or ae.get("reason") or "",
        }
    if status == "error":
        return {"status": "error", "error": ae.get("error") or "error"}
    if status == "cancelled":
        return {"status": "error", "error": ae.get("error") or "cancelled"}
    if status == "running":
        return {"status": "running"}
    return {"status": status}


def agentic_eval_summary(
    by_key: Dict[str, Any],
    gold_keys: Sequence[str],
) -> Dict[str, Any]:
    """Aggregate agentic-eval counts for one run."""
    n_total = len(gold_keys)
    n_done = 0
    n_correct = 0
    n_incorrect = 0
    n_error = 0
    n_running = 0

    for key in gold_keys:
        ae = by_key.get(key)
        if not ae:
            continue
        status = str(ae.get("status") or "")
        if status == "running":
            n_running += 1
            continue
        if status == "error":
            n_error += 1
            continue
        if status == "done" or ae.get("is_correct_answer"):
            n_done += 1
            verdict = str(ae.get("is_correct_answer") or "").lower()
            if verdict == "correct":
                n_correct += 1
            elif verdict == "incorrect":
                n_incorrect += 1

    n_pending = max(0, n_total - n_done - n_error - n_running)
    judged = n_correct + n_incorrect
    accuracy = round(n_correct / judged, 6) if judged else None

    return {
        "n_total": n_total,
        "n_done": n_done,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_error": n_error,
        "n_running": n_running,
        "n_pending": n_pending,
        "accuracy": accuracy,
    }


def _baseline_from_eval_report(
    report: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict) or not report.get("overall"):
        return None
    overall = report["overall"]
    return {
        "value_exact_match": overall.get("value_exact_match"),
        "page_f1_macro": overall.get("page_f1_macro"),
        "evidence_token_f1": overall.get("evidence_token_f1"),
        "n_keys": report.get("n_keys"),
        "document": report.get("document"),
    }


def build_evaluation_summary(
    run_ids: Sequence[str],
    runs_root: Path,
) -> Dict[str, Any]:
    """
    Combine baseline eval (05_eval.json or computed from 04_result.json) and
    agentic eval (06_agentic_eval/) for multiple runs.
    """
    runs_root = runs_root.resolve()
    if not run_ids:
        return {
            "run_ids": [],
            "documents": [],
            "document_warning": None,
            "per_run": [],
            "per_key": [],
            "keys": [],
        }

    per_run: List[Dict[str, Any]] = []
    documents: List[str] = []
    per_key_by_name: Dict[str, Dict[str, Any]] = {}

    for run_id in run_ids:
        run_dir = (runs_root / run_id).resolve()
        if not str(run_dir).startswith(str(runs_root)) or not run_dir.is_dir():
            raise KeyError(f"run not found: {run_id}")

        eval_report = load_or_compute_run_eval(
            run_dir, run_id=run_id, write_cache=True
        )
        baseline = _baseline_from_eval_report(
            eval_report if isinstance(eval_report, dict) else None
        )
        document = (baseline or {}).get("document")

        by_key_agentic = read_agentic_evals(run_dir)
        gold_keys: List[str] = []
        per_key_rows: Dict[str, Dict[str, Any]] = {}
        if isinstance(eval_report, dict):
            document = eval_report.get("document") or document
            for row in eval_report.get("per_key") or []:
                if not isinstance(row, dict) or "key" not in row:
                    continue
                key = str(row["key"])
                gold_keys.append(key)
                per_key_rows[key] = row

        if document:
            documents.append(str(document))

        agentic = agentic_eval_summary(by_key_agentic, gold_keys)

        per_run.append(
            {
                "run_id": run_id,
                "document": document,
                "has_baseline_eval": baseline is not None,
                "baseline": baseline,
                "agentic": agentic,
            }
        )

        for key in gold_keys:
            row = per_key_rows.get(key) or {}
            value = row.get("value") or {}
            entry = per_key_by_name.setdefault(
                key,
                {
                    "key": key,
                    "gold_value": value.get("gold"),
                    "by_run": {},
                },
            )
            if entry.get("gold_value") is None and value.get("gold") is not None:
                entry["gold_value"] = value.get("gold")
            entry["by_run"][run_id] = {
                "baseline_em": bool(value.get("exact_match")),
                "page_f1": (row.get("search_pages") or {}).get("f1"),
                "evidence_f1": (row.get("evidence_text") or {}).get("token_f1"),
                "pred_value": value.get("pred"),
                "agentic": _agentic_cell(by_key_agentic, key),
            }

    unique_docs = sorted(set(documents))
    document_warning = None
    if len(unique_docs) > 1:
        document_warning = (
            "Selected runs span multiple documents; compare metrics only within "
            f"the same document. Found: {', '.join(unique_docs)}"
        )

    per_key = [per_key_by_name[k] for k in sorted(per_key_by_name)]
    keys = [row["key"] for row in per_key]

    return {
        "run_ids": list(run_ids),
        "documents": unique_docs,
        "document_warning": document_warning,
        "per_run": per_run,
        "per_key": per_key,
        "keys": keys,
    }
