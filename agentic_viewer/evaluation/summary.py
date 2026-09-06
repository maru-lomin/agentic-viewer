"""Build cross-run evaluation summaries from baseline eval + agentic eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
from agentic_viewer.evaluation.live_progress import format_live_progress, read_eval_live_progress


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
        if status.get("status") == "running":
            # Running status always wins over stale cancelled/error result files.
            by_key[key] = status
        elif key not in by_key:
            by_key[key] = status
    return by_key


def _agentic_cell(
    by_key: Dict[str, Any],
    key: str,
    *,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ae = by_key.get(key)
    if not ae:
        return {"status": "pending"}
    status = str(ae.get("status") or "pending")
    summary = str(ae.get("reason_summary") or ae.get("reason") or "")
    detail = str(ae.get("reason_detail") or ae.get("text") or "")
    if "평가가 완료되지 않았습니다" in summary or "submit_evaluation을 호출하지 않았습니다" in detail:
        return {
            "status": "error",
            "error": "평가 미완료",
            "reason_summary": summary,
        }
    if status == "done" or ae.get("is_correct_answer") or ae.get("is_valid_gold"):
        verdict = str(ae.get("is_correct_answer") or "").lower()
        gold_verdict = str(ae.get("is_valid_gold") or "").lower()
        return {
            "status": "done",
            "is_correct_answer": verdict or None,
            "is_valid_gold": gold_verdict or None,
            "reason_summary": summary,
        }
    if status == "error":
        return {"status": "error", "error": ae.get("error") or "error"}
    if status == "cancelled":
        return {"status": "error", "error": ae.get("error") or "cancelled"}
    if status == "running":
        cell: Dict[str, Any] = {"status": "running"}
        if run_dir is not None:
            live = read_eval_live_progress(run_dir, key)
            if live:
                cell["live"] = live
                cell["live_label"] = format_live_progress(live)
        return cell
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
    n_gold_valid = 0
    n_gold_invalid = 0
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
        summary = str(ae.get("reason_summary") or ae.get("reason") or "")
        detail = str(ae.get("reason_detail") or ae.get("text") or "")
        if "평가가 완료되지 않았습니다" in summary or "submit_evaluation을 호출하지 않았습니다" in detail:
            n_error += 1
            continue
        if status == "done" or ae.get("is_correct_answer") or ae.get("is_valid_gold"):
            n_done += 1
            verdict = str(ae.get("is_correct_answer") or "").lower()
            if verdict == "correct":
                n_correct += 1
            elif verdict == "incorrect":
                n_incorrect += 1
            gold_verdict = str(ae.get("is_valid_gold") or "").lower()
            if gold_verdict == "valid":
                n_gold_valid += 1
            elif gold_verdict == "invalid":
                n_gold_invalid += 1

    n_pending = max(0, n_total - n_done - n_error - n_running)
    judged = n_correct + n_incorrect
    accuracy = round(n_correct / judged, 6) if judged else None
    gold_judged = n_gold_valid + n_gold_invalid
    gold_validity = round(n_gold_valid / gold_judged, 6) if gold_judged else None

    return {
        "n_total": n_total,
        "n_done": n_done,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_gold_valid": n_gold_valid,
        "n_gold_invalid": n_gold_invalid,
        "n_error": n_error,
        "n_running": n_running,
        "n_pending": n_pending,
        "accuracy": accuracy,
        "gold_validity": gold_validity,
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
            "average": None,
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
                "agentic": _agentic_cell(by_key_agentic, key, run_dir=run_dir),
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

    run_has_baseline = {r["run_id"]: bool(r.get("has_baseline_eval")) for r in per_run}
    for row in per_key:
        by_run = row.get("by_run") or {}
        n_correct = 0
        n_incorrect = 0
        for r_id in run_ids:
            if not run_has_baseline.get(r_id):
                continue
            cell = by_run.get(r_id)
            if not cell or "baseline_em" not in cell:
                continue
            if cell.get("baseline_em"):
                n_correct += 1
            else:
                n_incorrect += 1
        total = n_correct + n_incorrect
        row["overall"] = {
            "correct": n_correct,
            "incorrect": n_incorrect,
            "total": total,
            "rate": round(n_correct / total, 6) if total > 0 else None,
        }

    baseline_runs = [
        r for r in per_run if r.get("has_baseline_eval") and r.get("baseline")
    ]
    em_vals = [
        r["baseline"]["value_exact_match"]
        for r in baseline_runs
        if r["baseline"].get("value_exact_match") is not None
    ]
    page_f1_vals = [
        r["baseline"]["page_f1_macro"]
        for r in baseline_runs
        if r["baseline"].get("page_f1_macro") is not None
    ]
    evid_f1_vals = [
        r["baseline"]["evidence_token_f1"]
        for r in baseline_runs
        if r["baseline"].get("evidence_token_f1") is not None
    ]

    agentic_runs = [r for r in per_run if r.get("agentic")]
    acc_vals = [
        r["agentic"]["accuracy"]
        for r in agentic_runs
        if r["agentic"].get("accuracy") is not None
    ]
    gv_vals = [
        r["agentic"]["gold_validity"]
        for r in agentic_runs
        if r["agentic"].get("gold_validity") is not None
    ]
    total_done = sum(r.get("agentic", {}).get("n_done", 0) for r in per_run)
    total_keys = sum(r.get("agentic", {}).get("n_total", 0) for r in per_run)
    n_runs = len(per_run)

    average_metrics = {
        "value_exact_match": round(sum(em_vals) / len(em_vals), 6) if em_vals else None,
        "page_f1_macro": round(sum(page_f1_vals) / len(page_f1_vals), 6) if page_f1_vals else None,
        "evidence_token_f1": round(sum(evid_f1_vals) / len(evid_f1_vals), 6) if evid_f1_vals else None,
        "agentic_done_avg": round(total_done / n_runs, 2) if n_runs > 0 else 0,
        "agentic_total_avg": round(total_keys / n_runs, 2) if n_runs > 0 else 0,
        "agentic_done_total": total_done,
        "agentic_total_total": total_keys,
        "accuracy": round(sum(acc_vals) / len(acc_vals), 6) if acc_vals else None,
        "gold_validity": round(sum(gv_vals) / len(gv_vals), 6) if gv_vals else None,
    }

    return {
        "run_ids": list(run_ids),
        "documents": unique_docs,
        "document_warning": document_warning,
        "per_run": per_run,
        "average": average_metrics,
        "per_key": per_key,
        "keys": keys,
    }
