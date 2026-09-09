"""Evaluation API router and evaluation page."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse, Response

from agentic_viewer.evaluation.agentic_client import (
    AgenticEvalError,
    invoke_agentic_eval,
)
from agentic_viewer.evaluation.batch import enrich_batch_job_dict
from agentic_viewer.evaluation.status_cleanup import (
    cleanup_all_running_eval_statuses,
    mark_running_eval_status_cancelled,
)
from agentic_viewer.evaluation.summary import (
    build_evaluation_summary,
    read_agentic_evals,
)
from agentic_viewer.evaluation.trace_paths import (
    list_agentic_eval_keys,
    resolve_agentic_eval_trace_dir,
)
from agentic_viewer.evaluation.xlsx_export import generate_evaluation_xlsx
from agentic_viewer.evaluation_page import EVALUATION_HTML
from agentic_viewer.hierarchy import build_agent_tree
from agentic_viewer.timezone import kst_now
from agentic_viewer.timing import attach_timing_to_tree, build_timing_report

router = APIRouter()


def _get_app():
    from agentic_viewer import app as app_mod
    return app_mod


@router.get("/evaluation", response_class=HTMLResponse)
def evaluation_page() -> str:
    return EVALUATION_HTML


@router.get("/api/evaluation/summary")
@router.get("/api/evaluations/summary")
def get_evaluation_summary(run_ids: str = "") -> Dict[str, Any]:
    """Aggregate cached baseline + agentic eval across multiple runs."""
    app_mod = _get_app()
    runs_root = app_mod.RUNS_ROOT
    ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    if not ids:
        return build_evaluation_summary([], runs_root)
    try:
        return build_evaluation_summary(ids, runs_root)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/evaluation/batch-jobs/active")
@router.get("/api/batch-agentic-eval/active")
def get_active_batch_job() -> Dict[str, Any]:
    app_mod = _get_app()
    job = app_mod._BATCH_MANAGER.get_active_job()
    if job is None:
        return {"active": False, "job": None}
    return {
        "active": True,
        "job": enrich_batch_job_dict(job.to_dict(), app_mod.RUNS_ROOT),
    }


@router.get("/api/evaluation/batch-jobs/{job_id}")
@router.get("/api/batch-agentic-eval/{job_id}")
def get_batch_job(job_id: str) -> Dict[str, Any]:
    app_mod = _get_app()
    job = app_mod._BATCH_MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return enrich_batch_job_dict(job.to_dict(), app_mod.RUNS_ROOT)


@router.post("/api/evaluation/batch-agentic-eval")
@router.post("/api/batch-agentic-eval")
def post_batch_agentic_eval(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Run agentic-evaluation for all keys across selected runs (background job)."""
    app_mod = _get_app()
    run_ids = body.get("run_ids") or []
    if not isinstance(run_ids, list):
        raise HTTPException(status_code=400, detail="run_ids must be a list")
    skip_existing = bool(body.get("skip_existing", True))
    try:
        job = app_mod._BATCH_MANAGER.start(run_ids, skip_existing=skip_existing)
        return enrich_batch_job_dict(job.to_dict(), app_mod.RUNS_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/evaluation/batch-jobs/{job_id}/cancel")
@router.post("/api/batch-agentic-eval/{job_id}/cancel")
def cancel_batch_job(job_id: str) -> Dict[str, Any]:
    app_mod = _get_app()
    try:
        job = app_mod._BATCH_MANAGER.cancel(job_id)
        return enrich_batch_job_dict(job.to_dict(), app_mod.RUNS_ROOT)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/evaluation/cleanup-stale")
@router.post("/api/cleanup-all-stale-eval")
def post_cleanup_all_stale_eval(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Clean up stale running evaluation status files across runs."""
    app_mod = _get_app()
    run_ids = body.get("run_ids")
    if run_ids is not None and not isinstance(run_ids, list):
        raise HTTPException(status_code=400, detail="run_ids must be a list of strings")
    reason = str(body.get("reason") or "cancelled by user (stale cleanup)")
    cleaned = cleanup_all_running_eval_statuses(app_mod.RUNS_ROOT, run_ids=run_ids, reason=reason)
    total = sum(cleaned.values())
    return {"cleaned_by_run": cleaned, "total_cleaned": total}


@router.post("/api/runs/{run_id}/agentic-eval")
def post_agentic_eval(run_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Trigger EvalMasterAgent for one key via the inference-pipeline API."""
    app_mod = _get_app()
    app_mod._run_dir(run_id)
    key = str((body or {}).get("key") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    active = app_mod._BATCH_MANAGER.get_active_job()
    if active and active.status == "running" and run_id in active.run_ids:
        raise HTTPException(
            status_code=409,
            detail=f"batch agentic-evaluation job {active.job_id} is running for this run",
        )

    inflight = app_mod._AGENTIC_EVAL_INFLIGHT.setdefault(run_id, set())
    inflight.add(key)
    try:
        result = app_mod._call_inference_agentic_eval(run_id, key)
        return result
    finally:
        inflight.discard(key)
        if not inflight:
            app_mod._AGENTIC_EVAL_INFLIGHT.pop(run_id, None)


@router.post("/api/runs/{run_id}/cleanup-stale-eval")
@router.post("/api/runs/{run_id}/agentic-eval/cleanup-stale")
def post_cleanup_stale_eval(
    run_id: str, body: Dict[str, Any] = Body(default={})
) -> Dict[str, Any]:
    """Clean up stale running status for a specific run or key."""
    app_mod = _get_app()
    run_dir = app_mod._run_dir(run_id)
    key = body.get("key")
    reason = str(body.get("reason") or "cancelled by user (stale cleanup)")
    count = mark_running_eval_status_cancelled(run_dir, reason=reason, key=key)
    if key:
        inflight = app_mod._AGENTIC_EVAL_INFLIGHT.get(run_id)
        if inflight:
            inflight.discard(key)
            if not inflight:
                app_mod._AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
    else:
        app_mod._AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
    return {"run_id": run_id, "cleaned": count}


@router.get("/api/runs/{run_id}/eval")
def get_eval(run_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Score 04_result.json against dataset/answer_sheet.json; cache as 05_eval.json."""
    app_mod = _get_app()
    return app_mod._compute_run_eval(run_id, refresh=refresh)


@router.get("/api/runs/{run_id}/agentic-eval")
@router.get("/api/runs/{run_id}/agentic-evals")
def get_agentic_evals(run_id: str) -> Dict[str, Any]:
    """List cached per-key agentic-evaluation results under 06_agentic_eval/."""
    app_mod = _get_app()
    app_mod._run_dir(run_id)
    return app_mod._list_agentic_evals(run_id)


@router.get("/api/runs/{run_id}/agentic-eval/keys")
@router.get("/api/runs/{run_id}/agentic-eval-keys")
def list_agentic_eval_keys_api(run_id: str) -> Dict[str, Any]:
    """Per-key agentic-eval status + trace availability for one extraction run."""
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    return {"run_id": run_id, "keys": list_agentic_eval_keys(root)}


@router.get("/api/runs/{run_id}/agentic-eval/{key}/agent-tree")
@router.get("/api/runs/{run_id}/agentic-eval-tree")
def get_agentic_eval_tree(run_id: str, key: str = "") -> Dict[str, Any]:
    """EvalMaster → tools → SearchAgent tree for one key under 06_agentic_eval/."""
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    try:
        trace_dir = resolve_agentic_eval_trace_dir(root, key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tree = build_agent_tree(trace_dir)
    tree["agent_kind"] = "eval"
    tree["eval_key"] = key
    tree["parent_run_id"] = run_id
    payload_path = trace_dir.parent / f"{trace_dir.name}.json"
    tree["eval_result"] = app_mod._read_json(payload_path) if payload_path.is_file() else None
    timing = build_timing_report(trace_dir)
    return attach_timing_to_tree(tree, timing)


@router.get("/api/runs/{run_id}/agentic-eval/{key}/file")
@router.get("/api/runs/{run_id}/agentic-eval-file")
def get_agentic_eval_file(run_id: str, key: str = "", path: str = ""):
    """Read a file under ``06_agentic_eval/<key>/`` (for hierarchy step dumps)."""
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    try:
        trace_dir = resolve_agentic_eval_trace_dir(root, key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="invalid path")
    return app_mod._serve_run_file(trace_dir.resolve(), str(rel))


def _export_runs_to_xlsx_response(run_ids: List[str]) -> Response:
    app_mod = _get_app()
    runs_root = app_mod.RUNS_ROOT
    clean_ids = [str(x).strip() for x in run_ids if str(x).strip()]
    if not clean_ids:
        raise HTTPException(status_code=400, detail="run_ids is required")

    run_dirs: List[Path] = []
    for rid in clean_ids:
        p = (runs_root / rid).resolve()
        if str(p).startswith(str(runs_root)) and p.is_dir():
            run_dirs.append(p)

    if not run_dirs:
        raise HTTPException(status_code=404, detail="No valid runs found for the provided run_ids")

    buf = generate_evaluation_xlsx(run_dirs)
    ts = kst_now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{ts}.xlsx"
    encoded_filename = quote(filename)
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded_filename}',
    }
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@router.get("/api/evaluation/export/xlsx")
def get_export_evaluation_xlsx(run_ids: str = "") -> Response:
    """Download evaluation and extraction results for selected runs as Excel (.xlsx)."""
    ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    return _export_runs_to_xlsx_response(ids)


@router.post("/api/evaluation/export/xlsx")
def post_export_evaluation_xlsx(body: Dict[str, Any] = Body(...)) -> Response:
    """Download evaluation and extraction results for selected runs as Excel (.xlsx) via POST."""
    ids = body.get("run_ids") or []
    if isinstance(ids, str):
        ids = [x.strip() for x in ids.split(",") if x.strip()]
    elif not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="run_ids must be a list of strings")
    return _export_runs_to_xlsx_response(ids)
