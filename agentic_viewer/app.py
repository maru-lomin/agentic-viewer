"""Lightweight FastAPI viewer for agentic run traces under outputs/runs/."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from agentic_viewer.datasets import DatasetStore
from agentic_viewer.datasets_page import DATASETS_HTML
from agentic_viewer.eval.paths import answer_sheet_path
from agentic_viewer.evaluation.agentic_client import (
    AgenticEvalError,
    invoke_agentic_eval,
    invoke_agentic_eval_chat,
    get_agentic_eval_chat,
    delete_agentic_eval_chat,
)
from agentic_viewer.evaluation.batch import enrich_batch_job_dict, make_batch_manager
from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
from agentic_viewer.evaluation.status_cleanup import (
    cleanup_all_running_eval_statuses,
    mark_running_eval_status_cancelled,
)
from agentic_viewer.evaluation.summary import (
    agentic_eval_summary,
    build_evaluation_summary,
    read_agentic_evals,
)
from agentic_viewer.evaluation.trace_paths import (
    list_agentic_eval_keys,
    resolve_agentic_eval_trace_dir,
)
from agentic_viewer.evaluation_page import EVALUATION_HTML
from agentic_viewer.ground_truth import (
    get_document_gt,
    import_answer_sheet,
    invalidate_eval_caches_for_document,
    list_documents,
    update_gt_key,
)
from agentic_viewer.ground_truth_page import GROUND_TRUTH_HTML
from agentic_viewer.hierarchy import build_agent_tree
from agentic_viewer.inference_jobs import make_inference_job_manager
from agentic_viewer.highlights import chunk_highlights, page_highlights
from agentic_viewer.image_tokens import replace_base64_images
from agentic_viewer.pdf_source import infer_pdf_path, infer_run_document, pdf_info
from agentic_viewer.timing import attach_timing_to_tree, build_timing_report
from agentic_viewer.prompts import (
    compute_diff,
    get_backup,
    get_prompt,
    list_backups,
    list_prompts,
    restore_prompt,
    save_prompt,
)
from agentic_viewer.prompts_page import PROMPTS_HTML
from agentic_viewer.wrong_cases import (
    add_or_update_wrong_case,
    batch_add_wrong_cases,
    delete_wrong_case,
    get_wrong_case_detail,
    list_wrong_cases,
    update_wrong_case_status,
)
from agentic_viewer.wrong_cases_page import WRONG_CASES_HTML

def default_runs_root() -> Path:
    """Prefer shared repo outputs/runs, else legacy inference-pipeline path."""
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    candidates = [
        repo_root / "outputs" / "runs",
        repo_root / "inference-pipeline" / "outputs" / "runs",
        Path.cwd() / "runs",
        here.parents[1] / "runs",
    ]
    env = os.environ.get("AGENTIC_RUNS_DIR")
    if env:
        return Path(env).resolve()
    for c in candidates:
        if c.is_dir() or c.parent.is_dir():
            c.mkdir(parents=True, exist_ok=True)
            return c.resolve()
    candidates[-1].mkdir(parents=True, exist_ok=True)
    return candidates[-1].resolve()


RUNS_ROOT = default_runs_root()
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://127.0.0.1:8010").rstrip(
    "/"
)
# run_id -> set of keys currently evaluating (up to AGENTIC_EVAL_MAX_PARALLEL per run)
_AGENTIC_EVAL_INFLIGHT: Dict[str, Set[str]] = {}
_BATCH_MANAGER = make_batch_manager(
    RUNS_ROOT,
    INFERENCE_API_URL,
    _AGENTIC_EVAL_INFLIGHT,
)
_INFERENCE_JOB_MANAGER = make_inference_job_manager(
    INFERENCE_API_URL,
    runs_root=RUNS_ROOT,
    batch_manager=_BATCH_MANAGER,
)
_DATASET_STORE = DatasetStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: clean up stale running statuses from previous crashes or ungraceful stops
    cleaned = cleanup_all_running_eval_statuses(RUNS_ROOT)
    if cleaned:
        total = sum(cleaned.values())
        print(
            f"[agentic-viewer] Startup cleanup: marked {total} stale eval task(s) cancelled across {len(cleaned)} run(s): {cleaned}"
        )
    yield


app = FastAPI(title="Agentic Run Trace Viewer", version="0.3.0", lifespan=lifespan)


def _list_agentic_evals(run_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    inflight = _AGENTIC_EVAL_INFLIGHT.get(run_id)
    return {
        "by_key": read_agentic_evals(root),
        "inflight": sorted(inflight) if inflight else None,
    }


def _call_inference_agentic_eval(run_id: str, key: str) -> Dict[str, Any]:
    active = _BATCH_MANAGER.get_active_job()
    if active and active.status == "running" and run_id in active.run_ids:
        raise HTTPException(
            status_code=409,
            detail=f"batch agentic-evaluation job {active.job_id} is running for this run",
        )
    try:
        return invoke_agentic_eval(INFERENCE_API_URL, run_id, key)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _run_dir(run_id: str) -> Path:
    path = (RUNS_ROOT / run_id).resolve()
    if not str(path).startswith(str(RUNS_ROOT)) or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return path


def _assert_run_deletable(run_id: str, force: bool = False) -> None:
    if force:
        _AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
        active_infer = _INFERENCE_JOB_MANAGER.get_active_job()
        if active_infer and active_infer.status in {"queued", "running"}:
            for task in active_infer.tasks:
                if task.run_id == run_id and task.status in {"pending", "running"}:
                    task.status = "cancelled"
                    task.error = "run deleted by user"
        active_batch = _BATCH_MANAGER.get_active_job()
        if (
            active_batch
            and active_batch.status in {"queued", "running"}
            and run_id in active_batch.run_ids
        ):
            active_batch.run_ids = [r for r in active_batch.run_ids if r != run_id]
        return

    inflight = _AGENTIC_EVAL_INFLIGHT.get(run_id)
    if inflight:
        raise HTTPException(
            status_code=409,
            detail=f"agentic-evaluation is in progress for run {run_id}",
        )

    active_batch = _BATCH_MANAGER.get_active_job()
    if (
        active_batch
        and active_batch.status in {"queued", "running"}
        and run_id in active_batch.run_ids
    ):
        raise HTTPException(
            status_code=409,
            detail=f"batch job {active_batch.job_id} is using this run",
        )

    active_infer = _INFERENCE_JOB_MANAGER.get_active_job()
    if active_infer and active_infer.status in {"queued", "running"}:
        for task in active_infer.tasks:
            if task.run_id == run_id and task.status in {"pending", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail="KV extraction is in progress for this run",
                )


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_summary(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict):
        return None
    overall = report.get("overall")
    if not isinstance(overall, dict):
        return None
    return {
        "value_exact_match": overall.get("value_exact_match"),
        "page_f1_macro": overall.get("page_f1_macro"),
        "evidence_token_f1": overall.get("evidence_token_f1"),
        "n_keys": report.get("n_keys"),
        "document": report.get("document"),
    }


def _compute_run_eval(run_id: str, *, refresh: bool = False) -> Dict[str, Any]:
    root = _run_dir(run_id)
    report = load_or_compute_run_eval(
        root, run_id=run_id, refresh=refresh, write_cache=True
    )
    if report is not None:
        return report

    pred_path = root / "04_result.json"
    if not _read_json(pred_path):
        raise HTTPException(status_code=404, detail="04_result.json not found")
    raise HTTPException(
        status_code=400,
        detail="could not build eval report (missing document name or predictions)",
    )


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        s = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            return dt.astimezone()
        return dt
    except Exception:
        return None


def _format_display_ts(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M")


def _enrich_run_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure every run has a well-defined run_group_id and run_group_name.
    If already stored in meta.json, preserve them.
    If missing, group by dataset_id and cluster consecutive runs separated by
    <= 30 minutes into execution batches (e.g. evaluation-v2-run-v1 (2026-09-06 11:08)).
    """
    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("run_group_id"):
            continue
        ds_id = r.get("dataset_id")
        if ds_id:
            by_ds.setdefault(ds_id, []).append(r)

    for ds_id, group_runs in by_ds.items():
        # Sort chronologically (oldest first) to assign v1, v2, ...
        def _sort_key(item: Dict[str, Any]) -> float:
            dt = _parse_ts(item.get("started_at"))
            return dt.timestamp() if dt else 0.0

        group_runs.sort(key=_sort_key)

        # Check existing max version for this dataset among runs with explicit run_group_id
        pattern = re.compile(rf"{re.escape(ds_id)}-run-v(\d+)", re.IGNORECASE)
        v_idx = 0
        for r in rows:
            gid = str(r.get("run_group_id") or "")
            m = pattern.search(gid)
            if m:
                try:
                    v_idx = max(v_idx, int(m.group(1)))
                except ValueError:
                    pass

        # Split into sessions by gap > 1800s (30m)
        sessions: List[List[Dict[str, Any]]] = []
        current_session: List[Dict[str, Any]] = []
        last_ts: Optional[float] = None

        for r in group_runs:
            dt = _parse_ts(r.get("started_at"))
            cur_ts = dt.timestamp() if dt else None
            if current_session and cur_ts is not None and last_ts is not None:
                if (cur_ts - last_ts) > 1800:
                    sessions.append(current_session)
                    current_session = []
            current_session.append(r)
            if cur_ts is not None:
                last_ts = cur_ts

        if current_session:
            sessions.append(current_session)

        for sess in sessions:
            v_idx += 1
            first_dt = None
            ds_name = ds_id
            for r in sess:
                if not first_dt:
                    first_dt = _parse_ts(r.get("started_at"))
                if r.get("dataset_name"):
                    ds_name = r["dataset_name"]
            display_time = _format_display_ts(first_dt) if first_dt else ""
            gid = f"{ds_id}-run-v{v_idx}"
            gname = f"{ds_name}-run-v{v_idx}" + (f" ({display_time})" if display_time else "")
            for r in sess:
                r["run_group_id"] = gid
                r["run_group_name"] = gname

    return rows


def _next_dataset_run_version(dataset_id: str) -> int:
    max_v = 0
    pattern = re.compile(rf"{re.escape(dataset_id)}-run-v(\d+)", re.IGNORECASE)
    runs = list_runs()
    for r in runs:
        if r.get("dataset_id") != dataset_id:
            continue
        gid = str(r.get("run_group_id") or "")
        m = pattern.search(gid)
        if m:
            try:
                max_v = max(max_v, int(m.group(1)))
            except ValueError:
                pass
    return max_v + 1


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for child in sorted(RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        meta = _read_json(child / "meta.json") or {}
        result = _read_json(child / "04_result.json") or {}
        eval_report = load_or_compute_run_eval(
            child, run_id=child.name, write_cache=True
        )
        gold_keys: List[str] = []
        if isinstance(eval_report, dict):
            for row in eval_report.get("per_key") or []:
                if isinstance(row, dict) and "key" in row:
                    gold_keys.append(str(row["key"]))
        agentic_by_key = read_agentic_evals(child)
        status = meta.get("status")
        if not status:
            status = "running" if not meta.get("finished_at") else "unknown"
        rows.append(
            {
                "run_id": child.name,
                "document": infer_run_document(
                    child, eval_report=eval_report, result=result
                ),
                "status": status,
                "started_at": meta.get("started_at"),
                "finished_at": meta.get("finished_at"),
                "seconds": meta.get("seconds"),
                "n_kv": len(result.get("kv_results") or []),
                "page_count": (result.get("meta") or {}).get("page_count"),
                "dataset_id": meta.get("dataset_id"),
                "dataset_name": meta.get("dataset_name"),
                "dataset_source": meta.get("dataset_source"),
                "run_group_id": meta.get("run_group_id"),
                "run_group_name": meta.get("run_group_name"),
                "source_filename": meta.get("source_filename"),
                "eval_summary": _eval_summary(eval_report),
                "agentic_eval_summary": agentic_eval_summary(
                    agentic_by_key, gold_keys
                ),
            }
        )
    return _enrich_run_groups(rows)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    result = _read_json(root / "04_result.json")
    return {
        "run_id": run_id,
        "document": infer_run_document(root, result=result),
        "meta": _read_json(root / "meta.json"),
        "request": _read_json(root / "00_request.json"),
        "parse_summary": _read_json(root / "01_parse" / "summary.json"),
        "chunk_summary": _read_json(root / "02_chunk" / "summary.json"),
        "result": result,
        "error": _read_json(root / "04_error.json"),
    }


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str, force: bool = Query(default=False)) -> Dict[str, Any]:
    """Remove a run directory under outputs/runs/."""
    root = _run_dir(run_id)
    _assert_run_deletable(run_id, force=force)
    shutil.rmtree(root)
    _AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
    return {"ok": True, "run_id": run_id}


@app.post("/api/runs/delete-batch")
def delete_runs_batch(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Remove multiple run directories under outputs/runs/."""
    run_ids = body.get("run_ids")
    if not isinstance(run_ids, list):
        raise HTTPException(status_code=400, detail="run_ids must be a list of strings")
    force = bool(body.get("force", False))
    deleted: List[str] = []
    errors: Dict[str, str] = {}
    for rid in run_ids:
        rid_str = str(rid).strip()
        if not rid_str:
            continue
        try:
            root = _run_dir(rid_str)
            _assert_run_deletable(rid_str, force=force)
            shutil.rmtree(root)
            _AGENTIC_EVAL_INFLIGHT.pop(rid_str, None)
            deleted.append(rid_str)
        except Exception as exc:
            errors[rid_str] = str(exc)
    return {"deleted": deleted, "errors": errors, "total_deleted": len(deleted)}


@app.get("/api/runs/{run_id}/timeline")
def get_timeline(run_id: str) -> List[Dict[str, Any]]:
    path = _run_dir(run_id) / "timeline.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@app.get("/api/runs/{run_id}/steps")
def list_steps(run_id: str) -> List[str]:
    agent_dir = _run_dir(run_id) / "03_agent"
    if not agent_dir.is_dir():
        return []
    return sorted(p.name for p in agent_dir.glob("step_*.json"))


@app.get("/api/runs/{run_id}/steps/detail")
def list_steps_detail(run_id: str) -> List[Dict[str, Any]]:
    """
    Step dumps for the visualize tab.
    Omits messages_after (redundant with request + assistant + tools).
    """
    agent_dir = _run_dir(run_id) / "03_agent"
    if not agent_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(agent_dir.glob("step_*.json")):
        data = _read_json(path) or {}
        # Drop redundant full-history snapshot.
        data.pop("messages_after", None)
        data.pop("messages", None)  # legacy
        data["filename"] = path.name
        # Soften huge tool schemas for the UI list (full tools still available in file).
        tools = data.get("tools") or []
        data["tool_names"] = [
            (t.get("function") or {}).get("name")
            for t in tools
            if isinstance(t, dict)
        ]
        rows.append(data)
    return rows


@app.get("/api/runs/{run_id}/agent-tree")
def get_agent_tree(run_id: str) -> Dict[str, Any]:
    """Hierarchical Master → search_pages → SearchAgent sessions tree."""
    root = _run_dir(run_id)
    tree = build_agent_tree(root)
    timing = build_timing_report(root)
    return attach_timing_to_tree(tree, timing)


@app.get("/api/runs/{run_id}/agentic-eval/keys")
def list_agentic_eval_keys_api(run_id: str) -> Dict[str, Any]:
    """Per-key agentic-eval status + trace availability for one extraction run."""
    root = _run_dir(run_id)
    return {"run_id": run_id, "keys": list_agentic_eval_keys(root)}


@app.get("/api/runs/{run_id}/agentic-eval/{key}/agent-tree")
def get_agentic_eval_tree(run_id: str, key: str) -> Dict[str, Any]:
    """EvalMaster → tools → SearchAgent tree for one key under 06_agentic_eval/."""
    root = _run_dir(run_id)
    try:
        trace_dir = resolve_agentic_eval_trace_dir(root, key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tree = build_agent_tree(trace_dir)
    tree["agent_kind"] = "eval"
    tree["eval_key"] = key
    tree["parent_run_id"] = run_id
    payload_path = trace_dir.parent / f"{trace_dir.name}.json"
    tree["eval_result"] = _read_json(payload_path) if payload_path.is_file() else None
    timing = build_timing_report(trace_dir)
    return attach_timing_to_tree(tree, timing)


@app.get("/api/runs/{run_id}/agentic-eval/{key}/file")
def get_agentic_eval_file(run_id: str, key: str, path: str):
    """Read a file under ``06_agentic_eval/<key>/`` (for hierarchy step dumps)."""
    root = _run_dir(run_id)
    try:
        trace_dir = resolve_agentic_eval_trace_dir(root, key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=400, detail="invalid path")
    return _serve_run_file(trace_dir.resolve(), str(rel))


@app.get("/api/runs/{run_id}/timing")
def get_timing(run_id: str) -> Dict[str, Any]:
    """Agent / session / turn timing derived from timeline.jsonl."""
    return build_timing_report(_run_dir(run_id))


@app.get("/api/runs/{run_id}/eval")
def get_eval(run_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Score 04_result.json against dataset/answer_sheet.json; cache as 05_eval.json."""
    return _compute_run_eval(run_id, refresh=refresh)


@app.get("/api/runs/{run_id}/agentic-eval")
def get_agentic_evals(run_id: str) -> Dict[str, Any]:
    """List cached per-key agentic-evaluation results under 06_agentic_eval/."""
    _run_dir(run_id)
    return _list_agentic_evals(run_id)


@app.get("/api/ground-truth")
def get_ground_truth_index() -> Dict[str, Any]:
    try:
        documents = list_documents()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "path": str(answer_sheet_path()),
        "documents": documents,
    }


@app.get("/api/ground-truth/document")
def get_ground_truth_document(document: str) -> Dict[str, Any]:
    try:
        return get_document_gt(document)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/ground-truth/key")
@app.post("/api/ground-truth/key")
def put_ground_truth_key(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    document = str(body.get("document") or "").strip()
    key = str(body.get("key") or "").strip()
    if not document or not key:
        raise HTTPException(status_code=400, detail="document and key are required")
    entry = {
        "value": body.get("value"),
        "evidences": body.get("evidences"),
        "evidence_pages": body.get("evidence_pages"),
    }
    try:
        result = update_gt_key(document, key, entry)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    invalidated = invalidate_eval_caches_for_document(RUNS_ROOT, document)
    return {**result, "invalidated_eval_caches": invalidated}


@app.post("/api/ground-truth/upload")
async def upload_ground_truth(
    file: UploadFile = File(...),
    mode: str = Form("merge"),
) -> Dict[str, Any]:
    """
    Upload an answer_sheet.json file and merge/replace ground-truth entries.

    Expected shape matches dataset/answer_sheet.json:
      { "<document>.pdf": { "<key>": {value, evidences, evidence_pages}, ... }, ... }
    """
    name = (file.filename or "").strip() or "answer_sheet.json"
    if not name.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="only .json files are supported")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
    try:
        result = import_answer_sheet(payload, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    invalidated = 0
    for document in result.get("documents") or []:
        invalidated += invalidate_eval_caches_for_document(RUNS_ROOT, str(document))
    return {
        **result,
        "filename": name,
        "invalidated_eval_caches": invalidated,
        "documents_index": list_documents(),
    }


def _raise_dataset_error(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, FileExistsError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise exc


async def _read_pdf_uploads(files: List[UploadFile]) -> List[tuple[str, bytes]]:
    payload: List[tuple[str, bytes]] = []
    for upload in files:
        name = (upload.filename or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="each file must have a filename")
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty file: {name}")
        payload.append((name, data))
    return payload


@app.get("/api/datasets")
def list_datasets() -> Dict[str, Any]:
    return {"datasets": _DATASET_STORE.list_datasets()}


@app.post("/api/datasets")
async def create_dataset(
    name: str = Form(...),
    files: List[UploadFile] = File(default=[]),
) -> Dict[str, Any]:
    payload = await _read_pdf_uploads(files) if files else []
    try:
        return _DATASET_STORE.create(name, payload)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@app.get("/api/datasets/{source}/{dataset_id}")
def get_dataset(source: str, dataset_id: str) -> Dict[str, Any]:
    try:
        return _DATASET_STORE.get(source, dataset_id)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@app.post("/api/datasets/{source}/{dataset_id}/files")
async def add_dataset_files(
    source: str,
    dataset_id: str,
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    payload = await _read_pdf_uploads(files)
    try:
        return _DATASET_STORE.add_files(source, dataset_id, payload)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@app.delete("/api/datasets/{source}/{dataset_id}/files/{filename}")
def delete_dataset_file(source: str, dataset_id: str, filename: str) -> Dict[str, Any]:
    try:
        return _DATASET_STORE.delete_file(source, dataset_id, filename)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@app.delete("/api/datasets/{source}/{dataset_id}")
def delete_dataset(source: str, dataset_id: str) -> Dict[str, Any]:
    try:
        return _DATASET_STORE.delete_dataset(source, dataset_id)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@app.get("/api/evaluation/summary")
def get_evaluation_summary(run_ids: str = "") -> Dict[str, Any]:
    """Aggregate cached baseline + agentic eval across multiple runs."""
    ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    if not ids:
        return build_evaluation_summary([], RUNS_ROOT)
    try:
        return build_evaluation_summary(ids, RUNS_ROOT)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/evaluation/batch-jobs/active")
def get_active_batch_job() -> Dict[str, Any]:
    job = _BATCH_MANAGER.get_active_job()
    if job is None:
        return {"active": False, "job": None}
    return {
        "active": True,
        "job": enrich_batch_job_dict(job.to_dict(), RUNS_ROOT),
    }


@app.get("/api/evaluation/batch-jobs/{job_id}")
def get_batch_job(job_id: str) -> Dict[str, Any]:
    job = _BATCH_MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return enrich_batch_job_dict(job.to_dict(), RUNS_ROOT)


@app.post("/api/evaluation/batch-agentic-eval")
def post_batch_agentic_eval(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Run agentic-evaluation for all keys across selected runs (background job).

    Body: {run_ids: [...], skip_existing?: true}
    """
    run_ids = body.get("run_ids") or []
    if not isinstance(run_ids, list):
        raise HTTPException(status_code=400, detail="run_ids must be a list")
    skip_existing = bool(body.get("skip_existing", True))
    try:
        job = _BATCH_MANAGER.start(run_ids, skip_existing=skip_existing)
        return enrich_batch_job_dict(job.to_dict(), RUNS_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/evaluation/batch-jobs/{job_id}/cancel")
def cancel_batch_job(job_id: str) -> Dict[str, Any]:
    try:
        job = _BATCH_MANAGER.cancel(job_id)
        return enrich_batch_job_dict(job.to_dict(), RUNS_ROOT)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/evaluation/cleanup-stale")
def post_cleanup_all_stale_eval(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Clean up stale running evaluation status files across runs."""
    run_ids = body.get("run_ids")
    if run_ids is not None and not isinstance(run_ids, list):
        raise HTTPException(status_code=400, detail="run_ids must be a list of strings")
    reason = str(body.get("reason") or "cancelled by user (stale cleanup)")
    cleaned = cleanup_all_running_eval_statuses(RUNS_ROOT, run_ids=run_ids, reason=reason)
    total = sum(cleaned.values())
    return {"cleaned_by_run": cleaned, "total_cleaned": total}


@app.get("/api/inference/jobs/active")
def get_active_inference_job() -> Dict[str, Any]:
    job = _INFERENCE_JOB_MANAGER.get_active_job()
    return {"job": job.to_dict() if job else None}


@app.get("/api/inference/jobs/{job_id}")
def get_inference_job(job_id: str) -> Dict[str, Any]:
    job = _INFERENCE_JOB_MANAGER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return job.to_dict()


@app.post("/api/inference/jobs")
async def post_inference_job(
    files: List[UploadFile] = File(...),
    hooks: str = Form("agentic_config"),
    auto_eval: str = Form("true"),
) -> Dict[str, Any]:
    """
    Upload one or more PDFs and run KV extraction via the inference API.

    Mirrors inference-pipeline/client_dir.sh with base64 upload (--upload).
    """
    if not files:
        raise HTTPException(status_code=400, detail="at least one PDF file is required")

    payload: List[tuple[str, bytes]] = []
    for upload in files:
        name = (upload.filename or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="each file must have a filename")
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty file: {name}")
        payload.append((name, data))

    hooks_name = (hooks or "agentic_config").strip() or "agentic_config"
    auto_eval_bool = str(auto_eval).strip().lower() in {"true", "1", "yes"}
    now = datetime.now()
    display_time = now.strftime("%Y-%m-%d %H:%M")
    ts_slug = now.strftime("%Y%m%d-%H%M%S")
    upload_dataset_id = f"upload-{ts_slug}-{uuid.uuid4().hex[:6]}"
    upload_dataset_name = f"Upload ({display_time})"
    try:
        _DATASET_STORE.create(upload_dataset_id, payload)
        paths = _DATASET_STORE.pdf_paths("managed", upload_dataset_id)
        job = _INFERENCE_JOB_MANAGER.start(
            paths=paths,
            hooks=hooks_name,
            dataset_id=upload_dataset_id,
            dataset_name=upload_dataset_name,
            dataset_source="managed",
            run_group_id=upload_dataset_id,
            run_group_name=upload_dataset_name,
            auto_eval=auto_eval_bool,
        )
        result = job.to_dict()
        result["dataset_id"] = upload_dataset_id
        result["dataset_source"] = "managed"
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@app.post("/api/inference/jobs/from-dataset")
def post_inference_job_from_dataset(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Run KV extraction for every PDF in a named dataset."""
    dataset_id = str((body or {}).get("dataset_id") or "").strip()
    source = str((body or {}).get("source") or "").strip()
    if not dataset_id or not source:
        raise HTTPException(status_code=400, detail="dataset_id and source are required")
    try:
        info = _DATASET_STORE.get(source, dataset_id)
        paths = _DATASET_STORE.pdf_paths(source, dataset_id)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise
    if not paths:
        raise HTTPException(status_code=400, detail="dataset has no PDF files")
    hooks_name = str((body or {}).get("hooks") or "agentic_config").strip() or "agentic_config"
    auto_eval = bool((body or {}).get("auto_eval", True))
    next_v = _next_dataset_run_version(info["id"])
    now = datetime.now()
    display_time = now.strftime("%Y-%m-%d %H:%M")
    ts_slug = now.strftime("%Y%m%d-%H%M%S")
    ds_name = info.get("name") or info["id"]
    run_group_id = f"{info['id']}-run-v{next_v}-{ts_slug}"
    run_group_name = f"{ds_name}-run-v{next_v} ({display_time})"
    try:
        job = _INFERENCE_JOB_MANAGER.start(
            paths=paths,
            hooks=hooks_name,
            dataset_id=info["id"],
            dataset_name=ds_name,
            dataset_source=info["source"],
            run_group_id=run_group_id,
            run_group_name=run_group_name,
            auto_eval=auto_eval,
        )
        return job.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/agentic-eval")
def post_agentic_eval(run_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Trigger EvalMasterAgent for one key via the inference-pipeline API.

    Up to AGENTIC_EVAL_MAX_PARALLEL keys may run concurrently per run_id.
    Status files under 06_agentic_eval/ are written by the inference API
    (container user); the viewer only tracks in-memory inflight state.
    """
    _run_dir(run_id)
    key = str((body or {}).get("key") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    active = _BATCH_MANAGER.get_active_job()
    if active and active.status == "running" and run_id in active.run_ids:
        raise HTTPException(
            status_code=409,
            detail=f"batch agentic-evaluation job {active.job_id} is running for this run",
        )

    inflight = _AGENTIC_EVAL_INFLIGHT.setdefault(run_id, set())
    inflight.add(key)
    try:
        result = _call_inference_agentic_eval(run_id, key)
        return result
    finally:
        inflight.discard(key)
        if not inflight:
            _AGENTIC_EVAL_INFLIGHT.pop(run_id, None)


@app.post("/api/runs/{run_id}/agentic-eval/cleanup-stale")
def post_cleanup_stale_eval(
    run_id: str, body: Dict[str, Any] = Body(default={})
) -> Dict[str, Any]:
    """Clean up stale running status for a specific run or key."""
    run_dir = _run_dir(run_id)
    key = body.get("key")
    reason = str(body.get("reason") or "cancelled by user (stale cleanup)")
    count = mark_running_eval_status_cancelled(run_dir, reason=reason, key=key)
    if key:
        inflight = _AGENTIC_EVAL_INFLIGHT.get(run_id)
        if inflight:
            inflight.discard(key)
            if not inflight:
                _AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
    else:
        _AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
    return {"run_id": run_id, "cleaned": count}


@app.post("/api/runs/{run_id}/agentic-eval/chat")
def post_agentic_eval_chat(
    run_id: str, body: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Send a follow-up chat message to the EvalMasterAgent for an evaluated key."""
    _run_dir(run_id)
    key = str((body or {}).get("key") or "").strip()
    message = str((body or {}).get("message") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        return invoke_agentic_eval_chat(INFERENCE_API_URL, run_id, key, message)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/agentic-eval/chat")
def get_agentic_eval_chat_api(
    run_id: str,
    key: str = Query(...),
) -> Dict[str, Any]:
    """Retrieve chat history for an evaluated key."""
    root = _run_dir(run_id)
    key = str(key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    from agentic_viewer.evaluation.live_progress import _safe_key_filename
    safe = _safe_key_filename(key)
    chat_file = root / "06_agentic_eval" / safe / "chat_history.json"
    if chat_file.is_file():
        try:
            data = json.loads(chat_file.read_text(encoding="utf-8"))
            return {
                "ok": True,
                "key": key,
                "has_history": bool(data.get("messages")),
                "history": data.get("messages") or [],
            }
        except Exception:
            pass

    try:
        return get_agentic_eval_chat(INFERENCE_API_URL, run_id, key)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.delete("/api/runs/{run_id}/agentic-eval/chat")
def delete_agentic_eval_chat_api(
    run_id: str,
    key: str = Query(...),
) -> Dict[str, Any]:
    """Clear chat history for an evaluated key."""
    root = _run_dir(run_id)
    key = str(key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    from agentic_viewer.evaluation.live_progress import _safe_key_filename
    safe = _safe_key_filename(key)
    eval_dir = root / "06_agentic_eval" / safe
    for fname in ("chat_history.json", "chat_context.json"):
        p = eval_dir / fname
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass

    try:
        return delete_agentic_eval_chat(INFERENCE_API_URL, run_id, key)
    except Exception:
        return {"ok": True, "key": key, "cleared": True}



def _serve_run_file(root: Path, rel: str):
    """Serve a file under ``root`` (JSON as JSONResponse, text as HTML pre)."""
    rel = rel.lstrip("/")
    target = (root / rel).resolve()
    if not str(target).startswith(str(root.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return FileResponse(target)
    if target.suffix.lower() in {".json", ".jsonl", ".md", ".txt"}:
        text = target.read_text(encoding="utf-8", errors="replace")
        if target.suffix.lower() in {".md", ".txt"}:
            text = replace_base64_images(text)
        if target.suffix.lower() == ".json":
            return JSONResponse(json.loads(text))
        if target.suffix.lower() == ".jsonl":
            lines_out = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        obj["text"] = replace_base64_images(obj["text"])
                    if isinstance(obj, dict) and isinstance(obj.get("content"), str):
                        obj["content"] = replace_base64_images(obj["content"])
                    lines_out.append(json.dumps(obj, ensure_ascii=False))
                except json.JSONDecodeError:
                    lines_out.append(line)
            text = "\n".join(lines_out)
        return HTMLResponse(
            f"<pre style='white-space:pre-wrap;font-family:ui-monospace,monospace'>"
            f"{_escape(text)}</pre>"
        )
    return FileResponse(target)


@app.get("/api/runs/{run_id}/file")
def get_file(run_id: str, path: str):
    root = _run_dir(run_id)
    return _serve_run_file(root.resolve(), path)


@app.get("/api/runs/{run_id}/pages")
def list_pages(run_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    summary = _read_json(root / "01_parse" / "summary.json") or {}
    progress = _read_json(root / "01_parse" / "progress.json") or {}
    pages = list(summary.get("pages") or [])
    if not pages:
        # Mid-parse: summary not written yet — scan per-page meta dumps.
        parse_dir = root / "01_parse"
        if parse_dir.is_dir():
            for meta_path in sorted(parse_dir.glob("page_*.meta.json")):
                meta = _read_json(meta_path)
                if isinstance(meta, dict) and meta.get("page") is not None:
                    pages.append(meta)
    return {
        "pages": pages,
        "page_count": summary.get("page_count") or len(pages),
        "progress": progress or None,
        "seconds": summary.get("seconds"),
    }


@app.get("/api/runs/{run_id}/chunks")
def list_chunks(
    run_id: str, offset: int = 0, limit: int = 200, q: str = ""
) -> Dict[str, Any]:
    root = _run_dir(run_id)
    summary = _read_json(root / "02_chunk" / "summary.json") or {}
    progress = _read_json(root / "02_chunk" / "progress.json") or {}
    chunks = list(summary.get("chunks") or [])
    query = (q or "").strip().lower()
    if query:
        chunks = [
            c
            for c in chunks
            if query in str(c.get("chunk_id") or "").lower()
            or query in str(c.get("heading_path") or "").lower()
            or query in str(c.get("page") or "")
        ]
    offset = max(0, int(offset or 0))
    limit = max(1, min(1000, int(limit or 200)))
    slice_ = chunks[offset : offset + limit]
    return {
        "strategy": summary.get("strategy"),
        "chunk_count": summary.get("chunk_count") or len(summary.get("chunks") or []),
        "filtered_count": len(chunks),
        "offset": offset,
        "limit": limit,
        "total_chars": summary.get("total_chars"),
        "total_est_tokens": summary.get("total_est_tokens"),
        "progress": progress or None,
        "chunks": slice_,
    }


@app.get("/api/runs/{run_id}/chunks/{chunk_id}")
def get_chunk(run_id: str, chunk_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    path = root / "02_chunk" / "chunks.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="chunks.jsonl not found")
    want = str(chunk_id)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("chunk_id") or "") == want:
            text = row.get("text") or ""
            if isinstance(text, str):
                row["text"] = replace_base64_images(text)
            return row
    raise HTTPException(status_code=404, detail=f"chunk not found: {chunk_id}")


@app.get("/api/runs/{run_id}/chunks/{chunk_id}/highlights")
def get_chunk_highlights(run_id: str, chunk_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    try:
        return chunk_highlights(root, chunk_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/pages/{page_no}/highlights")
def get_page_highlights(
    run_id: str, page_no: int, q: Optional[str] = None
) -> Dict[str, Any]:
    root = _run_dir(run_id)
    try:
        return page_highlights(root, page_no, query=q)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/pdf/info")
def get_pdf_info(run_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    return pdf_info(root)


@app.get("/api/runs/{run_id}/pdf")
def get_pdf(run_id: str):
    root = _run_dir(run_id)
    path = infer_pdf_path(root)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found for this run")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/runs/{run_id}/conversation")
def get_conversation(run_id: str) -> List[Dict[str, Any]]:
    """Chat-style message transcript (preferred) or reconstructed from step dumps."""
    root = _run_dir(run_id)
    path = root / "03_agent" / "conversation.jsonl"
    if path.is_file():
        rows = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["i"] = i
            if isinstance(row.get("content"), str):
                row["content"] = replace_base64_images(row["content"])
            rows.append(row)
        return rows

    # Fallback: rebuild from last step's messages_after (older runs).
    agent_dir = root / "03_agent"
    steps = sorted(agent_dir.glob("step_*.json")) if agent_dir.is_dir() else []
    if not steps:
        return []
    last = _read_json(steps[-1]) or {}
    messages = last.get("messages_after") or last.get("messages") or []
    rows = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            content = replace_base64_images(content)
        rows.append(
            {
                "i": i,
                "t": None,
                "turn": None,
                "kind": m.get("role"),
                "role": m.get("role"),
                "content": content,
                "tool_calls": m.get("tool_calls"),
                "tool_call_id": m.get("tool_call_id"),
                "name": m.get("name"),
                "source": "reconstructed",
            }
        )
    return rows


# --- Wrong Cases endpoints ---

@app.get("/api/wrong-cases")
def api_list_wrong_cases(
    run_id: Optional[str] = None,
    document: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    return list_wrong_cases(
        run_id=run_id,
        document=document,
        status=status,
        search=search,
    )


@app.get("/api/wrong-cases/{case_id}")
def api_get_wrong_case_detail(case_id: str) -> Dict[str, Any]:
    detail = get_wrong_case_detail(RUNS_ROOT, case_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"wrong case not found: {case_id}")
    return detail


@app.post("/api/wrong-cases")
def api_create_wrong_case(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    run_id = str(body.get("run_id") or "").strip()
    key = str(body.get("key") or "").strip()
    note = str(body.get("note") or "").strip()
    status = str(body.get("status") or "open").strip()
    tags = body.get("tags")
    user_snapshot = body.get("snapshot")
    if not run_id or not key:
        raise HTTPException(status_code=400, detail="run_id and key are required")
    try:
        return add_or_update_wrong_case(
            RUNS_ROOT,
            run_id,
            key,
            note=note,
            status=status,
            tags=tags,
            user_snapshot=user_snapshot,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/wrong-cases/batch")
def api_batch_create_wrong_cases(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    run_id = str(body.get("run_id") or "").strip()
    keys = body.get("keys")
    note = str(body.get("note") or "").strip()
    status = str(body.get("status") or "open").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    if not isinstance(keys, list) or not keys:
        raise HTTPException(status_code=400, detail="keys must be a non-empty list of strings")
    try:
        added = batch_add_wrong_cases(
            RUNS_ROOT,
            run_id,
            keys,
            note=note,
            status=status,
        )
        return {"cases": added, "count": len(added)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/wrong-cases/{case_id}")
@app.put("/api/wrong-cases/{case_id}")
def api_update_wrong_case(case_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    status = body.get("status")
    note = body.get("note")
    tags = body.get("tags")
    updated = update_wrong_case_status(
        case_id,
        status=status,
        note=note,
        tags=tags,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"wrong case not found: {case_id}")
    return updated


@app.delete("/api/wrong-cases/{case_id}")
def api_delete_wrong_case(case_id: str) -> Dict[str, Any]:
    deleted = delete_wrong_case(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"wrong case not found: {case_id}")
    return {"ok": True, "id": case_id}


# --- Prompts endpoints ---

@app.get("/api/prompts")
def api_list_prompts() -> Dict[str, Any]:
    return {"prompts": list_prompts()}


@app.get("/api/prompts/{prompt_id}")
def api_get_prompt(prompt_id: str) -> Dict[str, Any]:
    try:
        return get_prompt(prompt_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/prompts/{prompt_id}")
def api_save_prompt(prompt_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    if "content" in body:
        content = body.get("content")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="'content' must be a string")
    elif "keys" in body and prompt_id == "kv_schema":
        keys = body.get("keys")
        if not isinstance(keys, list):
            raise HTTPException(status_code=400, detail="'keys' must be a list of objects")
        content = json.dumps(keys, ensure_ascii=False, indent=2) + "\n"
    else:
        raise HTTPException(status_code=400, detail="'content' field is required")
    try:
        return save_prompt(prompt_id, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/prompts/{prompt_id}/backups")
def api_list_prompt_backups(prompt_id: str) -> Dict[str, Any]:
    try:
        return {"backups": list_backups(prompt_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/prompts/{prompt_id}/backups/{timestamp}")
def api_get_prompt_backup(prompt_id: str, timestamp: str) -> Dict[str, Any]:
    try:
        return get_backup(prompt_id, timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/prompts/{prompt_id}/restore")
def api_restore_prompt(prompt_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    timestamp = str((body or {}).get("timestamp") or "").strip()
    if not timestamp:
        raise HTTPException(status_code=400, detail="'timestamp' is required")
    try:
        return restore_prompt(prompt_id, timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/prompts/{prompt_id}/diff")
def api_diff_prompt(prompt_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    modified = body.get("modified")
    if not isinstance(modified, str):
        raise HTTPException(status_code=400, detail="'modified' text is required")
    compare_with = body.get("compare_with", "current")
    try:
        if compare_with == "current":
            original = get_prompt(prompt_id)["content"]
            from_label = "current_saved"
        else:
            original = get_backup(prompt_id, str(compare_with))["content"]
            from_label = f"backup_{compare_with}"
        diff_text = compute_diff(original, modified, fromfile=from_label, tofile="modified")
        return {"diff": diff_text}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc



def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agentic Viewer — Inference</title>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2332;
      --line: #2d3a4d;
      --text: #e7ecf3;
      --muted: #9aa8bc;
      --accent: #3d9cf0;
      --ok: #3ecf8e;
      --err: #f07178;
      --sys: #6b7c93;
      --user: #2a4a6d;
      --asst: #1e3a2f;
      --tool: #3a2f1e;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--text);
      font-family: var(--sans); height: 100vh;
      display: flex; flex-direction: column; overflow: hidden;
    }
    header {
      flex: 0 0 auto;
      padding: 14px 20px; border-bottom: 1px solid var(--line);
      display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
    }
    header h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.02em; }
    header .meta { color: var(--muted); font-size: 13px; margin-left: auto; }
    .topnav { display: flex; gap: 4px; }
    .topnav a {
      color: var(--muted); text-decoration: none; font-size: 13px;
      padding: 6px 12px; border-radius: 999px; border: 1px solid transparent;
    }
    .topnav a:hover { color: var(--text); border-color: var(--line); }
    .topnav a.active {
      color: var(--text); background: var(--panel); border-color: var(--line);
    }
    main {
      flex: 1 1 0;
      min-height: 0;
      display: grid;
      grid-template-columns: 280px 1fr;
      overflow: hidden;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #121820;
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    #runList {
      flex: 1 1 0;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      scrollbar-width: thin;
      scrollbar-color: var(--line) transparent;
    }
    #runList::-webkit-scrollbar {
      width: 6px;
    }
    #runList::-webkit-scrollbar-track {
      background: transparent;
    }
    #runList::-webkit-scrollbar-thumb {
      background: var(--line);
      border-radius: 3px;
    }
    #runList::-webkit-scrollbar-thumb:hover {
      background: var(--muted);
    }
    .run {
      padding: 12px 14px; border-bottom: 1px solid var(--line); cursor: pointer;
      position: relative;
    }
    .run .run-head {
      display: flex; gap: 8px; align-items: flex-start; justify-content: space-between;
    }
    .run .run-delete {
      flex: 0 0 auto; background: transparent; border: 1px solid transparent;
      color: var(--muted); border-radius: 6px; cursor: pointer; font-size: 14px;
      line-height: 1; padding: 2px 6px;
    }
    .run .run-delete:hover:not(:disabled) {
      color: var(--err); border-color: #7a3a3f; background: rgba(240, 113, 120, 0.08);
    }
    .run .run-delete:disabled { opacity: 0.35; cursor: not-allowed; }
    .run:hover, .run.active { background: var(--panel); }
    .run .id { font-size: 12px; word-break: break-all; }
    .run-label { display: flex; flex-direction: column; gap: 2px; }
    .run-doc { font-size: 13px; font-weight: 600; line-height: 1.3; word-break: break-word; }
    .run-id { font-family: var(--mono); font-size: 11px; color: var(--muted); word-break: break-all; }
    .run .sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .badge {
      display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px;
      border: 1px solid var(--line);
    }
    .badge.ok { color: var(--ok); border-color: #2a6b4f; }
    .badge.error { color: var(--err); border-color: #7a3a3f; }
    .badge.warn { color: #e0a45c; border-color: #6b5530; }
    section {
      padding: 16px 20px;
      overflow-y: auto;
      overflow-x: hidden;
      height: 100%;
      min-height: 0;
    }
    .tabs { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
    .tab {
      background: transparent; border: 1px solid var(--line); color: var(--muted);
      padding: 6px 12px; border-radius: 6px; cursor: pointer; font: inherit;
    }
    .tab.active { color: var(--text); border-color: var(--accent); background: #152033; }
    pre, .code {
      background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
      padding: 12px; overflow: auto; font-family: var(--mono); font-size: 12px;
      line-height: 1.45; white-space: pre-wrap; word-break: break-word;
    }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 960px) {
      body { height: auto; min-height: 100vh; overflow: auto; }
      main { display: flex; flex-direction: column; height: auto; overflow: visible; }
      aside { height: auto; overflow: visible; }
      #runList { max-height: 380px; }
      section { height: auto; overflow: visible; }
      .grid2 { grid-template-columns: 1fr; }
    }
    .event {
      border-left: 2px solid var(--line); padding: 8px 0 8px 12px; margin: 0 0 8px;
    }
    .event .t { color: var(--muted); font-family: var(--mono); font-size: 11px; }
    .event .title { font-size: 13px; margin-top: 2px; }
    a { color: var(--accent); }
    .empty { color: var(--muted); padding: 40px 0; }
    .hint {
      color: var(--muted); font-size: 12px; margin: 0 0 14px; line-height: 1.45;
      max-width: 720px;
    }
    .chat { display: flex; flex-direction: column; gap: 12px; max-width: 860px; }
    .bubble {
      border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
      background: var(--panel);
    }
    .bubble.system { background: #151a22; border-color: #2a3344; }
    .bubble.user { background: var(--user); border-color: #3d6a9a; }
    .bubble.assistant { background: var(--asst); border-color: #2f6b52; }
    .bubble.tool { background: var(--tool); border-color: #6b5530; }
    .bubble .head {
      display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap;
      margin-bottom: 6px; font-size: 12px;
    }
    .bubble .role {
      font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
      font-size: 11px;
    }
    .bubble .meta { color: var(--muted); font-family: var(--mono); font-size: 11px; }
    .bubble .body {
      font-family: var(--mono); font-size: 12px; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word; max-height: 320px; overflow: auto;
    }
    .tool-call {
      margin-top: 8px; padding: 8px; border-radius: 6px;
      background: rgba(0,0,0,0.25); border: 1px dashed #4a6b55;
    }
    .tool-call .fn { color: var(--ok); font-family: var(--mono); font-size: 12px; }
    .turn-sep {
      display: flex; align-items: center; gap: 10px; color: var(--muted);
      font-size: 11px; font-family: var(--mono); margin: 4px 0;
    }
    .turn-sep::before, .turn-sep::after {
      content: ""; flex: 1; height: 1px; background: var(--line);
    }
    .viz-list { display: flex; flex-direction: column; gap: 16px; max-width: 920px; }
    .viz-card {
      border: 1px solid var(--line); border-radius: 10px; background: #121820;
      overflow: hidden;
    }
    .viz-card > summary {
      cursor: pointer; list-style: none; padding: 12px 14px;
      display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline;
      background: var(--panel); border-bottom: 1px solid var(--line);
    }
    .viz-card > summary::-webkit-details-marker { display: none; }
    .viz-card[open] > summary { border-bottom: 1px solid var(--line); }
    .viz-card .title { font-weight: 600; }
    .viz-body { padding: 12px 14px; display: flex; flex-direction: column; gap: 12px; }
    .viz-section h3 {
      margin: 0 0 6px; font-size: 12px; color: var(--accent);
      text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600;
    }
    .kv-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .kv-table th, .kv-table td {
      text-align: left; vertical-align: top; padding: 6px 8px;
      border-bottom: 1px solid var(--line); font-family: var(--mono);
    }
    .kv-table th { color: var(--muted); font-weight: 500; }
    .kv-table th:first-child { width: 180px; }
    .pdf-chunk-viewer { margin: 8px 0 12px; }
    .pdf-chunk-viewer .pdf-toolbar {
      display: flex; flex-direction: column; gap: 8px;
      margin-bottom: 8px; font-size: 12px; color: var(--muted);
      background: #0f1520; border: 1px solid var(--line); border-radius: 8px;
      padding: 8px 12px;
    }
    .pdf-toolbar-row {
      display: flex; gap: 12px; align-items: center; flex-wrap: wrap; justify-content: space-between;
    }
    .pdf-nav-row {
      justify-content: flex-start; gap: 10px;
    }
    .pdf-nav-group {
      display: inline-flex; align-items: center; gap: 6px;
    }
    .pdf-nav-btn {
      padding: 3px 9px; border-radius: 6px; border: 1px solid var(--line);
      background: #182232; color: var(--text); font-size: 11px; cursor: pointer;
      display: inline-flex; align-items: center; transition: all 0.15s ease;
    }
    .pdf-nav-btn:hover:not(:disabled) {
      border-color: var(--accent); color: var(--accent); background: #223044;
    }
    .pdf-nav-btn:disabled {
      opacity: 0.35; cursor: not-allowed;
    }
    .pdf-chunk-btn {
      background: #15273b; border-color: rgba(61, 156, 240, 0.4); color: #7cb9f7;
      font-weight: 500;
    }
    .pdf-chunk-btn:hover:not(:disabled) {
      background: #1b3450; border-color: var(--accent); color: #fff;
    }
    .pdf-chunk-btn.active {
      background: rgba(61, 156, 240, 0.25); border-color: var(--accent); color: #fff;
      box-shadow: 0 0 0 1px var(--accent) inset;
    }
    .pdf-nav-page-box {
      font-family: var(--mono); font-size: 11px; color: var(--muted);
      display: inline-flex; align-items: center; gap: 4px;
    }
    .pdf-page-input {
      width: 52px; padding: 2px 4px; border-radius: 4px; border: 1px solid var(--line);
      background: #0b1016; color: var(--text); font-family: var(--mono); font-size: 11px;
      text-align: center;
    }
    .pdf-page-input:focus {
      outline: none; border-color: var(--accent);
    }
    .pdf-status-badge {
      font-size: 11px; font-family: var(--mono); padding: 2px 8px; border-radius: 999px;
      margin-left: auto; display: inline-flex; align-items: center;
    }
    .pdf-status-badge.is-highlight {
      background: rgba(61, 156, 240, 0.18); border: 1px solid rgba(61, 156, 240, 0.6);
      color: #9cd0fc;
    }
    .pdf-status-badge.is-plain {
      background: #18202b; border: 1px solid var(--line); color: var(--muted);
    }
    .pdf-open-link {
      font-size: 11px; color: var(--accent); text-decoration: none;
    }
    .pdf-open-link:hover { text-decoration: underline; }
    .pdf-chunk-viewer .pdf-pages { display: flex; flex-direction: column; gap: 14px; }
    .pdf-page-wrap {
      border: 1px solid var(--line); border-radius: 8px; padding: 8px;
      background: #0b1016; max-width: 100%; overflow: auto;
    }
    .pdf-page-label {
      font-size: 11px; color: var(--muted); font-family: var(--mono);
      margin-bottom: 6px;
    }
    .pdf-canvas-wrap { position: relative; display: inline-block; line-height: 0; }
    .pdf-canvas-wrap canvas { display: block; max-width: 100%; height: auto; }
    .pdf-overlay {
      position: absolute; left: 0; top: 0; pointer-events: none;
    }
    .hl-box {
      position: absolute; box-sizing: border-box;
      border: 2px solid rgba(61, 156, 240, 0.95);
      background: rgba(61, 156, 240, 0.18);
      box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.25) inset;
    }
    .score-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px; margin: 0 0 16px;
    }
    .score-card {
      background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
      padding: 10px 12px;
    }
    .score-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    .score-card .value { font-family: var(--mono); font-size: 20px; margin-top: 4px; font-weight: 600; }
    .score-card .sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
    .eval-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .eval-table th, .eval-table td {
      text-align: left; vertical-align: top; padding: 8px 8px;
      border-bottom: 1px solid var(--line);
    }
    .eval-table th { color: var(--muted); font-weight: 500; position: sticky; top: 0; background: var(--bg); }
    .eval-table .key { font-family: var(--mono); font-size: 11px; max-width: 220px; word-break: break-word; }
    .eval-table details { margin-top: 4px; }
    .eval-table details summary { cursor: pointer; color: var(--accent); font-size: 11px; }
    .eval-table .ev-text {
      white-space: pre-wrap; word-break: break-word; font-family: var(--mono);
      font-size: 11px; max-height: 160px; overflow: auto; margin-top: 4px;
      background: #121820; border: 1px solid var(--line); border-radius: 6px; padding: 8px;
    }
    .eval-table .ev-block { margin-top: 6px; }
    .ev-label {
      display: inline-block; font-size: 10px; font-weight: 600; letter-spacing: 0.03em;
      text-transform: uppercase; padding: 1px 6px; border-radius: 4px; margin-bottom: 4px;
    }
    .ev-label.vlm { color: #9ad0ff; background: #1a2a3d; border: 1px solid #2a4a6a; }
    .ev-label.search { color: #b8e0a8; background: #1a2e1a; border: 1px solid #2a4a2a; }
    .ev-label.gold { color: #e0d0a0; background: #2a2618; border: 1px solid #4a4020; }
    .agentic-eval-btn {
      padding: 4px 10px; border-radius: 6px; border: 1px solid var(--accent);
      background: #152033; color: var(--accent); font-size: 11px; cursor: pointer;
      white-space: nowrap;
    }
    .agentic-eval-btn:disabled {
      opacity: 0.45; cursor: not-allowed; border-color: var(--line); color: var(--muted);
    }
    .agentic-eval-summary {
      font-size: 12px; margin: 0 0 6px; line-height: 1.35;
      max-width: 460px;
    }
    .agentic-eval-detail {
      margin-top: 6px;
      min-width: 280px;
      max-width: 460px;
    }
    .agentic-eval-text {
      white-space: pre-wrap; word-break: break-word; font-family: var(--mono);
      font-size: 11px; max-height: 220px; overflow: auto;
      background: #121820; border: 1px solid var(--line); border-radius: 6px; padding: 8px;
      min-width: 240px; max-width: 460px;
    }
    .agentic-eval-detail details summary {
      cursor: pointer; color: var(--accent); font-size: 11px;
    }
    .eval-chat-section {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
      min-width: 260px;
      max-width: 460px;
    }
    .eval-chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      font-weight: 600;
      color: var(--accent);
      margin-bottom: 6px;
    }
    .eval-chat-clear-btn {
      background: transparent;
      border: 1px solid var(--line);
      color: var(--muted);
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 10px;
      cursor: pointer;
    }
    .eval-chat-clear-btn:hover {
      color: var(--err);
      border-color: var(--err);
    }
    .eval-chat-msgs-box {
      max-height: 260px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding: 6px;
      background: #090d14;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .eval-chat-empty {
      font-size: 11px;
      color: var(--muted);
      padding: 8px 4px;
      line-height: 1.4;
      text-align: center;
    }
    .eval-chat-msg {
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.4;
      word-break: break-word;
    }
    .eval-chat-msg.user {
      align-self: flex-end;
      background: #19385c;
      color: #e2eeff;
      border: 1px solid #2b568c;
      max-width: 90%;
    }
    .eval-chat-msg.assistant {
      align-self: flex-start;
      background: #141c28;
      color: #dbe4ee;
      border: 1px solid #28374d;
      max-width: 95%;
    }
    .eval-chat-msg.assistant.loading {
      color: var(--accent);
      font-style: italic;
    }
    .eval-chat-msg-header {
      font-size: 10px;
      font-weight: 700;
      color: var(--muted);
      margin-bottom: 2px;
    }
    .eval-chat-msg.user .eval-chat-msg-header {
      color: #8bbce8;
      text-align: right;
    }
    .eval-chat-msg-body {
      white-space: pre-wrap;
    }
    .eval-chat-tools-badge {
      font-size: 10px;
      color: #6ee7b7;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 4px;
      padding: 2px 5px;
      margin-bottom: 4px;
      display: inline-block;
    }
    .eval-chat-error {
      color: var(--err);
      font-size: 11px;
      margin-top: 4px;
    }
    .eval-chat-input-row {
      display: flex;
      gap: 4px;
      margin-top: 6px;
    }
    .eval-chat-input {
      flex: 1;
      min-width: 0;
      background: #0f172a;
      border: 1px solid var(--line);
      border-radius: 4px;
      color: var(--text);
      font-size: 11px;
      padding: 4px 8px;
      font-family: inherit;
    }
    .eval-chat-input:focus {
      outline: none;
      border-color: var(--accent);
    }
    .eval-chat-send-btn {
      background: #1d4ed8;
      border: 1px solid #3b82f6;
      color: #fff;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
    }
    .eval-chat-send-btn:disabled, .eval-chat-input:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .agentic-eval-verdict {
      display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.03em;
      text-transform: uppercase; padding: 2px 8px; border-radius: 4px; margin-bottom: 6px;
    }
    .agentic-eval-verdict.correct {
      color: var(--ok); background: rgba(61, 214, 140, 0.12); border: 1px solid rgba(61, 214, 140, 0.35);
    }
    .agentic-eval-verdict.incorrect {
      color: var(--err); background: rgba(255, 107, 107, 0.12); border: 1px solid rgba(255, 107, 107, 0.35);
    }
    .agentic-eval-verdict.valid {
      color: var(--ok); background: rgba(61, 214, 140, 0.12); border: 1px solid rgba(61, 214, 140, 0.35);
    }
    .agentic-eval-verdict.invalid {
      color: var(--err); background: rgba(255, 107, 107, 0.12); border: 1px solid rgba(255, 107, 107, 0.35);
    }
    .agentic-eval-verdicts { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
    .agentic-eval-err { color: var(--err); font-size: 11px; }
    .gt-edit-btn {
      margin-top: 6px; padding: 3px 10px; border-radius: 999px;
      border: 1px solid var(--line); background: #152033; color: var(--text);
      font-size: 11px; cursor: pointer;
    }
    .wrong-case-btn {
      margin-top: 6px; padding: 3px 10px; border-radius: 999px;
      border: 1px solid var(--line); background: #152033; color: var(--muted);
      font-size: 11px; cursor: pointer; transition: all 0.15s ease;
      white-space: nowrap;
    }
    .wrong-case-btn:hover {
      border-color: var(--warn); color: var(--warn);
    }
    .wrong-case-btn.active {
      border-color: rgba(224, 164, 92, 0.6); background: rgba(224, 164, 92, 0.15);
      color: var(--warn); font-weight: 600;
    }
    .eval-row.row-highlight td {
      animation: evalRowPulse 3s ease-out;
    }
    @keyframes evalRowPulse {
      0% { background: rgba(224, 164, 92, 0.4); }
      30% { background: rgba(224, 164, 92, 0.25); }
      100% { background: transparent; }
    }
    .eval-page-chips {
      display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin: 4px 0 6px;
    }
    .pdf-page-btn {
      padding: 2px 8px; border-radius: 6px; border: 1px solid #2a4a6a;
      background: #152033; color: #9ad0ff; font-size: 11px; cursor: pointer;
      font-family: var(--mono); transition: all 0.15s ease;
      white-space: nowrap;
    }
    .pdf-page-btn:hover {
      background: #1e3352; border-color: var(--accent); color: #fff;
    }
    .pdf-page-inline-btn {
      display: inline-flex; align-items: center; gap: 2px;
      padding: 0 4px; border-radius: 4px; border: 1px solid rgba(77, 170, 252, 0.4);
      background: rgba(77, 170, 252, 0.12); color: var(--accent);
      font-size: 11px; cursor: pointer; font-family: var(--mono);
      vertical-align: baseline; margin: 0 2px;
    }
    .pdf-page-inline-btn:hover {
      background: rgba(77, 170, 252, 0.28); border-color: var(--accent);
    }
    .chunk-preview-inline {
      margin-top: 8px; padding: 8px 10px; background: #121820;
      border: 1px solid var(--line); border-radius: 8px;
      min-width: 360px; max-width: 100%; overflow-x: auto;
    }
    .preview-close-btn {
      cursor: pointer; background: transparent; border: none;
      color: var(--muted); font-size: 14px; font-weight: bold; padding: 2px 6px;
      border-radius: 4px; line-height: 1;
    }
    .preview-close-btn:hover { color: var(--text); background: rgba(255,255,255,0.1); }
    .gt-edit-btn:hover { border-color: var(--accent); }
    .gt-edit-btn.warn {
      border-color: #7a5530; color: #e0c090; background: rgba(224, 164, 92, 0.12);
    }
    .gt-modal-backdrop {
      position: fixed; inset: 0; background: rgba(0, 0, 0, 0.55); z-index: 200;
      display: flex; align-items: center; justify-content: center; padding: 20px;
    }
    .gt-modal {
      width: min(720px, 100%); max-height: 90vh; overflow: auto;
      background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
      padding: 16px 18px;
    }
    .gt-modal h3 { margin: 0 0 8px; font-size: 15px; }
    .gt-modal .sub { color: var(--muted); font-size: 12px; margin-bottom: 12px; word-break: break-word; }
    .gt-modal label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 4px; }
    .gt-modal input, .gt-modal textarea {
      width: 100%; padding: 8px 10px; border-radius: 8px;
      border: 1px solid var(--line); background: #0f1419; color: var(--text);
      font-family: var(--mono); font-size: 12px; line-height: 1.4;
    }
    .gt-modal textarea { min-height: 120px; resize: vertical; }
    .gt-modal-actions {
      display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px;
    }
    .gt-modal-actions button {
      padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line);
      background: #152033; color: var(--text); font-size: 12px; cursor: pointer;
    }
    .gt-modal-actions button.primary {
      background: #1a3a5c; border-color: #3d6a9a;
    }
    .gt-modal-actions button:disabled { opacity: 0.5; cursor: not-allowed; }
    .gt-modal-msg { font-size: 12px; margin-top: 8px; }
    .gt-modal-msg.ok { color: var(--ok); }
    .gt-modal-msg.err { color: var(--err); }
    .em-y { color: var(--ok); font-weight: 600; }
    .em-n { color: var(--err); font-weight: 600; }
    .run .eval-mini { color: var(--muted); font-size: 11px; margin-top: 3px; font-family: var(--mono); }

    .pill {
      display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0;
      border-radius: 999px; border: 1px solid var(--line); font-size: 11px;
      font-family: var(--mono); color: var(--text); background: #152033;
    }
    .flow {
      display: flex; flex-direction: column; gap: 8px;
    }
    .flow-item {
      border-left: 3px solid var(--line); padding: 8px 10px;
      background: var(--panel); border-radius: 0 8px 8px 0;
    }
    .flow-item.system { border-left-color: #6b7c93; }
    .flow-item.user { border-left-color: #3d9cf0; }
    .flow-item.assistant { border-left-color: #3ecf8e; }
    .flow-item.tool { border-left-color: #e0a45c; }
    .flow-item .label {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.04em; margin-bottom: 4px; color: var(--muted);
    }
    .pretty {
      font-family: var(--mono); font-size: 12px; line-height: 1.45;
      white-space: pre-wrap; word-break: break-word; max-height: 280px; overflow: auto;
      margin: 0;
    }
    .arrow { color: var(--muted); font-size: 12px; text-align: center; padding: 2px 0; }
    .tree { display: flex; flex-direction: column; gap: 10px; max-width: 980px; }
    .tree-node {
      border: 1px solid var(--line); border-radius: 10px; background: #121820;
      overflow: hidden;
    }
    .tree-node > summary {
      cursor: pointer; list-style: none; padding: 10px 12px;
      display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline;
      background: var(--panel);
    }
    .tree-node > summary::-webkit-details-marker { display: none; }
    .tree-node.master > summary { border-left: 4px solid var(--accent); }
    .tree-node.search > summary { border-left: 4px solid #e0a45c; margin-left: 16px; }
    .tree-node.session > summary { border-left: 4px solid #9b7bd4; margin-left: 32px; }
    .tree-node.session.shared > summary { border-left: 4px solid #7eb8da; margin-left: 32px; }
    .tree-node.key-result > summary { border-left: 4px solid #5a9b6a; margin-left: 32px; }
    .tree-node.turn > summary { border-left: 4px solid #6b7c93; margin-left: 48px; }
    .tree-node.search-prompts > summary { border-left: 4px solid #f0a85c; margin-left: 48px; }
    .search-cues-pill {
      display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
      background: rgba(224, 164, 92, 0.15); color: #f0c060; border: 1px solid rgba(224, 164, 92, 0.35);
      margin-right: 4px; margin-bottom: 2px; word-break: break-word;
    }
    .allowed-values-tag {
      display: inline-block; font-size: 11px; padding: 2px 6px; border-radius: 4px;
      background: rgba(62, 207, 142, 0.12); color: #3ecf8e; border: 1px solid rgba(62, 207, 142, 0.3);
      font-family: var(--mono); word-break: break-word;
    }
    .tree-node.output > summary { border-left: 4px solid #3ecf8e; }
    .tree-body { padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 8px; }
    .tree-tool {
      margin-left: 16px; padding: 8px 10px; border-radius: 8px;
      border: 1px dashed var(--line); background: rgba(0,0,0,0.15);
    }
    .tree-tool .name { color: var(--ok); font-family: var(--mono); font-size: 12px; font-weight: 600; }
    .tree-kv { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-top: 4px; }
    .tree-badge {
      font-size: 10px; padding: 1px 6px; border-radius: 999px;
      border: 1px solid var(--line); color: var(--muted); font-family: var(--mono);
    }
    .tree-badge.ok { color: var(--ok); border-color: #2a6b4f; }
    .tree-badge.warn { color: #e0a45c; border-color: #6b5530; }
    .tree-badge.err { color: var(--err); border-color: #7a3a3f; }
    .timing-panel { display: flex; flex-direction: column; gap: 16px; max-width: 980px; }
    .timing-row {
      display: grid; grid-template-columns: 180px 1fr 72px 52px; gap: 10px;
      align-items: center; font-size: 12px;
    }
    .timing-row .label { color: var(--muted); font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .timing-bar-wrap {
      height: 18px; background: #0f1419; border: 1px solid var(--line);
      border-radius: 999px; overflow: hidden;
    }
    .timing-bar {
      height: 100%; border-radius: 999px; min-width: 2px;
      background: linear-gradient(90deg, #3d9cf0, #3ecf8e);
    }
    .timing-bar.search { background: linear-gradient(90deg, #e0a45c, #f0c060); }
    .timing-bar.master { background: linear-gradient(90deg, #3d9cf0, #6eb6ff); }
    .timing-bar.parse { background: linear-gradient(90deg, #6b7c93, #9aa8bc); }
    .timing-bar.chunk { background: linear-gradient(90deg, #9b7bd4, #b89de8); }
    .timing-sub { margin-left: 20px; border-left: 2px solid var(--line); padding-left: 12px; display: flex; flex-direction: column; gap: 8px; }
    .timing-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .timing-table th, .timing-table td {
      text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line);
      font-family: var(--mono); vertical-align: top;
    }
    .timing-table th { color: var(--muted); font-weight: 500; }
    .timing-table tr.running td { background: rgba(224, 164, 92, 0.08); }
    .timing-live {
      border: 1px solid #6b5530; background: rgba(224, 164, 92, 0.12);
      border-radius: 10px; padding: 10px 12px; display: flex; flex-direction: column; gap: 6px;
    }
    .timing-live .live-title { color: #e0a45c; font-weight: 600; font-size: 13px; }
    .timing-live .live-row { font-family: var(--mono); font-size: 12px; }
    .timing-live .live-row .sess { color: #e0a45c; }
    body.embed { height: auto; min-height: 100vh; overflow: auto; }
    body.embed header,
    body.embed aside { display: none; }
    body.embed main { display: block; height: auto; min-height: 100vh; }
    body.embed section { padding: 12px 14px; height: auto; overflow: visible; }
    .upload-panel {
      flex: 0 0 auto;
      max-height: 50vh;
      overflow-y: auto;
      padding: 12px 14px; border-bottom: 1px solid var(--line);
      background: #121820; display: flex; flex-direction: column; gap: 8px;
    }
    .upload-panel h2 {
      margin: 0; font-size: 12px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted);
    }
    .upload-panel .hint { margin: 0; font-size: 11px; line-height: 1.4; }
    .upload-panel input[type="file"] {
      width: 100%; font-size: 12px; color: var(--muted);
    }
    .upload-panel .upload-actions {
      display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    }
    .upload-panel .upload-btn {
      background: #152033; border: 1px solid var(--accent); color: var(--text);
      padding: 6px 12px; border-radius: 6px; cursor: pointer; font: inherit; font-size: 12px;
    }
    .upload-panel .upload-btn:disabled {
      opacity: 0.5; cursor: not-allowed; border-color: var(--line);
    }
    .upload-panel select {
      width: 100%; padding: 6px 8px; border-radius: 6px;
      border: 1px solid var(--line); background: #0f1419; color: var(--text);
      font: inherit; font-size: 12px;
    }
    .upload-source {
      display: flex; gap: 12px; flex-wrap: wrap; font-size: 12px; color: var(--muted);
    }
    .upload-source label { display: flex; gap: 6px; align-items: center; cursor: pointer; }
    .dataset-group { border-bottom: 1px solid var(--line); }
    .dataset-group > summary {
      cursor: pointer; list-style: none; padding: 10px 14px;
      display: flex; gap: 8px; align-items: center; justify-content: space-between;
      background: #151c26; color: var(--text); font-size: 12px; font-weight: 600;
    }
    .dataset-group > summary::-webkit-details-marker { display: none; }
    .dataset-group > summary:hover { background: #1a2332; }
    .dataset-group > summary .group-title { flex: 1; word-break: break-word; }
    .dataset-group > summary .group-actions {
      display: flex; align-items: center; gap: 8px; margin-left: auto;
    }
    .dataset-group > summary .count { color: var(--muted); font-weight: 400; font-family: var(--mono); font-size: 11px; white-space: nowrap; }
    .dataset-group > summary .group-delete-btn {
      background: transparent; border: 1px solid transparent; color: var(--muted);
      border-radius: 4px; padding: 2px 6px; font-size: 11px; cursor: pointer; line-height: 1.2;
    }
    .dataset-group > summary .group-delete-btn:hover {
      color: var(--err); border-color: #7a3a3f; background: rgba(240, 113, 120, 0.08);
    }
    .dataset-group .run { padding-left: 18px; }
    .upload-status {
      font-size: 11px; color: var(--muted); line-height: 1.45;
      font-family: var(--mono); word-break: break-word;
    }
    .upload-status.running { color: #e0a45c; }
    .upload-status.error { color: var(--err); }
    .upload-status.done { color: var(--ok); }
  </style>
</head>
<body id="appBody">
  <header>
    <h1>Agentic Viewer</h1>
    <nav class="topnav">
      <a href="/" class="active">Inference</a>
      <a href="/datasets">Datasets</a>
      <a href="/evaluation">Evaluation</a>
      <a href="/ground-truth">Ground Truth</a>
      <a href="/wrong-cases">Wrong Cases</a>
      <a href="/prompts">Prompts</a>
    </nav>
    <div class="meta" id="headerMeta">Loading runs…</div>
  </header>
  <main>
    <aside>
      <div class="upload-panel" id="uploadPanel">
        <h2>KV extraction</h2>
        <p class="hint">Upload PDFs (saved as a new UUID dataset) or run an existing dataset from the <a href="/datasets">Datasets</a> page.</p>
        <div class="upload-source">
          <label><input type="radio" name="inferSource" value="files" checked /> Upload files</label>
          <label><input type="radio" name="inferSource" value="dataset" /> Dataset</label>
        </div>
        <input type="file" id="inferenceUploadInput" accept=".pdf,.PDF,application/pdf" multiple />
        <select id="inferenceDatasetSelect" hidden>
          <option value="">Select a dataset…</option>
        </select>
        <div style="margin: 6px 0 4px 0;">
          <label style="font-size: 11px; display: inline-flex; align-items: center; gap: 5px; cursor: pointer; color: var(--text);">
            <input type="checkbox" id="inferenceAutoEval" checked />
            Auto-evaluate via <code>-eval</code> model (asynchronous)
          </label>
        </div>
        <div class="upload-actions">
          <button type="button" class="upload-btn" id="inferenceUploadBtn">Run extraction</button>
          <button type="button" class="upload-btn" id="inferenceRefreshBtn">Refresh</button>
        </div>
        <div class="upload-status" id="inferenceUploadStatus"></div>
      </div>
      <div id="runList"></div>
    </aside>
    <section id="detail">
      <div class="empty" id="detailPlaceholder">Select a run</div>
    </section>
  </main>
<script>
const state = {
  runs: [], runId: null, tab: "hierarchy_kv",
  pages: [], chunks: null, pagesSubtab: "pages",
  agentTree: null, info: null,
  evalReport: null, evalError: null, evalLoading: false,
  agenticEvals: {}, agenticEvalInflight: [], agenticEvalError: null,
  agenticChats: {},
  evalOpenDetails: new Set(),
  gtEdit: null,
  batchJob: null, batchPollTimer: null,
  inferenceJob: null,
  datasets: [], inferSource: "files", inferDataset: "", expandedGroups: new Set(),
  evalKey: null, embed: false,
  evalHierarchyKeys: [], evalHierarchyKeysLoading: false,
  loadedRunId: null,
  agentTreeCache: { runId: null, kv: null, eval: {} },
  agentTreeLoading: false,
  pagesChunksLoadedFor: null,
  wrongCaseKeys: new Set(),
  wrongCasesByKey: {},
};

function showToast(msg, actionText, actionFn) {
  let toast = document.getElementById("appToast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "appToast";
    toast.style.cssText = "position:fixed;bottom:24px;right:24px;background:#152438;color:#e7ecf3;padding:10px 18px;border-radius:8px;border:1px solid #3d86c6;box-shadow:0 6px 20px rgba(0,0,0,0.5);font-size:12px;z-index:9999;display:flex;align-items:center;gap:12px;transition:opacity 0.2s ease;font-family:var(--sans);";
    document.body.appendChild(toast);
  }
  toast.innerHTML = `<span>${esc(msg)}</span>` + (actionText ? `<button type="button" style="background:#1a4971;border:1px solid #3d86c6;color:#fff;border-radius:999px;padding:3px 10px;cursor:pointer;font-size:11px">${esc(actionText)}</button>` : "");
  if (actionText && actionFn) {
    toast.querySelector("button").onclick = actionFn;
  }
  toast.style.opacity = "1";
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.style.opacity = "0"; }, 4500);
}

async function loadWrongCasesForRun() {
  if (!state.runId) {
    state.wrongCaseKeys = new Set();
    state.wrongCasesByKey = {};
    return;
  }
  try {
    const res = await api(`/api/wrong-cases?run_id=${encodeURIComponent(state.runId)}`);
    state.wrongCaseKeys = new Set((res.cases || []).map(c => c.key));
    state.wrongCasesByKey = {};
    (res.cases || []).forEach(c => { state.wrongCasesByKey[c.key] = c; });
  } catch (_) {
    state.wrongCaseKeys = new Set();
    state.wrongCasesByKey = {};
  }
}

function resetAgentTreeCache(runId) {
  state.agentTreeCache = { runId: runId, kv: null, eval: {} };
}

function getCachedAgentTree() {
  if (state.agentTreeCache.runId !== state.runId) return null;
  if (isEvalHierarchyView()) {
    return state.evalKey ? state.agentTreeCache.eval[state.evalKey] : null;
  }
  return state.agentTreeCache.kv;
}

function storeCachedAgentTree(tree) {
  if (!state.runId) return;
  if (state.agentTreeCache.runId !== state.runId) {
    resetAgentTreeCache(state.runId);
  }
  if (isEvalHierarchyView() && state.evalKey) {
    state.agentTreeCache.eval[state.evalKey] = tree;
  } else {
    state.agentTreeCache.kv = tree;
  }
}

function isEvalHierarchyView() {
  return state.tab === "hierarchy_eval" || (state.embed && state.evalKey);
}

function runFileUrl(relPath) {
  const encRun = encodeURIComponent(state.runId);
  const encPath = encodeURIComponent(relPath);
  if (isEvalHierarchyView() && state.evalKey) {
    const encKey = encodeURIComponent(state.evalKey);
    return `/api/runs/${encRun}/agentic-eval/${encKey}/file?path=${encPath}`;
  }
  return `/api/runs/${encRun}/file?path=${encPath}`;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch (_) { data = { detail: text }; }
    throw new Error(data.detail || text || r.statusText);
  }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) return r.json();
  return r.text();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

function shortJson(obj, limit=1200) {
  const s = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  if (s.length <= limit) return s;
  return s.slice(0, limit) + `\n… (${s.length - limit} more chars)`;
}

/** Parse nested JSON strings so escaped blobs become readable objects. */
function deepParse(value, depth=0) {
  if (depth > 6) return value;
  if (typeof value === "string") {
    const t = value.trim();
    if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
      try { return deepParse(JSON.parse(t), depth + 1); } catch (_) { return value; }
    }
    return value;
  }
  if (Array.isArray(value)) return value.map(v => deepParse(v, depth + 1));
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = deepParse(v, depth + 1);
    return out;
  }
  return value;
}

function pretty(value, limit=4000) {
  return shortJson(deepParse(value), limit);
}

function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(2);
}

function runRecord(id) {
  if (id && typeof id === "object") return id;
  return state.runs.find(r => r.run_id === id) || { run_id: String(id || "") };
}

function runDocument(runOrId) {
  const r = runRecord(runOrId);
  return r.document || (r.eval_summary && r.eval_summary.document) || null;
}

function runLabelHtml(runOrId) {
  const r = runRecord(runOrId);
  const id = esc(r.run_id);
  const doc = runDocument(r);
  if (!doc) return `<span class="run-id">${id}</span>`;
  return `<span class="run-label"><span class="run-doc">${esc(doc)}</span><span class="run-id">${id}</span></span>`;
}

function runLabelText(runOrId) {
  const r = runRecord(runOrId);
  const doc = runDocument(r);
  if (!doc) return r.run_id;
  return `${doc} · ${r.run_id}`;
}

function isRunBusy(runId) {
  // Allow deleting running or busy runs
  return false;
}

async function apiDelete(path) {
  const r = await fetch(path, { method: "DELETE" });
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch (_) { data = { detail: text }; }
  if (!r.ok) throw new Error(data.detail || text || r.statusText);
  return data;
}

async function deleteRun(runId, ev) {
  if (ev) {
    ev.preventDefault();
    ev.stopPropagation();
  }
  if (!runId) return;
  const r = runRecord(runId);
  const isRunning = r && r.status === "running";
  const label = runLabelText(runId);
  const promptMsg = isRunning
    ? `Delete this running run permanently?\n\n${label}\n\n(This will cancel any active tasks and remove the run directory.)`
    : `Delete this run permanently?\n\n${label}`;
  if (!confirm(promptMsg)) return;
  try {
    await apiDelete(`/api/runs/${encodeURIComponent(runId)}?force=true`);
    state.runs = state.runs.filter(item => item.run_id !== runId);
    if (state.runId === runId) {
      state.runId = null;
      const next = state.runs[0];
      if (next) {
        await selectRun(next.run_id);
      } else {
        const detail = document.getElementById("detail");
        if (detail) {
          detail.innerHTML = '<div class="empty" id="detailPlaceholder">Select a run</div>';
        }
        renderRuns();
      }
      return;
    }
    renderRuns();
  } catch (err) {
    alert(String(err.message || err));
  }
}

async function deleteGroup(groupKey, ev) {
  if (ev) {
    ev.preventDefault();
    ev.stopPropagation();
  }
  const groups = groupRuns(state.runs);
  const grp = groups.find(g => g.key === groupKey);
  if (!grp || !grp.runs.length) return;

  const nRuns = grp.runs.length;
  const hasRunning = grp.runs.some(r => r.status === "running");
  const msg = hasRunning
    ? `Delete all ${nRuns} run(s) in group "${grp.name}" permanently?\n\n(Includes running runs. This cannot be undone.)`
    : `Delete all ${nRuns} run(s) in group "${grp.name}" permanently?\n\nThis cannot be undone.`;
  if (!confirm(msg)) return;

  try {
    const runIds = grp.runs.map(r => r.run_id);
    const res = await apiPost("/api/runs/delete-batch", { run_ids: runIds, force: true });
    const deletedSet = new Set(res.deleted || runIds);
    state.runs = state.runs.filter(r => !deletedSet.has(r.run_id));
    state.expandedGroups.delete(groupKey);

    if (state.runId && deletedSet.has(state.runId)) {
      state.runId = null;
      const next = state.runs[0];
      if (next) {
        await selectRun(next.run_id);
      } else {
        const detail = document.getElementById("detail");
        if (detail) {
          detail.innerHTML = '<div class="empty" id="detailPlaceholder">Select a run</div>';
        }
        renderRuns();
      }
      return;
    }
    renderRuns();
  } catch (err) {
    alert("Failed to delete group: " + String(err.message || err));
  }
}

function renderUploadPanel() {
  if (state.embed) return;
  const statusEl = document.getElementById("inferenceUploadStatus");
  const btn = document.getElementById("inferenceUploadBtn");
  const input = document.getElementById("inferenceUploadInput");
  const select = document.getElementById("inferenceDatasetSelect");
  const job = state.inferenceJob;
  const active = job && (job.status === "queued" || job.status === "running");
  const useDataset = state.inferSource === "dataset";
  if (btn) btn.disabled = active;
  if (input) {
    input.disabled = active;
    input.hidden = useDataset;
  }
  if (select) {
    select.hidden = !useDataset;
    select.disabled = active;
    const current = state.inferDataset || select.value;
    const opts = [`<option value="">Select a dataset…</option>`].concat(
      (state.datasets || []).map(d => {
        const value = `${d.source}/${d.id}`;
        return `<option value="${esc(value)}">${esc(d.name || d.id)} (${d.n_files || 0} PDF${d.source === "folder" ? ", folder" : ""})</option>`;
      })
    );
    select.innerHTML = opts.join("");
    if (current && [...select.options].some(o => o.value === current)) {
      select.value = current;
      state.inferDataset = current;
    }
    select.onchange = () => { state.inferDataset = select.value; };
  }
  document.querySelectorAll('input[name="inferSource"]').forEach(radio => {
    radio.disabled = active;
    radio.checked = radio.value === state.inferSource;
  });
  const autoEvalCheckbox = document.getElementById("inferenceAutoEval");
  if (autoEvalCheckbox) autoEvalCheckbox.disabled = active;
  if (!statusEl) return;
  if (!job) {
    statusEl.className = "upload-status";
    statusEl.textContent = "";
    return;
  }
  const pct = job.progress_pct ?? (job.total ? Math.round(100 * job.completed / job.total) : 0);
  const cur = job.current ? ` · ${esc(job.current.filename)}` : "";
  const failed = job.failed ? ` · failed ${job.failed}` : "";
  const ds = (job.run_group_name || job.dataset_name) ? ` · ${esc(job.run_group_name || job.dataset_name)}` : "";
  const autoEvalTag = job.auto_eval ? ` · <span style="color:var(--accent,#4daafc)">auto-eval: on</span>` : "";
  statusEl.className = `upload-status ${job.status === "running" || job.status === "queued" ? "running" : (job.failed ? "error" : "done")}`;
  statusEl.innerHTML = `
    <div><b>${esc(job.status)}</b> ${job.completed}/${job.total} (${pct}%)${ds}${autoEvalTag}${cur}${failed}</div>
    ${(job.tasks || []).map(t => {
      const evalTag = t.eval_status ? ` · <span style="color:var(--accent,#4daafc)">eval: ${esc(t.eval_status)}</span>` : "";
      return `<div>${esc(t.filename)}: ${esc(t.status)}${t.run_id ? ` → ${esc(t.run_id)}` : ""}${evalTag}${t.error ? ` (${esc(t.error)})` : ""}</div>`;
    }).join("")}
    ${job.message ? `<div>${esc(job.message)}</div>` : ""}`;
}

async function refreshDatasets() {
  try {
    const ds = await api("/api/datasets");
    state.datasets = ds.datasets || [];
  } catch (_) {
    state.datasets = [];
  }
}

async function refreshInferenceUploadJob() {
  try {
    if (state.inferenceJob?.job_id) {
      state.inferenceJob = await api(`/api/inference/jobs/${encodeURIComponent(state.inferenceJob.job_id)}`);
    } else {
      const data = await api("/api/inference/jobs/active");
      state.inferenceJob = data.job || state.inferenceJob;
    }
    if (state.inferenceJob && (state.inferenceJob.status === "done" || state.inferenceJob.status === "partial" || state.inferenceJob.status === "error")) {
      await refreshDatasets();
    }
    if (state.inferenceJob?.eval_job_id && (!state.batchJob || state.batchJob.status !== "running")) {
      const bData = await api("/api/evaluation/batch-jobs/active");
      if (bData.job) {
        state.batchJob = bData.job;
      }
    }
    renderUploadPanel();
    state.runs = await api("/api/runs");
    renderRuns();
  } catch (err) {
    const statusEl = document.getElementById("inferenceUploadStatus");
    if (statusEl) {
      statusEl.className = "upload-status error";
      statusEl.textContent = String(err.message || err);
    }
  }
}

async function resumeInferenceUploadJob() {
  try {
    const data = await api("/api/inference/jobs/active");
    if (data.job) {
      state.inferenceJob = data.job;
      renderUploadPanel();
    }
  } catch (_) { /* ignore */ }
}

async function startInferenceUpload() {
  const statusEl = document.getElementById("inferenceUploadStatus");
  if (state.inferenceJob && (state.inferenceJob.status === "queued" || state.inferenceJob.status === "running")) {
    return;
  }
  const autoEvalCheckbox = document.getElementById("inferenceAutoEval");
  const autoEval = autoEvalCheckbox ? autoEvalCheckbox.checked : true;
  let start;
  if (state.inferSource === "dataset") {
    const select = document.getElementById("inferenceDatasetSelect");
    const value = select && select.value;
    if (!value || !value.includes("/")) {
      if (statusEl) {
        statusEl.className = "upload-status error";
        statusEl.textContent = "Select a dataset.";
      }
      return;
    }
    const slash = value.indexOf("/");
    const source = value.slice(0, slash);
    const datasetId = value.slice(slash + 1);
    start = () => apiPost("/api/inference/jobs/from-dataset", {
      source,
      dataset_id: datasetId,
      auto_eval: autoEval,
    });
  } else {
    const input = document.getElementById("inferenceUploadInput");
    if (!input || !input.files || !input.files.length) {
      if (statusEl) {
        statusEl.className = "upload-status error";
        statusEl.textContent = "Select one or more PDF files.";
      }
      return;
    }
    const form = new FormData();
    for (const file of input.files) {
      form.append("files", file, file.name);
    }
    form.append("auto_eval", autoEval ? "true" : "false");
    start = async () => {
      const r = await fetch("/api/inference/jobs", { method: "POST", body: form });
      const text = await r.text();
      let data;
      try { data = JSON.parse(text); } catch (_) { data = { detail: text }; }
      if (!r.ok) throw new Error(data.detail || text || r.statusText);
      input.value = "";
      return data;
    };
  }
  if (statusEl) {
    statusEl.className = "upload-status running";
    statusEl.textContent = "Starting…";
  }
  try {
    state.inferenceJob = await start();
    if (state.inferSource === "files" && state.inferenceJob.dataset_id) {
      await refreshDatasets();
    }
    renderUploadPanel();
  } catch (err) {
    if (statusEl) {
      statusEl.className = "upload-status error";
      statusEl.textContent = String(err.message || err);
    }
  }
}

function runItemHtml(r) {
  const es = r.eval_summary;
  const evalLine = es
    ? `<div class="eval-mini">EM ${fmtPct(es.value_exact_match)} · pageF1 ${fmtPct(es.page_f1_macro)} · evidF1 ${fmtPct(es.evidence_token_f1)}</div>`
    : "";
  const ae = r.agentic_eval_summary;
  const agenticLine = ae && ae.n_total
    ? `<div class="eval-mini">agentic ${ae.n_done}/${ae.n_total}${ae.accuracy != null ? ` · pred acc ${fmtPct(ae.accuracy)}` : ""}${ae.gold_validity != null ? ` · GT valid ${fmtPct(ae.gold_validity)}` : ""}</div>`
    : "";
  const busy = isRunBusy(r.run_id);
  return `
    <div class="run ${r.run_id === state.runId ? "active" : ""}" data-id="${esc(r.run_id)}">
      <div class="run-head">
        <div class="id">${runLabelHtml(r)}</div>
        <button type="button" class="run-delete" data-delete-run="${esc(r.run_id)}"
          title="Delete run" aria-label="Delete run" ${busy ? "disabled" : ""}>×</button>
      </div>
      <div class="sub">
        <span class="badge ${r.status === "ok" ? "ok" : (r.status === "error" ? "error" : (r.status === "running" ? "warn" : ""))}">${esc(r.status)}</span>
        ${r.seconds != null ? r.seconds + "s" : ""} · kv=${r.n_kv ?? "?"} · pages=${r.page_count ?? "?"}
      </div>
      ${evalLine}
      ${agenticLine}
    </div>`;
}

function groupRuns(runs) {
  const groups = [];
  const index = new Map();
  for (const r of runs) {
    const groupId = r.run_group_id || r.dataset_id;
    const grouped = Boolean(groupId);
    const key = grouped ? `${r.dataset_source || "managed"}/${groupId}` : "ungrouped";
    if (!index.has(key)) {
      index.set(key, groups.length);
      const name = grouped
        ? (r.run_group_name || r.dataset_name || r.dataset_id)
        : "Ungrouped";
      groups.push({
        key,
        name,
        dataset_id: r.dataset_id,
        run_group_id: r.run_group_id,
        runs: [],
        latest: 0,
      });
    }
    const g = groups[index.get(key)];
    g.runs.push(r);
    const ts = Date.parse(r.started_at || "") || 0;
    if (ts > g.latest) g.latest = ts;
  }
  groups.sort((a, b) => b.latest - a.latest);
  for (const g of groups) {
    g.runs.sort((a, b) => (Date.parse(b.started_at || "") || 0) - (Date.parse(a.started_at || "") || 0));
  }
  return groups;
}

function bindRunList(el) {
  el.querySelectorAll(".run").forEach(node => {
    node.onclick = () => selectRun(node.dataset.id);
  });
  el.querySelectorAll("[data-delete-run]").forEach(btn => {
    btn.onclick = (ev) => deleteRun(btn.dataset.deleteRun, ev);
  });
  el.querySelectorAll("[data-delete-group]").forEach(btn => {
    btn.onclick = (ev) => deleteGroup(btn.dataset.deleteGroup, ev);
  });
  el.querySelectorAll(".dataset-group").forEach(node => {
    node.addEventListener("toggle", () => {
      if (node.open) state.expandedGroups.add(node.dataset.group);
      else state.expandedGroups.delete(node.dataset.group);
    });
  });
}

function renderRuns() {
  if (state.embed) return;
  const el = document.getElementById("runList");
  if (!el) return;
  const prevScrollTop = el.scrollTop;
  if (!state.runs.length) {
    el.innerHTML = `<div class="empty" style="padding:16px">No runs in outputs/runs</div>`;
  } else {
    el.innerHTML = groupRuns(state.runs).map(g => {
      const open = state.expandedGroups.has(g.key);
      const nOk = g.runs.filter(r => r.status === "ok").length;
      return `
        <details class="dataset-group" data-group="${esc(g.key)}" ${open ? "open" : ""}>
          <summary>
            <span class="group-title">${esc(g.name)}</span>
            <span class="group-actions">
              <span class="count">${nOk}/${g.runs.length}</span>
              <button type="button" class="group-delete-btn" data-delete-group="${esc(g.key)}"
                title="Delete all ${g.runs.length} run(s) in this group">Delete group</button>
            </span>
          </summary>
          ${g.runs.map(runItemHtml).join("")}
        </details>`;
    }).join("");
    bindRunList(el);
    el.scrollTop = prevScrollTop;
  }
  document.getElementById("headerMeta").textContent =
    `${state.runs.length} run(s) · ${location.origin}`;
  renderUploadPanel();
}

async function selectRun(runId, opts = {}) {
  const keepTab = Boolean(opts.keepTab);
  const keepEvalKey = Boolean(opts.keepEvalKey);
  state.runId = runId;
  if (runId) {
    const groups = groupRuns(state.runs);
    const grp = groups.find(g => g.runs.some(r => r.run_id === runId));
    if (grp) state.expandedGroups.add(grp.key);
  }
  if (!state.embed && !keepTab) state.tab = "hierarchy_kv";
  state.pagesSubtab = "pages";
  state.pages = [];
  state.chunks = null;
  if (!keepEvalKey) {
    state.evalKey = null;
    state.evalHierarchyKeys = [];
  }
  state.loadedRunId = null;
  state.agentTree = null;
  state.agentTreeLoading = false;
  resetAgentTreeCache(runId);
  state.pagesChunksLoadedFor = null;
  if (!state.embed) {
    state.evalReport = null;
    state.evalError = null;
    state.evalLoading = false;
    state.agenticEvals = {};
    state.agenticEvalInflight = [];
    state.agenticEvalError = null;
    state.evalOpenDetails = new Set();
    state.wrongCaseKeys = new Set();
    state.wrongCasesByKey = {};
    state.batchJob = null;
    stopBatchPoll();
  }
  renderRuns();
  await renderDetail();
  const detail = document.getElementById("detail");
  if (detail) detail.scrollTop = 0;
}

async function loadEvalHierarchyKeys() {
  if (!state.runId) {
    state.evalHierarchyKeys = [];
    return;
  }
  state.evalHierarchyKeysLoading = true;
  try {
    const data = await api(
      `/api/runs/${encodeURIComponent(state.runId)}/agentic-eval/keys`
    );
    state.evalHierarchyKeys = data.keys || [];
  } catch (_) {
    state.evalHierarchyKeys = [];
  } finally {
    state.evalHierarchyKeysLoading = false;
  }
}

async function loadAgentTree() {
  if (!state.runId) return null;
  if (isEvalHierarchyView()) {
    if (!state.evalKey) return null;
    return api(
      `/api/runs/${encodeURIComponent(state.runId)}/agentic-eval/${encodeURIComponent(state.evalKey)}/agent-tree`
    );
  }
  return api(`/api/runs/${encodeURIComponent(state.runId)}/agent-tree`);
}

async function loadKvTreeCached() {
  if (!state.runId) return null;
  if (state.agentTreeCache.runId === state.runId && state.agentTreeCache.kv) {
    return state.agentTreeCache.kv;
  }
  const tree = await api(`/api/runs/${encodeURIComponent(state.runId)}/agent-tree`);
  if (state.agentTreeCache.runId !== state.runId) {
    resetAgentTreeCache(state.runId);
  }
  state.agentTreeCache.kv = tree;
  return tree;
}

async function loadPagesChunks() {
  if (!state.runId || state.pagesChunksLoadedFor === state.runId) return;
  const pagesPayload = await api(
    `/api/runs/${encodeURIComponent(state.runId)}/pages`
  );
  state.pages = Array.isArray(pagesPayload)
    ? pagesPayload
    : (pagesPayload?.pages || []);
  state.pagesMeta = Array.isArray(pagesPayload) ? null : pagesPayload;
  state.chunks = await api(
    `/api/runs/${encodeURIComponent(state.runId)}/chunks?limit=200`
  );
  state.pagesChunksLoadedFor = state.runId;
}

async function loadHierarchyTab() {
  if (!state.runId) return;
  if (state.tab === "hierarchy_eval") {
    await loadEvalHierarchyKeys();
    if (!state.evalKey && state.evalHierarchyKeys.length) {
      const done = state.evalHierarchyKeys.find(k => k.status === "done");
      state.evalKey = (done || state.evalHierarchyKeys[0]).key;
    }
    if (!state.evalKey) {
      state.agentTree = null;
      state.agentTreeLoading = false;
      paintDetail();
      return;
    }
  }

  const cached = getCachedAgentTree();
  if (cached) {
    state.agentTree = cached;
    state.agentTreeLoading = false;
    paintDetail();
    return;
  }

  state.agentTreeLoading = true;
  paintDetail();
  try {
    state.agentTree = await loadAgentTree();
    storeCachedAgentTree(state.agentTree);
  } finally {
    state.agentTreeLoading = false;
    paintDetail();
  }
}

async function loadTimingTab() {
  if (!state.runId) return;
  state.agentTreeLoading = true;
  paintDetail();
  try {
    state.agentTree = await loadKvTreeCached();
  } finally {
    state.agentTreeLoading = false;
    paintDetail();
  }
}

async function renderDetail(opts = {}) {
  const force = Boolean(opts.force);
  const detail = document.getElementById("detail");
  if (!state.runId) {
    detail.innerHTML = `<div class="empty">Select a run</div>`;
    return;
  }
  const keepTab = state.tab;
  const runChanged = state.loadedRunId !== state.runId;

  if (force) {
    resetAgentTreeCache(state.runId);
    state.pagesChunksLoadedFor = null;
    state.agentTree = null;
  }

  if (state.embed) {
    if (runChanged || force || !state.info) {
      detail.innerHTML = `<div class="empty">Loading ${esc(runLabelText(state.runId))}…</div>`;
      state.info = await api(`/api/runs/${encodeURIComponent(state.runId)}`);
      state.loadedRunId = state.runId;
    }
    state.tab = keepTab;
    await loadHierarchyTab();
    return;
  }

  if (runChanged || force || !state.info) {
    detail.innerHTML = `<div class="empty">Loading ${esc(runLabelText(state.runId))}…</div>`;
    state.info = await api(`/api/runs/${encodeURIComponent(state.runId)}`);
    state.loadedRunId = state.runId;
    if (runChanged) {
      resetAgentTreeCache(state.runId);
      state.pagesChunksLoadedFor = null;
      state.agentTree = null;
    }
  }

  state.tab = keepTab;

  if (state.tab === "hierarchy_kv" || state.tab === "hierarchy_eval") {
    await loadHierarchyTab();
    return;
  }

  if (state.tab === "timing") {
    await loadTimingTab();
    return;
  }

  if (state.tab === "pages") {
    if (state.pagesChunksLoadedFor !== state.runId) {
      state.agentTreeLoading = true;
      paintDetail();
      try {
        await loadPagesChunks();
      } finally {
        state.agentTreeLoading = false;
      }
    }
    paintDetail();
    return;
  }

  paintDetail();
}

function tabsHtml() {
  if (state.embed) return "";
  const tabs = [
    ["hierarchy_kv", "KV hierarchy"],
    ["hierarchy_eval", "Eval hierarchy"],
    ["timing", "Timing"],
    ["pages", "Pages / Chunks"],
    ["eval", "Eval"],
  ];
  return `<div class="tabs">${tabs.map(([id, label]) =>
    `<button class="tab ${state.tab===id?"active":""}" data-tab="${id}">${label}</button>`
  ).join("")}</div>`;
}

function renderEvalHierarchyKeyToolbar() {
  if (state.embed || state.tab !== "hierarchy_eval") return "";
  const batch = state.batchJob;
  const batchActive = batch && (batch.status === "queued" || batch.status === "running");
  const keyOpts = (state.evalHierarchyKeys || []).map(row => {
    const status = row.status || "pending";
    const verdict = row.is_correct_answer ? ` · pred ${row.is_correct_answer}` : "";
    const goldVerdict = row.is_valid_gold ? ` · GT ${row.is_valid_gold}` : "";
    return `<option value="${esc(row.key)}" ${row.key === state.evalKey ? "selected" : ""}>${esc(row.key)} (${esc(status)}${esc(verdict)}${esc(goldVerdict)})</option>`;
  }).join("");
  return `
    <div class="hierarchy-toolbar" style="display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap">
      <label style="font-size:12px;color:var(--muted)">Eval key
        <select id="evalHierarchyKey" style="min-width:220px;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:#0f1419;color:var(--text);font-family:var(--mono);font-size:12px"
          ${state.evalHierarchyKeysLoading ? "disabled" : ""}>
          ${keyOpts || `<option value="">—</option>`}
        </select>
      </label>
      ${state.evalKey ? `<button type="button" class="agentic-eval-btn" id="evalHierarchyRetry"
        ${batchActive ? "disabled" : ""}>Retry eval</button>` : ""}
      ${state.evalHierarchyKeysLoading ? `<span class="tree-kv">Loading keys…</span>` : ""}
    </div>`;
}

function masterAgentLabel() {
  return state.agentTree?.agent_kind === "eval" ? "EvalMaster" : "Master";
}

function fmtSec(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return n >= 100 ? `${n.toFixed(0)}s` : `${n.toFixed(1)}s`;
}

function runClockBase() {
  return state.agentTree?.timing?.run_started_at || state.info?.meta?.started_at;
}

function fmtRunClock(relativeSec) {
  if (relativeSec == null || Number.isNaN(Number(relativeSec))) return "—";
  const started = runClockBase();
  if (!started) {
    const n = Number(relativeSec);
    return n >= 100 ? `t+${n.toFixed(0)}s` : `t+${n.toFixed(1)}s`;
  }
  const ms = new Date(started).getTime() + Number(relativeSec) * 1000;
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

function fmtAbsDateTime(relativeSec) {
  if (relativeSec == null || Number.isNaN(Number(relativeSec))) return "—";
  const started = runClockBase();
  if (!started) {
    const n = Number(relativeSec);
    return n >= 100 ? `t+${n.toFixed(0)}s` : `t+${n.toFixed(1)}s`;
  }
  const ms = new Date(started).getTime() + Number(relativeSec) * 1000;
  return new Date(ms).toLocaleString([], {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
}

function masterTurnTimingBadge(timing) {
  if (!timing) return "";
  const wall = timing.wall_seconds ?? timing.llm_seconds;
  if (wall == null) return "";
  const start = timing.start_t;
  const end = timing.end_t;
  const range = (start != null && end != null)
    ? ` · ${fmtRunClock(start)} → ${fmtRunClock(end)}`
    : "";
  return `<span class="tree-badge master">${fmtSec(wall)}${range}</span>`;
}

function renderMasterTurnTimingDetail(mt) {
  const timing = mt.timing;
  if (!timing) return "";
  const wall = timing.wall_seconds ?? timing.llm_seconds;
  const start = timing.start_t;
  const end = timing.end_t;
  const bits = [
    wall != null ? `<div class="tree-kv"><b>Duration</b> ${esc(fmtSec(wall))}</div>` : "",
    start != null ? `<div class="tree-kv"><b>Start</b> ${esc(fmtAbsDateTime(start))}</div>` : "",
    end != null ? `<div class="tree-kv"><b>End</b> ${esc(fmtAbsDateTime(end))}</div>` : "",
  ].filter(Boolean);
  if (!bits.length) return "";
  const extra = [
    timing.llm_seconds != null ? `model ${fmtSec(timing.llm_seconds)}` : null,
    timing.tool_seconds != null ? `tools/overhead ${fmtSec(timing.tool_seconds)}` : null,
    timing.search_wall_seconds != null ? `search wall ${fmtSec(timing.search_wall_seconds)}` : null,
  ].filter(Boolean);
  return `<div class="viz-section" style="margin:0 0 10px">
    <h3 style="margin:0 0 6px">Timing</h3>
    ${bits.join("")}
    ${extra.length ? `<div class="tree-kv" style="margin-top:4px;color:var(--muted)">${esc(extra.join(" · "))}</div>` : ""}
  </div>`;
}

function timingBadge(timing, kind="") {
  if (!timing) return "";
  const wall = timing.wall_seconds ?? timing.llm_seconds;
  if (wall == null) return "";
  const pct = timing.pct != null ? ` · ${timing.pct}%` : "";
  const model = timing.llm_seconds != null && timing.wall_seconds != null
    ? ` · model ${fmtSec(timing.llm_seconds)}` : "";
  return `<span class="tree-badge ${kind}">${fmtSec(wall)}${model}${pct}</span>`;
}

function tokenLabel(node) {
  if (!node) return "";
  const inp = node.input_tokens;
  const out = node.output_tokens;
  if (inp != null || out != null) {
    return `in=${inp ?? "—"} out=${out ?? "—"}`;
  }
  if (node.prompt_est_tokens != null) {
    return `in≈${node.prompt_est_tokens}`;
  }
  return "";
}

function tokenBadge(node, kind="") {
  const label = tokenLabel(node);
  if (!label) return "";
  return `<span class="tree-badge ${kind}">${esc(label)}</span>`;
}

function fmtTok(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString();
}

function masterTurnTokenBadges(mt) {
  const label = tokenLabel(mt);
  let html = "";
  if (label) {
    html += `<span class="tree-badge master" title="Master LLM usage for this turn (out = assistant/tool-call generation)">${esc(label)}</span>`;
  }
  const toolMsgs = mt.tool_message_est_tokens;
  if (toolMsgs != null && toolMsgs > 0) {
    html += `<span class="tree-badge" title="Estimated tokens appended as tool-role messages before the next Master turn">tool msgs≈${fmtTok(toolMsgs)}</span>`;
  }
  return html;
}

function toolMessageTokenHint(tool) {
  const n = tool.message_est_tokens;
  if (n == null) return "";
  return ` · tool message≈${fmtTok(n)} tok (added to conversation)`;
}

function renderTimingBar(label, seconds, pct, cls="") {
  const width = Math.max(0.5, Math.min(100, Number(pct) || 0));
  return `<div class="timing-row">
    <div class="label" title="${esc(label)}">${esc(label)}</div>
    <div class="timing-bar-wrap"><div class="timing-bar ${cls}" style="width:${width}%"></div></div>
    <div>${fmtSec(seconds)}</div>
    <div>${esc(pct ?? 0)}%</div>
  </div>`;
}

function renderHandoffBox(title, payload) {
  if (payload == null || payload === "") return "";
  return `<div class="viz-section" style="margin-top:8px">
    <h3 style="margin:0 0 4px">${esc(title)}</h3>
    <pre class="pretty">${esc(pretty(payload, 3500))}</pre>
  </div>`;
}

function renderMasterPrompts(prompts) {
  if (!prompts) return "";
  const sys = prompts.system;
  const user = prompts.user;
  if (!sys && !user) return "";
  return `<details class="tree-node master-prompts">
    <summary>
      <span class="title">Master prompts</span>
      <span class="tree-badge">system + user</span>
    </summary>
    <div class="tree-body">
      ${sys ? `<div class="flow-item system" style="margin-bottom:8px">
        <div class="label">System prompt</div>
        <pre class="pretty" style="max-height:320px;margin:4px 0 0">${esc(sys)}</pre>
      </div>` : ""}
      ${user ? `<div class="flow-item user">
        <div class="label">User prompt</div>
        <pre class="pretty" style="max-height:200px;margin:4px 0 0">${esc(user)}</pre>
      </div>` : ""}
    </div>
  </details>`;
}

function renderSearchInitialState(initialState, prompts, opts = {}) {
  if (!initialState && !prompts) return "";
  const init = initialState || {};
  const p = prompts || {};
  const keys = Array.isArray(init.keys) ? init.keys : [];
  const outline = init.document_outline || "";
  const prior = init.prior_context || null;
  const sys = p.system || "";
  const user = p.user || "";

  if (!keys.length && !outline && !prior && !sys && !user) {
    return "";
  }

  const nKeys = keys.length;
  const outlineLines = outline ? outline.split("\n").length : 0;

  // 1. User Prompt Internal Sequence Bar
  const userSeqBar = `
    <div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;font-size:11px;margin:8px 0 10px;padding:6px 10px;background:rgba(0,0,0,0.3);border-radius:6px;border:1px solid var(--line);line-height:1.4">
      <span style="color:var(--muted);font-weight:600">User Prompt Section Sequence:</span>
      <span class="tree-badge ok" title="Search goal instructions">1. Task Instruction</span>
      <span style="color:var(--muted)">➔</span>
      <span class="tree-badge ok" title="Keys with search cues & allowed values">2. Target Keys & Schema Descriptions</span>
      <span style="color:var(--muted)">➔</span>
      <span class="tree-badge ok" title="Pre-injected Compact TOC">3. Document Outline TOC</span>
      ${prior ? '<span style="color:var(--muted)">➔</span><span class="tree-badge warn" title="Pages inspected & queries from prior session">4. Prior Session Progress</span>' : ''}
      <span style="color:var(--muted)">➔</span>
      <span class="tree-badge ok" title="Tool rules & submit_pages requirements">${prior ? '5' : '4'}. Tool Rules & Submit Instructions</span>
    </div>`;

  // 2. Raw LLM Messages (system first, then user)
  const messagesHtml = `
    <div style="display:flex;flex-direction:column;gap:10px;margin-top:4px">
      <!-- Message 0: system -->
      <details class="viz-section" style="margin:0">
        <summary style="cursor:pointer;font-size:12px;color:var(--text);font-weight:600;display:flex;align-items:center;gap:8px;user-select:none">
          <span class="tree-badge" style="color:#b197fc;border-color:#5c3e9e;background:rgba(155,123,212,0.12)">Message [0] · role: "system"</span>
          <span>SearchAgent System Prompt</span>
          <span class="tree-badge">Instructions & Tool Rules</span>
        </summary>
        <div style="margin-top:8px">
          <pre class="pretty" style="max-height:280px;margin:0;font-size:11px;line-height:1.45;white-space:pre-wrap">${esc(sys || "(system prompt not recorded in trace)")}</pre>
        </div>
      </details>

      <!-- Message 1: user -->
      <details class="viz-section" style="margin:0" open>
        <summary style="cursor:pointer;font-size:12px;color:var(--text);font-weight:600;display:flex;align-items:center;gap:8px;user-select:none">
          <span class="tree-badge" style="color:#74c0fc;border-color:#2a6a9e;background:rgba(61,156,240,0.12)">Message [1] · role: "user"</span>
          <span>SearchAgent Initial User Prompt</span>
          <span class="tree-badge ok">${esc(nKeys)} key(s)</span>
          ${outline ? `<span class="tree-badge">TOC outline</span>` : ""}
        </summary>
        <div style="margin-top:8px">
          ${userSeqBar}
          <pre class="pretty" style="max-height:420px;margin:0;font-size:11px;line-height:1.45;white-space:pre-wrap">${esc(user || "(user prompt not recorded in trace)")}</pre>
        </div>
      </details>
    </div>`;

  return `<details class="tree-node search-prompts" open>
    <summary>
      <span class="title">SearchAgent LLM Initial Messages (messages payload)</span>
      <span class="tree-badge ok">messages[0]=system, messages[1]=user</span>
      ${nKeys ? `<span class="tree-badge ok">${esc(nKeys)} key(s)</span>` : ""}
      ${outline ? `<span class="tree-badge">outline (${esc(outlineLines)}p)</span>` : ""}
      ${prior ? `<span class="tree-badge warn">prior handoff</span>` : ""}
    </summary>
    <div class="tree-body">
      <div class="hint" style="margin:0 0 6px;line-height:1.4">
        SearchAgent 실행 시 LLM에 전달되는 <b>messages 배열 (role: system ➔ role: user)</b>의 실제 원문 및 주입 순서입니다.
      </div>
      ${messagesHtml}
    </div>
  </details>`;
}

function renderSearchTurn(turn) {
  const err = turn.error ? `<span class="tree-badge err">error</span>` : "";
  const tools = (turn.tool_results || []).map(tr =>
    `<span class="pill">${esc(tr.name)}</span>`
  ).join("") || (turn.tool_calls || []).map(tc =>
    `<span class="pill">${esc(tc.name)}</span>`
  ).join("");
  const submit = turn.submit_output;
  const finishTool = (submit && submit.tool) || "submit_pages";
  const submitHtml = submit ? `
    <div class="viz-section" style="margin-top:8px">
      <h3 style="margin:0 0 4px">Output (${esc(finishTool)})</h3>
      <pre class="pretty">${esc(pretty(submit, 3500))}</pre>
    </div>` : "";
  const assistantHtml = turn.assistant_content ? `
    <div class="viz-section" style="margin-top:8px">
      <h3 style="margin:0 0 4px">Assistant</h3>
      <pre class="pretty">${esc(turn.assistant_content)}</pre>
    </div>` : "";
  return `<details class="tree-node turn">
    <summary>
      <span class="title">Search turn ${esc(turn.search_turn || turn.step)}</span>
      ${err}
      ${submit ? `<span class="tree-badge ok">output</span>` : ""}
      ${timingBadge(turn.timing)}
      ${tokenBadge(turn.timing || turn)}
      ${tools}
    </summary>
    <div class="tree-body">
      ${submitHtml}
      ${assistantHtml}
      <pre class="pretty">${esc(pretty({
        tool_calls: turn.tool_calls,
        tool_results: turn.tool_results,
        error: turn.error,
      }, 3000))}</pre>
      <a href="#" data-step="${esc(turn.filename)}">open step JSON</a>
    </div>
  </details>`;
}

function renderSearchSession(session, opts = {}) {
  const turns = session.turns || [];
  const err = session.error ? `<span class="tree-badge err">error</span>` : "";
  const status = session.status || "";
  const statusCls = status === "complete" ? "ok"
    : (status === "not_found" || String(status).startsWith("handoff") ? "warn" : "");
  const pages = session.pages || [];
  const shared = session.shared || opts.shared;
  const hasIn = session.prior_context_in != null;
  const nSessions = opts.nSessions || 1;
  const isHandoff = String(status).startsWith("handoff");
  const showPriorOut = nSessions > 1 || isHandoff;
  const showHandoffSummary = showPriorOut && !!session.handoff_summary;
  const showPriorOutBox = showPriorOut && session.prior_context_out != null;
  const hasOut = showHandoffSummary || showPriorOutBox;
  const title = shared
    ? (nSessions <= 1
      ? `Shared search (${esc(turns.length)} turn(s))`
      : `Shared search session ${esc(session.session_index)} (${esc(turns.length)} turn(s))`)
    : `Search session ${esc(session.session_index)}`;
  const sessionCls = shared ? "session shared" : "session";
  return `<details class="tree-node ${sessionCls}">
    <summary>
      <span class="title">${title}</span>
      ${!shared && session.key ? `<span class="tree-kv">${esc(session.key)}</span>` : ""}
      ${status ? `<span class="tree-badge ${statusCls}">${esc(status)}</span>` : ""}
      ${!shared ? `<span class="tree-badge">${esc(turns.length)} turn(s)</span>` : ""}
      ${timingBadge(session.timing)}
      ${!shared && pages.length ? `<span class="tree-badge ok">pages=${esc(pages.join(","))}</span>` : ""}
      ${hasIn ? `<span class="tree-badge">received prior</span>` : ""}
      ${hasOut ? `<span class="tree-badge warn">handoff out</span>` : ""}
      ${err}
    </summary>
    <div class="tree-body">
      ${renderSearchInitialState(session.initial_state, session.prompts, { shared })}
      ${renderHandoffBox(
        "Received from previous session (prior_context_in)",
        session.prior_context_in
      )}
      ${showHandoffSummary ? renderHandoffBox(
        "Produced for next session / master (handoff_summary)",
        session.handoff_summary
      ) : ""}
      ${showPriorOutBox ? renderHandoffBox(
        "Produced prior_context_out (structured handoff blob)",
        session.prior_context_out
      ) : ""}
      ${turns.map(renderSearchTurn).join("") || `<div class="empty">No turns linked</div>`}
    </div>
  </details>`;
}

function renderPageReasonsTable(reasons, chunkIds = null, title = "page_reasons") {
  if (!reasons || typeof reasons !== "object") reasons = {};
  const chunks = (chunkIds && typeof chunkIds === "object") ? chunkIds : {};
  const pages = new Set([
    ...Object.keys(reasons || {}),
    ...Object.keys(chunks || {}),
  ]);
  if (!pages.size) return "";
  const reasonRows = [...pages].sort((a, b) => {
    const ai = parseInt(a, 10), bi = parseInt(b, 10);
    if (!Number.isNaN(ai) && !Number.isNaN(bi)) return ai - bi;
    return String(a).localeCompare(String(b));
  }).map(page => {
    const cid = chunks[page] || "";
    const chunkCell = cid
      ? `<button type="button" class="chunk-jump" data-chunk-id="${esc(cid)}" style="padding:2px 8px;border-radius:999px;border:1px solid var(--line);background:#152033;color:var(--accent);font-size:11px;cursor:pointer">${esc(cid)}</button>`
      : "";
    return `<tr>
      <td>${esc(page)}</td>
      <td>${chunkCell}</td>
      <td>${esc(reasons[page] || "")}</td>
    </tr>`;
  }).join("");
  return `<div class="viz-section" style="margin-top:6px">
    <h3 style="margin:0 0 4px">${esc(title)}</h3>
    <table class="kv-table">
      <thead><tr><th>Page</th><th>chunk_id</th><th>Reason</th></tr></thead>
      <tbody>${reasonRows}</tbody>
    </table>
  </div>`;
}

let _pdfJsPromise = null;
const _pdfDocCache = {};

function loadScriptOnce(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === "1") resolve();
      else existing.addEventListener("load", () => resolve(), { once: true });
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.dataset.src = src;
    s.onload = () => { s.dataset.loaded = "1"; resolve(); };
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

async function ensurePdfJs() {
  if (window.pdfjsLib) return window.pdfjsLib;
  if (!_pdfJsPromise) {
    const ver = "3.11.174";
    const base = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${ver}`;
    _pdfJsPromise = loadScriptOnce(`${base}/pdf.min.js`).then(() => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = `${base}/pdf.worker.min.js`;
      return window.pdfjsLib;
    });
  }
  return _pdfJsPromise;
}

function regionToPctStyle(region, pageSize = null) {
  let x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  const hasBbox = Array.isArray(region.bbox) && region.bbox.length === 4;
  const isImagePx = (region.coord === "image_px") || (hasBbox && (region.bbox[2] > 1.5 || region.bbox[3] > 1.5));

  if (isImagePx && pageSize && pageSize.width > 0 && pageSize.height > 0) {
    const bb = region.bbox.map(Number);
    x0 = bb[0] / pageSize.width;
    y0 = bb[1] / pageSize.height;
    x1 = bb[2] / pageSize.width;
    y1 = bb[3] / pageSize.height;
  } else if (Array.isArray(region.bbox_norm) && region.bbox_norm.length === 4) {
    [x0, y0, x1, y1] = region.bbox_norm.map(Number);
  } else if (hasBbox) {
    const w = Number(region.width || 0);
    const h = Number(region.height || 0);
    if (w > 0 && h > 0) {
      const bb = region.bbox.map(Number);
      x0 = bb[0] / w; y0 = bb[1] / h; x1 = bb[2] / w; y1 = bb[3] / h;
    } else {
      return null;
    }
  } else {
    return null;
  }
  if (![x0, y0, x1, y1].every(n => Number.isFinite(n))) return null;
  const left = Math.min(x0, x1);
  const top = Math.min(y0, y1);
  const width = Math.abs(x1 - x0);
  const height = Math.abs(y1 - y0);
  if (width <= 0 || height <= 0) return null;
  return {
    left: `${left * 100}%`,
    top: `${top * 100}%`,
    width: `${width * 100}%`,
    height: `${height * 100}%`,
  };
}

async function getPdfDocument(runId) {
  if (!_pdfDocCache[runId]) {
    const pdfjs = await ensurePdfJs();
    const url = `/api/runs/${encodeURIComponent(runId)}/pdf`;
    _pdfDocCache[runId] = pdfjs.getDocument(url).promise;
  }
  return _pdfDocCache[runId];
}

async function mountChunkPdfViewer(host, runId, regions, fallbackPages=[]) {
  if (!host || !runId) return;
  const grouped = {};
  for (const r of (regions || [])) {
    const page = Number(r.page);
    if (!Number.isFinite(page) || page <= 0) continue;
    if (!grouped[page]) grouped[page] = [];
    grouped[page].push(r);
  }
  let chunkPages = Object.keys(grouped).map(Number).sort((a, b) => a - b);
  if (!chunkPages.length && Array.isArray(fallbackPages) && fallbackPages.length) {
    chunkPages = fallbackPages.map(Number).filter(p => Number.isFinite(p) && p > 0).sort((a, b) => a - b);
    for (const p of chunkPages) {
      if (!grouped[p]) grouped[p] = [];
    }
  }

  host.innerHTML = `<div class="empty">Loading PDF…</div>`;
  let info = null;
  try {
    info = await api(`/api/runs/${encodeURIComponent(runId)}/pdf/info`);
  } catch (_) {
    info = null;
  }
  if (!info || !info.available) {
    host.innerHTML = `<div class="empty">PDF not available for this run.</div>`;
    return;
  }

  let pdf;
  try {
    pdf = await getPdfDocument(runId);
  } catch (err) {
    host.innerHTML = `<div class="empty" style="color:var(--err)">${esc(err.message || err)} · <a href="/api/runs/${encodeURIComponent(runId)}/pdf" target="_blank">open full PDF</a></div>`;
    return;
  }

  const totalPages = Number(pdf.numPages || (info && info.page_count) || 1);
  let currentPage = (chunkPages.length > 0 && chunkPages[0] >= 1 && chunkPages[0] <= totalPages)
    ? chunkPages[0]
    : 1;

  const shell = document.createElement("div");
  shell.className = "pdf-chunk-viewer";

  const chunkButtonsHtml = chunkPages.map(p =>
    `<button type="button" class="pdf-nav-btn pdf-chunk-btn" data-page="${p}" title="Jump to chunk page ${p}">🎯 Chunk p.${p}</button>`
  ).join(" ");

  shell.innerHTML = `
    <div class="pdf-toolbar">
      <div class="pdf-toolbar-row">
        <span>Source: <b>${esc(info.filename || "document.pdf")}</b></span>
        <a href="/api/runs/${encodeURIComponent(runId)}/pdf" target="_blank" class="pdf-open-link">open full PDF ↗</a>
      </div>
      <div class="pdf-toolbar-row pdf-nav-row">
        <div class="pdf-nav-group">
          <button type="button" class="pdf-nav-btn pdf-prev-btn" title="Previous page">◀ Prev</button>
          <span class="pdf-nav-page-box">
            Page <input type="number" class="pdf-page-input" min="1" max="${totalPages}" value="${currentPage}"> / ${totalPages}
          </span>
          <button type="button" class="pdf-nav-btn pdf-next-btn" title="Next page">Next ▶</button>
        </div>
        ${chunkButtonsHtml ? `<div class="pdf-nav-group">${chunkButtonsHtml}</div>` : ""}
        <span class="pdf-status-badge"></span>
      </div>
    </div>
    <div class="pdf-pages">
      <div class="pdf-page-wrap">
        <div class="pdf-page-label"></div>
        <div class="pdf-canvas-wrap">
          <canvas></canvas>
          <div class="pdf-overlay"></div>
        </div>
      </div>
    </div>
  `;

  host.innerHTML = "";
  host.appendChild(shell);

  const prevBtn = shell.querySelector(".pdf-prev-btn");
  const nextBtn = shell.querySelector(".pdf-next-btn");
  const pageInput = shell.querySelector(".pdf-page-input");
  const statusBadge = shell.querySelector(".pdf-status-badge");
  const pageLabel = shell.querySelector(".pdf-page-label");
  const canvasWrap = shell.querySelector(".pdf-canvas-wrap");
  const canvas = shell.querySelector("canvas");
  const overlay = shell.querySelector(".pdf-overlay");

  let activeRenderTask = null;
  let currentRenderSeq = 0;

  async function renderCurrentPage() {
    const seq = ++currentRenderSeq;
    if (activeRenderTask) {
      try {
        activeRenderTask.cancel();
      } catch (_) {}
      activeRenderTask = null;
    }

    if (prevBtn) prevBtn.disabled = (currentPage <= 1);
    if (nextBtn) nextBtn.disabled = (currentPage >= totalPages);
    if (pageInput) pageInput.value = currentPage;

    const hlRegions = grouped[currentPage] || [];

    shell.querySelectorAll(".pdf-chunk-btn").forEach(btn => {
      const p = Number(btn.dataset.page);
      btn.classList.toggle("active", p === currentPage);
    });

    if (statusBadge) {
      if (hlRegions.length > 0) {
        statusBadge.className = "pdf-status-badge is-highlight";
        statusBadge.textContent = `🎯 Chunk highlight (${hlRegions.length})`;
        statusBadge.title = "Highlight active on this page for the selected chunk";
      } else {
        statusBadge.className = "pdf-status-badge is-plain";
        statusBadge.textContent = "Original PDF";
        statusBadge.title = "Showing original PDF without highlight";
      }
    }

    if (pageLabel) {
      pageLabel.textContent = `Page ${currentPage} of ${totalPages}${hlRegions.length > 0 ? " — Chunk highlight" : ""}`;
    }

    try {
      const page = await pdf.getPage(currentPage);
      if (seq !== currentRenderSeq) return;

      const scale = 1.35;
      const viewport = page.getViewport({ scale });
      const unscaled = page.getViewport({ scale: 1.0 });
      const pageSize = {
        width: unscaled.width * (300 / 72),
        height: unscaled.height * (300 / 72),
      };

      canvasWrap.style.width = `${Math.round(viewport.width)}px`;
      canvasWrap.style.height = `${Math.round(viewport.height)}px`;
      overlay.style.width = `${Math.round(viewport.width)}px`;
      overlay.style.height = `${Math.round(viewport.height)}px`;

      canvas.width = Math.round(viewport.width);
      canvas.height = Math.round(viewport.height);
      const ctx = canvas.getContext("2d");

      overlay.innerHTML = "";

      const renderTask = page.render({ canvasContext: ctx, viewport });
      activeRenderTask = renderTask;
      await renderTask.promise;
      if (seq !== currentRenderSeq) return;

      for (const region of hlRegions) {
        const style = regionToPctStyle(region, pageSize);
        if (!style) continue;
        const box = document.createElement("div");
        box.className = "hl-box";
        Object.assign(box.style, style);
        overlay.appendChild(box);
      }
    } catch (err) {
      if (err && err.name === "RenderingCancelledException") {
        return;
      }
      if (pageLabel) {
        pageLabel.textContent = `Error loading page ${currentPage}: ${err.message || err}`;
      }
    }
  }

  if (prevBtn) {
    prevBtn.onclick = () => {
      if (currentPage > 1) {
        currentPage--;
        renderCurrentPage();
      }
    };
  }

  if (nextBtn) {
    nextBtn.onclick = () => {
      if (currentPage < totalPages) {
        currentPage++;
        renderCurrentPage();
      }
    };
  }

  if (pageInput) {
    pageInput.onchange = () => {
      let p = parseInt(pageInput.value, 10);
      if (!Number.isFinite(p)) p = currentPage;
      p = Math.max(1, Math.min(totalPages, p));
      if (p !== currentPage) {
        currentPage = p;
        renderCurrentPage();
      } else {
        pageInput.value = currentPage;
      }
    };
    pageInput.onkeydown = (e) => {
      if (e.key === "Enter") {
        pageInput.blur();
      }
    };
  }

  shell.querySelectorAll(".pdf-chunk-btn").forEach(btn => {
    btn.onclick = () => {
      const p = Number(btn.dataset.page);
      if (Number.isFinite(p) && p >= 1 && p <= totalPages && p !== currentPage) {
        currentPage = p;
        renderCurrentPage();
      }
    };
  });

  await renderCurrentPage();
}

function renderChunkCard(container, row, hl, runId) {
  const regions = (hl && hl.regions) || row.regions || [];
  const regionRows = regions.length
    ? regions.map(r => {
        const bbox = Array.isArray(r.bbox) ? r.bbox.map(n => Number(n).toFixed(1)).join(", ") : "";
        const norm = Array.isArray(r.bbox_norm)
          ? r.bbox_norm.map(n => Number(n).toFixed(3)).join(", ")
          : "";
        const layout = (hl && hl.layout_paths && hl.layout_paths[String(r.page)]) || "";
        return `<tr>
          <td>${esc(String(r.page))}</td>
          <td><code>${esc(bbox)}</code></td>
          <td>${norm ? `<code>${esc(norm)}</code>` : "—"}</td>
          <td>${esc(String(r.n_elements || (r.element_ids || []).length || ""))}</td>
          <td>${layout ? `<a href="/api/runs/${encodeURIComponent(runId)}/file?path=${encodeURIComponent(layout)}" target="_blank">layout</a>` : "—"}</td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="5" class="empty">No highlight regions (layout missing or no match).</td></tr>`;
  const source = hl && hl.source ? ` · regions=${esc(hl.source)}` : "";
  container.innerHTML = `<details class="tree-node" open style="margin:0">
    <summary style="display:flex;align-items:center;justify-content:space-between">
      <div>
        <span class="title">${esc(row.chunk_id)}</span>
        <span class="tree-badge">${esc((row.pages || [row.page]).join(","))}</span>
        <span class="tree-kv">${esc(row.heading_path || "")}${source}</span>
      </div>
      <button type="button" class="preview-close-btn" title="닫기">✕</button>
    </summary>
    <div class="tree-body">
      <div class="viz-section" style="margin:6px 0">
        <h3 style="margin:0 0 4px">PDF highlight</h3>
        <div class="pdf-chunk-host"></div>
      </div>
      <div class="viz-section" style="margin:6px 0">
        <h3 style="margin:0 0 4px">Highlight regions</h3>
        <table class="kv-table">
          <thead><tr><th>Page</th><th>bbox (px)</th><th>bbox (norm)</th><th>#el</th><th>layout</th></tr></thead>
          <tbody>${regionRows}</tbody>
        </table>
      </div>
      <pre class="pretty">${esc(row.text || "")}</pre>
    </div>
  </details>`;
  const closeBtn = container.querySelector(".preview-close-btn");
  if (closeBtn) {
    closeBtn.onclick = (e) => {
      e.stopPropagation();
      container.remove();
    };
  }
  const pdfHost = container.querySelector(".pdf-chunk-host");
  const fallbackPages = (hl && hl.pages) || row.pages || (row.page ? [row.page] : []);
  if (pdfHost && (regions.length || (fallbackPages && fallbackPages.length))) {
    mountChunkPdfViewer(pdfHost, runId, regions, fallbackPages);
  }
  bindChunkJumpButtons(container);
  bindPageJumpButtons(container);
}

function renderPageCard(container, page, hl, runId, opts = {}) {
  const regions = (hl && hl.regions) || [];
  const chunks = (hl && hl.chunks) || [];
  const chunkCount = hl ? (hl.chunk_count || chunks.length) : 0;
  const keyLabel = opts.key ? ` · ${esc(opts.key)}` : "";
  const source = hl && hl.source ? ` · regions=${esc(hl.source)}` : "";

  let chunkSummaryHtml = "";
  if (chunks.length > 0) {
    chunkSummaryHtml = `
      <div class="viz-section" style="margin:6px 0">
        <h3 style="margin:0 0 4px">Page chunks (${chunks.length})</h3>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px">
          ${chunks.map(c => `
            <button type="button" class="chunk-jump" data-chunk-id="${esc(c.chunk_id)}" style="padding:2px 8px;border-radius:6px;border:1px solid var(--line);background:#152033;color:var(--accent);font-size:11px;cursor:pointer" title="${esc(c.heading_path || c.chunk_id)}">
              🎯 ${esc(c.chunk_id)}${c.heading_path ? ` (${esc(c.heading_path)})` : ""}
            </button>
          `).join("")}
        </div>
      </div>`;
  }

  const regionRows = regions.length
    ? regions.map(r => {
        const bbox = Array.isArray(r.bbox) ? r.bbox.map(n => Number(n).toFixed(1)).join(", ") : "";
        const norm = Array.isArray(r.bbox_norm) ? r.bbox_norm.map(n => Number(n).toFixed(3)).join(", ") : "";
        const cid = r.chunk_id ? `<code>${esc(r.chunk_id)}</code>` : "—";
        return `<tr>
          <td>${esc(String(r.page))}</td>
          <td>${cid}</td>
          <td><code>${esc(bbox)}</code></td>
          <td>${norm ? `<code>${esc(norm)}</code>` : "—"}</td>
          <td>${esc(String(r.n_elements || (r.element_ids || []).length || ""))}</td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="5" class="empty">No highlight regions on page ${page}.</td></tr>`;

  container.innerHTML = `<details class="tree-node" open style="margin:0">
    <summary style="display:flex;align-items:center;justify-content:space-between">
      <div>
        <span class="title">Page ${page}${keyLabel}</span>
        <span class="tree-badge ok">PDF Page ${page}</span>
        ${chunkCount ? `<span class="tree-badge">${chunkCount} chunk(s)</span>` : ""}
        <span class="tree-kv">${source}</span>
      </div>
      <button type="button" class="preview-close-btn" title="닫기">✕</button>
    </summary>
    <div class="tree-body">
      <div class="viz-section" style="margin:6px 0">
        <h3 style="margin:0 0 4px">PDF highlight</h3>
        <div class="pdf-chunk-host"></div>
      </div>
      ${chunkSummaryHtml}
      ${regions.length > 0 ? `
      <div class="viz-section" style="margin:6px 0">
        <h3 style="margin:0 0 4px">Highlight regions</h3>
        <table class="kv-table">
          <thead><tr><th>Page</th><th>Chunk</th><th>bbox (px)</th><th>bbox (norm)</th><th>#el</th></tr></thead>
          <tbody>${regionRows}</tbody>
        </table>
      </div>` : ""}
    </div>
  </details>`;

  const closeBtn = container.querySelector(".preview-close-btn");
  if (closeBtn) {
    closeBtn.onclick = (e) => {
      e.stopPropagation();
      container.remove();
    };
  }

  const pdfHost = container.querySelector(".pdf-chunk-host");
  if (pdfHost) {
    mountChunkPdfViewer(pdfHost, runId, regions, [page]);
  }

  bindChunkJumpButtons(container);
  bindPageJumpButtons(container);
}

function extractPageNumbersFromText(text) {
  if (!text || typeof text !== "string") return [];
  const found = new Set();
  
  // 1) 한국어 슬래시: 36/37페이지, 36/37/38페이지
  const pSlash = /(\d+)(?:\s*\/\s*(\d+))+\s*페이지/g;
  let m;
  while ((m = pSlash.exec(text)) !== null) {
    const nums = m[0].match(/\d+/g) || [];
    for (const n of nums) {
      const p = parseInt(n, 10);
      if (p > 0 && p < 2000) found.add(p);
    }
  }

  // 2) 한국어 쉼표: (10, 56페이지) 또는 10, 56페이지
  const pComma = /(\d+)(?:\s*,\s*\d+)+\s*페이지/g;
  while ((m = pComma.exec(text)) !== null) {
    const nums = m[0].match(/\d+/g) || [];
    for (const n of nums) {
      const p = parseInt(n, 10);
      if (p > 0 && p < 2000) found.add(p);
    }
  }

  // 3) 한국어 단순: 59페이지, 24 페이지
  const pSingle = /(\d+)\s*페이지/g;
  while ((m = pSingle.exec(text)) !== null) {
    const p = parseInt(m[1], 10);
    if (p > 0 && p < 2000) found.add(p);
  }

  // 4) 영어: pages 11, 59 또는 page 24 또는 p. 59
  const pEng = /\b(?:pages?|p\.)\s*(\d+(?:\s*(?:,|and|&)\s*\d+)*)/gi;
  while ((m = pEng.exec(text)) !== null) {
    const nums = m[1].match(/\d+/g) || [];
    for (const n of nums) {
      const p = parseInt(n, 10);
      if (p > 0 && p < 2000) found.add(p);
    }
  }

  return Array.from(found).sort((a, b) => a - b);
}

function formatAgenticDetailText(detailText, key="") {
  if (!detailText) return "";
  let safe = esc(String(detailText));

  // 한국어 슬래시: 36/37페이지
  safe = safe.replace(/(\d+)\s*\/\s*(\d+)\s*페이지/g, (match, p1, p2) => {
    return `<button type="button" class="pdf-page-inline-btn" data-page="${p1}" data-key="${esc(key)}" title="${p1}페이지 원문 PDF 보기">${p1}</button>/<button type="button" class="pdf-page-inline-btn" data-page="${p2}" data-key="${esc(key)}" title="${p2}페이지 원문 PDF 보기">${p2}페이지 📄</button>`;
  });

  // 한국어 "N페이지"
  safe = safe.replace(/(\d+)\s*페이지/g, (match, p) => {
    return `<button type="button" class="pdf-page-inline-btn" data-page="${p}" data-key="${esc(key)}" title="${p}페이지 원문 PDF 보기">${match} 📄</button>`;
  });

  // 영어 "pages 11, 59" 또는 "page 24"
  safe = safe.replace(/\b(pages?|p\.)\s*([0-9]+(?:\s*(?:,|&amp;|&)\s*[0-9]+)*)/gi, (fullMatch, prefix, numsStr) => {
    const parts = numsStr.replace(/&amp;/g, '&').split(/([,\s&]+)/);
    const linked = parts.map(part => {
      const num = parseInt(part.trim(), 10);
      if (Number.isFinite(num) && num > 0 && num < 2000) {
        return `<button type="button" class="pdf-page-inline-btn" data-page="${num}" data-key="${esc(key)}" title="${num}페이지 원문 PDF 보기">${num} 📄</button>`;
      }
      return part;
    }).join("");
    return `${prefix} ${linked}`;
  });

  return safe;
}

async function openChunkPreview(chunkId, anchorEl) {
  const id = String(chunkId || "").trim();
  if (!id || !state.runId) return;
  const detail = document.getElementById("detail");
  if (!detail) return;

  const host = anchorEl && anchorEl.closest
    ? (anchorEl.closest(".agentic-eval-detail, .agentic-eval-text, .ev-block, .tree-body, td, .viz-section") || anchorEl.parentElement)
    : detail;

  // Toggle closed if same preview already open
  const existing = host.querySelector(".chunk-preview-inline");
  if (existing && existing.dataset.previewChunk === id) {
    existing.remove();
    return;
  }

  detail.querySelectorAll(".chunk-preview-inline").forEach(r => r.remove());

  const box = document.createElement("div");
  box.className = "chunk-preview-inline";
  box.dataset.previewChunk = id;
  box.innerHTML = `<div class="empty">Loading ${esc(id)}…</div>`;
  if (host && host.appendChild) host.appendChild(box);
  else detail.appendChild(box);
  try {
    const [row, hl] = await Promise.all([
      api(`/api/runs/${encodeURIComponent(state.runId)}/chunks/${encodeURIComponent(id)}`),
      api(`/api/runs/${encodeURIComponent(state.runId)}/chunks/${encodeURIComponent(id)}/highlights`).catch(() => null),
    ]);
    renderChunkCard(box, row, hl, state.runId);
  } catch (err) {
    box.innerHTML = `<div class="empty" style="color:var(--err)">${esc(err.message || err)}</div>`;
  }
}

async function openPagePreview(pageNo, anchorEl, opts = {}) {
  const page = parseInt(pageNo, 10);
  if (!Number.isFinite(page) || page <= 0 || !state.runId) return;

  const detail = document.getElementById("detail");
  if (!detail) return;

  const host = anchorEl && anchorEl.closest
    ? (anchorEl.closest(".agentic-eval-detail, .agentic-eval-text, .ev-block, .tree-body, td, .viz-section") || anchorEl.parentElement)
    : detail;

  // Toggle closed if same preview already open
  const existing = host.querySelector(".chunk-preview-inline");
  if (existing && existing.dataset.previewPage === String(page)) {
    existing.remove();
    return;
  }

  detail.querySelectorAll(".chunk-preview-inline").forEach(r => r.remove());

  const box = document.createElement("div");
  box.className = "chunk-preview-inline";
  box.dataset.previewPage = String(page);
  box.innerHTML = `<div class="empty">Loading Page ${page}…</div>`;

  if (host && host.appendChild) host.appendChild(box);
  else detail.appendChild(box);

  try {
    const qParam = opts.quote ? `?q=${encodeURIComponent(opts.quote)}` : "";
    const hl = await api(`/api/runs/${encodeURIComponent(state.runId)}/pages/${page}/highlights${qParam}`).catch(() => null);
    renderPageCard(box, page, hl, state.runId, opts);
  } catch (err) {
    box.innerHTML = `<div class="empty" style="color:var(--err)">${esc(err.message || err)}</div>`;
  }
}

function bindChunkJumpButtons(root) {
  (root || document).querySelectorAll(".chunk-jump").forEach(btn => {
    btn.onclick = () => openChunkPreview(btn.dataset.chunkId, btn);
  });
}

function bindPageJumpButtons(root) {
  (root || document).querySelectorAll(".pdf-page-btn, .pdf-page-inline-btn").forEach(btn => {
    btn.onclick = (e) => {
      e.stopPropagation();
      const page = btn.dataset.page;
      const key = btn.dataset.key || "";
      const quote = btn.dataset.quote || "";
      openPagePreview(page, btn, { key, quote });
    };
  });
}

function renderSearchOutput(output) {
  if (!output) return "";
  const pages = output.pages || [];
  const reasons = output.page_reasons || {};
  const chunkIds = output.page_chunk_id || {};
  return `<div class="viz-section" style="margin:8px 0">
    <h3 style="margin:0 0 6px">SearchAgent output</h3>
    <div class="tree-kv">status=${esc(output.status || "?")} · pages=${pages.length ? esc(pages.join(", ")) : "∅"}</div>
    ${output.reason ? `<div class="tree-kv">reason=${esc(output.reason)}</div>` : ""}
    ${renderPageReasonsTable(reasons, chunkIds) || (pages.length ? `<pre class="pretty">${esc(pretty({pages}, 1200))}</pre>` : `<div class="tree-kv">No pages returned.</div>`)}
  </div>`;
}

function renderKeyResultRow(kr) {
  const pages = kr.pages || [];
  const status = kr.status || "?";
  const statusCls = status === "complete" ? "ok"
    : (status === "not_found" || String(status).startsWith("handoff") ? "warn" : "");
  const reasons = kr.page_reasons || kr.reasons || {};
  const chunkIds = kr.page_chunk_id || {};
  return `<details class="tree-node key-result">
    <summary>
      <span class="tree-kv">${esc(kr.key)}</span>
      <span class="tree-badge ${statusCls}">${esc(status)}</span>
      ${pages.length ? `<span class="tree-badge ok">pages=${esc(pages.join(","))}</span>` : `<span class="tree-badge warn">pages=∅</span>`}
    </summary>
    <div class="tree-body">
      ${kr.reason ? `<div class="tree-kv">reason=${esc(kr.reason)}</div>` : ""}
      ${renderPageReasonsTable(reasons, chunkIds) || (pages.length ? `<pre class="pretty">${esc(pretty({pages}, 800))}</pre>` : "")}
      ${kr.filename ? `<div class="tree-kv"><a href="#" data-tool-file="${esc(kr.filename)}">open per-key dump</a></div>` : ""}
    </div>
  </details>`;
}

function renderKeyResultsSection(keyResults) {
  if (!keyResults || !keyResults.length) return "";
  const nResolved = keyResults.filter(kr =>
    ["complete", "not_found", "handoff", "handoff_no_candidates"].includes(String(kr.status || ""))
  ).length;
  return `<div class="viz-section" style="margin:8px 0">
    <h3 style="margin:0 0 6px">Key results (${esc(nResolved)}/${esc(keyResults.length)} resolved)</h3>
    <p class="hint" style="margin:0 0 8px">
      Per-key pages from <code>submit_pages</code> / <code>no_relevant_pages</code>.
      Turns above are shared across all keys in this batch.
    </p>
    ${keyResults.map(renderKeyResultRow).join("")}
  </div>`;
}

function renderSearchAgent(node) {
  const res = node.result || {};
  const output = node.output || {
    pages: res.pages || [],
    page_reasons: res.page_reasons || res.reasons || {},
    page_chunk_id: res.page_chunk_id || {},
    status: res.status,
  };
  const shared = !!(node.shared || (node.batch && (node.key_results || []).length > 1));
  const keyResults = node.key_results || [];
  const sessions = node.sessions || [];
  const nTurns = res.n_search_steps
    ?? sessions.reduce((n, s) => n + ((s.turns || []).length), 0);
  const nRuntimeSessions = res.n_search_sessions || sessions.length || 0;
  const pages = output.pages || res.pages || [];
  const status = output.status || res.status || (pages.length ? "complete" : "unknown");
  const statusCls = status === "complete" ? "ok"
    : (status === "not_found" || String(status).startsWith("handoff") ? "warn" : "");
  const nResolved = output.n_resolved ?? res.n_resolved;
  const nKeys = output.n_keys ?? res.n_keys ?? keyResults.length;
  const summaryBadge = shared
    ? (nResolved != null && nKeys
      ? `<span class="tree-badge ok">${esc(nResolved)}/${esc(nKeys)} keys</span>`
      : `<span class="tree-badge">${esc(nKeys)} keys</span>`)
    : (pages.length
      ? `<span class="tree-badge ok">pages=${esc(pages.join(","))}</span>`
      : `<span class="tree-badge warn">pages=∅</span>`);
  const hasOutline = !!(node.initial_state && node.initial_state.document_outline);
  const outlineBadge = hasOutline ? `<span class="tree-badge">TOC outline</span>` : "";
  const sharedHint = shared
    ? `<p class="hint" style="margin:4px 0 8px">
        One shared SearchAgent ReAct loop for ${esc(nKeys)} keys
        (${esc(nRuntimeSessions || 1)} runtime session(s), ${esc(nTurns)} turn(s)).
        <code>submit_pages</code> / <code>no_relevant_pages</code> output per key is below.
      </p>`
    : `<p class="hint" style="margin:4px 0 8px">
        Turn-by-turn tool calls below. Final <code>submit_pages</code> /
        <code>no_relevant_pages</code> output is shown on the last turn.
      </p>`;
  return `<details class="tree-node search">
    <summary>
      <span class="title">SearchAgent</span>
      <span class="tree-badge ${statusCls}">${esc(status)}</span>
      <span class="tree-kv">${esc(node.key)}</span>
      ${timingBadge(node.timing, "warn")}
      ${summaryBadge}
      ${outlineBadge}
    </summary>
    <div class="tree-body">
      ${shared ? "" : renderSearchOutput(output)}
      <div class="tree-kv">
        ${nRuntimeSessions ? `runtime sessions=${esc(nRuntimeSessions)} · turns=${esc(nTurns)}` : ""}
        ${shared && nKeys ? ` · keys=${esc(nKeys)}` : ""}
      </div>
      ${node.note ? `<div class="tree-kv">${esc(node.note)}</div>` : ""}
      ${sharedHint}
      ${sessions.length ? sessions.map(s => renderSearchSession(s, {
        nSessions: sessions.length,
        shared,
      })).join("") :
        `${renderSearchInitialState(node.initial_state, node.prompts, { shared })}<div class="tree-kv">Search turn dumps not linked (legacy run).</div>`}
      ${shared ? renderKeyResultsSection(keyResults) : ""}
    </div>
  </details>`;
}

function renderLoadKvSchema(tool) {
  const args = tool.arguments || {};
  const result = tool.result || {};
  const items = Array.isArray(result.items) ? result.items : [];
  const keyFilter = args.key ? `key=${esc(args.key)}` : "all keys";
  const rows = items.map(it =>
    `<tr><td>${esc(it.key)}</td><td>${esc(it.description || "")}</td></tr>`
  ).join("");
  return `
    <div class="tree-kv">${keyFilter} · count=${esc(result.count != null ? result.count : items.length)}</div>
    ${rows ? `<table class="kv-table" style="margin-top:8px">
      <thead><tr><th>Key</th><th>Description</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>` : `<pre class="pretty" style="margin-top:6px">${esc(pretty(Object.keys(result).length ? result : (tool.result_preview || args), 4000))}</pre>`}
    ${tool.filename ? `<div class="tree-kv"><a href="#" data-tool-file="${esc(tool.filename)}">open tool dump</a></div>` : ""}
  `;
}

function renderExtractKvVlm(tool) {
  const args = tool.arguments || {};
  const result = tool.result || {};
  const pages = args.pages || result.pages || [];
  const keys = args.keys || result.keys || [];
  const hints = args.hints;
  const pageReasons = args.page_reasons || result.page_reasons || {};
  const pageChunkIds = args.page_chunk_id || result.page_chunk_id || {};
  const parsed = result.result || {};
  const extractions = Array.isArray(parsed.extractions) ? parsed.extractions : [];
  const covered = result.all_keys_covered;
  const rows = extractions.map(ex =>
    `<tr>
      <td>${esc(ex.key)}</td>
      <td>${esc(ex.value)}</td>
      <td>${esc(ex.value_reason || ex.evidence_quote || "")}</td>
    </tr>`
  ).join("");
  const files = tool.extra_files || {};
  const fileLinks = Object.entries(files).map(([label, rel]) =>
    `<a href="${runFileUrl(rel)}" target="_blank">${esc(label)}</a>`
  ).join(" · ");
  const hintsHtml = hints && String(hints).trim() ? `
    <div class="viz-section" style="margin-top:6px">
      <h3 style="margin:0 0 4px">Hints</h3>
      <pre class="pretty" style="max-height:160px;margin:0">${esc(String(hints))}</pre>
    </div>` : "";
  const nReasons = Object.keys(pageReasons || {}).length;
  return `
    <div class="tree-kv">
      pages=${esc(JSON.stringify(pages))} · keys=${esc(JSON.stringify(keys))}
      ${nReasons ? ` · page_reasons=${esc(nReasons)}` : ""}
      ${Object.keys(pageChunkIds || {}).length ? ` · page_chunk_id=${esc(Object.keys(pageChunkIds).length)}` : ""}
      ${result.input_tokens != null || result.output_tokens != null
        ? ` · VLM in=${esc(result.input_tokens ?? "—")} out=${esc(result.output_tokens ?? "—")}` : ""}
      ${covered === true ? `<span class="tree-badge ok">all_keys_covered</span>` : ""}
      ${covered === false ? `<span class="tree-badge warn">partial</span>` : ""}
    </div>
    ${renderPageReasonsTable(pageReasons, pageChunkIds)}
    ${hintsHtml}
    ${rows ? `<table class="kv-table" style="margin-top:8px">
      <thead><tr><th>Key</th><th>Value</th><th>Reason / Evidence</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>` : ""}
    ${!rows ? `<pre class="pretty" style="margin-top:6px">${esc(pretty(result.result || result || tool.result_preview || {}, 4000))}</pre>` : ""}
    ${fileLinks ? `<div class="tree-kv" style="margin-top:6px">pages: ${fileLinks}</div>` : ""}
    ${tool.filename ? `<div class="tree-kv"><a href="#" data-tool-file="${esc(tool.filename)}">open tool dump</a></div>` : ""}
  `;
}

function renderSubmitEvaluation(tool) {
  const args = tool.arguments || {};
  const result = tool.result || {};
  const payload = result.result || result || args;
  const verdict = String(payload.is_correct_answer || payload.verdict || "").toLowerCase();
  const goldVerdict = String(payload.is_valid_gold || "").toLowerCase();
  const cls = verdict === "correct" ? "ok" : (verdict === "incorrect" ? "warn" : "");
  const goldCls = goldVerdict === "valid" ? "ok" : (goldVerdict === "invalid" ? "warn" : "");
  return `
    <div class="tree-kv">
      key=${esc(payload.key || args.key || state.evalKey || "?")}
      ${verdict ? `<span class="tree-badge ${cls}">pred: ${esc(verdict)}</span>` : ""}
      ${goldVerdict ? `<span class="tree-badge ${goldCls}">GT: ${esc(goldVerdict)}</span>` : ""}
    </div>
    ${payload.reason_summary ? `<div class="tree-kv">${esc(payload.reason_summary)}</div>` : ""}
    ${payload.reason_detail || payload.text ? `<pre class="pretty" style="max-height:200px;margin-top:6px">${esc(payload.reason_detail || payload.text)}</pre>` : ""}
    ${tool.filename ? `<div class="tree-kv"><a href="#" data-tool-file="${esc(tool.filename)}">open tool dump</a></div>` : ""}
  `;
}

function renderPageImageChatVlm(tool) {
  const args = tool.arguments || {};
  const result = tool.result || {};
  const parsed = result.result || result.parsed || {};
  return `
    <div class="tree-kv">
      page=${esc(args.page ?? "?")}
      ${result.input_tokens != null ? ` · VLM in=${esc(result.input_tokens)} out=${esc(result.output_tokens ?? "—")}` : ""}
    </div>
    ${parsed.answer || parsed.summary ? `<pre class="pretty" style="max-height:200px;margin-top:6px">${esc(parsed.answer || parsed.summary)}</pre>` : ""}
    ${tool.filename ? `<div class="tree-kv"><a href="#" data-tool-file="${esc(tool.filename)}">open tool dump</a></div>` : ""}
  `;
}

function renderGenericToolResult(tool) {
  const args = tool.arguments || {};
  const result = tool.result != null ? tool.result : tool.result_preview;
  const tokenHint = toolMessageTokenHint(tool);
  let html = "";
  if (args && Object.keys(args).length) {
    html += `<div class="viz-section" style="margin-top:6px">
      <h3 style="margin:0 0 4px">Arguments</h3>
      <pre class="pretty" style="max-height:120px">${esc(pretty(args, 1200))}</pre>
    </div>`;
  }
  if (result != null && result !== "") {
    html += `<div class="viz-section" style="margin-top:6px">
      <h3 style="margin:0 0 4px">Result</h3>
      <pre class="pretty" style="max-height:280px">${esc(pretty(result, 4000))}</pre>
    </div>`;
  }
  if (!html) {
    html = `<div class="tree-kv">No arguments or result recorded.</div>`;
  }
  if (tokenHint) {
    html += `<div class="tree-kv" style="margin-top:4px">${esc(tokenHint.replace(/^ · /, ""))}</div>`;
  }
  return html;
}

function formatSearchKeys(tool) {
  const args = tool.arguments || {};
  const result = tool.result || {};
  const so = tool.search_output || {};
  let keys = args.key ?? args.keys ?? result.key ?? result.accepted ?? so.accepted;
  if (keys == null || keys === "") return "?";
  if (Array.isArray(keys)) {
    const clean = keys.map(k => (typeof k === "object" && k ? (k.key || JSON.stringify(k)) : String(k))).filter(Boolean);
    if (!clean.length) return "∅";
    if (clean.length === 1) return clean[0];
    return `${clean.length} keys · ${clean.join(" · ")}`;
  }
  return String(keys);
}

function failurePhaseLabel(phase) {
  const labels = {
    llm_request: "Master LLM API call failed (no assistant response received)",
    tool_execution: "Tool execution failed after assistant tool calls",
    response_processing: "Failed while processing assistant response / final JSON",
    unknown: "Turn failed",
  };
  return labels[phase] || labels.unknown;
}

function shortenErrorMessage(err) {
  if (!err) return "";
  const text = String(err);
  const msgMatch = text.match(/'message':\s*"([^"]+)"/);
  if (msgMatch) return msgMatch[1];
  const altMatch = text.match(/"message":\s*"([^"]+)"/);
  if (altMatch) return altMatch[1];
  return text.length > 500 ? text.slice(0, 500) + "…" : text;
}

function renderRequestTail(tail) {
  if (!tail || !tail.length) return "";
  const rows = tail.map((m, i) => {
    const tools = (m.tool_calls || []).filter(Boolean);
    const toolBit = tools.length ? `<div class="tree-kv">tools: ${esc(tools.join(", "))}</div>` : "";
    const preview = m.content_preview
      ? `<pre class="pretty" style="max-height:120px;margin:4px 0 0">${esc(m.content_preview)}</pre>`
      : `<div class="tree-kv" style="color:var(--muted)">(no text content)</div>`;
    return `<div class="flow-item ${esc(m.role)}" style="margin-top:6px">
      <div class="label">${esc(m.role)}${tail.length > 1 ? ` · tail ${i + 1}/${tail.length}` : ""}</div>
      ${toolBit}
      ${preview}
    </div>`;
  }).join("");
  return `<div class="viz-section" style="margin-top:8px">
    <h3 style="margin:0 0 4px">Request tail (messages sent to LLM)</h3>
    ${rows}
  </div>`;
}

function renderMasterTurnFailure(mt) {
  if (!mt.error && !(mt.request_summary || {}).n_messages) return "";
  const phase = mt.failure_phase || (mt.error ? "unknown" : "");
  const req = mt.request_summary || {};
  const roles = req.roles || {};
  const roleBits = Object.entries(roles).map(([r, n]) => `${r}=${n}`).join(" · ");
  const stats = [
    req.n_messages != null ? `${req.n_messages} messages` : null,
    roleBits || null,
    mt.prompt_est_tokens != null ? `prompt≈${mt.prompt_est_tokens}` : null,
    mt.budget_est_total != null ? `budget≈${mt.budget_est_total}` : null,
    mt.max_tokens != null ? `max_tokens=${mt.max_tokens}` : null,
    mt.n_tools != null ? `${mt.n_tools} tools` : null,
    mt.tool_choice ? `tool_choice=${mt.tool_choice}` : null,
  ].filter(Boolean).join(" · ");
  const errShort = shortenErrorMessage(mt.error);
  const assistant = (mt.assistant_content || "").trim();
  return `<div class="viz-section master-failure" style="margin:4px 0 8px">
    ${phase ? `<div class="tree-kv" style="color:var(--err);margin-bottom:6px"><b>${esc(failurePhaseLabel(phase))}</b></div>` : ""}
    ${errShort ? `<pre class="pretty" style="max-height:160px;border-color:#7a3a3f">${esc(errShort)}</pre>` : ""}
    ${stats ? `<div class="tree-kv" style="margin-top:8px">Request: ${esc(stats)}</div>` : ""}
    ${assistant ? `<div class="viz-section" style="margin-top:8px">
      <h3 style="margin:0 0 4px">Partial assistant output</h3>
      <pre class="pretty" style="max-height:160px">${esc(assistant)}</pre>
    </div>` : ""}
    ${renderRequestTail(req.tail)}
    ${mt.filename ? `<div class="tree-kv" style="margin-top:8px"><a href="#" data-step="${esc(mt.filename)}">open step JSON</a></div>` : ""}
  </div>`;
}

function renderMasterTurnBody(mt) {
  const tools = (mt.tools || []).map(renderMasterTool).join("");
  if (tools) return tools;
  const failure = renderMasterTurnFailure(mt);
  if (failure) return failure;
  return `<div class="empty">No tools on this turn</div>`;
}

function renderMasterTool(tool) {
  let inner = `<div class="name">${esc(tool.name)}</div>`;
  if (tool.name === "search_pages" || tool.name === "search_pages_start") {
    const so = tool.search_output || {};
    const status = so.status ? ` · ${esc(so.status)}` : "";
    inner += `<div class="tree-kv">key=${esc(formatSearchKeys(tool))}${status}</div>`;
    if (Array.isArray(so.accepted) && so.accepted.length) {
      inner += `<div class="tree-kv" style="margin-top:4px">accepted: ${esc(so.accepted.join(" · "))}</div>`;
    }
    inner += (tool.children || []).map(renderSearchAgent).join("");
  } else if (tool.name === "collect_search_results" || tool.name === "await_searches") {
    const so = tool.search_output || {};
    const n = so.n_keys != null ? so.n_keys : ((so.results || so.completed || []).length || null);
    inner += `<div class="tree-kv">policy=${esc((tool.arguments || {}).policy || "?")}${n != null ? ` · n_keys=${esc(n)}` : ""}${toolMessageTokenHint(tool)}</div>`;
    inner += (tool.children || []).map(renderSearchAgent).join("");
  } else if (tool.name === "extract_kv_vlm") {
    inner += renderExtractKvVlm(tool);
  } else if (tool.name === "load_kv_schema") {
    inner += renderLoadKvSchema(tool);
  } else if (tool.name === "submit_evaluation") {
    inner += renderSubmitEvaluation(tool);
  } else if (tool.name === "page_image_chat_vlm") {
    inner += renderPageImageChatVlm(tool);
  } else {
    inner += renderGenericToolResult(tool);
  }
  return `<div class="tree-tool">${inner}</div>`;
}

function renderTiming() {
  if (state.agentTreeLoading) {
    return `<div class="empty">Timing 데이터 로딩 중…</div>`;
  }
  const timing = state.agentTree?.timing;
  if (!timing) return `<div class="empty">No timing data for this run.</div>`;

  const total = timing.total_seconds || 0;
  const summary = timing.summary || {};
  const active = timing.active_searches || (timing.search_calls || []).filter(sc => sc.status === "running");
  const pipe = timing.pipeline_progress;
  const phaseLabel = (sc) => {
    if (sc.phase === "llm") return "waiting on LLM";
    if (sc.phase === "tools") return "running tools";
    if (sc.phase === "starting") return "starting";
    return "running";
  };
  const pipeBanner = pipe ? `
    <div class="timing-live">
      <div class="live-title">● ${esc(pipe.label || pipe.stage)}</div>
      <div class="live-row">
        stage=${esc(pipe.stage)} · elapsed ${fmtSec(pipe.seconds)}
        ${pipe.page != null ? ` · page ${esc(pipe.page)}${pipe.total_pages != null ? "/" + esc(pipe.total_pages) : ""}` : ""}
        ${pipe.strategy ? ` · strategy=${esc(pipe.strategy)}` : ""}
      </div>
    </div>` : "";
  const liveBanner = active.length ? `
    <div class="timing-live">
      <div class="live-title">● Now running · ${esc(active.length)} SearchAgent job(s)</div>
      ${active.map(sc => `
        <div class="live-row">
          <span class="sess">session ${esc(sc.current_session ?? "?")} · turn ${esc(sc.current_turn || "?")}</span>
          · ${esc(phaseLabel(sc))} · master step ${esc(sc.master_step || "?")}
          · ${esc(sc.label || sc.key)}
          ${sc.n_keys > 1 ? ` · ${esc(sc.n_keys)} keys` : ""}
        </div>`).join("")}
    </div>` : "";

  let html = `<div class="timing-panel">
    <p class="hint">
      Wall time from <code>timeline.jsonl</code>.
      Master turn wall includes nested SearchAgent work started on that step
      (async search is not just request→next-request).
      SearchAgent calls are collapsed by default — expand a session to see
      key outcomes and turns. Multi-key batches share one wall/model clock.
    </p>
    ${pipeBanner}
    ${liveBanner}
    <div class="tree-kv">
      total ${fmtSec(total)} · master model ${fmtSec(summary.master_llm_seconds)}
      · search model ${fmtSec(summary.search_llm_seconds)}
      · ${esc(summary.search_page_calls || 0)} search_pages
      ${summary.active_searches ? ` · <span style="color:#e0a45c">${esc(summary.active_searches)} active</span>` : ""}
    </div>
    <h3 style="margin:8px 0 6px">Pipeline stages</h3>
    ${(timing.stages || []).map(s => {
      const label = s.status === "running" ? `${s.stage} (running)` : s.stage;
      return renderTimingBar(label, s.seconds, s.pct, s.stage);
    }).join("") || `<div class="empty">No stage events yet</div>`}
    <h3 style="margin:16px 0 6px">Master turns</h3>
    ${(timing.master_turns || []).map(mt => {
      const searchBit = mt.n_search_calls
        ? ` · search wall ${fmtSec(mt.search_wall_seconds)} (max of ${esc(mt.n_search_calls)}) · search model ${fmtSec(mt.search_llm_seconds)}`
        : "";
      return `<div>
        ${renderTimingBar(`Turn ${mt.step}`, mt.wall_seconds, mt.pct, "master")}
        <div class="timing-sub tree-kv">model ${fmtSec(mt.llm_seconds)} · tools/overhead ${fmtSec(mt.tool_seconds)}${searchBit}${tokenLabel(mt) ? ` · ${esc(tokenLabel(mt))}` : ""}</div>
      </div>`;
    }).join("") || `<div class="empty">No master turns</div>`}
    <h3 style="margin:16px 0 6px">SearchAgent calls</h3>
    <div class="timing-search-list" style="display:flex;flex-direction:column;gap:8px">
      ${(timing.search_calls || []).map(sc => {
        const running = sc.status === "running";
        const keyOutcomes = sc.keys || (sc.key ? [{key: sc.key, status: sc.status}] : []);
        const shared = sc.shared || keyOutcomes.length > 1;
        const title = running
          ? `<span class="tree-badge warn">running</span> session ${esc(sc.current_session ?? "?")} · turn ${esc(sc.current_turn || "?")} · ${esc(sc.label || sc.key)}`
          : esc(sc.label || sc.key || "?");
        const keyRows = shared ? keyOutcomes.map(k => `
          <tr>
            <td style="padding-left:12px">${esc(k.key)}</td>
            <td><span class="tree-badge ${k.status === "complete" ? "ok" : (k.status === "pending" || String(k.status||"").startsWith("handoff") ? "warn" : "")}">${esc(k.status || "?")}</span></td>
            <td>${k.n_pages != null ? esc(k.n_pages) : "—"}</td>
          </tr>`).join("") : "";
        const turnRows = (sc.sessions || []).map(sess => (sess.turns || []).map(t => `
          <tr class="${t.status === "running" ? "running" : ""}">
            <td style="padding-left:12px">session ${esc(sess.session_index)} · turn ${esc(t.search_turn || t.step)}${t.status === "running" ? ` <span class="tree-badge warn">now</span>` : ""}</td>
            <td>${fmtSec(t.llm_seconds)}</td>
            <td>${esc(tokenLabel(t))}</td>
          </tr>`).join("")).join("");
        return `<details class="tree-node search" style="margin:0">
          <summary>
            <span class="title">m${esc(sc.master_step)}</span>
            ${shared ? `<span class="tree-badge">shared</span>` : ""}
            <span class="tree-kv" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis">${title}</span>
            <span class="tree-badge ${running ? "warn" : ""}">${fmtSec(sc.wall_seconds)}</span>
            <span class="tree-badge">model ${fmtSec(sc.llm_seconds)}</span>
            <span class="tree-badge">${esc(sc.n_turns || 0)} turns</span>
            ${running ? `<span class="tree-badge warn">${esc(phaseLabel(sc))}</span>` : ""}
          </summary>
          <div class="tree-body">
            <div class="tree-kv">overhead ${running ? "—" : fmtSec(sc.overhead_seconds)}${shared ? ` · ${esc(keyOutcomes.length)} keys in one SearchAgent session` : ` · key=${esc(sc.key)}`}</div>
            ${keyRows ? `<table class="timing-table" style="margin-top:8px">
              <thead><tr><th>Key</th><th>Status</th><th>Pages</th></tr></thead>
              <tbody>${keyRows}</tbody>
            </table>` : ""}
            ${turnRows ? `<table class="timing-table" style="margin-top:8px">
              <thead><tr><th>Turn</th><th>Model</th><th>Tokens</th></tr></thead>
              <tbody>${turnRows}</tbody>
            </table>` : `<div class="empty">No turns</div>`}
          </div>
        </details>`;
      }).join("") || `<div class="empty">No SearchAgent calls</div>`}
    </div>
  </div>`;
  return html;
}

function renderPagesChunks() {
  if (state.agentTreeLoading) {
    return `<div class="empty">Pages / chunks 로딩 중…</div>`;
  }
  const sub = state.pagesSubtab || "pages";
  const subTabs = `
    <div class="tabs" style="margin-bottom:10px">
      <button class="tab ${sub==="pages"?"active":""}" data-pages-sub="pages">Pages</button>
      <button class="tab ${sub==="chunks"?"active":""}" data-pages-sub="chunks">Chunks</button>
    </div>`;
  if (sub === "chunks") {
    const ch = state.chunks || {};
    const rows = ch.chunks || [];
    const prog = ch.progress;
    const progLine = prog && prog.status === "running"
      ? `<div class="timing-live" style="margin-bottom:10px"><div class="live-title">● Chunking (${esc(prog.strategy || "…")})</div></div>`
      : "";
    return `${subTabs}
      ${progLine}
      <p class="hint">strategy=${esc(ch.strategy || "?")} · ${esc(ch.chunk_count || 0)} chunks
        · ${esc(ch.total_est_tokens || "?")} est tokens
        · showing ${esc(rows.length)} / filtered ${esc(ch.filtered_count ?? rows.length)}</p>
      <div style="margin-bottom:10px;display:flex;gap:8px;align-items:center">
        <input id="chunkSearch" type="search" placeholder="filter chunk id / heading / page"
          style="flex:1;min-width:180px;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:#0f1419;color:var(--text);font-family:var(--mono);font-size:12px"
          value="${esc(state.chunkQuery || "")}" />
        <button type="button" id="chunkSearchBtn" style="padding:4px 12px;border-radius:999px;border:1px solid var(--line);background:#152033;color:var(--text);font-size:12px;cursor:pointer">Filter</button>
      </div>
      <table class="timing-table">
        <thead><tr><th>chunk_id</th><th>pages</th><th>heading</th><th>chars</th><th>tokens</th><th></th></tr></thead>
        <tbody>
          ${rows.map(c => `
            <tr class="chunk-row" data-chunk-id="${esc(c.chunk_id)}">
              <td>${esc(c.chunk_id)}</td>
              <td>${esc((c.pages || [c.page]).join(", "))}${c.page_end && c.page_end !== c.page ? ` → ${esc(c.page_end)}` : ""}</td>
              <td style="max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(c.heading_path || "")}</td>
              <td>${esc(c.chars)}</td>
              <td>${esc(c.est_tokens)}</td>
              <td><button type="button" class="chunk-open" data-chunk-id="${esc(c.chunk_id)}" style="padding:2px 8px;border-radius:999px;border:1px solid var(--line);background:#152033;color:var(--accent);font-size:11px;cursor:pointer">open</button></td>
            </tr>`).join("") || `<tr><td colspan="6" class="empty">No chunks yet</td></tr>`}
        </tbody>
      </table>`;
  }

  const prog = state.pagesMeta?.progress;
  const progLine = prog && prog.status === "running"
    ? `<div class="timing-live" style="margin-bottom:10px">
        <div class="live-title">● Parsing page ${esc(prog.page || 0)}${prog.total_pages != null ? "/" + esc(prog.total_pages) : ""}</div>
      </div>`
    : "";
  return `${subTabs}
    ${progLine}
    <p class="hint">${esc((state.pages || []).length)} page(s)
      ${state.pagesMeta?.seconds != null ? ` · parse ${esc(state.pagesMeta.seconds)}s` : ""}</p>
    <div class="grid2">${(state.pages || []).map(p => `
      <div>
        <div class="sub" style="margin-bottom:6px;color:var(--muted)">page ${esc(p.page)} · ${esc(p.chars)} chars · ~${esc(p.est_tokens)} tok</div>
        <a href="/api/runs/${encodeURIComponent(state.runId)}/file?path=${encodeURIComponent(p.md_path)}" target="_blank">open md</a>
        <pre class="code" data-md="${esc(p.md_path)}" style="max-height:220px"></pre>
      </div>`).join("") || `<div class="empty">No pages yet — parse may still be running</div>`}</div>`;
}

function renderMasterOutput() {
  const output = state.agentTree?.output || state.info?.result || {};
  const kv = Array.isArray(output.kv_results) ? output.kv_results : [];
  const err = output.error || state.info?.error?.error || state.info?.error;
  const nKv = output.n_kv != null ? output.n_kv : kv.length;

  if (err && !kv.length) {
    return `<details class="tree-node output" open>
      <summary>
        <span class="title">Output</span>
        <span class="tree-badge err">error</span>
      </summary>
      <div class="tree-body">
        <pre class="pretty">${esc(String(err))}</pre>
      </div>
    </details>`;
  }

  if (!kv.length) {
    return `<details class="tree-node output" open>
      <summary>
        <span class="title">Output</span>
        <span class="tree-badge warn">empty</span>
      </summary>
      <div class="tree-body">
        <div class="tree-kv">No kv_results in this run.</div>
        ${err ? `<pre class="pretty">${esc(String(err))}</pre>` : ""}
      </div>
    </details>`;
  }

  const rows = kv.map(item => {
    const evidence = Array.isArray(item.evidence) ? item.evidence : [];
    const evidenceText = evidence.map(ev => {
      const page = ev.page != null ? `p${ev.page}` : (ev.chunk_id || "");
      const text = ev.text || ev.value_reason || ev.evidence_quote || "";
      return page ? `[${page}] ${text}` : text;
    }).filter(Boolean).join(" · ") || (item.value_reason || item.evidence_quote || "");
    const reasons = item.search_reasons || item.page_reasons || (item.reason ? { not_found: item.reason } : {});
    const reasonText = (reasons && typeof reasons === "object")
      ? Object.entries(reasons).map(([p, t]) => (String(p).match(/^\d+$/) ? `p${p}: ${t}` : `${p}: ${t}`)).join(" · ")
      : (typeof reasons === "string" ? reasons : (item.reason || ""));
    const found = item.found;
    const foundBadge = found === true
      ? `<span class="tree-badge ok">found</span>`
      : (found === false ? `<span class="tree-badge warn">not found</span>` : "");
    return `<tr>
      <td>${esc(item.key)}</td>
      <td>${esc(item.value)} ${foundBadge}</td>
      <td>
        ${evidenceText ? `<div><span class="ev-label vlm">VLM</span> ${esc(evidenceText)}</div>` : ""}
        ${reasonText ? `<div style="margin-top:4px"><span class="ev-label search">Search</span> ${esc(reasonText)}</div>` : ""}
      </td>
    </tr>`;
  }).join("");

  return `<details class="tree-node output" open>
    <summary>
      <span class="title">Output</span>
      <span class="tree-badge ok">${esc(nKv)} keys</span>
    </summary>
    <div class="tree-body">
      <p class="hint" style="margin:0 0 8px">
        Final <code>kv_results</code> assembled from MasterAgent run
        (${esc(state.info?.meta?.status || "unknown")}).
      </p>
      <table class="kv-table">
        <thead><tr><th>Key</th><th>Value</th><th>Evidence</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <details style="margin-top:10px">
        <summary class="tree-kv" style="cursor:pointer">raw JSON</summary>
        <pre class="pretty">${esc(pretty({ kv_results: kv }, 12000))}</pre>
      </details>
    </div>
  </details>`;
}

function renderEvalOutput() {
  const er = state.agentTree?.eval_result || {};
  const verdict = String(er.is_correct_answer || "").toLowerCase();
  const goldVerdict = String(er.is_valid_gold || "").toLowerCase();
  const cls = verdict === "correct" ? "ok" : (verdict === "incorrect" ? "warn" : "");
  const goldCls = goldVerdict === "valid" ? "ok" : (goldVerdict === "invalid" ? "warn" : "");
  const summary = er.reason_summary || er.reason || "";
  const detail = er.reason_detail || er.text || "";
  if (!verdict && !goldVerdict && !summary && !detail) {
    return `<details class="tree-node output" open>
      <summary><span class="title">Eval output</span><span class="tree-badge warn">pending</span></summary>
      <div class="tree-body"><div class="tree-kv">No submit_evaluation verdict recorded yet.</div></div>
    </details>`;
  }
  return `<details class="tree-node output" open>
    <summary>
      <span class="title">Eval output</span>
      ${verdict ? `<span class="tree-badge ${cls}">pred: ${esc(verdict)}</span>` : ""}
      ${goldVerdict ? `<span class="tree-badge ${goldCls}">GT: ${esc(goldVerdict)}</span>` : ""}
      ${state.evalKey ? `<span class="tree-kv">${esc(state.evalKey)}</span>` : ""}
    </summary>
    <div class="tree-body">
      ${summary ? `<div class="tree-kv">${esc(summary)}</div>` : ""}
      ${detail ? `<pre class="pretty" style="max-height:280px">${esc(detail)}</pre>` : ""}
      <details style="margin-top:10px">
        <summary class="tree-kv" style="cursor:pointer">raw JSON</summary>
        <pre class="pretty">${esc(pretty(er, 12000))}</pre>
      </details>
    </div>
  </details>`;
}

function renderAgentHierarchy() {
  const isEval = isEvalHierarchyView();
  if (state.agentTreeLoading) {
    return `${isEval ? renderEvalHierarchyKeyToolbar() : ""}
      <div class="empty">Hierarchy 로딩 중…</div>`;
  }
  if (isEval && !state.embed && !state.evalKey) {
    return `${renderEvalHierarchyKeyToolbar()}
      <div class="empty">Select an eval key, or run agentic-evaluation from the Eval tab.</div>`;
  }
  const tree = state.agentTree;
  if (!tree || !(tree.master_turns || []).length) {
    return `${isEval ? renderEvalHierarchyKeyToolbar() : ""}
      <div class="empty">No ${esc(masterAgentLabel().toLowerCase())} turns found for this run.</div>`;
  }
  const treeIsEval = tree.agent_kind === "eval";
  let html = state.embed ? "" : (isEval
    ? `<p class="hint">
      <b>Eval hierarchy</b> = EvalMaster LLM turns → tools → nested SearchAgent sessions
      (<code>06_agentic_eval/</code>).
    </p>`
    : `<p class="hint">
    <b>KV hierarchy</b> = Master LLM turns → tools → nested SearchAgent sessions
    (<code>03_agent/</code>).<br/>
    Each <code>search_pages</code> call expands into a SearchAgent node.
    Multi-key batches share one ReAct loop; single-key searches show one session per handoff.
  </p>`);
  if (isEval && !state.embed) {
    html += renderEvalHierarchyKeyToolbar();
    if (state.evalKey) {
      html += `<p class="hint">Agentic evaluation trace for key <code>${esc(state.evalKey)}</code>
        (parent run ${runLabelHtml(state.runId)}).</p>`;
    }
  }
  html += `<div class="tree">`;

  html += renderMasterPrompts(tree.master_prompts);

  for (const mt of tree.master_turns) {
    const err = mt.error ? `<span class="tree-badge err">ERROR</span>` : "";
    const toolNames = (mt.tools || []).map(t => t.name).filter(Boolean);
    html += `<details class="tree-node master">
      <summary>
        <span class="title">${esc(masterAgentLabel())} turn ${esc(mt.step)}</span>
        ${err}
        ${masterTurnTimingBadge(mt.timing)}
        ${masterTurnTokenBadges(mt)}
        ${toolNames.map(n => `<span class="pill">${esc(n)}</span>`).join("")}
      </summary>
      <div class="tree-body">
        ${renderMasterTurnTimingDetail(mt)}
        ${renderMasterTurnBody(mt)}
      </div>
    </details>`;
  }

  const unassigned = tree.unassigned_search_steps || {};
  const keys = Object.keys(unassigned);
  if (keys.length) {
    html += `<details class="tree-node">
      <summary><span class="title">Unassigned search steps</span>
        <span class="tree-badge warn">${esc(keys.length)} group(s)</span></summary>
      <div class="tree-body"><pre class="pretty">${esc(pretty(unassigned, 4000))}</pre></div>
    </details>`;
  }

  html += treeIsEval ? renderEvalOutput() : renderMasterOutput();
  html += `</div>`;
  return html;
}


function evalDetailAttrs(kind, key) {
  const id = `${kind}:${key}`;
  const open = state.evalOpenDetails.has(id) ? " open" : "";
  return ` data-eval-detail="${esc(id)}"${open}`;
}

function bindEvalDetailToggles(root) {
  root.querySelectorAll("[data-eval-detail]").forEach(el => {
    el.addEventListener("toggle", () => {
      const id = el.getAttribute("data-eval-detail");
      if (!id) return;
      if (el.open) {
        state.evalOpenDetails.add(id);
        if (id.startsWith("agentic:")) {
          const key = id.slice("agentic:".length);
          ensureEvalChat(key);
        }
      } else {
        state.evalOpenDetails.delete(id);
      }
    });
  });
}

function parseGtPages(text) {
  const raw = String(text || "").trim();
  if (!raw) return [];
  return raw.split(/[,\s]+/).filter(Boolean).map(x => Number(x));
}

function parseGtEvidences(text) {
  return String(text || "").split("\n").map(s => s.trim()).filter(Boolean);
}

function renderGtModal() {
  const edit = state.gtEdit;
  if (!edit) return "";
  const saving = Boolean(edit.saving);
  const msg = edit.message
    ? `<div class="gt-modal-msg ${esc(edit.messageKind || "")}">${esc(edit.message)}</div>` : "";
  return `
    <div class="gt-modal-backdrop" id="gtModalBackdrop">
      <div class="gt-modal" role="dialog" aria-labelledby="gtModalTitle">
        <h3 id="gtModalTitle">${edit.isNew ? "Add ground truth" : "Edit ground truth"}</h3>
        <div class="sub">${esc(edit.document)} · ${esc(edit.key)}</div>
        <label for="gtModalValue">Value</label>
        <input id="gtModalValue" value="${esc(edit.value || "")}" ${saving ? "disabled" : ""} />
        <label for="gtModalEvidences">Evidences (one per line)</label>
        <textarea id="gtModalEvidences" ${saving ? "disabled" : ""}>${esc(edit.evidencesText || "")}</textarea>
        <label for="gtModalPages">Evidence pages (comma-separated)</label>
        <input id="gtModalPages" value="${esc(edit.pagesText || "")}" placeholder="1, 2, 3" ${saving ? "disabled" : ""} />
        ${msg}
        <div class="gt-modal-actions">
          <button type="button" class="primary" id="gtModalSave" ${saving ? "disabled" : ""}>
            ${saving ? "Saving…" : "Save"}
          </button>
          <button type="button" id="gtModalCancel" ${saving ? "disabled" : ""}>Cancel</button>
          <a href="/ground-truth?document=${encodeURIComponent(edit.document || "")}" target="_blank">
            Open in Ground Truth page
          </a>
        </div>
      </div>
    </div>`;
}

async function openGtEditor(key, opts = {}) {
  const document = state.evalReport?.document || runDocument(state.runId);
  if (!document) {
    alert("Document name is not available for this run.");
    return;
  }
  const predRows = state.info?.result?.kv_results || [];
  const predRow = predRows.find(row => row && row.key === key);
  const predValue = predRow && predRow.value != null ? String(predRow.value) : "";
  state.gtEdit = {
    document,
    key,
    value: predValue,
    evidencesText: "",
    pagesText: "",
    loading: true,
    saving: false,
    isNew: false,
    message: null,
    messageKind: null,
    highlightInvalid: Boolean(opts.highlightInvalid),
  };
  paintDetail();
  try {
    const data = await api(`/api/ground-truth/document?document=${encodeURIComponent(document)}`);
    const entry = (data.keys || []).find(row => row.key === key);
    if (entry) {
      state.gtEdit = {
        ...state.gtEdit,
        loading: false,
        value: entry.value || "",
        evidencesText: (entry.evidences || []).join("\n"),
        pagesText: (entry.evidence_pages || []).join(", "),
      };
    } else {
      state.gtEdit = {
        ...state.gtEdit,
        loading: false,
        isNew: true,
        message: data.exists === false
          ? "Document not in answer sheet yet — save to create GT."
          : "No GT for this key yet — save to create.",
        messageKind: "warn",
      };
    }
  } catch (err) {
    state.gtEdit = {
      ...state.gtEdit,
      loading: false,
      isNew: true,
      message: String(err.message || err),
      messageKind: "err",
    };
  }
  paintDetail();
}

function closeGtEditor() {
  state.gtEdit = null;
  paintDetail();
}

async function saveGtEditor() {
  const edit = state.gtEdit;
  if (!edit || edit.loading || edit.saving) return;
  const value = document.getElementById("gtModalValue")?.value ?? "";
  const evidences = parseGtEvidences(document.getElementById("gtModalEvidences")?.value ?? "");
  const pagesText = document.getElementById("gtModalPages")?.value ?? "";
  let evidence_pages;
  try {
    evidence_pages = parseGtPages(pagesText);
    if (pagesText.trim() && evidence_pages.some(n => Number.isNaN(n))) {
      throw new Error("invalid page numbers");
    }
  } catch (err) {
    edit.message = `Pages must be comma-separated integers (${err.message || err})`;
    edit.messageKind = "err";
    paintDetail();
    return;
  }
  edit.saving = true;
  edit.message = null;
  paintDetail();
  try {
    const result = await apiPut("/api/ground-truth/key", {
      document: edit.document,
      key: edit.key,
      value,
      evidences,
      evidence_pages,
    });
    edit.saving = false;
    edit.message = `Saved` + (result.invalidated_eval_caches
      ? ` · invalidated ${result.invalidated_eval_caches} eval cache(s)`
      : "");
    edit.messageKind = "ok";
    paintDetail();
    state.gtEdit = null;
    await ensureEval(true);
  } catch (err) {
    edit.saving = false;
    edit.message = String(err.message || err);
    edit.messageKind = "err";
    paintDetail();
  }
}

function bindGtEditor(root) {
  const backdrop = root.querySelector("#gtModalBackdrop");
  if (!backdrop) return;
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop && !state.gtEdit?.saving) closeGtEditor();
  });
  const cancel = root.querySelector("#gtModalCancel");
  if (cancel) cancel.onclick = () => closeGtEditor();
  const save = root.querySelector("#gtModalSave");
  if (save) save.onclick = () => saveGtEditor();
}

function renderEval() {
  if (state.evalLoading) {
    return `<div class="empty">Scoring against answer_sheet…</div>`;
  }
  if (state.evalError) {
    return `<div class="empty" style="color:var(--err)">Eval failed: ${esc(state.evalError)}</div>
      <button class="tab" id="evalRefresh">Retry</button>`;
  }
  const report = state.evalReport;
  if (!report) {
    return `<div class="empty">No eval report yet.</div>`;
  }
  const hasGt = report.has_gt !== false;
  const o = report.overall || {};
  const cards = [
    ["Value EM", hasGt ? o.value_exact_match : null, `${report.n_keys ?? "—"} keys`],
    ["Page F1 (macro)", hasGt ? o.page_f1_macro : null, hasGt ? `P ${fmtPct(o.page_precision_macro)} / R ${fmtPct(o.page_recall_macro)}` : "no GT"],
    ["Page F1 (micro)", hasGt ? o.page_f1_micro : null, hasGt ? `P ${fmtPct(o.page_precision_micro)} / R ${fmtPct(o.page_recall_micro)}` : "no GT"],
    ["Evidence token F1", hasGt ? o.evidence_token_f1 : null, report.document || ""],
  ].map(([label, val, sub]) => `
    <div class="score-card">
      <div class="label">${esc(label)}</div>
      <div class="value">${hasGt && val != null ? fmtPct(val) : "—"}</div>
      <div class="sub">${esc(sub)}</div>
    </div>`).join("");

  const batch = state.batchJob;
  const batchActive = batch && (batch.status === "queued" || batch.status === "running");
  const batchForRun = batch && batch.run_ids && batch.run_ids.includes(state.runId);
  const aeByKey = state.agenticEvals || {};
  const inflightKeys = Array.isArray(state.agenticEvalInflight)
    ? state.agenticEvalInflight
    : (state.agenticEvalInflight ? [state.agenticEvalInflight] : []);

  const rows = (report.per_key || []).map(row => {
    const em = row.value?.exact_match;
    const sp = row.search_pages || {};
    const et = row.evidence_text || {};
    const sr = row.search_reasons || {};
    const pc = row.page_chunk_id || {};
    const chunkMap = (pc.pred_map && typeof pc.pred_map === "object") ? pc.pred_map : {};
    const chunkJumpRows = Object.entries(chunkMap).map(([page, cid]) => {
      const id = String(cid || "").trim();
      if (!id) return "";
      return `<div class="ev-chunk-row">p${esc(page)}:
        <button type="button" class="chunk-jump" data-chunk-id="${esc(id)}">${esc(id)}</button>
      </div>`;
    }).filter(Boolean).join("");

    const predPages = Array.isArray(sp.pred) ? sp.pred : [];
    const goldPages = Array.isArray(sp.gold) ? sp.gold : [];

    let chunkBlockHtml;
    if (chunkJumpRows) {
      chunkBlockHtml = `<div class="ev-block">
        <span class="ev-label search">SearchAgent chunks</span>
        <div class="ev-text">${chunkJumpRows}</div>
      </div>`;
    } else {
      const predPageBtns = predPages.map(p =>
        `<button type="button" class="pdf-page-btn" data-page="${p}" data-key="${esc(row.key)}" title="Page ${p} 원문 PDF 보기">p.${p} 원문</button>`
      ).join(" ");
      chunkBlockHtml = `<div class="ev-block">
        <span class="ev-label search">SearchAgent chunks</span>
        <div class="ev-text" style="color:var(--muted)">
          ${predPages.length > 0
            ? `청크 미지정 (검색 페이지: ${predPageBtns})`
            : `검색된 청크 없음 <button type="button" class="pdf-page-btn" data-page="1" data-key="${esc(row.key)}" title="PDF 1페이지부터 원문 열기" style="margin-left:4px">PDF 원문 보기</button>`}
        </div>
      </div>`;
    }

    const goldPageBtns = goldPages.length > 0
      ? `<div style="margin-top:4px">
          <span style="color:var(--muted);font-size:10px">gold pages:</span>
          ${goldPages.map(p => `<button type="button" class="pdf-page-btn" data-page="${p}" data-key="${esc(row.key)}" title="Gold Page ${p} 원문 PDF 보기" style="margin-left:2px;border-color:#4a4020;background:#2a2618;color:#e0d0a0">p.${p}</button>`).join(" ")}
        </div>`
      : "";

    const ae = aeByKey[row.key];
    const keyInflight = inflightKeys.includes(row.key);
    const batchActiveForKey = Boolean(batchActive && batch && (
      (batch.current && batch.current.run_id === state.runId && batch.current.key === row.key) ||
      (batch.active || []).some(x => x.run_id === state.runId && x.key === row.key)
    ));
    const isActuallyRunning = Boolean(keyInflight || batchActiveForKey);
    const goldVerdictForBtn = String((ae && ae.is_valid_gold) || "").toLowerCase();
    const gtEditCls = goldVerdictForBtn === "invalid" ? " warn" : (hasGt ? "" : " warn");
    const gtEditBtn = `<button type="button" class="gt-edit-btn${gtEditCls}" data-gt-edit="${esc(row.key)}">
      ${hasGt ? "Edit GT" : "Add GT"}</button>`;
    let agenticCell;
    if (isActuallyRunning) {
      agenticCell = `<button type="button" class="agentic-eval-btn" disabled>Running…</button>`;
    } else if (ae && ae.status === "running") {
      agenticCell = `<div class="agentic-eval-err" title="Interrupted while running. Click Retry to re-run.">interrupted (running)</div>
        <button type="button" class="agentic-eval-btn" data-agentic-key="${esc(row.key)}"
          ${batchActive ? "disabled" : ""}>Retry</button>`;
    } else if (ae && ae.status === "error") {
      agenticCell = `<div class="agentic-eval-err">${esc(ae.error || "error")}</div>
        <button type="button" class="agentic-eval-btn" data-agentic-key="${esc(row.key)}"
          ${batchActive ? "disabled" : ""}>Retry</button>`;
    } else if (ae && (ae.status === "done" || ae.is_correct_answer || ae.is_valid_gold || ae.reason_summary || ae.reason || ae.text)) {
      const verdict = String(ae.is_correct_answer || "").toLowerCase();
      const goldVerdict = String(ae.is_valid_gold || "").toLowerCase();
      const verdictCls = verdict === "correct" ? "correct" : (verdict === "incorrect" ? "incorrect" : "");
      const goldCls = goldVerdict === "valid" ? "valid" : (goldVerdict === "invalid" ? "invalid" : "");
      const summary = ae.reason_summary || ae.reason || "";
      const detail = ae.reason_detail || ae.text || "";
      const isIncomplete = summary.includes("평가가 완료되지 않았습니다")
        || detail.includes("submit_evaluation을 호출하지 않았습니다");

      const mentionedPages = extractPageNumbersFromText(detail + " " + summary);
      const pageChipsHtml = mentionedPages.length > 0
        ? `<div class="eval-page-chips">
            <span style="color:var(--muted);font-size:11px">검증 페이지:</span>
            ${mentionedPages.map(p => `<button type="button" class="pdf-page-btn" data-page="${p}" data-key="${esc(row.key)}" title="Page ${p} 원문 PDF 열기">p.${p} 원문</button>`).join("")}
          </div>`
        : "";
      const formattedDetail = formatAgenticDetailText(detail, row.key);

      agenticCell = `
        <div class="agentic-eval-verdicts">
          ${verdict === "correct" || verdict === "incorrect"
            ? `<div class="agentic-eval-verdict ${verdictCls}">pred: ${esc(verdict)}</div>` : ""}
          ${goldVerdict === "valid" || goldVerdict === "invalid"
            ? `<div class="agentic-eval-verdict ${goldCls}">GT: ${esc(goldVerdict)}</div>` : ""}
          ${isIncomplete ? `<div class="agentic-eval-verdict warn" style="color:var(--warn,#e0a45c);background:rgba(224,164,92,0.15);border:1px solid rgba(224,164,92,0.4)">미완료</div>` : ""}
        </div>
        ${summary ? `<div class="agentic-eval-summary">${esc(summary)}</div>` : ""}
        ${pageChipsHtml}
        ${detail ? `<div class="agentic-eval-detail"><details${evalDetailAttrs("agentic", row.key)}>
          <summary>상세</summary>
          <div class="agentic-eval-text">${formattedDetail}</div>
          ${renderAgenticEvalChat(row.key)}
        </details></div>` : ""}
        <div style="margin-top:6px;display:flex;gap:6px;align-items:center">
          <button type="button" class="agentic-eval-btn" data-agentic-key="${esc(row.key)}"
            ${batchActive ? "disabled" : ""}>Retry</button>
          ${mentionedPages.length === 0 ? `<button type="button" class="pdf-page-btn" data-page="1" data-key="${esc(row.key)}" title="PDF 원문 열기">원문 PDF</button>` : ""}
        </div>`;
    } else {
      agenticCell = `<button type="button" class="agentic-eval-btn" data-agentic-key="${esc(row.key)}"
        ${batchActive ? "disabled" : ""}>agentic-evaluation</button>`;
    }
    const isWc = state.wrongCaseKeys && state.wrongCaseKeys.has(row.key);
    const wcCase = isWc ? state.wrongCasesByKey[row.key] : null;
    const wcBtn = isWc
      ? `<button type="button" class="wrong-case-btn active" data-wc-key="${esc(row.key)}" data-wc-id="${esc(wcCase?.id || "")}" title="Registered in Wrong Cases. Click to view or manage.">✓ In Wrong Cases</button>`
      : `<button type="button" class="wrong-case-btn" data-wc-key="${esc(row.key)}" title="Send this key to Wrong Cases">+ Wrong Case</button>`;

    return `<tr class="eval-row" data-key-row="${esc(row.key)}">
      <td class="key">${esc(row.key)}</td>
      <td class="${hasGt ? (em ? "em-y" : "em-n") : ""}">${hasGt ? (em ? "Y" : "N") : "—"}</td>
      <td>${hasGt ? fmtPct(sp.f1) : "—"}<div class="sub">pred [${esc((sp.pred||[]).join(", "))}]${hasGt ? ` · gold [${esc((sp.gold||[]).join(", "))}]` : ""}</div></td>
      <td>${hasGt ? fmtPct(et.token_f1) : "—"}</td>
      <td>
        <div><b>pred</b> ${esc(row.value?.pred ?? "")}</div>
        <div><b>gold</b> ${hasGt ? esc(row.value?.gold ?? "") : `<span style="color:var(--muted)">(none)</span>`}</div>
        <details${evalDetailAttrs("evidence", row.key)}>
          <summary>VLM reason / evidence · Search reasons</summary>
          <div class="ev-block">
            <span class="ev-label vlm">VLM value_reason</span>
            <div class="ev-text">${esc(et.pred || "(empty)")}</div>
          </div>
          <div class="ev-block">
            <span class="ev-label search">SearchAgent page_reasons</span>
            <div class="ev-text">${esc(sr.pred || row.reason || "(empty)")}</div>
          </div>
          ${chunkBlockHtml}
          <div class="ev-block">
            <span class="ev-label gold">gold evidences</span>
            <div class="ev-text">${esc(et.gold || "(empty)")}${goldPageBtns}</div>
          </div>
        </details>
        <div style="margin-top:6px;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
          ${gtEditBtn}
          ${wcBtn}
        </div>
      </td>
      <td>${agenticCell}</td>
    </tr>`;
  }).join("");

  const aeErr = state.agenticEvalError
    ? `<p class="hint" style="color:var(--err)">Agentic eval: ${esc(state.agenticEvalError)}</p>`
    : "";

  let batchHtml = "";
  if (batchForRun && batch) {
    const pct = batch.progress_pct ?? (batch.total ? Math.round(100 * batch.completed / batch.total) : 0);
    const cur = batch.current
      ? ` · ${esc(batch.current.key)}`
      : "";
    const activeN = Array.isArray(batch.active) ? batch.active.length : 0;
    const activeHint = activeN > 1 ? ` (${activeN} parallel)` : "";
    batchHtml = `
      <div class="hint" style="border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#152033">
        Batch agentic eval: <b>${esc(batch.status)}</b>
        ${batch.completed}/${batch.total} (${pct}%)${cur}${activeHint}
        ${batchActive ? `<button type="button" class="tab" id="batchRefresh" style="margin-left:8px">Refresh status</button>
        <button type="button" class="tab" id="batchCancel" style="margin-left:8px">Cancel</button>` : ""}
      </div>`;
  }

  const allKeysDisabled = batchActive;
  const hasStale = (report.per_key || []).some(row => {
    const ae = aeByKey[row.key];
    const keyInflight = inflightKeys.includes(row.key);
    const batchActiveForKey = Boolean(batchActive && batch && (
      (batch.current && batch.current.run_id === state.runId && batch.current.key === row.key) ||
      (batch.active || []).some(x => x.run_id === state.runId && x.key === row.key)
    ));
    return Boolean(ae && ae.status === "running" && !keyInflight && !batchActiveForKey);
  });

  const isIncompleteKey = (row) => {
    const ae = aeByKey[row.key];
    if (!ae) return true;
    if (ae.status === "error" || ae.status === "running") return true;
    const sum = String(ae.reason_summary || ae.reason || "");
    const det = String(ae.reason_detail || ae.text || "");
    return sum.includes("평가가 완료되지 않았습니다") || det.includes("submit_evaluation을 호출하지 않았습니다");
  };
  const incompleteRows = (report.per_key || []).filter(isIncompleteKey);
  const incompleteKeys = incompleteRows.map(r => r.key);
  state.incompleteEvalKeys = incompleteKeys;

  const noGtBanner = !hasGt
    ? `<div class="hint" style="border:1px solid var(--warn);border-radius:8px;padding:10px 12px;background:#2a2218;margin-bottom:12px">
        No ground truth in <code>answer_sheet.json</code> for <b>${esc(report.document)}</b>.
        Predictions are shown below. Use <b>Add GT</b> to create gold entries.
      </div>`
    : "";

  return `
    ${noGtBanner}
    <p class="hint">
      Baseline metrics vs <code>dataset/answer_sheet.json</code>.
      Cached as <code>05_eval.json</code> in the run directory.
      Evid F1 uses <b>VLM value_reason</b> only; SearchAgent <b>page_reasons</b> are shown separately.
      Agentic-evaluation runs up to 8 keys in parallel via the inference API and saves under <code>06_agentic_eval/</code>.
      Use <b>${hasGt ? "Edit GT" : "Add GT"}</b> when agentic eval marks gold as invalid or GT is missing.
      <button class="tab" id="evalRefresh" style="margin-left:8px">Recompute</button>
      <button type="button" class="agentic-eval-btn" id="evalAllKeys"
        style="margin-left:8px" ${allKeysDisabled ? "disabled" : ""}>Evaluate all keys</button>
      ${incompleteKeys.length > 0 ? `<button type="button" class="agentic-eval-btn" id="evalRetryIncomplete"
        style="margin-left:8px" ${allKeysDisabled ? "disabled" : ""}>Retry incomplete (${incompleteKeys.length})</button>` : ""}
      ${hasStale ? `<button type="button" class="tab" id="cleanStaleEval" style="margin-left:8px;color:var(--warn,#e0a45c)" title="Clean up interrupted or dead running tasks">Clear Stale</button>` : ""}
      <button type="button" class="tab" id="sendAllWrongCases" style="margin-left:8px;border-color:var(--warn,#e0a45c);color:var(--warn,#e0a45c)" title="Send all keys with EM=N to Wrong Cases">Send EM=N to Wrong Cases</button>
    </p>
    ${batchHtml}
    ${aeErr}
    <div class="score-grid">${cards}</div>
    <table class="eval-table">
      <thead>
        <tr>
          <th>Key</th><th>EM</th><th>Page F1</th><th>Evid F1</th><th>Values / reasons</th><th>Agentic eval</th>
        </tr>
      </thead>
      <tbody>${rows || `<tr><td colspan="6" class="empty">No keys</td></tr>`}</tbody>
    </table>`;
}

async function ensureEval(refresh=false) {
  if (!state.runId) return;
  if (state.evalLoading) return;
  if (!refresh && state.evalReport && !state.evalError) {
    await Promise.all([ensureAgenticEvals(), loadWrongCasesForRun()]);
    paintDetail();
    return;
  }
  state.evalLoading = true;
  state.evalError = null;
  paintDetail();
  try {
    const q = refresh ? "?refresh=1" : "";
    state.evalReport = await api(`/api/runs/${encodeURIComponent(state.runId)}/eval${q}`);
    state.evalError = null;
    const es = {
      value_exact_match: state.evalReport?.overall?.value_exact_match,
      page_f1_macro: state.evalReport?.overall?.page_f1_macro,
      evidence_token_f1: state.evalReport?.overall?.evidence_token_f1,
      n_keys: state.evalReport?.n_keys,
      document: state.evalReport?.document,
    };
    state.runs = state.runs.map(r => r.run_id === state.runId ? {
      ...r,
      eval_summary: es,
      document: state.evalReport?.document || r.document,
    } : r);
    renderRuns();
    await Promise.all([ensureAgenticEvals(), loadWrongCasesForRun()]);
  } catch (err) {
    state.evalReport = null;
    state.evalError = String(err.message || err);
  } finally {
    state.evalLoading = false;
    paintDetail();
  }
}

async function ensureEvalChat(key) {
  if (!state.runId || !key) return;
  if (!state.agenticChats[key]) {
    state.agenticChats[key] = { messages: [], loading: false, loaded: false, input: "", error: null };
  }
  const chat = state.agenticChats[key];
  if (chat.loaded || chat.loading) return;
  chat.loading = true;
  try {
    const res = await api(`/api/runs/${encodeURIComponent(state.runId)}/agentic-eval/chat?key=${encodeURIComponent(key)}`);
    if (res && res.history) {
      chat.messages = res.history;
    }
    chat.loaded = true;
  } catch (err) {
    console.warn("Failed to load chat history for key", key, err);
  } finally {
    chat.loading = false;
    paintDetail();
  }
}

async function sendEvalChatMessage(key) {
  if (!state.runId || !key) return;
  if (!state.agenticChats[key]) {
    state.agenticChats[key] = { messages: [], loading: false, loaded: true, input: "", error: null };
  }
  const chat = state.agenticChats[key];
  const text = (chat.input || "").trim();
  if (!text || chat.loading) return;

  chat.loading = true;
  chat.error = null;
  const now = new Date().toISOString();
  chat.messages.push({ role: "user", content: text, timestamp: now });
  chat.input = "";
  paintDetail();

  setTimeout(() => {
    try {
      const box = document.querySelector(`[data-chat-box="${CSS.escape(key)}"]`);
      if (box) box.scrollTop = box.scrollHeight;
    } catch (_) {}
  }, 10);

  try {
    const res = await apiPost(`/api/runs/${encodeURIComponent(state.runId)}/agentic-eval/chat`, {
      key: key,
      message: text,
    });
    if (res && res.history) {
      chat.messages = res.history;
    } else if (res && res.reply) {
      chat.messages.push({
        role: "assistant",
        content: res.reply,
        timestamp: new Date().toISOString(),
        tool_calls: res.tool_calls,
      });
    }
  } catch (err) {
    chat.error = err.message || "답변 생성 중 오류가 발생했습니다.";
  } finally {
    chat.loading = false;
    paintDetail();
    setTimeout(() => {
      try {
        const box = document.querySelector(`[data-chat-box="${CSS.escape(key)}"]`);
        if (box) box.scrollTop = box.scrollHeight;
        const inp = document.querySelector(`[data-chat-input="${CSS.escape(key)}"]`);
        if (inp) inp.focus();
      } catch (_) {}
    }, 20);
  }
}

async function clearEvalChat(key) {
  if (!state.runId || !key) return;
  if (!confirm(`"${key}"의 대화 기록을 초기화하시겠습니까?`)) return;
  try {
    await apiDelete(`/api/runs/${encodeURIComponent(state.runId)}/agentic-eval/chat?key=${encodeURIComponent(key)}`);
    state.agenticChats[key] = { messages: [], loading: false, loaded: true, input: "", error: null };
    showToast("대화 기록이 초기화되었습니다.");
    paintDetail();
  } catch (err) {
    showToast("대화 초기화 실패: " + err.message);
  }
}

function renderAgenticEvalChat(key) {
  if (!state.agenticChats[key]) {
    state.agenticChats[key] = { messages: [], loading: false, loaded: false, input: "", error: null };
  }
  const chat = state.agenticChats[key];
  if (!chat.loaded && !chat.loading && state.evalOpenDetails.has(`agentic:${key}`)) {
    setTimeout(() => ensureEvalChat(key), 0);
  }

  const msgs = chat.messages || [];
  const msgListHtml = msgs.length > 0
    ? msgs.map(m => {
        const isUser = m.role === "user";
        const bubbleCls = isUser ? "eval-chat-msg user" : "eval-chat-msg assistant";
        const roleLabel = isUser ? "👤 질문" : "🤖 평가 에이전트";
        let toolCallsHtml = "";
        if (!isUser && Array.isArray(m.tool_calls) && m.tool_calls.length > 0) {
          const names = m.tool_calls.map(tc => tc.name).join(", ");
          toolCallsHtml = `<div class="eval-chat-tools-badge" title="${esc(JSON.stringify(m.tool_calls))}">
            🔍 도구 실행: ${esc(names)}
          </div>`;
        }
        return `<div class="${bubbleCls}">
          <div class="eval-chat-msg-header">${esc(roleLabel)}</div>
          ${toolCallsHtml}
          <div class="eval-chat-msg-body">${esc(m.content || "")}</div>
        </div>`;
      }).join("")
    : `<div class="eval-chat-empty">
        채점 결과나 근거에 대해 질문해보세요.<br/>
        <span style="font-size:10px;color:var(--muted)">예: "Transformer라는 명시가 p.14에 실제로 어디에 있나요?", "무엇을 근거로 판단했나요?"</span>
      </div>`;

  return `
    <div class="eval-chat-section">
      <div class="eval-chat-header">
        <span>💬 평가 에이전트와 대화하기</span>
        ${msgs.length > 0 ? `<button type="button" class="eval-chat-clear-btn" data-chat-clear="${esc(key)}" title="대화 내역 초기화">대화 초기화</button>` : ""}
      </div>
      <div class="eval-chat-msgs-box" data-chat-box="${esc(key)}">
        ${msgListHtml}
        ${chat.loading ? `<div class="eval-chat-msg assistant loading"><div class="eval-chat-msg-header">🤖 평가 에이전트</div><div class="eval-chat-msg-body">답변을 생각하고 필요한 문서를 검색하는 중입니다… ⏳</div></div>` : ""}
      </div>
      ${chat.error ? `<div class="eval-chat-error">${esc(chat.error)}</div>` : ""}
      <div class="eval-chat-input-row">
        <input type="text" class="eval-chat-input" data-chat-input="${esc(key)}"
          placeholder="질문을 입력하세요 (Enter로 전송)"
          value="${esc(chat.input || "")}"
          ${chat.loading ? "disabled" : ""} />
        <button type="button" class="eval-chat-send-btn" data-chat-send="${esc(key)}"
          ${chat.loading ? "disabled" : ""}>
          ${chat.loading ? "전송 중…" : "전송"}
        </button>
      </div>
    </div>
  `;
}

async function ensureAgenticEvals() {
  if (!state.runId) return;
  try {
    const data = await api(`/api/runs/${encodeURIComponent(state.runId)}/agentic-eval`);
    state.agenticEvals = data.by_key || {};
    if (!state.agenticEvalInflight || !state.agenticEvalInflight.length) {
      state.agenticEvalInflight = data.inflight || [];
    }
    state.agenticEvalError = null;
  } catch (err) {
    state.agenticEvalError = String(err.message || err);
  }
}

async function runAgenticEval(key) {
  if (!state.runId || !key) return;
  if (state.batchJob && (state.batchJob.status === "queued" || state.batchJob.status === "running")) return;
  delete state.agenticChats[key];
  const inflightKeys = Array.isArray(state.agenticEvalInflight) ? [...state.agenticEvalInflight] : [];
  if (!inflightKeys.includes(key)) inflightKeys.push(key);
  state.agenticEvalInflight = inflightKeys;
  state.agenticEvalError = null;
  state.agenticEvals = {
    ...state.agenticEvals,
    [key]: { key, status: "running" },
  };
  paintDetail();
  try {
    const r = await fetch(
      `/api/runs/${encodeURIComponent(state.runId)}/agentic-eval`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      }
    );
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch (_) { data = { detail: text }; }
    if (!r.ok) {
      throw new Error(data.detail || text || r.statusText);
    }
    state.agenticEvals = { ...state.agenticEvals, [key]: data };
    if (state.agentTreeCache.runId === state.runId) {
      delete state.agentTreeCache.eval[key];
    }
  } catch (err) {
    state.agenticEvalError = String(err.message || err);
    state.agenticEvals = {
      ...state.agenticEvals,
      [key]: { key, status: "error", error: String(err.message || err) },
    };
  } finally {
    state.agenticEvalInflight = (state.agenticEvalInflight || []).filter(k => k !== key);
    await ensureAgenticEvals();
    await loadEvalHierarchyKeys();
    if (state.tab === "hierarchy_eval" && state.evalKey === key) {
      state.agentTree = await loadAgentTree();
    }
    try {
      state.runs = await api("/api/runs");
      renderRuns();
    } catch (_) {}
    paintDetail();
  }
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch (_) { data = { detail: text }; }
  if (!r.ok) throw new Error(data.detail || text || r.statusText);
  return data;
}

async function apiPut(path, body) {
  const r = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch (_) { data = { detail: text }; }
  if (!r.ok) throw new Error(data.detail || text || r.statusText);
  return data;
}

async function refreshInferenceBatchJob() {
  if (!state.batchJob?.job_id) return;
  try {
    state.batchJob = await api(`/api/evaluation/batch-jobs/${encodeURIComponent(state.batchJob.job_id)}`);
    await ensureAgenticEvals();
    paintDetail();
    if (state.batchJob.status !== "running" && state.batchJob.status !== "queued") {
      state.runs = await api("/api/runs");
      renderRuns();
    }
  } catch (err) {
    state.agenticEvalError = String(err.message || err);
    paintDetail();
  }
}

function stopBatchPoll() {
  if (state.batchPollTimer) {
    clearInterval(state.batchPollTimer);
    state.batchPollTimer = null;
  }
}

async function runAllAgenticEvals() {
  if (!state.runId) return;
  if (state.batchJob && (state.batchJob.status === "queued" || state.batchJob.status === "running")) return;
  state.agenticEvalError = null;
  try {
    state.batchJob = await apiPost("/api/evaluation/batch-agentic-eval", {
      run_ids: [state.runId],
      skip_existing: true,
    });
    paintDetail();
    if (state.batchJob.status !== "running" && state.batchJob.status !== "queued") {
      await ensureAgenticEvals();
      paintDetail();
    }
  } catch (err) {
    state.agenticEvalError = String(err.message || err);
    paintDetail();
  }
}

async function cancelInferenceBatch() {
  if (!state.batchJob?.job_id) return;
  try {
    state.batchJob = await apiPost(
      `/api/evaluation/batch-jobs/${encodeURIComponent(state.batchJob.job_id)}/cancel`, {}
    );
    await ensureAgenticEvals();
    paintDetail();
  } catch (err) {
    state.agenticEvalError = String(err.message || err);
    paintDetail();
  }
}

async function resumeInferenceBatchJob() {
  try {
    const data = await api("/api/evaluation/batch-jobs/active");
    if (data.active && data.job && (data.job.run_ids || []).includes(state.runId)) {
      state.batchJob = data.job;
    }
  } catch (_) {}
}

function paintDetail() {
  const detail = document.getElementById("detail");
  let body = "";
  if (state.tab === "hierarchy_kv" || state.tab === "hierarchy_eval") {
    body = renderAgentHierarchy();
  } else if (state.tab === "timing") {
    body = renderTiming();
  } else if (state.tab === "pages") {
    body = renderPagesChunks();
  } else if (state.tab === "eval") {
    body = renderEval();
  }
  detail.innerHTML = `
    ${state.embed ? "" : `<div class="meta" style="margin-bottom:10px;color:var(--muted);display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      ${runLabelHtml(state.runId)}
      ${state.tab === "hierarchy_eval" && state.evalKey ? `· eval key <code>${esc(state.evalKey)}</code>` : ""}
      · status=${esc(state.info?.meta?.status || (state.info?.meta?.finished_at ? "done" : "running"))}
      · ${esc(state.info?.meta?.seconds)}s
      <button type="button" id="runRefresh" style="margin-left:4px;padding:2px 10px;border-radius:999px;border:1px solid var(--line);background:#152033;color:var(--text);font-size:12px;cursor:pointer">Refresh</button>
      <button type="button" id="runDelete" style="margin-left:4px;padding:2px 10px;border-radius:999px;border:1px solid #7a3a3f;background:#2a1518;color:var(--err);font-size:12px;cursor:pointer"
        ${isRunBusy(state.runId) ? "disabled" : ""}>Delete run</button>
    </div>`}
    ${tabsHtml()}
    ${body}${renderGtModal()}`;
  detail.querySelectorAll(".tab").forEach(btn => {
    btn.onclick = () => {
      if (btn.dataset.pagesSub) {
        state.pagesSubtab = btn.dataset.pagesSub;
        paintDetail();
        return;
      }
      state.tab = btn.dataset.tab;
      if (state.tab === "hierarchy_kv" || state.tab === "hierarchy_eval") {
        loadHierarchyTab();
      } else if (state.tab === "timing") {
        loadTimingTab();
      } else if (state.tab === "pages") {
        renderDetail();
      } else {
        paintDetail();
      }
      if (state.tab === "eval") ensureEval(false);
    };
  });
  const evalKeySel = document.getElementById("evalHierarchyKey");
  if (evalKeySel) {
    evalKeySel.onchange = () => {
      state.evalKey = evalKeySel.value || null;
      loadHierarchyTab();
    };
  }
  const runRefresh = document.getElementById("runRefresh");
  if (runRefresh) runRefresh.onclick = () => renderDetail({ force: true });
  const runDelete = document.getElementById("runDelete");
  if (runDelete) runDelete.onclick = () => deleteRun(state.runId);
  const refreshBtn = document.getElementById("evalRefresh");
  if (refreshBtn) refreshBtn.onclick = () => ensureEval(true);
  if (state.tab === "eval" && !state.evalReport && !state.evalLoading && !state.evalError) {
    ensureEval(false);
  }
  detail.querySelectorAll("[data-agentic-key]").forEach(btn => {
    btn.onclick = () => runAgenticEval(btn.dataset.agenticKey);
  });
  const evalAllBtn = document.getElementById("evalAllKeys");
  if (evalAllBtn) evalAllBtn.onclick = () => runAllAgenticEvals();
  const evalRetryIncompleteBtn = document.getElementById("evalRetryIncomplete");
  if (evalRetryIncompleteBtn) {
    evalRetryIncompleteBtn.onclick = async () => {
      const keys = state.incompleteEvalKeys || [];
      if (keys.length === 1) {
        runAgenticEval(keys[0]);
      } else if (keys.length > 1) {
        runAllAgenticEvals();
      }
    };
  }
  const evalHierarchyRetry = document.getElementById("evalHierarchyRetry");
  if (evalHierarchyRetry && state.evalKey) {
    evalHierarchyRetry.onclick = () => runAgenticEval(state.evalKey);
  }
  const cleanStaleBtn = document.getElementById("cleanStaleEval");
  if (cleanStaleBtn) {
    cleanStaleBtn.onclick = async () => {
      cleanStaleBtn.disabled = true;
      try {
        await apiPost(`/api/runs/${encodeURIComponent(state.runId)}/agentic-eval/cleanup-stale`, {});
        await ensureEval(true);
      } catch (err) {
        alert("Failed to clear stale tasks: " + (err.message || err));
      }
    };
  }
  const batchRefresh = document.getElementById("batchRefresh");
  if (batchRefresh) batchRefresh.onclick = () => refreshInferenceBatchJob();
  const batchCancel = document.getElementById("batchCancel");
  if (batchCancel) batchCancel.onclick = () => cancelInferenceBatch();
  if (state.tab === "eval") {
    bindEvalDetailToggles(detail);
    detail.querySelectorAll("[data-gt-edit]").forEach(btn => {
      btn.onclick = () => openGtEditor(btn.dataset.gtEdit, {
        highlightInvalid: btn.classList.contains("warn"),
      });
    });
    bindGtEditor(detail);
    resumeInferenceBatchJob();

    detail.querySelectorAll(".wrong-case-btn").forEach(btn => {
      btn.onclick = async () => {
        const key = btn.dataset.wcKey;
        const isWc = btn.classList.contains("active");
        if (isWc) {
          const caseId = btn.dataset.wcId;
          const choice = confirm(`"${key}" is registered in Wrong Cases.\n\nClick OK to open the Wrong Cases page, or Cancel to remove it from Wrong Cases.`);
          if (choice) {
            window.location.href = `/wrong-cases?id=${encodeURIComponent(caseId)}`;
          } else {
            if (confirm(`Remove "${key}" from Wrong Cases?`)) {
              try {
                await api(`/api/wrong-cases/${encodeURIComponent(caseId)}`, { method: "DELETE" });
                state.wrongCaseKeys.delete(key);
                delete state.wrongCasesByKey[key];
                paintDetail();
                showToast(`Removed "${key}" from Wrong Cases`);
              } catch (err) {
                alert(`Failed to remove: ${err.message}`);
              }
            }
          }
          return;
        }

        try {
          btn.textContent = "Adding…";
          btn.disabled = true;
          const res = await apiPost("/api/wrong-cases", {
            run_id: state.runId,
            key: key,
          });
          state.wrongCaseKeys.add(key);
          state.wrongCasesByKey[key] = res;
          paintDetail();
          showToast(`Added "${key}" to Wrong Cases`, "View in Wrong Cases", () => {
            window.location.href = `/wrong-cases?id=${encodeURIComponent(res.id)}`;
          });
        } catch (err) {
          alert(`Failed to add to Wrong Cases: ${err.message}`);
          btn.textContent = "+ Wrong Case";
          btn.disabled = false;
        }
      };
    });

    const sendAllBtn = document.getElementById("sendAllWrongCases");
    if (sendAllBtn) {
      sendAllBtn.onclick = async () => {
        const report = state.evalReport;
        if (!report || !Array.isArray(report.per_key)) return;
        const incorrectKeys = report.per_key
          .filter(r => r.value?.exact_match === false)
          .map(r => r.key);
        if (incorrectKeys.length === 0) {
          alert("No incorrect keys (EM=N) found in this run.");
          return;
        }
        if (!confirm(`Send ${incorrectKeys.length} incorrect keys to Wrong Cases?`)) return;
        try {
          sendAllBtn.textContent = "Sending…";
          sendAllBtn.disabled = true;
          const res = await apiPost("/api/wrong-cases/batch", {
            run_id: state.runId,
            keys: incorrectKeys,
          });
          await loadWrongCasesForRun();
          paintDetail();
          showToast(`Added ${res.count} keys to Wrong Cases`, "View in Wrong Cases", () => {
            window.location.href = "/wrong-cases";
          });
        } catch (err) {
          alert(`Failed to batch add: ${err.message}`);
          sendAllBtn.textContent = "Send all EM=N to Wrong Cases";
          sendAllBtn.disabled = false;
        }
      };
    }

    detail.querySelectorAll("[data-chat-send]").forEach(btn => {
      btn.onclick = () => sendEvalChatMessage(btn.dataset.chatSend);
    });
    detail.querySelectorAll("[data-chat-clear]").forEach(btn => {
      btn.onclick = () => clearEvalChat(btn.dataset.chatClear);
    });
    detail.querySelectorAll("[data-chat-input]").forEach(input => {
      input.oninput = (e) => {
        const k = input.dataset.chatInput;
        if (!state.agenticChats[k]) state.agenticChats[k] = { messages: [], loading: false, loaded: true, input: "", error: null };
        state.agenticChats[k].input = e.target.value;
      };
      input.onkeydown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendEvalChatMessage(input.dataset.chatInput);
        }
      };
    });
    detail.querySelectorAll(".eval-chat-msgs-box").forEach(box => {
      box.scrollTop = box.scrollHeight;
    });

    const urlParams = new URLSearchParams(window.location.search);
    const targetKey = urlParams.get("key") || urlParams.get("highlight_key");
    if (targetKey) {
      setTimeout(() => {
        const row = detail.querySelector(`tr[data-key-row="${CSS.escape(targetKey)}"]`);
        if (row) {
          row.scrollIntoView({ behavior: "smooth", block: "center" });
          row.classList.add("row-highlight");
        }
      }, 100);
    }
  }
  const chunkSearchBtn = document.getElementById("chunkSearchBtn");
  if (chunkSearchBtn) {
    const runFilter = async () => {
      const input = document.getElementById("chunkSearch");
      state.chunkQuery = input ? input.value : "";
      const q = encodeURIComponent(state.chunkQuery || "");
      state.chunks = await api(`/api/runs/${encodeURIComponent(state.runId)}/chunks?limit=200&q=${q}`);
      paintDetail();
    };
    chunkSearchBtn.onclick = runFilter;
    const input = document.getElementById("chunkSearch");
    if (input) input.onkeydown = (e) => { if (e.key === "Enter") runFilter(); };
  }
  detail.querySelectorAll(".chunk-open").forEach(btn => {
    btn.onclick = async () => {
      const id = btn.dataset.chunkId;
      const dataRow = btn.closest("tr.chunk-row");
      if (!dataRow) {
        openChunkPreview(id, btn);
        return;
      }
      // Toggle closed if same preview already open under this row.
      const existing = dataRow.nextElementSibling;
      if (existing && existing.classList.contains("chunk-preview-row")
          && existing.dataset.chunkId === id) {
        existing.remove();
        return;
      }
      // Remove any other inline previews.
      detail.querySelectorAll("tr.chunk-preview-row").forEach(r => r.remove());
      const previewRow = document.createElement("tr");
      previewRow.className = "chunk-preview-row";
      previewRow.dataset.chunkId = id;
      previewRow.innerHTML = `<td colspan="6"><div class="empty">Loading ${esc(id)}…</div></td>`;
      dataRow.after(previewRow);
      try {
        const [row, hl] = await Promise.all([
          api(`/api/runs/${encodeURIComponent(state.runId)}/chunks/${encodeURIComponent(id)}`),
          api(`/api/runs/${encodeURIComponent(state.runId)}/chunks/${encodeURIComponent(id)}/highlights`).catch(() => null),
        ]);
        const cell = previewRow.querySelector("td");
        cell.style.cssText = "padding:8px 10px;background:#121820";
        renderChunkCard(cell, row, hl, state.runId);
      } catch (err) {
        previewRow.innerHTML = `<td colspan="6"><div class="empty" style="color:var(--err)">${esc(err.message || err)}</div></td>`;
      }
    };
  });
  bindChunkJumpButtons(detail);
  bindPageJumpButtons(detail);
  const openJsonDump = async (relPath) => {
    const data = await api(runFileUrl(relPath));
    const w = window.open("", "_blank");
    if (!w) return;
    w.document.write(`<pre style="white-space:pre-wrap;font-family:ui-monospace,monospace">${esc(pretty(data, 50000))}</pre>`);
    w.document.close();
  };
  detail.querySelectorAll("[data-step]").forEach(a => {
    a.onclick = async (e) => {
      e.preventDefault();
      openJsonDump(`03_agent/${a.dataset.step}`);
    };
  });
  detail.querySelectorAll("[data-tool-file]").forEach(a => {
    a.onclick = async (e) => {
      e.preventDefault();
      openJsonDump(`03_agent/tools/${a.dataset.toolFile}`);
    };
  });
  detail.querySelectorAll("[data-md]").forEach(async pre => {
    const path = pre.getAttribute("data-md");
    try {
      const r = await fetch(runFileUrl(path));
      const html = await r.text();
      const tmp = document.createElement("div");
      tmp.innerHTML = html;
      pre.textContent = tmp.textContent || html.slice(0, 4000);
    } catch (err) {
      pre.textContent = String(err);
    }
  });
}

(async function init() {
  const params = new URLSearchParams(location.search);
  state.embed = params.get("embed") === "1";
  const runParam = params.get("run");
  const tabParam = params.get("tab");
  const evalKeyParam = params.get("eval_key") || null;

  if (state.embed) {
    document.getElementById("appBody").classList.add("embed");
    state.tab = evalKeyParam ? "hierarchy_eval" : "hierarchy_kv";
    state.evalKey = evalKeyParam;
    if (runParam) {
      const detail = document.getElementById("detail");
      if (detail) detail.innerHTML = `<div class="empty">Loading ${esc(runParam)}…</div>`;
    }
  } else if (evalKeyParam) {
    state.evalKey = evalKeyParam;
    state.tab = "hierarchy_eval";
  } else if (tabParam === "hierarchy_eval") {
    state.tab = "hierarchy_eval";
  } else if (tabParam === "hierarchy" || tabParam === "hierarchy_kv") {
    state.tab = "hierarchy_kv";
  } else if (tabParam) {
    state.tab = tabParam;
  }

  state.runs = await api("/api/runs");
  if (!state.embed) {
    await refreshDatasets();
    const dsId = params.get("dataset");
    const dsSource = params.get("source");
    if (dsId) {
      state.inferSource = "dataset";
      const source = dsSource
        || (state.datasets.find(d => d.id === dsId) || {}).source
        || "folder";
      state.inferDataset = `${source}/${dsId}`;
    }
    renderRuns();
    await resumeInferenceUploadJob();
    const uploadBtn = document.getElementById("inferenceUploadBtn");
    if (uploadBtn) uploadBtn.onclick = () => startInferenceUpload();
    const refreshBtn = document.getElementById("inferenceRefreshBtn");
    if (refreshBtn) refreshBtn.onclick = () => refreshInferenceUploadJob();
    document.querySelectorAll('input[name="inferSource"]').forEach(radio => {
      radio.onchange = () => {
        if (radio.checked) {
          state.inferSource = radio.value;
          renderUploadPanel();
        }
      };
    });
  }
  if (runParam && (state.embed || state.runs.some(r => r.run_id === runParam))) {
    await selectRun(runParam, { keepTab: true, keepEvalKey: true });
  } else if (!state.embed && state.runs[0]) {
    await selectRun(state.runs[0].run_id);
  }
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/evaluation", response_class=HTMLResponse)
def evaluation_page() -> str:
    return EVALUATION_HTML


@app.get("/datasets", response_class=HTMLResponse)
def datasets_page() -> str:
    return DATASETS_HTML


@app.get("/ground-truth", response_class=HTMLResponse)
def ground_truth_page() -> str:
    return GROUND_TRUTH_HTML


@app.get("/wrong-cases", response_class=HTMLResponse)
@app.get("/wrong-case", response_class=HTMLResponse)
def wrong_cases_page() -> str:
    return WRONG_CASES_HTML


@app.get("/prompts", response_class=HTMLResponse)
@app.get("/prompt", response_class=HTMLResponse)
def prompts_page() -> str:
    return PROMPTS_HTML


def main() -> None:
    import uvicorn

    host = os.environ.get("TRACE_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("TRACE_VIEWER_PORT", "8099"))
    print(f"Trace viewer on http://{host}:{port}  runs_root={RUNS_ROOT}  datasets={_DATASET_STORE.managed_root}")
    uvicorn.run(
        "agentic_viewer.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
