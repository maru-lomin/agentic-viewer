"""Datasets API router and viewer page."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List

from agentic_viewer.timezone import kst_now

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from agentic_viewer.datasets import DatasetStore
from agentic_viewer.datasets_page import DATASETS_HTML

router = APIRouter()


def _get_dataset_store() -> DatasetStore:
    from agentic_viewer import app as app_mod
    store = getattr(app_mod, "_DATASET_STORE", None)
    if store is None:
        store = DatasetStore()
    return store


def _get_inference_job_manager() -> Any:
    from agentic_viewer import app as app_mod
    return getattr(app_mod, "_INFERENCE_JOB_MANAGER", None)


def _next_version(dataset_id: str) -> int:
    from agentic_viewer import app as app_mod
    fn = getattr(app_mod, "_next_dataset_run_version", None)
    return fn(dataset_id) if fn else 1


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


@router.get("/datasets", response_class=HTMLResponse)
def datasets_page() -> str:
    return DATASETS_HTML


@router.get("/api/datasets")
def list_datasets() -> Dict[str, Any]:
    return {"datasets": _get_dataset_store().list_datasets()}


@router.post("/api/datasets")
async def create_dataset(
    name: str = Form(...),
    files: List[UploadFile] = File(default=[]),
) -> Dict[str, Any]:
    payload = await _read_pdf_uploads(files) if files else []
    try:
        return _get_dataset_store().create(name, payload)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@router.get("/api/datasets/{source}/{dataset_id}")
def get_dataset(source: str, dataset_id: str) -> Dict[str, Any]:
    try:
        return _get_dataset_store().get(source, dataset_id)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@router.post("/api/datasets/{source}/{dataset_id}/files")
async def add_dataset_files(
    source: str,
    dataset_id: str,
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    payload = await _read_pdf_uploads(files)
    try:
        return _get_dataset_store().add_files(source, dataset_id, payload)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@router.delete("/api/datasets/{source}/{dataset_id}/files/{filename}")
def delete_dataset_file(source: str, dataset_id: str, filename: str) -> Dict[str, Any]:
    try:
        return _get_dataset_store().delete_file(source, dataset_id, filename)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@router.delete("/api/datasets/{source}/{dataset_id}")
def delete_dataset(source: str, dataset_id: str) -> Dict[str, Any]:
    try:
        return _get_dataset_store().delete_dataset(source, dataset_id)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise


@router.get("/api/inference/jobs/active")
def get_active_inference_job() -> Dict[str, Any]:
    mgr = _get_inference_job_manager()
    job = mgr.get_active_job() if mgr else None
    return {"job": job.to_dict() if job else None}


@router.get("/api/inference/jobs/{job_id}")
def get_inference_job(job_id: str) -> Dict[str, Any]:
    mgr = _get_inference_job_manager()
    job = mgr.get_job(job_id) if mgr else None
    if not job:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return job.to_dict()


@router.post("/api/inference/jobs")
async def post_inference_job(
    files: List[UploadFile] = File(...),
    hooks: str = Form("agentic_config"),
    auto_eval: str = Form("true"),
) -> Dict[str, Any]:
    """Upload one or more PDFs and run KV extraction via the inference API."""
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
    now = kst_now()
    display_time = now.strftime("%Y-%m-%d %H:%M")
    ts_slug = now.strftime("%Y%m%d-%H%M%S")
    upload_dataset_id = f"upload-{ts_slug}-{uuid.uuid4().hex[:6]}"
    upload_dataset_name = f"Upload ({display_time})"
    store = _get_dataset_store()
    mgr = _get_inference_job_manager()
    try:
        store.create(upload_dataset_id, payload)
        paths = store.pdf_paths("managed", upload_dataset_id)
        job = mgr.start(
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


@router.post("/api/inference/jobs/from-dataset")
def post_inference_job_from_dataset(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Run KV extraction for every PDF in a named dataset."""
    dataset_id = str((body or {}).get("dataset_id") or "").strip()
    source = str((body or {}).get("source") or "").strip()
    if not dataset_id or not source:
        raise HTTPException(status_code=400, detail="dataset_id and source are required")
    store = _get_dataset_store()
    mgr = _get_inference_job_manager()
    try:
        info = store.get(source, dataset_id)
        paths = store.pdf_paths(source, dataset_id)
    except Exception as exc:
        _raise_dataset_error(exc)
        raise
    if not paths:
        raise HTTPException(status_code=400, detail="dataset has no PDF files")
    hooks_name = str((body or {}).get("hooks") or "agentic_config").strip() or "agentic_config"
    auto_eval = bool((body or {}).get("auto_eval", True))
    next_v = _next_version(info["id"])
    now = kst_now()
    display_time = now.strftime("%Y-%m-%d %H:%M")
    ts_slug = now.strftime("%Y%m%d-%H%M%S")
    ds_name = info.get("name") or info["id"]
    run_group_id = f"{info['id']}-run-v{next_v}-{ts_slug}"
    run_group_name = f"{ds_name}-run-v{next_v} ({display_time})"
    try:
        job = mgr.start(
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
    except Exception as exc:
        _raise_dataset_error(exc)
        raise
