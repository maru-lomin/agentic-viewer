"""Ground Truth API router and management page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from agentic_viewer.eval.paths import answer_sheet_path
from agentic_viewer.ground_truth import (
    get_document_gt,
    import_answer_sheet,
    invalidate_eval_caches_for_document,
    list_documents,
    update_gt_key,
)
from agentic_viewer.ground_truth_page import GROUND_TRUTH_HTML

router = APIRouter()


def _get_runs_root() -> Path:
    from agentic_viewer import app as app_mod
    return getattr(app_mod, "RUNS_ROOT", Path("outputs/runs"))


@router.get("/ground-truth", response_class=HTMLResponse)
def ground_truth_page() -> str:
    return GROUND_TRUTH_HTML


@router.get("/api/ground-truth")
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


@router.get("/api/ground-truth/document")
def get_ground_truth_document(document: str) -> Dict[str, Any]:
    try:
        return get_document_gt(document)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/api/ground-truth/key")
@router.post("/api/ground-truth/key")
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
    invalidated = invalidate_eval_caches_for_document(_get_runs_root(), document)
    return {**result, "invalidated_eval_caches": invalidated}


@router.post("/api/ground-truth/upload")
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
    runs_root = _get_runs_root()
    for document in result.get("documents") or []:
        invalidated += invalidate_eval_caches_for_document(runs_root, str(document))
    return {
        **result,
        "filename": name,
        "invalidated_eval_caches": invalidated,
        "documents_index": list_documents(),
    }
