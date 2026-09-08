"""Wrong cases API router and viewer page."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse

from agentic_viewer.wrong_cases import (
    add_or_update_wrong_case,
    batch_add_wrong_cases,
    delete_wrong_case,
    get_wrong_case_detail,
    list_wrong_cases,
    update_wrong_case_status,
)
from agentic_viewer.wrong_cases_page import WRONG_CASES_HTML

router = APIRouter()


def _get_runs_root() -> Path:
    from agentic_viewer import app as app_mod
    return getattr(app_mod, "RUNS_ROOT", Path("outputs/runs"))


@router.get("/wrong-cases", response_class=HTMLResponse)
@router.get("/wrong-case", response_class=HTMLResponse)
def wrong_cases_page() -> str:
    return WRONG_CASES_HTML


@router.get("/api/wrong-cases")
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


@router.get("/api/wrong-cases/{case_id}")
def api_get_wrong_case_detail(case_id: str) -> Dict[str, Any]:
    detail = get_wrong_case_detail(_get_runs_root(), case_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"wrong case not found: {case_id}")
    return detail


@router.post("/api/wrong-cases")
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
            _get_runs_root(),
            run_id,
            key,
            note=note,
            status=status,
            tags=tags,
            user_snapshot=user_snapshot,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/wrong-cases/batch")
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
            _get_runs_root(),
            run_id,
            keys,
            note=note,
            status=status,
        )
        return {"cases": added, "count": len(added)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/api/wrong-cases/{case_id}")
@router.put("/api/wrong-cases/{case_id}")
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


@router.delete("/api/wrong-cases/{case_id}")
def api_delete_wrong_case(case_id: str) -> Dict[str, Any]:
    deleted = delete_wrong_case(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"wrong case not found: {case_id}")
    return {"ok": True, "id": case_id}
