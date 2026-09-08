"""Prompts API router and viewer page."""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse

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

router = APIRouter()


@router.get("/prompts", response_class=HTMLResponse)
@router.get("/prompt", response_class=HTMLResponse)
def prompts_page() -> str:
    return PROMPTS_HTML


@router.get("/api/prompts")
def api_list_prompts() -> Dict[str, Any]:
    return {"prompts": list_prompts()}


@router.get("/api/prompts/{prompt_id}")
def api_get_prompt(prompt_id: str) -> Dict[str, Any]:
    try:
        return get_prompt(prompt_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/api/prompts/{prompt_id}")
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


@router.get("/api/prompts/{prompt_id}/backups")
def api_list_prompt_backups(prompt_id: str) -> Dict[str, Any]:
    try:
        return {"backups": list_backups(prompt_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/prompts/{prompt_id}/backups/{timestamp}")
def api_get_prompt_backup(prompt_id: str, timestamp: str) -> Dict[str, Any]:
    try:
        return get_backup(prompt_id, timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/prompts/{prompt_id}/restore")
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


@router.post("/api/prompts/{prompt_id}/diff")
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
