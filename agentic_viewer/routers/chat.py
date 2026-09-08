"""Chat API router for EvalMaster, VLM, and SearchAgent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Query

from agentic_viewer.evaluation.agentic_client import (
    AgenticEvalError,
    delete_agentic_eval_chat,
    delete_search_chat,
    delete_vlm_chat,
    get_agentic_eval_chat,
    get_search_chat,
    get_vlm_chat,
    invoke_agentic_eval_chat,
    invoke_search_chat,
    invoke_vlm_chat,
)

router = APIRouter()

INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://127.0.0.1:8010")


def _run_dir(run_id: str) -> Path:
    from agentic_viewer import app as app_mod
    runs_root = getattr(app_mod, "RUNS_ROOT", Path("outputs/runs"))
    p = (runs_root / run_id).resolve()
    if not p.is_dir():
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return p


@router.post("/api/runs/{run_id}/agentic-eval/chat")
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


@router.get("/api/runs/{run_id}/agentic-eval/chat")
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


@router.delete("/api/runs/{run_id}/agentic-eval/chat")
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


@router.post("/api/runs/{run_id}/vlm-chat")
def post_vlm_chat_api(
    run_id: str, body: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Send a follow-up chat message to the VLM for an extracted key."""
    _run_dir(run_id)
    key = str((body or {}).get("key") or "").strip()
    message = str((body or {}).get("message") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        return invoke_vlm_chat(INFERENCE_API_URL, run_id, key, message)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/api/runs/{run_id}/vlm-chat")
def get_vlm_chat_api_endpoint(
    run_id: str,
    key: str = Query(...),
) -> Dict[str, Any]:
    """Retrieve VLM chat history for an extracted key."""
    root = _run_dir(run_id)
    key = str(key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    from agentic_viewer.evaluation.live_progress import _safe_key_filename
    safe = _safe_key_filename(key)
    chat_file = root / "03_agent" / "chats" / f"vlm_{safe}.json"
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
        return get_vlm_chat(INFERENCE_API_URL, run_id, key)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.delete("/api/runs/{run_id}/vlm-chat")
def delete_vlm_chat_api_endpoint(
    run_id: str,
    key: str = Query(...),
) -> Dict[str, Any]:
    """Clear VLM chat history for an extracted key."""
    root = _run_dir(run_id)
    key = str(key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    from agentic_viewer.evaluation.live_progress import _safe_key_filename
    safe = _safe_key_filename(key)
    chat_dir = root / "03_agent" / "chats"
    for fname in (f"vlm_{safe}.json", f"vlm_context_{safe}.json"):
        p = chat_dir / fname
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass

    try:
        return delete_vlm_chat(INFERENCE_API_URL, run_id, key)
    except Exception:
        return {"ok": True, "key": key, "cleared": True}


@router.post("/api/runs/{run_id}/search-chat")
def post_search_chat_api(
    run_id: str, body: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    """Send a follow-up chat message to SearchAgent for an extracted key."""
    _run_dir(run_id)
    key = str((body or {}).get("key") or "").strip()
    message = str((body or {}).get("message") or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        return invoke_search_chat(INFERENCE_API_URL, run_id, key, message)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/api/runs/{run_id}/search-chat")
def get_search_chat_api_endpoint(
    run_id: str,
    key: str = Query(...),
) -> Dict[str, Any]:
    """Retrieve SearchAgent chat history for an extracted key."""
    root = _run_dir(run_id)
    key = str(key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    from agentic_viewer.evaluation.live_progress import _safe_key_filename
    safe = _safe_key_filename(key)
    chat_file = root / "03_agent" / "chats" / f"search_{safe}.json"
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
        return get_search_chat(INFERENCE_API_URL, run_id, key)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.delete("/api/runs/{run_id}/search-chat")
def delete_search_chat_api_endpoint(
    run_id: str,
    key: str = Query(...),
) -> Dict[str, Any]:
    """Clear SearchAgent chat history for an extracted key."""
    root = _run_dir(run_id)
    key = str(key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="key is required")

    from agentic_viewer.evaluation.live_progress import _safe_key_filename
    safe = _safe_key_filename(key)
    chat_dir = root / "03_agent" / "chats"
    for fname in (f"search_{safe}.json", f"search_context_{safe}.json"):
        p = chat_dir / fname
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass

    try:
        return delete_search_chat(INFERENCE_API_URL, run_id, key)
    except Exception:
        return {"ok": True, "key": key, "cleared": True}
