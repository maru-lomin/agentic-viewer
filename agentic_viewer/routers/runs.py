"""Runs API router for inspecting and managing agentic extraction runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from agentic_viewer.hierarchy import build_agent_tree
from agentic_viewer.highlights import chunk_highlights, page_highlights
from agentic_viewer.image_tokens import replace_base64_images
from agentic_viewer.pdf_source import infer_pdf_path, infer_run_document, pdf_info
from agentic_viewer.run_cache import get_cached_run_row, invalidate_run_cache
from agentic_viewer.templates import get_template
from agentic_viewer.timing import attach_timing_to_tree, build_timing_report

router = APIRouter()

INDEX_HTML = get_template("index.html")


def _get_app():
    from agentic_viewer import app as app_mod
    return app_mod


@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


def _build_run_row(child: Path) -> Dict[str, Any]:
    app_mod = _get_app()
    meta = app_mod._read_json(child / "meta.json") or {}
    result = app_mod._read_json(child / "04_result.json") or {}
    eval_report = app_mod.load_or_compute_run_eval(
        child, run_id=child.name, write_cache=True
    )
    gold_keys: List[str] = []
    if isinstance(eval_report, dict):
        for row in eval_report.get("per_key") or []:
            if isinstance(row, dict) and "key" in row:
                gold_keys.append(str(row["key"]))
    agentic_by_key = app_mod.read_agentic_evals(child)
    status = meta.get("status")
    if not status:
        status = "running" if not meta.get("finished_at") else "unknown"
    return {
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
        "eval_summary": app_mod._eval_summary(eval_report),
        "agentic_eval_summary": app_mod.agentic_eval_summary(
            agentic_by_key, gold_keys
        ),
    }


@router.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    app_mod = _get_app()
    runs_root = app_mod.RUNS_ROOT
    runs_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for child in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        rows.append(get_cached_run_row(child, _build_run_row))
    return app_mod._enrich_run_groups(rows)


@router.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    result = app_mod._read_json(root / "04_result.json")
    return {
        "run_id": run_id,
        "document": infer_run_document(root, result=result),
        "meta": app_mod._read_json(root / "meta.json"),
        "request": app_mod._read_json(root / "00_request.json"),
        "parse_summary": app_mod._read_json(root / "01_parse" / "summary.json"),
        "chunk_summary": app_mod._read_json(root / "02_chunk" / "summary.json"),
        "result": result,
        "error": app_mod._read_json(root / "04_error.json"),
    }


@router.delete("/api/runs/{run_id}")
def delete_run(run_id: str, force: bool = Query(default=False)) -> Dict[str, Any]:
    """Remove a run directory under outputs/runs/."""
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    app_mod._assert_run_deletable(run_id, force=force)
    invalidate_run_cache(root)
    shutil.rmtree(root)
    app_mod._AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
    return {"ok": True, "run_id": run_id}


@router.post("/api/runs/delete-batch")
def delete_runs_batch(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Remove multiple run directories under outputs/runs/."""
    app_mod = _get_app()
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
            root = app_mod._run_dir(rid_str)
            app_mod._assert_run_deletable(rid_str, force=force)
            invalidate_run_cache(root)
            shutil.rmtree(root)
            app_mod._AGENTIC_EVAL_INFLIGHT.pop(rid_str, None)
            deleted.append(rid_str)
        except Exception as exc:
            errors[rid_str] = str(exc)
    return {"deleted": deleted, "errors": errors, "total_deleted": len(deleted)}


@router.get("/api/runs/{run_id}/timeline")
def get_timeline(run_id: str) -> List[Dict[str, Any]]:
    app_mod = _get_app()
    path = app_mod._run_dir(run_id) / "timeline.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@router.get("/api/runs/{run_id}/steps")
def list_steps(run_id: str) -> List[str]:
    app_mod = _get_app()
    agent_dir = app_mod._run_dir(run_id) / "03_agent"
    if not agent_dir.is_dir():
        return []
    return sorted(p.name for p in agent_dir.glob("step_*.json"))


@router.get("/api/runs/{run_id}/steps/detail")
def list_steps_detail(run_id: str) -> List[Dict[str, Any]]:
    """Step dumps for the visualize tab."""
    app_mod = _get_app()
    agent_dir = app_mod._run_dir(run_id) / "03_agent"
    if not agent_dir.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(agent_dir.glob("step_*.json")):
        data = app_mod._read_json(path) or {}
        data.pop("messages_after", None)
        data.pop("messages", None)
        data["filename"] = path.name
        tools = data.get("tools") or []
        data["tool_names"] = [
            (t.get("function") or {}).get("name")
            for t in tools
            if isinstance(t, dict)
        ]
        rows.append(data)
    return rows


@router.get("/api/runs/{run_id}/agent-tree")
def get_agent_tree(run_id: str) -> Dict[str, Any]:
    """Hierarchical Master → search_pages → SearchAgent sessions tree."""
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    tree = build_agent_tree(root)
    timing = build_timing_report(root)
    return attach_timing_to_tree(tree, timing)


@router.get("/api/runs/{run_id}/timing")
def get_timing(run_id: str) -> Dict[str, Any]:
    """Agent / session / turn timing derived from timeline.jsonl."""
    app_mod = _get_app()
    return build_timing_report(app_mod._run_dir(run_id))


@router.get("/api/runs/{run_id}/file")
def get_file(run_id: str, path: str):
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    return app_mod._serve_run_file(root.resolve(), path)


@router.get("/api/runs/{run_id}/pages")
def list_pages(run_id: str) -> Dict[str, Any]:
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    summary = app_mod._read_json(root / "01_parse" / "summary.json") or {}
    progress = app_mod._read_json(root / "01_parse" / "progress.json") or {}
    pages = list(summary.get("pages") or [])
    if not pages:
        parse_dir = root / "01_parse"
        if parse_dir.is_dir():
            for meta_path in sorted(parse_dir.glob("page_*.meta.json")):
                meta = app_mod._read_json(meta_path)
                if isinstance(meta, dict) and meta.get("page") is not None:
                    pages.append(meta)
    return {
        "pages": pages,
        "page_count": summary.get("page_count") or len(pages),
        "progress": progress or None,
        "seconds": summary.get("seconds"),
    }


@router.get("/api/runs/{run_id}/chunks")
def list_chunks(
    run_id: str, offset: int = 0, limit: int = 200, q: str = ""
) -> Dict[str, Any]:
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    summary = app_mod._read_json(root / "02_chunk" / "summary.json") or {}
    progress = app_mod._read_json(root / "02_chunk" / "progress.json") or {}
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


@router.get("/api/runs/{run_id}/chunks/{chunk_id}")
def get_chunk(run_id: str, chunk_id: str) -> Dict[str, Any]:
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
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


@router.get("/api/runs/{run_id}/chunks/{chunk_id}/highlights")
def get_chunk_highlights(run_id: str, chunk_id: str) -> Dict[str, Any]:
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    try:
        return chunk_highlights(root, chunk_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/runs/{run_id}/pages/{page_no}/highlights")
def get_page_highlights(
    run_id: str, page_no: int, q: Optional[str] = None
) -> Dict[str, Any]:
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    try:
        return page_highlights(root, page_no, query=q)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/runs/{run_id}/pdf/info")
@router.get("/api/runs/{run_id}/pdf-info")
def get_pdf_info(run_id: str) -> Dict[str, Any]:
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    return pdf_info(root)


@router.get("/api/runs/{run_id}/pdf")
def get_pdf(run_id: str):
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
    path = infer_pdf_path(root)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="PDF not found for this run")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/api/runs/{run_id}/conversation")
def get_conversation(run_id: str) -> List[Dict[str, Any]]:
    """Chat-style message transcript (preferred) or reconstructed from step dumps."""
    app_mod = _get_app()
    root = app_mod._run_dir(run_id)
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
    last = app_mod._read_json(steps[-1]) or {}
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
