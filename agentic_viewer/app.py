"""Lightweight FastAPI viewer for agentic run traces under outputs/runs/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from agentic_viewer.eval.paths import answer_sheet_path
from agentic_viewer.evaluation.agentic_client import AgenticEvalError, invoke_agentic_eval
from agentic_viewer.evaluation.batch import enrich_batch_job_dict, make_batch_manager
from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
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
    invalidate_eval_caches_for_document,
    list_documents,
    update_gt_key,
)
from agentic_viewer.ground_truth_page import GROUND_TRUTH_HTML
from agentic_viewer.hierarchy import build_agent_tree
from agentic_viewer.highlights import chunk_highlights
from agentic_viewer.image_tokens import replace_base64_images
from agentic_viewer.pdf_source import infer_pdf_path, infer_run_document, pdf_info
from agentic_viewer.timing import attach_timing_to_tree, build_timing_report

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

app = FastAPI(title="Agentic Run Trace Viewer", version="0.3.0")


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

    ans_path = answer_sheet_path()
    if not ans_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"answer sheet not found: {ans_path}",
        )
    raise HTTPException(status_code=400, detail="could not score run against answer sheet")


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
                "eval_summary": _eval_summary(eval_report),
                "agentic_eval_summary": agentic_eval_summary(
                    agentic_by_key, gold_keys
                ),
            }
        )
    return rows


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
      font-family: var(--sans); min-height: 100vh;
    }
    header {
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
    main { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 58px); }
    aside {
      border-right: 1px solid var(--line); overflow: auto; background: #121820;
    }
    .run {
      padding: 12px 14px; border-bottom: 1px solid var(--line); cursor: pointer;
    }
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
    section { padding: 16px 20px; overflow: auto; }
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
      main { grid-template-columns: 1fr; }
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
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
      margin-bottom: 8px; font-size: 12px; color: var(--muted);
    }
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
      max-width: 320px;
    }
    .agentic-eval-text {
      white-space: pre-wrap; word-break: break-word; font-family: var(--mono);
      font-size: 11px; max-height: 220px; overflow: auto;
      background: #121820; border: 1px solid var(--line); border-radius: 6px; padding: 8px;
      min-width: 180px; max-width: 320px;
    }
    .agentic-eval-detail details summary {
      cursor: pointer; color: var(--accent); font-size: 11px;
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
    body.embed header,
    body.embed aside { display: none; }
    body.embed main { grid-template-columns: 1fr; min-height: 100vh; }
    body.embed section { padding: 12px 14px; }
  </style>
</head>
<body id="appBody">
  <header>
    <h1>Agentic Viewer</h1>
    <nav class="topnav">
      <a href="/" class="active">Inference</a>
      <a href="/evaluation">Evaluation</a>
      <a href="/ground-truth">Ground Truth</a>
    </nav>
    <div class="meta" id="headerMeta">Loading runs…</div>
  </header>
  <main>
    <aside id="runList"></aside>
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
  evalOpenDetails: new Set(),
  gtEdit: null,
  batchJob: null, batchPollTimer: null,
  evalKey: null, embed: false,
  evalHierarchyKeys: [], evalHierarchyKeysLoading: false,
  loadedRunId: null,
  agentTreeCache: { runId: null, kv: null, eval: {} },
  agentTreeLoading: false,
  pagesChunksLoadedFor: null,
};

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

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(await r.text());
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

function renderRuns() {
  if (state.embed) return;
  const el = document.getElementById("runList");
  el.innerHTML = state.runs.map(r => {
    const es = r.eval_summary;
    const evalLine = es
      ? `<div class="eval-mini">EM ${fmtPct(es.value_exact_match)} · pageF1 ${fmtPct(es.page_f1_macro)} · evidF1 ${fmtPct(es.evidence_token_f1)}</div>`
      : "";
    const ae = r.agentic_eval_summary;
    const agenticLine = ae && ae.n_total
      ? `<div class="eval-mini">agentic ${ae.n_done}/${ae.n_total}${ae.accuracy != null ? ` · pred acc ${fmtPct(ae.accuracy)}` : ""}${ae.gold_validity != null ? ` · GT valid ${fmtPct(ae.gold_validity)}` : ""}</div>`
      : "";
    return `
    <div class="run ${r.run_id === state.runId ? "active" : ""}" data-id="${esc(r.run_id)}">
      <div class="id">${runLabelHtml(r)}</div>
      <div class="sub">
        <span class="badge ${r.status === "ok" ? "ok" : (r.status === "error" ? "error" : (r.status === "running" ? "warn" : ""))}">${esc(r.status)}</span>
        ${r.seconds != null ? r.seconds + "s" : ""} · kv=${r.n_kv ?? "?"} · pages=${r.page_count ?? "?"}
      </div>
      ${evalLine}
      ${agenticLine}
    </div>`;
  }).join("") || `<div class="empty" style="padding:16px">No runs in outputs/runs</div>`;
  el.querySelectorAll(".run").forEach(node => {
    node.onclick = () => selectRun(node.dataset.id);
  });
  document.getElementById("headerMeta").textContent =
    `${state.runs.length} run(s) · ${location.origin}`;
}

async function selectRun(runId, opts = {}) {
  const keepTab = Boolean(opts.keepTab);
  const keepEvalKey = Boolean(opts.keepEvalKey);
  state.runId = runId;
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
    state.batchJob = null;
    stopBatchPoll();
  }
  renderRuns();
  await renderDetail();
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

function regionToPctStyle(region) {
  let x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  if (Array.isArray(region.bbox_norm) && region.bbox_norm.length === 4) {
    [x0, y0, x1, y1] = region.bbox_norm.map(Number);
  } else if (Array.isArray(region.bbox) && region.bbox.length === 4) {
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

async function mountChunkPdfViewer(host, runId, regions) {
  if (!host || !runId) return;
  const grouped = {};
  for (const r of (regions || [])) {
    const page = Number(r.page);
    if (!Number.isFinite(page) || page <= 0) continue;
    if (!grouped[page]) grouped[page] = [];
    grouped[page].push(r);
  }
  const pages = Object.keys(grouped).map(Number).sort((a, b) => a - b);
  if (!pages.length) {
    host.innerHTML = `<div class="empty">No highlight regions for PDF overlay.</div>`;
    return;
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

  const shell = document.createElement("div");
  shell.className = "pdf-chunk-viewer";
  shell.innerHTML = `<div class="pdf-toolbar">
    <span>Source: ${esc(info.filename || "document.pdf")}</span>
    <a href="/api/runs/${encodeURIComponent(runId)}/pdf" target="_blank">open full PDF</a>
  </div><div class="pdf-pages"></div>`;
  host.innerHTML = "";
  host.appendChild(shell);
  const pagesHost = shell.querySelector(".pdf-pages");

  try {
    const pdf = await getPdfDocument(runId);
    const scale = 1.35;
    for (const pageNum of pages) {
      const page = await pdf.getPage(pageNum);
      const viewport = page.getViewport({ scale });
      const wrap = document.createElement("div");
      wrap.className = "pdf-page-wrap";

      const label = document.createElement("div");
      label.className = "pdf-page-label";
      label.textContent = `Page ${pageNum}`;

      const canvasWrap = document.createElement("div");
      canvasWrap.className = "pdf-canvas-wrap";
      canvasWrap.style.width = `${Math.round(viewport.width)}px`;
      canvasWrap.style.height = `${Math.round(viewport.height)}px`;

      const canvas = document.createElement("canvas");
      canvas.width = Math.round(viewport.width);
      canvas.height = Math.round(viewport.height);
      const ctx = canvas.getContext("2d");
      await page.render({ canvasContext: ctx, viewport }).promise;

      const overlay = document.createElement("div");
      overlay.className = "pdf-overlay";
      overlay.style.width = `${Math.round(viewport.width)}px`;
      overlay.style.height = `${Math.round(viewport.height)}px`;

      for (const region of grouped[pageNum]) {
        const style = regionToPctStyle(region);
        if (!style) continue;
        const box = document.createElement("div");
        box.className = "hl-box";
        Object.assign(box.style, style);
        overlay.appendChild(box);
      }

      canvasWrap.appendChild(canvas);
      canvasWrap.appendChild(overlay);
      wrap.appendChild(label);
      wrap.appendChild(canvasWrap);
      pagesHost.appendChild(wrap);
    }
    if (!pagesHost.children.length) {
      pagesHost.innerHTML = `<div class="empty">Could not render PDF pages for highlights.</div>`;
    }
  } catch (err) {
    host.innerHTML = `<div class="empty" style="color:var(--err)">${esc(err.message || err)}</div>`;
  }
}

async function openChunkPreview(chunkId, anchorEl) {
  const id = String(chunkId || "").trim();
  if (!id || !state.runId) return;
  const detail = document.getElementById("detail");
  if (!detail) return;
  detail.querySelectorAll(".chunk-preview-inline").forEach(r => r.remove());
  const host = anchorEl && anchorEl.closest
    ? (anchorEl.closest(".ev-block, .tree-body, td, .viz-section") || anchorEl.parentElement)
    : detail;
  const box = document.createElement("div");
  box.className = "chunk-preview-inline";
  box.style.cssText = "margin-top:8px;padding:8px 10px;background:#121820;border:1px solid var(--line);border-radius:8px";
  box.innerHTML = `<div class="empty">Loading ${esc(id)}…</div>`;
  if (host && host.appendChild) host.appendChild(box);
  else detail.appendChild(box);
  try {
    const [row, hl] = await Promise.all([
      api(`/api/runs/${encodeURIComponent(state.runId)}/chunks/${encodeURIComponent(id)}`),
      api(`/api/runs/${encodeURIComponent(state.runId)}/chunks/${encodeURIComponent(id)}/highlights`).catch(() => null),
    ]);
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
            <td>${layout ? `<a href="/api/runs/${encodeURIComponent(state.runId)}/file?path=${encodeURIComponent(layout)}" target="_blank">layout</a>` : "—"}</td>
          </tr>`;
        }).join("")
      : `<tr><td colspan="5" class="empty">No highlight regions (layout missing or no match).</td></tr>`;
    const source = hl && hl.source ? ` · regions=${esc(hl.source)}` : "";
    box.innerHTML = `<details class="tree-node" open style="margin:0">
      <summary><span class="title">${esc(row.chunk_id)}</span>
        <span class="tree-badge">${esc((row.pages || [row.page]).join(","))}</span>
        <span class="tree-kv">${esc(row.heading_path || "")}${source}</span>
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
    const pdfHost = box.querySelector(".pdf-chunk-host");
    if (pdfHost && regions.length) {
      mountChunkPdfViewer(pdfHost, state.runId, regions);
    }
  } catch (err) {
    box.innerHTML = `<div class="empty" style="color:var(--err)">${esc(err.message || err)}</div>`;
  }
}

function bindChunkJumpButtons(root) {
  (root || document).querySelectorAll(".chunk-jump").forEach(btn => {
    btn.onclick = () => openChunkPreview(btn.dataset.chunkId, btn);
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
        `<div class="tree-kv">Search turn dumps not linked (legacy run).</div>`}
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
      <td>${esc(ex.evidence_quote || "")}</td>
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
      <thead><tr><th>Key</th><th>Value</th><th>Evidence</th></tr></thead>
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
      const text = ev.text || ev.evidence_quote || "";
      return page ? `[${page}] ${text}` : text;
    }).filter(Boolean).join(" · ") || (item.evidence_quote || "");
    const reasons = item.search_reasons || item.page_reasons || {};
    const reasonText = (reasons && typeof reasons === "object")
      ? Object.entries(reasons).map(([p, t]) => `p${p}: ${t}`).join(" · ")
      : "";
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
      if (el.open) state.evalOpenDetails.add(id);
      else state.evalOpenDetails.delete(id);
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
        <h3 id="gtModalTitle">Edit ground truth</h3>
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
  const document = state.evalReport?.document;
  if (!document) {
    alert("Document name is not available for this run.");
    return;
  }
  state.gtEdit = {
    document,
    key,
    value: "",
    evidencesText: "",
    pagesText: "",
    loading: true,
    saving: false,
    message: null,
    messageKind: null,
    highlightInvalid: Boolean(opts.highlightInvalid),
  };
  paintDetail();
  try {
    const data = await api(`/api/ground-truth/document?document=${encodeURIComponent(document)}`);
    const entry = (data.keys || []).find(row => row.key === key);
    if (!entry) throw new Error(`GT key not found: ${key}`);
    state.gtEdit = {
      ...state.gtEdit,
      loading: false,
      value: entry.value || "",
      evidencesText: (entry.evidences || []).join("\n"),
      pagesText: (entry.evidence_pages || []).join(", "),
    };
  } catch (err) {
    state.gtEdit = {
      ...state.gtEdit,
      loading: false,
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
    const result = await apiPost("/api/ground-truth/key", {
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
  const o = report.overall || {};
  const cards = [
    ["Value EM", o.value_exact_match, `${report.n_keys ?? "—"} keys`],
    ["Page F1 (macro)", o.page_f1_macro, `P ${fmtPct(o.page_precision_macro)} / R ${fmtPct(o.page_recall_macro)}`],
    ["Page F1 (micro)", o.page_f1_micro, `P ${fmtPct(o.page_precision_micro)} / R ${fmtPct(o.page_recall_micro)}`],
    ["Evidence token F1", o.evidence_token_f1, report.document || ""],
  ].map(([label, val, sub]) => `
    <div class="score-card">
      <div class="label">${esc(label)}</div>
      <div class="value">${fmtPct(val)}</div>
      <div class="sub">${esc(sub)}</div>
    </div>`).join("");

  const batch = state.batchJob;
  const batchActive = batch && (batch.status === "queued" || batch.status === "running");
  const batchForRun = batch && batch.run_ids && batch.run_ids.includes(state.runId);

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
    const ae = (state.agenticEvals || {})[row.key];
    const inflightKeys = Array.isArray(state.agenticEvalInflight)
      ? state.agenticEvalInflight
      : (state.agenticEvalInflight ? [state.agenticEvalInflight] : []);
    const keyInflight = inflightKeys.includes(row.key);
    const batchActiveForKey = batchActive && batch && batch.active
      && batch.active.some(x => x.key === row.key);
    const goldVerdictForBtn = String((ae && ae.is_valid_gold) || "").toLowerCase();
    const gtEditCls = goldVerdictForBtn === "invalid" ? " warn" : "";
    const gtEditBtn = `<button type="button" class="gt-edit-btn${gtEditCls}" data-gt-edit="${esc(row.key)}">
      Edit GT</button>`;
    let agenticCell;
    if (ae && ae.status === "done" && (ae.is_correct_answer || ae.is_valid_gold || ae.reason_summary || ae.reason || ae.text)) {
      const verdict = String(ae.is_correct_answer || "").toLowerCase();
      const goldVerdict = String(ae.is_valid_gold || "").toLowerCase();
      const verdictCls = verdict === "correct" ? "correct" : (verdict === "incorrect" ? "incorrect" : "");
      const goldCls = goldVerdict === "valid" ? "valid" : (goldVerdict === "invalid" ? "invalid" : "");
      const summary = ae.reason_summary || ae.reason || "";
      const detail = ae.reason_detail || ae.text || "";
      agenticCell = `
        <div class="agentic-eval-verdicts">
          ${verdict === "correct" || verdict === "incorrect"
            ? `<div class="agentic-eval-verdict ${verdictCls}">pred: ${esc(verdict)}</div>` : ""}
          ${goldVerdict === "valid" || goldVerdict === "invalid"
            ? `<div class="agentic-eval-verdict ${goldCls}">GT: ${esc(goldVerdict)}</div>` : ""}
        </div>
        ${summary ? `<div class="agentic-eval-summary">${esc(summary)}</div>` : ""}
        ${detail ? `<div class="agentic-eval-detail"><details${evalDetailAttrs("agentic", row.key)}>
          <summary>상세</summary>
          <div class="agentic-eval-text">${esc(detail)}</div>
        </details></div>` : ""}`;
    } else if (ae && ae.status === "error") {
      agenticCell = `<div class="agentic-eval-err">${esc(ae.error || "error")}</div>
        <button type="button" class="agentic-eval-btn" data-agentic-key="${esc(row.key)}"
          ${batchActive ? "disabled" : ""}>Retry</button>`;
    } else if (keyInflight || batchActiveForKey || (ae && ae.status === "running")) {
      agenticCell = `<button type="button" class="agentic-eval-btn" disabled>Running…</button>`;
    } else {
      agenticCell = `<button type="button" class="agentic-eval-btn" data-agentic-key="${esc(row.key)}"
        ${batchActive ? "disabled" : ""}>agentic-evaluation</button>`;
    }
    return `<tr>
      <td class="key">${esc(row.key)}</td>
      <td class="${em ? "em-y" : "em-n"}">${em ? "Y" : "N"}</td>
      <td>${fmtPct(sp.f1)}<div class="sub">pred [${esc((sp.pred||[]).join(", "))}] · gold [${esc((sp.gold||[]).join(", "))}]</div></td>
      <td>${fmtPct(et.token_f1)}</td>
      <td>
        <div><b>pred</b> ${esc(row.value?.pred ?? "")}</div>
        <div><b>gold</b> ${esc(row.value?.gold ?? "")}</div>
        <details${evalDetailAttrs("evidence", row.key)}>
          <summary>VLM evidence · Search reasons</summary>
          <div class="ev-block">
            <span class="ev-label vlm">VLM evidence_quote</span>
            <div class="ev-text">${esc(et.pred || "(empty)")}</div>
          </div>
          <div class="ev-block">
            <span class="ev-label search">SearchAgent page_reasons</span>
            <div class="ev-text">${esc(sr.pred || "(empty)")}</div>
          </div>
          ${chunkJumpRows ? `<div class="ev-block">
            <span class="ev-label search">SearchAgent chunks</span>
            <div class="ev-text">${chunkJumpRows}</div>
          </div>` : ""}
          <div class="ev-block">
            <span class="ev-label gold">gold evidences</span>
            <div class="ev-text">${esc(et.gold || "(empty)")}</div>
          </div>
        </details>
        ${gtEditBtn}
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

  return `
    <p class="hint">
      Baseline metrics vs <code>dataset/answer_sheet.json</code>.
      Cached as <code>05_eval.json</code> in the run directory.
      Evid F1 uses <b>VLM evidence_quote</b> only; SearchAgent <b>page_reasons</b> are shown separately.
      Agentic-evaluation runs up to 8 keys in parallel via the inference API and saves under <code>06_agentic_eval/</code>.
      Use <b>Edit GT</b> when agentic eval marks gold as invalid.
      <button class="tab" id="evalRefresh" style="margin-left:8px">Recompute</button>
      <button type="button" class="agentic-eval-btn" id="evalAllKeys"
        style="margin-left:8px" ${allKeysDisabled ? "disabled" : ""}>Evaluate all keys</button>
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
  if (!refresh && state.evalReport && !state.evalError) {
    await ensureAgenticEvals();
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
    await ensureAgenticEvals();
  } catch (err) {
    state.evalReport = null;
    state.evalError = String(err.message || err);
  } finally {
    state.evalLoading = false;
    paintDetail();
  }
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
        const row = await api(`/api/runs/${encodeURIComponent(state.runId)}/chunks/${encodeURIComponent(id)}`);
        previewRow.innerHTML = `<td colspan="6" style="padding:8px 10px;background:#121820">
          <details class="tree-node" open style="margin:0">
            <summary><span class="title">${esc(row.chunk_id)}</span>
              <span class="tree-badge">${esc((row.pages || [row.page]).join(","))}</span>
              <span class="tree-kv">${esc(row.heading_path || "")}</span>
            </summary>
            <div class="tree-body"><pre class="pretty">${esc(row.text || "")}</pre></div>
          </details>
        </td>`;
      } catch (err) {
        previewRow.innerHTML = `<td colspan="6"><div class="empty" style="color:var(--err)">${esc(err.message || err)}</div></td>`;
      }
    };
  });
  bindChunkJumpButtons(detail);
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
  if (!state.embed) renderRuns();
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


@app.get("/ground-truth", response_class=HTMLResponse)
def ground_truth_page() -> str:
    return GROUND_TRUTH_HTML


def main() -> None:
    import uvicorn

    host = os.environ.get("TRACE_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("TRACE_VIEWER_PORT", "8099"))
    print(f"Trace viewer on http://{host}:{port}  runs_root={RUNS_ROOT}")
    uvicorn.run(
        "agentic_viewer.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
