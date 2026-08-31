"""Lightweight FastAPI viewer for agentic run traces under outputs/runs/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from agentic_viewer.eval.evaluate_kv import build_report, load_json
from agentic_viewer.eval.paths import answer_sheet_path
from agentic_viewer.hierarchy import build_agent_tree
from agentic_viewer.image_tokens import replace_base64_images
from agentic_viewer.timing import attach_timing_to_tree, build_timing_report

def default_runs_root() -> Path:
    """Prefer sibling inference-pipeline outputs, else ./runs."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "inference-pipeline" / "outputs" / "runs",
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

app = FastAPI(title="Agentic Run Trace Viewer", version="0.3.0")


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
    cache_path = root / "05_eval.json"
    if cache_path.is_file() and not refresh:
        cached = _read_json(cache_path)
        if isinstance(cached, dict) and cached.get("overall"):
            return cached

    pred_path = root / "04_result.json"
    pred = _read_json(pred_path)
    if not isinstance(pred, dict):
        raise HTTPException(status_code=404, detail="04_result.json not found")

    ans_path = answer_sheet_path()
    if not ans_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"answer sheet not found: {ans_path}",
        )
    answer_sheet = load_json(ans_path)
    if not isinstance(answer_sheet, dict):
        raise HTTPException(status_code=500, detail="invalid answer sheet JSON")

    try:
        report = build_report(
            pred,
            answer_sheet,
            pred_path=str(pred_path),
            answer_sheet_path=str(ans_path),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    report["run_id"] = run_id
    try:
        cache_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Runs dir may be read-only; still return the scored report.
        report["cache_write_error"] = str(cache_path)
    return report


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for child in sorted(RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        meta = _read_json(child / "meta.json") or {}
        result = _read_json(child / "04_result.json") or {}
        eval_cached = _read_json(child / "05_eval.json")
        status = meta.get("status")
        if not status:
            status = "running" if not meta.get("finished_at") else "unknown"
        rows.append(
            {
                "run_id": child.name,
                "status": status,
                "started_at": meta.get("started_at"),
                "finished_at": meta.get("finished_at"),
                "seconds": meta.get("seconds"),
                "n_kv": len(result.get("kv_results") or []),
                "page_count": (result.get("meta") or {}).get("page_count"),
                "eval_summary": _eval_summary(
                    eval_cached if isinstance(eval_cached, dict) else None
                ),
            }
        )
    return rows


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    return {
        "run_id": run_id,
        "meta": _read_json(root / "meta.json"),
        "request": _read_json(root / "00_request.json"),
        "parse_summary": _read_json(root / "01_parse" / "summary.json"),
        "chunk_summary": _read_json(root / "02_chunk" / "summary.json"),
        "result": _read_json(root / "04_result.json"),
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


@app.get("/api/runs/{run_id}/timing")
def get_timing(run_id: str) -> Dict[str, Any]:
    """Agent / session / turn timing derived from timeline.jsonl."""
    return build_timing_report(_run_dir(run_id))


@app.get("/api/runs/{run_id}/eval")
def get_eval(run_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Score 04_result.json against dataset/answer_sheet.json; cache as 05_eval.json."""
    return _compute_run_eval(run_id, refresh=refresh)


@app.get("/api/runs/{run_id}/file")
def get_file(run_id: str, path: str):
    root = _run_dir(run_id)
    rel = path.lstrip("/")
    target = (root / rel).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
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
  <title>Agentic Run Trace</title>
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
      padding: 16px 20px; border-bottom: 1px solid var(--line);
      display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap;
    }
    header h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.02em; }
    header .meta { color: var(--muted); font-size: 13px; }
    main { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 58px); }
    aside {
      border-right: 1px solid var(--line); overflow: auto; background: #121820;
    }
    .run {
      padding: 12px 14px; border-bottom: 1px solid var(--line); cursor: pointer;
    }
    .run:hover, .run.active { background: var(--panel); }
    .run .id { font-family: var(--mono); font-size: 12px; word-break: break-all; }
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
  </style>
</head>
<body>
  <header>
    <h1>Agentic Run Trace</h1>
    <div class="meta" id="headerMeta">Loading runs…</div>
  </header>
  <main>
    <aside id="runList"></aside>
    <section id="detail">
      <div class="empty">Select a run</div>
    </section>
  </main>
<script>
const state = {
  runs: [], runId: null, tab: "hierarchy",
  pages: [], chunks: null, pagesSubtab: "pages",
  agentTree: null, info: null,
  evalReport: null, evalError: null, evalLoading: false,
};

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

function renderRuns() {
  const el = document.getElementById("runList");
  el.innerHTML = state.runs.map(r => {
    const es = r.eval_summary;
    const evalLine = es
      ? `<div class="eval-mini">EM ${fmtPct(es.value_exact_match)} · pageF1 ${fmtPct(es.page_f1_macro)} · evidF1 ${fmtPct(es.evidence_token_f1)}</div>`
      : "";
    return `
    <div class="run ${r.run_id === state.runId ? "active" : ""}" data-id="${esc(r.run_id)}">
      <div class="id">${esc(r.run_id)}</div>
      <div class="sub">
        <span class="badge ${r.status === "ok" ? "ok" : (r.status === "error" ? "error" : (r.status === "running" ? "warn" : ""))}">${esc(r.status)}</span>
        ${r.seconds != null ? r.seconds + "s" : ""} · kv=${r.n_kv ?? "?"} · pages=${r.page_count ?? "?"}
      </div>
      ${evalLine}
    </div>`;
  }).join("") || `<div class="empty" style="padding:16px">No runs in outputs/runs</div>`;
  el.querySelectorAll(".run").forEach(node => {
    node.onclick = () => selectRun(node.dataset.id);
  });
  document.getElementById("headerMeta").textContent =
    `${state.runs.length} run(s) · ${location.origin}`;
}

async function selectRun(runId) {
  state.runId = runId;
  state.tab = "hierarchy";
  state.pagesSubtab = "pages";
  state.agentTree = null;
  state.pages = [];
  state.chunks = null;
  state.evalReport = null;
  state.evalError = null;
  state.evalLoading = false;
  renderRuns();
  await renderDetail();
}

async function renderDetail() {
  const detail = document.getElementById("detail");
  if (!state.runId) {
    detail.innerHTML = `<div class="empty">Select a run</div>`;
    return;
  }
  const keepTab = state.tab;
  detail.innerHTML = `<div class="empty">Loading ${esc(state.runId)}…</div>`;
  const [info, pagesPayload, agentTree, runs, chunksPayload] = await Promise.all([
    api(`/api/runs/${encodeURIComponent(state.runId)}`),
    api(`/api/runs/${encodeURIComponent(state.runId)}/pages`),
    api(`/api/runs/${encodeURIComponent(state.runId)}/agent-tree`),
    api("/api/runs"),
    api(`/api/runs/${encodeURIComponent(state.runId)}/chunks?limit=200`),
  ]);
  state.info = info;
  state.pages = Array.isArray(pagesPayload) ? pagesPayload : (pagesPayload?.pages || []);
  state.pagesMeta = Array.isArray(pagesPayload) ? null : pagesPayload;
  state.chunks = chunksPayload;
  state.agentTree = agentTree;
  state.runs = runs;
  state.tab = keepTab;
  renderRuns();
  paintDetail();
}

function tabsHtml() {
  const tabs = [
    ["hierarchy", "Agent hierarchy"],
    ["timing", "Timing"],
    ["pages", "Pages / Chunks"],
    ["eval", "Eval"],
  ];
  return `<div class="tabs">${tabs.map(([id, label]) =>
    `<button class="tab ${state.tab===id?"active":""}" data-tab="${id}">${label}</button>`
  ).join("")}</div>`;
}

function fmtSec(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return n >= 100 ? `${n.toFixed(0)}s` : `${n.toFixed(1)}s`;
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
  const hasIn = session.prior_context_in != null;
  // Single-session searches already expose progress via turns; hide redundant
  // prior_context_out snapshot. Keep it only when a later session may consume it.
  const nSessions = opts.nSessions || 1;
  const isHandoff = String(status).startsWith("handoff");
  const showPriorOut = nSessions > 1 || isHandoff;
  const showHandoffSummary = showPriorOut && !!session.handoff_summary;
  const showPriorOutBox = showPriorOut && session.prior_context_out != null;
  const hasOut = showHandoffSummary || showPriorOutBox;
  return `<details class="tree-node session">
    <summary>
      <span class="title">Search session ${esc(session.session_index)}</span>
      ${session.key ? `<span class="tree-kv">${esc(session.key)}</span>` : ""}
      ${status ? `<span class="tree-badge ${statusCls}">${esc(status)}</span>` : ""}
      <span class="tree-badge">${esc(turns.length)} turn(s)</span>
      ${timingBadge(session.timing)}
      ${pages.length ? `<span class="tree-badge ok">pages=${esc(pages.join(","))}</span>` : ""}
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

function renderPageReasonsTable(reasons, title = "page_reasons") {
  if (!reasons || typeof reasons !== "object") return "";
  const entries = Object.entries(reasons);
  if (!entries.length) return "";
  const reasonRows = entries.map(([page, reason]) =>
    `<tr><td>${esc(page)}</td><td>${esc(reason)}</td></tr>`
  ).join("");
  return `<div class="viz-section" style="margin-top:6px">
    <h3 style="margin:0 0 4px">${esc(title)}</h3>
    <table class="kv-table">
      <thead><tr><th>Page</th><th>Reason</th></tr></thead>
      <tbody>${reasonRows}</tbody>
    </table>
  </div>`;
}

function renderSearchOutput(output) {
  if (!output) return "";
  const pages = output.pages || [];
  const reasons = output.page_reasons || {};
  return `<div class="viz-section" style="margin:8px 0">
    <h3 style="margin:0 0 6px">SearchAgent output</h3>
    <div class="tree-kv">status=${esc(output.status || "?")} · pages=${pages.length ? esc(pages.join(", ")) : "∅"}</div>
    ${output.reason ? `<div class="tree-kv">reason=${esc(output.reason)}</div>` : ""}
    ${renderPageReasonsTable(reasons) || (pages.length ? `<pre class="pretty">${esc(pretty({pages}, 1200))}</pre>` : `<div class="tree-kv">No pages returned.</div>`)}
  </div>`;
}

function renderSearchAgent(node) {
  const res = node.result || {};
  const output = node.output || {
    pages: res.pages || [],
    page_reasons: res.page_reasons || res.reasons || {},
    status: res.status,
  };
  const pages = output.pages || res.pages || [];
  const status = output.status || res.status || (pages.length ? "complete" : "unknown");
  const statusCls = status === "complete" ? "ok"
    : (status === "not_found" || String(status).startsWith("handoff") ? "warn" : "");
  const sessions = node.sessions || [];
  return `<details class="tree-node search">
    <summary>
      <span class="title">SearchAgent</span>
      <span class="tree-badge ${statusCls}">${esc(status)}</span>
      <span class="tree-kv">${esc(node.key)}</span>
      ${timingBadge(node.timing, "warn")}
      ${pages.length ? `<span class="tree-badge ok">pages=${esc(pages.join(","))}</span>` : `<span class="tree-badge warn">pages=∅</span>`}
    </summary>
    <div class="tree-body">
      ${renderSearchOutput(output)}
      <div class="tree-kv">
        ${res.n_search_sessions ? `sessions=${esc(res.n_search_sessions)} · steps=${esc(res.n_search_steps)}` : ""}
      </div>
      ${node.note ? `<div class="tree-kv">${esc(node.note)}</div>` : ""}
      <p class="hint" style="margin:4px 0 8px">
        Turn-by-turn tool calls below. Final <code>submit_pages</code> /
        <code>no_relevant_pages</code> output is shown on the last turn.
      </p>
      ${sessions.length ? sessions.map(s => renderSearchSession(s, { nSessions: sessions.length })).join("") :
        `<div class="tree-kv">Search turn dumps not linked (legacy run).</div>`}
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
    `<a href="/api/runs/${encodeURIComponent(state.runId)}/file?path=${encodeURIComponent(rel)}" target="_blank">${esc(label)}</a>`
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
      ${result.input_tokens != null || result.output_tokens != null
        ? ` · VLM in=${esc(result.input_tokens ?? "—")} out=${esc(result.output_tokens ?? "—")}` : ""}
      ${covered === true ? `<span class="tree-badge ok">all_keys_covered</span>` : ""}
      ${covered === false ? `<span class="tree-badge warn">partial</span>` : ""}
    </div>
    ${renderPageReasonsTable(pageReasons)}
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

function renderGenericToolResult(tool) {
  const args = tool.arguments || {};
  const result = tool.result != null ? tool.result : tool.result_preview;
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
    inner += `<div class="tree-kv">policy=${esc((tool.arguments || {}).policy || "?")}${n != null ? ` · n_keys=${esc(n)}` : ""}</div>`;
    inner += (tool.children || []).map(renderSearchAgent).join("");
  } else if (tool.name === "extract_kv_vlm") {
    inner += renderExtractKvVlm(tool);
  } else if (tool.name === "load_kv_schema") {
    inner += renderLoadKvSchema(tool);
  } else {
    inner += renderGenericToolResult(tool);
  }
  return `<div class="tree-tool">${inner}</div>`;
}

function renderTiming() {
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
          · ${esc(sc.key)}
        </div>`).join("")}
    </div>` : "";

  let html = `<div class="timing-panel">
    <p class="hint">
      Wall time from <code>timeline.jsonl</code>.
      Master turn wall includes nested SearchAgent work started on that step
      (async search is not just request→next-request).
      SearchAgent calls are collapsed by default — expand a key to see turns.
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
        const title = running
          ? `<span class="tree-badge warn">running</span> session ${esc(sc.current_session ?? "?")} · turn ${esc(sc.current_turn || "?")} · ${esc(sc.key)}`
          : esc(sc.key);
        const turnRows = (sc.sessions || []).map(sess => (sess.turns || []).map(t => `
          <tr class="${t.status === "running" ? "running" : ""}">
            <td style="padding-left:12px">session ${esc(sess.session_index)} · turn ${esc(t.search_turn || t.step)}${t.status === "running" ? ` <span class="tree-badge warn">now</span>` : ""}</td>
            <td>${fmtSec(t.llm_seconds)}</td>
            <td>${esc(tokenLabel(t))}</td>
          </tr>`).join("")).join("");
        return `<details class="tree-node search" style="margin:0">
          <summary>
            <span class="title">m${esc(sc.master_step)}</span>
            <span class="tree-kv" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis">${title}</span>
            <span class="tree-badge ${running ? "warn" : ""}">${fmtSec(sc.wall_seconds)}</span>
            <span class="tree-badge">model ${fmtSec(sc.llm_seconds)}</span>
            <span class="tree-badge">${esc(sc.n_turns || 0)} turns</span>
            ${running ? `<span class="tree-badge warn">${esc(phaseLabel(sc))}</span>` : ""}
          </summary>
          <div class="tree-body">
            <div class="tree-kv">overhead ${running ? "—" : fmtSec(sc.overhead_seconds)} · key=${esc(sc.key)}</div>
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
    const found = item.found;
    const foundBadge = found === true
      ? `<span class="tree-badge ok">found</span>`
      : (found === false ? `<span class="tree-badge warn">not found</span>` : "");
    return `<tr>
      <td>${esc(item.key)}</td>
      <td>${esc(item.value)} ${foundBadge}</td>
      <td>${esc(evidenceText)}</td>
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

function renderAgentHierarchy() {
  const tree = state.agentTree;
  if (!tree || !(tree.master_turns || []).length) {
    return `<div class="empty">No master turns found for this run.</div>`;
  }
  let html = `<p class="hint">
    <b>Agent hierarchy</b> = Master LLM turns → tools → nested SearchAgent sessions.<br/>
    Each <code>search_pages</code> call expands into SearchAgent session(s) with their own turns.
  </p><div class="tree">`;

  for (const mt of tree.master_turns) {
    const err = mt.error ? `<span class="tree-badge err">ERROR</span>` : "";
    const toolNames = (mt.tools || []).map(t => t.name).filter(Boolean);
    html += `<details class="tree-node master">
      <summary>
        <span class="title">Master turn ${esc(mt.step)}</span>
        ${err}
        ${timingBadge(mt.timing, "master")}
        ${tokenBadge(mt, "master")}
        ${toolNames.map(n => `<span class="pill">${esc(n)}</span>`).join("")}
      </summary>
      <div class="tree-body">
        ${(mt.tools || []).map(renderMasterTool).join("") || `<div class="empty">No tools on this turn</div>`}
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

  html += renderMasterOutput();
  html += `</div>`;
  return html;
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

  const rows = (report.per_key || []).map(row => {
    const em = row.value?.exact_match;
    const sp = row.search_pages || {};
    const et = row.evidence_text || {};
    return `<tr>
      <td class="key">${esc(row.key)}</td>
      <td class="${em ? "em-y" : "em-n"}">${em ? "Y" : "N"}</td>
      <td>${fmtPct(sp.f1)}<div class="sub">pred [${esc((sp.pred||[]).join(", "))}] · gold [${esc((sp.gold||[]).join(", "))}]</div></td>
      <td>${fmtPct(et.token_f1)}</td>
      <td>
        <div><b>pred</b> ${esc(row.value?.pred ?? "")}</div>
        <div><b>gold</b> ${esc(row.value?.gold ?? "")}</div>
        <details>
          <summary>evidence text</summary>
          <div class="ev-text"><b>pred</b>\n${esc(et.pred || "(empty)")}\n\n<b>gold</b>\n${esc(et.gold || "(empty)")}</div>
        </details>
      </td>
    </tr>`;
  }).join("");

  return `
    <p class="hint">
      Baseline metrics vs <code>dataset/answer_sheet.json</code>.
      Cached as <code>05_eval.json</code> in the run directory.
      <button class="tab" id="evalRefresh" style="margin-left:8px">Recompute</button>
    </p>
    <div class="score-grid">${cards}</div>
    <table class="eval-table">
      <thead>
        <tr>
          <th>Key</th><th>EM</th><th>Page F1</th><th>Evid F1</th><th>Values / evidence</th>
        </tr>
      </thead>
      <tbody>${rows || `<tr><td colspan="5" class="empty">No keys</td></tr>`}</tbody>
    </table>`;
}

async function ensureEval(refresh=false) {
  if (!state.runId) return;
  if (!refresh && state.evalReport && !state.evalError) return;
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
    state.runs = state.runs.map(r => r.run_id === state.runId ? {...r, eval_summary: es} : r);
    renderRuns();
  } catch (err) {
    state.evalReport = null;
    state.evalError = String(err.message || err);
  } finally {
    state.evalLoading = false;
    paintDetail();
  }
}

function paintDetail() {
  const detail = document.getElementById("detail");
  let body = "";
  if (state.tab === "hierarchy") {
    body = renderAgentHierarchy();
  } else if (state.tab === "timing") {
    body = renderTiming();
  } else if (state.tab === "pages") {
    body = renderPagesChunks();
  } else if (state.tab === "eval") {
    body = renderEval();
  }
  detail.innerHTML = `
    <div class="meta" style="margin-bottom:10px;color:var(--muted);display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <code>${esc(state.runId)}</code>
      · status=${esc(state.info?.meta?.status || (state.info?.meta?.finished_at ? "done" : "running"))}
      · ${esc(state.info?.meta?.seconds)}s
      <button type="button" id="runRefresh" style="margin-left:4px;padding:2px 10px;border-radius:999px;border:1px solid var(--line);background:#152033;color:var(--text);font-size:12px;cursor:pointer">Refresh</button>
    </div>
    ${tabsHtml()}
    ${body}`;
  detail.querySelectorAll(".tab").forEach(btn => {
    btn.onclick = () => {
      if (btn.dataset.pagesSub) {
        state.pagesSubtab = btn.dataset.pagesSub;
        paintDetail();
        return;
      }
      state.tab = btn.dataset.tab;
      paintDetail();
      if (state.tab === "eval") ensureEval(false);
    };
  });
  const runRefresh = document.getElementById("runRefresh");
  if (runRefresh) runRefresh.onclick = () => renderDetail();
  const refreshBtn = document.getElementById("evalRefresh");
  if (refreshBtn) refreshBtn.onclick = () => ensureEval(true);
  if (state.tab === "eval" && !state.evalReport && !state.evalLoading && !state.evalError) {
    ensureEval(false);
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
      if (!dataRow) return;
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
  const openJsonDump = async (relPath) => {
    const data = await api(`/api/runs/${encodeURIComponent(state.runId)}/file?path=${encodeURIComponent(relPath)}`);
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
      const r = await fetch(`/api/runs/${encodeURIComponent(state.runId)}/file?path=${encodeURIComponent(path)}`);
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
  state.runs = await api("/api/runs");
  renderRuns();
  if (state.runs[0]) selectRun(state.runs[0].run_id);
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


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
