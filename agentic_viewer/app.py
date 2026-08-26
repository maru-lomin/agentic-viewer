"""Lightweight FastAPI viewer for agentic run traces under outputs/runs/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from agentic_viewer.image_tokens import replace_base64_images

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

app = FastAPI(title="Agentic Run Trace Viewer", version="0.2.0")


def _run_dir(run_id: str) -> Path:
    path = (RUNS_ROOT / run_id).resolve()
    if not str(path).startswith(str(RUNS_ROOT)) or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return path


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for child in sorted(RUNS_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        meta = _read_json(child / "meta.json") or {}
        result = _read_json(child / "04_result.json") or {}
        rows.append(
            {
                "run_id": child.name,
                "status": meta.get("status", "unknown"),
                "started_at": meta.get("started_at"),
                "finished_at": meta.get("finished_at"),
                "seconds": meta.get("seconds"),
                "n_kv": len(result.get("kv_results") or []),
                "page_count": (result.get("meta") or {}).get("page_count"),
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
        # Soften huge tool schemas for the UI list (full tools still available in file).
        tools = data.get("tools") or []
        data["tool_names"] = [
            (t.get("function") or {}).get("name")
            for t in tools
            if isinstance(t, dict)
        ]
        rows.append(data)
    return rows


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
def list_pages(run_id: str) -> List[Dict[str, Any]]:
    summary = _read_json(_run_dir(run_id) / "01_parse" / "summary.json") or {}
    return summary.get("pages") or []


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
    .kv-table th { width: 140px; color: var(--muted); font-weight: 500; }
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
  runs: [], runId: null, tab: "chat",
  conversation: [], steps: [], stepsDetail: [],
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

function renderRuns() {
  const el = document.getElementById("runList");
  el.innerHTML = state.runs.map(r => `
    <div class="run ${r.run_id === state.runId ? "active" : ""}" data-id="${esc(r.run_id)}">
      <div class="id">${esc(r.run_id)}</div>
      <div class="sub">
        <span class="badge ${r.status === "ok" ? "ok" : (r.status === "error" ? "error" : "")}">${esc(r.status)}</span>
        ${r.seconds != null ? r.seconds + "s" : ""} · kv=${r.n_kv ?? "?"} · pages=${r.page_count ?? "?"}
      </div>
    </div>`).join("") || `<div class="empty" style="padding:16px">No runs in outputs/runs</div>`;
  el.querySelectorAll(".run").forEach(node => {
    node.onclick = () => selectRun(node.dataset.id);
  });
  document.getElementById("headerMeta").textContent =
    `${state.runs.length} run(s) · ${location.origin}`;
}

async function selectRun(runId) {
  state.runId = runId;
  state.tab = "chat";
  state.stepsDetail = [];
  renderRuns();
  await renderDetail();
}

async function renderDetail() {
  const detail = document.getElementById("detail");
  if (!state.runId) {
    detail.innerHTML = `<div class="empty">Select a run</div>`;
    return;
  }
  detail.innerHTML = `<div class="empty">Loading ${esc(state.runId)}…</div>`;
  const [info, timeline, steps, pages, conversation, stepsDetail] = await Promise.all([
    api(`/api/runs/${encodeURIComponent(state.runId)}`),
    api(`/api/runs/${encodeURIComponent(state.runId)}/timeline`),
    api(`/api/runs/${encodeURIComponent(state.runId)}/steps`),
    api(`/api/runs/${encodeURIComponent(state.runId)}/pages`),
    api(`/api/runs/${encodeURIComponent(state.runId)}/conversation`),
    api(`/api/runs/${encodeURIComponent(state.runId)}/steps/detail`),
  ]);
  state.info = info;
  state.timeline = timeline;
  state.steps = steps;
  state.pages = pages;
  state.conversation = conversation;
  state.stepsDetail = stepsDetail;
  paintDetail();
}

function tabsHtml() {
  const tabs = [
    ["chat", "Chat"],
    ["stepsViz", "Agent steps (visualize)"],
    ["timeline", "Timeline"],
    ["turns", "LLM turns (raw)"],
    ["pages", "Pages"],
    ["result", "Result"],
    ["request", "Request"],
  ];
  return `<div class="tabs">${tabs.map(([id, label]) =>
    `<button class="tab ${state.tab===id?"active":""}" data-tab="${id}">${label}</button>`
  ).join("")}</div>`;
}

function renderChat() {
  const msgs = state.conversation || [];
  if (!msgs.length) {
    return `<div class="empty">No conversation yet. Re-run inference after the chat-log update, or open Agent steps.</div>`;
  }
  const reconstructed = msgs.some(m => m.source === "reconstructed");
  let html = `<p class="hint">
    Chat = messages in order (system → user → assistant → tool → …).<br/>
    An <b>LLM turn</b> is one completion call: assistant reply (text and/or tool_calls), then optional tool results go back in.<br/>
    ${reconstructed ? "<i>Reconstructed from last step dump (older run).</i>" : "From <code>03_agent/conversation.jsonl</code>."}
  </p><div class="chat">`;

  let lastTurn = null;
  for (const m of msgs) {
    if (m.turn != null && m.turn !== 0 && m.turn !== lastTurn && m.role === "assistant") {
      html += `<div class="turn-sep">LLM turn ${esc(m.turn)} starts (completion response)</div>`;
      lastTurn = m.turn;
    }
    const role = m.role || "unknown";
    const tMeta = [
      m.t != null ? `t=${m.t}s` : null,
      m.turn != null ? `turn=${m.turn}` : null,
      m.kind && m.kind !== role ? m.kind : null,
      m.name ? `tool=${m.name}` : null,
      m.tool_call_id ? `id=${String(m.tool_call_id).slice(0, 18)}…` : null,
    ].filter(Boolean).join(" · ");

    let body = "";
    if (role === "assistant" && (m.tool_calls || []).length) {
      if (m.content) body += `<div class="body">${esc(pretty(m.content, 800))}</div>`;
      else body += `<div class="body" style="color:var(--muted)">(no text content — tool call only)</div>`;
      for (const tc of m.tool_calls) {
        const fn = tc.function || {};
        body += `<div class="tool-call"><div class="fn">→ call ${esc(fn.name || "?")}</div>
          <div class="body" style="max-height:160px">${esc(pretty(fn.arguments, 800))}</div></div>`;
      }
    } else if (role === "tool") {
      body += `<div class="body">${esc(pretty(m.content, 2000))}</div>`;
    } else {
      body += `<div class="body">${esc(pretty(m.content || "", 2000))}</div>`;
    }

    html += `<div class="bubble ${esc(role)}">
      <div class="head">
        <span class="role">${esc(role)}</span>
        <span class="meta">#${esc(m.i)} ${esc(tMeta)}</span>
      </div>
      ${body}
    </div>`;
  }
  html += `</div>`;
  return html;
}

function renderMsgFlowItem(m, idx) {
  const role = m.role || "unknown";
  let inner = "";
  if (role === "assistant") {
    const tcs = m.tool_calls || [];
    if (m.content) inner += `<pre class="pretty">${esc(pretty(m.content, 1500))}</pre>`;
    else if (!tcs.length) inner += `<pre class="pretty" style="color:var(--muted)">(empty)</pre>`;
    for (const tc of tcs) {
      const fn = tc.function || {};
      inner += `<div class="tool-call"><div class="fn">tool_call → ${esc(fn.name || "?")}</div>
        <pre class="pretty" style="max-height:180px">${esc(pretty(fn.arguments, 1000))}</pre></div>`;
    }
  } else {
    inner = `<pre class="pretty">${esc(pretty(m.content || "", 2000))}</pre>`;
  }
  return `<div class="flow-item ${esc(role)}">
    <div class="label">${esc(role)} · msg[${esc(idx)}]</div>
    ${inner}
  </div>`;
}

function requestMessagesDelta(steps, idx) {
  /** New messages in this turn's request vs previous turn (append-only history). */
  const msgs = steps[idx].request_messages || [];
  if (idx <= 0) {
    return { msgs, omitted: 0, total: msgs.length };
  }
  const prevLen = (steps[idx - 1].request_messages || []).length;
  const omitted = Math.min(prevLen, msgs.length);
  return { msgs: msgs.slice(omitted), omitted, total: msgs.length };
}

function renderStepsViz() {
  const steps = state.stepsDetail || [];
  if (!steps.length) {
    return `<div class="empty">No agent steps found for this run.</div>`;
  }
  let html = `<p class="hint">
    Each card = one LLM completion. Flow: <b>new request msgs → assistant → tool results</b>.
    Prior-turn messages are omitted (history is append-only; see Chat tab for the full transcript).
    Nested JSON strings are unescaped for reading. <code>messages_after</code> is omitted (redundant).
  </p><div class="viz-list">`;

  for (let si = 0; si < steps.length; si++) {
    const step = steps[si];
    const nTools = (step.assistant && step.assistant.tool_calls || []).length;
    const nResults = (step.tool_results || []).length;
    const open = step.step === 1 ? " open" : "";
    const names = (step.tool_names || []).filter(Boolean);
    const delta = requestMessagesDelta(steps, si);
    const reqHeading = delta.omitted
      ? `1. New request messages (+${delta.omitted} prior omitted · ${delta.total} total sent)`
      : "1. Request messages (sent to LLM)";
    html += `<details class="viz-card"${open}>
      <summary>
        <span class="title">Turn ${esc(step.step)}</span>
        <span class="badge">${esc(nTools)} tool_call(s)</span>
        <span class="badge">${esc(nResults)} result(s)</span>
        <span class="meta" style="color:var(--muted);font-family:var(--mono);font-size:11px">
          prompt≈${esc(step.prompt_est_tokens)} · max_tokens=${esc(step.max_tokens)}
          ${step.error ? " · ERROR" : ""}
        </span>
      </summary>
      <div class="viz-body">
        <div class="viz-section">
          <h3>Meta</h3>
          <table class="kv-table">
            <tr><th>tool_choice</th><td>${esc(step.tool_choice)}</td></tr>
            <tr><th>available tools</th><td>${names.map(n => `<span class="pill">${esc(n)}</span>`).join("") || "—"}</td></tr>
            ${step.error ? `<tr><th>error</th><td style="color:var(--err)">${esc(step.error)}</td></tr>` : ""}
          </table>
        </div>

        <div class="viz-section">
          <h3>${esc(reqHeading)}</h3>
          <div class="flow">
            ${delta.msgs.map((m, i) => renderMsgFlowItem(m, delta.omitted + i)).join("") || "<div class='empty'>none</div>"}
          </div>
        </div>

        <div class="arrow">▼ completion</div>

        <div class="viz-section">
          <h3>2. Assistant (LLM output)</h3>
          <div class="flow">
            ${step.assistant ? renderMsgFlowItem(step.assistant, "out") : "<div class='empty'>none</div>"}
          </div>
        </div>

        ${(step.tool_results || []).length ? `
        <div class="arrow">▼ execute tools</div>
        <div class="viz-section">
          <h3>3. Tool results</h3>
          <div class="flow">
            ${(step.tool_results || []).map((tr, i) => `
              <div class="flow-item tool">
                <div class="label">tool · ${esc(tr.name || "?")} · #${esc(i)}</div>
                <table class="kv-table">
                  <tr><th>arguments</th><td><pre class="pretty" style="max-height:140px">${esc(pretty(tr.arguments, 800))}</pre></td></tr>
                  <tr><th>result</th><td><pre class="pretty">${esc(pretty(tr.result_preview != null ? tr.result_preview : tr.result, 2500))}</pre></td></tr>
                </table>
              </div>`).join("")}
          </div>
        </div>` : `
        <div class="viz-section">
          <h3>3. Tool results</h3>
          <div class="meta" style="color:var(--muted);font-size:12px">none (final text response or error)</div>
        </div>`}
      </div>
    </details>`;
  }
  html += `</div>`;
  return html;
}

function paintDetail() {
  const detail = document.getElementById("detail");
  let body = "";
  if (state.tab === "chat") {
    body = renderChat();
  } else if (state.tab === "stepsViz") {
    body = renderStepsViz();
  } else if (state.tab === "timeline") {
    body = (state.timeline || []).map(ev => `
      <div class="event">
        <div class="t">t=${esc(ev.t)}s · ${esc(ev.stage)}/${esc(ev.event)}</div>
        <div class="title">${esc(JSON.stringify(ev, null, 0).slice(0, 240))}</div>
      </div>`).join("") || `<div class="empty">No timeline yet</div>`;
  } else if (state.tab === "turns") {
    body = `<p class="hint">Raw JSON dumps per LLM completion. Prefer <b>Agent steps (visualize)</b>.</p>
      <div id="stepList">${(state.steps || []).map(name =>
      `<div style="margin-bottom:8px"><a href="#" data-step="${esc(name)}">${esc(name)}</a></div>`
    ).join("") || "No turns"}</div><div id="stepView"></div>`;
  } else if (state.tab === "pages") {
    body = `<div class="grid2">${(state.pages || []).map(p => `
      <div>
        <div class="sub" style="margin-bottom:6px;color:var(--muted)">page ${esc(p.page)} · ${esc(p.chars)} chars · ~${esc(p.est_tokens)} tok</div>
        <a href="/api/runs/${encodeURIComponent(state.runId)}/file?path=${encodeURIComponent(p.md_path)}" target="_blank">open md</a>
        <pre class="code" data-md="${esc(p.md_path)}" style="max-height:220px"></pre>
      </div>`).join("") || `<div class="empty">No pages</div>`}</div>`;
  } else if (state.tab === "result") {
    body = `<pre>${esc(JSON.stringify(state.info.result || state.info.error, null, 2))}</pre>`;
  } else if (state.tab === "request") {
    body = `<pre>${esc(JSON.stringify({meta: state.info.meta, request: state.info.request, parse: state.info.parse_summary, chunk: state.info.chunk_summary}, null, 2))}</pre>`;
  }
  detail.innerHTML = `
    <div class="meta" style="margin-bottom:10px;color:var(--muted)">
      <code>${esc(state.runId)}</code>
      · status=${esc(state.info?.meta?.status)}
      · ${esc(state.info?.meta?.seconds)}s
    </div>
    ${tabsHtml()}
    ${body}`;
  detail.querySelectorAll(".tab").forEach(btn => {
    btn.onclick = () => { state.tab = btn.dataset.tab; paintDetail(); };
  });
  detail.querySelectorAll("[data-step]").forEach(a => {
    a.onclick = async (e) => {
      e.preventDefault();
      const name = a.dataset.step;
      const data = await api(`/api/runs/${encodeURIComponent(state.runId)}/file?path=03_agent/${encodeURIComponent(name)}`);
      const view = document.getElementById("stepView");
      if (data && data.request_messages !== undefined) {
        // Raw dump also hides messages_after — use visualize tab instead.
        const sections = [
          ["request_messages", data.request_messages],
          ["tools / tool_choice", { tool_choice: data.tool_choice, tools: data.tools }],
          ["assistant", data.assistant],
          ["tool_results", data.tool_results],
        ];
        const meta = `turn=${esc(data.step)} · prompt_est=${esc(data.prompt_est_tokens)} · max_tokens=${esc(data.max_tokens)}` +
          (data.error ? ` · ERROR: ${esc(data.error)}` : "");
        view.innerHTML = `<div class="meta" style="margin:8px 0;color:var(--muted)">${meta}</div>` +
          sections.map(([title, payload]) => `
            <div style="margin:14px 0 6px;color:var(--accent);font-size:13px">${esc(title)}</div>
            <pre>${esc(pretty(payload, 20000))}</pre>
          `).join("");
      } else {
        view.innerHTML = `<pre>${esc(pretty(data, 20000))}</pre>`;
      }
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
