"""Evaluation aggregation page."""

EVALUATION_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agentic Evaluation</title>
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
      --warn: #e0a45c;
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
    header h1 { margin: 0; font-size: 18px; font-weight: 600; }
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
    main { display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 54px); }
    aside {
      border-right: 1px solid var(--line); overflow: auto; background: #121820;
      padding: 10px;
    }
    section { overflow: auto; padding: 16px 20px; }
    .toolbar {
      display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; align-items: center;
    }
    .toolbar button, .toolbar label.btn {
      padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line);
      background: #152033; color: var(--text); font-size: 12px; cursor: pointer;
    }
    .toolbar button:hover, .toolbar label.btn:hover { border-color: var(--accent); }
    .toolbar button:disabled { opacity: 0.5; cursor: not-allowed; }
    .toolbar button.primary {
      background: #1a3a5c; border-color: #3d6a9a; color: #e7ecf3;
    }
    .toolbar button.danger {
      background: #3a2024; border-color: #7a4048; color: #f0b0b4;
    }
    .toolbar label.btn { display: inline-flex; align-items: center; gap: 6px; }
    .run-item {
      border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
      margin-bottom: 8px; background: var(--panel); cursor: pointer;
    }
    .run-item:hover { border-color: #3d4f66; }
    .run-item.selected { border-color: var(--accent); background: #1a2a40; }
    .run-item .row1 { display: flex; gap: 8px; align-items: flex-start; }
    .run-item input { margin-top: 3px; }
    .run-item .id { font-family: var(--mono); font-size: 12px; word-break: break-all; }
    .run-item .sub { color: var(--muted); font-size: 11px; margin-top: 4px; }
    .badge {
      display: inline-block; padding: 1px 6px; border-radius: 4px;
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em;
    }
    .badge.ok { background: rgba(62, 207, 142, 0.15); color: var(--ok); }
    .badge.error { background: rgba(240, 113, 120, 0.15); color: var(--err); }
    .badge.warn { background: rgba(224, 164, 92, 0.15); color: var(--warn); }
    .empty { color: var(--muted); padding: 40px 0; }
    .warn-box {
      border: 1px solid #6b5530; background: rgba(224, 164, 92, 0.1);
      border-radius: 8px; padding: 10px 12px; color: #e0c090; font-size: 12px;
      margin-bottom: 14px; line-height: 1.45;
    }
    .batch-panel {
      border: 1px solid var(--line); border-radius: 10px; background: var(--panel);
      padding: 12px 14px; margin-bottom: 16px;
    }
    .batch-panel.running { border-color: #3d6a9a; background: #152033; }
    .batch-panel.clickable {
      cursor: pointer; transition: border-color 0.15s ease;
    }
    .batch-panel.clickable:hover { border-color: var(--accent); }
    .batch-panel.focused {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px rgba(61, 156, 240, 0.35);
    }
    .batch-panel .view-hint {
      color: var(--accent); font-size: 11px; margin-top: 8px;
    }
    .run-item.batch-run { border-color: #3d6a9a; background: #1a2a40; }
    .batch-panel .title { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
    .batch-panel .line { font-family: var(--mono); font-size: 12px; color: var(--muted); margin: 4px 0; }
    .batch-panel .line strong { color: var(--text); font-weight: 500; }
    .progress-wrap {
      height: 10px; background: #0f1419; border: 1px solid var(--line);
      border-radius: 999px; overflow: hidden; margin: 10px 0 8px;
    }
    .progress-bar {
      height: 100%; background: linear-gradient(90deg, #3d9cf0, #3ecf8e);
      border-radius: 999px; min-width: 2px; transition: width 0.3s ease;
    }
    .matrix-wrap { overflow: auto; max-width: 100%; }
    .matrix {
      width: 100%; border-collapse: collapse; font-size: 12px; min-width: 640px;
    }
    .matrix th, .matrix td {
      border: 1px solid var(--line); padding: 6px 8px; vertical-align: top;
    }
    .matrix th {
      background: var(--panel); color: var(--muted); font-weight: 500;
      position: sticky; top: 0; z-index: 1;
    }
    .matrix th.key-col, .matrix td.key-col {
      position: sticky; left: 0; z-index: 2; background: #121820;
      max-width: 220px; word-break: break-word; font-family: var(--mono); font-size: 11px;
    }
    .matrix th.key-col { z-index: 3; background: var(--panel); }
    .matrix .run-head {
      font-family: var(--mono); font-size: 11px; max-width: 140px; word-break: break-all;
    }
    .cell-em-y { color: var(--ok); }
    .cell-em-n { color: var(--err); }
    .cell-agent.correct { color: var(--ok); font-weight: 600; }
    .cell-agent.incorrect { color: var(--err); font-weight: 600; }
    .cell-agent.valid { color: var(--ok); font-weight: 600; }
    .cell-agent.invalid { color: var(--err); font-weight: 600; }
    .cell-agent.pending { color: var(--muted); }
    .cell-agent.running { color: var(--warn); }
    .live-progress { color: var(--warn); font-size: 12px; }
    .cell-agent.error { color: var(--err); font-size: 11px; }
    .cell-sub { color: var(--muted); font-size: 10px; margin-top: 2px; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.45; margin: 0 0 14px; }
    .run-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 16px; }
    .run-table th, .run-table td {
      text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line);
    }
    .run-table th { color: var(--muted); font-weight: 500; }
    .run-table td.mono { font-family: var(--mono); font-size: 11px; }
    .err-text { color: var(--err); font-size: 12px; margin-top: 6px; }
    .content-tabs {
      display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap;
    }
    .content-tabs button {
      background: transparent; border: 1px solid var(--line); color: var(--muted);
      padding: 6px 12px; border-radius: 6px; cursor: pointer; font: inherit; font-size: 12px;
    }
    .content-tabs button.active {
      color: var(--text); border-color: var(--accent); background: #152033;
    }
    .hierarchy-toolbar {
      display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 12px;
    }
    .hierarchy-toolbar select {
      min-width: 220px; padding: 6px 10px; border-radius: 8px;
      border: 1px solid var(--line); background: #0f1419; color: var(--text);
      font-family: var(--mono); font-size: 12px;
    }
    .hierarchy-frame {
      width: 100%; min-height: 72vh; border: 1px solid var(--line);
      border-radius: 10px; background: #0f1419;
    }
    .matrix td.clickable { cursor: pointer; }
    .matrix td.clickable:hover { background: rgba(61, 156, 240, 0.08); }
    .matrix .open-hierarchy {
      display: block; margin-top: 4px; font-size: 10px; color: var(--accent);
    }
    .cell-eval-btn {
      display: block; margin-top: 6px; padding: 2px 8px; border-radius: 999px;
      border: 1px solid var(--line); background: #152033; color: var(--text);
      font-size: 10px; cursor: pointer;
    }
    .cell-eval-btn:hover:not(:disabled) { border-color: var(--accent); }
    .cell-eval-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .cell-eval-btn.subtle { color: var(--muted); }
  </style>
</head>
<body>
  <header>
    <h1>Agentic Viewer</h1>
    <nav class="topnav">
      <a href="/">Inference</a>
      <a href="/evaluation" class="active">Evaluation</a>
    </nav>
    <div class="meta" id="headerMeta">Loading…</div>
  </header>
  <main>
    <aside>
      <div class="toolbar">
        <button type="button" id="selectFinished">Select finished</button>
        <button type="button" id="clearSelection">Clear</button>
      </div>
      <div id="runList"></div>
    </aside>
    <section id="content">
      <div class="empty">Select one or more runs to compare evaluation results.</div>
    </section>
  </main>
<script>
const state = {
  runs: [],
  selected: new Set(),
  summary: null,
  loading: false,
  error: null,
  batchJob: null,
  batchViewFocused: false,
  skipExisting: true,
  contentTab: "summary",
  hierarchyRunId: null,
  hierarchyKey: null,
  hierarchyKeys: [],
  hierarchyKeysLoading: false,
  hierarchyFollowBatch: true,
  hierarchyIframeSrc: null,
  agenticInflight: null,
  agenticEvalError: null,
};

function viewRunIds() {
  if (state.batchViewFocused && state.batchJob?.run_ids?.length) {
    return [...state.batchJob.run_ids];
  }
  return [...state.selected];
}

function batchRunSet() {
  const ids = state.batchJob?.run_ids;
  return ids?.length ? new Set(ids) : new Set();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(2);
}

function formatLiveLabel(live) {
  if (!live) return "";
  if (live.live_label) return live.live_label;
  const parts = [];
  if (live.master_turn != null) parts.push(`EvalMaster turn ${live.master_turn}`);
  if (live.search_session != null) {
    if (live.search_turn != null) {
      parts.push(`Search session ${live.search_session} turn ${live.search_turn}`);
    } else {
      parts.push(`Search session ${live.search_session}`);
    }
  }
  const activity = String(live.activity || "");
  if (activity === "waiting_llm") {
    parts.push(live.active_agent === "search" ? "waiting Search LLM" : "waiting EvalMaster LLM");
  } else if (activity.startsWith("tool:")) {
    parts.push(activity.replace("tool:", "tool "));
  }
  return parts.join(" · ");
}

function parseSelectedFromUrl() {
  const q = new URLSearchParams(location.search).get("runs");
  if (!q) return new Set();
  return new Set(q.split(",").map(s => s.trim()).filter(Boolean));
}

function parseContentTabFromUrl() {
  const tab = new URLSearchParams(location.search).get("tab");
  return tab === "hierarchy" ? "hierarchy" : "summary";
}

function syncUrl() {
  const params = new URLSearchParams();
  if (state.selected.size) params.set("runs", [...state.selected].join(","));
  if (state.batchJob?.job_id && state.batchViewFocused) {
    params.set("job", state.batchJob.job_id);
  }
  if (state.contentTab === "hierarchy") {
    params.set("tab", "hierarchy");
    if (state.hierarchyRunId) params.set("hrun", state.hierarchyRunId);
    if (state.hierarchyKey) params.set("hkey", state.hierarchyKey);
  }
  const qs = params.toString();
  const url = qs ? `/evaluation?${qs}` : "/evaluation";
  history.replaceState(null, "", url);
}

function hierarchyRunCandidates() {
  return viewRunIds();
}

function ensureHierarchySelection() {
  const runs = hierarchyRunCandidates();
  if (!runs.length) {
    state.hierarchyRunId = null;
    state.hierarchyKey = null;
    state.hierarchyKeys = [];
    return;
  }
  if (!state.hierarchyRunId || !runs.includes(state.hierarchyRunId)) {
    state.hierarchyRunId = runs[0];
    state.hierarchyKey = null;
  }
}

function hierarchyIframeSrc() {
  if (!state.hierarchyRunId || !state.hierarchyKey) return "";
  const q = new URLSearchParams({
    run: state.hierarchyRunId,
    tab: "hierarchy_eval",
    eval_key: state.hierarchyKey,
    embed: "1",
  });
  return `/?${q.toString()}`;
}

function hierarchyFullPageHref() {
  if (!state.hierarchyRunId || !state.hierarchyKey) return "";
  const q = new URLSearchParams({
    run: state.hierarchyRunId,
    tab: "hierarchy_eval",
    eval_key: state.hierarchyKey,
  });
  return `/?${q.toString()}`;
}

async function loadHierarchyKeys() {
  if (!state.hierarchyRunId) {
    state.hierarchyKeys = [];
    return;
  }
  state.hierarchyKeysLoading = true;
  try {
    const data = await api(`/api/runs/${encodeURIComponent(state.hierarchyRunId)}/agentic-eval/keys`);
    state.hierarchyKeys = data.keys || [];
    if (!state.hierarchyKey && state.hierarchyKeys.length) {
      const done = state.hierarchyKeys.find(k => k.status === "done");
      state.hierarchyKey = (done || state.hierarchyKeys[0]).key;
    } else if (state.hierarchyKey && !state.hierarchyKeys.some(k => k.key === state.hierarchyKey)) {
      state.hierarchyKey = state.hierarchyKeys[0]?.key || null;
    }
  } catch (_) {
    state.hierarchyKeys = [];
  } finally {
    state.hierarchyKeysLoading = false;
  }
}

function openHierarchy(runId, key) {
  state.contentTab = "hierarchy";
  state.hierarchyRunId = runId;
  state.hierarchyKey = key;
  state.hierarchyFollowBatch = false;
  state.hierarchyIframeSrc = null;
  syncUrl();
  renderContent();
  loadHierarchyKeys().then(() => renderContent());
}

function maybeFollowBatchHierarchy() {
  if (state.contentTab !== "hierarchy" || !state.hierarchyFollowBatch) return;
  const cur = state.batchJob?.current;
  if (!cur?.run_id || !cur?.key) return;
  if (state.hierarchyRunId === cur.run_id && state.hierarchyKey === cur.key) return;
  state.hierarchyRunId = cur.run_id;
  state.hierarchyKey = cur.key;
  state.hierarchyIframeSrc = null;
  syncUrl();
  loadHierarchyKeys().then(() => renderContent());
}

function contentTabsHtml() {
  return `<div class="content-tabs">
    <button type="button" data-content-tab="summary" class="${state.contentTab === "summary" ? "active" : ""}">Summary</button>
    <button type="button" data-content-tab="hierarchy" class="${state.contentTab === "hierarchy" ? "active" : ""}">Eval hierarchy</button>
  </div>`;
}

function renderHierarchyToolbar() {
  ensureHierarchySelection();
  const runs = hierarchyRunCandidates();
  if (!runs.length) {
    return `<div class="empty">Select one or more runs to inspect agentic-eval traces.</div>`;
  }
  const runOpts = runs.map(id =>
    `<option value="${esc(id)}" ${id === state.hierarchyRunId ? "selected" : ""}>${esc(id)}</option>`
  ).join("");
  const keyOpts = state.hierarchyKeys.map(row => {
    const status = row.status || "pending";
    const verdict = row.is_correct_answer ? ` · pred ${row.is_correct_answer}` : "";
    const goldVerdict = row.is_valid_gold ? ` · GT ${row.is_valid_gold}` : "";
    return `<option value="${esc(row.key)}" ${row.key === state.hierarchyKey ? "selected" : ""}>${esc(row.key)} (${esc(status)}${esc(verdict)}${esc(goldVerdict)})</option>`;
  }).join("");
  const follow = state.hierarchyFollowBatch
    ? `<label class="btn" style="font-size:12px"><input type="checkbox" id="followBatch" checked />
        Follow batch current key</label>`
    : `<label class="btn" style="font-size:12px"><input type="checkbox" id="followBatch" />
        Follow batch current key</label>`;
  return `
    <p class="hint">
      EvalMaster → SearchAgent trace for one key under <code>06_agentic_eval/</code>.
      Embedded from the Inference page <b>Eval hierarchy</b> tab.
    </p>
    <div class="hierarchy-toolbar">
      <label>Run
        <select id="hierarchyRun">${runOpts}</select>
      </label>
      <label>Key
        <select id="hierarchyKey" ${state.hierarchyKeysLoading ? "disabled" : ""}>
          ${keyOpts || `<option value="">—</option>`}
        </select>
      </label>
      ${follow}
      ${state.hierarchyRunId && state.hierarchyKey
        ? `<a href="${esc(hierarchyFullPageHref())}" target="_blank">Open full page</a>`
        : ""}
    </div>`;
}

function updateHierarchyFrame() {
  const host = document.getElementById("hierarchyFrameHost");
  if (!host) return;
  const src = hierarchyIframeSrc();
  if (!src) {
    state.hierarchyIframeSrc = null;
    host.innerHTML = `<div class="empty">${state.hierarchyKeysLoading ? "Loading keys…" : "No eval trace for this run/key yet."}</div>`;
    return;
  }
  if (state.hierarchyIframeSrc === src) return;
  state.hierarchyIframeSrc = src;
  host.innerHTML = `<iframe class="hierarchy-frame" src="${esc(src)}" title="Eval hierarchy"></iframe>`;
}

function renderHierarchyBody() {
  return `${renderHierarchyToolbar()}<div id="hierarchyFrameHost"></div>`;
}

function renderHierarchyLayout(batchHtml, tabs) {
  return `
    <div id="evalBatchHost">${batchHtml}</div>
    <div id="evalMainHost">
      ${tabs}
      ${renderHierarchyBody()}
    </div>`;
}

function refreshHierarchyDom(batchHtml, tabs) {
  const batchHost = document.getElementById("evalBatchHost");
  const tabsHost = document.getElementById("evalHierarchyTabsHost");
  const toolbarHost = document.getElementById("evalHierarchyToolbarHost");
  if (batchHost) batchHost.innerHTML = batchHtml;
  if (tabsHost) tabsHost.innerHTML = tabs;
  if (toolbarHost) toolbarHost.innerHTML = renderHierarchyToolbar();
  updateHierarchyFrame();
  bindBatchControls();
  bindContentTabs();
  bindHierarchyControls();
}

function mountHierarchyView(el, batchHtml, tabs) {
  el.innerHTML = `
    <div id="evalBatchHost">${batchHtml}</div>
    <div id="evalMainHost">
      <div id="evalHierarchyTabsHost">${tabs}</div>
      <div id="evalHierarchyToolbarHost">${renderHierarchyToolbar()}</div>
      <div id="hierarchyFrameHost"></div>
    </div>`;
  updateHierarchyFrame();
  bindBatchControls();
  bindContentTabs();
  bindHierarchyControls();
}

function bindHierarchyControls() {
  const runSel = document.getElementById("hierarchyRun");
  if (runSel) {
    runSel.onchange = async () => {
      state.hierarchyRunId = runSel.value;
      state.hierarchyKey = null;
      state.hierarchyFollowBatch = false;
      state.hierarchyIframeSrc = null;
      syncUrl();
      await loadHierarchyKeys();
      renderContent();
    };
  }
  const keySel = document.getElementById("hierarchyKey");
  if (keySel) {
    keySel.onchange = () => {
      state.hierarchyKey = keySel.value || null;
      state.hierarchyFollowBatch = false;
      state.hierarchyIframeSrc = null;
      syncUrl();
      renderContent();
    };
  }
  const followCb = document.getElementById("followBatch");
  if (followCb) {
    followCb.onchange = () => {
      state.hierarchyFollowBatch = followCb.checked;
      if (state.hierarchyFollowBatch) maybeFollowBatchHierarchy();
    };
  }
  document.querySelectorAll("td[data-matrix-cell]").forEach(td => {
    td.onclick = (e) => {
      if (e.target.closest("[data-run-eval]")) return;
      if (!td.dataset.runId || !td.dataset.key) return;
      openHierarchy(td.dataset.runId, td.dataset.key);
    };
  });
}

function bindMatrixControls() {
  document.querySelectorAll("[data-run-eval]").forEach(btn => {
    btn.onclick = (e) => {
      e.stopPropagation();
      runAgenticEval(btn.dataset.runId, btn.dataset.key);
    };
  });
}

function bindContentTabs() {
  document.querySelectorAll("[data-content-tab]").forEach(btn => {
    btn.onclick = () => {
      const next = btn.dataset.contentTab;
      if (next === state.contentTab) return;
      state.contentTab = next;
      if (next === "hierarchy") {
        ensureHierarchySelection();
        loadHierarchyKeys().then(() => {
          syncUrl();
          renderContent();
        });
      } else {
        state.hierarchyIframeSrc = null;
        syncUrl();
        renderContent();
      }
    };
  });
}

function persistBatchJobId(jobId) {
  try {
    if (jobId) sessionStorage.setItem("agentic_eval_batch_job", jobId);
    else sessionStorage.removeItem("agentic_eval_batch_job");
  } catch (_) { /* ignore */ }
}

function readPersistedBatchJobId() {
  try { return sessionStorage.getItem("agentic_eval_batch_job"); } catch (_) { return null; }
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
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

async function runAgenticEval(runId, key) {
  if (!runId || !key) return;
  if (batchIsActive()) return;
  const prev = state.agenticInflight;
  const keys = prev && prev.runId === runId && Array.isArray(prev.keys)
    ? [...prev.keys]
    : (prev && prev.runId === runId && prev.key ? [prev.key] : []);
  if (!keys.includes(key)) keys.push(key);
  state.agenticInflight = { runId, keys };
  state.agenticEvalError = null;
  renderContent();
  try {
    await apiPost(`/api/runs/${encodeURIComponent(runId)}/agentic-eval`, { key });
    await loadSummary(true);
    await refreshRuns();
  } catch (err) {
    state.agenticEvalError = String(err.message || err);
    await loadSummary(true);
  } finally {
    const cur = state.agenticInflight;
    if (cur && cur.runId === runId) {
      const keys = (Array.isArray(cur.keys) ? cur.keys : (cur.key ? [cur.key] : []))
        .filter(k => k !== key);
      state.agenticInflight = keys.length ? { runId, keys } : null;
    }
    renderContent();
  }
}

function renderMatrixEvalAction(runId, key, ae, isCurrent) {
  const inflight = state.agenticInflight;
  const inflightKeys = inflight && inflight.runId === runId
    ? (Array.isArray(inflight.keys) ? inflight.keys : (inflight.key ? [inflight.key] : []))
    : [];
  const isInflight = inflightKeys.includes(key);
  const batchDisabled = batchIsActive();
  if (ae.status === "running" || isCurrent || isInflight) {
    return `<button type="button" class="cell-eval-btn" disabled>Running…</button>`;
  }
  if (ae.status === "error") {
    return `<button type="button" class="cell-eval-btn" data-run-eval="1"
      data-run-id="${esc(runId)}" data-key="${esc(key)}"
      ${batchDisabled ? "disabled" : ""}>Retry</button>`;
  }
  if (ae.status === "done") {
    return `<button type="button" class="cell-eval-btn subtle" data-run-eval="1"
      data-run-id="${esc(runId)}" data-key="${esc(key)}"
      ${batchDisabled ? "disabled" : ""}>Re-run</button>`;
  }
  return `<button type="button" class="cell-eval-btn" data-run-eval="1"
    data-run-id="${esc(runId)}" data-key="${esc(key)}"
    ${batchDisabled ? "disabled" : ""}>agentic-eval</button>`;
}

function batchIsActive() {
  const j = state.batchJob;
  return j && (j.status === "queued" || j.status === "running");
}

function renderBatchStartControls() {
  const disabled = !state.selected.size || batchIsActive();
  return `
    <div class="batch-panel">
      <div class="title">Batch agentic-evaluation</div>
      <div class="toolbar" style="margin:0">
        <button type="button" class="primary" id="startBatch" ${disabled ? "disabled" : ""}>
          Run all keys (selected runs)
        </button>
        <label class="btn"><input type="checkbox" id="skipExisting" ${state.skipExisting ? "checked" : ""} />
          Skip existing</label>
      </div>
      <p class="hint" style="margin:8px 0 0">
        Evaluates every gold key via the inference API (up to 8 keys in parallel per run). Results save under
        <code>06_agentic_eval/</code>.
      </p>
    </div>`;
}

function renderBatchPanel() {
  const j = state.batchJob;
  if (!j) {
    return renderBatchStartControls();
  }

  const pct = j.progress_pct ?? (j.total ? Math.round(100 * j.completed / j.total) : 0);
  const running = j.status === "queued" || j.status === "running";
  const cur = j.current;
  const curHtml = cur
    ? `<div class="line">Current: <strong>${esc(cur.run_id)}</strong> · ${esc(cur.key)}</div>
       ${cur.live_label || formatLiveLabel(cur.live)
         ? `<div class="line live-progress">${esc(cur.live_label || formatLiveLabel(cur.live))}</div>` : ""}`
    : "";
  const errBlock = (j.errors || []).length
    ? `<div class="err-text">${j.errors.length} error(s) — latest: ${esc(j.errors[j.errors.length - 1].error)}</div>`
    : "";
  const cancelBtn = running
    ? `<button type="button" class="danger" id="cancelBatch">Cancel</button>`
    : "";
  const refreshBtn = running
    ? `<button type="button" class="btn" id="refreshBatch">Refresh</button>`
    : "";
  const dismissBtn = running
    ? ""
    : `<button type="button" class="btn" id="dismissBatch">Dismiss</button>`;
  const runList = (j.run_ids || []).map(id => esc(id)).join(", ");
  const focused = state.batchViewFocused;
  const viewHint = focused
    ? `<div class="view-hint">Showing summary & matrix for ${(j.run_ids || []).length} run(s) in this job</div>`
    : `<div class="view-hint">Click to view per-run summary and key × run matrix</div>`;

  const statusHtml = `
    <div class="batch-panel ${running ? "running" : ""} clickable ${focused ? "focused" : ""}"
         id="batchJobPanel" role="button" tabindex="0"
         title="View evaluation summary for batch runs">
      <div class="title">Batch job · ${esc(j.status)}</div>
      <div class="line">Runs: <strong>${runList || "—"}</strong></div>
      <div class="line">
        Progress: <strong>${j.completed}/${j.total}</strong>
        · skipped ${j.skipped}
        · failed ${j.failed}
        ${pct != null ? ` · ${pct}%` : ""}
      </div>
      ${curHtml}
      <div class="progress-wrap"><div class="progress-bar" style="width:${Math.max(0, Math.min(100, pct || 0))}%"></div></div>
      <div class="line">${esc(j.message || "")}</div>
      ${errBlock}
      ${running ? `<div class="view-hint">Progress updates manually — click Refresh</div>` : viewHint}
      <div class="toolbar" style="margin:8px 0 0">${cancelBtn}${refreshBtn}${dismissBtn}</div>
    </div>`;

  return running ? statusHtml : `${statusHtml}${renderBatchStartControls()}`;
}

function renderRunList() {
  const el = document.getElementById("runList");
  if (!state.runs.length) {
    el.innerHTML = `<div class="empty">No runs found</div>`;
    return;
  }
  el.innerHTML = state.runs.map(r => {
    const checked = state.selected.has(r.run_id) ? "checked" : "";
    const inBatch = batchRunSet().has(r.run_id);
    const sel = state.selected.has(r.run_id) ? "selected" : "";
    const batchCls = inBatch && state.batchViewFocused ? " batch-run" : "";
    const es = r.eval_summary;
    const ae = r.agentic_eval_summary;
    const baseline = es
      ? `EM ${fmtPct(es.value_exact_match)} · pageF1 ${fmtPct(es.page_f1_macro)}`
      : "no baseline eval";
    const agentic = ae
      ? `agentic ${ae.n_done}/${ae.n_total}${ae.accuracy != null ? ` · pred acc ${fmtPct(ae.accuracy)}` : ""}${ae.gold_validity != null ? ` · GT valid ${fmtPct(ae.gold_validity)}` : ""}`
      : "agentic —";
    const badgeCls = r.status === "ok" ? "ok" : (r.status === "error" ? "error" : "warn");
    return `
      <div class="run-item ${sel}${batchCls}" data-id="${esc(r.run_id)}">
        <div class="row1">
          <input type="checkbox" ${checked} data-run-check="${esc(r.run_id)}" />
          <div>
            <div class="id">${esc(r.run_id)}</div>
            <div class="sub">
              <span class="badge ${badgeCls}">${esc(r.status)}</span>
              ${r.seconds != null ? `${r.seconds}s` : ""} · kv=${r.n_kv ?? "?"}
            </div>
            <div class="sub">${esc(baseline)}</div>
            <div class="sub">${esc(agentic)}</div>
          </div>
        </div>
      </div>`;
  }).join("");

  el.querySelectorAll("[data-run-check]").forEach(cb => {
    cb.onclick = (e) => {
      e.stopPropagation();
      toggleRun(cb.dataset.runCheck, cb.checked);
    };
  });
  el.querySelectorAll(".run-item").forEach(node => {
    node.onclick = (e) => {
      if (e.target.matches("input")) return;
      const id = node.dataset.id;
      toggleRun(id, !state.selected.has(id));
    };
  });
}

function toggleRun(runId, on) {
  if (on) state.selected.add(runId);
  else state.selected.delete(runId);
  state.batchViewFocused = false;
  syncUrl();
  renderRunList();
  loadSummary();
}

function focusBatchJob() {
  if (!state.batchJob?.run_ids?.length) return;
  state.batchViewFocused = true;
  persistBatchJobId(state.batchJob.job_id);
  syncUrl();
  renderRunList();
  loadSummary();
}

function renderSummaryBody() {
  const s = state.summary;
  if (!s) return `<div class="empty">Loading summary…</div>`;

  const warn = s.document_warning
    ? `<div class="warn-box">${esc(s.document_warning)}</div>` : "";

  const runRows = (s.per_run || []).map(r => {
    const b = r.baseline || {};
    const a = r.agentic || {};
    return `<tr>
      <td class="mono"><a href="/?run=${encodeURIComponent(r.run_id)}&tab=eval">${esc(r.run_id)}</a></td>
      <td>${esc(r.document || "—")}</td>
      <td>${r.has_baseline_eval ? fmtPct(b.value_exact_match) : "—"}</td>
      <td>${r.has_baseline_eval ? fmtPct(b.page_f1_macro) : "—"}</td>
      <td>${r.has_baseline_eval ? fmtPct(b.evidence_token_f1) : "—"}</td>
      <td>${a.n_done ?? 0}/${a.n_total ?? 0}</td>
      <td>${a.accuracy != null ? fmtPct(a.accuracy) : "—"}</td>
      <td>${a.gold_validity != null ? fmtPct(a.gold_validity) : "—"}</td>
    </tr>`;
  }).join("");

  const runIds = s.run_ids || [];
  const cur = state.batchJob?.current;
  const headRuns = runIds.map(id =>
    `<th class="run-head"><a href="/?run=${encodeURIComponent(id)}&tab=eval">${esc(id)}</a></th>`
  ).join("");

  const matrixRows = (s.per_key || []).map(row => {
    const cells = runIds.map(runId => {
      const cell = (row.by_run || {})[runId];
      if (!cell) return `<td>—</td>`;
      const emCls = cell.baseline_em ? "cell-em-y" : "cell-em-n";
      const ae = cell.agentic || {};
      const isCurrent = cur && cur.run_id === runId && cur.key === row.key && ae.status !== "done";
      let agentHtml;
      const canOpen = ae.status === "done" || ae.status === "error" || ae.status === "running";
      if (ae.status === "done") {
        const v = String(ae.is_correct_answer || "").toLowerCase();
        const gv = String(ae.is_valid_gold || "").toLowerCase();
        const predCls = v === "correct" ? "correct" : (v === "incorrect" ? "incorrect" : "pending");
        const goldCls = gv === "valid" ? "valid" : (gv === "invalid" ? "invalid" : "pending");
        const predHtml = v ? `<div class="cell-agent ${predCls}">pred: ${esc(v)}</div>` : "";
        const goldHtml = gv ? `<div class="cell-agent ${goldCls}">GT: ${esc(gv)}</div>` : "";
        agentHtml = `${predHtml}${goldHtml}`;
      } else if (ae.status === "running" || isCurrent) {
        const liveLabel = ae.live_label || formatLiveLabel(ae.live)
          || (isCurrent ? formatLiveLabel(cur?.live) : "");
        agentHtml = liveLabel
          ? `<div class="cell-agent running">running</div><div class="cell-sub">${esc(liveLabel)}</div>`
          : `<div class="cell-agent running">running</div>`;
      } else if (ae.status === "error") {
        agentHtml = `<div class="cell-agent error">${esc(ae.error || "error")}</div>`;
      } else {
        agentHtml = `<div class="cell-agent pending">pending</div>`;
      }
      const openLink = canOpen
        ? `<span class="open-hierarchy">view eval hierarchy</span>` : "";
      const evalAction = renderMatrixEvalAction(runId, row.key, ae, isCurrent);
      return `<td class="${canOpen ? "clickable" : ""}" data-matrix-cell="1"
        data-run-id="${esc(runId)}" data-key="${esc(row.key)}">
        <div class="${emCls}">EM ${cell.baseline_em ? "Y" : "N"}</div>
        <div class="cell-sub">pageF1 ${fmtPct(cell.page_f1)} · evid ${fmtPct(cell.evidence_f1)}</div>
        ${agentHtml}
        ${evalAction}
        ${openLink}
      </td>`;
    }).join("");
    return `<tr>
      <td class="key-col">${esc(row.key)}</td>
      <td>${esc(row.gold_value ?? "")}</td>
      ${cells}
    </tr>`;
  }).join("");

  return `
    <p class="hint">
      Baseline metrics from <code>04_result.json</code> vs answer sheet.
      Agentic verdicts from <code>06_agentic_eval/</code> (refreshes while batch runs).
      Use <b>agentic-eval</b> per cell for a single key, or batch run above for all keys.
    </p>
    ${state.agenticEvalError
      ? `<div class="err-text">Agentic eval: ${esc(state.agenticEvalError)}</div>` : ""}
    ${warn}
    <h2 style="font-size:14px;margin:0 0 10px">Per-run summary</h2>
    <table class="run-table">
      <thead>
        <tr>
          <th>Run</th><th>Document</th><th>Value EM</th><th>Page F1</th>
          <th>Evid F1</th><th>Agentic done</th><th>Pred acc</th><th>GT valid</th>
        </tr>
      </thead>
      <tbody>${runRows || `<tr><td colspan="8" class="empty">No runs</td></tr>`}</tbody>
    </table>
    <h2 style="font-size:14px;margin:16px 0 10px">Key × run matrix</h2>
    <div class="matrix-wrap">
      <table class="matrix">
        <thead>
          <tr>
            <th class="key-col">Key</th>
            <th>Gold value</th>
            ${headRuns}
          </tr>
        </thead>
        <tbody>${matrixRows || `<tr><td colspan="${2 + runIds.length}" class="empty">No keys</td></tr>`}</tbody>
      </table>
    </div>`;
}

function renderContent() {
  const el = document.getElementById("content");
  const batchHtml = renderBatchPanel();
  const hasRuns = viewRunIds().length > 0;
  const tabs = contentTabsHtml();

  if (!hasRuns && !state.batchJob) {
    state.hierarchyIframeSrc = null;
    el.innerHTML = `<div class="empty">Select one or more runs to compare evaluation results.</div>`;
    return;
  }
  if (!hasRuns && state.batchJob) {
    state.hierarchyIframeSrc = null;
    el.innerHTML = batchHtml + `<div class="empty">Click the batch job panel above to view summary and matrix.</div>`;
    bindBatchControls();
    return;
  }

  if (state.contentTab === "hierarchy") {
    if (document.getElementById("evalBatchHost")) {
      refreshHierarchyDom(batchHtml, tabs);
      return;
    }
    mountHierarchyView(el, batchHtml, tabs);
    return;
  }

  state.hierarchyIframeSrc = null;

  if (state.loading && !state.summary && hasRuns) {
    el.innerHTML = batchHtml + tabs + `<div class="empty">Loading summary…</div>`;
    bindBatchControls();
    bindContentTabs();
    return;
  }
  if (state.error && !state.summary && hasRuns) {
    el.innerHTML = batchHtml + tabs + `<div class="empty" style="color:var(--err)">${esc(state.error)}</div>`;
    bindBatchControls();
    bindContentTabs();
    return;
  }

  el.innerHTML = batchHtml + tabs + renderSummaryBody();
  bindBatchControls();
  bindContentTabs();
  bindHierarchyControls();
  bindMatrixControls();
}

function renderSummary() {
  renderContent();
}

function bindBatchControls() {
  const startBtn = document.getElementById("startBatch");
  if (startBtn) startBtn.onclick = startBatch;
  const skipCb = document.getElementById("skipExisting");
  if (skipCb) skipCb.onchange = () => { state.skipExisting = skipCb.checked; };
  const cancelBtn = document.getElementById("cancelBatch");
  if (cancelBtn) {
    cancelBtn.onclick = (e) => {
      e.stopPropagation();
      cancelBatch();
    };
  }
  const dismissBtn = document.getElementById("dismissBatch");
  if (dismissBtn) {
    dismissBtn.onclick = (e) => {
      e.stopPropagation();
      dismissBatchJob();
    };
  }
  const refreshBtn = document.getElementById("refreshBatch");
  if (refreshBtn) {
    refreshBtn.onclick = (e) => {
      e.stopPropagation();
      refreshBatchJob();
    };
  }
  const panel = document.getElementById("batchJobPanel");
  if (panel) {
    panel.onclick = (e) => {
      if (e.target.closest("#cancelBatch, #dismissBatch, #refreshBatch")) return;
      focusBatchJob();
    };
    panel.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        focusBatchJob();
      }
    };
  }
}

function dismissBatchJob() {
  if (batchIsActive()) return;
  state.batchJob = null;
  state.batchViewFocused = false;
  persistBatchJobId(null);
  syncUrl();
  renderSummary();
  renderRunList();
}

async function loadSummary(silent=false) {
  const runIds = viewRunIds();
  if (!runIds.length) {
    state.summary = null;
    if (!silent) state.error = null;
    renderSummary();
    return;
  }
  if (!silent) {
    state.loading = true;
    state.error = null;
    renderSummary();
  }
  try {
    const q = runIds.map(encodeURIComponent).join(",");
    state.summary = await api(`/api/evaluation/summary?run_ids=${q}`);
    state.error = null;
  } catch (err) {
    if (!silent) {
      state.summary = null;
      state.error = String(err.message || err);
    }
  } finally {
    if (!silent) state.loading = false;
    renderSummary();
  }
}

async function refreshRuns() {
  state.runs = await api("/api/runs");
  renderRunList();
}

async function refreshBatchJob() {
  if (!state.batchJob?.job_id) return;
  try {
    state.batchJob = await api(`/api/evaluation/batch-jobs/${encodeURIComponent(state.batchJob.job_id)}`);
    renderSummary();
    renderRunList();
    maybeFollowBatchHierarchy();
    const running = state.batchJob.status === "running" || state.batchJob.status === "queued";
    if (!running) {
      persistBatchJobId(null);
    }
    if (viewRunIds().length) {
      await loadSummary(true);
    }
    if (!running) {
      await refreshRuns();
    }
  } catch (err) {
    state.error = String(err.message || err);
    renderSummary();
  }
}

async function startBatch() {
  if (!state.selected.size || batchIsActive()) return;
  try {
    state.batchJob = await apiPost("/api/evaluation/batch-agentic-eval", {
      run_ids: [...state.selected],
      skip_existing: state.skipExisting,
    });
    state.batchViewFocused = true;
    persistBatchJobId(state.batchJob.job_id);
    state.error = null;
    syncUrl();
    renderSummary();
    renderRunList();
    if (batchIsActive()) {
      await refreshBatchJob();
    } else {
      persistBatchJobId(null);
      await loadSummary(true);
    }
  } catch (err) {
    state.error = String(err.message || err);
    renderSummary();
  }
}

async function cancelBatch() {
  if (!state.batchJob?.job_id) return;
  try {
    state.batchJob = await apiPost(
      `/api/evaluation/batch-jobs/${encodeURIComponent(state.batchJob.job_id)}/cancel`,
      {}
    );
    await refreshBatchJob();
  } catch (err) {
    state.error = String(err.message || err);
    renderSummary();
  }
}

async function loadBatchJobById(jobId) {
  if (!jobId) return null;
  try {
    return await api(`/api/evaluation/batch-jobs/${encodeURIComponent(jobId)}`);
  } catch (_) {
    return null;
  }
}

async function resumeActiveJob() {
  const params = new URLSearchParams(location.search);
  const jobParam = params.get("job");

  let job = null;
  if (jobParam) {
    job = await loadBatchJobById(jobParam);
  }
  if (!job) {
    try {
      const data = await api("/api/evaluation/batch-jobs/active");
      if (data.active && data.job) job = data.job;
    } catch (_) { /* ignore */ }
  }
  if (!job) {
    const stored = readPersistedBatchJobId();
    if (stored) job = await loadBatchJobById(stored);
  }

  if (!job) {
    persistBatchJobId(null);
    return;
  }

  const running = job.status === "queued" || job.status === "running";
  if (!running && !jobParam) {
    persistBatchJobId(null);
    return;
  }

  state.batchJob = job;
  if (running) {
    persistBatchJobId(job.job_id);
  } else {
    persistBatchJobId(null);
  }
  if (jobParam) {
    state.batchViewFocused = true;
    syncUrl();
  }
  renderSummary();
  renderRunList();
  if (state.batchViewFocused) {
    await loadSummary(true);
  }
}

document.getElementById("selectFinished").onclick = () => {
  state.runs.filter(r => r.status === "ok").forEach(r => state.selected.add(r.run_id));
  state.batchViewFocused = false;
  syncUrl();
  renderRunList();
  loadSummary();
};

document.getElementById("clearSelection").onclick = () => {
  state.selected.clear();
  state.batchViewFocused = false;
  syncUrl();
  renderRunList();
  loadSummary();
};

(async function init() {
  const params = new URLSearchParams(location.search);
  state.selected = parseSelectedFromUrl();
  if (params.get("job")) state.batchViewFocused = true;
  state.contentTab = parseContentTabFromUrl();
  state.hierarchyRunId = params.get("hrun");
  state.hierarchyKey = params.get("hkey");
  state.runs = await api("/api/runs");
  document.getElementById("headerMeta").textContent =
    `${state.runs.length} run(s) · ${location.origin}`;
  renderRunList();
  await resumeActiveJob();
  if (state.contentTab === "hierarchy") {
    ensureHierarchySelection();
    await loadHierarchyKeys();
  }
  if (viewRunIds().length && !state.summary) await loadSummary();
  else renderContent();
})();
</script>
</body>
</html>
"""
