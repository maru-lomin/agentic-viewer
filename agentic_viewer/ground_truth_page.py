"""Ground-truth (answer sheet) management page."""

GROUND_TRUTH_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ground Truth</title>
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
    main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 54px); }
    aside {
      border-right: 1px solid var(--line); overflow: auto; background: #121820;
      padding: 10px;
    }
    section { overflow: auto; padding: 16px 20px; }
    .doc-item {
      border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
      margin-bottom: 8px; background: var(--panel); cursor: pointer;
    }
    .doc-item:hover { border-color: #3d4f66; }
    .doc-item.active { border-color: var(--accent); background: #1a2a40; }
    .doc-item .name { font-size: 13px; font-weight: 600; word-break: break-word; }
    .doc-item .sub { color: var(--muted); font-size: 11px; margin-top: 4px; }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.45; margin: 0 0 14px; }
    .empty { color: var(--muted); padding: 40px 0; }
    .gt-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .gt-table th, .gt-table td {
      border: 1px solid var(--line); padding: 8px 10px; vertical-align: top;
    }
    .gt-table th {
      background: var(--panel); color: var(--muted); font-weight: 500;
      position: sticky; top: 0; z-index: 1;
    }
    .gt-table td.key { font-family: var(--mono); font-size: 11px; max-width: 220px; word-break: break-word; }
    .gt-table textarea, .gt-table input {
      width: 100%; min-width: 120px; padding: 6px 8px; border-radius: 6px;
      border: 1px solid var(--line); background: #0f1419; color: var(--text);
      font-family: var(--mono); font-size: 11px; line-height: 1.4;
    }
    .gt-table textarea { min-height: 72px; resize: vertical; }
    .gt-actions { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
    .gt-actions button {
      padding: 4px 10px; border-radius: 999px; border: 1px solid var(--line);
      background: #152033; color: var(--text); font-size: 11px; cursor: pointer;
    }
    .gt-actions button:hover { border-color: var(--accent); }
    .gt-actions button.primary {
      background: #1a3a5c; border-color: #3d6a9a;
    }
    .gt-actions button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status-msg { font-size: 12px; margin: 8px 0 12px; }
    .status-msg.ok { color: var(--ok); }
    .status-msg.err { color: var(--err); }
    .search-box {
      width: 100%; margin-bottom: 10px; padding: 8px 10px; border-radius: 8px;
      border: 1px solid var(--line); background: #0f1419; color: var(--text);
      font-size: 12px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Agentic Viewer</h1>
    <nav class="topnav">
      <a href="/">Inference</a>
      <a href="/datasets">Datasets</a>
      <a href="/evaluation">Evaluation</a>
      <a href="/ground-truth" class="active">Ground Truth</a>
    </nav>
    <div class="meta" id="headerMeta">Loading…</div>
  </header>
  <main>
    <aside>
      <input type="search" class="search-box" id="docSearch" placeholder="Search documents…" />
      <div id="docList"></div>
    </aside>
    <section id="content">
      <div class="empty">Select a document to view and edit ground truth.</div>
    </section>
  </main>
<script>
const state = {
  documents: [],
  document: null,
  keys: [],
  loading: false,
  savingKey: null,
  message: null,
  messageKind: null,
  filter: "",
};

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function api(url, opts) {
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) {}
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : text || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function apiPut(url, body) {
  return api(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function syncUrl() {
  const params = new URLSearchParams();
  if (state.document) params.set("document", state.document);
  const qs = params.toString();
  history.replaceState(null, "", qs ? `/ground-truth?${qs}` : "/ground-truth");
}

function filteredDocuments() {
  const q = state.filter.trim().toLowerCase();
  if (!q) return state.documents;
  return state.documents.filter(d => d.document.toLowerCase().includes(q));
}

function renderDocList() {
  const el = document.getElementById("docList");
  const docs = filteredDocuments();
  if (!docs.length) {
    el.innerHTML = `<div class="empty">${state.documents.length ? "No matches" : "No documents"}</div>`;
    return;
  }
  el.innerHTML = docs.map(d => `
    <div class="doc-item ${d.document === state.document ? "active" : ""}" data-doc="${esc(d.document)}">
      <div class="name">${esc(d.document)}</div>
      <div class="sub">${d.n_keys} key(s)</div>
    </div>`).join("");
  el.querySelectorAll(".doc-item").forEach(node => {
    node.onclick = () => selectDocument(node.dataset.doc);
  });
}

function renderContent() {
  const el = document.getElementById("content");
  if (!state.document) {
    el.innerHTML = `<div class="empty">Select a document to view and edit ground truth.</div>`;
    return;
  }
  if (state.loading) {
    el.innerHTML = `<div class="empty">Loading ${esc(state.document)}…</div>`;
    return;
  }
  const msg = state.message
    ? `<div class="status-msg ${state.messageKind || ""}">${esc(state.message)}</div>` : "";
  const rows = state.keys.map(row => {
    const evidences = (row.evidences || []).join("\n");
    const pages = (row.evidence_pages || []).join(", ");
    const saving = state.savingKey === row.key;
    return `<tr data-gt-key="${esc(row.key)}">
      <td class="key">${esc(row.key)}</td>
      <td><input class="gt-value" value="${esc(row.value || "")}" /></td>
      <td><textarea class="gt-evidences">${esc(evidences)}</textarea></td>
      <td><input class="gt-pages" value="${esc(pages)}" placeholder="1, 2, 3" /></td>
      <td>
        <div class="gt-actions">
          <button type="button" class="primary gt-save" ${saving ? "disabled" : ""}>
            ${saving ? "Saving…" : "Save"}
          </button>
        </div>
      </td>
    </tr>`;
  }).join("");

  el.innerHTML = `
    <p class="hint">
      Edit <code>dataset/answer_sheet.json</code> entries for
      <strong>${esc(state.document)}</strong>.
      Saving creates a timestamped backup and invalidates cached <code>05_eval.json</code>
      for matching runs.
    </p>
    ${msg}
    <table class="gt-table">
      <thead>
        <tr>
          <th>Key</th><th>Value</th><th>Evidences</th><th>Pages</th><th></th>
        </tr>
      </thead>
      <tbody>${rows || `<tr><td colspan="5" class="empty">No keys</td></tr>`}</tbody>
    </table>`;

  el.querySelectorAll("tr[data-gt-key]").forEach(tr => {
    const key = tr.dataset.gtKey;
    const btn = tr.querySelector(".gt-save");
    if (!btn) return;
    btn.onclick = () => saveRow(tr, key);
  });
}

async function selectDocument(document) {
  state.document = document;
  state.message = null;
  state.loading = true;
  syncUrl();
  renderDocList();
  renderContent();
  try {
    const data = await api(`/api/ground-truth/document?document=${encodeURIComponent(document)}`);
    state.keys = data.keys || [];
  } catch (err) {
    state.keys = [];
    state.message = String(err.message || err);
    state.messageKind = "err";
  } finally {
    state.loading = false;
    renderContent();
  }
}

function parsePages(text) {
  const raw = String(text || "").trim();
  if (!raw) return [];
  return raw.split(/[,\s]+/).filter(Boolean).map(x => Number(x));
}

function parseEvidences(text) {
  return String(text || "").split("\n").map(s => s.trim()).filter(Boolean);
}

async function saveRow(tr, key) {
  const value = tr.querySelector(".gt-value")?.value ?? "";
  const evidences = parseEvidences(tr.querySelector(".gt-evidences")?.value ?? "");
  const pagesText = tr.querySelector(".gt-pages")?.value ?? "";
  let evidence_pages;
  try {
    evidence_pages = parsePages(pagesText);
    if (pagesText.trim() && evidence_pages.some(n => Number.isNaN(n))) {
      throw new Error("invalid page numbers");
    }
  } catch (err) {
    state.message = `Pages must be comma-separated integers (${err.message || err})`;
    state.messageKind = "err";
    renderContent();
    return;
  }

  state.savingKey = key;
  state.message = null;
  renderContent();
  try {
    const result = await apiPut("/api/ground-truth/key", {
      document: state.document,
      key,
      value,
      evidences,
      evidence_pages,
    });
    const idx = state.keys.findIndex(row => row.key === key);
    if (idx >= 0) state.keys[idx] = { key, ...result.entry };
    state.message = `Saved ${key}` + (result.invalidated_eval_caches
      ? ` · invalidated ${result.invalidated_eval_caches} eval cache(s)`
      : "");
    state.messageKind = "ok";
  } catch (err) {
    state.message = String(err.message || err);
    state.messageKind = "err";
  } finally {
    state.savingKey = null;
    renderContent();
  }
}

document.getElementById("docSearch").oninput = (e) => {
  state.filter = e.target.value || "";
  renderDocList();
};

(async function init() {
  const params = new URLSearchParams(location.search);
  const data = await api("/api/ground-truth");
  state.documents = data.documents || [];
  document.getElementById("headerMeta").textContent =
    `${state.documents.length} document(s) · ${esc(data.path || "")}`;
  renderDocList();
  const docParam = params.get("document");
  if (docParam && state.documents.some(d => d.document === docParam)) {
    await selectDocument(docParam);
  } else if (state.documents[0]) {
    await selectDocument(state.documents[0].document);
  } else {
    renderContent();
  }
})();
</script>
</body>
</html>
"""
