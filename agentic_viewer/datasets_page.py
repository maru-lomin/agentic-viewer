"""Named dataset management page."""

DATASETS_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agentic Viewer — Datasets</title>
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
    .create-box {
      border: 1px solid var(--line); border-radius: 8px; padding: 10px;
      background: var(--panel); margin-bottom: 12px;
    }
    .create-box h2 {
      margin: 0 0 8px; font-size: 12px; text-transform: uppercase;
      letter-spacing: 0.04em; color: var(--muted);
    }
    .create-box input[type="text"], .search-box {
      width: 100%; margin-bottom: 8px; padding: 8px 10px; border-radius: 8px;
      border: 1px solid var(--line); background: #0f1419; color: var(--text);
      font-size: 12px;
    }
    .create-box input[type="file"] { width: 100%; font-size: 11px; color: var(--muted); margin-bottom: 8px; }
    .ds-item {
      border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
      margin-bottom: 8px; background: var(--panel); cursor: pointer;
    }
    .ds-item:hover { border-color: #3d4f66; }
    .ds-item.active { border-color: var(--accent); background: #1a2a40; }
    .ds-item .name { font-size: 13px; font-weight: 600; word-break: break-word; }
    .ds-item .sub { color: var(--muted); font-size: 11px; margin-top: 4px; }
    .hint { color: var(--muted); font-size: 12px; line-height: 1.45; margin: 0 0 14px; }
    .empty { color: var(--muted); padding: 40px 0; }
    .file-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .file-table th, .file-table td {
      border: 1px solid var(--line); padding: 8px 10px; vertical-align: top;
      text-align: left;
    }
    .file-table th {
      background: var(--panel); color: var(--muted); font-weight: 500;
    }
    .file-table td.name { font-family: var(--mono); font-size: 11px; word-break: break-all; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 14px; }
    button, label.btn {
      padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line);
      background: #152033; color: var(--text); font-size: 12px; cursor: pointer;
    }
    button:hover, label.btn:hover { border-color: var(--accent); }
    button.primary { background: #1a3a5c; border-color: #3d6a9a; }
    button.danger { background: #3a2024; border-color: #7a4048; color: #f0b0b4; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status-msg { font-size: 12px; margin: 8px 0 12px; }
    .status-msg.ok { color: var(--ok); }
    .status-msg.err { color: var(--err); }
    .badge {
      display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 4px;
      border: 1px solid var(--line); color: var(--muted); text-transform: uppercase;
    }
  </style>
</head>
<body>
  <header>
    <h1>Agentic Viewer</h1>
    <nav class="topnav">
      <a href="/">Inference</a>
      <a href="/datasets" class="active">Datasets</a>
      <a href="/evaluation">Evaluation</a>
      <a href="/ground-truth">Ground Truth</a>
    </nav>
    <div class="meta" id="headerMeta">Loading…</div>
  </header>
  <main>
    <aside>
      <div class="create-box">
        <h2>New dataset</h2>
        <input type="text" id="newName" placeholder="Name (e.g. evaluation-v3)" />
        <input type="file" id="newFiles" accept=".pdf,.PDF,application/pdf" multiple />
        <button type="button" class="primary" id="createBtn">Create</button>
      </div>
      <input type="search" class="search-box" id="dsSearch" placeholder="Search datasets…" />
      <div id="dsList"></div>
    </aside>
    <section id="content">
      <div class="empty">Select a dataset.</div>
    </section>
  </main>
<script>
const state = {
  datasets: [],
  selected: null,
  detail: null,
  loading: false,
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

function dsKey(row) {
  return `${row.source}/${row.id}`;
}

function syncUrl() {
  const params = new URLSearchParams();
  if (state.selected) {
    params.set("source", state.selected.source);
    params.set("id", state.selected.id);
  }
  const qs = params.toString();
  history.replaceState(null, "", qs ? `/datasets?${qs}` : "/datasets");
}

function filteredDatasets() {
  const q = state.filter.trim().toLowerCase();
  if (!q) return state.datasets;
  return state.datasets.filter(d =>
    (d.name || "").toLowerCase().includes(q) || (d.id || "").toLowerCase().includes(q)
  );
}

function fmtSize(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (v < 1024) return `${v} B`;
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB`;
  return `${(v / (1024 * 1024)).toFixed(1)} MB`;
}

function renderList() {
  const el = document.getElementById("dsList");
  const rows = filteredDatasets();
  if (!rows.length) {
    el.innerHTML = `<div class="empty" style="padding:16px">${state.datasets.length ? "No matches" : "No datasets"}</div>`;
    return;
  }
  const sel = state.selected ? dsKey(state.selected) : "";
  el.innerHTML = rows.map(d => `
    <div class="ds-item ${dsKey(d) === sel ? "active" : ""}" data-source="${esc(d.source)}" data-id="${esc(d.id)}">
      <div class="name">${esc(d.name || d.id)}</div>
      <div class="sub">
        <span class="badge">${esc(d.source)}</span>
        ${d.n_files ?? 0} PDF(s)
        ${d.readonly ? " · read-only" : ""}
      </div>
    </div>`).join("");
  el.querySelectorAll(".ds-item").forEach(node => {
    node.onclick = () => selectDataset(node.dataset.source, node.dataset.id);
  });
}

function renderContent() {
  const el = document.getElementById("content");
  if (!state.selected) {
    el.innerHTML = `<div class="empty">Select a dataset, or create one from the left panel.</div>`;
    return;
  }
  if (state.loading) {
    el.innerHTML = `<div class="empty">Loading ${esc(state.selected.id)}…</div>`;
    return;
  }
  const d = state.detail || state.selected;
  const msg = state.message
    ? `<div class="status-msg ${state.messageKind || ""}">${esc(state.message)}</div>` : "";
  const files = d.files || [];
  const readonly = !!d.readonly;
  const rows = files.map(f => `
    <tr>
      <td class="name">${esc(f.filename)}</td>
      <td>${fmtSize(f.size)}</td>
      <td>${readonly ? "" : `<button type="button" class="danger" data-del-file="${esc(f.filename)}">Remove</button>`}</td>
    </tr>`).join("");
  const inferHref = `/?dataset=${encodeURIComponent(d.id)}&source=${encodeURIComponent(d.source)}`;
  el.innerHTML = `
    <p class="hint">
      ${d.source === "folder"
        ? `Folder dataset from <code>dataset/${esc(d.id)}</code> (used by <code>client_dir.sh</code>). Files are read-only here.`
        : `Managed dataset stored under <code>outputs/datasets/${esc(d.id)}</code>. Add PDFs and reuse it from Inference.`}
    </p>
    ${msg}
    <div class="actions">
      <a href="${inferHref}"><button type="button" class="primary">Run on Inference</button></a>
      ${readonly ? "" : `
        <label class="btn">Add PDFs
          <input type="file" id="addFiles" accept=".pdf,.PDF,application/pdf" multiple hidden />
        </label>
        <button type="button" class="danger" id="deleteDataset">Delete dataset</button>
      `}
    </div>
    <div class="hint">${esc(d.name || d.id)} · ${files.length} file(s)</div>
    <table class="file-table">
      <thead><tr><th>File</th><th>Size</th><th></th></tr></thead>
      <tbody>${rows || `<tr><td colspan="3" class="empty" style="padding:16px">No PDF files</td></tr>`}</tbody>
    </table>`;
  const add = document.getElementById("addFiles");
  if (add) add.onchange = () => addFiles(add.files);
  const delDs = document.getElementById("deleteDataset");
  if (delDs) delDs.onclick = () => deleteDataset();
  el.querySelectorAll("[data-del-file]").forEach(btn => {
    btn.onclick = () => deleteFile(btn.dataset.delFile);
  });
}

async function refreshList() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets || [];
  document.getElementById("headerMeta").textContent =
    `${state.datasets.length} dataset(s) · managed ${state.datasets.filter(d => d.source === "managed").length} · folder ${state.datasets.filter(d => d.source === "folder").length}`;
  renderList();
}

async function selectDataset(source, id) {
  state.selected = { source, id };
  state.message = null;
  state.loading = true;
  syncUrl();
  renderList();
  renderContent();
  try {
    state.detail = await api(`/api/datasets/${encodeURIComponent(source)}/${encodeURIComponent(id)}`);
  } catch (err) {
    state.detail = null;
    state.message = String(err.message || err);
    state.messageKind = "err";
  } finally {
    state.loading = false;
    renderContent();
  }
}

async function createDataset() {
  const name = document.getElementById("newName").value.trim();
  const files = document.getElementById("newFiles").files;
  if (!name) {
    alert("Enter a dataset name.");
    return;
  }
  const form = new FormData();
  form.append("name", name);
  for (const file of files || []) form.append("files", file, file.name);
  try {
    const created = await api("/api/datasets", { method: "POST", body: form });
    document.getElementById("newName").value = "";
    document.getElementById("newFiles").value = "";
    await refreshList();
    await selectDataset(created.source, created.id);
  } catch (err) {
    alert(String(err.message || err));
  }
}

async function addFiles(fileList) {
  if (!state.selected || !fileList || !fileList.length) return;
  const form = new FormData();
  for (const file of fileList) form.append("files", file, file.name);
  try {
    state.detail = await api(
      `/api/datasets/${encodeURIComponent(state.selected.source)}/${encodeURIComponent(state.selected.id)}/files`,
      { method: "POST", body: form }
    );
    state.message = `Added ${fileList.length} file(s)`;
    state.messageKind = "ok";
    await refreshList();
    renderContent();
  } catch (err) {
    state.message = String(err.message || err);
    state.messageKind = "err";
    renderContent();
  }
}

async function deleteFile(filename) {
  if (!state.selected) return;
  if (!confirm(`Remove ${filename} from this dataset?`)) return;
  try {
    state.detail = await api(
      `/api/datasets/${encodeURIComponent(state.selected.source)}/${encodeURIComponent(state.selected.id)}/files/${encodeURIComponent(filename)}`,
      { method: "DELETE" }
    );
    state.message = `Removed ${filename}`;
    state.messageKind = "ok";
    await refreshList();
    renderContent();
  } catch (err) {
    state.message = String(err.message || err);
    state.messageKind = "err";
    renderContent();
  }
}

async function deleteDataset() {
  if (!state.selected) return;
  if (!confirm(`Delete dataset ${state.selected.id}? PDFs in outputs/datasets will be removed.`)) return;
  try {
    await api(
      `/api/datasets/${encodeURIComponent(state.selected.source)}/${encodeURIComponent(state.selected.id)}`,
      { method: "DELETE" }
    );
    state.selected = null;
    state.detail = null;
    await refreshList();
    syncUrl();
    renderContent();
  } catch (err) {
    alert(String(err.message || err));
  }
}

document.getElementById("createBtn").onclick = () => createDataset();
document.getElementById("dsSearch").oninput = (e) => {
  state.filter = e.target.value || "";
  renderList();
};

(async function init() {
  const params = new URLSearchParams(location.search);
  await refreshList();
  const source = params.get("source");
  const id = params.get("id");
  if (source && id && state.datasets.some(d => d.source === source && d.id === id)) {
    await selectDataset(source, id);
  } else if (state.datasets[0]) {
    await selectDataset(state.datasets[0].source, state.datasets[0].id);
  } else {
    renderContent();
  }
})();
</script>
</body>
</html>
"""
