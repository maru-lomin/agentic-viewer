"""Resolve and serve source PDF files for agentic runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_runtime_path(path: Optional[str]) -> Optional[str]:
    """Map container paths to host checkout when the file is not local."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return str(p.resolve())

    text = str(path)
    repo = Path(__file__).resolve().parents[2]  # …/2608_poc_koreanre
    mappings = [
        ("/workspace/dataset/", repo / "dataset"),
        ("/workspace/inference_pipeline/", repo / "inference-pipeline"),
    ]
    for prefix, root in mappings:
        if text.startswith(prefix):
            cand = root / text[len(prefix) :]
            if cand.is_file():
                return str(cand.resolve())
    return path


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def infer_pdf_path(run_dir: Path, result: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Return a readable PDF path for a run, or None."""
    root = Path(run_dir).resolve()
    bundled = root / "00_source.pdf"
    if bundled.is_file():
        return bundled.resolve()

    req = _read_json(root / "00_request.json") or {}
    if not isinstance(req, dict):
        req = {}
    if result is None:
        result = _read_json(root / "04_result.json") or {}
    meta = (result or {}).get("meta") if isinstance(result, dict) else {}
    if not isinstance(meta, dict):
        meta = {}

    candidates = [
        req.get("pdf_path"),
        req.get("file_path"),
        meta.get("pdf_path"),
        meta.get("file_path"),
        meta.get("source_file"),
    ]
    for cand in candidates:
        resolved = resolve_runtime_path(str(cand) if cand else None)
        if resolved and Path(resolved).is_file():
            return Path(resolved).resolve()
    return None


def infer_run_document(
    run_dir: Path,
    *,
    eval_report: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Human-readable source document name for a run."""
    root = Path(run_dir).resolve()
    if isinstance(eval_report, dict):
        doc = eval_report.get("document")
        if doc:
            return str(doc)

    if result is None:
        result = _read_json(root / "04_result.json")
    if isinstance(result, dict):
        meta = result.get("meta") or {}
        if isinstance(meta, dict):
            for key in (
                "source_file",
                "file_name",
                "filename",
                "pdf_name",
                "pdf_path",
                "file_path",
            ):
                val = meta.get(key)
                if val:
                    return Path(str(val)).name

    req = _read_json(root / "00_request.json") or {}
    if isinstance(req, dict):
        for key in ("pdf_path", "file_path"):
            val = req.get(key)
            if val:
                return Path(str(val)).name

    filename = pdf_info(root).get("filename")
    return str(filename) if filename else None


def pdf_info(run_dir: Path) -> Dict[str, Any]:
    """Metadata for the PDF viewer API."""
    root = Path(run_dir).resolve()
    result = _read_json(root / "04_result.json") or {}
    path = infer_pdf_path(root, result if isinstance(result, dict) else None)
    bundled = (root / "00_source.pdf").is_file()
    if path is None:
        return {
            "available": False,
            "bundled": bundled,
            "filename": None,
            "path": None,
        }
    return {
        "available": True,
        "bundled": bundled,
        "filename": path.name,
        "path": str(path),
        "url": f"/api/runs/{root.name}/pdf",
    }
