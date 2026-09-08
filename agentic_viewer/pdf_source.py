"""Resolve and serve source PDF files for agentic runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return REPO_ROOT


def _datasets_roots() -> List[Path]:
    roots: List[Path] = []
    env_managed = os.environ.get("AGENTIC_DATASETS_DIR")
    if env_managed:
        p = Path(env_managed).expanduser().resolve()
        if p not in roots:
            roots.append(p)
    runs_env = os.environ.get("AGENTIC_RUNS_DIR")
    if runs_env:
        p = (Path(runs_env).expanduser().resolve().parent / "datasets").resolve()
        if p not in roots:
            roots.append(p)
    default_managed = (REPO_ROOT / "outputs" / "datasets").resolve()
    if default_managed not in roots:
        roots.append(default_managed)
    return roots


def _folder_roots() -> List[Path]:
    roots: List[Path] = []
    env_folder = os.environ.get("AGENTIC_DATASET_FOLDER_ROOT")
    if env_folder:
        p = Path(env_folder).expanduser().resolve()
        if p not in roots:
            roots.append(p)
    default_folder = (REPO_ROOT / "dataset").resolve()
    if default_folder not in roots:
        roots.append(default_folder)
    return roots


def resolve_runtime_path(path: Optional[str]) -> Optional[str]:
    """Map container paths to host checkout when the file is not local."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return str(p.resolve())

    text = str(path).strip()
    repo = _repo_root()
    mappings = [
        ("/workspace/dataset/", repo / "dataset"),
        ("/workspace/inference_pipeline/", repo / "inference-pipeline"),
        ("/workspace/inference-pipeline/", repo / "inference-pipeline"),
        ("/workspace/outputs/", repo / "outputs"),
    ]
    for prefix, root in mappings:
        if text.startswith(prefix):
            cand = root / text[len(prefix) :]
            if cand.is_file():
                return str(cand.resolve())
    return path


def find_pdf_by_filename(
    filename: str,
    *,
    dataset_id: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Search for a PDF file across run_dir, managed datasets, and dataset folders."""
    name = Path(filename).name.strip()
    if not name:
        return None
    if not name.lower().endswith(".pdf"):
        names = [name, f"{name}.pdf"]
    else:
        names = [name]

    # 1. Check inside run_dir
    if run_dir:
        run_dir = Path(run_dir).resolve()
        for cand_name in names:
            p = run_dir / cand_name
            if p.is_file():
                return p.resolve()

    # 2. Check specific dataset_id if provided
    dataset_roots = _datasets_roots()
    if dataset_id:
        ds_slug = str(dataset_id).strip()
        for root in dataset_roots:
            for cand_name in names:
                cand = root / ds_slug / "files" / cand_name
                if cand.is_file():
                    return cand.resolve()
                cand2 = root / ds_slug / cand_name
                if cand2.is_file():
                    return cand2.resolve()

    # 3. Check all managed datasets
    for root in dataset_roots:
        if not root.is_dir():
            continue
        for cand_name in names:
            matches = list(root.glob(f"*/files/{cand_name}"))
            if matches and matches[0].is_file():
                return matches[0].resolve()
            matches_direct = list(root.glob(f"*/{cand_name}"))
            if matches_direct and matches_direct[0].is_file():
                return matches_direct[0].resolve()

    # 4. Check folder roots (e.g. repo / dataset)
    folder_roots = _folder_roots()
    for root in folder_roots:
        if not root.is_dir():
            continue
        for cand_name in names:
            cand = root / cand_name
            if cand.is_file():
                return cand.resolve()
            sub_matches = list(root.glob(f"*/{cand_name}"))
            if sub_matches and sub_matches[0].is_file():
                return sub_matches[0].resolve()

    # 5. Fallback: check outputs root
    outputs_root = REPO_ROOT / "outputs"
    if outputs_root.is_dir():
        for cand_name in names:
            out_matches = list(outputs_root.glob(f"**/{cand_name}"))
            for m in out_matches:
                if m.is_file() and not m.name.startswith("00_"):
                    return m.resolve()

    return None


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _try_link_bundled(run_dir: Path, source_pdf: Path) -> None:
    """Optionally create 00_source.pdf symlink for fast future resolution."""
    bundled = run_dir / "00_source.pdf"
    if not bundled.exists():
        try:
            bundled.symlink_to(source_pdf.resolve())
        except OSError:
            pass


def infer_pdf_path(
    run_dir: Path,
    result: Optional[Dict[str, Any]] = None,
    eval_report: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Return a readable PDF path for a run, or None."""
    root = Path(run_dir).resolve()
    bundled = root / "00_source.pdf"
    if bundled.is_file():
        return bundled.resolve()

    req = _read_json(root / "00_request.json") or {}
    if not isinstance(req, dict):
        req = {}
    run_meta = _read_json(root / "meta.json") or {}
    if not isinstance(run_meta, dict):
        run_meta = {}
    if result is None:
        result = _read_json(root / "04_result.json") or {}
    meta = (result or {}).get("meta") if isinstance(result, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    if eval_report is None:
        eval_report = _read_json(root / "05_eval.json") or {}
    if not isinstance(eval_report, dict):
        eval_report = {}

    candidates = [
        req.get("pdf_path"),
        req.get("file_path"),
        req.get("source_filename"),
        meta.get("pdf_path"),
        meta.get("file_path"),
        meta.get("source_file"),
        meta.get("source_filename"),
        meta.get("filename"),
        meta.get("file_name"),
        run_meta.get("pdf_path"),
        run_meta.get("file_path"),
        run_meta.get("source_file"),
        run_meta.get("source_filename"),
        run_meta.get("filename"),
        run_meta.get("file_name"),
        eval_report.get("document") if isinstance(eval_report, dict) else None,
    ]

    # 1. Try direct / resolved path first
    for cand in candidates:
        if not cand:
            continue
        cand_str = str(cand).strip()
        resolved = resolve_runtime_path(cand_str)
        if resolved and Path(resolved).is_file():
            found = Path(resolved).resolve()
            _try_link_bundled(root, found)
            return found

    # 2. Search by filename across dataset roots
    dataset_id = req.get("dataset_id") or meta.get("dataset_id") or run_meta.get("dataset_id")
    for cand in candidates:
        if not cand:
            continue
        cand_str = str(cand).strip()
        filename = Path(cand_str).name
        if not filename or filename in {".", ".."}:
            continue
        found = find_pdf_by_filename(filename, dataset_id=dataset_id, run_dir=root)
        if found and found.is_file():
            _try_link_bundled(root, found)
            return found

    # 3. Document name from eval report or other meta
    doc_name = infer_run_document(root, eval_report=eval_report, result=result)
    if doc_name:
        found = find_pdf_by_filename(doc_name, dataset_id=dataset_id, run_dir=root)
        if found and found.is_file():
            _try_link_bundled(root, found)
            return found

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
                "source_filename",
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
        for key in ("source_filename", "pdf_path", "file_path"):
            val = req.get(key)
            if val:
                return Path(str(val)).name

    eval_file = root / "05_eval.json"
    if eval_file.is_file():
        eval_data = _read_json(eval_file)
        if isinstance(eval_data, dict) and eval_data.get("document"):
            return str(eval_data["document"])

    run_meta = _read_json(root / "meta.json") or {}
    if isinstance(run_meta, dict):
        for key in (
            "source_filename",
            "source_file",
            "filename",
            "file_name",
            "pdf_name",
            "pdf_path",
            "file_path",
            "document",
        ):
            val = run_meta.get(key)
            if val:
                return Path(str(val)).name

    bundled = root / "00_source.pdf"
    if bundled.is_file():
        try:
            resolved = bundled.resolve()
            if resolved.is_file() and not resolved.name.startswith("00_"):
                return resolved.name
        except OSError:
            pass

    return None


def pdf_info(run_dir: Path) -> Dict[str, Any]:
    """Metadata for the PDF viewer API."""
    root = Path(run_dir).resolve()
    result = _read_json(root / "04_result.json") or {}
    eval_report = _read_json(root / "05_eval.json") or {}
    path = infer_pdf_path(
        root,
        result if isinstance(result, dict) else None,
        eval_report if isinstance(eval_report, dict) else None,
    )
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
