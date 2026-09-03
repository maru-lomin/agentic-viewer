"""Persist named PDF groups and discover existing dataset/ folders."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentic_viewer.eval.paths import REPO_ROOT

MANIFEST_NAME = "manifest.json"
RESERVED_FOLDER_NAMES = {
    "",
    ".",
    "..",
    "docs",  # mixed sample files, not a named eval set
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_dataset_id(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raise ValueError("dataset name is required")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        raise ValueError("dataset name must contain letters, numbers, '.', '_' or '-'")
    if slug in {".", ".."}:
        raise ValueError("invalid dataset name")
    return slug


def default_managed_root() -> Path:
    env = os.environ.get("AGENTIC_DATASETS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    runs_env = os.environ.get("AGENTIC_RUNS_DIR")
    if runs_env:
        return (Path(runs_env).expanduser().resolve().parent / "datasets").resolve()
    return (REPO_ROOT / "outputs" / "datasets").resolve()


def default_folder_root() -> Path:
    env = os.environ.get("AGENTIC_DATASET_FOLDER_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "dataset").resolve()


def _is_pdf_name(name: str) -> bool:
    return Path(name).suffix.lower() == ".pdf"


def _list_pdfs(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and _is_pdf_name(p.name)
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _safe_child(root: Path, name: str) -> Path:
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"invalid name: {name}")
    path = (root / name).resolve()
    if not str(path).startswith(str(root.resolve())):
        raise ValueError(f"invalid name: {name}")
    return path


def _file_row(path: Path, *, uploaded_at: Optional[str] = None) -> Dict[str, Any]:
    stat = path.stat()
    row: Dict[str, Any] = {
        "filename": path.name,
        "size": stat.st_size,
    }
    if uploaded_at:
        row["uploaded_at"] = uploaded_at
    return row


class DatasetStore:
    def __init__(
        self,
        *,
        managed_root: Optional[Path] = None,
        folder_root: Optional[Path] = None,
    ) -> None:
        self.managed_root = (managed_root or default_managed_root()).resolve()
        self.folder_root = (folder_root or default_folder_root()).resolve()

    def list_datasets(self) -> List[Dict[str, Any]]:
        rows = [self._managed_summary(path) for path in self._iter_managed_dirs()]
        rows.extend(self._iter_folder_summaries())
        rows.sort(key=lambda r: (str(r.get("name") or "").lower(), r.get("id") or ""))
        return rows

    def get(self, source: str, dataset_id: str) -> Dict[str, Any]:
        source = self._normalize_source(source)
        dataset_id = slugify_dataset_id(dataset_id)
        if source == "managed":
            return self._read_managed(dataset_id)
        return self._read_folder(dataset_id)

    def create(
        self,
        name: str,
        files: Sequence[Tuple[str, bytes]] = (),
    ) -> Dict[str, Any]:
        dataset_id = slugify_dataset_id(name)
        dest = _safe_child(self.managed_root, dataset_id)
        if dest.exists():
            raise FileExistsError(f"dataset already exists: {dataset_id}")
        files_dir = dest / "files"
        files_dir.mkdir(parents=True, exist_ok=False)
        now = _utc_now()
        manifest: Dict[str, Any] = {
            "id": dataset_id,
            "name": name.strip() or dataset_id,
            "source": "managed",
            "created_at": now,
            "updated_at": now,
            "files": [],
        }
        self._write_manifest(dest, manifest)
        if files:
            self.add_files("managed", dataset_id, files)
            return self.get("managed", dataset_id)
        return self._summary_from_manifest(dest, manifest)

    def add_files(
        self,
        source: str,
        dataset_id: str,
        files: Sequence[Tuple[str, bytes]],
    ) -> Dict[str, Any]:
        source = self._normalize_source(source)
        if source != "managed":
            raise PermissionError("cannot modify folder datasets from the viewer")
        if not files:
            raise ValueError("at least one PDF file is required")
        dataset_id = slugify_dataset_id(dataset_id)
        dest = self._managed_dir(dataset_id)
        files_dir = dest / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest(dest)
        existing = {str(row.get("filename")) for row in (manifest.get("files") or [])}
        now = _utc_now()
        added: List[str] = []
        for filename, data in files:
            filename = Path(filename).name
            if not _is_pdf_name(filename):
                raise ValueError(f"only PDF files are supported: {filename}")
            if not data:
                raise ValueError(f"empty file: {filename}")
            if filename in existing:
                raise FileExistsError(f"file already in dataset: {filename}")
            path = _safe_child(files_dir, filename)
            path.write_bytes(data)
            row = _file_row(path, uploaded_at=now)
            files_list = list(manifest.get("files") or [])
            files_list.append(row)
            manifest["files"] = files_list
            existing.add(filename)
            added.append(filename)
        manifest["updated_at"] = now
        self._write_manifest(dest, manifest)
        summary = self._summary_from_manifest(dest, manifest)
        summary["added"] = added
        return summary

    def delete_file(self, source: str, dataset_id: str, filename: str) -> Dict[str, Any]:
        source = self._normalize_source(source)
        if source != "managed":
            raise PermissionError("cannot modify folder datasets from the viewer")
        dataset_id = slugify_dataset_id(dataset_id)
        dest = self._managed_dir(dataset_id)
        filename = Path(filename).name
        path = _safe_child(dest / "files", filename)
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {filename}")
        path.unlink()
        manifest = self._load_manifest(dest)
        manifest["files"] = [
            row
            for row in (manifest.get("files") or [])
            if str(row.get("filename")) != filename
        ]
        manifest["updated_at"] = _utc_now()
        self._write_manifest(dest, manifest)
        return self._summary_from_manifest(dest, manifest)

    def delete_dataset(self, source: str, dataset_id: str) -> Dict[str, Any]:
        source = self._normalize_source(source)
        if source != "managed":
            raise PermissionError("cannot delete folder datasets from the viewer")
        dataset_id = slugify_dataset_id(dataset_id)
        dest = self._managed_dir(dataset_id)
        shutil.rmtree(dest)
        return {"ok": True, "id": dataset_id, "source": "managed"}

    def pdf_paths(self, source: str, dataset_id: str) -> List[Path]:
        info = self.get(source, dataset_id)
        root = Path(info["path"])
        if info["source"] == "managed":
            root = root / "files"
        return [_safe_child(root, row["filename"]) for row in info.get("files") or []]

    def _normalize_source(self, source: str) -> str:
        value = (source or "").strip().lower()
        if value not in {"managed", "folder"}:
            raise ValueError("source must be 'managed' or 'folder'")
        return value

    def _iter_managed_dirs(self) -> List[Path]:
        if not self.managed_root.is_dir():
            return []
        dirs = [
            p
            for p in self.managed_root.iterdir()
            if p.is_dir() and (p / MANIFEST_NAME).is_file()
        ]
        return sorted(dirs, key=lambda p: p.name.lower())

    def _managed_summary(self, dest: Path) -> Dict[str, Any]:
        try:
            manifest = self._load_manifest(dest)
        except Exception:
            return {
                "id": dest.name,
                "name": dest.name,
                "source": "managed",
                "readonly": False,
                "n_files": 0,
                "path": str(dest),
                "error": "invalid manifest",
            }
        return self._summary_from_manifest(dest, manifest)

    def _iter_folder_summaries(self) -> List[Dict[str, Any]]:
        if not self.folder_root.is_dir():
            return []
        rows: List[Dict[str, Any]] = []
        for child in sorted(self.folder_root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir() or child.name in RESERVED_FOLDER_NAMES:
                continue
            pdfs = _list_pdfs(child)
            if not pdfs:
                continue
            rows.append(
                {
                    "id": child.name,
                    "name": child.name,
                    "source": "folder",
                    "readonly": True,
                    "n_files": len(pdfs),
                    "path": str(child),
                    "created_at": None,
                    "updated_at": None,
                }
            )
        return rows

    def _managed_dir(self, dataset_id: str) -> Path:
        dest = _safe_child(self.managed_root, dataset_id)
        if not dest.is_dir() or not (dest / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"dataset not found: {dataset_id}")
        return dest

    def _read_managed(self, dataset_id: str) -> Dict[str, Any]:
        dest = self._managed_dir(dataset_id)
        return self._summary_from_manifest(dest, self._load_manifest(dest), include_files=True)

    def _read_folder(self, dataset_id: str) -> Dict[str, Any]:
        dest = _safe_child(self.folder_root, dataset_id)
        if not dest.is_dir():
            raise FileNotFoundError(f"dataset folder not found: {dataset_id}")
        pdfs = _list_pdfs(dest)
        if not pdfs:
            raise FileNotFoundError(f"no PDF files in dataset folder: {dataset_id}")
        return {
            "id": dest.name,
            "name": dest.name,
            "source": "folder",
            "readonly": True,
            "n_files": len(pdfs),
            "path": str(dest),
            "created_at": None,
            "updated_at": None,
            "files": [_file_row(p) for p in pdfs],
        }

    def _load_manifest(self, dest: Path) -> Dict[str, Any]:
        path = dest / MANIFEST_NAME
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("manifest must be a JSON object")
        files_dir = dest / "files"
        # Keep manifest rows that still exist; refresh size from disk.
        by_name = {p.name: p for p in _list_pdfs(files_dir)}
        rows: List[Dict[str, Any]] = []
        seen = set()
        for row in data.get("files") or []:
            if not isinstance(row, dict) or not row.get("filename"):
                continue
            name = str(row["filename"])
            path = by_name.get(name)
            if path is None:
                continue
            merged = dict(row)
            merged.update(_file_row(path, uploaded_at=row.get("uploaded_at")))
            rows.append(merged)
            seen.add(name)
        for name, path in by_name.items():
            if name not in seen:
                rows.append(_file_row(path))
        data["files"] = rows
        data["id"] = dest.name
        data["source"] = "managed"
        return data

    def _write_manifest(self, dest: Path, manifest: Dict[str, Any]) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / MANIFEST_NAME
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _summary_from_manifest(
        self,
        dest: Path,
        manifest: Dict[str, Any],
        *,
        include_files: bool = False,
    ) -> Dict[str, Any]:
        files = list(manifest.get("files") or [])
        row: Dict[str, Any] = {
            "id": dest.name,
            "name": manifest.get("name") or dest.name,
            "source": "managed",
            "readonly": False,
            "n_files": len(files),
            "path": str(dest),
            "created_at": manifest.get("created_at"),
            "updated_at": manifest.get("updated_at"),
        }
        if include_files:
            row["files"] = files
        return row
