"""Load, validate, and persist dataset/answer_sheet.json."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_viewer.eval.paths import answer_sheet_path
from agentic_viewer.pdf_source import infer_run_document


def load_answer_sheet() -> Dict[str, Any]:
    path = answer_sheet_path()
    if not path.is_file():
        raise FileNotFoundError(f"answer sheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("answer sheet must be a JSON object")
    return data


def save_answer_sheet(data: Dict[str, Any]) -> Path:
    path = answer_sheet_path()
    if not isinstance(data, dict):
        raise ValueError("answer sheet must be a JSON object")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = path.with_name(f"{path.stem}.bak.{stamp}{path.suffix}")
        shutil.copy2(path, backup)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def normalize_gt_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("GT entry must be an object")
    value = str(raw.get("value") if raw.get("value") is not None else "")
    evidences_raw = raw.get("evidences")
    if evidences_raw is None:
        evidences: List[str] = []
    elif isinstance(evidences_raw, list):
        evidences = [str(x).strip() for x in evidences_raw if str(x).strip()]
    else:
        raise ValueError("evidences must be a list of strings")

    pages_raw = raw.get("evidence_pages")
    if pages_raw is None:
        pages: List[int] = []
    elif isinstance(pages_raw, list):
        pages = []
        for item in pages_raw:
            if item is None or item == "":
                continue
            try:
                pages.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid evidence page: {item!r}") from exc
    else:
        raise ValueError("evidence_pages must be a list of integers")

    return {
        "value": value,
        "evidences": evidences,
        "evidence_pages": pages,
    }


def list_documents() -> List[Dict[str, Any]]:
    sheet = load_answer_sheet()
    rows: List[Dict[str, Any]] = []
    for name in sorted(sheet):
        doc = sheet.get(name)
        n_keys = len(doc) if isinstance(doc, dict) else 0
        rows.append({"document": str(name), "n_keys": n_keys})
    return rows


def get_document_gt(document: str) -> Dict[str, Any]:
    doc_name = str(document or "").strip()
    if not doc_name:
        raise ValueError("document is required")
    sheet = load_answer_sheet()
    doc = sheet.get(doc_name)
    if not isinstance(doc, dict):
        raise KeyError(f"document not found: {doc_name}")
    keys: List[Dict[str, Any]] = []
    for key, entry in sorted(doc.items(), key=lambda kv: str(kv[0])):
        if not isinstance(entry, dict):
            continue
        normalized = normalize_gt_entry(entry)
        keys.append({"key": str(key), **normalized})
    return {"document": doc_name, "keys": keys}


def update_gt_key(document: str, key: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    doc_name = str(document or "").strip()
    key_name = str(key or "").strip()
    if not doc_name:
        raise ValueError("document is required")
    if not key_name:
        raise ValueError("key is required")

    sheet = load_answer_sheet()
    doc = sheet.get(doc_name)
    if not isinstance(doc, dict):
        raise KeyError(f"document not found: {doc_name}")
    if key_name not in doc:
        raise KeyError(f"key not found: {key_name}")

    normalized = normalize_gt_entry(entry)
    doc[key_name] = normalized
    sheet[doc_name] = doc
    path = save_answer_sheet(sheet)
    return {
        "document": doc_name,
        "key": key_name,
        "entry": normalized,
        "path": str(path),
    }


def invalidate_eval_caches_for_document(
    runs_root: Path,
    document: str,
) -> int:
    """Remove cached 05_eval.json for runs that match the document."""
    doc_name = str(document or "").strip()
    if not doc_name or not runs_root.is_dir():
        return 0
    removed = 0
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        cache = child / "05_eval.json"
        if not cache.is_file():
            continue
        cached_doc: Optional[str] = None
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cached_doc = payload.get("document")
        except (OSError, json.JSONDecodeError):
            cached_doc = None
        if cached_doc == doc_name or infer_run_document(child) == doc_name:
            try:
                cache.unlink()
                removed += 1
            except OSError:
                pass
    return removed
