"""Load, validate, and persist dataset/answer_sheet.json."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_viewer.eval.paths import answer_sheet_path
from agentic_viewer.pdf_source import infer_run_document
from agentic_viewer.timezone import kst_now


def load_answer_sheet() -> Dict[str, Any]:
    path = answer_sheet_path()
    if not path.is_file():
        raise FileNotFoundError(f"answer sheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("answer sheet must be a JSON object")
    return data


def load_answer_sheet_or_empty() -> Dict[str, Any]:
    try:
        return load_answer_sheet()
    except FileNotFoundError:
        return {}


def validate_answer_sheet_payload(payload: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Validate answer_sheet.json shape:

      {
        "<document>.pdf": {
          "<key>": {
            "value": "...",
            "evidences": ["..."],
            "evidence_pages": [1, 2]
          }
        }
      }
    """
    if not isinstance(payload, dict):
        raise ValueError("answer sheet must be a JSON object keyed by document name")
    if not payload:
        raise ValueError("answer sheet is empty")

    normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for doc_name, doc in payload.items():
        name = str(doc_name or "").strip()
        if not name:
            raise ValueError("document name must be a non-empty string")
        if not isinstance(doc, dict):
            raise ValueError(f"document {name!r} must be an object of keys")
        if not doc:
            raise ValueError(f"document {name!r} has no keys")
        keys: Dict[str, Dict[str, Any]] = {}
        for key_name, entry in doc.items():
            key = str(key_name or "").strip()
            if not key:
                raise ValueError(f"empty key under document {name!r}")
            if not isinstance(entry, dict):
                raise ValueError(f"entry for {name!r} / {key!r} must be an object")
            keys[key] = normalize_gt_entry(entry)
        normalized[name] = keys
    return normalized


def import_answer_sheet(
    payload: Any,
    *,
    mode: str = "merge",
) -> Dict[str, Any]:
    """
    Import GT documents from an answer_sheet.json payload.

    mode:
      - merge: add/overwrite keys per document (default)
      - replace: replace the entire answer sheet with the uploaded payload
    """
    mode_name = (mode or "merge").strip().lower()
    if mode_name not in {"merge", "replace"}:
        raise ValueError("mode must be 'merge' or 'replace'")

    incoming = validate_answer_sheet_payload(payload)
    if mode_name == "replace":
        sheet: Dict[str, Any] = {doc: dict(keys) for doc, keys in incoming.items()}
        created_documents = list(incoming.keys())
        updated_documents = []
        added_keys = sum(len(keys) for keys in incoming.values())
        updated_keys = 0
    else:
        sheet = load_answer_sheet_or_empty()
        created_documents = []
        updated_documents = []
        added_keys = 0
        updated_keys = 0
        for doc_name, keys in incoming.items():
            existing = sheet.get(doc_name)
            if not isinstance(existing, dict):
                created_documents.append(doc_name)
                sheet[doc_name] = dict(keys)
                added_keys += len(keys)
                continue
            updated_documents.append(doc_name)
            merged = dict(existing)
            for key, entry in keys.items():
                if key in merged:
                    updated_keys += 1
                else:
                    added_keys += 1
                merged[key] = entry
            sheet[doc_name] = merged

    path = save_answer_sheet(sheet)
    return {
        "path": str(path),
        "mode": mode_name,
        "documents": sorted(incoming.keys()),
        "n_documents": len(incoming),
        "n_keys": sum(len(keys) for keys in incoming.values()),
        "created_documents": created_documents,
        "updated_documents": updated_documents,
        "added_keys": added_keys,
        "updated_keys": updated_keys,
    }


def save_answer_sheet(data: Dict[str, Any]) -> Path:
    path = answer_sheet_path()
    if not isinstance(data, dict):
        raise ValueError("answer sheet must be a JSON object")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        stamp = kst_now().strftime("%Y%m%dT%H%M%S")
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
        return {"document": doc_name, "keys": [], "exists": False}
    keys: List[Dict[str, Any]] = []
    for key, entry in sorted(doc.items(), key=lambda kv: str(kv[0])):
        if not isinstance(entry, dict):
            continue
        normalized = normalize_gt_entry(entry)
        keys.append({"key": str(key), **normalized})
    return {"document": doc_name, "keys": keys, "exists": True}


def update_gt_key(document: str, key: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    doc_name = str(document or "").strip()
    key_name = str(key or "").strip()
    if not doc_name:
        raise ValueError("document is required")
    if not key_name:
        raise ValueError("key is required")

    sheet = load_answer_sheet()
    doc = sheet.get(doc_name)
    created_document = not isinstance(doc, dict)
    if created_document:
        doc = {}
    created_key = key_name not in doc

    normalized = normalize_gt_entry(entry)
    doc[key_name] = normalized
    sheet[doc_name] = doc
    path = save_answer_sheet(sheet)
    return {
        "document": doc_name,
        "key": key_name,
        "entry": normalized,
        "path": str(path),
        "created_document": created_document,
        "created_key": created_key,
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
