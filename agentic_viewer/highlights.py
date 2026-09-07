"""Chunk highlight regions from run artifacts (layout + chunks.jsonl)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_agentic_path() -> None:
    root = Path(__file__).resolve().parents[2] / "inference-pipeline"
    if root.is_dir():
        p = str(root)
        if p not in sys.path:
            sys.path.insert(0, p)


def _chunk_pages(row: Dict[str, Any]) -> List[int]:
    pages = row.get("pages")
    if isinstance(pages, list) and pages:
        return [int(p) for p in pages if int(p or 0) > 0]
    page = int(row.get("page") or 0)
    page_end = int(row.get("page_end") or page)
    if page <= 0:
        return []
    if page_end < page:
        page, page_end = page_end, page
    return list(range(page, page_end + 1))


def load_chunk_row(run_dir: Path, chunk_id: str) -> Optional[Dict[str, Any]]:
    path = run_dir / "02_chunk" / "chunks.jsonl"
    if not path.is_file():
        return None
    want = str(chunk_id)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and str(row.get("chunk_id") or "") == want:
            return row
    return None


def _enrich_regions(
    regions: List[Dict[str, Any]],
    layouts_by_page: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not regions:
        return []
    _ensure_agentic_path()
    from agentic.layout import enrich_region_bbox_norm

    out: List[Dict[str, Any]] = []
    for region in regions:
        page = int(region.get("page") or 0)
        layout = layouts_by_page.get(page) or {}
        width = int(layout.get("width") or region.get("width") or 0)
        height = int(layout.get("height") or region.get("height") or 0)
        out.append(
            enrich_region_bbox_norm(region, width=width, height=height)
        )
    return out


def chunk_highlights(run_dir: Path, chunk_id: str) -> Dict[str, Any]:
    """
    Return highlight regions for a chunk.

    Uses persisted ``regions`` when present; otherwise recomputes from
    ``01_parse/page_*.layout.json`` files.
    """
    root = Path(run_dir).resolve()
    row = load_chunk_row(root, chunk_id)
    if row is None:
        raise FileNotFoundError(f"chunk not found: {chunk_id}")

    pages = _chunk_pages(row)
    regions = list(row.get("regions") or [])
    page_char_ranges = list(row.get("page_char_ranges") or [])
    source = "persisted" if regions else "none"

    _ensure_agentic_path()
    from agentic.layout import (
        compute_chunk_regions,
        load_layouts_from_run_dir,
        regions_from_page_char_ranges,
    )
    from agentic_viewer.pdf_source import infer_pdf_path

    pdf_path = infer_pdf_path(root)
    layouts = load_layouts_from_run_dir(root, pdf_path=pdf_path)

    if page_char_ranges and layouts:
        regions = regions_from_page_char_ranges(page_char_ranges, layouts)
        source = "page_char_ranges"
    elif not regions and layouts:
        regions = compute_chunk_regions(row, layouts)
        source = "recomputed" if regions else "none"
    elif regions and layouts:
        regions = _enrich_regions(regions, layouts)
        if any(r.get("bbox_norm") for r in regions):
            source = "persisted+enriched" if source == "persisted" else source

    layout_paths: Dict[str, str] = {}
    parse_dir = root / "01_parse"
    if parse_dir.is_dir():
        for page_no in pages:
            rel = f"01_parse/page_{page_no:03d}.layout.json"
            if (root / rel).is_file():
                layout_paths[str(page_no)] = rel

    return {
        "chunk_id": str(row.get("chunk_id") or chunk_id),
        "pages": pages,
        "heading_path": str(row.get("heading_path") or ""),
        "regions": regions,
        "region_count": len(regions),
        "source": source,
        "layout_paths": layout_paths,
    }


def load_chunks_for_page(run_dir: Path, page_no: int) -> List[Dict[str, Any]]:
    """Return all chunk rows in ``02_chunk/chunks.jsonl`` that cover ``page_no``."""
    path = run_dir / "02_chunk" / "chunks.jsonl"
    if not path.is_file():
        return []
    want = int(page_no)
    results: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        pages = _chunk_pages(row)
        if want in pages:
            results.append(row)
    return results


def page_highlights(
    run_dir: Path,
    page_no: int,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return highlight regions and chunk summary for a specific page.

    Aggregates regions from chunks on this page, or matches query text against
    layout elements if query is provided.
    """
    root = Path(run_dir).resolve()
    page_i = int(page_no)
    if page_i <= 0:
        raise ValueError(f"invalid page number: {page_no}")

    chunks = load_chunks_for_page(root, page_i)

    _ensure_agentic_path()
    from agentic.layout import (
        bbox_normalize,
        bbox_union,
        load_layouts_from_run_dir,
        normalize_bbox,
        tokenize,
    )
    from agentic_viewer.pdf_source import infer_pdf_path

    pdf_path = infer_pdf_path(root)
    layouts = load_layouts_from_run_dir(root, pdf_path=pdf_path)
    page_layout = layouts.get(page_i) or {}

    regions: List[Dict[str, Any]] = []
    source = "none"

    # 1) If query text provided, try to match elements on this page
    if query and str(query).strip() and page_layout:
        q_tokens = set(tokenize(str(query)))
        elements = page_layout.get("elements") or []
        matched = []
        for el in elements:
            el_text = str(el.get("text") or "")
            el_tokens = set(tokenize(el_text))
            if q_tokens & el_tokens:
                bbox = normalize_bbox(el.get("bbox") or [])
                if bbox:
                    matched.append(
                        {
                            "element_id": el.get("id"),
                            "class": el.get("class") or "",
                            "bbox": bbox,
                        }
                    )
        if matched:
            union = bbox_union([m["bbox"] for m in matched])
            width = float(page_layout.get("width") or 0)
            height = float(page_layout.get("height") or 0)
            region: Dict[str, Any] = {
                "page": page_i,
                "bbox": union,
                "coord": "image_px",
                "element_ids": [m.get("element_id") for m in matched],
                "n_elements": len(matched),
                "source": "query_match",
            }
            if width > 0 and height > 0:
                region["bbox_norm"] = bbox_normalize(union, width=width, height=height)
                region["width"] = int(width)
                region["height"] = int(height)
            regions.append(region)
            source = "query_match"

    # 2) If no query match regions yet, aggregate regions from chunks on this page
    if not regions and chunks:
        for c in chunks:
            cid = str(c.get("chunk_id") or "")
            try:
                hl = chunk_highlights(root, cid)
                for r in hl.get("regions", []):
                    if int(r.get("page") or 0) == page_i:
                        r_copy = dict(r)
                        r_copy["chunk_id"] = cid
                        regions.append(r_copy)
            except Exception:
                pass
        if regions:
            source = "chunks"

    layout_paths: Dict[str, str] = {}
    rel = f"01_parse/page_{page_i:03d}.layout.json"
    if (root / rel).is_file():
        layout_paths[str(page_i)] = rel

    # Summarize chunk info for UI
    chunk_summaries = []
    for c in chunks:
        text = str(c.get("text") or "")
        chunk_summaries.append({
            "chunk_id": str(c.get("chunk_id") or ""),
            "heading_path": str(c.get("heading_path") or ""),
            "pages": _chunk_pages(c),
            "text": text[:300] + ("…" if len(text) > 300 else ""),
        })

    return {
        "page": page_i,
        "chunk_count": len(chunks),
        "chunks": chunk_summaries,
        "regions": regions,
        "region_count": len(regions),
        "source": source,
        "layout_paths": layout_paths,
    }

