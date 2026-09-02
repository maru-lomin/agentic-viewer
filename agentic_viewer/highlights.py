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
        width = int(region.get("width") or layout.get("width") or 0)
        height = int(region.get("height") or layout.get("height") or 0)
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

    layouts = load_layouts_from_run_dir(root)

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
