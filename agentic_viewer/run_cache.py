"""Lightweight caching layer for run summaries to avoid parsing large 04_result.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# In-memory cache mapping run_dir path -> (fingerprint_tuple, row_dict)
_RUN_ROW_CACHE: Dict[str, Tuple[Tuple[float, ...], Dict[str, Any]]] = {}


def get_run_fingerprint(run_dir: Path) -> Tuple[float, ...]:
    """Fast stat-based fingerprint of key run files."""
    try:
        meta_mtime = (run_dir / "meta.json").stat().st_mtime
    except OSError:
        meta_mtime = 0.0

    try:
        res_mtime = (run_dir / "04_result.json").stat().st_mtime
    except OSError:
        res_mtime = 0.0

    try:
        eval_mtime = (run_dir / "05_eval.json").stat().st_mtime
    except OSError:
        eval_mtime = 0.0

    try:
        agentic_mtime = (run_dir / "06_agentic_eval").stat().st_mtime
    except OSError:
        agentic_mtime = 0.0

    return (meta_mtime, res_mtime, eval_mtime, agentic_mtime)


def get_cached_run_row(
    run_dir: Path,
    compute_fn: Callable[[Path], Dict[str, Any]],
) -> Dict[str, Any]:
    """Return cached run row if fingerprint matches, else recompute and cache."""
    key = str(run_dir.resolve())
    fp = get_run_fingerprint(run_dir)

    cached = _RUN_ROW_CACHE.get(key)
    if cached is not None and cached[0] == fp:
        return dict(cached[1])

    # Check on-disk cache if exists
    cache_file = run_dir / ".run_summary.json"
    if cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("_fingerprint") == list(fp):
                row = data.get("row")
                if isinstance(row, dict):
                    _RUN_ROW_CACHE[key] = (fp, row)
                    return dict(row)
        except Exception:
            pass

    # Recompute
    row = compute_fn(run_dir)

    # Save to in-memory cache
    _RUN_ROW_CACHE[key] = (fp, row)

    # If run is completed, persist to on-disk cache
    if row.get("finished_at"):
        try:
            cache_file.write_text(
                json.dumps({"_fingerprint": list(fp), "row": row}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    return dict(row)


def invalidate_run_cache(run_dir: Optional[Path] = None) -> None:
    """Invalidate cache for a specific run or all runs."""
    if run_dir is None:
        _RUN_ROW_CACHE.clear()
    else:
        key = str(run_dir.resolve())
        _RUN_ROW_CACHE.pop(key, None)
        try:
            (run_dir / ".run_summary.json").unlink(missing_ok=True)
        except OSError:
            pass
