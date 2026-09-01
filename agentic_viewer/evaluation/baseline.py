"""Load or compute baseline KV eval (05_eval.json) from run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_viewer.eval.evaluate_kv import build_report, load_json
from agentic_viewer.eval.paths import answer_sheet_path


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def eval_cache_has_reason_split(report: Dict[str, Any]) -> bool:
    """True when cached eval separates VLM evidence vs SearchAgent reasons."""
    per_key = report.get("per_key")
    if not isinstance(per_key, list) or not per_key:
        return False
    first = per_key[0]
    if not isinstance(first, dict):
        return False
    return "search_reasons" in first


def load_or_compute_run_eval(
    run_dir: Path,
    *,
    run_id: Optional[str] = None,
    refresh: bool = False,
    write_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Return baseline eval report for a run directory.

    Uses cached ``05_eval.json`` when valid; otherwise scores ``04_result.json``
    against the answer sheet (same logic as the Inference Eval tab).
    """
    cache_path = run_dir / "05_eval.json"
    if cache_path.is_file() and not refresh:
        cached = _read_json(cache_path)
        if (
            isinstance(cached, dict)
            and cached.get("overall")
            and eval_cache_has_reason_split(cached)
        ):
            return cached

    pred_path = run_dir / "04_result.json"
    pred = _read_json(pred_path)
    if not isinstance(pred, dict):
        return None

    ans_path = answer_sheet_path()
    if not ans_path.is_file():
        return None
    answer_sheet = load_json(ans_path)
    if not isinstance(answer_sheet, dict):
        return None

    try:
        report = build_report(
            pred,
            answer_sheet,
            pred_path=str(pred_path),
            answer_sheet_path=str(ans_path),
        )
    except (KeyError, ValueError):
        return None

    if run_id:
        report["run_id"] = run_id
    if write_cache:
        try:
            cache_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            report["cache_write_error"] = str(cache_path)
    return report
