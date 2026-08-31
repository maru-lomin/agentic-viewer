"""Default paths for answer sheet / prediction / eval reports."""

from __future__ import annotations

import os
from pathlib import Path

VIEWER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = VIEWER_ROOT.parent

DEFAULT_PRED = REPO_ROOT / "inference-pipeline" / "outputs" / "result.json"
DEFAULT_ANSWER_SHEET = REPO_ROOT / "dataset" / "answer_sheet.json"
DEFAULT_EVAL_OUT = VIEWER_ROOT / "outputs" / "eval_report.json"


def answer_sheet_path() -> Path:
    env = os.environ.get("AGENTIC_ANSWER_SHEET")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_ANSWER_SHEET.resolve()
