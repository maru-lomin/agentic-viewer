"""HTML template loader for Agentic Viewer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=16)
def get_template(filename: str) -> str:
    """Read and cache an HTML template from the templates directory."""
    template_path = TEMPLATES_DIR / filename
    return template_path.read_text(encoding="utf-8")
