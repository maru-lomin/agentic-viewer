"""KV evaluation metrics and scoring (integrated into agentic-viewer)."""

from .metrics import as_page_set, exact_match, page_prf, token_f1

__all__ = [
    "as_page_set",
    "build_report",
    "evaluate_document",
    "exact_match",
    "page_prf",
    "token_f1",
]


def __getattr__(name: str):
    if name in {"build_report", "evaluate_document"}:
        from .evaluate_kv import build_report, evaluate_document

        return {
            "build_report": build_report,
            "evaluate_document": evaluate_document,
        }[name]
    raise AttributeError(name)
