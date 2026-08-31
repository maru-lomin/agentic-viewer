#!/usr/bin/env python3
"""Evaluate agentic KV results against answer_sheet.json.

Metrics (baseline):
  - value: exact match
  - search pages: precision / recall / F1 vs gold evidence_pages
  - evidence text: unordered whitespace token F1 vs gold evidences
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .metrics import as_page_set, exact_match, page_prf, token_f1
from .paths import (
    DEFAULT_ANSWER_SHEET,
    DEFAULT_EVAL_OUT,
    DEFAULT_PRED,
    answer_sheet_path,
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _join_evidence_texts(items: Sequence[Any]) -> str:
    parts: List[str] = []
    for item in items or []:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(
                item.get("text") or item.get("evidence_quote") or ""
            ).strip()
        else:
            text = ""
        if text:
            parts.append(text)
    return "\n".join(parts)


def _format_search_reasons(reasons: Any) -> str:
    """Format SearchAgent page_reasons as ``pN: reason`` lines."""
    if not reasons:
        return ""
    if isinstance(reasons, str):
        return reasons.strip()
    if isinstance(reasons, dict):
        def _page_key(item: Any) -> tuple:
            key = str(item)
            return (0, int(key)) if key.isdigit() else (1, key)

        lines: List[str] = []
        for page, text in sorted(reasons.items(), key=lambda kv: _page_key(kv[0])):
            note = str(text or "").strip()
            if note:
                lines.append(f"p{page}: {note}")
        return "\n".join(lines)
    if isinstance(reasons, list):
        return _join_evidence_texts(reasons)
    return str(reasons).strip()


def resolve_document_name(
    pred: Dict[str, Any],
    answer_sheet: Dict[str, Any],
    override: Optional[str] = None,
) -> str:
    if override:
        return override

    meta = pred.get("meta") or {}
    candidates: List[str] = []
    for key in ("source_file", "file_name", "filename", "pdf_name"):
        val = meta.get(key)
        if val:
            candidates.append(Path(str(val)).name)
    pdf_path = meta.get("pdf_path") or meta.get("file_path")
    if pdf_path:
        candidates.append(Path(str(pdf_path)).name)

    for name in candidates:
        if name in answer_sheet:
            return name

    if len(answer_sheet) == 1:
        return next(iter(answer_sheet))

    available = ", ".join(sorted(answer_sheet))
    raise KeyError(
        "Could not resolve document name for answer_sheet. "
        f"Tried {candidates or ['(none)']}. Available: {available}. "
        "Pass --document explicitly."
    )


def index_pred_by_key(
    kv_results: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in kv_results or []:
        if not isinstance(row, dict) or "key" not in row:
            continue
        out[str(row["key"])] = row
    return out


def evaluate_document(
    pred_rows: Dict[str, Dict[str, Any]],
    gold_doc: Dict[str, Any],
) -> Dict[str, Any]:
    per_key: List[Dict[str, Any]] = []
    em_flags: List[float] = []
    page_p: List[float] = []
    page_r: List[float] = []
    page_f: List[float] = []
    evid_f: List[float] = []
    micro_tp = micro_pred = micro_gold = 0

    for key, gold in gold_doc.items():
        if not isinstance(gold, dict):
            continue
        pred = pred_rows.get(key) or {}

        gold_value = str(gold.get("value") or "")
        pred_value = str(pred.get("value") or "")
        em = exact_match(pred_value, gold_value)

        gold_pages = gold.get("evidence_pages") or []
        pred_pages = pred.get("search_pages")
        if pred_pages is None:
            pred_pages = [
                e.get("page")
                for e in (pred.get("evidence") or [])
                if isinstance(e, dict)
            ]
        p_prec, p_rec, p_f1 = page_prf(pred_pages or [], gold_pages)

        pred_set = as_page_set(pred_pages or [])
        gold_set = as_page_set(gold_pages)
        micro_tp += len(pred_set & gold_set)
        micro_pred += len(pred_set)
        micro_gold += len(gold_set)

        gold_evid = _join_evidence_texts(gold.get("evidences") or [])
        # VLM extract_kv_vlm evidence_quote (not SearchAgent page_reasons).
        pred_evid = _join_evidence_texts(pred.get("evidence") or [])
        e_f1 = token_f1(pred_evid, gold_evid)
        search_reasons = _format_search_reasons(
            pred.get("search_reasons") or pred.get("page_reasons") or {}
        )

        em_flags.append(1.0 if em else 0.0)
        page_p.append(p_prec)
        page_r.append(p_rec)
        page_f.append(p_f1)
        evid_f.append(e_f1)

        per_key.append(
            {
                "key": key,
                "value": {
                    "pred": pred_value,
                    "gold": gold_value,
                    "exact_match": em,
                },
                "search_pages": {
                    "pred": sorted(pred_set),
                    "gold": sorted(gold_set),
                    "precision": round(p_prec, 6),
                    "recall": round(p_rec, 6),
                    "f1": round(p_f1, 6),
                },
                "evidence_text": {
                    # VLM evidence_quote only — used for token F1 vs gold evidences.
                    "pred": pred_evid,
                    "gold": gold_evid,
                    "token_f1": round(e_f1, 6),
                    "source": "vlm_evidence_quote",
                },
                "search_reasons": {
                    # SearchAgent submit_pages page_reasons (page selection rationale).
                    "pred": search_reasons,
                    "source": "search_agent_page_reasons",
                },
            }
        )

    if micro_pred == 0 and micro_gold == 0:
        micro_precision = micro_recall = micro_f1 = 1.0
    else:
        micro_precision = (micro_tp / micro_pred) if micro_pred else 0.0
        micro_recall = (micro_tp / micro_gold) if micro_gold else 0.0
        if micro_precision + micro_recall == 0:
            micro_f1 = 0.0
        else:
            micro_f1 = (
                2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
            )

    return {
        "n_keys": len(per_key),
        "overall": {
            "value_exact_match": round(_mean(em_flags), 6),
            "page_precision_macro": round(_mean(page_p), 6),
            "page_recall_macro": round(_mean(page_r), 6),
            "page_f1_macro": round(_mean(page_f), 6),
            "page_precision_micro": round(micro_precision, 6),
            "page_recall_micro": round(micro_recall, 6),
            "page_f1_micro": round(micro_f1, 6),
            "evidence_token_f1": round(_mean(evid_f), 6),
        },
        "per_key": per_key,
    }


def build_report(
    pred: Dict[str, Any],
    answer_sheet: Dict[str, Any],
    *,
    document: Optional[str] = None,
    pred_path: Optional[str] = None,
    answer_sheet_path: Optional[str] = None,
) -> Dict[str, Any]:
    doc_name = resolve_document_name(pred, answer_sheet, document)
    gold_doc = answer_sheet[doc_name]
    if not isinstance(gold_doc, dict):
        raise ValueError(f"invalid gold entry for {doc_name}")

    pred_rows = index_pred_by_key(pred.get("kv_results") or [])
    scored = evaluate_document(pred_rows, gold_doc)
    return {
        "document": doc_name,
        "pred_path": pred_path,
        "answer_sheet_path": answer_sheet_path,
        **scored,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def print_summary(report: Dict[str, Any]) -> None:
    overall = report["overall"]
    print(f"document: {report['document']}")
    print(f"n_keys:   {report['n_keys']}")
    print(f"value exact match:     {overall['value_exact_match']:.4f}")
    print(
        "page P/R/F1 (macro):   "
        f"{overall['page_precision_macro']:.4f} / "
        f"{overall['page_recall_macro']:.4f} / "
        f"{overall['page_f1_macro']:.4f}"
    )
    print(
        "page P/R/F1 (micro):   "
        f"{overall['page_precision_micro']:.4f} / "
        f"{overall['page_recall_micro']:.4f} / "
        f"{overall['page_f1_micro']:.4f}"
    )
    print(f"evidence token F1:     {overall['evidence_token_f1']:.4f}")
    print()
    print(f"{'key':<55} {'EM':>3} {'pageF1':>7} {'evidF1':>7}")
    print("-" * 75)
    for row in report["per_key"]:
        print(
            f"{row['key']:<55} "
            f"{'Y' if row['value']['exact_match'] else 'N':>3} "
            f"{row['search_pages']['f1']:>7.3f} "
            f"{row['evidence_text']['token_f1']:>7.3f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate agentic KV extraction against answer_sheet.json"
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=DEFAULT_PRED,
        help="Prediction JSON from client.py "
        "(default: ../inference-pipeline/outputs/result.json)",
    )
    parser.add_argument(
        "--answer-sheet",
        type=Path,
        default=None,
        help="Ground-truth answer sheet JSON "
        f"(default: {DEFAULT_ANSWER_SHEET} or $AGENTIC_ANSWER_SHEET)",
    )
    parser.add_argument(
        "--document",
        default=None,
        help="Document key in answer_sheet (default: infer from pred meta)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVAL_OUT,
        help="Where to write the eval report JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    pred_path = args.pred.expanduser().resolve()
    answer_path = (
        args.answer_sheet.expanduser().resolve()
        if args.answer_sheet
        else answer_sheet_path()
    )
    out_path = args.output.expanduser().resolve()

    if not pred_path.exists():
        print(f"prediction not found: {pred_path}", file=sys.stderr)
        return 1
    if not answer_path.exists():
        print(f"answer sheet not found: {answer_path}", file=sys.stderr)
        return 1

    pred = load_json(pred_path)
    answer_sheet = load_json(answer_path)
    if not isinstance(pred, dict) or not isinstance(answer_sheet, dict):
        print("pred and answer_sheet must be JSON objects", file=sys.stderr)
        return 1

    try:
        report = build_report(
            pred,
            answer_sheet,
            document=args.document,
            pred_path=str(pred_path),
            answer_sheet_path=str(answer_path),
        )
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(report)
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
