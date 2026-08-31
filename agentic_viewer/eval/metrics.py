"""Baseline metrics for agentic KV evaluation.

- Value: exact match
- Search pages: set precision / recall / F1
- Evidence text: unordered whitespace token F1 (multiset / Counter)
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence, Set, Tuple


def normalize_value(text: str) -> str:
    return " ".join(str(text or "").split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_value(pred) == normalize_value(gold)


def tokenize(text: str) -> Counter:
    tokens = normalize_value(text).split()
    return Counter(tokens)


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-tokens F1 (order ignored). Both empty -> 1.0."""
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    overlap = sum((pred_toks & gold_toks).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_toks.values())
    recall = overlap / sum(gold_toks.values())
    return 2.0 * precision * recall / (precision + recall)


def as_page_set(pages: Iterable) -> Set[int]:
    out: Set[int] = set()
    for p in pages or []:
        try:
            pi = int(p)
        except (TypeError, ValueError):
            continue
        if pi > 0:
            out.add(pi)
    return out


def page_prf(
    pred_pages: Sequence,
    gold_pages: Sequence,
) -> Tuple[float, float, float]:
    """Page-set precision, recall, F1. Both empty -> (1, 1, 1)."""
    pred = as_page_set(pred_pages)
    gold = as_page_set(gold_pages)
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = (tp / len(pred)) if pred else 1.0
    recall = (tp / len(gold)) if gold else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1
