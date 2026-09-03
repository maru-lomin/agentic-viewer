"""Tests for eval scoring when ground truth is missing."""

from __future__ import annotations

import unittest

from agentic_viewer.eval.evaluate_kv import build_pred_only_report, build_report


class EvalNoGtTests(unittest.TestCase):
    def test_build_pred_only_report(self) -> None:
        pred = {
            "meta": {"source_file": "unknown.pdf"},
            "kv_results": [
                {
                    "key": "Field A",
                    "value": "pred value",
                    "search_pages": [1, 2],
                    "evidence": [{"text": "quote"}],
                }
            ],
        }
        report = build_pred_only_report(pred, "unknown.pdf")
        self.assertFalse(report["has_gt"])
        self.assertEqual(report["document"], "unknown.pdf")
        self.assertEqual(report["n_keys"], 1)
        row = report["per_key"][0]
        self.assertEqual(row["key"], "Field A")
        self.assertEqual(row["value"]["pred"], "pred value")
        self.assertEqual(row["value"]["gold"], "")
        self.assertFalse(row["value"]["exact_match"])

    def test_build_report_marks_has_gt(self) -> None:
        pred = {
            "meta": {"source_file": "doc.pdf"},
            "kv_results": [{"key": "K", "value": "x"}],
        }
        answer_sheet = {
            "doc.pdf": {
                "K": {"value": "x", "evidences": [], "evidence_pages": []},
            }
        }
        report = build_report(pred, answer_sheet)
        self.assertTrue(report["has_gt"])
        self.assertEqual(report["overall"]["value_exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
