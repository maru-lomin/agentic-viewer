"""Tests for evaluation summary calculations and page rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_viewer.evaluation.summary import build_evaluation_summary
from agentic_viewer.evaluation_page import EVALUATION_HTML


class EvaluationSummaryTests(unittest.TestCase):
    def test_build_evaluation_summary_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            summary = build_evaluation_summary([], runs_root)
            self.assertEqual(summary["run_ids"], [])
            self.assertIsNone(summary["average"])
            self.assertEqual(summary["per_run"], [])
            self.assertEqual(summary["per_key"], [])

    def test_build_evaluation_summary_averages_and_overall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)

            # Create run 1
            run1 = runs_root / "run-1"
            run1.mkdir()
            eval1 = {
                "document": "docA.pdf",
                "has_gt": True,
                "overall": {
                    "value_exact_match": 1.0,
                    "page_f1_macro": 0.8,
                    "evidence_token_f1": 0.6,
                },
                "per_key": [
                    {
                        "key": "K1",
                        "value": {"pred": "v1", "gold": "v1", "exact_match": True},
                        "search_pages": {"f1": 0.8},
                        "evidence_text": {"token_f1": 0.6},
                        "search_reasons": {},
                    },
                    {
                        "key": "K2",
                        "value": {"pred": "v2", "gold": "v2", "exact_match": True},
                        "search_pages": {"f1": 1.0},
                        "evidence_text": {"token_f1": 1.0},
                        "search_reasons": {},
                    },
                ],
            }
            (run1 / "05_eval.json").write_text(json.dumps(eval1), encoding="utf-8")

            # Create run 2
            run2 = runs_root / "run-2"
            run2.mkdir()
            eval2 = {
                "document": "docA.pdf",
                "has_gt": True,
                "overall": {
                    "value_exact_match": 0.5,
                    "page_f1_macro": 0.4,
                    "evidence_token_f1": 0.2,
                },
                "per_key": [
                    {
                        "key": "K1",
                        "value": {"pred": "wrong", "gold": "v1", "exact_match": False},
                        "search_pages": {"f1": 0.4},
                        "evidence_text": {"token_f1": 0.2},
                        "search_reasons": {},
                    },
                    {
                        "key": "K2",
                        "value": {"pred": "v2", "gold": "v2", "exact_match": True},
                        "search_pages": {"f1": 0.6},
                        "evidence_text": {"token_f1": 0.4},
                        "search_reasons": {},
                    },
                ],
            }
            (run2 / "05_eval.json").write_text(json.dumps(eval2), encoding="utf-8")

            summary = build_evaluation_summary(["run-1", "run-2"], runs_root)

            # Test per-run list
            self.assertEqual(len(summary["per_run"]), 2)

            # Test average metrics
            avg = summary["average"]
            self.assertIsNotNone(avg)
            self.assertAlmostEqual(avg["value_exact_match"], 0.75)  # (1.0 + 0.5) / 2
            self.assertAlmostEqual(avg["page_f1_macro"], 0.6)  # (0.8 + 0.4) / 2
            self.assertAlmostEqual(avg["evidence_token_f1"], 0.4)  # (0.6 + 0.2) / 2

            # Test per_key overall EM calculation
            keys_by_name = {row["key"]: row for row in summary["per_key"]}
            self.assertIn("K1", keys_by_name)
            self.assertIn("K2", keys_by_name)

            k1_ov = keys_by_name["K1"]["overall"]
            self.assertEqual(k1_ov["correct"], 1)  # run-1 is True, run-2 is False
            self.assertEqual(k1_ov["incorrect"], 1)
            self.assertEqual(k1_ov["total"], 2)
            self.assertAlmostEqual(k1_ov["rate"], 0.5)

            k2_ov = keys_by_name["K2"]["overall"]
            self.assertEqual(k2_ov["correct"], 2)  # both True
            self.assertEqual(k2_ov["incorrect"], 0)
            self.assertEqual(k2_ov["total"], 2)
            self.assertAlmostEqual(k2_ov["rate"], 1.0)

    def test_evaluation_html_includes_avg_and_overall(self) -> None:
        # Check that Overall column header replaces Gold value
        self.assertIn('data-col="overall">Overall</th>', EVALUATION_HTML)
        self.assertNotIn("<th>Gold value</th>", EVALUATION_HTML)

        # Check that average summary row logic is present
        self.assertIn("summary-avg-row", EVALUATION_HTML)
        self.assertIn("avgRow", EVALUATION_HTML)

        # Check that overall cell renders EM Rate and Correct/Incorrect counts
        self.assertIn("matrix-overall-cell", EVALUATION_HTML)
        self.assertIn("EM Rate:", EVALUATION_HTML)
        self.assertIn("Correct:", EVALUATION_HTML)
        self.assertIn("Incorrect:", EVALUATION_HTML)


if __name__ == "__main__":
    unittest.main()
