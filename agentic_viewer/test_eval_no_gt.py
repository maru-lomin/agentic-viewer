"""Tests for eval scoring when ground truth is missing."""

from __future__ import annotations

import unittest

from agentic_viewer.eval.evaluate_kv import (
    _format_search_reasons,
    build_pred_only_report,
    build_report,
    extract_fallback_reasons,
)


class EvalNoGtTests(unittest.TestCase):
    def test_format_search_reasons_not_found(self) -> None:
        self.assertEqual(
            _format_search_reasons({"not_found": "no turbine"}),
            "not_found: no turbine",
        )
        self.assertEqual(
            _format_search_reasons({"18": "page 18", "not_found": "no turbine"}),
            "p18: page 18\nnot_found: no turbine",
        )
        self.assertEqual(
            _format_search_reasons("direct reason text"),
            "direct reason text",
        )

    def test_fallback_reasons_from_agent_trace(self) -> None:
        pred = {
            "meta": {"source_file": "doc.pdf"},
            "kv_results": [
                {
                    "key": "K_missing",
                    "value": "not_found",
                    "search_reasons": {},
                }
            ],
            "agent_trace": [
                {
                    "tool_calls": [
                        {
                            "name": "collect_search_results",
                            "result_preview": {
                                "completed": [
                                    {
                                        "key": "K_missing",
                                        "status": "not_found",
                                        "reason": "Not in document",
                                    }
                                ]
                            },
                        }
                    ]
                }
            ],
        }
        answer_sheet = {
            "doc.pdf": {
                "K_missing": {"value": "", "evidences": [], "evidence_pages": []},
            }
        }
        report = build_report(pred, answer_sheet)
        row = report["per_key"][0]
        self.assertEqual(
            row["search_reasons"]["pred"],
            "not_found: Not in document",
        )

    def test_not_found_reason_in_kv_results(self) -> None:
        pred = {
            "meta": {"source_file": "doc.pdf"},
            "kv_results": [
                {
                    "key": "K_missing",
                    "value": "not_found",
                    "search_reasons": {"not_found": "Explicit reason"},
                }
            ],
        }
        answer_sheet = {
            "doc.pdf": {
                "K_missing": {"value": "", "evidences": [], "evidence_pages": []},
            }
        }
        report = build_report(pred, answer_sheet)
        row = report["per_key"][0]
        self.assertEqual(
            row["search_reasons"]["pred"],
            "not_found: Explicit reason",
        )

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
