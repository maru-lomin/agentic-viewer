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

    def test_extract_searched_pages_with_bm25_queries(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from agentic_viewer.evaluation.baseline import (
            extract_searched_pages,
            enrich_eval_with_searched_pages,
            eval_cache_has_searched_pages,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pred = {
                "search_agent_traces": [
                    {
                        "key": "TestKey",
                        "prior_context_out": {
                            "pages_inspected": [10, 11],
                            "candidate_pages": [{"page": 10}, {"page": 12}],
                            "bm25_hits": [
                                {
                                    "query": "search query 1",
                                    "chunk_id": "10-1",
                                    "page": 10,
                                    "score": 15.5,
                                },
                                {
                                    "query": "search query 1",
                                    "chunk_id": "11-2",
                                    "page": 11,
                                    "score": 12.3,
                                },
                                {
                                    "query": "search query 2",
                                    "chunk_id": "12-1",
                                    "page": 12,
                                    "score": 9.8,
                                },
                            ],
                        },
                    }
                ]
            }
            (tmp_dir / "04_result.json").write_text(json.dumps(pred), encoding="utf-8")

            searched = extract_searched_pages(tmp_dir)
            self.assertIn("TestKey", searched)
            self.assertEqual(searched["TestKey"]["inspected"], [10, 11])
            self.assertEqual(searched["TestKey"]["bm25"], [10, 11, 12])
            queries = searched["TestKey"]["bm25_queries"]
            self.assertEqual(len(queries), 2)
            self.assertEqual(queries[0]["query"], "search query 1")
            self.assertEqual(len(queries[0]["hits"]), 2)
            self.assertEqual(queries[0]["hits"][0]["chunk_id"], "10-1")
            self.assertEqual(queries[1]["query"], "search query 2")
            self.assertEqual(len(queries[1]["hits"]), 1)
            self.assertEqual(queries[1]["hits"][0]["chunk_id"], "12-1")

            # Test enrich_eval_with_searched_pages
            eval_report = {
                "per_key": [
                    {
                        "key": "TestKey",
                        "search_pages": {"pred": [10], "gold": [10]},
                    }
                ]
            }
            enriched = enrich_eval_with_searched_pages(eval_report, tmp_dir)
            sp = enriched["per_key"][0]["search_pages"]
            self.assertEqual(sp["inspected"], [10, 11])
            self.assertEqual(sp["other_inspected"], [11])
            self.assertEqual(len(sp["bm25_queries"]), 2)
            self.assertTrue(eval_cache_has_searched_pages(enriched))


if __name__ == "__main__":
    unittest.main()
