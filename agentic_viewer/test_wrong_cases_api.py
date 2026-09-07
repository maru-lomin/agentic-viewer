"""Tests for Wrong Cases API handler functions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from agentic_viewer import app as app_module


class WrongCasesApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.runs_root = self.tmp_path / "runs"
        self.runs_root.mkdir()
        self.wrong_cases_file = self.tmp_path / "wrong_cases.json"

        self._orig_runs_root = app_module.RUNS_ROOT
        app_module.RUNS_ROOT = self.runs_root.resolve()

        self.patcher = mock.patch(
            "agentic_viewer.wrong_cases.store.wrong_cases_path",
            return_value=self.wrong_cases_file,
        )
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        app_module.RUNS_ROOT = self._orig_runs_root
        self.tmp.cleanup()

    def _create_mock_run(self, run_id: str, doc_name: str, key: str) -> None:
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(
            json.dumps({
                "run_id": run_id,
                "source_filename": doc_name,
                "dataset_id": "test-ds",
                "dataset_name": "Test Dataset",
                "started_at": "2026-09-07T02:30:00+00:00",
                "status": "ok",
            }),
            encoding="utf-8",
        )
        (run_dir / "05_eval.json").write_text(
            json.dumps({
                "overall": {"value_exact_match": 0.0},
                "per_key": [{
                    "key": key,
                    "value": {"pred": "not_found", "gold": "TIL 없음", "exact_match": False},
                    "search_pages": {"pred": [], "gold": [15], "f1": 0.0},
                    "evidence_text": {
                        "pred": "vlm evidence text",
                        "gold": "gold evidence text",
                        "token_f1": 0.0,
                    },
                    "search_reasons": {"pred": "search reason"},
                }],
            }),
            encoding="utf-8",
        )
        ae_dir = run_dir / "06_agentic_eval"
        ae_dir.mkdir(parents=True, exist_ok=True)
        (ae_dir / "key.json").write_text(
            json.dumps({
                "key": key,
                "status": "done",
                "is_correct_answer": "incorrect",
                "is_valid_gold": "valid",
                "reason_summary": "agentic eval summary",
                "reason_detail": "agentic eval detail",
            }),
            encoding="utf-8",
        )

    def test_pages_route_returns_html(self) -> None:
        html = app_module.wrong_cases_page()
        self.assertIn("Wrong Cases", html)
        self.assertIn("Open in Inference", html)

    def test_create_list_detail_update_delete_workflow(self) -> None:
        self._create_mock_run(
            "agentic-test-1",
            "test_doc.pdf",
            "Turbine TIL 발행 여부(현재 시점 기준)",
        )

        # 1. Create wrong case
        case_data = app_module.api_create_wrong_case({
            "run_id": "agentic-test-1",
            "key": "Turbine TIL 발행 여부(현재 시점 기준)",
            "note": "Investigation note",
        })
        case_id = case_data["id"]
        self.assertEqual(case_data["key"], "Turbine TIL 발행 여부(현재 시점 기준)")
        self.assertEqual(case_data["document"], "test_doc.pdf")
        self.assertEqual(case_data["snapshot"]["pred"], "not_found")
        self.assertEqual(case_data["snapshot"]["gold"], "TIL 없음")
        self.assertEqual(case_data["snapshot"]["agentic_eval"]["is_correct_answer"], "incorrect")

        # 2. List wrong cases
        list_data = app_module.api_list_wrong_cases()
        self.assertEqual(list_data["total"], 1)
        self.assertEqual(list_data["counts"]["open"], 1)
        self.assertEqual(list_data["cases"][0]["id"], case_id)

        # 3. Get detail
        detail_data = app_module.api_get_wrong_case_detail(case_id)
        self.assertEqual(detail_data["case"]["id"], case_id)
        self.assertEqual(detail_data["total_document_runs"], 1)

        # 4. Update status and note
        updated = app_module.api_update_wrong_case(case_id, {
            "status": "resolved",
            "note": "Resolved with better prompt",
        })
        self.assertEqual(updated["status"], "resolved")

        # 5. Delete
        deleted = app_module.api_delete_wrong_case(case_id)
        self.assertTrue(deleted["ok"])

        # 6. Verify deleted
        with self.assertRaises(HTTPException) as ctx:
            app_module.api_get_wrong_case_detail(case_id)
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
