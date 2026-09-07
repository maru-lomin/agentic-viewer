"""Unit tests for wrong cases store."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentic_viewer.wrong_cases.store import (
    add_or_update_wrong_case,
    batch_add_wrong_cases,
    delete_wrong_case,
    get_wrong_case_detail,
    list_wrong_cases,
    load_wrong_cases,
    save_wrong_cases,
    update_wrong_case_status,
)


class WrongCasesStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.wrong_cases_file = self.tmp_path / "wrong_cases.json"
        self.runs_root = self.tmp_path / "runs"
        self.runs_root.mkdir()

        # Patch wrong_cases_path to point to temporary file
        self.patcher = mock.patch(
            "agentic_viewer.wrong_cases.store.wrong_cases_path",
            return_value=self.wrong_cases_file,
        )
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        self.tmp_dir.cleanup()

    def _create_mock_run(
        self,
        run_id: str,
        doc_name: str,
        key: str,
        pred: str = "not_found",
        gold: str = "TIL 없음",
        exact_match: bool = False,
        agentic_correct: str = "incorrect",
    ) -> Path:
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(
            json.dumps({
                "run_id": run_id,
                "source_filename": doc_name,
                "dataset_id": "ds-1",
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
                    "value": {"pred": pred, "gold": gold, "exact_match": exact_match},
                    "search_pages": {"pred": [], "gold": [15], "f1": 0.0},
                    "evidence_text": {
                        "pred": "vlm reason text",
                        "gold": "gold evidence text",
                        "token_f1": 0.0,
                    },
                    "search_reasons": {"pred": "search reason text"},
                }],
            }),
            encoding="utf-8",
        )
        ae_dir = run_dir / "06_agentic_eval"
        ae_dir.mkdir(parents=True, exist_ok=True)
        (ae_dir / "test_key.json").write_text(
            json.dumps({
                "key": key,
                "status": "done",
                "is_correct_answer": agentic_correct,
                "is_valid_gold": "valid",
                "reason_summary": "agentic eval summary text",
                "reason_detail": "agentic eval detail text",
            }),
            encoding="utf-8",
        )
        return run_dir

    def test_add_and_list_wrong_case(self) -> None:
        self._create_mock_run("run-1", "doc1.pdf", "Key 1")
        case = add_or_update_wrong_case(
            self.runs_root,
            "run-1",
            "Key 1",
            note="Failed due to missing page",
        )
        self.assertIsNotNone(case)
        self.assertEqual(case["key"], "Key 1")
        self.assertEqual(case["document"], "doc1.pdf")
        self.assertEqual(case["status"], "open")
        self.assertEqual(case["note"], "Failed due to missing page")
        self.assertEqual(case["snapshot"]["pred"], "not_found")
        self.assertEqual(case["snapshot"]["gold"], "TIL 없음")
        self.assertEqual(case["snapshot"]["exact_match"], False)
        self.assertEqual(case["snapshot"]["vlm_reason"], "vlm reason text")
        self.assertEqual(case["snapshot"]["search_reasons"], "search reason text")
        self.assertEqual(case["snapshot"]["agentic_eval"]["is_correct_answer"], "incorrect")

        # List cases
        res = list_wrong_cases()
        self.assertEqual(res["total"], 1)
        self.assertEqual(res["counts"]["open"], 1)
        self.assertEqual(res["cases"][0]["id"], case["id"])

    def test_update_status_and_delete(self) -> None:
        self._create_mock_run("run-1", "doc1.pdf", "Key 1")
        case = add_or_update_wrong_case(self.runs_root, "run-1", "Key 1")
        cid = case["id"]

        updated = update_wrong_case_status(cid, status="resolved", note="Fixed in new prompt")
        self.assertIsNotNone(updated)
        self.assertEqual(updated["status"], "resolved")
        self.assertEqual(updated["note"], "Fixed in new prompt")

        counts = list_wrong_cases()["counts"]
        self.assertEqual(counts["resolved"], 1)
        self.assertEqual(counts["open"], 0)

        deleted = delete_wrong_case(cid)
        self.assertTrue(deleted)
        self.assertEqual(list_wrong_cases()["total"], 0)

    def test_get_detail_with_improvement_tracking(self) -> None:
        # Run 1: initial run with error
        self._create_mock_run("run-1", "doc1.pdf", "Key 1", pred="not_found", exact_match=False)
        case = add_or_update_wrong_case(self.runs_root, "run-1", "Key 1")

        # Run 2: newer run with fix
        run2_dir = self.runs_root / "run-2"
        run2_dir.mkdir(parents=True, exist_ok=True)
        (run2_dir / "meta.json").write_text(
            json.dumps({
                "run_id": "run-2",
                "source_filename": "doc1.pdf",
                "started_at": "2026-09-07T03:00:00+00:00",
                "status": "ok",
            }),
            encoding="utf-8",
        )
        (run2_dir / "05_eval.json").write_text(
            json.dumps({
                "overall": {"value_exact_match": 1.0},
                "per_key": [{
                    "key": "Key 1",
                    "value": {"pred": "TIL 없음", "gold": "TIL 없음", "exact_match": True},
                    "search_pages": {"pred": [15], "gold": [15], "f1": 1.0},
                    "evidence_text": {"pred": "found evidence", "gold": "gold evidence", "token_f1": 1.0},
                }],
            }),
            encoding="utf-8",
        )

        detail = get_wrong_case_detail(self.runs_root, case["id"])
        self.assertIsNotNone(detail)
        self.assertEqual(detail["case"]["id"], case["id"])
        self.assertEqual(detail["total_document_runs"], 2)

        latest = detail["latest_run"]
        self.assertIsNotNone(latest)
        self.assertEqual(latest["run_id"], "run-2")
        self.assertEqual(latest["exact_match"], True)
        self.assertTrue(latest["improved"])

    def test_batch_add_wrong_cases(self) -> None:
        self._create_mock_run("run-1", "doc1.pdf", "Key 1")
        self._create_mock_run("run-1", "doc1.pdf", "Key 2")
        added = batch_add_wrong_cases(self.runs_root, "run-1", ["Key 1", "Key 2"])
        self.assertEqual(len(added), 2)
        self.assertEqual(list_wrong_cases()["total"], 2)


if __name__ == "__main__":
    unittest.main()
