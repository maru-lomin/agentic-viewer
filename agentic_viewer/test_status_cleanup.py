"""Tests for stale running evaluation status cleanup."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_viewer.app import post_cleanup_all_stale_eval, post_cleanup_stale_eval
from agentic_viewer.evaluation_page import EVALUATION_HTML
from agentic_viewer.evaluation.status_cleanup import (
    cleanup_all_running_eval_statuses,
    mark_running_eval_status_cancelled,
)


class StatusCleanupTests(unittest.TestCase):
    def test_mark_running_eval_status_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            eval_dir = run_dir / "06_agentic_eval"
            eval_dir.mkdir(parents=True)

            status_1 = eval_dir / "Key_One.status.json"
            status_1.write_text(
                json.dumps({"key": "Key One", "status": "running"}), encoding="utf-8"
            )

            status_2 = eval_dir / "Key_Two.status.json"
            status_2.write_text(
                json.dumps({"key": "Key Two", "status": "done"}), encoding="utf-8"
            )

            count = mark_running_eval_status_cancelled(run_dir)
            self.assertEqual(count, 1)

            data_1 = json.loads(status_1.read_text(encoding="utf-8"))
            self.assertEqual(data_1["status"], "cancelled")
            self.assertIn("cancelled", data_1["error"])

            data_2 = json.loads(status_2.read_text(encoding="utf-8"))
            self.assertEqual(data_2["status"], "done")

    def test_cleanup_all_running_eval_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            run1 = runs_root / "run-1"
            (run1 / "06_agentic_eval").mkdir(parents=True)
            (run1 / "06_agentic_eval" / "K1.status.json").write_text(
                json.dumps({"key": "K1", "status": "running"}), encoding="utf-8"
            )

            run2 = runs_root / "run-2"
            (run2 / "06_agentic_eval").mkdir(parents=True)
            (run2 / "06_agentic_eval" / "K2.status.json").write_text(
                json.dumps({"key": "K2", "status": "running"}), encoding="utf-8"
            )

            res = cleanup_all_running_eval_statuses(runs_root)
            self.assertEqual(res, {"run-1": 1, "run-2": 1})

    def test_ui_includes_stale_handling(self) -> None:
        # Evaluation dashboard HTML should include clearStaleTasks and Retry (stale) logic
        self.assertIn("clearStaleTasks", EVALUATION_HTML)
        self.assertIn("Retry (stale)", EVALUATION_HTML)


if __name__ == "__main__":
    unittest.main()
