"""Tests for run deletion API."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from agentic_viewer import app as app_module


class DeleteRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.tmp.name) / "runs"
        self.runs_root.mkdir(parents=True)
        self._orig_runs_root = app_module.RUNS_ROOT
        app_module.RUNS_ROOT = self.runs_root.resolve()

    def tearDown(self) -> None:
        app_module.RUNS_ROOT = self._orig_runs_root
        self.tmp.cleanup()

    def _make_run(self, run_id: str) -> Path:
        run_dir = self.runs_root / run_id
        run_dir.mkdir()
        (run_dir / "meta.json").write_text(
            json.dumps({"run_id": run_id, "status": "ok", "finished_at": "2026-01-01T00:00:00Z"}),
            encoding="utf-8",
        )
        return run_dir

    def test_delete_run_removes_directory(self) -> None:
        run_dir = self._make_run("agentic-test-delete")
        result = app_module.delete_run("agentic-test-delete")
        self.assertTrue(result["ok"])
        self.assertFalse(run_dir.exists())

    def test_delete_run_not_found(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            app_module.delete_run("missing-run")
        self.assertEqual(ctx.exception.status_code, 404)

    @patch.object(app_module._BATCH_MANAGER, "get_active_job")
    def test_delete_run_blocked_when_batch_active(self, mock_active) -> None:
        from agentic_viewer.evaluation.batch import BatchJob

        self._make_run("agentic-busy")
        mock_active.return_value = BatchJob(
            job_id="batch-1",
            run_ids=["agentic-busy"],
            skip_existing=True,
            status="running",
        )
        with self.assertRaises(HTTPException) as ctx:
            app_module.delete_run("agentic-busy", force=False)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertTrue((self.runs_root / "agentic-busy").is_dir())

    @patch.object(app_module._BATCH_MANAGER, "get_active_job")
    def test_delete_run_force_when_batch_active(self, mock_active) -> None:
        from agentic_viewer.evaluation.batch import BatchJob

        run_dir = self._make_run("agentic-busy-force")
        mock_active.return_value = BatchJob(
            job_id="batch-1",
            run_ids=["agentic-busy-force"],
            skip_existing=True,
            status="running",
        )
        result = app_module.delete_run("agentic-busy-force", force=True)
        self.assertTrue(result["ok"])
        self.assertFalse(run_dir.exists())

    def test_delete_running_status_run(self) -> None:
        run_dir = self.runs_root / "agentic-running-run"
        run_dir.mkdir()
        (run_dir / "meta.json").write_text(
            json.dumps({"run_id": "agentic-running-run", "status": "running"}),
            encoding="utf-8",
        )
        result = app_module.delete_run("agentic-running-run")
        self.assertTrue(result["ok"])
        self.assertFalse(run_dir.exists())

    def test_delete_runs_batch(self) -> None:
        r1 = self._make_run("batch-run-1")
        r2 = self._make_run("batch-run-2")
        r3 = self._make_run("batch-run-3")

        result = app_module.delete_runs_batch({"run_ids": ["batch-run-1", "batch-run-2"]})
        self.assertEqual(result["total_deleted"], 2)
        self.assertIn("batch-run-1", result["deleted"])
        self.assertIn("batch-run-2", result["deleted"])
        self.assertFalse(r1.exists())
        self.assertFalse(r2.exists())
        self.assertTrue(r3.exists())

    def test_delete_runs_batch_invalid_input(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            app_module.delete_runs_batch({"run_ids": "not-a-list"})
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
