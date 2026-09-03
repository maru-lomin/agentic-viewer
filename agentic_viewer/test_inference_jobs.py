"""Tests for inference upload job manager."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_viewer.inference_jobs import InferenceJobManager


class InferenceJobManagerTests(unittest.TestCase):
    def test_rejects_non_pdf(self) -> None:
        mgr = InferenceJobManager("http://127.0.0.1:8010")
        with self.assertRaises(ValueError):
            mgr.start([("notes.txt", b"hello")])

    def test_rejects_empty_file_list(self) -> None:
        mgr = InferenceJobManager("http://127.0.0.1:8010")
        with self.assertRaises(ValueError):
            mgr.start([])

    @patch("agentic_viewer.inference_jobs.wait_for_inference_api")
    @patch("agentic_viewer.inference_jobs.invoke_inference")
    def test_runs_files_sequentially(self, mock_invoke, mock_wait) -> None:
        mock_invoke.side_effect = [
            {"kv_results": [{"key": "a"}], "meta": {"run_id": "run-1", "seconds": 1.2}},
            {"kv_results": [], "meta": {"run_id": "run-2", "seconds": 2.1}},
        ]
        mgr = InferenceJobManager("http://127.0.0.1:8010")
        job = mgr.start(
            [
                ("one.pdf", b"%PDF-1"),
                ("two.pdf", b"%PDF-2"),
            ]
        )
        job_id = job.job_id

        import time

        deadline = time.time() + 5
        while time.time() < deadline:
            job = mgr.get_job(job_id)
            assert job is not None
            if job.status in {"done", "partial", "error"}:
                break
            time.sleep(0.05)
        else:
            self.fail("job did not finish")

        self.assertEqual(job.status, "done")
        self.assertEqual(job.tasks[0].status, "done")
        self.assertEqual(job.tasks[0].run_id, "run-1")
        self.assertEqual(job.tasks[1].run_id, "run-2")
        self.assertEqual(mock_invoke.call_count, 2)
        mock_wait.assert_called_once()

    @patch("agentic_viewer.inference_jobs.wait_for_inference_api")
    @patch("agentic_viewer.inference_jobs.invoke_inference")
    def test_runs_from_paths_and_annotates_dataset(self, mock_invoke, mock_wait) -> None:
        import json
        import tempfile
        import time
        from pathlib import Path

        tmp = tempfile.TemporaryDirectory()
        try:
            root = Path(tmp.name)
            pdf = root / "doc.pdf"
            pdf.write_bytes(b"%PDF-1")
            runs_root = root / "runs"
            runs_root.mkdir()

            def fake_invoke(*_args, **kwargs):
                run_id = kwargs.get("request_id") or "run-ds-1"
                dest = runs_root / run_id
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / "meta.json"
                meta = {}
                if path.is_file():
                    meta = json.loads(path.read_text(encoding="utf-8"))
                meta["run_id"] = run_id
                path.write_text(json.dumps(meta), encoding="utf-8")
                return {"kv_results": [], "meta": {"run_id": run_id, "seconds": 0.5}}

            mock_invoke.side_effect = fake_invoke
            mgr = InferenceJobManager("http://127.0.0.1:8010", runs_root=runs_root)
            job = mgr.start(
                paths=[pdf],
                dataset_id="evaluation-v2",
                dataset_name="evaluation-v2",
                dataset_source="folder",
            )
            deadline = time.time() + 5
            while time.time() < deadline:
                job = mgr.get_job(job.job_id)
                assert job is not None
                if job.status in {"done", "partial", "error"}:
                    break
                time.sleep(0.05)
            else:
                self.fail("job did not finish")
            self.assertEqual(job.status, "done")
            run_id = job.tasks[0].run_id
            self.assertTrue(run_id)
            meta = json.loads((runs_root / run_id / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["dataset_id"], "evaluation-v2")
            self.assertEqual(meta["source_filename"], "doc.pdf")
            extra = mock_invoke.call_args.kwargs.get("extra") or {}
            self.assertEqual(extra.get("dataset_id"), "evaluation-v2")
        finally:
            tmp.cleanup()

    @patch("agentic_viewer.inference_jobs.wait_for_inference_api")
    def test_marks_all_tasks_error_when_api_unavailable(self, mock_wait) -> None:
        from agentic_viewer.inference_client import InferenceError

        mock_wait.side_effect = InferenceError("down", status_code=502)
        mgr = InferenceJobManager("http://127.0.0.1:8010")
        job = mgr.start([("doc.pdf", b"%PDF")])

        import time

        deadline = time.time() + 5
        while time.time() < deadline:
            job = mgr.get_job(job.job_id)
            assert job is not None
            if job.status == "error":
                break
            time.sleep(0.05)
        else:
            self.fail("job did not fail")

        self.assertEqual(job.tasks[0].status, "error")
        self.assertIn("down", job.tasks[0].error or "")


if __name__ == "__main__":
    unittest.main()
