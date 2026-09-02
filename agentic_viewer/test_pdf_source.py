"""Tests for PDF path resolution in the viewer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_viewer.pdf_source import infer_pdf_path, infer_run_document, pdf_info, resolve_runtime_path


class PdfSourceTests(unittest.TestCase):
    def test_resolve_runtime_path_maps_container_dataset(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        sample = repo / "dataset" / "evaluation-v1"
        pdfs = list(sample.glob("*.pdf")) if sample.is_dir() else []
        if not pdfs:
            self.skipTest("no sample pdf in dataset")
        name = pdfs[0].name
        mapped = resolve_runtime_path(f"/workspace/dataset/evaluation-v1/{name}")
        self.assertTrue(Path(mapped).is_file())

    def test_infer_pdf_path_from_request(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        runs = repo / "outputs" / "runs"
        if not runs.is_dir():
            self.skipTest("no outputs/runs")
        candidates = sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = next((p for p in candidates if (p / "00_request.json").is_file()), None)
        if run_dir is None:
            self.skipTest("no run with 00_request.json")
        path = infer_pdf_path(run_dir)
        if path is None:
            self.skipTest("pdf not resolvable in this environment")
        self.assertTrue(path.is_file())
        info = pdf_info(run_dir)
        self.assertTrue(info["available"])
        self.assertTrue(info["filename"])
        doc = infer_run_document(run_dir)
        self.assertTrue(doc)

    def test_infer_run_document_from_eval_cache(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        runs = repo / "outputs" / "runs"
        if not runs.is_dir():
            self.skipTest("no outputs/runs")
        for run_dir in sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            eval_path = run_dir / "05_eval.json"
            if not eval_path.is_file():
                continue
            expected = json.loads(eval_path.read_text(encoding="utf-8")).get("document")
            if not expected:
                continue
            self.assertEqual(infer_run_document(run_dir), expected)
            return
        self.skipTest("no run with 05_eval.json document")


if __name__ == "__main__":
    unittest.main()
