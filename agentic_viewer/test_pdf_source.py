"""Tests for PDF path resolution in the viewer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_viewer.pdf_source import infer_pdf_path, pdf_info, resolve_runtime_path


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


if __name__ == "__main__":
    unittest.main()
