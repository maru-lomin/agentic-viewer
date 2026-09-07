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

    def test_infer_pdf_path_for_target_run(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        target = repo / "outputs" / "runs" / "agentic-bc9df067-1fa8-4d12-a107-1e6b85c3e005"
        if not target.is_dir():
            self.skipTest("target run not found")
        path = infer_pdf_path(target)
        self.assertIsNotNone(path)
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "Albanesi - CT ROCA - 2024 (ING-1718-23-AR) - VF (English).pdf")
        info = pdf_info(target)
        self.assertTrue(info["available"])
        self.assertEqual(info["filename"], "Albanesi - CT ROCA - 2024 (ING-1718-23-AR) - VF (English).pdf")

    def test_ui_includes_chunk_pdf_highlight_support(self) -> None:
        from agentic_viewer.app import INDEX_HTML

        self.assertIn("renderChunkCard", INDEX_HTML)
        self.assertIn("mountChunkPdfViewer", INDEX_HTML)
        self.assertIn("pdf-chunk-host", INDEX_HTML)
        self.assertIn("pdf-prev-btn", INDEX_HTML)
        self.assertIn("pdf-next-btn", INDEX_HTML)
        self.assertIn("pdf-page-input", INDEX_HTML)
        self.assertIn("pdf-chunk-btn", INDEX_HTML)
        self.assertIn("renderCurrentPage", INDEX_HTML)
        self.assertIn("renderPageCard", INDEX_HTML)
        self.assertIn("openPagePreview", INDEX_HTML)
        self.assertIn("extractPageNumbersFromText", INDEX_HTML)
        self.assertIn("formatAgenticDetailText", INDEX_HTML)
        self.assertIn("pdf-page-btn", INDEX_HTML)
        self.assertIn("pdf-page-inline-btn", INDEX_HTML)

    def test_page_highlights_api_endpoint(self) -> None:
        from agentic_viewer.app import get_page_highlights

        repo = Path(__file__).resolve().parents[2]
        target = repo / "outputs" / "runs" / "agentic-183ead9e-83df-4410-aaf3-4c567988c79b"
        if not target.is_dir():
            self.skipTest("target run not found")

        data = get_page_highlights("agentic-183ead9e-83df-4410-aaf3-4c567988c79b", 4)
        self.assertEqual(data.get("page"), 4)
        self.assertGreater(data.get("chunk_count", 0), 0)
        self.assertIsInstance(data.get("regions"), list)
        self.assertGreater(len(data.get("regions")), 0)

    def test_chunk_highlights_calibrated_from_pdf(self) -> None:
        from agentic_viewer.highlights import chunk_highlights

        repo = Path(__file__).resolve().parents[2]
        target = repo / "outputs" / "runs" / "agentic-183ead9e-83df-4410-aaf3-4c567988c79b"
        if not target.is_dir():
            self.skipTest("target run not found")
        hl = chunk_highlights(target, "4-1")
        self.assertGreater(len(hl.get("regions", [])), 0)
        r = hl["regions"][0]
        self.assertEqual(r.get("width"), 2481)
        self.assertIn(r.get("height"), (3508, 3509))
        norm = r.get("bbox_norm", [])
        self.assertEqual(len(norm), 4)
        # Verify right boundary is calibrated (~89.7%) rather than uncalibrated marginless (~99.9%)
        self.assertLess(norm[2], 0.92)
        self.assertGreater(norm[2], 0.88)
        # Verify bottom boundary is calibrated (~87.8%) rather than uncalibrated (~91.7%)
        self.assertLess(norm[3], 0.89)
        self.assertGreater(norm[3], 0.86)

    def test_page_highlights_without_chunk_search(self) -> None:
        from agentic_viewer.highlights import page_highlights

        repo = Path(__file__).resolve().parents[2]
        target = repo / "outputs" / "runs" / "agentic-183ead9e-83df-4410-aaf3-4c567988c79b"
        if not target.is_dir():
            self.skipTest("target run not found")
        hl = page_highlights(target, 4)
        self.assertEqual(hl.get("page"), 4)
        self.assertGreater(hl.get("chunk_count", 0), 0)
        self.assertGreater(len(hl.get("regions", [])), 0)
        self.assertIn("4", hl.get("layout_paths", {}))


if __name__ == "__main__":
    unittest.main()
