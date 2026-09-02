"""Tests for ground-truth store helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentic_viewer.ground_truth.store import (
    get_document_gt,
    list_documents,
    normalize_gt_entry,
    update_gt_key,
)


class GroundTruthStoreTests(unittest.TestCase):
    def test_normalize_gt_entry(self) -> None:
        entry = normalize_gt_entry(
            {
                "value": "설치",
                "evidences": ["line one", "", "line two"],
                "evidence_pages": ["14", 15],
            }
        )
        self.assertEqual(entry["value"], "설치")
        self.assertEqual(entry["evidences"], ["line one", "line two"])
        self.assertEqual(entry["evidence_pages"], [14, 15])

    def test_update_gt_key_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer_sheet.json"
            sheet = {
                "doc.pdf": {
                    "Distance between GTG": {
                        "value": "old",
                        "evidences": ["a"],
                        "evidence_pages": [1],
                    }
                }
            }
            path.write_text(json.dumps(sheet), encoding="utf-8")
            with mock.patch(
                "agentic_viewer.ground_truth.store.answer_sheet_path",
                return_value=path,
            ):
                before = get_document_gt("doc.pdf")
                self.assertEqual(before["keys"][0]["value"], "old")
                result = update_gt_key(
                    "doc.pdf",
                    "Distance between GTG",
                    {
                        "value": "new",
                        "evidences": ["b", "c"],
                        "evidence_pages": [2, 3],
                    },
                )
                self.assertEqual(result["entry"]["value"], "new")
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    saved["doc.pdf"]["Distance between GTG"]["value"],
                    "new",
                )
                docs = list_documents()
                self.assertEqual(docs[0]["document"], "doc.pdf")
                self.assertEqual(docs[0]["n_keys"], 1)


if __name__ == "__main__":
    unittest.main()
