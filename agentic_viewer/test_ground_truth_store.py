"""Tests for ground-truth store helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentic_viewer.ground_truth.store import (
    get_document_gt,
    import_answer_sheet,
    list_documents,
    normalize_gt_entry,
    update_gt_key,
    validate_answer_sheet_payload,
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
                self.assertFalse(result["created_document"])
                self.assertFalse(result["created_key"])
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    saved["doc.pdf"]["Distance between GTG"]["value"],
                    "new",
                )
                docs = list_documents()
                self.assertEqual(docs[0]["document"], "doc.pdf")
                self.assertEqual(docs[0]["n_keys"], 1)

    def test_get_document_gt_missing_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer_sheet.json"
            path.write_text(json.dumps({"existing.pdf": {}}), encoding="utf-8")
            with mock.patch(
                "agentic_viewer.ground_truth.store.answer_sheet_path",
                return_value=path,
            ):
                data = get_document_gt("missing.pdf")
                self.assertEqual(data["document"], "missing.pdf")
                self.assertEqual(data["keys"], [])
                self.assertFalse(data["exists"])

    def test_update_gt_key_creates_document_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer_sheet.json"
            path.write_text(json.dumps({}), encoding="utf-8")
            with mock.patch(
                "agentic_viewer.ground_truth.store.answer_sheet_path",
                return_value=path,
            ):
                result = update_gt_key(
                    "new.pdf",
                    "Some key",
                    {"value": "42", "evidences": ["line"], "evidence_pages": [3]},
                )
                self.assertTrue(result["created_document"])
                self.assertTrue(result["created_key"])
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["new.pdf"]["Some key"]["value"], "42")
                data = get_document_gt("new.pdf")
                self.assertTrue(data["exists"])
                self.assertEqual(data["keys"][0]["key"], "Some key")

    def test_import_answer_sheet_merge_and_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer_sheet.json"
            path.write_text(
                json.dumps(
                    {
                        "keep.pdf": {
                            "Old key": {
                                "value": "old",
                                "evidences": ["a"],
                                "evidence_pages": [1],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            payload = {
                "keep.pdf": {
                    "Old key": {
                        "value": "updated",
                        "evidences": ["b"],
                        "evidence_pages": [2],
                    },
                    "New key": {
                        "value": "42",
                        "evidences": ["c"],
                        "evidence_pages": [3],
                    },
                },
                "other.pdf": {
                    "Only key": {
                        "value": "x",
                        "evidences": [],
                        "evidence_pages": [],
                    }
                },
            }
            with mock.patch(
                "agentic_viewer.ground_truth.store.answer_sheet_path",
                return_value=path,
            ):
                validated = validate_answer_sheet_payload(payload)
                self.assertEqual(validated["keep.pdf"]["Old key"]["value"], "updated")
                merged = import_answer_sheet(payload, mode="merge")
                self.assertEqual(merged["n_documents"], 2)
                self.assertEqual(merged["added_keys"], 2)
                self.assertEqual(merged["updated_keys"], 1)
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["keep.pdf"]["Old key"]["value"], "updated")
                self.assertEqual(saved["keep.pdf"]["New key"]["value"], "42")
                self.assertEqual(saved["other.pdf"]["Only key"]["value"], "x")

                replaced = import_answer_sheet(
                    {
                        "solo.pdf": {
                            "K": {
                                "value": "1",
                                "evidences": ["e"],
                                "evidence_pages": [9],
                            }
                        }
                    },
                    mode="replace",
                )
                self.assertEqual(replaced["mode"], "replace")
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(list(saved.keys()), ["solo.pdf"])

    def test_import_answer_sheet_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_answer_sheet_payload([])
        with self.assertRaises(ValueError):
            validate_answer_sheet_payload({"doc.pdf": "bad"})
        with self.assertRaises(ValueError):
            validate_answer_sheet_payload(
                {"doc.pdf": {"key": {"value": "x", "evidences": "bad", "evidence_pages": []}}}
            )


if __name__ == "__main__":
    unittest.main()
