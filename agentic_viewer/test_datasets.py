"""Tests for named PDF dataset storage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_viewer.datasets.store import DatasetStore, slugify_dataset_id


class DatasetStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.managed = root / "managed"
        self.folder = root / "folder"
        self.folder.mkdir()
        self.store = DatasetStore(managed_root=self.managed, folder_root=self.folder)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_create_with_uuid_name(self) -> None:
        ds_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        created = self.store.create(ds_id, [("doc.pdf", b"%PDF")])
        self.assertEqual(created["id"], ds_id)
        self.assertEqual(created["name"], ds_id)
        self.assertEqual(slugify_dataset_id("evaluation-v2"), "evaluation-v2")
        self.assertEqual(slugify_dataset_id(" Eval Set 1 "), "Eval-Set-1")
        with self.assertRaises(ValueError):
            slugify_dataset_id("   ")

    def test_create_and_add_files(self) -> None:
        created = self.store.create("Eval Set 1", [("a.pdf", b"%PDF-a")])
        self.assertEqual(created["id"], "Eval-Set-1")
        self.assertEqual(created["n_files"], 1)
        self.assertFalse(created["readonly"])

        updated = self.store.add_files("managed", "Eval-Set-1", [("b.pdf", b"%PDF-b")])
        self.assertEqual(updated["n_files"], 2)
        detail = self.store.get("managed", "Eval-Set-1")
        names = [row["filename"] for row in detail["files"]]
        self.assertEqual(names, ["a.pdf", "b.pdf"])

    def test_rejects_duplicate_file(self) -> None:
        self.store.create("dup", [("a.pdf", b"%PDF")])
        with self.assertRaises(FileExistsError):
            self.store.add_files("managed", "dup", [("a.pdf", b"%PDF2")])

    def test_delete_file_and_dataset(self) -> None:
        self.store.create("gone", [("a.pdf", b"%PDF"), ("b.pdf", b"%PDF")])
        self.store.delete_file("managed", "gone", "a.pdf")
        detail = self.store.get("managed", "gone")
        self.assertEqual([row["filename"] for row in detail["files"]], ["b.pdf"])
        self.store.delete_dataset("managed", "gone")
        with self.assertRaises(FileNotFoundError):
            self.store.get("managed", "gone")

    def test_lists_folder_datasets(self) -> None:
        v2 = self.folder / "evaluation-v2"
        v2.mkdir()
        (v2 / "one.pdf").write_bytes(b"%PDF-1")
        (v2 / "two.PDF").write_bytes(b"%PDF-2")
        (self.folder / "docs").mkdir()
        (self.folder / "docs" / "sample.pdf").write_bytes(b"%PDF")
        empty = self.folder / "empty"
        empty.mkdir()

        rows = self.store.list_datasets()
        ids = {(r["source"], r["id"]) for r in rows}
        self.assertIn(("folder", "evaluation-v2"), ids)
        self.assertNotIn(("folder", "docs"), ids)
        self.assertNotIn(("folder", "empty"), ids)

        detail = self.store.get("folder", "evaluation-v2")
        self.assertTrue(detail["readonly"])
        self.assertEqual(detail["n_files"], 2)
        paths = self.store.pdf_paths("folder", "evaluation-v2")
        self.assertEqual([p.name for p in paths], ["one.pdf", "two.PDF"])

    def test_folder_datasets_are_readonly(self) -> None:
        v1 = self.folder / "evaluation-v1"
        v1.mkdir()
        (v1 / "a.pdf").write_bytes(b"%PDF")
        with self.assertRaises(PermissionError):
            self.store.add_files("folder", "evaluation-v1", [("b.pdf", b"%PDF")])
        with self.assertRaises(PermissionError):
            self.store.delete_dataset("folder", "evaluation-v1")


if __name__ == "__main__":
    unittest.main()
