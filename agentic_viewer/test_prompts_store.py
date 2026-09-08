"""Unit tests for prompts store module."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from agentic_viewer.prompts.store import (
    PROMPT_CONFIGS,
    compute_diff,
    get_backup,
    get_prompt,
    list_backups,
    list_prompts,
    load_schema_keys,
    prompts_dir,
    restore_prompt,
    save_prompt,
)


class TestPromptsStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.prompts_path = Path(self.tmp_dir.name)
        self.orig_env = os.environ.get("AGENTIC_PROMPTS_DIR")
        os.environ["AGENTIC_PROMPTS_DIR"] = str(self.prompts_path)

        # Seed initial prompt files
        (self.prompts_path / "master_system_prompt.txt").write_text(
            "Original Master Prompt\nLine 2", encoding="utf-8"
        )
        (self.prompts_path / "search_system_prompt.txt").write_text(
            "Original Search Prompt", encoding="utf-8"
        )
        (self.prompts_path / "eval_master_system_prompt.txt").write_text(
            "Original Eval Master Prompt", encoding="utf-8"
        )
        (self.prompts_path / "extract_kv_vlm_prompt.txt").write_text(
            "Original Extract KV VLM Prompt\nLine 2", encoding="utf-8"
        )
        # Seed dummy kv_description.json
        schema = [
            {"key": "Key A", "description": "Desc A"},
            {"key": "Key B", "description": "Desc B"},
        ]
        (self.prompts_path / "kv_description.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )

    def tearDown(self) -> None:
        if self.orig_env is not None:
            os.environ["AGENTIC_PROMPTS_DIR"] = self.orig_env
        else:
            os.environ.pop("AGENTIC_PROMPTS_DIR", None)
        self.tmp_dir.cleanup()

    def test_prompts_dir_resolution(self) -> None:
        self.assertEqual(prompts_dir(), self.prompts_path)

    def test_list_prompts(self) -> None:
        prompts = list_prompts()
        self.assertEqual(len(prompts), 5)
        ids = {p["id"] for p in prompts}
        self.assertEqual(ids, {"master", "search", "eval_master", "kv_schema", "extract_kv_vlm"})

        master = next(p for p in prompts if p["id"] == "master")
        self.assertTrue(master["exists"])
        self.assertEqual(master["line_count"], 2)
        self.assertEqual(master["backup_count"], 0)

        extract_tool = next(p for p in prompts if p["id"] == "extract_kv_vlm")
        self.assertTrue(extract_tool["exists"])
        self.assertEqual(extract_tool["line_count"], 2)
        self.assertEqual(extract_tool["category"], "extraction")

        schema = next(p for p in prompts if p["id"] == "kv_schema")
        self.assertTrue(schema["exists"])
        self.assertEqual(schema["category"], "schema")

    def test_get_prompt_master(self) -> None:
        data = get_prompt("master")
        self.assertEqual(data["id"], "master")
        self.assertEqual(data["content"], "Original Master Prompt\nLine 2")
        self.assertIn("contract", data)
        self.assertEqual(len(data["schema_keys"]), 2)
        self.assertEqual(data["schema_keys"][0]["key"], "Key A")

    def test_get_and_save_kv_schema(self) -> None:
        data = get_prompt("kv_schema")
        self.assertEqual(data["id"], "kv_schema")
        self.assertEqual(len(data["schema_keys"]), 2)

        # Save updated schema keys
        updated_keys = [
            {"key": "Key A", "description": "Updated Desc A with cues"},
            {"key": "Key B", "description": "Desc B"},
            {"key": "Key C", "description": "New Key C"},
        ]
        res = save_prompt("kv_schema", json.dumps(updated_keys))
        self.assertEqual(res["id"], "kv_schema")
        self.assertIsNotNone(res["backup"])

        # Check backup created
        backups = list_backups("kv_schema")
        self.assertEqual(len(backups), 1)
        backup_content = get_backup("kv_schema", backups[0]["timestamp"])
        self.assertIn("Key A", backup_content["content"])

        # Check reload
        reloaded = get_prompt("kv_schema")
        self.assertEqual(len(reloaded["schema_keys"]), 3)
        self.assertEqual(reloaded["schema_keys"][0]["description"], "Updated Desc A with cues")
        self.assertEqual(reloaded["schema_keys"][2]["key"], "Key C")

    def test_save_kv_schema_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            save_prompt("kv_schema", "not a json")
        with self.assertRaises(ValueError):
            save_prompt("kv_schema", json.dumps({"not": "a list"}))
        with self.assertRaises(ValueError):
            save_prompt("kv_schema", json.dumps([{"no_key": 1}]))

    def test_get_prompt_invalid_id(self) -> None:
        with self.assertRaises(ValueError):
            get_prompt("non_existent")

    def test_save_prompt_creates_backup_and_updates(self) -> None:
        res = save_prompt("master", "Updated Master Prompt\nNew line\nLine 3")
        self.assertEqual(res["id"], "master")
        self.assertIsNotNone(res["backup"])
        self.assertEqual(res["line_count"], 3)

        # File on disk updated
        updated_content = (self.prompts_path / "master_system_prompt.txt").read_text(encoding="utf-8")
        self.assertEqual(updated_content, "Updated Master Prompt\nNew line\nLine 3")

        # Backup file exists and has original content
        backups = list_backups("master")
        self.assertEqual(len(backups), 1)
        backup_content = get_backup("master", backups[0]["timestamp"])
        self.assertEqual(backup_content["content"], "Original Master Prompt\nLine 2")

    def test_restore_prompt(self) -> None:
        save_prompt("search", "Version 2 Search")
        backups = list_backups("search")
        self.assertEqual(len(backups), 1)
        orig_ts = backups[0]["timestamp"]

        # Save a third version
        save_prompt("search", "Version 3 Search")
        self.assertEqual(get_prompt("search")["content"], "Version 3 Search")

        # Restore to version 1
        restore_prompt("search", orig_ts)
        self.assertEqual(get_prompt("search")["content"], "Original Search Prompt")

    def test_compute_diff(self) -> None:
        orig = "Line 1\nLine 2\nLine 3"
        mod = "Line 1\nModified Line 2\nLine 3\nLine 4"
        diff = compute_diff(orig, mod, fromfile="saved", tofile="editing")
        self.assertIn("-Line 2", diff)
        self.assertIn("+Modified Line 2", diff)
        self.assertIn("+Line 4", diff)

    def test_load_schema_keys(self) -> None:
        keys = load_schema_keys()
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0]["key"], "Key A")
        self.assertEqual(keys[1]["key"], "Key B")


if __name__ == "__main__":
    unittest.main()
