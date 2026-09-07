"""Unit tests for Prompts API endpoints in agentic_viewer.app."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from agentic_viewer import app as app_module


class PromptsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.prompts_path = Path(self.tmp_dir.name)
        self.orig_env = os.environ.get("AGENTIC_PROMPTS_DIR")
        os.environ["AGENTIC_PROMPTS_DIR"] = str(self.prompts_path)

        # Seed initial prompt files
        (self.prompts_path / "master_system_prompt.txt").write_text(
            "Master Initial Content\nLine 2", encoding="utf-8"
        )
        (self.prompts_path / "search_system_prompt.txt").write_text(
            "Search Initial Content", encoding="utf-8"
        )
        (self.prompts_path / "eval_master_system_prompt.txt").write_text(
            "Eval Master Initial Content", encoding="utf-8"
        )
        schema = [
            {"key": "Test Key", "description": "Test Desc"},
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

    def test_prompts_page_html(self) -> None:
        html = app_module.prompts_page()
        self.assertIn("Prompts", html)

    def test_api_list_prompts(self) -> None:
        res = app_module.api_list_prompts()
        self.assertIn("prompts", res)
        prompts = res["prompts"]
        self.assertEqual(len(prompts), 4)
        ids = [p["id"] for p in prompts]
        self.assertIn("master", ids)
        self.assertIn("search", ids)
        self.assertIn("eval_master", ids)
        self.assertIn("kv_schema", ids)

    def test_api_get_prompt_success(self) -> None:
        res = app_module.api_get_prompt("master")
        self.assertEqual(res["id"], "master")
        self.assertEqual(res["content"], "Master Initial Content\nLine 2")
        self.assertIn("contract", res)
        self.assertEqual(len(res["schema_keys"]), 1)
        self.assertEqual(res["schema_keys"][0]["key"], "Test Key")

    def test_api_save_kv_schema_keys_array(self) -> None:
        body = {
            "keys": [
                {"key": "Key 1", "description": "Desc 1"},
                {"key": "Key 2", "description": "Desc 2"},
            ]
        }
        res = app_module.api_save_prompt("kv_schema", body)
        self.assertEqual(res["id"], "kv_schema")
        self.assertIsNotNone(res["backup"])

        reloaded = app_module.api_get_prompt("kv_schema")
        self.assertEqual(len(reloaded["schema_keys"]), 2)
        self.assertEqual(reloaded["schema_keys"][0]["key"], "Key 1")

    def test_api_get_prompt_not_found(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            app_module.api_get_prompt("invalid_id")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_api_save_prompt(self) -> None:
        body = {"content": "New Master Content\nLine A\nLine B"}
        res = app_module.api_save_prompt("master", body)
        self.assertEqual(res["id"], "master")
        self.assertEqual(res["line_count"], 3)
        self.assertIsNotNone(res["backup"])

        # Check content updated
        updated = app_module.api_get_prompt("master")
        self.assertEqual(updated["content"], "New Master Content\nLine A\nLine B")

        # Check backups
        backups = app_module.api_list_prompt_backups("master")
        self.assertEqual(len(backups["backups"]), 1)
        ts = backups["backups"][0]["timestamp"]

        b_content = app_module.api_get_prompt_backup("master", ts)
        self.assertEqual(b_content["content"], "Master Initial Content\nLine 2")

    def test_api_save_prompt_validation(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            app_module.api_save_prompt("master", {})
        self.assertEqual(ctx.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx2:
            app_module.api_save_prompt("master", {"content": 123})
        self.assertEqual(ctx2.exception.status_code, 400)

    def test_api_restore_prompt(self) -> None:
        # Save version 2
        app_module.api_save_prompt("search", {"content": "Version 2 Search"})
        backups = app_module.api_list_prompt_backups("search")
        self.assertGreaterEqual(len(backups["backups"]), 1)
        orig_ts = backups["backups"][0]["timestamp"]

        # Restore
        res = app_module.api_restore_prompt("search", {"timestamp": orig_ts})
        self.assertEqual(res["id"], "search")
        curr = app_module.api_get_prompt("search")
        self.assertEqual(curr["content"], "Search Initial Content")

    def test_api_diff_prompt(self) -> None:
        res = app_module.api_diff_prompt(
            "eval_master",
            {"modified": "Eval Master Initial Content\nNew diff line", "compare_with": "current"},
        )
        self.assertIn("diff", res)
        self.assertIn("+New diff line", res["diff"])


if __name__ == "__main__":
    unittest.main()
