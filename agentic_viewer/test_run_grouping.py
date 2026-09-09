"""Tests for run grouping and dataset execution versioning."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from agentic_viewer.app import _enrich_run_groups, _format_display_ts, _parse_ts


class RunGroupingTests(unittest.TestCase):
    def test_parse_and_format_ts(self) -> None:
        dt = _parse_ts("2026-09-06T02:08:19.310525+00:00")
        self.assertIsNotNone(dt)
        s = _format_display_ts(dt)
        self.assertTrue(len(s) > 0)
        self.assertIn(":", s)
        # UTC 02:08 should be formatted as KST 11:08
        self.assertEqual(s, "2026-09-06 11:08")

        # KST input should also format correctly
        dt_kst = _parse_ts("2026-09-06T11:08:19.310525+09:00")
        self.assertIsNotNone(dt_kst)
        self.assertEqual(_format_display_ts(dt_kst), "2026-09-06 11:08")

    def test_enrich_run_groups_assigns_v1_for_continuous_runs(self) -> None:
        base = datetime(2026, 9, 6, 2, 0, 0, tzinfo=timezone.utc)
        rows: List[Dict[str, Any]] = [
            {
                "run_id": f"run-{i}",
                "dataset_id": "evaluation-v2",
                "dataset_name": "evaluation-v2",
                "started_at": (base + timedelta(minutes=i * 2)).isoformat(),
            }
            for i in range(5)
        ]
        enriched = _enrich_run_groups(rows)
        for r in enriched:
            self.assertEqual(r["run_group_id"], "evaluation-v2-run-v1")
            self.assertTrue(r["run_group_name"].startswith("evaluation-v2-run-v1"))

    def test_enrich_run_groups_splits_sessions_with_large_time_gap(self) -> None:
        base1 = datetime(2026, 9, 6, 2, 0, 0, tzinfo=timezone.utc)
        base2 = datetime(2026, 9, 6, 6, 0, 0, tzinfo=timezone.utc)  # 4 hours later

        rows: List[Dict[str, Any]] = [
            {
                "run_id": "run-batch1-1",
                "dataset_id": "evaluation-v2",
                "dataset_name": "evaluation-v2",
                "started_at": base1.isoformat(),
            },
            {
                "run_id": "run-batch1-2",
                "dataset_id": "evaluation-v2",
                "dataset_name": "evaluation-v2",
                "started_at": (base1 + timedelta(minutes=3)).isoformat(),
            },
            {
                "run_id": "run-batch2-1",
                "dataset_id": "evaluation-v2",
                "dataset_name": "evaluation-v2",
                "started_at": base2.isoformat(),
            },
            {
                "run_id": "run-batch2-2",
                "dataset_id": "evaluation-v2",
                "dataset_name": "evaluation-v2",
                "started_at": (base2 + timedelta(minutes=2)).isoformat(),
            },
        ]
        enriched = _enrich_run_groups(rows)
        batch1_runs = [r for r in enriched if r["run_id"].startswith("run-batch1")]
        batch2_runs = [r for r in enriched if r["run_id"].startswith("run-batch2")]

        self.assertEqual(len(batch1_runs), 2)
        self.assertEqual(len(batch2_runs), 2)

        self.assertEqual(batch1_runs[0]["run_group_id"], "evaluation-v2-run-v1")
        self.assertEqual(batch1_runs[1]["run_group_id"], "evaluation-v2-run-v1")
        self.assertTrue("run-v1" in batch1_runs[0]["run_group_name"])

        self.assertEqual(batch2_runs[0]["run_group_id"], "evaluation-v2-run-v2")
        self.assertEqual(batch2_runs[1]["run_group_id"], "evaluation-v2-run-v2")
        self.assertTrue("run-v2" in batch2_runs[0]["run_group_name"])

    def test_enrich_run_groups_preserves_explicit_groups(self) -> None:
        rows: List[Dict[str, Any]] = [
            {
                "run_id": "explicit-1",
                "dataset_id": "evaluation-v2",
                "run_group_id": "custom-group-id",
                "run_group_name": "Custom Run Group",
                "started_at": "2026-09-06T01:00:00Z",
            },
            {
                "run_id": "legacy-1",
                "dataset_id": "evaluation-v2",
                "dataset_name": "evaluation-v2",
                "started_at": "2026-09-06T02:00:00Z",
            },
        ]
        enriched = _enrich_run_groups(rows)
        self.assertEqual(enriched[0]["run_group_id"], "custom-group-id")
        self.assertEqual(enriched[0]["run_group_name"], "Custom Run Group")
        self.assertEqual(enriched[1]["run_group_id"], "evaluation-v2-run-v1")

    def test_ungrouped_runs_remain_without_run_group(self) -> None:
        rows: List[Dict[str, Any]] = [
            {
                "run_id": "standalone-1",
                "started_at": "2026-09-06T01:00:00Z",
            }
        ]
        enriched = _enrich_run_groups(rows)
        self.assertIsNone(enriched[0].get("run_group_id"))


if __name__ == "__main__":
    unittest.main()
