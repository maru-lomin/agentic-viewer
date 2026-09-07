import unittest
from agentic_viewer.hierarchy import (
    _parse_key_description,
    _parse_search_user_prompt,
)


class TestHierarchyPrompts(unittest.TestCase):
    def test_parse_key_description(self):
        desc = (
            "Extract separation distance between GTG. "
            "Search cues: separation distance, turbine enclosures, CT, GTG, 90', feet. "
            "Convert feet to meters if needed. "
            "Allowed values (choose exactly one): '25m 초과' / '20~25m' / '20m'. "
            "If the document has no usable information, output '20m'."
        )
        meta = _parse_key_description(desc)
        self.assertEqual(
            meta["cues"],
            "separation distance, turbine enclosures, CT, GTG, 90', feet",
        )
        self.assertEqual(meta["allowed"], "'25m 초과' / '20~25m' / '20m'")

    def test_parse_multi_key_user_prompt(self):
        prompt = """Find evidence page(s) for EACH of the following extraction keys in one shared search session. Reuse page reads across keys — do not re-open a page you already inspected unless needed.
Keys:
- key: Distance between GTG
  description: Extract distance between GTG. Search cues: GTG, enclosure, 90'. Allowed values: '20m' / '25m'.
- key: Configuration
  description: Extract plant configuration. Search cues: 2 X 1, combined cycle. Allowed values: 'Multi Shaft' / 'Single Shaft'.
Document outline (compact table of contents):
```
  p1 | Title Page
  p2 | Executive Summary
```
Use the outline above to understand document structure and target likely section pages directly with bm25_search / get_page_text.
Use tools as needed. As soon as a key has 1–3 good pages, call submit_pages..."""

        res = _parse_search_user_prompt(prompt)
        self.assertEqual(len(res["keys"]), 2)
        self.assertEqual(res["keys"][0]["key"], "Distance between GTG")
        self.assertEqual(res["keys"][0]["search_cues"], "GTG, enclosure, 90'")
        self.assertEqual(res["keys"][0]["allowed_values"], "'20m' / '25m'")
        self.assertEqual(res["keys"][1]["key"], "Configuration")
        self.assertEqual(res["keys"][1]["search_cues"], "2 X 1, combined cycle")
        self.assertTrue("Document outline" not in res.get("document_outline", ""))
        self.assertIn("p1 | Title Page", res.get("document_outline", ""))
        self.assertIn("Find evidence page(s)", res.get("task_instruction", ""))

    def test_parse_single_key_user_prompt(self):
        prompt = """Find the best evidence page(s) for this single extraction key.
key: Nearest Fire Brigade
description: Extract distance to the nearest fire brigade. Search cues: Fire Brigade, Distance, miles. Allowed values: '5km 이내' / '5~10km'.
Document outline (compact table of contents):
```
  p1 | Plant Overview
```
Use tools as needed."""

        res = _parse_search_user_prompt(prompt)
        self.assertEqual(len(res["keys"]), 1)
        self.assertEqual(res["keys"][0]["key"], "Nearest Fire Brigade")
        self.assertEqual(res["keys"][0]["search_cues"], "Fire Brigade, Distance, miles")
        self.assertEqual(res["keys"][0]["allowed_values"], "'5km 이내' / '5~10km'")
        self.assertIn("p1 | Plant Overview", res.get("document_outline", ""))

    def test_parse_prior_context_in_prompt(self):
        prompt = """Find evidence page(s)...
Keys:
- key: Distance
  description: Search cues: dist.
Prior search session progress (continue — do not repeat blindly):
{
  "queries_tried": ["dist"],
  "pages_inspected": [1, 2],
  "candidate_pages": [2]
}
Use tools as needed."""

        res = _parse_search_user_prompt(prompt)
        self.assertIsNotNone(res.get("prior_context"))
        self.assertEqual(res["prior_context"]["queries_tried"], ["dist"])
        self.assertEqual(res["prior_context"]["candidate_pages"], [2])


if __name__ == "__main__":
    unittest.main()
