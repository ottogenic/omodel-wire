#!/usr/bin/env python3
"""
Offline tests for the config LOADER (the generic configs themselves live in
omodel-manager and are validated there). CI-safe: uses a temp fixture, so it does
not depend on the sibling omodel-manager checkout being present.

Run:  python3 -m unittest test_configs   (or the whole suite: python3 -m unittest)
"""

import os
import pathlib
import shutil
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

# omodel-wire.py is hyphenated -> load by path.
_path = pathlib.Path(__file__).resolve().parent / "omodel-wire.py"
mw = SourceFileLoader("omodel_wire", str(_path)).load_module()

FIXTURE = """# fixture-model — tuning notes

```json
{
  "match": ["fixture-model", "org/Fixture"],
  "capabilities": {"vision": false, "reasoning": true, "tool_call": true,
                   "thinking_control": "enable_thinking"},
  "presets": {
    "reason":   {"thinking": true,  "sampling": {"temperature": 0.6, "top_p": 0.95}},
    "code":     {"thinking": true,  "sampling": {"temperature": 0.6}},
    "agent":    {"thinking": true,  "sampling": {"temperature": 0.6}},
    "instruct": {"thinking": false, "sampling": {"temperature": 0.7}}
  }
}
```
"""


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        (pathlib.Path(self.tmp) / "fixture-model.md").write_text(FIXTURE, encoding="utf-8")
        # a README with its own json block must NOT be loaded as a recipe
        (pathlib.Path(self.tmp) / "README.md").write_text(
            "```json\n{\"match\": [\"should-not-load\"]}\n```\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_and_skips_readme(self):
        cfg = mw.load_configs(self.tmp)
        files = [r["_file"] for r in cfg["recipes"]]
        self.assertIn("fixture-model.md", files)
        self.assertNotIn("README.md", files)

    def test_match_resolves_config(self):
        cfg = mw.load_configs(self.tmp)
        r = mw.match_recipe("org/Fixture", cfg)
        self.assertIsNotNone(r)
        self.assertEqual(r["_file"], "fixture-model.md")

    def test_caps_from_capabilities(self):
        cfg = mw.load_configs(self.tmp)
        caps = mw.caps_from_capabilities(mw.match_recipe("fixture-model", cfg))
        self.assertTrue(caps["reasoning"])
        self.assertTrue(caps["can_disable"])   # enable_thinking supports off
        self.assertFalse(caps["effort_ok"])
        self.assertEqual(set(caps),
                         {"reasoning", "can_disable", "effort_ok", "graded", "reason"})

    def test_missing_dir_is_empty(self):
        cfg = mw.load_configs(os.path.join(self.tmp, "nope"))
        self.assertEqual(cfg["recipes"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
