#!/usr/bin/env python3
"""
Offline tests for the config LOADER (the generic configs themselves live in
omodel-manager and are validated there). CI-safe: uses a temp TOML fixture, so it
does not depend on the sibling omodel-manager checkout being present.

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

FIXTURE = """# fixture-model — tuning notes live here as comments
match = ["fixture-model", "org/Fixture"]

[capabilities]
vision = false
reasoning = true
tool_call = true
thinking_control = "enable_thinking"

[presets.reason]
thinking = true
[presets.reason.sampling]
temperature = 0.6
top_p = 0.95

[presets.code]
thinking = true
[presets.code.sampling]
temperature = 0.6

[presets.agent]
thinking = true
[presets.agent.sampling]
temperature = 0.6

[presets.instruct]
thinking = false
[presets.instruct.sampling]
temperature = 0.7
"""


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        (pathlib.Path(self.tmp) / "fixture-model.toml").write_text(FIXTURE, encoding="utf-8")
        # non-.toml files (e.g. the folder README) must be ignored
        (pathlib.Path(self.tmp) / "README.md").write_text("not toml\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_toml_ignores_nontoml(self):
        cfg = mw.load_configs(self.tmp)
        self.assertEqual([r["_file"] for r in cfg["recipes"]], ["fixture-model.toml"])

    def test_match_resolves_config(self):
        r = mw.match_recipe("org/Fixture", mw.load_configs(self.tmp))
        self.assertIsNotNone(r)
        self.assertEqual(r["_file"], "fixture-model.toml")

    def test_caps_from_capabilities(self):
        caps = mw.caps_from_capabilities(mw.match_recipe("fixture-model", mw.load_configs(self.tmp)))
        self.assertTrue(caps["reasoning"])
        self.assertTrue(caps["can_disable"])   # enable_thinking supports off
        self.assertFalse(caps["effort_ok"])
        self.assertEqual(set(caps),
                         {"reasoning", "can_disable", "effort_ok", "graded", "reason"})

    def test_missing_dir_is_empty(self):
        self.assertEqual(mw.load_configs(os.path.join(self.tmp, "nope"))["recipes"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
