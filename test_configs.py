#!/usr/bin/env python3
"""
Offline tests for the declared per-model configs (configs/*.md) and their loader.

Run:  python3 -m unittest test_configs   (or the whole suite: python3 -m unittest)

No network, no probing, no $HOME writes.
"""

import pathlib
import unittest
from importlib.machinery import SourceFileLoader

# omodel-wire.py is hyphenated -> load by path.
_path = pathlib.Path(__file__).resolve().parent / "omodel-wire.py"
mw = SourceFileLoader("omodel_wire", str(_path)).load_module()

REQUIRED_PRESETS = {"reason", "code", "agent", "instruct"}
KNOWN_THINKING_CONTROL = {"enable_thinking", "reasoning_effort", "soft_switch", "none"}


class ConfigLoaderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = mw.load_configs()
        self.recipes = self.cfg["recipes"]

    def test_configs_load(self):
        self.assertTrue(self.recipes, "no configs loaded from configs/")

    def test_readme_not_loaded_as_recipe(self):
        # README.md has an example ```json block; it must be skipped.
        files = [r.get("_file", "") for r in self.recipes]
        self.assertNotIn("README.md", files)

    def test_every_config_is_valid(self):
        for r in self.recipes:
            with self.subTest(config=r.get("_file")):
                match = r.get("match")
                self.assertTrue(match, "match required")
                self.assertIsInstance(match if isinstance(match, list) else [match], list)
                caps = r.get("capabilities", {})
                self.assertIn("reasoning", caps)
                self.assertIn("tool_call", caps)
                tc = caps.get("thinking_control", r.get("thinking_control"))
                if tc is not None:
                    self.assertIn(tc, KNOWN_THINKING_CONTROL)
                presets = r.get("presets", {})
                self.assertTrue(REQUIRED_PRESETS.issubset(presets),
                                f"missing presets: {REQUIRED_PRESETS - set(presets)}")
                for pk, preset in presets.items():
                    self.assertIn("sampling", preset, f"{pk} needs sampling")
                    self.assertIn("thinking", preset, f"{pk} needs thinking flag")


class MatchAndCapsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = mw.load_configs()

    def test_match_by_served_id(self):
        # a discovered served-model-id should resolve to the qwen 35b config
        r = mw.match_recipe("nvidia/Qwen3.6-35B-A3B-NVFP4", self.cfg)
        self.assertIsNotNone(r)
        self.assertEqual(r["_file"], "qwen3.6-35b-nvfp4.md")

    def test_match_by_manager_key(self):
        r = mw.match_recipe("qwen3.6-35b-nvfp4", self.cfg)
        self.assertIsNotNone(r)

    def test_caps_from_capabilities(self):
        r = mw.match_recipe("qwen3.6-35b-nvfp4", self.cfg)
        caps = mw.caps_from_capabilities(r)
        self.assertTrue(caps["reasoning"])
        self.assertTrue(caps["can_disable"])      # enable_thinking supports off
        self.assertFalse(caps["effort_ok"])       # not reasoning_effort
        # keys must mirror probe_reasoning's shape (consumed by oc_build_recipe_agents)
        self.assertEqual(set(caps), {"reasoning", "can_disable", "effort_ok", "graded", "reason"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
