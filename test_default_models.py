#!/usr/bin/env python3
"""
Test suite for default_models.json functionality.

Tests the load_default_models, get_preferred_model, and _apply_default_models functions.
"""

import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "omodel-wire.py")

import importlib.util
_spec = importlib.util.spec_from_file_location("omodel-wire", MODULE_PATH)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


class TestLoadDefaultModels(unittest.TestCase):
    """Test load_default_models() function."""

    def test_load_creates_template_if_missing(self):
        """Should create template file if default_models.json doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = os.path.join(tmpdir, "default_models.json")
            
            # Temporarily override DEFAULT_MODELS_FILE
            original = m.DEFAULT_MODELS_FILE
            m.DEFAULT_MODELS_FILE = fake_file
            
            try:
                result = m.load_default_models()
                
                # Should have created the file
                self.assertTrue(os.path.exists(fake_file))
                
                # Should have correct structure
                self.assertIn("agents", result)
                self.assertIn("subagents", result)
                
                self.assertEqual(result, m.DEFAULT_MODELS_TEMPLATE)
                # Workhorse agents lead with the local (~free) Qwen; the two
                # expensive-but-low-volume workers lead with a top-tier paid model.
                qwen_first = {"team", "loom", "research", "code", "agent",
                              "agent-research", "agent-code", "agent-test", "agent-instruct"}
                for section in ("agents", "subagents"):
                    for name, preferences in result[section].items():
                        if name in qwen_first:
                            self.assertEqual(preferences[0], "qwen3.6-35b-a3b-nvfp4-unsloth")
                        else:  # agent-architect, agent-review
                            self.assertEqual(preferences[0], "github-copilot/claude-opus-4.8")
            finally:
                m.DEFAULT_MODELS_FILE = original

    def test_load_without_create_does_not_write_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = os.path.join(tmpdir, "default_models.json")
            original = m.DEFAULT_MODELS_FILE
            m.DEFAULT_MODELS_FILE = fake_file
            try:
                self.assertEqual(m._load_default_models(create=False), m.DEFAULT_MODELS_TEMPLATE)
                self.assertFalse(os.path.exists(fake_file))
            finally:
                m.DEFAULT_MODELS_FILE = original

    def test_load_returns_existing_config(self):
        """Should return existing config without overwriting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = os.path.join(tmpdir, "default_models.json")
            
            # Create a custom config with lowercase keys matching actual agent names
            custom_config = {
                "agents": {"team": ["custom-model"]},
                "subagents": {"agent-code": ["another-model"]}
            }
            with open(fake_file, "w") as f:
                json.dump(custom_config, f)
            
            # Temporarily override DEFAULT_MODELS_FILE
            original = m.DEFAULT_MODELS_FILE
            m.DEFAULT_MODELS_FILE = fake_file
            
            try:
                result = m.load_default_models()
                
                # Should return the existing config
                self.assertEqual(result["agents"]["team"], ["custom-model"])
                self.assertEqual(result["subagents"]["agent-code"], ["another-model"])
            finally:
                m.DEFAULT_MODELS_FILE = original


class TestGetPreferredModel(unittest.TestCase):
    """Test get_preferred_model() function."""

    def test_single_available_model(self):
        """Should return the only available model."""
        available = ["qwen3-coder-next-fp8"]
        prefs = ["some-other-model"]
        result = m.get_preferred_model(available, prefs)
        self.assertEqual(result, "qwen3-coder-next-fp8")

    def test_first_preference_matches(self):
        """Should return first preference if it's available."""
        available = ["model1", "model2", "model3"]
        prefs = ["model2", "model1", "model3"]
        result = m.get_preferred_model(available, prefs)
        self.assertEqual(result, "model2")

    def test_second_preference_matches(self):
        """Should return second preference if first is not available."""
        available = ["model1", "model2", "model3"]
        prefs = ["model4", "model2", "model1"]
        result = m.get_preferred_model(available, prefs)
        self.assertEqual(result, "model2")

    def test_no_matching_preference(self):
        """Should return first available model if no preferences match."""
        available = ["model1", "model2", "model3"]
        prefs = ["model4", "model5", "model6"]
        result = m.get_preferred_model(available, prefs)
        self.assertEqual(result, "model1")

    def test_empty_preferences(self):
        """Should return first available model if preferences list is empty."""
        available = ["model1", "model2"]
        prefs = []
        result = m.get_preferred_model(available, prefs)
        self.assertEqual(result, "model1")

    def test_empty_available(self):
        """Should return None if no models available."""
        available = []
        prefs = ["model1"]
        result = m.get_preferred_model(available, prefs)
        self.assertIsNone(result)

    def test_case_sensitivity(self):
        """Model matching should be case-sensitive."""
        available = ["Qwen3-Coder-Next-FP8", "qwen3-coder-next-fp8"]
        prefs = ["qwen3-coder-next-fp8"]
        result = m.get_preferred_model(available, prefs)
        self.assertEqual(result, "qwen3-coder-next-fp8")


class TestApplyDefaultModels(unittest.TestCase):
    """Test _apply_default_models() function."""

    def test_single_model_available(self):
        """Should set all agents to the single available model."""
        agents = {
            "Team": {"model": "dgx-n1-8000/some-model", "mode": "primary"},
            "research": {"model": "dgx-n1-8000/some-model", "mode": "primary"},
            "agent-code": {"model": "dgx-n1-8000/some-model", "mode": "subagent"}
        }
        default_models = {
            "agents": {"Team": ["qwen3-coder-next-fp8"], "research": ["other-model"]},
            "subagents": {"agent-code": ["another-model"]}
        }
        reasoning_caps = {
            "dgx-n1-8000/qwen3-coder-next-fp8": {"reasoning": True}
        }
        
        available_models = {
            "dgx-n1-8000/qwen3-coder-next-fp8": {},
            "dgx-n1-8000/other-model": {},
            "dgx-n1-8000/another-model": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        
        # Should use qwen3-coder-next-fp8 for Team
        self.assertEqual(result["Team"]["model"], "dgx-n1-8000/qwen3-coder-next-fp8")
        # Should use other-model for research (explicit preference)
        self.assertEqual(result["research"]["model"], "dgx-n1-8000/other-model")
        # Should use another-model for agent-code (explicit preference)
        self.assertEqual(result["agent-code"]["model"], "dgx-n1-8000/another-model")

    def test_multiple_models_selects_preference(self):
        """Should select preferred model when multiple are available."""
        agents = {
            "Team": {"model": "dgx-n1-8000/some-model", "mode": "primary"},
            "agent-code": {"model": "dgx-n1-8000/some-model", "mode": "subagent"}
        }
        default_models = {
            "agents": {"Team": ["model3", "model2", "model1"]},
            "subagents": {"agent-code": ["model2", "model1"]}
        }
        reasoning_caps = {
            "dgx-n1-8000/model1": {"reasoning": True},
            "dgx-n1-8000/model2": {"reasoning": True},
            "dgx-n1-8000/model3": {"reasoning": True}
        }
        
        available_models = {
            "dgx-n1-8000/model1": {},
            "dgx-n1-8000/model2": {},
            "dgx-n1-8000/model3": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        
        # Team prefers model3, model2, model1 -> should get model3
        self.assertEqual(result["Team"]["model"], "dgx-n1-8000/model3")
        # agent-code prefers model2, model1 -> should get model2
        self.assertEqual(result["agent-code"]["model"], "dgx-n1-8000/model2")

    def test_no_matching_preference_fallback(self):
        """Should fallback to first available when no preferences match."""
        agents = {
            "Team": {"model": "dgx-n1-8000/some-model", "mode": "primary"}
        }
        default_models = {
            "agents": {"Team": ["model4", "model5"]}
        }
        reasoning_caps = {
            "dgx-n1-8000/model1": {"reasoning": True},
            "dgx-n1-8000/model2": {"reasoning": True}
        }
        
        available_models = {
            "dgx-n1-8000/model1": {},
            "dgx-n1-8000/model2": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        
        # No match, should fallback to first available
        self.assertEqual(result["Team"]["model"], "dgx-n1-8000/model1")

    def test_selects_correct_provider_for_preferred_model(self):
        """Should find and use the correct provider for the preferred model."""
        agents = {
            "Team": {"model": "dgx-n2-8001/some-model", "mode": "primary"}
        }
        default_models = {
            "agents": {"Team": ["qwen3-coder-next-fp8"]}
        }
        reasoning_caps = {
            "dgx-n1-8000/qwen3-coder-next-fp8": {"reasoning": True},
            "dgx-n2-8001/qwen3-coder-next-fp8": {"reasoning": True}
        }
        
        available_models = {
            "dgx-n1-8000/qwen3-coder-next-fp8": {},
            "dgx-n2-8001/qwen3-coder-next-fp8": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        
        # Should select qwen3-coder-next-fp8 from available (first match)
        # When multiple providers have the same model, it picks the first one
        self.assertEqual(result["Team"]["model"], "dgx-n1-8000/qwen3-coder-next-fp8")

    def test_no_agents(self):
        """Should return empty dict if no agents."""
        result = m._apply_default_models({}, {}, {}, {})
        self.assertEqual(result, {})

    def test_no_reasoning_caps(self):
        """Should return agents unchanged if no reasoning_caps."""
        agents = {"Team": {"model": "dgx-n1-8000/qwen3"}}
        result = m._apply_default_models(agents, {}, {}, {})
        self.assertEqual(result, agents)

    def test_subagent_preferences(self):
        """Should use subagents section for subagents."""
        agents = {
            "agent-code": {"model": "dgx-n1-8000/some-model", "mode": "subagent"},
            "agent-research": {"model": "dgx-n1-8000/some-model", "mode": "subagent"}
        }
        default_models = {
            "agents": {},
            "subagents": {
                "agent-code": ["model2"],
                "agent-research": ["model1"]
            }
        }
        reasoning_caps = {
            "dgx-n1-8000/model1": {"reasoning": True},
            "dgx-n1-8000/model2": {"reasoning": True}
        }
        
        available_models = {
            "dgx-n1-8000/model1": {},
            "dgx-n1-8000/model2": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        
        self.assertEqual(result["agent-code"]["model"], "dgx-n1-8000/model2")
        self.assertEqual(result["agent-research"]["model"], "dgx-n1-8000/model1")

    def test_non_dgx_model_selected_when_remote(self):
        """Should select remote models when in available_models pool."""
        agents = {
            "Team": {"model": "dgx-n1-8000/some-model", "mode": "primary"}
        }
        default_models = {
            "agents": {"Team": ["anthropic/claude-opus", "dgx-n1-8000/qwen3"]}
        }
        reasoning_caps = {
            "dgx-n1-8000/qwen3": {"reasoning": True}
        }
        
        # Remote model NOT in pool -> skipped, falls back to first available
        available_models = {
            "dgx-n1-8000/qwen3": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["Team"]["model"], "dgx-n1-8000/qwen3")

    def test_remote_model_in_pool_accepted(self):
        """Remote models in available_models pool are accepted."""
        agents = {
            "Team": {"model": "dgx-n1-8000/some-model", "mode": "primary"}
        }
        default_models = {
            "agents": {"Team": ["anthropic/claude-opus", "dgx-n1-8000/qwen3"]}
        }
        reasoning_caps = {}
        
        # Remote model IN pool -> accepted as first preference
        available_models = {
            "anthropic/claude-opus": {},
            "dgx-n1-8000/qwen3": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["Team"]["model"], "anthropic/claude-opus")

    def test_remote_model_present_in_pool_uses_full_ref(self):
        """A remote model surfaced via the existing-config pool (full slug key)
        is applied with that full provider/model ref."""
        agents = {"team": {"model": "dgx-n1-8000/some-model", "mode": "primary"}}
        default_models = {"agents": {"team": ["openai/gpt-5.5", "dgx-n1-8000/qwen3"]}}
        reasoning_caps = {}
        # all_available_models includes the remote provider's model under its full slug
        available_models = {"dgx-n1-8000/qwen3": {}, "openai/gpt-5.5": {}}
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["team"]["model"], "openai/gpt-5.5")

    def test_authed_remote_pref_resolves_without_discovery(self):
        """Regression: agent-review was downgraded to the local qwen on EVERY sync
        because native-auth cloud prefs (openai/..., github-copilot/...) never
        appear in the discovered pool. With the provider authed (remote_ok), the
        remote pref must win over a later local pref."""
        agents = {"agent-review": {"model": "openai/gpt-5.6-sol", "mode": "subagent"}}
        dm = {"subagents": {"agent-review": [
            "openai/gpt-5.6-sol", "github-copilot/claude-opus-4.8",
            "qwen3.6-35b-a3b-nvfp4-unsloth"]}}
        avail = {"dgx-localhost-8000/qwen3.6-35b-a3b-nvfp4-unsloth": {}}
        out = m._apply_default_models(agents, dm, {}, avail,
                                      remote_ok=frozenset({"openai"}))
        self.assertEqual(out["agent-review"]["model"], "openai/gpt-5.6-sol")
        # without auth, the old (local-fallback) behavior is preserved
        out2 = m._apply_default_models(dict(agents), dm, {}, avail)
        self.assertEqual(out2["agent-review"]["model"],
                         "dgx-localhost-8000/qwen3.6-35b-a3b-nvfp4-unsloth")

    def test_agent_review_remote_subagent_preference(self):
        """agent-review can prefer a remote reviewer model (anthropic/claude-opus-4-8) if in pool."""
        agents = {"agent-review": {"model": "dgx-n1-8000/qwen3", "mode": "subagent"}}
        default_models = {"subagents": {
            "agent-review": ["anthropic/claude-opus-4-8", "qwen3-coder-next-fp8"]}}
        reasoning_caps = {}
        # Remote model NOT in pool -> skipped, falls back to first available
        available_models = {"dgx-n1-8000/qwen3-coder-next-fp8": {}}
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["agent-review"]["model"], "dgx-n1-8000/qwen3-coder-next-fp8")

    def test_agent_review_remote_in_pool(self):
        """agent-review accepts remote reviewer model when in available_models pool."""
        agents = {"agent-review": {"model": "dgx-n1-8000/qwen3", "mode": "subagent"}}
        default_models = {"subagents": {
            "agent-review": ["anthropic/claude-opus-4-8", "qwen3-coder-next-fp8"]}}
        reasoning_caps = {}
        # Remote model IN pool -> accepted as first preference
        available_models = {"anthropic/claude-opus-4-8": {}, "dgx-n1-8000/qwen3-coder-next-fp8": {}}
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["agent-review"]["model"], "anthropic/claude-opus-4-8")

    def test_all_available_models_merged_pool(self):
        """Simulates the oc_sync scenario where all_available_models includes both
        local probes and remote models from existing config. Team and agent-review
        should both prefer remote models when they're in the merged pool."""
        agents = {
            "team": {"model": "dgx-n1-8000/qwen3", "mode": "primary"},
            "agent-review": {"model": "dgx-n1-8000/qwen3", "mode": "subagent"}
        }
        default_models = {
            "agents": {"team": ["openai/gpt-5.5", "anthropic/claude-opus-4-8", "qwen3-coder-next-fp8"]},
            "subagents": {"agent-review": ["anthropic/claude-opus-4-8", "qwen3-coder-next-fp8"]}
        }
        reasoning_caps = {}
        
        # all_available_models = merged pool (local probes + existing config providers)
        # This simulates what oc_sync builds at line 1752-1758
        all_available_models = {
            "dgx-n1-8000/qwen3-coder-next-fp8": {},  # local model
            "openai/gpt-5.5": {},  # remote model from existing config
            "anthropic/claude-opus-4-8": {}  # remote model from existing config
        }
        
        result = m._apply_default_models(agents, default_models, reasoning_caps, all_available_models)
        
        # Team should get openai/gpt-5.5 (first preference that's in pool)
        self.assertEqual(result["team"]["model"], "openai/gpt-5.5")
        # agent-review should get anthropic/claude-opus-4-8 (first preference that's in pool)
        self.assertEqual(result["agent-review"]["model"], "anthropic/claude-opus-4-8")

    def test_all_available_models_skips_unavailable_remote_ref(self):
        """When a remote ref is NOT in all_available_models, it should be skipped
        and the next preference should be tried."""
        agents = {
            "team": {"model": "dgx-n1-8000/qwen3", "mode": "primary"}
        }
        default_models = {
            "agents": {"team": ["openai/gpt-5.5", "anthropic/claude-opus-4-8", "qwen3-coder-next-fp8"]}
        }
        reasoning_caps = {}
        
        # openai/gpt-5.5 is NOT in pool, but anthropic/claude-opus-4-8 is
        all_available_models = {
            "dgx-n1-8000/qwen3-coder-next-fp8": {},
            "anthropic/claude-opus-4-8": {}
        }
        
        result = m._apply_default_models(agents, default_models, reasoning_caps, all_available_models)
        
        # openai/gpt-5.5 is skipped (not in pool), anthropic/claude-opus-4-8 is accepted
        self.assertEqual(result["team"]["model"], "anthropic/claude-opus-4-8")

    def test_agent_review_skips_anthropic_selects_openai(self):
        """Regression test: agent-review should skip unavailable anthropic and select
        available openai/gpt-5.5 when both are in preferences."""
        agents = {
            "agent-review": {"model": "dgx-n1-8000/qwen3-coder-next-fp8", "mode": "subagent"}
        }
        default_models = {
            "subagents": {
                "agent-review": ["anthropic/claude-opus-4-8", "openai/gpt-5.5", "google/gemini-3.5-flash"]
            }
        }
        reasoning_caps = {}
        
        # Runtime models include openai/gpt-5.5 but NOT Anthropic
        # This simulates the real scenario where opencode models shows openai/gpt-5.5
        # but anthropic/claude-opus-4-8 is not available
        all_available_models = {
            "dgx-n1-8000/qwen3-coder-next-fp8": {},  # local model
            "openai/gpt-5.5": {},  # runtime model (available)
            "google/gemini-3.5-flash": {}  # also available
        }
        
        model_notes = []
        result = m._apply_default_models(agents, default_models, reasoning_caps, all_available_models, model_notes)
        
        # anthropic/claude-opus-4-8 should be skipped (not in pool)
        # openai/gpt-5.5 should be selected (first available preference)
        self.assertEqual(result["agent-review"]["model"], "openai/gpt-5.5")
        self.assertIn("Pref skipped (not available): anthropic/claude-opus-4-8", model_notes)

    def test_github_copilot_gpt_5_5_in_pool_accepted(self):
        """Regression test: github-copilot/gpt-5.5 should be accepted when in available_models pool."""
        agents = {
            "team": {"model": "dgx-n1-8000/qwen3", "mode": "primary"}
        }
        default_models = {
            "agents": {"team": ["github-copilot/gpt-5.5", "openai/gpt-5.5", "qwen3-coder-next-fp8"]}
        }
        reasoning_caps = {}

        # github-copilot/gpt-5.5 IS in pool -> accepted as first preference
        available_models = {
            "github-copilot/gpt-5.5": {},
            "openai/gpt-5.5": {},
            "dgx-n1-8000/qwen3-coder-next-fp8": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["team"]["model"], "github-copilot/gpt-5.5")

    def test_github_copilot_gpt_5_5_skipped_when_not_in_pool(self):
        """Regression test: github-copilot/gpt-5.5 should be skipped when not in pool."""
        agents = {
            "team": {"model": "dgx-n1-8000/qwen3", "mode": "primary"}
        }
        default_models = {
            "agents": {"team": ["github-copilot/gpt-5.5", "openai/gpt-5.5", "qwen3-coder-next-fp8"]}
        }
        reasoning_caps = {}

        # github-copilot/gpt-5.5 is NOT in pool -> skipped, falls back to first available
        available_models = {
            "openai/gpt-5.5": {},
            "dgx-n1-8000/qwen3-coder-next-fp8": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["team"]["model"], "openai/gpt-5.5")

    def test_github_copilot_claude_opus_4_8_in_pool_accepted(self):
        """Regression test: github-copilot/claude-opus-4.8 should be accepted when in available_models pool."""
        agents = {
            "agent-review": {"model": "dgx-n1-8000/qwen3", "mode": "subagent"}
        }
        default_models = {
            "subagents": {"agent-review": ["github-copilot/claude-opus-4.8", "openai/gpt-5.5", "google/gemini-3.5-flash"]}
        }
        reasoning_caps = {}

        # github-copilot/claude-opus-4.8 IS in pool -> accepted as first preference
        available_models = {
            "github-copilot/claude-opus-4.8": {},
            "openai/gpt-5.5": {},
            "google/gemini-3.5-flash": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        self.assertEqual(result["agent-review"]["model"], "github-copilot/claude-opus-4.8")


if __name__ == "__main__":
    unittest.main()
