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
                
                # Should have team and agent-code defaults
                self.assertIn("team", result["agents"])
                self.assertIn("agent-code", result["subagents"])
                
                # Should have qwen3-coder-next-fp8 as default
                self.assertEqual(result["agents"]["team"], ["qwen3-coder-next-fp8"])
                self.assertEqual(result["subagents"]["agent-code"], ["qwen3-coder-next-fp8"])
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
            "agent-plan": {"model": "dgx-n1-8000/some-model", "mode": "subagent"}
        }
        default_models = {
            "agents": {},
            "subagents": {
                "agent-code": ["model2"],
                "agent-plan": ["model1"]
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
        self.assertEqual(result["agent-plan"]["model"], "dgx-n1-8000/model1")

    def test_non_dgx_model_selected_when_remote(self):
        """Should select remote models (not in available_models) when they're preferences."""
        agents = {
            "Team": {"model": "dgx-n1-8000/some-model", "mode": "primary"}
        }
        default_models = {
            "agents": {"Team": ["anthropic/claude-opus", "dgx-n1-8000/qwen3"]}
        }
        reasoning_caps = {
            "dgx-n1-8000/qwen3": {"reasoning": True}
        }
        
        available_models = {
            "dgx-n1-8000/qwen3": {}
        }
        result = m._apply_default_models(agents, default_models, reasoning_caps, available_models)
        
        # Remote models are now supported (e.g., openai/gpt-5.5 from OpenCode's built-in provider)
        self.assertEqual(result["Team"]["model"], "anthropic/claude-opus")


if __name__ == "__main__":
    unittest.main()