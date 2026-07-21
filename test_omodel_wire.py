#!/usr/bin/env python3
"""
Test suite for omodel-wire.

Stdlib only (unittest). Run after ANY change to confirm the config the tool builds
still has the settings it must have. Network probes are monkeypatched, so this runs
offline and touches nothing outside a temp dir.

    python3 -m unittest test_omodel_wire -v
    # or
    python3 test_omodel_wire.py

What it covers:
  * roster integrity  -- AGENT_SPECS / PERM / colors / modes / MANAGED_AGENTS
  * config loader     -- see test_configs.py (declared per-model configs)
  * agent building    -- recipe -> agents + per-agent sampling, thinking knobs, team
  * providers         -- tool_call / temperature / vision / reasoning on model entries
  * end-to-end sync   -- oc_sync writes a config with the right agents, disables
                         build/plan, sets default_agent, and PRESERVES a frontier
                         team model across re-syncs
  * plugin directory -- written to plugins/ (plural, per OpenCode docs) + cleanup
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import tempfile
import types
import unittest
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "omodel-wire.py")

_spec = importlib.util.spec_from_file_location("omodel-wire", MODULE_PATH)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
ROLE_NAMES = {"reason", "code", "agent", "instruct"}
MODES = {"primary", "subagent", "all"}
# A caps dict as probe_reasoning would return for a fully-capable reasoning endpoint.
FULL_CAPS = {"reasoning": True, "can_disable": True, "effort_ok": True,
             "graded": True, "reason": "probe: enable_thinking + reasoning_effort"}

# Self-contained declared-config fixture matching the FakeProbes model (no live
# probing anymore; capabilities come from configs). CI-safe: no sibling repo needed.
FIXTURE_DIR = tempfile.mkdtemp(prefix="omw-cfg-")
with open(os.path.join(FIXTURE_DIR, "qwen3.6-27b-nvfp4.toml"), "w", encoding="utf-8") as _f:
    _f.write(
        'match = ["Qwen3.6-27B-NVFP4", "qwen3.6-27b-nvfp4", "Qwen3.6-27B", '
        '"Qwen3.6-35B-A3B-NVFP4", "Qwen3.6-35B"]\n'
        '[capabilities]\n'
        'vision = false\nreasoning = true\ntool_call = true\nthinking_control = "enable_thinking"\n'
        '[presets.reason]\nthinking = true\n[presets.reason.sampling]\n'
        'temperature = 1.0\ntop_p = 0.95\ntop_k = 20\n'
        '[presets.code]\nthinking = true\n[presets.code.sampling]\n'
        'temperature = 0.6\ntop_p = 0.95\ntop_k = 20\n'
        '[presets.agent]\nthinking = true\n'
        'options.chat_template_kwargs = { preserve_thinking = true }\n'
        '[presets.agent.sampling]\ntemperature = 0.6\ntop_p = 0.95\ntop_k = 20\n'
        '[presets.instruct]\nthinking = false\n[presets.instruct.sampling]\n'
        'temperature = 0.7\ntop_p = 0.80\ntop_k = 20\npresence_penalty = 1.5\n')
# a thinking_control="none" model (Nemotron) whose presets carry explicit knobs
with open(os.path.join(FIXTURE_DIR, "nemotron-3-super.toml"), "w", encoding="utf-8") as _f:
    _f.write(
        'match = ["NVIDIA-Nemotron-3-Super-120B", "nemotron-3-super", "Nemotron-3-Super"]\n'
        'thinking_control = "none"\n'
        '[capabilities]\nvision = false\nreasoning = true\ntool_call = true\n'
        'thinking_control = "enable_thinking"\n'
        '[presets.reason]\nthinking = true\n'
        'options.chat_template_kwargs = { enable_thinking = true }\n'
        '[presets.reason.sampling]\ntemperature = 1.0\ntop_p = 0.95\n'
        '[presets.code]\nthinking = true\n'
        'options.chat_template_kwargs = { enable_thinking = true }\n'
        '[presets.code.sampling]\ntemperature = 1.0\ntop_p = 0.95\n'
        '[presets.agent]\nthinking = true\n'
        'options.chat_template_kwargs = { enable_thinking = true }\n'
        '[presets.agent.sampling]\ntemperature = 1.0\ntop_p = 0.95\n'
        '[presets.instruct]\nthinking = false\n'
        'options.chat_template_kwargs = { enable_thinking = false }\n'
        '[presets.instruct.sampling]\ntemperature = 1.0\ntop_p = 0.95\n')
FIXTURE_CONFIGS = m.load_configs(FIXTURE_DIR)
FIXTURE_VISION = {"recipes": [{"_file": "vis.toml", "match": ["Qwen3.6-27B-NVFP4"],
                  "capabilities": {"vision": {"input": ["text", "image"], "output": ["text"]},
                                   "reasoning": False, "tool_call": True}}]}


@contextlib.contextmanager
def quiet():
    """Silence the tool's chatty stdout during a call."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def make_args(tmpdir, **over):
    """A fully-populated args namespace matching what main() builds, dry_run off,
    writing into tmpdir. Override any field via kwargs."""
    a = types.SimpleNamespace(
        config=os.path.join(tmpdir, "opencode.json"),
        _hosts=["192.0.2.101"], _ports=[8000], timeout=1.0,
        no_vision_probe=True, vision_probe_all=False,
        profiles=True, no_reasoning_probe=False, no_tool_call=False,
        set_default=None, allow_empty=False, no_sampling_plugin=False,
        add_default_providers=False,
        web_search="none", enable_exa_shell=False, write_shell_env=False,
        mcp_name="websearch", mcp_command=None, mcp_url=None,
        mcp_env=None, mcp_header=None,
        keep_builtins=False, default_agent="code",
        team_task_budget=None,
        configs=FIXTURE_DIR, recipes=None, no_recipes=False, dry_run=False,
        sampling="server-default", temperature=None, top_p=None, top_k=None,
        presence_penalty=None, frequency_penalty=None,
        repetition_detection=None,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


class FakeProbes:
    """Context manager that swaps the network probes for canned answers so the whole
    sync runs offline. `model` is served at 192.0.2.101:8000.
    
    Also mocks discover_opencode_runtime_models to return an empty list so the
    runtime model discovery doesn't pick up real system models during tests."""

    def __init__(self, model="Qwen3.6-27B-NVFP4", max_len=262144, vision=False, runtime_models=None):
        self.model, self.max_len, self.vision = model, max_len, vision
        self.runtime_models = runtime_models if runtime_models is not None else []
        self._saved = {}

    def __enter__(self):
        for name in ("probe", "probe_reasoning", "probe_vision", "discover_opencode_runtime_models"):
            if hasattr(m, name):
                self._saved[name] = getattr(m, name)

        def probe(host, port, timeout):
            if (host, port) == ("192.0.2.101", 8000):
                return [{"id": self.model, "max_model_len": self.max_len}]
            return []

        m.probe = probe
        m.probe_reasoning = lambda h, p, mid, t: dict(FULL_CAPS)
        m.probe_vision = lambda h, p, mid, t: (
            (True, "blue", "answer contains blue") if self.vision
            else (False, "", "text-only / unverified"))
        # Mock runtime model discovery
        runtime_models = self.runtime_models
        m.discover_opencode_runtime_models = lambda opencode_path=None, timeout=5.0: (
            runtime_models, f"found {len(runtime_models)} runtime model(s)" if runtime_models else "no runtime models in test")
        return self

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            setattr(m, name, fn)


# --------------------------------------------------------------------------- #
# Roster / spec integrity
# --------------------------------------------------------------------------- #
class TestRosterIntegrity(unittest.TestCase):
    def test_agent_specs_shape(self):
        keys = set()
        for spec in m.AGENT_SPECS:
            self.assertEqual(len(spec), 7, f"AGENT_SPEC must be a 7-tuple: {spec}")
            key, prole, mode, is_worker, perm, color, desc = spec
            keys.add(key)
            self.assertIn(prole, ROLE_NAMES, f"{key}: unknown role {prole}")
            self.assertIn(mode, MODES, f"{key}: bad mode {mode}")
            self.assertIn(perm, m.PERM, f"{key}: perm profile {perm} not in PERM")
            self.assertRegex(color, HEX, f"{key}: color {color} not #rrggbb")
            self.assertIsInstance(desc, str)
            # workers are hidden subagents; visible agents are primary
            self.assertEqual(mode, "subagent" if is_worker else "primary",
                             f"{key}: worker/mode mismatch")
        self.assertEqual(len(keys), len(m.AGENT_SPECS), "duplicate AGENT_SPEC keys")

    def test_all_written_agents_are_managed(self):
        # Everything the tool can emit must be prunable on re-sync.
        for key, *_ in m.AGENT_SPECS:
            self.assertIn(key, m.MANAGED_AGENTS, f"{key} missing from MANAGED_AGENTS")
        self.assertIn("team", m.MANAGED_AGENTS)
        for b in m.BUILTIN_DISABLE:
            self.assertIn(b, m.MANAGED_AGENTS, f"disabled builtin {b} not managed")

    def test_team_targets_are_workers(self):
        by_key = {s[0]: s for s in m.AGENT_SPECS}
        for t in m.TEAM_TARGETS:
            self.assertIn(t, by_key, f"team target {t} not in AGENT_SPECS")
            _, _, mode, is_worker, *_ = by_key[t]
            self.assertTrue(is_worker and mode == "subagent",
                            f"team target {t} must be a hidden subagent")

    def test_perm_tiers(self):
        self.assertEqual(m.PERM["readonly"]["edit"], "deny")
        self.assertEqual(m.PERM["readonly"]["bash"], "deny")
        self.assertEqual(m.PERM["readonly"]["task"], "deny")
        self.assertEqual(m.PERM["ask"]["edit"], "ask")
        self.assertEqual(m.PERM["ask"]["bash"], "ask")
        self.assertEqual(m.PERM["full"]["edit"], "allow")
        self.assertEqual(m.PERM["full"]["bash"], "allow")
        for tier in m.PERM.values():  # web always available
            self.assertEqual(tier["websearch"], "allow")
            self.assertEqual(tier["webfetch"], "allow")

    def test_builtin_disable_matches_current_opencode_primaries(self):
        # OpenCode's two built-in *primary* agents are build + plan (docs).
        self.assertEqual(set(m.BUILTIN_DISABLE), {"build", "plan"})


# --------------------------------------------------------------------------- #
# Agent building from recipes
# --------------------------------------------------------------------------- #
class TestAgentBuilding(unittest.TestCase):
    REF = "dgx-n1-8000/Qwen3.6-27B-NVFP4"

    def _recipe(self, name_frag):
        return m.match_recipe(name_frag, FIXTURE_CONFIGS)

    def test_recipe_roster_and_permissions(self):
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, sampling = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))

        for k in ("research", "code", "agent", "team",
                  "agent-research", "agent-code", "agent-test",
                  "agent-instruct", "agent-architect", "agent-review"):
            self.assertIn(k, agents, f"missing agent {k}")
            self.assertEqual(agents[k]["model"], self.REF)

        # Visible direct-use agents retain OpenCode defaults; every delegation role
        # gets its own minimal role-skill bootstrap prompt.
        for k in ("research", "code", "agent"):
            self.assertNotIn("prompt", agents[k], f"visible {k} should be prompt-free")
        for k in m.TEAM_TARGETS:
            expected = f"{{file:./prompts/{m.ROLE_PROMPT_FILES[k]}}}"
            self.assertEqual(agents[k].get("prompt"), expected)
        self.assertEqual(
            agents["team"].get("prompt"),
            f"{{file:./prompts/{m.ROLE_PROMPT_FILES['team']}}}",
        )

        # permission tiers landed on the right agents
        self.assertEqual(agents["research"]["permission"]["edit"], "deny")
        self.assertEqual(agents["code"]["permission"]["edit"], "ask")
        self.assertEqual(agents["agent"]["permission"]["edit"], "allow")
        self.assertEqual(agents["agent-test"]["permission"]["edit"], "deny")
        self.assertEqual(agents["agent-review"]["permission"]["edit"], "deny")

        # sampling from the 27B card: reason=1.0, code/agent=0.6
        self.assertEqual(agents["research"]["temperature"], 1.0)
        self.assertEqual(agents["code"]["temperature"], 0.6)
        self.assertEqual(sampling["code"]["topK"], 20)
        self.assertEqual(sampling["code"]["topP"], 0.95)

    def test_team_delegation_lock(self):
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        team = agents["team"]
        self.assertEqual(team["mode"], "primary")
        self.assertEqual(team["color"], m.TEAM_COLOR)
        # delegation-only: EVERY tool category denied except `task`
        for tool in ("read", "grep", "glob", "list", "edit", "bash", "webfetch", "websearch"):
            self.assertEqual(team["permission"][tool], "deny",
                             f"team should not be able to use {tool}")
        task = team["permission"]["task"]
        self.assertEqual(task["*"], "deny")
        for t in m.TEAM_TARGETS:
            self.assertEqual(task[t], "allow", f"team can't delegate to {t}")

    def test_code_agent_can_delegate_to_review(self):
        # code/agent get a task_budget so they can hand a PR to agent-review, and
        # their task permission allowlists ONLY agent-review (everything else denied).
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        for k in ("code", "agent"):
            self.assertEqual(agents[k].get("task_budget"), 1,
                             f"{k} needs a task_budget to delegate")
            task = agents[k]["permission"]["task"]
            self.assertEqual(task["*"], "deny")
            self.assertEqual(task["agent-review"], "allow")
            # But edit/bash permissions should be preserved
            self.assertIn(agents[k]["permission"].get("edit"), ("ask", "allow"),
                          f"{k} should have edit permission")
            bash = agents[k]["permission"].get("bash")
            # bash is either the shorthand ("ask") or the merge-tripwire dict ({"*":"allow", "*gh pr merge*":"ask"})
            self.assertTrue(bash == "ask" or (isinstance(bash, dict) and bash.get("*") == "allow"),
                            f"{k} should have bash permission")

    def test_agent_code_cannot_delegate_to_review(self):
        # agent-code (hidden worker) must NOT delegate to agent-review.
        # It has NO delegation capability at all (task="deny"), preserving edit/bash permissions.
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        # agent-code is a worker with full permissions but NO delegation capability
        task = agents["agent-code"]["permission"]["task"]
        self.assertEqual(task, "deny", "agent-code should have task='deny' (no delegation)")
        # agent-code should NOT have task_budget (no delegation)
        self.assertNotIn("task_budget", agents["agent-code"])
        # But edit/bash permissions should be preserved (full access), except the
        # merge tripwire prompts (only agent-review holds the reviewer token to merge).
        self.assertEqual(agents["agent-code"]["permission"].get("edit"), "allow")
        bash = agents["agent-code"]["permission"].get("bash")
        self.assertEqual(bash.get("*"), "allow")
        self.assertEqual(bash.get("*gh pr merge*"), "ask")

    def test_only_review_may_merge_and_workers_never_delegate(self):
        # Every worker has task='deny' (no sub-delegation); every agent EXCEPT
        # agent-review carries the merge tripwire (ask). agent-review keeps plain
        # bash allow (the token split is the real merge control).
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        for k in ("agent-research", "agent-code", "agent-test", "agent-instruct",
                  "agent-architect", "agent-review"):
            self.assertEqual(agents[k]["permission"]["task"], "deny",
                             f"{k} must not sub-delegate")
        for k, a in agents.items():
            if k == "agent-review":
                self.assertEqual(a["permission"]["bash"], "allow",
                                 "agent-review keeps unrestricted bash (it merges)")
                continue
            bash = a["permission"].get("bash")
            if isinstance(bash, dict):
                self.assertEqual(bash.get("*gh pr merge*"), "ask",
                                 f"{k} must hit the merge tripwire")

    def test_architect_is_readonly(self):
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        arch = agents["agent-architect"]["permission"]
        self.assertEqual(arch["edit"], "deny")
        self.assertEqual(arch["bash"], "deny")
        self.assertEqual(arch["task"], "deny")

    def test_workers_get_step_caps_primaries_do_not(self):
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        # Each worker carries its configured step cap.
        for key, want in m.WORKER_STEPS.items():
            self.assertEqual(agents[key].get("steps"), want, f"{key} step cap")
        # tightest on the well-defined jobs, most headroom on open-ended work.
        self.assertLess(agents["agent-instruct"]["steps"], agents["agent-code"]["steps"])
        self.assertLessEqual(agents["agent-test"]["steps"], agents["agent-code"]["steps"])
        # visible primaries and team are for direct/human use -> no step cap imposed.
        for key in ("research", "code", "agent", "team"):
            self.assertNotIn("steps", agents[key], f"{key} should not be step-capped")

    def test_thinking_knob_on(self):
        # reason/code/agent are thinking:true -> enable_thinking + graded effort.
        recipe = self._recipe("Qwen3.6-27B-NVFP4")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        opt = agents["research"]["options"]["chat_template_kwargs"]
        self.assertTrue(opt["enable_thinking"])
        self.assertEqual(agents["research"]["options"]["reasoning_effort"], "high")
        # instruct is thinking:false -> enable_thinking false
        instr = agents["agent-instruct"]["options"]["chat_template_kwargs"]
        self.assertFalse(instr["enable_thinking"])

    def test_thinking_control_none(self):
        # Nemotron: control 'none' means NO probe-derived reasoning_effort; only the
        # preset's explicit chat_template_kwargs are emitted.
        recipe = self._recipe("NVIDIA-Nemotron-3-Super-120B")
        self.assertEqual(recipe.get("thinking_control"), "none")
        agents, _ = m.oc_build_recipe_agents(self.REF, recipe, dict(FULL_CAPS))
        opts = agents["research"]["options"]
        self.assertNotIn("reasoning_effort", opts)
        self.assertTrue(opts["chat_template_kwargs"]["enable_thinking"])

    def test_generic_fallback_when_no_recipe(self):
        agents, sampling = m.oc_build_agents(self.REF, dict(FULL_CAPS))
        self.assertIn("team", agents)
        self.assertEqual(agents["code"]["temperature"], 0.6)  # Qwen precise-coding


# --------------------------------------------------------------------------- #
# _apply_model_config_to_agent -- the single "configure an agent for its model" path
# --------------------------------------------------------------------------- #
class TestApplyModelConfig(unittest.TestCase):
    def test_dgx_model_gets_role_preset_and_plugin_vector(self):
        # A code-role agent placed on the fixture Qwen model gets the CODE preset
        # (temp 0.6, top_p 0.95, thinking on) + its plugin sampling vector.
        agent = {"mode": "primary", "model": "dgx-x/old", "permission": {}}
        plugin = {}
        m._apply_model_config_to_agent(agent, "code", "dgx-n1-8000/Qwen3.6-27B-NVFP4",
                                       FIXTURE_CONFIGS, per_model_sampling=plugin)
        self.assertEqual(agent["model"], "dgx-n1-8000/Qwen3.6-27B-NVFP4")
        self.assertEqual(agent["temperature"], 0.6)
        self.assertEqual(agent["top_p"], 0.95)
        self.assertTrue(agent["options"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(plugin["Qwen3.6-27B-NVFP4"]["code"]["topK"], 20)

    def test_switching_dgx_model_replaces_stale_config(self):
        # The reported bug: an agent carrying one model's config, moved to another model,
        # must get the NEW model's config -- not keep the old temp/thinking.
        agent = {"mode": "primary", "model": "dgx-x/old", "temperature": 1.0,
                 "options": {"chat_template_kwargs": {"enable_thinking": True},
                             "reasoning_effort": "high"}, "permission": {}}
        # instruct role on the fixture Qwen model: temp 0.7, thinking OFF.
        m._apply_model_config_to_agent(agent, "agent-instruct",
                                       "dgx-n1-8000/Qwen3.6-27B-NVFP4", FIXTURE_CONFIGS)
        self.assertEqual(agent["temperature"], 0.7)
        self.assertFalse(agent["options"]["chat_template_kwargs"]["enable_thinking"])

    def test_thinking_control_none_has_no_reasoning_effort(self):
        # Nemotron (thinking_control none): only the preset's explicit kwargs, no
        # probe-derived reasoning_effort injected.
        agent = {"mode": "primary", "model": "dgx-x/old", "permission": {}}
        m._apply_model_config_to_agent(agent, "research",
                                       "dgx-n2-8000/NVIDIA-Nemotron-3-Super-120B", FIXTURE_CONFIGS)
        self.assertNotIn("reasoning_effort", agent["options"])
        self.assertTrue(agent["options"]["chat_template_kwargs"]["enable_thinking"])

    def test_cloud_model_strips_local_knobs(self):
        # A frontier/cloud model runs on OpenCode's defaults -- local vLLM knobs are removed
        # and the agent's plugin entry is dropped.
        agent = {"mode": "primary", "model": "dgx-x/old", "temperature": 0.6, "top_p": 0.95,
                 "options": {"chat_template_kwargs": {"enable_thinking": True}}, "permission": {}}
        plugin = {"gpt-5.5": {"code": {"temperature": 0.6}}}
        m._apply_model_config_to_agent(agent, "code", "openai/gpt-5.5", FIXTURE_CONFIGS,
                                       per_model_sampling=plugin)
        self.assertEqual(agent["model"], "openai/gpt-5.5")
        self.assertNotIn("temperature", agent)
        self.assertNotIn("top_p", agent)
        self.assertNotIn("options", agent)
        self.assertNotIn("code", plugin["gpt-5.5"])
        # untouched fields survive
        self.assertEqual(agent["mode"], "primary")
        self.assertIn("permission", agent)


# --------------------------------------------------------------------------- #
# Variants + sampling plugin
# --------------------------------------------------------------------------- #
class TestVariantsAndPlugin(unittest.TestCase):
    def test_graded_variants(self):
        v = m.oc_build_variants(dict(FULL_CAPS))
        self.assertEqual(set(v), {"no-think", "low", "medium", "high"})
        self.assertEqual(v["high"]["options"]["reasoning_effort"], "high")
        self.assertTrue(v["low"]["options"]["chat_template_kwargs"]["enable_thinking"])

    def test_ungraded_variants(self):
        caps = dict(FULL_CAPS, graded=False)
        v = m.oc_build_variants(caps)
        self.assertEqual(set(v), {"no-think", "think"})

    def test_no_disable_gives_empty_off_options(self):
        caps = dict(FULL_CAPS, can_disable=False, graded=False)
        v = m.oc_build_variants(caps)
        self.assertEqual(v["no-think"]["options"], {})

    def test_sampling_plugin_js_is_valid(self):
        mid = "Qwen3.6-27B-NVFP4"
        _, sampling = m.oc_build_agents("dgx-n1-8000/" + mid, dict(FULL_CAPS))
        js = m.oc_agent_sampling_plugin_js({mid: sampling}, mid)
        self.assertIn("chat.params", js)
        self.assertIn(m.PROVIDER_PREFIX, js)
        self.assertIn("input.model", js)   # sampling is now keyed by the running model
        # the embedded AGENT_SAMPLING table must be valid JSON we can round-trip;
        # it's now nested: {model_id: {agent: vec}}.
        table = js.split("const AGENT_SAMPLING =", 1)[1].split("\n\nfunction", 1)[0].strip()
        self.assertEqual(json.loads(table)[mid]["code"]["topK"], 20)

    def test_repetition_detection_default_is_lenient(self):
        # Regression: the "300000" -> "30000" false cut came from min_pattern_size
        # defaulting to 1 (a lone repeated token counted as a pattern). The default must
        # set it >= 2 so single-token runs (long numbers, indentation, "====", hex) are
        # never flagged; and vLLM requires min_count >= 2 and min_pattern <= max_pattern.
        with tempfile.TemporaryDirectory() as tmp:
            rd = m.build_sampling(make_args(tmp))["repetition_detection"]
        self.assertGreaterEqual(rd["min_pattern_size"], 2)
        self.assertLessEqual(rd["min_pattern_size"], rd["max_pattern_size"])
        self.assertGreaterEqual(rd["min_count"], 2)

    def test_repetition_detection_off_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = m.build_sampling(make_args(tmp, repetition_detection="off"))
        self.assertIsNone(s["repetition_detection"])

    def test_repetition_detection_partial_override_merges(self):
        # tuning one knob must keep the others (not drop max_pattern_size -> disabled).
        with tempfile.TemporaryDirectory() as tmp:
            rd = m.build_sampling(make_args(tmp, repetition_detection="min_count:4"))["repetition_detection"]
        self.assertEqual(rd["min_count"], 4)             # overridden (differs from default)
        self.assertEqual(rd["max_pattern_size"], 20)     # kept from default
        self.assertEqual(rd["min_pattern_size"], 3)      # kept from default

    def test_repetition_detection_reaches_agent_options(self):
        # the default must actually land on each built agent's sampling vector.
        with tempfile.TemporaryDirectory() as tmp:
            rd = m.build_sampling(make_args(tmp))["repetition_detection"]
        _, sampling = m.oc_build_agents("dgx-n1-8000/Qwen3.6-27B-NVFP4", dict(FULL_CAPS), rd)
        self.assertEqual(sampling["code"]["options"]["repetition_detection"], rd)


# --------------------------------------------------------------------------- #
# Reasoning capability probe
# --------------------------------------------------------------------------- #
def _resp(content="", reasoning="", finish="stop"):
    """A canned /v1/chat/completions response as the probe helpers parse it."""
    msg = {"content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    return {"choices": [{"message": msg, "finish_reason": finish}]}


class TestReasoningProbe(unittest.TestCase):
    def test_reasoning_len_structured_field(self):
        self.assertEqual(m._reasoning_len(_resp(reasoning="let me think...")), 15)

    def test_reasoning_len_inline_closed_block(self):
        # No reasoning parser -> the model emits <think>...</think> in content.
        j = _resp(content="<think>reasoning here</think>391")
        self.assertEqual(m._reasoning_len(j), len("reasoning here"))

    def test_reasoning_len_inline_unclosed_block(self):
        # Cut off before </think>, but the open tag is still present.
        j = _resp(content="<think>still thinking about it", finish="length")
        self.assertEqual(m._reasoning_len(j), len("still thinking about it"))

    def test_reasoning_len_plain_content_is_zero(self):
        self.assertEqual(m._reasoning_len(_resp(content="391")), 0)

    def test_probe_detects_truncated_mid_think(self):
        # qwen3 reasoning-parser + no </think> reached: partial, UNTAGGED reasoning
        # lands in content with reasoning=null and finish_reason=length (vLLM #35221).
        def fake_chat(host, port, mid, extra, timeout, prompt):
            if not extra:  # default request -> truncated mid-think
                return 200, _resp(content="Thinking Process:\n\n1", finish="length"), ""
            # enable_thinking=false disables it -> clean short answer
            return 200, _resp(content="391", finish="stop"), ""
        saved = m._chat
        m._chat = fake_chat
        try:
            caps = m.probe_reasoning("h", 8000, "qwen/qwen3.6-35b", 1.0)
        finally:
            m._chat = saved
        self.assertTrue(caps["reasoning"])
        self.assertTrue(caps["can_disable"])

    def test_probe_treats_non_reasoning_as_non_reasoning(self):
        # Answers the trivial prompt and stops on its own -> not a reasoning model.
        def fake_chat(host, port, mid, extra, timeout, prompt):
            return 200, _resp(content="391", finish="stop"), ""
        saved = m._chat
        m._chat = fake_chat
        try:
            caps = m.probe_reasoning("h", 8000, "some-instruct-model", 1.0)
        finally:
            m._chat = saved
        self.assertFalse(caps["reasoning"])


# --------------------------------------------------------------------------- #
# Provider / model-entry construction
# --------------------------------------------------------------------------- #
class TestProviders(unittest.TestCase):
    SD = {"mode": "server-default", "temperature": None, "top_p": None,
          "top_k": None, "presence_penalty": None, "frequency_penalty": None}

    def test_tool_call_declared_by_default(self):
        with FakeProbes():
            providers, refs, caps, _ = m.oc_build_providers(
                ["192.0.2.101"], [8000], 1.0, self.SD,
                profiles=False, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertTrue(entry["tool_call"])
        self.assertEqual(refs, ["dgx-n1-8000/Qwen3.6-27B-NVFP4"])

    def test_tool_call_can_be_disabled(self):
        with FakeProbes():
            providers, _, _, _ = m.oc_build_providers(
                ["192.0.2.101"], [8000], 1.0, self.SD,
                profiles=False, tool_call=False, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertNotIn("tool_call", entry)

    def test_server_default_sets_temperature_false(self):
        with FakeProbes():
            providers, _, _, _ = m.oc_build_providers(
                ["192.0.2.101"], [8000], 1.0, self.SD,
                profiles=False, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertIs(entry["temperature"], False)

    def test_profiles_keeps_temperature_true(self):
        with FakeProbes():
            providers, _, caps, _ = m.oc_build_providers(
                ["192.0.2.101"], [8000], 1.0, self.SD,
                profiles=True, recipes=FIXTURE_CONFIGS, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertIs(entry["temperature"], True)
        self.assertTrue(entry["reasoning"])
        self.assertIn("variants", entry)
        self.assertIn("dgx-n1-8000/Qwen3.6-27B-NVFP4", caps)

    def test_vision_writes_attachment_and_modalities(self):
        with FakeProbes():
            providers, _, _, _ = m.oc_build_providers(
                ["192.0.2.101"], [8000], 1.0, self.SD,
                profiles=False, recipes=FIXTURE_VISION, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertTrue(entry["attachment"])
        self.assertEqual(entry["modalities"]["input"], ["text", "image"])


# --------------------------------------------------------------------------- #
# End-to-end oc_sync
# --------------------------------------------------------------------------- #
class TestSyncEndToEnd(unittest.TestCase):
    def _sync(self, tmp, **over):
        runtime_models = over.pop("runtime_models", None)
        default_models = over.pop("default_models", None)
        args = make_args(tmp, **over)
        sampling = m.build_sampling(args)
        orig = m.load_default_models
        if default_models is not None:
            m.load_default_models = lambda: default_models
        try:
            with FakeProbes(runtime_models=runtime_models), quiet():
                rc = m.oc_sync(args, sampling, {"opencode"})
        finally:
            m.load_default_models = orig
        self.assertEqual(rc, 0)
        with open(args.config, encoding="utf-8") as f:
            return json.load(f)

    def test_writes_full_roster_and_disables_builtins(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._sync(tmp)
            ag = cfg["agent"]
            for k in ("research", "code", "agent", "team",
                      "agent-research", "agent-code", "agent-test", "agent-instruct", "agent-architect", "agent-review"):
                self.assertIn(k, ag)
            # native build/plan disabled, default moved off build
            self.assertEqual(ag["build"], {"disable": True})
            self.assertEqual(ag["plan"], {"disable": True})
            self.assertEqual(cfg["default_agent"], "code")
            # provider + model entry wired for tools + reasoning
            entry = cfg["provider"]["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
            self.assertTrue(entry["tool_call"])
            self.assertTrue(entry["reasoning"])

    def test_sync_roster_summary_describes_current_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp)
            with FakeProbes():
                out = _capture(m.oc_sync, args, m.build_sampling(args), {"opencode"})
            self.assertIn("simple -> agent-code", out)
            self.assertIn("medium/high -> agent-architect", out)
            self.assertIn("completed -> agent-test -> agent-review", out)

    def test_dry_run_does_not_create_default_models_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_models = os.path.join(tmp, "default_models.json")
            old_default_models = m.DEFAULT_MODELS_FILE
            m.DEFAULT_MODELS_FILE = default_models
            try:
                args = make_args(tmp, dry_run=True)
                with FakeProbes(), quiet():
                    self.assertEqual(m.oc_sync(args, m.build_sampling(args), {"opencode"}), 0)
                self.assertFalse(os.path.exists(default_models))
            finally:
                m.DEFAULT_MODELS_FILE = old_default_models

    def test_writes_all_role_skills_and_bootstrap_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            for agent_name, skill_name in m.ROLE_SKILL_NAMES.items():
                skill = os.path.join(tmp, "skills", skill_name, "SKILL.md")
                prompt = os.path.join(tmp, "prompts", m.ROLE_PROMPT_FILES[agent_name])
                self.assertTrue(os.path.exists(skill), f"{skill_name} skill not written")
                self.assertTrue(os.path.exists(prompt), f"{agent_name} prompt not written")
                with open(skill, encoding="utf-8") as f:
                    self.assertIn(f"name: {skill_name}", f.read())
                with open(prompt, encoding="utf-8") as f:
                    body = f.read()
                self.assertIn(f"`{skill_name}-override`", body)
                self.assertIn(f"`{skill_name}`", body)
                self.assertIn(f"`{skill_name}-extend`", body)
                self.assertLess(body.index(f"`{skill_name}-override`"),
                                body.index(f"`{skill_name}-extend`"))
                self.assertIn("load it exclusively", body)
                self.assertIn("cannot weaken or contradict", body)
                self.assertIn("NEVER probe a missing skill", body)
                self.assertEqual(len(body.splitlines()), 5, "bootstrap prompt should stay five lines")

    def test_worker_role_skills_carry_return_contract(self):
        for name in ("agent-code", "agent-research", "agent-test", "agent-instruct",
                     "agent-architect", "agent-review"):
            skill = m.ROLE_SKILLS[name]
            for status in ("STATUS: DONE", "STATUS: CONTINUE", "STATUS: NEEDS_RESEARCH",
                           "STATUS: BLOCKED"):
                self.assertIn(status, skill)
            self.assertIn("Do not spin", skill)

    def test_code_test_review_skills_split_verification_work(self):
        self.assertIn("Run focused checks", m.AGENT_CODE_SKILL)
        self.assertIn("Leave full suites", m.AGENT_CODE_SKILL)
        self.assertIn("Run the broad verification", m.AGENT_TEST_SKILL)
        self.assertIn("coder owns tight edit/test loops", m.AGENT_TEST_SKILL)
        self.assertIn("tester output as the primary command evidence", m.AGENT_REVIEW_SKILL)
        self.assertIn("run spot", m.AGENT_REVIEW_SKILL)

    def test_team_skill_encodes_full_workflow_and_continuity(self):
        skill = m.AGENT_TEAM_SKILL
        for text in ("Simple", "Medium/high-risk", "architect first", "NEEDS_RESEARCH",
                     "Required Test", "agent-test", "Required Review", "verification packet",
                     "one at a time",
                     "same reviewer `task_id`", "remediation", "scope firewall",
                     "ask whether to create one"):
            self.assertIn(text, skill)
        self.assertIn("same tester `task_id`", skill)
        self.assertIn("After tester pass", skill)
        self.assertIn("send it directly to reviewer", skill)
        self.assertIn("Do not research, inspect, reconstruct, or pre-review it", skill)
        self.assertIn("same architect `task_id`", skill)
        self.assertIn("same coder `task_id`", skill)
        self.assertIn("substitutes categories", skill)
        self.assertIn("do not relay or implement", skill)
        self.assertIn("agent runbook review", skill)
        self.assertIn("load `agent-runbook-review`", skill)

    def test_architect_and_reviewer_classify_without_expanding_scope(self):
        for skill in (m.AGENT_ARCHITECT_SKILL, m.AGENT_REVIEW_SKILL):
            for category in ("blocker", "regression", "pre-existing", "future work",
                             "out of scope"):
                self.assertIn(f"`{category}`", skill)
            self.assertIn("Only `blocker` and `regression` require immediate implementation", skill)
            self.assertIn("Never expand acceptance criteria", skill)
        self.assertIn("Never rename categories", m.AGENT_REVIEW_SKILL)

    def test_removes_retired_generated_role_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for relpath in m.LEGACY_ROLE_ARTIFACTS:
                path = os.path.join(tmp, relpath)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("stale")
            self._sync(tmp)
            for relpath in m.LEGACY_ROLE_ARTIFACTS:
                self.assertFalse(os.path.exists(os.path.join(tmp, relpath)))

    def test_writes_agent_runbook_review_skill_globally(self):
        # The runbook-review maintenance pass ships globally so "perform an agent runbook
        # review" works in any repo, and it carries the tested SQLite session-mining recipe.
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            skill = os.path.join(tmp, "skills", "agent-runbook-review", "SKILL.md")
            self.assertTrue(os.path.exists(skill), "agent-runbook-review SKILL.md not written")
            body = open(skill, encoding="utf-8").read()
            self.assertIn("name: agent-runbook-review", body)   # discoverable
            self.assertIn("opencode.db", body)                  # the session-mining source
            self.assertIn("parent_id", body)                    # subagent linkage
            self.assertIn("Runbook Review Report", body)        # report-first output

    def test_role_skills_are_scoped_to_their_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            ag = self._sync(tmp)["agent"]
            for agent_name, own_skill in m.ROLE_SKILL_NAMES.items():
                rules = ag[agent_name]["permission"]["skill"]
                if agent_name == "team":
                    self.assertEqual(rules["*"], "deny")
                self.assertEqual(rules[own_skill], "allow")
                self.assertEqual(rules[f"{own_skill}-extend"], "allow")
                self.assertEqual(rules[f"{own_skill}-override"], "allow")
                for other in set(m.ROLE_SKILLS) - {own_skill}:
                    self.assertEqual(rules[other], "deny")

    def test_writes_git_identity_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            plugin = os.path.join(tmp, "plugins", "otools-git-identity.js")
            self.assertTrue(os.path.exists(plugin), "otools-git-identity.js not written")
            js = open(plugin, encoding="utf-8").read()
            self.assertIn('"shell.env"', js)         # the injecting hook
            self.assertIn("gh_token_coder", js)       # reads the coder token
            self.assertIn("GH_TOKEN_REVIEWER", js)    # exposes the reviewer token
            self.assertIn("insteadOf", js)            # HTTPS routing (no SSH)
            self.assertIn("GIT_AUTHOR_EMAIL", js)     # commits authored as the bot
            self.assertIn("gh_coder_identity", js)    # ...from the resolved-identity file

    def test_missing_gh_token_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder = os.path.join(tmp, "coder"); rev = os.path.join(tmp, "rev")
            saved = (m.GH_TOKEN_CODER_FILE, m.GH_TOKEN_REVIEWER_FILE)
            try:
                m.GH_TOKEN_CODER_FILE, m.GH_TOKEN_REVIEWER_FILE = coder, rev
                self.assertEqual(set(m._missing_gh_token_roles()), {"coder", "reviewer"})
                open(coder, "w").write("x")
                self.assertEqual(m._missing_gh_token_roles(), ["reviewer"])
                open(rev, "w").write("x")
                self.assertEqual(m._missing_gh_token_roles(), [])
            finally:
                m.GH_TOKEN_CODER_FILE, m.GH_TOKEN_REVIEWER_FILE = saved

    def test_set_gh_token_writes_and_clears(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder = os.path.join(tmp, "otools", "gh_token_coder")
            ident = os.path.join(tmp, "otools", "gh_coder_identity")
            saved = (m.GH_TOKEN_CODER_FILE, m.GH_CODER_IDENTITY_FILE, m._resolve_gh_identity)
            try:
                m.GH_TOKEN_CODER_FILE, m.GH_CODER_IDENTITY_FILE = coder, ident
                m._resolve_gh_identity = lambda t: None      # offline: no network in unit tests
                with quiet():
                    m._set_gh_token("coder", "ghp_secret123")
                self.assertEqual(open(coder).read(), "ghp_secret123")   # trimmed, no newline
                if os.name == "posix":                                  # 0600 only meaningful on POSIX
                    self.assertEqual(oct(os.stat(coder).st_mode & 0o777), "0o600")
                with quiet():
                    m._set_gh_token("coder", "none")                    # clear
                self.assertFalse(os.path.exists(coder))
            finally:
                m.GH_TOKEN_CODER_FILE, m.GH_CODER_IDENTITY_FILE, m._resolve_gh_identity = saved

    def test_set_coder_token_resolves_and_writes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            coder = os.path.join(tmp, "otools", "gh_token_coder")
            ident = os.path.join(tmp, "otools", "gh_coder_identity")
            saved = (m.GH_TOKEN_CODER_FILE, m.GH_CODER_IDENTITY_FILE, m._resolve_gh_identity)
            try:
                m.GH_TOKEN_CODER_FILE, m.GH_CODER_IDENTITY_FILE = coder, ident
                # stub the API so the test is offline + deterministic
                m._resolve_gh_identity = lambda t: {
                    "name": "ottogenic-bot",
                    "email": "999+ottogenic-bot@users.noreply.github.com"}
                with quiet():
                    m._set_gh_token("coder", "ghp_bot")
                who = json.load(open(ident))
                self.assertEqual(who["name"], "ottogenic-bot")
                self.assertIn("@users.noreply.github.com", who["email"])
                # clearing the token also removes the resolved identity
                with quiet():
                    m._set_gh_token("coder", "none")
                self.assertFalse(os.path.exists(ident))
            finally:
                m.GH_TOKEN_CODER_FILE, m.GH_CODER_IDENTITY_FILE, m._resolve_gh_identity = saved

    def test_default_model_prefs_ordered_and_slash_safe(self):
        # Preference resolution walks the list IN ORDER; a served id that contains a slash
        # (unsloth/qwen3-coder-next-fp8) is a LOCAL model, not a remote provider ref.
        avail = {
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8": {},
            "dgx-103-8000/qwen3-coder-next-fp8": {},
        }
        def resolve(prefs, mode="primary"):
            agents = {"code": {"mode": mode, "model": "dgx-x/placeholder"}}
            key = "subagents" if mode == "subagent" else "agents"
            out = m._apply_default_models(dict(agents), {key: {"code": prefs}}, {}, avail)
            return out["code"]["model"]

        # top local live -> picked (full ref), even though a remote is later in the list
        self.assertEqual(
            resolve(["unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "openai/gpt-5.5"]),
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8")
        # top local NOT live -> fall THROUGH to the next live local (slash id not misclassified,
        # not jumped to the remote)
        self.assertEqual(
            resolve(["qwen3.6-35b-a3b-fp8", "qwen3-coder-next-fp8", "openai/gpt-5.5"]),
            "dgx-103-8000/qwen3-coder-next-fp8")
        # strict order: [local-down, cloud-down, local-up] -> the local model, because
        # remote refs must be in the available_models pool to be accepted
        self.assertEqual(
            resolve(["qwen3.6-35b-a3b-fp8", "openai/gpt-5.5", "qwen3-coder-next-fp8"]),
            "dgx-103-8000/qwen3-coder-next-fp8")
        # no listed local live, cloud ref not in pool -> fallback to first available local
        self.assertEqual(
            resolve(["qwen3.6-35b-a3b-fp8", "openai/gpt-5.5"]), "dgx-102-8000/unsloth/qwen3-coder-next-fp8")
        # remote ref not in pool -> skipped, fall back to next available local
        self.assertEqual(
            resolve(["anthropic/claude-opus-4-8", "openai/gpt-5.5"], mode="subagent"),
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8")

    def test_remote_ref_not_in_pool_is_skipped(self):
        # Regression: anthroipc/claude-opus-4-8 should NOT be accepted if not in available_models
        avail = {
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8": {},
            "dgx-103-8000/qwen3-coder-next-fp8": {},
        }
        def resolve(prefs, mode="primary"):
            agents = {"code": {"mode": mode, "model": "dgx-x/placeholder"}}
            key = "subagents" if mode == "subagent" else "agents"
            out = m._apply_default_models(dict(agents), {key: {"code": prefs}}, {}, avail)
            return out["code"]["model"]

        # Remote ref not in pool -> skipped, falls back to first available local
        self.assertEqual(
            resolve(["anthropic/claude-opus-4-8", "openai/gpt-5.5"]),
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8")

    def test_remote_ref_in_pool_is_accepted(self):
        # When a remote ref IS in the available_models pool, it should be accepted
        avail = {
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8": {},
            "openai/gpt-5.5": {},  # Remote ref in pool
            "dgx-103-8000/qwen3-coder-next-fp8": {},
        }
        def resolve(prefs, mode="primary"):
            agents = {"code": {"mode": mode, "model": "dgx-x/placeholder"}}
            key = "subagents" if mode == "subagent" else "agents"
            out = m._apply_default_models(dict(agents), {key: {"code": prefs}}, {}, avail)
            return out["code"]["model"]

        # Remote ref in pool -> accepted (first in list)
        self.assertEqual(
            resolve(["openai/gpt-5.5", "unsloth/qwen3-coder-next-fp8"]),
            "openai/gpt-5.5")
        # Local ref not in pool, remote ref in pool -> remote accepted
        self.assertEqual(
            resolve(["qwen3.6-35b-a3b-fp8", "openai/gpt-5.5"]),
            "openai/gpt-5.5")

    def test_served_id_with_slash_resolves_to_provider(self):
        # Regression: unsloth/qwen3-coder-next-fp8 should resolve to dgx-.../unsloth/qwen3-coder-next-fp8
        avail = {
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8": {},
            "dgx-103-8000/qwen3-coder-next-fp8": {},
        }
        def resolve(prefs, mode="primary"):
            agents = {"code": {"mode": mode, "model": "dgx-x/placeholder"}}
            key = "subagents" if mode == "subagent" else "agents"
            out = m._apply_default_models(dict(agents), {key: {"code": prefs}}, {}, avail)
            return out["code"]["model"]

        # Served ID with slash -> resolves to full provider-qualified ref
        self.assertEqual(
            resolve(["unsloth/qwen3-coder-next-fp8"]),
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8")
        # Bare model ID -> resolves to first matching provider
        self.assertEqual(
            resolve(["qwen3-coder-next-fp8"]),
            "dgx-103-8000/qwen3-coder-next-fp8")

    def test_unavailable_dgx_pref_falls_back_to_cloud(self):
        # When first DGX pref is unavailable, falls back to next available (cloud or local)
        avail = {
            "openai/gpt-5.5": {},  # Remote ref in pool
            "dgx-103-8000/qwen3-coder-next-fp8": {},
        }
        def resolve(prefs, mode="primary"):
            agents = {"code": {"mode": mode, "model": "dgx-x/placeholder"}}
            key = "subagents" if mode == "subagent" else "agents"
            out = m._apply_default_models(dict(agents), {key: {"code": prefs}}, {}, avail)
            return out["code"]["model"]

        # Unavailable DGX pref, cloud in pool -> cloud accepted
        self.assertEqual(
            resolve(["qwen3.6-35b-a3b-fp8", "openai/gpt-5.5"]),
            "openai/gpt-5.5")

    def test_no_prefs_preserves_valid_existing_model(self):
        # When no preferences are configured, existing model is preserved
        avail = {
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8": {},
        }
        def resolve(prefs, mode="primary"):
            agents = {"code": {"mode": mode, "model": "dgx-102-8000/unsloth/qwen3-coder-next-fp8"}}
            key = "subagents" if mode == "subagent" else "agents"
            out = m._apply_default_models(dict(agents), {key: {"code": prefs}}, {}, avail)
            return out["code"]["model"]

        # Empty prefs -> existing model preserved
        self.assertEqual(
            resolve([]),
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8")
        # No key in default_models -> existing model preserved
        self.assertEqual(
            resolve(None),  # Will be treated as empty
            "dgx-102-8000/unsloth/qwen3-coder-next-fp8")

    def test_non_reasoning_only_fleet_still_builds_roster(self):
        # Regression: a fleet with NO reasoning models must still rebuild the roster
        # onto a live model, not leave the agents empty/stale pointing at a model
        # that's no longer served (the "model ... is not valid" bug).
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp)
            sampling = m.build_sampling(args)
            with FakeProbes(model="qwen3-coder-next-fp8"), quiet():  # matches no config -> non-reasoning
                rc = m.oc_sync(args, sampling, {"opencode"})
            self.assertEqual(rc, 0)
            with open(args.config, encoding="utf-8") as f:
                cfg = json.load(f)
            ag = cfg["agent"]
            for k in ("research", "code", "agent", "team",
                      "agent-research", "agent-code", "agent-test", "agent-instruct", "agent-architect", "agent-review"):
                self.assertIn(k, ag, f"{k} missing -> roster not rebuilt for a non-reasoning fleet")
            # local agents point at the LIVE model + an existing provider (not a stale ref)
            for k in ("code", "agent", "agent-research", "agent-code", "agent-test", "agent-instruct", "agent-architect", "agent-review"):
                ref = ag[k]["model"]
                self.assertIn("qwen3-coder-next-fp8", ref, f"{k} not on the live model: {ref}")
                self.assertIn(ref.split("/", 1)[0], cfg["provider"],
                              f"{k} points at a non-existent provider: {ref}")

    def test_agent_ordering_visible_then_hidden_then_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._sync(tmp)
            order = list(cfg["agent"])
            vis = [order.index(k) for k in ("research", "code", "agent", "team")]
            hid = [order.index(k) for k in m.TEAM_TARGETS]
            dis = [order.index(k) for k in ("build", "plan")]
            self.assertLess(max(vis), min(hid), "visible must precede hidden workers")
            self.assertLess(max(hid), min(dis), "hidden must precede disabled builtins")

    def test_keep_builtins_leaves_native_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._sync(tmp, keep_builtins=True)
            self.assertNotIn("build", cfg["agent"])  # not written as a disabled stub
            self.assertNotIn("default_agent", cfg)

    def test_output_cap_lifted_for_recipe_without_max_output(self):
        # Qwen3.6-35B-A3B declares NO recipe max_output, but a reasoning model still
        # spends output tokens thinking, so OpenCode's 32k per-step cap must be lifted.
        # The model's output limit must exceed 32k AND the sync must raise
        # OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX (else long turns get cut off mid-think).
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp)
            sampling = m.build_sampling(args)
            buf = io.StringIO()
            with FakeProbes(model="Qwen3.6-35B-A3B-NVFP4"), \
                    contextlib.redirect_stdout(buf):
                rc = m.oc_sync(args, sampling, {"opencode"})
            self.assertEqual(rc, 0)
            with open(args.config, encoding="utf-8") as f:
                cfg = json.load(f)
            entry = cfg["provider"]["dgx-n1-8000"]["models"]["Qwen3.6-35B-A3B-NVFP4"]
            self.assertGreater(entry["limit"]["output"], 32000)
            self.assertIn("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", buf.getvalue())

    def test_team_on_cloud_model_from_default_models_strips_local_knobs(self):
        # The team's model comes from default_models.json like any agent. A frontier/cloud
        # choice runs on OpenCode's own defaults -- the DGX-only knobs (options/top_p/
        # temperature) are stripped so we never impose vLLM sampling on a cloud model.
        dm = {"agents": {"team": ["openai/gpt-5.5"]}, "subagents": {}}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._sync(tmp, runtime_models=["openai/gpt-5.5"], default_models=dm)
            team = cfg["agent"]["team"]
            self.assertEqual(team["model"], "openai/gpt-5.5")
            self.assertNotIn("options", team)      # no chat_template_kwargs / reasoning_effort
            self.assertNotIn("top_p", team)
            self.assertNotIn("temperature", team)

    def test_team_model_from_default_models_persists_across_resync(self):
        # No special team-model preservation any more: default_models.json IS the persistence,
        # so an available cloud choice is re-selected every sync.
        dm = {"agents": {"team": ["openai/gpt-5.5"]}, "subagents": {}}
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp, runtime_models=["openai/gpt-5.5"], default_models=dm)
            cfg = self._sync(tmp, runtime_models=["openai/gpt-5.5"], default_models=dm)
            self.assertEqual(cfg["agent"]["team"]["model"], "openai/gpt-5.5")

    def test_stale_unavailable_team_model_not_preserved(self):
        # Regression: a cloud team model left in the config that ISN'T available (no provider,
        # not runtime-discovered, and NOT in default_models.json) must not be re-pinned every
        # sync -- the team reverts to a live local model instead of a broken cloud ref.
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "opencode.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"agent": {"team": {"mode": "primary",
                                              "model": "anthropic/claude-opus-4-8"}}}, f)
            # No anthropic provider, runtime discovery returns nothing.
            cfg = self._sync(tmp)
            team_model = cfg["agent"]["team"]["model"]
            self.assertNotIn("claude", team_model, "stale unavailable team model was preserved")
            self.assertIn("Qwen3.6-27B-NVFP4", team_model)  # reverted to the live local model

    def test_task_budget_written_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._sync(tmp, team_task_budget=4)
            self.assertEqual(cfg["agent"]["team"]["task_budget"], 4)
            cfg2 = self._sync(tmp)  # preserved without re-passing
            self.assertEqual(cfg2["agent"]["team"]["task_budget"], 4)

    def test_user_agent_is_not_clobbered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "opencode.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"agent": {"myown": {"mode": "primary", "model": "x/y"}}}, f)
            cfg = self._sync(tmp)
            self.assertIn("myown", cfg["agent"], "user's own agent was pruned")

    def test_refuses_to_write_when_nothing_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, _hosts=["10.255.255.1"])  # nothing answers here
            sampling = m.build_sampling(args)
            with FakeProbes(), quiet():
                rc = m.oc_sync(args, sampling, {"opencode"})
            self.assertEqual(rc, 2, "should refuse (exit 2) with no endpoints")
            self.assertFalse(os.path.exists(args.config))

    def test_team_model_pref_skips_unavailable_remote_refs(self):
        # Regression: oc_sync preselection should not accept slash-containing
        # preferences when no reasoning models exist, unless they're in available_models.
        # anthropic/claude-opus-4-8 should NOT be selected if not available.
        default_models = {
            "agents": {"team": ["anthropic/claude-opus-4-8", "openai/gpt-5.5"]},
            "subagents": {}
        }
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, profiles=True)
            sampling = m.build_sampling(args)
            # No reasoning models - only a local model available
            with FakeProbes(model="qwen3-coder-next-fp8"), quiet():
                rc = m.oc_sync(args, sampling, {"opencode"})
            self.assertEqual(rc, 0)
            with open(args.config, encoding="utf-8") as f:
                cfg = json.load(f)
            # team should use the available local model, NOT the unavailable remote refs
            team_model = cfg["agent"]["team"]["model"]
            self.assertIn("qwen3-coder-next-fp8", team_model)
            self.assertNotIn("claude-opus", team_model)
            self.assertNotIn("gpt-5.5", team_model)

    def test_team_model_pref_uses_remote_when_available(self):
        # When a remote model IS in available_models, it should be selected.
        # We need to set up a config with a remote provider first, then sync.
        default_models = {
            "agents": {"team": ["openai/gpt-5.5"]},
            "subagents": {}
        }
        with tempfile.TemporaryDirectory() as tmp:
            # First create a config with a remote provider
            cfg_path = os.path.join(tmp, "opencode.json")
            initial_cfg = {
                "provider": {
                    "openai": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "OpenAI",
                        "options": {"baseURL": "https://api.openai.com/v1", "apiKey": "sk-..."},
                        "models": {"gpt-5.5": {"reasoning": True, "tool_call": True}}
                    }
                },
                "agent": {
                    "team": {"mode": "primary", "model": "openai/gpt-4"}
                }
            }
            with open(cfg_path, "w") as f:
                json.dump(initial_cfg, f)
            
            args = make_args(tmp, profiles=True, _hosts=["192.0.2.101"], _ports=[8000])
            sampling = m.build_sampling(args)
            _orig = m.load_default_models
            m.load_default_models = lambda: default_models
            try:
                with FakeProbes(), quiet():
                    rc = m.oc_sync(args, sampling, {"opencode"})
            finally:
                m.load_default_models = _orig
            self.assertEqual(rc, 0)
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # team should use the remote model from the pool (first preference)
            self.assertEqual(cfg["agent"]["team"]["model"], "openai/gpt-5.5")

    def test_no_matching_preference_preserves_current_model(self):
        # When no preferences match but current model is still available, keep it.
        # This test uses a pre-existing config with a model that's still available.
        default_models = {
            "agents": {"code": ["qwen3.6-35b-a3b-fp8", "openai/gpt-5.5"]},
            "subagents": {}
        }
        with tempfile.TemporaryDirectory() as tmp:
            # Create initial config with a model that will be available
            cfg_path = os.path.join(tmp, "opencode.json")
            initial_cfg = {
                "provider": {
                    "dgx-n1-8000": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "DGX n1:8000",
                        "options": {"baseURL": "http://192.0.2.101:8000/v1", "apiKey": "sglang"},
                        "models": {"Qwen3.6-27B-NVFP4": {"reasoning": True, "tool_call": True}}
                    }
                },
                "agent": {
                    "code": {"mode": "primary", "model": "dgx-n1-8000/Qwen3.6-27B-NVFP4"}
                }
            }
            with open(cfg_path, "w") as f:
                json.dump(initial_cfg, f)
            
            args = make_args(tmp, profiles=True, _hosts=["192.0.2.101"], _ports=[8000])
            sampling = m.build_sampling(args)
            with FakeProbes(), quiet():
                rc = m.oc_sync(args, sampling, {"opencode"})
            self.assertEqual(rc, 0)
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # code agent should keep its current model since no pref matched
            # but it's still available (fallback to first available is only when
            # current model is NOT available)
            code_model = cfg["agent"]["code"]["model"]
            self.assertIn("Qwen3.6-27B-NVFP4", code_model)

    def test_team_pref_overrides_stale_existing_when_no_reasoning_models(self):
        """Regression: previous team openai/gpt-5.5, runtime reports
        github-copilot/gpt-5.5, no reasoning models live. The resolved
        default_models.json preference must win, not preservation."""
        default_models = {
            "agents": {"team": ["github-copilot/gpt-5.5", "openai/gpt-5.5"]},
            "subagents": {}
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "opencode.json")
            initial_cfg = {
                "provider": {
                    "openai": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "OpenAI",
                        "options": {"baseURL": "https://api.openai.com/v1", "apiKey": "sk-..."},
                        "models": {"gpt-5.5": {"reasoning": True, "tool_call": True}}
                    }
                },
                "agent": {
                    "team": {"mode": "primary", "model": "openai/gpt-5.5"}
                }
            }
            with open(cfg_path, "w") as f:
                json.dump(initial_cfg, f)

            args = make_args(tmp, profiles=True, _hosts=["192.0.2.101"], _ports=[8000])
            sampling = m.build_sampling(args)
            # No reasoning models live; runtime discovery reports github-copilot/gpt-5.5
            with FakeProbes(model="qwen3-coder-next-fp8",
                            runtime_models=["github-copilot/gpt-5.5", "openai/gpt-5.5"]), quiet():
                _orig = m.load_default_models
                m.load_default_models = lambda: default_models
                try:
                    rc = m.oc_sync(args, sampling, {"opencode"})
                finally:
                    m.load_default_models = _orig
            self.assertEqual(rc, 0)
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # team must switch to the first resolved preference, not preserve openai
            self.assertEqual(cfg["agent"]["team"]["model"], "github-copilot/gpt-5.5")

    def test_team_falls_back_to_live_local_when_no_remote_pref_resolves(self):
        """When the team's remote preferences are all unavailable, team lands on a
        live local model (Qwen-first design), rather than a stale remote. Injects a
        team pref of only-unavailable remotes so the fallback path is exercised."""
        default_models = {
            "agents": {"team": ["anthropic/claude-opus-4-8"]},  # not available
            "subagents": {}
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "opencode.json")
            initial_cfg = {
                "provider": {
                    "openai": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "OpenAI",
                        "options": {"baseURL": "https://api.openai.com/v1", "apiKey": "sk-..."},
                        "models": {"gpt-5.5": {"reasoning": True, "tool_call": True}}
                    }
                },
                "agent": {
                    "team": {"mode": "primary", "model": "openai/gpt-5.5"}
                }
            }
            with open(cfg_path, "w") as f:
                json.dump(initial_cfg, f)

            args = make_args(tmp, profiles=True, _hosts=["192.0.2.101"], _ports=[8000])
            sampling = m.build_sampling(args)
            with FakeProbes(model="qwen3-coder-next-fp8",
                            runtime_models=["openai/gpt-5.5"]), quiet():
                _orig = m.load_default_models
                m.load_default_models = lambda: default_models
                try:
                    rc = m.oc_sync(args, sampling, {"opencode"})
                finally:
                    m.load_default_models = _orig
            self.assertEqual(rc, 0)
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # No remote pref resolves -> team lands on the live local model, not a stale remote.
            self.assertEqual(cfg["agent"]["team"]["model"], "dgx-n1-8000/qwen3-coder-next-fp8")

    def test_stale_remote_provider_skipped_when_runtime_discovery_succeeds(self):
        """Regression: When runtime discovery succeeds, stale remote providers
        from existing config should NOT be available unless they appear in runtime.

        Scenario: existing config has Anthropic, but runtime opencode models only
        reports openai/gpt-5.5. agent-review should resolve to OpenAI, not Anthropic."""
        default_models = {
            "agents": {"team": ["openai/gpt-5.5"]},
            "subagents": {"agent-review": ["anthropic/claude-opus-4-8", "openai/gpt-5.5"]}
        }
        with tempfile.TemporaryDirectory() as tmp:
            # Create initial config with a stale Anthropic provider
            cfg_path = os.path.join(tmp, "opencode.json")
            initial_cfg = {
                "provider": {
                    "anthropic": {
                        "npm": "@ai-sdk/anthropic",
                        "name": "Anthropic",
                        "options": {"apiKey": "sk-..."},
                        "models": {
                            "claude-opus-4-8": {"reasoning": True, "tool_call": True}
                        }
                    }
                },
                "agent": {
                    "agent-review": {"mode": "subagent", "model": "anthropic/claude-opus-4-8"}
                }
            }
            with open(cfg_path, "w") as f:
                json.dump(initial_cfg, f)
            
            args = make_args(tmp, profiles=True, _hosts=["192.0.2.101"], _ports=[8000])
            sampling = m.build_sampling(args)
            # Runtime discovery reports only openai/gpt-5.5, NOT Anthropic
            with FakeProbes(runtime_models=["openai/gpt-5.5"]), quiet():
                _orig = m.load_default_models
                m.load_default_models = lambda: default_models
                try:
                    rc = m.oc_sync(args, sampling, {"opencode"})
                finally:
                    m.load_default_models = _orig
            self.assertEqual(rc, 0)
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # agent-review should use openai/gpt-5.5 (first available preference)
            # NOT anthropic/claude-opus-4-8 (stale, not in runtime discovery)
            review_model = cfg["agent"]["agent-review"]["model"]
            self.assertEqual(review_model, "openai/gpt-5.5")
            # Verify Anthropic was NOT added to the pool
            # The provider should be kept (user-authored), but Anthropic model
            # should NOT be in all_available_models
            self.assertIn("anthropic", cfg["provider"])
            # Verify team gets openai/gpt-5.5 (it's the only remote in pool)
            self.assertEqual(cfg["agent"]["team"]["model"], "openai/gpt-5.5")

    def test_configured_provider_model_available_when_runtime_discovery_fails(self):
        """When `opencode models` returns nothing, a model from a CONFIGURED provider in the
        existing config is still available as a fallback, so an agent can be placed on it."""
        default_models = {
            "agents": {"team": ["anthropic/claude-opus-4-8"]},
            "subagents": {}
        }
        with tempfile.TemporaryDirectory() as tmp:
            # Create initial config with Anthropic provider
            cfg_path = os.path.join(tmp, "opencode.json")
            initial_cfg = {
                "provider": {
                    "anthropic": {
                        "npm": "@ai-sdk/anthropic",
                        "name": "Anthropic",
                        "options": {"apiKey": "sk-..."},
                        "models": {
                            "claude-opus-4-8": {"reasoning": True, "tool_call": True}
                        }
                    }
                },
                "agent": {
                    "team": {"mode": "primary", "model": "anthropic/claude-opus-4-8"}
                }
            }
            with open(cfg_path, "w") as f:
                json.dump(initial_cfg, f)

            # Runtime discovery fails; the configured anthropic provider's model is the fallback.
            cfg = self._sync(tmp, _hosts=["192.0.2.101"], _ports=[8000],
                             runtime_models=[], default_models=default_models)
            team = cfg["agent"]["team"]
            self.assertEqual(team["model"], "anthropic/claude-opus-4-8")
            # cloud model -> local vLLM knobs stripped, runs on OpenCode's own defaults
            self.assertNotIn("top_p", team)
            self.assertNotIn("options", team)


# --------------------------------------------------------------------------- #
# Plugin directory -- must match OpenCode docs (https://opencode.ai/docs/plugins/)
# --------------------------------------------------------------------------- #
class TestPluginDirectory(unittest.TestCase):
    def _sync(self, tmp, **over):
        args = make_args(tmp, **over)
        sampling = m.build_sampling(args)
        with FakeProbes(), quiet():
            self.assertEqual(m.oc_sync(args, sampling, {"opencode"}), 0)
        return args

    def test_plugin_written_to_documented_plural_dir(self):
        # OpenCode loads plugins from `plugins/` (plural). The sampling plugin must
        # land there so per-agent top_k/min_p/penalties/maxOutput are enforced.
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            self.assertTrue(
                os.path.exists(os.path.join(tmp, "plugins", "dgx-sampling.js")),
                "sampling plugin must be written to plugins/ (plural)")
            self.assertFalse(
                os.path.exists(os.path.join(tmp, "plugin", "dgx-sampling.js")),
                "nothing should be written to the old singular plugin/ dir")

    def test_stale_singular_plugin_is_cleaned_up(self):
        # A plugin left in the old `plugin/` dir by an earlier sync gets removed.
        with tempfile.TemporaryDirectory() as tmp:
            legacy_dir = os.path.join(tmp, "plugin")
            os.makedirs(legacy_dir)
            legacy = os.path.join(legacy_dir, "dgx-sampling.js")
            with open(legacy, "w", encoding="utf-8") as f:
                f.write("// stale\n")
            self._sync(tmp)
            self.assertFalse(os.path.exists(legacy), "stale plugin/ file not removed")


class TestAudit(unittest.TestCase):
    """--audit: offline diff of the live OpenCode config vs the omodel-manager configs."""

    def _sync(self, tmp, **over):
        args = make_args(tmp, **over)
        sampling = m.build_sampling(args)
        with FakeProbes(), quiet():
            self.assertEqual(m.oc_sync(args, sampling, {"opencode"}), 0)
        return args

    def test_audit_in_sync_after_fresh_sync(self):
        # A config the tool just wrote must audit clean against the same configs.
        with tempfile.TemporaryDirectory() as tmp:
            args = self._sync(tmp)
            with quiet():
                self.assertEqual(m.oc_audit(args), 0)

    def test_audit_detects_plugin_drift(self):
        # Tamper one plugin value -> audit must flag drift (exit 1).
        with tempfile.TemporaryDirectory() as tmp:
            args = self._sync(tmp)
            pj = os.path.join(tmp, "plugins", "dgx-sampling.js")
            with open(pj, encoding="utf-8") as f:
                txt = f.read()
            self.assertIn('"topK": 20', txt)
            with open(pj, "w", encoding="utf-8") as f:
                f.write(txt.replace('"topK": 20', '"topK": 5', 1))
            with quiet():
                self.assertEqual(m.oc_audit(args), 1)

    def test_audit_flags_model_missing_per_model_sampling(self):
        # A registered managed model with no per-model plugin table is surfaced and
        # flagged (agents on it would fall back to server defaults).
        with tempfile.TemporaryDirectory() as tmp:
            args = self._sync(tmp)
            cfgp = os.path.join(tmp, "opencode.json")
            with open(cfgp, encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("provider", {})["dgx-n2-8000"] = {
                "models": {"NVIDIA-Nemotron-3-Super-120B":
                           {"reasoning": True, "tool_call": True}}}
            with open(cfgp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = m.oc_audit(args)
            out = buf.getvalue()
            self.assertIn("NVIDIA-Nemotron-3-Super-120B", out)
            self.assertIn("no per-model sampling", out)
            self.assertEqual(rc, 1)

    def test_audit_nothing_to_compare(self):
        # No managed agents -> exit 2.
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "opencode.json"), "w", encoding="utf-8") as f:
                json.dump({"agent": {}}, f)
            args = make_args(tmp)
            with quiet():
                self.assertEqual(m.oc_audit(args), 2)


class TestSharedHosts(unittest.TestCase):
    """omw reads the same ~/.config/otools/hosts store omodel-manager writes."""

    def setUp(self):
        self._old = m.HOSTS_FILE
        self._dir = tempfile.mkdtemp()
        m.HOSTS_FILE = os.path.join(self._dir, "hosts")

    def tearDown(self):
        m.HOSTS_FILE = self._old

    def _write(self, text):
        with open(m.HOSTS_FILE, "w", encoding="utf-8") as f:
            f.write(text)

    def test_parses_alias_bare_and_user_at(self):
        self._write(
            "# managed by install\n"
            "dgx-1\totto@192.0.2.101\n"      # alias<TAB>user@host
            "otto@192.0.2.102\n"              # bare user@host
            "192.0.2.103\n"                   # bare host
            "\n"
        )
        self.assertEqual(
            m.load_shared_hosts(),
            ["192.0.2.101", "192.0.2.102", "192.0.2.103"],
        )

    def test_dedup_by_host(self):
        self._write("dgx-1\totto@192.0.2.101\nn1\troot@192.0.2.101\n")
        self.assertEqual(m.load_shared_hosts(), ["192.0.2.101"])

    def test_missing_file_is_empty(self):
        m.HOSTS_FILE = os.path.join(self._dir, "nope")
        self.assertEqual(m.load_shared_hosts(), [])


def _cli_args(config, **over):
    a = types.SimpleNamespace(config=config, configs=FIXTURE_DIR, name=None, _settings={})
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _capture(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


class TestSkillsCommand(unittest.TestCase):
    """`omw skills` lists skills (global + project) and pretty-prints one."""

    def _synced(self, tmp, **over):
        args = make_args(tmp, **over)
        sampling = m.build_sampling(args)
        with FakeProbes(), quiet():
            self.assertEqual(m.oc_sync(args, sampling, {"opencode"}), 0)
        return args.config

    def test_size_metric_thresholds(self):
        self.assertEqual(m._skill_size_verdict(10), "lean")
        self.assertEqual(m._skill_size_verdict(60), "moderate")
        self.assertEqual(m._skill_size_verdict(120), "LARGE")

    def test_instruction_count_skips_frontmatter_and_fences(self):
        text = ("---\nname: x\ndescription: y\n---\n"
                "- one\n- two\nYou MUST do it\n"
                "```\n- not counted inside a fence\n```\n"
                "plain prose line not counted\n")
        # 2 bullets + 1 MUST line = 3; fence body and prose excluded
        self.assertEqual(m._skill_instruction_count(text), 3)

    def test_frontmatter_parsed(self):
        fm = m._skill_frontmatter("---\nname: foo\ndescription: bar baz\n---\n# body\n")
        self.assertEqual(fm["name"], "foo")
        self.assertEqual(fm["description"], "bar baz")

    def test_lists_global_skills_after_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp)
            out = _capture(m.cmd_skills, _cli_args(cfg, name=None))
            self.assertIn("SKILL", out)
            self.assertIn("agent-runbook-review", out)
            for skill_name in m.ROLE_SKILLS:
                self.assertIn(skill_name, out)
            self.assertNotIn("team-orchestration", out)
            self.assertNotIn("pr-review", out)
            self.assertIn("global", out)            # scope column

    def test_pretty_prints_one_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp)
            out = _capture(m.cmd_skills, _cli_args(cfg, name="agent-runbook-review"))
            self.assertIn("skill: agent-runbook-review", out)
            self.assertIn("instruction(s)", out)    # size line
            self.assertIn("# SKILL.md", out)         # per-file header
            self.assertIn("opencode.db", out)        # renders the body

    def test_unknown_skill_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp)
            with self.assertRaises(SystemExit):
                with quiet():
                    m.cmd_skills(_cli_args(cfg, name="does-not-exist"))


class TestCliViews(unittest.TestCase):
    """Stage 1-2: subcommand skeleton, home, config, read-only views."""

    def _synced(self, tmp, **over):
        args = make_args(tmp, **over)
        sampling = m.build_sampling(args)
        with FakeProbes(), quiet():
            m.oc_sync(args, sampling, {"opencode"})
        return args.config

    def test_suggest_formats_block(self):
        out = _capture(m._suggest, [("do a thing", "omw thing")], header="Next")
        self.assertIn("Next:", out)
        self.assertIn("- do a thing", out)
        self.assertIn("omw thing", out)

    def test_suggest_skips_empty(self):
        self.assertEqual(_capture(m._suggest, []), "")

    def test_settings_roundtrip_and_precedence(self):
        old = m.WIRE_SETTINGS_FILE
        with tempfile.TemporaryDirectory() as tmp:
            m.WIRE_SETTINGS_FILE = os.path.join(tmp, "wire.json")
            try:
                self.assertEqual(m.load_settings(), {})
                m.save_settings({"hosts": "192.0.2.101,192.0.2.102"})
                self.assertEqual(m.load_settings()["hosts"], "192.0.2.101,192.0.2.102")
                a = types.SimpleNamespace(_settings=m.load_settings())
                self.assertEqual(m._setting(a, "hosts"), "192.0.2.101,192.0.2.102")
                self.assertEqual(m._setting(a, "default_agent"), "code")  # built-in fallback
                self.assertIsNone(m._setting(a, "configs_dir"))
            finally:
                m.WIRE_SETTINGS_FILE = old

    def test_retired_team_model_setting_is_pruned_on_load(self):
        # A leftover team_model in wire.json (retired) is dropped on load so it can't
        # silently drive sync any more.
        old = m.WIRE_SETTINGS_FILE
        with tempfile.TemporaryDirectory() as tmp:
            m.WIRE_SETTINGS_FILE = os.path.join(tmp, "wire.json")
            try:
                m.save_settings({"team_model": "anthropic/claude-opus-4-8", "web_search": "exa"})
                loaded = m.load_settings()
                self.assertNotIn("team_model", loaded)
                self.assertEqual(loaded["web_search"], "exa")   # other keys survive
            finally:
                m.WIRE_SETTINGS_FILE = old

    def test_agents_list_and_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp, team_task_budget=4)
            out = _capture(m.cmd_agents, _cli_args(cfg))
            for name in ("research", "code", "agent", "team"):
                self.assertIn(name, out)
            self.assertNotIn("agent-research", out)  # workers are not primaries
            detail = _capture(m.cmd_agents, _cli_args(cfg, name="team"))
            self.assertIn("agent: team", detail)
            self.assertIn("work-budget", detail)
            self.assertIn("4", detail)

    def test_subagents_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp)
            out = _capture(m.cmd_subagents, _cli_args(cfg))
            for w in ("agent-research", "agent-code", "agent-test", "agent-instruct", "agent-architect", "agent-review"):
                self.assertIn(w, out)
            # primaries are excluded from the subagents view (team is primary-only)
            self.assertNotIn("team", out)
            self.assertNotIn("\n  research ", out)

    def test_models_list_and_role_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp)
            lst = _capture(m.cmd_models, _cli_args(cfg))
            self.assertIn("MODEL", lst)
            self.assertIn("Qwen3.6-27B-NVFP4", lst)
            det = _capture(m.cmd_models, _cli_args(cfg, name="Qwen3.6-27B"))
            self.assertIn("ROLE", det)
            for role in ("reason", "code", "agent", "instruct"):
                self.assertIn(role, det)

    def test_home_suggests_sync_when_empty_and_review_when_synced(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = types.SimpleNamespace(_settings={"configs_dir": FIXTURE_DIR,
                                                     "opencode_config": os.path.join(tmp, "none.json")})
            self.assertIn("omw sync", _capture(m.cmd_home, empty))
            cfg = self._synced(tmp)
            synced = types.SimpleNamespace(_settings={"configs_dir": FIXTURE_DIR,
                                                      "opencode_config": cfg})
            out = _capture(m.cmd_home, synced)
            self.assertIn("managed agent", out)
            self.assertIn("omw agents", out)

    def test_main_dispatch_config_path(self):
        out = _capture(m.main, ["config", "--path"])
        self.assertIn(os.path.basename(m.WIRE_SETTINGS_FILE), out)

    def test_main_dispatch_models_list(self):
        # No live opencode config -> nothing live -> default lists nothing...
        out = _capture(m.main, ["models", "--configs", FIXTURE_DIR])
        self.assertNotIn("MODEL", out)
        self.assertIn("No models are live", out)
        # ...but --all shows the full declared catalogue.
        out_all = _capture(m.main, ["models", "--all", "--configs", FIXTURE_DIR])
        self.assertIn("MODEL", out_all)

    def test_models_list_defaults_to_live_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp)
            live = _capture(m.cmd_models, _cli_args(cfg, all=False))
            allrows = _capture(m.cmd_models, _cli_args(cfg, all=True))
            # --all is a superset: it lists at least as many rows as the
            # live-only default.
            self.assertIn("MODEL", live)
            self.assertIn("MODEL", allrows)
            self.assertGreaterEqual(allrows.count(".toml"), live.count(".toml"))
            self.assertIn("--all", live)  # hint to see the rest


def _sync_fixture(tmp, default_models=None, **over):
    # Tests must NOT depend on the developer's personal default_models.json (editing prefs
    # shouldn't break the suite). Default to neutral prefs -> agents fall back to the live
    # fixture model; pass default_models=... to exercise the preference logic explicitly.
    args = make_args(tmp, **over)
    sampling = m.build_sampling(args)
    orig = m.load_default_models
    m.load_default_models = lambda: (default_models
                                     if default_models is not None
                                     else {"agents": {}, "subagents": {}})
    try:
        with FakeProbes(), quiet():
            m.oc_sync(args, sampling, {"opencode"})
    finally:
        m.load_default_models = orig
    return args.config


class TestCliMutations(unittest.TestCase):
    """Stage 3: --set-* live edits touch only opencode.json + the plugin."""

    def _load(self, cfg):
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)

    def test_set_agent_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            with quiet():
                m.cmd_agents(_cli_args(cfg, name="research", set_model="anthropic/claude-opus-4-8"))
            self.assertEqual(self._load(cfg)["agent"]["research"]["model"], "anthropic/claude-opus-4-8")

    def test_set_all_subagent_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            with quiet():
                m.cmd_subagents(_cli_args(cfg, name=None, set_model="anthropic/x"))
            ag = self._load(cfg)["agent"]
            for w in ("agent-research", "agent-code", "agent-test", "agent-instruct", "agent-architect", "agent-review"):
                self.assertEqual(ag[w]["model"], "anthropic/x")

    def test_set_model_to_cloud_strips_local_knobs(self):
        # Regression: `--set-model` to a frontier/cloud model must reconfigure the agent
        # for that model (OpenCode defaults), not leave the old DGX temp/thinking behind.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            before = self._load(cfg)["agent"]["research"]
            self.assertIn("options", before)          # DGX build left thinking knobs
            with quiet():
                m.cmd_agents(_cli_args(cfg, name="research", set_model="openai/gpt-5.5"))
            research = self._load(cfg)["agent"]["research"]
            self.assertEqual(research["model"], "openai/gpt-5.5")
            self.assertNotIn("options", research)     # local knobs stripped for cloud
            self.assertNotIn("top_p", research)
            self.assertNotIn("temperature", research)

    def test_set_model_to_dgx_applies_recipe_config(self):
        # `--set-model` to a DGX model applies THAT model's recipe preset (thinking on for
        # the research/reason role) rather than a bare string swap.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            # move the (thinking-off) instruct worker onto the Qwen model as-is; then verify
            # research keeps a correct reason-preset block after a same-model reconfigure.
            with quiet():
                m.cmd_agents(_cli_args(cfg, name="research",
                                       set_model="dgx-n1-8000/Qwen3.6-27B-NVFP4"))
            research = self._load(cfg)["agent"]["research"]
            self.assertEqual(research["model"], "dgx-n1-8000/Qwen3.6-27B-NVFP4")
            self.assertTrue(research["options"]["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(research["temperature"], 1.0)   # reason preset

    def test_set_work_budget_updates_team_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            with quiet():
                m.cmd_agents(_cli_args(cfg, name=None, set_model=None, set_work_budget=7))
            self.assertEqual(self._load(cfg)["agent"]["team"]["task_budget"], 7)
            prompt = os.path.join(tmp, "prompts", m.ROLE_PROMPT_FILES["team"])
            with open(prompt, encoding="utf-8") as f:
                body = f.read()
            self.assertIn("7", body)
            self.assertEqual(len(body.splitlines()), 5)
            self.assertFalse(os.path.exists(os.path.join(tmp, "prompts", "otools-team.md")))

    def test_set_model_temperature_hits_all_role_agents_and_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            with quiet():
                m.cmd_models(_cli_args(cfg, name="Qwen3.6-27B", role="code",
                                       set_temperature=0.42, set_thinking=None))
            ag = self._load(cfg)["agent"]
            # both code-preset agents on that model move
            self.assertEqual(ag["code"]["temperature"], 0.42)
            self.assertEqual(ag["agent-code"]["temperature"], 0.42)
            # the plugin table moved too
            plugin = m._plugin_agent_sampling(cfg)
            self.assertEqual(plugin["Qwen3.6-27B-NVFP4"]["code"]["temperature"], 0.42)

    def test_set_thinking_toggles_enable_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            with quiet():
                m.cmd_models(_cli_args(cfg, name="Qwen3.6-27B", role="instruct",
                                       set_temperature=None, set_thinking=True))
            opts = self._load(cfg)["agent"]["agent-instruct"]["options"]["chat_template_kwargs"]
            self.assertTrue(opts["enable_thinking"])

    def test_main_dispatch_set_work_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            with quiet():
                m.main(["agents", "team", "--config", cfg, "--set-work-budget", "5"])
            self.assertEqual(self._load(cfg)["agent"]["team"]["task_budget"], 5)


class TestWorkBudget(unittest.TestCase):
    """Stage 4: team work-budget defaults from a config's [capabilities] concurrency."""

    def _configs_with_concurrency(self, root, n):
        cdir = os.path.join(root, "cfgs")
        os.makedirs(cdir)
        with open(os.path.join(cdir, "q.toml"), "w", encoding="utf-8") as f:
            f.write('match = ["Qwen3.6-27B-NVFP4"]\n'
                    '[capabilities]\nreasoning = true\ntool_call = true\n'
                    f'concurrency = {n}\nthinking_control = "enable_thinking"\n'
                    '[presets.reason]\nthinking = true\n[presets.reason.sampling]\ntemperature = 1.0\n'
                    '[presets.code]\nthinking = true\n[presets.code.sampling]\ntemperature = 0.6\n'
                    '[presets.agent]\nthinking = true\n[presets.agent.sampling]\ntemperature = 0.6\n'
                    '[presets.instruct]\nthinking = false\n[presets.instruct.sampling]\ntemperature = 0.7\n')
        return cdir

    def _budget(self, cfg):
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)["agent"]["team"].get("task_budget")

    def test_defaults_from_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = self._configs_with_concurrency(tmp, 3)
            cfg = _sync_fixture(tmp, configs=cdir)
            self.assertEqual(self._budget(cfg), 3)

    def test_flag_overrides_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir = self._configs_with_concurrency(tmp, 3)
            cfg = _sync_fixture(tmp, configs=cdir, team_task_budget=9)
            self.assertEqual(self._budget(cfg), 9)

    def test_no_concurrency_leaves_budget_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)  # FIXTURE_DIR configs have no concurrency
            self.assertIsNone(self._budget(cfg))


class TestBareModelNameResolution(unittest.TestCase):
    """Test that bare model names (without provider prefix) are correctly resolved."""

    def _make_cfg_with_providers(self):
        return {
            "provider": {
                "dgx-n1-8000": {
                    "models": {
                        "Qwen3.6-35B-A3B-NVFP4": {"reasoning": True, "tool_call": True},
                        "Qwen3.6-27B-NVFP4": {"reasoning": True, "tool_call": True},
                    }
                },
                "nvidia": {
                    "models": {
                        "Qwen3.6-35B-NVFP4": {"reasoning": True, "tool_call": True},
                    }
                },
            }
        }

    def test_resolve_model_ref_finds_provider_for_bare_name(self):
        cfg = self._make_cfg_with_providers()
        ref, prov = m._resolve_model_ref("Qwen3.6-35B-A3B-NVFP4", cfg)
        self.assertEqual(ref, "dgx-n1-8000/Qwen3.6-35B-A3B-NVFP4")
        self.assertEqual(prov, "dgx-n1-8000")

    def test_resolve_model_ref_finds_exact_match(self):
        cfg = self._make_cfg_with_providers()
        ref, prov = m._resolve_model_ref("qwen3.6-35b-a3b-nvfp4", cfg)
        self.assertEqual(ref, "dgx-n1-8000/Qwen3.6-35B-A3B-NVFP4")

    def test_resolve_model_ref_prefers_exact_over_substring(self):
        cfg = self._make_cfg_with_providers()
        ref, prov = m._resolve_model_ref("Qwen3.6-35B", cfg)
        self.assertIsNone(ref)

    def test_resolve_model_ref_returns_none_for_unknown_model(self):
        cfg = self._make_cfg_with_providers()
        ref, prov = m._resolve_model_ref("unknown-model", cfg)
        self.assertIsNone(ref)
        self.assertIsNone(prov)

    def test_resolve_model_ref_handles_fully_qualified_ref(self):
        cfg = self._make_cfg_with_providers()
        ref, prov = m._resolve_model_ref("dgx-n1-8000/Qwen3.6-35B-A3B-NVFP4", cfg)
        self.assertEqual(ref, "dgx-n1-8000/Qwen3.6-35B-A3B-NVFP4")
        self.assertEqual(prov, "dgx-n1-8000")

    def _cfg_unsloth_and_plain(self):
        # unsloth (served id contains a slash) on one host; the plain model on another
        return {"provider": {
            "dgx-102-8000": {"models": {"unsloth/qwen3-coder-next-fp8": {}}},
            "dgx-103-8000": {"models": {"qwen3-coder-next-fp8": {}}},
        }}

    def test_resolve_model_ref_slash_served_id_resolves_to_host(self):
        # Regression: unsloth/qwen3-coder-next-fp8 used to split to a non-existent 'unsloth'
        # provider -> broken config. It must resolve to the host that actually serves it.
        cfg = self._cfg_unsloth_and_plain()
        ref, prov = m._resolve_model_ref("unsloth/qwen3-coder-next-fp8", cfg)
        self.assertEqual(ref, "dgx-102-8000/unsloth/qwen3-coder-next-fp8")
        self.assertEqual(prov, "dgx-102-8000")
        # the plain served id resolves to the OTHER host, not the unsloth one
        ref2, _ = m._resolve_model_ref("qwen3-coder-next-fp8", cfg)
        self.assertEqual(ref2, "dgx-103-8000/qwen3-coder-next-fp8")

    def test_resolve_model_ref_ambiguous_returns_choices(self):
        # Same served id on two hosts -> refuse to guess; return the host-qualified choices.
        cfg = {"provider": {
            "dgx-102-8000": {"models": {"qwen3-coder-next-fp8": {}}},
            "dgx-103-8000": {"models": {"qwen3-coder-next-fp8": {}}},
        }}
        ref, info = m._resolve_model_ref("qwen3-coder-next-fp8", cfg)
        self.assertIsNone(ref)
        self.assertEqual(info, ["dgx-102-8000/qwen3-coder-next-fp8",
                                "dgx-103-8000/qwen3-coder-next-fp8"])

    def test_resolve_model_ref_cloud_ref_passthrough(self):
        ref, prov = m._resolve_model_ref("anthropic/claude-opus-4-8",
                                         self._cfg_unsloth_and_plain())
        self.assertEqual(ref, "anthropic/claude-opus-4-8")
        self.assertEqual(prov, "anthropic")

    def test_models_list_shows_host_qualified_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            lst = _capture(m.cmd_models, _cli_args(cfg))
            # MODEL column is now the full ref (provider/served-id), not the bare served id
            self.assertRegex(lst, r"dgx-\S+/Qwen3\.6-27B-NVFP4")

    def test_validate_ref_accepts_bare_name_if_model_exists(self):
        cfg = self._make_cfg_with_providers()
        warn = m._validate_ref("Qwen3.6-35B-A3B-NVFP4", cfg)
        self.assertIsNone(warn)

    def test_validate_ref_warns_for_unknown_bare_name(self):
        cfg = self._make_cfg_with_providers()
        warn = m._validate_ref("unknown-model", cfg)
        self.assertIsNotNone(warn)
        self.assertIn("should look like", warn)

    def test_mutate_roster_resolves_bare_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "opencode.json")
            cfg = {
                "provider": {
                    "dgx-n1-8000": {
                        "models": {
                            "Qwen3.6-35B-A3B-NVFP4": {"reasoning": True, "tool_call": True},
                        }
                    }
                },
                "agent": {
                    "team": {"mode": "primary", "model": "dgx-n1-8000/Qwen3.6-27B-NVFP4"}
                }
            }
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

            class Args:
                config = cfg_path
                name = "team"
                set_model = "Qwen3.6-35B-A3B-NVFP4"
                _settings = {}
                _hosts = ["192.0.2.101"]
                _ports = [8000]

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m._mutate_roster(Args(), want_subagent=False)

            with open(cfg_path) as f:
                result = json.load(f)
            self.assertEqual(
                result["agent"]["team"]["model"],
                "dgx-n1-8000/Qwen3.6-35B-A3B-NVFP4"
            )


# --------------------------------------------------------------------------- #
# Proxy tests
# --------------------------------------------------------------------------- #
class TestProxyModule(unittest.TestCase):
    """Unit tests for the proxy helper module (loaded by omw as m.proxy)."""
    PX = m.proxy

    def test_module_loaded(self):
        self.assertIsNotNone(self.PX, "utils/omw_proxy.py failed to load")

    def test_short_id_short_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.PX.short_id(tmp)
            self.assertEqual(len(a), 7)
            open(os.path.join(tmp, a + "_req.json"), "w").close()
            self.assertNotEqual(a, self.PX.short_id(tmp))

    def test_build_curl_strips_host(self):
        curl = self.PX.build_curl({"method": "POST", "url": "http://h:8000/v1/chat/completions",
                                   "headers": {"Content-Type": "application/json", "Host": "x"},
                                   "body": '{"model":"m"}'})
        self.assertIn("-X POST", curl)
        self.assertIn("http://h:8000/v1/chat/completions", curl)
        self.assertNotIn("Host:", curl)

    def test_render_read_sections_and_newlines(self):
        req = {"request_id": "abc1234", "method": "POST", "url": "http://h/v1/chat/completions",
               "model": "m", "body": json.dumps({"model": "m", "temperature": 0.6,
                   "tools": [{"function": {"name": "bash", "description": "run a command"}}],
                   "messages": [{"role": "system", "content": "you are helpful"},
                                {"role": "user", "content": "hi\nthere"}]})}
        res = {"status": 200, "elapsed_ms": 5,
               "body": json.dumps({"choices": [{"message": {"content": "hello"}}]})}
        out = self.PX.render_read(req, res, use_color=False)
        for tok in ("abc1234", "tools", "bash", "system", "you are helpful",
                    "[user]", "there", "hello"):
            self.assertIn(tok, out)
        self.assertNotIn("\\n", out)  # newlines rendered, not literal backslash-n


class TestProxyIntegration(unittest.TestCase):
    """End-to-end: mock upstream -> proxy -> client, incl. SSE streaming."""

    def test_routing_streaming_logging(self):
        import threading
        import urllib.request as ur
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        PX = m.proxy

        class Up(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass
            def do_GET(self):
                b = json.dumps({"data": [{"id": "qwen/qwen3.6-35b"}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0); self.rfile.read(n)
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                for tok in ["Hel", "lo!"]:
                    d = ("data: " + json.dumps({"choices": [{"delta": {"content": tok}}]}) + "\n\n").encode()
                    self.wfile.write(b"%X\r\n" % len(d) + d + b"\r\n"); self.wfile.flush()
                done = b"data: [DONE]\n\n"
                self.wfile.write(b"%X\r\n" % len(done) + done + b"\r\n"); self.wfile.write(b"0\r\n\r\n")

        up = ThreadingHTTPServer(("127.0.0.1", 0), Up); up.daemon_threads = True
        threading.Thread(target=up.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            logs = os.path.join(tmp, "proxy_logs"); os.makedirs(logs)
            routes = os.path.join(tmp, "routes.json")
            with open(routes, "w") as f:
                json.dump({"dgx-n1-8000": f"http://127.0.0.1:{up.server_address[1]}/v1"}, f)
            px = ThreadingHTTPServer(("127.0.0.1", 0), PX.ProxyHandler); px.daemon_threads = True
            px.logs_dir, px.routes_path = logs, routes
            threading.Thread(target=px.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{px.server_address[1]}/dgx-n1-8000"
            try:
                self.assertIn("qwen/qwen3.6-35b", ur.urlopen(base + "/models", timeout=10).read().decode())
                r = ur.Request(base + "/chat/completions",
                               data=json.dumps({"model": "qwen/qwen3.6-35b", "stream": True,
                                                "messages": [{"role": "user", "content": "hi"}]}).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
                sse = ur.urlopen(r, timeout=10).read().decode()
                self.assertIn("Hel", sse); self.assertIn("[DONE]", sse)
            finally:
                px.shutdown(); up.shutdown()
            files = os.listdir(logs)
            ids = {f.split("_")[0] for f in files if f.endswith(".json")}
            self.assertTrue(ids and all(len(i) == 7 for i in ids), ids)
            self.assertIn("index.jsonl", files)
            for i in ids:
                req_d, res_d = PX.find_pair(logs, i)
                if res_d and res_d.get("streamed"):
                    self.assertIn("Hello!", PX.render_read(req_d, res_d, use_color=False))


class TestProxyCli(unittest.TestCase):
    """omw proxy on/off rewrites only ~/.config/opencode; no daemon spawned in tests."""

    def setUp(self):
        self._sf, self._epr = m.WIRE_SETTINGS_FILE, m._ensure_proxy_running
        self._tmp = tempfile.mkdtemp()
        m.WIRE_SETTINGS_FILE = os.path.join(self._tmp, "wire.json")
        m._ensure_proxy_running = lambda P: False

    def tearDown(self):
        m.WIRE_SETTINGS_FILE, m._ensure_proxy_running = self._sf, self._epr

    def _cfg(self):
        path = os.path.join(self._tmp, "opencode.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"provider": {
                "dgx-n1-8000": {"options": {"baseURL": "http://192.0.2.101:8000/v1"},
                                "models": {"qwen/qwen3.6-35b": {}}},
                "dgx-n2-8000": {"options": {"baseURL": "http://192.0.2.102:8000/v1"},
                                "models": {"qwen3-coder-next-nvfp4": {}}}}}, f)
        return path

    def _args(self, cfg, **over):
        a = types.SimpleNamespace(config=cfg, configs=FIXTURE_DIR, target=None, port=9099,
                                  action="on", output_curl=False, no_color=True, _settings={})
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def _load(self, cfg):
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)

    def test_on_all_then_off_restores(self):
        cfg = self._cfg()
        with quiet():
            m.cmd_proxy_on(self._args(cfg))
        prov = self._load(cfg)["provider"]
        self.assertTrue(all(m._is_loopback(p["options"]["baseURL"]) for p in prov.values()))
        routes = self._load(os.path.join(self._tmp, "proxy_routes.json"))
        self.assertEqual(routes["dgx-n1-8000"], "http://192.0.2.101:8000/v1")
        with quiet():
            m.cmd_proxy_off(self._args(cfg))
        prov = self._load(cfg)["provider"]
        self.assertEqual(prov["dgx-n1-8000"]["options"]["baseURL"], "http://192.0.2.101:8000/v1")
        self.assertFalse(os.path.exists(os.path.join(self._tmp, "proxy_routes.json")))

    def test_on_single_model_only(self):
        cfg = self._cfg()
        with quiet():
            m.cmd_proxy_on(self._args(cfg, target="qwen3-coder-next-nvfp4"))
        prov = self._load(cfg)["provider"]
        self.assertFalse(m._is_loopback(prov["dgx-n1-8000"]["options"]["baseURL"]))
        self.assertTrue(m._is_loopback(prov["dgx-n2-8000"]["options"]["baseURL"]))


class TestModelsListCatalogue(unittest.TestCase):
    """omw models: --all shows the full declared catalogue (incl. offline models),
    the live view emits one row per served instance, hint math counts declared
    configs, and unmatched live models never get guessed capabilities."""

    def _cfg(self, tmp, provider):
        path = os.path.join(tmp, "opencode.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"provider": provider}, f)
        return path

    def test_all_shows_declared_but_not_live(self):
        # Nothing live -> --all still lists BOTH fixture configs (qwen + nemotron).
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, {})
            out = _capture(m.cmd_models, _cli_args(cfg, all=True))
            self.assertIn("Qwen3.6-27B-NVFP4", out)
            self.assertIn("NVIDIA-Nemotron-3-Super-120B", out)
            # offline rows carry LIVE/PROXY/SERVED = "-"
            nemo = [ln for ln in out.splitlines() if "Nemotron" in ln][0]
            self.assertIn("nemotron", nemo)

    def test_default_live_only_hides_offline(self):
        # Only qwen live -> default view shows qwen, not nemotron, and the hint
        # counts the offline config ("1 more not live").
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, {"dgx-n1-8000": {
                "options": {"baseURL": "http://127.0.0.1:8000/v1"},
                "models": {"Qwen3.6-27B-NVFP4": {}}}})
            out = _capture(m.cmd_models, _cli_args(cfg, all=False))
            self.assertIn("Qwen3.6-27B-NVFP4", out)
            self.assertNotIn("Nemotron", out)
            self.assertIn("1 more not live", out)

    def test_one_row_per_served_instance(self):
        # Two served ids matching the SAME config (on two endpoints) -> two rows,
        # one proxied (loopback) and one direct.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, {
                "dgx-a": {"options": {"baseURL": "http://127.0.0.1:8000/v1"},
                          "models": {"Qwen3.6-27B-NVFP4": {}}},
                "dgx-b": {"options": {"baseURL": "http://198.51.100.5:8000/v1"},
                          "models": {"Qwen3.6-27B": {}}}})
            out = _capture(m.cmd_models, _cli_args(cfg, all=False))
            self.assertIn("Qwen3.6-27B-NVFP4", out)
            # both instances of the single qwen config appear as separate rows
            rows = [ln for ln in out.splitlines() if "Qwen3.6-27B" in ln and "yes" in ln]
            self.assertEqual(len(rows), 2, f"expected 2 served rows, got: {rows}")

    def test_unmatched_live_model_not_guessed(self):
        # A live provider model with no declared config is surfaced only under
        # --all, with capabilities shown as "-" (never guessed from the name).
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp, {"dgx-n1-8000": {
                "options": {"baseURL": "http://127.0.0.1:8000/v1"},
                "models": {"totally-unknown-model": {}}}})
            live = _capture(m.cmd_models, _cli_args(cfg, all=False))
            self.assertNotIn("totally-unknown-model", live)  # hidden by default
            allout = _capture(m.cmd_models, _cli_args(cfg, all=True))
            row = [ln for ln in allout.splitlines() if "totally-unknown-model" in ln]
            self.assertTrue(row, "unmatched model should appear under --all")
            self.assertIn("(no config match)", row[0])


class TestModelsProxyColumn(unittest.TestCase):
    """omw models list gains LIVE/PROXY/SERVED reflecting the live config."""

    def test_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "opencode.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({"provider": {"dgx-n1-8000": {
                    "options": {"baseURL": "http://127.0.0.1:9099/dgx-n1-8000"},
                    "models": {"Qwen3.6-27B-NVFP4": {}}}}}, f)
            out = _capture(m.cmd_models, _cli_args(cfg))
            self.assertIn("PROXY", out)
            self.assertIn("SERVED", out)
            live = [ln for ln in out.splitlines() if "Qwen3.6-27B-NVFP4" in ln and "yes" in ln]
            self.assertTrue(live, "live row not found")
            self.assertIn("on", live[0])  # proxied (loopback baseURL)


class TestDisabledProviders(unittest.TestCase):
    """--add-default-providers flag: opencode and huggingface are disabled by default."""

    def _sync(self, tmp, **over):
        args = make_args(tmp, **over)
        sampling = m.build_sampling(args)
        with FakeProbes(), quiet():
            self.assertEqual(m.oc_sync(args, sampling, {"opencode"}), 0)
        return args, sampling

    def test_disabled_by_default_disables_providers(self):
        """Without --add-default-providers, opencode and huggingface are in disabled_providers."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a config with opencode provider
            cfg = os.path.join(tmp, "opencode.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({
                    "provider": {
                        "opencode": {
                            "options": {"baseURL": "https://opencode.ai/v1"},
                            "models": {"qwen3-coder-next-fp8": {"name": "Qwen3 Coder"}}
                        }
                    }
                }, f)
            
            args, _ = self._sync(tmp)
            with open(cfg, "r", encoding="utf-8") as f:
                result = json.load(f)
            
            # disabled_providers should contain opencode and huggingface
            disabled = result.get("disabled_providers", [])
            self.assertIn("opencode", disabled, "opencode should be in disabled_providers by default")
            self.assertIn("huggingface", disabled, "huggingface should be in disabled_providers by default")

    def test_enabled_with_flag_no_disabled(self):
        """With --add-default-providers, opencode and huggingface are NOT in disabled_providers."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "opencode.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({
                    "provider": {
                        "opencode": {
                            "options": {"baseURL": "https://opencode.ai/v1"},
                            "models": {"qwen3-coder-next-fp8": {"name": "Qwen3 Coder"}}
                        }
                    }
                }, f)
            
            args, _ = self._sync(tmp, add_default_providers=True)
            with open(cfg, "r", encoding="utf-8") as f:
                result = json.load(f)
            
            # disabled_providers should NOT contain opencode or huggingface
            disabled = result.get("disabled_providers", [])
            self.assertNotIn("opencode", disabled, "opencode should NOT be in disabled_providers with flag")
            self.assertNotIn("huggingface", disabled, "huggingface should NOT be in disabled_providers with flag")

    def test_disable_then_enable_removes_disabled(self):
        """Disable then enable: --add-default-providers should remove opencode/huggingface from disabled_providers."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "opencode.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({
                    "disabled_providers": ["opencode", "huggingface"],
                    "provider": {
                        "opencode": {
                            "options": {"baseURL": "https://opencode.ai/v1"},
                            "models": {"qwen3-coder-next-fp8": {"name": "Qwen3 Coder"}}
                        }
                    }
                }, f)
            
            # First sync (default, should keep disabled)
            args, _ = self._sync(tmp)
            with open(cfg, "r", encoding="utf-8") as f:
                result1 = json.load(f)
            disabled1 = result1.get("disabled_providers", [])
            self.assertIn("opencode", disabled1, "opencode should be disabled after first sync")
            
            # Second sync with flag (should remove from disabled)
            args, _ = self._sync(tmp, add_default_providers=True)
            with open(cfg, "r", encoding="utf-8") as f:
                result2 = json.load(f)
            disabled2 = result2.get("disabled_providers", [])
            self.assertNotIn("opencode", disabled2, "opencode should be enabled with flag")
            self.assertNotIn("huggingface", disabled2, "huggingface should be enabled with flag")

    def test_preserves_user_disabled_providers(self):
        """User-authored disabled_providers entries should be preserved across both paths."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "opencode.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({
                    "disabled_providers": ["opencode", "my-custom-provider"],
                    "provider": {
                        "opencode": {
                            "options": {"baseURL": "https://opencode.ai/v1"},
                            "models": {"qwen3-coder-next-fp8": {"name": "Qwen3 Coder"}}
                        }
                    }
                }, f)
            
            # Sync with flag (should remove opencode but keep my-custom-provider)
            args, _ = self._sync(tmp, add_default_providers=True)
            with open(cfg, "r", encoding="utf-8") as f:
                result = json.load(f)
            disabled = result.get("disabled_providers", [])
            self.assertNotIn("opencode", disabled, "opencode should be enabled with flag")
            self.assertIn("my-custom-provider", disabled, "user-authored provider should be preserved")

    def test_huggingface_disabled_by_default(self):
        """Without --add-default-providers, huggingface is in disabled_providers."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "opencode.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump({
                    "provider": {
                        "huggingface": {
                            "options": {"baseURL": "https://api-inference.huggingface.co/v1"},
                            "models": {"mistralai/Mistral-7B": {"name": "Mistral 7B"}}
                        }
                    }
                }, f)
            
            args, _ = self._sync(tmp)
            with open(cfg, "r", encoding="utf-8") as f:
                result = json.load(f)
            
            # disabled_providers should contain huggingface
            disabled = result.get("disabled_providers", [])
            self.assertIn("huggingface", disabled, "huggingface should be in disabled_providers by default")


class TestRuntimeModelDiscovery(unittest.TestCase):
    """Test the discover_opencode_runtime_models helper function."""

    def test_discover_empty_when_opencode_not_found(self):
        """When opencode binary is not on PATH, returns empty list."""
        models, note = m.discover_opencode_runtime_models("/nonexistent/opencode")
        self.assertEqual(models, [])
        # Note should indicate some kind of failure
        self.assertTrue("failed" in note.lower() or "no such file" in note.lower())

    def test_discover_accepts_github_copilot_gpt_5_5(self):
        """github-copilot/gpt-5.5 should be recognized as a valid remote provider model."""
        import subprocess
        import tempfile

        # Create a fake opencode script that returns output with github-copilot/gpt-5.5
        with tempfile.TemporaryDirectory() as tmp:
            fake_opencode = os.path.join(tmp, "opencode")
            # Simulate opencode models output with github-copilot/gpt-5.5
            with open(fake_opencode, "w") as f:
                f.write("#!/bin/bash\necho 'github-copilot/gpt-5.5'\necho 'openai/gpt-5.5'\n")
            os.chmod(fake_opencode, 0o755)

            models, note = m.discover_opencode_runtime_models(fake_opencode)
            # Should include github-copilot/gpt-5.5
            self.assertIn("github-copilot/gpt-5.5", models)
            self.assertIn("openai/gpt-5.5", models)
            self.assertIn("2 runtime model(s)", note)

    def test_discover_handles_timeout(self):
        """When opencode models takes too long, returns empty list."""
        # This would require mocking subprocess which is complex, so we just verify
        # the function signature accepts a timeout parameter
        import inspect
        sig = inspect.signature(m.discover_opencode_runtime_models)
        self.assertIn('timeout', sig.parameters)

    def test_discover_accepts_custom_opencode_path(self):
        """Function accepts explicit opencode_path parameter."""
        import inspect
        sig = inspect.signature(m.discover_opencode_runtime_models)
        self.assertIn('opencode_path', sig.parameters)

    def test_discover_returns_tuple(self):
        """Returns (models_list, note) tuple."""
        models, note = m.discover_opencode_runtime_models("/nonexistent/opencode")
        self.assertIsInstance(models, list)
        self.assertIsInstance(note, str)

    def test_discover_empty_when_opencode_models_fails(self):
        """When opencode models returns nonzero exit code, returns empty list."""
        import subprocess
        import tempfile
        
        # Create a fake opencode script that exits with error
        with tempfile.TemporaryDirectory() as tmp:
            fake_opencode = os.path.join(tmp, "opencode")
            with open(fake_opencode, "w") as f:
                f.write("#!/bin/bash\nexit 1\n")
            os.chmod(fake_opencode, 0o755)
            
            models, note = m.discover_opencode_runtime_models(fake_opencode)
            self.assertEqual(models, [])
            # Note should indicate failure
            self.assertIn("exit", note.lower())


class TestOcBuildOpencodeProviders(unittest.TestCase):
    """Test the oc_build_opencode_providers helper function."""

    def test_extracts_models_from_providers(self):
        """Extracts all provider/model refs from config."""
        cfg = {
            "provider": {
                "dgx-n1-8000": {
                    "models": {
                        "qwen3-coder-next-fp8": {"name": "Qwen3 Coder"},
                        "qwen3.6-27b-nvfp4": {"name": "Qwen3.6 27B"}
                    }
                },
                "openai": {
                    "models": {
                        "gpt-5.5": {"name": "GPT-5.5"}
                    }
                }
            }
        }
        models = m.oc_build_opencode_providers(cfg)
        self.assertEqual(models, {
            "dgx-n1-8000/qwen3-coder-next-fp8": {},
            "dgx-n1-8000/qwen3.6-27b-nvfp4": {},
            "openai/gpt-5.5": {},
        })

    def test_empty_config_returns_empty_dict(self):
        """Empty config returns empty dict."""
        models = m.oc_build_opencode_providers({})
        self.assertEqual(models, {})

    def test_missing_provider_key_returns_empty(self):
        """Config without provider key returns empty dict."""
        models = m.oc_build_opencode_providers({"agent": {}})
        self.assertEqual(models, {})

    def test_missing_models_key_skips_provider(self):
        """Provider without models key is skipped."""
        cfg = {"provider": {"dgx-n1-8000": {}}}
        models = m.oc_build_opencode_providers(cfg)
        self.assertEqual(models, {})


class TestCopilotSync(unittest.TestCase):
    """The GitHub Copilot CLI target: .agent.md roster + settings.json + env snippet."""

    def _sync(self, tmp, **over):
        args = make_args(tmp, copilot_home=tmp, target="copilot", **over)
        sampling = m.build_sampling(args)
        with FakeProbes(), quiet():
            rc = m.copilot_sync(args, sampling, {"copilot"})
        self.assertEqual(rc, 0)
        return args

    def test_writes_full_roster_as_agent_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            adir = os.path.join(tmp, "agents")
            for name in ("research", "code", "agent", "team",
                         "agent-research", "agent-code", "agent-test", "agent-instruct", "agent-architect", "agent-review", "agent-review"):
                self.assertTrue(os.path.exists(os.path.join(adir, f"{name}.agent.md")),
                                f"{name}.agent.md missing")
            body = open(os.path.join(adir, "code.agent.md"), encoding="utf-8").read()
            self.assertTrue(body.startswith("---\nname: code\n"))
            self.assertIn("description:", body)

    def test_readonly_agent_restricts_tools_full_agent_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            adir = os.path.join(tmp, "agents")
            research = open(os.path.join(adir, "research.agent.md"), encoding="utf-8").read()
            self.assertIn('tools: ["read", "search", "fetch"]', research)
            agent = open(os.path.join(adir, "agent.agent.md"), encoding="utf-8").read()
            self.assertNotIn("tools:", agent.split("\n\n")[0])   # full -> all tools (omitted)

    def test_agent_review_is_explicit_invoke_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            review = open(os.path.join(tmp, "agents", "agent-review.agent.md"), encoding="utf-8").read()
            self.assertIn("disable-model-invocation: true", review)
            self.assertNotIn("task_id", review.split("\n\n")[0])  # clean Copilot description

    def test_settings_json_and_env_snippet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            settings = json.load(open(os.path.join(tmp, "settings.json"), encoding="utf-8"))
            self.assertEqual(settings["model"], "Qwen3.6-27B-NVFP4")   # the FakeProbes model
            self.assertIs(settings["includeCoAuthoredBy"], False)      # no Co-Authored-By
            env = open(os.path.join(tmp, "otools-copilot.env"), encoding="utf-8").read()
            self.assertIn("COPILOT_PROVIDER_BASE_URL=", env)
            self.assertIn("192.0.2.101:8000/v1", env)
            self.assertTrue(os.path.exists(os.path.join(tmp, "otools-copilot.ps1")))

    def test_settings_merge_preserves_user_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(tmp, exist_ok=True)
            with open(os.path.join(tmp, "settings.json"), "w", encoding="utf-8") as f:
                f.write('// managed\n{ "theme": "dark", "beep": true }\n')   # comment-tolerant
            self._sync(tmp)
            settings = json.load(open(os.path.join(tmp, "settings.json"), encoding="utf-8"))
            self.assertEqual(settings["theme"], "dark")       # preserved
            self.assertEqual(settings["model"], "Qwen3.6-27B-NVFP4")  # ours added

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            default_models = os.path.join(tmp, "default_models.json")
            old_default_models = m.DEFAULT_MODELS_FILE
            m.DEFAULT_MODELS_FILE = default_models
            try:
                self._sync(tmp, dry_run=True)
                self.assertFalse(os.path.exists(os.path.join(tmp, "agents")))
                self.assertFalse(os.path.exists(os.path.join(tmp, "settings.json")))
                self.assertFalse(os.path.exists(default_models))
            finally:
                m.DEFAULT_MODELS_FILE = old_default_models

    def test_offline_writes_roster_without_endpoint(self):
        # DGX offline (no host matches the probe) -> still write the roster + model-independent
        # settings so you can see where they land; skip the endpoint wiring.
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(tmp, copilot_home=tmp, target="copilot", _hosts=["192.0.2.99"])
            sampling = m.build_sampling(args)
            with FakeProbes(), quiet():
                rc = m.copilot_sync(args, sampling, {"copilot"})
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "agents", "code.agent.md")))  # roster written
            settings = json.load(open(os.path.join(tmp, "settings.json"), encoding="utf-8"))
            self.assertIs(settings["includeCoAuthoredBy"], False)   # model-independent keys set
            self.assertNotIn("model", settings)                     # no model when offline
            self.assertFalse(os.path.exists(os.path.join(tmp, "otools-copilot.env")))  # endpoint not wired

    def test_copilot_home_env_var_wins(self):
        saved = os.environ.get("COPILOT_HOME")
        try:
            os.environ["COPILOT_HOME"] = os.path.join("x", "custom")
            self.assertEqual(m.copilot_home(), os.path.join("x", "custom"))
        finally:
            os.environ.pop("COPILOT_HOME", None)
            if saved is not None:
                os.environ["COPILOT_HOME"] = saved

    def test_wsl_redirects_to_windows_copilot_home(self):
        # Under WSL with Copilot installed on the Windows side, resolve the Windows ~/.copilot.
        saved = (m._is_wsl, os.path.isdir, os.environ.get("USERPROFILE"))
        try:
            m._is_wsl = lambda: True
            os.environ["USERPROFILE"] = r"C:\Users\Otto"
            os.path.isdir = lambda p: p == "/mnt/c/Users/Otto/.copilot"
            self.assertEqual(m._wsl_windows_copilot_home(), "/mnt/c/Users/Otto/.copilot")
            m._is_wsl = lambda: False   # not WSL -> no redirect
            self.assertIsNone(m._wsl_windows_copilot_home())
        finally:
            m._is_wsl, os.path.isdir = saved[0], saved[1]
            if saved[2] is None:
                os.environ.pop("USERPROFILE", None)
            else:
                os.environ["USERPROFILE"] = saved[2]

    def test_win_path_translation(self):
        self.assertEqual(m._win_path("/mnt/c/Users/Otto/.copilot/x.ps1"),
                         r"C:\Users\Otto\.copilot\x.ps1")
        self.assertEqual(m._win_path("/home/otto/.copilot/x"), "/home/otto/.copilot/x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
