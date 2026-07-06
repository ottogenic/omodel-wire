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
        web_search="none", enable_exa_shell=False, write_shell_env=False,
        mcp_name="websearch", mcp_command=None, mcp_url=None,
        mcp_env=None, mcp_header=None,
        keep_builtins=False, default_agent="code",
        team_model=None, team_reasoning=None, team_task_budget=None,
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
    sync runs offline. `model` is served at 192.0.2.101:8000."""

    def __init__(self, model="Qwen3.6-27B-NVFP4", max_len=262144, vision=False):
        self.model, self.max_len, self.vision = model, max_len, vision
        self._saved = {}

    def __enter__(self):
        for name in ("probe", "probe_reasoning", "probe_vision"):
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
                  "agent-plan", "agent-code", "agent-instruct"):
            self.assertIn(k, agents, f"missing agent {k}")
            self.assertEqual(agents[k]["model"], self.REF)

        # visible agents carry no worker prompt; hidden workers do.
        for k in ("research", "code", "agent"):
            self.assertNotIn("prompt", agents[k], f"visible {k} should be prompt-free")
        # agent-review has its own review prompt; other workers use worker prompt
        for k in ("agent-plan", "agent-code", "agent-instruct"):
            self.assertEqual(agents[k].get("prompt"), "{file:./prompts/otools-worker.md}")
        self.assertEqual(agents["agent-review"].get("prompt"), "{file:./prompts/otools-review.md}")

        # permission tiers landed on the right agents
        self.assertEqual(agents["research"]["permission"]["edit"], "deny")
        self.assertEqual(agents["code"]["permission"]["edit"], "ask")
        self.assertEqual(agents["agent"]["permission"]["edit"], "allow")

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
        # visible research (readonly) and the workers never get a task budget.
        self.assertNotIn("task_budget", agents["research"])
        self.assertNotIn("task_budget", agents["agent-plan"])

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
        args = make_args(tmp, **over)
        sampling = m.build_sampling(args)
        with FakeProbes(), quiet():
            rc = m.oc_sync(args, sampling, {"opencode"})
        self.assertEqual(rc, 0)
        with open(args.config, encoding="utf-8") as f:
            return json.load(f)

    def test_writes_full_roster_and_disables_builtins(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._sync(tmp)
            ag = cfg["agent"]
            for k in ("research", "code", "agent", "team",
                      "agent-plan", "agent-code", "agent-instruct"):
                self.assertIn(k, ag)
            # native build/plan disabled, default moved off build
            self.assertEqual(ag["build"], {"disable": True})
            self.assertEqual(ag["plan"], {"disable": True})
            self.assertEqual(cfg["default_agent"], "code")
            # provider + model entry wired for tools + reasoning
            entry = cfg["provider"]["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
            self.assertTrue(entry["tool_call"])
            self.assertTrue(entry["reasoning"])

    def test_writes_team_orchestration_skill_globally(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp)
            skill = os.path.join(tmp, "skills", "team-orchestration", "SKILL.md")
            self.assertTrue(os.path.exists(skill), "team-orchestration SKILL.md not written")
            body = open(skill, encoding="utf-8").read()
            self.assertIn("name: team-orchestration", body)   # frontmatter -> discoverable
            self.assertIn("PARALLEL", body)                    # the batching guidance

    def test_team_orchestration_scoped_to_team_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ag = self._sync(tmp)["agent"]
            # every non-team agent denies the team skill...
            for k in ("research", "code", "agent", "agent-plan", "agent-code",
                      "agent-instruct", "agent-review"):
                self.assertEqual(ag[k]["permission"]["skill"], {"team-orchestration": "deny"})
            # ...and team does NOT (so it alone can load it)
            self.assertNotIn("skill", ag["team"]["permission"])

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
                      "agent-plan", "agent-code", "agent-instruct"):
                self.assertIn(k, ag, f"{k} missing -> roster not rebuilt for a non-reasoning fleet")
            # local agents point at the LIVE model + an existing provider (not a stale ref)
            for k in ("code", "agent", "agent-plan", "agent-code", "agent-instruct"):
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

    def test_anthropic_team_model_thinking_and_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._sync(tmp, team_model="anthropic/claude-opus-4-8",
                             team_reasoning="high")
            team = cfg["agent"]["team"]
            self.assertEqual(team["model"], "anthropic/claude-opus-4-8")
            self.assertEqual(team["options"]["thinking"],
                             {"type": "enabled", "budgetTokens": 32000})
            self.assertEqual(team["temperature"], 1.0)
            # local vLLM knobs stripped for the cloud model
            self.assertNotIn("chat_template_kwargs", team.get("options", {}))
            self.assertNotIn("top_p", team)

    def test_frontier_team_model_preserved_across_resync(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sync(tmp, team_model="anthropic/claude-opus-4-8", team_reasoning="high")
            # re-sync WITHOUT the flags: the anthropic choice must survive.
            cfg = self._sync(tmp)
            team = cfg["agent"]["team"]
            self.assertEqual(team["model"], "anthropic/claude-opus-4-8")
            self.assertEqual(team["temperature"], 1.0)

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
                m.save_settings({"team_model": "anthropic/claude-opus-4-8"})
                self.assertEqual(m.load_settings()["team_model"], "anthropic/claude-opus-4-8")
                a = types.SimpleNamespace(_settings=m.load_settings())
                self.assertEqual(m._setting(a, "team_model"), "anthropic/claude-opus-4-8")
                self.assertEqual(m._setting(a, "default_agent"), "code")  # built-in fallback
                self.assertIsNone(m._setting(a, "configs_dir"))
            finally:
                m.WIRE_SETTINGS_FILE = old

    def test_agents_list_and_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp, team_task_budget=4)
            out = _capture(m.cmd_agents, _cli_args(cfg))
            for name in ("research", "code", "agent", "team"):
                self.assertIn(name, out)
            self.assertNotIn("agent-plan", out)  # workers are not primaries
            detail = _capture(m.cmd_agents, _cli_args(cfg, name="team"))
            self.assertIn("agent: team", detail)
            self.assertIn("work-budget", detail)
            self.assertIn("4", detail)

    def test_subagents_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._synced(tmp)
            out = _capture(m.cmd_subagents, _cli_args(cfg))
            for w in ("agent-plan", "agent-code", "agent-instruct"):
                self.assertIn(w, out)
            self.assertNotIn("research", out)  # primaries excluded

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


def _sync_fixture(tmp, **over):
    args = make_args(tmp, **over)
    sampling = m.build_sampling(args)
    with FakeProbes(), quiet():
        m.oc_sync(args, sampling, {"opencode"})
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
            for w in ("agent-plan", "agent-code", "agent-instruct"):
                self.assertEqual(ag[w]["model"], "anthropic/x")

    def test_set_work_budget_updates_team_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _sync_fixture(tmp)
            with quiet():
                m.cmd_agents(_cli_args(cfg, name=None, set_model=None, set_work_budget=7))
            self.assertEqual(self._load(cfg)["agent"]["team"]["task_budget"], 7)
            with open(os.path.join(tmp, "prompts", "otools-team.md"), encoding="utf-8") as f:
                self.assertIn("7", f.read())

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
