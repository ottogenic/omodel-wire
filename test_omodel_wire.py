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
  * recipes           -- DEFAULT_RECIPES == model_recipes.json, shape, name-matching
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
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "omodel-wire.py")
RECIPES_PATH = os.path.join(HERE, "model_recipes.json")

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
        _hosts=["192.168.50.101"], _ports=[8000], timeout=1.0,
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
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


class FakeProbes:
    """Context manager that swaps the network probes for canned answers so the whole
    sync runs offline. `model` is served at 192.168.50.101:8000."""

    def __init__(self, model="Qwen3.6-27B-NVFP4", max_len=262144, vision=False):
        self.model, self.max_len, self.vision = model, max_len, vision
        self._saved = {}

    def __enter__(self):
        for name in ("probe", "probe_reasoning", "probe_vision"):
            self._saved[name] = getattr(m, name)

        def probe(host, port, timeout):
            if (host, port) == ("192.168.50.101", 8000):
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
# Recipes
# --------------------------------------------------------------------------- #
class TestRecipes(unittest.TestCase):
    def test_default_recipes_match_json_file(self):
        with open(RECIPES_PATH, encoding="utf-8") as f:
            disk = json.load(f)
        self.assertEqual(m.DEFAULT_RECIPES, disk,
                         "DEFAULT_RECIPES (in script) must equal model_recipes.json")

    def test_recipe_shape(self):
        for r in m.DEFAULT_RECIPES["recipes"]:
            self.assertIn("match", r)
            self.assertTrue(r["match"], "empty match patterns")
            self.assertIn("presets", r)
            for role, preset in r["presets"].items():
                self.assertIn(role, ROLE_NAMES, f"unknown role {role}")
                self.assertIn("thinking", preset)
                self.assertIn("sampling", preset)

    def test_match_recipe(self):
        recs = m.DEFAULT_RECIPES
        self.assertIn("27B", m.match_recipe("Qwen3.6-27B-NVFP4", recs)["source"])
        self.assertIn("35B", m.match_recipe("Qwen3.6-35B-A3B-NVFP4", recs)["source"])
        self.assertIn("Nemotron",
                      m.match_recipe("NVIDIA-Nemotron-3-Super-120B-A12B", recs)["source"])
        self.assertIn("GLM", m.match_recipe("GLM-4.7-Flash", recs)["source"])
        self.assertIsNone(m.match_recipe("totally-unknown-model", recs))

    def test_match_is_case_insensitive(self):
        self.assertIsNotNone(m.match_recipe("qwen3.6-27b-nvfp4", m.DEFAULT_RECIPES))


# --------------------------------------------------------------------------- #
# Agent building from recipes
# --------------------------------------------------------------------------- #
class TestAgentBuilding(unittest.TestCase):
    REF = "dgx-n1-8000/Qwen3.6-27B-NVFP4"

    def _recipe(self, name_frag):
        return m.match_recipe(name_frag, m.DEFAULT_RECIPES)

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
        for k in m.TEAM_TARGETS:
            self.assertEqual(agents[k].get("prompt"), "{file:./prompts/otools-worker.md}")

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
        self.assertEqual(team["permission"]["edit"], "deny")
        self.assertEqual(team["permission"]["bash"], "deny")
        task = team["permission"]["task"]
        self.assertEqual(task["*"], "deny")
        for t in m.TEAM_TARGETS:
            self.assertEqual(task[t], "allow", f"team can't delegate to {t}")

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
        _, sampling = m.oc_build_agents("dgx-n1-8000/Qwen3.6-27B-NVFP4", dict(FULL_CAPS))
        js = m.oc_agent_sampling_plugin_js(sampling)
        self.assertIn("chat.params", js)
        self.assertIn(m.PROVIDER_PREFIX, js)
        # the embedded AGENT_SAMPLING table must be valid JSON we can round-trip
        table = js.split("const AGENT_SAMPLING =", 1)[1].split("\n\nfunction", 1)[0].strip()
        self.assertEqual(json.loads(table)["code"]["topK"], 20)


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
            providers, refs, caps = m.oc_build_providers(
                ["192.168.50.101"], [8000], 1.0, self.SD,
                profiles=False, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertTrue(entry["tool_call"])
        self.assertEqual(refs, ["dgx-n1-8000/Qwen3.6-27B-NVFP4"])

    def test_tool_call_can_be_disabled(self):
        with FakeProbes():
            providers, _, _ = m.oc_build_providers(
                ["192.168.50.101"], [8000], 1.0, self.SD,
                profiles=False, tool_call=False, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertNotIn("tool_call", entry)

    def test_server_default_sets_temperature_false(self):
        with FakeProbes():
            providers, _, _ = m.oc_build_providers(
                ["192.168.50.101"], [8000], 1.0, self.SD,
                profiles=False, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertIs(entry["temperature"], False)

    def test_profiles_keeps_temperature_true(self):
        with FakeProbes():
            providers, _, caps = m.oc_build_providers(
                ["192.168.50.101"], [8000], 1.0, self.SD,
                profiles=True, recipes=FIXTURE_CONFIGS, verbose=False)
        entry = providers["dgx-n1-8000"]["models"]["Qwen3.6-27B-NVFP4"]
        self.assertIs(entry["temperature"], True)
        self.assertTrue(entry["reasoning"])
        self.assertIn("variants", entry)
        self.assertIn("dgx-n1-8000/Qwen3.6-27B-NVFP4", caps)

    def test_vision_writes_attachment_and_modalities(self):
        with FakeProbes():
            providers, _, _ = m.oc_build_providers(
                ["192.168.50.101"], [8000], 1.0, self.SD,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
