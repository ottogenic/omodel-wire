#!/usr/bin/env python3
"""
omodel-wire.py

Two jobs on one laptop:

  1) DETECT which agentic-dev tools you have installed (OpenCode today;
     pi.dev / Claude Code / others are stubbed for the future).

  2) For the tools that support it, SYNC the OpenAI-compatible model endpoints
     running on your DGX Spark nodes into that tool's config.

Right now only OpenCode is wired up for syncing. The detection layer and the
"configurator" registry are deliberately pluggable so adding pi.dev / Claude
Code later is just one more entry.

Stdlib only -- no pip install. See README.md for usage and AGENTS.md for the full
architecture, rules, and how to extend.

Quick start:
  omodel-wire.py --install-aliases     # add the `omw` shell alias (re-open shell after)
  omw                                 # detect tools + sync (default sampling)
  omw --profiles                      # + build the agent roster for reasoning models
  omw --profiles --web-search exa --write-shell-env
  omw --dry-run                       # preview opencode.json + plugin, write nothing

--profiles builds, per reasoning model, an agent roster from omodel-manager's
declared per-model configs (configs/*.toml) -- capabilities + per-mode sampling are
DECLARED, not probed (fast). Roster: visible `research` / `code` / `agent` + the
`loom` pipeline lead (whose conductor dispatches the hidden `agent-research` /
`agent-code` / `agent-test` / `agent-architect` / `agent-review` workers) -- plus
Ctrl+T thinking variants and an agent-aware chat.params plugin that pins sampling.
"""

__version__ = "0.2.0"

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib   # stdlib (Python 3.11+); reads the generic per-model configs
import urllib.error
import urllib.parse
import urllib.request

# Proxy helper module (utils/omw_proxy.py) loaded BY PATH so it works whether omw is
# run as a script or imported by the tests via importlib (the file name has no bearing).
import importlib.util as _ilu
_PROXY_MOD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "omw_proxy.py")
try:
    _pspec = _ilu.spec_from_file_location("omw_proxy", _PROXY_MOD_PATH)
    proxy = _ilu.module_from_spec(_pspec)
    _pspec.loader.exec_module(proxy)
except (OSError, ImportError, AttributeError):
    proxy = None

# Loom conductor module (utils/omw_loom.py) -- the v2 deterministic pipeline
# orchestrator. Loaded by path like the proxy; `omw loom` and the generated
# OpenCode `loom` tool both dispatch into it. None if missing (loom disabled).
_LOOM_MOD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "omw_loom.py")
try:
    _lspec = _ilu.spec_from_file_location("omw_loom", _LOOM_MOD_PATH)
    loom_mod = _ilu.module_from_spec(_lspec)
    _lspec.loader.exec_module(loom_mod)
except (OSError, ImportError, AttributeError):
    loom_mod = None

# ----------------------------------------------------------------------------
# Endpoint discovery defaults -- placeholder examples only. Set your real hosts
# via `omm install` / `~/.config/otools/hosts`; these are the no-store fallback.
# (192.0.2.0/24 is the RFC 5737 documentation range -- replace with your own.)
# ----------------------------------------------------------------------------
DEFAULT_HOSTS = ["192.0.2.101", "192.0.2.102"]
# DEFAULT_PORTS = [8000, 8001, 8002, 8888, 30000, 11434]  # 11434 = ollama
DEFAULT_PORTS = [8000, 8001, 8002]
HOST_LABELS = {                       # friendly short labels for provider keys
    "192.0.2.101": "n1",
    "192.0.2.102": "n2",
    "192.0.2.103": "n3",
}
# Shared with omodel-manager: `omm install`/`ps` manage this file; omw reads it so
# both tools see the same fleet. Absent/empty -> fall back to DEFAULT_HOSTS.
HOSTS_FILE = os.path.expanduser("~/.config/otools/hosts")


def load_shared_hosts():
    """Bare host IPs from the shared omodel-manager store (HOSTS_FILE). Each line is
    `alias<TAB>user@host`, a bare `user@host`, or a bare host; we take the host part
    (after any `user@`) for HTTP probing. Returns [] if the file is absent/empty."""
    out = []
    try:
        with open(HOSTS_FILE) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                target = ln.split(None, 1)[-1].strip()   # drop the alias column if present
                host = target.split("@")[-1]             # drop user@ -> bare host/IP
                if host and host not in out:
                    out.append(host)
    except OSError:
        pass
    return out
PROBE_TIMEOUT = 2.0                    # seconds per /v1/models probe
VISION_TIMEOUT = 30.0                  # seconds for the image probe (first call is slow)
VISION_MAXTOKENS = 2048                # a vision *reasoning* model can burn a lot of tokens
                                        # thinking before it emits the answer; too low ->
                                        # truncated mid-think -> empty content -> false negative.
REASONING_TIMEOUT = 45.0               # seconds per reasoning-capability probe call (a generous
                                        # budget can take longer to generate on a busy server)
REASONING_MAXTOKENS = 8192             # max_tokens is only a CEILING -- a model stops on its own
                                        # (finish_reason=stop) long before this on a trivial prompt,
                                        # so a generous cap costs nothing but removes the truncation
                                        # trap: too small and a reasoning model is cut off mid-think,
                                        # and with a qwen3-style --reasoning-parser vLLM then drops
                                        # the partial thinking into `content` with reasoning=null
                                        # (vLLM issue #35221) -- see probe_reasoning's length tell.

# Qwen's own per-mode sampling recommendations (qwen.readthedocs.io quickstart):
#   thinking      -> temperature 0.6, top_p 0.95, top_k 20
#   non-thinking  -> temperature 0.7, top_p 0.80, presence_penalty 1.5 (anti-repeat)
QWEN_THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95}
QWEN_NOTHINK_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}


def match_recipe(model_id, recipes):
    mid = (model_id or "").lower()
    for r in recipes.get("recipes", []):
        pats = r.get("match")
        pats = [pats] if isinstance(pats, str) else (pats or [])
        if any(str(p).lower() in mid for p in pats):
            return r
    return None


def _configs_dir(path=None):
    """Directory of GENERIC per-model configs. omodel-manager OWNS these; this tool
    is an adapter that consumes them. Resolution order:
      --configs PATH  >  $OMODEL_CONFIGS  >  sibling ../omodel-manager/configs."""
    if path:
        return os.path.expanduser(path)
    env = os.environ.get("OMODEL_CONFIGS")
    if env:
        return os.path.expanduser(env)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "omodel-manager", "configs"))


def load_configs(configs_dir=None):
    """Load the generic per-model configs from <manager>/configs/*.toml (owned by
    omodel-manager). Returns {"recipes": [...]} compatible with match_recipe().
    Non-.toml files are ignored; files that fail to parse are warned and skipped."""
    d = _configs_dir(configs_dir)
    recipes = []
    if not os.path.isdir(d):
        print(f"  note: model configs dir not found: {d}\n"
              f"        point it with --configs PATH or $OMODEL_CONFIGS "
              f"(omodel-manager's configs/).")
        return {"recipes": recipes}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".toml"):
            continue
        p = os.path.join(d, fn)
        try:
            with open(p, "rb") as f:
                recipe = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError) as e:
            print(f"  warning: {fn} did not parse: {e}; skipping")
            continue
        recipe.setdefault("_file", fn)
        recipes.append(recipe)
    return {"recipes": recipes}


def caps_from_capabilities(recipe):
    """Synthesize the probe-style caps dict from a recipe's DECLARED capabilities,
    so oc_build_recipe_agents / oc_build_providers work without live probing.
    Mirrors probe_reasoning's output keys."""
    cap = (recipe or {}).get("capabilities", {}) or {}
    tc = cap.get("thinking_control", recipe.get("thinking_control", "enable_thinking"))
    return {
        "reasoning": bool(cap.get("reasoning", False)),
        # both enable_thinking and reasoning_effort setups can force thinking OFF
        # (reasoning_effort turns it on; enable_thinking:false turns it off).
        "can_disable": tc in ("enable_thinking", "reasoning_effort"),
        "effort_ok": tc == "reasoning_effort",
        "graded": bool(cap.get("graded", False)),
        "reason": f"declared (thinking_control={tc})",
    }


def discover_opencode_runtime_models(opencode_path=None, timeout=5.0):
    """Discover OpenCode runtime models by running `opencode models` and parsing output.
    
    Returns:
        (models_list, note) where models_list is a list of provider/model refs
        (e.g., ["openai/gpt-5.5", "anthropic/claude-opus-4-8"]) and note is a
        human-readable status string.
    
    The helper is safe: it handles missing binary, nonzero exit, timeout, and
    unexpected output without crashing; returns ([], note) on any failure."""
    if opencode_path is None:
        # Try to find opencode on PATH
        opencode_path = shutil.which("opencode")
        if not opencode_path:
            return [], "no opencode binary found on PATH"
    
    # Run `opencode models` with a short timeout
    try:
        result = subprocess.run(
            [opencode_path, "models"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as e:
        return [], f"opencode models failed: {e}"
    
    # Check for nonzero exit
    if result.returncode != 0:
        return [], f"opencode models exited with code {result.returncode}"
    
    # Parse output lines that look like provider/model refs
    # Expected format: lines with "provider/model" pattern (e.g., "openai/gpt-5.5")
    models = []
    seen = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Look for provider/model pattern (must have a known provider prefix)
        # Pattern: provider/model where provider is in REMOTE_PROVIDERS or starts with dgx-
        parts = line.split("/")
        if len(parts) == 2:
            provider, model = parts[0].strip(), parts[1].strip()
            # Only accept if provider is a known remote provider or dgx provider
            if provider in REMOTE_PROVIDERS or provider.startswith("dgx-"):
                ref = f"{provider}/{model}"
                if ref not in seen:
                    seen.add(ref)
                    models.append(ref)
    
    if not models:
        return [], "no provider/model refs found in opencode models output"
    
    return models, f"found {len(models)} runtime model(s)"


def oc_build_opencode_providers(cfg):
    """Extract models from OpenCode's existing config providers.
    
    Returns dict of {provider/model_ref: {}} for all models in managed and
    unmanaged providers in the config. This is used as a fallback/back-compat
    source when runtime discovery fails or to augment the runtime pool."""
    providers = cfg.get("provider", {}) or {}
    models = {}
    for prov_key, prov_cfg in providers.items():
        if not isinstance(prov_cfg, dict):
            continue
        model_entries = prov_cfg.get("models", {}) or {}
        for mid in model_entries.keys():
            models[f"{prov_key}/{mid}"] = {}
    return models


DEFAULT_CONTEXT = 200000              # used if endpoint doesn't report max_model_len
DEFAULT_OUTPUT = 65536               # OpenCode requires limit.output
API_KEY = "sglang"                    # dummy; vLLM/SGLang ignore it
PROVIDER_PREFIX = "dgx-"             # all managed providers start with this
# Cloud providers OpenCode routes to directly. Lets a default_models.json preference tell a
# real remote ref (openai/gpt-5.5) apart from a DGX served-model-id that merely contains a
# slash (e.g. unsloth/qwen3-coder-next-fp8 -- an HF org/model id, NOT a provider ref).
REMOTE_PROVIDERS = {"anthropic", "openai", "google", "openrouter", "azure",
                    "mistral", "xai", "groq", "deepseek", "cohere", "bedrock", "vertex",
                    "github-copilot"}


def _authed_remote_providers():
    """Cloud providers the user has authenticated in OpenCode (auth.json).

    Native-auth providers never appear in the discovered/available model pool
    (it only holds probed DGX endpoints + config-file providers), so without
    this, a default_models remote pref like openai/gpt-5.6-sol is skipped on
    EVERY sync and the agent silently falls through to a local model -- seen
    live: agent-review repeatedly downgraded from its frontier reviewer model
    to the local qwen. A remote pref for an authed provider resolves directly."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"),
                                                           ".local", "share")
    try:
        with open(os.path.join(base, "opencode", "auth.json"), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return frozenset()
        return frozenset(k for k in data if isinstance(k, str)) & REMOTE_PROVIDERS
    except (OSError, ValueError):
        return frozenset()
# vLLM repetition_detection (RepetitionDetectionParams): terminate a generation once a
# token N-gram keeps repeating, so a degenerate loop can't burn the whole output budget.
# Three knobs (vLLM's own defaults in parens):
#   min_pattern_size (0->1)   smallest repeating unit to flag, in TOKENS
#   max_pattern_size (0=off)  largest repeating unit to flag; min_pattern<=max_pattern
#   min_count        (>=2)    consecutive repeats before terminating
# Tuned LENIENT on purpose -- only cut genuine long loops, never legitimate short repeats.
# Note the trip point for a repeated unit of pattern_len tokens is pattern_len*min_count
# tokens (smallest checked pattern_len is min_pattern_size), so:
#   * min_pattern_size=3  -> single/short-token runs only trip at min_pattern_size*min_count
#     = 30 identical tokens in a row, so "300000", indentation, "====" / "----" rules,
#     hex/base64 stay well under it. A long rule trips only if the tokenizer emits 30+
#     IDENTICAL tokens; BPE packs "----" runs into multi-char tokens, so in practice that's
#     a ~hundreds-of-chars run. (The earlier default left this at 1, which cut "300000" at
#     "30000" after just five 0s.)
#   * max_pattern_size=20 -> vLLM's own auto-enable ceiling; catches up to ~sentence-length
#     loops (the expensive kind).
#   * min_count=10        -> the unit must repeat 10x before we stop. Well past any
#     legitimate repetition (identical array rows / boilerplate / ASCII tables trip 6 but
#     not 10), yet a genuine runaway repeats hundreds of times so it's still cut within
#     ~30-40 tokens; the extra headroom costs <=80 tokens worst case on a 32k budget.
DEFAULT_REPETITION_DETECTION = {"min_pattern_size": 3, "max_pattern_size": 20, "min_count": 10}
LEGACY_KEYS = {"dgx"}                 # also clean up the old single "dgx" provider
# Agents emitted per recipe. We explicitly override OpenCode's built-in plan/build
# (full defs, our sampling + permissions). Two tiers:
#   VISIBLE (Tab, mode primary, provider-default prompt) -- direct-use agents.
#   HIDDEN  (mode subagent, minimal bootstrap prompt) -- the loom conductor's
#           workers; role guidance is injected per-dispatch from LOOM_SKILLS_DIR.
# Permission profiles (risk tiers). websearch/webfetch allowed on all.
#   readonly: no edits/bash/delegation (research only)
#   ask:      full access but PROMPTS for confirmation on edits/bash
#   full:     edits/bash run without prompting (autonomous)
PERM = {
    "readonly": {"edit": "deny",  "bash": "deny",  "task": "deny",
                 "websearch": "allow", "webfetch": "allow"},
    "ask":      {"edit": "ask",   "bash": "ask",   "websearch": "allow", "webfetch": "allow",
                 "task": {"*": "deny", "agent-review": "allow"}},
    "full":     {"edit": "allow", "bash": "allow", "websearch": "allow", "webfetch": "allow",
                 "task": {"*": "deny", "agent-review": "allow"}},
    # Test and review agents may execute checks and inspect git, but never edit the
    # implementation. Only agent-review receives the reviewer token at runtime.
    "test":     {"edit": "deny", "bash": "allow", "websearch": "allow", "webfetch": "allow",
                 "task": "deny"},
    "review":   {"edit": "deny", "bash": "allow", "websearch": "allow", "webfetch": "allow",
                 "task": "deny"},
}
# Each spec: (key, preset role, mode, is_worker, perm profile, color, description).
# color = FIXED hex by risk: green (read-only) -> yellow-green (ask) -> orange
# (autonomous). Tweak the hexes here.
# Visible names avoid the reserved built-ins `build`/`plan` (which OpenCode won't
# let you override -- they'd show as "native" and ignore our settings). We instead
# disable the natives and use `research` (planner) + `code` (coder).
AGENT_SPECS = [
    ("research", "reason", "primary", False, "readonly", "#22c55e", "research & reasoning, read-only + web"),
    ("code",     "code",   "primary", False, "ask",      "#a3e635", "interactive coder -- asks before edits/bash"),
    ("agent",    "agent",  "primary", False, "full",     "#f97316", "autonomous worker, full access (no prompts)"),
    ("agent-research", "reason",   "subagent", True, "readonly", "#22c55e", "[worker] research & data-gathering, read-only + web (fetch/summarize)"),
    ("agent-code",     "code",     "subagent", True, "full",     "#f97316", "[worker] coding / implementation / debugging, full access"),
    ("agent-test",     "code",     "subagent", True, "test",     "#eab308", "[worker] runs broad lint/test/build/script verification and reports exact results; never edits"),
    ("agent-architect","reason",   "subagent", True, "readonly", "#8b5cf6", "[worker] read-only planner/verifier + escalation target for hard problems; returns a plan, research list, and acceptance criteria"),
    ("agent-review",   "reason",   "subagent", True, "review", "#3b82f6", "[worker] verifies completed implementations and reviews Pull Requests; classifies findings and never edits"),
]
# Per-worker `steps` cap (max agentic iterations before OpenCode forces a text-only
# summary -- see opencode.ai/docs/agents#max-steps). Tightest on the well-defined jobs
# (test), most headroom on open-ended work (code/review). This bounds runaway
# TOOL-ACTION loops; combined with the DONE/CONTINUE/BLOCKED exit contract in the worker
# packets, a worker that hits the wall reports back so the loom conductor can continue
# it (same session) or escalate to agent-architect. Not set on visible primaries.
WORKER_STEPS = {
    "agent-research":  10,  # fetch + read a few sources + summarize (read-only)
    "agent-test":      12,  # run harness + read result files + report (runaway-prone)
    "agent-architect": 15,  # read-and-reason across the code, but read-only
    "agent-code":      20,  # implement/debug: edit -> run -> inspect -> fix cycles
    "agent-review":    20,  # checkout + run checks + read the whole diff
}
# Built-in agents we disable (can't be overridden; replaced by research/code).
BUILTIN_DISABLE = ["build", "plan"]
LOOM_COLOR = "#ec4899"   # pink -- pipeline lead (deterministic conductor)
# Agent keys this tool may write under --profiles (current + legacy). Used to prune
# stale ones on re-sync -- incl. old plan/build OVERRIDES and RETIRED names (team,
# agent-instruct) -- so re-syncing converges. Won't touch the user's own agents.
MANAGED_AGENTS = {"research", "code", "agent", "team",
                  "agent-research", "agent-code", "agent-test", "agent-instruct",
                  "agent-architect", "agent-review",
                  "loom",
                  "agent-plan",
                  "plan", "build", "instruct", "architect", "reason", "chat", "fast",
                  "general", "webdev", "agentic"}

# Default models config file (user preferences for agent/subagent models)
DEFAULT_MODELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_models.json")
DEFAULT_MODELS_TEMPLATE = {
    "agents": {
        # Qwen-first: the local DGX 35B-A3B is ~free at the margin, so it leads every
        # workhorse agent; paid github-copilot models are fallbacks only.
        # loom's LLM half only does intake + relay, so a cheap local model leads; the
        # pipeline workers keep their own per-role rankings below.
        "loom": ["qwen3.6-35b-a3b-nvfp4-unsloth",
                 "unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "qwen3-coder-next-nvfp4",
                 "github-copilot/gpt-5.4", "github-copilot/gemini-2.5-pro",
                 "github-copilot/claude-sonnet-4.6"],
        "research": ["qwen3.6-35b-a3b-nvfp4-unsloth",
                     "unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "qwen3-coder-next-nvfp4",
                     "github-copilot/gemini-3.5-flash", "github-copilot/claude-haiku-4.5"],
        "code": ["qwen3.6-35b-a3b-nvfp4-unsloth",
                 "unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "qwen3-coder-next-nvfp4",
                 "github-copilot/kimi-k2.7-code", "github-copilot/gpt-5.3-codex"],
        "agent": ["qwen3.6-35b-a3b-nvfp4-unsloth",
                  "unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "qwen3-coder-next-nvfp4",
                  "github-copilot/kimi-k2.7-code", "github-copilot/gpt-5.3-codex"],
    },
    "subagents": {
        "agent-research": ["qwen3.6-35b-a3b-nvfp4-unsloth",
                           "unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "qwen3-coder-next-nvfp4",
                           "github-copilot/gemini-3.5-flash", "github-copilot/claude-haiku-4.5"],
        "agent-code": ["qwen3.6-35b-a3b-nvfp4-unsloth",
                       "unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "qwen3-coder-next-nvfp4",
                       "github-copilot/kimi-k2.7-code", "github-copilot/gpt-5.3-codex"],
        "agent-test": ["qwen3.6-35b-a3b-nvfp4-unsloth",
                       "unsloth/qwen3-coder-next-fp8", "qwen3-coder-next-fp8", "qwen3-coder-next-nvfp4",
                       "github-copilot/claude-haiku-4.5", "github-copilot/gpt-5.4-mini"],
        # Expensive-but-low-volume: architect (read-only planner/verifier) and review
        # (merge gate) are the only agents that lead with a top-tier paid model.
        "agent-architect": ["github-copilot/claude-opus-4.8", "github-copilot/gpt-5.6-sol",
                            "github-copilot/gemini-3.1-pro-preview",
                            "qwen3.6-35b-a3b-nvfp4-unsloth"],
        "agent-review": ["github-copilot/claude-opus-4.8", "github-copilot/claude-sonnet-5",
                         "github-copilot/gpt-5.6-sol",
                         "qwen3.6-35b-a3b-nvfp4-unsloth"],
    },
}


def load_default_models():
    """Load default_models.json, creating it with the built-in template if missing."""
    return _load_default_models(create=True)


def _load_default_models(create):
    """Load default_models.json preferences. Returns {agents: {...}, subagents: {...}}.
    Auto-creates with template if missing unless create is false. Never overwrites an
    existing user config."""
    if not os.path.exists(DEFAULT_MODELS_FILE):
        return _create_default_models_template(write=create)
    try:
        with open(DEFAULT_MODELS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _create_default_models_template(write=create)
        if "agents" not in data:
            data["agents"] = {}
        if "subagents" not in data:
            data["subagents"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return _create_default_models_template(write=create)


def _create_default_models_template(write=True):
    """Create default_models.json with the built-in agent preference ordering."""
    template = {
        section: {agent: list(preferences) for agent, preferences in agents.items()}
        for section, agents in DEFAULT_MODELS_TEMPLATE.items()
    }
    if not write:
        return template
    try:
        with open(DEFAULT_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
    except OSError:
        pass
    return template


def _resolve_model_ref_from_prefs(pref, available_models, remote_ok=frozenset()):
    """Resolve a model preference to its full provider-qualified ref if available.

    Args:
        pref: model preference string - can be:
            - Full ref: "openai/gpt-5.5" (must match exactly in available_models)
            - Served ID with slash: "unsloth/qwen3-coder-next-fp8" (DGX model, must match model ID)
            - Bare model ID: "qwen3-coder-next-fp8" (must match model ID)
        available_models: dict mapping full refs to {} (e.g., {"dgx-1-8000/qwen3.6-27b": {}})
        remote_ok: providers whose remote refs resolve WITHOUT appearing in
            available_models (native-auth clouds -- see _authed_remote_providers)

    Returns:
        Full provider-qualified ref if resolvable, None otherwise.
    """
    if not pref or not (available_models or remote_ok):
        return None
    available_models = available_models or {}

    # Check if pref is already a full ref (contains / with a known provider)
    if "/" in pref:
        provider_part = pref.split("/", 1)[0]
        # If it's a known remote provider or dgx provider, check exact match in available_models
        if provider_part in REMOTE_PROVIDERS or provider_part.startswith("dgx-"):
            if pref in available_models:
                return pref
            if provider_part in remote_ok:
                return pref  # authed native cloud: OpenCode routes it directly
            return None
    
    # pref is a bare model ID or served ID without a known provider - match against model IDs
    # For served IDs like "unsloth/qwen3-coder-next-fp8", we need to match the full served ID
    # For bare IDs like "qwen3-coder-next-fp8", we match just the model part
    
    # First, check if pref contains a slash - if so, it's a served ID that should match exactly
    if "/" in pref:
        # This is a served ID like "unsloth/qwen3-coder-next-fp8"
        # Look for an available model that ends with this served ID
        for model_ref in available_models.keys():
            if model_ref.endswith("/" + pref):
                return model_ref
        return None
    
    # Bare model ID - match against model ID part of refs
    pref_model_id = pref
    
    # Find a matching ref in available_models
    for model_ref in available_models.keys():
        # Extract model ID from the ref (everything after the first /)
        ref_model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
        if ref_model_id == pref_model_id:
            return model_ref
    
    return None


def get_preferred_model(available_models, preferences):
    """Pick the best model from preferences that's in available_models.
    
    Args:
        available_models: list of model IDs available from the endpoint
        preferences: list of preferred model IDs (in order of preference)
    
    Returns:
        The first preference that's in available_models, or the first available model,
        or None if no models available.
    """
    if not available_models:
        return None
    if not preferences:
        return available_models[0]
    
    available_set = set(available_models)
    for pref in preferences:
        if pref in available_set:
            return pref
    return available_models[0]


def _apply_default_models(agents, default_models, reasoning_caps, available_models=None,
                          notes=None, remote_ok=frozenset()):
    """Apply user's default model preferences to agents based on available models.
    
    Args:
        agents: dict of agents from oc_build_agents/oc_build_recipe_agents
        default_models: loaded default_models.json data
        reasoning_caps: dict mapping model_ref -> caps (for reasoning models)
        available_models: dict mapping model_ref -> {} (all models, including non-reasoning)
        notes: optional list to append warning/notes messages
    
    Returns:
        agents dict with models updated according to preferences
    """
    if not agents:
        return agents
    
    # Use available_models if provided, otherwise fall back to reasoning_caps
    if available_models:
        model_refs = list(available_models.keys())
    elif reasoning_caps:
        model_refs = list(reasoning_caps.keys())
    else:
        return agents
    
    if not model_refs:
        return agents
    
    # Build list of available model IDs and refs for fallback
    available_model_ids = []
    for model_ref in model_refs:
        # Extract just the model ID part (after /)
        model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
        available_model_ids.append(model_id)
    
    if not available_model_ids:
        return agents
    
    # Build a set of available refs for quick lookup
    available_refs_set = set(model_refs)
    
    # Apply preferences for each agent
    agents_copy = dict(agents)
    for agent_name, agent_cfg in agents_copy.items():
        if not isinstance(agent_cfg, dict):
            continue
        
        # Determine which preferences to use (agents or subagents)
        if agent_cfg.get("mode") == "subagent":
            prefs = (default_models.get("subagents") or {}).get(agent_name, [])
        else:
            prefs = (default_models.get("agents") or {}).get(agent_name, [])
        
        if prefs:
            # Resolve this agent's model by walking its preferences IN ORDER and taking the FIRST
            # that resolves against the available_models pool. List order is authoritative.
            # A served ID that contains a slash (unsloth/qwen3-coder-next-fp8) is a DGX model
            # — it matches ONLY the local-pool check. Remote refs (openai/..., anthropic/...)
            # must appear EXACTLY in available_models to be accepted.
            preferred = None
            resolved_ref = None
            for pref in prefs:
                resolved = _resolve_model_ref_from_prefs(pref, available_models, remote_ok)
                if resolved is not None:
                    preferred = pref  # Use the original preference string for display
                    resolved_ref = resolved
                    break
                # Pref is not available - emit a visible warning
                if notes is not None:
                    notes.append(f"Pref skipped (not available): {pref}")
            
            if preferred is None:
                # Nothing matched - check if current model is still available
                current_ref = agent_cfg.get("model", "")
                if current_ref and current_ref in available_refs_set:
                    # Current model is still available - keep it
                    if notes is not None:
                        notes.append(f"Using current model: {current_ref} (no preference resolved)")
                elif available_model_ids:
                    # Fall back to first available model
                    preferred = available_model_ids[0]
                    resolved_ref = _resolve_model_ref_from_prefs(preferred, available_models,
                                                                 remote_ok)
                    if notes is not None:
                        notes.append(f"Fallback to first available: {preferred}")
                # If no available models, leave unchanged (no update to agent_cfg)
            
            if resolved_ref:
                agent_cfg["model"] = resolved_ref
            elif preferred and "/" in preferred:
                # Remote model reference (e.g., openai/gpt-5.5) - use directly
                agent_cfg["model"] = preferred
            elif preferred:
                # Fallback: keep the provider from current model
                current_ref = agent_cfg.get("model", "")
                if "/" in current_ref:
                    provider = current_ref.split("/", 1)[0]
                    agent_cfg["model"] = f"{provider}/{preferred}"
                else:
                    agent_cfg["model"] = preferred
    
    return agents_copy


# Agent -> basename for the {file:./prompts/otools-<name>.md} bootstrap prompts
# written next to opencode.json; edit them there to tune behavior (sync resets).
ROLE_SKILL_NAMES = {
    "loom": "agent-loom",
    "agent-research": "agent-research",
    "agent-code": "agent-code",
    "agent-test": "agent-test",
    "agent-architect": "agent-architect",
    "agent-review": "agent-review",
}


def _role_bootstrap_prompt(agent_name, skill_name):
    """Minimal role system prompt (replaces OpenCode's default entirely).

    The opener MUST stay action-first and MUST come before any output/summary
    framing: leading with summary wording makes Qwen3-family models (vLLM
    --tool-call-parser qwen3_coder) NARRATE tool calls as text instead of emitting
    native calls -> the parser drops them and the worker fabricates (verified on
    n1: 1/8 leading with summary framing vs 15/15 action-first).
    The closing paragraph counters OpenCode #18423 (orchestrator receives the
    subagent's last text part even when empty). Role instructions and the Return
    Contract are NOT in this prompt: the loom conductor injects both into every
    dispatch packet deterministically (from LOOM_SKILLS_DIR), so they live in
    exactly one place and never depend on model-invoked skill loading.
    """
    if agent_name == "loom":
        # The loom agent cannot load skills (permission.skill = deny), so this
        # prompt IS its entire operating method. Keep it short and imperative:
        # a router with exactly one legal move per user message.
        return """You are `loom`, a ROUTER. You never answer, investigate, plan, or fix anything
yourself. You have no workspace. Every user message gets EXACTLY ONE loom tool
call, then a report relay. Nothing else.

Pick the action with one rule:

1. The message requests ANY change -- feature, bug fix, update, add, remove,
   rewrite, content edit: call the loom tool with action "run" and
   packet = the user's message VERBATIM. Do not rewrite, summarize, shorten,
   or enrich it. risk = "high" if the user says high risk, else "medium".
2. The message is ONLY a question (answering it changes nothing): call the
   loom tool with action "ask", packet = the question verbatim.
3. The message asks about a job's progress or outcome: action "status".

When the tool returns, relay its report word-for-word and STOP. No second tool
call, no commentary. If the report says paused: tell the user why and wait;
when they reply with guidance, call action "resume" with the job id and their
reply as note.

YOU ARE NEVER ALLOWED TO FIX PROBLEMS. If the tool errors or the result looks
wrong, show the user exactly what happened and stop.
"""
    return f"""You are `{agent_name}`, a delegated worker in a deterministic pipeline. Complete the
task by calling the provided tools. Act, inspect each tool result, then continue until
the task is done. Your dispatch message carries your role instructions and the required
format of your final reply (the Return Contract); both are mandatory.
When the work is finished, end with a plain-text message containing your concrete
results -- your caller receives ONLY this final message. Never stop on a bare tool call
or an empty message.
"""


ROLE_PROMPT_FILES = {key: f"otools-{skill}.md" for key, skill in ROLE_SKILL_NAMES.items()}


WORKER_STATUS_CONTRACT = """
## Return Contract

Put your FULL deliverable (the plan, findings, or description of changes) in the
message body FIRST. Then end every reply with exactly this block, replacing each
<placeholder> with your actual content -- NEVER repeat the placeholder text itself:

    RESULT: <one-paragraph summary of what you did or found, with files as path:line>
    EVIDENCE: <each command or check you ran and what it RETURNED, or your sources>
    STATUS: <one of DONE | CONTINUE | NEEDS_RESEARCH | BLOCKED>

- `STATUS: DONE` -- assigned work is complete.
- `STATUS: CONTINUE` -- making progress with a concrete next step.
- `STATUS: NEEDS_RESEARCH` -- blocked on specific facts; include a focused `RESEARCH REQUEST:` list of questions.
- `STATUS: BLOCKED` -- no sound path forward; EVIDENCE shows attempts and the exact error.

Do not spin: if you would return `CONTINUE` a second time with nothing new in
EVIDENCE, return `BLOCKED` instead. Escalating early is correct behavior, not failure.
"""


FINDING_CLASSIFICATION = """
## Finding Classification

Classify every finding as exactly one of:
- `blocker` -- violates caller-provided acceptance criteria or breaks the requested feature.
- `regression` -- behavior newly broken by the current change.
- `pre-existing` -- unrelated issue present before this task.
- `future work` -- valid improvement not required for this request.
- `out of scope` -- outside the caller's stated goal.

Only `blocker` and `regression` require immediate implementation. Report the other categories
separately. Never expand acceptance criteria unless the caller explicitly requests a broader audit.
"""


ROLE_PROMPTS = {key: _role_bootstrap_prompt(key, skill)
                for key, skill in ROLE_SKILL_NAMES.items()}


AGENT_LOOM_SKILL = """---
name: agent-loom
description: Global operating method for `loom`: gather goal, criteria, scope, and risk; run the pipeline with one loom tool call; relay the report verbatim; gate PR creation on explicit user approval.
---

# loom (pipeline lead)

You run feature work through a deterministic pipeline (agent-architect -> agent-code ->
agent-test -> agent-review) with ONE call to the `loom` tool. The tool performs all
delegation, looping, and escalation; you do intake and reporting only.

## Feature Or Question? (decide this first)

DEFAULT TO A FEATURE. Use `action: "ask"` ONLY when the entire message is a question
and requests no change to the code or repo. If the message asks you to add, implement,
build, create, change, expand, wire up, or refactor ANYTHING, it is a feature -- use
`action: "run"`, even when the same message also says "compare to the PRD", "check the
docs", or "look for any existing X". Comparing, checking, and researching are steps the
pipeline performs DURING the build; they are never a reason to route to `ask`. When in
doubt between the two, choose `run`. 

`ask` answers THE USER's questions -- it is never for you. Never call `ask` to gather
context, inspect files, or prepare a packet for a feature request: you do not need repo
knowledge to write the packet. Pass the user's requirements as given; the pipeline's
planning step reads the code itself and requests research when it needs facts. In a feature
conversation never call `ask` at all -- not to prepare, and not to verify afterward:
the pipeline tests and reviews its own work, and your report already contains that
evidence. Relay the report and stop. `ask` is only for conversations that ARE questions.

For a pure question (nothing in the request would change code or files):
1. Call the `loom` tool with `action: "ask"` and the question as `packet`.
2. Relay the cited answer.
Never answer such questions from your own knowledge -- you have no web access and no
workspace tools, but `action: "ask"` routes to `agent-research`, which has both.

## Procedure

1. Restate the goal, explicit acceptance criteria, and scope boundary.
2. Ask one concise clarification only when ambiguity materially changes the result.
3. Classify risk:
   - `simple` -- localized, clear, low-risk change (the pipeline skips planning).
   - `medium` -- a design choice, multiple interacting files, or an unfamiliar API.
   - `high` -- persistence/migration, concurrency, or security work.
   When uncertain, use `medium`.
4. Call the `loom` tool once with:
   - `action`: `"run"`
   - `packet`: the goal, acceptance criteria, and scope boundary as plain text.
   - `risk`: your classification.
5. The tool streams progress and returns the final report. Relay the report to the
   user: implementation summary, exact check results, and non-blocking findings
   listed separately. Quote check results; do not paraphrase them.
6. If the report says `paused`, tell the user exactly why and show the resume
   command from the report. To continue after their answer, call the `loom` tool
   with `action: "resume"`, the `job` id, and their guidance as `note`.

## Troubleshooting

When the user asks what happened, why a run failed, or where a job stands:
1. Call the `loom` tool with `action: "status"` (add the `job` id when you know it;
   without it you get the recent job list -- then call again with the right id).
2. Relay the report and event log it returns. Do not speculate about failures;
   the log is the truth.

## Rules

- One feature = one `loom` call. Never split a feature across calls, and never
  start a second call for a feature that is already running.
- If the user asks for MULTIPLE independent features that touch different files,
  make one `loom` call per feature in the same turn; each runs as its own job with
  its own progress card. If the features overlap in any file, run them one at a
  time instead.
- Never create a PR without explicit user approval. On approval, call the `loom`
  tool with `action: "pr"` and the `job` id from the report.
- When the report arrives, relaying it ENDS YOUR TURN. Do not call the `loom` tool
  again -- no `pr`, no `status`, no re-running the job -- unless the user asks in a
  new message. If the report looks incomplete or the working tree looks wrong, say so
  in your reply; never try to fix it by re-running the pipeline.
- You cannot inspect the workspace. If the user asks a code question, suggest the
  `research` agent; your only job is running the pipeline.
"""


# Per-role instruction bodies (plain markdown, no frontmatter). The loom conductor
# injects these into every dispatch packet from LOOM_SKILLS_DIR -- deterministic
# delivery, unlike the retired model-invoked role skills. The Return Contract is
# delivered separately (see WORKER_STATUS_CONTRACT), never restated here.
ROLE_INSTRUCTIONS = {
    "agent-code": """# agent-code

Implement exactly the delegated goal -- no more, no less.

## Before Editing

1. Read the project instructions (AGENTS.md or equivalent) if present.
2. Read the relevant files, neighboring code, and existing tests.
3. Never assume a library, framework, or convention exists -- confirm it in the code.

## Rules

- Make the smallest complete change. No unrelated refactors.
- Do not add backwards-compatibility shims unless the task requires them.
- Preserve unrelated changes already in the worktree. Never revert user work.
- Never run destructive git commands (reset --hard, checkout --, force push, clean).
- Use the read/search/edit tools, not shell file operations.
- Run independent reads and checks in parallel.
- Match the existing code style. Keep comments rare and useful.
- Never expose, log, or commit secrets.
- Do not commit or push unless the delegated task explicitly says to. Never open, approve, or
  merge a PR; those steps are dispatched separately by the pipeline.
- Do not guess URLs, APIs, versions, or other external facts -- return
  `STATUS: NEEDS_RESEARCH` instead.

## Verify Your Change

Run focused checks that give fast feedback: a syntax check, the targeted tests, or a narrow
smoke command. Do NOT run full suites, long scripts, or broad lint/build passes --
broad verification runs later in the pipeline -- unless the delegated task explicitly
asks you to. State any risk you could not verify.
""",
    "agent-research": """# agent-research

Answer the delegated questions only. You are read-only: never edit files or run side-effecting
commands. Search local code first when relevant; use authoritative/current documentation for
external claims. Never invent URLs, APIs, versions, file contents, or certainty.

- Prefer Glob/Grep/Read for local discovery and WebFetch/search for external sources.
- Parallelize independent searches.
- Separate verified facts, reasonable inference, and unresolved uncertainty.
- Return concise findings, absolute or workspace-relative `path:line` references, and source URLs.
- If evidence is unavailable or conflicting, state exactly what remains unknown.
""",
    "agent-test": """# agent-test

Run the broad verification delegated to you after implementation. Read project instructions,
review the changed files and the focused check results supplied in your packet, then run the relevant full suites,
lint/typecheck/build commands, or scripted workflows. You may inspect files and execute checks, but
never edit implementation or tests.

- Never delete, skip, weaken, or rewrite a test/assertion to make it pass.
- Prefer one comprehensive pass over repeated narrow reruns; tight edit/test loops belong to the implementation step.
- Label a failure pre-existing only with evidence: it also fails without this change, or the
  task listed it as known. Otherwise report it as unclassified; never guess.
- Report each command, exit result, failing test name, and the shortest useful output excerpt.
- Do not commit, push, or open a PR unless the dispatched task explicitly delegates it.
- When PR creation is delegated to you: create the branch, commit exactly the change's
  files, push, and open the PR with your verification evidence in the body. Use the
  default (bot) token -- never `GH_TOKEN_REVIEWER`, or the PR cannot later be approved.
- Never approve or merge a PR; that happens in a later review step.
""",
    "agent-architect": """# agent-architect

You are read-only. Inspect project instructions, relevant code, tests, and supplied evidence;
never edit files or run side-effecting commands. You never review completed work --
that happens later in the pipeline. You do exactly three jobs:

## 1. Plan

Produce an implementation plan for the delegated goal. Return:
1. relevant current-state findings;
2. a numbered implementation plan;
3. explicit, checkable acceptance criteria;
4. scope exclusions and verification commands.

## 2. Request Research

For unfamiliar medium/high-risk work, do not guess: return `STATUS: NEEDS_RESEARCH` with the
focused questions you need answered before you can plan.

## 3. Diagnose A Blocker

When you are sent a `BLOCKED` implementation report, diagnose the smallest correction
that unblocks the original goal. Do not redesign unrelated areas or expand the plan. Use only
caller-provided criteria and project bars supplied in the task.

Report observations outside the delegated goal under the Finding Classification taxonomy
(delivered with your Return Contract); only `blocker` and `regression` block.
""",
    "agent-review": """# agent-review

Review the completed implementation against the dispatched packet: original goal, acceptance
criteria, scope, plan/research when applicable, changed files, the implementation's
focused checks, and the supplied test evidence. If essential context is missing, return `BLOCKED` and list it; do not
invent acceptance criteria.
The caller's criteria, taxonomy, and scope are authoritative. Never rename categories, substitute a
different review framework, or promote an unrelated observation into a blocking finding.

- Read project instructions and `REVIEW.md`/CONTRIBUTING/CI when relevant.
- Inspect changed code and callers. Use the supplied test evidence as the primary command evidence; run spot
  checks only for missing, suspicious, or PR/base-diff-specific coverage.
- Preserve unrelated work; use a clean checkout/worktree and never reset or revert user changes.
- Check correctness, regressions, invariants, scope creep, stale-branch reversions, secrets, and
  required tests/changelog entries.
- Never edit or fix code. Give each finding `path:line`, classification, impact, and concrete pass
  condition. If clean, say `No blocking findings.`
- On re-review, check the specified issue and regression surface in the same session. Do not
  restart a broad audit or add acceptance criteria.

## PR Review

You never open PRs; creation happens in an earlier pipeline step when the operator approves one.
For a pull request, inspect the PR claim and the full diff against the base branch. Then:
1. Approve or merge ONLY when all three hold: the user or an applicable project
   policy explicitly authorized it; all checks are acceptable; and no
   `blocker` or `regression` remains.
2. Use `GH_TOKEN_REVIEWER` for approve and merge actions.
3. Never push directly to the base branch.
""",
}


# The five pipeline roles, in dispatch-precedence order. Single source of truth for
# the loom instruction/contract files and the role_models map handed to the conductor.
LOOM_ROLES = ("agent-architect", "agent-code", "agent-test", "agent-review",
              "agent-research")

# loom-managed instruction files (regenerated on every sync; python constants are the
# source of truth). Lives under the same ~/.config root as this tool's own settings
# (WIRE_SETTINGS_FILE) rather than under opencode/ -- these files feed the loom
# conductor, not OpenCode.
LOOM_SKILLS_DIR = os.path.expanduser("~/.config/otools/loom/skills")


def _role_contract_text(role):
    """Contract packet for one role. Review and architect also carry the finding
    taxonomy -- mirroring the retired role skills, which appended it for those two."""
    if role in ("agent-review", "agent-architect"):
        return WORKER_STATUS_CONTRACT + FINDING_CLASSIFICATION
    return WORKER_STATUS_CONTRACT


def write_loom_role_files():
    """Write <LOOM_SKILLS_DIR>/agent-<role>-instructions-default.md + -contract.md for
    every pipeline role. Returns the paths written. Repo-local overlays
    (.loom/skills/agent-<role>-instructions-local.md) are hand-authored, never touched."""
    os.makedirs(LOOM_SKILLS_DIR, exist_ok=True)
    written = []
    for role in LOOM_ROLES:
        for suffix, text in ((f"{role}-instructions-default.md", ROLE_INSTRUCTIONS[role]),
                             (f"{role}-contract.md", _role_contract_text(role))):
            path = os.path.join(LOOM_SKILLS_DIR, suffix)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            written.append(path)
    return written


# Global OpenCode skills still written by sync: loom's own operating method plus the
# user-kicked session-review pass. Role guidance no longer ships as skills.
GLOBAL_SKILLS = {
    "agent-loom": AGENT_LOOM_SKILL,
}


LEGACY_ROLE_ARTIFACTS = (
    os.path.join("skills", "team-orchestration", "SKILL.md"),
    os.path.join("skills", "pr-review", "SKILL.md"),
    os.path.join("prompts", "otools-team.md"),
    os.path.join("prompts", "otools-worker.md"),
    os.path.join("prompts", "otools-test.md"),
    os.path.join("prompts", "otools-architect.md"),
    os.path.join("prompts", "otools-review.md"),
    # Retired with the team orchestrator / model-invoked role skills:
    os.path.join("prompts", "otools-agent-team.md"),
    os.path.join("prompts", "otools-agent-instruct.md"),
)

# Whole skill DIRECTORIES retired by the loom-managed instruction rework: role
# guidance now ships in LOOM_SKILLS_DIR, team/instruct/runbook-review are gone.
# agent-loom (and session-review) stay.
LEGACY_SKILL_DIRS = ("agent-code", "agent-test", "agent-architect", "agent-review",
                     "agent-research", "agent-instruct", "agent-team",
                     "agent-runbook-review")


def _role_skill_path(cfg_path, skill_name):
    return os.path.join(os.path.dirname(cfg_path), "skills", skill_name, "SKILL.md")


def _role_prompt_path(cfg_path, agent_name):
    return os.path.join(os.path.dirname(cfg_path), "prompts", ROLE_PROMPT_FILES[agent_name])


def oc_cleanup_stale_role_artifacts(config_path):
    """Delete deployed artifacts of the retired layout: the per-role/team/instruct/
    runbook skill directories and the retired prompt files. Returns removed paths."""
    root = os.path.dirname(config_path)
    removed = []
    for name in LEGACY_SKILL_DIRS:
        d = os.path.join(root, "skills", name)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    for relpath in LEGACY_ROLE_ARTIFACTS:
        stale = os.path.join(root, relpath)
        if not os.path.exists(stale):
            continue
        try:
            os.remove(stale)
            removed.append(stale)
            try:
                os.rmdir(os.path.dirname(stale))
            except OSError:
                pass
        except OSError:
            pass
    return removed


def oc_refresh_role_artifacts(config_path):
    """Rewrite the model-independent GENERATED artifacts from current code: global
    skills (agent-loom, session-review), the loom-managed instruction/contract files,
    role prompts (for roster agents present in the existing config), and the loom
    tool plugin (when the loom agent exists). Also removes retired-layout artifacts.

    Called on the sync paths that build no roster (no live models). Rationale,
    learned the hard way: `git pull` updates these artifacts' SOURCE but the
    deployed copies only change on a successful sync -- so with the model server
    down, a freshly pulled conductor ran under a stale plugin and old skills.
    Skills/prompts/plugin carry no model data; refreshing them is always safe.
    Returns the list of paths written."""
    cfg = oc_load_config(config_path) if os.path.exists(config_path) else {}
    agent_cfg = cfg.get("agent") or {}
    written = []

    def _write(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(path)

    for skill_name, text in GLOBAL_SKILLS.items():
        _write(_role_skill_path(config_path, skill_name), text)
    written.extend(write_loom_role_files())
    for agent_name in ROLE_SKILL_NAMES:
        if agent_name not in agent_cfg:
            continue
        _write(_role_prompt_path(config_path, agent_name), ROLE_PROMPTS[agent_name])
    if "loom" in agent_cfg and loom_mod is not None:
        _write(os.path.join(os.path.dirname(config_path), "plugins", "otools-loom.js"),
               oc_loom_plugin_js())
    oc_cleanup_stale_role_artifacts(config_path)
    return written


# `session-review` skill -- a user-kicked maintenance pass ("run a session review").
# Written GLOBALLY by sync so it is available in every repo. It mines the loom ledger
# (and optionally the OpenCode session store) for failures, timings, and instruction
# gaps, then proposes repo-local instruction updates -- REPORT-FIRST, approval-gated.
SESSION_REVIEW_SKILL = r"""---
name: session-review
description: Audit THIS session's loom run -- mine loom.db and opencode.db for the current job's failures, timing, and instruction gaps; propose updates to the repo's .loom/skills/*-local.md instruction files. Reviews other/multiple jobs only when the user names them.
---

# Session review

An audit pass over THIS session's loom pipeline run. You mine the deterministic ledger
for what actually happened, diagnose why workers failed or stalled, and turn each
diagnosis into a concrete repo-local instruction update. REPORT-FIRST: propose every
change and write only after approval.

## 0. Scope: the current run only

Default scope is the loom job launched from THIS conversation -- not "recent jobs" and
never the whole ledger. Resolve the job id in this order:

1. The loom tool result earlier in this conversation states it ("job <id>"). Use that.
2. Else match the invoking session: the conductor stamps the launching session id on
   the job row -- `SELECT id FROM jobs WHERE parent_session = '<this session id>'
   ORDER BY id DESC LIMIT 1`.
3. Else take the newest job for this checkout -- `SELECT id, goal FROM jobs WHERE
   dir = '<absolute repo dir>' ORDER BY id DESC LIMIT 1` -- and SAY which job you
   picked and why before proceeding.

Audit multiple jobs or another job ONLY when the user names them or explicitly asks for
a cross-run audit. If the pick is ambiguous (e.g. parallel runs in the same directory),
list the candidates with timestamps and goals and ask; do not audit them all.

## 1. Read the loom ledger

The conductor records every job in a SQLite ledger. Open it READ-ONLY so a live run is
never disturbed:

    DB="file:$HOME/.local/share/otools/loom.db?mode=ro"

Tables (current schema): `jobs(id, created, updated, dir, parent_session, goal, risk,
phase, status, plan, worker_model, report)`; `tasks(job_id, role, purpose, session_id,
attempts, model, last_status, spin)`; `findings(job_id, n, text, status)`;
`events(job_id, ts, kind, detail)`; `notes(job_id, ts, text)`.

For YOUR job id, pull:
- Symptom events: `malformed`, `testfail`, `findings`, `fix`, `escalate`, `blocked`,
  `antispin`, `timeout`, `stale`, `paused`, `error` -- each is a bad reply format, a
  failed test cycle, a review round, a hand-off up the ladder, or a stall.
- Timeline events: `phase`, `status`, `dispatch`, `resume`, `session`. Diff consecutive
  timestamps for per-phase durations (plan vs code vs test vs review vs fix rounds).
- Review findings from the `findings` table: the reviewer's blockers, numbered per fix
  round (`n = round*100 + k`), `status` pending or fixed. The fix loop batches EVERY
  open blocker back to the coder each round, re-tests, then re-reviews the COMPLETE
  change; rounds repeat until the review is clean or `max_fix_rounds` pauses the job.
  (A job that shows `done` with rows still `pending` predates this loop -- note it as
  historical, don't diagnose it as a fresh failure.)

If `sqlite3` isn't on PATH, Python's stdlib `sqlite3` module reads the same file.

## 2. Mine the worker transcripts -- this job's sessions only

Each task row carries the worker's OpenCode session id; the workers were created as
children of the same parent session. For any suspicious task, open
`~/.local/share/opencode/opencode.db` (read-only, same `?mode=ro` trick) and read the
`message` / `part` tables for THAT session id to see what the worker actually said and
did -- tool calls, errors, and its final reply.

Confine transcript reads to the session ids on THIS job's task rows (plus the loom
lead's own session). Other sessions in opencode.db belong to other runs; reading them
wastes context and pollutes the diagnosis.

## 3. Diagnose

Look for patterns, not one-offs:
- repeated malformed replies (contract not followed) -- which role, which field;
- spins: `CONTINUE` loops with nothing new, or `antispin` events;
- fix-round churn: blockers that took multiple rounds because a fix was partial or a
  test was hollow -- quote the reviewer's re-raised finding;
- escalations and timeouts -- and whether the phase budget or the task was at fault;
- instruction gaps: the worker did X wrong because nothing in its instructions said
  otherwise. These are the highest-value findings.

## 4. Propose instruction updates (approval-gated)

Turn each diagnosis into a specific line of instruction, quote the exact text, name the
file it belongs in, and write it ONLY after approval.

### Where each kind of instruction belongs

Route by AUDIENCE, not by topic. Ask: "which agents must know this?"

| The instruction is... | Goes in | Notes |
|---|---|---|
| Needed by EVERY agent (repo layout, what is/isn't our code, language/platform rules that constrain all work, ToS or safety constraints) | `<repo>/AGENTS.md` | Loaded into every session on every turn -- the most expensive file in the system. Add only what every role needs. |
| How to WRITE code here (language gotchas that bite implementers, file/module placement, required focused checks) | `.loom/skills/agent-code-instructions-local.md` | |
| How to VERIFY here (test commands, harness flags, pass/fail bars, what the harness can/cannot prove) | `.loom/skills/agent-test-instructions-local.md` | |
| The REVIEW bar (what blocks a change, finding classification, PR approve/merge policy and tokens) | `.loom/skills/agent-review-instructions-local.md` | |
| PLANNING scope (what counts as in/out of scope, repo-specific design constraints, standing exceptions to scope) | `.loom/skills/agent-architect-instructions-local.md` | |
| Where to LOOK THINGS UP (trusted doc sources, dead/blocked sites, local reference checkouts) | `.loom/skills/agent-research-instructions-local.md` | |
| Human-only setup (build steps, one-time installs, credentials, data downloads) | `README.md` / `docs/` | NOT an agent instruction file. Agents cannot perform these and pay context to read them. |

The ONLY files you may write are `<repo>/AGENTS.md` and
`<repo>/.loom/skills/agent-<role>-instructions-local.md`. Everything else in the table is
either generated (see rule 5) or human-owned documentation.

### Rules that keep these files honest

1. **One canonical home per fact.** If a rule already exists in `AGENTS.md`, a role file
   must POINT at it, not restate it. Duplication drifts and doubles the context cost.
2. **Demote anything role-specific out of `AGENTS.md`.** A rule only one role can act on
   (a test harness flag, a PR token, a doc source) is charged to every other role for
   nothing. Moving it out is a valid, high-value finding on its own.
3. **Promote only on repeated evidence.** Move a rule INTO `AGENTS.md` only when two or
   more roles demonstrably needed it in the ledger -- not on a hunch.
4. **Write the consequence, not the principle.** "Declare helpers before the function that
   calls them; a later `local function` resolves to a nil global and crashes at runtime"
   beats "follow good Lua practice". Observed-failure wording is what workers actually obey.
5. **Never edit the generated files.** `-default.md` and `-contract.md` under
   `~/.config/otools/loom/skills/` are regenerated by `omw sync`; hand edits are silently
   lost. A finding that warrants a GLOBAL change (every repo benefits) is reported as a
   suggestion for the omodel-wire maintainer -- never edited in place.
6. **An `agent-<role>-instructions-override.md` replaces global + local for that role
   entirely.** Propose one only when the default role behavior is actively wrong for this
   repo; prefer adding to `-local.md`.

## 5. Report (your final message)

- **Findings** -- each failure pattern with its evidence (job id, event kind, excerpt).
- **Per-phase timings** -- where the pipeline spends its time; outliers flagged.
- **Changes made** -- every `-local.md` line added, with the diagnosis it addresses.
- **Follow-ups** -- global suggestions for the maintainer, and anything unresolved.
"""

GLOBAL_SKILLS["session-review"] = SESSION_REVIEW_SKILL


# The blue test image + the word we expect a real vision model to say back.
# A text-only server typically 200s and ignores the image -> answer won't say "blue".
BLUE_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
EXPECTED_COLOR = "blue"

# ----------------------------------------------------------------------------
# Agentic-dev tool registry (detection). Add tools here as you adopt them.
#   cli      : candidate executable names to look for on PATH
#   config   : where that tool keeps its config (for reference / future sync)
#   sync     : name of the configurator implemented below, or None (stub)
# NOTE: opencode/claude binaries are confirmed; "pi" is a best-effort guess for
#       pi.dev -- adjust once you know the real binary name.
# ----------------------------------------------------------------------------
TOOLS = [
    {
        "key": "opencode",
        "display": "OpenCode",
        "cli": ["opencode"],
        "config": "~/.config/opencode/opencode.json",
        "sync": "opencode",
    },
    {
        "key": "copilot",
        "display": "GitHub Copilot",
        "cli": ["copilot"],
        "config": "~/.copilot/  (agents/ + settings.json)",
        "sync": "copilot",
    },
    {
        "key": "claude-code",
        "display": "Claude Code",
        "cli": ["claude"],
        "config": "~/.claude/settings.json",
        "sync": None,            # future
    },
    {
        "key": "pi",
        "display": "pi.dev",
        "cli": ["pi"],           # TODO: confirm real binary name for pi.dev
        "config": "~/.config/pi/config.json",
        "sync": None,            # future
    },
    # A few other common agentic CLIs -- handy in the detection report.
    {"key": "aider", "display": "Aider", "cli": ["aider"], "config": "~/.aider.conf.yml", "sync": None},
    {"key": "gemini", "display": "Gemini CLI", "cli": ["gemini"], "config": "~/.gemini/settings.json", "sync": None},
    {"key": "crush", "display": "Crush", "cli": ["crush"], "config": "~/.config/crush/crush.json", "sync": None},
    {"key": "codex", "display": "Codex CLI", "cli": ["codex"], "config": "~/.codex/config.toml", "sync": None},
    {"key": "cursor-agent", "display": "Cursor Agent", "cli": ["cursor-agent"], "config": "~/.cursor/", "sync": None},
]


# ============================================================================
# Tool detection
# ============================================================================
def _version_of(path):
    """Best-effort `<tool> --version`. Returns short string or ''."""
    for flag in ("--version", "version", "-v"):
        try:
            out = subprocess.run([path, flag], capture_output=True, text=True, timeout=5)
            blob = (out.stdout or out.stderr or "").strip()
            if blob:
                return blob.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def detect_tools():
    """Return list of {tool..., installed, path, version} for every known tool."""
    results = []
    for tool in TOOLS:
        found_path = None
        for name in tool["cli"]:
            p = shutil.which(name)
            if p:
                found_path = p
                break
        entry = dict(tool)
        entry["installed"] = found_path is not None
        entry["path"] = found_path or ""
        entry["version"] = _version_of(found_path) if found_path else ""
        results.append(entry)
    return results


def print_detection(detected):
    print("Installed agentic-dev tools:")
    name_w = max(len(t["display"]) for t in detected)
    for t in detected:
        if t["installed"]:
            sync = "sync: yes" if t["sync"] else "sync: (planned)"
            ver = f"  {t['version']}" if t["version"] else ""
            print(f"  [x] {t['display']:<{name_w}}  {t['path']}{ver}   [{sync}]")
        else:
            print(f"  [ ] {t['display']:<{name_w}}  (not found on PATH)")
    print()


# ============================================================================
# DGX endpoint probing
# ============================================================================
def host_label(host):
    return HOST_LABELS.get(host, host.split(".")[-1])


def probe(host, port, timeout):
    """Return list of {id, max_model_len} for a live endpoint, or None if dead."""
    url = f"http://{host}:{port}/v1/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            json.JSONDecodeError, ValueError):
        return None
    models = []
    for m in data.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        models.append({"id": mid, "max_model_len": m.get("max_model_len")})
    return models or None


def looks_visual(model_id):
    """Cheap name pre-filter so we only image-probe likely vision models."""
    s = model_id.lower()
    tokens = (
        "-vl", "vl-", "/vl", "vl/", "vision", "visual", "multimodal", "omni",
        "llava", "pixtral", "internvl", "minicpm-v", "-v-", "qwen-vl", "qwen2-vl",
        "qwen2.5-vl", "qwen3-vl", "gemma-3", "molmo", "idefics", "phi-3-vision",
        "phi-3.5-vision", "phi-4-multimodal", "kimi-vl", "glm-4v",
    )
    return any(t in s for t in tokens)


def probe_vision(host, port, model_id, timeout):
    """Send the BLUE test image + 'what color?'.

    Returns (is_vision, answer, reason) where reason is a short human string
    that always explains the verdict (so failures aren't silently swallowed):
      * HTTP 200 AND answer mentions the expected color -> (True, answer, ...)
        (model genuinely decoded the image)
      * HTTP 200 but answer does NOT mention the color  -> (False, answer, ...)
        (server very likely ignored the image -> treat as text-only / unverified)
      * server error mentioning 'decode'                -> (True, None, ...)
        (it tried to decode the image -> vision pipeline is live)
      * server error mentioning image-not-supported     -> (False, None, ...)
      * any other failure                               -> (False, None, ...)
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model_id,
        "max_tokens": VISION_MAXTOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is this image? Answer in one word."},
            {"type": "image_url", "image_url": {"url": BLUE_PNG_DATA_URL}},
        ]}],
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
        content, reasoning, finish = "", "", ""
        try:
            j = json.loads(body)
            choice = (j.get("choices") or [{}])[0]
            msg = choice.get("message", {}) or {}
            content = (msg.get("content") or "").strip()
            # reasoning models put their chain-of-thought in a separate field
            reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
            finish = choice.get("finish_reason") or ""
        except Exception:
            pass
        # The visible answer (prefer content; fall back to reasoning).
        answer = content or reasoning
        # Trust it if EITHER content or reasoning mentions the image's color --
        # a reasoning model that "sees" the image says blue while thinking.
        if EXPECTED_COLOR in (content + " " + reasoning).lower():
            where = "content" if EXPECTED_COLOR in content.lower() else "reasoning"
            return True, answer, f'HTTP 200, model named the color in {where}'
        if answer:
            return False, answer, f'HTTP 200 but answer was "{answer[:120]}" (expected {EXPECTED_COLOR})'
        hint = f" (finish_reason={finish}; raise VISION_MAXTOKENS)" if finish == "length" else ""
        return False, answer, f"HTTP 200 with empty content (server accepted but said nothing){hint}"
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", "ignore")
        except Exception:
            raw = ""
        body = raw.lower()
        snippet = " ".join(raw.split())[:200]
        # explicit text-only signal (parenthesized to avoid and/or precedence bug)
        if ("not support image" in body) or ("image input" in body and "decode" not in body):
            return False, None, f"HTTP {e.code}: model reports no image support -- {snippet}"
        if "decode" in body:           # it tried to decode -> vision pipeline live
            return True, None, f"HTTP {e.code}: image decode error -> vision pipeline is live -- {snippet}"
        return False, None, f"HTTP {e.code}: ambiguous server error -- {snippet}"
    except (urllib.error.URLError, OSError) as e:
        return False, None, f"connection/timeout error: {e}"


# ============================================================================
# Reasoning / thinking capability probe
# ============================================================================
def _chat(host, port, model_id, extra_body, timeout, prompt):
    """POST a chat completion with extra top-level body fields.
    Returns (status_int_or_None, parsed_json_or_None, error_text)."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "max_tokens": REASONING_MAXTOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(extra_body)
    data = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200), json.loads(resp.read().decode("utf-8", "ignore")), ""
    except urllib.error.HTTPError as e:
        try:
            return e.code, None, e.read().decode("utf-8", "ignore")
        except Exception:
            return e.code, None, ""
    except (urllib.error.URLError, OSError) as e:
        return None, None, str(e)


# Inline chain-of-thought, for endpoints served WITHOUT a reasoning parser (the
# model emits the tags itself). A closed <think>...</think> block, or an unclosed
# <think>... run when the model was cut off before finishing its thought.
_THINK_BLOCK = re.compile(r"<think\s*>(.*?)</\s*think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think\s*>(.*)$", re.DOTALL | re.IGNORECASE)


def _reasoning_len(j):
    """Length of the model's chain-of-thought in a response (0 if none).

    Handles the server layouts we see in the wild:
      * reasoning parser configured  -> chain-of-thought in `message.reasoning`
        (or `reasoning_content`), final answer in `message.content`;
      * no reasoning parser          -> the model emits `<think>...</think>`
        (or an unclosed `<think>...`) inline in `message.content`.
    The remaining case -- a qwen3-style parser that runs out of tokens before the
    closing `</think>` and drops partial, *untagged* reasoning into `content` with
    `reasoning=null` (vLLM #35221) -- leaves nothing to measure here; probe_reasoning
    catches it via the `finish_reason=length` tell instead."""
    try:
        msg = (j.get("choices") or [{}])[0].get("message", {}) or {}
        structured = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        if structured:
            return len(structured)
        content = msg.get("content") or ""
        blocks = _THINK_BLOCK.findall(content)
        if blocks:
            return sum(len(b.strip()) for b in blocks)
        opened = _THINK_OPEN.search(content)
        if opened:
            return len(opened.group(1).strip())
    except Exception:
        pass
    return 0


def _finish_reason(j):
    """The response's finish_reason ('' if absent). 'length' means the model was
    cut off at max_tokens rather than stopping on its own."""
    try:
        return (j.get("choices") or [{}])[0].get("finish_reason") or ""
    except Exception:
        return ""


def probe_reasoning(host, port, model_id, timeout):
    """Determine, against the LIVE endpoint, what thinking knobs it honors.

    Returns caps dict:
      reasoning   : model emits chain-of-thought by default
      can_disable : chat_template_kwargs.enable_thinking=false actually suppresses it
      effort_ok   : reasoning_effort low/high are accepted (no 4xx)
      graded      : high produces materially more thinking than low (true depth dial)
      reason      : human summary
    """
    caps = {"reasoning": False, "can_disable": False, "effort_ok": False,
            "graded": False, "reason": ""}
    prompt = "Compute 23 * 17. Think step by step, then give the final number."

    # A) default request -> does it think at all?
    _, j, err = _chat(host, port, model_id, {}, timeout, prompt)
    if j is None:
        caps["reason"] = f"default request failed: {err[:160]}"
        return caps
    if _reasoning_len(j) == 0:
        # A model with a qwen3-style --reasoning-parser that runs out of tokens BEFORE
        # closing </think> returns its partial thinking in `content` with reasoning=null
        # (vLLM #35221), so _reasoning_len sees nothing. But a non-reasoning model
        # answers this trivial prompt in a few tokens and stops on its own, so a
        # `length` cutoff here is itself the tell that the model was still thinking.
        if _finish_reason(j) != "length":
            caps["reason"] = "no `reasoning` field by default -> treat as non-reasoning model"
            return caps
    caps["reasoning"] = True

    # B) can we turn thinking OFF via chat_template_kwargs?
    _, j_off, _ = _chat(host, port, model_id,
                        {"chat_template_kwargs": {"enable_thinking": False}}, timeout, prompt)
    if j_off is not None and _reasoning_len(j_off) == 0:
        caps["can_disable"] = True

    # C) does reasoning_effort work, and do levels differ?
    _, j_lo, _ = _chat(host, port, model_id, {"reasoning_effort": "low"}, timeout, prompt)
    _, j_hi, _ = _chat(host, port, model_id, {"reasoning_effort": "high"}, timeout, prompt)
    if j_lo is not None and j_hi is not None:
        caps["effort_ok"] = True
        lo, hi = _reasoning_len(j_lo), _reasoning_len(j_hi)
        caps["graded"] = hi > lo * 1.4 and (hi - lo) > 200

    caps["reason"] = (
        f"reasoning=yes; disable={'ok' if caps['can_disable'] else 'NO'}; "
        f"reasoning_effort={'accepted' if caps['effort_ok'] else 'rejected'}; "
        f"levels={'graded' if caps['graded'] else 'binary (low~high)'}")
    return caps


def oc_off_options(caps):
    """Body options that turn thinking OFF for this endpoint ({} if it can't)."""
    if caps["can_disable"]:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def oc_effort_options(caps, level):
    """Body options that turn thinking ON at `level` (low/medium/high).

    ALWAYS sets enable_thinking:true so a `think` variant reliably overrides an
    agent that set enable_thinking:false -- they must share the same key, since
    vLLM gives chat_template_kwargs.enable_thinking priority over reasoning_effort.
    reasoning_effort is added for graded depth when the endpoint supports it."""
    opts = {"chat_template_kwargs": {"enable_thinking": True}}
    if caps["effort_ok"]:
        opts["reasoning_effort"] = level
    return opts


def oc_build_variants(caps):
    """Ctrl+T-cyclable thinking-depth presets for one reasoning model."""
    variants = {"no-think": {"options": oc_off_options(caps)}}
    if caps["graded"]:
        for lvl in ("low", "medium", "high"):
            variants[lvl] = {"options": oc_effort_options(caps, lvl)}
    else:
        variants["think"] = {"options": oc_effort_options(caps, "high")}
    return variants


# Built-in fallback for a reasoning model with no curated recipe: the 4 roles
# using Qwen's recommended numbers (thinking ON for reason/code/agent -- matches
# Qwen's precise-coding guidance -- and a fast no-think instruct mode).
GENERIC_QWEN_RECIPE = {
    "thinking_control": "auto",
    "presets": {
        "reason": {
            "desc": "Research & Q&A (Qwen general thinking: temp 1.0)",
            "thinking": True,
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        },
        "code": {
            "desc": "Interactive coding (Qwen precise-coding: thinking ON, temp 0.6)",
            "thinking": True,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        },
        "agent": {
            "desc": "Unattended agent (coding sampling + preserved thinking)",
            "thinking": True,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            "options": {"chat_template_kwargs": {"preserve_thinking": True}},
        },
        "instruct": {
            "desc": "Fast, no thinking (Qwen instruct: temp 0.7)",
            "thinking": False,
            "sampling": {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                         "min_p": 0.0, "presence_penalty": 1.5},
        },
    },
}


def _preset_options(preset, control, caps):
    """Thinking-knob options dict for a recipe preset (shared by build + reconfigure).

    control='none' leaves the model at its own default; otherwise a thinking preset gets
    enable_thinking + reasoning_effort (per caps) and a non-thinking preset gets the
    off-options. Explicit per-preset raw kwargs (e.g. preserve_thinking for the agent role,
    or Nemotron's {enable_thinking, low_effort}) are merged on top."""
    if control == "none":
        o = {}
    elif preset.get("thinking"):
        o = dict(oc_effort_options(caps, "high"))
    else:
        o = dict(oc_off_options(caps))
    for k, v in (preset.get("options") or {}).items():
        if k == "chat_template_kwargs" and isinstance(o.get(k), dict) and isinstance(v, dict):
            o[k] = {**o[k], **v}
        else:
            o[k] = v
    return o


def _preset_vec(preset, repetition_detection):
    """Plugin sampling vector for a recipe preset (shared by build + reconfigure)."""
    s = preset.get("sampling", {})
    vec = {}
    if "temperature" in s: vec["temperature"] = s["temperature"]
    if "top_p" in s: vec["topP"] = s["top_p"]
    if "top_k" in s: vec["topK"] = s["top_k"]
    if preset.get("max_output"): vec["maxOutputTokens"] = preset["max_output"]
    body = {}
    if s.get("presence_penalty") is not None: body["presence_penalty"] = s["presence_penalty"]
    if s.get("min_p"): body["min_p"] = s["min_p"]
    if s.get("repetition_penalty") not in (None, 1.0): body["repetition_penalty"] = s["repetition_penalty"]
    if repetition_detection is not None:
        body["repetition_detection"] = repetition_detection
    if body: vec["options"] = body
    return vec


def oc_build_agents(model_ref, caps, repetition_detection=None):
    """Generic (no curated recipe) reasoning model -> the 4 standard roles via
    Qwen's recommended numbers. Returns (agents, agent_sampling) like the recipe
    path, so the agent-aware sampling plugin is written for these too."""
    return oc_build_recipe_agents(model_ref, GENERIC_QWEN_RECIPE, caps, repetition_detection)


def oc_build_recipe_agents(model_ref, recipe, caps, repetition_detection=None):
    """Turn a matched recipe's presets into OpenCode agents + an agent->sampling
    map for the agent-aware chat.params plugin.

    Returns (agents, agent_sampling):
      agents[name]         -> {description, model, options{thinking knob,...}}
      agent_sampling[name] -> {temperature, topP, topK, maxOutputTokens, options{...}}
                               (the plugin enforces these so top_k/min_p land reliably)
    """
    # thinking_control: "auto" (default) injects Qwen-style knobs the probe
    # confirmed (reasoning_effort / enable_thinking / preserve_thinking);
    # "none" leaves the model at its own default -- use for non-Qwen models
    # (e.g. Nemotron super_v3) whose templates don't take those kwargs.
    control = recipe.get("thinking_control", "auto")

    presets = recipe.get("presets", {})
    agents, agent_sampling = {}, {}
    for key, prole, mode, is_worker, perm, color, sdesc in AGENT_SPECS:
        preset = presets.get(prole)
        if not preset:
            continue
        s = preset.get("sampling", {})
        agent = {
            "description": f"{key}: {sdesc}",
            "mode": mode,
            "model": model_ref,
            "color": color,
            "options": _preset_options(preset, control, caps),
            "permission": dict(PERM[perm]),
        }
        # Delegation roles get tiny bootstrap prompts; their role instructions and
        # Return Contract arrive in every dispatch packet (injected by the loom
        # conductor from LOOM_SKILLS_DIR). Visible direct-use agents stay prompt-free
        # and retain OpenCode's provider default system prompts.
        if is_worker:
            agent["prompt"] = f"{{file:./prompts/{ROLE_PROMPT_FILES[key]}}}"
        # Workers never sub-delegate -- the loom conductor does all dispatching. Deny
        # task on every worker (the PERM `full`/`readonly` profiles allow agent-review,
        # which is correct for the VISIBLE code/agent primaries but not the workers).
        if is_worker:
            agent["permission"]["task"] = "deny"
        # Merge tripwire. The REAL control that only agent-review can merge is the
        # TOKEN split: non-reviewer agents don't hold $GH_TOKEN_REVIEWER, and GitHub
        # rejects an unauthorized/self merge. This bash rule is defense-in-depth ONLY --
        # it is trivially bypassable (e.g. `GH_TOKEN=x gh pr merge`, extra spaces, or
        # `cd d && gh pr merge` don't match the glob, since OpenCode matches the parsed
        # command string). So we set it to "ask" (a visible prompt), not "deny" (false
        # assurance), and cover the common merge spellings. See opencode.ai/docs/permissions.
        if key != "agent-review":
            merge_gate = {"*gh pr merge*": "ask", "*gh  pr  merge*": "ask"}
            base_bash = agent["permission"].get("bash")
            if base_bash == "allow":
                agent["permission"]["bash"] = {"*": "allow", **merge_gate}
            elif isinstance(base_bash, dict):
                base_bash.update(merge_gate)
        # Reliable sampling lives in the agent config too (correct even without the
        # plugin); the plugin additionally enforces top_k/min_p/penalties/maxOutput.
        if "temperature" in s: agent["temperature"] = s["temperature"]
        if "top_p" in s: agent["top_p"] = s["top_p"]
        if not preset.get("thinking") and not caps["can_disable"] and not recipe.get("soft_switch"):
            agent["description"] += "  [WARN: endpoint can't disable thinking]"
        # Visible code/agent get task_budget to delegate (only to agent-review)
        if key in ("code", "agent"):
            agent["task_budget"] = 1
        # Per-worker step cap: bound runaway tool-action loops. On the wall OpenCode
        # forces a text summary, which our DONE/CONTINUE/BLOCKED contract turns into a
        # continue-or-escalate signal for the loom conductor.
        if key in WORKER_STEPS:
            agent["steps"] = WORKER_STEPS[key]
        # The `loom` custom tool (registered globally by plugins/otools-loom.js) is
        # for the loom agent ONLY -- hide it from every other agent.
        agent["tools"] = {"loom": False}
        agents[key] = agent
        agent_sampling[key] = _preset_vec(preset, repetition_detection)

    # --- `loom`: the pipeline lead. Holds exactly ONE tool -- the `loom` custom tool
    # from plugins/otools-loom.js -- whose Python conductor runs the plan/code/test/
    # review pipeline deterministically over the hidden agent-* workers. Built when a
    # reason preset + at least one worker exist. Model comes from default_models. ---
    rp = presets.get("reason")
    workers = [k for k in agents if agents[k].get("mode") == "subagent"]
    if rp and workers:
        rs = rp.get("sampling", {})
        looma = {
            "description": "loom: pipeline lead -- one loom-tool call runs "
                           "plan/code/test/review deterministically over the agent-* workers",
            "mode": "primary",
            "model": model_ref,
            "color": LOOM_COLOR,
            "prompt": f"{{file:./prompts/{ROLE_PROMPT_FILES['loom']}}}",
            "options": _preset_options(rp, control, caps),
            # Intake-and-relay only: deny every workspace tool AND the task tool; the
            # pipeline (not this agent) does all delegation via the server API.
            "permission": {"read": "deny", "grep": "deny", "glob": "deny", "list": "deny",
                           "edit": "deny", "bash": "deny",
                           "webfetch": "deny", "websearch": "deny",
                           "task": "deny",
                           # "skill" deny does double duty: hides the skill tool AND
                           # removes the <available_skills> block from the system
                           # prompt (Skill.available filters on this permission).
                           # Without it the personality-creator skill advertises
                           # "use whenever editing dialogue lines" -- a direct
                           # invitation for intake to wander (observed 2026-07-26).
                           "skill": "deny"},
            "tools": {"loom": True, "skill": False, "todowrite": False},
        }
        if "temperature" in rs: looma["temperature"] = rs["temperature"]
        if "top_p" in rs: looma["top_p"] = rs["top_p"]
        agents["loom"] = looma
        agent_sampling["loom"] = _preset_vec(rp, repetition_detection)
    return agents, agent_sampling


# Cloud/frontier providers get their known-good config from OpenCode's model registry,
# NOT local vLLM knobs. When an agent lands on such a model we strip the DGX-only agent
# fields so it runs on the model's own correct defaults instead of stale local sampling.
_LOCAL_ONLY_AGENT_FIELDS = ("temperature", "top_p", "options")


def _apply_model_config_to_agent(agent_cfg, agent_name, model_ref, configs,
                                 reasoning_caps=None, repetition_detection=None,
                                 per_model_sampling=None):
    """Configure ONE agent for `model_ref` with THAT model's known-good settings.

    The single place every model-assignment path routes through (roster build,
    default_models application, and `--set-model`), so an agent ALWAYS carries the right
    sampling + thinking config for whatever model it runs on:
      * DGX-hosted model     -> the omodel-manager recipe preset for the agent's role
        (temperature/top_p/thinking knobs on the block + the plugin sampling vector);
        a model with no curated recipe falls back to the generic Qwen numbers.
      * Frontier/cloud model -> OpenCode's registry defaults; the DGX-only knobs are
        stripped so we never impose vLLM sampling/thinking on a cloud model.

    Mutates `agent_cfg` in place (only model + sampling fields; never touches
    permission/mode/prompt/task_budget) and updates
    per_model_sampling[model_id][agent_name] when a dict is passed."""
    role = AGENT_ROLE.get(agent_name, "code")
    model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
    provider = model_ref.split("/", 1)[0] if "/" in model_ref else ""
    agent_cfg["model"] = model_ref

    if provider.startswith(PROVIDER_PREFIX):
        recipe = match_recipe(model_id, configs) or GENERIC_QWEN_RECIPE
        caps = (reasoning_caps or {}).get(model_ref) or caps_from_capabilities(
            match_recipe(model_id, configs) or {})
        control = recipe.get("thinking_control", "auto")
        preset = (recipe.get("presets") or {}).get(role)
        if not preset:
            # Matched recipe lacks this role -> use the generic Qwen preset for it rather
            # than leaving the agent on the previous model's (now-wrong) knobs.
            control = GENERIC_QWEN_RECIPE.get("thinking_control", "auto")
            preset = (GENERIC_QWEN_RECIPE.get("presets") or {}).get(role)
        if not preset:
            return
        agent_cfg["options"] = _preset_options(preset, control, caps)
        s = preset.get("sampling", {})
        if "temperature" in s:
            agent_cfg["temperature"] = s["temperature"]
        else:
            agent_cfg.pop("temperature", None)
        if "top_p" in s:
            agent_cfg["top_p"] = s["top_p"]
        else:
            agent_cfg.pop("top_p", None)
        if per_model_sampling is not None:
            per_model_sampling.setdefault(model_id, {})[agent_name] = _preset_vec(
                preset, repetition_detection)
    else:
        # Frontier/cloud model: strip DGX-only knobs; OpenCode uses the model's own defaults.
        for field in _LOCAL_ONLY_AGENT_FIELDS:
            agent_cfg.pop(field, None)
        if per_model_sampling is not None and model_id in per_model_sampling:
            per_model_sampling[model_id].pop(agent_name, None)


def oc_loom_plugin_js():
    """plugins/otools-loom.js -- registers the `loom` custom tool (v2 orchestrator).

    The tool spawns the stdlib-Python loom conductor (this script's `loom` subcommand)
    against THIS server (`serverUrl` from PluginInput) with the calling session as the
    parent, then relays the conductor's --json-events lines into the live TUI tool card
    (ctx.metadata). Esc in the parent fires ctx.abort -> SIGTERM -> the job pauses
    resumably. Absolute python/script paths are baked at sync time."""
    python = sys.executable or "python3"
    script = os.path.abspath(__file__)
    return f"""// otools-loom.js  --  AUTO-GENERATED by omodel-wire.py (loom v2 orchestrator).
// Registers the `loom` custom tool: one call runs the plan/code/test/review pipeline
// deterministically over the agent-* workers, as REAL child sessions of the caller.
import {{ tool }} from "@opencode-ai/plugin"

const PYTHON = {json.dumps(python)}
const SCRIPT = {json.dumps(script)}

export const OtoolsLoom = async ({{ serverUrl, directory }}) => {{
  return {{
    // The loom agent is intake-and-relay: it never touches the workspace, so the
    // project AGENTS.md (repo layout, Lua gotchas) is pure context waste for it --
    // and skill advertisements there have derailed intake. Strip instruction-file
    // blocks from ITS system prompt only; workers keep theirs. The loom agent is
    // identified by its own prompt marker, so other agents are untouched.
    "experimental.chat.system.transform": async (_input, output) => {{
      // NOTE: by the time this hook fires, opencode has already JOINED the system
      // array into one string -- filtering array entries does nothing. Instead,
      // parse each "Instructions from: <path>" header, read that file, and remove
      // the exact block. Precise: only bytes that came from instruction files go.
      const fs = await import("node:fs")
      const sys = output.system || []
      if (!sys.some((s) => typeof s === "string" && s.includes("You are `loom`, a ROUTER"))) return
      // MUTATE IN PLACE: the caller keeps its own reference to this array and
      // ignores a reassigned output.system (verified in llm/request.ts prepare()).
      for (let i = 0; i < sys.length; i++) {{
        let s = sys[i]
        if (typeof s !== "string" || !s.includes("Instructions from: ")) continue
        for (const m of s.matchAll(/Instructions from: ([^\\n]+)\\n/g)) {{
          try {{
            const content = fs.readFileSync(m[1], "utf8")
            s = s.replace(m[0] + content, "").replace(m[0] + content.trimEnd(), "")
          }} catch {{}}
        }}
        sys[i] = s
      }}
    }},
    tool: {{
      loom: tool({{
        description:
          "Deterministic feature pipeline (plan/code/test/review over the agent-* workers). " +
          "action 'run' starts a job (needs packet, risk); 'ask' routes one question to " +
          "agent-research and returns a cited answer (needs packet=the question); 'status' " +
          "reports a job's phase, workers, errors, and event log (job optional: latest); " +
          "'resume' continues a paused job (needs job, optional note); 'pr' creates and " +
          "reviews the PR for a finished job (needs job).",
        args: {{
          action: tool.schema.enum(["run", "ask", "status", "resume", "pr"]).describe("what to do"),
          packet: tool.schema.string().optional()
            .describe("run: goal + acceptance criteria + scope boundary; ask: the question"),
          risk: tool.schema.enum(["simple", "medium", "high"]).optional()
            .describe("run: risk class (default medium)"),
          job: tool.schema.number().optional().describe("resume/pr: job id from the report"),
          note: tool.schema.string().optional().describe("resume: operator guidance to fold in"),
        }},
        async execute(args, ctx) {{
          // Resolve the CALLER's project directory per call. The plugin-level
          // `directory` is the instance cwd -- a dedicated `opencode serve` may run
          // from anywhere -- but jobs must anchor to the calling session's repo
          // (job.dir drives .loom/skills overlay lookup and worker cwd).
          const base0 = String(serverUrl).replace(/\\/$/, "")
          const auth0 = process.env.OPENCODE_SERVER_PASSWORD
            ? {{ Authorization: "Basic " +
                 btoa("opencode:" + process.env.OPENCODE_SERVER_PASSWORD) }}
            : {{}}
          let dir = directory
          try {{
            const sr = await fetch(`${{base0}}/session/${{ctx.sessionID}}`, {{ headers: auth0 }})
            if (sr.ok) {{
              const sj = await sr.json()
              if (sj && sj.directory) dir = sj.directory
            }}
          }} catch {{}}
          const argv = [PYTHON, SCRIPT, "loom"]
          if (args.action === "run") {{
            if (!args.packet) return "loom: 'run' needs a packet (goal + criteria + scope)."
            argv.push("run", "--attach", String(serverUrl), "--parent", ctx.sessionID,
                      "--risk", args.risk || "medium", "--dir", dir,
                      "--json-events", "--packet", args.packet)
          }} else if (args.action === "ask") {{
            if (!args.packet) return "loom: 'ask' needs the question in packet."
            argv.push("ask", "--attach", String(serverUrl), "--parent", ctx.sessionID,
                      "--dir", dir, "--json-events", "--question", args.packet)
          }} else if (args.action === "status") {{
            argv.push("status")
            if (args.job != null) argv.push("--job", String(args.job))
            const st = Bun.spawn(argv, {{ cwd: dir, stdout: "pipe", stderr: "pipe" }})
            const sout = await new Response(st.stdout).text()
            const serr = await new Response(st.stderr).text()
            await st.exited
            let extra = ""
            if (args.job != null) {{
              const lg = Bun.spawn([PYTHON, SCRIPT, "loom", "log", "--job", String(args.job)],
                                   {{ cwd: dir, stdout: "pipe" }})
              const logText = await new Response(lg.stdout).text()
              await lg.exited
              const tail = logText.trim().split("\\n").slice(-30).join("\\n")
              if (tail) extra = "\\n\\nEVENT LOG (last 30):\\n" + tail
            }}
            return {{ title: "loom status", output: (sout + serr).trim() + extra }}
          }} else if (args.action === "resume") {{
            if (args.job == null) return "loom: 'resume' needs the job id."
            argv.push("resume", "--job", String(args.job),
                      "--attach", String(serverUrl), "--json-events")
            if (args.note) argv.push("--note", args.note)
          }} else {{
            if (args.job == null) return "loom: 'pr' needs the job id."
            argv.push("pr", "--job", String(args.job), "--attach", String(serverUrl),
                      "--json-events")
          }}
          // ---- live tool-card updates ------------------------------------------
          // Upstream, plugin ctx.metadata() is a silent no-op (registry.ts bridges
          // `ask` into a Promise but spreads the Effect-returning `metadata` through
          // unrun), and the TUI's GenericTool renders ONLY state.input while a tool
          // runs. So we self-serve: PATCH our own tool part over the server API
          // (the same session.updatePart primitive the internal callback uses) and
          // carry the live status in state.input, which GenericTool repaints via
          // SSE. Original input is restored before the call completes.
          const base = String(serverUrl).replace(/\\/$/, "")
          const auth = process.env.OPENCODE_SERVER_PASSWORD
            ? {{ Authorization: "Basic " +
                 btoa("opencode:" + process.env.OPENCODE_SERVER_PASSWORD) }}
            : {{}}
          let partCache
          let origInput
          let origTool
          const findPart = async () => {{
            const r = await fetch(
              `${{base}}/session/${{ctx.sessionID}}/message/${{ctx.messageID}}`,
              {{ headers: auth }})
            if (!r.ok) return null
            const msg = await r.json()
            return (msg.parts || []).find(
              (p) => p.type === "tool" && p.callID === ctx.callID) || null
          }}
          const patchPart = async (mutate) => {{
            try {{
              if (partCache === undefined) {{
                partCache = await findPart()
                if (partCache) {{
                  origInput = partCache.state?.input
                  origTool = partCache.tool
                }}
              }}
              if (!partCache || partCache.state?.status !== "running") return
              const p = {{ ...partCache, state: {{ ...partCache.state }} }}
              mutate(p)
              const r = await fetch(
                `${{base}}/session/${{p.sessionID}}/message/${{p.messageID}}/part/${{p.id}}`,
                {{ method: "PATCH",
                   headers: {{ "content-type": "application/json", ...auth }},
                   body: JSON.stringify(p) }})
              if (r.ok) partCache = await r.json()
            }} catch {{}}
          }}
          const updateCard = (title) => patchPart((p) => {{
            p.state.title = title
            // once morphed to a task card, the child session provides the live
            // detail itself -- don't fight it by rewriting the input
            if (p.tool !== "task") p.state.input = {{ loom: title }}
          }})
          // The native subagent experience: morph our part into a `task` tool part
          // pointing metadata.sessionId at the ACTIVE worker session. The TUI's Task
          // renderer then does everything v1 does -- live child tool line, spinner,
          // duration, click/keys into the worker -- and follows each handoff.
          const followWorker = (session, title) => patchPart((p) => {{
            p.tool = "task"
            p.state.title = title
            p.state.metadata = {{ ...(p.state.metadata || {{}}),
                                  sessionId: session, parentSessionId: ctx.sessionID }}
            p.state.input = {{ description: title, subagent_type: "loom" }}
          }})
          // Completion card: a one-line summary. Restoring the original input
          // would repaint the ENTIRE packet -- the goal's third appearance in
          // the transcript (prompt, tool call, report). A finished card needs
          // neither the packet nor the task morph.
          const finishCard = (summary) => patchPart((p) => {{
            if (origTool !== undefined) p.tool = origTool
            p.state.title = summary
            p.state.input = {{ loom: summary }}
          }})

          const proc = Bun.spawn(argv, {{ cwd: dir, stdout: "pipe", stderr: "pipe" }})
          const onAbort = () => {{ try {{ proc.kill("SIGTERM") }} catch {{}} }}
          ctx.abort.addEventListener("abort", onAbort, {{ once: true }})
          let report = ""
          let dispatches = 0
          const lines = []
          // Milestones only: worker sessions are directly clickable in the TUI now,
          // so the status/dispatch/session/skills/compose chatter adds nothing.
          const KEEP = new Set(["start", "phase", "done", "error", "paused", "malformed"])
          const dec = new TextDecoder()
          let buf = ""
          for await (const chunk of proc.stdout) {{
            buf += dec.decode(chunk, {{ stream: true }})
            let i
            while ((i = buf.indexOf("\\n")) >= 0) {{
              const line = buf.slice(0, i)
              buf = buf.slice(i + 1)
              if (!line.trim()) continue
              let ev
              try {{ ev = JSON.parse(line) }} catch {{ lines.push(line); continue }}
              if (ev.type === "report") {{ report = ev.report; continue }}
              if (ev.type === "dispatch") dispatches++
              if (KEEP.has(ev.type)) lines.push(`${{ev.type}}: ${{ev.detail || ""}}`)
              const title = ev.title || ev.detail || ev.type
              try {{ ctx.metadata({{ title }}) }} catch {{}}   // no-op today; upstream fix welcome
              if (ev.session) await followWorker(ev.session, title)
              else await updateCard(title)
            }}
          }}
          const err = await new Response(proc.stderr).text()
          const code = await proc.exited
          const summary =
            code === 0
              ? (dispatches ? `pipeline complete (${{dispatches}} dispatches)` : "complete")
              : `stopped (exit ${{code}})`
          await finishCard(summary)
          if (report)
            return {{ title: summary, output: report, metadata: {{ exitCode: code }} }}
          if (code !== 0) {{
            const tail = lines.slice(-10).join("\\n")
            return {{ title: summary, output: (tail + "\\n" + err).trim() || `exit ${{code}}` }}
          }}
          return {{ title: summary, output: lines.join("\\n") || "(no output)" }}
        }},
      }}),
    }},
  }}
}}
"""


def oc_agent_sampling_plugin_js(per_model_sampling, default_model_id):
    """chat.params plugin that sets the FULL sampling vector per (model, agent).
    The vector applied depends on BOTH the running agent's role AND the model it is
    on, so switching an agent onto another model applies THAT model's card sampling.
    Keyed by served model id, then agent name; DEFAULT_MODEL is the fallback for a
    managed model with no table of its own. Mutates `output` in place."""
    table = json.dumps(per_model_sampling, indent=2)
    return f"""// dgx-sampling.js  --  AUTO-GENERATED by omodel-wire.py (recipe profiles).
// Per-(model, agent) sampling from the model-card recipes. The chat.params hook is
// the one client-side place that reliably sets temperature/topP/topK/maxOutputTokens
// and body options (min_p, presence_penalty), regardless of openai-compatible quirks.
// The vector is chosen by the CURRENT model (input.model.id) AND agent, so switching
// an agent's model re-tunes it per that model's card. Thinking stays in agent options.

const MANAGED_PREFIX = {json.dumps(PROVIDER_PREFIX)}
const SCOPE_ALL = false
const DEFAULT_MODEL = {json.dumps(default_model_id)}
const AGENT_SAMPLING = {table}

function isManaged(input) {{
  if (SCOPE_ALL) return true
  const p = input && input.provider ? input.provider : {{}}
  const m = input && input.model ? input.model : {{}}
  const ids = [p.id, p.name, p.providerID, m.providerID, m.id].filter(Boolean)
  return ids.some((x) => String(x).startsWith(MANAGED_PREFIX))
}}

export const DgxSampling = async () => {{
  return {{
    "chat.params": async (input, output) => {{
      if (!isManaged(input)) return
      const m = input && input.model ? input.model : {{}}
      const table = AGENT_SAMPLING[m.id] || AGENT_SAMPLING[DEFAULT_MODEL] || {{}}
      const s = table[input.agent]
      if (!s) return
      if ("temperature" in s) output.temperature = s.temperature
      if ("topP" in s) output.topP = s.topP
      if ("topK" in s) output.topK = s.topK
      if ("maxOutputTokens" in s) output.maxOutputTokens = s.maxOutputTokens
      if (s.options) {{
        output.options = output.options || {{}}
        for (const k in s.options) output.options[k] = s.options[k]
      }}
    }},
  }}
}}
"""


# ============================================================================
# OpenCode configurator
# ============================================================================
def oc_build_providers(hosts, ports, timeout, sampling, profiles=False,
                       tool_call=True, recipes=None, verbose=True):
    """Discover endpoints; return (providers dict, flat refs list, reasoning_caps).

    Capabilities (reasoning / thinking-knob / vision) are DECLARED in the matched
    per-model config (omodel-manager's configs/*.toml) -- no live probing here.
    reasoning_caps maps a model ref -> its synthesized caps dict, so the caller
    can build matching agents."""
    providers = {}
    refs = []
    reasoning_caps = {}
    available_models = {}
    for host in hosts:
        for port in ports:
            found = probe(host, port, timeout)
            if not found:
                continue
            key = f"{PROVIDER_PREFIX}{host_label(host)}-{port}"
            base = f"http://{host}:{port}/v1"
            model_entries = {}
            for m in found:
                ctx = m["max_model_len"] or DEFAULT_CONTEXT
                out = min(DEFAULT_OUTPUT, max(4096, ctx // 2))
                entry = {
                    "name": f"{m['id']} ({host_label(host)}:{port})",
                    "limit": {"context": ctx, "output": out},
                }
                rec = match_recipe(m["id"], recipes) if recipes else None

                # ---- FIX #3: declare tool-call capability --------------------
                # OpenCode does NOT send tool definitions to a custom
                # openai-compatible model unless it declares this capability.
                # Without it, websearch/edit/bash/etc. are invisible to the model.
                if tool_call:
                    entry["tool_call"] = True

                # ---- FIX #2: stop OpenCode injecting temperature -------------
                # In profiles mode, agents own sampling, so KEEP the temperature
                # capability ON -- it's REQUIRED for agent.temperature to be
                # honored (OpenCode gates: capabilities.temperature ? agent.temp
                # : undefined). Set it for every model here, not just reasoning ones.
                if profiles:
                    entry["temperature"] = True
                elif sampling["mode"] != "opencode-default":
                    if sampling["mode"] == "server-default":
                        # capabilities.temperature=false -> OpenCode sends none
                        entry["temperature"] = False
                    elif sampling["mode"] == "fixed" and sampling["temperature"] is not None:
                        # keep the capability on; the plugin pins the value
                        entry["temperature"] = True

                # ---- PROFILES: declared reasoning + thinking variants (config) ----
                if profiles and rec and (rec.get("capabilities") or {}).get("reasoning"):
                    caps = caps_from_capabilities(rec)
                    entry["reasoning"] = True
                    entry["temperature"] = True   # let agent/variant temps apply
                    entry["variants"] = oc_build_variants(caps)
                    reasoning_caps[f"{key}/{m['id']}"] = caps
                    if verbose:
                        print(f"    reasoning: {m['id']} -> declared ({rec['_file']})")
                elif profiles and verbose:
                    print(f"    reasoning: {m['id']} -> skipped "
                          f"({'config: non-reasoning' if rec else 'no config match'})")

                # ---- vision: declared in the config (no probing) -------------
                vis = (rec.get("capabilities") or {}).get("vision") if rec else None
                if vis:
                    entry["attachment"] = True
                    entry["modalities"] = vis if isinstance(vis, dict) else \
                        {"input": ["text", "image"], "output": ["text"]}
                    if verbose:
                        print(f"    vision: {m['id']} -> ENABLED (declared)")

                model_entries[m["id"]] = entry
                refs.append(f"{key}/{m['id']}")
                # Track all available models (not just reasoning) for default model selection
                available_models[f"{key}/{m['id']}"] = {}

            providers[key] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"DGX {host_label(host)}:{port}",
                "options": {"baseURL": base, "apiKey": API_KEY},
                "models": model_entries,
            }
            if verbose:
                ids = ", ".join(m["id"] for m in found)
                print(f"  [up]   {host}:{port}  ->  {ids}")
    return providers, refs, reasoning_caps, available_models


def oc_sampling_plugin_js(sampling):
    """Generate the chat.params plugin that owns the sampling params.

    Hook contract (packages/plugin/src/index.ts): `chat.params(input, output)`
    returns void -> we MUTATE `output` in place. temperature/topP/topK are
    top-level on `output`; penalties live in `output.options`.
    """
    mode = sampling["mode"]
    rep_det = sampling.get("repetition_detection")
    rep_det_json = json.dumps(rep_det) if rep_det else "undefined"
    if mode == "fixed":
        t = "undefined" if sampling["temperature"] is None else json.dumps(sampling["temperature"])
        p = "undefined" if sampling["top_p"] is None else json.dumps(sampling["top_p"])
        k = "undefined" if sampling["top_k"] is None else json.dumps(sampling["top_k"])
        pp = sampling["presence_penalty"]
        fp = sampling["frequency_penalty"]
        body = f"""      // --- fixed-sampling policy ---
      output.temperature = {t}
      output.topP = {p}
      output.topK = {k}
      output.options = output.options || {{}}
"""
        if pp is None:
            body += "      delete output.options.presence_penalty\n"
        else:
            body += f"      output.options.presence_penalty = {json.dumps(pp)}\n"
        if fp is None:
            body += "      delete output.options.frequency_penalty\n"
        else:
            body += f"      output.options.frequency_penalty = {json.dumps(fp)}\n"
    else:  # server-default
        body = """      // --- server-default policy: let the inference server decide ---
      output.temperature = undefined
      output.topP = undefined
      output.topK = undefined
      output.options = output.options || {}
      delete output.options.presence_penalty
      delete output.options.frequency_penalty
"""

    return f"""// dgx-sampling.js  --  AUTO-GENERATED by omodel-wire.py. Do not hand-edit;
// re-run the script to regenerate, or set --sampling opencode-default to remove.
//
// Why this exists: OpenCode injects topP/topK (and for Qwen a 0.55 temperature)
// that are NOT controllable from opencode.json. The chat.params hook is the one
// client-side place that can override/remove them before the request is built.
// We mutate `output` in place (the hook returns void).

// Apply only to the providers this script manages. If your provider context
// field differs, flip SCOPE_ALL to true (safe if ALL your providers are DGX).
const MANAGED_PREFIX = {json.dumps(PROVIDER_PREFIX)}
const SCOPE_ALL = false

function isManaged(input) {{
  if (SCOPE_ALL) return true
  const p = input && input.provider ? input.provider : {{}}
  const m = input && input.model ? input.model : {{}}
  const ids = [p.id, p.name, p.providerID, m.providerID, m.id].filter(Boolean)
  return ids.some((x) => String(x).startsWith(MANAGED_PREFIX))
}}

export const DgxSampling = async () => {{
  return {{
    "chat.params": async (input, output) => {{
      if (!isManaged(input)) return
{body}      // --- repetition_detection: terminate degenerate N-gram loops ---
      output.options.repetition_detection = {rep_det_json}
    }},
  }}
}}
"""


def _kv_list_to_dict(pairs):
    """['K=V', ...] -> {'K': 'V'}. Value may contain '='."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            print(f"  warning: ignoring malformed KEY=VAL: {p}")
            continue
        k, v = p.split("=", 1)
        out[k] = v
    return out


def _shell_rc():
    """Best-effort (rc_path, kind) for the user's shell."""
    shell = os.environ.get("SHELL", "")
    home = os.path.expanduser("~")
    if shell.endswith("zsh"):
        return os.path.join(home, ".zshrc"), "posix"
    if shell.endswith("fish"):
        return os.path.join(home, ".config", "fish", "config.fish"), "fish"
    return os.path.join(home, ".bashrc"), "posix"


def _export_line(kind, var, val):
    return f"set -gx {var} {val}\n" if kind == "fish" else f"export {var}={val}\n"


def _ensure_shell_env(var, val, do_write):
    """Idempotently make VAR=val available to OpenCode via the shell rc.
    Returns a human note. Only writes when do_write is true."""
    if os.environ.get(var):
        return f"{var} already set in this shell."
    rc, kind = _shell_rc()
    if not do_write:
        return (f"set {var} to use this:  {_export_line(kind, var, val).strip()}  "
                f"(add to {rc}, or re-run with --write-shell-env)")
    try:
        existing = ""
        if os.path.exists(rc):
            with open(rc) as f:
                existing = f.read()
        if var in existing:
            return f"{var} already present in {rc}."
        os.makedirs(os.path.dirname(rc), exist_ok=True)
        with open(rc, "a") as f:
            f.write(f"\n# added by omodel-wire.py\n{_export_line(kind, var, val)}")
        return f"Appended {var}={val} to {rc} (open a new shell to apply)."
    except OSError as e:
        return f"could not edit {rc} ({e}); set {var}={val} yourself."


def install_aliases():
    """Add the `omw` alias (this sync tool) to the shell rc."""
    sync = os.path.abspath(__file__)
    rc, kind = _shell_rc()

    def aline(name, cmd):
        return f"alias {name} '{cmd}'" if kind == "fish" else f"alias {name}='{cmd}'"

    block = "\n".join([
        "# omodel-wire alias (added by omodel-wire.py --install-aliases)",
        aline("omw", f'python3 "{sync}"'),
    ]) + "\n"

    try:
        existing = ""
        if os.path.exists(rc):
            with open(rc) as f:
                existing = f.read()
        if "alias omw" in existing:
            print(f"omw alias already present in {rc} (leaving as-is).")
            return
        os.makedirs(os.path.dirname(rc), exist_ok=True)
        with open(rc, "a") as f:
            f.write("\n" + block)
        print(f"Added alias to {rc}:")
        print(f"  omw -> python3 {sync}")
        print(f"Run:  source {rc}   (or open a new shell) to start using it.")
    except OSError as e:
        print(f"ERROR: could not write {rc}: {e}", file=sys.stderr)
        sys.exit(1)


def oc_apply_web_search(cfg, args):
    """Mutate cfg to expose a web-search tool to all models. Returns notes[]."""
    notes = []
    mode = args.web_search
    if mode == "none":
        return notes

    perm = cfg.setdefault("permission", {})

    if mode == "exa":
        # Built-in Exa websearch (keyless) + always-on webfetch. Gated behind
        # the OPENCODE_ENABLE_EXA env var for non-OpenCode providers.
        perm.setdefault("websearch", "allow")
        perm.setdefault("webfetch", "allow")
        notes.append(_ensure_shell_env("OPENCODE_ENABLE_EXA", "1",
                                       args.enable_exa_shell or args.write_shell_env))
        return notes

    if mode == "mcp":
        name = args.mcp_name
        mcp = cfg.setdefault("mcp", {})
        if args.mcp_url:
            server = {"type": "remote", "url": args.mcp_url, "enabled": True}
            headers = _kv_list_to_dict(args.mcp_header)
            if headers:
                server["headers"] = headers
        elif args.mcp_command:
            server = {"type": "local", "command": shlex.split(args.mcp_command),
                      "enabled": True}
            env = _kv_list_to_dict(args.mcp_env)
            if env:
                server["environment"] = env
        else:
            print("ERROR: --web-search mcp needs --mcp-command or --mcp-url.", file=sys.stderr)
            sys.exit(1)
        mcp[name] = server
        perm.setdefault(f"{name}_*", "allow")
        perm.setdefault("webfetch", "allow")
        notes.append(f"Added MCP server '{name}'; its tools (and webfetch) are exposed to all models.")
        return notes

    return notes


def _print_roster(agents):
    """Print the resulting agent roster (Tab cycle + hidden workers)."""
    primary = [k for k in agents if agents[k].get("mode") in ("primary", "all")]
    hidden = [k for k in agents if agents[k].get("mode") == "subagent"]
    print(f"  Tab cycle (visible): {', '.join(primary)}")
    if hidden:
        print(f"  hidden workers (delegation-only, not in Tab): {', '.join(hidden)}")
    if "loom" in agents:
        print("  loom pipeline: agent-architect -> agent-code -> agent-test -> agent-review")


def oc_verify(args):
    """Opt-in: probe live endpoints and compare real capabilities to the declared
    configs. Writes nothing. Exit 0 = all match, 1 = mismatch(es), 2 = none found."""
    configs = load_configs(args.configs)
    found_any, mismatches = False, 0
    for host in args._hosts:
        for port in args._ports:
            found = probe(host, port, args.timeout)
            if not found:
                continue
            found_any = True
            for mm in found:
                mid = mm["id"]
                rec = match_recipe(mid, configs)
                print(f"\n{host}:{port}  {mid}")
                if not rec:
                    print("  [MISS] no config matches -- add one to configs/ or fix `match`")
                    mismatches += 1
                    continue
                cap = rec.get("capabilities", {}) or {}
                print(f"  config: {rec['_file']}")
                # reasoning
                dr = bool(cap.get("reasoning"))
                pr = probe_reasoning(host, port, mid, REASONING_TIMEOUT)["reasoning"]
                ok = dr == pr
                mismatches += 0 if ok else 1
                print(f"  {'[ok] ' if ok else '[DIFF]'} reasoning: declared={dr} probed={pr}")
                # vision (probe regardless, to catch under-declared multimodal models)
                dv = bool(cap.get("vision"))
                pv, _, why = probe_vision(host, port, mid, VISION_TIMEOUT)
                ok = dv == pv
                mismatches += 0 if ok else 1
                print(f"  {'[ok] ' if ok else '[DIFF]'} vision:    declared={dv} probed={pv}  [{why}]")
    if not found_any:
        print("No live endpoints found to verify.")
        return 2
    print(f"\n{'All declared capabilities match.' if not mismatches else f'{mismatches} mismatch(es) -- update the config(s) or the model.'}")
    return 0 if not mismatches else 1


# ---- audit: live OpenCode config vs omodel-manager recommendations (offline) ----
_AUDIT_FIELDS = ["temperature", "top_p", "top_k", "min_p", "presence_penalty",
                 "max_output", "enable_thinking", "preserve_thinking"]


def _audit_color(s, code):
    return s if not sys.stdout.isatty() else "\x1b[%sm%s\x1b[0m" % (code, s)


def _audit_fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "%g" % v
    return str(v)


def _audit_norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    return v


def _plugin_agent_sampling(config_path):
    """Extract the AGENT_SAMPLING {...} object from the generated
    plugins/dgx-sampling.js. Returns the dict, {} if present-but-unparseable, or
    None if the plugin file is absent (its knobs then fall back to server defaults)."""
    p = os.path.join(os.path.dirname(config_path), "plugins", "dgx-sampling.js")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            txt = f.read()
    except OSError:
        return None
    m = re.search(r"AGENT_SAMPLING\s*=\s*\{", txt)
    if not m:
        return {}
    start = txt.index("{", m.start())
    depth = 0
    for j in range(start, len(txt)):
        if txt[j] == "{":
            depth += 1
        elif txt[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[start:j + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _audit_vec(agent_cfg, plugin_vec):
    """Flatten one agent's EFFECTIVE sampling: temp/top_p + thinking from the
    opencode.json agent block, top_k/min_p/presence/max_output from its plugin
    vector. Works for the live side and the expected side (same shapes)."""
    ck = ((agent_cfg or {}).get("options") or {}).get("chat_template_kwargs") or {}
    s = plugin_vec or {}
    o = s.get("options") or {}
    return {
        "temperature": s.get("temperature", (agent_cfg or {}).get("temperature")),
        "top_p": s.get("topP", (agent_cfg or {}).get("top_p")),
        "top_k": s.get("topK"),
        "min_p": o.get("min_p"),
        "presence_penalty": o.get("presence_penalty"),
        "max_output": s.get("maxOutputTokens"),
        "enable_thinking": ck.get("enable_thinking"),
        "preserve_thinking": ck.get("preserve_thinking"),
    }


def oc_audit(args):
    """Offline: compare the LIVE OpenCode agent sampling to the omodel-manager configs
    it was generated from -- side by side, per model and agent -- and highlight drift.
    Writes nothing. Exit 0 = in sync, 1 = drift found, 2 = nothing to compare."""
    config_path = os.path.expanduser(args.config)
    cfg = oc_load_config(config_path)
    configs = load_configs(args.configs)
    agents = cfg.get("agent") or {}
    providers = cfg.get("provider") or {}
    plugin = _plugin_agent_sampling(config_path)
    plugin_missing = plugin is None
    plugin = plugin or {}
    # An old (pre-per-model) plugin has agent names at the top level, not model ids.
    flat_plugin = any(k in MANAGED_AGENTS for k in plugin)

    # Managed provider models (registered endpoints) -- shown even if no agent uses them.
    prov = {}
    for pkey, pentry in providers.items():
        if not str(pkey).startswith(PROVIDER_PREFIX):
            continue
        for mid, mentry in ((pentry or {}).get("models") or {}).items():
            prov["%s/%s" % (pkey, mid)] = mentry or {}
    # Managed roster agents grouped by the model they run on.
    by_model = {}
    for k in agents:
        a = agents[k] or {}
        if k in MANAGED_AGENTS and a.get("model") and not a.get("disable"):
            by_model.setdefault(a["model"], []).append(k)

    if not prov and not by_model:
        print(f"No managed models/agents in {config_path} -- run `omw --profiles` first.")
        return 2

    print("omodel-wire audit -- live OpenCode config vs omodel-manager recommendations")
    print(f"  config: {config_path}")
    print("  plugin: " + ("(MISSING -- top_k/penalties/max_output revert to server "
                          "defaults)" if plugin_missing else "plugins/dgx-sampling.js"))

    diffs = 0

    def row(scope, param, av, ev):
        nonlocal diffs
        d = _audit_norm(av) != _audit_norm(ev)
        diffs += 1 if d else 0
        line = "    %-16s %-17s %10s %10s  %s" % (
            scope, param, _audit_fmt(av), _audit_fmt(ev),
            _audit_color("DIFF", "31") if d else "")
        print(_audit_color(line, "1") if d else line)

    for model_ref in sorted(set(prov) | set(by_model)):
        provider = model_ref.split("/", 1)[0] if "/" in model_ref else "?"
        served = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
        names = sorted(by_model.get(model_ref, []))
        print(f"\nModel: {served}   [{provider}]")
        if not str(model_ref).startswith(PROVIDER_PREFIX):
            print(f"  external model (not omodel-manager-managed): "
                  f"{', '.join(names)}  [skipped]")
            continue
        rec = match_recipe(served, configs)
        if not rec:
            print(f"  no omodel-manager config matches '{served}' -- "
                  f"add one to configs/ or fix `match`  [skipped]")
            continue
        print(f"  config: {rec.get('_file')}")
        caps = caps_from_capabilities(rec)
        cap = rec.get("capabilities", {}) or {}
        mentry = prov.get(model_ref, {})
        print(f"    {'scope':16} {'param':17} {'OpenCode':>10} {'omodel-mgr':>10}")
        # model-level capabilities (shown for every managed model)
        oc_vision = "image" in ((mentry.get("modalities") or {}).get("input") or [])
        row("[model]", "reasoning", bool(mentry.get("reasoning")), bool(cap.get("reasoning")))
        row("[model]", "vision", oc_vision, bool(cap.get("vision")))
        row("[model]", "tool_call", bool(mentry.get("tool_call")), bool(cap.get("tool_call")))
        # per-(model, agent) sampling: what each role gets WHEN RUN ON THIS MODEL.
        if flat_plugin:
            print("    (plugin is the OLD flat format -- run `omw --profiles` to upgrade "
                  "to per-model sampling)")
            diffs += 1
            continue
        pm = plugin.get(served, {})   # this model's per-agent vectors from the plugin
        if not pm:
            print("    (no per-model sampling in the plugin for this model -- agents run "
                  "on it use server defaults; run `omw --profiles`)")
            diffs += 1
            continue
        exp_agents, exp_sampling = oc_build_recipe_agents(model_ref, rec, caps)
        for name in sorted(exp_sampling):
            if name == "loom":  # orchestrator; audited on its own model, not here
                continue
            act = _audit_vec(agents.get(name, {}), pm.get(name))
            exp = _audit_vec(exp_agents.get(name, {}), exp_sampling.get(name))
            for fld in _AUDIT_FIELDS:
                av, ev = act.get(fld), exp.get(fld)
                if av is None and ev is None:
                    continue
                row(name, fld, av, ev)

    print()
    if diffs:
        print(_audit_color(
            f"{diffs} difference(s) -- run `omw --profiles` to re-sync OpenCode to the "
            f"configs.", "33"))
        return 1
    print("In sync: OpenCode matches the omodel-manager recommendations.")
    return 0


def oc_sync(args, sampling, detected_installed):
    """Run the OpenCode sync. Returns exit code (0 ok, 2 nothing found)."""
    config_path = os.path.expanduser(args.config)

    configs = load_configs(args.configs) if not args.no_recipes else {"recipes": []}
    
    print(f"Probing {len(args._hosts)} host(s) x {len(args._ports)} port(s) for OpenCode ...")
    providers, refs, reasoning_caps, available_models = oc_build_providers(
        args._hosts, args._ports, args.timeout, sampling,
        profiles=args.profiles, tool_call=not args.no_tool_call, recipes=configs)

    if not providers:
        print("  (no live endpoints found)")
        if not args.allow_empty:
            print("\nRefusing to rewrite config because nothing was discovered.")
            print("If you really stopped all models, re-run with --allow-empty.")
            # The generated artifacts (skills/prompts/loom plugin) carry no model
            # data -- refresh them anyway so a `git pull` takes effect even while
            # the model server is down.
            if args.dry_run:
                print("(dry run: would still refresh role skills/prompts/plugin)")
            else:
                refreshed = oc_refresh_role_artifacts(config_path)
                if refreshed:
                    print(f"Refreshed {len(refreshed)} generated artifact(s) "
                          "(role skills/prompts/loom plugin) from current code; "
                          "config untouched.")
            return 2

    cfg = oc_load_config(config_path)
    cfg.setdefault("$schema", "https://opencode.ai/config.json")
    existing = cfg.get("provider", {})
    kept = {k: v for k, v in existing.items() if not oc_is_managed(k)}
    removed = [k for k in existing if oc_is_managed(k)]
    kept.update(providers)
    
    # Use disabled_providers array to disable built-in providers (OpenCode and Hugging Face)
    # This is the proper OpenCode way to hide providers at the provider level
    # Managed providers we control: always remove them from existing_disabled first,
    # then add back only when the flag is absent (to preserve user-authored entries)
    MANAGED_DISABLED = {"opencode", "huggingface"}
    existing_disabled = set(cfg.get("disabled_providers", []))
    
    if args.add_default_providers:
        # Remove managed providers from disabled list (enable them)
        disabled_providers_list = sorted(existing_disabled - MANAGED_DISABLED)
    else:
        # Add managed providers to disabled list (disable them), preserving user entries
        disabled_providers_list = sorted(existing_disabled | MANAGED_DISABLED)
    
    cfg["disabled_providers"] = disabled_providers_list if disabled_providers_list else []
    
    cfg["provider"] = kept

    # Collect all available models from multiple sources:
    # 1. Local probes (DGX endpoints)
    # 2. OpenCode runtime models (from `opencode models`)
    # 3. Existing config providers (fallback/back-compat)
    # 
    # The runtime models source is authoritative for OpenCode's built-in providers
    # (like openai/) that are NOT in opencode.json's provider block.
    # 
    # Build this BEFORE the profiles block so the condition check at line 1878
    # can see remote models that are available via runtime discovery.
    
    # Start with local probes
    all_available_models = dict(available_models)
    
    # Discover OpenCode runtime models (built-in providers like openai/, anthropic/)
    runtime_models, runtime_note = discover_opencode_runtime_models()
    runtime_success = bool(runtime_models)
    if runtime_success:
        for ref in runtime_models:
            all_available_models[ref] = {}
        print(f"  runtime: {runtime_note}")
    else:
        print(f"  runtime: {runtime_note}")
    
    # Also include models from existing config providers (for fallback/back-compat)
    # For REMOTE_PROVIDERS, only include if runtime discovery succeeded AND the model
    # appears in runtime_models. If runtime discovery failed, include all existing
    # config models as fallback. Local managed providers (dgx-) are unaffected by
    # runtime discovery and use live probe results.
    for prov_key, prov_cfg in existing.items():
        if prov_key in kept:  # only include providers we're keeping
            models = prov_cfg.get("models", {})
            for mid, mcfg in models.items():
                ref = f"{prov_key}/{mid}"
                if ref in all_available_models:
                    continue  # already added (from local probes or runtime)
                # For remote providers, only include if runtime discovery succeeded
                # AND this model is in the runtime list. If runtime failed, include
                # existing config models as fallback/back-compat.
                if prov_key in REMOTE_PROVIDERS:
                    if runtime_success and ref not in runtime_models:
                        continue  # stale remote model not in runtime discovery
                all_available_models[ref] = {}

    # Default model handling
    if args.set_default:
        cfg["model"] = args.set_default
    else:
        cur = cfg.get("model")
        valid = set(refs)
        if cur and cur not in valid and oc_is_managed(cur.split("/")[0]):
            if refs:
                cfg["model"] = refs[0]
                print(f"\nNote: previous default '{cur}' is no longer running; "
                      f"switched default to '{refs[0]}'.")
            else:
                cfg.pop("model", None)

    # ---- PROFILES: build role-agents (replacing built-in plan/build) ----------
    # Emits the visible research/code/agent primaries, the loom pipeline lead, and
    # the hidden agent-* workers the conductor dispatches.
    # Curated recipe supplies per-model sampling; else generic Qwen numbers.
    agents = {}
    agent_model_ref = None
    per_model_sampling = None
    matched_recipe = None
    env_notes = []
    # Use all_available_models (merged pool) for the condition check so we can build
    # the roster even when only remote models are available (via runtime discovery).
    # Fall back to available_models (local-only) if all_available_models is empty.
    roster_pool = all_available_models if all_available_models else available_models
    if args.profiles and roster_pool:
        # Build the roster from whatever is LIVE. Prefer a reasoning model for the
        # primary ref when one exists; otherwise fall back to any live model (e.g. a
        # coder-only fleet) so the roster is rebuilt from the live endpoints instead
        # of leaving stale agents pointing at a model that is no longer served.
        cur = cfg.get("model")
        # Use all_available_models (merged pool of local probes + existing config providers)
        # instead of available_models (local-only) so remote refs like openai/gpt-5.5
        # from existing config are considered for orchestrator selection.
        pool = reasoning_caps if reasoning_caps else all_available_models

        # Build available_model_ids for remote model detection
        available_model_ids = []
        for model_ref in all_available_models.keys():
            model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
            available_model_ids.append(model_id)

        # Check if user prefers a remote model (e.g., openai/gpt-5.5) for loom
        default_models = (_load_default_models(create=False) if args.dry_run
                          else load_default_models())
        loom_prefs = (default_models.get("agents") or {}).get("loom", [])

        # Prefer a reasoning model first, then check loom preferences
        agent_model_ref = None
        if reasoning_caps:
            agent_model_ref = cur if cur in reasoning_caps else sorted(reasoning_caps)[0]
        else:
            # No reasoning models - validate loom preferences against all_available_models
            # Use the same resolver logic as _apply_default_models so unavailable
            # cloud refs like anthropic/claude-opus-4-8 and unavailable local served IDs
            # are not selected.
            for pref in loom_prefs:
                resolved = _resolve_model_ref_from_prefs(pref, all_available_models)
                if resolved is not None:
                    agent_model_ref = resolved
                    break

            if not agent_model_ref:
                agent_model_ref = cur if cur in pool else sorted(pool)[0]
        model_id = agent_model_ref.split("/", 1)[1] if "/" in agent_model_ref else agent_model_ref
        matched_recipe = match_recipe(model_id, configs)
        # Reasoning models carry declared caps; a non-reasoning model derives them
        # from its config (caps_from_capabilities handles a None/{} recipe -> a valid
        # non-reasoning caps dict, so build never KeyErrors on an unmatched model).
        caps = reasoning_caps.get(agent_model_ref) or caps_from_capabilities(matched_recipe or {})
        if matched_recipe:
            agents, _ = oc_build_recipe_agents(
                agent_model_ref, matched_recipe, caps,
                sampling.get("repetition_detection"))
            try:
                key, mid = agent_model_ref.split("/", 1)
                mentry = cfg["provider"][key]["models"][mid]
                # Recipe-declared variants (raw, model-native kwargs) override the
                # generic probe-derived ones -- e.g. Nemotron's enable_thinking /
                # low_effort, which the Qwen-style probe variants can't express.
                if matched_recipe.get("variants"):
                    mentry["variants"] = matched_recipe["variants"]
                ctx = mentry["limit"]["context"]   # auto-probed: matches running 128K/256K server
                # If the recipe declares a larger per-request budget, raise the output
                # cap so the largest preset isn't clamped; never exceed the context.
                max_out = max((p.get("max_output", 0)
                               for p in matched_recipe.get("presets", {}).values()), default=0)
                if max_out:
                    mentry["limit"]["output"] = min(max(mentry["limit"].get("output", 0), max_out), ctx)
                # context sufficiency warning (card: keep >= min_thinking for thinking)
                need = matched_recipe.get("context", {}).get("min_thinking")
                if need and ctx < need:
                    print(f"  warning: max_model_len {ctx} < recommended {need} for thinking "
                          f"(per {matched_recipe.get('source','recipe')})")
            except (KeyError, ValueError):
                pass
        else:
            agents, _ = oc_build_agents(
                agent_model_ref, caps, sampling.get("repetition_detection"))

        # Apply default models preferences to agents (default_models already loaded above)
        model_notes = []
        agents = _apply_default_models(agents, default_models, reasoning_caps,
                                       all_available_models, model_notes,
                                       remote_ok=_authed_remote_providers())
        for note in model_notes:
            print(f"  {note}")

        # Per-(model, agent) sampling: one table per discovered model, so switching
        # an agent onto another model applies THAT model's card sampling (10+ models
        # with different recommended temps each get their own vector). Covers
        # non-reasoning models too, so a coder-only fleet still gets its sampling.
        _rep_det = sampling.get("repetition_detection")
        per_model_sampling = {}
        for _mref in available_models:
            _mid = _mref.split("/", 1)[1] if "/" in _mref else _mref
            _mrec = match_recipe(_mid, configs)
            _mcaps = reasoning_caps.get(_mref) or caps_from_capabilities(_mrec or {})
            if _mrec:
                _, _msamp = oc_build_recipe_agents(_mref, _mrec, _mcaps, _rep_det)
            else:
                _, _msamp = oc_build_agents(_mref, _mcaps, _rep_det)
            per_model_sampling[_mid] = _msamp

        # Re-derive EACH agent's block for the model it actually ended up on. The roster
        # was built on one primary model, but _apply_default_models may have moved agents
        # onto other models; without this, an agent keeps the primary's thinking knobs /
        # sampling (wrong for a different DGX model) or stale local knobs (wrong on a cloud
        # model). This is the single guarantee that every agent carries its model's
        # known-good config -- DGX recipe preset or frontier defaults -- and keeps the
        # plugin's per-(model, agent) table in sync.
        for _name, _a in agents.items():
            if isinstance(_a, dict) and _a.get("model"):
                _apply_model_config_to_agent(_a, _name, _a["model"], configs,
                                             reasoning_caps, _rep_det, per_model_sampling)

        # OpenCode caps per-step output at 32k UNLESS this env var is raised. A
        # reasoning model spends output tokens *thinking* before it answers, so one
        # coding turn easily exceeds 32k and OpenCode cuts it off mid-thought. Raise
        # the cap to the model's real output limit (set from the recipe/context
        # above, else the auto default) -- NOT gated on a recipe declaring max_output,
        # since a reasoning model without a curated output rec still needs it.
        try:
            key, mid = agent_model_ref.split("/", 1)
            out_limit = cfg["provider"][key]["models"][mid]["limit"]["output"]
            if out_limit > 32000:
                env_notes.append(_ensure_shell_env(
                    "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", str(out_limit),
                    args.write_shell_env))
        except (KeyError, ValueError, TypeError):
            pass
        existing_agents = cfg.get("agent", {}) or {}
        # Rebuild the agent map so OUR agents are written in a FIXED canonical
        # order every sync (research, code, agent, loom, ...workers...). OpenCode's
        # Tab cycle follows the config object order in practice, and dict.update()
        # would otherwise preserve a STALE order from a prior sync. User-created
        # agents are kept (in their order). Retired managed agents (team,
        # agent-instruct) drop out here: they are MANAGED but no longer built.
        user_agents = {k: v for k, v in existing_agents.items() if k not in MANAGED_AGENTS}
        # Write VISIBLE primaries first (research, code, agent, loom) as a contiguous
        # block, THEN the hidden agent-* workers, then disabled built-ins, then the
        # user's own agents.
        prim = {k: v for k, v in agents.items() if v.get("mode") != "subagent"}
        subs = {k: v for k, v in agents.items() if v.get("mode") == "subagent"}
        # Disable the reserved built-ins build/plan (OpenCode won't let us override
        # them, and they clutter the menu as "native"). We replace them with the
        # custom `code`/`research` agents above. --keep-builtins opts out.
        disabled = {} if args.keep_builtins else {b: {"disable": True} for b in BUILTIN_DISABLE}
        cfg["agent"] = {**prim, **subs, **disabled, **user_agents}
        # `build` was OpenCode's default agent; since it's now disabled, point the
        # default at one of our agents so startup doesn't fall back to a disabled one.
        if not args.keep_builtins and args.default_agent in cfg["agent"]:
            cfg["default_agent"] = args.default_agent

    # Retired per-role skill permission blocks: role guidance no longer ships as
    # skills, so a stale `permission.skill` map on a managed agent (from an earlier
    # sync) would only hide genuinely optional skills. Scrub it even when no roster
    # was rebuilt this run.
    for _name, _a in (cfg.get("agent") or {}).items():
        # "loom" is exempt: its permission.skill = "deny" is deliberate -- it hides
        # the skill tool AND strips <available_skills> from its system prompt.
        if _name == "loom":
            continue
        if _name in MANAGED_AGENTS and isinstance(_a, dict):
            _perm = _a.get("permission")
            if isinstance(_perm, dict):
                _perm.pop("skill", None)

    # ---- Minimal role prompts + global skills + loom instruction files -------
    role_prompt_paths, role_skill_paths = {}, {}
    if args.profiles:
        for agent_name in ROLE_SKILL_NAMES:
            if agent_name not in agents:
                continue
            role_prompt_paths[agent_name] = _role_prompt_path(config_path, agent_name)
        for skill_name in GLOBAL_SKILLS:
            role_skill_paths[skill_name] = _role_skill_path(config_path, skill_name)

    # ---- Web search / tool exposure -----------------------------------------
    web_notes = oc_apply_web_search(cfg, args)

    # ---- Sampling plugin ------------------------------------------------------
    # recipe profiles -> agent-aware plugin; plain (non-profiles) -> mode plugin.
    # OpenCode loads plugins from the `plugins/` directory (plural) next to the
    # config -- see https://opencode.ai/docs/plugins/. (Was `plugin/`; fixed so the
    # sampling plugin actually gets picked up.)
    plugin_path = os.path.join(os.path.dirname(config_path), "plugins", "dgx-sampling.js")
    if args.profiles and per_model_sampling and not args.no_sampling_plugin:
        _def_mid = (agent_model_ref.split("/", 1)[1]
                    if agent_model_ref and "/" in agent_model_ref else agent_model_ref)
        plugin_js = oc_agent_sampling_plugin_js(per_model_sampling, _def_mid)
        write_plugin = True
    elif (not args.profiles and sampling["mode"] != "opencode-default"
          and not args.no_sampling_plugin):
        plugin_js = oc_sampling_plugin_js(sampling)
        write_plugin = True
    else:
        plugin_js, write_plugin = None, False

    # Loom tool plugin: written whenever the roster includes the loom agent (and the
    # conductor module is importable), so the `loom` custom tool exists server-side.
    loom_plugin_path = os.path.join(os.path.dirname(config_path), "plugins", "otools-loom.js")
    write_loom_plugin = bool(cfg.get("agent", {}).get("loom")) and loom_mod is not None
    loom_plugin_js = oc_loom_plugin_js() if write_loom_plugin else None

    print(f"\nDiscovered {len(refs)} model(s); removed {len(removed)} stale provider(s).")
    if args.profiles:
        rc = len(reasoning_caps)
        if not agents:
            print("Profiles mode: no models discovered")
        else:
            if not rc:
                print(f"Profiles mode: no reasoning models live -> roster on "
                      f"{agent_model_ref} (per default_models.json)")
            elif matched_recipe:
                print(f"Profiles mode: recipe match for {agent_model_ref}")
                print(f"  source: {matched_recipe.get('source','(recipe)')}")
            else:
                print(f"Profiles mode: {rc} reasoning model(s); no recipe -> generic Qwen numbers")
                print(f"  model: {agent_model_ref}")
            _print_roster(agents)
            for n in env_notes:
                print(f"  output: {n}")
    else:
        print(f"Sampling mode: {sampling['mode']}"
              + (" (+ chat.params plugin)" if write_plugin else ""))
    print(f"Tool calls: {'declared (tool_call=true) on all models' if not args.no_tool_call else 'left unset'}")
    if args.web_search != "none":
        print(f"Web search: {args.web_search}")
        for n in web_notes:
            print(f"  - {n}")
    if refs:
        print("OpenCode model references now available:")
        for r in refs:
            star = "  <- default" if cfg.get("model") == r else ""
            print(f"  - {r}{star}")

    if args.dry_run:
        print("\n--- DRY RUN: would write opencode.json ---")
        print(json.dumps(cfg, indent=2))
        if write_plugin:
            print(f"\n--- DRY RUN: would write {plugin_path} ---")
            print(plugin_js)
        if write_loom_plugin:
            print(f"\n--- DRY RUN: would write {loom_plugin_path} ---")
            print(loom_plugin_js)
        for agent_name, prompt_path in role_prompt_paths.items():
            print(f"\n--- DRY RUN: would write {prompt_path} ---")
            print(ROLE_PROMPTS[agent_name])
        for skill_name, skill_path in role_skill_paths.items():
            print(f"\n--- DRY RUN: would write {skill_path} ---")
            print(GLOBAL_SKILLS[skill_name])
        if args.profiles:
            print(f"\n--- DRY RUN: would regenerate loom instruction/contract files "
                  f"under {LOOM_SKILLS_DIR} for: {', '.join(LOOM_ROLES)} ---")
        return 0

    # Write config (+ one-shot backup)
    parent = os.path.dirname(config_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(config_path):
        try:
            with open(config_path) as a, open(config_path + ".bak", "w") as b:
                b.write(a.read())
        except OSError:
            pass
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"\nWrote {config_path}  (backup at {config_path}.bak)")

    # Clean up a plugin left in the old singular `plugin/` dir by earlier syncs,
    # so OpenCode doesn't ignore it (or load a stale copy) now that we use `plugins/`.
    legacy_plugin = os.path.join(os.path.dirname(config_path), "plugin", "dgx-sampling.js")
    if os.path.exists(legacy_plugin):
        try:
            os.remove(legacy_plugin)
            print(f"Removed stale {legacy_plugin} (moved to plugins/)")
        except OSError:
            pass

    if write_plugin:
        os.makedirs(os.path.dirname(plugin_path), exist_ok=True)
        with open(plugin_path, "w") as f:
            f.write(plugin_js)
        print(f"Wrote {plugin_path}")
    if write_loom_plugin:
        os.makedirs(os.path.dirname(loom_plugin_path), exist_ok=True)
        with open(loom_plugin_path, "w") as f:
            f.write(loom_plugin_js)
        print(f"Wrote {loom_plugin_path}  (loom tool)")
    elif os.path.exists(loom_plugin_path) and not cfg.get("agent", {}).get("loom"):
        try:
            os.remove(loom_plugin_path)
            print(f"Removed {loom_plugin_path} (no loom agent in roster)")
        except OSError:
            pass
    elif sampling["mode"] == "opencode-default" and os.path.exists(plugin_path):
        try:
            os.remove(plugin_path)
            print(f"Removed {plugin_path} (sampling=opencode-default)")
        except OSError:
            pass

    # Git identity plugin: coder token for all agents, reviewer token for agent-review.
    # Always written (independent of sampling/roster); reads the token files at runtime.
    identity_path = _git_identity_plugin_path(config_path)
    os.makedirs(os.path.dirname(identity_path), exist_ok=True)
    with open(identity_path, "w") as f:
        f.write(GIT_IDENTITY_PLUGIN_JS)
    print(f"Wrote {identity_path}")
    # If the coder token is set but its commit identity hasn't been resolved yet
    # (e.g. the token file was created by hand), resolve+cache it now (one API call).
    if os.path.exists(GH_TOKEN_CODER_FILE) and not os.path.exists(GH_CODER_IDENTITY_FILE):
        try:
            _write_coder_identity(open(GH_TOKEN_CODER_FILE).read().strip(), quiet=True)
        except OSError:
            pass

    for agent_name, prompt_path in role_prompt_paths.items():
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(ROLE_PROMPTS[agent_name])
        print(f"Wrote {prompt_path}")

    for skill_name, skill_path in role_skill_paths.items():
        os.makedirs(os.path.dirname(skill_path), exist_ok=True)
        with open(skill_path, "w") as f:
            f.write(GLOBAL_SKILLS[skill_name])
        print(f"Wrote {skill_path}")

    # loom-managed instruction/contract files: regenerated on EVERY sync so the
    # python constants stay the single source of truth. Repo-local overlays
    # (.loom/skills/*-local.md) are hand-authored and never touched.
    if args.profiles:
        loom_files = write_loom_role_files()
        print(f"Wrote {len(loom_files)} loom instruction/contract file(s) "
              f"under {LOOM_SKILLS_DIR}")

    # --allow-empty with zero models builds no roster, which used to skip these
    # writes entirely -- refresh the generated artifacts from current code anyway.
    if args.profiles and not role_prompt_paths:
        refreshed = oc_refresh_role_artifacts(config_path)
        if refreshed:
            print(f"Refreshed {len(refreshed)} generated artifact(s) "
                  "(global skills/prompts/loom files/plugin; no roster this sync)")

    # Remove artifacts of the retired layout: the per-role/team/instruct/runbook
    # skill directories and the retired prompt files. Project overlays are never
    # touched; agent-loom and session-review are kept.
    if args.profiles:
        for stale in oc_cleanup_stale_role_artifacts(config_path):
            print(f"Removed stale {stale}")

    _warn_missing_gh_tokens()
    print("Restart / reload OpenCode to pick up the changes.")
    return 0


def oc_load_config(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: could not read {path}: {e}", file=sys.stderr)
        sys.exit(1)


def oc_is_managed(key):
    return key in LEGACY_KEYS or key.startswith(PROVIDER_PREFIX)


# ============================================================================
# CLI
# ============================================================================
def build_sampling(args):
    out = {
        "mode": args.sampling,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
    }
    # repetition_detection: parse --repetition-detection VAL.
    #   None (flag absent)    -> DEFAULT_REPETITION_DETECTION
    #   "off"                  -> None (disabled)
    #   "K1:V1,K2:V2"          -> partial overrides MERGED onto the default, so tuning one
    #                             knob (e.g. "min_count:10") keeps the others -- rather than
    #                             dropping max_pattern_size and silently disabling detection.
    raw = getattr(args, "repetition_detection", None)
    if raw is None:
        out["repetition_detection"] = dict(DEFAULT_REPETITION_DETECTION)
    elif raw.lower() == "off":
        out["repetition_detection"] = None
    else:
        merged = dict(DEFAULT_REPETITION_DETECTION)
        for kv in raw.split(","):
            if ":" not in kv:
                print(f"  warning: ignoring malformed repetition_detection entry: {kv}",
                      file=sys.stderr)
                continue
            k, v = kv.split(":", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            merged[k.strip()] = v
        out["repetition_detection"] = merged
    return out


# ============================================================================
# Persisted settings (~/.config/otools/wire.json) -- flag > wire.json > default
# ============================================================================
WIRE_SETTINGS_FILE = os.path.expanduser("~/.config/otools/wire.json")

# key -> (one-line description, built-in fallback)
SETTINGS_KEYS = {
    "opencode_config": ("path to opencode.json", "~/.config/opencode/opencode.json"),
    "configs_dir":     ("omodel-manager configs/ dir", None),
    "hosts":           ("comma-separated host IPs to probe", None),
    "ports":           ("comma-separated ports", ",".join(map(str, DEFAULT_PORTS))),
    "default_agent":   ("startup agent (research/code/agent/loom)", "code"),
    "web_search":      ("none|exa|mcp", "none"),
    "proxy_port":      ("proxy listen port (default: 9099)", 9099),
    "proxy_active":    ("is proxy currently active?", False),
}
# Retired settings -- silently dropped from wire.json on load so a stale value can't
# linger and mislead (these belonged to the removed team orchestrator; per-agent model
# routing lives in default_models.json).
RETIRED_SETTINGS = ("team_model", "team_reasoning")


def load_settings():
    """Read wire.json; {} if absent or unparseable. Retired keys are dropped so a stale
    value (e.g. an old `team_model`) can't linger and silently drive behavior."""
    try:
        with open(WIRE_SETTINGS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {}
    except (OSError, json.JSONDecodeError):
        return {}
    if any(k in d for k in RETIRED_SETTINGS):
        for k in RETIRED_SETTINGS:
            d.pop(k, None)
        try:
            save_settings(d)   # rewrite without the retired keys
        except OSError:
            pass
    return d


def save_settings(d):
    os.makedirs(os.path.dirname(WIRE_SETTINGS_FILE), exist_ok=True)
    with open(WIRE_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


def _setting(args, key):
    """Resolve a setting: wire.json value, else the built-in fallback."""
    s = getattr(args, "_settings", {}) or {}
    if s.get(key) is not None:
        return s[key]
    return SETTINGS_KEYS[key][1]


def _suggest(items, header="Next steps"):
    """items: list of (description, command-string). Prints a breadcrumb block.
    Mirrors omodel-manager's _suggest so the two tools feel the same."""
    items = [it for it in items if it]
    if not items:
        return
    print(f"\n{header}:")
    for desc, cmd in items:
        print(f"  - {desc}")
        print(f"      {cmd}")


def _resolve_io(args):
    """Fill args.config / args.configs from flag > wire.json > default."""
    if not getattr(args, "config", None):
        args.config = _setting(args, "opencode_config")
    if not getattr(args, "configs", None):
        args.configs = _setting(args, "configs_dir")   # None -> _configs_dir() default
    return args


def _resolve_hosts_ports(args):
    """Fill args._hosts / args._ports from flag > wire.json > shared store/default."""
    hosts = (getattr(args, "hosts", None)
             or _setting(args, "hosts")
             or ",".join(load_shared_hosts() or DEFAULT_HOSTS))
    ports = getattr(args, "ports", None) or _setting(args, "ports")
    args._hosts = [h.strip() for h in hosts.split(",") if h.strip()]
    args._ports = [int(p) for p in ports.split(",") if p.strip()]
    return args


# ============================================================================
# Subcommands
# ============================================================================
def cmd_home(args):
    """No-subcommand landing screen: status + suggested next steps. No network."""
    cfg_path = os.path.expanduser(_setting(args, "opencode_config"))
    cfg = oc_load_config(cfg_path) if os.path.exists(cfg_path) else {}
    agents = cfg.get("agent", {}) or {}
    managed = [k for k, v in agents.items()
               if k in MANAGED_AGENTS and not (isinstance(v, dict) and v.get("disable"))]
    cdir = _configs_dir(_setting(args, "configs_dir"))
    ntoml = len([f for f in os.listdir(cdir) if f.endswith(".toml")]) if os.path.isdir(cdir) else 0
    hosts = _setting(args, "hosts") or ",".join(load_shared_hosts() or DEFAULT_HOSTS)
    loom = (agents.get("loom") or {}).get("model") or "(from default_models.json)"

    print("omodel-wire -- wire local model endpoints into OpenCode (omw)\n")
    print("Status:")
    cfg_note = "" if ntoml else "  [not found -- omw config --set configs_dir PATH]"
    print(f"  configs : {cdir}  ({ntoml} model config(s)){cfg_note}")
    print(f"  hosts   : {hosts}")
    dfl = f", default @{cfg.get('default_agent')}" if cfg.get("default_agent") else ""
    print(f"  opencode: {cfg_path}  ({len(managed)} managed agent(s){dfl})")
    print(f"  loom    : {loom}")
    print(f"  settings: {WIRE_SETTINGS_FILE}{'' if os.path.exists(WIRE_SETTINGS_FILE) else '  (none yet)'}")

    if not managed:
        items = [("Sync the OpenCode agent roster from the model configs", "omw sync")]
    else:
        items = [
            ("Review the agent roster (models, permissions)", "omw agents"),
            ("Review per-model sampling", "omw models"),
            ("Check for drift vs the known-good configs", "omw audit"),
            ("Re-sync to known-good presets", "omw sync"),
        ]
    _suggest(items, header="Suggested next steps")


def cmd_config(args):
    """Show or persist wire settings (~/.config/otools/wire.json)."""
    settings = args._settings
    if args.path:
        print(WIRE_SETTINGS_FILE)
        return
    if args.edit:
        if not os.path.exists(WIRE_SETTINGS_FILE):
            save_settings(settings)
        editor = os.environ.get("EDITOR", "nano")
        sys.exit(subprocess.run([editor, WIRE_SETTINGS_FILE]).returncode)
    handled_token = False
    for role in ("coder", "reviewer"):
        raw = getattr(args, f"set_gh_token_{role}", None)
        if raw is None:
            continue
        handled_token = True
        val = raw
        if val == "__PROMPT__":
            import getpass
            val = getpass.getpass(f"Paste the {role} GitHub token (input hidden): ")
        msg = _set_gh_token(role, val)
        if msg:
            print(msg)
    if handled_token:
        return
    if args.set:
        key, val = args.set
        if key not in SETTINGS_KEYS:
            print(f"unknown setting '{key}'. known: {', '.join(SETTINGS_KEYS)}", file=sys.stderr)
            sys.exit(1)
        if val.strip().lower() in ("", "none", "unset", "-", "default"):
            settings.pop(key, None)
            save_settings(settings)
            print(f"cleared {key} (back to default)")
        else:
            settings[key] = val
            save_settings(settings)
            print(f"set {key} = {val}")
        return
    # default: show resolved settings + source
    exists = os.path.exists(WIRE_SETTINGS_FILE)
    print(f"omodel-wire settings ({WIRE_SETTINGS_FILE}{'' if exists else '  -- not created yet'}):\n")
    for k, (desc, dflt) in SETTINGS_KEYS.items():
        if settings.get(k) is not None:
            print(f"  {k:16} {settings[k]}   ({desc})")
        else:
            print(f"  {k:16} (default: {dflt})   ({desc})")
    # GitHub identity (stored as token files, not in wire.json)
    print("\nGitHub identity (used by the otools-git-identity OpenCode plugin):")
    for role in ("coder", "reviewer"):
        state = "set" if os.path.exists(_gh_token_path(role)) else "unset"
        print(f"  gh_token_{role:8} {state}   ({_gh_token_path(role)})")
    _suggest([("Persist a value (VALUE 'none' to clear)", "omw config --set default_agent code"),
              ("Set the shared-bot GitHub token (prompts, hidden)", "omw config --set-gh-token-coder"),
              ("Set your reviewer GitHub token", "omw config --set-gh-token-reviewer")],
             header="Set")


def cmd_detect(args):
    detected = detect_tools()
    print_detection(detected)


def cmd_shell_init(args):
    install_aliases()


def cmd_audit(args):
    _resolve_io(args)
    sys.exit(oc_audit(args))


def cmd_verify(args):
    _resolve_io(args)
    _resolve_hosts_ports(args)
    sys.exit(oc_verify(args))


# ============================================================================
# GitHub Copilot CLI target
# ============================================================================
# Copilot's config home is ~/.copilot (override: $COPILOT_HOME). It reads custom
# agents from <home>/agents/<name>.agent.md (markdown + YAML frontmatter; the body
# is the agent's system prompt) and user settings from <home>/settings.json.
#
# KEY CONSTRAINT: the custom model ENDPOINT is environment-variable-only
# (COPILOT_PROVIDER_BASE_URL / _API_KEY / _TYPE) -- it can NOT be persisted in
# settings.json -- and the CLI takes a SINGLE custom provider. So Copilot runs the
# whole roster on ONE DGX model: we pick that model, write its name + behavior into
# settings.json, write the roster as .agent.md files, and emit an env snippet for the
# endpoint. Delegation in Copilot is runtime-global by description (no per-parent
# allowlist), and subagents cannot spawn subagents -- so primaries become top-level
# agents, workers become subagents, and agent-review is explicit-invoke-only.
# Docs: docs.github.com/en/copilot/{how-tos/copilot-cli/customize-copilot/use-byok-models,
#       reference/custom-agents-configuration, reference/copilot-cli-reference/cli-config-dir-reference}

def _is_wsl():
    """True when running under WSL (which looks like Linux but hosts Windows on /mnt/c)."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _wsl_windows_copilot_home():
    """Under WSL, the Windows-side ~/.copilot (Copilot installed on Windows, not in WSL), or None.

    Tries $USERPROFILE (if WSL interop exposes it), then /mnt/c/Users/<wsl-user>, then any single
    real user under /mnt/c/Users that already has a `.copilot`."""
    if not _is_wsl():
        return None
    cands = []
    up = os.environ.get("USERPROFILE")            # e.g. C:\Users\Otto
    if up and len(up) > 2 and up[1] == ":":
        cands.append(f"/mnt/{up[0].lower()}{up[2:].replace(chr(92), '/')}/.copilot")
    user = os.environ.get("USER")
    if user:
        cands.append(f"/mnt/c/Users/{user}/.copilot")
    try:
        for name in sorted(os.listdir("/mnt/c/Users")):
            if name.lower() in ("public", "default", "default user", "all users", "defaultuser0"):
                continue
            cands.append(f"/mnt/c/Users/{name}/.copilot")
    except OSError:
        pass
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def copilot_home():
    """Copilot CLI config home, auto-detected per platform.

    Precedence: $COPILOT_HOME > a native ~/.copilot that already exists (Copilot is installed
    right here -- correct on native Windows/macOS/Linux, where expanduser is already per-OS) >
    a Windows-side ~/.copilot when running under WSL (Copilot installed on Windows) > native
    ~/.copilot (created on first write)."""
    env = os.environ.get("COPILOT_HOME")
    if env:
        return env
    native = os.path.expanduser(os.path.join("~", ".copilot"))
    if os.path.isdir(native):
        return native
    return _wsl_windows_copilot_home() or native


def _copilot_home_for(args):
    return getattr(args, "copilot_home", None) or copilot_home()


# readonly agents are restricted to read/search/fetch; ask/full agents get all tools
# (omitting `tools` in the frontmatter means "all tools available").
COPILOT_READONLY_TOOLS = ["read", "search", "fetch"]


def _copilot_tools_for(perm):
    """A Copilot `tools:` list for a PERM tier, or None to mean 'all tools'."""
    return list(COPILOT_READONLY_TOOLS) if perm == "readonly" else None


def _yaml_frontmatter(fields):
    """Emit YAML frontmatter for the flat scalar/bool/list values we use (order preserved)."""
    lines = ["---"]
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, list):
            lines.append(f"{k}: [{', '.join(json.dumps(x) for x in v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _copilot_agent_md(name, description, body, tools=None, model=None, disable_model_invocation=None):
    """One Copilot custom-agent file: YAML frontmatter + markdown body (the system prompt)."""
    fm = _yaml_frontmatter({
        "name": name,
        "description": description,
        "tools": tools,
        "model": model,
        "disable-model-invocation": disable_model_invocation,
    })
    return f"{fm}\n\n{body.strip()}\n"


# Copilot-appropriate agent bodies. Concise and tool-agnostic (Copilot's runtime
# handles delegation), unlike the OpenCode prompts which reference the `task` tool.
COPILOT_BODIES = {
    "research": (
        "You are a research & reasoning agent. Investigate the codebase, read docs and the web, and "
        "reason problems through before any implementation. You are READ-ONLY: gather information and "
        "explain your findings; do not edit files or run commands with side effects."),
    "code": (
        "You are an interactive coding agent. Implement, refactor, and debug. Prefer small, verifiable "
        "changes, match the surrounding style, and confirm before destructive edits or commands with "
        "side effects."),
    "agent": (
        "You are an autonomous coding agent with full access. Take the task from start to finish: edit "
        "files, run commands, and VERIFY your work (build/tests) before reporting done."),
    "agent-research": (
        "You are a read-only research/data-gathering worker. Gather exactly the information asked "
        "for -- read files, search, fetch docs/web -- and return a concrete, self-contained summary "
        "with sources. Do not edit files or run commands with side effects."),
    "agent-code": (
        "You are the implementation worker. Inspect project instructions and surrounding code first; "
        "make the smallest correct change, preserve unrelated work, avoid secrets, and run focused "
        "checks for fast feedback. Leave broad suites/scripts to agent-test unless asked. Never commit "
        "or open a PR unless explicitly asked. Return files changed, exact results, and DONE, CONTINUE, "
        "NEEDS_RESEARCH, or BLOCKED."),
    "agent-test": (
        "You run broad verification after implementation: full suites, lint, typecheck, build commands, "
        "and scripted workflows. Never edit source or weaken tests. Report each command, PASS/FAIL, "
        "failing names, and useful output."),
    "agent-architect": (
        "You are read-only and do two things: produce an implementation plan, or request the research you "
        "need before planning. Return a bounded numbered plan, research requests, acceptance criteria, scope "
        "exclusions, and checks. Diagnose a blocked agent-code without expanding the original goal. "
        "You never review completed code -- agent-review owns that."),
    "agent-review": (
        "You are a read-only implementation and PR reviewer. Verify the caller's goal, acceptance criteria, "
        "scope, diff, and agent-test evidence; spot-check only missing or suspicious coverage. Classify findings "
        "as blocker, regression, pre-existing, future work, or out of scope; only blocker/regression require "
        "immediate fixes. Never edit. Approve or merge only when explicitly authorized and no blocking "
        "finding remains."),
}


# Copilot descriptions drive the runtime's auto-delegation, so they're purpose-written
# (not the OpenCode `sdesc`, which references OpenCode-isms like task_id).
COPILOT_DESCRIPTIONS = {
    "research": "Read-only research & reasoning: investigate the codebase, read docs and the web, and reason through problems before implementation.",
    "code": "Interactive coding: implement, refactor, and debug, confirming before destructive edits or side-effecting commands.",
    "agent": "Autonomous coding with full access: take a task end to end -- edit, run, and verify.",
    "agent-research": "Read-only research/data-gathering worker: fetch docs/web, gather information, return a concrete summary with sources; no edits.",
    "agent-code": "Full-access implementation worker: complete a change and run focused checks for fast feedback.",
    "agent-test": "Broad verification worker: runs full suites/scripts and reports exact pass/fail evidence; never edits source or tests.",
    "agent-architect": "Read-only planner and escalation target: plans hard changes or requests the research needed to plan, and diagnoses a blocked agent-code; no edits, never reviews completed work.",
    "agent-review": "Reviews tested implementations and PRs against supplied criteria; classifies findings and never edits.",
}


def copilot_build_agents():
    """{filename: agent-md content} for the full roster in Copilot's .agent.md format.

    Primaries (research/code/agent) are top-level agents; the agent-* workers are
    subagents. agent-review is `disable-model-invocation: true` (explicit-invoke-only) so
    it isn't auto-dispatched -- the closest Copilot has to a gated review step."""
    out = {}
    for key, prole, mode, is_worker, perm, color, sdesc in AGENT_SPECS:
        body = COPILOT_BODIES.get(key, sdesc)
        dmi = True if key == "agent-review" else None
        out[f"{key}.agent.md"] = _copilot_agent_md(
            name=key, description=COPILOT_DESCRIPTIONS.get(key, sdesc.replace("[worker] ", "")),
            body=body, tools=_copilot_tools_for(perm), disable_model_invocation=dmi)
    return out


def _copilot_pick_model(providers, available_models, reasoning_caps, default_models=None):
    """Pick the ONE model Copilot's single BYOK endpoint will serve.

    Prefers the coder the `code` agent would use (Copilot is a coding assistant), then any
    reasoning model, then the first live model. Returns (model_ref, base_url, served_id)."""
    if not available_models:
        return None, None, None
    refs = sorted(available_models)
    served = {ref: (ref.split("/", 1)[1] if "/" in ref else ref) for ref in refs}
    chosen = None
    if default_models is None:
        default_models = load_default_models()
    for pref in (default_models.get("agents") or {}).get("code", []):
        pl = pref.lower()
        for ref in refs:
            sid = served[ref].lower()
            if sid == pl or sid.rsplit("/", 1)[-1] == pl.rsplit("/", 1)[-1]:
                chosen = ref
                break
        if chosen:
            break
    if not chosen:
        chosen = sorted(reasoning_caps)[0] if reasoning_caps else refs[0]
    pk = chosen.split("/", 1)[0]
    base_url = ((providers.get(pk) or {}).get("options") or {}).get("baseURL", "")
    return chosen, base_url, served[chosen]


def _copilot_load_settings(path):
    """Load settings.json, tolerating //-line comments (Copilot writes a commented file)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return {}
    lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("//")]
    try:
        return json.loads("\n".join(lines)) or {}
    except ValueError:
        return {}


def copilot_merge_settings(settings, served_id=None):
    """Set the omw-managed keys in Copilot's settings.json (non-destructive to other keys).
    `model` is only set when a live model was found (served_id); the rest is model-independent."""
    settings["includeCoAuthoredBy"] = False   # matches the no-Co-Authored-By rule
    settings["stream"] = True
    if served_id:
        settings["model"] = served_id
    return settings


def copilot_env_files(base_url, served_id, api_key="dgx"):
    """(sh, ps1) env snippets that point Copilot's single BYOK provider at the DGX endpoint.
    The endpoint can't live in settings.json, so this is the one piece that must be env."""
    sh = ("# otools -- Copilot BYOK provider for your DGX endpoint (auto-generated by `omw sync`).\n"
          "# The custom endpoint can't be stored in settings.json, so source this before running\n"
          "# `copilot`, or add it to your shell rc (~/.bashrc, ~/.zshrc).\n"
          'export COPILOT_PROVIDER_TYPE="openai"\n'
          f'export COPILOT_PROVIDER_BASE_URL="{base_url}"\n'
          f'export COPILOT_PROVIDER_API_KEY="{api_key}"\n'
          f'export COPILOT_MODEL="{served_id}"\n')
    ps1 = ("# otools -- Copilot BYOK provider (PowerShell). Dot-source before `copilot`, or add to\n"
           "# your $PROFILE. For a persistent user env var use: setx COPILOT_PROVIDER_BASE_URL \"...\"\n"
           '$env:COPILOT_PROVIDER_TYPE = "openai"\n'
           f'$env:COPILOT_PROVIDER_BASE_URL = "{base_url}"\n'
           f'$env:COPILOT_PROVIDER_API_KEY = "{api_key}"\n'
           f'$env:COPILOT_MODEL = "{served_id}"\n')
    return sh, ps1


def _win_path(p):
    """Translate a /mnt/<drive>/... WSL path to a Windows path (C:\\...) for display; else return p."""
    mnt = re.match(r"^/mnt/([a-z])/(.*)$", p)
    return f"{mnt.group(1).upper()}:\\{mnt.group(2).replace('/', chr(92))}" if mnt else p


def _copilot_print_where_to_find(home, n):
    """Print where the synced agents show up -- all three Copilot surfaces read <home>/agents/."""
    print(f"\nYour {n} agents are in {os.path.join(home, 'agents')} -- all three Copilot surfaces read it:")
    print("  * VS Code:      Copilot Chat -> the agents dropdown at the bottom of the panel.")
    print("                  Reload first: Ctrl+Shift+P -> \"Developer: Reload Window\".")
    print("  * CLI / TUI:    run `copilot`, then type /agent  (or `copilot --agent code -p \"...\"`).")
    print("  * Desktop app:  the agent picker (shares this same ~/.copilot config).")


def _copilot_print_env_instructions(home, sh_path, ps1_path):
    win = home.startswith("/mnt/")   # WSL wrote to the Windows-side Copilot home
    print("\nOne more step -- point Copilot at the DGX endpoint (it can't live in settings.json):")
    if win:
        # Copilot runs on Windows; give the Windows path for its shell.
        print(f"  PowerShell (Windows):    . {_win_path(ps1_path)}")
        print(f"  (or from WSL if you run copilot there:  source {sh_path})")
    else:
        print(f"  bash/zsh (Linux/macOS):  source {sh_path}")
        print(f"  PowerShell (Windows):    . {ps1_path}")
    print("  Add that line to your shell rc / $PROFILE to persist it, then run")
    print("  `copilot --agent code`.")


def copilot_sync(args, sampling, detected_installed):
    """Write the GitHub Copilot CLI target: agent roster (.agent.md) + settings.json + the
    BYOK provider env snippet.

    The roster and the model-independent settings are ALWAYS written -- so you can sync (and see
    where the files land) even with the DGX offline. The provider endpoint (settings `model` +
    the env snippet) needs a live model, so it's wired only when one is discovered; otherwise the
    sync completes with a note to re-run when the DGX is reachable. Returns 0."""
    home = _copilot_home_for(args)
    configs = load_configs(args.configs) if not getattr(args, "no_recipes", False) else {"recipes": []}
    print(f"Probing {len(args._hosts)} host(s) x {len(args._ports)} port(s) for Copilot ...")
    providers, refs, reasoning_caps, available_models = oc_build_providers(
        args._hosts, args._ports, args.timeout, sampling, profiles=True,
        tool_call=not getattr(args, "no_tool_call", False), recipes=configs, verbose=True)
    default_models = (_load_default_models(create=False) if args.dry_run
                      else load_default_models())
    model_ref, base_url, served_id = _copilot_pick_model(
        providers, available_models, reasoning_caps, default_models)
    agents = copilot_build_agents()

    if model_ref:
        print(f"\nGitHub Copilot target -- the whole roster runs on ONE endpoint:")
        print(f"  model:    {model_ref}")
        print(f"  base URL: {base_url}")
    else:
        print("\nGitHub Copilot target -- no live model right now; writing the roster + settings "
              "only (the endpoint gets wired when the DGX is reachable).")

    if args.dry_run:
        print(f"\n--- DRY RUN: would write to {home} ---")
        print(f"  agents/  ({len(agents)}): {', '.join(sorted(agents))}")
        print(f"  settings.json (includeCoAuthoredBy=false, stream=true"
              + (f", model={served_id}" if served_id else "; model when a DGX is live") + ")")
        if base_url:
            print(f"  otools-copilot.env / .ps1  (COPILOT_PROVIDER_BASE_URL={base_url})")
        return 0

    # 1. Agent roster -- model-independent, always written.
    agents_dir = os.path.join(home, "agents")
    os.makedirs(agents_dir, exist_ok=True)
    for fn, content in sorted(agents.items()):
        with open(os.path.join(agents_dir, fn), "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Wrote {len(agents)} agent(s) to {agents_dir}")

    # 2. settings.json -- model-independent keys always; `model` only when a model is live.
    settings_path = os.path.join(home, "settings.json")
    settings = copilot_merge_settings(_copilot_load_settings(settings_path), served_id)
    os.makedirs(home, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    print(f"Wrote {settings_path}")

    _copilot_print_where_to_find(home, len(agents))

    # 3. Provider endpoint -- needs a live model/base URL.
    if not model_ref:
        print("\n  NOTE: the endpoint is NOT wired yet (no live model), so the agents will use")
        print("  Copilot's default hosted model. Re-run `omw sync --target copilot` with your DGX")
        print("  up to point them at your local model.")
        return 0
    sh, ps1 = copilot_env_files(base_url, served_id)
    sh_path = os.path.join(home, "otools-copilot.env")
    ps1_path = os.path.join(home, "otools-copilot.ps1")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    with open(ps1_path, "w", encoding="utf-8") as f:
        f.write(ps1)
    print(f"Wrote {sh_path} and {ps1_path}")
    _copilot_print_env_instructions(home, sh_path, ps1_path)
    return 0


def cmd_sync(args):
    """Full profile sync: roster + providers + plugin + prompts (was --profiles)."""
    _resolve_io(args)
    _resolve_hosts_ports(args)
    if not args.default_agent:
        args.default_agent = _setting(args, "default_agent")
    if not args.web_search:
        args.web_search = _setting(args, "web_search")
    args.profiles = True   # the roster is the whole point of `sync`

    sampling = build_sampling(args)
    detected = detect_tools()
    print_detection(detected)
    installed = {t["key"] for t in detected if t["installed"]}

    # Route to the requested target(s). --target opencode (default) preserves prior behavior;
    # copilot writes the Copilot CLI config; all does every implemented target.
    sync_funcs = {"opencode": oc_sync, "copilot": copilot_sync}
    target = getattr(args, "target", None) or "opencode"
    wanted = list(sync_funcs) if target == "all" else [target]
    ran_any = False
    for name in wanted:
        func = sync_funcs.get(name)
        if not func:
            continue
        ran_any = True
        tool = next((t for t in detected if t["sync"] == name), None)
        if tool and not tool["installed"]:
            print(f"{tool['display']} not found on PATH -- writing config anyway "
                  f"(install it, or point --config, to use it).")
        rc = func(args, sampling, installed)
        if rc not in (0,):
            sys.exit(rc)
    if not ran_any:
        print(f"No sync target named {target!r}. (Implemented: opencode, copilot.)")
        return
    if not args.dry_run:
        _suggest([
            ("Review the agent roster", "omw agents"),
            ("Review per-model sampling", "omw models"),
            ("Confirm live config matches the known-good configs", "omw audit"),
        ])


def _add_sync_args(p):
    """All sync-time knobs live under `omw sync` (moved off the top level).
    Settings-backed flags default to None so cmd_sync can resolve wire.json."""
    p.add_argument("--hosts", default=None,
                   help="comma-separated host IPs to probe "
                        "(default: wire.json hosts / shared ~/.config/otools/hosts / built-in)")
    p.add_argument("--ports", default=None,
                   help="comma-separated ports to probe on each host")
    p.add_argument("--set-default", metavar="REF",
                   help="set OpenCode top-level default model, e.g. dgx-n1-8000/qwen3-coder")
    p.add_argument("--timeout", type=float, default=PROBE_TIMEOUT)

    # Sampling control
    p.add_argument("--sampling", choices=["server-default", "fixed", "opencode-default"],
                   default="server-default",
                   help="server-default: server decides temp/topP/topK/penalties; "
                        "fixed: pin values via flags below; opencode-default: OpenCode's own.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--frequency-penalty", type=float, default=None)
    p.add_argument("--no-sampling-plugin", action="store_true",
                   help="don't write the chat.params plugin (only set temperature:false)")
    p.add_argument("--repetition-detection", default=None, metavar="VAL",
                   help="vLLM repetition_detection to terminate degenerate N-gram loops "
                        "(default lenient). 'off' to disable, or 'K:V,...' to override knobs.")

    # Tool calling + web search
    p.add_argument("--no-tool-call", action="store_true",
                   help="don't declare tool_call (OpenCode then sends NO tools to these models)")
    p.add_argument("--add-default-providers", action="store_true",
                   help="enable built-in OpenCode and Hugging Face providers (by default, they are disabled)")
    p.add_argument("--web-search", choices=["none", "exa", "mcp"], default=None,
                   help="expose a web-search tool. exa: keyless Exa; mcp: an MCP server "
                        "(default: wire.json web_search, else none)")
    p.add_argument("--enable-exa-shell", action="store_true",
                   help="(--web-search exa) append OPENCODE_ENABLE_EXA=1 to your shell rc")
    p.add_argument("--write-shell-env", action="store_true",
                   help="append needed OpenCode env vars (EXA, output-token max) to your shell rc")
    p.add_argument("--mcp-name", default="websearch")
    p.add_argument("--mcp-command", help="(--web-search mcp) stdio command")
    p.add_argument("--mcp-url", help="(--web-search mcp) remote MCP URL")
    p.add_argument("--mcp-env", action="append", metavar="KEY=VAL")
    p.add_argument("--mcp-header", action="append", metavar="KEY=VAL")

    # Roster
    p.add_argument("--no-reasoning-probe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--keep-builtins", action="store_true",
                   help="keep OpenCode's native build/plan agents (normally disabled)")
    p.add_argument("--default-agent", default=None,
                   help="startup agent when build/plan are disabled "
                        "(default: wire.json default_agent, else code)")
    p.add_argument("--no-recipes", action="store_true",
                   help="ignore the omodel-manager configs; yields generic behavior")
    p.add_argument("--dry-run", action="store_true", help="print result, do not write")
    p.add_argument("--allow-empty", action="store_true",
                   help="write even if NOTHING was discovered (default: refuse)")


# ============================================================================
# Read-only views: agents / subagents / models
# ============================================================================
def _table(headers, rows):
    """Print an aligned table (omm-style column widths). Rows are tuples of cells."""
    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    w = [max(len(str(c)) for c in col) for col in cols]
    print("  " + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  " + "  ".join(str(r[i]).ljust(w[i]) for i in range(len(headers))))


def _perm_label(perm):
    """Map a permission block back to its tier name (readonly/ask/full)."""
    if not isinstance(perm, dict):
        return "?"
    e, b = perm.get("edit"), perm.get("bash")
    if e == "deny" and b == "deny":
        return "readonly"
    if e == "ask" or b == "ask":
        return "ask"
    if e == "allow" and b == "allow":
        return "full"
    return "custom"


def _short_model(ref):
    """dgx-n1-8000/Qwen3.6-35B-A3B-NVFP4 -> Qwen3.6-35B-A3B-NVFP4 (keep provider for cloud)."""
    if not ref:
        return "-"
    return ref.split("/", 1)[1] if ref.startswith(PROVIDER_PREFIX) and "/" in ref else ref


def _model_id(ref):
    return ref.split("/", 1)[1] if ref and "/" in ref else (ref or "")


def _agent_vec(name, a, plugin):
    """Effective flattened sampling for one agent (opencode agent block + plugin vector)."""
    pv = (plugin or {}).get(_model_id(a.get("model", "")), {}).get(name)
    return _audit_vec(a, pv)


def _fmt(v):
    return "-" if v is None else ("on" if v is True else ("off" if v is False else str(v)))


def _load_live(args):
    """(cfg_path, cfg, agents, plugin) for the resolved opencode.json."""
    cfg_path = os.path.expanduser(args.config)
    cfg = oc_load_config(cfg_path) if os.path.exists(cfg_path) else {}
    return cfg_path, cfg, (cfg.get("agent", {}) or {}), (_plugin_agent_sampling(cfg_path) or {})


def _roster_view(args, want_subagent):
    """Shared implementation for `agents` (primaries) and `subagents` (workers)."""
    _resolve_io(args)
    cfg_path, cfg, agents, plugin = _load_live(args)
    label = "subagent" if want_subagent else "primary"
    if not agents:
        print(f"No agents found in {cfg_path}.")
        _suggest([("Sync the roster first", "omw sync")])
        return

    def _is(name, a):
        if not isinstance(a, dict) or a.get("disable"):
            return False
        if name not in MANAGED_AGENTS:
            return False
        mode = a.get("mode")
        return (mode == "subagent") if want_subagent else (mode in ("primary", "all"))

    picked = {k: v for k, v in agents.items() if _is(k, v)}
    if args.name:
        if args.name not in picked:
            print(f"{label} '{args.name}' not found. Available: {', '.join(picked) or '(none)'}",
                  file=sys.stderr)
            sys.exit(1)
        _show_agent_detail(args.name, picked[args.name], plugin)
        return

    rows = []
    for name, a in picked.items():
        vec = _agent_vec(name, a, plugin)
        # Show task_budget for agents that can delegate (code, agent)
        budget = a.get("task_budget", "-") if name in ("code", "agent") else "-"
        rows.append((name, _short_model(a.get("model")), _fmt(vec["temperature"]),
                     _fmt(vec["enable_thinking"]), _perm_label(a.get("permission")), budget))
    print(f"{'Sub-agents (delegation workers)' if want_subagent else 'Primary agents (Tab cycle)'} "
          f"in {cfg_path}:\n")
    _table(("AGENT", "MODEL", "TEMP", "THINK", "PERM", "BUDGET"), rows)
    nxt = [("Show one agent's full config", f"omw {'subagents' if want_subagent else 'agents'} <name>"),
           ("Review per-model sampling", "omw models")]
    _suggest(nxt)


def _show_agent_detail(name, a, plugin):
    vec = _agent_vec(name, a, plugin)
    print(f"agent: {name}\n")
    print(f"  model      : {a.get('model', '-')}")
    print(f"  mode       : {a.get('mode', '-')}")
    print(f"  permission : {_perm_label(a.get('permission'))}  ({a.get('permission')})")
    print("  effective sampling (opencode.json + plugin):")
    for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
              "max_output", "enable_thinking", "preserve_thinking"):
        print(f"    {k:18} {_fmt(vec.get(k))}")


def cmd_agents(args):
    if _has_roster_mutation(args):
        return _mutate_roster(args, want_subagent=False)
    _roster_view(args, want_subagent=False)


def cmd_subagents(args):
    if _has_roster_mutation(args):
        return _mutate_roster(args, want_subagent=True)
    _roster_view(args, want_subagent=True)


def _find_model_config(name, configs):
    """Lenient match of a user-typed model name to a declared config (either direction)."""
    n = (name or "").lower()
    for r in configs.get("recipes", []):
        pats = r.get("match") or []
        pats = [pats] if isinstance(pats, str) else pats
        if any(n in str(p).lower() or str(p).lower() in n for p in pats):
            return r
        if r.get("_file", "").lower().rsplit(".", 1)[0] == n:
            return r
    return None


ROLE_ORDER = ["reason", "code", "agent", "instruct"]


def cmd_models(args):
    _resolve_io(args)
    cfg_path, cfg, agents, plugin = _load_live(args)
    if getattr(args, "set_temperature", None) is not None or getattr(args, "set_thinking", None) is not None:
        return _mutate_model(args, cfg_path, cfg, agents, plugin)
    configs = load_configs(args.configs)
    live_ids = {mid for pv in (cfg.get("provider") or {}).values()
                for mid in (pv.get("models") or {})}

    if args.name:
        r = _find_model_config(args.name, configs)
        if not r:
            avail = ", ".join((rr.get("match") or ["?"])[0] for rr in configs.get("recipes", []))
            print(f"no model config matches '{args.name}'. Known: {avail}", file=sys.stderr)
            sys.exit(1)
        title = (r.get("match") or [args.name])[0]
        cap = r.get("capabilities", {}) or {}
        print(f"model: {title}   (config: {r.get('_file', '?')})")
        conc = cap.get("concurrency")
        conc_s = f", concurrency={conc}" if conc else ""
        print(f"  capabilities: reasoning={_fmt(bool(cap.get('reasoning')))}, "
              f"vision={_fmt(bool(cap.get('vision')))}, tool_call={_fmt(bool(cap.get('tool_call')))}, "
              f"thinking_control={cap.get('thinking_control', r.get('thinking_control', '-'))}{conc_s}\n")
        rows = []
        for role in ROLE_ORDER:
            ps = (r.get("presets") or {}).get(role)
            if not ps:
                continue
            s = ps.get("sampling", {}) or {}
            rows.append((role, _fmt(s.get("temperature")), _fmt(s.get("top_p")), _fmt(s.get("top_k")),
                         "on" if ps.get("thinking") else "off", _fmt(ps.get("max_output"))))
        print("  Per-role presets (the known-good sampling, from the config):\n")
        _table(("ROLE", "TEMP", "TOP_P", "TOP_K", "THINK", "MAX_OUT"), rows)
        # which live agents currently run on this model + their effective temp
        on_model = []
        for aname, a in agents.items():
            if not isinstance(a, dict) or a.get("disable"):
                continue
            if _find_model_config(_model_id(a.get("model", "")), {"recipes": [r]}) is r:
                on_model.append(f"{aname}({_fmt(_agent_vec(aname, a, plugin)['temperature'])})")
        if on_model:
            print(f"\n  live agents on this model: {', '.join(on_model)}")
        _suggest([("Tweak a role for testing (live-only; omw sync resets)",
                   f"omw models {title} --role code --set-temperature 0.5")])
        return

    # list
    # Full catalogue; LIVE/PROXY/SERVED reflect what's actually running now.
    # We iterate the DECLARED recipes (so --all shows the full catalogue, incl.
    # offline models), then emit ONE row per live served instance of each config
    # -- so the same config served on two endpoints shows as two rows. Configs
    # with no live instance collapse to a single row (served "-"), shown only
    # under --all. Capabilities always come from the config; never guessed.
    # One entry per live served instance -- (provider_key, served_id, is_proxied) -- so the
    # SAME model on two hosts shows as two rows with distinct, host-qualified refs.
    live_instances = []
    for pk, pv in (cfg.get("provider") or {}).items():
        prox = _is_loopback((pv.get("options") or {}).get("baseURL", ""))
        for mid in (pv.get("models") or {}):
            live_instances.append((pk, mid, prox))
    rows = []
    total = 0            # declared configs seen (drives the "N more not live" hint)
    live_configs = 0     # declared configs with at least one live instance
    show_all = getattr(args, "all", False)
    matched_keys = set()  # (provider, served_id) instances that matched some config
    for r in configs.get("recipes", []):
        total += 1
        cap = r.get("capabilities", {}) or {}
        pats = r.get("match") or []
        pats = [pats] if isinstance(pats, str) else pats
        matched = sorted((pk, mid, prox) for (pk, mid, prox) in live_instances
                         if any(str(p).lower() in mid.lower() for p in pats))
        matched_keys.update((pk, mid) for pk, mid, _ in matched)
        reason = _fmt(bool(cap.get("reasoning")))
        vision = _fmt(bool(cap.get("vision")))
        if matched:
            live_configs += 1
            # One row per served instance. MODEL is the full host-qualified ref you pass to
            # `--set-model`; SERVED is the bare served id (same model, different hosts differ).
            for pk, mid, prox in matched:
                proxy = "on" if prox else "off"
                rows.append((f"{pk}/{mid}", reason, vision, "yes", proxy, mid, r.get("_file", "")))
        elif show_all:
            title = (r.get("match") or ["?"])[0]
            rows.append((title, reason, vision, "-", "-", "-", r.get("_file", "")))
    # Live provider instances with NO declared config -- surfaced under --all only, so
    # they're not silently invisible, but without guessing their capabilities.
    if show_all:
        for pk, mid, prox in sorted(live_instances):
            if (pk, mid) in matched_keys:
                continue
            proxy = "on" if prox else "off"
            rows.append((f"{pk}/{mid}", "-", "-", "yes", proxy, mid, "(no config match)"))
    if not total:
        print("No model configs found.")
        _suggest([("Point at omodel-manager's configs", "omw config --set configs_dir PATH")])
        return
    if not rows:  # configs exist, but nothing live and no --all
        print("No models are live right now.")
        _suggest([("Show every declared model", "omw models --all")])
        return
    scope = "declared in the omodel-manager configs" if show_all else "live now"
    hidden = "" if show_all else f"  ({total - live_configs} more not live; --all to show)"
    print(f"Models {scope} (MODEL = the host-qualified ref for --set-model):{hidden}\n")
    _table(("MODEL", "REASON", "VISION", "LIVE", "PROXY", "SERVED", "CONFIG"), rows)
    _suggest([("Pin an agent to a model", "omw agents code --set-model dgx-<host>/<served-id>"),
              ("Show a model's per-role sampling", "omw models <served-id>"),
              ("List every declared model", "omw models --all")] if not show_all
             else [("Show a model's per-role sampling", "omw models <served-id>")])


def _find_model_config_live(recipe, live_ids):
    pats = recipe.get("match") or []
    pats = [pats] if isinstance(pats, str) else pats
    return any(any(str(p).lower() in mid.lower() for p in pats) for mid in live_ids)


# ============================================================================
# Live tweaks (--set-*): edit ONLY ~/.config/opencode/. `omw sync` resets.
# ============================================================================
# agent name -> preset role it runs (research=reason, code=code, ... loom=reason)
AGENT_ROLE = {spec[0]: spec[1] for spec in AGENT_SPECS}
AGENT_ROLE.setdefault("loom", "reason")

RESET_NOTE = "  (live edit only; run `omw sync` to reset to the known-good configs)"


def _boolish(s):
    v = str(s).strip().lower()
    if v in ("true", "on", "yes", "1"):
        return True
    if v in ("false", "off", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def _has_roster_mutation(args):
    return bool(getattr(args, "set_model", None))


# Two GitHub tokens (chmod 600), read by the identity plugin and checked by `omw sync`.
GH_TOKEN_CODER_FILE = os.path.expanduser("~/.config/otools/gh_token_coder")
GH_TOKEN_REVIEWER_FILE = os.path.expanduser("~/.config/otools/gh_token_reviewer")
# Resolved bot identity ({name,email}) so agent COMMITS read as the bot, not just pushes/PRs.
# Written when the coder token is set (one API call); the plugin only reads it (no net at runtime).
GH_CODER_IDENTITY_FILE = os.path.expanduser("~/.config/otools/gh_coder_identity")

# OpenCode plugin (shell.env hook) that gives agents a GitHub identity. Static —
# reads the two token files at runtime, so re-syncing never needs the tokens present.
GIT_IDENTITY_PLUGIN_JS = r'''// otools-git-identity -- auto-generated by `omw sync`; DO NOT EDIT (regenerated each sync).
//
// Gives OpenCode agents a GitHub identity via the shell.env hook, so there's no per-user
// git config to juggle:
//   * every agent shell gets GH_TOKEN = the CODER token -> commits & PRs are the coder
//     (bot) account. All coding agents (research/code/agent/agent-*) use this.
//   * GH_TOKEN_REVIEWER is also exposed; the `agent-review` subagent overrides GH_TOKEN
//     with it to review + MERGE as your own account -> real two-party review (GitHub lets
//     you approve a bot's PR).
//   * github.com git ops are routed over HTTPS + token per-shell (no SSH, no remote edits,
//     no ~/.gitconfig changes).
//
// Files (chmod 600), all under ~/.config/otools/ :
//   gh_token_coder       (shared bot account token)
//   gh_token_reviewer    (your account token)
//   gh_coder_identity    (JSON {name,email} for the coder -- so COMMITS read as the bot too;
//                         written by `omw config --set-gh-token-coder` / `omw sync`, no net here)
import { readFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";

const cfg = (name) => join(homedir(), ".config", "otools", name);
const tok = (name) => {
  try { return readFileSync(cfg(name), "utf8").trim(); }
  catch { return ""; }
};
const ident = () => {
  try { return JSON.parse(readFileSync(cfg("gh_coder_identity"), "utf8")); }
  catch { return null; }
};

export const OtoolsGitIdentity = async () => ({
  "shell.env": async (_input, output) => {
    const coder = tok("gh_token_coder");
    const reviewer = tok("gh_token_reviewer");
    if (coder) {
      output.env.GH_TOKEN = coder;          // default: agents act as the coder (bot) account
      output.env.GH_TOKEN_CODER = coder;
      // route github.com over HTTPS + token, per shell -- no SSH, no ~/.gitconfig edits.
      output.env.GIT_CONFIG_COUNT = "2";
      output.env.GIT_CONFIG_KEY_0 = "url.https://github.com/.insteadOf";
      output.env.GIT_CONFIG_VALUE_0 = "git@github.com:";
      output.env.GIT_CONFIG_KEY_1 = "credential.https://github.com.helper";
      output.env.GIT_CONFIG_VALUE_1 = '!f() { echo username=x; echo "password=$GH_TOKEN"; }; f';
      // author/committer -> the bot, so the commit itself (not just the push/PR) reads as the bot.
      const who = ident();
      if (who && who.name && who.email) {
        output.env.GIT_AUTHOR_NAME = who.name;
        output.env.GIT_AUTHOR_EMAIL = who.email;
        output.env.GIT_COMMITTER_NAME = who.name;
        output.env.GIT_COMMITTER_EMAIL = who.email;
      }
    }
    if (reviewer) output.env.GH_TOKEN_REVIEWER = reviewer;  // agent-review sets GH_TOKEN=$GH_TOKEN_REVIEWER
  },
});
'''


def _git_identity_plugin_path(cfg_path):
    return os.path.join(os.path.dirname(cfg_path), "plugins", "otools-git-identity.js")


def _missing_gh_token_roles():
    """Which of the two token files are absent (for the sync warning). Testable."""
    return [role for role, path in (("coder", GH_TOKEN_CODER_FILE),
                                    ("reviewer", GH_TOKEN_REVIEWER_FILE))
            if not os.path.exists(path)]


def _gh_token_path(role):
    return GH_TOKEN_CODER_FILE if role == "coder" else GH_TOKEN_REVIEWER_FILE


def _write_private_file(path, content):
    """Write `content` to `path` with 0600 perms from creation (never world-readable)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content.encode())
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _resolve_gh_identity(token):
    """Ask GitHub who a token belongs to. Returns {name,email} or None (best-effort, offline-safe).

    email is the account's attributable no-reply address (`<id>+<login>@users.noreply.github.com`),
    which links the commit to the account regardless of its email-privacy setting.
    """
    if not token:
        return None
    req = urllib.request.Request(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "otools-git-identity"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            u = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    login, uid = u.get("login"), u.get("id")
    if not login or uid is None:
        return None
    return {"name": login, "email": f"{uid}+{login}@users.noreply.github.com"}


def _write_coder_identity(token, quiet=False):
    """Resolve the coder token's identity and cache it to GH_CODER_IDENTITY_FILE. Returns it or None."""
    who = _resolve_gh_identity(token)
    if who:
        _write_private_file(GH_CODER_IDENTITY_FILE, json.dumps(who))
        if not quiet:
            print(f"   commits will be authored as {who['name']} <{who['email']}>")
    elif not quiet:
        print("   (couldn't reach GitHub to resolve the bot's commit identity — "
              "pushes/PRs still work; re-run `omw sync` when online to set it)")
    return who


def _set_gh_token(role, value):
    """Persist (or clear) the coder/reviewer GitHub token file with 0600 perms.

    For the coder, also resolve+cache the bot's commit identity (one API call) so agent
    COMMITS read as the bot too. Returns a human status string.
    """
    path = _gh_token_path(role)
    if value is None or value.strip().lower() in ("", "none", "unset", "-", "clear"):
        if os.path.exists(path):
            os.remove(path)
            if role == "coder" and os.path.exists(GH_CODER_IDENTITY_FILE):
                os.remove(GH_CODER_IDENTITY_FILE)
            return f"cleared {role} token ({path})"
        return f"{role} token already unset ({path})"
    token = value.strip()
    _write_private_file(path, token)
    if role == "coder":
        print(f"set coder token -> {path}  (reload OpenCode to apply)")
        _write_coder_identity(token)
        return None  # already printed
    return f"set {role} token -> {path}  (reload OpenCode to apply)"


def _warn_missing_gh_tokens():
    missing = _missing_gh_token_roles()
    if not missing:
        return
    print("\n⚠  GitHub identity not fully set — agents fall back to your logged-in gh account.")
    print("   Set the missing token(s), then reload OpenCode:")
    for role in missing:
        who = ("shared bot account -- every coding agent commits/opens PRs as this"
               if role == "coder" else
               "your account -- you + @agent-review review & merge as this")
        print(f"     omw config --set-gh-token-{role}   # {who} (prompts, input hidden)")


def _plugin_js_path(cfg_path):
    return os.path.join(os.path.dirname(cfg_path), "plugins", "dgx-sampling.js")


def _default_model_id(cfg):
    d = _model_id(cfg.get("model") or "")
    if d:
        return d
    for pv in (cfg.get("provider") or {}).values():
        for mid in (pv.get("models") or {}):
            return mid
    return ""


def _write_cfg(cfg_path, cfg):
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def _write_plugin(cfg_path, plugin, cfg):
    p = _plugin_js_path(cfg_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(oc_agent_sampling_plugin_js(plugin, _default_model_id(cfg)))


def _managed_instances(cfg):
    """[(provider_key, served_id), ...] over every model on a managed (dgx-) provider.

    One entry per served instance, so the SAME served id on two hosts yields two entries
    with distinct provider keys -- which is how we tell them apart and refuse to guess."""
    return [(pk, mid)
            for pk, pv in (cfg.get("provider") or {}).items()
            if str(pk).startswith(PROVIDER_PREFIX)
            for mid in (pv.get("models") or {})]


def _resolve_model_ref(ref, cfg, configs=None):
    """Resolve a model name to a full ``provider/served-id`` reference.

    Returns ``(resolved_ref, provider)``. When the name is AMBIGUOUS -- it maps to the same
    served model on more than one live host -- returns ``(None, [candidate_full_refs...])`` so
    the caller can list the choices instead of silently picking one (which used to produce a
    broken config). Returns ``(None, None)`` when the name can't be resolved at all.

    Accepted, in order: a full managed ref (``dgx-.../served-id``); a cloud provider ref
    (``openai/…``, ``anthropic/…`` -- OpenCode routes these directly); a whole served id,
    INCLUDING one that itself contains a slash such as ``unsloth/qwen3-coder-next-fp8``; a bare
    model name matched against served-id tails; finally an omodel-manager recipe match-pattern."""
    instances = _managed_instances(cfg)
    if "/" in ref:
        prov, mid = ref.split("/", 1)
        if (prov, mid) in instances:            # exact full managed ref
            return ref, prov
        if prov in REMOTE_PROVIDERS:            # cloud ref -- routed directly by OpenCode
            return ref, prov
    n = ref.lower()

    def pick(hits):
        if len(hits) == 1:
            pk, mid = hits[0]
            return f"{pk}/{mid}", pk
        if len(hits) > 1:
            return None, sorted(f"{pk}/{mid}" for pk, mid in hits)
        return None, None

    # whole ref == a served id (covers served ids that themselves contain a slash)
    hit = pick([(pk, mid) for pk, mid in instances if mid.lower() == n])
    if hit != (None, None):
        return hit
    # bare name == a served id's tail (the part after the last slash)
    hit = pick([(pk, mid) for pk, mid in instances if mid.rsplit("/", 1)[-1].lower() == n])
    if hit != (None, None):
        return hit
    # omodel-manager recipe match-pattern -> its target served id
    if configs:
        for recipe in configs.get("recipes", []):
            pats = recipe.get("match", [])
            pats = [pats] if isinstance(pats, str) else pats
            if not any(str(p).lower() == n for p in pats):
                continue
            target = n
            for p in pats:
                if "/" in str(p):
                    target = str(p).rsplit("/", 1)[-1].lower()
                    break
            hit = pick([(pk, mid) for pk, mid in instances
                        if mid.rsplit("/", 1)[-1].lower() == target])
            if hit != (None, None):
                return hit
    return None, None


def _validate_ref(ref, cfg):
    """Return a warning string if REF isn't a live managed provider/model (None if ok
    or a cloud ref like anthropic/...)."""
    if "/" not in ref:
        resolved, _ = _resolve_model_ref(ref, cfg)
        if resolved:
            return None
        return f"'{ref}' should look like provider/model-id"
    prov, mid = ref.split("/", 1)
    if prov.startswith(PROVIDER_PREFIX):
        models = ((cfg.get("provider") or {}).get(prov) or {}).get("models") or {}
        if mid not in models:
            return f"'{ref}' isn't among the live providers (setting anyway; sync it to use it)"
    return None


def _resolve_live_model(name, cfg, plugin):
    ids = set(plugin or {})
    for pv in (cfg.get("provider") or {}).values():
        ids |= set(pv.get("models") or {})
    n = (name or "").lower()
    exact = [i for i in ids if i.lower() == n]
    if exact:
        return exact[0], sorted(ids)
    sub = [i for i in ids if n and n in i.lower()]
    if len(sub) == 1:
        return sub[0], sorted(ids)
    return None, sorted(ids)


def _mutate_roster(args, want_subagent):
    _resolve_io(args)
    cfg_path, cfg, agents, plugin = _load_live(args)
    if not agents:
        print("No agents to edit. Run `omw sync` first.", file=sys.stderr)
        sys.exit(1)

    ref = args.set_model
    if getattr(args, "name", None):
        if args.name not in agents:
            print(f"agent '{args.name}' not found.", file=sys.stderr)
            sys.exit(1)
        targets = [args.name]
    elif want_subagent:
        targets = [k for k, a in agents.items()
                   if isinstance(a, dict) and not a.get("disable")
                   and a.get("mode") == "subagent" and k in MANAGED_AGENTS]
    else:
        print("name an agent: omw agents <name> --set-model REF", file=sys.stderr)
        sys.exit(1)
    
    # Load omodel-manager configs to support bare model name resolution
    configs = load_configs(getattr(args, "configs", None))
    resolved_ref, info = _resolve_model_ref(ref, cfg, configs)
    if resolved_ref:
        ref = resolved_ref
    elif isinstance(info, list):   # ambiguous -- the same model on multiple hosts
        print(f"'{ref}' matches more than one live model -- pass the full host-qualified ref:",
              file=sys.stderr)
        for r in info:
            print(f"    {r}", file=sys.stderr)
        print("(see `omw models` for the exact refs)", file=sys.stderr)
        sys.exit(1)
    else:
        warn = _validate_ref(ref, cfg)
        if warn:
            print(f"  note: {warn}")
    # Configure each target for the new model with its known-good config -- DGX recipe
    # preset (temperature/top_p/thinking + plugin sampling vector) or, on a frontier/cloud
    # model, OpenCode's own defaults (local vLLM knobs stripped). A plain model swap would
    # leave the OLD model's sampling/thinking on the agent, which is the whole thing this
    # tool exists to prevent.
    for nm in targets:
        _apply_model_config_to_agent(agents[nm], nm, ref, configs,
                                     repetition_detection=dict(DEFAULT_REPETITION_DETECTION),
                                     per_model_sampling=plugin)
    _write_cfg(cfg_path, cfg)
    if os.path.exists(_plugin_js_path(cfg_path)):
        _write_plugin(cfg_path, plugin, cfg)
    print(f"set model -> {ref}  for: {', '.join(targets)}")
    print(RESET_NOTE)
    _suggest([("See the change", "omw subagents" if want_subagent else "omw agents"),
              ("Reset to known-good", "omw sync")])


def _mutate_model(args, cfg_path, cfg, agents, plugin):
    if not getattr(args, "role", None):
        print("specify a role: --role reason|code|agent|instruct "
              "(see `omw models <name>`)", file=sys.stderr)
        sys.exit(1)
    mid, ids = _resolve_live_model(getattr(args, "name", None), cfg, plugin)
    if not mid:
        avail = ", ".join(ids) or "(none -- run `omw sync` first)"
        print(f"no single live model matches '{args.name}'. live models: {avail}", file=sys.stderr)
        sys.exit(1)
    targets = [nm for nm, a in agents.items()
               if isinstance(a, dict) and not a.get("disable")
               and _model_id(a.get("model", "")) == mid and AGENT_ROLE.get(nm) == args.role]
    if not targets:
        print(f"no live agent runs {mid} with role '{args.role}'.", file=sys.stderr)
        sys.exit(1)
    plugin_exists = os.path.exists(_plugin_js_path(cfg_path))
    for nm in targets:
        a = agents[nm]
        if args.set_temperature is not None:
            a["temperature"] = args.set_temperature
            plugin.setdefault(mid, {}).setdefault(nm, {})["temperature"] = args.set_temperature
        if args.set_thinking is not None:
            a.setdefault("options", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = args.set_thinking
    _write_cfg(cfg_path, cfg)
    if plugin_exists and args.set_temperature is not None:
        _write_plugin(cfg_path, plugin, cfg)
    what = []
    if args.set_temperature is not None:
        what.append(f"temperature={args.set_temperature}")
    if args.set_thinking is not None:
        what.append(f"thinking={args.set_thinking}")
    print(f"{mid} [{args.role}] -> {', '.join(what)}   (agents: {', '.join(targets)})")
    print(RESET_NOTE)
    _suggest([("See it", f"omw models {args.name}"),
              ("Check drift vs known-good", "omw audit"),
              ("Reset to known-good", "omw sync")])


# ============================================================================
# Skills views: list installed agent skills (global + project) and inspect one
# ============================================================================
SKILL_SIZE_THRESHOLDS = (40, 80)   # <=40 lean, 41-80 moderate, >80 LARGE


def _read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _skill_frontmatter(text):
    """Parse the leading `--- ... ---` YAML-ish block of a SKILL.md into a dict of the
    simple `key: value` lines (name, description, ...). {} if there's no frontmatter."""
    out = {}
    if not text.startswith("---"):
        return out
    end = text.find("\n---", 3)
    if end == -1:
        return out
    for ln in text[3:end].splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _skill_instruction_count(text):
    """Transparent size metric: number of normative lines -- bullets / numbered steps plus
    lines containing MUST/NEVER/ALWAYS/DO NOT/DON'T -- excluding frontmatter and fenced code."""
    body = text
    if body.startswith("---"):
        e = body.find("\n---", 3)
        if e != -1:
            body = body[e + 4:]
    imperative = re.compile(r"\b(MUST|NEVER|ALWAYS|DO NOT|DON'T)\b")
    bullet = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)")
    n, in_fence = 0, False
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not s:
            continue
        if bullet.match(ln) or imperative.search(ln):
            n += 1
    return n


def _skill_size_verdict(n):
    lean, moderate = SKILL_SIZE_THRESHOLDS
    if n <= lean:
        return "lean"
    if n <= moderate:
        return "moderate"
    return "LARGE"


def _skill_roots(args):
    """[(scope, skills_root), ...] -- the OpenCode GLOBAL skills dir (next to opencode.json)
    then the PROJECT skills dir (./.agents/skills), matching where the roster looks."""
    roots = []
    cfg = os.path.expanduser(args.config) if getattr(args, "config", None) else ""
    if cfg:
        roots.append(("global", os.path.join(os.path.dirname(cfg), "skills")))
    roots.append(("project", os.path.join(os.getcwd(), ".agents", "skills")))
    return roots


def _find_skills(args):
    """{name: [(scope, skill_dir, skill_md), ...]} across the global + project roots."""
    found = {}
    for scope, root in _skill_roots(args):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            md = os.path.join(root, name, "SKILL.md")
            if os.path.isfile(md):
                found.setdefault(name, []).append((scope, os.path.join(root, name), md))
    return found


def _skill_md_files(skill_dir):
    """All .md files in a skill dir, SKILL.md first, then the rest alphabetically."""
    files = [os.path.join(skill_dir, f) for f in os.listdir(skill_dir) if f.endswith(".md")]
    files.sort(key=lambda p: (os.path.basename(p) != "SKILL.md", os.path.basename(p).lower()))
    return files


def _show_skill_detail(name, entries):
    """Pretty-print every .md file of a skill (each scope it appears in)."""
    for scope, skill_dir, md in entries:
        text = _read_text(md)
        n = _skill_instruction_count(text)
        fm = _skill_frontmatter(text)
        print("=" * 78)
        print(f"skill: {name}   [{scope}]")
        print(f"  path: {skill_dir}")
        print(f"  size: {n} instruction(s) -> {_skill_size_verdict(n)}")
        if fm.get("description"):
            print(f"  description: {fm['description']}")
        print("=" * 78)
        for f in _skill_md_files(skill_dir):
            print(f"\n{'-' * 78}\n# {os.path.basename(f)}\n{'-' * 78}\n")
            print(_read_text(f).rstrip())
        print()


def cmd_skills(args):
    """List agent skills (global + project); `omw skills <name>` pretty-prints one."""
    _resolve_io(args)
    skills = _find_skills(args)
    if getattr(args, "name", None):
        if args.name not in skills:
            print(f"skill '{args.name}' not found. Available: "
                  f"{', '.join(sorted(skills)) or '(none)'}", file=sys.stderr)
            sys.exit(1)
        _show_skill_detail(args.name, skills[args.name])
        return
    if not skills:
        print("No skills found (looked in the OpenCode global skills/ and ./.agents/skills/).")
        _suggest([("Sync the roster (writes the global skills)", "omw sync")])
        return
    rows = []
    for name in sorted(skills):
        for scope, skill_dir, md in skills[name]:
            text = _read_text(md)
            n = _skill_instruction_count(text)
            desc = (_skill_frontmatter(text).get("description", "") or "")
            rows.append((name, scope, f"{n} ({_skill_size_verdict(n)})",
                         desc[:58] + ("\u2026" if len(desc) > 58 else "")))
    print("Agent skills (OpenCode global + project ./.agents/skills):\n")
    _table(("SKILL", "SCOPE", "SIZE", "DESCRIPTION"), rows)
    _suggest([("Inspect a skill (pretty-print all its .md files)", "omw skills <name>"),
              ("Re-sync the global skills", "omw sync")])


def _build_parser():
    ap = argparse.ArgumentParser(
        prog="omodel-wire",
        description="Wire local/OpenAI-compatible model endpoints into OpenCode (omw).")
    ap.add_argument("--version", action="version", version=f"omodel-wire {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")

    io_parent = argparse.ArgumentParser(add_help=False)
    io_parent.add_argument("--config", default=None,
                           help="path to opencode.json (default: wire.json / ~/.config/opencode/opencode.json)")
    io_parent.add_argument("--configs", metavar="PATH", default=None,
                           help="omodel-manager configs dir (default: wire.json / $OMODEL_CONFIGS / sibling)")

    ps = sub.add_parser("sync", parents=[io_parent],
                        help="sync the agent roster from the model configs (OpenCode and/or Copilot)")
    _add_sync_args(ps)
    ps.add_argument("--target", choices=["opencode", "copilot", "all"], default="opencode",
                    help="which tool to configure (default: opencode)")
    ps.add_argument("--copilot-home", default=None,
                    help="override Copilot's config home (default: $COPILOT_HOME or ~/.copilot)")
    ps.set_defaults(func=cmd_sync)

    pa = sub.add_parser("audit", parents=[io_parent],
                        help="compare live OpenCode sampling vs declared configs (offline drift check)")
    pa.set_defaults(func=cmd_audit)

    pv = sub.add_parser("verify", parents=[io_parent],
                        help="probe live endpoints vs declared capabilities (slow, opt-in)")
    pv.add_argument("--hosts", default=None)
    pv.add_argument("--ports", default=None)
    pv.add_argument("--timeout", type=float, default=PROBE_TIMEOUT)
    pv.add_argument("--no-vision-probe", action="store_true")
    pv.add_argument("--vision-probe-all", action="store_true")
    pv.set_defaults(func=cmd_verify)

    pc = sub.add_parser("config", help="show or persist wire settings (wire.json)")
    pc.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="persist one setting")
    pc.add_argument("--edit", action="store_true", help="open wire.json in $EDITOR")
    pc.add_argument("--path", action="store_true", help="print the settings file path")
    pc.add_argument("--set-gh-token-coder", nargs="?", const="__PROMPT__", default=None,
                    metavar="TOKEN", dest="set_gh_token_coder",
                    help="store the shared-bot GitHub token (coding agents commit/PR as this); "
                         "omit TOKEN to be prompted with hidden input, pass 'none' to clear")
    pc.add_argument("--set-gh-token-reviewer", nargs="?", const="__PROMPT__", default=None,
                    metavar="TOKEN", dest="set_gh_token_reviewer",
                    help="store your GitHub token (agent-review approves/merges the bot's PRs as "
                         "this); omit TOKEN to be prompted, pass 'none' to clear")
    pc.set_defaults(func=cmd_config)

    pag = sub.add_parser("agents", parents=[io_parent],
                         help="list primary agents; show/tweak one (`omw agents code`)")
    pag.add_argument("name", nargs="?", help="agent name to show or edit")
    pag.add_argument("--set-model", metavar="REF", help="live-set this agent's model")
    pag.set_defaults(func=cmd_agents)

    psa = sub.add_parser("subagents", parents=[io_parent],
                         help="list hidden workers; show/tweak (no name = all workers)")
    psa.add_argument("name", nargs="?", help="worker name to show or edit")
    psa.add_argument("--set-model", metavar="REF",
                     help="live-set model (no name -> all workers)")
    psa.set_defaults(func=cmd_subagents)

    pm = sub.add_parser("models", parents=[io_parent],
                        help="list models; show/tweak per-role sampling (`omw models qwen`)")
    pm.add_argument("name", nargs="?", help="model name to show or edit")
    pm.add_argument("--all", action="store_true",
                    help="list every declared model, not just the live ones")
    pm.add_argument("--role", choices=ROLE_ORDER, help="which role to edit (with --set-*)")
    pm.add_argument("--set-temperature", type=float, metavar="T", help="live-set temperature")
    pm.add_argument("--set-thinking", type=_boolish, metavar="BOOL",
                    help="live-set thinking on/off (true|false)")
    pm.set_defaults(func=cmd_models)

    psk = sub.add_parser("skills", parents=[io_parent],
                         help="list agent skills (global + project); show one with `omw skills <name>`")
    psk.add_argument("name", nargs="?", help="skill to pretty-print (all its .md files)")
    psk.set_defaults(func=cmd_skills)

    pd = sub.add_parser("detect", aliases=["doctor"],
                        help="report which agentic-dev tools are installed")
    pd.set_defaults(func=cmd_detect)

    psi = sub.add_parser("shell-init", aliases=["install-aliases"],
                         help="install the `omw` shell alias")
    psi.set_defaults(func=cmd_shell_init)

    if loom_mod is not None:
        pl = loom_mod.add_parser(sub)
        pl.set_defaults(func=cmd_loom)

    pp = sub.add_parser("proxy", parents=[io_parent],
                        help="debug proxy: log OpenCode<->model traffic (on|off|replay|read|status)")
    pp.add_argument("action", choices=["on", "off", "replay", "read", "status"],
                    help="on/off [model] | replay <id> | read <id> | status")
    pp.add_argument("target", nargs="?",
                    help="model name (on/off) or request_id (replay/read)")
    pp.add_argument("--port", type=int, default=None,
                    help="proxy port (default: wire.json proxy_port, else 9099)")
    pp.add_argument("--output-curl", action="store_true",
                    help="(replay) print a copy-pasteable curl instead of running the request")
    pp.add_argument("--no-color", action="store_true", help="(read) disable ANSI colors")
    pp.set_defaults(func=cmd_proxy)

    pc = sub.add_parser("cleanup", parents=[io_parent],
                        help="delete old OpenCode sessions (messages/parts cascade; "
                             "then VACUUM)")
    pc.add_argument("--keep", type=int, default=0,
                    help="retain the newest N root sessions with their workers "
                         "(default 0: delete ALL sessions)")
    pc.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted, delete nothing")
    pc.add_argument("--force", action="store_true",
                    help="run even while an opencode process is alive")
    pc.add_argument("--db", help="opencode.db path (default: the standard one)")
    pc.set_defaults(func=cmd_cleanup)

    return ap


# ============================================================================
# Proxy commands (debug proxy: log OpenCode <-> model traffic)
# ============================================================================
def _proxy_paths(args):
    config_path = os.path.expanduser(args.config)
    d = os.path.dirname(config_path)
    port = getattr(args, "port", None) or _setting(args, "proxy_port") or 9099
    return {
        "config_path": config_path,
        "pid": os.path.join(d, ".omw-proxy.pid"),
        "routes": os.path.join(d, "proxy_routes.json"),
        "backup": config_path + ".proxy-bak",
        "logs": os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy_logs"),
        "port": int(port),
    }


def _is_loopback(url):
    try:
        host = urllib.parse.urlsplit(url or "").hostname or ""
    except (ValueError, AttributeError):
        return False
    return host in ("127.0.0.1", "localhost", "::1")


def _is_proxied(url, port):
    """True when `url` points at the debug proxy itself. Loopback alone is NOT
    proof of proxying -- on a single box the real upstream (vllm/llama.cpp) is
    loopback too; the distinguishing mark is the proxy's own port."""
    try:
        sp = urllib.parse.urlsplit(url or "")
    except (ValueError, AttributeError):
        return False
    return _is_loopback(url) and sp.port == int(port)


def _read_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _read_pid(pid_file):
    try:
        with open(pid_file) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _proxy_pid(P):
    """Return the live daemon pid, or None (stale pid files read as not-running)."""
    pid = _read_pid(P["pid"])
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def _providers_for_target(cfg, target):
    """Managed provider keys hosting model `target` (bare name ok); all dgx- if None."""
    providers = cfg.get("provider") or {}
    managed = [k for k in providers if k.startswith(PROVIDER_PREFIX)]
    if not target:
        return managed
    mid, _ids = _resolve_live_model(target, cfg, {})
    if not mid:
        return []
    return [k for k in managed if mid in ((providers[k].get("models") or {}))]


def _loom_role_models(wire_settings):
    """{worker: 'provider/model'} for the pipeline roles, read from the CURRENT
    opencode.json agent entries (same config resolution as sync: wire.json override,
    else XDG_CONFIG_HOME, else the built-in default path). {} when the config is
    missing or unreadable -- the conductor then falls back to its own defaults.

    XDG_CONFIG_HOME matters for isolated harness instances: a serve launched with
    a cloned config dir spawns this conductor with that env, and worker dispatches
    must route by the CLONE's agent models, not the machine-global config.
    (Observed 2026-07-26: parallel benchmark runs all dispatched workers on the
    global config's model despite per-instance clones.)"""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    default = (os.path.join(xdg, "opencode", "opencode.json") if xdg
               else SETTINGS_KEYS["opencode_config"][1])
    path = os.path.expanduser(wire_settings.get("opencode_config") or default)
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    out = {}
    for role in LOOM_ROLES:
        entry = (cfg.get("agent") or {}).get(role)
        if isinstance(entry, dict) and entry.get("model"):
            out[role] = entry["model"]
    return out


def cmd_loom(args):
    """Dispatch `omw loom ...` into the conductor module (utils/omw_loom.py)."""
    if loom_mod is None:
        print("loom: utils/omw_loom.py is missing or unloadable", file=sys.stderr)
        return 2
    wire_settings = load_settings()
    return loom_mod.dispatch(args, wire_settings=wire_settings,
                             default_models_path=DEFAULT_MODELS_FILE,
                             role_models=_loom_role_models(wire_settings),
                             loom_dir=LOOM_SKILLS_DIR,
                             status_contract=WORKER_STATUS_CONTRACT)


def cmd_proxy(args):
    _resolve_io(args)
    if proxy is None:
        print("proxy helper (utils/omw_proxy.py) not found.", file=sys.stderr)
        sys.exit(1)
    return {"on": cmd_proxy_on, "off": cmd_proxy_off, "replay": cmd_proxy_replay,
            "read": cmd_proxy_read, "status": cmd_proxy_status}[args.action](args)


def _ensure_proxy_running(P):
    """Launch the proxy daemon if not already up. Returns True if it started it."""
    if _proxy_pid(P):
        return False
    os.makedirs(P["logs"], exist_ok=True)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "omw_proxy.py")
    env = os.environ.copy()
    env["OMW_PROXY_LOGS_DIR"] = P["logs"]
    # Redirect to a file (NOT a PIPE -- an unread PIPE fills and stalls the daemon).
    out = open(os.path.join(P["logs"], "proxy.log"), "a", encoding="utf-8")
    kw = {"start_new_session": True} if os.name == "posix" else {}
    proc = subprocess.Popen(
        [sys.executable, script, "--port", str(P["port"]),
         "--logs-dir", P["logs"], "--routes", P["routes"]],
        stdout=out, stderr=subprocess.STDOUT, env=env, **kw)
    with open(P["pid"], "w") as f:
        f.write(str(proc.pid))
    return True


def cmd_proxy_on(args):
    P = _proxy_paths(args)
    cfg = oc_load_config(P["config_path"])
    providers = cfg.get("provider") or {}
    targets = _providers_for_target(cfg, args.target)
    if not targets:
        if args.target:
            print(f"no live managed model matches '{args.target}'. See `omw models`.", file=sys.stderr)
        else:
            print("no managed (dgx-) providers in opencode.json. Run `omw sync` first.", file=sys.stderr)
        sys.exit(1)

    routes = _read_json_file(P["routes"]) or {}
    if not os.path.exists(P["backup"]) and os.path.exists(P["config_path"]):
        shutil.copy2(P["config_path"], P["backup"])

    changed = []
    for key in targets:
        opts = providers[key].get("options") or {}
        cur = opts.get("baseURL")
        if not cur or _is_proxied(cur, P["port"]):
            continue
        routes[key] = cur                       # remember the real upstream
        opts["baseURL"] = f"http://127.0.0.1:{P['port']}/{key}"
        providers[key]["options"] = opts
        changed.append(key)

    _dump_json(P["routes"], routes)
    _write_cfg(P["config_path"], cfg)
    settings = load_settings()
    settings["proxy_port"] = P["port"]
    settings["proxy_active"] = True
    save_settings(settings)

    started = _ensure_proxy_running(P)
    print(f"proxy ON for: {', '.join(changed)}" if changed
          else f"already proxied: {', '.join(routes)}")
    print(f"  daemon: {'started' if started else 'already running'} on 127.0.0.1:{P['port']} "
          f"(pid {_read_pid(P['pid'])})")
    print(f"  logs  : {P['logs']}")
    print("  reload OpenCode to route through the proxy.")
    _suggest([("Read a logged exchange (id shown by the proxy / in proxy_logs/index.jsonl)",
               "omw proxy read <id>"),
              ("Turn the proxy off", "omw proxy off")])


def cmd_proxy_off(args):
    P = _proxy_paths(args)
    cfg = oc_load_config(P["config_path"])
    providers = cfg.get("provider") or {}
    routes = _read_json_file(P["routes"]) or {}
    if not routes:
        print("proxy is not on (no proxy_routes.json).")
        _stop_proxy(P)   # tidy any stray daemon/pid
        return

    if args.target:
        want = set(_providers_for_target(cfg, args.target))
        keys = [k for k in list(routes) if k in want]
        if not keys:
            print(f"'{args.target}' is not currently proxied. Proxied: {', '.join(routes)}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        keys = list(routes)

    for k in keys:
        if k in providers and k in routes:
            (providers[k].setdefault("options", {}))["baseURL"] = routes[k]
        routes.pop(k, None)
    _write_cfg(P["config_path"], cfg)
    print(f"proxy OFF for: {', '.join(keys)}")

    if routes:
        _dump_json(P["routes"], routes)
        print(f"  still proxied: {', '.join(routes)}")
    else:
        _stop_proxy(P)
        for f in (P["routes"], P["backup"]):
            if os.path.exists(f):
                os.remove(f)
        settings = load_settings()
        settings["proxy_active"] = False
        save_settings(settings)
        print("  all models unproxied; daemon stopped, config restored.")
    print("  reload OpenCode to pick up the change.")


def _stop_proxy(P):
    pid = _read_pid(P["pid"])
    if pid is not None:
        try:
            os.kill(pid, 15)   # SIGTERM (Windows: terminates regardless of sig)
        except OSError:
            pass
    if os.path.exists(P["pid"]):
        os.remove(P["pid"])


def cmd_proxy_status(args):
    P = _proxy_paths(args)
    cfg = oc_load_config(P["config_path"])
    providers = cfg.get("provider") or {}
    pid = _proxy_pid(P)
    print(f"daemon : {('running (pid ' + str(pid) + ')') if pid else 'not running'} "
          f"on 127.0.0.1:{P['port']}")
    print(f"logs   : {P['logs']}")
    proxied = [k for k, v in providers.items()
               if _is_proxied((v.get("options") or {}).get("baseURL", ""), P["port"])]
    print(f"proxied: {', '.join(proxied) or '(none)'}")
    if not pid and proxied:
        _suggest([("Restart the daemon", "omw proxy on")])


def cmd_proxy_replay(args):
    P = _proxy_paths(args)
    rid = args.target
    if not rid:
        print("usage: omw proxy replay <request_id> [--output-curl]", file=sys.stderr)
        sys.exit(1)
    req, _res = proxy.find_pair(P["logs"], rid)
    if not req:
        print(f"request '{rid}' not found in {P['logs']}", file=sys.stderr)
        sys.exit(1)
    if args.output_curl:
        print(proxy.build_curl(req))
        return
    method = req.get("method", "GET")
    url = req.get("url", "")
    headers = {k: v for k, v in (req.get("headers") or {}).items()
               if k.lower() not in ("host", "content-length", "accept-encoding", "connection")}
    body = req.get("body") or ""
    print(f"replaying {rid}: {method} {url}\n")
    try:
        r = urllib.request.Request(url, data=body.encode("utf-8") if body else None,
                                   headers=headers, method=method)
        with urllib.request.urlopen(r, timeout=120) as resp:
            status, out = resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, out = e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"replay failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"-> {status}\n{out}")


def cmd_proxy_read(args):
    P = _proxy_paths(args)
    rid = args.target
    if not rid:
        print("usage: omw proxy read <request_id>", file=sys.stderr)
        sys.exit(1)
    req, res = proxy.find_pair(P["logs"], rid)
    if not req:
        print(f"request '{rid}' not found in {P['logs']}", file=sys.stderr)
        sys.exit(1)
    use_color = (sys.stdout.isatty() and not getattr(args, "no_color", False)
                 and not os.environ.get("NO_COLOR"))
    print(proxy.render_read(req, res, use_color=use_color))


# ============================================================================
# Main entry point
# ============================================================================
def _opencode_db_path():
    return os.path.expanduser("~/.local/share/opencode/opencode.db")


def _opencode_running():
    try:
        out = subprocess.run(["pgrep", "-x", "opencode"], capture_output=True)
        return out.returncode == 0
    except OSError:
        return False  # no pgrep (non-Linux dev box): skip the guard


def cmd_cleanup(args):
    """Delete old OpenCode sessions. --keep N retains the newest N root
    sessions (each with ALL its worker/child sessions); default 0 deletes
    everything. FK cascade removes messages, parts, and session_* aux rows;
    VACUUM reclaims the disk afterwards."""
    import sqlite3

    db = getattr(args, "db", None) or _opencode_db_path()
    if not os.path.exists(db):
        print(f"cleanup: no OpenCode db at {db}")
        return
    if not getattr(args, "force", False) and _opencode_running():
        print("cleanup: an opencode process is running -- close it first "
              "(or pass --force if you are sure nothing is mid-write).")
        raise SystemExit(2)

    before = os.path.getsize(db)
    con = sqlite3.connect(db)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        total, = con.execute("SELECT count(*) FROM session").fetchone()
        keep_n = max(0, getattr(args, "keep", 0) or 0)
        roots = [r[0] for r in con.execute(
            "SELECT id FROM session WHERE parent_id IS NULL "
            "ORDER BY time_updated DESC")]
        keep_roots = roots[:keep_n]
        if keep_roots:
            marks = ",".join("?" * len(keep_roots))
            kept = [r[0] for r in con.execute(
                "WITH RECURSIVE kept(id) AS ("
                f"  SELECT id FROM session WHERE id IN ({marks})"
                "  UNION"
                "  SELECT s.id FROM session s JOIN kept k ON s.parent_id = k.id)"
                "SELECT id FROM kept", keep_roots)]
        else:
            kept = []
        if getattr(args, "dry_run", False):
            print(f"cleanup (dry run): would delete {total - len(kept)} of "
                  f"{total} sessions, keeping {len(keep_roots)} root(s) "
                  f"({len(kept)} sessions incl. workers).")
            return
        if kept:
            marks = ",".join("?" * len(kept))
            cur = con.execute(
                f"DELETE FROM session WHERE id NOT IN ({marks})", kept)
        else:
            cur = con.execute("DELETE FROM session")
        deleted = cur.rowcount
        con.commit()
        con.execute("VACUUM")
        con.commit()
    finally:
        con.close()
    after = os.path.getsize(db)
    print(f"cleanup: deleted {deleted} of {total} sessions "
          f"(kept {len(keep_roots)} root(s), {len(kept)} total incl. workers); "
          f"db {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")


def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)
    args._settings = load_settings()
    if not args.cmd:
        cmd_home(args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
