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
  omw --profiles --team-model anthropic/claude-opus-4-8 --team-reasoning high \
       --team-task-budget 4 --web-search exa --write-shell-env
  omw --dry-run                       # preview opencode.json + plugin, write nothing

--profiles builds, per reasoning model, an agent roster from omodel-manager's
declared per-model configs (configs/*.toml) -- capabilities + per-mode sampling are
DECLARED, not probed (fast). Roster: visible `research` / `code` / `agent` + a
`team` orchestrator (which
delegates to hidden `agent-plan` / `agent-code` / `agent-instruct` workers) -- plus
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
                    "mistral", "xai", "groq", "deepseek", "cohere", "bedrock", "vertex"}
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
#   VISIBLE (Tab, mode primary, NO worker prompt) -- your direct-use agents.
#   HIDDEN  (mode subagent, WITH worker prompt) -- the team's delegation targets;
#           kept out of the Tab cycle so your direct agents stay prompt-free.
# Permission profiles (risk tiers). websearch/webfetch allowed on all.
#   readonly: no edits/bash/delegation (research only)
#   ask:      full access but PROMPTS for confirmation on edits/bash
#   full:     edits/bash run without prompting (autonomous)
# `team-orchestration` is the team lead's methodology skill (written globally by sync);
# deny it to every non-team agent so it only surfaces for `team` (team has its own
# permission block below and does NOT carry this deny).
PERM = {
    "readonly": {"edit": "deny",  "bash": "deny",  "task": "deny",
                 "websearch": "allow", "webfetch": "allow",
                 "skill": {"team-orchestration": "deny"}},
    "ask":      {"edit": "ask",   "bash": "ask",   "websearch": "allow", "webfetch": "allow",
                 "task": {"*": "deny", "agent-review": "allow"},
                 "skill": {"team-orchestration": "deny"}},
    "full":     {"edit": "allow", "bash": "allow", "websearch": "allow", "webfetch": "allow",
                 "task": {"*": "deny", "agent-review": "allow"},
                 "skill": {"team-orchestration": "deny"}},
}
# Each spec: (key, preset role, mode, is_worker, perm profile, color, description).
# color = FIXED hex by risk: green (read-only) -> yellow-green (ask) -> orange
# (autonomous) -> red (team). Tweak the hexes here.
# Visible names avoid the reserved built-ins `build`/`plan` (which OpenCode won't
# let you override -- they'd show as "native" and ignore our settings). We instead
# disable the natives and use `research` (planner) + `code` (coder).
AGENT_SPECS = [
    ("research", "reason", "primary", False, "readonly", "#22c55e", "research & reasoning, read-only + web"),
    ("code",     "code",   "primary", False, "ask",      "#a3e635", "interactive coder -- asks before edits/bash"),
    ("agent",    "agent",  "primary", False, "full",     "#f97316", "autonomous worker, full access (no prompts)"),
    ("agent-plan",     "reason",   "subagent", True, "readonly", "#22c55e", "[worker] research & reasoning, read-only + web"),
    ("agent-code",     "code",     "subagent", True, "full",     "#f97316", "[worker] coding / implementation / debugging, full access"),
    ("agent-instruct", "instruct", "subagent", True, "full",     "#eab308", "[worker] fast mechanical subtasks, no thinking"),
    ("agent-review",   "reason",   "subagent", True, "full", "#3b82f6", "[worker] handles reviewing Pull Requests — always re-use task_id when reviewing related PRs in the same session to maintain context"),
]
# Hidden workers the team may delegate to (its permission.task allowlist).
TEAM_TARGETS = ["agent-plan", "agent-code", "agent-instruct", "agent-review"]
# Built-in agents we disable (can't be overridden; replaced by research/code).
BUILTIN_DISABLE = ["build", "plan"]
TEAM_COLOR = "#ef4444"   # red -- highest risk (orchestrates, spends $, delegates)
# Agent keys this tool may write under --profiles (current + legacy). Used to prune
# stale ones on re-sync -- incl. old plan/build OVERRIDES and old names -- so
# re-syncing converges. Won't touch the user's own agents.
MANAGED_AGENTS = {"research", "code", "agent", "team",
                  "agent-plan", "agent-code", "agent-instruct", "agent-review",
                  "plan", "build", "instruct", "architect", "reason", "chat", "fast",
                  "general", "webdev", "agentic"}

# Default models config file (user preferences for agent/subagent models)
DEFAULT_MODELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_models.json")


def load_default_models():
    """Load default_models.json preferences. Returns {agents: {...}, subagents: {...}}.
    Auto-creates with template if missing. Never overwrites existing user config."""
    if not os.path.exists(DEFAULT_MODELS_FILE):
        return _create_default_models_template()
    try:
        with open(DEFAULT_MODELS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _create_default_models_template()
        if "agents" not in data:
            data["agents"] = {}
        if "subagents" not in data:
            data["subagents"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return _create_default_models_template()


def _create_default_models_template():
    """Create default_models.json with template (all agents set to qwen3-coder-next-fp8)."""
    template = {
        "agents": {
            "team": ["qwen3-coder-next-fp8"],
            "research": ["qwen3-coder-next-fp8"],
            "code": ["qwen3-coder-next-fp8"],
            "agent": ["qwen3-coder-next-fp8"]
        },
        "subagents": {
            "agent-plan": ["qwen3-coder-next-fp8"],
            "agent-code": ["qwen3-coder-next-fp8"],
            "agent-instruct": ["qwen3-coder-next-fp8"],
            "agent-review": ["qwen3-coder-next-fp8"]
        }
    }
    try:
        with open(DEFAULT_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
    except OSError:
        pass
    return template


def _resolve_model_ref_from_prefs(pref, available_models):
    """Resolve a model preference to its full provider-qualified ref if available.
    
    Args:
        pref: model preference string - can be:
            - Full ref: "openai/gpt-5.5" (must match exactly in available_models)
            - Served ID with slash: "unsloth/qwen3-coder-next-fp8" (DGX model, must match model ID)
            - Bare model ID: "qwen3-coder-next-fp8" (must match model ID)
        available_models: dict mapping full refs to {} (e.g., {"dgx-1-8000/qwen3.6-27b": {}})
    
    Returns:
        Full provider-qualified ref if resolvable, None otherwise.
    """
    if not pref or not available_models:
        return None
    
    # Check if pref is already a full ref (contains / with a known provider)
    if "/" in pref:
        provider_part = pref.split("/", 1)[0]
        # If it's a known remote provider or dgx provider, check exact match in available_models
        if provider_part in REMOTE_PROVIDERS or provider_part.startswith("dgx-"):
            if pref in available_models:
                return pref
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


def _apply_default_models(agents, default_models, reasoning_caps, available_models=None, notes=None):
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
                resolved = _resolve_model_ref_from_prefs(pref, available_models)
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
                    resolved_ref = _resolve_model_ref_from_prefs(preferred, available_models)
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


# System prompt for the `team` lead/orchestrator. Written to a file next to
# opencode.json and referenced via {file:...}; edit it there to tune behavior.
TEAM_PROMPT = """You are the Team Lead -- an orchestrator. You have NO tools of your own: you
cannot read, grep, edit, or run commands -- the ONLY thing you can do is delegate
by calling the `task` tool. Your job is to break work down, hand each piece to a
worker, and verify the results they report back.

You delegate by CALLING THE `task` TOOL -- choose a subagent by NAME and give it
an instruction. Do NOT just type "@agent ..." in your reply; that does nothing.
You do not format the call yourself: the task tool's schema handles that -- you
only pick the subagent and write the instruction.

Subagents you can delegate to (use the exact name as the task tool's subagent):
- agent-plan     -- research & reasoning, READ-ONLY (web search/fetch + reading
                    files; cannot edit or run commands). For gathering info and
                    reasoning through problems BEFORE implementation.
- agent-code     -- capable worker, full edit/shell + reasoning. For anything
                    non-trivial: implementation, refactors, debugging, investigating.
- agent-instruct -- fast, no-reasoning worker. ONLY for simple, well-specified,
                    mechanical subtasks: one obvious edit, rename, format, boilerplate.
- agent-review   -- reviews a pull request against the repo's bar. Delegate when the
                    user asks to review / approve / merge a PR. It returns issues +
                    suggested fixes (route those to agent-code, then re-delegate with
                    the SAME task_id to re-review); it merges only when clean.

**Before you plan, load the `team-orchestration` skill and follow it** -- it holds how to
decompose a request, dispatch independent work to subagents in PARALLEL, sequence only
true dependencies, and verify results. Never edit or run anything yourself; if the request
is ambiguous, ask one round of clarifying questions before delegating.
"""


# `team-orchestration` skill -- the Team Lead's methodology. Written GLOBALLY by sync to
# <opencode-config>/skills/team-orchestration/SKILL.md (auto-scanned), and scoped to `team`
# via permission.skill (see PERM). Thin prompt + this skill = the agent-review/pr-review pattern.
TEAM_ORCHESTRATION_SKILL = """---
name: team-orchestration
description: How the Team Lead decomposes a request into subtasks, dispatches independent work to subagents in PARALLEL, sequences only true dependencies, and verifies results. Load at the start of every request you orchestrate.
---

# Orchestrating a request

You are the Team Lead. You do not do the work yourself -- you decompose it, delegate each
piece to the right subagent via the `task` tool, and verify what comes back.

## The loop
1. **Restate the goal** in one line and list explicit, checkable acceptance criteria.
2. **Decompose** into the smallest independent subtasks.
3. **Parallelize first.** Identify every subtask that does NOT depend on another's output and
   dispatch them *together* (multiple `task` calls in one turn) rather than one at a time.
   Prefer a wide first wave of independent work over a serial chain -- this is the single
   biggest lever on wall-clock time.
4. **Sequence only true dependencies.** A subtask waits only if it genuinely needs a prior
   result; everything else runs concurrently.
5. **Write self-contained instructions.** Each worker cannot see this conversation -- put the
   exact files/paths, what to do, and the acceptance criteria in the task itself.
6. **Verify against the criteria.** When workers report back, check their results. If
   something is missing or wrong, delegate a focused follow-up -- never fix it yourself.
7. **Summarize** for the user once all criteria are met, plus any follow-ups.

## Routing
- research / info-gathering / reading docs -> **agent-plan** (read-only)
- implementation / refactors / debugging / anything non-trivial -> **agent-code**
- one obvious mechanical edit (rename, format, boilerplate) -> **agent-instruct**
- review / approve / merge a PR -> **agent-review** (it returns issues+fixes; route fixes to
  agent-code, then re-delegate to agent-review with the SAME task_id)

## Rules
- You have no read/edit/shell tools -- never try to do the work yourself.
- Keep every delegated instruction scoped and independently verifiable.
- Spend your delegation budget (task_budget) on the widest set of independent subtasks that
  cleanly separate -- don't serialize work that could run in parallel.
- If the request is ambiguous, ask ONE round of clarifying questions before delegating.
"""


def _team_skill_path(cfg_path):
    return os.path.join(os.path.dirname(cfg_path), "skills", "team-orchestration", "SKILL.md")


def team_prompt_text(task_budget=None):
    """The team system prompt, with a soft note about the delegation budget so the
    model knows how many subtasks it can run (OpenCode's task_budget enforces it)."""
    text = TEAM_PROMPT
    if task_budget is not None:
        text += (f"\nDelegation budget: you have up to {task_budget} sub-task delegations per "
                 f"run -- feel free to use all {task_budget} when the work cleanly separates into "
                 f"that many independent pieces.\n")
    return text


# Worker system prompt for delegation targets (plan/agent/instruct). Counters the
# OpenCode local-subagent bug (#18423 / PR #18429) where the orchestrator gets the
# subagent's LAST text part even when it's empty -- by forcing a non-empty,
# results-bearing final message. Written next to opencode.json; edit there to tune.
#
# ORDERING IS LOAD-BEARING -- do not front-load the "final message must be plain text"
# instruction. Leading with output/summary framing makes Qwen3-family models (served via
# vLLM --tool-call-parser qwen3_coder) NARRATE tool calls as text (<invoke .../>, bash(...),
# etc.) instead of emitting native tool calls -> the parser drops them, the loop exits, and
# the worker fabricates/leaks (verified on n1: 1/8 vs 15/15 with action-first wording).
# Lead with "call the tools", THEN state the summary rule.
WORKER_PROMPT = """Complete the task by calling the provided tools. Act, inspect each tool result, then continue until the task is done.

When the work is finished, send a final plain-text message that summarizes what you did and includes the concrete results that matter: command output, files changed, key findings, or the exact error if something failed. That final message is the only thing your caller (the orchestrator) receives back, so:
- Never stop on a bare tool call or an empty message -- always finish with a text summary.
- Do not just restate the command you ran; include what it RETURNED.
- If you couldn't complete the task, say so plainly and why.
"""

# Review prompt for agent-review. Written next to opencode.json; edit there to tune.
# This agent handles reviewing Pull Requests and provides feedback to the parent agent.
REVIEW_PROMPT = """You are agent-review -- the pull-request reviewer for whatever repository you're working in.

Load the `pr-review` skill and follow it end-to-end. In short: review the open PR against the repo's REVIEW.md, then hand the parent agent an itemized list of issues, each with a suggested fix. You do NOT fix issues yourself, and you do NOT merge while any remain -- the parent's coding agents apply the fixes and delegate back to you (reuse the task_id) to re-review. Merge ONLY when the review is clean, and only as the REVIEWER account (`GH_TOKEN="$GH_TOKEN_REVIEWER" gh ...`, per the skill).

Your final message is the review report the parent agent acts on.
"""

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

    def _opts_for(preset):
        """Build the thinking-knob options dict for a preset."""
        if control == "none":
            o = {}
        elif preset.get("thinking"):
            o = dict(oc_effort_options(caps, "high"))
        else:
            o = dict(oc_off_options(caps))
        # Explicit per-preset raw kwargs win (e.g. preserve/clear_thinking for the
        # agent role, or Nemotron's {enable_thinking, low_effort}).
        for k, v in (preset.get("options") or {}).items():
            if k == "chat_template_kwargs" and isinstance(o.get(k), dict) and isinstance(v, dict):
                o[k] = {**o[k], **v}
            else:
                o[k] = v
        return o

    def _vec_for(preset):
        """Plugin sampling vector for a preset."""
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
            "options": _opts_for(preset),
            "permission": dict(PERM[perm]),
        }
        # Hidden delegation workers get the worker prompt so they return a
        # non-empty results summary (works around OpenCode #18423). The VISIBLE
        # twins (plan/build/agent) stay prompt-free for clean direct use.
        if is_worker:
            # agent-review has its own review prompt; others use worker prompt
            if key == "agent-review":
                agent["prompt"] = "{file:./prompts/otools-review.md}"
            else:
                agent["prompt"] = "{file:./prompts/otools-worker.md}"
        # Reliable sampling lives in the agent config too (correct even without the
        # plugin); the plugin additionally enforces top_k/min_p/penalties/maxOutput.
        if "temperature" in s: agent["temperature"] = s["temperature"]
        if "top_p" in s: agent["top_p"] = s["top_p"]
        if not preset.get("thinking") and not caps["can_disable"] and not recipe.get("soft_switch"):
            agent["description"] += "  [WARN: endpoint can't disable thinking]"
        # code and agent need task_budget to delegate (only to agent-review)
        if key in ("code", "agent"):
            agent["task_budget"] = 1
        agents[key] = agent
        agent_sampling[key] = _vec_for(preset)

    # --- `team`: lead orchestrator with the reason preset's sampling/thinking
    # that DELEGATES to the hidden agent-* workers instead of editing. Built when
    # a reason preset + at least one worker exist. Model overridable via --team-model. ---
    rp = presets.get("reason")
    targets = [k for k in TEAM_TARGETS if k in agents]
    if rp and targets:
        task_map = {"*": "deny"}
        for k in targets:
            task_map[k] = "allow"
        rs = rp.get("sampling", {})
        team = {
            "description": "team: lead orchestrator -- plans and delegates to the agent-* "
                           "workers, validates; has NO tools of its own (delegation only)",
            "mode": "primary",
            "model": model_ref,
            "color": TEAM_COLOR,
            "prompt": "{file:./prompts/otools-team.md}",
            "options": _opts_for(rp),
            # Delegation-only: deny EVERY tool category (incl. the read-only ones --
            # read/grep/glob/list have their own permission keys and default to allow,
            # so denying edit/bash alone still lets the orchestrator grep/read). The
            # only thing it can do is spawn its workers via `task`.
            "permission": {"read": "deny", "grep": "deny", "glob": "deny", "list": "deny",
                           "edit": "deny", "bash": "deny",
                           "webfetch": "deny", "websearch": "deny",
                           "task": task_map},
        }
        if "temperature" in rs: team["temperature"] = rs["temperature"]
        if "top_p" in rs: team["top_p"] = rs["top_p"]
        agents["team"] = team
        agent_sampling["team"] = _vec_for(rp)
    return agents, agent_sampling


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
    if "team" in agents:
        print(f"  team delegates via @agent-plan (research) / @agent-code (hard) / "
              f"@agent-instruct (fast); Ctrl+T toggles thinking")


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
            if name == "team":       # orchestrator; audited on its own model, not here
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

    # ---- PROFILES: build role-agents (alongside built-in plan/build) ----------
    # Emits: code (primary), agent (all), instruct (subagent, hidden), architect
    # (orchestrator). Tab cycle = plan, build, code, agent, architect.
    # Curated recipe supplies per-model sampling; else generic Qwen numbers.
    agents = {}
    agent_model_ref = None
    agent_sampling = None
    per_model_sampling = None
    matched_recipe = None
    env_notes = []
    team_budget = None
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
        # from existing config are considered for team selection.
        pool = reasoning_caps if reasoning_caps else all_available_models
        
        # Build available_model_ids for remote model detection
        available_model_ids = []
        for model_ref in all_available_models.keys():
            model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
            available_model_ids.append(model_id)
        
        # Check if user prefers a remote model (e.g., openai/gpt-5.5) for team
        default_models = load_default_models()
        team_prefs = (default_models.get("agents") or {}).get("team", [])
        
        # Prefer a reasoning model first, then check team preferences
        agent_model_ref = None
        if reasoning_caps:
            agent_model_ref = cur if cur in reasoning_caps else sorted(reasoning_caps)[0]
        else:
            # No reasoning models - validate team preferences against all_available_models
            # Use the same resolver logic as _apply_default_models so unavailable
            # cloud refs like anthropic/claude-opus-4-8 and unavailable local served IDs
            # are not selected.
            for pref in team_prefs:
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
            agents, agent_sampling = oc_build_recipe_agents(
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
            agents, agent_sampling = oc_build_agents(
                agent_model_ref, caps, sampling.get("repetition_detection"))

        # Apply default models preferences to agents (default_models already loaded above)
        model_notes = []
        agents = _apply_default_models(agents, default_models, reasoning_caps, all_available_models, model_notes)
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
        # Capture a previously-set frontier team model + thinking config BEFORE we
        # overwrite, so a re-sync without the flags doesn't reset your choices.
        prev_team = existing_agents.get("team") or {}
        prev_team_model = prev_team.get("model")
        prev_team_thinking = (prev_team.get("options") or {}).get("thinking")
        prev_team_budget = prev_team.get("task_budget")
        # Rebuild the agent map so OUR agents are written in a FIXED canonical
        # order every sync (plan, build, agent, ...workers..., team). OpenCode's
        # Tab cycle follows the config object order in practice, and dict.update()
        # would otherwise preserve a STALE order from a prior sync (e.g. team
        # landing before build). User-created agents are kept (in their order).
        user_agents = {k: v for k, v in existing_agents.items() if k not in MANAGED_AGENTS}
        # Write VISIBLE primaries first (research, code, agent, team) as a contiguous
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

        # ---- Team model (frontier planner). Flag wins; else PRESERVE whatever
        # frontier model was already set (so we never wipe your anthropic choice).
        # Workers keep their own pinned local models, so they stay on the DGX. ----
        # Check if team preference resolved to a valid model - if so, use that instead of preserving
        team_agent_model = agents.get("team", {}).get("model") if agents else None
        team_model = args.team_model
        if not team_model:
            # No flag passed - check if preference resolved to a valid model
            if team_agent_model and team_agent_model != agent_model_ref:
                # Preference resolved to a different model - use it
                team_model = team_agent_model
            elif prev_team_model and not prev_team_model.split("/", 1)[0].startswith(PROVIDER_PREFIX):
                # No preference or preference is same as current - preserve existing
                team_model = prev_team_model
                print(f"  team model preserved from existing config: {team_model}")
        if team_model and "team" in agents:
            tm = cfg["agent"]["team"]
            tm["model"] = team_model
            provider = team_model.split("/", 1)[0]
            if not provider.startswith(PROVIDER_PREFIX):
                # non-dgx (e.g. anthropic): local vLLM options/top_p are meaningless
                # there and the dgx-scoped plugin won't touch it -- drop them.
                tm.pop("options", None)
                tm.pop("top_p", None)
                agent_sampling.pop("team", None)
                # Anthropic extended thinking. Flag wins; else preserve previous.
                # (Anthropic uses options.thinking budgetTokens, NOT reasoningEffort.)
                budget = {"low": 10000, "medium": 24000, "high": 32000}.get(args.team_reasoning)
                if budget and provider == "anthropic":
                    tm["options"] = {"thinking": {"type": "enabled", "budgetTokens": budget}}
                    tm["temperature"] = 1.0   # Anthropic REQUIRES temp=1 when thinking is on
                    print(f"  team reasoning -> {args.team_reasoning} (thinking budgetTokens {budget})")
                elif prev_team_thinking:
                    tm["options"] = {"thinking": prev_team_thinking}
                    tm["temperature"] = 1.0
                    print(f"  team reasoning preserved (budgetTokens "
                          f"{prev_team_thinking.get('budgetTokens')})")
            if args.team_model:
                print(f"  team model -> {team_model} (workers stay local)")

        # Cap how many sub-agents the team spawns (task_budget). Precedence:
        # flag > previously-set budget > the worker model's declared `concurrency`
        # (its max-num-seqs -- you can't usefully run more parallel workers than the
        # server has sequence slots). The effective value is injected into the team
        # prompt so the model can self-limit.
        team_budget = args.team_task_budget if args.team_task_budget is not None else prev_team_budget
        budget_src = "flag" if args.team_task_budget is not None else ("preserved" if prev_team_budget is not None else None)
        if team_budget is None and matched_recipe:
            conc = (matched_recipe.get("capabilities") or {}).get("concurrency")
            if isinstance(conc, int) and conc > 0:
                team_budget, budget_src = conc, "concurrency"
        if team_budget is not None and "team" in agents:
            cfg["agent"]["team"]["task_budget"] = team_budget
            if budget_src == "flag":
                print(f"  team task_budget -> {team_budget} delegations/session")
            elif budget_src == "concurrency":
                print(f"  team task_budget defaulted from model concurrency "
                      f"(max-num-seqs): {team_budget}")
            else:
                print(f"  team task_budget preserved: {team_budget}")

    # ---- Team orchestrator + worker prompt files ----------------------------
    team_prompt_path = None
    team_skill_path = None
    if args.profiles and "team" in agents:
        team_prompt_path = os.path.join(os.path.dirname(config_path),
                                        "prompts", "otools-team.md")
        # The team's methodology lives in a GLOBAL skill (auto-scanned from
        # <config-dir>/skills); the thin team prompt tells it to load this.
        team_skill_path = _team_skill_path(config_path)
    worker_prompt_path = None
    if args.profiles and any(k in agents for k in TEAM_TARGETS):
        worker_prompt_path = os.path.join(os.path.dirname(config_path),
                                          "prompts", "otools-worker.md")
    review_prompt_path = None
    if args.profiles and "agent-review" in agents:
        review_prompt_path = os.path.join(os.path.dirname(config_path),
                                          "prompts", "otools-review.md")

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
        if team_prompt_path:
            print(f"\n--- DRY RUN: would write {team_prompt_path} ---")
            print(team_prompt_text(team_budget))
        if worker_prompt_path:
            print(f"\n--- DRY RUN: would write {worker_prompt_path} ---")
            print(WORKER_PROMPT)
        if review_prompt_path:
            print(f"\n--- DRY RUN: would write {review_prompt_path} ---")
            print(REVIEW_PROMPT)
        if team_skill_path:
            print(f"\n--- DRY RUN: would write {team_skill_path} ---")
            print(TEAM_ORCHESTRATION_SKILL)
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

    if team_prompt_path:
        os.makedirs(os.path.dirname(team_prompt_path), exist_ok=True)
        with open(team_prompt_path, "w") as f:
            f.write(team_prompt_text(team_budget))
        print(f"Wrote {team_prompt_path}")

    if team_skill_path:
        os.makedirs(os.path.dirname(team_skill_path), exist_ok=True)
        with open(team_skill_path, "w") as f:
            f.write(TEAM_ORCHESTRATION_SKILL)
        print(f"Wrote {team_skill_path}")

    if worker_prompt_path:
        os.makedirs(os.path.dirname(worker_prompt_path), exist_ok=True)
        with open(worker_prompt_path, "w") as f:
            f.write(WORKER_PROMPT)
        print(f"Wrote {worker_prompt_path}")

    if review_prompt_path:
        os.makedirs(os.path.dirname(review_prompt_path), exist_ok=True)
        with open(review_prompt_path, "w") as f:
            f.write(REVIEW_PROMPT)
        print(f"Wrote {review_prompt_path}")

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
    "team_model":      ("model ref for the team orchestrator", None),
    "team_reasoning":  ("low|medium|high (Anthropic team thinking)", None),
    "default_agent":   ("startup agent (research/code/agent/team)", "code"),
    "web_search":      ("none|exa|mcp", "none"),
    "proxy_port":      ("proxy listen port (default: 9099)", 9099),
    "proxy_active":    ("is proxy currently active?", False),
}


def load_settings():
    """Read wire.json; {} if absent or unparseable."""
    try:
        with open(WIRE_SETTINGS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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
    team = (agents.get("team") or {}).get("model") or _setting(args, "team_model") or "(local worker model)"

    print("omodel-wire -- wire local model endpoints into OpenCode (omw)\n")
    print("Status:")
    cfg_note = "" if ntoml else "  [not found -- omw config --set configs_dir PATH]"
    print(f"  configs : {cdir}  ({ntoml} model config(s)){cfg_note}")
    print(f"  hosts   : {hosts}")
    dfl = f", default @{cfg.get('default_agent')}" if cfg.get("default_agent") else ""
    print(f"  opencode: {cfg_path}  ({len(managed)} managed agent(s){dfl})")
    print(f"  team    : {team}")
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
    _suggest([("Persist a value (VALUE 'none' to clear)", "omw config --set team_model anthropic/claude-opus-4-8"),
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

def copilot_home():
    """Copilot CLI config home: $COPILOT_HOME or ~/.copilot (cross-platform)."""
    return os.environ.get("COPILOT_HOME") or os.path.expanduser(os.path.join("~", ".copilot"))


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
    "agent-plan": (
        "You are a read-only research/reasoning worker. Gather exactly the information asked for -- read "
        "files, search, fetch docs -- and return a concrete, self-contained answer. Do not edit files or "
        "run commands with side effects."),
    "agent-code": (
        "You are a full-access implementation worker. Complete the assigned change end to end and verify "
        "it. Your final message must state what you changed and the concrete result: command output, "
        "files touched, or the exact error if it failed."),
    "agent-instruct": (
        "You are a fast worker for simple, well-specified, mechanical tasks (one obvious edit, rename, "
        "format, boilerplate). Do exactly what's asked and report the concrete result."),
    "team": (
        "You are the Team Lead -- an orchestrator. Break the request into the smallest independent "
        "pieces and delegate each to the specialist subagent that fits: research -> agent-plan, "
        "implementation -> agent-code, mechanical edits -> agent-instruct, PR review -> agent-review. "
        "Verify their results against explicit acceptance criteria before reporting back, and prefer "
        "dispatching independent work in parallel. Copilot routes delegation for you based on each "
        "subagent's description, so keep every delegated instruction scoped and self-contained."),
}


# Copilot descriptions drive the runtime's auto-delegation, so they're purpose-written
# (not the OpenCode `sdesc`, which references OpenCode-isms like task_id).
COPILOT_DESCRIPTIONS = {
    "research": "Read-only research & reasoning: investigate the codebase, read docs and the web, and reason through problems before implementation.",
    "code": "Interactive coding: implement, refactor, and debug, confirming before destructive edits or side-effecting commands.",
    "agent": "Autonomous coding with full access: take a task end to end -- edit, run, and verify.",
    "agent-plan": "Read-only research/reasoning worker: gather information and return a concrete answer; no edits.",
    "agent-code": "Full-access implementation worker: complete a change end to end and verify it.",
    "agent-instruct": "Fast worker for simple, well-specified mechanical edits (rename, format, boilerplate).",
    "agent-review": "Reviews a pull request against the repo's bar and reports issues + suggested fixes; invoke it explicitly for PR review / approve / merge.",
    "team": "Lead orchestrator: decompose a request, delegate pieces to the specialist subagents, and verify results.",
}


def copilot_build_agents():
    """{filename: agent-md content} for the full roster in Copilot's .agent.md format.

    Primaries (research/code/agent) + team are top-level agents; the agent-* workers are
    subagents. agent-review is `disable-model-invocation: true` (explicit-invoke-only) so it
    isn't auto-dispatched -- the closest Copilot has to "only the team delegates to review"."""
    out = {}
    for key, prole, mode, is_worker, perm, color, sdesc in AGENT_SPECS:
        body = REVIEW_PROMPT if key == "agent-review" else COPILOT_BODIES.get(key, sdesc)
        dmi = True if key == "agent-review" else None
        out[f"{key}.agent.md"] = _copilot_agent_md(
            name=key, description=COPILOT_DESCRIPTIONS.get(key, sdesc.replace("[worker] ", "")),
            body=body, tools=_copilot_tools_for(perm), disable_model_invocation=dmi)
    out["team.agent.md"] = _copilot_agent_md(
        name="team", description=COPILOT_DESCRIPTIONS["team"],
        body=COPILOT_BODIES["team"], tools=_copilot_tools_for("full"))
    return out


def _copilot_pick_model(providers, available_models, reasoning_caps):
    """Pick the ONE model Copilot's single BYOK endpoint will serve.

    Prefers the coder the `code` agent would use (Copilot is a coding assistant), then any
    reasoning model, then the first live model. Returns (model_ref, base_url, served_id)."""
    if not available_models:
        return None, None, None
    refs = sorted(available_models)
    served = {ref: (ref.split("/", 1)[1] if "/" in ref else ref) for ref in refs}
    chosen = None
    for pref in (load_default_models().get("agents") or {}).get("code", []):
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


def copilot_merge_settings(settings, served_id):
    """Set the omw-managed keys in Copilot's settings.json (non-destructive to other keys)."""
    settings["model"] = served_id
    settings["includeCoAuthoredBy"] = False   # matches the no-Co-Authored-By rule
    settings["stream"] = True
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


def _copilot_print_env_instructions(sh_path, ps1_path):
    print("\nOne more step -- point Copilot at the DGX endpoint (it can't live in settings.json):")
    print(f"  bash/zsh (Linux/macOS):  source {sh_path}")
    print(f"  PowerShell (Windows):    . {ps1_path}")
    print("  (add the source/dot-source line to your shell rc / $PROFILE to persist it,")
    print("   then run `copilot` and pick an agent, e.g. `copilot --agent code`.)")


def copilot_sync(args, sampling, detected_installed):
    """Write the GitHub Copilot CLI target: agent roster (.agent.md) + settings.json + the
    BYOK provider env snippet. The whole roster runs on ONE model (single custom endpoint).
    Returns an exit code (0 ok, 2 = nothing live)."""
    home = _copilot_home_for(args)
    configs = load_configs(args.configs) if not getattr(args, "no_recipes", False) else {"recipes": []}
    providers, refs, reasoning_caps, available_models = oc_build_providers(
        args._hosts, args._ports, args.timeout, sampling, profiles=True,
        tool_call=not getattr(args, "no_tool_call", False), recipes=configs, verbose=True)
    model_ref, base_url, served_id = _copilot_pick_model(providers, available_models, reasoning_caps)
    if not model_ref:
        print("No live models discovered -- launch one with `omm launch <profile> --host <h>` first; "
              "Copilot's single BYOK endpoint needs a live model.", file=sys.stderr)
        return 2

    print(f"\nGitHub Copilot target -- the whole roster runs on ONE endpoint:")
    print(f"  model:    {model_ref}")
    print(f"  base URL: {base_url}")
    agents = copilot_build_agents()

    if args.dry_run:
        print(f"\n--- DRY RUN: would write to {home} ---")
        print(f"  agents/  ({len(agents)}): {', '.join(sorted(agents))}")
        print(f"  settings.json (model={served_id}, includeCoAuthoredBy=false, stream=true)")
        print(f"  otools-copilot.env / .ps1  (COPILOT_PROVIDER_BASE_URL={base_url})")
        return 0

    agents_dir = os.path.join(home, "agents")
    os.makedirs(agents_dir, exist_ok=True)
    for fn, content in sorted(agents.items()):
        with open(os.path.join(agents_dir, fn), "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Wrote {len(agents)} agent(s) to {agents_dir}")

    settings_path = os.path.join(home, "settings.json")
    settings = copilot_merge_settings(_copilot_load_settings(settings_path), served_id)
    os.makedirs(home, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    print(f"Wrote {settings_path}")

    sh, ps1 = copilot_env_files(base_url, served_id)
    sh_path = os.path.join(home, "otools-copilot.env")
    ps1_path = os.path.join(home, "otools-copilot.ps1")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    with open(ps1_path, "w", encoding="utf-8") as f:
        f.write(ps1)
    print(f"Wrote {sh_path} and {ps1_path}")
    _copilot_print_env_instructions(sh_path, ps1_path)
    return 0


def cmd_sync(args):
    """Full profile sync: roster + providers + plugin + prompts (was --profiles)."""
    _resolve_io(args)
    _resolve_hosts_ports(args)
    if args.team_model is None:
        args.team_model = _setting(args, "team_model")
    if args.team_reasoning is None:
        args.team_reasoning = _setting(args, "team_reasoning")
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

    # Roster / team
    p.add_argument("--no-reasoning-probe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--keep-builtins", action="store_true",
                   help="keep OpenCode's native build/plan agents (normally disabled)")
    p.add_argument("--default-agent", default=None,
                   help="startup agent when build/plan are disabled "
                        "(default: wire.json default_agent, else code)")
    p.add_argument("--team-model", "--architect-model", metavar="REF", dest="team_model",
                   default=None,
                   help="put the `team` orchestrator on a specific model (e.g. "
                        "anthropic/claude-opus-4-8). Workers stay local; preserved across syncs.")
    p.add_argument("--team-task-budget", "--architect-task-budget", type=int, metavar="N",
                   dest="team_task_budget", default=None,
                   help="cap how many sub-agents the team may spawn per session")
    p.add_argument("--team-reasoning", choices=["low", "medium", "high"], dest="team_reasoning",
                   default=None,
                   help="(Anthropic team model) extended-thinking budget: low/medium/high")
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
        # Show task_budget for team and agents that can delegate (code, agent)
        budget = a.get("task_budget", "-") if name in ("team", "code", "agent") else "-"
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
    if name == "team":
        print(f"  work-budget: {a.get('task_budget', '(unset)')}")
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
        conc_s = f", concurrency={conc} (team work-budget default)" if conc else ""
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
# agent name -> preset role it runs (research=reason, code=code, ... team=reason)
AGENT_ROLE = {spec[0]: spec[1] for spec in AGENT_SPECS}
AGENT_ROLE.setdefault("team", "reason")

RESET_NOTE = "  (live edit only; run `omw sync` to reset to the known-good configs)"


def _boolish(s):
    v = str(s).strip().lower()
    if v in ("true", "on", "yes", "1"):
        return True
    if v in ("false", "off", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def _has_roster_mutation(args):
    return bool(getattr(args, "set_model", None)) or getattr(args, "set_work_budget", None) is not None


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
//     (bot) account. All coding agents (research/team/code/agent/agent-*) use this.
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
    if getattr(args, "set_work_budget", None) is not None:
        return _set_work_budget(cfg_path, cfg, agents, args.set_work_budget)

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
    for nm in targets:
        agents[nm]["model"] = ref
    _write_cfg(cfg_path, cfg)
    print(f"set model -> {ref}  for: {', '.join(targets)}")
    print(RESET_NOTE)
    _suggest([("See the change", "omw subagents" if want_subagent else "omw agents"),
              ("Reset to known-good", "omw sync")])


def _set_work_budget(cfg_path, cfg, agents, n):
    team = agents.get("team")
    if not isinstance(team, dict) or team.get("disable"):
        print("no `team` agent in this config (nothing delegates).", file=sys.stderr)
        sys.exit(1)
    team["task_budget"] = n
    tp = os.path.join(os.path.dirname(cfg_path), "prompts", "otools-team.md")
    os.makedirs(os.path.dirname(tp), exist_ok=True)
    with open(tp, "w", encoding="utf-8") as f:
        f.write(team_prompt_text(n))
    _write_cfg(cfg_path, cfg)
    print(f"team work-budget -> {n} delegations/session")
    print(RESET_NOTE)
    _suggest([("Review", "omw agents team"), ("Reset to known-good", "omw sync")])


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
                         help="list primary agents; show/tweak one (`omw agents team`)")
    pag.add_argument("name", nargs="?", help="agent name to show or edit")
    pag.add_argument("--set-model", metavar="REF", help="live-set this agent's model")
    pag.add_argument("--set-work-budget", type=int, metavar="N",
                     help="live-set the team's delegation budget (task_budget)")
    pag.set_defaults(func=cmd_agents)

    psa = sub.add_parser("subagents", parents=[io_parent],
                         help="list hidden workers; show/tweak (no name = all workers)")
    psa.add_argument("name", nargs="?", help="worker name to show or edit")
    psa.add_argument("--set-model", metavar="REF",
                     help="live-set model (no name -> all workers)")
    psa.add_argument("--set-work-budget", type=int, metavar="N",
                     help="live-set the team's delegation budget (forwards to team)")
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

    pd = sub.add_parser("detect", aliases=["doctor"],
                        help="report which agentic-dev tools are installed")
    pd.set_defaults(func=cmd_detect)

    psi = sub.add_parser("shell-init", aliases=["install-aliases"],
                         help="install the `omw` shell alias")
    psi.set_defaults(func=cmd_shell_init)

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
        if not cur or _is_loopback(cur):
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
               if _is_loopback((v.get("options") or {}).get("baseURL", ""))]
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
