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
import urllib.request

# ----------------------------------------------------------------------------
# DGX endpoint discovery defaults -- edit if your layout changes
# ----------------------------------------------------------------------------
DEFAULT_HOSTS = ["192.168.50.101", "192.168.50.102"]
# DEFAULT_PORTS = [8000, 8001, 8002, 8888, 30000, 11434]  # 11434 = ollama
DEFAULT_PORTS = [8000, 8001, 8002]
HOST_LABELS = {                       # friendly short labels for provider keys
    "192.168.50.101": "n1",
    "192.168.50.102": "n2",
    "192.168.50.103": "n3",
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


DEFAULT_CONTEXT = 200000              # used if endpoint doesn't report max_model_len
DEFAULT_OUTPUT = 65536               # OpenCode requires limit.output
API_KEY = "sglang"                    # dummy; vLLM/SGLang ignore it
PROVIDER_PREFIX = "dgx-"             # all managed providers start with this
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
PERM = {
    "readonly": {"edit": "deny",  "bash": "deny",  "task": "deny",
                 "websearch": "allow", "webfetch": "allow"},
    "ask":      {"edit": "ask",   "bash": "ask",   "websearch": "allow", "webfetch": "allow"},
    "full":     {"edit": "allow", "bash": "allow", "websearch": "allow", "webfetch": "allow"},
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
]
# Hidden workers the team may delegate to (its permission.task allowlist).
TEAM_TARGETS = ["agent-plan", "agent-code", "agent-instruct"]
# Built-in agents we disable (can't be overridden; replaced by research/code).
BUILTIN_DISABLE = ["build", "plan"]
TEAM_COLOR = "#ef4444"   # red -- highest risk (orchestrates, spends $, delegates)
# Agent keys this tool may write under --profiles (current + legacy). Used to prune
# stale ones on re-sync -- incl. old plan/build OVERRIDES and old names -- so
# re-syncing converges. Won't touch the user's own agents.
MANAGED_AGENTS = {"research", "code", "agent", "team",
                  "agent-plan", "agent-code", "agent-instruct",
                  "plan", "build", "instruct", "architect", "reason", "chat", "fast",
                  "general", "webdev", "agentic"}

# System prompt for the `team` lead/orchestrator. Written to a file next to
# opencode.json and referenced via {file:...}; edit it there to tune behavior.
TEAM_PROMPT = """You are the Team Lead -- an orchestrator. You do NOT write code, edit files, or
run commands yourself (you have no edit/bash access). Your job is to break work
down, delegate it to your workers, and verify the result.

You delegate by CALLING THE `task` TOOL -- choose a subagent by NAME and give it
an instruction. Do NOT just type "@agent ..." in your reply; that does nothing.
You do not format the call yourself: the task tool's schema handles that -- you
only pick the subagent and write the instruction.

Subagents you can delegate to (use the exact name as the task tool's subagent):
- agent-plan     -- research & reasoning, READ-ONLY (web search/fetch + reading
                    files; cannot edit or run commands). Use for: gathering info,
                    reading docs/web, and reasoning through problems that need
                    research BEFORE implementation.
- agent-code     -- capable worker, full edit/shell + reasoning. Use for anything
                    non-trivial: implementation, refactors, debugging, reading
                    logs, investigating problems.
- agent-instruct -- fast, no-reasoning worker. Use ONLY for simple, well-specified,
                    mechanical subtasks: one obvious edit, rename, format,
                    summarize a file/log, boilerplate.

For every request:
1. Restate the goal in one line and list explicit acceptance criteria.
2. Decompose into the smallest independent subtasks.
3. For each, call the task tool with the right subagent name and a SELF-CONTAINED
   instruction -- exact files/paths, what to do, and acceptance criteria. The
   worker cannot see this conversation, so include everything it needs.
4. Sequence only true dependencies; keep subtasks independent where possible.
5. When workers report back, check results against the acceptance criteria. If
   something is missing or wrong, delegate a focused follow-up -- never fix it
   yourself.
6. When all criteria are met, give the user a concise summary + any follow-ups.

Rules: never edit/run directly; route research/info-gathering -> agent-plan,
hard/ambiguous implementation -> agent-code, trivial mechanical -> agent-instruct;
keep every delegated instruction scoped and verifiable; if the request is
ambiguous, ask one round of clarifying questions before delegating.
"""


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
            agent["prompt"] = "{file:./prompts/otools-worker.md}"
        # Reliable sampling lives in the agent config too (correct even without the
        # plugin); the plugin additionally enforces top_k/min_p/penalties/maxOutput.
        if "temperature" in s: agent["temperature"] = s["temperature"]
        if "top_p" in s: agent["top_p"] = s["top_p"]
        if not preset.get("thinking") and not caps["can_disable"] and not recipe.get("soft_switch"):
            agent["description"] += "  [WARN: endpoint can't disable thinking]"
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
            "description": "team: lead orchestrator -- plans, delegates to the agent-* workers, "
                           "validates; does not edit directly",
            "mode": "primary",
            "model": model_ref,
            "color": TEAM_COLOR,
            "prompt": "{file:./prompts/otools-team.md}",
            "options": _opts_for(rp),
            "permission": {"edit": "deny", "bash": "deny",
                           "websearch": "allow", "webfetch": "allow",
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

            providers[key] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": f"DGX {host_label(host)}:{port}",
                "options": {"baseURL": base, "apiKey": API_KEY},
                "models": model_entries,
            }
            if verbose:
                ids = ", ".join(m["id"] for m in found)
                print(f"  [up]   {host}:{port}  ->  {ids}")
    return providers, refs, reasoning_caps


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
    providers, refs, reasoning_caps = oc_build_providers(
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
    cfg["provider"] = kept

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
    if args.profiles and reasoning_caps:
        cur = cfg.get("model")
        agent_model_ref = cur if cur in reasoning_caps else sorted(reasoning_caps)[0]
        caps = reasoning_caps[agent_model_ref]
        model_id = agent_model_ref.split("/", 1)[1] if "/" in agent_model_ref else agent_model_ref
        matched_recipe = match_recipe(model_id, configs)
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

        # Per-(model, agent) sampling: one table per discovered reasoning model, so
        # switching an agent onto another model applies THAT model's card sampling
        # (10+ models with different recommended temps each get their own vector).
        _rep_det = sampling.get("repetition_detection")
        per_model_sampling = {}
        for _mref, _mcaps in reasoning_caps.items():
            _mid = _mref.split("/", 1)[1] if "/" in _mref else _mref
            _mrec = match_recipe(_mid, configs)
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
        team_model = args.team_model
        if not team_model and prev_team_model and \
                not prev_team_model.split("/", 1)[0].startswith(PROVIDER_PREFIX):
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

        # Cap how many sub-agents the team spawns (task_budget). Flag wins; else
        # PRESERVE a previously-set budget so re-syncs don't drop it. The effective
        # value is injected into the team prompt so the model can self-limit.
        team_budget = args.team_task_budget if args.team_task_budget is not None else prev_team_budget
        if team_budget is not None and "team" in agents:
            cfg["agent"]["team"]["task_budget"] = team_budget
            if args.team_task_budget is not None:
                print(f"  team task_budget -> {team_budget} delegations/session")
            else:
                print(f"  team task_budget preserved: {team_budget}")

    # ---- Team orchestrator + worker prompt files ----------------------------
    team_prompt_path = None
    if args.profiles and "team" in agents:
        team_prompt_path = os.path.join(os.path.dirname(config_path),
                                        "prompts", "otools-team.md")
    worker_prompt_path = None
    if args.profiles and any(k in agents for k in TEAM_TARGETS):
        worker_prompt_path = os.path.join(os.path.dirname(config_path),
                                          "prompts", "otools-worker.md")

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
        if not rc:
            print("Profiles mode: no reasoning models found")
        elif matched_recipe:
            print(f"Profiles mode: recipe match for {agent_model_ref}")
            print(f"  source: {matched_recipe.get('source','(recipe)')}")
            _print_roster(agents)
            for n in env_notes:
                print(f"  output: {n}")
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

    if team_prompt_path:
        os.makedirs(os.path.dirname(team_prompt_path), exist_ok=True)
        with open(team_prompt_path, "w") as f:
            f.write(team_prompt_text(team_budget))
        print(f"Wrote {team_prompt_path}")

    if worker_prompt_path:
        os.makedirs(os.path.dirname(worker_prompt_path), exist_ok=True)
        with open(worker_prompt_path, "w") as f:
            f.write(WORKER_PROMPT)
        print(f"Wrote {worker_prompt_path}")

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


def main():
    ap = argparse.ArgumentParser(
        description="Detect agentic-dev tools and sync DGX models into them (OpenCode today).")
    ap.add_argument("--version", action="version",
                    version=f"omodel-wire {__version__}")
    ap.add_argument("--detect-only", action="store_true",
                    help="only report which agentic-dev tools are installed; do not sync")
    ap.add_argument("--install-aliases", action="store_true",
                    help="add the `omw` shell alias (this sync tool) to your shell rc, then exit")
    ap.add_argument("--hosts", default=",".join(load_shared_hosts() or DEFAULT_HOSTS),
                    help="comma-separated host IPs to probe "
                         "(default: the shared ~/.config/otools/hosts, else built-in)")
    ap.add_argument("--ports", default=",".join(map(str, DEFAULT_PORTS)),
                    help="comma-separated ports to probe on each host")
    ap.add_argument("--config", default="~/.config/opencode/opencode.json",
                    help="path to opencode.json")
    ap.add_argument("--set-default", metavar="REF",
                    help="set OpenCode top-level default model, e.g. dgx-n1-8000/qwen3-coder")
    ap.add_argument("--timeout", type=float, default=PROBE_TIMEOUT)
    ap.add_argument("--no-vision-probe", action="store_true",
                    help="skip the image test entirely (don't auto-detect vision)")
    ap.add_argument("--vision-probe-all", action="store_true",
                    help="image-probe EVERY model, not just name-matched ones (slower)")

    # Sampling control (FIX #2)
    ap.add_argument("--sampling", choices=["server-default", "fixed", "opencode-default"],
                    default="server-default",
                    help="server-default: stop OpenCode sending temp/topP/topK/penalties "
                         "(server decides); fixed: pin values via flags below; "
                         "opencode-default: keep OpenCode's behavior (old).")
    ap.add_argument("--temperature", type=float, default=None,
                    help="(--sampling fixed) temperature to pin; omit to leave unset")
    ap.add_argument("--top-p", type=float, default=None,
                    help="(--sampling fixed) top_p to pin; omit to leave unset")
    ap.add_argument("--top-k", type=int, default=None,
                    help="(--sampling fixed) top_k to pin; omit to leave unset")
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="(--sampling fixed) presence_penalty to set; omit to drop it")
    ap.add_argument("--frequency-penalty", type=float, default=None,
                    help="(--sampling fixed) frequency_penalty to set; omit to drop it")
    ap.add_argument("--no-sampling-plugin", action="store_true",
                    help="don't write the chat.params plugin (only set temperature:false)")
    ap.add_argument("--repetition-detection", default=None, metavar="VAL",
                    help="vLLM repetition_detection to terminate degenerate N-gram loops "
                         "(default: min_pattern_size:3, max_pattern_size:20, min_count:10 -- "
                         "lenient, only cuts long stuck loops). Use 'off' to disable, or "
                         "'K1:V1,...' to override knobs (merged onto the default, so "
                         "'min_count:14' raises just that one), e.g. 'min_pattern_size:2,min_count:6'.")

    # Tool calling + web search exposure
    ap.add_argument("--no-tool-call", action="store_true",
                    help="don't declare tool_call capability (OpenCode then won't send "
                         "ANY tools -- websearch/edit/bash -- to these custom models)")
    ap.add_argument("--web-search", choices=["none", "exa", "mcp"], default="none",
                    help="expose a web-search tool to all models. exa: built-in keyless Exa "
                         "(needs OPENCODE_ENABLE_EXA env, see --enable-exa-shell); "
                         "mcp: add an MCP server via --mcp-command or --mcp-url.")
    ap.add_argument("--enable-exa-shell", action="store_true",
                    help="(--web-search exa) append OPENCODE_ENABLE_EXA=1 to your shell rc")
    ap.add_argument("--write-shell-env", action="store_true",
                    help="append any needed OpenCode env vars (e.g. EXA, "
                         "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX for >32k output) to your shell rc")
    ap.add_argument("--mcp-name", default="websearch",
                    help="(--web-search mcp) name for the MCP server entry")
    ap.add_argument("--mcp-command",
                    help="(--web-search mcp) stdio command, e.g. 'npx -y exa-mcp-server'")
    ap.add_argument("--mcp-url", help="(--web-search mcp) remote MCP URL instead of a command")
    ap.add_argument("--mcp-env", action="append", metavar="KEY=VAL",
                    help="(--web-search mcp local) env var for the server; repeatable. "
                         "Use {env:NAME} to reference a shell var without storing the secret.")
    ap.add_argument("--mcp-header", action="append", metavar="KEY=VAL",
                    help="(--web-search mcp remote) HTTP header; repeatable.")

    # Per-task profiles (agents + thinking-depth variants)
    ap.add_argument("--profiles", action="store_true",
                    help="generate the agent roster (research/code/agent + a team orchestrator "
                         "delegating to hidden agent-plan/agent-code/agent-instruct workers), "
                         "plus Ctrl+T thinking variants. Probes each endpoint for the right "
                         "thinking knob and disables native build/plan. Supersedes --sampling.")
    ap.add_argument("--no-reasoning-probe", action="store_true",
                    help="(with --profiles) skip the reasoning capability probe")
    ap.add_argument("--keep-builtins", action="store_true",
                    help="(with --profiles) keep OpenCode's native build/plan agents instead "
                         "of disabling them (we replace them with code/research).")
    ap.add_argument("--default-agent", default="code",
                    help="startup agent when native build/plan are disabled (default: code). "
                         "e.g. research / code / agent / team.")
    ap.add_argument("--team-model", "--architect-model", metavar="REF", dest="team_model",
                    help="put the `team` orchestrator on a specific model, e.g. "
                         "anthropic/claude-sonnet-4-6 or anthropic/claude-opus-4-8 (frontier "
                         "planner). Workers stay on their local models. A previously-set "
                         "frontier model is PRESERVED across re-syncs even without this flag. "
                         "Non-dgx refs drop the local chat_template_kwargs automatically.")
    ap.add_argument("--team-task-budget", "--architect-task-budget", type=int, metavar="N",
                    dest="team_task_budget",
                    help="cap how many sub-agents (task/delegation calls) the team may spawn "
                         "per session. Omit for no cap (depth still bounded by level_limit).")
    ap.add_argument("--team-reasoning", choices=["low", "medium", "high"], dest="team_reasoning",
                    help="(Anthropic team model) set extended-thinking budget: low=10000, "
                         "medium=24000, high=32000 budgetTokens. Preserved across re-syncs.")
    ap.add_argument("--configs", metavar="PATH",
                    help="omodel-manager's generic per-model configs dir (default: "
                         "$OMODEL_CONFIGS, else sibling ../omodel-manager/configs)")
    ap.add_argument("--no-recipes", action="store_true",
                    help="(with --profiles) ignore the configs; yields generic behavior")

    ap.add_argument("--audit", action="store_true",
                    help="compare the LIVE OpenCode agent sampling against the omodel-manager "
                         "configs (offline, no probing): a side-by-side table per model + agent, "
                         "highlighting drift and suggesting `--profiles` to re-sync. Writes nothing.")
    ap.add_argument("--verify", action="store_true",
                    help="probe live endpoints and compare their real capabilities to the "
                         "declared configs (opt-in; writes nothing). Slow -- re-runs the "
                         "vision/reasoning probes the normal path no longer uses.")
    ap.add_argument("--dry-run", action="store_true", help="print result, do not write")
    ap.add_argument("--allow-empty", action="store_true",
                    help="write even if NOTHING was discovered (default: refuse, as a safety net)")
    args = ap.parse_args()

    if args.install_aliases:
        install_aliases()
        return

    args._hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    args._ports = [int(p) for p in args.ports.split(",") if p.strip()]

    if args.audit:
        sys.exit(oc_audit(args))

    if args.verify:
        sys.exit(oc_verify(args))

    # 1) Detection
    detected = detect_tools()
    print_detection(detected)
    if args.detect_only:
        return

    installed = {t["key"] for t in detected if t["installed"]}

    # 2) Sync the tools that (a) support it and (b) are installed.
    sampling = build_sampling(args)
    ran_any = False
    for tool in detected:
        if tool["sync"] != "opencode":
            continue
        ran_any = True
        if not tool["installed"]:
            print(f"OpenCode not found on PATH -- writing config anyway "
                  f"(remove --config or install opencode to use it).")
        rc = oc_sync(args, sampling, installed)
        if rc not in (0,):
            sys.exit(rc)

    if not ran_any:
        print("No configurable tools matched. (Only OpenCode sync is implemented today.)")


if __name__ == "__main__":
    main()
