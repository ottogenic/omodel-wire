# AGENTS.md — architecture & working notes for `omodel-wire`

This file is the machine-and-human oriented map of the codebase. Read it before
editing. It complements `README.md` (user-facing) — here we cover *how* the tool is
built and the invariants you must not break.

## What this is

A single-file, stdlib-only Python CLI (`omodel-wire.py`) that (1) detects installed
agentic-dev tools and (2) syncs OpenAI-compatible model endpoints into their configs.
OpenCode is the only wired-up target today. Curated model settings (capabilities +
per-mode sampling) live in **omodel-manager**'s `configs/*.toml`; this tool consumes
them — it does not own them.

**Constraints (do not violate):**
- **Standard library only.** No third-party imports (uses `tomllib`, stdlib 3.11+).
  Runs with a bare `python3` (3.11+).
- **Single script, no config data.** The tool is `omodel-wire.py`; the generic model
  configs live in omodel-manager, read via `--configs` / `$OMODEL_CONFIGS` / sibling.
- **Idempotent & non-destructive.** Every run merges into existing config and prunes
  only what it manages. Never clobber user-authored agents/providers or preserved
  cloud settings.
- **Cross-platform paths.** Runs from WSL/Linux and Windows; use `os.path`/`expanduser`
  and never assume a POSIX-only home.

## Layout of `omodel-wire.py`

Top-to-bottom, the meaningful sections:

- **Discovery defaults** — `DEFAULT_HOSTS` (fallback only), `DEFAULT_PORTS`, `HOST_LABELS`,
  probe timeouts/limits, and `load_shared_hosts()` which reads the shared
  `~/.config/otools/hosts` store (managed by omodel-manager) as the `--hosts` default so
  both tools see the same fleet. Edit `DEFAULT_HOSTS`/`HOST_LABELS` only for the no-store
  fallback / provider-key labels.
- **Configs (consumed, not owned)** — `_configs_dir()` (resolves `--configs` /
  `$OMODEL_CONFIGS` / sibling `../omodel-manager/configs`), `load_configs()` (reads
  `omodel-manager`'s `configs/*.toml` via `tomllib`), `caps_from_capabilities()`
  (synthesizes the probe-style caps dict from a config's DECLARED capabilities),
  `match_recipe()`. **omodel-manager owns and validates these configs**; this tool is
  an adapter. There is NO live probing on the sync path — capabilities are declared.
- **Agent model** —
  - `PERM`: permission tiers `readonly` / `ask` / `full` (edit/bash/task/web policy).
  - `AGENT_SPECS`: list of `(key, preset_role, mode, is_worker, perm_profile, color, desc)`.
    This is the single source of truth for the roster.
  - `TEAM_TARGETS` = the three hidden workers the team may delegate to.
  - `BUILTIN_DISABLE` = `["build", "plan"]` — native agents we suppress.
  - `TEAM_COLOR`, and risk-based colors carried per spec.
  - `MANAGED_AGENTS`: the set the tool is allowed to prune. **Anything not in this set
    is left untouched.** Add new managed agent keys here or stale entries won't be
    cleaned.
  - `TEAM_PROMPT` / `team_prompt_text()` (injects the soft task-budget line),
    `WORKER_PROMPT` (forces a non-empty final summary on hidden workers).
- **Tool registry** — `TOOLS` list + `detect_tools()` / `print_detection()`. This is the
  extension point for new tools: add a `TOOLS` entry with a `sync` key and a matching
  configurator.
- **Probes (verify-only)** — `probe()` (/v1/models, still used for discovery),
  `probe_vision()` (blue-image), `probe_reasoning()` (reasoning + thinking-knob),
  `_chat()` helper. These are NO LONGER on the sync path (capabilities are declared);
  they run only under `--verify` (`oc_verify()`), which compares a live endpoint to
  its declared config.
  `_reasoning_len()` measures chain-of-thought from the structured `reasoning`/
  `reasoning_content` field **and** inline `<think>...</think>` in `content` (endpoints
  without a reasoning parser); `_finish_reason()` lets `probe_reasoning` treat a
  `finish_reason=length` cutoff as "still thinking" — this is how a qwen3-`--reasoning-parser`
  model truncated before `</think>` (vLLM #35221 dumps partial reasoning into `content`
  with `reasoning=null`) is still detected. `REASONING_MAXTOKENS` must stay generous
  enough for a trivial prompt to reach `</think>`.
- **OpenCode config builders** — `oc_build_variants()`, `oc_build_agents()`,
  `oc_build_recipe_agents()` (recipe → per-role agents + synthesized `team`),
  `oc_build_providers()`, the two `chat.params` plugin emitters
  (`oc_agent_sampling_plugin_js`, `oc_sampling_plugin_js`), and `oc_apply_web_search()`.
- **Assembly** — `oc_sync()` is the heart: loads existing config, builds providers +
  agents, orders them (visible primaries → hidden subagents → disabled build/plan
  stubs → user agents), preserves the frontier team model / reasoning / task budget
  across runs, sets `default_agent`, and writes (unless `--dry-run`).
- **Shell integration** — `_shell_rc()`, `_ensure_shell_env()`, `install_aliases()`
  (installs the `omw` alias only).
- **CLI (subcommands, omm-style)** — `_build_parser()` wires argparse subparsers with
  `.set_defaults(func=cmd_x)`; `main(argv=None)` dispatches `args.func(args)` (bare `omw`
  → `cmd_home`). Commands: `cmd_sync` (was the `--profiles` path; sets `args.profiles`),
  `cmd_agents`/`cmd_subagents` (`_roster_view` + `_mutate_roster`), `cmd_models`
  (`_mutate_model`), `cmd_audit`/`cmd_verify`/`cmd_detect`/`cmd_shell_init`, `cmd_config`.
  `build_sampling()` still builds the sampling dict from the `sync` flags.
- **Settings + UX helpers** — `~/.config/otools/wire.json` via `load_settings`/
  `save_settings`/`_setting` (precedence flag > wire.json > `SETTINGS_KEYS` default;
  resolved by `_resolve_io`/`_resolve_hosts_ports`). `_suggest()` (ported from omm) prints
  the "Next steps" breadcrumbs; `_table()` renders aligned tables.
- **Live tweaks** — `--set-*` handlers edit ONLY `~/.config/opencode/` (opencode.json +
  re-emitted `plugins/dgx-sampling.js` via `_write_cfg`/`_write_plugin`); declared configs
  are never touched. `AGENT_ROLE` maps an agent name → its preset role for `omw models
  --role`. `[capabilities] concurrency` in a config seeds the team `task_budget` default.
- **Invariant:** tests build their own args namespace and call `oc_sync`/`cmd_*` directly
  (not through `main()`), so keep arg **dest-names** and function signatures stable.
  `__version__` is near the top of the file.

## OpenCode reference (how to modify agents / settings)

Editing this tool means knowing how OpenCode itself consumes the config. This section
is the distilled, **vetted** map so you don't have to re-derive it. Source URLs are
below as ready-to-use WebFetch prompts — fetch them when you need the current detail
(OpenCode moves fast; treat the docs as authoritative over this file if they diverge).

### Docs — WebFetch pointer lines

Paste one of these to an AI tool (or fetch yourself) when you need up-to-date detail:

- use webfetch to find documentation/info on topic **agents & subagents (fields, modes, built-ins, disabling)**: https://opencode.ai/docs/agents/
- use webfetch to find documentation/info on topic **top-level config (provider, model, agent blocks, schema)**: https://opencode.ai/docs/config/
- use webfetch to find documentation/info on topic **permissions (edit/bash/task, ask/allow/deny, --auto)**: https://opencode.ai/docs/permissions/
- use webfetch to find documentation/info on topic **tools (which tools exist, enabling/disabling per agent)**: https://opencode.ai/docs/tools/
- use webfetch to find documentation/info on topic **custom tools**: https://opencode.ai/docs/custom-tools/
- use webfetch to find documentation/info on topic **plugins (directory, hooks, npm plugins)**: https://opencode.ai/docs/plugins/
- use webfetch to find documentation/info on topic **providers (openai-compatible, model capabilities)**: https://opencode.ai/docs/providers/
- config JSON schema (referenced as `$schema`): https://opencode.ai/config.json
- source + issue tracker (bugs / limitations below): https://github.com/sst/opencode  (also served as `github.com/anomalyco/opencode`)

### Agent config fields (from the agents docs)

An agent is an object under the top-level `agent` block (or a markdown file in
`~/.config/opencode/agents/` with YAML frontmatter + body-as-prompt). Fields:

| Field | Meaning |
| ----- | ------- |
| `description` | What it does / when to use it (subagents are auto-picked by this). |
| `mode` | `primary` (Tab-cyclable, can spawn subagents), `subagent` (delegation target), or `all`. |
| `model` | Model ref `provider/model-id`. |
| `temperature` | 0.0–1.0. Only honored if the model declares the `temperature` capability. |
| `top_p` | Alternative diversity control. |
| `prompt` | System prompt, usually a file ref: `"{file:./prompts/x.md}"` **relative to the config file**. |
| `permission` | Per-tool policy (see below). |
| `disable` | `true` deactivates a (built-in or custom) agent. |
| `steps` | Max agentic iterations. |
| `color` | Hex (`#22c55e`) or theme name: `primary`/`secondary`/`accent`/`success`/`warning`/`error`/`info`. |
| `hidden` | Hide a **subagent** from the `@` menu (this tool does NOT set it — workers stay `@`-mentionable). |

- **Built-in agents (current):** primaries `build` (default, all tools) and `plan`
  (restricted); subagents `general`, `explore`, `scout`; hidden system agents
  `compaction`, `title`, `summary`. This tool disables `build`/`plan`
  (`BUILTIN_DISABLE`) and replaces them with `research`/`code`. (Note: several names in
  `MANAGED_AGENTS` — `webdev`, `agentic`, `chat`, `fast`, `architect`, `reason` — are
  **legacy** from earlier versions of this tool, kept only so re-syncs prune them.)
- **Permissions** (`permission` block): keys like `edit`, `bash`, `task`, plus
  `websearch`/`webfetch`; values `allow` / `ask` / `deny`. `task` may be a map
  (`{"*":"deny","agent-code":"allow"}`) to restrict delegation targets — how `team`
  is locked to the hidden workers. CLI `--auto` auto-approves anything not explicitly
  denied.

### Gotchas this tool works around (vetted: source-read + issue tracker)

- Custom OpenAI-compatible models get **no tools at all** unless `tool_call: true` is
  declared on the model — hence it's on by default (`--no-tool-call` to disable).
- Vision needs **both** `modalities` **and** `attachment: true` on the model entry.
- `temperature: false` on a model makes OpenCode send **no** temperature; `topP`/`topK`/
  penalties are **not** config-gated — the only client-side override is the `chat.params`
  plugin hook (why the tool emits one). In `--profiles` mode `temperature` is kept `true`
  so per-agent `temperature`/`top_p` are honored.
- OpenCode **reserves** `build`/`plan`; overrides on those names are ignored (they show
  as "native"). So the tool disables them and moves the startup agent
  (`default_agent`, via `--default-agent`).
- Tab/menu **order is not configurable** — it's internal (upstream feature requests
  open). Config object order is written canonically anyway (visible → hidden → disabled).
- A `team`/orchestrator can receive a local subagent's **empty** last text part instead
  of its result — the `WORKER_PROMPT` on hidden workers forces a non-empty final summary
  to dodge this. (Cloud/Anthropic workers are unaffected; local vLLM workers were.)
- Per-step output is capped at **32k** unless `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`
  is set (`--write-shell-env` appends it).
- Anthropic extended thinking uses `options.thinking = {type:"enabled", budgetTokens:N}`
  (NOT `reasoningEffort`) and **requires `temperature: 1.0`**. Non-`dgx-` (cloud) team
  models have their local `chat_template_kwargs` / `reasoning_effort` / `top_p` stripped
  automatically.
- Anthropic Pro/Max subscription OAuth was removed from OpenCode (ToS) — use an API key
  (`ANTHROPIC_API_KEY` / `/connect`).

### Plugin directory (resolved)

The sampling plugin is written to `<config_dir>/plugins/dgx-sampling.js` (**plural**),
matching the OpenCode docs (https://opencode.ai/docs/plugins/). Earlier builds used the
singular `plugin/` — `oc_sync` now removes a stale `plugin/dgx-sampling.js` on re-sync so
OpenCode doesn't ignore the plugin or load a stale copy. The test suite asserts both
(written to `plugins/`, and the legacy singular file is cleaned up).

### Other items to VERIFY against your installed OpenCode

Source-derived or undocumented mechanisms — confirm before relying on them:

1. **`chat.params` hook.** The sampling plugin uses a `chat.params` hook. That hook is
   **not listed** on the current plugins docs page (which enumerates event hooks like
   `tool.execute.before`). It was vetted from OpenCode source in development, so it is
   likely real-but-undocumented — but confirm it still fires in your version.
2. **`task_budget` agent field.** The team's delegation cap is written as `task_budget`
   on the agent. This is **not** in the documented agent fields (docs list `steps`).
   Treat it as experimental; confirm it's honored.

## How to extend

- **New model:** add a `configs/<key>.toml` **in omodel-manager** (match patterns,
  capabilities, context, per-mode presets). This adapter just consumes it. Confirm the
  declaration with `omodel-wire --verify` against the live endpoint.
- **New role/agent:** add a row to `AGENT_SPECS`; if it should be prunable, add its key
  to `MANAGED_AGENTS`; wire delegation via `TEAM_TARGETS` if it's a worker.
- **New target tool (pi.dev, Claude Code):** add a `TOOLS` entry and a configurator
  analogous to the `oc_*` functions; branch on `tool["sync"]` in `main()`.

## Verify after changes

```bash
python3 -m unittest test_omodel_wire -v   # full offline suite (probes are mocked)
python3 -m py_compile omodel-wire.py      # syntax
python3 omodel-wire.py --dry-run          # inspect config + plugin, writes nothing
python3 omodel-wire.py --profiles --dry-run
```

`test_omodel_wire.py` is the regression suite — run it after ANY change. It monkeypatches
the network probes, so it runs offline and writes only to a temp dir. It asserts roster
integrity, config loading (`test_configs.py`), recipe matching, agent/plugin
construction, provider capability flags, the plugin directory (`plugins/`, plural, per
docs) + stale-`plugin/` cleanup, and a full `oc_sync` round-trip (roster written,
build/plan disabled, `default_agent` set, frontier team model preserved across re-syncs).

Prefer `--dry-run` over touching a real `opencode.json` while iterating. Do not write to
`$HOME` dotfiles from tests — `install_aliases()` / shell-env writers append to the real
rc file (the suite always passes `write_shell_env=False`, which never writes).

## Contributing changes (for AI agents / local models)

If you are an AI agent (e.g. a local model via OpenCode) opening a pull request against this
repo, follow these rules. PRs that ignore them get sent back.

**You open PRs — you do NOT merge.** Never push to `main`, never merge a PR (not even your
own, not even when it's green), and never approve. Your job ends at `gh pr create`; a separate
reviewer runs the checks and merges. A direct push to `main` is a hard-rule violation.

**Only claim what's actually in your diff.** Don't write "added tests" (or describe any change)
in the commit/PR unless it's really in the diff — the reviewer verifies, and false claims get
the PR bounced.

**Scope & hygiene**
- One concern per PR. Small, focused diffs — no unrelated reformatting or churn.
- Branch `feat/<slug>` | `fix/<slug>` | `chore/<slug>`; Conventional-Commit messages.
- Never touch `LICENSE`, `__version__` / release tags, or `.github/` CI unless explicitly asked.

**Before you push (required)**
- `python -m py_compile omodel-wire.py` — must pass.
- `python -m unittest` — the full offline suite must pass. Add or adjust tests for your change.
- Update `CHANGELOG.md` under `[Unreleased]` for anything user-facing.
- Paste the py_compile + unittest output into the PR body (the template asks for it).

**Respect the invariants above:** stdlib-only, single-script, LF endings, kebab/snake naming,
`plugins/` (plural), and model configs owned by omodel-manager (never reintroduce
`model_recipes.json`).

**Open the PR** with `gh pr create`, fill in the template, then **stop** — do not merge; wait for review.

### For the reviewer (Claude)

If the user asks to **review / approve / merge open PRs**, read **`REVIEW.md`** and execute it
against every open PR: run the checks, make critical fixes, squash-merge the ones that pass the
bar, and leave anything risky open with a change-request review.
