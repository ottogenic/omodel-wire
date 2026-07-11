---
name: code-changes
description: Make a code change to the omodel-wire.py script — the top-to-bottom file/section map, how to extend (new model config, role/agent, or target tool), and the verify-after-changes checks. Use before editing omodel-wire's Python.
---

# Making code changes to `omodel-wire.py`

Read this before editing `omodel-wire.py`. For how OpenCode itself consumes the config this tool generates, see the **opencode-reference** skill.

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
- **Debug proxy** — `utils/omw_proxy.py` (loaded by path as `proxy`) is a threaded stdlib
  HTTP proxy: **path-prefix routing** (`/<provider>/…` → real upstream via
  `proxy_routes.json`, re-read per request), **SSE streaming passthrough** that tees the
  stream into the log, short 7-char ids, flat `proxy_logs/`. Importable helpers:
  `proxy_logs_dir` / `find_pair` / `build_curl` / `render_read`. The `cmd_proxy_*` handlers
  in `omodel-wire.py` rewrite live `dgx-` provider baseURLs to the proxy and back (backup
  `.proxy-bak`, map `proxy_routes.json`, pid `.omw-proxy.pid`); daemon stdout goes to a
  **file** (never a PIPE — an unread PIPE stalls the proxy). Streaming + threading + the
  file redirect are the fixes for the earlier "proxy dies" behavior.
- **Invariant:** tests build their own args namespace and call `oc_sync`/`cmd_*` directly
  (not through `main()`), so keep arg **dest-names** and function signatures stable.
  `__version__` is near the top of the file.

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
build/plan disabled, `default_agent` set, team model uses resolved preference or preserves frontier only when no preference resolves).

Prefer `--dry-run` over touching a real `opencode.json` while iterating. Do not write to
`$HOME` dotfiles from tests — `install_aliases()` / shell-env writers append to the real
rc file (the suite always passes `write_shell_env=False`, which never writes).
