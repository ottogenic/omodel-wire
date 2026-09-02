# omodel-wire

Discover omodel-manager card, node, and cluster deployments and add them to OpenCode.

`omodel-wire` is a Python 3.11+ standard-library CLI. It probes the hosts registered by
`omodel-manager` in `~/.config/otools/hosts` (or explicit hosts from `wire.json`/CLI),
then refreshes providers and native Build/Plan defaults in OpenCode's global config.
Discovery is independent of which machine launched or stopped a model.

It does not install custom agents, subagents, prompts, skills, permissions, MCP
servers, web search, or workflow state. Those concerns belong in the separate
`omodel-pipeline` project. It only profiles OpenCode's built-in `build` and `plan`
agents from the model TOMLs.

## Quick Start

```bash
# Install the omw shell alias, then start a new shell.
python3 omodel-wire.py shell-init

# Preview the discovered providers and models.
omw sync --dry-run

# Update ~/.config/opencode/opencode.json.
omw sync

# Restart OpenCode and select an otools-* model.
opencode
```

Sync probes this machine plus each registered host/port combination and includes only endpoints whose
`/v1/models` responds. Register the same serving hosts on every controller where you want
to run `omw sync`; no deployment-state file needs to be copied between machines.

Persist legacy fallback discovery settings in `~/.config/otools/wire.json`:

```bash
omw config --set hosts 192.168.50.101,192.168.50.102
omw config --set ports 8000,8001
omw config
```

## Sync Ownership

`omw sync`:

- Adds one provider per live host/port endpoint.
- Adds every model currently returned by that endpoint's `/v1/models` response.
- Replaces stale `otools-*` providers and prunes old `dgx-*` / `dgx` / `dgxN` keys.
- Maps native Build to each model's TOML `build` preset and native Plan to its
  `plan` preset while keeping their config blocks free of model-specific sampling.
- Writes `plugins/dgx-sampling.js` so temperature, `top_p`, `top_k`, penalties,
  native reasoning options, and output limits follow the current model and built-in agent.
- Preserves unrelated agents, agent fields, OpenCode settings, and providers.
- Removes stale managed top-level model references and old disabled Build/Plan stubs.
- Refuses to write when no endpoint responds unless `--allow-empty` is explicit.
- Creates `opencode.json.bak` before replacing an existing config.

Model capability declarations are read recursively from the sibling
`omodel-manager` project's `configs/` tree, including `configs/card/`, `configs/node/`,
and `configs/cluster/`. They supply context/output limits, tool calling, vision,
reasoning, model-native thinking variants, and Build/Plan sampling. Use
`--no-recipes` to omit those declarations or `--configs PATH` to select another
config directory. Use `--build-model` and `--plan-model` to select different live
defaults; otherwise a still-live existing choice is preserved, then the first live
model with the required preset is used. If a live preset requires more than OpenCode's
default 32k output cap, sync prints the required environment setting;
`--write-shell-env` persists it.

Each live served model is matched against the manager-owned TOML declarations by model ID.

## Commands

| Command | Purpose |
| --- | --- |
| `omw sync` | Sync live managed card/node/cluster endpoints and native Build/Plan TOML defaults. |
| `omw verify` | Compare live endpoint capabilities with model declarations. |
| `omw config` | Show config paths, legacy fallback discovery, and proxy settings. |
| `omw proxy` | Temporarily route managed endpoint traffic through the local debug logger. |
| `omw cleanup` | Remove old OpenCode session data. |
| `omw detect` | Report installed agentic development tools. |
| `omw shell-init` | Install the `omw` shell alias. |

Run `omw <command> --help` for command-specific options.

## Safety

- Preview writes with `omw sync --dry-run`.
- Without `--allow-empty`, an offline fleet leaves the current config untouched.
- `--allow-empty` deliberately removes managed providers while preserving all
  unrelated config.
- Restart OpenCode after a successful sync; configuration is loaded only at startup.

See [AGENTS.md](AGENTS.md) for development invariants and
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution checks.
