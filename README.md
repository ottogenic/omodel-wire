# omodel-wire

Discover OpenAI-compatible models on DGX Spark nodes and add them to OpenCode.

`omodel-wire` is a Python 3.11+ standard-library CLI. It reads the host inventory
shared with `omodel-manager`, probes each `/v1/models` endpoint, and refreshes the
managed DGX providers and native Build/Plan defaults in OpenCode's global config.

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

# Restart OpenCode and select a dgx-* model.
opencode
```

By default, hosts come from `~/.config/otools/hosts`, which is maintained by
`omodel-manager`, and ports `8000`, `8001`, and `8002` are probed. Override either
for one run:

```bash
omw sync --hosts 192.168.50.101,192.168.50.102 --ports 8000,8001
```

Persist common settings in `~/.config/otools/wire.json`:

```bash
omw config --set hosts 192.168.50.101,192.168.50.102
omw config --set ports 8000,8001
omw config
```

## Sync Ownership

`omw sync`:

- Adds one OpenCode provider per live host and port.
- Adds every model currently returned by that endpoint's `/v1/models` response.
- Replaces stale providers named `dgx-*` and legacy `dgx`/`dgxN` keys.
- Maps native Build to each model's TOML `code` preset and native Plan to its
  `reason` preset.
- Writes `plugins/dgx-sampling.js` so `top_k`, `min_p`, penalties, native reasoning
  options, and output limits follow the current model and built-in agent.
- Preserves unrelated agents, agent fields, OpenCode settings, and providers.
- Removes stale managed top-level model references and old disabled Build/Plan stubs.
- Refuses to write when no endpoint responds unless `--allow-empty` is explicit.
- Creates `opencode.json.bak` before replacing an existing config.

Model capability declarations are read from the sibling `omodel-manager` project's
`configs/*.toml` files. They supply context/output limits, tool calling, vision,
reasoning, model-native thinking variants, and Build/Plan sampling. Use
`--no-recipes` to omit those declarations or `--configs PATH` to select another
config directory. Use `--build-model` and `--plan-model` to select different live
defaults; otherwise a still-live existing choice is preserved, then the first live
model with the required preset is used. If a live preset requires more than OpenCode's
default 32k output cap, sync prints the required environment setting;
`--write-shell-env` persists it.

No local router is required: each DGX host appears as a separate provider because
an OpenCode provider has one `baseURL`.

## Commands

| Command | Purpose |
| --- | --- |
| `omw sync` | Sync live DGX providers/models and native Build/Plan TOML defaults. |
| `omw verify` | Compare live endpoint capabilities with model declarations. |
| `omw config` | Show or persist discovery paths, hosts, ports, and proxy settings. |
| `omw proxy` | Temporarily route DGX traffic through the local debug logger. |
| `omw cleanup` | Remove old OpenCode session data. |
| `omw detect` | Report installed agentic development tools. |
| `omw shell-init` | Install the `omw` shell alias. |

Run `omw <command> --help` for command-specific options.

## Safety

- Preview writes with `omw sync --dry-run`.
- Without `--allow-empty`, an offline fleet leaves the current config untouched.
- `--allow-empty` deliberately removes managed DGX providers while preserving all
  unrelated config.
- Restart OpenCode after a successful sync; configuration is loaded only at startup.

See [AGENTS.md](AGENTS.md) for development invariants and
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution checks.
