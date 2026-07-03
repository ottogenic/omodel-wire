# omodel-wire

Sync OpenAI-compatible model endpoints into your agentic-dev tools — and optionally
build a curated, per-model **agent roster** on top of them.

`omodel-wire` runs on your laptop, probes the model servers you point it at (e.g.
vLLM / SGLang on your DGX nodes), figures out what each model can actually do
(vision, reasoning, which thinking knob it honors), and writes the matching config.
Today it targets [OpenCode](https://opencode.ai); the detection and "configurator"
layers are pluggable so other tools (pi.dev, Claude Code, …) can be added later.

- **Stdlib only.** No `pip install`, no dependencies — just Python 3.
- **Idempotent.** Re-run any time; it merges into your existing config and preserves
  the settings it shouldn't touch (see [Safety](#safety)).
- **Config-driven, no probing.** Model capabilities + per-mode sampling are DECLARED
  in **omodel-manager**'s `configs/*.toml` (read via `--configs` / `$OMODEL_CONFIGS` /
  sibling `../omodel-manager/configs`). No slow per-model vision/reasoning probes on the
  sync path — use `--verify` to check a declaration against a live endpoint.

---

## Quick start

```bash
# 1) add the `omw` shell alias (re-open your shell afterwards)
python3 omodel-wire.py --install-aliases

# 2) detect installed tools + sync discovered models (server decides sampling)
omw

# 3) preview without writing anything
omw --dry-run

# 4) build the full agent roster for reasoning models
omw --profiles

# 5) a frontier orchestrator that delegates to your local workers
omw --profiles \
     --team-model anthropic/claude-opus-4-8 --team-reasoning high --team-task-budget 4 \
     --web-search exa --write-shell-env
```

By default the tool probes the hosts registered in the shared
`~/.config/otools/hosts` store (managed by omodel-manager's `install`/`ps`), on ports
`8000/8001/8002` — so `omm install user@ip dgx-3` makes the box visible here too. If that
file is absent it falls back to a built-in list. Override either with `--hosts` / `--ports`.

---

## What it does

1. **Detect** which agentic-dev tools are installed (OpenCode today).
2. **Probe** each `/v1/models` endpoint, then per model:
   - a **vision** check — actually sends a blue test image and confirms the answer
     mentions "blue" (a text-only server that silently ignores images won't fool it);
   - with `--profiles`, a **reasoning** check plus which thinking knob it honors
     (`chat_template_kwargs.enable_thinking` vs `reasoning_effort`).
3. **Write** the OpenCode config: providers, models (with `tool_call`, `attachment` +
   `modalities` for vision, context/output limits), a `chat.params` plugin that pins
   sampling, and — with `--profiles` — the agent roster and Ctrl+T thinking variants.

### The agent roster (`--profiles`)

Built from `AGENT_SPECS`, with per-mode sampling from omodel-manager's configs:

| Agent           | Visibility | Role     | Permissions            |
| --------------- | ---------- | -------- | ---------------------- |
| `research`      | Tab        | reason   | read-only + web        |
| `code`          | Tab        | code     | edit/bash **ask** + web |
| `agent`         | Tab        | agent    | full (edit/bash allow) |
| `team`          | Tab        | orchestrator | read-only, delegates |
| `agent-plan`    | hidden     | reason   | read-only + web        |
| `agent-code`    | hidden     | code     | full                   |
| `agent-instruct`| hidden     | instruct | full                   |

`team` is a read-only orchestrator that **delegates** to the three hidden
`agent-*` workers (`permission.task` is limited to them). The hidden workers carry a
"worker prompt" that forces a non-empty final summary — a workaround for an OpenCode
bug where the orchestrator otherwise receives an empty result. The visible agents
stay clean (no worker prompt) so they're pleasant to drive by hand.

OpenCode reserves the names `build`/`plan` and ignores overrides on them, so
`--profiles` **disables** the native pair and ships `research`/`code` in their place,
setting `--default-agent` (default `code`) since `build` used to be the startup agent.
Use `--keep-builtins` to opt out.

Put the `team` orchestrator on a frontier model with `--team-model`
(e.g. `anthropic/claude-opus-4-8`); workers stay local. `--team-reasoning`
(`low`/`medium`/`high` → 10000/24000/32000 thinking-budget tokens) and
`--team-task-budget N` are preserved across re-syncs.

### Model configs

Curated capabilities + per-mode sampling live in **omodel-manager**'s `configs/*.toml`
(one commented, `cat`/`vi`-friendly file per model, keyed by name-match). When a
discovered model matches, its per-mode presets become the agents above with sampling
enforced by the plugin. No match → generic `code`/`reason`. **Add a model by dropping a
`.toml` in omodel-manager's `configs/`** — see that repo's `configs/README.md`.
`--no-recipes` opts out. Configs ship for Qwen3.6-27B, Qwen3.6-35B-A3B,
NVIDIA-Nemotron-3-Super, and GLM-4.7-Flash. `--verify` checks a live model against its config.

---

## Common options

| Flag | Purpose |
| ---- | ------- |
| `--dry-run` | Print the resulting config + plugin; write nothing. |
| `--detect-only` | Just report which tools are installed. |
| `--hosts` / `--ports` | Comma-separated endpoints to probe. |
| `--profiles` | Build the agent roster + thinking variants. |
| `--sampling {server-default,fixed,opencode-default}` | Who decides sampling (default `server-default` = let the model server decide). `fixed` pins values via `--temperature`/`--top-p`/`--top-k`/`--presence-penalty`/`--frequency-penalty`. |
| `--vision-probe-all` | Image-probe every model, not just name-matched ones. |
| `--no-vision-probe` | Skip vision detection entirely. |
| `--web-search {none,exa,mcp}` | Expose a web-search tool. `exa` needs `OPENCODE_ENABLE_EXA` (see `--enable-exa-shell`); `mcp` adds a server via `--mcp-command`/`--mcp-url`. |
| `--team-model REF` | Put `team` on a specific (often frontier) model. |
| `--team-reasoning {low,medium,high}` | Extended-thinking budget for an Anthropic team model. |
| `--team-task-budget N` | Cap delegation calls per session. |
| `--default-agent NAME` | Startup agent when native build/plan are disabled (default `code`). |
| `--keep-builtins` | Keep native build/plan instead of replacing them. |
| `--configs PATH` | omodel-manager's configs dir (or set `$OMODEL_CONFIGS`; default sibling). |
| `--no-recipes` | Ignore the configs (generic behavior). |
| `--audit` | Offline side-by-side of the live OpenCode config vs the omodel-manager configs, per model + agent; highlights sampling drift and suggests `--profiles`. Writes nothing. |
| `--verify` | Probe live endpoints and compare to the declared configs; writes nothing. |
| `--install-aliases` | Add the `omw` shell alias, then exit. |
| `--write-shell-env` | Append needed OpenCode env vars (Exa, >32k output) to your shell rc. |
| `--version` | Print version and exit. |

Run `python3 omodel-wire.py --help` for the complete list.

---

## Safety

- **Non-destructive merge.** The tool merges into your existing `opencode.json`; it
  only prunes agents it manages (`MANAGED_AGENTS`), leaving your hand-written agents
  and providers alone.
- **Anthropic config is preserved.** A previously-set frontier team model — and its
  reasoning budget and task budget — survive re-syncs even without re-passing the flags,
  so your cloud config isn't wiped.
- **Preview first.** `--dry-run` shows exactly what would be written.
- Use `--allow-empty` only if you deliberately want to write when nothing was discovered.

---

## Requirements

- Python 3 (standard library only).
- Reachable OpenAI-compatible endpoints.
- OpenCode installed (the config is still written if it isn't, so you can stage it).

See [AGENTS.md](AGENTS.md) for the architecture and extension points, and
[CONTRIBUTING.md](CONTRIBUTING.md) for the branching/commit workflow.
