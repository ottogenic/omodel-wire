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

`omw` is a verb-first CLI (like its sibling `omm`). Run it with no arguments for a
guided home screen; each command suggests the next step.

```bash
# 1) add the `omw` shell alias (re-open your shell afterwards)
python3 omodel-wire.py shell-init

# 2) see status + suggested next steps
omw

# 3) sync the OpenCode agent roster from the model configs
omw sync

# 4) inspect what you got
omw agents            # primary agents (research/code/agent/team)
omw models qwen       # a model's per-role sampling table

# 5) experiment live (edits are disposable — `omw sync` resets to known-good)
omw models qwen --role code --set-temperature 0.5
omw agents team --set-work-budget 4
omw audit             # show what has drifted from the known-good configs
```

Common settings you don't want to retype live in `~/.config/otools/wire.json` — set
them once and every command picks them up (precedence: **flag > wire.json > default**):

```bash
omw config --set team_model anthropic/claude-opus-4-8
omw config --set default_agent code
omw config                       # show resolved settings + their source
```

By default `omw sync` discovers the hosts in the shared `~/.config/otools/hosts` store
(managed by omm's `install`/`ps`), on ports `8000/8001/8002` — so `omm install user@ip
dgx-3` makes the box visible here too. Override with `omw sync --hosts` / `--ports`.

---

## What it does

1. **Detect** which agentic-dev tools are installed (OpenCode today).
2. **Discover** live models via each `/v1/models` endpoint (fast), then read each
   model's capabilities (vision / reasoning / thinking-knob) and per-mode sampling from
   **omodel-manager**'s declared `configs/*.toml` — **no slow per-model probing** on the
   sync path. (`omw verify` re-runs the real probes to check a declaration.)
3. **Write** the OpenCode config: providers, models (with `tool_call`, `attachment` +
   `modalities` for vision, context/output limits), a `(model, agent)`-keyed `chat.params`
   plugin that pins sampling, the agent roster, and Ctrl+T thinking variants.

### The agent roster (`omw sync`)

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
`omw sync` **disables** the native pair and ships `research`/`code` in their place,
pointing the startup agent at `--default-agent` (default `code`, or wire.json
`default_agent`). `omw sync --keep-builtins` opts out.

Put the `team` orchestrator on a frontier model with `omw sync --team-model` — or persist
it with `omw config --set team_model anthropic/claude-opus-4-8`; workers stay local.
`--team-reasoning` (`low`/`medium`/`high` → 10000/24000/32000 thinking-budget tokens) and
`--team-task-budget N` are preserved across re-syncs. If you don't set a budget, `sync`
defaults it to the worker model's declared `concurrency` (its `max-num-seqs`) so it won't
spawn more parallel workers than the server has sequence slots.

### Model configs

Curated capabilities + per-mode sampling live in **omodel-manager**'s `configs/*.toml`
(one commented, `cat`/`vi`-friendly file per model, keyed by name-match). When a
discovered model matches, its per-mode presets become the agents above with sampling
enforced by the plugin. No match → generic `code`/`reason`. **Add a model by dropping a
`.toml` in omodel-manager's `configs/`** — see that repo's `configs/README.md`.
`omw sync --no-recipes` opts out. Configs ship for Qwen3.6-27B, Qwen3.6-35B-A3B,
NVIDIA-Nemotron-3-Super, GLM-4.7-Flash, and the Gemma-4 pair. `omw verify` checks a live
model against its config.

---

## Commands

| Command | What it does |
| ------- | ------------ |
| `omw` | Home screen: status + suggested next steps. |
| `omw sync` | Build/refresh the OpenCode roster from the model configs. Carries all the sync knobs (`--hosts/--ports`, `--team-model/--team-task-budget/--team-reasoning`, `--default-agent/--keep-builtins`, `--web-search`, `--sampling …`, `--dry-run`, …). This is the "reset to known-good" button. |
| `omw agents [<name>]` | List primary agents; show one in detail. |
| `omw agents <name> --set-model REF` | Live-set an agent's model. |
| `omw agents team --set-work-budget N` | Live-set the team's delegation budget. |
| `omw subagents [<name>] [--set-model REF]` | List/inspect hidden workers; no name → set all workers. |
| `omw models [<name>]` | List models; `<name>` shows the per-role sampling table. |
| `omw models <name> --role R --set-temperature T` / `--set-thinking B` | Live-tweak one role's sampling/thinking. |
| `omw audit` | Offline side-by-side of the live config vs the known-good configs; flags drift. |
| `omw verify` | Probe live endpoints and compare to the declared capabilities (slow, opt-in). |
| `omw config [--set KEY VAL] [--edit] [--path]` | Show or persist settings in `~/.config/otools/wire.json`. |
| `omw detect` | Report which agentic-dev tools are installed. |
| `omw shell-init` | Install the `omw` shell alias. |

Every `--set-*` edit touches **only** `~/.config/opencode/` (opencode.json + the sampling
plugin) — the declared configs stay pristine, so `omw sync` always restores a known-good
state and `omw audit` shows exactly what you've changed. Run `omw <command> --help` for
the full flag list of any command.

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
