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
omw config --set default_agent code
omw config --set web_search exa
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

| Agent             | Visibility | Role         | Permissions             |
| ----------------- | ---------- | ------------ | ----------------------- |
| `research`        | Tab        | reason       | read-only + web         |
| `code`            | Tab        | code         | edit/bash **ask** + web |
| `agent`           | Tab        | agent        | full                    |
| `team`            | Tab        | orchestrator | delegation-only         |
| `agent-research`  | hidden     | research     | read-only + web         |
| `agent-code`      | hidden     | implementation | full                  |
| `agent-test`      | hidden     | verification | read + checks, no edits |
| `agent-instruct`  | hidden     | mechanical   | full                    |
| `agent-architect` | hidden     | architecture | read-only               |
| `agent-review`    | hidden     | review       | read + checks, no edits |

`team` has no workspace tools; it can only delegate to the six hidden workers through `task`.
Team and every worker load a visible global role skill from
`~/.config/opencode/skills/agent-*/SKILL.md`; their system prompts contain only the bootstrap.
A repo can add `.agents/skills/agent-<role>-extend/SKILL.md` to append role guidance, or
`agent-<role>-override` to replace both the global role and extend skill. Override wins; otherwise
the global role loads before extend. Visible `research`/`code`/`agent` remain prompt-free and keep
OpenCode's model-specific default prompts.

The standard flow sends simple work to `agent-code`, routes medium/high-risk work through
`agent-architect` for research, plan, and acceptance criteria, sends completed implementation to
`agent-test` for broad/scripted verification, then sends the tested packet to `agent-review`.
Only findings classified as blockers or regressions enter the one-at-a-time fix loop. Once verified,
Team asks whether to create a PR and perform PR review.

OpenCode reserves the names `build`/`plan` and ignores overrides on them, so
`omw sync` **disables** the native pair and ships `research`/`code` in their place,
pointing the startup agent at `--default-agent` (default `code`, or wire.json
`default_agent`). `omw sync --keep-builtins` opts out.

Agent model preferences live in local `default_models.json`. `--team-task-budget N` is preserved
across re-syncs; without it, sync defaults the budget from the worker model's declared concurrency
(`max-num-seqs`).

### Disabled providers by default

By default, the built-in **OpenCode** and **Hugging Face** providers are **disabled** using
OpenCode's `disabled_providers` array. This keeps the provider definitions in your config
but prevents users from accidentally selecting them.

To enable these providers (e.g., if you're using their API keys), pass `--add-default-providers`:

```bash
omw sync --add-default-providers
```

This is idempotent: re-running without the flag re-applies the disabled list; with the flag,
providers are enabled. The providers remain managed by `omw` and will be pruned on the next
sync unless they're also running on your DGX endpoints (in which case they're kept as DGX
providers with their own `dgx-*` keys).

### Model configs

Curated capabilities + per-mode sampling live in **omodel-manager**'s `configs/*.toml`
(one commented, `cat`/`vi`-friendly file per model, keyed by name-match). When a
discovered model matches, its per-mode presets become the agents above with sampling
enforced by the plugin. No match → generic `code`/`reason`. **Add a model by dropping a
`.toml` in omodel-manager's `configs/`** — see that repo's `configs/README.md`.
`omw sync --no-recipes` opts out. Configs ship for Qwen3.6-27B, Qwen3.6-35B-A3B,
NVIDIA-Nemotron-3-Super, GLM-4.7-Flash, and the Gemma-4 pair. `omw verify` checks a live
model against its config.

### GitHub Copilot target (experimental)

`omw sync --target copilot` writes the same roster to the **GitHub Copilot CLI** instead of (or,
with `--target all`, alongside) OpenCode. It writes to Copilot's config home — the roster as
`.agent.md` files under `agents/`, `settings.json` (model + `includeCoAuthoredBy: false`), and an
`otools-copilot.env`/`.ps1` snippet. The home is **auto-detected**: `~/.copilot` on native
Windows/macOS/Linux, and — when you run `omw` in **WSL** but Copilot is installed on Windows — the
Windows-side `C:\Users\<you>\.copilot` (via `/mnt/c`). `$COPILOT_HOME` or `--copilot-home` overrides.

**All three Copilot surfaces read `~/.copilot/agents/`**, so the same roster shows up in the
**CLI/TUI** (`copilot` → `/agent`), **VS Code** (Copilot Chat → the agents dropdown; reload the
window after syncing), and the **desktop app** (agent picker). The roster + settings are written even
with the DGX offline (so you can see them land); the endpoint is wired only when a model is live —
until then the agents run on Copilot's default hosted model.

Two things differ from OpenCode by Copilot's design: its CLI takes a **single custom endpoint**, so
the whole roster runs on **one DGX model** (the endpoint is env-only — hence the snippet you `source`
before `copilot`); and delegation is runtime-global by description with **no subagent nesting**, so
primaries become top-level agents, the workers become subagents, and `agent-review` is
explicit-invoke-only. A multi-model **VS Code** target is a planned follow-up.

```bash
omw sync --target copilot            # write the Copilot roster + settings + env snippet
source ~/.copilot/otools-copilot.env # point Copilot at your DGX endpoint (bash/zsh)
copilot --agent code                 # run it
```

---

## Commands

| Command | What it does |
| ------- | ------------ |
| `omw` | Home screen: status + suggested next steps. |
| `omw sync` | Build/refresh the OpenCode roster from the model configs. Carries the sync knobs (`--hosts/--ports`, `--team-task-budget`, `--default-agent/--keep-builtins`, `--web-search`, `--sampling …`, `--dry-run`, …). This is the "reset to known-good" button. |
| `omw sync --add-default-providers` | Enable built-in OpenCode and Hugging Face providers (by default, these are disabled and their models are hidden from the picker). |
| `omw agents [<name>]` | List primary agents; show one in detail. |
| `omw agents <name> --set-model REF` | Live-set an agent's model. |
| `omw agents team --set-work-budget N` | Live-set the team's delegation budget. |
| `omw subagents [<name>] [--set-model REF]` | List/inspect hidden workers; no name → set all workers. |
| `omw models [<name>]` | List models; `<name>` shows the per-role sampling table. |
| `omw models <name> --role R --set-temperature T` / `--set-thinking B` | Live-tweak one role's sampling/thinking. |
| `omw audit` | Offline side-by-side of the live config vs the known-good configs; flags drift. |
| `omw verify` | Probe live endpoints and compare to the declared capabilities (slow, opt-in). |
| `omw config [--set KEY VAL] [--edit] [--path]` | Show or persist settings in `~/.config/otools/wire.json`. |
| `omw config --set-gh-token-coder` / `--set-gh-token-reviewer` | Store the two GitHub identity tokens (see below). Prompts with hidden input; pass `none` to clear. |
| `omw proxy on [<model>]` / `off [<model>]` | Route live models through a local debug proxy that logs every request/response (no name = all; a name = one). |
| `omw proxy read <id>` / `replay <id> [--output-curl]` / `status` | Read a logged exchange (colored, sectioned); re-issue it directly to the API (or emit curl); show proxy state. |
| `omw detect` | Report which agentic-dev tools are installed. |
| `omw shell-init` | Install the `omw` shell alias. |

Every `--set-*` edit touches **only** `~/.config/opencode/` (opencode.json + the sampling
plugin) — the declared configs stay pristine, so `omw sync` always restores a known-good
state and `omw audit` shows exactly what you've changed. Run `omw <command> --help` for
the full flag list of any command.

### GitHub identity (two tokens, then you're done)

So agents commit and open PRs under a **shared bot** account while **you** review and
merge them — real two-party review — `omw sync` writes an OpenCode plugin
(`plugins/otools-git-identity.js`) that hands each shell a GitHub token. You set two tokens
once:

```bash
omw config --set-gh-token-coder      # paste the BOT account's token (hidden input)
omw config --set-gh-token-reviewer   # paste YOUR account's token
```

Each is a GitHub PAT with **Contents: Read/Write** + **Pull requests: Read/Write** on the
repos. They're stored (chmod 600) at `~/.config/otools/gh_token_{coder,reviewer}`. Coding
agents (research/team/code/agent/workers) commit and open PRs as the **coder** (bot);
`agent-review` approves and merges as the **reviewer** (you). github.com git is routed over
HTTPS+token automatically — no SSH keys, no `~/.gitconfig` edits. Until both are set,
`omw sync` warns and agents fall back to your logged-in `gh`. Reload OpenCode after setting
them. Verify from inside OpenCode: `gh api user --jq .login` (→ bot) and
`GH_TOKEN="$GH_TOKEN_REVIEWER" gh api user --jq .login` (→ you).

### Debugging with the proxy

`omw proxy on` transparently inserts a local proxy between OpenCode and your model
endpoints — no config guesswork, it reads the live provider list and rewrites those
baseURLs, mapping each back to the real upstream. It **streams SSE through untouched**
(so the TUI keeps working) while teeing the full request/response into `proxy_logs/`
(short 7-char ids, one `<id>_req.json`/`<id>_res.json` pair each, plus `index.jsonl`).
Then reload OpenCode, reproduce the issue, and inspect:

```bash
omw proxy on                       # or: omw proxy on qwen3.6-35b   (one model)
# ... reproduce in OpenCode ...
omw proxy read <id>                # human-readable, colored view of one exchange
omw proxy replay <id>              # re-run that exact request straight at the API
omw proxy replay <id> --output-curl   # ... or copy-paste it as curl
omw proxy off                      # restore config, stop the proxy
```

---

## Safety

- **Non-destructive merge.** The tool merges into your existing `opencode.json`; it
  only prunes agents it manages (`MANAGED_AGENTS`), leaving your hand-written agents
  and providers alone.
- **Disabled providers by default.** The built-in OpenCode and Hugging Face providers
  are disabled by default using the `disabled_providers` array. Pass `--add-default-providers`
  to enable them (e.g., if you're using their API keys).
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
