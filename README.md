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
| `team`          | Tab        | orchestrator | delegation-only (no tools) |
| `agent-plan`    | hidden     | reason   | read-only + web        |
| `agent-code`    | hidden     | code     | full                   |
| `agent-instruct`| hidden     | instruct | full                   |

`team` is a **delegation-only** orchestrator: every tool category is denied
(`read`/`grep`/`glob`/`list`/`edit`/`bash`/`webfetch`/`websearch`), so the only thing it
can do is **delegate** to the three hidden `agent-*` workers via `task` (its
`permission.task` is limited to them). The hidden workers carry a
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
