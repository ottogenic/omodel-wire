# AGENTS.md — `omodel-wire`

Single-file, stdlib-only Python CLI (`omodel-wire.py`) that (1) detects installed
agentic-dev tools and (2) syncs OpenAI-compatible model endpoints into their configs.
OpenCode is the only wired-up target today. Curated model settings (capabilities +
per-mode sampling) live in **omodel-manager**'s recursive `configs/{card,node,cluster}/`
tree; this tool **consumes**
them — it does not own them.

This file is intentionally short: the invariants below bind **every** task; anything
task-specific lives in a **skill** (see the index at the bottom) that loads on demand.

## Invariants — never violate

- **Stdlib only.** No third-party imports (uses `tomllib`, stdlib 3.11+). Bare `python3` 3.11+.
- **Single script, no config data.** The tool is `omodel-wire.py`; the generic model configs
  live in omodel-manager, read via `--configs` / `$OMODEL_CONFIGS` / sibling
  `../omodel-manager/configs`. Never copy them here or reintroduce `model_recipes.json` /
  `DEFAULT_RECIPES` (retired in 0.2.0).
- **Idempotent & field-scoped.** `omw sync` owns managed `otools-*` and migration
  `dgx-*` provider entries, the
  native Build/Plan fields `model`/`temperature`/`top_p`/`options`, and the generated
  `plugins/dgx-sampling.js`. Preserve all unrelated agents, agent fields, permissions,
  built-in providers, and user settings.
- **Discovery is controller-local and live.** `omw sync` probes this machine plus hosts registered by
  `omodel-manager` in `~/.config/otools/hosts`, or explicit `wire.json`/CLI hosts and ports.
  It must never depend on which machine launched a model or on `deployments.json`.
- **No custom workflow config.** `tool_call` is declared on every discovered model and
  Build/Plan consume omodel-manager's code/reason presets. Never emit custom agents,
  prompts, skills, or state-machine plugins; those belong to omodel-pipeline.
- **Publishing follows the maintainer's requested destination.** This is a single-maintainer repo;
  when publishing is requested, the default is a tested direct push to `main`. Open a PR only when
  the maintainer explicitly requests one. Never force-push or bypass required checks.
- **Cross-platform paths** (WSL/Linux + Windows): use `os.path` / `expanduser`; never assume
  a POSIX-only home.

## Working agreement

- **Todo tracking.** For any task with >2 discrete steps, use your harness's todo tool (not a
  chat-markdown plan). One item `in_progress` at a time; mark `completed` as you go.
- **Plan approval is for the whole plan.** After a "go"/"proceed", execute all remaining steps
  without stopping to re-ask between them.
- **Prefer `--dry-run`** while iterating; never write `$HOME` dotfiles from tests. Tests build
  their own args namespace and call `oc_sync`/`cmd_*` directly — keep arg **dest-names** and
  function signatures stable.
- **Solo git by default.** Work in the canonical checkout. Use a branch/worktree only when the
  maintainer requests it, concurrent work is actually active, or isolation protects a working
  version. Start clean, stage **explicit paths** (never `git add -A`), and inspect status, diff,
  and recent history before committing or pushing. Test via `python3 ./omodel-wire.py …`; when in
  a worktree, pass `--configs <path>` explicitly rather than relying on the sibling default.
- **Publishing is not complete until the canonical checkout is current.** After branch/worktree
  work, integrate it into the canonical checkout (the one the `omw` alias targets — resolve it,
  don't assume a path), push the requested destination, and verify canonical `HEAD`, local `main`,
  and `origin/main` agree. Do not tell the maintainer to run `omw` while the alias-target checkout
  is behind the published change.

## Skills — load the one matching your task first

Skill bodies load on demand; load the match before you start (OpenCode also surfaces them
via the `skill` tool).

| To… | Skill |
| --- | --- |
| Make a code change (file layout, how to extend, checks to run) | **`code-changes`** |
| Modify how the tool generates OpenCode config (agent/permission fields, vetted gotchas, plugin dir, doc pointers) | **`opencode-reference`** |
| Prove the generated sampling actually reaches the model server | **`validate-opencode`** |
| Open an explicitly requested pull request | **`open-a-pr`** |
| Review / approve / merge an explicitly requested PR | delegate to **`agent-review`** (see the note below the table) |

Other docs (read directly when relevant): `README.md` (user-facing), `CHANGELOG.md`.

**When a PR is explicitly requested**, reviews are handled by **`agent-review`**, whose global
role is extended here by
`agent-review-extend`. It checks tested work against the caller's criteria and `REVIEW.md`,
classifies findings, and merges a PR only when authorized and clean. Delegate by name through the
`task` tool; fix one blocker/regression at a time and reuse the same reviewer `task_id`.
