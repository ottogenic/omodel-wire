# AGENTS.md — `omodel-wire`

Single-file, stdlib-only Python CLI (`omodel-wire.py`) that (1) detects installed
agentic-dev tools and (2) syncs OpenAI-compatible model endpoints into their configs.
OpenCode is the only wired-up target today. Curated model settings (capabilities +
per-mode sampling) live in **omodel-manager**'s `configs/*.toml`; this tool **consumes**
them — it does not own them.

This file is intentionally short: the invariants below bind **every** task; anything
task-specific lives in a **skill** (see the index at the bottom) that loads on demand.

## Invariants — never violate

- **Stdlib only.** No third-party imports (uses `tomllib`, stdlib 3.11+). Bare `python3` 3.11+.
- **Single script, no config data.** The tool is `omodel-wire.py`; the generic model configs
  live in omodel-manager, read via `--configs` / `$OMODEL_CONFIGS` / sibling
  `../omodel-manager/configs`. Never copy them here or reintroduce `model_recipes.json` /
  `DEFAULT_RECIPES` (retired in 0.2.0).
- **Idempotent & non-destructive.** Every run merges into the existing config and prunes only
  what it manages (`MANAGED_AGENTS`). Never clobber user-authored agents/providers or
  preserved cloud settings.
- **Sampling plugin path is `plugins/` (plural)**; `tool_call` is declared on every model.
- **You open PRs — you never merge, push to `main`, or approve.** (See the `open-a-pr` skill.)
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
- **Parallel-safe git.** Other agents may share this repo. Work in your own `git worktree`,
  branch first (`git switch -c …`), stage **explicit paths** (never `git add -A`), and start
  from a clean tree — if `git status` shows work that isn't yours, stop and surface it. Test
  via `python3 ./omodel-wire.py …` from your checkout, **not** the `omw` alias; pass
  `--configs <path>` explicitly (the sibling default may not resolve from a worktree). Full
  flow: the `open-a-pr` skill.

## Skills — load the one matching your task first

Skill bodies load on demand; load the match before you start (OpenCode also surfaces them
via the `skill` tool).

| To… | Skill |
| --- | --- |
| Make a code change (file layout, how to extend, checks to run) | **`code-changes`** |
| Modify how the tool generates OpenCode config (agent/permission fields, vetted gotchas, plugin dir, doc pointers) | **`opencode-reference`** |
| Prove the generated sampling actually reaches the model server | **`validate-opencode`** |
| Commit + open a pull request | **`open-a-pr`** |
| Review / approve / merge an open PR | delegate to **`agent-review`** (see the note below the table) |

Other docs (read directly when relevant): `README.md` (user-facing), `CHANGELOG.md`.

**Reviews** are handled by **`agent-review`**, whose global role is extended here by
`agent-review-extend`. It checks completed work against the caller's criteria and `REVIEW.md`,
classifies findings, and merges a PR only when authorized and clean. Delegate by name through the
`task` tool; fix one blocker/regression at a time and reuse the same reviewer `task_id`.
