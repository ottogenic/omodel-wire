---
name: open-a-pr
description: Open a pull request against omodel-wire (for AI agents / local models) — branch + Conventional-Commit conventions, the required py_compile + unittest checks, CHANGELOG, the PR template, and the you-open-PRs-never-merge rule. Use when committing or opening a PR.
---

# Open a PR against omodel-wire (for AI agents / local models)

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

**Respect the invariants** (deeper invariants — stdlib-only, single-script, idempotent, etc. —
live in `AGENTS.md`; refer to it rather than repeating in full): stdlib-only, single-script, LF
endings, kebab/snake naming, `plugins/` (plural), and model configs owned by omodel-manager
(never reintroduce `model_recipes.json`).

**Working in parallel (worktrees).** Other agents may be running against this repo at once, so
never assume you're alone in the working tree:
- **One worktree per agent.** Do your work in your own `git worktree` (a separate folder on its
  own branch), not the shared clone — different folder means you can't sweep up another agent's
  uncommitted work, and they can't sweep yours. Make one with `./git-new-worktree <folder> <branch>`
  (e.g. `./git-new-worktree omw-fix-x fix/x`).
- **Branch first, stage explicit paths.** `git switch -c fix/<slug>` before editing; stage what
  you changed by name (`git add omodel-wire.py test_omodel_wire.py`). **Never `git add -A` /
  `git add .`** — that's how one agent's commit swallows another's uncommitted changes.
- **Clean-tree precondition.** Run `git status` before you start. If it's dirty with work that
  isn't yours, STOP and surface it — don't absorb it into your commit.
- **Test from YOUR checkout, not the `omw` alias.** Run `python3 ./omodel-wire.py <cmd>` from
  your worktree — the `omw` alias points at the main clone's copy, and agent shells often don't
  load `.bashrc` (alias undefined). Also pass **`--configs <path>`** (or `$OMODEL_CONFIGS`)
  explicitly: omw's default is the sibling `../omodel-manager/configs`, which from a worktree may
  not resolve to the omodel-manager copy you intend.

**Open the PR** with `gh pr create`, fill in the template, then **stop** — do not merge; wait for review.
