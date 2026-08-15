---
name: open-a-pr
description: Open a pull request against omodel-wire only when the maintainer explicitly requests one — branch conventions, required checks, CHANGELOG, and the PR template.
---

# Open a PR against omodel-wire (for AI agents / local models)

Use this skill only when the maintainer explicitly requests a pull request. The normal
single-maintainer publishing path is tested integration directly to `main`, as documented in
`AGENTS.md`. If a PR is requested, follow these rules.

**Follow the requested end state.** If the maintainer asks only to open a PR, stop after
`gh pr create`. If review or merge is also requested, delegate the review according to
`AGENTS.md` and carry the clean result through the requested end state.

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

**Branch hygiene.** A PR requires a focused branch, but a separate worktree is optional unless
concurrent work is active or isolation is useful:
- **Stage explicit paths.** Stage what you changed by name (`git add omodel-wire.py
  test_omodel_wire.py`). **Never `git add -A` / `git add .`.**
- **Clean-tree precondition.** Run `git status` before you start. If it's dirty with work that
  isn't yours, STOP and surface it — don't absorb it into your commit.
- **Test from the development checkout.** Run `python3 ./omodel-wire.py <cmd>` there rather than
  the `omw` alias. In a worktree, pass **`--configs <path>`** (or `$OMODEL_CONFIGS`) explicitly.

**Open the PR** with `gh pr create` and fill in the template. Then follow the maintainer's
explicit request about whether to stop, review, or merge.
