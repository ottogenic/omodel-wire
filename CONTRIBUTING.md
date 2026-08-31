# Contributing to omodel-wire

A small, private, single-maintainer project. The workflow below keeps history clean and
releases traceable without ceremony.

## Solo publishing workflow

- **`main` is always releasable.** This is a single-maintainer repo, so tested direct pushes to
  `main` are the default when the maintainer requests publishing.
- **Branches and worktrees are optional isolation.** Use them when the maintainer requests one,
  concurrent work is active, or preserving the currently working version is useful. Pull requests
  are opt-in, never automatic.
- Optional short-lived branches off `main` are named by type:
  - `feat/<slug>` — a new capability (e.g. `feat/pi-dev-target`)
  - `fix/<slug>` — a bug fix (e.g. `fix/vision-false-positive`)
  - `chore/<slug>` / `docs/<slug>` / `refactor/<slug>` — everything else
- Keep a branch focused on one change. Integrate it locally into canonical `main`, push `main`,
  verify local and remote agree, then delete the temporary branch/worktree.

## Commits — Conventional Commits

Format: `type(scope): summary` in the imperative mood.

```
feat(recipes): add GLM-4.7-Flash preset
fix(vision): require "blue" in probe answer to avoid false positives
docs(agents): document role skill overlays
chore(cli): add --version flag
```

Types: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`. `scope` is optional
(e.g. `recipes`, `agents`, `opencode`, `cli`). Add a body when the "why" isn't obvious.

## Publishing to main

- AI contributors may commit and push directly to `main` when the maintainer requests publishing.
- Before pushing, run the required checks and inspect status, diff, and recent history. Never
  force-push; stage only explicit intended paths.
- If development used a branch/worktree, finish by updating the canonical checkout (the one the
  `omw` alias targets — resolve it rather than assuming a path), integrating the tested commit
  into local `main`, pushing `main`, and verifying canonical `HEAD`, local `main`, and
  `origin/main` agree.
- Keep `main` to one clean Conventional Commit per focused change. Delete temporary branches and
  worktrees after integration.
- Use the PR/reviewer workflow only when the maintainer explicitly requests a PR.

## Releases — SemVer + tags

Version lives in `__version__` inside `omodel-wire.py`.

- `MAJOR` — a breaking change to CLI flags, config output, or the consumed config schema.
- `MINOR` — a backward-compatible feature (new flag, new target tool).
- `PATCH` — a backward-compatible fix.

To cut a release: bump `__version__`, add a dated section to `CHANGELOG.md`, commit as
`chore(release): vX.Y.Z`, then tag:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
```

## Before you publish

1. `python3 -m unittest test_omodel_wire` passes (expected-failures are fine; hard
   failures are not), and `python3 -m py_compile omodel-wire.py` passes.
2. `python3 omodel-wire.py --dry-run` and `--profiles --dry-run` produce sane output.
3. Model configs are owned by **omodel-manager** (recursive `configs/**/*.toml`); this tool only
   consumes them. Adding/tuning a model happens there, not here.
4. Docs updated (`README.md` for user-facing flags, `AGENTS.md` for architecture) and a
   `CHANGELOG.md` entry added under **Unreleased**.
5. No new third-party imports — stdlib only.

## Style

- Match the surrounding code: comment density, naming, and idioms already in
  `omodel-wire.py`.
- Keep it a single script with one companion JSON. Don't split into a package unless the
  tool genuinely outgrows one file.

## Naming conventions

Two standards, split by **how the name is used** — the same split every Python CLI uses
(pytest's `test_*.py`, the `pre-commit` / `ruff` / `pip` commands, …). Pick by that rule,
not by taste:

- **kebab-case for anything you type at a shell** — the executable (`omodel-wire.py` is
  invoked directly) and every flag (`--dry-run`, `--team-task-budget`, `--no-vision-probe`).
  No underscores on the CLI surface.
- **snake_case for Python files that get imported** — modules and tests
  (`test_omodel_wire.py`, `test_configs.py`). This is **required**, not stylistic:
  CI runs `python -m unittest test_omodel_wire`, and `import` treats the filename as a
  module name — a hyphen there is a syntax error (parsed as minus). PEP 8 also mandates
  snake for modules.

Rule of thumb: **type it at a shell → kebab; Python imports it → snake.** `omodel-wire.py`
is a hyphenated file only because it's run by path or loaded via
`spec_from_file_location(<logical name>, path)` in the tests — never imported by its
filename, so it never has to be a valid module name.

## Code review (AI reviewer)

Pull requests are reviewed and merged by an AI reviewer (Claude) following **`REVIEW.md`**.
Open a Claude Code chat in this repo and say **"review open PRs"** (or run `/review-prs`): it
lists the open PRs, runs the checks, makes critical fixes, and squash-merges the ones that pass
the bar — leaving anything risky open with a change-request review. Authors follow the rules in
`AGENTS.md` → *Contributing changes*. Requires the GitHub CLI (`gh auth status` clean).

## Add PR review to a new otools repo

1. Copy `REVIEW.md`; edit §1 (checks) and §2 (invariants) for that repo.
2. Add the *Contributing changes (for AI agents)* section + the reviewer trigger line to its
   `AGENTS.md`.
3. Add `.github/pull_request_template.md` and a `.github/workflows/ci.yml` that runs the
   offline suite.
4. The `/review-prs` command is global (`~/.claude/commands/review-prs.md`) — nothing to copy.
