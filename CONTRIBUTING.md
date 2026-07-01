# Contributing to omodel-wire

A small, private, single-maintainer project. The workflow below keeps history clean and
releases traceable without ceremony.

## Branching — GitHub Flow

- **`main` is always releasable.** Never commit directly to `main`.
- **Short-lived feature branches** off `main`, named by type:
  - `feat/<slug>` — a new capability (e.g. `feat/pi-dev-target`)
  - `fix/<slug>` — a bug fix (e.g. `fix/vision-false-positive`)
  - `chore/<slug>` / `docs/<slug>` / `refactor/<slug>` — everything else
- Keep a branch focused on one change. Rebase on `main` before merging.

## Commits — Conventional Commits

Format: `type(scope): summary` in the imperative mood.

```
feat(recipes): add GLM-4.7-Flash preset
fix(vision): require "blue" in probe answer to avoid false positives
docs(readme): document --team-reasoning
chore(cli): add --version flag
```

Types: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`. `scope` is optional
(e.g. `recipes`, `agents`, `opencode`, `cli`). Add a body when the "why" isn't obvious.

## Merging

- **Squash-merge** feature branches into `main` so `main` reads as one clean commit per
  change. Delete the branch after merging.
- The squash commit's subject should itself be a Conventional Commit.

## Releases — SemVer + tags

Version lives in `__version__` inside `omodel-wire.py`.

- `MAJOR` — a breaking change to CLI flags, config output, or `model_recipes.json` shape.
- `MINOR` — a backward-compatible feature (new flag, new recipe, new target tool).
- `PATCH` — a backward-compatible fix.

To cut a release: bump `__version__`, add a dated section to `CHANGELOG.md`, commit as
`chore(release): vX.Y.Z`, then tag:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
```

## Before you open a PR / merge

1. `python3 -m unittest test_omodel_wire` passes (expected-failures are fine; hard
   failures are not), and `python3 -m py_compile omodel-wire.py` passes.
2. `python3 omodel-wire.py --dry-run` and `--profiles --dry-run` produce sane output.
3. If you touched `model_recipes.json`, you also updated `DEFAULT_RECIPES` in the script
   (they must stay identical).
4. Docs updated (`README.md` for user-facing flags, `AGENTS.md` for architecture) and a
   `CHANGELOG.md` entry added under **Unreleased**.
5. No new third-party imports — stdlib only.

## Style

- Match the surrounding code: comment density, naming, and idioms already in
  `omodel-wire.py`.
- Keep it a single script with one companion JSON. Don't split into a package unless the
  tool genuinely outgrows one file.
