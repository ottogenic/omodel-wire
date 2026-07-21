# REVIEW.md — the code-review bar for omodel-wire

The repo-specific standard a pull request must meet to merge. The review process and finding
taxonomy live in the global **`agent-review`** skill; this repo's checks and merge policy extend it
through **`agent-review-extend`**. This file is only the bar those skills check against.

## Checks (must pass)

    python3 -m py_compile omodel-wire.py
    python3 -m unittest           # test_omodel_wire.py + test_configs.py — offline, no network

## Invariants — a diff that breaks any of these is NOT mergeable

- **Stdlib only** — no third-party imports.
- **Single script** — the tool stays in `omodel-wire.py` (+ `test_*.py`). Model configs are
  owned by omodel-manager (`configs/*.toml`); never copy them here or reintroduce
  `model_recipes.json` / `DEFAULT_RECIPES` (retired in 0.2.0).
- **LF endings** (`.gitattributes` enforces) — the shebang must stay runnable on Linux.
- **Naming** — kebab-case CLI surface, snake_case importable Python (`test_*.py`).
- Sampling plugin writes to `plugins/` (plural); `tool_call` is declared on every model.

## Mergeable when ALL hold

1. Both checks pass.
2. The diff does only what the PR claims — no unrelated churn.
3. No correctness bug (read the diff; green tests aren't enough).
4. `CHANGELOG.md [Unreleased]` updated for anything user-facing.
5. Doesn't touch `LICENSE`, `__version__` / tags, `.github/`, or security-sensitive paths
   without explicit user approval.
