# REVIEW.md — code-review bar for omodel-wire

Playbook for reviewing pull requests against **omodel-wire**. When the user says
"review open PRs" (or runs `/review-prs`), follow this end-to-end.

**Authority:** fix + auto-merge PRs that clear the bar below; leave risky or unfixable ones
open with a change-request review. Every merge leaves an audit trail.

## 1. Checks to run (must pass, after your fixes)

```
python -m py_compile omodel-wire.py
python -m unittest            # test_omodel_wire.py + test_configs.py — offline, no network
```

## 2. Repo invariants — a diff that breaks any of these does NOT merge until fixed

- **Stdlib only** — no third-party imports.
- **Single script** — the tool stays in `omodel-wire.py` (+ `test_*.py`). Model configs are
  **owned by omodel-manager** (`configs/*.toml`); never copy them here or reintroduce
  `model_recipes.json` / `DEFAULT_RECIPES` (retired in 0.2.0).
- **LF endings** (`.gitattributes` enforces) — no CRLF; the shebang must stay runnable on Linux.
- **Naming** — kebab-case CLI surface, snake_case importable Python (`test_*.py`).
- Sampling plugin writes to `plugins/` (plural); `tool_call` is declared on every model.

## 3. Auto-merge bar — ALL must hold

1. Both checks in §1 pass.
2. The diff does what the PR description claims — no unrelated changes or churn.
3. No correctness bug you couldn't fix (read the diff; don't just trust green tests).
4. `CHANGELOG.md [Unreleased]` updated for anything user-facing.
5. Does **not** touch `LICENSE`, `__version__`/tags, `.github/`, or security-sensitive
   paths without explicit user approval.

## 4. Fixes you may make directly, then merge

Correctness bugs, failing/missing tests, a missing CHANGELOG entry, small style/consistency,
trimming scope creep. **Larger rewrites or ambiguous design → request changes, don't merge.**

## 5. Merge mechanics

`gh pr review <n> --approve` with a 2–4 line summary of what you checked and fixed (the audit
trail), then `gh pr merge <n> --squash --delete-branch`. Conventional-Commit squash title.
**No `Co-Authored-By` trailer** (maintainer preference).

## 6. Reject (leave open, `gh pr review <n> --request-changes`) when

Tests red and not trivially fixable · a security issue · diff ≠ stated intent · it touches
LICENSE / version / CI · scope too large to safely fix inline. Say specifically what's needed
so the next attempt can pass.
