---
name: agent-review-extend
description: Repo-specific extension for reviewing omodel-wire changes and PRs. Adds required checks, invariants, and this repo's reviewer-authorized merge policy to the global agent-review role.
---

# omodel-wire Review Extension

Read `REVIEW.md`; it is this repo's acceptance bar.

## Required Checks

Run from the PR/change checkout:

    python3 -m py_compile omodel-wire.py
    python3 -m unittest

Use `python3`, not `python`. The known `test_main_dispatch_models_list` failure is `pre-existing`
only when a live localhost model is the sole cause and the same test fails on `main`; verify both.

## Repo Invariants

- Stdlib only; the implementation stays in `omodel-wire.py` plus tests.
- omodel-manager owns model configs; never add `model_recipes.json` or `DEFAULT_RECIPES`.
- Preserve idempotent, non-destructive merging and cross-platform paths.
- Keep LF endings, `plugins/` plural, and `tool_call` on every model.
- Require a focused diff and `[Unreleased]` changelog entry for user-visible behavior.
- Changes to `LICENSE`, `__version__`/tags, `.github/`, or security-sensitive paths need explicit
  user approval.

## PR Merge Policy

This repo authorizes `agent-review` to approve and squash-merge a clean PR in the same delegated
PR-review pass. Merge only when no `blocker` or `regression` remains and required checks are
acceptable:

    GH_TOKEN="$GH_TOKEN_REVIEWER" gh pr review <n> --approve
    GH_TOKEN="$GH_TOKEN_REVIEWER" gh pr merge <n> --squash --delete-branch

Use a Conventional-Commit squash title and no `Co-Authored-By` trailer. Never push to `main`.
If the local post-merge step complains that `main` is checked out in another worktree, verify the
server state with `gh pr view <n> --json state,mergedAt`; do not assume the merge failed.
