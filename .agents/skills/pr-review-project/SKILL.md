---
name: pr-review-project
description: Review a pull request against THIS repo (omodel-wire) and its REVIEW.md. Use when asked to review / approve / merge a PR here (this is what the agent-review subagent runs in this repo). It runs the checks, reads the diff for bugs, quality, and invariant violations, and returns an itemized list of issues + suggested fixes to the parent agent. It merges ONLY when the review is clean.
---

# Review a pull request (omodel-wire)

This is omodel-wire's project-specific review skill; agent-review prefers it over the generic
global `pr-review` skill. It follows the same review -> report -> merge loop, with this repo's
exact checks, invariants, and gotchas below.

You are the reviewer for THIS repository. The rule is simple: **review first, hand the findings
back to the parent agent, and merge only when there is nothing left to fix.** You do **not** fix
issues yourself — the parent (coding) agents do that and then ask you to re-review.

## 1. Load the bar
Read this repo's **`REVIEW.md`** — its checks and invariants are the standard you review against.

## 2. Review the PR (comprehensively — green tests are necessary, not sufficient)
- `gh pr view <n>` and `gh pr diff <n>` — what it *claims* vs. what it *changes*.
- Get the PR's code, then run the checks from `REVIEW.md` (py_compile + `python3 -m unittest` —
  the env has `python3`, not `python`). Use `gh pr checkout <n>` when the clone has a GitHub
  remote; if it doesn't (a worktree/clone made from a local path) fetch the ref directly with
  `git fetch origin pull/<n>/head && git switch --detach FETCH_HEAD`, or if you're already on the
  PR branch with a clean tree, just run the checks in place.
- Read the diff yourself, looking for:
  - **correctness bugs / logic errors** — trace the changed paths and their callers;
  - **code quality** — clarity, matches the surrounding style, no dead code or debug leftovers;
  - **invariant violations** — every rule in `REVIEW.md`;
  - **scope creep** — unrelated changes or churn beyond the stated intent;
  - **stale-branch regressions** — does the diff *revert* something already on `main`? (a branch
    cut before a recent merge, committed whole-file, silently undoes it — rebase onto `main` and
    re-check);
  - **secrets / leaked private info** — tokens, keys, internal IPs, hostnames, real emails;
  - **missing tests or `CHANGELOG` entry**.

## 3. Report to the parent agent — ALWAYS
Your final message is a review report the parent agent acts on. Return an **itemized** list; for
each issue give:

> **`file:line` · severity · what's wrong · a concrete suggested fix.**

Do not fix the issues and do not merge while any remain — the parent's coding agents apply the
fixes, then delegate back to you (reuse the task_id) to re-review. If the review is clean, say so
plainly: **"No issues found."**

## 4. Merge — only when the review is clean
If (and only if) there are no issues, merge **as the reviewer account** so the approval is
genuine two-party review (GitHub lets a different account approve the coder's PR):

    GH_TOKEN="$GH_TOKEN_REVIEWER" gh pr review <n> --approve
    GH_TOKEN="$GH_TOKEN_REVIEWER" gh pr merge  <n> --squash --delete-branch

Conventional-Commit squash title, **no `Co-Authored-By` trailer**. If `$GH_TOKEN_REVIEWER` is
unset, do **not** self-approve (GitHub blocks it) — post the summary with `gh pr comment` and
tell the user to set the reviewer token.

The merge is server-side. In a shared git worktree (with `main` checked out in a sibling
worktree), `gh pr merge` may still print `fatal: 'main' is already used by worktree …` from its
local post-merge step even though the PR merged. Confirm with `gh pr view <n> --json state,mergedAt`
(expect `MERGED`); if `--delete-branch` was skipped, delete the remote ref yourself:
`gh api -X DELETE repos/<owner>/<repo>/git/refs/heads/<branch>`.

Never push to `main` directly. Never merge anything touching `LICENSE`, `__version__` / tags, or
`.github/` CI without explicit user approval.
