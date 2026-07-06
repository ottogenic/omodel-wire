---
name: pr-review
description: Review, fix, and merge open pull requests against this repo, per its REVIEW.md. Use when asked to review / approve / merge open PRs. Reviews each PR comprehensively for code quality, bugs, and invariant violations FIRST, itemizes any issues with a concrete suggested fix, and only proceeds to merge when the review is clean.
---

# Review open PRs

Act as the code reviewer + approver for THIS repository. You have standing authority to fix and
merge PRs that clear the bar; stop only for the guardrail cases. The gate is simple: **review
first, surface every issue with a suggested fix, and only merge a PR that comes back clean.**

## Setup

1. Read this repo's **`REVIEW.md`** — it defines the exact checks to run, the repo invariants,
   and the auto-merge bar. (No `REVIEW.md`? Fall back to: checks pass, the diff matches its
   stated intent, no secrets, and respect anything `AGENTS.md` / `CONTRIBUTING.md` marks as an
   invariant.)
2. `gh auth status` — if `gh` is missing or unauthenticated, STOP and tell the user how to fix
   it (`gh auth login`).
3. `gh pr list --state open` — review open, non-draft PRs oldest-first (or only the PR numbers
   the user gave).

## Per PR — REVIEW comprehensively FIRST, then decide

### 1. Understand it
`gh pr view <n>` and `gh pr diff <n>` — get clear on what it *claims* versus what it actually
*changes*.

### 2. Comprehensive review (always, before any merge)
Check the branch out (`gh pr checkout <n>`), run the repo's checks from **`REVIEW.md` §1**
(e.g. py_compile + `python -m unittest`), then **read the diff yourself**. Green tests are
necessary, not sufficient. Look for:

- **Correctness bugs / logic errors** — trace the changed code paths; consider edge cases and
  the callers of anything changed.
- **Code quality** — clarity, matches the surrounding style/idiom, no dead code, no debug
  leftovers, sensible names.
- **Invariant violations** — every rule in **`REVIEW.md` §2**.
- **Scope creep** — unrelated changes, reformatting, or churn beyond the stated intent.
- **Stale-branch regressions** — does the diff *revert* something already on `main`? A branch
  cut before a recent merge, committed whole-file, silently undoes it. Rebase onto `main` and
  re-check the diff.
- **Secrets / security / privacy** — tokens, keys, injected shell, weakened auth, or leaked
  private info (internal IPs, hostnames, real emails) — especially on public repos.
- **Missing tests / CHANGELOG** — new behavior needs a test; user-facing changes need a
  `CHANGELOG [Unreleased]` entry.

### 3. Itemize the findings — ALWAYS, before proceeding
Produce an explicit, numbered list. For each issue give:

> **`file:line` · severity · what's wrong · a concrete suggested fix.**

If the review is clean, say so plainly (**"no issues found"**) — don't skip this step.

### 4. Decide

- **Issues found → do NOT merge yet.**
  - Issues you can safely fix inline (per **`REVIEW.md` §4** — correctness bugs, missing
    test/CHANGELOG, small style, trimming scope creep): apply the fix on the branch, commit with
    a Conventional-Commit message and **no `Co-Authored-By` trailer**, re-run the checks, then
    re-review.
  - Issues needing a design decision or a rewrite bigger than a critical fix:
    `gh pr review <n> --request-changes` with the itemized list. Leave the branch as-is. Stop.

- **No issues → proceed to merge.**
  `gh pr review <n> --approve` with a 2–4 line summary of what you checked and fixed (the audit
  trail), then `gh pr merge <n> --squash --delete-branch` with a Conventional-Commit squash
  title.
  > If the PR author is the same identity you review as, GitHub blocks self-approval — post the
  > summary via `gh pr comment <n>` instead, then squash-merge.

## Guardrails — never auto-merge these; request changes or ask the user

- Failing tests you couldn't fix, or a diff that doesn't match its stated intent.
- Security issues (injected shell, leaked secrets/tokens, weakened auth).
- Changes to `LICENSE`, `__version__` / release tags, or `.github/` CI.
- Anything needing a design decision or a rewrite larger than a critical fix.

## Report

End with a summary table — **PR# · title · verdict (MERGED / CHANGES-REQUESTED / SKIPPED) ·
issues found & what you fixed · why** — then stop. Never push to `main` directly; every change
flows through the PR branch and the squash-merge.