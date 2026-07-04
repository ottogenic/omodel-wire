<!-- Read AGENTS.md → "Contributing changes (for AI agents / local models)" before opening. -->

## What & why
<!-- One or two sentences: what this changes and why. Link the issue/task if any. -->

## Scope
- [ ] One concern; small, focused diff
- [ ] No unrelated reformatting or churn
- [ ] Does not touch `LICENSE` / `__version__` / release tags / `.github/` (or explicitly approved)

## Tests (paste the output)
```
$ python -m py_compile omodel-wire.py
$ python -m unittest
...
```

## Checklist
- [ ] Stdlib only — no new third-party imports
- [ ] `CHANGELOG.md` `[Unreleased]` updated (if user-facing)
- [ ] Followed the `AGENTS.md` invariants (single-script, LF, kebab/snake naming, `plugins/`)
- [ ] Branch named `feat/|fix/|chore/…`, Conventional-Commit title
