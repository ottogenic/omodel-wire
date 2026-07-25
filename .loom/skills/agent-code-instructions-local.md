# omodel-wire Coding Overlay

## Repo Invariants

- Stdlib only; the implementation stays in `omodel-wire.py` plus tests (`utils/` holds
  the loom conductor and proxy helpers).
- omodel-manager owns model configs; never add `model_recipes.json` or `DEFAULT_RECIPES`.
- Preserve idempotent, non-destructive config merging and cross-platform paths.
- Keep LF endings, `plugins/` plural, and `tool_call` on every model.
- Add an `[Unreleased]` changelog entry for user-visible behavior.
- Changes to `LICENSE`, `__version__`/tags, `.github/`, or security-sensitive paths need
  explicit user approval.

## Focused Check

Run `python3 -m py_compile omodel-wire.py` plus the targeted `python3 -m unittest
<TestClass>` for the area you changed. Use `python3`, not `python`. The full suite
runs in a later verification step.
