# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Per-model sampling.** The `--profiles` `chat.params` plugin (`dgx-sampling.js`) is now
  keyed by `(model, agent)`, not agent alone. It resolves the sampling vector from the
  CURRENT model (`input.model.id`), so switching an agent onto a different model applies
  THAT model's card-recommended sampling — e.g. a different `reason` temperature per model
  across a fleet of 10+ endpoints. Falls back to `DEFAULT_MODEL` for a managed model with
  no table. Verified live: one `research` agent ran temp 1.0 on FP8 and 0.9 on NVFP4
  simultaneously. `--audit` now shows each model's per-agent sampling and flags a model
  whose per-model table is missing (agents on it would use server defaults).
- **Capabilities are now DECLARED, not probed.** `--profiles` reads vision / reasoning /
  thinking-knob and per-mode sampling from omodel-manager's `configs/*.toml` (consumed
  via `--configs` / `$OMODEL_CONFIGS` / sibling `../omodel-manager/configs`). Removes the
  slow per-model vision/reasoning warmups on every sync.

### Added
- `--audit`: offline diff of the LIVE OpenCode config vs the omodel-manager configs it
  was generated from. Prints a side-by-side table for **every registered managed model**
  (including endpoints no agent is bound to, shown with a note), covering model-level
  capabilities (reasoning / vision / tool_call) and per-agent sampling, and highlights
  drift (e.g. after editing a preset) with a suggestion to `--profiles` re-sync. Reads
  `opencode.json` + `plugins/dgx-sampling.js`; no probing. Exit 1 on drift, 2 if nothing
  to compare.
- `--verify`: opt-in — probe live endpoints and diff their real capabilities against the
  declared configs (writes nothing). The probe functions now run only here.

### Removed
- `model_recipes.json` + `DEFAULT_RECIPES` + `load_recipes()` (+ `--recipes` /
  `$OMODEL_WIRE_RECIPES`). Curated model configs now live in and are owned by
  omodel-manager (this tool is a consuming adapter).

### Fixed
- Reasoning probe no longer misdetects reasoning models (e.g. Qwen3.6-35B-A3B) as
  non-reasoning. When a qwen3-style `--reasoning-parser` is configured and the model is
  cut off before closing `</think>`, vLLM returns the partial thinking in `content` with
  `reasoning: null` (vLLM #35221); the probe now treats a `finish_reason=length` cutoff
  on the trivial probe prompt as the tell that the model was still thinking. Also detects
  inline `<think>...</think>` chain-of-thought for endpoints served without a reasoning
  parser.
- Reasoning models no longer get cut off mid-thought during real use. OpenCode caps
  per-step output at 32k unless `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` is raised; the
  tool now raises it based on the model's actual `limit.output` (not only when a recipe
  preset declared `max_output > 32000`), so a reasoning model whose recipe carries no
  curated output length (e.g. Qwen3.6-35B-A3B) still gets the cap lifted.

### Changed
- Probe budgets sized realistically instead of just-enough: `REASONING_MAXTOKENS`
  384 -> 8192, `VISION_MAXTOKENS` 512 -> 2048, `REASONING_TIMEOUT` 30s -> 45s. These are
  ceilings (a model stops on its own well before them on the trivial probe prompts), so
  the extra headroom costs nothing on normal responses but removes truncation traps.

## [0.1.0] - 2026-06-30

Initial packaged release. Extracted from the `otools` suite and renamed to
`omodel-wire`.

### Added
- OpenCode detection + model sync from OpenAI-compatible endpoints (vLLM/SGLang).
- Verified vision probe (blue-image check) writing `attachment` + `modalities`.
- `tool_call: true` on custom models so OpenCode actually sends tools to them.
- Sampling control: `temperature: false` + a `chat.params` plugin written to the
  docs-correct `plugins/` directory (`--sampling server-default|fixed|opencode-default`);
  a stale `plugin/dgx-sampling.js` from earlier singular-dir syncs is cleaned up on sync.
- `--profiles` agent roster: visible `research` / `code` / `agent` + a `team`
  orchestrator delegating to hidden `agent-plan` / `agent-code` / `agent-instruct`
  workers, driven by an editable `model_recipes.json`.
- Native `build`/`plan` disabled and replaced; `--default-agent`, `--keep-builtins`.
- Frontier team model via `--team-model`, with `--team-reasoning` and
  `--team-task-budget` preserved across re-syncs.
- Web search via `--web-search exa|mcp`; `--write-shell-env` for OpenCode env vars.
- Ctrl+T thinking variants; reasoning-knob probing.
- `--install-aliases` (installs the `omw` alias), `--dry-run`, `--version`.
- Recipes for Qwen3.6-27B, Qwen3.6-35B-A3B, NVIDIA-Nemotron-3-Super, GLM-4.7-Flash.
- Offline regression suite `test_omodel_wire.py` (stdlib `unittest`, network probes
  mocked) covering roster integrity, recipes, agent/plugin building, provider flags,
  and a full `oc_sync` round-trip.
- `AGENTS.md` OpenCode reference section with vetted config-field tables, doc WebFetch
  pointer lines, and a list of source-derived/undocumented mechanisms to verify against
  the installed OpenCode (the `chat.params` hook; the `task_budget` field).

[Unreleased]: https://example.invalid/omodel-wire/compare/v0.1.0...HEAD
[0.1.0]: https://example.invalid/omodel-wire/releases/tag/v0.1.0
