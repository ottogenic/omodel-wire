---
name: opencode-reference
description: Reference for modifying how omodel-wire generates OpenCode config — agent/permission/config fields, the vetted gotchas the tool works around, the plugins/ directory, items to verify against your installed OpenCode, and OpenCode doc pointers. Use when changing agent/provider/plugin generation.
---

## OpenCode reference (how to modify agents / settings)

Editing this tool means knowing how OpenCode itself consumes the config. This section
is the distilled, **vetted** map so you don't have to re-derive it. Source URLs are
below as ready-to-use WebFetch prompts — fetch them when you need the current detail
(OpenCode moves fast; treat the docs as authoritative over this file if they diverge).

### Docs — WebFetch pointer lines

Paste one of these to an AI tool (or fetch yourself) when you need up-to-date detail:

- use webfetch to find documentation/info on topic **agents & subagents (fields, modes, built-ins, disabling)**: https://opencode.ai/docs/agents/
- use webfetch to find documentation/info on topic **top-level config (provider, model, agent blocks, schema)**: https://opencode.ai/docs/config/
- use webfetch to find documentation/info on topic **permissions (edit/bash/task, ask/allow/deny, --auto)**: https://opencode.ai/docs/permissions/
- use webfetch to find documentation/info on topic **tools (which tools exist, enabling/disabling per agent)**: https://opencode.ai/docs/tools/
- use webfetch to find documentation/info on topic **custom tools**: https://opencode.ai/docs/custom-tools/
- use webfetch to find documentation/info on topic **plugins (directory, hooks, npm plugins)**: https://opencode.ai/docs/plugins/
- use webfetch to find documentation/info on topic **providers (openai-compatible, model capabilities)**: https://opencode.ai/docs/providers/
- config JSON schema (referenced as `$schema`): https://opencode.ai/config.json
- source + issue tracker (bugs / limitations below): https://github.com/sst/opencode  (also served as `github.com/anomalyco/opencode`)

### Agent config fields (from the agents docs)

An agent is an object under the top-level `agent` block (or a markdown file in
`~/.config/opencode/agents/` with YAML frontmatter + body-as-prompt). Fields:

| Field | Meaning |
| ----- | ------- |
| `description` | What it does / when to use it (subagents are auto-picked by this). |
| `mode` | `primary` (Tab-cyclable, can spawn subagents), `subagent` (delegation target), or `all`. |
| `model` | Model ref `provider/model-id`. |
| `temperature` | 0.0–1.0. Only honored if the model declares the `temperature` capability. |
| `top_p` | Alternative diversity control. |
| `prompt` | System prompt, usually a file ref: `"{file:./prompts/x.md}"` **relative to the config file**. |
| `permission` | Per-tool policy (see below). |
| `disable` | `true` deactivates a (built-in or custom) agent. |
| `steps` | Max agentic iterations. |
| `color` | Hex (`#22c55e`) or theme name: `primary`/`secondary`/`accent`/`success`/`warning`/`error`/`info`. |
| `hidden` | Hide a **subagent** from the `@` menu (this tool does NOT set it — workers stay `@`-mentionable). |

- **Built-in agents (current):** primaries `build` (default, all tools) and `plan`
  (restricted); subagents `general`, `explore`, `scout`; hidden system agents
  `compaction`, `title`, `summary`. This tool disables `build`/`plan`
  (`BUILTIN_DISABLE`) and replaces them with `research`/`code`. (Note: several names in
  `MANAGED_AGENTS` — `webdev`, `agentic`, `chat`, `fast`, `architect`, `reason` — are
  **legacy** from earlier versions of this tool, kept only so re-syncs prune them.)
- **Permissions** (`permission` block): keys like `edit`, `bash`, `task`, plus
  `websearch`/`webfetch`; values `allow` / `ask` / `deny`. `task` may be a map
  (`{"*":"deny","agent-code":"allow"}`) to restrict delegation targets — how `team`
  is locked to the hidden workers. CLI `--auto` auto-approves anything not explicitly
  denied.

### Gotchas this tool works around (vetted: source-read + issue tracker)

- Custom OpenAI-compatible models get **no tools at all** unless `tool_call: true` is
  declared on the model — hence it's on by default (`--no-tool-call` to disable).
- Vision needs **both** `modalities` **and** `attachment: true` on the model entry.
- `temperature: false` on a model makes OpenCode send **no** temperature; `topP`/`topK`/
  penalties are **not** config-gated — the only client-side override is the `chat.params`
  plugin hook (why the tool emits one). In `--profiles` mode `temperature` is kept `true`
  so per-agent `temperature`/`top_p` are honored.
- OpenCode **reserves** `build`/`plan`; overrides on those names are ignored (they show
  as "native"). So the tool disables them and moves the startup agent
  (`default_agent`, via `--default-agent`).
- Tab/menu **order is not configurable** — it's internal (upstream feature requests
  open). Config object order is written canonically anyway (visible → hidden → disabled).
- A `team`/orchestrator can receive a local subagent's **empty** last text part instead
  of its result — the `WORKER_PROMPT` on hidden workers forces a non-empty final summary
  to dodge this. (Cloud/Anthropic workers are unaffected; local vLLM workers were.)
- Per-step output is capped at **32k** unless `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`
  is set (`--write-shell-env` appends it).
- Anthropic extended thinking uses `options.thinking = {type:"enabled", budgetTokens:N}`
  (NOT `reasoningEffort`) and **requires `temperature: 1.0`**. Non-`dgx-` (cloud) team
  models have their local `chat_template_kwargs` / `reasoning_effort` / `top_p` stripped
  automatically.
- Anthropic Pro/Max subscription OAuth was removed from OpenCode (ToS) — use an API key
  (`ANTHROPIC_API_KEY` / `/connect`).

### Plugin directory (resolved)

The sampling plugin is written to `<config_dir>/plugins/dgx-sampling.js` (**plural**),
matching the OpenCode docs (https://opencode.ai/docs/plugins/). Earlier builds used the
singular `plugin/` — `oc_sync` now removes a stale `plugin/dgx-sampling.js` on re-sync so
OpenCode doesn't ignore the plugin or load a stale copy. The test suite asserts both
(written to `plugins/`, and the legacy singular file is cleaned up).

### Other items to VERIFY against your installed OpenCode

Source-derived or undocumented mechanisms — confirm before relying on them:

1. **`chat.params` hook.** The sampling plugin uses a `chat.params` hook. That hook is
   **not listed** on the current plugins docs page (which enumerates event hooks like
   `tool.execute.before`). It was vetted from OpenCode source in development, so it is
   likely real-but-undocumented — but confirm it still fires in your version.
2. **`task_budget` agent field.** The team's delegation cap is written as `task_budget`
   on the agent. This is **not** in the documented agent fields (docs list `steps`).
   Treat it as experimental; confirm it's honored.
