# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Per-worker step caps + a DONE/CONTINUE/BLOCKED exit contract, so workers escalate instead of
  spinning.** Each delegation worker now carries a `steps` cap (agent-instruct 5, agent-research
  10, agent-test 12, agent-architect 15, agent-code 20, agent-review 20); OpenCode forces a
  text summary when the cap is hit. Worker prompts end with one of three statuses — `DONE`
  (finished), `CONTINUE` (working plan, just needs another round), or `BLOCKED` (no working
  hypothesis) — and the `team-orchestration` skill acts on each: `CONTINUE` → resume the SAME
  worker (same task_id); `BLOCKED` → route to `agent-architect` for a fix. An anti-spin guard
  escalates a worker that returns `CONTINUE` ~twice without converging. Visible primaries and
  `team` are not step-capped (direct human use). Values are sensible defaults; a tuning flag can
  come later.
- **Optional per-worker project skills.** Every worker prompt now tells the worker to load a
  repo-local skill named after itself (`<agent-name>-project`, e.g. `agent-code-project`,
  `agent-architect-project`) IF one exists — the same try-project-first pattern `agent-review`
  already uses with `pr-review-project`. These skills are absent by default (a no-op line in a
  fresh repo), hand-authored under `.agents/skills/`, and never written by `omw sync`, so a repo
  can give one worker repo-specific guidance without bloating the shared global prompt. The
  `agent-runbook-review` skill now recommends and drafts `agent-<role>-project` skills when it
  finds recurring role-specific guidance.
- **Two new subagents — `agent-test` and `agent-architect` — for model-tiered delegation.**
  `agent-test` runs lint/tests and reports structured PASS/FAIL (opening a PR only when explicitly
  asked, and never weakening tests to pass); `agent-architect` is a read-only planner/verifier and
  the escalation target when `agent-code` reports BLOCKED. Each has its own system prompt
  (`prompts/otools-test.md`, `prompts/otools-architect.md`); architect is purple (`#8b5cf6`).

### Changed
- **Sync roster summary now names the current research worker.** The printed delegation hint says
  `@agent-research` instead of the retired `@agent-plan` name.
- **Roster reworked for cost-tiered delegation.** `agent-plan` is renamed to `agent-research`
  (read-only web fetch + summarize). The team now delegates to six workers
  (`agent-research`, `agent-code`, `agent-test`, `agent-instruct`, `agent-architect`,
  `agent-review`). Workers never sub-delegate — every worker now has `task: "deny"` and the team
  is the sole orchestrator/courier (escalation flows code → BLOCKED → architect → code through the
  team, never subagent-to-subagent).
- **Only `agent-review` may merge.** The real control is the **token split** — non-reviewer
  agents don't hold `$GH_TOKEN_REVIEWER`, so GitHub rejects an unauthorized/self merge. As
  defense-in-depth, a bash **merge tripwire** (`*gh pr merge*` → `ask`) is added to every other
  agent; it's a visible prompt, not a hard block (glob matching is bypassable, so `deny` would be
  false assurance). A new `test` permission profile backs `agent-test`.
- **`default_models.json` template is now Qwen-first.** Every workhorse agent leads with the local
  `qwen3.6-35b-a3b-nvfp4-unsloth` (≈free at the margin) and falls back to paid GitHub-Copilot
  models only when needed; the two low-volume, high-stakes workers (`agent-architect`,
  `agent-review`) lead with `github-copilot/claude-opus-4.8`.
- **`team-orchestration` skill expanded** with per-worker delegation contracts (what to ask each
  worker to return), the escalation loop, session-continuity rules (reuse a worker's own `task_id`;
  carry payloads across workers rather than sharing task_ids), and an explicit-only git rule: the
  normal loop stops at working, tested code and never opens a PR / reviews / merges unless the user
  asks. The global `pr-review` skill notes it runs only on explicit request; this repo's
  `pr-review-project` skill documents its opt-in aggressive review-and-merge-in-one-pass flow.

- **Every model assignment now applies that model's known-good config — and the redundant
  `--team-model` flag is gone.** Previously only the roster build configured a model correctly;
  `omw agents/subagents --set-model` was a bare string swap and `default_models.json` reassignments
  weren't re-derived, so an agent kept the *previous* model's temperature/top_p and (critically)
  thinking knobs. Now a single path (`_apply_model_config_to_agent`) configures every agent for the
  model it lands on — via `--set-model`, `default_models.json`, or the roster build:
    - **DGX-hosted models** get the omodel-manager recipe preset for the agent's role
      (temperature/top_p/thinking + the plugin sampling vector).
    - **Frontier/cloud models** run on OpenCode's own registry defaults; the DGX-only vLLM knobs
      (`options`/`top_p`/`temperature`) are stripped so we never impose local sampling on them.
  The `--team-model` / `--team-reasoning` flags and the `team_model`/`team_reasoning` wire.json
  settings are **removed** (a leftover `team_model` is pruned from wire.json on load). Set the
  team's model in `default_models.json` (persistent) or with `omw agents team --set-model`
  (transient); `--team-task-budget` is unchanged.

### Added
- **`agent-runbook-review` skill + `omw skills` command.** A user-kicked maintenance pass
  ("perform an agent runbook review") shipped globally by `omw sync` (like `team-orchestration`
  and `pr-review`). It compacts and de-duplicates a repo's agent-facing docs, mines the **current
  OpenCode session — including subagent sessions** — from the SQLite store
  (`~/.local/share/opencode/opencode.db`, read-only; subagents linked via `session.parent_id`) for
  recurring tool failures and drafts fixes into the right file, audits skill sizes (a transparent
  instruction-count metric with lean/moderate/LARGE verdicts), recommends whether a new
  `*-project` skill is warranted, and authors first drafts of missing key files (AGENTS.md,
  REVIEW.md) — all **report-first**, never a silent rewrite. New `omw skills` lists skills (global
  + project `.agents/skills`) with their size verdict; `omw skills <name>` pretty-prints every
  `.md` file in the skill.
- **Generic global `pr-review` skill + project-override pattern.** `omw sync` now writes a
  repo-agnostic `pr-review` skill globally to `<config-dir>/skills/pr-review/SKILL.md` (like
  `team-orchestration`), so `agent-review` has a working review method in a **fresh repo**. The
  `agent-review` prompt now loads the repo's own `pr-review-project` skill first and, if it's
  absent, prints "No project specific PR skill found, using global default" and falls back to the
  global `pr-review`. The global skill includes a short section on authoring a `pr-review-project`
  skill for a repo. omodel-wire's own review skill is renamed `pr-review` → `pr-review-project`.
- **GitHub Copilot CLI is now a sync target** — `omw sync --target copilot` (or `--target all`)
  writes the agent roster to `~/.copilot/` as `.agent.md` files (Markdown + YAML frontmatter),
  merges `settings.json` (`model`, `includeCoAuthoredBy: false`, `stream: true`), and emits an
  `otools-copilot.env`/`.ps1` snippet for the BYOK provider endpoint. The config home is
  **auto-detected**: native `~/.copilot` on Windows/macOS/Linux, or the Windows-side
  `C:\Users\<you>\.copilot` (via `/mnt/c`) when `omw` runs in WSL but Copilot is on Windows;
  `$COPILOT_HOME`/`--copilot-home` override. **Copilot's CLI takes a single
  custom endpoint**, so the whole roster runs on ONE DGX model (the endpoint can't live in
  `settings.json`, hence the env snippet). Delegation is Copilot-runtime-global by description and
  subagents can't spawn subagents, so primaries → top-level agents, workers → subagents, and
  `agent-review` is `disable-model-invocation: true` (explicit-invoke-only). VS Code multi-model
  target is a planned follow-up.
- **`.claude/skills/getting-started` onboarding skill** — an end-to-end setup guide (shell aliases,
  DGX provisioning, launching a first model, the HF token, installing OpenCode, syncing + tweaking
  the roster) for Claude to walk a new user through, with copy-paste commands at each step.

### Fixed
- **Sync no longer re-pins the `team` agent to a stale, unavailable cloud model.** The team-model
  preservation step kept whatever non-DGX model was previously in `opencode.json` (e.g. a leftover
  `anthropic/claude-opus-4-8`) on every sync, without checking it was reachable and ignoring
  `default_models.json` entirely — so an unavailable model that was never in your preferences would
  stick forever. Preservation now applies only when the previous team model still resolves against
  the available pool (runtime-discovered or a configured provider); otherwise the team reverts to a
  live model.
- **Team model now honors `default_models.json` when no reasoning models are live.** Previously,
  if the previous team model was a non-DGX frontier (e.g., `openai/gpt-5.5`) and the resolved
  preference in `default_models.json` pointed elsewhere (e.g., `github-copilot/gpt-5.5`), sync
  would preserve the stale existing model because the "preference resolved" check only fired when
  the resolved model differed from the already-resolved preference. Sync now explicitly checks
  whether any team preference resolves and uses it; preservation only applies when no preference
  resolves.
- **`omw agents <name> --set-model` now round-trips with the names `omw models` shows.** The
  models list printed the bare served id (e.g. `unsloth/qwen3-coder-next-fp8`), but passing that
  to `--set-model` split it into a non-existent `unsloth` provider (broken config), and a bare
  name silently resolved to the first tail match (possibly the wrong host). Now `omw models`
  prints the **host-qualified ref** (`dgx-<host>/<served-id>`) as the MODEL column — one row per
  live instance, so the same model on two hosts is two distinct refs — and `--set-model` accepts
  that ref, resolves a slash-containing served id to its host, and **errors with the exact choices**
  when a name maps to more than one host instead of guessing.
- **PR-review tooling:** the `REVIEW.md` checks and the `pr-review` skill now use `python3` (the
  WSL env has no `python`), and the skill documents worktree-safe PR checkout plus a fallback for
  `gh pr merge`'s local post-merge error when `main` is checked out in a sibling worktree.

### Changed
- **`default_models.json` is now a local preference file.** It is ignored by Git, so custom
  model priorities no longer block repository updates. `omw sync` recreates a missing file with
  the built-in per-agent preference ordering, which includes `gemma4-31b-it-nvfp4` as each
  role's second-choice model.
- **Built-in OpenCode and Hugging Face providers are disabled by default.** The tool now writes a
  top-level `disabled_providers` array (e.g., `["opencode", "huggingface"]`) to `opencode.json`.
  These providers remain in the config but their models are hidden from the `/models` picker.
  Pass `--add-default-providers` to enable them (e.g., if you're using their API keys).
- **`disabled_providers` uses OpenCode's native schema.** The tool now uses the documented
  `disabled_providers` array rather than model-level `blacklist`, which is the proper way to
  hide providers at the provider level.
- **`code` and `agent` retain delegation to `agent-review`.** They have `task_budget = 1` and
  `permission.task: {"*": "deny", "agent-review": "allow"}` so they can hand PR reviews to
  `agent-review` without opening up general delegation. `agent-code` (hidden worker) has
  `permission.task: "deny"` (no delegation at all) to prevent it from delegating to `agent-review`
  or any other agent, preserving edit/bash permissions (`allow`). The `team` orchestrator
  retains full delegation capability to all workers (`agent-plan`, `agent-code`,
  `agent-instruct`, `agent-review`).

### Added
- **`--add-default-providers` flag.** Enable built-in OpenCode and Hugging Face providers when
  they are disabled by default.

### Fixed
- **OpenCode runtime model discovery via `opencode models`.** The tool now discovers available
  OpenCode built-in providers (e.g., `openai/`, `anthropic/`) at runtime via `opencode models`,
  so `default_models.json` preferences can fall through from unavailable cloud refs (e.g.,
  `anthropic/claude-opus-4-8` offline) to available ones (e.g., `openai/gpt-5.5`). The roster
  is rebuilt from the live endpoints even when no reasoning model is running — a coder-only
  fleet (all non-reasoning) gets a full, valid roster instead of leaving agents pointing at
  a model that's no longer served (which OpenCode rejected as "not valid"). `opencode.json` is
  no longer treated as authoritative for remote models.
- **`default_models.json` preference resolution now walks the list in order and no longer
  misclassifies local served ids.** Two bugs surfaced when a preference list mixed local models
  first with a cloud fallback last: (1) any preference containing a `/` that wasn't in the local
  pool was treated as a remote provider ref — so a DGX served id like `unsloth/qwen3-coder-next-fp8`
  (an HF org/model id) pointed the roster at a non-existent `unsloth` provider whenever that model
  was down; (2) the old two-pass logic grabbed the first cloud model in the list regardless of
  order, so a local-first list still resolved to `openai/gpt-5.5` even with a local model live and
  listed first. Now a single ordered pass takes the first live-local **or** known-remote-provider
  preference (see `REMOTE_PROVIDERS`), falling back to the first available local only if nothing in
  the list resolves. (Tests no longer depend on the developer's personal `default_models.json`.)

### Changed
- **`default_models.json` updated** with per-agent preference lists: the coding/reasoning agents
  prefer local DGX models (`unsloth/qwen3-coder-next-fp8` → the fp8/nvfp4 coders → the 35B-A3B pair)
  then cloud (`openai/gpt-5.5`, `google/gemini-3.5-flash`); `agent-review` prefers
  `anthropic/claude-opus-4-8` then the two cloud models.

### Added
- **`team-orchestration` skill** (written to the global `<config-dir>/skills/`, which OpenCode
  auto-scans) holds the Team Lead's methodology — decompose, **dispatch independent work to
  subagents in parallel**, sequence only true dependencies, verify. Scoped to `team` via
  `permission.skill` (denied to every other agent). `TEAM_PROMPT` is slimmed to identity +
  delegation mechanics + "load this skill," mirroring the `agent-review`/`pr-review`
  thin-prompt-plus-skill pattern. (Verified live via `opencode debug skill`; note OpenCode
  auto-loads project `.agents/skills/` and global `~/.config/opencode/skills/`, so no
  `skills.paths` config is needed.)
- **Per-agent GitHub identity via an auto-generated OpenCode plugin.** `omw sync` writes
  `plugins/otools-git-identity.js` (a `shell.env` hook): every coding agent gets `GH_TOKEN` = the
  **coder** token (shared bot account), so commits/PRs are the bot; `agent-review` uses
  `GH_TOKEN_REVIEWER` (your account) to review + merge — real two-party review. All github.com git
  ops route over HTTPS+token per-shell (no SSH, no `~/.gitconfig` edits). Tokens live in
  `~/.config/otools/gh_token_coder` and `gh_token_reviewer` (chmod 600); `omw sync` warns if either
  is missing.
- **`omw config --set-gh-token-coder` / `--set-gh-token-reviewer`** to store those tokens (writes the
  file with 0600 perms). Pass the token inline, omit it to be prompted with hidden input, or pass
  `none` to clear. `omw config` (no args) now shows each token's set/unset state and path.
- **Commits are authored as the bot too, not just pushed by it.** Setting the coder token resolves the
  bot's login + id from the GitHub API once and caches `~/.config/otools/gh_coder_identity`
  (`{name, email}`, using the attributable `<id>+<login>@users.noreply.github.com`); the plugin reads
  that file and exports `GIT_AUTHOR_*`/`GIT_COMMITTER_*` (no network in the per-shell hot path). Fully
  best-effort — if GitHub is unreachable, pushes/PRs still work and `omw sync` retries the resolve.

### Changed
- **`AGENTS.md` now tells agents to *delegate* PR reviews to `agent-review`** (via the `task` tool,
  by name) instead of reviewing inline. Placed in `AGENTS.md` — which augments every agent's context —
  so it reaches the prompt-free `code`/`agent` primaries too, without clobbering their default prompt
  (setting a per-agent `prompt` would replace it). Wording is **capability-neutral** ("if you have the
  `task` tool, delegate…") since AGENTS.md is read by every agent but only some can delegate; it does
  not tell agents to route fixes to `agent-code` (only `team` can reach it). Also drops the incorrect
  "delegate to agent-review" line from the worker prompt (`agent-plan` is read-only and cannot delegate).
- **PR-review workflow split: `REVIEW.md` is now just the repo's *bar*; the review *process* moved
  to the `pr-review` skill.** `REVIEW.md` keeps only the checks, invariants, and merge conditions;
  the skill holds the process — review first, hand the parent agent an itemized list of issues +
  suggested fixes, and merge (as the reviewer) only when clean. `agent-review`'s prompt is now thin
  (it loads the skill), and the team delegates PR reviews to `agent-review` via the task tool (by
  name, not `@`). Removes the process duplication across REVIEW.md / skill / prompt.
- **`git-new-worktree --delete` now also cleans up orphaned worktree folders.** If a folder is a
  sibling of the repo but git no longer tracks it as a worktree (its registration was pruned — e.g.
  by a cross-OS `git worktree prune`), or it's just a stray sibling directory, `--delete` offers to
  remove the folder after a clear "deletion is PERMANENT / can't verify unsaved work" warning.
  Restricted to siblings; refuses `.`/`..`.
- **New `git-sync-main` helper + renamed `new-worktree` → `git-new-worktree`.** `./git-sync-main`
  brings the current clone's `main` up to date with origin (fetch --prune → switch to main →
  fast-forward) — "make sure this is up to date." It **refuses inside a linked worktree and on a
  dirty tree**, so it never disturbs feature work. Both helpers now carry the `git-` prefix (also
  usable as `git new-worktree` / `git sync-main` when the repo is on `PATH`).
- **`./git-new-worktree` gains teardown**: `--delete <folder>` (aliases `--undo`/`--rm`) removes the
  worktree + its **local** branch only — **safe by default**: it never touches an open PR or the
  remote branch, so a submitted-but-unmerged PR (or a merged one) is untouched. `--abort <folder>`
  is the throw-it-all-away version — it also closes the open PR + deletes the remote branch. `-y`
  skips the prompt.
- **Genericized example host addresses** to the RFC 5737 documentation range (`192.0.2.0/24`)
  in `DEFAULT_HOSTS`/`HOST_LABELS`, docstrings, and tests — the shipped fallback no longer
  hardcodes a specific private LAN. Configure real hosts via `omm install` /
  `~/.config/otools/hosts` as before. Also ignore `wire.json`/`hosts` defensively.
- **`AGENTS.md` slimmed to invariants + a skill index; task detail moved to lazy-loaded
  skills.** The full `AGENTS.md` was injected into every model request (~4.4k tokens) even for
  trivial turns. It's now a lean always-on core; the layout / OpenCode-reference / contributing
  sections moved into OpenCode **skills** under `.agents/skills/` (the vendor-neutral discovery
  path): `code-changes`, `opencode-reference`, `validate-opencode`, `open-a-pr`. Only each
  skill's name + description is advertised up front; the body loads on demand via the `skill`
  tool. `VALIDATE_OPENCODE.md` moved into the `validate-opencode` skill. Cuts per-request
  prompt overhead substantially with no loss of guidance.

### Fixed
- **`omw proxy` no longer logs a traceback when a client disconnects.** A client (OpenCode)
  cancelling or timing out mid-request raises a connection error on the proxy's response write;
  the request handler now swallows the client-disconnect family (`ConnectionResetError`,
  `BrokenPipeError`, `ConnectionAbortedError`) instead of trying to send a doomed 502.
  `ConnectionRefusedError` (an *upstream* failure) is deliberately left to still return 502.
- **`team` is now truly delegation-only.** Its permission block denied only `edit`/`bash`,
  but OpenCode gates the read-only tools (`read`/`grep`/`glob`/`list`) under their own
  permission keys that default to *allow* — so the orchestrator could (and did) grep/read
  files directly instead of delegating. The team now denies **every** tool category
  (`read`/`grep`/`glob`/`list`/`edit`/`bash`/`webfetch`/`websearch`); the only action it
  can take is `task` (spawn a worker). Re-run `omw sync` to apply. (Uses `permission`, the
  supported mechanism — the old `tools` field is deprecated in OpenCode.)

### Added
- **`agent-review` subagent.** A new hidden worker that handles reviewing Pull Requests. It uses
  `anthropic/claude-opus-4-8` by default, delegates only to `agent-review`, and has a task budget
  of 1. The review prompt is written to `prompts/otools-review.md` and guides the agent to
  identify issues, provide fixes, and inform the parent agent when a PR is ready to merge. The
  `code` and `agent` primaries now carry `task_budget = 1` and a `task` permission that allowlists
  **only** `agent-review`, so they can hand a PR to the reviewer without opening up general
  delegation.
- **`pr-review` skill.** A new skill at `.agents/skills/pr-review/SKILL.md` that defines the
  end-to-end workflow for reviewing and merging PRs against this repo. It follows the rules in
  `REVIEW.md`, runs the repo's checks, reads the diff, and only merges when the review is clean.
  PRs that fail checks, have security issues, or need design decisions get `--request-changes`.
- **`default_models.json` — user-editable model preferences for agents/subagents.** `omw sync`
  selects the highest-preferred *available* model from `default_models.json` for each
  agent/subagent (ordered lists; fall back to the first available model if none match).
  **The roster is now rebuilt from the live endpoints even when no reasoning model is running**
  — a coder-only fleet (all non-reasoning) gets a full, valid roster instead of leaving agents
  pointing at a model that's no longer served (which OpenCode rejected as "not valid").
  Per-model sampling is emitted for non-reasoning models too. Template auto-created on first run.
- **Runtime discovery/failure tests.** `test_default_models.py` now covers the merged-pool scenario
  where local probes and remote models from `opencode models` are combined; tests validate that
  unavailable cloud refs (e.g., `anthropic/claude-opus-4-8`) are skipped and the first available
  preference (e.g., `openai/gpt-5.5`) is selected. Tests also verify that remote models in the
  pool are accepted as preferences, not just local DGX models.
- **`omw proxy` — a debug proxy that logs OpenCode ↔ model traffic** (stdlib only):
  - `omw proxy on [<model>]` — route live models through the proxy with **no `--upstream`
    needed**: rewrites the `dgx-` provider baseURLs to `127.0.0.1:<port>/<route>` and maps
    each route → the real endpoint in `proxy_routes.json`. No arg = all live models; a model
    name = just that one. Per-model toggles compose. Launches the daemon in the background.
  - `omw proxy off [<model>]` — restore the selected (or all) providers; stops the daemon
    when nothing is left proxied.
  - `omw proxy replay <id> [--output-curl]` — re-issue a logged request **directly** to the
    real API (the logged URL is the upstream, not the proxy), or print a copy-paste curl.
  - `omw proxy read <id>` — **NEW**: colored, section-headed view of a logged exchange
    (model & params, tools, system prompt, messages with real newlines, assembled response).
  - `omw proxy status` — daemon + which providers are proxied.
  - **Robust core:** threaded server with true **SSE/streaming passthrough** (tees the
    stream to the log instead of buffering — fixes the stalls/crashes), **path-prefix
    routing** to the correct model, short 7-char ids, and **flat `proxy_logs/`** (no date
    subfolder) with an `index.jsonl`.
- **`omw models` gains `LIVE` / `PROXY` / `SERVED` columns** — the list now shows, per model,
  whether it's live, whether the proxy is on for it, and the real served id from the live
  endpoint (e.g. `qwen/qwen3.6-35b`), so you can see what's actually running vs which config
  it matched. Columns: `MODEL · REASON · VISION · LIVE · PROXY · SERVED · CONFIG`.

### Changed
- **`omw models` now lists only LIVE models by default; `--all` shows the full catalogue.**
  The declared roster can be long and mostly-offline, so the bare `omw models` view is now
  scoped to what's actually running; it prints how many more are hidden and hints `--all`.
  `omw models --all` restores the every-declared-model table. `omw models <name>` detail is
  unchanged.
- **`omw models` now emits one row per live *served instance*, not per config.** When the same
  config is served under more than one id / on more than one endpoint, each instance gets its own
  row (with its own `PROXY`/`SERVED`), so you can see every running copy. `--all` still lists the
  full declared catalogue (offline configs included); live provider models with no matching config
  surface under `--all` with capabilities shown as `-` (never guessed from the model name).
- **CLI redesigned into `omm`-style subcommands.** The ~40 flat top-level flags are
  replaced by verbs: `omw` (guided home screen), `omw sync` (the roster sync — all former
  sync flags live here), `omw agents` / `omw subagents` / `omw models` (list/inspect + live
  tweaks), plus `omw audit` / `omw verify` / `omw config` / `omw detect` / `omw shell-init`.
  Bare `omw` prints status + suggested next steps, and every command suggests the next
  step (ported omm's `_suggest` breadcrumb helper). `sync` always builds the roster (was
  `--profiles`). Old top-level flags (`--profiles`, `--audit`, `--install-aliases`, …) are
  gone — use the subcommands.

### Added
- **`omw config` + `~/.config/otools/wire.json`** — persist the settings you keep
  retyping (opencode path, configs dir, hosts, `team_model`, `team_reasoning`,
  `default_agent`, `web_search`). Precedence everywhere: **flag > wire.json > built-in**.
- **Live tweak commands** — `omw agents <name> --set-model`, `omw subagents [--set-model]`,
  `omw agents team --set-work-budget N`, and `omw models <name> --role R --set-temperature
  T` / `--set-thinking B`. Every edit touches ONLY `~/.config/opencode/` (opencode.json +
  a re-emitted `plugins/dgx-sampling.js`); the declared configs stay pristine, so
  `omw sync` restores known-good and `omw audit` shows exactly what drifted. `--set-model`
  accepts a bare model name (e.g. `qwen3.6-35b-nvfp4`), auto-resolved to the live
  `provider/model-id`.
- **Per-model work-budget default.** An optional `[capabilities] concurrency` in a model
  config (mirrors the launch profile's `max-num-seqs`) becomes the default team
  `task_budget` when none is set — the team won't spawn more parallel workers than the
  server has sequence slots. Surfaced in `omw models <name>`.

## [0.2.0] - 2026-07-03

### Added
- **Shared host discovery.** `omw` now defaults its probe host list from the same
  `~/.config/otools/hosts` store that omodel-manager's `install`/`ps` manage, so adding a
  box once (`omm install user@ip dgx-3`) makes it visible to both tools — no more editing
  the hardcoded `DEFAULT_HOSTS`. Parses `alias<TAB>user@host`, bare `user@host`, or bare
  host lines and strips `user@` to the bare IP for HTTP probing; falls back to the built-in
  `DEFAULT_HOSTS` when the file is absent/empty. `--hosts` still overrides. Added an `n3`
  label (192.168.50.103) so the third node gets a clean provider key.
- **`--repetition-detection`** — sets vLLM's `repetition_detection` (RepetitionDetectionParams)
  on every managed request via the sampling plugin, terminating a generation once a token
  N-gram loops so a degenerate loop can't burn the whole output budget. Default is tuned
  **lenient** — `min_pattern_size:3, max_pattern_size:20, min_count:10`: it only cuts a unit
  that repeats 10× and ignores single/double-token runs, so long numbers (`300000`),
  indentation, `====` rules, hex/base64, and short repeated array rows/boilerplate are never
  flagged, while a genuine runaway (which repeats hundreds of times) is still cut within
  ~30–40 tokens. (The initial cut left `min_pattern_size` at vLLM's default of 1, which
  clipped `300000` to `30000`.) Pass `off` to disable, or `K:V,…` to override individual
  knobs — merged onto the default, so `min_count:14` raises just that one instead of
  dropping `max_pattern_size` (→ disabled).

### Documentation
- **Naming conventions codified** in `CONTRIBUTING.md`: kebab-case for the CLI surface
  (executable + flags), snake_case for imported Python files (modules/tests) — the latter
  required, since CI's `python -m unittest test_omodel_wire` / `import` reject hyphenated
  module names. Documents why the existing split is intentional, not accidental.

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
- **Worker prompt ordering (load-bearing).** The hidden workers' prompt now leads with the
  tool-calling instruction, then the plain-text-summary rule. Front-loading the summary rule
  made Qwen3-family workers (served via vLLM `--tool-call-parser qwen3_coder`) narrate tool
  calls as text (`<invoke .../>`, `bash(...)`) instead of emitting native tool calls — the
  parser dropped them, the loop exited early, and workers fabricated/leaked results. Verified
  on n1: 1/8 → 15/15 successful worker runs.
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

[Unreleased]: https://github.com/ottogenic/omodel-wire/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ottogenic/omodel-wire/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ottogenic/omodel-wire/releases/tag/v0.1.0
