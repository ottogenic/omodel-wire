---
name: validate-opencode
description: Validate that the OpenCode config omodel-wire generated actually reaches the model server — each agent's sampling matches its omodel-manager preset and OpenCode isn't falling back to model defaults. Use to prove the sampling handoff via live request logs.
---

# Validate the OpenCode sampling handoff (AI guide)

An AI-runnable procedure to prove that the OpenCode config `omodel-wire` generated
**actually reaches the model server** — i.e. each agent's sampling matches the
omodel-manager preset it maps to, and OpenCode isn't silently falling back to the
model's own defaults.

This is a **guide, not a script**: the exact agents, hosts, discriminators, and
quirks change per setup. Reason about each step; the commands below are starting
points to adapt, not a fixed recipe.

**Golden rule:** *a value in the log only proves a handoff if it differs from the
model default.* Always baseline the default first (Part B/1). Matching a value that
happens to equal the default proves nothing.

---

## Prerequisites

- The model(s) are running via **omodel-manager with `--enable-log-requests`** so vLLM
  logs the merged `SamplingParams(...)` per request (`configs` ship this flag).
- `omodel-wire` has been synced (`omw --profiles`) so `opencode.json` **and**
  `plugins/dgx-sampling.js` exist and are current. If you changed an omodel-manager
  config, **re-run `omw --profiles` first** — the `.toml` is not live until you re-sync.
- You can reach the model endpoint directly (same LAN) to send raw baseline/marker
  requests.
- Locate the binaries/paths (they vary): `opencode` (often `~/.opencode/bin/opencode`),
  `~/.config/opencode/opencode.json`, `~/.config/opencode/plugins/dgx-sampling.js`.

## Part A — Static compare (declared vs generated)

> **Shortcut:** `omw --audit` automates this whole section — it prints the side-by-side
> table (per model + agent) comparing the live OpenCode config to the omodel-manager
> presets, flags drift, and suggests `--profiles` to re-sync. Run it first; drop to the
> manual steps below only to understand a specific diff or to check something it doesn't.

1. **Source of truth:** the matched model's preset table in omodel-manager
   `configs/<key>.toml` (`[presets.reason|code|agent|instruct]` → sampling + `max_output`
   + `options`).
2. **Generated config:** read BOTH files — the sampling is split:
   - `opencode.json` agent blocks carry `temperature`, `top_p`, and the thinking
     `options` (`enable_thinking` / `preserve_thinking`) inline.
   - `plugins/dgx-sampling.js` `AGENT_SAMPLING` carries `topK`, `maxOutputTokens`, and
     body `options` (`presence_penalty`, `min_p`). **This plugin is load-bearing** — if
     it's absent/`--pure`, those knobs revert to server defaults.
3. **Map agent → preset** from the generated roster (don't hardcode — derive it):
   `research`/`agent-research`/`agent-architect`/`agent-review`/`team` → `reason`;
   `code`/`agent-code`/`agent-test` → `code`; `agent` → `agent`; and
   `agent-instruct` → `instruct`. Compare every param.
4. **Expected non-issues:** `min_p = 0.0` is *omitted* from the generated output because
   it equals the vLLM default — a missing `min_p` is not a bug. `temperature`/`top_p`
   appearing in both files (same value) is intentional redundancy.

## Part B — Live handoff proof (the part that matters)

1. **Baseline the model default.** Send ONE chat request to the endpoint with **no
   sampling params** (just `model` + `messages` + a small `max_tokens`), then read the
   logged `SamplingParams`. Record `temperature`, `top_p`, `top_k`, `presence_penalty`.
   *Do not skip this* — the default is not always obvious (observed: `temperature=1.0`,
   not the card's 0.6).
2. **Pick a discriminator per agent** — a param whose preset value ≠ the baseline.
   `presence_penalty` is ideal (preset `0.8`/`1.5` vs default `0.0`); `temperature` works
   when the preset differs from baseline. If an agent's whole vector equals the baseline,
   you cannot prove its handoff from logs — say so.
3. **For each PRIMARY agent** (`research`, `code`, `agent`, `team`):
   1. **Marker:** send a raw request with a unique `seed` (e.g. `seed=900001`,
      `max_tokens=3`) to bracket the log.
   2. **Drive it:** `opencode run --agent <name> --dir <scratch> --auto "<trivial prompt,
      no tools/edits>"`. **Never `--pure`** (it disables the plugin). Use a throwaway
      `--dir` and a prompt that won't trigger edits/bash (e.g. *"Reply with exactly the
      word: ping. Do not use any tools."*).
   3. **Read the log after the marker:** `omodel-manager logs <key> --remote <host>
      --tail N`; find the `SamplingParams` line whose `seed=` matches your marker, take
      the lines *after* it.
   4. **Ignore the aux call.** OpenCode fires a title/summary request that **bypasses the
      plugin** (`temperature≈0.5, top_p=1.0`, large `max_tokens`) — one extra line per
      run. The *agent turn* is the line whose vector matches the recipe.
   5. **Verdict:** PASS if the agent turn matches the expected preset **and** the
      discriminator differs from the baseline. If it equals the baseline instead, the
      handoff is broken (OpenCode is using defaults).

## Subagents (`mode: subagent`) — special handling

`opencode run --agent <subagent>` **does not run the subagent** — OpenCode prints
`agent "<x>" is a subagent, not a primary agent. Falling back to default agent` and runs
`default_agent` instead. So subagents can't be validated by direct invocation.

- **Verify by construction (usually enough):** the plugin sets sampling from
  `AGENT_SAMPLING[input.agent]` — it keys on the *running* agent's name, so a subagent
  gets its own vector when it runs via delegation. Since the primaries prove that exact
  code path, the subagents are covered by inspection.
- **To exercise one live (optional):** either (a) drive the `team`/a primary with a
  prompt that forces delegation to that specific worker, then find the worker's request
  in the log; or (b) copy `opencode.json`, flip the subagent to `mode: primary` in the
  copy, run with `--config <copy>`, then discard it. **Never edit the live config in
  place.**

## Gotchas (reason about these)

- **Baseline first, always.** The "model default" is empirical, not assumed.
- **Sampling is split** across `opencode.json` (inline) + the plugin — check both.
- **Aux title/summary call** bypasses the plugin — expect one extra `SamplingParams`
  line per run; don't mistake it for the agent turn.
- **`--pure` kills plugins.** Never use it here.
- **Isolate with the seed marker.** Other traffic on the box will interleave.
- **Re-sync after config edits.** A `.toml` change is invisible until `omw --profiles`.
- **`--enable-log-requests` must be on** or there is no `SamplingParams` line to read.

## Report

Emit a matrix: `agent | actual turn (T/top_p/top_k/pp/max) | expected preset | PASS/FAIL`,
plus the baseline row and any agent whose vector can't be distinguished from default.
Call out subagents as "verified by construction (not directly exercised)".

## Appendix — adaptable orchestrator (starting point, not the recipe)

```python
# marker(seed) -> raw request with unique seed; sp_after(seed) -> SamplingParams lines
# after that marker in `omodel-manager logs`. Drive each primary with opencode run,
# diff the agent turn against the expected preset vector. Adapt fields/agents per setup.
import json, urllib.request, subprocess, re, os, tempfile
BASE="http://<host>:8000"; MID="<served-model-id>"; OC=os.path.expanduser("~/.opencode/bin/opencode")
def post(body): urllib.request.urlopen(urllib.request.Request(
    BASE+"/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type":"application/json"}), timeout=60).read()
def marker(seed): post({"model":MID,"messages":[{"role":"user","content":"marker"}],"max_tokens":3,"seed":seed})
def sp_after(seed):
    out=subprocess.run(["python3","omodel-manager","logs","<key>","--remote","<host>","--tail","200"],
                       capture_output=True,text=True).stdout
    L=[re.sub(r"\x1b\[[0-9;]*m","",x) for x in out.splitlines()]
    i=max([j for j,x in enumerate(L) if ("seed=%d"%seed) in x and "SamplingParams(" in x] or [-1])
    return [x.split("SamplingParams(")[1] for x in L[i+1:] if "SamplingParams(" in x]
# baseline: post model+messages+max_tokens only (NO sampling) then sp_after -> the default
# per agent: marker(seed); subprocess.run([OC,"run","--agent",name,"--dir",tempfile.mkdtemp(),
#            "--auto","Reply with exactly the word: ping. Do not use any tools."]); sp_after(seed)
```
