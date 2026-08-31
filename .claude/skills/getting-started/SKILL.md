---
name: getting-started
description: End-to-end onboarding for the otools model stack — serve local models on managed card, node, or cluster devices with omodel-manager (omm) and wire them into OpenCode with omodel-wire (omw).
---

# Getting started with the otools model stack

Two sibling tools, run in this order:

| Tool | Alias | Job |
| --- | --- | --- |
| **omodel-manager** | `omm` | Launches and manages vLLM deployments on card, node, and cluster devices. |
| **omodel-wire** | `omw` | Probes the manager's deployment registry and wires live models into **OpenCode**. |

`omw` reads `omm`'s recursive `configs/card`, `configs/node`, and `configs/cluster`
model configs plus `~/.config/otools/deployments.json` (override
`OMODEL_MANAGER_DEPLOYMENTS`). Keep both repos checked out side by side.

## How to drive this guide (for the AI assistant)

- **Run the read-only / inspection commands yourself and show the output** — `omm list`, `omm ps`, `omm health …`, `omw agents`, `omw models`, `omw audit`. These are safe.
- For anything that **touches a remote host, starts a container, or is interactive** — `omm install`, `omm launch`, the OpenCode installer, `shell-init` (edits their rc file) — **give a copy-paste block and offer to run it**, but let the user confirm first (these do real work / may prompt).
- The shell env is **WSL/Linux with `python3`** (there is no bare `python`). Use `python3` for the one-time bootstrap; after `shell-init` the `omm`/`omw` aliases work from anywhere.

---

## 1. Install the shell aliases (one-time)

Run each from inside its repo, then re-open the shell:

```bash
# in the omodel-manager repo
python3 omodel-manager shell-init      # installs the `omm` alias
# in the omodel-wire repo
python3 omodel-wire.py shell-init      # installs the `omw` alias
```

This appends an alias to your shell rc (`~/.bashrc` / `~/.zshrc` / fish config). Idempotent — safe to re-run. **Open a new shell** so `omm` and `omw` resolve.

---

## 2. Provision a DGX host (one-time per host)

`omm install` bootstraps a remote host over SSH — generates a dedicated key (`~/.ssh/otools_model_manager_ed25519`), installs Docker + checks the NVIDIA runtime/CDI, and **registers the host under a short alias** in `~/.config/otools/hosts`.

```bash
omm install user@<host-ip> <alias> --fix     # e.g. user@192.0.2.101 dgx1 --fix
```

- `<alias>` (e.g. `dgx1`) is the short name you'll pass as `--host dgx1` everywhere after.
- `--fix` remediates what's missing (installs Docker, adds you to the `docker` group, sets up the scoped sudo rule, and **prompts for a HuggingFace token if none is set** — see step 4).
- This is interactive and SSHes into the box — **the user should run it.** Offer the command; don't run it for them.

Check what's registered / reachable:

```bash
omm ps            # all registered hosts + any running containers (safe — run this to confirm)
```

---

## 3. Launch your first model + watch it load

Recommended first model: **`unsloth-qwen3-coder-next-fp8`** (a fast, capable coder).

```bash
omm launch unsloth-qwen3-coder-next-fp8 --host dgx1     # start the model container
omm logs   unsloth-qwen3-coder-next-fp8 --host dgx1 -f  # follow vLLM's startup logs (Ctrl-C detaches; container keeps running)
omm health unsloth-qwen3-coder-next-fp8 --host dgx1     # confirm it's serving
```

- `launch` returns immediately if the image isn't cached yet (it pulls in the background — poll with `omm pull-status`), then starts serving. Watch `logs -f` until vLLM prints that it's ready.
- See every available model to launch: `omm list` (safe — run it to show the catalogue).
- `launch`/`logs` reach the DGX — offer the commands and run them **if the user is ready to start a container**; otherwise hand them the copy-paste.

---

## 4. HuggingFace token (optional)

Only needed for **gated** models (e.g. `nvidia/*`). The recommended `unsloth-qwen3-coder-next-fp8` does **not** need one.

- Provide it via `$HF_TOKEN`, or let `omm install … --fix` store it at `~/.config/otools/hf_token` (chmod 600).
- Launches forward it automatically. It's never written into the repo or shown in dry-run output.

```bash
# optional — only for gated models:
umask 077; printf %s '<hf_token>' > ~/.config/otools/hf_token
```

---

## 5. Install OpenCode

OpenCode is the agent runtime that `omw` configures. Install (recommended one-liner):

```bash
curl -fsSL https://opencode.ai/install | bash      # installs to ~/.opencode/bin
```

Alternatives: `npm install -g opencode-ai` (or `bun`/`pnpm`). Docs: <https://opencode.ai/docs/>. Re-open your shell so `opencode` is on `PATH`.

---

## 6. Sync OpenCode (omw)

With at least one model live (step 3) and OpenCode installed, wire it up:

```bash
omw sync         # probes your live endpoints, reads omm's configs, writes ~/.config/opencode/opencode.json
```

`omw sync` probes each recorded deployment `base_url` exactly, ignores stale/dead
records, and refreshes only managed `otools-*` providers plus native OpenCode Build/Plan
model selections and their sampling plugin. Re-run it whenever deployments change.
`--dry-run` previews the result without writing. Hosts and ports in `wire.json` are
retained only for unmanaged fallback when the deployment registry is absent or valid-empty.
An invalid registry fails closed instead of probing unrelated legacy endpoints.

Confirm it worked (safe — run these and show output):

```bash
omw verify       # optional slow live capability check against declarations
opencode models  # confirm otools-<device>/<served-id> refs are present
opencode         # launch OpenCode; native Build and Plan use the synced presets
```

---

## 7. Select Build and Plan models

```bash
opencode models       # includes otools-<device>/<served-id>
omw sync --dry-run    # shows the Build/Plan choices without writing
```

Use `omw sync --build-model REF --plan-model REF` to select explicit device-qualified
refs. Without those flags, sync preserves a still-live native selection or chooses a
compatible live model. Legacy unmanaged discovery may still produce `dgx-*` refs during
migration, but manager deployments use `otools-*`.

---

## Quick recap (copy-paste the whole path)

```bash
# 1. aliases (once, then re-open shell)
python3 omodel-manager shell-init ; python3 omodel-wire.py shell-init
# 2. provision a host (once per host — user runs; interactive)
omm install user@<host-ip> dgx1 --fix
# 3. launch + watch
omm launch unsloth-qwen3-coder-next-fp8 --host dgx1
omm logs   unsloth-qwen3-coder-next-fp8 --host dgx1 -f
# 4. install OpenCode
curl -fsSL https://opencode.ai/install | bash
# 5. wire it up + verify
omw sync ; omw agents ; omw models ; opencode
```
