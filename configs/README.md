# `configs/` — per-model declared configs

These files replace the slow live **probing** (vision / reasoning detection) with
**declared capabilities + tuning**, one file per model. They are the source of
truth `omodel-wire` reads to build OpenCode providers and the agent roster.

- **One file per model**, named to match an **omodel-manager** profile/model key
  (e.g. `qwen3.6-35b-nvfp4.md` ↔ the `qwen3.6-35b-nvfp4` launch profile).
- Each file is **both a human tuning README and a machine config**: prose docs +
  a single fenced ` ```json ` block the tool parses. Edit the block to tune; the
  prose around it explains what each value does (JSON has no comments — the doc is
  the comment).
- The tool extracts the **first ` ```json ` block** in the file, `json.loads` it,
  and matches a discovered served-model-id against its `match` list.

## The machine block

````markdown
```json
{
  "match": ["qwen3.6-35b-nvfp4", "nvidia/Qwen3.6-35B-A3B-NVFP4", "Qwen3.6-35B"],
  "source": "https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4",
  "capabilities": {
    "vision": false,
    "reasoning": true,
    "tool_call": true,
    "thinking_control": "enable_thinking"
  },
  "context": { "native": 262144, "min_thinking": 131072 },
  "thinking_default": true,
  "soft_switch": false,
  "presets": {
    "reason":   { "thinking": true,  "max_output": 16384, "sampling": { ... } },
    "code":     { "thinking": true,  "max_output": 32768, "sampling": { ... } },
    "agent":    { "thinking": true,  "options": { "chat_template_kwargs": { "preserve_thinking": true } }, "sampling": { ... } },
    "instruct": { "thinking": false, "sampling": { ... } }
  },
  "variants": { "think": { ... }, "no-think": { ... } }
}
```
````

### Field reference

| Key | Meaning |
|-----|---------|
| `match` | List of substrings matched (case-insensitive) against the served-model-id from `/v1/models`. Include the omodel-manager key **and** the HF/served id(s) it answers to. |
| `source` | Model-card URL the tuning came from (documentation only). |
| `capabilities.vision` | `false`, or an object `{ "input": ["text","image"], "output": ["text"] }`. When truthy the tool writes `modalities` + `attachment: true`. **Replaces the vision probe.** |
| `capabilities.reasoning` | `true`/`false` — does the model emit chain-of-thought. **Replaces the reasoning probe.** |
| `capabilities.tool_call` | `true`/`false` — written as `tool_call` on the model (required or OpenCode sends no tools to custom openai-compatible models). |
| `capabilities.thinking_control` | How thinking is toggled: `"enable_thinking"` (Qwen `chat_template_kwargs`), `"reasoning_effort"` (vLLM low/med/high), `"soft_switch"` (`/think` `/no_think` in the prompt), or `"none"` (leave the model at its template default; per-preset `options` carry any knobs). Drives the caps the roster builder used to get from probing. |
| `context.native` / `min_thinking` | Context length facts (documentation + sanity). |
| `presets` | The **roles** the roster is built from (`reason`, `code`, `agent`, `instruct`). Each: `thinking` (bool), optional `max_output`, a `sampling` block, and optional raw `options` (e.g. `chat_template_kwargs`). Multiple agents/subagents map onto these presets — see AGENTS.md. |
| `variants` | Optional Ctrl+T thinking-depth presets (raw `options` only). |

### Supported `sampling` params

The tool forwards these to OpenCode (top-level where the API supports it, else via
the agent-aware `chat.params` plugin's `options` → provider body):

`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
`repetition_penalty`, `max_output` (→ `maxOutputTokens`).

Non-standard body params (e.g. `thinking_token_budget`, `chat_template_kwargs`) go
under a preset's `options` and are passed through the plugin as-is.

## Verifying declared capabilities

Declared configs trust you got the capabilities right. To confirm against a live
endpoint occasionally (not every run), use the opt-in probe:

```bash
omodel-wire verify <model-or-host>     # re-runs the vision/reasoning probes and
                                        # reports mismatches vs the declared config
```

## Conventions

- Keep the filename == the omodel-manager profile key so the two tools line up.
- If two launch profiles serve the **same** model (e.g. 256k vs 512k context),
  they share one config here — sampling is per-model, not per-launch.
- A model with no matching config falls back to `_default.md`.
- The test suite (`test_omodel_wire.py`) parses every config and checks required
  keys — run it after editing.
