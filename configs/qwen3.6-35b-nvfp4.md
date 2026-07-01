# qwen3.6-35b-nvfp4 — sampling & thinking reference

NVIDIA NVFP4 quant of Qwen3.6-35B-A3B (MoE). Reasoning model, thinking **on by
default**, tool-calling. Serve/launch via the omodel-manager `qwen3.6-35b-nvfp4`
profile. Numbers below are **starting points to sweep from**, not guarantees.

- **vision:** not enabled (declared `false` — unverified on this checkpoint; flip to
  a `modalities` object and run `omodel-wire verify` if a build turns out multimodal).
- **reasoning:** yes. **thinking control:** `enable_thinking` (Qwen `chat_template_kwargs`).
- **tool_call:** yes.

## Thinking on/off (`chat_template_kwargs`)

Thinking is a **chat-template** toggle, not a sampling param. Four levers:

1. **Per-request (raw HTTP):** top-level `"chat_template_kwargs": {"enable_thinking": false}`.
2. **Per-request (OpenAI client):** wrap in `extra_body={"chat_template_kwargs": {...}}`.
3. **Server default:** `vllm serve … --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'` (request-level always overrides).
4. **Soft switch:** `/think` or `/no_think` in a message; the model follows the most recent one.

Middle ground: cap reasoning length with `thinking_token_budget` instead of disabling.

## Sampling tuning table

For raw curl each param is a top-level JSON key. Values are sweep starting points.

| Parameter | Default | Practical tuning |
|-----------|---------|------------------|
| temperature | 1.0 | Qwen thinking baseline **0.6**. Do NOT set 0 — greedy decoding loops on thinking models. 0.6 = focus; 0.8–1.0 = variety. |
| top_p | 1.0 | Qwen baseline **0.95**. Lower to ~0.8 to tighten; keep ≥0.9 for reasoning. |
| top_k | -1 (off) | Qwen baseline **20**. Lower (10–20) tightens cheaply. |
| min_p | 0.0 | Alt tail cutoff; try 0.05–0.1 instead of aggressive top_p/top_k. |
| presence_penalty | 0.0 | **Anti-loop.** +0.5…+1.5 breaks repetition/topic-collapse. >1.5 risks language mixing. |
| frequency_penalty | 0.0 | 0.1–0.5 curbs verbatim repetition; >1.0 degrades fluency. |
| repetition_penalty | 1.0 | 1.05–1.15 discourages repeats; >1.2 hurts quality. Don't stack hard with presence/frequency. |
| max_tokens / max_output | until EOS | 16k–32k for thinking mode; the reasoning trace eats budget. |
| thinking_token_budget | none | Cap reasoning length without disabling thinking. |

Anti-loop line that matters: **temperature 0.6 (NOT 0) + presence_penalty ~0.8**.

## Per-mode baselines (the roster presets)

- **reason** (research/planning, thinking on): temp 0.6, top_p 0.95, top_k 20,
  presence_penalty 0.8, max 16k. Bump temp toward 0.8–1.0 for more exploratory research.
- **code** (precise coding, thinking on): temp 0.6, top_p 0.95, top_k 20, max 32k.
- **agent** (unattended worker): coding sampling + `preserve_thinking` so the trace
  survives across tool calls.
- **instruct** (fast, no thinking): temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5.

## machine config (omodel-wire parses the block below)

```json
{
  "match": ["qwen3.6-35b-nvfp4", "nvidia/Qwen3.6-35B-A3B-NVFP4", "Qwen3.6-35B-A3B", "Qwen3.6-35B"],
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
    "reason": {
      "desc": "Research & planning (thinking on). Anti-loop baseline temp 0.6 + presence_penalty 0.8.",
      "thinking": true,
      "max_output": 16384,
      "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.8 }
    },
    "code": {
      "desc": "Precise coding (thinking on). Card SciCode temp 0.6.",
      "thinking": true,
      "max_output": 32768,
      "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0 }
    },
    "agent": {
      "desc": "Unattended worker: coding sampling + preserved thinking across tool calls.",
      "thinking": true,
      "max_output": 32768,
      "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0 },
      "options": { "chat_template_kwargs": { "preserve_thinking": true } }
    },
    "instruct": {
      "desc": "Fast, no thinking. Qwen instruct baseline.",
      "thinking": false,
      "max_output": 32768,
      "sampling": { "temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5 }
    }
  },
  "variants": {
    "think":    { "options": { "chat_template_kwargs": { "enable_thinking": true } } },
    "no-think": { "options": { "chat_template_kwargs": { "enable_thinking": false } } }
  }
}
```

## Validating what actually ran

Start the server with `--enable-log-requests`, send a request, then read the merged
`SamplingParams(...)` line (your request merged over `generation_config.json`):

```bash
docker logs -f otools-vllm-qwen3.6-35b-nvfp4 2>&1 | grep -i sampling
```

If a value you sent isn't there, `generation_config.json` or a server default overrode it.
