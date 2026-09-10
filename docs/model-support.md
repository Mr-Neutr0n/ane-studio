# Model support

## Admission rules

A checkpoint can enter ANE Studio when all of these hold:

- It is a causal text-generation model on Hugging Face.
- Its exact Transformers architecture has a registered adapter.
- Tokenizer and model load without remote custom code.
- The fixed-shape trace converts to a Core ML ML Program.
- Every parity fixture reaches cosine similarity `0.999` or higher and chooses the same top token.
- Both source and Core ML logits are finite and nonzero.

An architecture name does not bypass parity. The app pins model metadata and downloads to an exact Hugging Face revision. A failed artifact cannot run chat or benchmarks.

Converter version 3 checks four prompts against the original, unpatched Transformers forward:

| Fixture | Text | Selected position in the two added checkpoints |
| --- | --- | ---: |
| `completion` | `The capital of France is` | 4 |
| `short` | `The opposite of hot is` | 4 |
| `first-token` | `Hello` | 0 |
| `context-boundary` | `A short sentence about the sky and the sea. ` repeated and truncated to the graph length | 63 |

Every check must pass. The manifest stores individual checks. Its aggregate cosine is the minimum, maximum absolute error is the maximum, and `topTokenMatches` is true only when every check agrees. The top-level token IDs refer to `completion`.

## Registered adapters

### LFM2

`Lfm2ForCausalLM` combines short convolutions and grouped-query attention. The trace adapter replaces the dynamic causal mask with a fixed additive mask and uses the declared convolution kernel size as a trace constant. Patches restore the original functions even when tracing fails.

LFM2 keeps FP16 source loading and FP16 conversion. Its parity reference is the unpatched FP16 Transformers model. Historical validated checkpoints:

- `LiquidAI/LFM2.5-230M` at `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
- `LiquidAI/LFM2.5-350M` at `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.

These weights use the LFM Open License, not Apache 2.0 or MIT.

### Qwen2 and Llama

`Qwen2ForCausalLM` and `LlamaForCausalLM` use eager grouped-query attention with the same scoped fixed causal-mask rewrite. Both load the source in FP32.

Qwen2 converts in full FP32. Mixed FP16 matrix compute passed three prompts but failed the one-token `Hello` fixture at position zero, with cosine `0.977045`. Restricting FP16 to linear projections still failed at `0.982327`. Full FP32 passed all four without changing the threshold or model math.

Llama uses mixed precision. Core ML's FP16 pass selects only `linear` and `matmul` operations. Embeddings, RMS normalization, residual operations and softmax retain FP32. The fixed-position RoPE angles fold to FP32 sine/cosine constants before that pass. This is not a claim about future dynamic-position RoPE graphs.

The adapter selects the current hidden-state row before projecting to the vocabulary. It does not compute a full sequence of vocabulary vectors for each generation step.

The two useful Apache 2.0 checkpoints are:

- `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`.
- `HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

See [compatibility results](compatibility-results.md) for exact parity, chat, policy results and compilation limits. These small models can produce poor answers even when conversion parity passes. ANE Studio does not reproduce their advertised long-context limits.

`yujiepan/qwen2-tiny-random` is a converter regression fixture, not a useful chat model. `WeiboAI/VibeThinker-3B` has a registered architecture but has not passed full-checkpoint conversion. Its FP32 source alone needs about 12 GB, before tracing and conversion copies. Do not attempt it on a busy 16 GB Mac.

## Numerical failure history

The same pinned Qwen2.5 and SmolLM2 revisions failed the earlier blanket-FP16 conversion. Qwen2.5 had cosine `0.903762` and a different top token. SmolLM2 returned zero logits.

Full-FP32 and selective-matrix-FP16 probes isolated the failure to precision lowering. A separate synthetic RMS normalization check reproduced finite zero outputs when an FP16 square overflowed. That explains a possible mechanism, not the exact failing operator in the original full checkpoint. No epsilon, weight, cosine threshold, or top-token requirement was changed.

The revision blocks were removed only after the repaired conversion passed all four real unpatched-source fixtures on the Mac. New revisions still pass through conversion and the same gate.

## Runtime and resource limits

- The default graph has a total context of 64 tokens, including the chat template and generated tokens. It is not a KV-cache decoder.
- Input token IDs pass directly into Core ML. Generation never decodes and re-encodes its history.
- A prompt that fills the graph returns an explicit error. The app does not silently discard the chat template.
- Generation reports `eos`, `contextLimit`, or `maxTokens`, plus token and context counts.
- The fixed mask is equivalent only for a contiguous, right-padded valid prefix. It does not support left padding or interior mask holes.
- Mixed precision uses more disk than blanket FP16. Current packages are about 383 MB for mixed-precision SmolLM2 and 1.98 GB for FP32 Qwen2.5.
- Start the added checkpoints with CPU or GPU. First CPU+ANE compilation can take minutes and may fall back to CPU.
- Requested compute policy and planned placement do not prove observed accelerator activity. No Instruments trace was captured for these tests.

## Networked smoke commands

These commands use the repository's native worker environment. Generated artifacts stay outside Git. The legacy `ANE_STUDIO_HOME` name is intentional and preserves existing data paths.

```bash
export ANE_STUDIO_HOME="$(mktemp -d /tmp/ane-model-smoke.XXXXXX)"
printf '%s\n' '{"id":"smol","command":"prepare","payload":{"modelId":"HuggingFaceTB/SmolLM2-135M-Instruct","sequenceLength":64}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
```

Prepare Qwen2.5 in the same isolated home, after the first command exits:

```bash
printf '%s\n' '{"id":"qwen","command":"prepare","payload":{"modelId":"Qwen/Qwen2.5-0.5B-Instruct","sequenceLength":64}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
```

Chat and compare policies only after preparation returns `state: ready`:

```bash
printf '%s\n' '{"id":"chat","command":"chat","payload":{"modelId":"HuggingFaceTB/SmolLM2-135M-Instruct","policy":"cpuOnly","messages":[{"role":"user","content":"Say hello in one short sentence."}],"maxNewTokens":16,"temperature":0}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
printf '%s\n' '{"id":"bench","command":"benchmark","payload":{"modelId":"HuggingFaceTB/SmolLM2-135M-Instruct","prompt":"Count from one to ten.","policies":["cpuOnly","cpuAndGPU","cpuAndNeuralEngine","all"]}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
```

Regression preparations, one at a time:

```bash
printf '%s\n' '{"id":"lfm","command":"prepare","payload":{"modelId":"LiquidAI/LFM2.5-230M","sequenceLength":64}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
printf '%s\n' '{"id":"lfm350","command":"prepare","payload":{"modelId":"LiquidAI/LFM2.5-350M","sequenceLength":64}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
printf '%s\n' '{"id":"tiny","command":"prepare","payload":{"modelId":"yujiepan/qwen2-tiny-random","sequenceLength":64}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
```

A successful response contains passing parity and an artifact digest. Keep model-family tests separate from model-quality claims. Run chat and all four policy attempts on Apple Silicon before reporting a checkpoint as tested. Record timeouts or fallback instead of inferring ANE execution from an allowed policy.
