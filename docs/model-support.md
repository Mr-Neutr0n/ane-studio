# Model support

## Admission rules

A checkpoint can enter ANE Studio when all of these hold:

- It is a causal text-generation model on Hugging Face.
- Its exact Transformers architecture has a registered adapter.
- The tokenizer and model load without remote custom code.
- A fixed-shape FP16 trace converts to a Core ML ML Program.
- Source and Core ML logits reach cosine similarity `0.999` or higher.
- Both runtimes choose the same top token for the parity prompt.

The app rejects unsupported architectures before downloading weights. A model name, parameter count, or shared base family does not bypass the adapter gate.

## Registered adapters

### LFM2

`Lfm2ForCausalLM` combines short convolutions and grouped-query attention. Transformers currently exposes two trace gaps that the adapter rewrites without changing model math:

- The dynamic causal-mask helper becomes a fixed upper-triangular additive mask.
- The short-convolution fallback uses the model's declared kernel size as a trace constant.

Tested checkpoints:

- `LiquidAI/LFM2.5-230M`: passed at revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45`.
- `LiquidAI/LFM2.5-350M`: passed at revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.

### Qwen2

`Qwen2ForCausalLM` uses eager grouped-query attention and a fixed causal mask during tracing.

The adapter passed conversion and parity with `yujiepan/qwen2-tiny-random`. Architecture support does not imply that a larger checkpoint passes. `Qwen/Qwen2.5-0.5B-Instruct` converted structurally but failed parity with cosine `0.903762` and a different top token.

`WeiboAI/VibeThinker-3B` uses the same architecture, but its full checkpoint has not been converted on the 16 GB development Mac because source and FP16 artifact storage require more free disk and memory than were available.

## Known failed checkpoints

Failures are pinned to exact Hugging Face revisions in `catalog.py` so the app rejects them before another large download:

- `HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f77366fa6b3b4b768ec3050bf629380bac`: the converted Llama graph returned zero logits. Llama remains unregistered.
- `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`: logit and top-token parity failed.

A new model revision or adapter fix can remove a block only after repeating conversion and parity on the Mac.

## Networked smoke commands

These commands download models and write generated artifacts outside Git.

Test the small LFM2 checkpoint end to end:

```bash
printf '%s\n' \
  '{"id":"lfm","command":"prepare","payload":{"modelId":"LiquidAI/LFM2.5-230M","sequenceLength":64}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
```

Test the second validated LFM2 checkpoint:

```bash
printf '%s\n' \
  '{"id":"lfm350","command":"prepare","payload":{"modelId":"LiquidAI/LFM2.5-350M","sequenceLength":64}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
```

Test the Qwen2 adapter with a small fixture:

```bash
export ANE_STUDIO_HOME="$(mktemp -d /tmp/ane-studio-qwen.XXXXXX)"
printf '%s\n' \
  '{"id":"qwen","command":"prepare","payload":{"modelId":"yujiepan/qwen2-tiny-random","sequenceLength":16}}' \
  | uv run --project worker --extra coreml --locked ane-studio-worker
```

A successful final response has `state: ready`, `parity.status: passed`, and an artifact digest.

## Adding an adapter

1. Add the exact Transformers architecture to `worker/src/ane_studio_worker/catalog.py`.
2. Add only the trace rewrites needed for equivalent model math.
3. Convert a tiny public checkpoint for the architecture.
4. Verify logit cosine similarity and top-token agreement.
5. Test a useful checkpoint at its intended size.
6. Record model-specific licenses and resource limits.
7. Run chat and all four compute policies on Apple Silicon.

Keep architecture support separate from checkpoint validation. Passing a tiny fixture proves the conversion path, not the quality or resource behavior of every checkpoint in that family.
