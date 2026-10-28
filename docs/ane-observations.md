# ANE execution checks

Tested on 2026-10-28 using an Apple M5 MacBook Air, 16 GB unified memory,
macOS 26.4.1 build 25E253, and Instruments 16.0 build 17F113.

These are execution checks through ANE Studio's production Python generation
code. Timing collected under Instruments is profiling data, not a clean
performance benchmark. The fixed graph has a 64-token total context.

## Measured results

| Checkpoint | Worker predictions | ANE predictions overlapping generation | Finding |
| --- | ---: | ---: | --- |
| LFM2.5 230M | 80 across 8 replies | 80 | ANE execution observed |
| LFM2.5 350M | 24 across 8 replies | 24 | ANE execution observed |
| SmolLM2 135M Instruct | 24 across 8 replies | 0 | CPU fallback under CPU+ANE |
| Qwen2.5 0.5B Instruct | 40 across 4 replies | 0 | CPU fallback under CPU+ANE |
| LFM3 1.7B, FP32 | 2 one-token probes | 0 | CPU fallback under CPU+ANE; GPU chat passed |

The LFM CPU-only controls generated the same replies with zero ANE prediction
events. SmolLM2's CPU-only control also recorded zero ANE predictions. The ANE
lane was present in each export. Compilation and model-load events do not count
as prediction events.

Both LFM2.5 artifacts retained their historical passing parity records and passed
the worker's artifact-digest check before generation. This run did not repeat
their conversion or upgrade their historical single-fixture parity to the newer
four-fixture gate. Qwen and SmolLM2 passed fresh four-fixture conversion against
the unpatched FP32 source before profiling. Their minimum cosines were
`0.999999999817` and `0.999827711881` respectively; all top tokens matched.

LFM3 passed the four-fixture gate only with full FP32 source and conversion:
minimum cosine `0.999999999446`, all top tokens matching. Its GPU-policy chat
returned `Hello! How can I help you today?`, nine visible tokens followed by EOS.
The final runtime completed three GPU-policy replies and two Automatic-policy
replies. A real Electron CPU-policy test asked `Name one planet.` and received
`One planet is **Earth**.`, six visible tokens followed by EOS. The test closed
the app and confirmed the worker session temporary directory was removed.
The completed CPU+ANE trace contained two one-token generations and zero ANE
predictions. A longer earlier capture exceeded its recording window and is
retained as incomplete, not included in the table. One diagnostic run also
crashed in native input cleanup. See the runtime note below.

ANE use does not establish that every operation ran on ANE. The CPU+ANE policy
permits CPU fallback, and the LFM plans included CPU operations. The evidence is
specific to these packages, this workload, and this Mac.

Small machine-readable records are retained in `docs/evidence/`. Full local
workloads, parity records, trace exports, traces, and cleanup records are under
the ignored `artifacts/ane-audit-2026-10-28/` directory. Original model manifests
and benchmark reports were preserved before model removal.
The tested source weights, converted packages, and identified compiled orphans
were removed after the checks. The app now requires a fresh download/conversion
before running one of these models again. Raw traces are preserved in the local
`traces.tar.gz` archive; extract it in that directory to restore the original
trace paths before opening them or repeating export analysis.

## Repeat a check

Prepare a model using the app or the worker's `prepare` command in an isolated
`ANE_STUDIO_HOME`. A failed parity result must not proceed to chat or profiling.
Compile the public signpost helper once:

```bash
mkdir -p artifacts/profile
xcrun clang -dynamiclib -O2 scripts/inference-signpost.c \
  -o artifacts/profile/inference-signpost.dylib
```

Run one model and policy at a time:

```bash
uv run --project worker --extra coreml --locked python scripts/profile-model.py \
  --home "$ANE_STUDIO_HOME" \
  --model LiquidAI/LFM2.5-230M \
  --policy cpuAndNeuralEngine --repeats 8 \
  --signpost-library artifacts/profile/inference-signpost.dylib \
  --trace artifacts/profile/lfm230-ane.trace \
  --output artifacts/profile/lfm230-ane.json

uv run --project worker --extra coreml --locked python scripts/summarize-profile.py \
  --trace artifacts/profile/lfm230-ane.trace \
  --workload artifacts/profile/lfm230-ane.json \
  --output artifacts/profile/lfm230-observed.json
```

The profiler loads the model before attaching Instruments. A Darwin notification
confirms that recording has started before generation begins. Public signposts
mark each generation, and the workload records its PID, source revision, package
digest, policy, replies, and prediction count. Recording must remain active for
every generation; `--trace-seconds` controls the safety limit.
For slow FP32 graphs, use `--max-new-tokens 1 --repeats 2` to keep the observed
interval within that limit. A complete short probe establishes execution
placement, not long-reply latency or model quality.

The summary exports both lanes from the same trace, validates the expected XML
schemas, checks the worker PID, and requires one complete signpost interval per
reply. Missing instrumentation is an error, not a negative ANE result. Repeat
with `--policy cpuOnly` and different output paths as an exclusion control.
ANE hardware events are system-wide, so retain the control and correlate events
with the worker intervals. Preserve the raw trace when other ANE workloads could
make attribution ambiguous.

## Runtime and storage findings

The LFM3 diagnostic crash matched the delayed NumPy-owner release described in
[Apple's September 2026 fix](https://github.com/apple/coremltools/pull/2829).
That native fix is merged upstream but is absent from the pinned 9.0 wheel.
The worker retains immutable prediction inputs until Core ML has released its
references. It does not mutate or reuse arrays still borrowed by native code.
This is a caller-side mitigation; it does not replace the upstream GIL fix.
The owner registry is specific to the bundled CPython 3.12 layout. A bounded
normal-exit drain is best effort. Adopting a released wheel containing the native
fix remains a follow-up; no general interpreter-shutdown guarantee is claimed.

Two interrupted diagnostic processes left roughly 6.4 GiB each of compiled
weights in Foundation's default temporary directory. Foundation ignored
`TMPDIR`. Completed compilations are now explicitly moved to a private worker
session directory before loading `CompiledMLModel`. The Electron supervisor
waits for worker termination before removing that session directory, including
crash and cancellation paths. Native compilation initially uses Foundation's
directory, so interruption before that move can still leave a native orphan.
Only the audit's identified orphans were removed; unrelated temp files were
left alone. The cleanup ledger records the exact paths and sizes.

## Current references

- [Apple's WWDC26 introduction to Core AI](https://developer.apple.com/videos/play/wwdc2026/324/)
  and [2026 Instruments guidance](https://developer.apple.com/documentation/coreai/analyzing-model-runtime-performance-with-instruments)
  describe correlation of model activity with Neural Engine and GPU activity.
  Those Core AI examples use a different runtime from this app's Core ML worker.
  The installed Core ML Instruments template supplies the hardware lane used here.
- [Shahir's September 2026 revised measurement study](https://arxiv.org/abs/2608.22110v2)
  measures placement together with weight encoding and execution controls. It
  corrected earlier byte accounting and does not establish a general 25M-30M
  parameter limit. Do not reuse the withdrawn residency interpretations.
- [Liquid AI's LFM3 model card](https://huggingface.co/LiquidAI/LFM3-1.7B)
  documents the new tri-hybrid architecture. Native MLX and GGUF availability
  does not establish ANE execution through this app's Core ML converter.
