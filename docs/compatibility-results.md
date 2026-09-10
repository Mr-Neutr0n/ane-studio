# Apple M5 compatibility results

Test date: 2026-09-08. Host: Apple M5 MacBook Air, 16 GB unified memory, macOS 26.4.1 build 25E253. These are local compatibility checks, not cross-machine performance claims.

Environment: Python 3.12.13, PyTorch 2.7.0, Transformers 5.16.1, coremltools 9.0, NumPy 2.2.6. Jobs ran sequentially. Generated weights, packages, logs and machine-readable results are outside Git under `/tmp/ane-audit/`.

## Source parity

The graph has 64 tokens. All four fixtures must have finite, nonzero logits, cosine similarity at least `0.999`, and the same source/Core ML top token. Reference logits come from the original Transformers forward before the scoped trace patches. The fixtures test positions 0, 4 and 63.

| Model | Exact revision | Precision | Package | Minimum cosine | Top tokens |
| --- | --- | --- | ---: | ---: | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | FP16 linear/matmul, FP32 sensitive operations and source | 383,167,129 B | `0.9998536161` | All four match |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | FP32 | 1,977,228,431 B | `0.9999999998` | All four match |

The production `Service.prepare` path completed for Qwen with converter version 3 in 55.3 seconds. Its FP32 artifact digest is `e4da509e15ebb1e3d1ca7d07377ed0f013ebe30251695cefcdb57749af4038e1`. All four requested policies then passed the same four source fixtures and chat against that digest. Its earlier mixed-precision packages were rejected and removed.

### Why FP32 is necessary for Qwen

The earlier blanket-FP16 adapter failed Qwen at cosine `0.903762` and returned zero SmolLM2 logits. The matrix-only FP16 rewrite fixed SmolLM2 and initially passed three Qwen prompts. The new one-token `Hello` fixture exposed a remaining Qwen error:

| Qwen conversion | Position-zero cosine | Top token | Decision |
| --- | ---: | --- | --- |
| FP16 linear and attention matmul, sensitive operations FP32 | `0.9770449016` | Match | Reject |
| FP16 linear only, other operations FP32 | `0.9823271174` | Match | Reject |
| Full FP32 | `0.9999999997` | Match | Pass four source fixtures |

Neither top-token agreement alone nor passing the other prompts is enough. The gate was not relaxed. Converter version 3 uses FP32 for Qwen2.

The precision diagnosis also reproduced RMS normalization collapse with a synthetic FP16 square overflow. That is a possible mechanism for the old zero-logit result, not proof of the specific failing operator in the full SmolLM2 graph.

The final SmolLM2 `Service.prepare` completed in 21.9 seconds. Its converter-v3 artifact digest is `3ce36e44d897208ec1228eb25662fe724c61575be36a7021dd47fbff70bd40fe`. Its CPU parity minimum is `0.9998536161` across all four fixtures. The earlier mixed-precision artifact also passed, but its values are not used as final-package parity evidence.

## SmolLM2 chat and policies

Prompt: `Say hello in one short sentence.` Every completed policy returned `Hello!`.

The final converter-v3 artifact passed all four source fixtures and chat with CPU only, CPU+GPU and Automatic. Its cold CPU+ANE retry exceeded the 90-second supervised budget and remains incomplete. The earlier v2 artifact completed all four requested policies, but that historical fallback does not validate the refreshed artifact's CPU+ANE path.

Chat generated two visible tokens, then EOS, using 39 of 64 context tokens. A warm timing run retained eight samples for `Count from one to ten.` The model's answer began `I'm sorry for the confusion, but`. This is a compatibility smoke, not a quality endorsement.

| Requested policy | Result | Planned operations |
| --- | --- | --- |
| CPU only | Passed | 2,206 CPU, 1 unknown |
| CPU + GPU | Passed | 2,206 GPU, 1 unknown |
| CPU + ANE | Final artifact timed out after 90 s | No final result |
| Automatic | Passed all four fixtures and chat | 2,206 GPU, 1 unknown |

The earlier artifact's ANE compiler emitted a failure during a 419.9-second cold CPU+ANE policy run. The requested policy still allowed CPU fallback and produced valid logits. This does not demonstrate ANE acceleration. Use CPU or GPU for faster startup. That earlier four-policy run took 482.6 seconds. The final retry used a native 90-second process timer. No system service was killed.

No Instruments trace was captured. Requested policy, planned placement and observed activity remain separate. Planned operation counts are not runtime device measurements.

## Qwen2.5 chat and policies

Every policy returned `Hello! How can I assist you today?`, nine visible tokens followed by EOS, with 45 of 64 context tokens consumed. Timing retained eight tokens after one warmup for `Count from one to ten.` Its reply began `Sure, here's a simple Python code`. This is not a model-quality benchmark.

| Requested policy | Median token time | Throughput | Planned operations |
| --- | ---: | ---: | --- |
| CPU only | 453.1 ms | 2.21 tok/s | 1,360 CPU, 1 unknown |
| CPU + GPU | 87.0 ms | 11.49 tok/s | 1,360 GPU, 1 unknown |
| CPU + ANE | 136.8 ms | 7.31 tok/s | 1,360 CPU, 1 unknown |
| Automatic | 67.5 ms | 14.82 tok/s | 1,360 GPU, 1 unknown |

Each complete policy attempt stayed below its 90-second supervised limit. These timings varied as the Mac's load and caches changed. The different CPU-only and CPU+ANE timings do not prove ANE activity. Both plans selected CPU, with zero planned ANE operations. GPU or Automatic was more practical for this FP32 graph on the test Mac.

## Historical LFM2 results

The earlier 2026-08-29 checks used a single completion fixture, FP16 conversion and CPU-only Core ML parity:

| Checkpoint | Exact revision | Package | Cosine | Top token |
| --- | --- | ---: | ---: | --- |
| `LiquidAI/LFM2.5-230M` | `40cb2ad3b3044d5a41eee083a6103c8b523afa45` | 459.8 MB | `0.999936` | Match |
| `LiquidAI/LFM2.5-350M` | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` | 709.4 MB | `0.999706` | Match |
| `yujiepan/qwen2-tiny-random` | `b01b9c82aaf1efb4c26c94b6342d611b397245ff` | 5.2 MB | `0.999995` | Match |

The 350M model returned `Hello!` in that earlier chat smoke. CPU+ANE planned 699 ANE and 17 CPU operations. That plan did not prove observed ANE execution. Current regression evidence will be listed separately rather than replacing historical values.

## Storage and safety

Compilation uses a managed temporary directory beside each package. The worker removes it after loading or compute-plan inspection, including failed calls. Conversion defers model loading until source and trace objects are released. Policy comparisons release each runtime before compute-plan inspection and before loading the next policy.

The audit kept existing user data intact. It removed only its own superseded diagnostic packages to recover disk. Logs and result JSON remained available. No model weights or converted packages belong in Git or the app distribution.

Machine-readable index: `/tmp/ane-audit/model-summary.json`. Environment: `/tmp/ane-audit/environment.json`. Numerical review: `/tmp/ane-audit/numerical-review.md`.
