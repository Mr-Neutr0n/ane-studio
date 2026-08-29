# M5 compatibility results

Test date: 2026-08-29

Host: Apple M5 MacBook Air, 16 GB unified memory, macOS 26.4.1 build 25E253. The Mac was on AC power and reported no thermal warning. These are local development checks, not cross-machine benchmark claims.

## Conversion results

All attempts used a fixed 64-token graph, FP16 conversion, a CPU-only Core ML parity prediction, minimum cosine similarity `0.999`, and top-token agreement.

| Model and pinned revision | Architecture | Artifact | Cosine | Top token | Result |
| --- | --- | ---: | ---: | --- | --- |
| `LiquidAI/LFM2.5-230M` `40cb2ad3` | LFM2 | 459.8 MB | `0.999936` | Match | Passed |
| `LiquidAI/LFM2.5-350M` `9e6c6ccf` | LFM2 | 709.4 MB | `0.999706` | Match | Passed |
| `Qwen/Qwen2.5-0.5B-Instruct` `7ae55760` | Qwen2 | 988.8 MB | `0.903762` | Mismatch | Failed |
| `HuggingFaceTB/SmolLM2-135M-Instruct` `12fd25f7` | Llama | 269.7 MB | `0.000000` | Mismatch | Failed |
| `yujiepan/qwen2-tiny-random` `b01b9c82` | Qwen2 fixture | 5.2 MB | `0.999995` | Match | Passed fixture |

The Qwen2.5 checkpoint was also tried with FP32-sensitive normalization and softmax operations. That probe reached cosine `0.886621` and still chose a different top token, so weakening parity was rejected.

The SmolLM2 package returned finite but entirely zero logits from Core ML. A constant-position diagnostic produced nonzero logits but only `0.790244` cosine. Llama remains unregistered.

`google/gemma-3-270m-it` was inspected but not downloaded because the repository requires accepted access terms and an authenticated token.

## LFM2.5-350M chat

Prompt: `Say hello in one short sentence.`

Output: `Hello!`

CPU+ANE generation reported a 34.1 ms first token and 57.8 tok/s median throughput for this short response.

## LFM2.5-350M policy comparison

The app ran one warmup generation and retained eight token samples per policy.

| Policy | Median token | Throughput | Planned operations |
| --- | ---: | ---: | --- |
| CPU only | 35.5 ms | 28.2 tok/s | 716 CPU |
| CPU + GPU | 12.3 ms | 81.5 tok/s | 716 GPU |
| CPU + ANE | 20.9 ms | 47.9 tok/s | 699 ANE, 17 CPU |
| Automatic | 35.3 ms | 28.4 tok/s | 699 ANE, 12 GPU, 5 CPU |

GPU was fastest in this run even though the CPU+ANE plan placed most operations on ANE. No Instruments trace was captured, so these are requested-policy and planned-placement results only.

## Storage finding

Repeated `MLComputePlan` inspection originally left host-local `.mlmodelc` directories in the system temporary directory. Four-policy tests accumulated several gigabytes. The worker now redirects compilation into a managed directory beside the model package and removes it after loading or inspection. Regression tests cover normal and failed cleanup.
