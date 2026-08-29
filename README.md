# ANE Studio

A local macOS app for bringing small Hugging Face language models to Core ML, checking whether conversion preserved their behavior, and measuring which Apple Silicon compute policy is fastest.

![ANE Studio policy comparison](docs/assets/benchmark.png)

ANE Studio does not call a selected CPU+ANE policy "ANE-only." Core ML can partition a graph or fall back. The app keeps requested policy, planned placement, and observed activity as separate claims.

## What works

- Native desktop experience through Electron on Apple Silicon
- Hugging Face model inspection and pinned downloads
- FP16 Core ML conversion for LFM2 and Qwen2 causal language models
- Source-versus-Core-ML logit parity before execution
- Local streaming chat through Core ML
- CPU, CPU+GPU, CPU+ANE, and automatic policy comparisons
- Raw token latency samples and `MLComputePlan` operation counts
- Encrypted Hugging Face credentials through Electron `safeStorage` and macOS Keychain
- No network traffic after model download

The first curated checkpoints are:

| Model | Architecture | Status |
| --- | --- | --- |
| [`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M) | LFM2 hybrid convolution and attention | Tested end to end on an M5 Mac |
| [`WeiboAI/VibeThinker-3B`](https://huggingface.co/WeiboAI/VibeThinker-3B) | Qwen2 | Adapter tested with a tiny Qwen2 fixture; full 3B conversion needs more free disk and memory |

Other checkpoints using those exact Transformers architectures can be inspected from the model ID field. Unsupported architectures fail before downloading weights.

## Measured M5 smoke result

`LiquidAI/LFM2.5-230M` was downloaded at revision `40cb2ad3b3044d5a41eee083a6103c8b523afa45`, converted at a fixed 64-token shape, and run on an Apple M5 MacBook Air with 16 GB unified memory.

- Core ML artifact: 459.8 MB
- Logit cosine similarity: `0.999936`
- Top-token agreement: passed
- Chat output for "Say hello in one short sentence": `Hello! How can I assist you today?`
- Planned CPU+ANE placement: 641 of 658 non-constant operations on ANE
- Observed placement: not captured because full Xcode and an Instruments trace were unavailable

One interactive policy run produced:

| Policy | Median token | Throughput | Planned placement |
| --- | ---: | ---: | --- |
| CPU only | 20.6 ms | 48.6 tok/s | 658 CPU operations |
| CPU + GPU | 8.7 ms | 115.2 tok/s | 658 GPU operations |
| CPU + ANE | 12.7 ms | 78.9 tok/s | 641 ANE, 17 CPU operations |
| Automatic | 17.4 ms | 57.5 tok/s | 79 ANE, 574 GPU, 5 CPU operations |

These are development smoke numbers, not publication-grade results. The current runtime recomputes a fixed context without a KV cache, and the policy test runs sequentially in one worker process. The result is useful for local backend selection, not a universal hardware ranking.

## Requirements

- Apple Silicon Mac
- macOS 15 or newer
- [Bun](https://bun.sh/)
- [uv](https://docs.astral.sh/uv/)
- About 2 GB free for development dependencies
- Additional disk space for source weights and FP16 Core ML packages

Full Xcode is optional for app development. It is required for Instruments profiling and observed-placement evidence.

## Run from source

```bash
git clone https://github.com/Mr-Neutr0n/ane-studio.git
cd ane-studio
bun install
bun run worker:setup
bun run dev
```

Select `LFM2.5 230M`, click **Convert to Core ML**, and wait for download, conversion, and parity. Public models do not require a Hugging Face token.

Local data lives under:

```text
~/Library/Application Support/ANE Studio/
```

Removing a model in the app deletes its downloaded snapshot, Core ML package, manifest, and benchmark reports.

## Build the unsigned app

```bash
bun run package
open "dist/mac-arm64/ANE Studio.app"
```

Local builds are free. Public notarization requires Apple's paid developer program. This repository does not require it.

## Verify

```bash
bun run verify
```

That command runs TypeScript checks, renderer tests, Python worker tests, the production build, and the Electron startup test. Networked model conversion smoke commands are documented in [`docs/model-support.md`](docs/model-support.md).

## Architecture and privacy

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/model-support.md`](docs/model-support.md)
- [`docs/privacy.md`](docs/privacy.md)
- [`MODEL_LICENSES.md`](MODEL_LICENSES.md)

The Electron renderer has no Node.js access. It talks through a narrow preload API to the main process. The main process supervises a local Python worker that owns Hugging Face, PyTorch, Transformers, and Core ML Tools.

## Limitations

- LFM2 and Qwen2 are the only registered architectures.
- Conversion uses FP16 and a fixed 64-token context by default.
- Generation currently recomputes the fixed context each token instead of using a stateful KV cache.
- The app reports planned placement automatically. Observed placement still requires Instruments.
- Unsigned local builds trigger normal macOS Gatekeeper behavior when copied to another machine.
- Model licenses remain separate from this repository's MIT code license.

## License

ANE Studio source code is MIT licensed. Downloaded models retain their original licenses. See [`LICENSE`](LICENSE), [`MODEL_LICENSES.md`](MODEL_LICENSES.md), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
