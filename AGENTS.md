# Project instructions

## Product boundary

ANE Studio is a local Electron app for converting, validating, benchmarking, and chatting with small Hugging Face causal language models on Apple Silicon. Keep model weights, prompts, credentials, converted packages, and results on the user's Mac.

## Architecture

- Keep untrusted renderer code behind Electron context isolation and the typed preload bridge.
- Keep Hugging Face and Core ML dependencies in `worker/`. The renderer must not invoke Python or the shell directly.
- Add model families through explicit adapters and parity fixtures. A shared architecture name is not enough evidence that a checkpoint works.
- Read `docs/architecture.md` before changing IPC, worker lifecycle, conversion, or local storage.
- Read `docs/model-support.md` before adding a model or changing parity requirements.

## Evidence

Report Core ML placement as requested, planned, or observed. Only an Instruments trace aligned with inference proves observed accelerator activity. Keep profiling separate from timing. A model must pass source-to-Core-ML parity before chat or benchmarks run.

## Verification

Run `bun run verify` after source changes. Run the networked model smoke commands in `docs/model-support.md` when changing an adapter. Generated model weights, `.mlpackage`, `.mlmodelc`, results, virtual environments, and build output stay out of Git.
