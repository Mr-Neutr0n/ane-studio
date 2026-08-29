# Contributing

ANE Studio accepts focused fixes, model adapters, measurement improvements, and documentation corrections.

## Setup

```bash
bun install
bun run worker:setup
bun run verify
```

Use an Apple Silicon Mac for Core ML conversion and runtime work. Unit tests can cover worker protocol and metadata behavior without downloading a model, but adapter changes need the networked smoke commands in `docs/model-support.md`.

## Pull requests

- Keep renderer access behind the typed preload bridge.
- Add parity evidence for model conversion changes.
- Label requested, planned, and observed placement correctly.
- Keep model weights, generated Core ML packages, results, virtual environments, and build output out of Git.
- Update model and third-party license notes when adding a dependency or curated checkpoint.

By submitting a contribution, you agree to license it under this repository's MIT license.
