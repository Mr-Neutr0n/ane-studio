# Contributing

ANE Studio accepts focused fixes, model adapters, measurement improvements, and documentation corrections. Keep app ID `dev.harikrishna.anestudio`, the `ane-studio-worker` command, and existing `ANE Studio` data paths compatible unless a change includes a tested migration.

## Setup and checks

Use an Apple Silicon Mac with macOS 15 or newer. Install Bun and uv, then run:

```bash
bun install --frozen-lockfile
bun run worker:setup
bun install --cwd website --frozen-lockfile
cd website
bunx playwright install chromium webkit
cd ..
bun run verify
```

The website has its own dependency lock and verification command, `(cd website && bun run verify)`. Root verification runs it too. Browser downloads and dependency installs need network access. Unit tests do not download model weights.

## Changes to models

- Read `docs/model-support.md` and `docs/architecture.md` first.
- Add an explicit adapter and tiny parity fixture for a new architecture.
- Require cosine similarity at least `0.999` and matching top tokens. Never lower this gate to admit a model.
- Run the networked smoke commands for each advertised checkpoint revision.
- Record conversion, chat, and all four compute policies on Apple Silicon. A tiny random fixture is not a useful chat model.
- Label requested, planned, and observed placement correctly. Timing does not prove observed ANE use.
- Record model licenses and memory or storage limits.

## Pull requests

Keep renderer access behind the typed preload bridge. Do not add shell or Python access to the renderer. Add tests for request validation, cancellation, startup, or storage changes.

Keep weights, converted packages, prompts, credentials, result directories, virtual environments, and build output out of Git. Use isolated `ANE_STUDIO_HOME` directories for tests. Do not attach private prompts or model weights to issues.

Dependency versions are exact in `package.json`, `bun.lock`, and `worker/uv.lock`. Update the relevant license notices with dependency changes. Do not hand-edit generated runtime inventories.

## Release checks

`bun run release:local` builds an unsigned, unnotarized Apple Silicon app with a bundled runtime. It requires Apple's command-line tools and the pinned release tooling in the README. Generated inventories live in `artifacts/runtime-notices/` and ship under the app's `Contents/Resources/licenses/runtime/` directory.

A matching version tag creates a draft through GitHub Actions. Before publication, run the packaged clean-data test and networked model smoke tests against the final worker. Review the checksum, runtime notices, known limits, and website release URL. Never describe an ad-hoc native signature as Apple Developer ID signing or notarization.

By submitting a contribution, you agree to license it under this repository's MIT license.
