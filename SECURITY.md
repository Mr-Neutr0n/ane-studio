# Security

## Report a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/Mr-Neutr0n/ane-studio/security/advisories/new). Do not put live tokens, private model weights, or user prompts in public issues. Include the app version, macOS version, steps to reproduce, and a minimal example with no credentials.

ANE Studio is an early preview. Security fixes target the current release. There is no automatic update service; check the repository for updates.

## Process boundaries

The Electron renderer uses sandboxing and context isolation with Node.js integration disabled. It can call only the typed methods in `src/preload/index.ts`. The main process validates the caller, command, and payload before it sends JSONL to the Python worker. External navigation is restricted.

The main process starts one local worker. Cancellation stops its process group before replacement. Requests have time limits, pending-request limits, protocol validation, and log redaction. The packaged app launches isolated bundled Python directly; it does not search PATH for Python or uv.

Model files remain untrusted input. Transformers and Core ML still parse model-controlled files. Download checkpoints from publishers you trust. The worker disables remote custom code. Do not bypass parity failures or install an unreviewed adapter to admit a model.

## Credentials and local data

Hugging Face tokens use Electron `safeStorage`, backed by macOS Keychain. The app does not fall back to plaintext storage if encryption is unavailable. The main process passes the selected token through the worker environment, not through renderer events or command-line arguments.

The worker has an app-scoped Hugging Face cache. It disables implicit cached login and Hugging Face telemetry. Deleting a token does not silently reuse another tool's Hugging Face login. Logs redact token-shaped secrets, but do not treat a log as safe to publish without review.

Models, converted packages, and benchmark reports remain under the local app data directory. Reports can contain prompts. `ANE_STUDIO_HOME` is a developer/test override, not a remote storage setting. The `ANE Studio` directory and app ID stay stable across updates.

## Release trust

Current release automation produces an unsigned, unnotarized Apple Silicon preview. Native runtime relocation uses local ad-hoc signatures where macOS requires them. Those signatures do not prove publisher identity. There is no Apple Developer ID signature or notarization ticket.

Download only from this repository's release page. Verify its ZIP against the `SHA256SUMS` file from the same release. A checksum detects corruption or mismatched files; it does not replace a trusted publisher signature. If you cannot accept that limit, review and run the source instead. Do not disable Gatekeeper globally.

GitHub Actions use pinned action revisions and locked dependencies. The release workflow creates a draft for maintainer review. Runtime and desktop license notices ship inside the app. Model weights and credentials do not ship in release artifacts.
