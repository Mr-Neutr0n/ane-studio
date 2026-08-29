# Security

## Reporting

Report a vulnerability privately through GitHub's security advisory feature for this repository. Do not include live Hugging Face tokens, private model weights, or user prompts in an issue.

## Security model

The Electron renderer runs with sandboxing, context isolation, and Node.js integration disabled. It can call only the methods exposed by `src/preload/index.ts`. The main process validates worker command names before writing JSONL to the local Python worker.

Hugging Face tokens use Electron `safeStorage`, backed by macOS Keychain. The worker receives a token only through its environment. ANE Studio does not execute Hugging Face remote custom code.

Generated models remain local and are not trusted as application code. Core ML and Transformers still parse model-controlled files, so users should download checkpoints from publishers they trust.
