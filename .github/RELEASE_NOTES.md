ANE Studio for Apple Silicon, macOS 15 or newer.

This ZIP includes Python 3.12 and the locked Core ML worker dependencies. It does not need Bun, uv, Homebrew, or a system Python. No model weights or credentials are included. The first model download needs an internet connection.

This is an unsigned, unnotarized preview. It is not signed with an Apple Developer ID. macOS may block its first launch. Verify the ZIP against `SHA256SUMS`, extract it, move ANE Studio.app to Applications, then use System Settings > Privacy & Security > Open Anyway if you trust this release. Do not disable Gatekeeper globally. See the README for the source-install alternative.

Model execution requires logit cosine similarity at least 0.999 and matching top tokens. Requested and planned accelerator placement are not proof of observed ANE activity.

Before publishing this draft, the maintainer must confirm:

- Source and packaged startup checks passed on Apple Silicon.
- Networked conversion, chat, and policy evidence matches the shipped worker.
- The artifact checksum and runtime license inventory are attached and correct.
- Public model claims match docs/model-support.md.
- The website links to this exact release asset only after publication.
