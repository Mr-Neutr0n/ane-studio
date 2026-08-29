# Roadmap

## Working alpha

- [x] Electron model library, chat, tests, details, and technical logs
- [x] Local Python worker with typed JSONL protocol
- [x] Hugging Face metadata, pinned downloads, and credential handling
- [x] LFM2 and Qwen2 FP16 Core ML adapters
- [x] Source-to-Core-ML parity gate
- [x] Streaming Core ML chat
- [x] Four-policy interactive benchmark
- [x] Planned placement through `MLComputePlan`
- [x] M5 end-to-end LFM2.5-230M and LFM2.5-350M validation
- [x] Revision-specific rejection for failed SmolLM2-135M and Qwen2.5-0.5B conversions

## Next

- [ ] Stateful prefill and KV-cache decode
- [ ] Fresh-process benchmark rounds with randomized policy order
- [ ] Instruments launch and trace association from the app
- [ ] Full VibeThinker-3B conversion on a machine with enough free disk and memory
- [ ] Additional architecture adapters with tiny and useful checkpoint fixtures
- [ ] Signed and notarized release workflow after there is demand
