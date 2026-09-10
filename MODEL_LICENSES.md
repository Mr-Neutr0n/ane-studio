# Model licenses

ANE Studio code is MIT licensed. Downloaded weights and converted Core ML packages keep their source model licenses. The app does not bundle either.

## Apache 2.0 models

The two compact instruction checkpoints use the OSI-approved Apache License 2.0. Neither repository required gated access in the 2026-09-08 checks.

| Checkpoint | Tested revision | License source |
| --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | [Pinned LICENSE](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE) |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | [Pinned model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) |

The app includes [Apache 2.0 license text](licenses/APACHE-2.0.txt). SmolLM2's pinned repository declares `apache-2.0` in its model card rather than a separate `LICENSE` file. Preserve upstream attribution, license text, modification notices, and any applicable `NOTICE` when redistributing weights or converted derivatives.

These license checks do not prove training-data rights or model quality. Conversion and runtime evidence is in [compatibility results](docs/compatibility-results.md).

## LiquidAI LFM2.5 checkpoints

- Sources: <https://huggingface.co/LiquidAI/LFM2.5-230M> and <https://huggingface.co/LiquidAI/LFM2.5-350M>
- License: LFM Open License v1.0
- Full text: [LFM Open License 1.0](licenses/LFM-OPEN-LICENSE-1.0.txt)

The LFM license includes attribution and commercial-use conditions, including a revenue threshold. It is not the same license as ANE Studio's MIT code or the Apache-licensed models above. Review the full terms before using or redistributing an LFM model or converted derivative.

## Converter fixture

`yujiepan/qwen2-tiny-random` reports no license identifier. It is a local converter regression fixture, not a useful chat model or a bundled OSS model. No weights are redistributed by this project.

## WeiboAI/VibeThinker-3B

- Source: <https://huggingface.co/WeiboAI/VibeThinker-3B>
- Model card license identifier: MIT
- Base model: `Qwen/Qwen2.5-Coder-3B`
- Apache 2.0 text used by the Qwen base family: [Apache 2.0](licenses/APACHE-2.0.txt)

The full checkpoint has not passed local conversion. Verify its current model card, base-model terms, and access conditions before use. ANE Studio records the Hugging Face license identifier in each local manifest.

## User-added models

ANE Studio displays the Hugging Face license identifier when available. It does not decide whether the terms permit a particular use or redistribution. Do not publish converted packages without preserving the source model's required notices and terms.
