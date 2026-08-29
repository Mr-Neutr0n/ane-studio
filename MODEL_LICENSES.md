# Model licenses

ANE Studio code is MIT licensed. Models downloaded or converted by the user remain governed by their own licenses. The application does not bundle model weights or converted Core ML packages.

## LiquidAI LFM2.5 checkpoints

- Sources: <https://huggingface.co/LiquidAI/LFM2.5-230M> and <https://huggingface.co/LiquidAI/LFM2.5-350M>
- License: LFM Open License v1.0
- Full text: [`licenses/LFM-OPEN-LICENSE-1.0.txt`](licenses/LFM-OPEN-LICENSE-1.0.txt)

The LFM license includes attribution and commercial-use conditions, including a revenue threshold. Users must review the full terms before using or redistributing the model or a converted derivative.

## Apache-licensed compatibility checkpoints

ANE Studio's compatibility tests also reference:

- <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>
- <https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct>
- <https://huggingface.co/yujiepan/qwen2-tiny-random>

The first two model cards identify Apache 2.0. The tiny random Qwen2 fixture reports no license identifier and is used only for local converter testing. See [`licenses/APACHE-2.0.txt`](licenses/APACHE-2.0.txt).

## WeiboAI/VibeThinker-3B

- Source: <https://huggingface.co/WeiboAI/VibeThinker-3B>
- Model card license identifier: MIT
- Base model: `Qwen/Qwen2.5-Coder-3B`
- Apache 2.0 text used by the Qwen base family: [`licenses/APACHE-2.0.txt`](licenses/APACHE-2.0.txt)

Users must verify the current model card, base-model terms, and any access conditions at download time. ANE Studio records the license identifier returned by Hugging Face in the local manifest.

## User-added models

ANE Studio displays the Hugging Face license identifier when available. It does not decide whether a model's terms permit a particular use or redistribution. Converted Core ML packages should not be published without preserving the source model's required notices and terms.
