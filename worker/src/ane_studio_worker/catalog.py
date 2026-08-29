from __future__ import annotations

CURATED_MODELS = (
    {
        "id": "LiquidAI/LFM2.5-230M",
        "label": "LFM2.5 230M",
        "description": "Liquid AI's compact conversational model for edge devices.",
        "parameters": 229_693_184,
        "architecture": "Lfm2ForCausalLM",
        "license": "LFM Open License 1.0",
        "releasedAt": "2026-06-24",
        "recommended": True,
    },
    {
        "id": "LiquidAI/LFM2.5-350M",
        "label": "LFM2.5 350M",
        "description": "A larger LFM2 edge checkpoint for testing architecture scaling.",
        "parameters": 354_483_968,
        "architecture": "Lfm2ForCausalLM",
        "license": "LFM Open License 1.0",
        "releasedAt": "2026-03-31",
        "recommended": False,
    },
    {
        "id": "WeiboAI/VibeThinker-3B",
        "label": "VibeThinker 3B",
        "description": "A compact reasoning and coding model based on Qwen2.",
        "parameters": 3_085_938_688,
        "architecture": "Qwen2ForCausalLM",
        "license": "MIT",
        "releasedAt": "2026-06-12",
        "recommended": False,
        "caution": "Conversion needs roughly 12 GB of free memory and may use swap on a 16 GB Mac.",
    },
)

SUPPORTED_ARCHITECTURES = {
    "Lfm2ForCausalLM": "lfm2",
    "Qwen2ForCausalLM": "qwen2",
}

KNOWN_FAILED_REVISIONS = {
    (
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    ): "The current Llama conversion produces invalid zero logits in Core ML.",
    (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
    ): "The current FP16 conversion fails logit parity and top-token agreement.",
}
