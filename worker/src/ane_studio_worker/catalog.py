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
