from __future__ import annotations

CURATED_MODELS = (
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5 0.5B Instruct",
        "description": "An Apache-licensed instruction model for short local conversations.",
        "parameters": 494_032_768,
        "architecture": "Qwen2ForCausalLM",
        "license": "Apache 2.0",
        "releasedAt": "2024-09-19",
        "recommended": True,
        "caution": "FP32 preserves parity. Use CPU or GPU. The Core ML package is about 2 GB.",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "label": "SmolLM2 135M Instruct",
        "description": "A small Apache-licensed checkpoint for quick conversion and simple chat.",
        "parameters": 134_515_008,
        "architecture": "LlamaForCausalLM",
        "license": "Apache 2.0",
        "releasedAt": "2024-11-01",
        "recommended": False,
        "caution": "Limited answer quality and 64-token graph. CPU or GPU starts faster than ANE.",
    },
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
        "caution": "Not validated. FP32 conversion can exceed 16 GB of unified memory.",
    },
)

SUPPORTED_ARCHITECTURES = {
    "Lfm2ForCausalLM": "lfm2",
    "Qwen2ForCausalLM": "qwen2",
    "LlamaForCausalLM": "llama",
}

# These exact revisions pass the v3 four-fixture gate against unpatched source logits.
# Keep future failures revision-specific. Architecture support is not a parity bypass.
KNOWN_FAILED_REVISIONS: dict[tuple[str, str], str] = {}
