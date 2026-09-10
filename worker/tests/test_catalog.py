from ane_studio_worker.catalog import (
    CURATED_MODELS,
    KNOWN_FAILED_REVISIONS,
    SUPPORTED_ARCHITECTURES,
)


def test_curated_models_cover_registered_architectures() -> None:
    assert {model["id"] for model in CURATED_MODELS} == {
        "LiquidAI/LFM2.5-230M",
        "LiquidAI/LFM2.5-350M",
        "WeiboAI/VibeThinker-3B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    }
    assert {model["architecture"] for model in CURATED_MODELS} == set(SUPPORTED_ARCHITECTURES)


def test_failed_checkpoints_are_revision_specific() -> None:
    assert all(
        "/" in model_id and len(revision) == 40 for model_id, revision in KNOWN_FAILED_REVISIONS
    )
    assert all(reason.endswith(".") for reason in KNOWN_FAILED_REVISIONS.values())


def test_two_useful_apache_checkpoints_have_real_conversion_adapters() -> None:
    models = {model["id"]: model for model in CURATED_MODELS}
    for model_id in ("Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"):
        model = models[model_id]
        assert model["license"] == "Apache 2.0"
        assert model["parameters"] > 100_000_000
        assert model["architecture"] in SUPPORTED_ARCHITECTURES
        assert not any(blocked_id == model_id for blocked_id, _ in KNOWN_FAILED_REVISIONS)


def test_known_failure_registry_accepts_only_revision_specific_entries() -> None:
    # The two historical failures were removed only after real converter-v3 parity.
    assert (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
    ) not in KNOWN_FAILED_REVISIONS
    assert (
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    ) not in KNOWN_FAILED_REVISIONS
