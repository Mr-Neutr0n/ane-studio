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
    }
    assert {model["architecture"] for model in CURATED_MODELS} == set(SUPPORTED_ARCHITECTURES)


def test_failed_checkpoints_are_revision_specific() -> None:
    assert all(
        "/" in model_id and len(revision) == 40 for model_id, revision in KNOWN_FAILED_REVISIONS
    )
    assert all(reason.endswith(".") for reason in KNOWN_FAILED_REVISIONS.values())
