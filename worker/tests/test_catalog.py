from ane_studio_worker.catalog import CURATED_MODELS, SUPPORTED_ARCHITECTURES


def test_curated_models_are_recent_distinct_architectures() -> None:
    assert {model["id"] for model in CURATED_MODELS} == {
        "LiquidAI/LFM2.5-230M",
        "WeiboAI/VibeThinker-3B",
    }
    assert len({model["architecture"] for model in CURATED_MODELS}) == 2
    assert all(model["architecture"] in SUPPORTED_ARCHITECTURES for model in CURATED_MODELS)
