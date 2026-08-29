from __future__ import annotations

from pathlib import Path

import pytest

from ane_studio_worker import service
from ane_studio_worker.service import Service, bounded_float, bounded_int, required_model_id


def test_validates_model_identifier() -> None:
    assert required_model_id({"modelId": "LiquidAI/LFM2.5-230M"}) == "LiquidAI/LFM2.5-230M"
    with pytest.raises(ValueError):
        required_model_id({"modelId": "local-model"})


def test_bounds_generation_values() -> None:
    assert bounded_int("64", 16, 256, "sequence") == 64
    assert bounded_float("0.2", 0, 2, "temperature") == pytest.approx(0.2)
    with pytest.raises(ValueError):
        bounded_int(512, 16, 256, "sequence")
    with pytest.raises(ValueError):
        bounded_float(-1, 0, 2, "temperature")


def test_unknown_converter_fails_before_download(monkeypatch, tmp_path: Path) -> None:
    events = []
    worker = Service(events.append)
    monkeypatch.setattr(
        service,
        "inspect_model",
        lambda model_id: {
            "id": model_id,
            "supported": False,
            "reason": "No converter is registered for MysteryLM.",
            "architecture": "MysteryLM",
            "parameterCount": 1,
            "license": "mit",
            "revision": "abc",
        },
    )
    with pytest.raises(ValueError, match="MysteryLM"):
        worker.prepare("request", {"modelId": "org/model"})
    assert not events
