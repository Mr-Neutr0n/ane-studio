from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ane_studio_worker import huggingface as hub

REVISION = "a" * 40


def mock_hub(monkeypatch, tmp_path: Path, config=None, revision=REVISION):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(config if config is not None else {"architectures": ["Lfm2ForCausalLM"]})
    )
    calls = []
    info = SimpleNamespace(
        sha=revision,
        safetensors=SimpleNamespace(total=123),
        card_data=None,
        private=False,
        gated=False,
    )

    class Api:
        def __init__(self, **kwargs):
            calls.append(("api", kwargs))

        def model_info(self, model_id, **kwargs):
            calls.append(("info", {"model_id": model_id, **kwargs}))
            return info

    def download(model_id, filename, **kwargs):
        calls.append(("config", {"model_id": model_id, "filename": filename, **kwargs}))
        return str(path)

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    return calls, path


def test_inspection_pins_config_to_metadata_revision(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls, _ = mock_hub(monkeypatch, tmp_path)
    result = hub.inspect_model("org/model")
    assert result["supported"] is True
    assert result["revision"] == REVISION
    assert calls[0] == ("api", {"token": False})
    assert calls[2][1]["revision"] == REVISION
    assert calls[2][1]["token"] is False


def test_only_explicit_token_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", " explicit-test-credential ")
    calls, _ = mock_hub(monkeypatch, tmp_path)
    hub.inspect_model("org/model")
    assert calls[0][1]["token"] == "explicit-test-credential"
    assert calls[2][1]["token"] == "explicit-test-credential"


@pytest.mark.parametrize(
    "config",
    [
        [],
        1,
        "bad",
        {"architectures": "Lfm2ForCausalLM"},
        {"architectures": [{}]},
        {"architectures": [7]},
    ],
)
def test_inspection_rejects_untrusted_config_shapes(monkeypatch, tmp_path: Path, config) -> None:
    mock_hub(monkeypatch, tmp_path, config)
    with pytest.raises(ValueError, match="config"):
        hub.inspect_model("org/model")


def test_inspection_rejects_large_config_before_parse(monkeypatch, tmp_path: Path) -> None:
    _, path = mock_hub(monkeypatch, tmp_path)
    path.write_text(" " * 1_048_577)
    with pytest.raises(ValueError, match="too large"):
        hub.inspect_model("org/model")


@pytest.mark.parametrize("revision", [None, "main", "abc", "../main"])
def test_inspection_requires_exact_revision_before_config_fetch(
    monkeypatch, tmp_path: Path, revision
) -> None:
    calls, _ = mock_hub(monkeypatch, tmp_path, revision=revision)
    with pytest.raises(ValueError, match="revision"):
        hub.inspect_model("org/model")
    assert [name for name, _ in calls] == ["api", "info"]


def test_unknown_architecture_is_not_supported(monkeypatch, tmp_path: Path) -> None:
    mock_hub(monkeypatch, tmp_path, {"architectures": ["CustomModel"]})
    result = hub.inspect_model("org/model")
    assert result["supported"] is False
    assert "CustomModel" in result["reason"]


def test_known_failed_revision_is_blocked(monkeypatch, tmp_path: Path) -> None:
    mock_hub(monkeypatch, tmp_path)
    monkeypatch.setattr(hub, "KNOWN_FAILED_REVISIONS", {("org/model", REVISION): "parity failed"})
    result = hub.inspect_model("org/model")
    assert result["supported"] is False
    assert result["reason"] == "parity failed"


def test_download_is_pinned_and_does_not_fetch_executable_weights(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(hub, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    destination = tmp_path / "source"
    assert hub.download_model("org/model", destination, REVISION) == destination
    assert calls[0]["revision"] == REVISION
    assert calls[0]["token"] is False
    assert "*.safetensors" in calls[0]["allow_patterns"]
    assert "*.bin" not in calls[0]["allow_patterns"]
    assert "*.py" not in calls[0]["allow_patterns"]


def test_download_rejects_unpinned_revision_before_creating_files(tmp_path: Path) -> None:
    destination = tmp_path / "source"
    with pytest.raises(ValueError, match="revision"):
        hub.download_model("org/model", destination, "main")
    assert not destination.exists()


def test_download_rejects_existing_symlink_target(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "source"
    destination.mkdir()
    outside = tmp_path / "private"
    outside.write_text("keep")
    (destination / "config.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        hub.download_model("org/model", destination, REVISION)
    assert outside.read_text() == "keep"


def test_bad_identifier_is_rejected_before_network(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("invalid ID reached Hub")

    monkeypatch.setattr(hub, "HfApi", unexpected)
    with pytest.raises(ValueError):
        hub.inspect_model("org/../model")
