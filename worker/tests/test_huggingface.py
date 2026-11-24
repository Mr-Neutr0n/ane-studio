from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ane_studio_worker import huggingface as hub

REVISION = "a" * 40


def mock_hub(monkeypatch, tmp_path: Path, config=None, revision=REVISION, model_id="org/model"):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(config if config is not None else {"architectures": ["Lfm2ForCausalLM"]})
    )
    calls = []
    info = SimpleNamespace(
        id=model_id,
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
    staged = hub.download_model("org/model", destination, REVISION)
    assert staged == calls[0]["local_dir"] == tmp_path / "source.partial"
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


def test_inspection_returns_canonical_hub_id(monkeypatch, tmp_path: Path) -> None:
    calls, _ = mock_hub(monkeypatch, tmp_path, model_id="Org/Model")
    assert hub.inspect_model("org/model")["id"] == "Org/Model"
    assert calls[2][1]["model_id"] == "Org/Model"


def test_inspection_keeps_requested_id_when_hub_id_is_invalid(monkeypatch, tmp_path: Path) -> None:
    mock_hub(monkeypatch, tmp_path, model_id="not a model id")
    assert hub.inspect_model("org/model")["id"] == "org/model"


def fake_snapshot(monkeypatch, files):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        for name in files:
            (kwargs["local_dir"] / name).write_text(kwargs["revision"])

    monkeypatch.setattr(hub, "snapshot_download", download)
    return calls


def test_same_revision_downloads_in_place(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "source"
    fake_snapshot(monkeypatch, ["config.json"])
    hub.replace_source(hub.download_model("org/model", destination, REVISION), destination)
    calls = fake_snapshot(monkeypatch, ["config.json"])
    assert hub.download_model("org/model", destination, REVISION) == destination
    assert calls[0]["local_dir"] == destination


def test_new_revision_replaces_stale_source_files(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "source"
    fake_snapshot(monkeypatch, ["config.json", "chat_template.jinja"])
    hub.replace_source(hub.download_model("org/model", destination, REVISION), destination)
    newer = "b" * 40
    fake_snapshot(monkeypatch, ["config.json"])
    staged = hub.download_model("org/model", destination, newer)
    # The old source stays intact until the new revision is complete.
    assert (destination / "chat_template.jinja").read_text() == REVISION
    hub.replace_source(staged, destination)
    assert sorted(path.name for path in destination.iterdir()) == [
        hub.REVISION_MARKER,
        "config.json",
    ]
    assert (destination / "config.json").read_text() == newer
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source"]


def test_interrupted_staging_resumes_only_for_its_revision(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "source"
    staging = tmp_path / "source.partial"
    staging.mkdir()
    (staging / hub.REVISION_MARKER).write_text(REVISION)
    (staging / "model.safetensors").write_text("partial")
    fake_snapshot(monkeypatch, [])
    hub.download_model("org/model", destination, REVISION)
    assert (staging / "model.safetensors").exists()
    hub.download_model("org/model", destination, "b" * 40)
    assert not (staging / "model.safetensors").exists()


def test_replace_source_clears_a_leftover_previous_source(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "source"
    fake_snapshot(monkeypatch, ["config.json"])
    hub.replace_source(hub.download_model("org/model", destination, REVISION), destination)
    # A stop during the last removal leaves source.old behind, sometimes as a file.
    (tmp_path / "source.old").write_text("stale")
    fake_snapshot(monkeypatch, ["config.json"])
    hub.replace_source(hub.download_model("org/model", destination, "b" * 40), destination)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["source"]


def test_previous_source_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (tmp_path / "source.old").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        hub.remove_previous_source(tmp_path / "source")
    assert (outside / "keep.txt").exists()


def test_symlinked_destination_is_rejected_before_it_is_read(monkeypatch, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "source").symlink_to(outside)
    calls = fake_snapshot(monkeypatch, ["config.json"])
    with pytest.raises(ValueError, match="symbolic links"):
        hub.download_model("org/model", tmp_path / "source", REVISION)
    assert calls == []
    assert list(outside.iterdir()) == []


def test_unreadable_marker_and_file_staging_path_restage_cleanly(
    monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "source"
    destination.mkdir()
    (destination / hub.REVISION_MARKER).write_bytes(b"\xff\xfe")
    (tmp_path / "source.partial").write_text("not a directory")
    fake_snapshot(monkeypatch, ["config.json"])
    staged = hub.download_model("org/model", destination, REVISION)
    assert staged == tmp_path / "source.partial"
    assert (staged / hub.REVISION_MARKER).read_text().strip() == REVISION
