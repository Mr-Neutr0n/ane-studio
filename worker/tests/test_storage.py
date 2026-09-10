from __future__ import annotations

import json
from pathlib import Path

import pytest

from ane_studio_worker import storage
from ane_studio_worker.storage import directory_digest, model_slug


def test_model_slug_is_readable_and_collision_resistant() -> None:
    first = model_slug("LiquidAI/LFM2.5-230M")
    second = model_slug("LiquidAI/LFM2.5-230M-Base")
    assert first.startswith("liquidai-lfm2-5-230m-")
    assert first != second
    assert len(first) <= 63


def test_directory_digest_ignores_directory_name(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "weights").mkdir(parents=True)
        (root / "Manifest.json").write_text("model", encoding="utf-8")
        (root / "weights/data.bin").write_bytes(b"weights")
    assert directory_digest(left) == directory_digest(right)


def test_directory_digest_changes_with_content(tmp_path: Path) -> None:
    root = tmp_path / "model.mlpackage"
    root.mkdir()
    path = root / "Manifest.json"
    path.write_text("one", encoding="utf-8")
    before = directory_digest(root)
    path.write_text("two", encoding="utf-8")
    assert directory_digest(root) != before


@pytest.fixture(autouse=True)
def local_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANE_STUDIO_HOME", str(tmp_path / "studio"))


def test_manifest_round_trip_and_private_atomic_file() -> None:
    record = {"id": "org/model", "state": "ready"}
    storage.write_manifest("org/model", record)
    assert storage.load_manifest("org/model") == record
    assert storage.list_manifests() == [record]
    assert storage.manifest_path("org/model").stat().st_mode & 0o777 == 0o600
    assert not list(storage.model_directory("org/model").glob("*.tmp"))


def test_failed_manifest_serialization_keeps_old_record() -> None:
    record = {"id": "org/model", "state": "ready"}
    storage.write_manifest("org/model", record)
    with pytest.raises(ValueError):
        storage.write_manifest("org/model", {**record, "score": float("nan")})
    assert storage.load_manifest("org/model") == record


def test_fixed_temporary_symlink_cannot_overwrite_external_file(tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.write_text("keep")
    directory = storage.model_directory("org/model")
    directory.mkdir(parents=True)
    (directory / "manifest.tmp").symlink_to(target)
    storage.write_manifest("org/model", {"id": "org/model"})
    assert target.read_text() == "keep"


@pytest.mark.parametrize("level", ["models", "model", "manifest"])
def test_managed_storage_rejects_symlink_escape(tmp_path: Path, level) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "manifest.json"
    external.write_text('{"id":"org/model"}')
    directory = storage.model_directory("org/model")
    if level == "models":
        directory.parent.parent.mkdir(parents=True)
        directory.parent.symlink_to(outside, target_is_directory=True)
    elif level == "model":
        directory.parent.mkdir(parents=True)
        directory.symlink_to(outside, target_is_directory=True)
    else:
        directory.mkdir(parents=True)
        (directory / "manifest.json").symlink_to(external)
    with pytest.raises(ValueError, match="Symbolic"):
        storage.load_manifest("org/model")
    with pytest.raises(ValueError, match="Symbolic"):
        storage.write_manifest("org/model", {"id": "org/model"})
    assert storage.list_manifests() == []
    assert external.read_text() == '{"id":"org/model"}'


def test_remove_rejects_symlinked_parent_without_deleting_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep")
    directory = storage.model_directory("org/model")
    directory.parent.mkdir(parents=True)
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        storage.remove_model("org/model")
    assert sentinel.read_text() == "keep"


def test_remove_is_idempotent_but_reports_permission_errors(monkeypatch) -> None:
    storage.remove_model("org/missing")

    def fail(path):
        raise PermissionError("denied")

    monkeypatch.setattr(storage.shutil, "rmtree", fail)
    with pytest.raises(PermissionError):
        storage.remove_model("org/model")


@pytest.mark.parametrize("value", [[], 1, "invalid", {"id": "org/other"}])
def test_bad_manifest_records_do_not_break_catalog(value) -> None:
    path = storage.manifest_path("org/model")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    assert storage.list_manifests() == []
    with pytest.raises(ValueError):
        storage.load_manifest("org/model")


def test_corrupt_manifest_is_skipped() -> None:
    path = storage.manifest_path("org/model")
    path.parent.mkdir(parents=True)
    path.write_text("{not-json")
    assert storage.list_manifests() == []


def test_digest_streams_files_and_preserves_existing_digest(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    package = tmp_path / "package"
    package.mkdir()
    content = b"x" * (2 * 1_048_576 + 4)
    (package / "weights.bin").write_bytes(content)
    relative = b"weights.bin"
    expected = hashlib.sha256(
        len(relative).to_bytes(8, "big")
        + relative
        + len(content).to_bytes(8, "big")
        + hashlib.sha256(content).digest()
    ).hexdigest()

    def fail(*args):
        pytest.fail("digest must not read an entire weight shard at once")

    monkeypatch.setattr(Path, "read_bytes", fail)
    assert storage.directory_digest(package) == (expected, len(content))


@pytest.mark.parametrize("helper", [storage.directory_digest, storage.directory_signature])
def test_artifact_helpers_reject_symlink_files(tmp_path: Path, helper) -> None:
    outside = tmp_path / "outside"
    outside.write_text("private")
    package = tmp_path / "package"
    package.mkdir()
    (package / "weights.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="Symbolic"):
        helper(package)


def test_directory_signature_detects_same_size_content_change(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    path = package / "weights.bin"
    path.write_bytes(b"aaa")
    before = storage.directory_signature(package)
    path.write_bytes(b"bbb")
    assert storage.directory_signature(package) != before


def test_explicit_model_path_cannot_escape() -> None:
    with pytest.raises(ValueError, match="inside"):
        storage.model_path("org/model", "../outside")
    with pytest.raises(ValueError, match="inside"):
        storage.model_path("org/model", "/outside")


def test_legacy_null_optional_fields_are_omitted_without_rewriting_disk() -> None:
    value = {
        "id": "org/model",
        "state": "failed",
        "parameterCount": None,
        "architecture": None,
        "parity": {"status": "failed", "cosineSimilarity": None},
    }
    path = storage.manifest_path("org/model")
    path.parent.mkdir(parents=True)
    text = json.dumps(value)
    path.write_text(text)
    record = storage.load_manifest("org/model")
    assert "parameterCount" not in record
    assert "architecture" not in record
    assert record["parity"] == {"status": "failed"}
    assert storage.list_manifests() == [record]
    assert path.read_text() == text


def test_new_manifests_omit_null_optional_fields() -> None:
    storage.write_manifest("org/model", {"id": "org/model", "parameterCount": None})
    assert json.loads(storage.manifest_path("org/model").read_text()) == {"id": "org/model"}
