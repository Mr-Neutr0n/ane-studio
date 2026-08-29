from __future__ import annotations

from pathlib import Path

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
