from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


def data_root() -> Path:
    explicit = os.environ.get("ANE_STUDIO_HOME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path.home() / "Library/Application Support/ANE Studio").resolve()


def model_slug(model_id: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")[:52]
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def model_directory(model_id: str) -> Path:
    return data_root() / "models" / model_slug(model_id)


def manifest_path(model_id: str) -> Path:
    return model_directory(model_id) / "manifest.json"


def load_manifest(model_id: str) -> dict[str, Any] | None:
    path = manifest_path(model_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(model_id: str, value: dict[str, Any]) -> None:
    path = manifest_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def list_manifests() -> list[dict[str, Any]]:
    root = data_root() / "models"
    if not root.exists():
        return []
    values = []
    for path in sorted(root.glob("*/manifest.json")):
        try:
            values.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return values


def remove_model(model_id: str) -> None:
    shutil.rmtree(model_directory(model_id), ignore_errors=True)


def directory_digest(root: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        total += len(content)
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(hashlib.sha256(content).digest())
    return hasher.hexdigest(), total
