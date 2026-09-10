from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import suppress
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


def model_path(model_id: str, *parts: str) -> Path:
    root = data_root()
    path = root / "models" / model_slug(model_id)
    for part in parts:
        if Path(part).is_absolute() or ".." in Path(part).parts:
            raise ValueError("Model paths must stay inside the local model directory.")
        path /= part
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symbolic links are not allowed in managed model paths.")
    return path


def model_directory(model_id: str) -> Path:
    return model_path(model_id)


def manifest_path(model_id: str) -> Path:
    return model_path(model_id, "manifest.json")


def normalize_manifest(value: dict[str, Any]) -> dict[str, Any]:
    # Optional fields cross a typed JSON bridge as absent, not null.
    record = {key: item for key, item in value.items() if item is not None}
    if isinstance(record.get("parity"), dict):
        record["parity"] = {key: item for key, item in record["parity"].items() if item is not None}
    return record


def load_manifest(model_id: str) -> dict[str, Any] | None:
    path = manifest_path(model_id)
    if not path.exists():
        return None
    if path.stat().st_size > 1_048_576:
        raise ValueError("The model manifest is too large.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("id") != model_id:
        raise ValueError("The model manifest has an invalid model identifier.")
    return normalize_manifest(value)


def write_manifest(model_id: str, value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or value.get("id") != model_id:
        raise ValueError("The model manifest has an invalid model identifier.")
    content = (
        json.dumps(normalize_manifest(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    path = manifest_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def list_manifests() -> list[dict[str, Any]]:
    root = data_root() / "models"
    if not root.exists() or root.is_symlink():
        return []
    values = []
    for path in sorted(root.glob("*/manifest.json")):
        try:
            if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 1_048_576:
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("id"), str):
                continue
            if path.parent.name != model_slug(value["id"]):
                continue
            record = load_manifest(value["id"])
            if record is not None:
                values.append(record)
        except (OSError, ValueError, RecursionError):
            continue
    return values


def remove_model(model_id: str) -> None:
    with suppress(FileNotFoundError):
        shutil.rmtree(model_directory(model_id))


def directory_signature(root: Path) -> tuple[tuple[Any, ...], ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("The artifact must be a local directory, not a symbolic link.")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolic links are not allowed in model artifacts.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Model artifacts must contain only regular files.")
        status = path.stat()
        entries.append(
            (
                path.relative_to(root).as_posix(),
                status.st_dev,
                status.st_ino,
                status.st_size,
                status.st_mtime_ns,
                status.st_ctime_ns,
            )
        )
    return tuple(entries)


def directory_digest(root: Path) -> tuple[str, int]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("The artifact must be a local directory, not a symbolic link.")
    hasher = hashlib.sha256()
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolic links are not allowed in model artifacts.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Model artifacts must contain only regular files.")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content_hasher = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1_048_576):
                size += len(chunk)
                content_hasher.update(chunk)
        total += size
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(size.to_bytes(8, "big"))
        hasher.update(content_hasher.digest())
    return hasher.hexdigest(), total
