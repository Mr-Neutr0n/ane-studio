from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id

from .catalog import KNOWN_FAILED_REVISIONS, SUPPORTED_ARCHITECTURES

# Names the one revision whose files a source directory may contain.
REVISION_MARKER = ".ane-studio-revision"


def token() -> str | bool:
    # None tells the Hub to use a cached login, even after the app token was deleted.
    value = os.environ.get("HF_TOKEN", "").strip()
    return value or False


def validate_model_id(model_id: Any) -> str:
    if not isinstance(model_id, str):
        raise ValueError("modelId must be a Hugging Face organization/model identifier")
    value = model_id.strip()
    if value.count("/") != 1 or len(value) > 200:
        raise ValueError("modelId must be a Hugging Face organization/model identifier")
    validate_repo_id(value)
    return value


def pinned_revision(revision: Any) -> str:
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("Hugging Face must return an exact model commit revision.")
    return revision


def inspect_model(model_id: str) -> dict[str, Any]:
    model_id = validate_model_id(model_id)
    access_token = token()
    info = HfApi(token=access_token).model_info(model_id)
    revision = pinned_revision(info.sha)
    # The Hub resolves case variants and renames. Store models under one canonical ID.
    with suppress(ValueError):
        model_id = validate_model_id(getattr(info, "id", None))
    config_path = Path(
        hf_hub_download(model_id, "config.json", revision=revision, token=access_token)
    )
    if config_path.stat().st_size > 1_048_576:
        raise ValueError("The model config is too large.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The model config must be a JSON object.")
    architectures = config.get("architectures")
    if architectures is None:
        architectures = []
    if (
        not isinstance(architectures, list)
        or len(architectures) > 16
        or any(not isinstance(item, str) or not item or len(item) > 128 for item in architectures)
    ):
        raise ValueError("The model config architectures must be a list of names.")
    architecture = architectures[0] if architectures else "unknown"
    parameter_count = None
    if info.safetensors and info.safetensors.total:
        parameter_count = int(info.safetensors.total)
    card = info.card_data.to_dict() if info.card_data else {}
    license_name = card.get("license")
    if not isinstance(license_name, str) or not license_name:
        license_name = "unknown"
    known_failure = KNOWN_FAILED_REVISIONS.get((model_id, info.sha))
    supported = architecture in SUPPORTED_ARCHITECTURES and known_failure is None
    if known_failure:
        reason = known_failure
    elif architecture not in SUPPORTED_ARCHITECTURES:
        reason = f"No converter is registered for {architecture}."
    else:
        reason = None
    return {
        "id": model_id,
        "architecture": architecture,
        "parameterCount": parameter_count,
        "license": license_name,
        "revision": revision,
        "private": info.private,
        "gated": bool(info.gated),
        "supported": supported,
        "adapter": SUPPORTED_ARCHITECTURES.get(architecture),
        "reason": reason,
    }


def _marked_revision(directory: Path) -> str | None:
    marker = directory / REVISION_MARKER
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 64:
        return None
    return marker.read_text(encoding="utf-8").strip()


def download_model(model_id: str, destination: Path, revision: str) -> Path:
    """Download one pinned revision and return the directory that holds it.

    A local directory only collects missing files. Files from another revision
    would remain, so a different revision downloads into a separate staging
    directory. Call replace_source() to move it into place.
    """
    model_id = validate_model_id(model_id)
    revision = pinned_revision(revision)
    target = destination
    if not destination.is_dir() or _marked_revision(destination) != revision:
        target = destination.with_name(f"{destination.name}.partial")
    for directory in (destination, target):
        if directory.is_symlink() or (
            directory.exists() and any(path.is_symlink() for path in directory.rglob("*"))
        ):
            raise ValueError("The model download directory must not contain symbolic links.")
    if target != destination and target.exists() and _marked_revision(target) != revision:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / REVISION_MARKER).write_text(revision + "\n", encoding="utf-8")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=target,
        token=token(),
        max_workers=2,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.model",
            "*.txt",
            "*.safetensors",
            "LICENSE*",
            "NOTICE*",
            "README*",
        ],
    )
    return target


def replace_source(staged: Path, destination: Path) -> None:
    if staged == destination:
        return
    previous = destination.with_name(f"{destination.name}.old")
    shutil.rmtree(previous, ignore_errors=True)
    if destination.exists():
        destination.rename(previous)
    staged.rename(destination)
    shutil.rmtree(previous, ignore_errors=True)
