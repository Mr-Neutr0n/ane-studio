from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .catalog import SUPPORTED_ARCHITECTURES


def token() -> str | None:
    value = os.environ.get("HF_TOKEN", "").strip()
    return value or None


def inspect_model(model_id: str) -> dict[str, Any]:
    info = HfApi(token=token()).model_info(model_id, files_metadata=True)
    config_path = hf_hub_download(model_id, "config.json", token=token())
    import json

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else "unknown"
    parameter_count = None
    if info.safetensors and info.safetensors.total:
        parameter_count = int(info.safetensors.total)
    card = info.card_data.to_dict() if info.card_data else {}
    license_name = card.get("license") or "unknown"
    return {
        "id": model_id,
        "architecture": architecture,
        "parameterCount": parameter_count,
        "license": license_name,
        "revision": info.sha,
        "private": info.private,
        "gated": bool(info.gated),
        "supported": architecture in SUPPORTED_ARCHITECTURES,
        "adapter": SUPPORTED_ARCHITECTURES.get(architecture),
        "reason": None
        if architecture in SUPPORTED_ARCHITECTURES
        else f"No converter is registered for {architecture}.",
    }


def download_model(model_id: str, destination: Path, revision: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
        token=token(),
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
    return destination
