from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

EventWriter = Callable[[dict[str, Any]], None]


def now() -> str:
    return datetime.now(UTC).isoformat()


def redact_credentials(text: str) -> str:
    access_token = os.environ.get("HF_TOKEN", "").strip()
    if access_token:
        text = text.replace(access_token, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]{20,}", "[redacted]", text)


def event(
    write: EventWriter,
    request_id: str,
    kind: str,
    *,
    stage: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    token: str | None = None,
    level: str = "info",
    model: dict[str, Any] | None = None,
) -> None:
    value: dict[str, Any] = {
        "kind": kind,
        "requestId": request_id,
        "at": now(),
        "level": level,
    }
    for key, item in (
        ("stage", stage),
        ("progress", progress),
        ("message", message),
        ("token", token),
        ("model", model),
    ):
        if item is not None:
            value[key] = item
    write(value)
