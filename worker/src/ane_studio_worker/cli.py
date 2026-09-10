from __future__ import annotations

import json
import sys
import threading
import traceback
from typing import Any

from .events import redact_credentials
from .service import Service

_LOCK = threading.Lock()
MAX_REQUEST_LENGTH = 1_048_576


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON numbers are not allowed: {value}")


def redact_values(value: Any) -> Any:
    if isinstance(value, str):
        return redact_credentials(value)
    if isinstance(value, dict):
        return {key: redact_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_values(item) for item in value]
    return value


def write(value: dict[str, Any]) -> None:
    with _LOCK:
        text = json.dumps(
            redact_values(value), separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


def process(service: Service, request: Any) -> None:
    request_id = request.get("id") if isinstance(request, dict) else None
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
        write(
            {
                "id": "unknown",
                "ok": False,
                "error": {
                    "code": "invalid_request",
                    "message": "Request id must contain between 1 and 128 characters.",
                },
            }
        )
        return
    try:
        command = request.get("command")
        payload = request.get("payload", {})
        if not isinstance(command, str) or not isinstance(payload, dict):
            raise ValueError("Request requires a command and object payload.")
        result = service.handle(request_id, command, payload)
        write({"id": request_id, "ok": True, "result": result})
    except Exception as error:
        write(
            {
                "id": request_id,
                "ok": False,
                "error": {
                    "code": type(error).__name__,
                    "message": str(error) or type(error).__name__,
                    "detail": traceback.format_exc(),
                },
            }
        )


def main() -> None:
    service = Service(write)
    while line := sys.stdin.readline(MAX_REQUEST_LENGTH + 1):
        if len(line) > MAX_REQUEST_LENGTH:
            while not line.endswith("\n"):
                line = sys.stdin.readline(MAX_REQUEST_LENGTH + 1)
                if not line:
                    break
            write(
                {
                    "id": "unknown",
                    "ok": False,
                    "error": {"code": "invalid_request", "message": "Request is too large."},
                }
            )
            continue
        if not line.strip():
            continue
        try:
            request = json.loads(line, parse_constant=reject_constant)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object.")
        except (ValueError, RecursionError) as error:
            write(
                {
                    "id": "unknown",
                    "ok": False,
                    "error": {"code": "invalid_json", "message": str(error)},
                }
            )
            continue
        process(service, request)


if __name__ == "__main__":
    main()
