from __future__ import annotations

import json
import sys
import threading
import traceback
from typing import Any

from .service import Service

_LOCK = threading.Lock()


def write(value: dict[str, Any]) -> None:
    with _LOCK:
        sys.stdout.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")
        sys.stdout.flush()


def process(service: Service, request: dict[str, Any]) -> None:
    request_id = request.get("id")
    if not isinstance(request_id, str):
        write(
            {
                "id": "unknown",
                "ok": False,
                "error": {"code": "invalid_request", "message": "Request id must be text."},
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
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object.")
        except (json.JSONDecodeError, ValueError) as error:
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
