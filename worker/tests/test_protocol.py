from __future__ import annotations

import io
import json

import pytest

from ane_studio_worker import cli


class StubService:
    def handle(self, request_id, command, payload):
        return {"request": request_id, "command": command, "payload": payload}


def test_process_writes_one_response(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(cli, "write", written.append)
    cli.process(StubService(), {"id": "one", "command": "health", "payload": {}})
    assert written == [
        {
            "id": "one",
            "ok": True,
            "result": {"request": "one", "command": "health", "payload": {}},
        }
    ]


def test_process_serializes_errors(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(cli, "write", written.append)
    cli.process(StubService(), {"id": "one", "payload": {}})
    assert written[0]["ok"] is False
    assert written[0]["error"]["code"] == "ValueError"


@pytest.mark.parametrize(
    "candidate", [None, [], 7, "bad", {}, {"id": ""}, {"id": " "}, {"id": "x" * 129}]
)
def test_process_rejects_bad_request_ids(monkeypatch, candidate) -> None:
    written = []
    monkeypatch.setattr(cli, "write", written.append)
    cli.process(StubService(), candidate)
    assert len(written) == 1
    assert written[0]["ok"] is False
    assert written[0]["error"]["code"] == "invalid_request"


def run_cli(monkeypatch, text):
    written = []
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(text))
    monkeypatch.setattr(cli, "write", written.append)
    monkeypatch.setattr(cli, "Service", lambda write: StubService())
    cli.main()
    return written


@pytest.mark.parametrize(
    "bad_line",
    [
        "[]",
        "null",
        "{bad}",
        '{"id":"bad","command":"health","payload":{"x":NaN}}',
        '{"id":"bad","command":"health","payload":{"x":Infinity}}',
    ],
)
def test_main_recovers_after_invalid_json(monkeypatch, bad_line) -> None:
    written = run_cli(monkeypatch, bad_line + '\n{"id":"good","command":"health"}\n')
    assert len(written) == 2
    assert written[0]["ok"] is False
    assert written[1]["id"] == "good"
    assert written[1]["ok"] is True


def test_main_bounds_and_drains_oversized_line(monkeypatch) -> None:
    monkeypatch.setattr(cli, "MAX_REQUEST_LENGTH", 64)
    written = run_cli(monkeypatch, "x" * 200 + '\n{"id":"good","command":"health"}\n')
    assert len(written) == 2
    assert written[0]["error"]["message"] == "Request is too large."
    assert written[1]["id"] == "good"


def test_main_recovers_after_deeply_nested_json(monkeypatch) -> None:
    written = run_cli(monkeypatch, "[" * 1200 + "]" * 1200 + '\n{"id":"good","command":"health"}\n')
    assert len(written) == 2
    assert written[0]["ok"] is False
    assert written[1]["ok"] is True


def test_write_is_compact_strict_json_and_redacts_credentials(monkeypatch) -> None:
    output = io.StringIO()
    secret = "hf_" + "testcredential" * 3
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(cli.sys, "stdout", output)
    cli.write({"id": "test", "message": "error " + secret})
    assert secret not in output.getvalue()
    assert json.loads(output.getvalue())["message"] == "error [redacted]"
    with pytest.raises(ValueError):
        cli.write({"invalid": float("nan")})
    assert len(output.getvalue().splitlines()) == 1


def test_main_processes_requests_in_order(monkeypatch) -> None:
    written = run_cli(
        monkeypatch, '\n{"id":"first","command":"health"}\n{"id":"second","command":"catalog"}\n'
    )
    assert [value["id"] for value in written] == ["first", "second"]
