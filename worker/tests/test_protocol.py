from __future__ import annotations

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
