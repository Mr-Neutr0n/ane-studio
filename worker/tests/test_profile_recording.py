from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_expired_recording_still_saving_cannot_publish_workload(monkeypatch, tmp_path: Path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "profile-model.py"
    spec = importlib.util.spec_from_file_location("profile_recording_regression", script)
    assert spec and spec.loader
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)

    clock = SimpleNamespace(now=99.0)
    calls = []

    class SavingRecorder:
        def poll(self):
            # xctrace has stopped collecting, but remains alive saving its trace.
            calls.append("recorder-poll")
            return None

    recorder = SavingRecorder()

    @contextmanager
    def recording(_path, _duration):
        try:
            yield recorder, 100.0
        finally:
            calls.append("recording-cleanup")

    class Runtime:
        def __init__(self, *_args):
            pass

        def generate(self, *_args):
            calls.append("generate")
            clock.now = 101.0
            return {"text": "Hello", "samples": [2_000.0]}

    class Signpost:
        def __init__(self, name):
            self.name = name

        def __call__(self):
            calls.append(self.name)

    library = SimpleNamespace(inference_begin=Signpost("begin"), inference_end=Signpost("end"))
    record = {"maxSequenceLength": 64}
    service = SimpleNamespace(_paths=lambda _model: (record, None, "source", "package"))
    output = tmp_path / "authoritative-workload.json"
    monkeypatch.setattr(profile, "Service", lambda _write: service)
    monkeypatch.setattr(profile, "CoreMLRuntime", Runtime)
    monkeypatch.setattr(profile, "record_trace", recording)
    monkeypatch.setattr(profile, "ctypes", SimpleNamespace(CDLL=lambda _path: library))
    monkeypatch.setattr(
        profile,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, monotonic_ns=lambda: 99_000_000_000),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--home",
            str(tmp_path / "models"),
            "--model",
            "test/local-model",
            "--trace",
            str(tmp_path / "expired.trace"),
            "--signpost-library",
            str(tmp_path / "signposts.dylib"),
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
    )
    for name in ("ANE_STUDIO_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "test-original-value")

    with pytest.raises(RuntimeError, match="Recording ended during generation"):
        profile.main()

    assert calls == [
        "recorder-poll",
        "begin",
        "generate",
        "end",
        "recorder-poll",
        "recording-cleanup",
    ]
    assert not output.exists()
