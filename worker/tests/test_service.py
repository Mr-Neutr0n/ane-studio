from __future__ import annotations

from pathlib import Path

import pytest

from ane_studio_worker import service
from ane_studio_worker.service import Service, bounded_float, bounded_int, required_model_id


def test_validates_model_identifier() -> None:
    assert required_model_id({"modelId": "LiquidAI/LFM2.5-230M"}) == "LiquidAI/LFM2.5-230M"
    with pytest.raises(ValueError):
        required_model_id({"modelId": "local-model"})


def test_bounds_generation_values() -> None:
    assert bounded_int("64", 16, 256, "sequence") == 64
    assert bounded_float("0.2", 0, 2, "temperature") == pytest.approx(0.2)
    with pytest.raises(ValueError):
        bounded_int(512, 16, 256, "sequence")
    with pytest.raises(ValueError):
        bounded_float(-1, 0, 2, "temperature")


def test_unknown_converter_fails_before_download(monkeypatch, tmp_path: Path) -> None:
    events = []
    worker = Service(events.append)
    monkeypatch.setattr(
        service,
        "inspect_model",
        lambda model_id: {
            "id": model_id,
            "supported": False,
            "reason": "No converter is registered for MysteryLM.",
            "architecture": "MysteryLM",
            "parameterCount": 1,
            "license": "mit",
            "revision": "abc",
        },
    )
    with pytest.raises(ValueError, match="MysteryLM"):
        worker.prepare("request", {"modelId": "org/model"})
    assert not events


@pytest.fixture(autouse=True)
def local_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANE_STUDIO_HOME", str(tmp_path / "studio"))


def passed_parity(**changes):
    return {
        "status": "passed",
        "cosineSimilarity": 0.999,
        "topTokenMatches": True,
        "sourceToken": 42,
        "coremlToken": 42,
        **changes,
    }


def ready_model(model_id="org/model", **changes):
    directory = service.model_directory(model_id)
    source = directory / "source"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    package = directory / "artifacts" / "model.mlpackage"
    package.mkdir(parents=True)
    (package / "Manifest.json").write_text("model")
    digest, size = service.directory_digest(package)
    record = {
        "id": model_id,
        "state": "ready",
        "parity": passed_parity(),
        "maxSequenceLength": 64,
        "artifactSha256": digest,
        "artifactBytes": size,
        **changes,
    }
    service.write_manifest(model_id, record)
    return record, source, package


@pytest.mark.parametrize(
    "model_id",
    [
        "../model",
        "org/../model",
        "/org/model",
        "org/",
        "org/a b",
        "org/a--b",
        "https://huggingface.co/org/model",
        7,
        None,
    ],
)
def test_rejects_invalid_model_paths(model_id) -> None:
    with pytest.raises(ValueError):
        required_model_id({"modelId": model_id})


@pytest.mark.parametrize("value", [True, False, 16.5, float("nan"), float("inf"), [], None])
def test_rejects_non_integer_bounds(value) -> None:
    with pytest.raises(ValueError):
        bounded_int(value, 1, 256, "sequence")


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), [], None])
def test_rejects_non_finite_or_boolean_floats(value) -> None:
    with pytest.raises(ValueError):
        bounded_float(value, 0, 2, "temperature")


@pytest.mark.parametrize(
    "parity",
    [
        None,
        [],
        {"status": "passed"},
        passed_parity(cosineSimilarity=0.998999),
        passed_parity(cosineSimilarity=float("nan")),
        passed_parity(cosineSimilarity=float("inf")),
        passed_parity(cosineSimilarity=True),
        passed_parity(cosineSimilarity="1"),
        passed_parity(topTokenMatches=False),
        passed_parity(topTokenMatches=1),
        passed_parity(coremlToken=43),
        passed_parity(sourceToken=True, coremlToken=1),
        passed_parity(sourceToken=-1, coremlToken=-1),
    ],
)
def test_quantitative_parity_gate_rejects_invalid_results(parity) -> None:
    assert service.parity_passed(parity) is False


def test_quantitative_parity_gate_keeps_exact_threshold() -> None:
    assert service.parity_passed(passed_parity()) is True
    assert service.parity_passed(passed_parity(cosineSimilarity=1)) is True


@pytest.mark.parametrize("operation", ["chat", "benchmark", "placement"])
def test_every_execution_path_requires_quantitative_parity(operation) -> None:
    ready_model(parity={"status": "passed"})
    worker = Service(lambda value: None)
    with pytest.raises(ValueError, match="parity"):
        worker.handle(
            "request",
            operation,
            {"modelId": "org/model", "messages": [{"role": "user", "content": "hi"}]},
        )


def test_cached_runtime_rechecks_failed_manifest(monkeypatch) -> None:
    record, _, _ = ready_model()
    monkeypatch.setattr(service, "CoreMLRuntime", lambda *args: object())
    worker = Service(lambda value: None)
    runtime = worker._runtime("org/model", "cpuOnly")
    assert worker._runtime("org/model", "cpuOnly") is runtime
    record["state"] = "failed"
    service.write_manifest("org/model", record)
    with pytest.raises(ValueError, match="parity"):
        worker._runtime("org/model", "cpuOnly")


def test_cached_runtime_rechecks_missing_artifact(monkeypatch) -> None:
    _, _, package = ready_model()
    monkeypatch.setattr(service, "CoreMLRuntime", lambda *args: object())
    worker = Service(lambda value: None)
    worker._runtime("org/model", "cpuOnly")
    service.remove_model("org/model")
    assert not package.exists()
    with pytest.raises(ValueError, match="parity"):
        worker._runtime("org/model", "cpuOnly")


def test_cached_runtime_rejects_modified_artifact(monkeypatch) -> None:
    _, _, package = ready_model()
    monkeypatch.setattr(service, "CoreMLRuntime", lambda *args: object())
    worker = Service(lambda value: None)
    worker._runtime("org/model", "cpuOnly")
    (package / "Manifest.json").write_text("other")
    with pytest.raises(ValueError, match="changed after parity"):
        worker._runtime("org/model", "cpuOnly")


def test_digest_is_reused_only_while_artifact_signature_matches(monkeypatch) -> None:
    ready_model()
    original = service.directory_digest
    calls = []

    def digest(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(service, "directory_digest", digest)
    worker = Service(lambda value: None)
    worker._paths("org/model")
    worker._paths("org/model")
    assert len(calls) == 1


def test_runtime_cache_keeps_only_one_policy(monkeypatch) -> None:
    ready_model()
    monkeypatch.setattr(service, "CoreMLRuntime", lambda *args: object())
    worker = Service(lambda value: None)
    first = worker._runtime("org/model", "cpuOnly")
    second = worker._runtime("org/model", "all")
    assert first is not second
    assert len(worker.runtimes) == 1


def mock_prepare(monkeypatch, parity):
    monkeypatch.setattr(
        service,
        "inspect_model",
        lambda model_id: {
            "supported": True,
            "architecture": "Lfm2ForCausalLM",
            "parameterCount": 1,
            "license": "mit",
            "revision": "a" * 40,
        },
    )

    def download(model_id, source, revision):
        source.mkdir(parents=True, exist_ok=True)
        (source / "config.json").write_text("{}")

    def convert(model_id, source, package, *args):
        package.mkdir(parents=True, exist_ok=True)
        (package / "Manifest.json").write_text("converted")
        return {"parity": parity}

    monkeypatch.setattr(service, "download_model", download)
    monkeypatch.setattr(service, "convert_model", convert)


@pytest.mark.parametrize(
    "parity",
    [
        passed_parity(cosineSimilarity=0.9),
        passed_parity(coremlToken=3),
        passed_parity(cosineSimilarity=float("nan")),
    ],
)
def test_prepare_does_not_trust_passed_status(monkeypatch, parity) -> None:
    mock_prepare(monkeypatch, parity)
    events = []
    result = Service(events.append).prepare("request", {"modelId": "org/model"})
    assert result["state"] == "failed"
    assert result["parity"]["status"] == "failed"
    assert service.load_manifest("org/model")["state"] == "failed"
    assert events[-1]["level"] == "error"


def test_prepare_succeeds_only_with_numeric_parity(monkeypatch) -> None:
    mock_prepare(monkeypatch, passed_parity())
    worker = Service(lambda value: None)
    result = worker.prepare("request", {"modelId": "org/model"})
    assert result["state"] == "ready"
    assert len(result["artifactSha256"]) == 64
    assert worker._paths("org/model")[0]["parity"]["cosineSimilarity"] == 0.999


def test_failed_reprepare_evicts_cache_and_records_error(monkeypatch) -> None:
    ready_model()
    mock_prepare(monkeypatch, passed_parity())
    monkeypatch.setattr(service, "CoreMLRuntime", lambda *args: object())
    worker = Service(lambda value: None)
    worker._runtime("org/model", "cpuOnly")

    def fail(*args):
        raise OSError("download interrupted")

    monkeypatch.setattr(service, "download_model", fail)
    with pytest.raises(OSError, match="interrupted"):
        worker.prepare("request", {"modelId": "org/model"})
    assert not worker.runtimes
    assert service.load_manifest("org/model")["state"] == "failed"
    with pytest.raises(ValueError, match="parity"):
        worker._runtime("org/model", "cpuOnly")


@pytest.mark.parametrize(
    "payload",
    [
        {"policy": []},
        {"policy": "bogus"},
        {"temperature": True},
        {"maxNewTokens": 1.5},
        {"messages": [{"role": "tool", "content": "hi"}]},
    ],
)
def test_chat_rejects_bad_inputs_before_loading_runtime(monkeypatch, payload) -> None:
    worker = Service(lambda value: None)

    def unexpected(*args):
        pytest.fail("runtime must not load for invalid input")

    monkeypatch.setattr(worker, "_runtime", unexpected)
    with pytest.raises(ValueError):
        worker.chat(
            "request",
            {"modelId": "org/model", "messages": [{"role": "user", "content": "hi"}], **payload},
        )


@pytest.mark.parametrize("policies", [[], ["all", "all"], ["bad"], [None], "all", [["all"]]])
def test_benchmark_rejects_bad_policies_before_loading(monkeypatch, policies) -> None:
    worker = Service(lambda value: None)
    with pytest.raises(ValueError, match="policies"):
        worker.run_benchmark("request", {"modelId": "org/model", "policies": policies})


def test_runtime_local_execution_does_not_contact_hub(monkeypatch) -> None:
    ready_model()

    def forbidden(*args, **kwargs):
        pytest.fail("local execution contacted the Hub")

    monkeypatch.setattr(service, "inspect_model", forbidden)
    monkeypatch.setattr(service, "download_model", forbidden)

    class Runtime:
        def __init__(self, *args):
            pass

        def generate(self, messages, tokens, temperature, callback):
            callback("local")
            return {"text": "local", "tokens": 1}

    monkeypatch.setattr(service, "CoreMLRuntime", Runtime)
    events = []
    result = Service(events.append).chat(
        "request", {"modelId": "org/model", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert result["text"] == "local"
    assert events[-1]["kind"] == "token"


def test_uncached_runtime_emits_loading_progress_before_constructor(monkeypatch) -> None:
    ready_model()
    events = []

    def construct(*args):
        assert len(events) == 1
        assert events[0]["stage"] == "loading"
        assert events[0]["requestId"] == "first"
        assert "progress" not in events[0]
        assert "cancel" in events[0]["message"]
        return object()

    monkeypatch.setattr(service, "CoreMLRuntime", construct)
    worker = Service(events.append)
    first = worker._runtime("org/model", "cpuOnly", request_id="first")
    assert worker._runtime("org/model", "cpuOnly", request_id="second") is first
    assert len(events) == 1


def test_prepare_events_omit_unknown_parameter_count(monkeypatch) -> None:
    mock_prepare(monkeypatch, passed_parity())
    monkeypatch.setattr(
        service,
        "inspect_model",
        lambda model_id: {
            "supported": True,
            "architecture": "Lfm2ForCausalLM",
            "parameterCount": None,
            "license": "mit",
            "revision": "a" * 40,
        },
    )
    events = []
    record = Service(events.append).prepare("request", {"modelId": "org/model"})
    assert "parameterCount" not in record
    assert all(
        "parameterCount" not in value["model"] for value in events if value["kind"] == "model"
    )
