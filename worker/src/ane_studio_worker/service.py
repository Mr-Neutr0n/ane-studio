from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .catalog import CURATED_MODELS
from .coreml_backend import CoreMLRuntime, benchmark, convert_model, planned_placement
from .events import EventWriter, event, now
from .hardware import hardware_info
from .huggingface import download_model, inspect_model
from .storage import (
    directory_digest,
    list_manifests,
    load_manifest,
    model_directory,
    model_slug,
    remove_model,
    write_manifest,
)


class Service:
    def __init__(self, write: EventWriter) -> None:
        self.write = write
        self.runtimes: dict[tuple[str, str], CoreMLRuntime] = {}

    def handle(self, request_id: str, command: str, payload: dict[str, Any]) -> Any:
        handlers = {
            "health": self.health,
            "catalog": self.catalog,
            "inspect": self.inspect,
            "prepare": self.prepare,
            "chat": self.chat,
            "benchmark": self.run_benchmark,
            "placement": self.placement,
            "remove": self.remove,
        }
        handler = handlers.get(command)
        if handler is None:
            raise ValueError(f"Unsupported worker command: {command}")
        return handler(request_id, payload)

    def health(self, _request_id: str, _payload: dict[str, Any]) -> dict[str, object]:
        return hardware_info()

    def catalog(self, _request_id: str, _payload: dict[str, Any]) -> dict[str, Any]:
        return {"curatedModels": list(CURATED_MODELS), "models": list_manifests()}

    def inspect(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = required_model_id(payload)
        event(
            self.write,
            request_id,
            "progress",
            stage="inspect",
            progress=0.05,
            message="Reading Hugging Face metadata",
        )
        result = inspect_model(model_id)
        event(
            self.write,
            request_id,
            "progress",
            stage="inspect",
            progress=1.0,
            message="Model metadata loaded",
        )
        return result

    def prepare(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = required_model_id(payload)
        sequence_length = bounded_int(payload.get("sequenceLength", 64), 16, 256, "sequenceLength")
        info = inspect_model(model_id)
        if not info["supported"]:
            raise ValueError(str(info["reason"]))
        directory = model_directory(model_id)
        source = directory / "source"
        package = directory / "artifacts" / "model.mlpackage"
        record: dict[str, Any] = {
            "id": model_id,
            "slug": model_slug(model_id),
            "state": "downloading",
            "architecture": info["architecture"],
            "parameterCount": info["parameterCount"],
            "license": info["license"],
            "revision": info["revision"],
            "maxSequenceLength": sequence_length,
            "parity": {"status": "pending"},
            "updatedAt": now(),
        }
        write_manifest(model_id, record)
        event(self.write, request_id, "model", model=record)
        try:
            event(
                self.write,
                request_id,
                "progress",
                stage="download",
                progress=0.12,
                message="Downloading pinned model files",
            )
            download_model(model_id, source, info["revision"])
            record.update(state="converting", updatedAt=now())
            write_manifest(model_id, record)
            event(self.write, request_id, "model", model=record)
            converted = convert_model(
                model_id,
                source,
                package,
                info["architecture"],
                sequence_length,
                request_id,
                self.write,
            )
            digest, size = directory_digest(package)
            parity = converted["parity"]
            state = "ready" if parity["status"] == "passed" else "failed"
            record.update(
                state=state,
                artifactBytes=size,
                artifactSha256=digest,
                parity=parity,
                updatedAt=now(),
            )
            if state == "failed":
                record["error"] = "Core ML output did not pass the source-model parity gate."
            write_manifest(model_id, record)
            event(self.write, request_id, "model", model=record)
            event(
                self.write,
                request_id,
                "progress",
                stage="complete",
                progress=1.0,
                message="Model is ready" if state == "ready" else "Model failed parity",
                level="info" if state == "ready" else "error",
            )
            return record
        except Exception as error:
            record.update(
                state="failed",
                error=f"{type(error).__name__}: {error}",
                updatedAt=now(),
            )
            write_manifest(model_id, record)
            event(self.write, request_id, "model", model=record)
            raise

    def chat(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = required_model_id(payload)
        policy = str(payload.get("policy", "cpuAndNeuralEngine"))
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        normalized = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "user",
                "assistant",
            }:
                raise ValueError("Each message needs a valid role and content.")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Each message content must be text.")
            normalized.append({"role": message["role"], "content": content})
        runtime = self._runtime(model_id, policy)
        return runtime.generate(
            normalized,
            bounded_int(payload.get("maxNewTokens", 48), 1, 128, "maxNewTokens"),
            bounded_float(payload.get("temperature", 0.2), 0, 2, "temperature"),
            lambda token: event(self.write, request_id, "token", token=token),
        )

    def run_benchmark(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = required_model_id(payload)
        prompt = payload.get("prompt", "Explain why the sky appears blue in one sentence.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        policies = payload.get("policies", ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"])
        if not isinstance(policies, list) or not all(isinstance(item, str) for item in policies):
            raise ValueError("policies must be a list")
        record, directory, source, package = self._paths(model_id)
        event(
            self.write,
            request_id,
            "progress",
            stage="benchmark",
            progress=0.05,
            message="Starting clean policy comparisons",
        )
        report = benchmark(
            package,
            source,
            int(record["maxSequenceLength"]),
            model_id,
            prompt,
            policies,
        )
        results = directory / "results"
        results.mkdir(parents=True, exist_ok=True)
        path = results / f"benchmark-{now().replace(':', '-')}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        event(
            self.write,
            request_id,
            "progress",
            stage="benchmark",
            progress=1.0,
            message="Benchmark complete",
        )
        return report

    def placement(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = required_model_id(payload)
        policy = str(payload.get("policy", "cpuAndNeuralEngine"))
        _, _, _, package = self._paths(model_id)
        event(
            self.write,
            request_id,
            "progress",
            stage="placement",
            progress=0.2,
            message="Compiling the compute plan",
        )
        result = planned_placement(package, policy)
        event(
            self.write,
            request_id,
            "progress",
            stage="placement",
            progress=1.0,
            message="Compute plan ready",
        )
        return result

    def remove(self, _request_id: str, payload: dict[str, Any]) -> dict[str, bool]:
        model_id = required_model_id(payload)
        self.runtimes = {key: value for key, value in self.runtimes.items() if key[0] != model_id}
        remove_model(model_id)
        return {"removed": True}

    def _runtime(self, model_id: str, policy: str) -> CoreMLRuntime:
        key = (model_id, policy)
        runtime = self.runtimes.get(key)
        if runtime:
            return runtime
        record, _, source, package = self._paths(model_id)
        runtime = CoreMLRuntime(package, source, int(record["maxSequenceLength"]), policy)
        self.runtimes[key] = runtime
        return runtime

    @staticmethod
    def _paths(model_id: str) -> tuple[dict[str, Any], Path, Path, Path]:
        record = load_manifest(model_id)
        if (
            not record
            or record.get("state") != "ready"
            or record.get("parity", {}).get("status") != "passed"
        ):
            raise ValueError("The model must pass conversion parity before it can run.")
        directory = model_directory(model_id)
        source = directory / "source"
        package = directory / "artifacts" / "model.mlpackage"
        if not source.exists() or not package.exists():
            raise ValueError("The model record exists, but its local artifacts are missing.")
        return record, directory, source, package


def required_model_id(payload: dict[str, Any]) -> str:
    model_id = payload.get("modelId")
    if not isinstance(model_id, str) or "/" not in model_id or len(model_id) > 200:
        raise ValueError("modelId must be a Hugging Face organization/model identifier")
    return model_id.strip()


def bounded_int(value: Any, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def bounded_float(value: Any, minimum: float, maximum: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result
