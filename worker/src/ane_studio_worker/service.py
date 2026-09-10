from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any

from .catalog import CURATED_MODELS
from .coreml_backend import POLICIES, CoreMLRuntime, benchmark, convert_model, planned_placement
from .events import EventWriter, event, now, redact_credentials
from .hardware import hardware_info
from .huggingface import download_model, inspect_model, validate_model_id
from .storage import (
    directory_digest,
    directory_signature,
    list_manifests,
    load_manifest,
    model_directory,
    model_path,
    model_slug,
    normalize_manifest,
    remove_model,
    write_manifest,
)


class Service:
    def __init__(self, write: EventWriter) -> None:
        self.write = write
        self.runtimes: dict[tuple[str, str, str, int], CoreMLRuntime] = {}
        self.verified_artifacts: dict[Path, tuple[str, tuple[tuple[Any, ...], ...]]] = {}

    def handle(self, request_id: str, command: str, payload: dict[str, Any]) -> Any:
        if not isinstance(command, str) or not isinstance(payload, dict):
            raise ValueError("Request requires a command and object payload.")
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
        source = model_path(model_id, "source")
        package = model_path(model_id, "artifacts", "model.mlpackage")
        self._clear_runtimes()
        self.verified_artifacts.clear()
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
        record = normalize_manifest(record)
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
            passed = parity_passed(parity)
            parity = dict(parity) if isinstance(parity, dict) else {}
            parity["status"] = "passed" if passed else "failed"
            # Strict JSON cannot store NaN/Infinity from a failed numerical check.
            parity = {
                key: value
                for key, value in parity.items()
                if value is not None and (not isinstance(value, float) or math.isfinite(value))
            }
            state = "ready" if passed else "failed"
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
                error=redact_credentials(f"{type(error).__name__}: {error}"),
                updatedAt=now(),
            )
            write_manifest(model_id, record)
            event(self.write, request_id, "model", model=record)
            raise

    def chat(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = required_model_id(payload)
        policy = required_policy(payload.get("policy", "cpuAndNeuralEngine"))
        max_new_tokens = bounded_int(payload.get("maxNewTokens", 48), 1, 128, "maxNewTokens")
        temperature = bounded_float(payload.get("temperature", 0.2), 0, 2, "temperature")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) > 128:
            raise ValueError("messages must contain between 1 and 128 messages")
        normalized = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or message["role"] not in {"system", "user", "assistant"}
            ):
                raise ValueError("Each message needs a valid role and content.")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Each message content must be text.")
            normalized.append({"role": message["role"], "content": content})
        if sum(len(message["content"]) for message in normalized) > 32_768:
            raise ValueError("messages must contain at most 32768 characters in total")
        runtime = self._runtime(model_id, policy, request_id=request_id)
        return runtime.generate(
            normalized,
            max_new_tokens,
            temperature,
            lambda token: event(self.write, request_id, "token", token=token),
        )

    def run_benchmark(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_id = required_model_id(payload)
        prompt = payload.get("prompt", "Explain why the sky appears blue in one sentence.")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32_768:
            raise ValueError("prompt must be non-empty text of at most 32768 characters")
        policies = payload.get("policies", ["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"])
        if (
            not isinstance(policies, list)
            or not 1 <= len(policies) <= len(POLICIES)
            or any(not isinstance(item, str) or item not in POLICIES for item in policies)
            or len(set(policies)) != len(policies)
        ):
            raise ValueError("policies must contain between 1 and 4 distinct compute policies")
        record, _, source, package = self._paths(model_id)
        self._clear_runtimes()
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
        results = model_path(model_id, "results")
        results.mkdir(parents=True, exist_ok=True)
        path = results / f"benchmark-{now().replace(':', '-')}.json"
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
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
        policy = required_policy(payload.get("policy", "cpuAndNeuralEngine"))
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

    def _clear_runtimes(self) -> None:
        self.runtimes.clear()
        gc.collect()

    def _runtime(
        self, model_id: str, policy: str, *, request_id: str | None = None
    ) -> CoreMLRuntime:
        policy = required_policy(policy)
        record, _, source, package = self._paths(model_id)
        key = (model_id, policy, record["artifactSha256"], record["maxSequenceLength"])
        runtime = self.runtimes.get(key)
        if runtime is not None:
            return runtime
        # Keeping every model/policy loaded can exhaust a 16 GB Mac.
        self._clear_runtimes()
        if request_id is not None:
            event(
                self.write,
                request_id,
                "progress",
                stage="loading",
                message=(
                    "Loading the local Core ML runtime. First use of a compute policy can take "
                    "several minutes. You can cancel."
                ),
            )
        runtime = CoreMLRuntime(package, source, record["maxSequenceLength"], policy)
        self.runtimes[key] = runtime
        return runtime

    def _paths(self, model_id: str) -> tuple[dict[str, Any], Path, Path, Path]:
        record = load_manifest(model_id)
        if not record or record.get("state") != "ready" or not parity_passed(record.get("parity")):
            raise ValueError("The model must pass conversion parity before it can run.")
        record["maxSequenceLength"] = bounded_int(
            record.get("maxSequenceLength"), 16, 256, "maxSequenceLength"
        )
        directory = model_directory(model_id)
        source = model_path(model_id, "source")
        package = model_path(model_id, "artifacts", "model.mlpackage")
        if not source.is_dir() or not package.is_dir():
            raise ValueError("The model record exists, but its local artifacts are missing.")
        directory_signature(source)
        expected = record.get("artifactSha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("The model artifact has no valid parity digest. Prepare it again.")
        signature = directory_signature(package)
        if self.verified_artifacts.get(package) != (expected, signature):
            digest, size = directory_digest(package)
            if digest != expected or not size:
                raise ValueError(
                    "The model artifact changed after parity validation. Prepare it again."
                )
            self.verified_artifacts.clear()
            self.verified_artifacts[package] = (expected, signature)
        return record, directory, source, package


def parity_passed(parity: Any) -> bool:
    if not isinstance(parity, dict) or parity.get("status") != "passed":
        return False
    cosine = parity.get("cosineSimilarity")
    source = parity.get("sourceToken")
    converted = parity.get("coremlToken")
    if type(cosine) not in (int, float):
        return False
    try:
        finite = math.isfinite(cosine)
    except OverflowError:
        return False
    return (
        finite
        and cosine >= 0.999
        and parity.get("topTokenMatches") is True
        and type(source) is int
        and type(converted) is int
        and source >= 0
        and source == converted
    )


def required_model_id(payload: dict[str, Any]) -> str:
    return validate_model_id(payload.get("modelId"))


def required_policy(value: Any) -> str:
    if not isinstance(value, str) or value not in POLICIES:
        raise ValueError("Unknown compute policy.")
    return value


def bounded_int(value: Any, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def bounded_float(value: Any, minimum: float, maximum: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result
