from __future__ import annotations

import gc
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .events import EventWriter, event

POLICIES = {
    "cpuOnly": "CPU_ONLY",
    "cpuAndGPU": "CPU_AND_GPU",
    "cpuAndNeuralEngine": "CPU_AND_NE",
    "all": "ALL",
}


def _imports() -> tuple[Any, Any, Any, Any]:
    try:
        import coremltools as ct
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The Core ML environment is incomplete. Run `bun run worker:setup` "
            "from the repository root."
        ) from error
    return ct, np, torch, (AutoModelForCausalLM, AutoTokenizer)


@contextmanager
def _patch_trace_gaps(
    architecture: str,
    sequence_length: int,
    torch: Any,
    *,
    convolution_kernel_size: int = 3,
):
    # Valid-token outputs are unchanged for contiguous, right-padded input only.
    # Pads occur in the causal future of the selected token.
    import importlib

    module_names = {
        "Lfm2ForCausalLM": "lfm2",
        "Qwen2ForCausalLM": "qwen2",
        "LlamaForCausalLM": "llama",
    }
    family = module_names.get(architecture)
    if family is None:
        raise ValueError(f"No trace adapter for {architecture}.")
    modeling = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    mask = torch.zeros(1, 1, sequence_length, sequence_length, dtype=torch.float32)
    if sequence_length > 1:
        mask = torch.triu(
            torch.full((sequence_length, sequence_length), -10_000.0, dtype=torch.float32),
            diagonal=1,
        )[None, None]
    original_mask = modeling.create_causal_mask
    original_conv = None
    try:
        modeling.create_causal_mask = lambda *args, **kwargs: mask
        if architecture == "Lfm2ForCausalLM":
            import torch.nn.functional as functional

            original_conv = modeling.causal_conv1d_fn

            def traceable_causal_conv1d(
                hidden_states: Any,
                weight: Any,
                bias: Any = None,
                activation: str | None = None,
                **kwargs: Any,
            ) -> Any:
                sequence = hidden_states.shape[-1]
                output = functional.conv1d(
                    hidden_states.to(weight.dtype),
                    weight=weight.unsqueeze(1),
                    bias=bias,
                    padding=convolution_kernel_size - 1,
                    groups=weight.shape[0],
                )[:, :, :sequence]
                if activation is not None:
                    output = modeling.ACT2FN[activation](output)
                return output.to(hidden_states.dtype)

            modeling.causal_conv1d_fn = traceable_causal_conv1d
        yield
    finally:
        modeling.create_causal_mask = original_mask
        if architecture == "Lfm2ForCausalLM":
            modeling.causal_conv1d_fn = original_conv


def _inputs(tokenizer: Any, text: str, sequence_length: int, np: Any) -> tuple[dict[str, Any], int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=sequence_length)
    return _token_inputs(tokenizer, list(encoded["input_ids"]), sequence_length, np)


def _token_inputs(
    tokenizer: Any, token_ids: list[int], sequence_length: int, np: Any
) -> tuple[dict[str, Any], int]:
    if not token_ids:
        first = tokenizer.bos_token_id
        if first is None:
            first = tokenizer.eos_token_id
        token_ids = [first if first is not None else 0]
    token_ids = token_ids[-sequence_length:]
    position = len(token_ids) - 1
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    padded = token_ids + [pad] * (sequence_length - len(token_ids))
    mask = [1] * len(token_ids) + [0] * (sequence_length - len(token_ids))
    return {
        "input_ids": np.asarray([padded], dtype=np.int32),
        "attention_mask": np.asarray([mask], dtype=np.int32),
        "position": np.asarray([position], dtype=np.int32),
    }, position


def _cosine(left: Any, right: Any, np: Any) -> float:
    if left.shape != right.shape or left.size == 0:
        return 0.0
    a = left.astype(np.float64).reshape(-1)
    b = right.astype(np.float64).reshape(-1)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    # Zero logits are not evidence of a functioning language model.
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _parity_metrics(reference: Any, prediction: Any, np: Any) -> dict[str, Any]:
    valid = bool(
        reference.shape == prediction.shape
        and reference.size > 0
        and np.isfinite(reference).all()
        and np.isfinite(prediction).all()
        and np.linalg.norm(reference.astype(np.float64)) > 0
        and np.linalg.norm(prediction.astype(np.float64)) > 0
    )
    cosine = _cosine(reference, prediction, np)
    source_token = int(reference.argmax()) if valid else None
    coreml_token = int(prediction.argmax()) if valid else None
    matches = valid and source_token == coreml_token
    metrics = {
        "status": "passed" if cosine >= 0.999 and matches else "failed",
        "cosineSimilarity": cosine,
        "topTokenMatches": matches,
    }
    if valid:
        metrics.update(
            maximumAbsoluteError=float(
                np.max(np.abs(reference.astype(np.float64) - prediction.astype(np.float64)))
            ),
            sourceToken=source_token,
            coremlToken=coreml_token,
        )
    return metrics


def _fp16_matrix_operation(operation: Any) -> bool:
    # Preserve FP32 normalization, softmax and residual math. Fixed-position RoPE
    # folds to constants before this pass. All surviving matmuls use FP16.
    # Casting RMS statistics to FP16 can overflow x**2 and collapse logits to zero.
    return operation.op_type in {"linear", "matmul"}


def _compute_precision(architecture: str, ct: Any) -> Any:
    if architecture == "Qwen2ForCausalLM":
        # Qwen2.5's position-zero fixture fails even with FP16 linear only.
        return ct.precision.FLOAT32
    if architecture == "LlamaForCausalLM":
        return ct.transform.FP16ComputePrecision(op_selector=_fp16_matrix_operation)
    return ct.precision.FLOAT16


def convert_model(
    model_id: str,
    source: Path,
    output: Path,
    architecture: str,
    sequence_length: int,
    request_id: str,
    write: EventWriter,
) -> dict[str, Any]:
    ct, np, torch, auto = _imports()
    AutoModelForCausalLM, AutoTokenizer = auto
    mixed_precision = architecture == "LlamaForCausalLM"
    fp32_conversion = architecture == "Qwen2ForCausalLM"
    fp32_source = mixed_precision or fp32_conversion
    event(
        write,
        request_id,
        "progress",
        stage="loading",
        progress=0.38,
        message="Loading FP32 source weights" if fp32_source else "Loading FP16 weights",
    )
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32 if fp32_source else torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

    class LastTokenModel(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any, attention_mask: Any, position: Any) -> Any:
            hidden = self.wrapped.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )[0]
            # Select before the vocabulary projection. This avoids materializing
            # sequence_length vocabulary vectors on every decode step.
            last = torch.index_select(hidden, 1, position.to(torch.int64)).squeeze(1)
            return self.wrapped.lm_head(last)

    prompts = [
        ("completion", "The capital of France is"),
        ("short", "The opposite of hot is"),
        ("first-token", "Hello"),
        ("context-boundary", "A short sentence about the sky and the sea. " * sequence_length),
    ]
    feeds = [_inputs(tokenizer, text, sequence_length, np)[0] for _, text in prompts]
    torch_feeds = [
        tuple(
            torch.from_numpy(feed[key]).to(torch.int64)
            for key in ("input_ids", "attention_mask", "position")
        )
        for feed in feeds
    ]
    references = []
    # Use the original, unpatched Transformers forward as the parity reference.
    with torch.no_grad():
        for args in torch_feeds:
            logits = model(
                input_ids=args[0], attention_mask=args[1], use_cache=False, return_dict=False
            )[0]
            references.append(logits[:, int(args[2][0])].float().cpu().numpy().copy())
        del logits

    wrapper = LastTokenModel(model).eval()
    event(
        write,
        request_id,
        "progress",
        stage="tracing",
        progress=0.48,
        message="Tracing a fixed-shape decoder",
    )
    with (
        torch.no_grad(),
        _patch_trace_gaps(
            architecture,
            sequence_length,
            torch,
            convolution_kernel_size=int(config.get("conv_L_cache", 3)),
        ),
    ):
        for reference, args in zip(references, torch_feeds, strict=True):
            adapted = wrapper(*args).float().cpu().numpy()
            if _parity_metrics(reference, adapted, np)["status"] != "passed":
                raise RuntimeError("Trace adapter changed the source model output.")
        traced = torch.jit.trace(wrapper, torch_feeds[0], strict=False, check_trace=False)

    event(
        write,
        request_id,
        "progress",
        stage="converting",
        progress=0.58,
        message="Converting to a Core ML ML Program",
    )
    # Do not compile the default compute policy while source/traced weights are
    # resident. The CPU parity runtime loads only after those objects are freed.
    with _managed_compile_directory(output.parent):
        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, sequence_length), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, sequence_length), dtype=np.int32),
                ct.TensorType(name="position", shape=(1,), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=_compute_precision(architecture, ct),
            skip_model_load=True,
        )
        shutil.rmtree(output, ignore_errors=True)
        converted.save(output)
    del converted, traced, wrapper, model, adapted
    gc.collect()

    event(
        write,
        request_id,
        "progress",
        stage="validating",
        progress=0.82,
        message="Checking Core ML output against the unpatched source model",
    )
    checks = []
    with _managed_compile_directory(output.parent):
        coreml_model = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
    for (name, _), feed, reference in zip(prompts, feeds, references, strict=True):
        prediction = coreml_model.predict(feed)["logits"]
        checks.append(
            {
                "name": name,
                "position": int(feed["position"][0]),
                **_parity_metrics(reference, prediction, np),
            }
        )
    parity = {
        **checks[0],
        "status": "passed" if all(check["status"] == "passed" for check in checks) else "failed",
        "cosineSimilarity": min(check["cosineSimilarity"] for check in checks),
        "topTokenMatches": all(check["topTokenMatches"] for check in checks),
        "converterVersion": 3,
        "reference": "unpatched-transformers-fp32"
        if fp32_source
        else "unpatched-transformers-fp16",
        "computePrecision": "fp32"
        if fp32_conversion
        else "fp16-matrices-fp32-sensitive-ops"
        if mixed_precision
        else "fp16",
        "checks": checks,
    }
    errors = [check["maximumAbsoluteError"] for check in checks if "maximumAbsoluteError" in check]
    if errors:
        parity["maximumAbsoluteError"] = max(errors)
    del coreml_model
    gc.collect()
    return {"parity": parity, "position": int(feeds[0]["position"][0])}


@contextmanager
def _managed_compile_directory(parent: Path):
    parent.mkdir(parents=True, exist_ok=True)
    previous = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix=".ane-studio-compile-", dir=parent) as root:
        tempfile.tempdir = root
        try:
            yield Path(root)
        finally:
            tempfile.tempdir = previous


class CoreMLRuntime:
    def __init__(self, package: Path, source: Path, sequence_length: int, policy: str) -> None:
        ct, np, _, auto = _imports()
        _, AutoTokenizer = auto
        self.ct = ct
        self.np = np
        self.sequence_length = sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.padding_side = "right"
        unit_name = POLICIES.get(policy)
        if not unit_name:
            raise ValueError(f"Unknown compute policy: {policy}")
        with _managed_compile_directory(package.parent):
            self.model = ct.models.MLModel(
                str(package),
                compute_units=getattr(ct.ComputeUnit, unit_name),
            )

    def prompt_text(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ValueError, TypeError, AttributeError):
            lines = [f"{message['role'].title()}: {message['content']}" for message in messages]
            return "\n".join(lines) + "\nAssistant:"

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        text = self.prompt_text(messages)
        encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = list(encoded)
        if len(ids) >= self.sequence_length:
            raise ValueError(
                f"The chat prompt uses {len(ids)} tokens, but this fixed graph has a "
                f"{self.sequence_length}-token context. Shorten the prompt or start a new chat."
            )
        generated: list[int] = []
        durations: list[float] = []
        eos = self.tokenizer.eos_token_id
        rng = self.np.random.default_rng(941731)
        capacity = self.sequence_length - len(ids)
        stop_reason = "contextLimit" if capacity < max_new_tokens else "maxTokens"
        for _ in range(min(max_new_tokens, capacity)):
            feed, _ = _token_inputs(self.tokenizer, ids, self.sequence_length, self.np)
            started = time.perf_counter_ns()
            logits = self.model.predict(feed)["logits"].astype(self.np.float32)[0]
            duration = (time.perf_counter_ns() - started) / 1_000_000
            if (
                logits.ndim != 1
                or logits.size == 0
                or not self.np.isfinite(logits).all()
                or not self.np.any(logits)
            ):
                raise RuntimeError("Core ML returned invalid or zero logits.")
            durations.append(duration)
            if temperature <= 0:
                token_id = int(logits.argmax())
            else:
                scaled = logits / max(temperature, 1e-4)
                scaled -= scaled.max()
                probabilities = self.np.exp(scaled)
                probabilities /= probabilities.sum()
                token_id = int(rng.choice(len(probabilities), p=probabilities))
            if eos is not None and token_id == eos:
                stop_reason = "eos"
                break
            ids.append(token_id)
            generated.append(token_id)
            piece = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if on_token and piece:
                on_token(piece)
        output = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "text": output,
            "tokens": len(generated),
            "stopReason": stop_reason,
            "contextTokens": len(ids),
            "sequenceLength": self.sequence_length,
            "firstTokenMs": durations[0] if durations else None,
            "medianTokenMs": float(self.np.median(durations)) if durations else None,
            "tokensPerSecond": 1000.0 / float(self.np.median(durations)) if durations else None,
            "samples": durations,
        }


def planned_placement(package: Path, policy: str) -> dict[str, Any]:
    ct, _, _, _ = _imports()
    note = "MLComputePlan reports anticipated placement. It does not prove runtime execution."
    try:
        from coremltools.models.compute_device import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )
        from coremltools.models.compute_plan import MLComputePlan
        from coremltools.models.utils import compile_model

        unit_name = POLICIES[policy]
        compiled: str | None = None
        with _managed_compile_directory(package.parent):
            try:
                compiled = compile_model(str(package))
                plan = MLComputePlan.load_from_path(
                    compiled,
                    compute_units=getattr(ct.ComputeUnit, unit_name),
                )
                counts = {"CPU": 0, "GPU": 0, "ANE": 0, "unknown": 0}
                structure = plan.model_structure.program
                for function in structure.functions.values():
                    for operation in function.block.operations:
                        if operation.operator_name.startswith("const"):
                            continue
                        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                        device = getattr(usage, "preferred_compute_device", None)
                        if isinstance(device, MLNeuralEngineComputeDevice):
                            counts["ANE"] += 1
                        elif isinstance(device, MLGPUComputeDevice):
                            counts["GPU"] += 1
                        elif isinstance(device, MLCPUComputeDevice):
                            counts["CPU"] += 1
                        else:
                            counts["unknown"] += 1
            finally:
                if compiled:
                    shutil.rmtree(compiled, ignore_errors=True)
        return {
            "evidence": "planned",
            "policy": policy,
            "observed": False,
            "operationCounts": counts,
            "note": note,
        }
    except Exception as error:
        return {
            "evidence": "planned",
            "policy": policy,
            "observed": False,
            "note": f"Compute plan unavailable: {type(error).__name__}: {error}",
        }


def benchmark(
    package: Path,
    source: Path,
    sequence_length: int,
    model_id: str,
    prompt: str,
    policies: list[str],
) -> dict[str, Any]:
    reports = []
    messages = [{"role": "user", "content": prompt}]
    for policy in policies:
        runtime = None
        try:
            runtime = CoreMLRuntime(package, source, sequence_length, policy)
            runtime.generate(messages, 1, 0)
            result = runtime.generate(messages, 8, 0)
            runtime = None
            gc.collect()
            reports.append(
                {
                    "policy": policy,
                    "valid": True,
                    "firstTokenMs": result["firstTokenMs"],
                    "medianTokenMs": result["medianTokenMs"],
                    "tokensPerSecond": result["tokensPerSecond"],
                    "samples": result["samples"],
                    "placement": [
                        {
                            "evidence": "requested",
                            "policy": policy,
                            "observed": False,
                            "note": "The selected Core ML policy allowed these compute units.",
                        },
                        planned_placement(package, policy),
                        {
                            "evidence": "observed",
                            "policy": policy,
                            "observed": False,
                            "note": "No Instruments trace was captured for this timing run.",
                        },
                    ],
                }
            )
        except Exception as error:
            reports.append(
                {
                    "policy": policy,
                    "valid": False,
                    "samples": [],
                    "placement": [],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            del runtime
            gc.collect()
    return {
        "modelId": model_id,
        "prompt": prompt,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policies": reports,
    }
