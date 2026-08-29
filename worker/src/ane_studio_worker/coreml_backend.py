from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
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


def _patch_trace_gaps(
    architecture: str,
    sequence_length: int,
    torch: Any,
    *,
    convolution_kernel_size: int = 3,
) -> None:
    mask = torch.zeros(1, 1, sequence_length, sequence_length, dtype=torch.float32)
    if sequence_length > 1:
        mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                torch.finfo(torch.float32).min,
                dtype=torch.float32,
            ),
            diagonal=1,
        )[None, None]
    if architecture == "Lfm2ForCausalLM":
        import torch.nn.functional as functional
        import transformers.models.lfm2.modeling_lfm2 as modeling

        modeling.create_causal_mask = lambda *args, **kwargs: mask

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
    elif architecture == "Qwen2ForCausalLM":
        import transformers.models.qwen2.modeling_qwen2 as modeling

        if hasattr(modeling, "create_causal_mask"):
            modeling.create_causal_mask = lambda *args, **kwargs: mask


def _inputs(tokenizer: Any, text: str, sequence_length: int, np: Any) -> tuple[dict[str, Any], int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=True, max_length=sequence_length)
    token_ids = list(encoded["input_ids"])
    if not token_ids:
        token_ids = [tokenizer.bos_token_id or tokenizer.eos_token_id or 0]
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
    a = left.astype(np.float32).reshape(-1)
    b = right.astype(np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


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
    event(
        write,
        request_id,
        "progress",
        stage="loading",
        progress=0.38,
        message="Loading FP16 weights",
    )
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    _patch_trace_gaps(
        architecture,
        sequence_length,
        torch,
        convolution_kernel_size=int(config.get("conv_L_cache", 3)),
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

    class LastTokenModel(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: Any, attention_mask: Any, position: Any) -> Any:
            logits = self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )[0]
            index = position.to(torch.int64).reshape(1, 1, 1).expand(1, 1, logits.shape[-1])
            return torch.gather(logits, 1, index).squeeze(1)

    prompt = "The capital of France is"
    feed, position = _inputs(tokenizer, prompt, sequence_length, np)
    torch_feed = (
        torch.from_numpy(feed["input_ids"]).to(torch.int64),
        torch.from_numpy(feed["attention_mask"]).to(torch.int64),
        torch.from_numpy(feed["position"]).to(torch.int64),
    )
    wrapper = LastTokenModel(model).eval()
    event(
        write,
        request_id,
        "progress",
        stage="tracing",
        progress=0.48,
        message="Tracing a fixed-shape decoder",
    )
    with torch.no_grad():
        reference = wrapper(*torch_feed).float().cpu().numpy()
        traced = torch.jit.trace(wrapper, torch_feed, strict=False, check_trace=False)

    event(
        write,
        request_id,
        "progress",
        stage="converting",
        progress=0.58,
        message="Converting to a Core ML ML Program",
    )
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, sequence_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, sequence_length), dtype=np.int32),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        import shutil

        shutil.rmtree(output)
    converted.save(output)
    del converted, traced, wrapper, model
    gc.collect()

    event(
        write,
        request_id,
        "progress",
        stage="validating",
        progress=0.82,
        message="Checking Core ML output against the source model",
    )
    coreml_model = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
    prediction = coreml_model.predict(feed)["logits"]
    cosine = _cosine(reference, prediction, np)
    maximum_error = float(
        np.max(np.abs(reference.astype(np.float32) - prediction.astype(np.float32)))
    )
    source_token = int(reference.argmax(-1)[0])
    coreml_token = int(prediction.argmax(-1)[0])
    parity = {
        "status": "passed" if cosine >= 0.999 and source_token == coreml_token else "failed",
        "cosineSimilarity": cosine,
        "maximumAbsoluteError": maximum_error,
        "topTokenMatches": source_token == coreml_token,
        "sourceToken": source_token,
        "coremlToken": coreml_token,
    }
    del coreml_model
    gc.collect()
    return {"parity": parity, "position": position}


class CoreMLRuntime:
    def __init__(self, package: Path, source: Path, sequence_length: int, policy: str) -> None:
        ct, np, _, auto = _imports()
        _, AutoTokenizer = auto
        self.ct = ct
        self.np = np
        self.sequence_length = sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.tokenizer.padding_side = "right"
        unit_name = POLICIES.get(policy)
        if not unit_name:
            raise ValueError(f"Unknown compute policy: {policy}")
        self.model = ct.models.MLModel(
            str(package), compute_units=getattr(ct.ComputeUnit, unit_name)
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
        ids = list(encoded)[-(self.sequence_length - 1) :]
        generated: list[int] = []
        durations: list[float] = []
        eos = self.tokenizer.eos_token_id
        rng = self.np.random.default_rng(941731)
        for _ in range(min(max_new_tokens, self.sequence_length - len(ids))):
            current_text = self.tokenizer.decode(ids, skip_special_tokens=False)
            feed, _ = _inputs(self.tokenizer, current_text, self.sequence_length, self.np)
            started = time.perf_counter_ns()
            logits = self.model.predict(feed)["logits"].astype(self.np.float32)[0]
            durations.append((time.perf_counter_ns() - started) / 1_000_000)
            if temperature <= 0:
                token_id = int(logits.argmax())
            else:
                scaled = logits / max(temperature, 1e-4)
                scaled -= scaled.max()
                probabilities = self.np.exp(scaled)
                probabilities /= probabilities.sum()
                token_id = int(rng.choice(len(probabilities), p=probabilities))
            if eos is not None and token_id == eos:
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

        compiled = compile_model(str(package))
        unit_name = POLICIES[policy]
        plan = MLComputePlan.load_from_path(
            compiled, compute_units=getattr(ct.ComputeUnit, unit_name)
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
        try:
            runtime = CoreMLRuntime(package, source, sequence_length, policy)
            runtime.generate(messages, 1, 0)
            result = runtime.generate(messages, 8, 0)
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
    return {
        "modelId": model_id,
        "prompt": prompt,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policies": reports,
    }
