from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ane_studio_worker import coreml_backend


def test_managed_compile_directory_is_removed_and_restores_tempdir(tmp_path: Path) -> None:
    previous = tempfile.tempdir
    parent = tmp_path / "artifacts"

    with coreml_backend._managed_compile_directory(parent) as managed:
        assert Path(tempfile.gettempdir()) == managed
        (managed / "model.mlmodelc").mkdir()
        (managed / "model.mlmodelc/model.bin").write_bytes(b"compiled")

    assert tempfile.tempdir == previous
    assert not managed.exists()


def test_managed_compile_directory_cleans_up_after_failure(tmp_path: Path) -> None:
    with (
        pytest.raises(RuntimeError, match="compile failed"),
        coreml_backend._managed_compile_directory(tmp_path) as managed,
    ):
        (managed / "partial.mlmodelc").mkdir()
        raise RuntimeError("compile failed")

    assert not managed.exists()


def test_planned_placement_removes_compiler_output_after_failure(
    monkeypatch, tmp_path: Path
) -> None:
    compiled = tmp_path / "system-temp" / "model.mlmodelc"

    compute_device = ModuleType("coremltools.models.compute_device")
    compute_device.MLCPUComputeDevice = type("MLCPUComputeDevice", (), {})
    compute_device.MLGPUComputeDevice = type("MLGPUComputeDevice", (), {})
    compute_device.MLNeuralEngineComputeDevice = type("MLNeuralEngineComputeDevice", (), {})

    compute_plan = ModuleType("coremltools.models.compute_plan")

    class FailingPlan:
        @staticmethod
        def load_from_path(*args, **kwargs):
            raise RuntimeError("plan failed")

    compute_plan.MLComputePlan = FailingPlan
    utils = ModuleType("coremltools.models.utils")

    def compile_model(_package: str) -> str:
        compiled.mkdir(parents=True)
        (compiled / "model.bin").write_bytes(b"compiled")
        return str(compiled)

    utils.compile_model = compile_model
    monkeypatch.setitem(sys.modules, compute_device.__name__, compute_device)
    monkeypatch.setitem(sys.modules, compute_plan.__name__, compute_plan)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    fake_ct = SimpleNamespace(ComputeUnit=SimpleNamespace(CPU_AND_NE="cpu-and-neural-engine"))
    monkeypatch.setattr(
        coreml_backend,
        "_imports",
        lambda: (fake_ct, None, None, None),
    )

    result = coreml_backend.planned_placement(tmp_path / "source.mlpackage", "cpuAndNeuralEngine")

    assert "plan failed" in result["note"]
    assert not compiled.exists()


@pytest.fixture
def np():
    return pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "family,architecture",
    [
        ("qwen2", "Qwen2ForCausalLM"),
        ("llama", "LlamaForCausalLM"),
    ],
)
def test_trace_mask_patch_is_scoped_even_on_failure(monkeypatch, family, architecture) -> None:
    name = f"transformers.models.{family}.modeling_{family}"
    modeling = ModuleType(name)

    def original(**kwargs):
        return "original-mask"

    modeling.create_causal_mask = original
    monkeypatch.setitem(sys.modules, name, modeling)
    fake_torch = SimpleNamespace(float32="fp32", zeros=lambda *args, **kwargs: "fixed-mask")

    with (
        pytest.raises(RuntimeError, match="trace failed"),
        coreml_backend._patch_trace_gaps(architecture, 1, fake_torch),
    ):
        assert modeling.create_causal_mask() == "fixed-mask"
        raise RuntimeError("trace failed")

    assert modeling.create_causal_mask is original


def test_unknown_trace_adapter_is_rejected() -> None:
    with (
        pytest.raises(ValueError, match="No trace adapter"),
        coreml_backend._patch_trace_gaps("UnknownForCausalLM", 64, None),
    ):
        pass


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("linear", True),
        ("matmul", True),
        ("square", False),
        ("pow", False),
        ("reduce_mean", False),
        ("rsqrt", False),
        ("add", False),
        ("mul", False),
        ("softmax", False),
        ("sin", False),
        ("cos", False),
    ],
)
def test_mixed_precision_preserves_sensitive_operations(operation, expected) -> None:
    assert coreml_backend._fp16_matrix_operation(SimpleNamespace(op_type=operation)) is expected


@pytest.mark.parametrize(
    "right",
    [
        [0.0, 0.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [],
    ],
)
def test_invalid_logits_fail_closed(np, right) -> None:
    reference = np.asarray([[1.0, 2.0]])
    prediction = np.asarray([right])
    parity = coreml_backend._parity_metrics(reference, prediction, np)
    assert parity["status"] == "failed"
    assert parity["topTokenMatches"] is False
    assert parity["cosineSimilarity"] == 0
    assert "maximumAbsoluteError" not in parity
    assert "sourceToken" not in parity
    assert "coremlToken" not in parity


def test_equal_zero_logits_do_not_pass_parity(np) -> None:
    logits = np.zeros((1, 10))
    parity = coreml_backend._parity_metrics(logits, logits, np)
    assert parity["status"] == "failed"
    assert parity["topTokenMatches"] is False


def test_parity_requires_both_cosine_and_top_token(np) -> None:
    reference = np.asarray([[1.001, 1.0]])
    prediction = np.asarray([[1.0, 1.001]])
    parity = coreml_backend._parity_metrics(reference, prediction, np)
    assert parity["cosineSimilarity"] >= 0.999
    assert parity["status"] == "failed"
    assert parity["topTokenMatches"] is False

    parity = coreml_backend._parity_metrics(reference, np.asarray([[1.001, -1.0]]), np)
    assert parity["topTokenMatches"] is True
    assert parity["status"] == "failed"


def test_passing_parity_keeps_exact_threshold(np) -> None:
    logits = np.asarray([[1.0, 2.0]])
    parity = coreml_backend._parity_metrics(logits, logits.copy(), np)
    assert parity["status"] == "passed"
    assert parity["cosineSimilarity"] >= 0.999
    assert parity["sourceToken"] == parity["coremlToken"] == 1


class NonRoundTripTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    bos_token_id = None

    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return "<template>user: hello assistant:"

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        # Decoding is deliberately not an inverse of encoding.
        return {"input_ids": self.ids if text.startswith("<template>") else [99]}

    def decode(self, ids, **kwargs):
        return "decoded text is not the original token stream"


def test_generate_preserves_token_ids_without_round_trip(np) -> None:
    runtime = coreml_backend.CoreMLRuntime.__new__(coreml_backend.CoreMLRuntime)
    runtime.np = np
    runtime.sequence_length = 8
    runtime.tokenizer = NonRoundTripTokenizer([10, 20])
    feeds = []

    def predict(feed):
        feeds.append(feed)
        return {"logits": np.asarray([[0.0, 0.0, 1.0]])}

    runtime.model = SimpleNamespace(predict=predict)
    result = runtime.generate([{"role": "user", "content": "hello"}], 2, 0)

    assert len(runtime.tokenizer.calls) == 1
    assert feeds[0]["input_ids"].tolist() == [[10, 20, 0, 0, 0, 0, 0, 0]]
    assert feeds[1]["input_ids"].tolist() == [[10, 20, 2, 0, 0, 0, 0, 0]]
    assert result["tokens"] == 2


def test_context_exhaustion_is_explicit_and_does_not_drop_chat_template(np) -> None:
    runtime = coreml_backend.CoreMLRuntime.__new__(coreml_backend.CoreMLRuntime)
    runtime.np = np
    runtime.sequence_length = 8
    runtime.tokenizer = NonRoundTripTokenizer(list(range(8)))
    runtime.model = SimpleNamespace(predict=lambda _: pytest.fail("must not run exhausted prompt"))

    with pytest.raises(ValueError, match="uses 8 tokens.*8-token context"):
        runtime.generate([{"role": "user", "content": "hello"}], 2, 0)

    assert runtime.tokenizer.calls == ["<template>user: hello assistant:"]


def test_token_inputs_right_pad_and_preserve_selected_position(np) -> None:
    tokenizer = NonRoundTripTokenizer([])
    feed, position = coreml_backend._token_inputs(tokenizer, [50, 51, 52], 8, np)
    assert position == 2
    assert feed["position"].tolist() == [2]
    assert feed["input_ids"].tolist() == [[50, 51, 52, 0, 0, 0, 0, 0]]
    assert feed["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]


def test_empty_input_preserves_bos_token_zero(np) -> None:
    tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=9, pad_token_id=None)
    feed, position = coreml_backend._token_inputs(tokenizer, [], 4, np)
    assert feed["input_ids"].tolist() == [[0, 9, 9, 9]]
    assert position == 0


@pytest.mark.parametrize("logits", [[[0.0, 0.0]], [[float("nan"), 1.0]]])
def test_generation_rejects_invalid_or_zero_policy_logits(np, logits) -> None:
    runtime = coreml_backend.CoreMLRuntime.__new__(coreml_backend.CoreMLRuntime)
    runtime.np = np
    runtime.sequence_length = 8
    runtime.tokenizer = NonRoundTripTokenizer([1])
    runtime.model = SimpleNamespace(predict=lambda _: {"logits": np.asarray(logits)})
    with pytest.raises(RuntimeError, match="invalid or zero logits"):
        runtime.generate([{"role": "user", "content": "hello"}], 2, 0)


def test_generation_reports_context_limit_without_silent_prompt_truncation(np) -> None:
    runtime = coreml_backend.CoreMLRuntime.__new__(coreml_backend.CoreMLRuntime)
    runtime.np = np
    runtime.sequence_length = 8
    runtime.tokenizer = NonRoundTripTokenizer([10, 11, 12, 13, 14, 15, 16])
    runtime.model = SimpleNamespace(predict=lambda _: {"logits": np.asarray([[0.0, 1.0]])})
    result = runtime.generate([{"role": "user", "content": "hello"}], 3, 0)
    assert result["tokens"] == 1
    assert result["stopReason"] == "contextLimit"
    assert result["contextTokens"] == result["sequenceLength"] == 8


def test_generation_reports_eos_even_without_visible_tokens(np) -> None:
    runtime = coreml_backend.CoreMLRuntime.__new__(coreml_backend.CoreMLRuntime)
    runtime.np = np
    runtime.sequence_length = 8
    runtime.tokenizer = NonRoundTripTokenizer([1])
    logits = np.zeros((1, 10))
    logits[0, 9] = 1
    runtime.model = SimpleNamespace(predict=lambda _: {"logits": logits})
    result = runtime.generate([{"role": "user", "content": "hello"}], 3, 0)
    assert result["tokens"] == 0
    assert result["stopReason"] == "eos"
    assert result["contextTokens"] == 1


def test_qwen_uses_fp32_to_preserve_position_zero_parity() -> None:
    fake_ct = SimpleNamespace(
        precision=SimpleNamespace(FLOAT32="fp32", FLOAT16="fp16"),
        transform=SimpleNamespace(FP16ComputePrecision=lambda **kwargs: kwargs),
    )
    assert coreml_backend._compute_precision("Qwen2ForCausalLM", fake_ct) == "fp32"
    assert coreml_backend._compute_precision("Lfm2ForCausalLM", fake_ct) == "fp16"
    assert coreml_backend._compute_precision("LlamaForCausalLM", fake_ct) == {
        "op_selector": coreml_backend._fp16_matrix_operation
    }
