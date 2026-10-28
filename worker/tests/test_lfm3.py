from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pytest

from ane_studio_worker import coreml_backend


@pytest.fixture
def tiny_lfm3():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from ane_studio_worker.vendor.lfm3 import register

    register()
    register()  # Worker requests can register the same local adapter again.
    config = transformers.AutoConfig.for_model(
        "lfm3",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["conv", "full_attention", "sliding_attention"],
        sliding_window=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = transformers.AutoModelForCausalLM.from_config(
            config, trust_remote_code=False
        ).eval()
    return torch, transformers, model


def test_local_lfm3_config_preserves_layer_labels_and_nested_rope(tiny_lfm3):
    _, _, model = tiny_lfm3
    assert model.config.layer_types == ["conv", "full_attention", "sliding_attention"]
    assert set(model.config.rope_parameters) == {"full_attention", "sliding_attention"}
    assert model.config.rope_parameters["full_attention"]["partial_rotary_factor"] == 0.5
    assert model.config.rope_parameters["full_attention"]["rope_theta"] == 1_000_000.0
    assert model.config.rope_parameters["sliding_attention"]["rope_theta"] == 10_000.0


def test_registered_lfm3_uses_local_classes_despite_remote_auto_map(tiny_lfm3, tmp_path: Path):
    torch, transformers, model = tiny_lfm3
    model.save_pretrained(tmp_path)
    _add_remote_auto_map(tmp_path)
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
    ).eval()
    assert type(reloaded).__module__ == "ane_studio_worker.vendor.lfm3.modeling_lfm3"
    inputs = torch.tensor([[1, 3, 5]])
    with torch.no_grad():
        expected = model(input_ids=inputs, use_cache=False).logits
        actual = reloaded(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("length", [1, 5, 16])
def test_lfm3_fixed_masks_preserve_valid_token_outputs(tiny_lfm3, length: int):
    torch, _, model = tiny_lfm3
    from ane_studio_worker.vendor.lfm3 import modeling_lfm3

    original = modeling_lfm3.create_masks_by_layer_type
    model = model.to(dtype=torch.float16)
    ids = torch.zeros((1, 16), dtype=torch.long)
    ids[0, :length] = torch.arange(1, length + 1)
    mask = torch.zeros_like(ids)
    mask[0, :length] = 1
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with coreml_backend._patch_trace_gaps("Lfm3ForCausalLM", 16, torch):
            actual = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual[:, :length], expected[:, :length], atol=0, rtol=0)
    assert modeling_lfm3.create_masks_by_layer_type is original


def test_lfm3_mask_patch_restores_source_after_failure(tiny_lfm3):
    torch, _, _ = tiny_lfm3
    from ane_studio_worker.vendor.lfm3 import modeling_lfm3

    original = modeling_lfm3.create_masks_by_layer_type
    with (
        pytest.raises(RuntimeError, match="trace failed"),
        coreml_backend._patch_trace_gaps("Lfm3ForCausalLM", 16, torch),
    ):
        assert modeling_lfm3.create_masks_by_layer_type is not original
        raise RuntimeError("trace failed")
    assert modeling_lfm3.create_masks_by_layer_type is original


def test_lfm3_mask_patch_rejects_graph_beyond_sliding_window(tiny_lfm3):
    torch, _, model = tiny_lfm3
    from ane_studio_worker.vendor.lfm3 import modeling_lfm3

    original = modeling_lfm3.create_masks_by_layer_type
    with (
        pytest.raises(ValueError, match="must fit its sliding attention window"),
        coreml_backend._patch_trace_gaps("Lfm3ForCausalLM", 32, torch),
    ):
        modeling_lfm3.create_masks_by_layer_type(
            config=model.config,
            inputs_embeds=torch.zeros((1, 32, 32)),
            attention_mask=torch.ones((1, 32)),
        )
    assert modeling_lfm3.create_masks_by_layer_type is original


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Real Core ML parity requires an Apple Silicon Mac.",
)
def test_production_lfm3_conversion_passes_four_real_cpu_checks(tiny_lfm3, tmp_path: Path):
    pytest.importorskip("coremltools")
    tokenizers = pytest.importorskip("tokenizers")
    _, transformers, model = tiny_lfm3
    source = tmp_path / "source"
    model.save_pretrained(source)
    _add_remote_auto_map(source)
    vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    words = "The capital of France is opposite hot Hello A short sentence about the sky and sea ."
    for word in words.split():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    for index in range(len(vocabulary), 64):
        vocabulary[f"unused{index}"] = index
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    fast.save_pretrained(source)
    result = coreml_backend.convert_model(
        "test/tiny-lfm3",
        source,
        tmp_path / "artifacts/model.mlpackage",
        "Lfm3ForCausalLM",
        16,
        "local-regression",
        lambda _event: None,
    )
    parity = result["parity"]
    assert parity["status"] == "passed"
    assert parity["reference"] == "unpatched-transformers-fp32"
    assert parity["computePrecision"] == "fp32"
    assert {check["position"] for check in parity["checks"]} == {0, 4, 15}
    assert len(parity["checks"]) == 4
    assert all(check["status"] == "passed" for check in parity["checks"])
    assert not list((tmp_path / "artifacts").glob(".ane-studio-compile-*"))


def _add_remote_auto_map(source: Path) -> None:
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["auto_map"] = {
        "AutoConfig": "remote_config.Lfm3Config",
        "AutoModelForCausalLM": "remote_model.Lfm3ForCausalLM",
    }
    path.write_text(json.dumps(config))
    for name in ("remote_config.py", "remote_model.py"):
        (source / name).write_text('raise RuntimeError("remote model code must not execute")\n')
