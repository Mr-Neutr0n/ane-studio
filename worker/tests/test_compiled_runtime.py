from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ane_studio_worker import coreml_backend


def test_compiled_runtime_owns_destination_until_model_is_released(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(coreml_backend.tempfile, "tempdir", str(tmp_path))
    utils = ModuleType("coremltools.models.utils")
    destinations = []

    def compile_model(package, *, destination_path):
        assert package == str(tmp_path / "source.mlpackage")
        destination = Path(destination_path)
        assert destination.name == "model.mlmodelc"
        assert destination.parent.parent == tmp_path
        assert destination.parent.name.startswith("ane-studio-runtime-")
        destination.mkdir()
        (destination / "weights.bin").write_bytes(b"native compiled weights")
        destinations.append(destination)
        return str(destination)

    class CompiledModel:
        def __init__(self, path, *, compute_units):
            assert compute_units == "cpu-and-neural-engine"
            self.path = Path(path)
            assert (self.path / "weights.bin").is_file()

    utils.compile_model = compile_model
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    ct = SimpleNamespace(models=SimpleNamespace(CompiledMLModel=CompiledModel))
    model = coreml_backend._load_compiled_model(
        tmp_path / "source.mlpackage", "cpu-and-neural-engine", ct
    )
    reference = weakref.ref(model)
    gc.collect()
    assert reference() is model
    assert destinations[0].is_dir()
    del model
    gc.collect()
    assert reference() is None
    assert not destinations[0].parent.exists()


@pytest.mark.parametrize("failure_stage", ["compile", "load"])
def test_compiled_runtime_failure_removes_only_owned_destination(
    monkeypatch, tmp_path: Path, failure_stage: str
):
    monkeypatch.setattr(coreml_backend.tempfile, "tempdir", str(tmp_path))
    unrelated = tmp_path / "unrelated.mlmodelc"
    unrelated.mkdir()
    marker = unrelated / "weights.bin"
    marker.write_bytes(b"keep existing model")
    utils = ModuleType("coremltools.models.utils")
    destinations = []

    def compile_model(_package, *, destination_path):
        destination = Path(destination_path)
        destination.mkdir()
        (destination / "partial.bin").write_bytes(b"partial compiled weights")
        destinations.append(destination)
        if failure_stage == "compile":
            raise RuntimeError("compile failed")
        return str(destination)

    def load_model(_path, *, compute_units):
        raise RuntimeError("load failed")

    utils.compile_model = compile_model
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    ct = SimpleNamespace(models=SimpleNamespace(CompiledMLModel=load_model))
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        coreml_backend._load_compiled_model(tmp_path / "source.mlpackage", "cpu-only", ct)
    assert not destinations[0].parent.exists()
    assert marker.read_bytes() == b"keep existing model"
