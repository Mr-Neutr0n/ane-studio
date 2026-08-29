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
