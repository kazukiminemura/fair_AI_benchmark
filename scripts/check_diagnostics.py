"""Run with python -m scripts.check_diagnostics."""

import math
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fairbench import cli
from fairbench.backends.base import CompiledModel
from fairbench.backends.openvino_backend import OpenVinoBackend
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.models import build_precision_gemm
from fairbench.report import format_execution_precision
from fairbench.runner import _aggregate, _grade
from fairbench.stats import summarise, summarise_repeats
from fairbench.system import PowerInfo


def main() -> None:
    for values in ([], [float("nan"), float("inf"), -float("inf"), None]):
        empty = summarise(values)
        assert empty.count == 0 and empty.outliers == 0 and type(empty.outliers) is int
        assert all(math.isnan(value) for key, value in empty.to_dict().items() if key not in ("count", "outliers"))
    for value in (0.0, 2.5):
        single = summarise([value])
        assert single.count == 1 and single.outliers == 0 and type(single.outliers) is int
        assert all(getattr(single, key) == value for key in ("mean_ms", "median_ms", "p90_ms", "p99_ms", "min_ms", "max_ms"))
        assert single.stdev_ms == single.mad_ms == single.iqr_ms == 0
        if value:
            assert single.cv == single.cv_trimmed == 0
        else:
            assert math.isnan(single.cv) and math.isnan(single.cv_trimmed)
    print("Empty/single-sample statistics passed: integer outliers, unavailable floats, unchanged singleton boundaries.")
    assert math.isnan(summarise_repeats([10], [1]).spread_pct)
    assert summarise_repeats([10, 10, 10], [1, 1, 1]).spread_pct == 0
    args = cli.build_parser().parse_args(["run", "--quick"])
    assert cli._config_from_args(args).repeats == 3
    cell = {"ok": True, "target_id": "test", "kind": "gpu", "vendor": "unknown", "workload": "precision_gemm_fp16",
            "units_per_s": 10, "model": {"flops": 100}, "latency": {"median_ms": 1, "cv_trimmed": 0.10},
            "session": {"compute_precisions": ["f32"]}, "iterations": 100,
            "timing_breakdown": {"available": True, "method": "test", "h2d_ms": 1, "compute_ms": 2, "d2h_ms": 3}}
    single = _aggregate([cell])[0]
    assert single["repeat_spread_pct"] is None
    triple = _aggregate([cell, deepcopy(cell), deepcopy(cell)])[0]
    assert triple["timing_breakdown"]["repeats_measured"] == 3
    assert triple["timing_breakdown"]["compute_ms"] == 2
    assert format_execution_precision(triple) == "FP32"
    unknown = deepcopy(cell)
    unknown["session"] = {}
    mixed = _aggregate([cell, unknown])[0]
    assert not mixed["compute_precision_complete"]
    assert "partially unverified" in format_execution_precision(mixed)
    fingerprint = SimpleNamespace(power=PowerInfo())
    for row, expected in ((single, "skip"), (triple, "pass"), ({**triple, "repeats_ok": 1}, "skip")):
        report = {"results": [row], "cells": [cell], "power_sources": []}
        checks = _grade(report, fingerprint, RunConfig())["checks"]
        assert next(check for check in checks if check["name"] == "Repeatability")["status"] == expected
        if expected == "skip":
            assert not any("repeats agree" in check["detail"] for check in checks)

    backend = OrtBackend()
    binding = MagicMock()
    output = np.ones((2, 2), dtype=np.float32)
    binding.get_outputs.return_value = [MagicMock()]
    binding.copy_outputs_to_cpu.return_value = [output]
    session = MagicMock()
    session.io_binding.return_value = binding
    session.run.return_value = [output]
    compiled = CompiledModel(session, 0, "fp16", ["y"])
    target = Target("gpu", "ort", "DmlExecutionProvider", "gpu", "unknown", "GPU")
    event = {"cat": "Node", "name": "DmlFusedNode_0_0_kernel_time", "args": {
        "provider": target.device, "op_name": "DmlFusedNode_0_0", "input_type_shape": [{"float": [512, 512]}]}}
    for precision, dtype in (("fp16", "float16"), ("int8", "float")):
        evidence = backend._gpu_precision_evidence(build_precision_gemm(precision, 512), target, [event])
        assert evidence["available"] and evidence["tensor_precisions"] == [dtype]
        assert evidence["arithmetic_precision"] == "unverified"
        assert evidence["accumulator_precision"] == "unverified"
        assert evidence["evidence"] == "profiled_fused_graph"
    cpu_event = deepcopy(event)
    cpu_event["args"]["provider"] = "CPUExecutionProvider"
    built = build_precision_gemm("fp16", 512)
    for events in ([], [cpu_event], [event, cpu_event], [event, event]):
        assert not backend._gpu_precision_evidence(built, target, events)["available"]
    direct = deepcopy(event)
    direct["args"].update(op_name="MatMul", input_type_shape=[{"float16": [512, 512]}, {"float16": [512, 512]}])
    assert backend._gpu_precision_evidence(built, target, [direct])["tensor_precisions"] == ["float16"]
    direct["args"]["input_type_shape"] = [{"float16": [512, 512]}]
    assert not backend._gpu_precision_evidence(built, target, [direct])["available"]
    gpu_cell = deepcopy(cell)
    gpu_cell["session"] = {"gpu_precision": evidence}
    gpu_triple = _aggregate([gpu_cell, deepcopy(gpu_cell), deepcopy(gpu_cell)])[0]
    assert gpu_triple["gpu_precision"]["repeats_measured"] == 3
    assert "FP32 (GPU-assigned graph operands" in format_execution_precision(gpu_triple)
    missing = deepcopy(gpu_cell)
    missing["session"] = {}
    assert not _aggregate([gpu_cell, missing])[0]["gpu_precision"]["available"]
    with patch.object(backend.ort.OrtValue, "ortvalue_from_numpy", return_value=MagicMock()):
        timing = backend.diagnose(compiled, SimpleNamespace(precision="fp16"), target, RunConfig(), {"x": output})
    assert timing["available"] and timing["samples"] >= 10
    assert binding.synchronize_inputs.call_count == timing["samples"]
    assert binding.synchronize_outputs.call_count >= timing["samples"]
    assert compiled.meta["compute_precisions"] == []
    binding.copy_outputs_to_cpu.return_value = [output * 2]
    with patch.object(backend.ort.OrtValue, "ortvalue_from_numpy", return_value=MagicMock()):
        timing = backend.diagnose(compiled, SimpleNamespace(precision="fp16"), target, RunConfig(), {"x": output})
    assert not timing["available"] and "does not match" in timing["reason"]
    binding.copy_outputs_to_cpu.return_value = [output]
    for kind, provider in (("cpu", "CPUExecutionProvider"), ("npu", "VitisAIExecutionProvider")):
        phase_target = Target(kind, "ort", provider, kind, "unknown", kind.upper())
        with patch.object(backend, "_profile_events", return_value=[]), patch.object(backend.ort.OrtValue, "ortvalue_from_numpy", return_value=MagicMock()):
            timing = backend.diagnose(compiled, built, phase_target, RunConfig(), {"x": output})
        assert timing["available"] and timing["transfer_scope"] == "host_buffers"
        assert timing["internal_transfers_in_compute"] == (kind == "npu")
    ov_backend = OpenVinoBackend.__new__(OpenVinoBackend)
    request = MagicMock()
    input_buffer = np.zeros_like(output)
    request.get_tensor.side_effect = lambda name: SimpleNamespace(data=input_buffer if name == "x" else output)
    session.create_infer_request.return_value = request
    for kind in ("cpu", "npu"):
        phase_target = Target(kind, "openvino", kind.upper(), kind, "intel", kind.upper())
        with patch.object(ov_backend, "run", return_value=[output]):
            timing = ov_backend.diagnose(compiled, built, phase_target, RunConfig(), {"x": output})
        assert timing["available"] and timing["transfer_scope"] == "host_buffers"
        assert timing["internal_transfers_in_compute"] == (kind != "cpu")
        assert np.array_equal(input_buffer, output)
        assert all(len(values) == timing["samples"] for values in timing["phase_samples_ms"].values())
    with patch.object(ov_backend, "run", return_value=[output * 2]):
        assert not ov_backend.diagnose(compiled, built, phase_target, RunConfig(), {"x": output})["available"]
    print("Repeatability, precision evidence, synchronized GPU phases, output validation and repeat aggregation passed.")


if __name__ == "__main__":
    main()
