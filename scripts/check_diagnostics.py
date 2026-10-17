"""Run with python -m scripts.check_diagnostics."""

import math
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fairbench import cli
from fairbench.backends.base import CompiledModel
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.report import format_execution_precision
from fairbench.runner import _aggregate, _grade
from fairbench.stats import summarise_repeats
from fairbench.system import PowerInfo


def main() -> None:
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
    print("Repeatability, precision evidence, synchronized GPU phases, output validation and repeat aggregation passed.")


if __name__ == "__main__":
    main()
