"""Regression checks for grading inputs and failure/precision disclosures."""

from copy import deepcopy

from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.models import build_precision_gemm
from fairbench.report import format_execution_precision
from fairbench.runner import _aggregate, _grade, _score
from fairbench.system import PowerInfo, SystemFingerprint


def cell(target="cpu", workload="precision_gemm_fp16", cv=0.02, count=100, units=100.0):
    return {"ok": True, "target_id": target, "kind": "cpu" if target == "cpu" else "gpu",
            "vendor": "unknown", "backend": "ort", "workload": workload, "units_per_s": units,
            "latency": {"median_ms": 1000 / units, "p90_ms": 1100 / units, "cv_trimmed": cv, "count": count},
            "iterations": count, "model": {"flops": 1000},
            "correctness": {"checked": True, "valid": True, "method": "deterministic_element_samples_v2"}}


def main() -> None:
    fp = SystemFingerprint(power=PowerInfo())
    cfg = RunConfig(repeats=3)
    for cv, expected in ((0.03, "pass"), (0.15, "warn"), (0.30, "warn")):
        cells = [cell(units=100), cell(cv=cv, count=90, units=99), cell(units=101)]
        rows = _aggregate(cells)
        before = deepcopy(rows)
        report = {"results": rows, "cells": cells, "config": cfg.to_dict()}
        grade = _grade(report, fp, cfg)
        timing = next(check for check in grade["checks"] if check["name"] == "Timing stability")
        assert timing["status"] == expected and f"{cv:.1%}" in timing["detail"], timing
        assert "over 90 iterations in the worst repeat" in timing["detail"]
        assert rows[0]["latency"] == before[0]["latency"] == cells[2]["latency"]
        assert rows[0]["units_per_s"] == before[0]["units_per_s"] == 100

    crash = "worker crashed: ValueError: invalid GEMM search size"
    cells = []
    for workload, gpu_repeats in (("precision_gemm_fp16", 1), ("precision_gemm_int8", 2)):
        cells.extend(cell(workload=workload, units=10) for _ in range(3))
        cells.extend(cell("gpu", workload, units=20) for _ in range(gpu_repeats))
        cells.extend({"ok": False, "target_id": "gpu", "workload": workload, "error": crash}
                     for _ in range(3 - gpu_repeats))
    report = {"results": _aggregate(cells), "cells": cells, "config": cfg.to_dict(), "warnings": ["existing warning"]}
    _grade(report, fp, cfg)
    assert report["warnings"] == ["existing warning", f"3 failed cell(s): {crash}"]
    _grade(report, fp, cfg)
    assert len(report["warnings"]) == 2, "Repeated grading duplicated the warning"
    index = {entry["target_id"]: entry for entry in _score(report)["ai_index"]}
    assert (index["cpu"]["repeats_ok"], index["cpu"]["repeats_requested"]) == (6, 6)
    assert (index["gpu"]["repeats_ok"], index["gpu"]["repeats_requested"]) == (3, 6)
    assert index["cpu"]["ai_index"] == 1 and index["gpu"]["ai_index"] == 2

    target = Target("gpu", "ort", "DmlExecutionProvider", "gpu", "unknown", "GPU")
    event = {"cat": "Node", "name": "DmlFusedNode_0_kernel_time", "args": {
        "provider": target.device, "op_name": "DmlFusedNode_0", "input_type_shape": [{"float": [512, 512]}]}}
    evidence = OrtBackend._gpu_precision_evidence(build_precision_gemm("fp16", 512), target, [event])
    assert evidence["evidence"] == "profiled_fused_graph" and evidence["tensor_precisions"] == ["float16"]
    assert "pre-optimization ONNX" in evidence["note"] and "not the optimized/fused graph" in evidence["note"]
    assert "pre-optimization ONNX" in format_execution_precision({"gpu_precision": evidence})
    print("Grade disclosure checks passed: worst-repeat populations, unchanged public values, grouped/idempotent failures, index repeat totals and precision sources.")


if __name__ == "__main__":
    main()
