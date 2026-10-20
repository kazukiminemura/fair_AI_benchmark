"""Run with python -m scripts.check_tensorrt; no NVIDIA SDK needed."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from fairbench import devices, interp
from fairbench.backends.base import BackendError, CompiledModel
from fairbench.backends.tensorrt_backend import int8_evidence
from fairbench.config import RunConfig
from fairbench.report import format_execution_precision
from fairbench.runner import _aggregate
from fairbench.stats import summarise
from fairbench.system import SystemFingerprint


def main():
    # TensorRT 10.13 inspector layout observed on RTX 3060 Ti. A constant
    # weight is embedded; its operand type appears in the selected tactic.
    gemm = {"Name": "precision_matmul", "LayerType": "gemm",
            "Inputs": [{"Format/Datatype": "Int8"}], "Outputs": [{"Format/Datatype": "Int8"}],
            "TacticName": "sm80_xmma_fprop_implicit_gemm_interleaved_i8i8_i8i32_f32"}
    good = {"Layers": [gemm]}
    probe = int8_evidence(good, "engine-hash")
    assert probe["arithmetic_precision"] == "int8" and not probe["available"]
    assert not probe["runtime_execution_verified"]
    failures = [{"Layers": []}, {"Layers": [gemm, gemm]},
                {"Layers": [{"LayerType": "Reformat", "Inputs": [{"Format/Datatype": "Int8"}]}]}]
    for field, value in (("Inputs", [{"Format/Datatype": "Float"}]),
                         ("TacticName", "fp32_gemm"),
                         ("Inputs", [{"Format/Datatype": "Int8"}, {"Format/Datatype": "Float"}]),
                         ("Weights", {"Type": "Float"})):
        changed = deepcopy(gemm)
        changed[field] = value
        failures.append({"Layers": [changed]})
    for bad in failures:
        try:
            int8_evidence(bad, "bad")
        except BackendError:
            pass
        else:
            raise AssertionError(f"Unverified engine accepted: {bad}")

    RunConfig(backend="tensorrt")
    info = interp.Interpreter("trt-python", available=True, tensorrt_version="10.13.3.9",
                             tensorrt_devices=[{"device_id": 0, "name": "RTX 3060 Ti", "compute_capability": [8, 6]}])
    with patch.object(devices, "_ort_targets", return_value=([], "", [])), \
            patch.object(interp, "probe_all", return_value=[info, info]), \
            patch("fairbench.backends.vulkan_backend.probe_runtime", return_value={}):
        discovery = devices.discover(SystemFingerprint(), include_openvino=False)
    assert len(discovery.available()) == 1
    assert discovery.available()[0].python == "trt-python"
    assert discovery.available()[0].backend == "tensorrt"
    probe.update(available=True, runtime_execution_verified=True)
    cell = {"ok": True, "target_id": "trt", "workload": "precision_gemm_int8", "precision": "int8",
            "units_per_s": 10, "flops_per_s": 1000, "session": {"gpu_precision": probe},
            "latency": summarise([1., 1., 1.]).to_dict()}
    row = _aggregate([cell, deepcopy(cell)])[0]
    assert row["gpu_precision"]["arithmetic_precision"] == "int8"
    assert "execution verified" in format_execution_precision(row)
    missing = deepcopy(cell)
    missing["session"]["gpu_precision"]["runtime_execution_verified"] = False
    row = _aggregate([cell, missing])[0]
    assert row["gpu_precision"]["arithmetic_precision"] == "unverified"
    assert "execution verified" not in format_execution_precision(row)
    # Even --no-check must not publish TensorRT INT8 when correctness is unavailable.
    from fairbench import worker
    fake = SimpleNamespace(compile=lambda *args: CompiledModel(None, 0, "int8", ["y"]),
                           run=lambda *args: [])
    with patch.object(worker, "get_backend", return_value=fake), \
            patch.object(worker, "reference_fingerprint", return_value=None):
        result = worker.run_cell({"target": discovery.available()[0].to_dict(),
                                  "workload": "precision_gemm_int8", "precision": "int8",
                                  "cfg": RunConfig(gemm_size=512, check_correctness=False).to_dict()})
    assert not result["ok"] and result["phase"] == "correctness" and "flops_per_s" not in result
    print("TensorRT checks passed: reject missing/FP32 evidence, runtime gate, discovery, repeat aggregation and report.")


if __name__ == "__main__":
    main()
