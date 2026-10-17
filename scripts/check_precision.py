"""Run with python -m scripts.check_precision."""

from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort

from fairbench import models
from fairbench.backends.ort_backend import OrtBackend
from fairbench.cli import _config_from_args, build_parser
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.runner import DEFAULT_WORKLOADS, precision_of, resolve_workloads


def main() -> None:
    assert RunConfig().gemm_size == _config_from_args(build_parser().parse_args(["run"])).gemm_size == 2048
    assert _config_from_args(build_parser().parse_args(["run", "--gemm-size", "512"])).gemm_size == 512
    for size in (512, 2048):
        sized = models.build_one("precision_gemm_fp16", size)
        assert sized.gemm_shape == (size, size, size)
        assert sized.flops == models.OPERATION_CONVENTION["operations_per_mac"] * size ** 3
    reference = models.build_precision_gemm()
    assert reference.gemm_shape == (512, 1024, 1024)
    inputs = reference.make_inputs()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    expected = ort.InferenceSession(reference.serialise(), options, providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    for precision, dtype in (("fp16", onnx.TensorProto.FLOAT16), ("int8", onnx.TensorProto.INT8)):
        built = models.build_precision_gemm(precision)
        onnx.checker.check_model(built.model, full_check=True)
        assert built.sha256 == models.build_one(built.key).sha256
        assert precision_of(built.key) == built.precision == precision
        if precision in ("fp16", "int8"):
            assert built.key in DEFAULT_WORKLOADS
            assert built.key in resolve_workloads(RunConfig(suite="precision"))
        assert next(weight for weight in built.model.graph.initializer if weight.name == "weight").data_type == dtype
        assert built.flops == 2 * 512 * 1024 * 1024
        if precision in ("fp16", "int8"):
            session = ort.InferenceSession(built.serialise(), options, providers=["CPUExecutionProvider"])
            actual = session.run(None, inputs)[0]
            error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert np.isfinite(actual).all() and error < (0.001 if precision == "fp16" else 0.03), error
            if precision == "int8":
                assert "QLinearMatMul" not in [node.op_type for node in built.model.graph.node]
                cpu = Target("cpu", "ort", "CPUExecutionProvider", "cpu", "intel", "CPU")
                backend = OrtBackend()
                compiled = backend.compile(built, cpu, RunConfig(), "int8")
                backend.diagnose(compiled, built, cpu, RunConfig(), inputs)
                assert "MatMulIntegerToFloat" in compiled.meta["compute_kernels"], compiled.meta

    backend = OrtBackend()
    assert resolve_workloads(RunConfig()) == ["precision_gemm_fp16", "precision_gemm_int8"]
    target = Target("test-gpu", "ort", "DmlExecutionProvider", "gpu", "unknown", "GPU")
    session = MagicMock()
    session.get_providers.return_value = ["DmlExecutionProvider"]
    session.get_outputs.return_value = [MagicMock(name="output")]
    with patch.object(backend.ort, "InferenceSession", return_value=session) as create:
        backend.compile(models.build_precision_gemm("fp16"), target, RunConfig(), "fp16")
        assert create.call_args.kwargs["providers"] == ["DmlExecutionProvider"]
        assert create.call_args.kwargs["sess_options"].get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
    for precision in ("fp8", "int4"):
        assert f"precision_gemm_{precision}" not in models.BASE_BUILDERS
        try:
            models.build_one(f"precision_gemm_{precision}")
        except ValueError as error:
            assert f"Native {precision.upper()}" in str(error)
        else:
            raise AssertionError("Unverified low precision arithmetic was offered as throughput")
    print("Precision graphs, deterministic hashes, full output accuracy, scheduling defaults and strict GPU fallback passed.")


if __name__ == "__main__":
    main()
