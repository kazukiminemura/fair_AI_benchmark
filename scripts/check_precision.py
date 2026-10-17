"""Run with python -m scripts.check_precision."""

from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort

from fairbench import models
from fairbench.backends import BackendError
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.runner import DEFAULT_WORKLOADS, precision_of, resolve_workloads


def main() -> None:
    reference = models.build_precision_gemm()
    inputs = reference.make_inputs()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    expected = ort.InferenceSession(reference.serialise(), options, providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    for precision, dtype in (("fp16", onnx.TensorProto.FLOAT16), ("fp8", onnx.TensorProto.FLOAT8E4M3FN),
                             ("int8", onnx.TensorProto.INT8), ("int4", onnx.TensorProto.INT4)):
        built = models.build_precision_gemm(precision)
        onnx.checker.check_model(built.model, full_check=True)
        assert built.sha256 == models.build_one(built.key).sha256
        assert precision_of(built.key) == built.precision == precision
        assert built.key in DEFAULT_WORKLOADS
        assert built.key in resolve_workloads(RunConfig(suite="precision"))
        assert next(weight for weight in built.model.graph.initializer if weight.name == "weight").data_type == dtype
        assert built.flops == 2 * 512 * 1024 * 1024
        if precision in ("fp16", "int8"):
            session = ort.InferenceSession(built.serialise(), options, providers=["CPUExecutionProvider"])
            actual = session.run(None, inputs)[0]
            error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert np.isfinite(actual).all() and error < (0.001 if precision == "fp16" else 0.03), error

    backend = OrtBackend()
    target = Target("test-gpu", "ort", "DmlExecutionProvider", "gpu", "unknown", "GPU")
    session = MagicMock()
    session.get_providers.return_value = ["DmlExecutionProvider"]
    session.get_outputs.return_value = [MagicMock(name="output")]
    with patch.object(backend.ort, "InferenceSession", return_value=session) as create:
        backend.compile(models.build_precision_gemm("fp16"), target, RunConfig(), "fp16")
        assert create.call_args.kwargs["providers"] == ["DmlExecutionProvider"]
        assert create.call_args.kwargs["sess_options"].get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
    try:
        backend.compile(models.build_precision_gemm("int4"), target, RunConfig(), "int4")
    except BackendError as error:
        assert "native INT4" in str(error)
    else:
        raise AssertionError("INT4 dequantization was reported as native INT4 throughput")
    print("Precision graphs, deterministic hashes, full output accuracy, scheduling defaults and strict GPU fallback passed.")


if __name__ == "__main__":
    main()
