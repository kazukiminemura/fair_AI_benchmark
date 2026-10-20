"""Regression check: uv run python -m scripts.check_npu_requests (CPU only)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from fairbench.backends.base import BackendError
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.models import build_precision_gemm


def main():
    backend = OrtBackend()
    cpu = Target("cpu", "ort", "CPUExecutionProvider", "cpu", "amd", "CPU")
    npu = Target("npu", "winml", "VitisAIExecutionProvider", "npu", "amd", "NPU")
    built = build_precision_gemm("int8", 64)
    original_sha = built.sha256
    opts = backend._model_provider_options(built, npu, "int8")
    profile = json.loads(Path(opts["config_file"]).read_text())
    assert opts["target"] == "X2"
    dpu = next(p["passDpuParam"] for p in profile["passes"] if "passDpuParam" in p)
    assert dpu["xcompilerAttrs"]["enable_matmul_to_conv"] == {"boolValue": False}
    assert profile["targets"][0]["targetOpts"]["xcompilerAttrs"]["enable_matmul_to_conv"] == {"boolValue": False}
    assert built.sha256 == original_sha and npu.options == {}
    # Do not override explicit compiler profiles or tune unrelated workloads.
    for explicit in ({"config_file": "user.json"}, {"target_opts": "{}"},
                     {"target": "X1"}, {"target": "X2-Benchmark"}, {"xclbin": "user.xclbin"}):
        custom = replace(npu, options={"provider_options": explicit})
        assert backend._model_provider_options(built, custom, "int8") == explicit
    custom = replace(npu, options={"config_file": "user.json"})
    assert backend._model_provider_options(built, custom, "int8") == {"config_file": "user.json"}
    assert "config_file" not in backend._model_provider_options(built, cpu, "int8")
    assert "config_file" not in backend._model_provider_options(built, npu, "fp16")
    assert "config_file" not in backend._model_provider_options(replace(built, key="other_int8"), npu, "int8")
    feed = built.make_inputs()
    # A CPU session exercises the same host OrtValue / I/O binding lifecycle
    # without requiring AMD hardware in the regression suite.
    compiled = backend.compile(built, cpu, RunConfig(), "int8")
    expected = backend.run(compiled, feed)
    assert backend.io_device(npu) == "cpu"
    requests = [backend.prepare_resident(compiled, feed, npu) for _ in range(2)]
    for _ in range(3):
        for run, read_outputs in requests:
            run()
            for actual, reference in zip(read_outputs(), expected, strict=True):
                np.testing.assert_array_equal(actual, reference)
    # Separate requests own their output storage; reading one must not damage
    # another request or the next inference.
    first = requests[0][1]()
    first[0].fill(np.nan)
    requests[1][0]()
    np.testing.assert_array_equal(requests[1][1]()[0], expected[0])
    unsupported = Target("other", "ort", "UnknownProvider", "npu", "unknown", "Other")
    try:
        backend.prepare_resident(compiled, feed, unsupported)
    except BackendError:
        pass
    else:
        raise AssertionError("Unknown providers must not silently acquire a resident path")
    print("NPU host bindings: repeated results, independent outputs and unsupported-provider rejection passed.")


if __name__ == "__main__":
    main()
