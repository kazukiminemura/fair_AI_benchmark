"""CPU regression checks; add --hardware to check AMD integer matrix execution."""

import argparse
import hashlib
import json
import struct
from unittest.mock import patch

import numpy as np
from onnx import numpy_helper

from fairbench import models
from fairbench.backends.base import BackendError
from fairbench.backends.vulkan_backend import (
    SHADERS,
    VulkanBackend,
    _int8_property,
    _model_parameters,
    _pack_weights,
    _spirv_int8_evidence,
    _vector_grid,
    probe_runtime,
)
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.report import format_execution_precision
from fairbench.runner import _aggregate, verify_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    # Check tile coordinates, including nonsquare matrices and signed values.
    for k, n in ((32, 64), (96, 32)):
        weights = np.random.default_rng(42).integers(-128, 128, (k, n), dtype=np.int8)
        packed = _pack_weights(weights).reshape(n//16, k//16, 16, 16)
        for column in range(n//16):
            for inner in range(k//16):
                np.testing.assert_array_equal(packed[column, inner].T,
                                              weights[inner*16:(inner+1)*16, column*16:(column+1)*16])
    profile = json.loads((SHADERS / "manifest.json").read_text())
    for elements in (64**2, 512*1024, 4096**2):
        x, y = _vector_grid(elements)
        assert x <= 65535 and y <= 65535 and 0 <= x*y*256-elements < 256*x
    for stem, record in profile["shaders"].items():
        assert hashlib.sha256((SHADERS / f"{stem}.comp").read_text().encode()).hexdigest() == record["source_sha256"]
        assert hashlib.sha256((SHADERS / f"{stem}.spv").read_bytes()).hexdigest() == record["spirv_sha256"]
    binary = (SHADERS / "int8_gemm.spv").read_bytes()
    assert _spirv_int8_evidence(binary)["operand_precision"] == "int8"
    # An altered shader must fail even when the GPU supports INT8 and the
    # benchmark's graph still carries an INT8 label.
    words = list(struct.unpack(f"<{len(binary)//4}I", binary))
    index = 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 65535
        if opcode == 21 and words[index+2] == 8:
            words[index+2] = 16
        index += count
    for bad in (b"invalid", (SHADERS / "int8_output.spv").read_bytes(), struct.pack(f"<{len(words)}I", *words)):
        try:
            _spirv_int8_evidence(bad)
        except BackendError:
            pass
        else:
            raise AssertionError("Wrong/missing compiled INT8 arithmetic was accepted")
    # A graph label alone must never authorize lowering different arithmetic.
    built = models.build_precision_gemm("int8", 64)
    _model_parameters(built, "int8")
    for model, precision in ((built, "fp16"), (models.build_precision_gemm("fp16", 64), "int8")):
        try:
            _model_parameters(model, precision)
        except BackendError:
            pass
        else:
            raise AssertionError("Unverified arithmetic accepted")
    built.model.graph.node[-1].input[1] = "input_scale"
    try:
        _model_parameters(built, "int8")
    except BackendError:
        pass
    else:
        raise AssertionError("Modified quantization was silently lowered")
    target = Target("vk", "vulkan", "VulkanCooperativeMatrix", "gpu", "amd", "AMD Vulkan INT8 GPU")
    with patch("fairbench.runner.spawn_worker", return_value={"ok": True}) as spawn:
        verify_targets([target], RunConfig())
    job = spawn.call_args.args[0]
    assert job["workload"] == "precision_gemm_int8" and job["cfg"]["check_correctness"]
    assert job["cfg"]["gemm_size"] == 512
    print("Vulkan shader integrity, canonical-model gate and integer device verification passed.")
    if not args.hardware:
        return
    import onnxruntime as ort
    import vulkan as v

    runtime = probe_runtime()
    assert runtime.get("vulkan_devices"), runtime
    device = runtime["vulkan_devices"][0]
    target.detail, target.options = device["name"], device
    prop = device["int8_matrix"]
    assert _int8_property(v, [prop])
    for name in ("AType", "BType", "CType", "ResultType"):
        assert _int8_property(v, [{**prop, name: v.VK_COMPONENT_TYPE_FLOAT32_KHR}]) is None
    backend = VulkanBackend()
    for size in (32, 64, 96, 512):
        model = models.build_precision_gemm("int8", size)
        compiled = backend.compile(model, target, RunConfig(), "int8")
        feed = model.make_inputs()
        # Exercise rounding ties, negative values, and both clipping limits.
        feed["x"][0, :10] = np.array([-6, -4, -1.5/32, -.5/32, .5/32, 1.5/32, 2.5/32, 3.5/32, 4, 6], np.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        expected = ort.InferenceSession(model.serialise(), options, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        actual = backend.run(compiled, feed)[0]
        # Verify the integer product itself, before output rounding can hide a
        # layout error. This also proves that input/weight packing is lossless.
        values = {t.name: numpy_helper.to_array(t) for t in model.model.graph.initializer}
        quantized = np.clip(np.rint(feed["x"] / values["input_scale"]), -128, 127).astype(np.int32)
        product = quantized @ values["weight"].astype(np.int32)
        accumulated = np.frombuffer(compiled.handle.buffers[3][1], dtype=np.int32).reshape(size, size)
        np.testing.assert_array_equal(accumulated, product)
        assert np.max(np.abs(actual-expected)) <= 1/16
        assert np.linalg.norm(actual-expected) / np.linalg.norm(expected) < .002
        run, read = backend.prepare_resident(compiled, feed, target)
        for _ in range(3):
            run()
            np.testing.assert_array_equal(read()[0], actual)
        assert compiled.meta["gpu_precision"]["runtime_execution_verified"]
        cell = {"ok": True, "target_id": "vk", "workload": model.key, "precision": "int8",
                "units_per_s": 10, "flops_per_s": 1000, "session": compiled.meta,
                "latency": {"median_ms": 100}}
        row = _aggregate([cell, cell])[0]
        assert row["gpu_precision"]["accumulator_precision"] == "int32"
        text = format_execution_precision(row)
        assert "Vulkan" in text and "INT32" in text and "TensorRT" not in text
        compiled.handle.close()
    print("Hardware INT8 / INT32 pipeline: full outputs, rounding/clipping, static buffers and repeated execution passed.")


if __name__ == "__main__":
    main()
