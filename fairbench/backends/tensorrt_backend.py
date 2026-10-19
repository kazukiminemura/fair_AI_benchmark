"""Native TensorRT execution with evidence from the engine being measured."""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import numpy as np

from ..stats import inference_samples, median, summarise
from .base import Backend, BackendError, CompiledModel


def _cuda(result):
    error, *values = result
    if int(error):
        raise BackendError(f"CUDA runtime error: {error}")
    return values[0] if len(values) == 1 else values


def probe_runtime() -> dict:
    """Executed by interpreter discovery, including isolated environments."""
    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cuda

        devices = []
        for index in range(_cuda(cuda.cudaGetDeviceCount())):
            props = _cuda(cuda.cudaGetDeviceProperties(index))
            devices.append({"device_id": index, "name": props.name.decode(),
                            "compute_capability": [props.major, props.minor]})
        return {"tensorrt_version": trt.__version__, "tensorrt_devices": devices}
    except ImportError:
        return {}
    except Exception as exc:
        return {"tensorrt_error": str(exc)}


def int8_evidence(information: dict, engine_sha256: str) -> dict:
    """Fail closed: quantizers, boundary tensors and builder flags prove nothing.

    This backend accepts only the single-MatMul precision GEMM for INT8. Require
    exactly one identifiable compiled GEMM, INT8 activation and weight operands,
    and an INT8 tactic. Unknown/fused representations require a new verifier.
    """
    layers = information.get("Layers", [])
    compute = [layer for layer in layers if re.search(
        r"gemm|matrixmultiply|fullyconnected|convolution", str(layer.get("LayerType", "")), re.I)]
    if len(compute) != 1:
        raise BackendError("TensorRT INT8 verification failed: expected one identifiable compiled GEMM.")
    layer = compute[0]
    inputs = layer.get("Inputs", [])
    activation_int8 = bool(inputs) and "int8" in inputs[0].get("Format/Datatype", "").lower()
    tactic = layer.get("TacticName", "")
    # TensorRT may embed constant weights, omitting them from Inputs/Weights.
    # The selected i8i8 tactic explicitly identifies both operand types.
    if len(inputs) >= 2:
        weights_int8 = "int8" in inputs[1].get("Format/Datatype", "").lower()
    elif layer.get("Weights"):
        weights_int8 = str(layer["Weights"].get("Type", "")).lower() == "int8"
    else:
        weights_int8 = bool(re.search(r"i8i8|s8s8", tactic, re.I))
    tactic_int8 = bool(re.search(r"i8i8|i16832|s8s8|int8", tactic, re.I))
    if not (activation_int8 and weights_int8 and tactic_int8):
        raise BackendError("TensorRT INT8 verification failed: compiled GEMM operands/tactic do not prove INT8. "
                           + json.dumps(layer))
    return {"available": False, "tensor_precisions": ["int8"], "evidence": "tensorrt_engine_inspector",
            "arithmetic_precision": "int8", "accumulator_precision": "unverified",
            "engine_sha256": engine_sha256, "compute_layers": compute,
            "runtime_execution_verified": False,
            "note": "INT8 operands and tactic verified in the measured TensorRT engine; "
                    "successful synchronized execution required. No instruction disassembly or accumulator verification."}


class _Execution:
    """Each request owns its CUDA stream, execution context and static buffers."""

    def __init__(self, engine, trt, cuda, device_id):
        self.engine, self.cuda, self.device_id = engine, cuda, device_id
        self.stream, self.buffers = None, {}
        _cuda(cuda.cudaSetDevice(device_id))
        self.context = engine.create_execution_context()
        if self.context is None:
            raise BackendError("TensorRT could not create an execution context.")
        self.stream = _cuda(cuda.cudaStreamCreate())
        self.inputs, self.outputs = [], []
        try:
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                shape = tuple(engine.get_tensor_shape(name))
                if any(dim <= 0 for dim in shape):
                    raise BackendError("TensorRT benchmark requires static positive I/O shapes.")
                host = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(name)))
                pointer = _cuda(cuda.cudaMalloc(host.nbytes))
                self.buffers[name] = (pointer, host)
                if not self.context.set_tensor_address(name, pointer):
                    raise BackendError(f"TensorRT could not bind {name}.")
                (self.inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(name)
        except Exception:
            self.close()
            raise

    def upload(self, feed):
        _cuda(self.cuda.cudaSetDevice(self.device_id))
        if set(feed) != set(self.inputs):
            raise BackendError("TensorRT input names do not match the engine.")
        for name in self.inputs:
            pointer, host = self.buffers[name]
            value = feed[name]
            if value.shape != host.shape or value.dtype != host.dtype:
                raise BackendError(f"TensorRT input shape/type mismatch: {name}")
            np.copyto(host, value)
            _cuda(self.cuda.cudaMemcpyAsync(pointer, host.ctypes.data, host.nbytes,
                                           self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        _cuda(self.cuda.cudaStreamSynchronize(self.stream))

    def execute(self):
        _cuda(self.cuda.cudaSetDevice(self.device_id))
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise BackendError("TensorRT execution failed.")
        _cuda(self.cuda.cudaStreamSynchronize(self.stream))

    def download(self):
        _cuda(self.cuda.cudaSetDevice(self.device_id))
        for name in self.outputs:
            pointer, host = self.buffers[name]
            _cuda(self.cuda.cudaMemcpyAsync(host.ctypes.data, pointer, host.nbytes,
                                           self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        _cuda(self.cuda.cudaStreamSynchronize(self.stream))
        return [self.buffers[name][1].copy() for name in self.outputs]

    def close(self):
        _cuda(self.cuda.cudaSetDevice(self.device_id))
        if self.stream is not None:
            _cuda(self.cuda.cudaStreamSynchronize(self.stream))
        self.context = None
        for pointer, _ in self.buffers.values():
            _cuda(self.cuda.cudaFree(pointer))
        self.buffers.clear()
        if self.stream is not None:
            _cuda(self.cuda.cudaStreamDestroy(self.stream))
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TensorRTBackend(Backend):
    id = "tensorrt"
    name = "NVIDIA TensorRT"
    _IO_DEVICES = {"TensorRT": "cuda"}

    @staticmethod
    def is_installed():
        import importlib.util
        return importlib.util.find_spec("tensorrt") is not None and importlib.util.find_spec("cuda") is not None

    def __init__(self):
        import tensorrt as trt
        from cuda.bindings import runtime as cuda

        self.trt, self.cuda = trt, cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engines = {}

    def compile(self, built, target, cfg, precision):
        trt = self.trt
        if precision == "int8" and built.key != "precision_gemm_int8":
            raise BackendError("Verified TensorRT INT8 measurement supports precision_gemm_int8 only.")
        device_id = int(target.options.get("device_id", 0))
        _cuda(self.cuda.cudaSetDevice(device_id))
        started = time.perf_counter()
        key = (built.sha256, device_id)
        if key not in self.engines:
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse(built.serialise()):
                raise BackendError("TensorRT ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            options = builder.create_builder_config()
            options.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            options.clear_flag(trt.BuilderFlag.TF32)
            options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024**2)
            plan = builder.build_serialized_network(network, options)
            if plan is None:
                raise BackendError("TensorRT engine build failed; no fallback is allowed.")
            engine = self.runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise BackendError("TensorRT engine deserialization failed.")
            self.engines[key] = (engine, hashlib.sha256(bytes(plan)).hexdigest())
        engine, sha = self.engines[key]
        information = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
        evidence = int8_evidence(information, sha) if precision == "int8" else None
        execution = _Execution(engine, trt, self.cuda, device_id)
        props = _cuda(self.cuda.cudaGetDeviceProperties(device_id))
        meta: dict[str, Any] = {"requested_provider": "TensorRT", "execution_providers": ["TensorRT"],
            "fallback_used": False, "tensorrt_version": trt.__version__, "device_name": props.name.decode(),
            "device_id": device_id, "engine_sha256": sha, "engine_inspector": information,
            "strongly_typed": True, "tf32_enabled": False, "workspace_limit_bytes": 512 * 1024**2}
        if evidence:
            meta["gpu_precision"] = evidence
            meta["compute_precisions"] = ["int8"]
        return CompiledModel(execution, (time.perf_counter() - started) * 1000, precision, execution.outputs, meta)

    @staticmethod
    def io_device(target):
        return "cuda"

    def _execute(self, compiled):
        compiled.handle.execute()
        if "gpu_precision" in compiled.meta:
            compiled.meta["gpu_precision"].update(available=True, runtime_execution_verified=True)

    def run(self, compiled, feed):
        compiled.handle.upload(feed)
        self._execute(compiled)
        return compiled.handle.download()

    def prepare_resident(self, compiled, feed, target):
        compiled.handle.upload(feed)
        return lambda: self._execute(compiled), compiled.handle.download

    def diagnose(self, compiled, built, target, cfg, feed):
        handle = compiled.handle
        phases = {"h2d_ms": [], "compute_ms": [], "d2h_ms": [], "e2e_ms": []}
        for _ in inference_samples(cfg, phases["compute_ms"]):
            start = time.perf_counter()
            handle.upload(feed)
            uploaded = time.perf_counter()
            self._execute(compiled)
            computed = time.perf_counter()
            handle.download()
            end = time.perf_counter()
            for name, value in (("h2d_ms", uploaded-start), ("compute_ms", computed-uploaded),
                                ("d2h_ms", end-computed), ("e2e_ms", end-start)):
                phases[name].append(value * 1000)
        return {"available": True, "method": "synchronized_host_wall_cuda_stream", "transfer_scope": "device_io",
                "internal_transfers_in_compute": False, "samples": len(phases["compute_ms"]),
                "compute_statistics": summarise(phases["compute_ms"]).to_dict(), "phase_samples_ms": phases,
                **{name: median(values) for name, values in phases.items()},
                "note": "Measured engine; synchronized host wall time includes dispatch, Q/DQ and synchronization. "
                        "Resident compute excludes host/device transfers; not isolated kernel time."}
