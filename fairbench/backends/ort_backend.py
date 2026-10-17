"""ONNX Runtime backend.

This is the portable path: the same runtime, the same graph, the same input,
different Execution Provider underneath. It is available for **both** vendors -
AMD CPUs run CPUExecutionProvider, AMD GPUs run DirectML, Intel NPUs run the
OpenVINO EP, AMD NPUs run Vitis AI - which is why it is the backbone of the
suite.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RunConfig
from ..devices import Target
from ..models import BuiltModel
from ..stats import median
from .base import Backend, BackendError, CompiledModel


class OrtBackend(Backend):
    id = "ort"
    name = "ONNX Runtime"
    _IO_DEVICES = {"DmlExecutionProvider": "dml", "CUDAExecutionProvider": "cuda",
                   "ROCMExecutionProvider": "cuda", "CPUExecutionProvider": "cpu"}

    @staticmethod
    def is_installed() -> bool:
        try:
            import onnxruntime  # noqa: F401

            return True
        except Exception:
            return False

    def __init__(self) -> None:
        import onnxruntime as ort

        self.ort = ort
        self.version = getattr(ort, "__version__", "?")

    # ------------------------------------------------------------------
    def _provider_options(self, target: Target, precision: str) -> dict[str, Any]:
        ep = target.device
        opts: dict[str, Any] = dict(target.options.get("provider_options", {}))
        if ep == "OpenVINOExecutionProvider":
            # Route the EP at the right Intel device and precision.
            dev = target.options.get("openvino_device")
            if dev:
                opts.setdefault("device_type", dev)
            opts.setdefault("precision", {"int8": "INT8", "fp16": "FP16", "fp32": "FP32"}.get(precision, "FP32"))
            opts.setdefault("num_of_threads", "0")
        elif ep == "DmlExecutionProvider":
            opts.setdefault("device_id", "0")
            opts.setdefault("disable_graph_capture", "0")
        elif ep == "VitisAIExecutionProvider":
            for key in ("config_file", "cache_dir", "cache_key", "ai_analyzer"):
                if target.options.get(key):
                    opts.setdefault(key, target.options[key])
        elif ep in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
            if precision == "fp16":
                opts.setdefault("arena_extend_strategy", "kSameAsRequested")
        elif ep == "QNNExecutionProvider":
            opts.setdefault("backend_path", "QnnHtp.dll")
        return opts

    def _session_options(self, target: Target, cfg: RunConfig) -> Any:
        so = self.ort.SessionOptions()
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        # DirectML requires memory-pattern optimisation to be disabled.
        so.enable_mem_pattern = target.device != "DmlExecutionProvider"
        so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL

        threads = cfg.resolve_threads()
        so.intra_op_num_threads = threads
        # Single inference stream: we measure latency/throughput of one stream,
        # not the throughput of a saturated server. Anything else would reward
        # bigger core counts instead of better AI hardware.
        so.inter_op_num_threads = 1

        if cfg.thread_affinity and threads > 1:
            try:
                # Off by default: see the note in RunConfig.thread_affinity.
                # ORT numbers logical processors from 1, and expects
                # (thread_pool_size - 1) entries.
                affinity = ";".join(str(i) for i in range(1, threads))
                so.add_session_config_entry("session.intra_op_thread_affinities", affinity)
            except Exception:
                # Not fatal: the option is not honoured by every build/platform.
                pass
        return so

    # ------------------------------------------------------------------
    def compile(self, built: BuiltModel, target: Target, cfg: RunConfig, precision: str) -> CompiledModel:
        providers = [target.device]
        provider_options = [self._provider_options(target, precision)]
        # Always keep the CPU EP as a silent fallback so a partially assignable
        # graph still runs; unassigned nodes are visible via `fallback_used`.
        if target.device != "CPUExecutionProvider" and not built.key.startswith("precision_gemm_"):
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        so = self._session_options(target, cfg)
        if built.key.startswith("precision_gemm_"):
            if precision == "int4":
                raise BackendError("INT4 QDQ GEMM requires verified native INT4 execution; this ONNX Runtime backend cannot verify it. Dequantized FP32 throughput is not reported as INT4 TOPS.")
            if target.kind != "cpu":
                so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        started = time.perf_counter()
        try:
            session = self.ort.InferenceSession(
                built.serialise(),
                sess_options=so,
                providers=providers,
                provider_options=provider_options,
            )
        except Exception as exc:
            raise BackendError(f"{target.device} could not load '{built.key}': {exc}") from exc
        compile_ms = (time.perf_counter() - started) * 1000

        active = session.get_providers()
        return CompiledModel(
            handle=session,
            compile_ms=compile_ms,
            precision=precision,
            output_names=[o.name for o in session.get_outputs()],
            meta={
                "execution_providers": active,
                "requested_provider": target.device,
                "fallback_used": bool(active and active[0] != target.device),
                "intra_op_threads": so.intra_op_num_threads,
                "inter_op_threads": so.inter_op_num_threads,
                "graph_optimization": "ORT_ENABLE_ALL",
                "precision_note": "Model input/weight precision; native hardware arithmetic is not verified. FP16 may be emulated by the runtime.",
            },
        )

    def run(self, compiled: CompiledModel, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return compiled.handle.run(None, feed)

    def prepare_resident(self, compiled: CompiledModel, feed: dict[str, np.ndarray], target: Target):
        if target.device not in self._IO_DEVICES:
            return super().prepare_resident(compiled, feed, target)
        binding, tensors = self._resident_binding(compiled, feed, target, warmup=1)

        def run(inputs=tensors):
            compiled.handle.run_with_iobinding(binding)
            binding.synchronize_outputs()

        return run, binding.copy_outputs_to_cpu

    def _resident_binding(self, compiled: CompiledModel, feed: dict[str, np.ndarray],
                          target: Target, warmup: int):
        device = self._IO_DEVICES[target.device]
        device_id = int(self._provider_options(target, compiled.precision).get("device_id", 0))
        binding = compiled.handle.io_binding()
        tensors = {name: self.ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value), device, device_id)
                   for name, value in feed.items()}
        for name, value in tensors.items():
            binding.bind_ortvalue_input(name, value)
        for name in compiled.output_names:
            binding.bind_output(name, device, device_id)
        for _ in range(warmup):
            compiled.handle.run_with_iobinding(binding)
            binding.synchronize_outputs()
        for name, value in zip(compiled.output_names, binding.get_outputs(), strict=True):
            binding.bind_ortvalue_output(name, value)

        return binding, tensors

    def diagnose(self, compiled: CompiledModel, built: BuiltModel, target: Target,
                 cfg: RunConfig, feed: dict[str, np.ndarray]) -> dict[str, Any]:
        compiled.meta["compute_precisions"] = []
        compiled.meta["compute_precision_source"] = "Runtime does not expose arithmetic precision."
        if target.kind == "cpu":
            try:
                events = self._profile_events(built, target, cfg, feed)
                types = set()
                for event in events:
                    arguments = event.get("args", {})
                    if arguments.get("op_name") in ("MatMul", "Gemm", "FusedMatMul", "MatMulInteger", "QLinearMatMul"):
                        input_types = arguments.get("input_type_shape", [])
                        for index in ((0, 3) if arguments.get("op_name") == "QLinearMatMul" else (0, 1)):
                            if index < len(input_types):
                                types.update(input_types[index])
                compiled.meta["compute_precisions"] = sorted(types)
                compiled.meta["compute_precision_source"] = "ORT CPU compute-node input types; instruction/accumulator precision unverified"
            except Exception as exc:
                compiled.meta["compute_precision_source"] = f"Precision probe unavailable: {exc}"
            return {"available": False, "reason": "CPU execution has no host/GPU transfer phases."}
        if target.kind != "gpu" or target.device not in self._IO_DEVICES:
            return {"available": False, "reason": "Device-resident I/O binding is unavailable for this provider."}
        compiled.meta["gpu_precision"] = self._probe_gpu_precision(built, target, cfg, feed)
        try:
            binding, tensors = self._resident_binding(compiled, feed, target, warmup=max(2, cfg.warmup))
            reference = compiled.handle.run(None, feed)
            phases = {"h2d_ms": [], "compute_ms": [], "d2h_ms": []}
            sample_count = max(10, min(30, cfg.min_iterations))
            for _ in range(sample_count):
                started = time.perf_counter()
                for name, tensor in tensors.items():
                    tensor.update_inplace(np.ascontiguousarray(feed[name]))
                binding.synchronize_inputs()
                uploaded = time.perf_counter()
                compiled.handle.run_with_iobinding(binding)
                binding.synchronize_outputs()
                computed = time.perf_counter()
                outputs = binding.copy_outputs_to_cpu()
                downloaded = time.perf_counter()
                for actual, expected in zip(outputs, reference, strict=True):
                    if not np.allclose(actual, expected, rtol=1e-3, atol=1e-3):
                        raise BackendError("Device-resident output does not match the end-to-end result.")
                phases["h2d_ms"].append((uploaded - started) * 1000)
                phases["compute_ms"].append((computed - uploaded) * 1000)
                phases["d2h_ms"].append((downloaded - computed) * 1000)
            return {"available": True, "method": "synchronized_host_wall_io_binding", "samples": sample_count,
                    **{name: median(values) for name, values in phases.items()}, "phase_samples_ms": phases,
                    "note": "Synchronized host wall time, not GPU event/kernel time. Compute includes dispatch, synchronization, casts and quantization; model weights and I/O allocations are already resident. End-to-end inference is measured separately."}
        except Exception as exc:
            return {"available": False, "reason": f"I/O binding timing unavailable: {exc}"}

    def _probe_gpu_precision(self, built: BuiltModel, target: Target, cfg: RunConfig,
                             feed: dict[str, np.ndarray]) -> dict[str, Any]:
        try:
            return self._gpu_precision_evidence(built, target, self._profile_events(built, target, cfg, feed))
        except Exception as exc:
            return {"available": False, "reason": f"GPU precision probe unavailable: {exc}"}

    def _profile_events(self, built: BuiltModel, target: Target, cfg: RunConfig,
                        feed: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="fairbench-precision-") as directory:
            options = self._session_options(target, cfg)
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "precision")
            provider_options = {}
            if target.kind == "gpu":
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                provider_options["provider_options"] = [self._provider_options(target, built.precision)]
            session = self.ort.InferenceSession(built.serialise(), options, providers=[target.device], **provider_options)
            try:
                session.run(None, feed)
            finally:
                profile_path = session.end_profiling()
            return json.loads(Path(profile_path).read_text(encoding="utf-8"))

    @staticmethod
    def _gpu_precision_evidence(built: BuiltModel, target: Target,
                                events: list[dict[str, Any]]) -> dict[str, Any]:
        import onnx

        kernels = [event.get("args", {}) for event in events
                   if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
        if not kernels or any(kernel.get("provider") != target.device for kernel in kernels):
            return {"available": False, "reason": "Profile did not confirm exclusive execution on the requested GPU provider."}
        operators = {"MatMul", "Gemm", "FusedMatMul", "MatMulInteger", "QLinearMatMul"}
        nodes = [kernel for kernel in kernels if kernel.get("op_name") in operators]
        types = set()
        evidence = "profiled_compute_inputs"
        if nodes:
            for node in nodes:
                inputs = node.get("input_type_shape", [])
                for index in ((0, 3) if node["op_name"] == "QLinearMatMul" else (0, 1)):
                    if index >= len(inputs) or not inputs[index]:
                        return {"available": False, "reason": "Profile omitted a compute operand type."}
                    types.update(inputs[index])
            if len(nodes) != len([node for node in built.model.graph.node if node.op_type in operators]):
                return {"available": False, "reason": "Profile does not expose every compute operator."}
        elif (len(kernels) == 1 and kernels[0].get("op_name", "").startswith("DmlFusedNode")
              and built.key.startswith("precision_gemm_")):
            graph = onnx.shape_inference.infer_shapes(built.model).graph
            tensor_types = {value.name: value.type.tensor_type.elem_type
                            for value in [*graph.input, *graph.value_info, *graph.output]}
            tensor_types.update({value.name: value.data_type for value in graph.initializer})
            nodes = [node for node in graph.node if node.op_type in operators]
            if len(nodes) != 1:
                return {"available": False, "reason": "Fused precision graph must contain exactly one compute operator."}
            for node in nodes:
                for index in ((0, 3) if node.op_type == "QLinearMatMul" else (0, 1)):
                    dtype = tensor_types.get(node.input[index])
                    if not dtype:
                        return {"available": False, "reason": "Assigned graph omitted a compute operand type."}
                    types.add(onnx.TensorProto.DataType.Name(dtype).lower())
            evidence = "profiled_fused_graph"
        if not types:
            return {"available": False, "reason": "GPU profile did not expose supported compute operators."}
        return {"available": True, "provider": target.device, "tensor_precisions": sorted(types),
                "evidence": evidence, "profiled_operators": [kernel.get("op_name") for kernel in kernels],
                "arithmetic_precision": "unverified", "accumulator_precision": "unverified",
                "note": "GPU provider assignment verified in a separate session with identical model/options. "
                        "Types describe compute operands in the assigned graph, not native GPU instructions or accumulator precision. "
                        "Fused graph boundary input/output types are not used as compute precision."}
