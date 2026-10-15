"""ONNX Runtime backend.

This is the portable path: the same runtime, the same graph, the same input,
different Execution Provider underneath. It is available for **both** vendors -
AMD CPUs run CPUExecutionProvider, AMD GPUs run DirectML, Intel NPUs run the
OpenVINO EP, AMD NPUs run Vitis AI - which is why it is the backbone of the
suite.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import RunConfig
from ..devices import Target
from ..models import BuiltModel
from .base import Backend, BackendError, CompiledModel


class OrtBackend(Backend):
    id = "ort"
    name = "ONNX Runtime"

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
        so.enable_mem_pattern = True
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
        if target.device != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
            provider_options.append({})

        so = self._session_options(target, cfg)
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
            },
        )

    def run(self, compiled: CompiledModel, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return compiled.handle.run(None, feed)
