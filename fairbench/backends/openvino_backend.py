"""OpenVINO backend (Intel CPU / GPU / NPU).

OpenVINO is the only runtime that exposes Intel's NPU as a first-class device,
so it is required for any Intel NPU number. It is used for CPU and GPU too,
which is deliberate: the same runtime on all three devices means the
CPU-vs-GPU-vs-NPU comparison inside one machine is apples-to-apples, and it
gives Intel's own toolchain a fair shot on CPU.

Everything is forced to a **single inference stream at LATENCY performance
mode** so it can be compared with the single-stream ONNX Runtime numbers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RunConfig
from ..devices import Target
from ..models import BuiltModel
from .base import Backend, BackendError, CompiledModel

CACHE_DIR = Path(".cache") / "openvino"


def _static_shape(port) -> list[int]:
    """Extract a concrete shape from an OpenVINO input port.

    The OpenVINO Python API has renamed this accessor more than once; the three
    fallbacks below cover 2023.x through 2026.x.
    """
    pshape = port.get_partial_shape()
    for getter in (
        lambda ps: list(ps.to_shape()),
        lambda ps: [int(d.get_length()) for d in ps],
        lambda ps: [int(d) for d in ps],
    ):
        try:
            shape = getter(pshape)
        except Exception:
            continue
        if shape and all(isinstance(v, int) and v >= 0 for v in shape):
            return shape
    raise ValueError(f"OpenVINO input '{port.get_any_name()}' is not statically shaped: {pshape}")


def _reshape_map(model) -> dict[str, list[int]]:
    """Static shapes for every input that can carry one.

    A *presence* input is a legitimate zero-length tensor: the prefill workload
    feeds an empty KV cache. OpenVINO's reshape API rejects a 0 in the target
    shape, so those inputs are left alone - their shape is already correct in
    the converted model.
    """
    shapes: dict[str, list[int]] = {}
    for inp in model.inputs:
        shape = _static_shape(inp)
        if any(dim == 0 for dim in shape):
            continue
        shapes[inp.get_any_name()] = shape
    return shapes


class OpenVinoBackend(Backend):
    id = "openvino"
    name = "OpenVINO"

    @staticmethod
    def is_installed() -> bool:
        try:
            import openvino  # noqa: F401

            return True
        except Exception:
            return False

    def __init__(self) -> None:
        import openvino as ov

        self.ov = ov
        self.core = ov.Core()
        self.version = getattr(ov, "__version__", "?")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self.core.set_property({"CACHE_DIR": str(CACHE_DIR.resolve())})
        except Exception:
            pass

    # ------------------------------------------------------------------
    def compile(self, built: BuiltModel, target: Target, cfg: RunConfig, precision: str) -> CompiledModel:
        ov = self.ov

        started = time.perf_counter()
        try:
            model = ov.convert_model(str(built.materialise()))
        except Exception as exc:
            raise BackendError(f"OpenVINO could not convert '{built.key}': {exc}") from exc

        try:
            # NPUs consume static shapes; every graph in the suite is static by
            # construction, so this is shape re-assertion, not a real reshape.
            shapes = _reshape_map(model)
            if shapes:
                model.reshape(shapes)
        except Exception as exc:
            raise BackendError(f"OpenVINO could not fix shapes for '{built.key}': {exc}") from exc

        props = self._properties(target, cfg, precision)
        try:
            compiled = self.core.compile_model(model, target.device, props)
        except Exception as exc:
            raise BackendError(f"OpenVINO could not compile for {target.device}: {exc}") from exc
        compile_ms = (time.perf_counter() - started) * 1000

        return CompiledModel(
            handle=compiled,
            compile_ms=compile_ms,
            precision=precision,
            output_names=[o.get_any_name() for o in compiled.outputs],
            meta={
                "execution_providers": [f"OpenVINO {target.device}"],
                "input_graph_precision": built.precision,
                "inference_precision_hint": "fp16" if target.kind in ("gpu", "npu") else "runtime_default",
                "precision_note": "Hint is a compiler preference, not proof of every operator's execution precision.",
                "requested_provider": f"OpenVINO:{target.device}",
                "openvino_properties": {k: str(v) for k, v in props.items()},
                "intra_op_threads": cfg.resolve_threads() if target.kind == "cpu" else None,
            },
        )

    def _properties(self, target: Target, cfg: RunConfig, precision: str) -> dict[str, Any]:
        ov = self.ov
        props: dict[str, Any] = {}
        try:
            props[ov.properties.hint.performance_mode()] = ov.properties.hint.PerformanceMode.LATENCY
        except Exception:
            pass

        if target.kind == "cpu":
            try:
                props[ov.properties.inference_num_threads()] = cfg.resolve_threads()
            except Exception:
                pass
        elif target.kind in ("gpu", "npu"):
            # Both accelerators receive the same hint. Previously the GPU was
            # forced to FP32 while the NPU used FP16 for the same FP32 graph.
            try:
                props[ov.properties.hint.inference_precision()] = ov.Type.f16
            except Exception:
                pass
        return props

    def run(self, compiled: CompiledModel, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        inputs = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feed.items()}
        result = compiled.handle(inputs, share_outputs=False)
        names = compiled.output_names
        return [np.asarray(result[name]) for name in names]
