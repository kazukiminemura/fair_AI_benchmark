"""ONNX inference on an explicitly selected Windows ML EP device."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

from .. import windowsml
from .base import BackendError, CompiledModel
from .ort_backend import OrtBackend


class WindowsMLBackend(OrtBackend):
    id = "winml"
    name = "Windows ML"

    @staticmethod
    def is_installed() -> bool:
        return sys.platform == "win32" and importlib.util.find_spec("windowsml") is not None

    def __init__(self) -> None:
        super().__init__()
        self.notes = windowsml.register()

    def _create_session(self, built, target, cfg, *, profile_prefix=None, precision=None):
        selected = next((device for device in self.ort.get_ep_devices()
                         if windowsml.device_info(device)["key"] == target.options.get("winml_key")), None)
        if selected is None:
            raise BackendError("Windows ML device is no longer available; refresh device discovery.")
        options = self._session_options(target, cfg)
        if target.kind != "cpu":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = profile_prefix
        provider_options = (self._model_provider_options(built, target, precision or built.precision)
                            if target.device == "VitisAIExecutionProvider"
                            else dict(target.options.get("provider_options", {})))
        if target.device.casefold() == windowsml.TENSORRT_RTX_PROVIDER.casefold():
            # Host buffers are the portable WinML path. CUDA graph capture
            # requires stable device bindings, which this path does not have.
            provider_options.setdefault("enable_cuda_graph", "0")
        options.add_provider_for_devices([selected], provider_options)
        started = time.perf_counter()
        try:
            session = self.ort.InferenceSession(self._model_source(built, target), sess_options=options, enable_fallback=False)
        except Exception as exc:
            raise BackendError(f"Windows ML {target.device} ({target.kind}) could not load {built.key}: {exc}") from exc
        active = session.get_providers()
        if target.device not in active:
            raise BackendError(f"Windows ML did not activate the requested provider: {active}")
        return session, selected, options, provider_options, (time.perf_counter() - started) * 1000

    def compile(self, built, target, cfg, precision) -> CompiledModel:
        gpu_precision = None
        if target.device.casefold() == windowsml.TENSORRT_RTX_PROVIDER.casefold():
            # Profile this very session once and stop profiling before timing.
            # Its fused FP32 boundary does not reveal the internal GEMM type.
            with tempfile.TemporaryDirectory(prefix="fairbench-winml-verify-") as directory:
                session, selected, options, provider_options, compile_ms = self._create_session(
                    built, target, cfg, profile_prefix=str(Path(directory) / "profile"), precision=precision)
                try:
                    session.run(None, built.make_inputs(cfg.input_seed))
                finally:
                    profile_path = session.end_profiling()
                events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            gpu_precision = self._gpu_precision_evidence(built, target, events)
            kernels = [event.get("args", {}) for event in events
                       if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")]
            if not kernels or any(kernel.get("provider") != target.device for kernel in kernels):
                raise BackendError("Windows ML TensorRT-RTX GPU assignment could not be verified; no throughput is reported.")
        else:
            session, selected, options, provider_options, compile_ms = self._create_session(built, target, cfg, precision=precision)
        return CompiledModel(
            handle=session, compile_ms=compile_ms,
            precision=precision, output_names=[output.name for output in session.get_outputs()],
            meta={"execution_providers": session.get_providers(), "requested_provider": target.device,
                  "windowsml_device": windowsml.device_info(selected), "fallback_used": False,
                  "provider_options": provider_options,
                  **({"gpu_precision": gpu_precision, "execution_assignment": {
                      "available": True, "provider": target.device, "source": "measured_session_profile",
                      "profiled_operators": [kernel.get("op_name") for kernel in kernels],
                      "profiling_stopped_before_timing": True}} if gpu_precision is not None else {}),
                  "cpu_fallback_disabled": target.kind != "cpu",
                  "intra_op_threads": options.intra_op_num_threads,
                  "inter_op_threads": options.inter_op_num_threads},
        )

    def _profile_events(self, built, target, cfg, feed):
        # Use the same V2 device selection as measurement. A providers=[name]
        # session cannot prove assignment to a catalog-registered WinML device.
        with tempfile.TemporaryDirectory(prefix="fairbench-winml-profile-") as directory:
            session, *_ = self._create_session(built, target, cfg, profile_prefix=str(Path(directory) / "profile"))
            try:
                session.run(None, feed)
            finally:
                profile_path = session.end_profiling()
            return json.loads(Path(profile_path).read_text(encoding="utf-8"))
