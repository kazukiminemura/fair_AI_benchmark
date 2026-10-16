"""ONNX inference on an explicitly selected Windows ML EP device."""

from __future__ import annotations

import importlib.util
import sys
import time

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

    def compile(self, built, target, cfg, precision) -> CompiledModel:
        selected = next((device for device in self.ort.get_ep_devices()
                         if windowsml.device_info(device)["key"] == target.options.get("winml_key")), None)
        if selected is None:
            raise BackendError("Windows ML device is no longer available; refresh device discovery.")
        options = self._session_options(target, cfg)
        if target.kind != "cpu":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices([selected], target.options.get("provider_options", {}))
        started = time.perf_counter()
        try:
            session = self.ort.InferenceSession(built.serialise(), sess_options=options, enable_fallback=False)
        except Exception as exc:
            raise BackendError(f"Windows ML {target.device} ({target.kind}) could not load {built.key}: {exc}") from exc
        active = session.get_providers()
        if target.device not in active:
            raise BackendError(f"Windows ML did not activate the requested provider: {active}")
        return CompiledModel(
            handle=session, compile_ms=(time.perf_counter() - started) * 1000,
            precision=precision, output_names=[output.name for output in session.get_outputs()],
            meta={"execution_providers": active, "requested_provider": target.device,
                  "windowsml_device": windowsml.device_info(selected), "fallback_used": False,
                  "cpu_fallback_disabled": target.kind != "cpu",
                  "intra_op_threads": options.intra_op_num_threads,
                  "inter_op_threads": options.inter_op_num_threads},
        )
