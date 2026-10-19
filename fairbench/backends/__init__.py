"""Backend registry."""

from __future__ import annotations

from .base import Backend, BackendError, CompiledModel

__all__ = ["Backend", "BackendError", "CompiledModel", "get_backend", "installed_backends"]


def installed_backends() -> dict[str, Backend]:
    """Instantiate every backend whose runtime is importable."""
    from .openvino_backend import OpenVinoBackend
    from .ort_backend import OrtBackend
    from .tensorrt_backend import TensorRTBackend
    from .windowsml_backend import WindowsMLBackend

    out: dict[str, Backend] = {}
    for cls in (OrtBackend, OpenVinoBackend, WindowsMLBackend, TensorRTBackend):
        try:
            if cls.is_installed():
                out[cls.id] = cls()
        except Exception:
            continue
    return out


def get_backend(backend_id: str) -> Backend:
    backends = installed_backends()
    if backend_id not in backends:
        raise BackendError(
            f"Backend '{backend_id}' is not installed. Available: {sorted(backends) or 'none'}"
        )
    return backends[backend_id]
