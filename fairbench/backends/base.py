"""Backend abstraction.

A backend is a *runtime*, not a vendor: ONNX Runtime and OpenVINO both serve
several device kinds. Keeping them behind one interface is what lets the suite
measure an Intel NPU (OpenVINO plugin) and an AMD NPU (Vitis AI EP) with the
same workload code.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RunConfig
from ..devices import Target
from ..models import BuiltModel


@dataclass
class CompiledModel:
    """A device-resident, ready-to-run model."""

    handle: Any
    compile_ms: float
    precision: str
    output_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


class BackendError(RuntimeError):
    """Raised when a target cannot host a workload. Recorded, not fatal."""


class Backend(abc.ABC):
    id: str = "base"
    name: str = "base"

    @staticmethod
    def is_installed() -> bool:  # pragma: no cover - trivial
        return False

    @abc.abstractmethod
    def compile(self, built: BuiltModel, target: Target, cfg: RunConfig, precision: str) -> CompiledModel:
        """Load and warm the graph onto the device.

        Compilation is timed separately from inference: on an NPU it can take
        tens of seconds, and mixing it into the latency would be dishonest.
        """

    @abc.abstractmethod
    def run(self, compiled: CompiledModel, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Execute one inference."""

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name}

    def diagnose(self, compiled: CompiledModel, built: BuiltModel, target: Target,
                 cfg: RunConfig, feed: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"available": False, "reason": "Transfer/compute separation is not available for this backend."}

    def prepare_resident(self, compiled: CompiledModel, feed: dict[str, np.ndarray], target: Target):
        raise BackendError("Persistent device buffers are not supported by this backend.")
