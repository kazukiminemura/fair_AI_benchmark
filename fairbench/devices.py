"""Device discovery.

``fairbench`` never guesses where a model will run. It enumerates the runtimes
that are actually installed, asks each one which devices it exposes, and only
then offers those devices as benchmark targets. A device that is advertised but
not usable (no driver, unsupported precision, NPU absent) is reported as
**unavailable with a reason**, never silently skipped - an empty column is a
result too.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from . import system

IS_WINDOWS = sys.platform.startswith("win")


# --------------------------------------------------------------------------
# the six cells the suite is designed around
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    id: str
    vendor: str
    kind: str
    label: str
    access: str

    def to_dict(self) -> dict:
        return asdict(self)


SLOTS: list[Slot] = [
    Slot("intel:cpu", "intel", "cpu", "Intel CPU", "ONNX Runtime CPU EP / OpenVINO CPU plugin"),
    Slot("intel:gpu", "intel", "gpu", "Intel GPU (Iris Xe / Arc)", "OpenVINO GPU plugin / DirectML"),
    Slot("intel:npu", "intel", "npu", "Intel NPU (AI Boost)", "OpenVINO NPU plugin (Core Ultra 100/200V/200H)"),
    Slot("amd:cpu", "amd", "cpu", "AMD CPU", "ONNX Runtime CPU EP / OpenVINO CPU plugin"),
    Slot("amd:gpu", "amd", "gpu", "AMD GPU (Radeon / iGPU)", "DirectML (Windows) / ROCm (Linux)"),
    Slot("amd:npu", "amd", "npu", "AMD NPU (XDNA / Ryzen AI)", "Vitis AI EP (ONNX Runtime) / Ryzen AI SW"),
]

# Devices we can benchmark but that are not part of the Intel-vs-AMD comparison.
EXTRA_SLOTS: list[Slot] = [
    Slot("nvidia:gpu", "nvidia", "gpu", "NVIDIA GPU", "CUDA / TensorRT EP"),
    Slot("qualcomm:npu", "qualcomm", "npu", "Qualcomm NPU (Hexagon)", "QNN EP"),
]


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
@dataclass
class Target:
    """A concrete (runtime, device) pair the runner can execute on."""

    id: str
    backend: str  # "ort" | "openvino"
    device: str  # EP name, or OpenVINO device name ("CPU", "GPU.0", "NPU")
    kind: str  # cpu | gpu | npu
    vendor: str  # intel | amd | nvidia | qualcomm | unknown
    label: str
    detail: str = ""
    available: bool = True
    precisions: list[str] = field(default_factory=lambda: ["fp32"])
    runtime_version: str = ""
    # Interpreter that actually owns this device. Empty means "the one running
    # fairbench"; otherwise the worker is re-launched there, which is how a
    # DirectML or CUDA wheel can be used without displacing the user's
    # onnxruntime install.
    python: str = ""
    # ORT provider options / OpenVINO hints, kept in the result file so a
    # reviewer can see exactly what was configured.
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Discovery:
    targets: list[Target] = field(default_factory=list)
    capabilities: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    runtimes: dict[str, str] = field(default_factory=dict)
    interpreters: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "targets": [t.to_dict() for t in self.targets],
            "capabilities": self.capabilities,
            "notes": self.notes,
            "runtimes": self.runtimes,
            "interpreters": self.interpreters,
            "slots": [s.to_dict() for s in SLOTS + EXTRA_SLOTS],
        }

    def available(self) -> list[Target]:
        return [t for t in self.targets if t.available]

    def get(self, target_id: str) -> Target | None:
        for t in self.targets:
            if t.id == target_id:
                return t
        return None


# --------------------------------------------------------------------------
# ONNX Runtime
# --------------------------------------------------------------------------
_ORT_EP_TO_KIND = {
    "CPUExecutionProvider": ("cpu", None),
    "CUDAExecutionProvider": ("gpu", "nvidia"),
    "TensorrtExecutionProvider": ("gpu", "nvidia"),
    "ROCMExecutionProvider": ("gpu", "amd"),
    "MIGraphXExecutionProvider": ("gpu", "amd"),
    "DmlExecutionProvider": ("gpu", None),  # vendor resolved from the adapter list
    "OpenVINOExecutionProvider": ("gpu", "intel"),
    "VitisAIExecutionProvider": ("npu", "amd"),
    "QNNExecutionProvider": ("npu", "qualcomm"),
    "CoreMLExecutionProvider": ("gpu", "apple"),
}

_ORT_EP_LABEL = {
    "CPUExecutionProvider": "CPU",
    "CUDAExecutionProvider": "CUDA GPU",
    "TensorrtExecutionProvider": "TensorRT GPU",
    "ROCMExecutionProvider": "ROCm GPU",
    "MIGraphXExecutionProvider": "MIGraphX GPU",
    "DmlExecutionProvider": "DirectML GPU",
    "OpenVINOExecutionProvider": "OpenVINO",
    "VitisAIExecutionProvider": "Vitis AI NPU",
    "QNNExecutionProvider": "Qualcomm QNN NPU",
    "CoreMLExecutionProvider": "CoreML",
}

def _ort_targets(fp: system.SystemFingerprint) -> tuple[list[Target], str, list[str]]:
    """Enumerate ONNX Runtime devices across every interpreter that has one.

    ``onnxruntime``, ``onnxruntime-gpu`` and ``onnxruntime-directml`` all install
    into the same import name, so a machine can only ever have one of them
    *active*. Probing a list of interpreters lets a DirectML or CUDA wheel live
    in its own virtual environment and still be benchmarked - without touching
    the environment the user already relies on.
    """
    from . import interp

    targets: list[Target] = []
    notes: list[str] = []
    interpreters = interp.probe_all()
    primary = next((i for i in interpreters if i.primary), None)
    primary_version = primary.onnxruntime if primary else "?"

    total_providers = sorted({p for i in interpreters if i.available for p in i.providers})

    gpu_vendor = None
    for a in fp.adapters:
        if a.kind == "gpu" and a.vendor in ("amd", "intel"):
            gpu_vendor = a.vendor
            break
    if gpu_vendor is None:
        g = fp.find_adapter("gpu")
        gpu_vendor = g.vendor if g else "unknown"

    seen: set[tuple[str, str]] = set()
    for info in interpreters:
        if not info.available:
            if not info.primary:
                notes.append(f"interpreter {info.path} could not be probed: {info.error}")
            continue
        for ep in info.providers:
            if ep in ("AzureExecutionProvider",):
                continue
            kind, vendor = _ORT_EP_TO_KIND.get(ep, ("unknown", None))
            if kind == "unknown":
                continue
            if vendor is None:
                vendor = fp.cpu.vendor if kind == "cpu" else gpu_vendor
            key = (ep, vendor if kind != "cpu" else "cpu")
            if key in seen:
                continue
            seen.add(key)

            label = _ORT_EP_LABEL.get(ep, ep)
            precisions = ["fp32"]
            if ep in ("CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider", "DmlExecutionProvider"):
                precisions = ["fp32", "fp16"]
            if ep in ("VitisAIExecutionProvider", "QNNExecutionProvider"):
                precisions = ["int8", "fp16"]
            if ep == "CPUExecutionProvider":
                precisions = ["fp32", "int8"]

            detail = fp.cpu.name if kind == "cpu" else _adapter_detail(fp, kind, vendor)
            target_id = f"ort:{ep}"
            if not info.primary:
                target_id += f"@{info.label}"

            targets.append(
                Target(
                    id=target_id,
                    backend="ort",
                    device=ep,
                    kind=kind,
                    vendor=vendor,
                    label=f"ORT {label}" + ("" if info.primary else f" ({info.label})"),
                    detail=detail,
                    available=True,
                    precisions=precisions,
                    runtime_version=info.onnxruntime or "?",
                    python="" if info.primary else info.path,
                    options={"interpreter": info.path, "interpreter_label": info.label},
                )
            )

    missing = [
        ("DmlExecutionProvider", "onnxruntime-directml", "any DirectX 12 GPU on Windows, including NVIDIA and AMD Radeon, with no CUDA install"),
        ("CUDAExecutionProvider", "onnxruntime-gpu (+ CUDA/cuDNN 12)", "NVIDIA GPUs"),
        ("TensorrtExecutionProvider", "onnxruntime-gpu (+ TensorRT)", "NVIDIA GPUs, fastest path"),
        ("OpenVINOExecutionProvider", "onnxruntime-openvino", "Intel CPU/GPU/NPU through ONNX Runtime"),
        ("VitisAIExecutionProvider", "onnxruntime-vitisai", "AMD Ryzen AI (XDNA) NPU"),
        ("QNNExecutionProvider", "onnxruntime-qnn", "Qualcomm Snapdragon NPU"),
    ]
    for ep, pkg, what in missing:
        if ep not in total_providers:
            setup = "Use 'fairbench setup-gpu --cuda' for an isolated CUDA environment." if ep in (
                "CUDAExecutionProvider", "TensorrtExecutionProvider"
            ) else (
                "Run 'uv sync' on a supported platform, or use a separate runtime via FAIRBENCH_PYTHON "
                f"({pkg.split(' ')[0]})."
            )
            notes.append(f"{ep} unavailable -> {what}. {setup}")

    if any(t.vendor == "nvidia" and t.kind == "gpu" for t in targets) is False and fp.find_adapter("gpu", "nvidia"):
        notes.append(
            "An NVIDIA GPU is present but no ONNX Runtime build here exposes a GPU provider. "
            "Run 'fairbench setup-gpu' (or setup-gpu.cmd in the packaged build) to create an "
            "isolated DirectML environment - it does not modify this environment."
        )
    return targets, primary_version, notes


def _adapter_detail(fp: system.SystemFingerprint, kind: str, vendor: str) -> str:
    a = fp.find_adapter(kind, vendor) or fp.find_adapter(kind)
    return a.name if a else ""


# --------------------------------------------------------------------------
# OpenVINO
# --------------------------------------------------------------------------
_OV_PROP_WHITELIST = (
    "FULL_DEVICE_NAME",
    "DEVICE_TYPE",
    "DEVICE_ARCHITECTURE",
    "DEVICE_ID",
    "NUM_STREAMS",
    "INFERENCE_PRECISION_HINT",
    "SUPPORTED_PROPERTIES",
)


def _openvino_targets(fp: system.SystemFingerprint) -> tuple[list[Target], str, list[str]]:
    targets: list[Target] = []
    notes: list[str] = []
    try:
        import openvino as ov
    except Exception:
        return [], "", [
            "openvino not installed -> Intel CPU/GPU/NPU plugin not measurable. "
            "Install with: pip install openvino"
        ]

    version = getattr(ov, "__version__", "?")
    try:
        core = ov.Core()
        devices = list(core.available_devices)
    except Exception as exc:
        return [], version, [f"openvino.Core() failed: {exc}"]

    for dev in devices:
        base = dev.split(".")[0]
        kind = {"CPU": "cpu", "GPU": "gpu", "NPU": "npu", "VPU": "npu"}.get(base, "unknown")
        if kind == "unknown":
            continue
        detail = ""
        arch = ""
        try:
            detail = str(core.get_property(dev, "FULL_DEVICE_NAME"))
        except Exception:
            detail = dev
        try:
            arch = str(core.get_property(dev, "DEVICE_ARCHITECTURE"))
        except Exception:
            arch = ""

        # OpenVINO's CPU plugin is architecture neutral; the GPU and NPU plugins
        # only target Intel silicon. Anything else that shows up (some builds
        # enumerate other vendors' adapters) is declared unusable up front rather
        # than producing a misleading number later.
        if kind == "cpu":
            vendor = fp.cpu.vendor
            usable = True
        else:
            vendor = system.classify_vendor(arch) if system.classify_vendor(arch) != "unknown" else "intel"
            if system.classify_vendor(detail) != "unknown":
                vendor = system.classify_vendor(detail)
            usable = vendor == "intel"

        precisions = ["fp32", "fp16"]
        if kind == "npu":
            precisions = ["fp16", "int8"]
        targets.append(
            Target(
                id=f"openvino:{dev}",
                backend="openvino",
                device=dev,
                kind=kind,
                vendor=vendor,
                label={
                    "cpu": "OpenVINO CPU",
                    "gpu": "OpenVINO GPU",
                    "npu": "OpenVINO NPU",
                }[kind],
                detail=detail,
                available=usable,
                precisions=precisions,
                runtime_version=version,
                options={
                    **({"DEVICE_ARCHITECTURE": arch} if arch else {}),
                    **({"unavailable_reason": "OpenVINO's GPU/NPU plugins only target Intel hardware"} if not usable else {}),
                },
            )
        )

    if not any(t.kind == "npu" for t in targets):
        notes.append(
            "No NPU device exposed by OpenVINO. Intel NPUs appear as 'NPU' on Core Ultra "
            "parts with the AI Boost driver; AMD XDNA NPUs need the Ryzen AI software stack "
            "(onnxruntime-vitisai), not OpenVINO."
        )
    return targets, version, notes


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def discover(fingerprint: system.SystemFingerprint | None = None, include_openvino: bool = True) -> Discovery:
    fp = fingerprint or system.collect()
    d = Discovery()

    ort_targets, ort_version, ort_notes = _ort_targets(fp)
    d.targets += ort_targets
    d.notes += ort_notes
    if ort_version:
        d.runtimes["onnxruntime"] = ort_version

    try:
        from . import interp as _interp

        d.interpreters = [i.to_dict() for i in _interp.probe_all()]
    except Exception:
        d.interpreters = []

    if include_openvino:
        ov_targets, ov_version, ov_notes = _openvino_targets(fp)
        d.targets += ov_targets
        d.notes += ov_notes
        if ov_version:
            d.runtimes["openvino"] = ov_version

    # Build the capability matrix the dashboard renders: for every Intel/AMD
    # slot, is there a target that fills it?
    caps: dict[str, dict] = {}
    for slot in SLOTS + EXTRA_SLOTS:
        matches = [t for t in d.targets if t.vendor == slot.vendor and t.kind == slot.kind and t.available]
        unavailable = [t for t in d.targets if t.vendor == slot.vendor and t.kind == slot.kind and not t.available]
        reason = matches[0].detail if matches else _reason_for(slot, fp)
        if not matches and unavailable:
            reason = unavailable[0].options.get("unavailable_reason") or reason
        caps[slot.id] = {
            "slot": slot.to_dict(),
            "available": bool(matches),
            "targets": [t.id for t in matches],
            "detail": reason,
        }
    d.capabilities = caps
    return d


def _reason_for(slot: Slot, fp: system.SystemFingerprint) -> str:
    if slot.kind == "npu":
        if slot.vendor == "intel":
            return "No Intel NPU present (requires Core Ultra with AI Boost), or its driver/OpenVINO NPU plugin is unavailable."
        if slot.vendor == "amd":
            return "No AMD XDNA NPU present (requires Ryzen AI 300/7040+), or onnxruntime-vitisai is not installed."
        return "No NPU device found."
    if slot.kind == "gpu":
        a = fp.find_adapter("gpu", slot.vendor)
        if a:
            return (
                f"{a.name} is present but no runtime here exposes it. "
                "Run 'fairbench setup-gpu' for an isolated DirectML environment "
                "(no CUDA needed), or install onnxruntime-gpu / the OpenVINO GPU plugin."
            )
        return f"No {slot.vendor} GPU found on this system."
    if slot.kind == "cpu":
        if fp.cpu.vendor == slot.vendor:
            return "This CPU is present; CPUExecutionProvider should be available."
        return f"No {slot.vendor} CPU (this system has {fp.cpu.name})."
    return "Not available."
