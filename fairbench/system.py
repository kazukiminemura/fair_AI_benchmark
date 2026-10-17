"""System fingerprint.

Collects everything needed to decide whether two benchmark runs are actually
comparable: OS build, CPU, memory, GPU/NPU adapters, driver versions, power
source and the active Windows power plan.

Anything we cannot read is recorded as ``None`` rather than skipped, so the
report always shows what was *not* known.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

IS_WINDOWS = sys.platform.startswith("win")


# --------------------------------------------------------------------------
# low level helpers
# --------------------------------------------------------------------------
def _decode(raw: bytes) -> str:
    """Windows console tools emit OEM/ANSI bytes; try UTF-8 then the local codepage."""
    for enc in ("utf-8-sig", "utf-8", "cp932", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _run(args: list[str], timeout: int = 25) -> str:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return _decode(proc.stdout or b"") + _decode(proc.stderr or b"")
    except Exception:
        return ""


def _powershell_json(script: str, timeout: int = 30) -> Any:
    """Run a PowerShell snippet and parse its JSON output. Returns None on failure."""
    if not IS_WINDOWS:
        return None
    raw = _run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + script,
        ],
        timeout=timeout,
    )
    raw = raw.strip()
    if not raw:
        return None
    # PowerShell emits warnings before JSON sometimes; keep the last JSON block.
    start = min(
        [i for i in (raw.find("{"), raw.find("[")) if i >= 0] or [-1]
    )
    if start > 0:
        raw = raw[start:]
    try:
        return json.loads(raw)
    except Exception:
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class CpuInfo:
    name: str = "unknown"
    vendor: str = "unknown"  # intel | amd | other
    family: str | None = None
    physical_cores: int = 0
    logical_cores: int = 0
    base_clock_mhz: int | None = None
    current_clock_mhz: int | None = None
    l3_cache_kb: int | None = None


@dataclass
class MemoryInfo:
    total_gib: float | None = None
    speed_mts: int | None = None
    modules: list[dict[str, Any]] = field(default_factory=list)
    bandwidth: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterInfo:
    name: str
    vendor: str  # intel | amd | nvidia | qualcomm | microsoft | unknown
    kind: str  # gpu | npu | unknown
    driver: str | None = None
    memory_mib: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerInfo:
    power_plan: str | None = None
    on_ac: bool | None = None
    battery_present: bool = False
    battery_percent: float | None = None
    battery_full_wh: float | None = None
    battery_design_wh: float | None = None


@dataclass
class SystemFingerprint:
    schema: str = "fairbench.system/1"
    hostname: str = ""
    os_name: str = ""
    os_release: str = ""
    os_version: str = ""
    os_build: str = ""
    arch: str = ""
    python: str = ""
    cpu: CpuInfo = field(default_factory=CpuInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    adapters: list[AdapterInfo] = field(default_factory=list)
    power: PowerInfo = field(default_factory=PowerInfo)
    scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- convenience used by the scorer / UI -----------------------------
    @property
    def vendor(self) -> str:
        return self.cpu.vendor

    def find_adapter(self, kind: str, vendor: str | None = None) -> AdapterInfo | None:
        for a in self.adapters:
            if a.kind != kind:
                continue
            if vendor and a.vendor != vendor:
                continue
            return a
        return None


# --------------------------------------------------------------------------
# vendor detection
# --------------------------------------------------------------------------
def classify_vendor(text: str) -> str:
    t = (text or "").lower()
    if "intel" in t or "genuineintel" in t:
        return "intel"
    if "amd" in t or "authenticamd" in t or "ryzen" in t or "radeon" in t:
        return "amd"
    if "nvidia" in t or "geforce" in t or "quadro" in t or "rtx" in t:
        return "nvidia"
    if "qualcomm" in t or "snapdragon" in t or "hexagon" in t or "adreno" in t:
        return "qualcomm"
    if "microsoft" in t or "basic" in t or "warp" in t:
        return "microsoft"
    return "unknown"


_NPU_RE = re.compile(
    r"\bnpu\b|\bvpu\b|ai boost|neural processor|neural engine|compute accelerator"
    r"|hexagon|\bxdna\b|ryzen ai|ai engine|snapdragon",
    re.IGNORECASE,
)
_GPU_RE = re.compile(
    r"geforce|radeon|\barc\b|iris|\buhd graphics\b|graphics|\brtx\b|\bgpu\b|adreno|vega",
    re.IGNORECASE,
)


def classify_kind(name: str) -> str:
    t = name or ""
    if _NPU_RE.search(t):
        return "npu"
    if _GPU_RE.search(t):
        return "gpu"
    return "unknown"


# --------------------------------------------------------------------------
# collectors
# --------------------------------------------------------------------------
def _collect_cpu() -> CpuInfo:
    info = CpuInfo()
    info.name = platform.processor() or "unknown"
    if IS_WINDOWS:
        data = _powershell_json(
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
            "MaxClockSpeed,CurrentClockSpeed,L3CacheSize,Manufacturer | ConvertTo-Json -Compress"
        )
        rows = _as_list(data)
        if rows:
            r = rows[0] or {}
            info.name = (r.get("Name") or info.name).strip()
            info.physical_cores = int(r.get("NumberOfCores") or 0)
            info.logical_cores = int(r.get("NumberOfLogicalProcessors") or 0)
            info.base_clock_mhz = r.get("MaxClockSpeed")
            info.current_clock_mhz = r.get("CurrentClockSpeed")
            info.l3_cache_kb = r.get("L3CacheSize")
    if not info.logical_cores:
        info.logical_cores = os.cpu_count() or 0
    if not info.physical_cores:
        info.physical_cores = max(1, info.logical_cores // 2)
    info.vendor = classify_vendor(info.name)
    return info


def _collect_memory() -> MemoryInfo:
    mem = MemoryInfo()
    if IS_WINDOWS:
        data = _powershell_json(
            "Get-CimInstance Win32_PhysicalMemory | "
            "Select-Object Capacity,Speed,ConfiguredClockSpeed,DataWidth,DeviceLocator,Manufacturer,PartNumber | ConvertTo-Json -Compress"
        )
        total = 0
        speeds: list[int] = []
        for r in _as_list(data):
            r = r or {}
            cap = r.get("Capacity") or 0
            total += int(cap)
            speed = r.get("ConfiguredClockSpeed") or r.get("Speed")
            if speed:
                speeds.append(int(speed))
            mem.modules.append(
                {
                    "capacity_gib": round(int(cap) / 1024**3, 1) if cap else None,
                    "speed_mts": speed,
                    "data_width_bits": r.get("DataWidth") if r.get("DataWidth") not in (None, 0, 65535) else None,
                    "slot": r.get("DeviceLocator"),
                    "part": (r.get("PartNumber") or "").strip() or None,
                }
            )
        if total:
            mem.total_gib = round(total / 1024**3, 1)
        if speeds:
            mem.speed_mts = max(speeds)
    if mem.total_gib is None:
        try:
            import psutil  # type: ignore

            mem.total_gib = round(psutil.virtual_memory().total / 1024**3, 1)
        except Exception:
            pass
    return mem


def measure_memory_bandwidth(mem: MemoryInfo) -> None:
    """Measure host RAM copy traffic (read + write), not NPU DMA or peak.

    Called before the inference suite, never during discovery or inference.
    Sizes and sample rates are recorded for reproducibility.
    """
    if mem.bandwidth:
        return
    import numpy as np

    size = max(1024**2, min(256 * 1024**2, int((mem.total_gib or 2) * 1024**3 / 32)))
    try:
        source = np.ones(size, dtype=np.uint8)
        destination = np.empty_like(source)
        np.copyto(destination, source)
        rates = []
        for _ in range(5):
            started = time.perf_counter()
            np.copyto(destination, source)
            elapsed = time.perf_counter() - started
            rates.append(2 * size / elapsed / 1e9)
        mem.bandwidth = {"available": True, "gb_per_s": float(np.median(rates)),
                         "method": "numpy.copyto: host RAM read + write; not NPU bandwidth",
                         "buffer_bytes": size, "traffic_bytes_per_copy": 2 * size,
                         "samples_gb_per_s": rates, "samples": len(rates)}
    except (MemoryError, ValueError, RuntimeError) as exc:
        mem.bandwidth = {"available": False, "error": str(exc)}


def _collect_adapters() -> list[AdapterInfo]:
    found: list[AdapterInfo] = []
    if IS_WINDOWS:
        gpus = _powershell_json(
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,DriverVersion,AdapterRAM,VideoProcessor | ConvertTo-Json -Compress"
        )
        for r in _as_list(gpus):
            r = r or {}
            name = (r.get("Name") or "").strip()
            if not name:
                continue
            vram = r.get("AdapterRAM")
            found.append(
                AdapterInfo(
                    name=name,
                    vendor=classify_vendor(name),
                    kind=classify_kind(name) if classify_kind(name) != "unknown" else "gpu",
                    driver=r.get("DriverVersion"),
                    memory_mib=round(int(vram) / 1024**2) if vram else None,
                )
            )

        # NPUs show up as PnP devices, not as video controllers. The pattern is
        # deliberately word-bounded: a loose "ipu"/"npu" match also hits
        # "USB Input Device".
        pnp = _powershell_json(
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.Name -match '\\bNPU\\b|\\bVPU\\b|AI Boost|Neural Processor|Neural Engine|Compute Accelerator|Hexagon|\\bXDNA\\b|Ryzen AI|AI Engine' } | "
            "Select-Object Name,Manufacturer,DeviceID | ConvertTo-Json -Compress"
        )
        for r in _as_list(pnp):
            r = r or {}
            name = (r.get("Name") or "").strip()
            if not name:
                continue
            if any(a.name == name for a in found):
                continue
            found.append(
                AdapterInfo(
                    name=name,
                    vendor=classify_vendor(name + " " + (r.get("Manufacturer") or "")),
                    kind="npu",
                    driver=None,
                    extra={"device_id": r.get("DeviceID")},
                )
            )
    return found


KNOWN_POWER_PLANS = {
    "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
    "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
    "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate performance",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Ultimate performance",
}


def _parse_power_plan(raw: str) -> str | None:
    """Normalise ``powercfg /getactivescheme`` into a locale-independent label."""
    if not raw.strip():
        return None
    line = raw.strip().splitlines()[0]
    match = re.search(
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        line,
    )
    if not match:
        return line
    guid = match.group(1).lower()
    label = KNOWN_POWER_PLANS.get(guid, "Custom")
    return f"{label} ({guid})"


def _collect_power() -> PowerInfo:
    p = PowerInfo()
    if not IS_WINDOWS:
        return p
    p.power_plan = _parse_power_plan(_run(["powercfg", "/getactivescheme"]))
    bat = _powershell_json(
        "Get-CimInstance Win32_Battery | "
        "Select-Object BatteryStatus,EstimatedChargeRemaining,DesignCapacity,FullChargeCapacity | ConvertTo-Json -Compress"
    )
    rows = _as_list(bat)
    if rows:
        r = rows[0] or {}
        p.battery_present = True
        status = r.get("BatteryStatus")
        # 2 = AC connected, 1 = discharging, 3-5 = fully charged on AC, 6-9 charging
        p.on_ac = status in (2, 3, 6, 7, 8, 9) if status is not None else None
        p.battery_percent = r.get("EstimatedChargeRemaining")
        if r.get("DesignCapacity"):
            p.battery_design_wh = round(int(r["DesignCapacity"]) / 1000, 1)
        if r.get("FullChargeCapacity"):
            p.battery_full_wh = round(int(r["FullChargeCapacity"]) / 1000, 1)
    elif shutil.which("powercfg"):
        # No WMI battery => almost certainly a desktop / VM.
        p.on_ac = True
    return p


def probe_background_load(samples: int = 3) -> float | None:
    """Average total CPU utilisation *before* the benchmark starts.

    Background load is the most common reason two laptops with identical
    silicon disagree by 30 %. Measuring it takes three seconds, once, outside
    every timed region, so it cannot influence the numbers it explains.
    """
    if not IS_WINDOWS:
        return None
    script = (
        f"$s = (Get-Counter -Counter '\\Processor(_Total)\\% Processor Time' "
        f"-MaxSamples {int(samples)} -SampleInterval 1).CounterSamples | "
        "Measure-Object -Property CookedValue -Average; "
        "if ($s) { [math]::Round($s.Average, 2) } else { '' }"
    )
    raw = _run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script], timeout=30)
    text = raw.strip().splitlines()[-1].strip() if raw.strip() else ""
    try:
        return float(text)
    except ValueError:
        return None


def collect() -> SystemFingerprint:
    version = platform.version()
    fp = SystemFingerprint(
        hostname=platform.node(),
        os_name=platform.system(),
        os_release=platform.release(),
        os_version=version,
        os_build=version.rsplit(".", 1)[-1] if "." in version else version,
        arch=platform.machine(),
        python=platform.python_version(),
    )
    fp.cpu = _collect_cpu()
    fp.memory = _collect_memory()
    fp.adapters = _collect_adapters()
    fp.power = _collect_power()
    return fp


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(collect().to_dict(), indent=2, ensure_ascii=False))
