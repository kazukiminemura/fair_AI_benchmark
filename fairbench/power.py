"""Power and CPU-cost sampling.

Two independent things are measured, and they matter for different reasons.

**Host CPU cost** (``process_time``) is always available, needs no drivers and
no privileges, and answers the question "how much host CPU does this offload
still burn?" - the real advantage of an AI PC is the offload freeing the CPU,
and for a badly dispatched NPU or GPU path that advantage can quietly disappear.

**System power** is optional and best-effort, and only ever measured from
instruments that are identical on Intel and AMD:

* the Windows **power meter** counter (instantaneous watts, present on many
  modern laptops) - the preferred source, 1 W resolution;
* the **battery discharge rate** counter (milliwatts) - fine grained enough for
  a per-workload window;
* the **battery charge percentage** - only usable for long runs, because 1 % of
  a 60 Wh battery is 36 s of runtime at 60 W. It is kept as a last resort and
  the report says so.

The sampler refuses to run when none of these exist. That is not laziness: the
sampling loop costs host CPU, and paying that cost on a CPU benchmark to
measure nothing would be a self-inflicted unfairness.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IS_WINDOWS = sys.platform.startswith("win")

PROBE_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$out = @()
try {
  $s = (Get-Counter -Counter '\Power Meter(*)\Power' -MaxSamples 1).CounterSamples
  if ($s) { $out += 'power_meter' }
} catch {}
try {
  $s = (Get-Counter -Counter '\Battery Status(*)\Discharge Rate' -MaxSamples 1).CounterSamples
  if ($s) { $out += 'battery_rate' }
} catch {}
$b = Get-CimInstance -ClassName Win32_Battery | Select-Object -First 1
if ($b) { $out += 'battery_percent' }
$out -join ','
"""

SAMPLER_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
while ($true) {
  $m = $null; $r = $null; $p = $null; $cap = $null
  try {
    $s = (Get-Counter -Counter '\Power Meter(*)\Power' -MaxSamples 1).CounterSamples
    if ($s) { $m = ($s | Measure-Object -Property CookedValue -Sum).Sum }
  } catch {}
  try {
    $s = (Get-Counter -Counter '\Battery Status(*)\Discharge Rate' -MaxSamples 1).CounterSamples
    if ($s) { $r = ($s | Measure-Object -Property CookedValue -Sum).Sum }
  } catch {}
  $b = Get-CimInstance -ClassName Win32_Battery | Select-Object -First 1
  if ($b) { $p = $b.EstimatedChargeRemaining; $cap = $b.FullChargeCapacity }
  "{0}|{1}|{2}|{3}" -f $m, $r, $p, $cap
  Start-Sleep -Milliseconds 1000
}
"""

PROBE_CACHE = Path(".cache") / "power_probe.json"


# --------------------------------------------------------------------------
def probe_power_sources(use_cache: bool = True) -> list[str]:
    """One cheap PowerShell call to learn which instruments exist.

    Run once by the parent process and handed to every worker, so the expensive
    probe never happens inside a timed region.
    """
    if not IS_WINDOWS:
        return []
    if use_cache and PROBE_CACHE.exists():
        try:
            cached = json.loads(PROBE_CACHE.read_text())
            if isinstance(cached, list):
                return cached
        except Exception:
            pass
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", PROBE_PS],
            capture_output=True,
            timeout=60,
        )
        text = (proc.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()
        sources = [s.strip() for s in (text[-1] if text else "").split(",") if s.strip()]
    except Exception:
        sources = []
    try:
        PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PROBE_CACHE.write_text(json.dumps(sources))
    except Exception:
        pass
    return sources


# --------------------------------------------------------------------------
@dataclass
class PowerSample:
    t: float
    meter_w: float | None
    battery_w: float | None
    percent: float | None
    capacity_mwh: float | None


@dataclass
class PowerSummary:
    available: bool = False
    method: str | None = None
    source: str | None = None
    average_watts: float | None = None
    samples: int = 0
    spread_watts: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "method": self.method,
            "source": self.source,
            "average_watts": self.average_watts,
            "spread_watts": self.spread_watts,
            "samples": self.samples,
            "note": self.note,
        }


class PowerSampler:
    """Background sampler. Start before the timed region, stop after it."""

    def __init__(self, sources: list[str] | None = None, enabled: bool = True) -> None:
        self.sources = list(sources or [])
        self.enabled = bool(enabled and IS_WINDOWS and self.sources)
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[PowerSample] = []
        self._stop = threading.Event()
        self.note = "" if self.enabled else "no power instrument on this machine"

    def __enter__(self) -> PowerSampler:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", SAMPLER_PS],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            self.note = f"could not start power sampler: {exc}"
            self.enabled = False
            return
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc, self._proc = self._proc, None
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _reader(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            meter = _to_float(parts[0])
            rate = _to_float(parts[1])
            self._samples.append(
                PowerSample(
                    t=time.perf_counter(),
                    meter_w=(meter / 1000.0) if (meter is not None and abs(meter) > 200) else meter,
                    # The counter is documented in milliwatts; values in the
                    # hundreds are watts reported by other driver versions.
                    battery_w=(rate / 1000.0) if (rate is not None and abs(rate) > 200) else rate,
                    percent=_to_float(parts[2]),
                    capacity_mwh=_to_float(parts[3]),
                )
            )

    # -- analysis -------------------------------------------------------
    def summary(self, window: tuple[float, float]) -> PowerSummary:
        if not self.enabled:
            return PowerSummary(available=False, note=self.note or "power sampling unavailable")
        t0, t1 = window
        inner = [s for s in self._samples if t0 <= s.t <= t1]
        if len(inner) < 2:
            return PowerSummary(available=False, samples=len(self._samples), note="not enough samples inside the timed window")

        meter = [s.meter_w for s in inner if s.meter_w and s.meter_w > 0]
        if len(meter) >= 2:
            return PowerSummary(
                available=True,
                method="power_meter",
                source="Windows Power Meter counter (whole system)",
                average_watts=sum(meter) / len(meter),
                spread_watts=max(meter) - min(meter),
                samples=len(meter),
            )

        rates = [s.battery_w for s in inner if s.battery_w and s.battery_w > 0]
        if len(rates) >= 2:
            return PowerSummary(
                available=True,
                method="battery_rate",
                source="battery discharge-rate counter (whole system)",
                average_watts=sum(rates) / len(rates),
                spread_watts=max(rates) - min(rates),
                samples=len(rates),
            )

        points = [(s.t, s.percent, s.capacity_mwh) for s in inner if s.percent is not None]
        if len(points) >= 3:
            capacity = next((p[2] for p in reversed(points) if p[2]), None)
            slope = _slope([(p[0], p[1]) for p in points])
            if capacity and slope < 0:
                return PowerSummary(
                    available=True,
                    method="battery_discharge",
                    source="battery charge percentage slope (whole system)",
                    average_watts=-slope / 100.0 * (capacity / 1000.0) * 3600.0,
                    samples=len(points),
                    note="1 % battery resolution: only trustworthy for runs of several minutes",
                )
            if slope >= 0:
                return PowerSummary(available=False, samples=len(points), note="battery is not discharging (plugged in)")

        return PowerSummary(available=False, samples=len(inner), note="no usable power reading in the timed window")


def _to_float(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _slope(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    num = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    den = sum((p[0] - mean_x) ** 2 for p in points)
    return num / den if den else 0.0


class HostCpuMeter:
    """Host CPU time consumed while a workload runs.

    ``process_time`` counts every thread in this process, so on a GPU or NPU run
    it reveals how much of the work was still done on the host. Free, exact, and
    identical on Intel and AMD.
    """

    def __enter__(self) -> HostCpuMeter:
        self._w0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, *exc: object) -> None:
        self.wall = time.perf_counter() - self._w0
        self.cpu = time.process_time() - self._c0
