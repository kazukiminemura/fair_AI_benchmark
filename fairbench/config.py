"""Run configuration.

Every knob that can change a number is in here, defaults are chosen to be
defensible, and the whole object is dumped into the result file. If two runs
differ, the difference is visible in the JSON rather than buried in code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunConfig:
    # ---- work selection ---------------------------------------------------
    backend: str = "all"  # all | ort | openvino | winml
    suite: str = "standard"
    workloads: list[str] = field(default_factory=list)  # empty => all
    include_int8: bool = True

    # ---- measurement policy ----------------------------------------------
    mode: str = "latency"  # latency | throughput
    requests: int = 0  # throughput: 0 sweeps 1/2/4/8; otherwise fixed concurrency
    warmup: int = 5
    min_iterations: int = 15
    max_iterations: int = 20000
    min_seconds: float = 4.0
    max_seconds: float = 45.0
    repeats: int = 3
    cooldown_seconds: float = 3.0

    # ---- execution policy -------------------------------------------------
    threads: int | None = None  # None => physical core count
    # Off by default, and deliberately so. Pinning ONNX Runtime's intra-op
    # threads to fixed logical CPUs measured *worse* than leaving them free on
    # the development machine (trimmed CV 9.4 % vs 7.2 % over 1300 iterations of
    # the decode workload), because a pinned pool also has to share those CPUs
    # with the main thread and with the OS. It stays available for machines
    # where the OS scheduler misbehaves, but it never quietly changes a number.
    thread_affinity: bool = False
    input_seed: int = 1337
    precision: str = "auto"  # auto | fp32 | fp16 | int8
    # Resolved from the host at run time and recorded, so a reader can see the
    # thread budget the CPU numbers were produced with.
    physical_cores: int = 0
    logical_cores: int = 0

    # ---- power / validity -------------------------------------------------
    measure_power: bool = True
    allow_battery: bool = False
    check_correctness: bool = True
    relative_tolerance: float = 0.05  # 5% on fp compare (covers fp16/int8 drift)

    def __post_init__(self) -> None:
        if self.backend not in ("all", "ort", "openvino", "winml"):
            raise ValueError("backend must be all, ort, openvino or winml")
        if self.mode not in ("latency", "throughput"):
            raise ValueError("mode must be latency or throughput")
        if not isinstance(self.requests, int) or not 0 <= self.requests <= 32:
            raise ValueError("requests must be an integer between 0 (auto) and 32")

    def resolve_threads(self) -> int:
        if self.threads and self.threads > 0:
            return self.threads
        # Physical cores, not SMT threads: ONNX Runtime's CPU EP gets slower when
        # it is given hyperthreads on matrix workloads, and the penalty differs
        # between Intel and AMD. Pinning both vendors to physical cores is the
        # only way the CPU comparison means anything.
        cores = self.physical_cores or (os.cpu_count() or 4) // 2
        return max(1, cores)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
