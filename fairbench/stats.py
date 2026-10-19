"""Measurement aggregation.

Fairness rule 2 lives here: every device is aggregated the same way. The
primary figure of merit is the **median** of the steady-state iteration
latencies, because it is robust to the one or two scheduler hiccups that any
laptop will produce, and it is not inflated by an unlucky first iteration.

We also publish ``cv`` (coefficient of variation) and a MAD-based outlier count
so a reader can tell a clean number from a noisy one.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field


def inference_samples(cfg, compute_samples):
    """Bounded sustained sampling; budget counts inference, not output validation."""
    started = time.perf_counter()
    minimum = 5 if cfg.quick else max(30, cfg.min_iterations)
    seconds = 0 if cfg.quick else max(0, cfg.min_seconds)
    limit = max(minimum, min(cfg.max_iterations, 2000))
    for index in range(limit):
        if index >= minimum and (sum(compute_samples) / 1000 >= seconds
                                 or (cfg.max_seconds and time.perf_counter() - started >= cfg.max_seconds)):
            break
        yield index


@dataclass
class StatSummary:
    count: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    cv: float  # stdev / mean, unitless; >0.05 means "noisy, treat with care"
    mad_ms: float
    outliers: int
    iqr_ms: float
    # Coefficient of variation after dropping the top and bottom 1 %. A single
    # OS scheduler hiccup on a 150-sample run can push the raw CV past 50 %
    # while the distribution itself is perfectly tight, so the validity grade
    # uses this one. The raw CV is still reported next to it.
    cv_trimmed: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy-compatible, no numpy needed)."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summarise(latencies_ms: list[float]) -> StatSummary:
    values = [v for v in latencies_ms if v is not None and math.isfinite(v)]
    n = len(values)
    if n == 0:
        return StatSummary(
            count=0,
            mean_ms=float("nan"),
            median_ms=float("nan"),
            p90_ms=float("nan"),
            p99_ms=float("nan"),
            min_ms=float("nan"),
            max_ms=float("nan"),
            stdev_ms=float("nan"),
            cv=float("nan"),
            mad_ms=float("nan"),
            outliers=0,
            iqr_ms=float("nan"),
            cv_trimmed=float("nan"),
        )
    ordered = sorted(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    stdev = math.sqrt(var)
    median = _percentile(ordered, 0.5)
    deviations = sorted(abs(v - median) for v in values)
    mad = _percentile(deviations, 0.5)
    # 3.5 * MAD is the standard robust outlier fence; scaled for normal data.
    outliers = 0
    if mad > 0:
        outliers = sum(1 for v in values if abs(v - median) / (1.4826 * mad) > 3.5)

    # Trimmed view: drop 1 % from each end (at least one sample each side).
    cut = max(1, n // 100)
    trimmed = ordered[cut : n - cut] if n - 2 * cut >= 3 else ordered
    t_mean = sum(trimmed) / len(trimmed)
    t_var = sum((v - t_mean) ** 2 for v in trimmed) / (len(trimmed) - 1) if len(trimmed) > 1 else 0.0
    cv_trimmed = (math.sqrt(t_var) / t_mean) if t_mean else float("nan")

    return StatSummary(
        count=n,
        mean_ms=mean,
        median_ms=median,
        p90_ms=_percentile(ordered, 0.90),
        p99_ms=_percentile(ordered, 0.99),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        stdev_ms=stdev,
        cv=stdev / mean if mean else float("nan"),
        mad_ms=mad,
        outliers=outliers,
        iqr_ms=_percentile(ordered, 0.75) - _percentile(ordered, 0.25),
        cv_trimmed=cv_trimmed,
    )


def median(values: list[float]) -> float:
    return _percentile(sorted(v for v in values if v is not None and math.isfinite(v)), 0.5)


def geometric_mean(values: list[float]) -> float:
    """Used for composite scores; a geometric mean stops one huge category
    (a 300 GFLOP/s GEMM) from drowning out everything else."""
    clean = [v for v in values if v and math.isfinite(v) and v > 0]
    if not clean:
        return float("nan")
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def ppm_error(measured: float, reference: float) -> float:
    """Relative difference in parts per million, used by the correctness gate."""
    if reference == 0:
        return 0.0 if measured == 0 else float("inf")
    return abs(measured - reference) / abs(reference) * 1e6


@dataclass
class RepeatSummary:
    """Aggregate of several independent repeats of the same workload.

    A laptop's throughput drifts as the chassis heats up, so the suite runs the
    same workload more than once with a cooldown in between and reports the
    spread. If the repeats disagree badly the run is flagged rather than
    silently averaged.
    """

    repeats: int
    per_repeat_median_ms: list[float] = field(default_factory=list)
    per_repeat_units_per_s: list[float] = field(default_factory=list)
    best_units_per_s: float = float("nan")
    median_units_per_s: float = float("nan")
    worst_units_per_s: float = float("nan")
    spread_pct: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def summarise_repeats(units_per_s: list[float], medians_ms: list[float]) -> RepeatSummary:
    clean = [v for v in units_per_s if math.isfinite(v) and v > 0]
    if not clean:
        return RepeatSummary(repeats=len(units_per_s))
    med = median(clean)
    return RepeatSummary(
        repeats=len(clean),
        per_repeat_median_ms=list(medians_ms),
        per_repeat_units_per_s=list(units_per_s),
        best_units_per_s=max(clean),
        median_units_per_s=med,
        worst_units_per_s=min(clean),
        spread_pct=(max(clean) - min(clean)) / med * 100 if len(clean) >= 2 and med else float("nan"),
    )
