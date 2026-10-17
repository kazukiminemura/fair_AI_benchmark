"""Regression gates for the numbers the dashboard and the CLI publish.

Run with::

    uv run --no-sync python -m scripts.check_dashboard_consistency

Both gates encode a promise the product already makes about itself:

1. ``fairbench/web/app.js`` METHOD says "Same warm-up, same time budget, same
   aggregation: the median, never the best case."  Two *separate* things must
   hold at once:

   * The published ``row["latency"]`` block is the **fastest** repeat's
     statistics (the README documents the distribution columns - p90/p99/CV/
     count - as one representative, fastest repeat), so a fix to the grade must
     not disturb it.
   * The validity grade is a reliability verdict, so its Timing-stability input
     is the **worst** ``cv_trimmed`` across every repeat of the run, and the
     iteration population its message names describes that same run.

2. A report that requested three repeats per cell must never explain a missing
   repeatability verdict with "single repeat per cell": it must name the
   requested and the successful repeat counts, surface the crash reasons in
   ``report["warnings"]``, and, when the index is built from fewer repeats than
   requested, say so on the index entry itself.

These are self-consistency gates, not new policy: the Timing-stability check
calls its own input the *worst* trimmed CV (``runner._grade``) and the
Repeatability check has the requested repeat count in ``report["config"]``.
"""

from __future__ import annotations

import re

from fairbench.config import RunConfig
from fairbench.runner import _aggregate, _grade, _score
from fairbench.system import PowerInfo, SystemFingerprint

FINGERPRINT = SystemFingerprint(power=PowerInfo())


def _cell(median_ms: float, cv_trimmed: float, iterations: int, units: float) -> dict:
    """One successful repeat of one (target, workload) cell."""
    return {
        "ok": True,
        "target_id": "ort:CPUExecutionProvider",
        "target_label": "ORT CPU",
        "kind": "cpu",
        "vendor": "intel",
        "backend": "ort",
        "device": "CPUExecutionProvider",
        "workload": "precision_gemm_fp16",
        "precision": "fp16",
        "model": {"flops": 1_000_000, "unit": "sample", "sha256": "0" * 64},
        "units_per_s": units,
        "iterations": iterations,
        "latency": {"median_ms": median_ms, "p90_ms": median_ms * 1.1, "cv": cv_trimmed,
                    "cv_trimmed": cv_trimmed, "count": iterations},
        "session": {"compute_precisions": ["float"]},
        "correctness": {"checked": True, "valid": True, "method": "deterministic_element_samples_v2"},
        "power": {"available": False},
    }


def _timing_stability(report: dict) -> dict:
    checks = _grade(report, FINGERPRINT, RunConfig(repeats=3))["checks"]
    return next(check for check in checks if check["name"] == "Timing stability")


def _graded_percent(detail: str) -> float | None:
    """The jitter percentage the Timing-stability message reports, if it names one."""
    match = re.search(r"(\d+(?:\.\d+)?)%", detail)
    return float(match.group(1)) if match else None


def _reported_iterations(detail: str) -> int | None:
    """The iteration population the Timing-stability message names, if it names one."""
    match = re.search(r"over ([\d,]+) iterations", detail)
    return int(match.group(1).replace(",", "")) if match else None


def _mentions_number(text: str, number: int) -> bool:
    """True when ``text`` contains ``number`` as a standalone count (not 0.1 or 13)."""
    return re.search(rf"(?<![\d.]){number}(?![\d.])", text) is not None


def _assert_grade_uses_worst_repeat(cells: list[dict], *, label: str) -> None:
    """Check one three-repeat cell against the corrected claims."""
    row = _aggregate(cells)[0]
    assert row["repeats_ok"] == len(cells), row["repeats_ok"]

    # (2) The published latency block must stay the fastest repeat's statistics.
    # This is the documented dashboard behaviour; correcting the grade must not
    # change what the table prints.
    fastest = min(cells, key=lambda cell: cell["latency"]["median_ms"])
    for key in ("median_ms", "p90_ms", "cv_trimmed", "count"):
        assert row["latency"][key] == fastest["latency"][key], (
            f"{label}: published latency.{key} is {row['latency'][key]!r}, but the fastest "
            f"repeat's is {fastest['latency'][key]!r}; the row must keep the fastest repeat's "
            "statistics (the documented dashboard behaviour)")

    per_repeat = [latency["cv_trimmed"] for latency in row["repeat_latencies"]]
    worst_index = max(range(len(per_repeat)), key=per_repeat.__getitem__)
    worst = per_repeat[worst_index]
    worst_iterations = row["repeat_latencies"][worst_index]["count"]
    total_iterations = sum(latency["count"] for latency in row["repeat_latencies"])
    assert worst != row["latency"]["cv_trimmed"], (
        f"{label}: the test data must separate the fastest repeat's jitter from the worst")

    report = {"results": [row], "cells": cells, "warnings": [], "config": {"repeats": 3}}
    check = _timing_stability(report)
    graded = _graded_percent(check["detail"])
    shown = f"{graded:.1f}%" if graded is not None else "no value"
    print(f"  [{label}] per-repeat trimmed CV {per_repeat}; graded {shown}; worst repeat {worst:.4f}")
    print(f"  [{label}] Timing stability: [{check['status']}] {check['detail']}")

    # (1) The grade's input is the run's worst repeat, never the best case.
    assert graded is not None and abs(graded - worst * 100) < 0.05, (
        f"{label}: the Timing-stability grade used {shown} instead of the run's worst repeat "
        f"{worst:.4f} ({worst * 100:.1f}%); the dashboard METHOD promises "
        "'the median, never the best case'")

    # (3) The iteration population the message names must describe the same run
    # the statistic was taken from: either the worst repeat's own iterations or
    # the whole run's. A "pass"/"warn" verdict always names one.
    reported = _reported_iterations(check["detail"])
    if check["status"] in ("pass", "warn"):
        assert reported is not None, (
            f"{label}: a {check['status']} verdict must name the iteration population its "
            f"jitter was measured over: {check['detail']!r}")
        assert reported in (worst_iterations, total_iterations), (
            f"{label}: the verdict names {reported:,} iterations, which is neither the "
            f"{worst:.1%} repeat's own {worst_iterations:,} nor the run's {total_iterations:,}; "
            "the statistic and its population must describe the same measurement")


def _check_graded_jitter_is_the_worst_repeat() -> None:
    # Two configurations of the same three-repeat cell. The slowest repeat is the
    # jitteriest and the fastest repeat is the quietest, so a grade taken from the
    # published (fastest) block would look better than the run really was.
    #
    # The 30 % case is the original red: 30 % jitter on the slowest repeat, which
    # the grade must treat as a failure. The 15 % case lands the verdict in the
    # "warn" band, whose message names the iteration population, so claim 3 is
    # actually exercised rather than skipped.
    _assert_grade_uses_worst_repeat(
        [_cell(median_ms=10.0, cv_trimmed=0.02, iterations=100, units=100.0),   # fastest, quietest
         _cell(median_ms=12.0, cv_trimmed=0.30, iterations=90, units=80.0),     # slowest, 30 % jitter
         _cell(median_ms=11.0, cv_trimmed=0.15, iterations=95, units=90.0)],
        label="30% worst")
    _assert_grade_uses_worst_repeat(
        [_cell(median_ms=10.0, cv_trimmed=0.02, iterations=100, units=100.0),
         _cell(median_ms=12.0, cv_trimmed=0.15, iterations=90, units=80.0),
         _cell(median_ms=11.0, cv_trimmed=0.08, iterations=95, units=90.0)],
        label="15% worst")
    print("Graded jitter comes from the worst repeat, its population matches, and the published block stays the fastest repeat's.")


def _entry_repeat_counts(entry: dict) -> set[int]:
    """The repeat counts an index entry exposes through repeat-named fields."""
    counts: set[int] = set()
    for key, value in entry.items():
        if "repeat" not in str(key).lower():
            continue
        for item in (value.values() if isinstance(value, dict) else [value]):
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                counts.add(item)
            elif isinstance(item, str):
                counts.update(int(number) for number in re.findall(r"\d+", item))
    return counts


def _check_index_repeat_coverage() -> None:
    # The CPU baseline was repeated three times; the GPU only survived once. Both
    # entries must say how many of the requested repeats fed the index.
    def row(target_id: str, kind: str, repeats_ok: int, units: float) -> dict:
        return {"ok": True, "target_id": target_id, "target_label": target_id, "kind": kind,
                "vendor": "unknown", "backend": "ort", "workload": "dense_gemm",
                "units_per_s": units, "units_per_joule": None,
                "repeats_requested": 3, "repeats_ok": repeats_ok}

    scores = _score({"results": [row("cpu", "cpu", 3, 10.0), row("gpu", "gpu", 1, 20.0)],
                     "config": {"mode": "latency"}})
    index = {entry["target_id"]: entry for entry in scores["ai_index"]}
    assert set(index) == {"cpu", "gpu"}, index.keys()
    for target_id, repeats_ok in (("cpu", 3), ("gpu", 1)):
        counts = _entry_repeat_counts(index[target_id])
        assert 3 in counts and repeats_ok in counts, (
            f"the {target_id} index entry does not report that {repeats_ok} of 3 requested "
            f"repeats fed the index: {index[target_id]!r}")


def _check_repeatability_message_matches_the_request() -> None:
    # The run asked for three repeats; two of them crashed. The surviving cell is
    # not a "single repeat per cell" configuration problem, and the crash reasons
    # must not vanish with the failed cells.
    ok = _cell(median_ms=10.0, cv_trimmed=0.02, iterations=100, units=100.0)
    crash = "worker crashed: ValueError: invalid GEMM search size"
    failed = {"ok": False, "target_id": ok["target_id"], "workload": ok["workload"],
              "error": crash, "phase": "spawn"}
    cells = [ok, dict(failed), dict(failed)]
    row = _aggregate(cells)[0]
    assert (row["repeats_ok"], row["repeats_requested"]) == (1, 3), row
    report = {"results": [row], "cells": cells, "warnings": [], "power_sources": [],
              "config": {"repeats": 3}}
    check = next(c for c in _grade(report, FINGERPRINT, RunConfig(repeats=3))["checks"]
                 if c["name"] == "Repeatability")
    print(f"  config.repeats={report['config']['repeats']} vs the verdict above")
    print(f"  Repeatability: [{check['status']}] {check['detail']}")
    assert "single repeat per cell" not in check["detail"], (
        "the run requested 3 repeats and 2 of them failed; the verdict blames the "
        "configuration instead of the failed repeats")
    # (1) The verdict must name how many of the requested repeats survived.
    assert _mentions_number(check["detail"], 3) and _mentions_number(check["detail"], 1), (
        f"the verdict does not state how many of the 3 requested repeats survived: "
        f"{check['detail']!r}")

    # (2) The crash reason must be aggregated into the report's warnings, not lost
    # with the failed cells.
    assert any(crash in warning for warning in report["warnings"]), (
        f"the crash reason {crash!r} never reached report['warnings']: {report['warnings']!r}")

    # (3) The index must say when it was built from fewer repeats than requested.
    _check_index_repeat_coverage()
    print("Repeatability explains the missing repeats, surfaces the crash, and the index reports its repeat coverage.")


if __name__ == "__main__":
    _check_graded_jitter_is_the_worst_repeat()
    _check_repeatability_message_matches_the_request()
