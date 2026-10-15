"""Suite orchestration.

The parent process never runs a model itself. It decides *what* to measure,
dispatches each cell to a fresh child process, aggregates the repeats and
grading, and writes one self-describing JSON file.

Order of operations, and why:

1. **Fingerprint the machine** before anything runs, so the numbers carry the
   hardware they came from.
2. **Verify every target** with a sub-millisecond probe in a child process. A
   device that hangs or aborts is dropped with a reason instead of taking the
   suite down.
3. **Measure** each (target, workload, precision) ``repeats`` times with a
   cooldown between repeats, alternating targets so that a thermal drift hits
   every device equally.
4. **Grade** the run: power source, power plan, idle noise, repeat spread.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__, models, paths, system
from .config import RunConfig
from .devices import Discovery, Target, discover
from .power import probe_power_sources as power_probe
from .stats import geometric_mean, median, summarise_repeats

JOB_DIR = Path(".cache") / "jobs"

ProgressFn = Callable[[dict[str, Any]], None]


# --------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------
DEFAULT_WORKLOADS = [
    "llm_decode",
    "dense_gemm",
    "depthwise_mobile",
    "conv_vision",
    "transformer",
    "llm_prefill",
]

WORKLOAD_META: dict[str, dict[str, str]] = {
    "dense_gemm": {"category": "dense", "blurb": "Wide MLP; pure matrix throughput. The NPU's home turf."},
    "conv_vision": {"category": "vision", "blurb": "Convolutional backbone; classic image inference."},
    "depthwise_mobile": {"category": "vision", "blurb": "MobileNet-style depthwise CNN; what NPUs are specified for."},
    "transformer": {"category": "nlp", "blurb": "Transformer encoder; NLP representation workloads."},
    "llm_prefill": {"category": "generative", "blurb": "LLM prefill of 192 tokens; compute bound, drives TTFT."},
    "llm_decode": {"category": "generative", "blurb": "Single-token LLM decode with a 768-token KV cache; bandwidth bound."},
    "dense_gemm_int8": {"category": "quantised", "blurb": "Same MLP with INT8 weights; the format NPUs natively run."},
    "transformer_int8": {"category": "quantised", "blurb": "INT8 transformer encoder."},
    "llm_decode_int8": {"category": "quantised", "blurb": "INT8 LLM decode; the realistic edge-model profile."},
}


def resolve_workloads(cfg: RunConfig, requested: list[str] | None = None) -> list[str]:
    names = requested or cfg.workloads or list(DEFAULT_WORKLOADS)
    if "all" in names:
        names = list(models.BASE_BUILDERS)
    out: list[str] = []
    for name in names:
        if name in models.BASE_BUILDERS:
            if name not in out:
                out.append(name)
        elif name.endswith("_int8") and name[: -len("_int8")] in models.BASE_BUILDERS:
            if name not in out:
                out.append(name)
        else:
            raise ValueError(f"unknown workload '{name}'")
    if cfg.include_int8:
        for name in list(out):
            if models.has_int8(name) and f"{name}_int8" not in out:
                out.append(f"{name}_int8")
    return out


def precision_of(workload: str) -> str:
    return "int8" if workload.endswith("_int8") else "fp32"


# --------------------------------------------------------------------------
# child process plumbing
# --------------------------------------------------------------------------
def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(paths.project_root())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _worker_command(python: str, job_path: Path, out_path: Path) -> list[str]:
    """How to launch a measurement cell.

    In a packaged build the "interpreter" is the executable itself, so the
    module entry point is replaced by a hidden ``--worker`` flag. A device that
    lives in a *real* external interpreter (a DirectML venv) still uses
    ``-m fairbench.worker``, which is why the packaged build ships the source.
    """
    if paths.is_own_interpreter(python) and paths.FROZEN:
        return [python, "--worker", str(job_path), str(out_path)]
    return [python, "-X", "faulthandler", "-m", "fairbench.worker", str(job_path), str(out_path)]


def _spawn_worker(job: dict[str, Any], timeout: float, tag: str) -> dict[str, Any]:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    # Absolute paths: the child may run with a different working directory.
    job_path = (JOB_DIR / f"{tag}.job.json").resolve()
    out_path = (JOB_DIR / f"{tag}.out.json").resolve()
    job_path.write_text(json.dumps(job))
    if out_path.exists():
        out_path.unlink()

    python = job.get("target", {}).get("python") or sys.executable
    if paths.FROZEN and not paths.is_own_interpreter(python) and not paths.worker_module_available():
        return {
            "ok": False,
            "target_id": job["target"]["id"],
            "workload": job["workload"],
            "error": (
                f"this device needs the external interpreter {python}, but the packaged build "
                "cannot find the fairbench worker source next to the executable"
            ),
            "phase": "spawn",
        }
    cmd = _worker_command(python, job_path, out_path)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            env=_worker_env(),
            capture_output=True,
            timeout=timeout,
        )
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        returncode = proc.returncode
    except FileNotFoundError:
        return {
            "ok": False,
            "target_id": job["target"]["id"],
            "workload": job["workload"],
            "error": f"interpreter for this device is missing: {python}",
            "phase": "spawn",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "target_id": job["target"]["id"],
            "workload": job["workload"],
            "error": f"device did not respond within {timeout:.0f}s (hung or extremely slow)",
            "phase": "timeout",
            "elapsed_s": time.perf_counter() - started,
        }

    if out_path.exists():
        try:
            result = json.loads(out_path.read_text())
            result.setdefault("worker_returncode", returncode)
            if not result.get("ok") and not result.get("error"):
                result["error"] = _crash_hint(returncode, stderr)
            return result
        except Exception as exc:
            return {
                "ok": False,
                "target_id": job["target"]["id"],
                "workload": job["workload"],
                "error": f"could not parse worker output: {exc}",
                "phase": "parse",
            }
    return {
        "ok": False,
        "target_id": job["target"]["id"],
        "workload": job["workload"],
        "error": _crash_hint(returncode, stderr),
        "phase": "crash",
        "elapsed_s": time.perf_counter() - started,
    }


def _crash_hint(returncode: int | None, stderr: str) -> str:
    if returncode is not None and returncode < 0:
        codes = {
            -1073741819: "access violation (0xC0000005)",
            -1073740791: "stack buffer overrun / abort (0xC0000409)",
            -1073741571: "stack overflow (0xC00000FD)",
        }
        base = f"device driver aborted the process: {codes.get(returncode, hex(returncode & 0xFFFFFFFF))}"
    else:
        base = f"worker exited with code {returncode}"
    tail = "\n".join(line for line in stderr.strip().splitlines()[-4:] if line.strip())
    return f"{base}\n{tail}" if tail else base


# --------------------------------------------------------------------------
# target verification
# --------------------------------------------------------------------------
def verify_targets(targets: list[Target], cfg: RunConfig, progress: ProgressFn | None = None) -> dict[str, dict]:
    """Prove each device can execute a trivial graph before committing to the suite."""
    probe_cfg = RunConfig(
        warmup=1,
        min_iterations=3,
        max_iterations=3,
        min_seconds=0.0,
        max_seconds=10.0,
        repeats=1,
        check_correctness=False,
        measure_power=False,
        physical_cores=cfg.physical_cores,
        logical_cores=cfg.logical_cores,
    )
    verdicts: dict[str, dict] = {}
    for target in targets:
        job = {
            "target": target.to_dict(),
            "workload": "smoke",
            "precision": "fp32",
            "cfg": probe_cfg.to_dict(),
            "repeat_index": 0,
            "power_sources": [],
        }
        result = _spawn_worker(job, timeout=180.0, tag=f"verify-{_slug(target.id)}")
        verdicts[target.id] = {
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
            "compile_ms": result.get("compile_ms"),
            "latency_ms": (result.get("latency") or {}).get("median_ms"),
        }
        if progress:
            progress(
                {
                    "event": "verify",
                    "target": target.id,
                    "ok": bool(result.get("ok")),
                    "error": result.get("error"),
                }
            )
    return verdicts


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text)[:80]


# --------------------------------------------------------------------------
# the suite
# --------------------------------------------------------------------------
def run_suite(
    cfg: RunConfig,
    target_ids: list[str] | None = None,
    workloads: list[str] | None = None,
    *,
    verify: bool = True,
    progress: ProgressFn | None = None,
    fingerprint: system.SystemFingerprint | None = None,
    discovery: Discovery | None = None,
) -> dict[str, Any]:
    started = time.time()
    report_warnings: list[str] = []
    fp = fingerprint or system.collect()
    cfg.physical_cores = fp.cpu.physical_cores
    cfg.logical_cores = fp.cpu.logical_cores

    disc = discovery or discover(fp)
    all_targets = disc.available()
    if target_ids:
        wanted = set(target_ids)
        targets = [t for t in all_targets if t.id in wanted or t.device in wanted]
        missing = wanted - {t.id for t in targets} - {t.device for t in targets}
        if missing:
            raise SystemExit(f"unknown or unavailable target(s): {sorted(missing)}")
    else:
        targets = all_targets

    workload_names = resolve_workloads(cfg, workloads)

    # Pre-flight: build every planned graph once, here, in the parent. A graph
    # that cannot be built (a packaged build without the ONNX Runtime quantiser,
    # for example) is dropped now with a reason, instead of being retried - and
    # failing - in every child process of every target.
    buildable: list[str] = []
    for name in workload_names:
        why = models.describe_build_error(name)
        if why is None:
            buildable.append(name)
        else:
            report_warnings.append(f"workload '{name}' was dropped: {why}")
    workload_names = buildable

    # Probe once, here, and hand the answer to every worker: the probe costs a
    # PowerShell start-up, and doing it inside a timed region would be unfair.
    power_sources = power_probe() if cfg.measure_power else []
    if not power_sources:
        report_warnings.append(
            "no power instrument detected (Windows Power Meter or battery counters); "
            "energy per inference will be unavailable"
        )

    report: dict[str, Any] = {
        "schema": "fairbench.result/1",
        "tool": {"name": "fairbench", "version": __version__},
        "started_at": started,
        "finished_at": None,
        "duration_s": None,
        "config": cfg.to_dict(),
        "system": fp.to_dict(),
        "discovery": disc.to_dict(),
        "workloads": [
            {
                "key": w,
                "precision": precision_of(w),
                "int8_variant": w.endswith("_int8"),
                **WORKLOAD_META.get(w, {"category": "other", "blurb": ""}),
            }
            for w in workload_names
        ],
        "targets": [],
        "cells": [],
        "results": [],
        "scores": {},
        "grade": {},
        "warnings": [*disc.notes, *report_warnings],
        "power_sources": power_sources,
    }

    # ---- verify -------------------------------------------------------
    verdicts = verify_targets(targets, cfg, progress) if verify else {t.id: {"ok": True} for t in targets}
    runnable: list[Target] = []
    for target in targets:
        verdict = verdicts.get(target.id, {})
        entry = target.to_dict()
        entry["verified"] = bool(verdict.get("ok"))
        entry["verify_error"] = verdict.get("error")
        entry["verify_latency_ms"] = verdict.get("latency_ms")
        report["targets"].append(entry)
        if verdict.get("ok"):
            runnable.append(target)
        else:
            report["warnings"].append(
                f"{target.label} ({target.id}) was dropped before measurement: {verdict.get('error')}"
            )

    # ---- measure ------------------------------------------------------
    # Interleave the repeats across targets so that a warming chassis or a
    # background update hits every device, not just the one measured last.
    schedule: list[tuple[int, Target, str]] = []
    for repeat in range(max(1, cfg.repeats)):
        for workload in workload_names:
            for target in runnable:
                schedule.append((repeat, target, workload))

    total = len(schedule)

    # Sample the machine's idle state once, after verification and before any
    # timed region, so the report can say whether a background process stole
    # the numbers. Costs ~3 s and cannot influence what it explains.
    background_load: float | None = None
    if total and not getattr(cfg, "_skip_load_probe", False):
        background_load = system.probe_background_load()
        report["background_load_percent"] = background_load
    if total == 0:
        report["warnings"].append("no runnable targets - nothing was measured")

    for index, (repeat, target, workload) in enumerate(schedule, start=1):
        precision = precision_of(workload)
        job = {
            "target": target.to_dict(),
            "workload": workload,
            "precision": precision,
            "cfg": cfg.to_dict(),
            "repeat_index": repeat,
            "power_sources": power_sources,
        }
        tag = f"{_slug(target.id)}--{workload}--r{repeat}"
        timeout = cfg.max_seconds + 420.0
        result = _spawn_worker(job, timeout=timeout, tag=tag)
        result["repeat_index"] = repeat
        result["precision"] = precision
        report["cells"].append(result)

        if progress:
            progress(
                {
                    "event": "cell",
                    "index": index,
                    "total": total,
                    "repeat": repeat,
                    "target": target.id,
                    "target_label": target.label,
                    "vendor": target.vendor,
                    "kind": target.kind,
                    "workload": workload,
                    "ok": bool(result.get("ok")),
                    "units_per_s": result.get("units_per_s"),
                    "median_ms": (result.get("latency") or {}).get("median_ms"),
                    "error": result.get("error"),
                }
            )

        if cfg.cooldown_seconds and index < total:
            time.sleep(cfg.cooldown_seconds)

    # ---- aggregate ----------------------------------------------------
    report["results"] = _aggregate(report["cells"])
    report["scores"] = _score(report)
    report["grade"] = _grade(report, fp, cfg)
    report["finished_at"] = time.time()
    report["duration_s"] = report["finished_at"] - started

    if progress:
        progress({"event": "done", "report": _slim(report)})
    return report


# --------------------------------------------------------------------------
def _aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for cell in cells:
        key = (cell.get("target_id") or "?", cell.get("workload") or "?")
        groups.setdefault(key, []).append(cell)

    rows: list[dict[str, Any]] = []
    for (target_id, workload), items in groups.items():
        good = [c for c in items if c.get("ok")]
        base = (items[0] if items else {}) or {}
        row: dict[str, Any] = {
            "target_id": target_id,
            "workload": workload,
            "vendor": base.get("vendor"),
            "kind": base.get("kind"),
            "backend": base.get("backend"),
            "device": base.get("device"),
            "target_label": base.get("target_label"),
            "precision": base.get("precision") or precision_of(workload),
            "repeats_requested": len(items),
            "repeats_ok": len(good),
        }
        if not good:
            row.update(
                {
                    "ok": False,
                    "error": base.get("error") or "all repeats failed",
                    "phase": base.get("phase"),
                }
            )
            rows.append(row)
            continue

        units = [c["units_per_s"] for c in good if c.get("units_per_s")]
        medians = [(c.get("latency") or {}).get("median_ms") for c in good if (c.get("latency") or {}).get("median_ms")]
        lat = [c["latency"] for c in good if c.get("latency")]
        rep = summarise_repeats(units, medians)
        model_info = base.get("model") or {}
        host_cpu = [c.get("host_cpu_seconds_per_inference") for c in good if c.get("host_cpu_seconds_per_inference")]
        power = [c.get("power") or {} for c in good]
        power_ok = [p for p in power if p.get("available")]

        best_latency = min(lat, key=lambda s: s["median_ms"]) if lat else {}
        row.update(
            {
                "ok": True,
                "unit": model_info.get("unit", "sample"),
                "model_sha256": model_info.get("sha256"),
                "model_size_bytes": model_info.get("size_bytes"),
                "flops": model_info.get("flops"),
                "compile_ms": median([c["compile_ms"] for c in good if c.get("compile_ms") is not None]),
                "latency_ms": median(medians) if medians else None,
                "iterations": sum(int(c.get("iterations") or 0) for c in good),
                "latency": best_latency,
                "units_per_s": rep.median_units_per_s,
                "units_per_s_best": rep.best_units_per_s,
                "units_per_s_worst": rep.worst_units_per_s,
                "repeat_spread_pct": rep.spread_pct,
                "flops_per_s": (model_info.get("flops") or 0) * rep.median_units_per_s
                if rep.median_units_per_s == rep.median_units_per_s
                else None,
                "host_cpu_seconds_per_inference": median(host_cpu) if host_cpu else None,
                "repeat_latencies": [c["latency"] for c in good],
                "repeats": rep.to_dict(),
            }
        )
        if power_ok:
            watts = [p["average_watts"] for p in power_ok if p.get("average_watts")]
            row["power"] = {
                "average_watts": median(watts) if watts else None,
                "method": power_ok[0].get("method"),
                "source": power_ok[0].get("source"),
                "note": power_ok[0].get("note", ""),
            }
            if watts and row["units_per_s"]:
                row["units_per_joule"] = row["units_per_s"] / median(watts)
        rows.append(row)

    rows.sort(key=lambda r: (r["workload"], r["kind"], r["vendor"], r["target_id"]))
    return rows


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _score(report: dict[str, Any]) -> dict[str, Any]:
    """Normalise each workload against this machine's own CPU baseline.

    Absolute throughput is the honest cross-machine number and it is right there
    in every row. The index exists to answer a different question - "does the
    accelerator actually pay for itself on this laptop?" - and it is computed
    as a geometric mean so that a 300 GFLOP/s GEMM cannot drown out a decoder.
    Ratios are taken against the *same machine's* CPU EP result, which keeps the
    index immune to whatever CPU the laptop happens to have.
    """
    rows = [r for r in report["results"] if r.get("ok")]
    cpus = [r for r in rows if r.get("kind") == "cpu" and r.get("backend") == "ort"]
    if not cpus:
        cpus = [r for r in rows if r.get("kind") == "cpu"]
    baseline = {r["workload"]: r["units_per_s"] for r in cpus if r.get("units_per_s")}
    if not baseline:
        return {"baseline": None, "note": "no CPU baseline was measured; relative index unavailable"}

    per_target: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        base = baseline.get(row["workload"])
        if not base or not row.get("units_per_s"):
            continue
        per_target.setdefault(row["target_id"], []).append((row["workload"], row["units_per_s"] / base))

    index: list[dict[str, Any]] = []
    for target_id, pairs in per_target.items():
        sample = next(r for r in rows if r["target_id"] == target_id)
        index.append(
            {
                "target_id": target_id,
                "label": sample.get("target_label"),
                "vendor": sample.get("vendor"),
                "kind": sample.get("kind"),
                "ai_index": geometric_mean([v for _, v in pairs]),
                "workloads": len(pairs),
            }
        )
    index.sort(key=lambda r: -(r["ai_index"] or 0))

    efficiency: list[dict[str, Any]] = []
    cpu_ai = next((i["ai_index"] for i in index if i["kind"] == "cpu"), None)
    per_target_w: dict[str, list[float]] = {}
    for row in rows:
        if row.get("units_per_joule"):
            per_target_w.setdefault(row["target_id"], []).append(row["units_per_joule"])
    for target_id, values in per_target_w.items():
        sample = next(r for r in rows if r["target_id"] == target_id)
        efficiency.append(
            {
                "target_id": target_id,
                "label": sample.get("target_label"),
                "vendor": sample.get("vendor"),
                "kind": sample.get("kind"),
                "units_per_joule": geometric_mean(values),
                "workloads": len(values),
            }
        )
    efficiency.sort(key=lambda r: -(r["units_per_joule"] or 0))

    return {
        "baseline": {
            "target_id": cpus[0]["target_id"] if cpus else None,
            "label": cpus[0].get("target_label") if cpus else None,
            "definition": "ONNX Runtime CPU EP on this machine",
            "workloads": len(baseline),
        },
        "ai_index": index,
        "efficiency": efficiency,
        "cpu_index": cpu_ai,
    }


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------
def _grade(report: dict[str, Any], fp: system.SystemFingerprint, cfg: RunConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    # power source
    if fp.power.battery_present:
        if fp.power.on_ac:
            add("Power source", "pass", "running on AC power; the CPU/GPU are allowed to boost fully")
        elif cfg.allow_battery:
            add("Power source", "warn", "running on battery; performance will be reduced by the power limit")
        else:
            add("Power source", "fail", "running on battery - plug in for a comparable result (override with --allow-battery)")
    else:
        add("Power source", "pass", "no battery detected (desktop or VM)")

    # power plan
    plan = (fp.power.power_plan or "").lower()
    if "power saver" in plan:
        add("Power plan", "fail", f"'{fp.power.power_plan}' caps clocks; use Balanced or High performance")
    elif plan:
        add("Power plan", "pass", fp.power.power_plan or "unknown")
    else:
        add("Power plan", "skip", "could not read the active Windows power plan")

    # repeat consistency. This is the *decisive* stability signal: independent
    # repeats of the same work bracket everything from thermal drift to a
    # background update, which in-loop jitter cannot see.
    spreads = [r.get("repeat_spread_pct") for r in report["results"] if r.get("ok") and r.get("repeat_spread_pct") is not None]
    worst = max(spreads) if spreads else None
    if worst is None:
        add("Repeatability", "skip", "single repeat per cell; run with --repeats 3 for a stability check")
    elif worst <= 5:
        add("Repeatability", "pass", f"worst repeat spread {worst:.1f}% (<=5%)")
    elif worst <= 12:
        add("Repeatability", "warn", f"worst repeat spread {worst:.1f}%; thermal drift or background load")
    else:
        add("Repeatability", "fail", f"worst repeat spread {worst:.1f}%; the numbers are not yet stable")

    # In-loop jitter. On a GPU an operation can finish in 250 us, where one
    # kernel dispatch costs as much as the kernel itself, so the per-iteration
    # spread is large even when the *throughput* over the window is rock solid.
    # Throughput is what gets published, so this only escalates to a failure
    # when the repeats disagree too.
    total_iterations = sum(int(r.get("iterations") or 0) for r in report["results"] if r.get("ok"))
    trimmed = [
        r["latency"].get("cv_trimmed")
        for r in report["results"]
        if r.get("ok") and r.get("latency") and r["latency"].get("cv_trimmed") is not None
    ]
    worst_trimmed = max((v for v in trimmed if v is not None), default=None)
    if worst_trimmed is None:
        add("Timing stability", "skip", "no per-iteration statistics available")
    elif worst_trimmed <= 0.05:
        add("Timing stability", "pass", f"worst trimmed CV {worst_trimmed:.1%} over {total_iterations:,} iterations")
    elif worst_trimmed <= 0.20 or (worst is not None and worst <= 5):
        add(
            "Timing stability",
            "warn",
            f"per-iteration jitter {worst_trimmed:.1%} over {total_iterations:,} iterations"
            + (
                f", but repeats agree within {worst:.1f}%, so throughput is stable"
                if worst is not None and worst <= 5
                else ""
            ),
        )
    else:
        add("Timing stability", "fail", f"per-iteration jitter {worst_trimmed:.1%}; results are not stable")

    # correctness
    checked = [
        c for c in report["cells"] if (c.get("correctness") or {}).get("checked")
    ]
    invalid = [c for c in checked if (c.get("correctness") or {}).get("valid") is False]
    if not checked:
        add("Correctness", "skip", "output verification was disabled")
    elif invalid:
        add("Correctness", "fail", f"{len(invalid)} cell(s) produced output that differs from the CPU reference")
    else:
        add("Correctness", "pass", f"{len(checked)} cell(s) matched the CPU reference within tolerance")

    # fallbacks - a provider that quietly fell back to CPU is a fake result
    fallbacks = [
        c
        for c in report["cells"]
        if (c.get("session") or {}).get("fallback_used")
    ]
    if fallbacks:
        seen = sorted({f"{(c.get('session') or {}).get('execution_providers', ['?'])[0]}" for c in fallbacks})
        add("Provider fallback", "fail", f"{len(fallbacks)} cell(s) silently fell back to {seen}")

    # unfinished cells
    failed = [c for c in report["cells"] if not c.get("ok")]
    if failed:
        add("Cell failures", "warn", f"{len(failed)} of {len(report['cells'])} cells did not complete")

    # error measurement
    with_units = [r for r in report["results"] if r.get("ok") and r.get("units_per_joule")]
    if with_units:
        add("Energy", "pass", f"{len(with_units)} cell(s) measured Joules per inference from the battery/power meter")
    elif not report.get("power_sources"):
        add("Energy", "skip", "no power instrument on this machine (a laptop on battery exposes one; desktops usually do not)")
    else:
        add("Energy", "skip", f"power instruments were available ({', '.join(report['power_sources'])}) but no reading landed in a timed window")

    # background load, sampled before the run started
    load = report.get("background_load_percent")
    if load is None:
        add("Background load", "skip", "idle CPU utilisation could not be sampled on this platform")
    elif load <= 10:
        add("Background load", "pass", f"{load:.1f}% CPU busy before the run; the machine was quiet")
    elif load <= 25:
        add("Background load", "warn", f"{load:.1f}% CPU busy before the run; results will be noisier than they should be")
    else:
        add("Background load", "fail", f"{load:.1f}% CPU busy before the run; close other work and repeat")

    # coverage - an empty run must never grade as an A
    good = [r for r in report["results"] if r.get("ok")]
    if not good:
        add("Coverage", "fail", "no measurement completed; nothing in this report can be trusted")
    else:
        kinds = sorted({r["kind"] for r in good if r.get("kind")})
        add("Coverage", "pass", f"{len(good)} device/workload combinations measured ({', '.join(kinds)})")

    weights = {"pass": 1.0, "warn": 0.6, "skip": 1.0, "fail": 0.0}
    scored = [c for c in checks if c["status"] != "skip"]
    score = 100.0 * sum(weights[c["status"]] for c in scored) / len(scored) if scored else 100.0
    grade = "A" if score >= 95 else "B" if score >= 80 else "C" if score >= 60 else "D"
    return {"score": round(score, 1), "letter": grade, "checks": checks}


def _slim(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k != "cells"}
