"""One measurement cell, executed in its own process.

Isolation is not paranoia. Vendor runtimes really do abort the process: during
development, OpenVINO's GPU plugin took the whole interpreter down on a machine
where it misidentified an NVIDIA adapter. Running each (device, workload,
repeat) cell in a child process means

* a crash or a driver hang costs exactly one cell, not the suite;
* every cell starts from an identical, un-fragmented memory state, so an early
  GPU run cannot make a later one look slower;
* the parent can enforce a wall-clock timeout on a hung device.

Usage:  python -m fairbench.worker <job.json> <result.json>
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import models
from .backends import BackendError, get_backend
from .config import RunConfig
from .devices import Target
from .power import PowerSampler
from .stats import summarise

RESULT_MARKER = "@@FAIRBENCH-RESULT@@"


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------
def _fingerprint(outputs: list[np.ndarray]) -> dict[str, Any]:
    y = np.asarray(outputs[0], dtype=np.float64).ravel()
    if y.size == 0:
        return {"size": 0, "l2": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "size": int(y.size),
        "l2": float(np.linalg.norm(y)),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }


def _reference_path(sha: str, seed: int, shape: tuple) -> Path:
    cache = Path(".cache") / "reference"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{sha[:16]}-{seed}-{','.join(map(str, shape))}.json"


def reference_fingerprint(base_key: str, seed: int) -> dict[str, Any] | None:
    """Ground truth for a workload: the float32 graph, on ONNX Runtime's CPU EP.

    Using ORT CPU as the reference is deliberate - it is the one path that
    behaves identically on Intel and AMD silicon, so it cannot favour a vendor.
    """
    built = models.build_one(base_key)
    shape = tuple(d.dim_value for d in built.model.graph.output[0].type.tensor_type.shape.dim)
    path = _reference_path(built.sha256, seed, shape)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        session = ort.InferenceSession(built.serialise(), so, providers=["CPUExecutionProvider"])
        outs = session.run(None, built.make_inputs(seed))
        fp = _fingerprint([np.asarray(o) for o in outs])
        path.write_text(json.dumps(fp))
        return fp
    except Exception:
        return None


def relative_l2(outputs: list[np.ndarray], reference: dict[str, Any]) -> float | None:
    current = _fingerprint(outputs)
    if not reference or reference.get("l2", 0) == 0 or current["l2"] == 0:
        return None
    return abs(current["l2"] - reference["l2"]) / reference["l2"]


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------
def _run_throughput(backend, compiled, built, target, cfg, feed, result, job):
    from dataclasses import replace

    from .throughput import measure, measure_processes

    candidates = [cfg.requests] if cfg.requests else ([1, 2, 4, 8, 16, 32] if target.kind == "npu" else [1, 2, 4, 8])
    trials = []
    successful = []
    sampler = PowerSampler(sources=job.get("power_sources") or [], enabled=cfg.measure_power)
    sampler.start()
    try:
        for count in candidates:
            try:
                runners = []
                lane_cfg = replace(cfg, threads=max(1, cfg.resolve_threads() // count))
                for _ in range(0 if target.device == "DmlExecutionProvider" else count):
                    if target.backend == "openvino":
                        # CompiledModel.__call__ shares a cached InferRequest;
                        # explicit requests avoid sharing mutable request state.
                        request = compiled.handle.create_infer_request()
                        def run(request=request):
                            output = request.infer(feed, share_inputs=False, share_outputs=False)
                            return [np.asarray(output[name]) for name in compiled.output_names]
                    else:
                        # DirectML forbids concurrent Run calls on one session.
                        # Give every lane a separate session on all ORT providers.
                        model = backend.compile(built, target, lane_cfg, job.get("precision", "fp32"))
                        if model.meta.get("fallback_used"):
                            raise BackendError("Requested accelerator fell back to another provider")
                        def run(model=model):
                            return backend.run(model, feed)
                    for _ in range(max(1, cfg.warmup)):
                        run()
                    runners.append(run)
                trial = (measure_processes(job, cfg, count) if target.device == "DmlExecutionProvider"
                         else measure(runners, cfg))
                trial["power"] = sampler.summary((trial["start"], trial["end"])).to_dict()
                trial["threads_per_session"] = lane_cfg.resolve_threads() if target.backend == "ort" else None
                successful.append(trial)
                trials.append({k: v for k, v in trial.items() if k != "latencies_ms"})
            except Exception as exc:
                trials.append({"requests": count, "error": str(exc), "ok": False})
        if not successful:
            raise BackendError(f"All throughput trials failed: {trials}")
        if len(successful) != len(trials):
            result["warnings"].append("Some concurrency candidates failed; the reported maximum covers successful candidates only.")
        best = max(successful, key=lambda trial: trial["units_per_s"])
        if best["wall_s"] < cfg.min_seconds:
            result["warnings"].append("Selected throughput trial ended before the requested minimum duration; increase max_iterations or the time budget.")
        stats = summarise(best["latencies_ms"])
        power = best["power"]
        units = best["units_per_s"]
        result.update({
            "ok": True, "phase": "done", "measurement_mode": "throughput",
            "concurrent_requests": best["requests"], "throughput_trials": trials,
            "throughput_search": "fixed" if cfg.requests else f"best observed among {candidates} requests",
            "iterations": best["iterations"], "wall_s": best["wall_s"],
            "latency": stats.to_dict(), "units_per_s": units,
            "flops_per_s": built.flops * units, "power": power,
            "host_cpu_seconds": best["host_cpu_seconds"],
            "host_cpu_seconds_per_inference": best["host_cpu_seconds"] / best["iterations"],
            "host_cpu_cores_used": best["host_cpu_seconds"] / best["wall_s"],
            "energy_per_inference_j": power.get("average_watts") / units if power.get("available") and power.get("average_watts") else None,
        })
        result["warnings"].append("Throughput is completed requests / wall time including drain; latency is per-request under concurrent load. Maximum is limited to tested request counts.")
        return result
    except Exception as exc:
        result.update({"error": str(exc), "phase": "throughput"})
        return result
    finally:
        sampler.stop()


def run_cell(job: dict[str, Any]) -> dict[str, Any]:
    target = Target(**job["target"])
    key = job["workload"]
    precision = job.get("precision", "fp32")
    cfg = RunConfig(**job["cfg"])
    repeat_index = int(job.get("repeat_index", 0))

    result: dict[str, Any] = {
        "ok": False,
        "target_id": target.id,
        "vendor": target.vendor,
        "kind": target.kind,
        "backend": target.backend,
        "device": target.device,
        "target_label": target.label,
        "workload": key,
        "precision": precision,
        "repeat_index": repeat_index,
        "measurement_mode": cfg.mode,
        "error": None,
        "warnings": [],
    }

    try:
        backend = get_backend(target.backend)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    warnings: list[str] = result["warnings"]
    started = time.perf_counter()

    try:
        built = models.build_one(key)
    except Exception as exc:
        result["error"] = f"could not build workload: {exc}"
        return result

    result["model"] = {
        "key": built.key,
        "sha256": built.sha256,
        "size_bytes": built.size_bytes,
        "flops": built.flops,
        "unit": built.unit,
        "description": built.description,
        "precision": built.precision,
    }

    # ---- compile ------------------------------------------------------
    try:
        compiled = backend.compile(built, target, cfg, precision)
    except BackendError as exc:
        result["error"] = str(exc)
        result["phase"] = "compile"
        result["elapsed_s"] = time.perf_counter() - started
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["phase"] = "compile"
        result["traceback"] = traceback.format_exc()[-1500:]
        result["elapsed_s"] = time.perf_counter() - started
        return result

    result["session"] = compiled.meta
    result["compile_ms"] = compiled.compile_ms
    if compiled.meta.get("fallback_used"):
        warnings.append(
            f"requested {compiled.meta.get('requested_provider')} but "
            f"{compiled.meta.get('execution_providers', ['?'])[0]} executed the graph"
        )

    feed = built.make_inputs(cfg.input_seed)

    # ---- correctness --------------------------------------------------
    if cfg.check_correctness:
        base_key = "precision_gemm" if key.startswith("precision_gemm_") else (key[: -len("_int8")] if key.endswith("_int8") else key)
        ref = reference_fingerprint(base_key, cfg.input_seed)
        try:
            outs = [np.asarray(o) for o in backend.run(compiled, feed)]
            diff = relative_l2(outs, ref) if ref else None
            tolerance = cfg.relative_tolerance * (5.0 if precision in ("int8", "int4") else 1.0)
            result["correctness"] = {
                "checked": ref is not None,
                "relative_l2_error": diff,
                "tolerance": tolerance,
                "valid": (diff is None) or (diff <= tolerance),
                "reference": "onnxruntime CPUExecutionProvider, fp32 graph",
            }
            if diff is not None and diff > tolerance:
                warnings.append(
                    f"output differs from the CPU reference by {diff:.2%} "
                    f"(tolerance {tolerance:.1%}) - numbers may be meaningless"
                )
        except Exception as exc:
            result["correctness"] = {"checked": False, "valid": None, "error": str(exc)}
            warnings.append(f"correctness probe failed: {exc}")

    if cfg.mode == "throughput":
        measured = _run_throughput(backend, compiled, built, target, cfg, feed, result, job)
        if measured.get("ok"):
            measured["timing_breakdown"] = backend.diagnose(compiled, built, target, cfg, feed)
        measured["elapsed_s"] = time.perf_counter() - started
        return measured

    # ---- warm-up ------------------------------------------------------
    try:
        for _ in range(max(0, cfg.warmup)):
            backend.run(compiled, feed)
    except Exception as exc:
        result["error"] = f"warm-up failed: {exc}"
        result["phase"] = "warmup"
        return result

    # ---- timed region -------------------------------------------------
    # Python's cyclic collector can pause for tens of milliseconds on a large
    # heap, which would show up as a fake latency spike on whichever device
    # happened to be running. It is not needed here (the hot loop creates no
    # reference cycles), so it is parked for the duration and restored after.
    import gc

    gc.collect()
    gc.disable()
    sampler = PowerSampler(sources=job.get("power_sources") or [], enabled=cfg.measure_power)
    sampler.start()
    latencies_ms: list[float] = []
    cpu0 = time.process_time()
    host_cpu_raw = 0.0
    try:
        t_start = time.perf_counter()
        budget = max(cfg.min_seconds, 0.0)
        deadline = t_start + cfg.max_seconds if cfg.max_seconds else None
        iterations = 0
        while True:
            t0 = time.perf_counter()
            backend.run(compiled, feed)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            iterations += 1
            if iterations >= cfg.min_iterations and (t1 - t_start) >= budget:
                break
            if iterations >= cfg.max_iterations:
                break
            if deadline and time.perf_counter() > deadline:
                warnings.append(f"stopped at the {cfg.max_seconds:g}s budget")
                break
        t_end = time.perf_counter()
    except Exception as exc:
        result["error"] = f"inference failed: {exc}"
        result["phase"] = "measure"
        result["traceback"] = traceback.format_exc()[-1500:]
        host_cpu_raw = time.process_time() - cpu0
        gc.enable()
        sampler.stop()
        return result
    finally:
        host_cpu_raw = time.process_time() - cpu0
        gc.enable()

    window = (t_start, t_end)
    host_cpu = host_cpu_raw
    power = sampler.summary(window)
    sampler.stop()

    # ---- aggregate ----------------------------------------------------
    stats = summarise(latencies_ms)
    wall = t_end - t_start
    units_per_s = len(latencies_ms) / wall if wall > 0 else float("nan")
    flops_per_s = built.flops * units_per_s
    cpu_seconds_per_inference = host_cpu / len(latencies_ms) if latencies_ms else float("nan")

    result.update(
        {
            "ok": True,
            "phase": "done",
            "elapsed_s": time.perf_counter() - started,
            "iterations": len(latencies_ms),
            "wall_s": wall,
            "latency": stats.to_dict(),
            "units_per_s": units_per_s,
            "flops_per_s": flops_per_s,
            "host_cpu_seconds": host_cpu,
            "host_cpu_seconds_per_inference": cpu_seconds_per_inference,
            "host_cpu_cores_used": cpu_seconds_per_inference / (stats.median_ms / 1000.0)
            if stats.median_ms
            else float("nan"),
            "power": power.to_dict(),
            "energy_per_inference_j": (power.average_watts / units_per_s)
            if (power.available and power.average_watts and units_per_s)
            else None,
        }
    )
    if stats.cv_trimmed > 0.05:
        warnings.append(f"high variance (trimmed CV {stats.cv_trimmed:.1%}); treat the median with care")
    result["timing_breakdown"] = backend.diagnose(compiled, built, target, cfg, feed)
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python -m fairbench.worker <job.json> <result.json>", file=sys.stderr)
        return 2
    job_path, out_path = Path(argv[1]), Path(argv[2])
    job = json.loads(job_path.read_text())
    try:
        result = run_cell(job)
    except Exception as exc:
        result = {
            "ok": False,
            "target_id": job.get("target", {}).get("id"),
            "workload": job.get("workload"),
            "error": f"worker crashed: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
    out_path.write_text(json.dumps(result, default=_json_default))
    return 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
