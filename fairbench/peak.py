"""Bounded, isolated tuning followed by independent confirmation trials."""

from collections.abc import Callable
from dataclasses import replace
from math import isfinite

from .config import RunConfig
from .devices import Target
from .throughput import parallel_memory_budget
from .worker_process import spawn_worker


def search(target: Target, workload: str, cfg: RunConfig, progress: Callable | None = None) -> dict:
    precision = workload.rsplit("_", 1)[-1]
    if workload not in ("precision_gemm_fp16", "precision_gemm_int8"):
        raise ValueError("Peak search requires the FP16/INT8 precision GEMM workloads.")
    budget = cfg.resolve_threads()
    trials = []
    resident = target.backend == "tensorrt" or target.backend in ("ort", "openvino") and (
        target.backend == "openvino" or target.device in (
            "CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider"))
    scope = ("device_resident" if resident and target.backend in ("ort", "tensorrt") and target.kind == "gpu"
             else "persistent_requests" if resident else "host_to_host")

    def probe(size, requests, threads):
        settings = {"gemm_size": size, "requests": requests, "threads": threads, "resident_io": resident}
        if any(trial["settings"] == settings for trial in trials):
            return
        probe_cfg = replace(cfg, suite="custom", mode="throughput", **settings,
                            warmup=1, min_iterations=1, min_seconds=min(0.5, cfg.min_seconds),
                            max_seconds=max(2.0, min(cfg.max_seconds, 10.0)), repeats=1, measure_power=False)
        job = {"target": target.to_dict(), "workload": workload, "precision": precision,
               "cfg": probe_cfg.to_dict(), "repeat_index": 0, "power_sources": []}
        memory = parallel_memory_budget(4 * size**2, 8 * size**2)
        result = (spawn_worker(job, timeout=240, tag=f"peak-{target.id.replace(':', '-')}-{precision}-{size}-{requests}-{threads}")
                  if requests <= memory["max_requests"] else {
                      "ok": False, "skipped": True,
                      "error": f"RAM budget allows {memory['max_requests']} request(s), requested {requests}."})
        correct = result.get("correctness") or {}
        trial = {"settings": settings, "ok": bool(result.get("ok")) and (not cfg.check_correctness or bool(correct.get("checked") and correct.get("valid"))),
                 "flops_per_s": result.get("flops_per_s"), "error": result.get("error"),
                 "correctness": correct, "model": result.get("model"), "wall_s": result.get("wall_s"),
                 "iterations": result.get("iterations"), "warnings": result.get("warnings", [])}
        trial.update({"memory_budget": memory, "skipped": bool(result.get("skipped"))})
        trials.append(trial)
        if progress:
            progress({"event": "tuning", "target": target.id, "workload": workload,
                      "size": size, "requests": requests, "threads": threads,
                      "ok": trial["ok"], "flops_per_s": trial["flops_per_s"]})

    def best():
        valid = [trial for trial in trials if trial["ok"] and isfinite(trial.get("flops_per_s") or 0) and (trial.get("flops_per_s") or 0) > 0]
        return max(valid, key=lambda trial: trial["flops_per_s"]) if valid else None

    for size in (1024, 2048, 4096):
        if size <= cfg.peak_max_size:
            probe(size, cfg.requests or 1, budget)
    selected = best()
    if selected and not cfg.requests:
        size = selected["settings"]["gemm_size"]
        for requests in ((2, 4, 8, 16, 32) if target.kind == "npu" else (2, 4)):
            probe(size, requests, budget)
    selected = best()
    if selected and target.kind == "cpu" and cfg.threads is None and cfg.logical_cores > budget:
        probe(selected["settings"]["gemm_size"], selected["settings"]["requests"], cfg.logical_cores)
        probe(selected["settings"]["gemm_size"], 1, cfg.logical_cores)
    selected = best()
    return {"target_id": target.id, "workload": workload, "scope": scope,
            "method": "bounded coordinate search: size, concurrency, CPU thread budget",
            "selected": selected["settings"] if selected else None, "trials": trials,
            "note": "Best observed among tested candidates; confirmation repeats use fixed settings. Not a theoretical peak or an exhaustive global maximum."}
