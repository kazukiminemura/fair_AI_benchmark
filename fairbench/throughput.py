"""Sustained concurrent inference; each lane owns its request/session."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .config import RunConfig


def measure(runners: list[Callable], cfg: RunConfig) -> dict:
    ready = threading.Barrier(len(runners) + 1)
    start = threading.Event()
    lock = threading.Lock()
    issued = 0
    failed = threading.Event()
    t_start = 0.0

    def lane(run):
        nonlocal issued
        timings = []
        ready.wait()
        start.wait()
        try:
            while not failed.is_set():
                with lock:
                    elapsed = time.perf_counter() - t_start
                    if (issued >= cfg.max_iterations
                        or (cfg.max_seconds and elapsed >= cfg.max_seconds)
                        or (issued >= cfg.min_iterations and elapsed >= cfg.min_seconds)):
                        break
                    issued += 1
                t0 = time.perf_counter()
                run()
                timings.append((time.perf_counter() - t0) * 1000)
        except BaseException:
            failed.set()
            raise
        return timings

    with ThreadPoolExecutor(max_workers=len(runners)) as pool:
        futures = [pool.submit(lane, run) for run in runners]
        ready.wait()
        cpu0 = time.process_time()
        t_start = time.perf_counter()
        start.set()
        latencies = [value for future in futures for value in future.result()]
        t_end = time.perf_counter()
        cpu = time.process_time() - cpu0
    if not latencies:
        raise RuntimeError("No inference completed in the throughput window")
    return {"latencies_ms": latencies, "start": t_start, "end": t_end,
            "host_cpu_seconds": cpu, "requests": len(runners),
            "iterations": len(latencies), "wall_s": t_end - t_start,
            "units_per_s": len(latencies) / (t_end - t_start)}


def _process_lane(connection, job, cfg, count, issued, start, stop):
    # A separate process also isolates DirectML's native device/session state.
    try:
        from dataclasses import replace

        from .backends import get_backend
        from .devices import Target
        from .models import build_one

        target = Target(**job["target"])
        backend = get_backend(target.backend)
        lane_cfg = replace(cfg, threads=max(1, cfg.resolve_threads() // count))
        built = build_one(job["workload"])
        model = backend.compile(built, target, lane_cfg, job.get("precision", "fp32"))
        if model.meta.get("fallback_used"):
            raise RuntimeError("Requested accelerator fell back to another provider")
        feed = built.make_inputs(cfg.input_seed)
        for _ in range(max(1, cfg.warmup)):
            backend.run(model, feed)
        connection.send({"ready": True})
        t_start = connection.recv()
        start.wait()
        cpu0 = time.process_time()
        timings = []
        while not stop.is_set():
            with issued.get_lock():
                elapsed = time.perf_counter() - t_start
                if (issued.value >= cfg.max_iterations
                    or (cfg.max_seconds and elapsed >= cfg.max_seconds)
                    or (issued.value >= cfg.min_iterations and elapsed >= cfg.min_seconds)):
                    break
                issued.value += 1
            t0 = time.perf_counter()
            backend.run(model, feed)
            timings.append((time.perf_counter() - t0) * 1000)
        connection.send({"latencies_ms": timings, "host_cpu_seconds": time.process_time() - cpu0})
    except BaseException as exc:
        stop.set()
        connection.send({"error": str(exc)})
    finally:
        connection.close()


def measure_processes(job: dict, cfg: RunConfig, count: int) -> dict:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    issued = context.Value("i", 0)
    start, stop = context.Event(), context.Event()
    processes, pipes = [], []
    try:
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_process_lane, args=(child, job, cfg, count, issued, start, stop), daemon=True)
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
        for pipe in pipes:
            if not pipe.poll(180):
                raise RuntimeError("Timed out preparing a DirectML process")
            response = pipe.recv()
            if not response.get("ready"):
                raise RuntimeError(response.get("error", "DirectML process did not become ready"))
        t_start = time.perf_counter()
        for pipe in pipes:
            pipe.send(t_start)
        start.set()
        responses = []
        for pipe in pipes:
            if not pipe.poll((cfg.max_seconds or cfg.min_seconds) + 60):
                raise RuntimeError("Timed out draining a DirectML process")
            response = pipe.recv()
            if "error" in response:
                raise RuntimeError(response["error"])
            responses.append(response)
        t_end = time.perf_counter()
        latencies = [value for response in responses for value in response["latencies_ms"]]
        if not latencies:
            raise RuntimeError("No DirectML inference completed")
        return {"latencies_ms": latencies, "start": t_start, "end": t_end,
                "host_cpu_seconds": sum(response["host_cpu_seconds"] for response in responses),
                "requests": count, "iterations": len(latencies), "wall_s": t_end - t_start,
                "units_per_s": len(latencies) / (t_end - t_start), "isolation": "process"}
    finally:
        stop.set()
        start.set()
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for pipe in pipes:
            pipe.close()
