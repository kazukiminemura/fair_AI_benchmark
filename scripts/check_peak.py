"""Run with python -m scripts.check_peak."""

from unittest.mock import MagicMock, patch

import numpy as np

from fairbench import models, peak
from fairbench.backends.base import CompiledModel
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Discovery, Target
from fairbench.runner import _score, run_suite
from fairbench.system import CpuInfo, SystemFingerprint


@patch("fairbench.throughput.available_memory_bytes", return_value=64 * 1024**3)
def main(_available_memory) -> None:
    target = Target("cpu", "ort", "CPUExecutionProvider", "cpu", "unknown", "CPU")
    cfg = RunConfig(suite="peak", mode="peak", physical_cores=6, logical_cores=12)
    jobs = []

    def measured(job, **kwargs):
        jobs.append(job)
        settings = job["cfg"]
        rate = {1024: 5, 2048: 10, 4096: 8}[settings["gemm_size"]]
        rate *= {1: 1, 2: 2, 4: 1.2}[settings["requests"]]
        rate *= settings["threads"] / 6
        return {"ok": True, "flops_per_s": rate * 1e12, "correctness": {"checked": True, "valid": True}}

    with patch.object(peak, "spawn_worker", side_effect=measured):
        result = peak.search(target, "precision_gemm_int8", cfg)
    assert cfg.mode == "peak" and cfg.gemm_size == 4096
    assert result["selected"] == {"gemm_size": 2048, "requests": 2, "threads": 12, "resident_io": True}
    assert len(result["trials"]) == len(jobs) and len(jobs) >= 6
    assert all(job["cfg"]["mode"] == "throughput" and job["cfg"]["repeats"] == 1 for job in jobs)
    assert {job["cfg"]["gemm_size"] for job in jobs} == {1024, 2048, 4096}
    with patch.object(peak, "spawn_worker", side_effect=measured):
        capped = peak.search(target, "precision_gemm_fp16", RunConfig(suite="peak", peak_max_size=4096))
    assert all(trial["settings"]["gemm_size"] <= 4096 for trial in capped["trials"])
    with patch.object(peak, "spawn_worker", side_effect=lambda job, **kwargs: {
        "ok": True, "flops_per_s": job["cfg"]["gemm_size"] * 1e9,
        "correctness": {"checked": True, "valid": True},
    }):
        largest = peak.search(target, "precision_gemm_fp16", RunConfig(suite="peak", requests=1))
    assert largest["selected"]["gemm_size"] == 4096
    with patch.object(peak, "spawn_worker", return_value={"ok": True, "flops_per_s": 1e30,
                                                       "correctness": {"checked": True, "valid": False}}):
        rejected = peak.search(target, "precision_gemm_fp16", cfg)
    assert rejected["selected"] is None and not any(trial["ok"] for trial in rejected["trials"])
    fixed = RunConfig(suite="peak", mode="peak", requests=4, threads=3, peak_max_size=1024)
    with patch.object(peak, "spawn_worker", side_effect=measured):
        result = peak.search(target, "precision_gemm_int8", fixed)
    assert len(result["trials"]) == 1 and result["selected"]["requests"] == 4
    assert _score({"config": {"mode": "peak"}})["baseline"] is None
    built = models.build_one("precision_gemm_fp16", 512)
    assert built.flops == 2 * 512 ** 3
    assert built.make_inputs()["x"].shape == (512, 512)
    assert built.sha256 == models.build_one("precision_gemm_fp16", 512).sha256
    assert built.sha256 != models.build_one("precision_gemm_fp16").sha256
    large = models.build_one("precision_gemm_fp16", 4096)
    assert RunConfig(gemm_size=4096).gemm_size == 4096
    assert large.flops == 2 * 4096 ** 3
    assert large.make_inputs()["x"].shape == (4096, 4096)
    for settings in ({"gemm_size": 4196}, {"peak_max_size": 4196}):
        try:
            RunConfig(**settings)
        except ValueError:
            pass
        else:
            raise AssertionError("Removed GEMM size must be rejected")

    selected = {"gemm_size": 1024, "requests": 2, "threads": 6, "resident_io": True}
    confirmation_jobs = []

    def confirm(job, **kwargs):
        confirmation_jobs.append(job)
        return {"ok": True, "target_id": target.id, "kind": target.kind, "vendor": target.vendor,
                "workload": job["workload"], "units_per_s": 10, "iterations": 10,
                "model": {"flops": 2 * 1024 ** 3, "gemm_shape": [1024, 1024, 1024], "gemm_size": 1024,
                          "operation_convention": models.OPERATION_CONVENTION}, "latency": {"median_ms": 1}}

    with patch("fairbench.peak.search", return_value={"selected": selected, "scope": "persistent_requests", "trials": [1, 2]}) as tune, \
            patch("fairbench.runner.spawn_worker", side_effect=confirm), \
            patch("fairbench.runner.system.measure_memory_bandwidth"), \
            patch("fairbench.runner.system.probe_background_load", return_value=0), \
            patch("fairbench.runner.models.describe_build_error", return_value=None):
        report = run_suite(RunConfig(suite="peak", mode="peak", repeats=3, cooldown_seconds=0, measure_power=False),
                           verify=False, discovery=Discovery(targets=[target]),
                           fingerprint=SystemFingerprint(cpu=CpuInfo(physical_cores=6, logical_cores=12)))
    assert tune.call_count == 2 and len(confirmation_jobs) == 6
    assert all(job["cfg"]["gemm_size"] == 1024 and job["cfg"]["requests"] == 2 for job in confirmation_jobs)
    assert all(row["repeats_ok"] == 3 and row["peak_settings"] == selected for row in report["results"])
    assert report["operation_convention"]["operations_per_mac"] == 2
    assert all(row["gemm_shape"] == [1024, 1024, 1024] and row["gemm_size"] == 1024 for row in report["results"])

    backend = OrtBackend()
    session, binding = MagicMock(), MagicMock()
    session.io_binding.return_value = binding
    binding.get_outputs.return_value = [MagicMock()]
    output = np.ones((2, 2), dtype=np.float32)
    binding.copy_outputs_to_cpu.return_value = [output]
    compiled = CompiledModel(session, 0, "fp16", ["y"])
    with patch.object(backend.ort.OrtValue, "ortvalue_from_numpy", return_value=MagicMock()):
        run, read_outputs = backend.prepare_resident(compiled, {"x": output}, target)
        for _ in range(5):
            run()
        assert not binding.copy_outputs_to_cpu.called
        assert binding.synchronize_outputs.call_count == 6
        assert np.array_equal(read_outputs()[0], output)
    print("Peak search, correctness rejection, fixed settings, size accounting and transfer-free timed runner passed.")


if __name__ == "__main__":
    main()
