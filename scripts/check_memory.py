"""Run with python -m scripts.check_memory; no large allocation or NPU required."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fairbench import peak, throughput, worker
from fairbench.config import RunConfig
from fairbench.devices import Target


def main() -> None:
    size = 4096
    with patch.object(throughput, "available_memory_bytes", return_value=4 * 1024**3):
        budget = throughput.parallel_memory_budget(4 * size**2, 8 * size**2)
    assert budget["max_requests"] == 3
    assert budget["budget_bytes"] == 1536 * 1024**2
    for available, expected in ((0, 0), (512 * 1024**2, 0), (8 * 1024**3, 7), (16 * 1024**3, 15), (None, 1)):
        with patch.object(throughput, "available_memory_bytes", return_value=available):
            assert throughput.parallel_memory_budget(4 * size**2, 8 * size**2)["max_requests"] == expected

    target = Target("npu", "openvino", "NPU", "npu", "intel", "NPU")
    config = RunConfig(suite="peak", mode="peak")

    def measured(job, **kwargs):
        return {"ok": True, "flops_per_s": job["cfg"]["gemm_size"] * job["cfg"]["requests"],
                "correctness": {"checked": True, "valid": True}}

    with patch.object(throughput, "available_memory_bytes", return_value=4 * 1024**3), \
            patch.object(peak, "spawn_worker", side_effect=measured) as launch:
        result = peak.search(target, "precision_gemm_int8", config)
    assert result["selected"]["gemm_size"] == 4096 and result["selected"]["requests"] == 2
    assert {trial["settings"]["requests"] for trial in result["trials"] if trial["skipped"]} == {4, 8, 16, 32}
    assert all(job.args[0]["cfg"]["requests"] <= 2 for job in launch.call_args_list)
    assert all(trial["memory_budget"] for trial in result["trials"])
    with patch.object(throughput, "available_memory_bytes", return_value=0), patch.object(peak, "spawn_worker") as launch:
        blocked = peak.search(target, "precision_gemm_fp16", config)
    assert blocked["selected"] is None and not launch.called

    with patch.object(throughput, "available_memory_bytes", side_effect=[64 * 1024**3] * 3 + [0] * 6), \
            patch.object(peak, "spawn_worker", side_effect=measured):
        shrinking = peak.search(target, "precision_gemm_int8", config)
    assert shrinking["selected"]["requests"] == 1
    assert all(trial["skipped"] for trial in shrinking["trials"] if trial["settings"]["requests"] > 1)
    fixed = RunConfig(suite="peak", mode="peak", gemm_size=size, requests=32)
    with patch.object(throughput, "available_memory_bytes", return_value=4 * 1024**3), patch.object(peak, "spawn_worker") as launch:
        blocked = peak.search(target, "precision_gemm_int8", fixed)
    assert blocked["selected"] is None and not launch.called

    built = SimpleNamespace(size_bytes=4 * size**2, model=SimpleNamespace(graph=SimpleNamespace(output=[])))
    cfg = RunConfig(mode="throughput", requests=32, measure_power=False)
    with patch.object(throughput, "parallel_memory_budget", return_value=budget), \
            patch.object(throughput, "measure") as measure:
        result = worker._run_throughput(MagicMock(), MagicMock(), built, target, cfg,
                                        {}, {"ok": False, "warnings": []}, {})
    assert not result["ok"] and result["phase"] == "throughput" and "RAM budget" in result["error"]
    assert not measure.called
    print("RAM guards passed: 4096 sizing, low/unknown RAM, changing availability, skipped trials and explicit requests.")


if __name__ == "__main__":
    main()
