"""Regression gates for correctness, process timing and CLI precedence."""

import json
import subprocess
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from fairbench import cli, models, throughput, worker, worker_process
from fairbench.backends.base import CompiledModel
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.system import CpuInfo
from fairbench.worker import _fingerprint, relative_l2


def main() -> None:
    def reject_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")

    payload = {"ok": True, "latency": {"mean_ms": float("nan")},
               "nested": [float("inf"), (float("-inf"), {"value": np.float32("nan")})],
               "array": np.array([np.nan, np.inf, -np.inf, 1.25]),
               "numpy_float": np.float64("nan"), "numpy_int": np.int64(3),
               "literal": "NaN is text", "finite": 2.5}
    with TemporaryDirectory() as directory, patch.object(worker_process, "JOB_DIR", Path(directory)):
        def launch(cmd, **kwargs):
            with patch.object(worker, "run_cell", return_value=payload):
                assert worker.main(["worker", cmd[-2], cmd[-1]]) == 0
            parsed = json.loads(Path(cmd[-1]).read_text(), parse_constant=reject_constant)
            assert parsed["latency"]["mean_ms"] is None
            assert parsed["nested"] == [None, [None, {"value": None}]]
            assert parsed["array"] == [None, None, None, 1.25]
            assert parsed["numpy_float"] is None and parsed["numpy_int"] == 3
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with patch.object(worker_process.subprocess, "run", side_effect=launch):
            result = worker_process.spawn_worker({"target": {"id": "test"}, "workload": "smoke"}, 1, "nonfinite")
        assert result["ok"] and result["worker_returncode"] == 0
        assert result["latency"]["mean_ms"] is None and result["numpy_float"] is None
        assert result["literal"] == payload["literal"] and result["finite"] == 2.5
    print("Worker strict JSON checks passed: nested NaN/Infinity become null, NumPy conversions and parent readback.")
    values = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
    reference = _fingerprint([values, values * 2])
    assert relative_l2([values.copy(), values * 2], reference) == 0
    for outputs in ([values[::-1], values * 2], [-values, values * 2],
                    [values, -values * 2], [values.T, values * 2], [values]):
        assert relative_l2(outputs, reference) > 0.05
    assert relative_l2([values], {"l2": float(np.linalg.norm(values))}) is None
    assert relative_l2([np.zeros(4)], _fingerprint([np.zeros(4)])) == 0
    assert relative_l2([np.full((4, 6), np.nan), values], reference) == float("inf")
    large = np.arange(20000, dtype=np.float32)
    assert len(_fingerprint([large])["outputs"][0]["values"]) == 16384
    target = Target("cpu", "ort", "CPUExecutionProvider", "cpu", "intel", "CPU")
    backend = MagicMock()
    backend.compile.return_value = CompiledModel(None, 0, "fp16", ["y"])
    backend.run.return_value = [np.ones(4)]
    with patch.object(worker, "get_backend", return_value=backend), patch.object(worker, "reference_fingerprint", return_value=None):
        result = worker.run_cell({"target": target.to_dict(), "workload": "precision_gemm_fp16", "precision": "fp16", "cfg": RunConfig().to_dict()})
    assert not result["ok"] and result["phase"] == "correctness"
    for kernel, executed, expected_code in (("MatMulIntegerToFloat", True, 0), ("MatMul", True, 1), ("MatMulIntegerToFloat", False, 1)):
        output = StringIO()
        result = {"ok": executed, "correctness": {"checked": True, "valid": True},
                  "session": {"compute_kernels": [kernel]}}
        with (patch.dict(models.BASE_BUILDERS, {"smoke": None}, clear=True), patch.object(models, "QUANTISABLE", ()),
              patch.object(models, "build_one", return_value=SimpleNamespace(flops=1, size_bytes=1, sha256="abc")),
              patch("fairbench.backends.installed_backends", return_value={"ort": MagicMock()}),
              patch.object(cli.system, "collect", return_value=SimpleNamespace(cpu=CpuInfo(physical_cores=6))),
              patch.object(cli, "discover", return_value=SimpleNamespace(available=lambda: [target])),
              patch.object(cli, "verify_targets", return_value={target.id: {"ok": True}}),
              patch("fairbench.worker_process.spawn_worker", return_value=result) as launch,
              redirect_stdout(output)):
            assert cli.cmd_selftest(SimpleNamespace(verbose=False)) == expected_code
            assert launch.call_count == 2
        assert ("all supported-workload execution checks passed" in output.getvalue()) == (expected_code == 0)

    context = MagicMock()
    processes = [MagicMock(), MagicMock()]
    for process in processes:
        process.is_alive.return_value = False
    context.Process.side_effect = processes
    parents = [MagicMock(), MagicMock()]
    for parent in parents:
        parent.poll.return_value = True
        parent.recv.side_effect = [{"ready": True}, {"latencies_ms": [1] * 50,
                                   "finished": 12.0, "host_cpu_seconds": 0.5}]
    context.Pipe.side_effect = [(parent, MagicMock()) for parent in parents]
    with patch("multiprocessing.get_context", return_value=context), patch.object(throughput.time, "perf_counter", return_value=10):
        result = throughput.measure_processes({}, RunConfig(), 2)
    assert result["wall_s"] == 2 and result["units_per_s"] == 50

    for stage in ("preparing", "draining"):
        context = MagicMock()
        process = context.Process.return_value
        process.is_alive.return_value = False
        parent = MagicMock()
        parent.poll.return_value = True
        parent.recv.side_effect = [EOFError()] if stage == "preparing" else [{"ready": True}, EOFError()]
        context.Pipe.return_value = parent, MagicMock()
        with patch("multiprocessing.get_context", return_value=context):
            try:
                throughput.measure_processes({}, RunConfig(), 1)
            except RuntimeError as error:
                assert "exited without a result" in str(error) and stage in str(error)
            else:
                raise AssertionError("A dead child process was accepted")
    print("Element checks reject permutations/sign flips/shape errors; IPC timing and EOF isolation passed.")


if __name__ == "__main__":
    main()
