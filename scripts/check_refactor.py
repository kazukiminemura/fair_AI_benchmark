"""Regression checks: python -m scripts.check_refactor (no extra dependencies)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fairbench import worker_process
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.runner import _score, verify_targets


def check_worker_failures() -> None:
    job = {"target": {"id": "cpu"}, "workload": "smoke"}
    cases = [
        (None, FileNotFoundError(), "spawn"),
        (None, subprocess.TimeoutExpired("worker", 1), "timeout"),
        (None, None, "crash"),
        ("invalid json", None, "parse"),
        ({"ok": True}, None, None),
        ({"ok": False}, None, None),
        ({"ok": False, "error": "compile failed"}, None, None),
    ]
    with TemporaryDirectory() as directory, patch.object(worker_process, "JOB_DIR", Path(directory)):
        for payload, error, phase in cases:
            def launch(cmd, payload=payload, error=error, **kwargs):
                assert kwargs["timeout"] == 1
                assert Path(cmd[-2]).is_absolute() and Path(cmd[-1]).is_absolute()
                assert json.loads(Path(cmd[-2]).read_text()) == job
                assert not Path(cmd[-1]).exists(), "stale output was not removed"
                if error:
                    raise error
                if payload is not None:
                    content = json.dumps(payload) if isinstance(payload, dict) else payload
                    Path(cmd[-1]).write_text(content)
                return subprocess.CompletedProcess(cmd, 1, b"", b"driver error")

            # Reuse the same tag to verify that old output cannot mask a crash.
            (Path(directory) / "check.out.json").write_text('{"ok": true}')
            with patch.object(worker_process.subprocess, "run", side_effect=launch):
                result = worker_process.spawn_worker(job, timeout=1, tag="check")
            assert result.get("phase") == phase, result
            if phase:
                assert not result["ok"] and result["error"], result
                assert result["target_id"] == "cpu" and result["workload"] == "smoke"
            else:
                assert result["worker_returncode"] == 1
                if payload.get("ok"):
                    assert result["ok"]
                else:
                    assert result["error"] == payload.get("error", "worker exited with code 1\ndriver error")

    with patch.object(worker_process.paths, "FROZEN", True), patch.object(
        worker_process.paths, "is_own_interpreter", return_value=True
    ):
        assert worker_process._worker_command("app.exe", Path("job"), Path("out")) == [
            "app.exe", "--worker", "job", "out"
        ]
    assert "access violation" in worker_process._crash_hint(-1073741819, "")
    print("Worker checks passed: success, errors, stale output, frozen command, driver crash.")


def check_scores() -> None:
    def row(target, kind, workload, units, energy=None, backend="ort"):
        return {
            "ok": True, "target_id": target, "kind": kind, "workload": workload,
            "units_per_s": units, "units_per_joule": energy, "backend": backend,
            "target_label": target, "vendor": "unknown",
        }

    rows = [
        row("cpu", "cpu", "a", 10), row("cpu", "cpu", "b", 20),
        row("gpu", "gpu", "a", 20, 4), row("gpu", "gpu", "b", 160, 16),
        row("npu", "npu", "unmatched", 30, 32),
        row("ov-cpu", "cpu", "a", 1000, backend="openvino"),
        {**row("failed", "gpu", "a", 10000, 10000), "ok": False},
    ]
    scores = _score({"results": rows})
    index = {entry["target_id"]: entry for entry in scores["ai_index"]}
    assert abs(index["gpu"]["ai_index"] - 4) < 1e-12
    assert index["gpu"]["workloads"] == 2
    assert math.isclose(scores["cpu_index"], 100)  # First CPU in the descending ranking.
    assert scores["baseline"]["target_id"] == "cpu"
    assert "failed" not in index and "npu" not in index
    assert [entry["target_id"] for entry in scores["efficiency"]] == ["npu", "gpu"]
    assert abs(scores["efficiency"][1]["units_per_joule"] - 8) < 1e-12
    assert _score({"results": [rows[2]]})["baseline"] is None
    fallback = _score({"results": [rows[5]]})
    assert fallback["baseline"]["target_id"] == "ov-cpu"
    assert fallback["cpu_index"] == 1
    print("Score checks passed: CPU baseline, geometric means, ranking, missing workloads, failures.")


def check_cpu_worker() -> None:
    target = Target("check-cpu", "ort", "CPUExecutionProvider", "cpu", "unknown", "CPU")
    with TemporaryDirectory() as directory, patch.object(worker_process, "JOB_DIR", Path(directory)):
        verdict = verify_targets([target], RunConfig(threads=1))[target.id]
    assert verdict["ok"], verdict
    assert verdict["compile_ms"] >= 0 and verdict["latency_ms"] > 0
    print(f"CPU child process passed ({Path(sys.executable).name}).")


if __name__ == "__main__":
    check_worker_failures()
    check_scores()
    check_cpu_worker()
