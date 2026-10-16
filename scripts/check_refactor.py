"""Regression checks: python -m scripts.check_refactor (no extra dependencies)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fairbench import cli, interp, worker_process
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Target
from fairbench.runner import _score, verify_targets
from fairbench.system import PowerInfo, SystemFingerprint


def check_throughput() -> None:
    import threading
    import time

    from fairbench.throughput import measure, measure_processes

    lock = threading.Lock()
    active = peak = completed = 0
    def run():
        nonlocal active, peak, completed
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005)
        with lock:
            active -= 1
            completed += 1
    cfg = RunConfig(mode="throughput", requests=4, min_iterations=12,
                    max_iterations=12, min_seconds=0, max_seconds=2)
    result = measure([run] * 4, cfg)
    assert completed == result["iterations"] == len(result["latencies_ms"]) == 12
    assert peak > 1 and active == 0
    assert result["units_per_s"] == completed / result["wall_s"]
    def broken():
        raise RuntimeError("inference failed")
    try:
        measure([broken, run], cfg)
    except RuntimeError as exc:
        assert str(exc) == "inference failed"
    else:
        raise AssertionError("inference failure was swallowed")
    target = Target("throughput-cpu", "ort", "CPUExecutionProvider", "cpu", "unknown", "CPU")
    cfg.requests = 2
    cfg.warmup = 1
    cfg.measure_power = False
    cfg.threads = 1
    with TemporaryDirectory() as directory, patch.object(worker_process, "JOB_DIR", Path(directory)):
        measured = worker_process.spawn_worker({"target": target.to_dict(), "workload": "smoke", "cfg": cfg.to_dict()}, 60, "throughput")
    assert measured["ok"], measured
    assert measured["concurrent_requests"] == 2 and measured["iterations"] == 12
    assert measured["measurement_mode"] == "throughput"
    assert measured["correctness"]["valid"]
    isolated = measure_processes({"target": target.to_dict(), "workload": "smoke"}, cfg, 2)
    assert isolated["iterations"] == 12 and isolated["isolation"] == "process"
    print("Throughput checks passed: concurrent execution, completed counts, drain, exceptions, real CPU worker.")


def check_conda_launch() -> None:
    with TemporaryDirectory() as directory:
        base = Path(directory)
        prefix = base / "envs" / "ryzen-ai-test"
        meta = prefix / "conda-meta"
        meta.mkdir(parents=True)
        history = meta / "history"
        history.write_text("before")
        python = prefix / "python.exe"
        python.touch()
        conda = base / "Scripts" / "conda.exe"
        conda.parent.mkdir()
        conda.touch()
        with patch.dict("os.environ", {"CONDA_EXE": str(conda)}):
            command = interp.python_command(str(python), ["-c", "pass"])
            assert command == [str(conda), "run", "--no-capture-output", "--prefix", str(prefix), str(python), "-c", "pass"]
            worker_cmd = worker_process._worker_command(str(python), Path("job"), Path("out"))
            assert worker_cmd[:6] == command[:6]
            assert worker_cmd[-2:] == ["job", "out"]
            payload = (interp.MARKER + json.dumps({"providers": ["VitisAIExecutionProvider"]})).encode()
            with patch.object(interp.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, payload, b"")) as run:
                info = interp.probe(str(python))
            assert info.providers == ["VitisAIExecutionProvider"]
            assert info.label == "ryzen-ai-test"
            assert run.call_args.args[0][:6] == command[:6]
        before = interp._signature(str(python))
        history.write_text("after SDK installation")
        assert before != interp._signature(str(python))
    print("Conda launch passed: activation for discovery/workers, environment label, SDK cache invalidation.")


def check_npu_setup() -> None:
    args = cli.build_parser().parse_args(["setup-npu", "--installer", "sdk.exe"])
    with patch.object(cli.sys, "platform", "linux"), patch("subprocess.run") as run:
        assert args.func(args) == 2
        run.assert_not_called()
    with patch.object(cli.sys, "platform", "win32"), patch("subprocess.run") as run:
        run.return_value.returncode = 7
        assert args.func(args) == 7
        command = run.call_args.args[0]
        assert command[-2:] == ["-Installer", str(Path("sdk.exe").resolve())]
        assert Path(command[command.index("-File") + 1]).is_file()
    print("NPU setup checks passed: platform guard, installer path, failure propagation.")


def check_conda_discovery() -> None:
    with TemporaryDirectory() as directory:
        home = Path(directory)
        registry = home / ".conda" / "environments.txt"
        registry.parent.mkdir()
        env = home / "custom-ryzen-env"
        env.mkdir()
        python = env / "python.exe"
        python.touch()
        registry.write_text(str(env) + "\n" + str(home / "missing"), encoding="utf-8")
        with patch.object(interp.Path, "home", return_value=home):
            candidates = interp.candidates()
        assert str(python.resolve()) in candidates
        assert not any("missing" in path for path in candidates)
    print("Conda discovery passed: custom paths and missing environments.")


def check_runtime_refresh() -> None:
    path = sys.executable
    stale = interp.Interpreter(path, primary=True, available=True, providers=["CPUExecutionProvider"])
    current = interp.Interpreter(path, primary=True, available=True,
                                 providers=["CPUExecutionProvider", "DmlExecutionProvider"])
    cache = {path.lower(): {"signature": interp._signature(path), "result": stale.to_dict()}}
    with (
        patch.object(interp, "candidates", return_value=[path]),
        patch.object(interp, "_load_cache", return_value=cache),
        patch.object(interp, "_save_cache"),
        patch.object(interp, "probe", return_value=current) as probe,
    ):
        assert "DmlExecutionProvider" in interp.probe_all()[0].providers
        probe.assert_called_once_with(path)
    print("Runtime refresh passed: new providers are detected after uv sync.")


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
    assert abs(index["gpu"]["ai_index"] - 2) < 1e-12
    assert index["gpu"]["workloads"] == 1
    assert math.isclose(scores["cpu_index"], 1)  # The chosen CPU baseline is always 1.
    assert scores["baseline"]["target_id"] == "cpu"
    assert "failed" not in index and "npu" not in index
    assert [entry["target_id"] for entry in scores["efficiency"]] == ["gpu"]
    assert abs(scores["efficiency"][0]["units_per_joule"] - 4) < 1e-12
    assert scores["comparison_workloads"] == ["a"]  # ov-cpu only completed a.
    partial = _score({"results": rows[:4] + [
        row("npu", "npu", "a", 30), {**row("npu", "npu", "b", 1), "ok": False}
    ]})
    assert partial["comparison_workloads"] == ["a"]
    assert partial["excluded_workloads"] == ["b"]
    ranks = {item["target_id"]: item for item in partial["ai_index"]}
    assert ranks["gpu"]["ai_index"] == 2
    assert all(item["workloads"] == 1 for item in ranks.values())
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


def check_battery_cli() -> None:
    for present, on_ac, allow, expected_code, label in (
        (True, False, True, 0, "battery"),
        (True, False, False, 2, "battery"),
        (True, True, False, 0, "AC"),
        (False, False, False, 0, "desktop"),
        (True, None, True, 0, "battery"),
    ):
        fingerprint = SystemFingerprint(power=PowerInfo(
            battery_present=present, on_ac=on_ac, power_plan="Balanced"
        ))
        arguments = ["run", "--no-open"]
        if allow:
            arguments.append("--allow-battery")
        output = StringIO()
        with (
            patch.object(cli.system, "collect", return_value=fingerprint),
            patch.object(cli, "run_suite", return_value={"results": []}) as run,
            patch.object(cli.report_mod, "save", return_value=Path("check.json")) as save,
            redirect_stdout(output),
        ):
            assert cli.main(arguments) == expected_code
        assert f"({label})" in output.getvalue()
        assert run.call_count == save.call_count == (1 if expected_code == 0 else 0)
        if run.called:
            assert run.call_args.args[0].allow_battery == allow
    print("Battery CLI checks passed: battery override, AC, desktop, unknown AC, default refusal.")


def check_openvino_precision() -> None:
    try:
        from fairbench.backends.openvino_backend import OpenVinoBackend
        if not OpenVinoBackend.is_installed():
            return
        backend = OpenVinoBackend()
        prop = backend.ov.properties.hint.inference_precision()
        for kind, device in (("gpu", "GPU"), ("npu", "NPU")):
            target = Target(device, "openvino", device, kind, "intel", device)
            assert backend._properties(target, RunConfig(), "fp32")[prop] == backend.ov.Type.f16
    except ImportError:
        return
    print("OpenVINO accelerator precision hints match (FP16 for GPU and NPU).")


def check_directml_options() -> None:
    backend = OrtBackend()
    for provider in ("CPUExecutionProvider", "DmlExecutionProvider"):
        target = Target("check", "ort", provider, "cpu", "unknown", "check")
        options = backend._session_options(target, RunConfig(threads=1))
        assert options.enable_mem_pattern == (provider != "DmlExecutionProvider")
        assert options.execution_mode == backend.ort.ExecutionMode.ORT_SEQUENTIAL
    print("DirectML session options passed (no GPU required for this check).")


if __name__ == "__main__":
    check_throughput()
    check_conda_launch()
    check_npu_setup()
    check_conda_discovery()
    check_runtime_refresh()
    check_worker_failures()
    check_scores()
    check_cpu_worker()
    check_battery_cli()
    check_directml_options()
    check_openvino_precision()
