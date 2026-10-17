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
from unittest.mock import MagicMock, patch

from fairbench import cli, interp, worker_process
from fairbench.backends.ort_backend import OrtBackend
from fairbench.config import RunConfig
from fairbench.devices import Discovery, Target
from fairbench.runner import _score, verify_targets
from fairbench.system import PowerInfo, SystemFingerprint


def check_backend_selection() -> None:
    from fairbench.runner import select_targets

    targets = [Target("ort-cpu", "ort", "CPUExecutionProvider", "cpu", "intel", "CPU"),
               Target("ov-gpu", "openvino", "GPU", "gpu", "intel", "GPU"),
               Target("wm-npu", "winml", "VitisAIExecutionProvider", "npu", "amd", "NPU")]
    assert select_targets(targets) == targets
    assert select_targets(targets, "winml") == [targets[2]]
    assert select_targets(targets, "openvino", ["ov-gpu"]) == [targets[1]]
    for pool, backend, selected in ((targets, "winml", ["ov-gpu"]), (targets[:1], "winml", None)):
        try:
            select_targets(pool, backend, selected)
        except ValueError:
            pass
        else:
            raise AssertionError("backend mismatch silently accepted")
    args = cli.build_parser().parse_args(["run", "--backend", "winml", "--suite", "vision"])
    cfg = cli._config_from_args(args)
    assert cfg.backend == "winml" and cfg.suite == "vision"
    assert cfg.to_dict()["backend"] == "winml"
    print("Backend selection passed: runtime filter, target intersection, unavailable runtime, recorded CLI config.")


def check_measurement_sets() -> None:
    from fairbench.runner import DEFAULT_WORKLOADS, WORKLOAD_SETS, resolve_workloads

    for key, suite in WORKLOAD_SETS.items():
        assert resolve_workloads(RunConfig(suite=key, include_int8=False)) == suite["workloads"]
    assert resolve_workloads(RunConfig(include_int8=False)) == DEFAULT_WORKLOADS
    assert resolve_workloads(RunConfig(suite="compute")) == ["dense_gemm", "dense_gemm_int8"]
    args = cli.build_parser().parse_args(["run", "--suite", "vision", "--no-int8"])
    cfg = cli._config_from_args(args)
    assert resolve_workloads(cfg) == ["conv_vision", "depthwise_mobile"]
    args = cli.build_parser().parse_args(["run", "--suite", "vision", "--workloads", "transformer", "--no-int8"])
    cfg = cli._config_from_args(args)
    assert cfg.suite == "custom" and resolve_workloads(cfg, args.workloads) == ["transformer"]
    for suite in ("missing", "custom"):
        try:
            resolve_workloads(RunConfig(suite=suite))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid/empty measurement set accepted")
    print("Measurement sets passed: presets, default compatibility, INT8, CLI overrides, validation.")


def check_compute_units() -> None:
    from fairbench.report import format_compute_rate

    for kind in ("cpu", "gpu"):
        assert format_compute_rate({"kind": kind, "flops_per_s": 2e12}) == "2.000 TFLOPS"
    assert format_compute_rate({"kind": "npu", "flops_per_s": 2e12}) == "2.000 TOPS"
    assert format_compute_rate({"kind": "npu", "flops_per_s": 0}) == "0.000 TOPS"
    assert format_compute_rate({"kind": "npu", "flops_per_s": float("nan")}) == "—"
    assert format_compute_rate({"kind": "cpu"}) == "—"
    print("Compute units passed: CPU/GPU TFLOPS, NPU TOPS, zero/missing values.")


def check_windowsml_npu_requirement() -> None:
    from fairbench import windowsml

    data = {"winml_version": "test", "winml_notes": [], "winml_devices": [
        {"provider": "VitisAIExecutionProvider", "kind": "gpu", "description": "GPU"}
    ]}
    with patch("sys.argv", ["winml", "--setup", "--provider", "VitisAIExecutionProvider", "--require-npu"]), \
            patch.object(windowsml, "probe", return_value=data) as probe, redirect_stdout(StringIO()):
        assert windowsml.main() == 1
        data["winml_devices"][0]["kind"] = "npu"
        assert windowsml.main() == 0
        assert probe.call_args.kwargs == {"download": True, "provider_name": "VitisAIExecutionProvider"}
    print("Windows ML NPU requirement passed: GPU-only installation cannot report success.")


def check_windowsml_selection() -> None:
    from types import SimpleNamespace

    from fairbench import windowsml
    from fairbench.backends.base import BackendError
    from fairbench.backends.windowsml_backend import WindowsMLBackend

    def device(kind):
        return SimpleNamespace(ep_name="OpenVINOExecutionProvider", ep_options={}, ep_metadata={},
                               device=SimpleNamespace(type=SimpleNamespace(name=kind), vendor_id=0x8086,
                                                      device_id=1, vendor="Intel", metadata={"Description": kind}))
    gpu, npu = device("GPU"), device("NPU")
    assert windowsml.device_info(gpu)["key"] != windowsml.device_info(npu)["key"]
    backend = object.__new__(WindowsMLBackend)
    options = MagicMock()
    session = MagicMock()
    session.get_providers.return_value = ["OpenVINOExecutionProvider"]
    session.get_outputs.return_value = [SimpleNamespace(name="output")]
    factory = MagicMock(return_value=session)
    backend.ort = SimpleNamespace(get_ep_devices=lambda: [gpu, npu], InferenceSession=factory)
    target = Target("winml-test", "winml", "OpenVINOExecutionProvider", "npu", "intel", "NPU",
                    options={"winml_key": windowsml.device_info(npu)["key"]})
    built = SimpleNamespace(serialise=lambda: b"model", key="smoke")
    with patch.object(backend, "_session_options", return_value=options):
        compiled = backend.compile(built, target, RunConfig(), "fp32")
        options.add_provider_for_devices.assert_called_once_with([npu], {})
        options.add_session_config_entry.assert_called_with("session.disable_cpu_ep_fallback", "1")
        assert factory.call_args.kwargs["enable_fallback"] is False
        assert compiled.meta["windowsml_device"]["kind"] == "npu"
        session.get_providers.return_value = ["CPUExecutionProvider"]
        try:
            backend.compile(built, target, RunConfig(), "fp32")
        except BackendError:
            pass
        else:
            raise AssertionError("CPU fallback was reported as NPU execution")
    print("Windows ML checks passed: explicit GPU/NPU distinction, strict CPU fallback rejection.")


def check_report_json() -> None:
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen

    from fairbench import report, server

    def reject_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")
    data = {"schema": "fairbench.result/1", "system": {"hostname": "日本語PC"},
            "results": [], "cells": [{"error": float("nan"), "nested": [float("inf"), -float("inf")]}],
            "literal": "NaN is text", "finite": 1.25}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        legacy = root / "legacy.json"
        legacy.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with patch.object(report, "RESULTS_DIR", root):
            saved = report.save(data, root / "new.json")
            parsed = json.loads(saved.read_text(encoding="utf-8"), parse_constant=reject_constant)
            assert parsed["cells"] == [{"error": None, "nested": [None, None]}]
            assert parsed["literal"] == data["literal"] and parsed["finite"] == 1.25
            assert report.load(legacy) == parsed
            assert len(report.list_results()) == 2
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{httpd.server_port}/api/result/legacy.json") as response:
                    body = json.loads(response.read().decode("utf-8"), parse_constant=reject_constant)
                assert body == parsed
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()
        legacy.write_text(json.dumps(data, ensure_ascii=False), encoding="cp932")
        assert report.load(legacy) == parsed
    print("Report JSON checks passed: strict JSON, legacy NaN/Infinity, Japanese encoding, actual result API.")


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
    from types import SimpleNamespace

    default = cli.build_parser().parse_args(["setup-npu"])
    npu = Target("amd-npu", "winml", "VitisAIExecutionProvider", "npu", "amd", "AMD NPU")
    with (
        patch.object(cli.sys, "platform", "win32"),
        patch.object(cli.system, "collect", return_value=SimpleNamespace(cpu=SimpleNamespace(vendor="amd"))),
        patch.object(cli, "cmd_setup_winml", return_value=0) as setup,
        patch.object(interp, "CACHE", MagicMock()),
        patch.object(interp, "probe_all", return_value=[]),
        patch.object(cli, "discover", return_value=Discovery(targets=[npu])) as discovery,
        patch.object(cli, "verify_targets", return_value={npu.id: {"ok": True}}) as verify,
    ):
        assert default.func(default) == 0
        assert setup.call_args.args[0].provider == "VitisAIExecutionProvider"
        assert setup.call_args.args[0].require_npu is True
        assert verify.call_args.args[0] == [npu]
        discovery.return_value = Discovery(targets=[])
        assert default.func(default) == 1  # An installed GPU does not complete NPU setup.
        discovery.return_value = Discovery(targets=[npu])
        verify.return_value = {npu.id: {"ok": False, "error": "compile failed"}}
        assert default.func(default) == 1
    print("NPU setup checks passed: default Vitis AI acquisition, device and inference gates, legacy installer.")


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


def check_npu_int8_and_memory() -> None:
    from fairbench import models, system, worker
    from fairbench.backends.openvino_backend import OpenVinoBackend
    from fairbench.runner import resolve_workloads

    cfg = RunConfig(suite="npu_peak", include_int8=False, warmup=1, measure_power=False,
                    min_seconds=0.05, max_seconds=2, repeats=1, threads=1)
    assert cfg.mode == "throughput" and cfg.include_int8
    assert resolve_workloads(cfg) == ["dense_static_int8"]
    built = models.build_one("dense_static_int8")
    ops = [n.op_type for n in built.model.graph.node]
    assert ops.count("MatMul") == 2 and "DynamicQuantizeLinear" not in ops
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    target = Target("ov-cpu", "openvino", "CPU", "cpu", "intel", "CPU")
    result = worker.run_cell({"target": target.to_dict(), "workload": built.key,
                              "precision": "int8", "cfg": {**cfg.to_dict(), "requests": 2}})
    assert result["ok"], result
    assert result["correctness"]["valid"], result["correctness"]
    assert result["concurrent_requests"] == 2 and result["flops_per_s"] > 0
    assert result["session"]["quantization"] == "static_qdq"
    backend = OpenVinoBackend()
    npu = Target("npu", "openvino", "NPU", "npu", "intel", "NPU")
    assert backend.ov.properties.hint.inference_precision() not in backend._properties(npu, cfg, "int8")
    counts = []
    def measured(runners, config):
        counts.append(len(runners))
        return {"latencies_ms": [1.0], "start": 0, "end": 1, "wall_s": 1,
                "requests": len(runners), "iterations": 1, "units_per_s": len(runners),
                "host_cpu_seconds": 0.1}
    with patch("fairbench.throughput.measure", side_effect=measured):
        sweep = worker._run_throughput(MagicMock(), MagicMock(output_names=[]), built, npu, cfg,
                                       {}, {"warnings": []}, {"precision": "int8"})
    assert counts == [1, 2, 4, 8, 16, 32] and sweep["concurrent_requests"] == 32
    mem = system.MemoryInfo(total_gib=0.03125)
    with patch.object(system.time, "perf_counter", side_effect=range(10)):
        system.measure_memory_bandwidth(mem)
    assert mem.bandwidth["gb_per_s"] == 2 * mem.bandwidth["buffer_bytes"] / 1e9
    assert mem.bandwidth["samples"] == 5
    with patch.object(system, "IS_WINDOWS", True), patch.object(system, "_powershell_json", return_value=[
        {"Capacity": 8 * 1024**3, "Speed": 6400, "ConfiguredClockSpeed": 5600, "DataWidth": 64}]):
        info = system._collect_memory()
    assert info.speed_mts == 5600 and info.modules[0]["data_width_bits"] == 64
    print("NPU preset, static QDQ INT8 correctness on CPU, parallel sweep, precision policy and RAM accounting passed.")


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
    check_npu_int8_and_memory()
    check_backend_selection()
    check_measurement_sets()
    check_compute_units()
    check_windowsml_npu_requirement()
    check_windowsml_selection()
    check_report_json()
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
