"""Command line interface: ``python -m fairbench <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__, models, paths, system
from . import report as report_mod
from .config import RunConfig
from .devices import EXTRA_SLOTS, SLOTS, discover
from .runner import WORKLOAD_META, WORKLOAD_SETS, resolve_workloads, run_suite, verify_targets

# --------------------------------------------------------------------------
# pretty printing
# --------------------------------------------------------------------------
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def _colour_enabled() -> bool:
    return sys.stdout.isatty()


def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if _colour_enabled() else text


def banner() -> None:
    art = r"""
   __       _        _                     _
  / _| __ _(_)_ __  | |__   ___ _ __   ___| |__
 | |_ / _` | | '__| | '_ \ / _ \ '_ \ / __| '_ \
 |  _| (_| | | |    | |_) |  __/ | | | (__| | | |
 |_|  \__,_|_|_|    |_.__/ \___|_| |_|\___|_| |_|
"""
    print(c(art, MAGENTA))
    print(f"  {c('vendor-neutral AI benchmark', BOLD)} for laptop CPU / GPU / NPU")
    print(f"  {c('Intel vs AMD, identical work, identical measurement', DIM)}")
    print(f"  {c('v' + __version__, DIM)}\n")


def cmd_devices(args: argparse.Namespace) -> int:
    fp = system.collect()
    disc = discover(fp)
    print(f"{c('MACHINE', BOLD)}")
    print(f"  CPU        {fp.cpu.name}  ({fp.cpu.physical_cores}C/{fp.cpu.logical_cores}T)")
    print(f"  Memory     {fp.memory.total_gib} GiB" + (f" @ {fp.memory.speed_mts} MT/s" if fp.memory.speed_mts else ""))
    print(f"  OS         {fp.os_name} {fp.os_release} (build {fp.os_build})")
    if fp.power.power_plan:
        print(f"  Power      {fp.power.power_plan}")
    print(f"  Battery    {'present' if fp.power.battery_present else 'none'}"
          + (f", {'AC' if fp.power.on_ac else 'discharging'}" if fp.power.battery_present else ""))
    for a in fp.adapters:
        print(f"  {a.kind.upper():10s} {a.name}  [{a.vendor}]" + (f"  driver {a.driver}" if a.driver else ""))

    print(f"\n{c('RUNTIMES', BOLD)}")
    for name, version in disc.runtimes.items():
        print(f"  {name:12s} {version}")

    print(f"\n{c('TARGET SLOTS', BOLD)}")
    for slot in SLOTS + EXTRA_SLOTS:
        cap = disc.capabilities.get(slot.id, {})
        ok = cap.get("available")
        mark = c("AVAILABLE", GREEN) if ok else c("--------", DIM)
        print(f"  [{mark}] {slot.label:32s} {c(slot.access, DIM)}")
        if ok:
            for tid in cap.get("targets", []):
                target = disc.get(tid)
                print(f"              {c('->', CYAN)} {tid:34s} {target.detail if target else ''}")
        else:
            print(f"              {c('->', DIM)} {cap.get('detail', '')}")

    if disc.notes:
        print(f"\n{c('NOTES', BOLD)}")
        for note in disc.notes:
            print(f"  - {note}")
    return 0


def cmd_workloads(args: argparse.Namespace) -> int:
    print(f"{c('WORKLOADS', BOLD)}  (identical ONNX graphs on every device)\n")
    for key in list(models.BASE_BUILDERS) + ["smoke"]:
        built = models.build_one(key)
        meta = WORKLOAD_META.get(key, {})
        print(f"  {c(key, CYAN):30s} {built.flops / 1e9:6.2f} GFLOP  {built.size_bytes / 1e6:6.1f} MB  [{built.unit}]")
        print(f"      {meta.get('blurb', '')}")
    print(f"\n  {c('INT8 variants', BOLD)} (post-training dynamic quantisation, reproducible)")
    for base in models.QUANTISABLE:
        q = models.build_one(f"{base}_int8")
        print(f"  {c(base + '_int8', CYAN):30s} {q.flops / 1e9:6.2f} GFLOP  {q.size_bytes / 1e6:6.1f} MB")
    return 0


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(
        backend=args.backend,
        suite="custom" if args.workloads else args.suite,
        mode=args.mode,
        requests=args.requests,
        warmup=args.warmup,
        min_iterations=args.iterations,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        repeats=args.repeats,
        cooldown_seconds=args.cooldown,
        threads=args.threads,
        measure_power=not args.no_power,
        allow_battery=args.allow_battery,
        check_correctness=not args.no_check,
        include_int8=not args.no_int8,
    )
    if args.quick:
        cfg.warmup = 2
        cfg.min_iterations = 5
        cfg.min_seconds = 1.0
        cfg.max_seconds = 15.0
        cfg.repeats = 1
        cfg.cooldown_seconds = 0.5
    if args.full:
        cfg.warmup = 8
        cfg.min_iterations = 30
        cfg.min_seconds = 10.0
        cfg.max_seconds = 90.0
        cfg.repeats = 5
        cfg.cooldown_seconds = 5.0
    if args.threads == 0:
        cfg.threads = None
    return cfg


def _print_progress(ev: dict) -> None:
    kind = ev.get("event")
    if kind == "verify":
        mark = c(" ok ", GREEN) if ev["ok"] else c("fail", RED)
        suffix = "" if ev["ok"] else f"  {c(str(ev.get('error')).splitlines()[0][:90], DIM)}"
        print(f"  [{mark}] verify {ev['target']}{suffix}")
    elif kind == "cell":
        mark = c(" ok ", GREEN) if ev["ok"] else c("fail", RED)
        if ev["ok"]:
            detail = f"{ev['units_per_s']:9.2f} units/s   {ev['median_ms']:8.3f} ms"
        else:
            detail = c(str(ev.get("error")).splitlines()[0][:90], DIM)
        print(f"  [{mark}] {ev['index']:>3}/{ev['total']} {ev['target_label']:<26s} {ev['workload']:<18s} {detail}")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    fp = system.collect()
    cfg.physical_cores = fp.cpu.physical_cores
    cfg.logical_cores = fp.cpu.logical_cores

    banner()
    print(f"{c('HOST', BOLD)}  {fp.cpu.name}  {fp.cpu.physical_cores}C/{fp.cpu.logical_cores}T  "
          f"{fp.memory.total_gib} GiB")
    if fp.power.power_plan:
        print(f"{c('POWER', BOLD)} {fp.power.power_plan}  ({'AC' if fp.power.on_ac else 'battery' if fp.power.battery_present else 'desktop'})")
    if fp.power.battery_present and not fp.power.on_ac and not args.allow_battery:
        print(c("\n  Refusing to run: the laptop is on battery, which caps clocks.", YELLOW))
        print("  Plug in, or pass --allow-battery to record the number anyway.")
        return 2

    workloads = resolve_workloads(cfg, args.workloads)
    print(f"{c('PLAN', BOLD)}  {len(workloads)} workloads x {cfg.repeats} repeat(s), "
          f"{cfg.min_seconds:g}s minimum per trial, mode={cfg.mode}, requests={cfg.requests or 'auto'}, threads={cfg.resolve_threads()}")
    print(f"{c('WORK', BOLD)}  {', '.join(workloads)}\n")

    t0 = time.time()
    report = run_suite(cfg, target_ids=args.targets, workloads=workloads, verify=not args.no_verify, progress=_print_progress, fingerprint=fp)

    elapsed = time.time() - t0
    print(f"\n{c('DONE', BOLD)} in {elapsed:.0f}s\n")
    _print_summary(report)

    path = report_mod.save(report, args.out)
    print(f"\n{c('saved', BOLD)} {path}")

    if not args.no_open:
        _serve_and_open(path, args.port, background=True)
    return 0


def _print_summary(report: dict) -> None:
    rows = [r for r in report.get("results", []) if r.get("ok")]
    if not rows:
        print(c("  Nothing completed successfully.", RED))
        return
    print(f"  {'device':<26s} {'workload':<18s} {'unit/s':>10s} {'median ms':>10s} {'compute':>18s} {'idx':>6s}")
    print(f"  {'-' * 26} {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 18} {'-' * 6}")
    index = {i["target_id"]: i["ai_index"] for i in (report.get("scores") or {}).get("ai_index", [])}
    for r in sorted(rows, key=lambda r: (r["workload"], -(r.get("units_per_s") or 0))):
        idx = index.get(r["target_id"])
        print(
            f"  {r['target_label']:<26s} {r['workload']:<18s} "
            f"{r.get('units_per_s') or 0:>10.2f} {r.get('latency_ms') or 0:>10.3f} "
            f"{report_mod.format_compute_rate(r):>18s} "
            f"{(f'{idx:.2f}x' if idx else ''):>6s}"
        )
    grade = report.get("grade") or {}
    letter = grade.get("letter", "?")
    colour = GREEN if letter in ("A", "B") else YELLOW if letter == "C" else RED
    print(f"\n  {c('Environment grade', BOLD)}: {c(letter, colour)}  ({grade.get('score')}/100)")
    for check in grade.get("checks", []):
        mark = {"pass": c("pass", GREEN), "warn": c("warn", YELLOW), "fail": c("fail", RED), "skip": c("skip", DIM)}[check["status"]]
        print(f"    [{mark}] {check['name']:<18s} {check['detail']}")


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = RunConfig(repeats=1, measure_power=False, physical_cores=system.collect().cpu.physical_cores)
    fp = system.collect()
    disc = discover(fp)
    targets = [t for t in disc.available() if not args.targets or t.id in args.targets or t.backend in args.targets]
    banner()
    print(f"{c('VERIFYING', BOLD)} {len(targets)} target(s)\n")
    verdicts = verify_targets(targets, cfg, _print_progress)
    ok = [k for k, v in verdicts.items() if v["ok"]]
    print(f"\n  {len(ok)}/{len(targets)} targets can execute models.")
    for k, v in verdicts.items():
        if not v["ok"]:
            print(f"  {c('dropped', RED)} {k}: {v['error']}")
    return 0


def _serve_and_open(path: Path | None, port: int, background: bool = False) -> int:
    from .server import serve

    url = f"http://127.0.0.1:{port}/"
    print(f"{c('dashboard', BOLD)} {url}")
    if path:
        print(f"  opening {path.name}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    serve(port=port)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    banner()
    if not args.no_open:
        try:
            webbrowser.open(f"http://127.0.0.1:{args.port}/")
        except Exception:
            pass
    serve(port=args.port)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    data = report_mod.load(args.file)
    _print_summary(data)
    return 0


# --------------------------------------------------------------------------
def cmd_setup_winml(args: argparse.Namespace) -> int:
    import shutil
    import subprocess

    from . import interp, worker_process

    if sys.platform != "win32" or sys.getwindowsversion().build < 26100:
        print("Windows ML EP Catalog requires Windows 11 24H2 (build 26100) or newer.")
        return 2
    uv = shutil.which("uv")
    if not uv:
        print("Install uv first, then retry setup-winml.")
        return 2
    target = paths.work_root() / ".venv-winml"
    python = target / "Scripts" / "python.exe"
    if not python.exists():
        result = subprocess.run([uv, "venv", "--python", "3.12", str(target)])
        if result.returncode:
            return result.returncode
    result = subprocess.run([uv, "pip", "install", "--python", str(python),
                             "windowsml[with-ort]==2.4.89", "onnx>=1.16"])
    if result.returncode:
        return result.returncode
    command = [str(python), "-m", "fairbench.windowsml", "--setup"]
    if args.provider:
        command += ["--provider", args.provider]
    if args.require_npu:
        command += ["--require-npu"]
    result = subprocess.run(command, env=worker_process._worker_env())
    interp.CACHE.unlink(missing_ok=True)
    if result.returncode:
        return result.returncode
    print("Windows ML providers prepared. Run 'uv run fairbench verify --targets winml' before measurement.")
    return 0


def cmd_setup_npu(args: argparse.Namespace) -> int:
    """Prepare AMD Vitis AI and prove an NPU inference can complete."""
    import subprocess

    if sys.platform != "win32":
        print("Automatic AMD NPU setup currently supports Windows only.")
        return 2
    if not args.check:
        if args.sdk or args.installer:
            script = Path(__file__).with_name("setup_amd_npu.ps1")
            command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
            if args.installer:
                command += ["-Installer", str(Path(args.installer).resolve())]
            result = subprocess.run(command)
            if result.returncode:
                return result.returncode
        else:
            fp = system.collect()
            if fp.cpu.vendor != "amd":
                print("Run setup-npu on the AMD laptop. No installation was started.")
                return 2
            result = cmd_setup_winml(argparse.Namespace(provider="VitisAIExecutionProvider", require_npu=True))
            if result:
                return result
    from . import interp

    interp.CACHE.unlink(missing_ok=True)
    interpreters = interp.probe_all(refresh=True)
    for info in interpreters:
        print(f"{info.path}: {', '.join(info.providers) or 'no providers'}")
        for device in info.winml_devices:
            print(f"  Windows ML {device['kind'].upper()}: {device['provider']} - {device['description']}")
        for note in info.winml_notes:
            print(f"  {note}")
        if info.error:
            print(f"  {info.error}")
    disc = discover()
    targets = [target for target in disc.available()
               if target.vendor == "amd" and target.kind == "npu" and target.device == "VitisAIExecutionProvider"]
    if not targets:
        print("AMD Vitis AI NPU was not detected. Install/update the AMD NPU driver and Windows, then retry setup-npu. "
              "For an existing Ryzen AI SDK environment, set FAIRBENCH_PYTHON to its python.exe. "
              "https://ryzenai.docs.amd.com/projects/WinML/en/latest/installation.html")
        return 1
    verdicts = verify_targets(targets, RunConfig(measure_power=False), _print_progress)
    if not any(verdict.get("ok") for verdict in verdicts.values()):
        print("Vitis AI was found but NPU inference failed. See the verification error above; setup is incomplete.")
        return 1
    print("AMD Vitis AI NPU inference verified.")
    return 0


def cmd_setup_gpu(args: argparse.Namespace) -> int:
    """Create an isolated interpreter that has a GPU-capable ONNX Runtime.

    ONNX Runtime's wheels all install into the same ``onnxruntime`` package, so
    adding DirectML or CUDA to an environment that already has the CPU build
    would displace it. A throwaway virtual environment avoids that entirely and
    costs nothing at benchmark time, because fairbench already runs every cell
    in a child process and can simply launch the right one.

    ``uv`` is used when it is on PATH - it is dramatically faster and it is what
    this project is set up around - with the stdlib ``venv`` as a fallback so the
    packaged build still works on a machine with no uv.
    """
    import shutil
    import subprocess
    import venv as venv_mod

    root = paths.work_root()
    name = args.name or (".venv-cuda" if args.cuda else ".venv-gpu")
    target = Path(name) if Path(name).is_absolute() else root / name
    python = target / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")

    banner()
    uv = shutil.which("uv")
    if python.exists() and not args.force:
        print(f"{c('exists', YELLOW)}  {target}  (pass --force to rebuild)")
    elif uv:
        print(f"{c('creating', BOLD)} {target}  {c('via uv', DIM)}")
        if args.force and target.exists():
            shutil.rmtree(target, ignore_errors=True)
        if subprocess.run([uv, "venv", "--python", sys.executable, str(target)]).returncode != 0:
            print(c("  uv venv failed", RED))
            return 1
    else:
        print(f"{c('creating', BOLD)} {target}  {c('via python -m venv', DIM)}")
        venv_mod.EnvBuilder(with_pip=True, clear=args.force).create(str(target))

    packages = ["numpy", "onnx"]
    if args.cuda:
        packages += ["onnxruntime-gpu", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
    else:
        packages += ["onnxruntime-directml"]
    print(f"{c('installing', BOLD)} {' '.join(packages)}")
    if uv:
        cmd = [uv, "pip", "install", "--python", str(python), *packages]
    else:
        cmd = [str(python), "-m", "pip", "install", "--upgrade", *packages]
    if subprocess.run(cmd).returncode != 0:
        print(c("  install failed", RED))
        return 1

    # Invalidate the probe cache so discovery sees the new interpreter at once.
    from . import interp

    interp.CACHE.unlink(missing_ok=True)
    print(f"\n{c('probing', BOLD)}")
    found = False
    for info in interp.probe_all(refresh=True):
        mark = c("ok  ", GREEN) if info.available else c("fail", RED)
        provs = ", ".join(p for p in info.providers if p != "AzureExecutionProvider") or (info.error or "-")
        print(f"  [{mark}] {info.path}")
        print(f"         onnxruntime {info.onnxruntime or '-'}  providers: {provs}")
        if not info.primary and info.available and info.providers:
            found = True

    if found:
        print(f"\n{c('done', GREEN)} - run '{c('fairbench devices', CYAN)}' to see the new devices.")
    else:
        print(f"\n{c('no new devices appeared.', YELLOW)}")
        if args.cuda:
            print("  CUDA needs a matching driver, CUDA runtime and cuDNN. On Windows, DirectML")
            print("  (no --cuda) reaches an NVIDIA GPU without any of that and is usually the")
            print("  better choice for a benchmark, because it is the same runtime on AMD hardware.")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show how this build is running - useful in a bug report."""
    import platform

    banner()
    print(f"{c('BUILD', BOLD)}")
    print(f"  fairbench     {__version__}")
    print(f"  packaged      {'yes (frozen executable)' if paths.FROZEN else 'no (running from source)'}")
    print(f"  executable    {sys.executable}")
    print(f"  app root      {paths.app_root()}")
    print(f"  module root   {paths.project_root()}")
    print(f"  resources     {paths.resource_dir()}  {'ok' if paths.resource_dir().exists() else 'MISSING'}")
    print(f"  worker source {'available' if paths.worker_module_available() else 'missing (external GPU interpreters disabled)'}")
    print(f"  python        {platform.python_version()} on {platform.platform()}")
    print(f"  cwd           {Path.cwd()}")

    from . import interp

    print(f"\n{c('INTERPRETERS', BOLD)}")
    for info in interp.probe_all():
        mark = c("ok  ", GREEN) if info.available else c("fail", RED)
        provs = ", ".join(p for p in info.providers if p != "AzureExecutionProvider") or (info.error or "-")
        print(f"  [{mark}] {info.label:22s} {info.path}")
        print(f"         onnxruntime {info.onnxruntime or '-'}  providers: {provs}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Prove this build works end to end, and say exactly what is broken if not.

    Written for the packaged build, where the user cannot inspect a traceback
    easily and where a missing optional module turns into a silently empty
    column rather than an error.
    """
    import traceback

    from .backends import installed_backends

    banner()
    ok = True

    print(f"{c('1. workloads', BOLD)}")
    keys = [*models.BASE_BUILDERS, *(f"{k}_int8" for k in models.QUANTISABLE), "smoke"]
    for key in keys:
        try:
            built = models.build_one(key)
            print(f"  [{c('ok  ', GREEN)}] {key:22s} {built.flops / 1e9:7.2f} GFLOP  {built.size_bytes / 1e6:7.1f} MB  {built.sha256[:12]}")
        except Exception as exc:
            ok = False
            print(f"  [{c('fail', RED)}] {key:22s} {exc}")

    print(f"\n{c('2. backends', BOLD)}")
    backends = installed_backends()
    for bid, backend in backends.items():
        print(f"  [{c('ok  ', GREEN)}] {backend.name} ({bid})")
    if not backends:
        ok = False
        print(f"  [{c('fail', RED)}] no backend is importable")

    print(f"\n{c('3. devices', BOLD)}")
    fp = system.collect()
    disc = discover(fp)
    runnable = []
    for target in disc.available():
        backend = backends.get(target.backend)
        if backend is None:
            print(f"  [{c('skip', DIM)}] {target.id:38s} backend '{target.backend}' not installed")
            continue
        try:
            cfg = RunConfig(physical_cores=fp.cpu.physical_cores, logical_cores=fp.cpu.logical_cores)
            built = models.build_one("smoke")
            compiled = backend.compile(built, target, cfg, "fp32")
            backend.run(compiled, built.make_inputs())
            print(f"  [{c('ok  ', GREEN)}] {target.id:38s} {target.detail[:40]}")
            runnable.append(target)
        except Exception as exc:
            ok = False
            print(f"  [{c('fail', RED)}] {target.id:38s} {str(exc).splitlines()[0][:90]}")
            if args.verbose:
                print(traceback.format_exc()[-800:])
    if not runnable:
        ok = False
        print(f"  {c('no device could execute a model', RED)}")

    print(f"\n{c('4. worker process', BOLD)}")
    if runnable:
        try:
            verdict = verify_targets([runnable[0]], RunConfig(measure_power=False, repeats=1, physical_cores=fp.cpu.physical_cores))[runnable[0].id]
            if verdict.get("ok"):
                print(f"  [{c('ok  ', GREEN)}] isolated child process handled the probe ({verdict.get('latency_ms'):.2f} ms)")
            else:
                ok = False
                print(f"  [{c('fail', RED)}] child process failed: {verdict.get('error')}")
        except Exception as exc:
            ok = False
            print(f"  [{c('fail', RED)}] {exc}")
    else:
        print(f"  [{c('skip', DIM)}] no device to test with")

    print(f"\n{c('5. dashboard assets', BOLD)}")
    for name in ("index.html", "style.css", "app.js"):
        exists = (paths.resource_dir() / name).is_file()
        ok = ok and exists
        print(f"  [{c('ok  ', GREEN) if exists else c('fail', RED)}] {name}")

    print()
    if ok:
        print(f"{c('all checks passed', GREEN)}")
        return 0
    print(f"{c('some checks failed', RED)} - see above")
    return 1


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fairbench", description="Vendor-neutral AI benchmark for laptop CPU / GPU / NPU")
    parser.add_argument("--version", action="version", version=f"fairbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dev = sub.add_parser("devices", help="list this machine's AI devices and runtimes")
    p_dev.set_defaults(func=cmd_devices)

    p_wl = sub.add_parser("workloads", help="list the benchmark workloads")
    p_wl.set_defaults(func=cmd_workloads)

    p_ver = sub.add_parser("verify", help="check which devices can actually execute a model")
    p_ver.add_argument("--targets", nargs="*", default=None)
    p_ver.set_defaults(func=cmd_verify)

    p_run = sub.add_parser("run", help="run the benchmark suite")
    p_run.add_argument("--backend", choices=["all", "ort", "openvino", "winml"], default="all", help="runtime to benchmark (default: all available)")
    p_run.add_argument("--suite", choices=list(WORKLOAD_SETS), default="standard", help="measurement workload set; --workloads overrides it")
    p_run.add_argument("--mode", choices=["latency", "throughput"], default="latency")
    p_run.add_argument("--requests", type=int, choices=range(33), default=0, metavar="0..32", help="throughput concurrency; 0 sweeps 1, 2, 4, 8")
    p_run.add_argument("--targets", nargs="*", default=None, help="target ids from `fairbench devices`")
    p_run.add_argument("--workloads", nargs="*", default=None, help="workload keys (default: a representative set)")
    p_run.add_argument("--out", default=None, help="output json path")
    p_run.add_argument("--quick", action="store_true", help="fast, low-confidence settings")
    p_run.add_argument("--full", action="store_true", help="slow, publication-grade settings")
    p_run.add_argument("--repeats", type=int, default=3)
    p_run.add_argument("--warmup", type=int, default=5)
    p_run.add_argument("--iterations", type=int, default=15, help="minimum timed iterations per cell")
    p_run.add_argument("--min-seconds", type=float, default=4.0)
    p_run.add_argument("--max-seconds", type=float, default=45.0)
    p_run.add_argument("--cooldown", type=float, default=3.0, help="seconds between cells")
    p_run.add_argument("--threads", type=int, default=None, help="CPU threads (default: physical cores)")
    p_run.add_argument("--no-int8", action="store_true", help="skip the INT8 variants")
    p_run.add_argument("--no-check", action="store_true", help="skip output verification")
    p_run.add_argument("--no-power", action="store_true", help="skip power sampling")
    p_run.add_argument("--no-verify", action="store_true", help="skip device verification")
    p_run.add_argument("--allow-battery", action="store_true")
    p_run.add_argument("--port", type=int, default=8765)
    p_run.add_argument("--no-open", action="store_true", help="do not open the dashboard")
    p_run.set_defaults(func=cmd_run)

    p_serve = sub.add_parser("serve", help="start the dashboard web app")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--no-open", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_show = sub.add_parser("show", help="print a saved result")
    p_show.add_argument("file")
    p_show.set_defaults(func=cmd_show)

    p_gpu = sub.add_parser("setup-gpu", help="create an isolated environment with a GPU-capable ONNX Runtime")
    p_gpu.add_argument("--name", default=None, help="virtual environment directory (default .venv-gpu / .venv-cuda)")
    p_gpu.add_argument("--cuda", action="store_true", help="install onnxruntime-gpu + CUDA runtime instead of DirectML")
    p_gpu.add_argument("--force", action="store_true", help="rebuild an existing environment")
    p_gpu.set_defaults(func=cmd_setup_gpu)

    p_winml = sub.add_parser("setup-winml", help="prepare Windows ML and download compatible certified EPs")
    p_winml.add_argument("--provider", help="download only this execution provider (e.g. VitisAIExecutionProvider)")
    p_winml.add_argument("--require-npu", action="store_true", help="fail if the requested NPU is not exposed")
    p_winml.set_defaults(func=cmd_setup_winml)

    p_npu = sub.add_parser("setup-npu", help="prepare and verify AMD Vitis AI NPU through Windows ML")
    p_npu.add_argument("--sdk", action="store_true", help="use the legacy AMD driver/SDK installer instead of Windows ML")
    p_npu.add_argument("--check", action="store_true", help="refresh AMD runtime diagnostics without installing anything")
    p_npu.add_argument("--installer", help="previously downloaded official Ryzen AI installer (.exe)")
    p_npu.set_defaults(func=cmd_setup_npu)

    p_info = sub.add_parser("info", help="show where this build lives and how it is running")
    p_info.set_defaults(func=cmd_info)

    p_self = sub.add_parser("selftest", help="verify this build end to end and diagnose what is missing")
    p_self.add_argument("-v", "--verbose", action="store_true")
    p_self.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Hidden entry points. In a packaged build these are how the app re-enters
    # itself: `--worker` runs one measurement cell, `--probe-env` reports the
    # bundled ONNX Runtime providers to the interpreter registry.
    if argv and argv[0] == "--worker":
        from . import worker

        return worker.main([sys.argv[0], *argv[1:]])
    if argv and argv[0] == "--probe-env":
        from . import interp

        sys.stdout.write(interp.MARKER + json.dumps(interp.payload()))
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
