"""Launch isolated measurement workers and translate process failures into results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import interp, paths

JOB_DIR = Path(".cache") / "jobs"


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(paths.project_root())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _worker_command(python: str, job_path: Path, out_path: Path) -> list[str]:
    """How to launch a measurement cell.

    In a packaged build the "interpreter" is the executable itself, so the
    module entry point is replaced by a hidden ``--worker`` flag. A device that
    lives in a *real* external interpreter (a DirectML venv) still uses
    ``-m fairbench.worker``, which is why the packaged build ships the source.
    """
    if paths.is_own_interpreter(python) and paths.FROZEN:
        return [python, "--worker", str(job_path), str(out_path)]
    return interp.python_command(python, ["-X", "faulthandler", "-m", "fairbench.worker", str(job_path), str(out_path)])


def spawn_worker(job: dict[str, Any], timeout: float, tag: str) -> dict[str, Any]:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    # Absolute paths: the child may run with a different working directory.
    job_path = (JOB_DIR / f"{tag}.job.json").resolve()
    out_path = (JOB_DIR / f"{tag}.out.json").resolve()
    job_path.write_text(json.dumps(job))
    if out_path.exists():
        out_path.unlink()

    python = job.get("target", {}).get("python") or sys.executable
    if paths.FROZEN and not paths.is_own_interpreter(python) and not paths.worker_module_available():
        return {
            "ok": False,
            "target_id": job["target"]["id"],
            "workload": job["workload"],
            "error": (
                f"this device needs the external interpreter {python}, but the packaged build "
                "cannot find the fairbench worker source next to the executable"
            ),
            "phase": "spawn",
        }
    started = time.perf_counter()
    try:
        cmd = _worker_command(python, job_path, out_path)
        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            env=_worker_env(),
            capture_output=True,
            timeout=timeout,
        )
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        returncode = proc.returncode
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "target_id": job["target"]["id"],
            "workload": job["workload"],
            "error": f"interpreter or Conda launcher for this device is missing: {python}: {exc}",
            "phase": "spawn",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "target_id": job["target"]["id"],
            "workload": job["workload"],
            "error": f"device did not respond within {timeout:.0f}s (hung or extremely slow)",
            "phase": "timeout",
            "elapsed_s": time.perf_counter() - started,
        }

    if out_path.exists():
        try:
            result = json.loads(out_path.read_text())
            result.setdefault("worker_returncode", returncode)
            if not result.get("ok") and not result.get("error"):
                result["error"] = _crash_hint(returncode, stderr)
            if not result.get("ok"):
                log_path = job_path.with_suffix('.log')
                log_path.write_text(f"stdout:\n{stdout}\nstderr:\n{stderr}\nresult:\n"
                                    + json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                result["worker_log"] = str(log_path)
                result["error"] = (result.get("error") or "worker failed") + f"\nWorker log: {log_path}"
                if stderr.strip():
                    result["error"] += "\n" + stderr.strip()[-8000:]
            return result
        except Exception as exc:
            return {
                "ok": False,
                "target_id": job["target"]["id"],
                "workload": job["workload"],
                "error": f"could not parse worker output: {exc}",
                "phase": "parse",
            }
    return {
        "ok": False,
        "target_id": job["target"]["id"],
        "workload": job["workload"],
        "error": _crash_hint(returncode, stderr),
        "phase": "crash",
        "elapsed_s": time.perf_counter() - started,
    }


def _crash_hint(returncode: int | None, stderr: str) -> str:
    if returncode is not None and returncode < 0:
        codes = {
            -1073741819: "access violation (0xC0000005)",
            -1073740791: "stack buffer overrun / abort (0xC0000409)",
            -1073741571: "stack overflow (0xC00000FD)",
        }
        base = f"device driver aborted the process: {codes.get(returncode, hex(returncode & 0xFFFFFFFF))}"
    else:
        base = f"worker exited with code {returncode}"
    tail = "\n".join(line for line in stderr.strip().splitlines()[-4:] if line.strip())
    return f"{base}\n{tail}" if tail else base
