"""Interpreter registry.

ONNX Runtime ships one Execution-Provider set per wheel, and the wheels collide:
``onnxruntime``, ``onnxruntime-gpu``, ``onnxruntime-directml`` and
``onnxruntime-vitisai`` all install into the same ``onnxruntime`` package. A
user who already has ``onnxruntime`` for other tools therefore cannot simply
``pip install onnxruntime-directml`` without displacing it.

fairbench solves that the way it solves everything else - by measuring rather
than assuming. It probes every interpreter it can find, records which Execution
Providers each one exposes, and runs each device in the interpreter that
actually has it. The main environment is never modified.

The packaged build has one wrinkle: the "interpreter" running the app is not
Python at all, so it is probed through the app's own ``--probe-env`` entry point
instead of ``python -c``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import paths

CACHE = Path(".cache") / "interpreters.json"
MARKER = "@@FAIRBENCH-ENV@@"

# Executed in a *candidate* interpreter. Kept short and dependency-free so it can
# run in an environment that has nothing but ONNX Runtime installed.
PROBE_SOURCE = (
    "import json,sys,platform\n"
    "p={'python':sys.version.split()[0],'platform':platform.platform()}\n"
    "try:\n"
    " import onnxruntime as ort\n"
    " p['onnxruntime']=getattr(ort,'__version__','?')\n"
    " p['providers']=list(ort.get_available_providers())\n"
    "except Exception as e:\n"
    " p['onnxruntime']=None;p['providers']=[]\n"
    " p['error']=type(e).__name__+': '+str(e)\n"
    "try:\n"
    " import importlib.util\n"
    " if importlib.util.find_spec('windowsml') is not None:\n"
    "  from fairbench.windowsml import probe\n"
    "  p.update(probe())\n"
    "except Exception as e:\n"
    " p['winml_notes']=[str(e)]\n"
    "sys.stdout.write(" + repr(MARKER) + "+json.dumps(p))\n"
)

# Names we look for automatically, in priority order.
VENV_NAMES = (".venv-winml", ".venv-gpu", ".venv-dml", ".venv-cuda", ".venv-ryzenai", ".venv")


@dataclass
class Interpreter:
    path: str
    label: str = ""
    available: bool = False  # the probe ran at all
    python: str | None = None
    onnxruntime: str | None = None
    providers: list[str] = field(default_factory=list)
    platform: str | None = None
    error: str | None = None
    primary: bool = False
    bundled: bool = False
    winml_devices: list[dict] = field(default_factory=list)
    winml_notes: list[str] = field(default_factory=list)
    winml_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------
def payload() -> dict[str, Any]:
    """What this interpreter has. Shared by the subprocess and the frozen path."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        import onnxruntime as ort

        info["onnxruntime"] = getattr(ort, "__version__", "?")
        info["providers"] = list(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001 - reporting the failure is the point
        info["onnxruntime"] = None
        info["providers"] = []
        info["error"] = f"{type(exc).__name__}: {exc}"
    if sys.platform == "win32":
        import importlib.util

        if importlib.util.find_spec("windowsml") is not None:
            try:
                from .windowsml import probe

                info.update(probe())
            except Exception as exc:
                info["winml_notes"] = [str(exc)]
    return info


def _venv_python(root: Path, name: str) -> Path | None:
    for candidate in (root / name / "Scripts" / "python.exe", root / name / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def candidates() -> list[str]:
    """Every interpreter worth probing, primary first, deduplicated.

    Virtual environments are looked for both next to the application and in the
    working directory, because an installed package has no natural home of its
    own.
    """
    found: list[str] = [sys.executable]

    extra = os.environ.get("FAIRBENCH_PYTHON", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            found.append(item.strip())

    # AMD's installer creates a Conda environment, not a project-local venv.
    # Conda records custom installation paths here as well as default ones.
    registry = Path.home() / ".conda" / "environments.txt"
    prefixes = [os.environ.get("CONDA_PREFIX", "")]
    try:
        prefixes += registry.read_text(encoding="utf-8").splitlines()
    except OSError:
        pass
    for prefix in prefixes:
        if prefix.strip():
            root = Path(prefix.strip())
            for python in (root / "python.exe", root / "bin" / "python"):
                if python.is_file():
                    found.append(str(python))
                    break

    roots: list[Path] = []
    for root in (paths.work_root(), Path.cwd()):
        if root not in roots:
            roots.append(root)
    for root in roots:
        for name in VENV_NAMES:
            py = _venv_python(root, name)
            if py:
                found.append(str(py))

    seen: set[str] = set()
    unique: list[str] = []
    for path in found:
        try:
            key = str(Path(path).resolve())
        except Exception:
            key = path
        if key.lower() in seen:
            continue
        seen.add(key.lower())
        unique.append(key)
    return unique


def python_command(interpreter: str, arguments: list[str]) -> list[str]:
    """Activate external Conda runtimes for both discovery and measurement.

    DLL search paths and vendor activate.d scripts are part of a Conda runtime;
    launching its python.exe directly does not reproduce conda activate.
    """
    python = Path(interpreter)
    prefix = python.parent
    if prefix.name.lower() in ("bin", "scripts"):
        prefix = prefix.parent
    if not paths.is_own_interpreter(interpreter) and (prefix / "conda-meta").is_dir():
        base = prefix.parent.parent if prefix.parent.name.lower() == "envs" else prefix
        candidates = [
            os.environ.get("CONDA_EXE", ""),
            str(base / "Scripts" / "conda.exe"),
            str(base / "bin" / "conda"),
            shutil.which("conda.exe") or shutil.which("conda") or "",
        ]
        conda = next((path for path in candidates if path and Path(path).is_file()), None)
        if not conda:
            raise FileNotFoundError(
                f"Conda environment found at {prefix}, but its conda executable was not found. "
                "Run fairbench from the Miniforge Prompt or set CONDA_EXE to the base installation's conda.exe."
            )
        return [conda, "run", "--no-capture-output", "--prefix", str(prefix), interpreter, *arguments]
    return [interpreter, *arguments]


def _signature(path: str) -> str:
    python = Path(path)
    prefix = python.parent.parent if python.parent.name.lower() in ("bin", "scripts") else python.parent
    # Installing SDK packages leaves python.exe unchanged. Include Conda's
    # transaction history and ORT packages in both Windows and POSIX layouts.
    files = [python, prefix / "conda-meta" / "history",
             prefix / "Lib" / "site-packages" / "onnxruntime" / "__init__.py"]
    files.extend(sorted(prefix.glob("lib/python*/site-packages/onnxruntime/__init__.py")))
    signatures = []
    for file in files:
        try:
            st = file.stat()
            signatures.append(f"{file}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            signatures.append("missing")
    return ":".join(signatures)


def _load_cache() -> dict[str, Any]:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(data: dict[str, Any]) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, indent=1))
    except Exception:
        pass


def probe(interpreter: str, timeout: float = 120.0) -> Interpreter:
    """Ask one interpreter what it has. Never raises."""
    own = paths.is_own_interpreter(interpreter)
    info = Interpreter(path=interpreter, primary=own)
    info.bundled = own and paths.FROZEN
    if info.bundled:
        info.label = "bundled runtime"
    elif own:
        info.label = "main environment"
    else:
        parent = Path(interpreter).parent
        info.label = (parent.parent if parent.name.lower() in ("scripts", "bin") else parent).name

    if own:
        # No subprocess needed: we already are that interpreter.
        data = payload()
        info.available = True
        info.python = data.get("python")
        info.onnxruntime = data.get("onnxruntime")
        info.providers = data.get("providers") or []
        info.platform = data.get("platform")
        info.error = data.get("error")
        info.winml_devices = data.get("winml_devices", [])
        info.winml_notes = data.get("winml_notes", [])
        info.winml_version = data.get("winml_version", "")
        return info

    if not Path(interpreter).exists():
        info.error = "interpreter not found"
        return info

    try:
        cmd = python_command(interpreter, ["-c", PROBE_SOURCE])
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=str(paths.project_root()))
    except Exception as exc:
        info.error = f"could not run probe: {exc}"
        return info

    text = (proc.stdout or b"").decode("utf-8", errors="replace")
    marker = text.rfind(MARKER)
    if marker < 0:
        tail = ((proc.stderr or b"").decode("utf-8", errors="replace").strip().splitlines() or [""])[-1]
        info.error = f"probe produced no result{': ' + tail if tail else ''}"
        return info
    try:
        data = json.loads(text[marker + len(MARKER) :].strip())
    except Exception as exc:
        info.error = f"probe output was not JSON: {exc}"
        return info

    info.available = True
    info.python = data.get("python")
    info.onnxruntime = data.get("onnxruntime")
    info.providers = data.get("providers") or []
    info.platform = data.get("platform")
    info.error = data.get("error")
    info.winml_devices = data.get("winml_devices", [])
    info.winml_notes = data.get("winml_notes", [])
    info.winml_version = data.get("winml_version", "")
    return info


def probe_all(refresh: bool = False) -> list[Interpreter]:
    """Probe every candidate, using the on-disk cache where it is still valid."""
    cache = _load_cache()
    out: list[Interpreter] = []
    dirty = False
    for path in candidates():
        key = path.lower()
        sig = _signature(path)
        entry = cache.get(key)
        # The executable does not change when uv sync replaces runtime wheels.
        # Always read the current environment so an old CPU-only cache cannot hide GPUs.
        if not refresh and not paths.is_own_interpreter(path) and entry and entry.get("signature") == sig and entry.get("result") and not entry["result"].get("error") and not entry["result"].get("winml_version"):
            info = Interpreter(**entry["result"])
        else:
            info = probe(path)
            cache[key] = {"signature": _signature(path), "result": info.to_dict()}
            dirty = True
        out.append(info)
    if dirty:
        _save_cache(cache)
    return out


def provider_homes(refresh: bool = False) -> dict[str, str]:
    """Map each available Execution Provider to the interpreter that has it.

    The primary interpreter wins ties, so the tool keeps using whatever the user
    already set up whenever that is sufficient.
    """
    homes: dict[str, str] = {}
    for info in probe_all(refresh=refresh):
        if not info.available:
            continue
        for ep in info.providers:
            if ep in ("AzureExecutionProvider",):
                continue
            homes.setdefault(ep, info.path)
    return homes
