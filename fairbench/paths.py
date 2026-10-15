"""Filesystem layout, resolved for both a source checkout and a frozen build.

The packaged build (PyInstaller) changes two things that the rest of the code
must not guess about:

* **Resources** (the dashboard assets) live in the bootloader's extraction
  directory, not next to the ``.py`` files.
* **The working tree** is whatever directory the user launched from, not the
  bundle. ``cache/`` and ``results/`` stay relative to the working directory, so
  a user can benchmark from anywhere and keep their results together.

There is a third case worth being explicit about: the frozen build ships the
``fairbench`` source under ``src/`` on purpose. ONNX Runtime wheels all install
into the same ``onnxruntime`` package, so a GPU runtime often lives in its own
interpreter - and that interpreter needs the worker code to run at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

FROZEN: bool = bool(getattr(sys, "frozen", False))


def _bundle_dir() -> Path:
    """Where PyInstaller unpacked the app, or the package directory in source."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    """Read-only assets shipped with the app (the dashboard)."""
    return _bundle_dir() / "web"


def app_root() -> Path:
    """Where the application lives.

    In a source checkout this is the repository root; in a frozen build it is
    the directory holding the executable. Created virtual environments
    (``.venv-gpu``) and user-visible files belong here.
    """
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """Directory put on ``PYTHONPATH`` so any interpreter can import the worker.

    In a frozen build this is ``app_root()/src``, which deliberately is *not* the
    executable's own directory: keeping the shipped source one level down stops
    it from shadowing the modules the bootloader already provides.
    """
    if FROZEN:
        return app_root() / "src"
    return Path(__file__).resolve().parent.parent


def is_checkout() -> bool:
    """True when running from a source tree rather than an installed package."""
    root = app_root()
    return (root / "pyproject.toml").is_file() and (root / "fairbench").is_dir()


def work_root() -> Path:
    """Where user-created virtual environments belong.

    A source checkout or a packaged build has an obvious home for them; an
    installed package does not, because it lives inside site-packages. In that
    case the working directory is the only honest answer, and the interpreter
    registry searches there too.
    """
    if FROZEN or is_checkout():
        return app_root()
    return Path.cwd()


def is_own_interpreter(path: str) -> bool:
    """True when ``path`` is the interpreter currently running this code."""
    try:
        return Path(path).resolve() == Path(sys.executable).resolve()
    except Exception:
        return path == sys.executable


def worker_module_available(root: Path | None = None) -> bool:
    """Whether an *external* interpreter could run ``-m fairbench.worker``."""
    root = root or project_root()
    return (root / "fairbench" / "worker.py").is_file()
