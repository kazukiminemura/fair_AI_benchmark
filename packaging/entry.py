"""Entry point for the frozen (PyInstaller) build.

Kept deliberately trivial: the real dispatch lives in ``fairbench.cli.main``,
including the hidden ``--worker`` and ``--probe-env`` re-entry points that a
packaged build needs because ``sys.executable`` is the bundle, not Python.
"""

import multiprocessing

from fairbench.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
