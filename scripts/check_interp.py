"""Regression checks for interpreter cache invalidation without installing ORT."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fairbench import interp


def main() -> None:
    with TemporaryDirectory() as directory:
        prefix = Path(directory)
        python = prefix / "bin" / "python"
        python.parent.mkdir()
        python.touch()
        packages = [prefix / layout / "onnxruntime" / "__init__.py" for layout in (
            "Lib/site-packages", "lib/python3.12/site-packages", "lib/python3.13/site-packages")]
        for package in packages:
            package.parent.mkdir(parents=True)
            package.write_text("# fake ORT\n")
        stale = interp.Interpreter(str(python), available=True, providers=["CPUExecutionProvider"])
        current = interp.Interpreter(str(python), available=True, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])
        cache = {str(python).lower(): {"signature": interp._signature(str(python)), "result": stale.to_dict()}}
        with (patch.object(interp, "candidates", return_value=[str(python)]),
              patch.object(interp, "_load_cache", return_value=cache),
              patch.object(interp, "_save_cache"),
              patch.object(interp, "probe", return_value=current) as probe):
            assert interp.probe_all()[0].providers == stale.providers
            probe.assert_not_called()
            for package in packages:
                before = interp._signature(str(python))
                stat = package.stat()
                os.utime(package, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
                assert interp._signature(str(python)) != before, package
                probe.reset_mock()
                assert interp.probe_all()[0].providers == current.providers
                probe.assert_called_once_with(str(python))
            probe.reset_mock()
            interp.probe_all()
            probe.assert_not_called()

        before = interp._signature(str(python))
        extra = prefix / "lib/python3.14/site-packages/onnxruntime/__init__.py"
        extra.parent.mkdir(parents=True)
        extra.write_text("# fake ORT\n")
        assert interp._signature(str(python)) != before
        extra.unlink()
        assert interp._signature(str(python)) == before
        history = prefix / "conda-meta/history"
        history.parent.mkdir()
        history.write_text("installed ORT")
        assert interp._signature(str(python)) != before

    path = sys.executable
    stale = interp.Interpreter(path, primary=True, available=True, providers=["CPUExecutionProvider"])
    cache = {path.lower(): {"signature": interp._signature(path), "result": stale.to_dict()}}
    with (patch.object(interp, "candidates", return_value=[path]),
          patch.object(interp, "_load_cache", return_value=cache),
          patch.object(interp, "_save_cache"),
          patch.object(interp, "probe", return_value=stale) as probe):
        interp.probe_all()
        interp.probe_all()
        assert probe.call_count == 2
    print("Interpreter cache checks passed: Windows/POSIX package mtimes, all versions, cache hits, Conda history, main always probed.")


if __name__ == "__main__":
    main()
