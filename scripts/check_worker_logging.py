"""Run with python -m scripts.check_worker_logging."""

import json
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fairbench import worker_process


def main():
    with TemporaryDirectory() as directory:
        def fail(command, **kwargs):
            Path(command[-1]).write_text(json.dumps({"ok": False, "error": "could not load precision_gemm_int8"}))
            return CompletedProcess(command, 0, stdout=b"compiler output", stderr=b"NPU compiler failure\nfull reason")

        with patch.object(worker_process, "JOB_DIR", Path(directory)), patch.object(worker_process.subprocess, "run", side_effect=fail):
            result = worker_process.spawn_worker({"target": {"id": "amd-npu"}, "workload": "precision_gemm_int8"}, 10, "failure")
        assert "NPU compiler failure\nfull reason" in result["error"]
        log = Path(result["worker_log"]).read_text(encoding="utf-8")
        assert "compiler output" in log and "NPU compiler failure" in log
        assert "could not load precision_gemm_int8" in log
    print("Worker failure diagnostics preserve runtime output and expose the log path.")


if __name__ == "__main__":
    main()
