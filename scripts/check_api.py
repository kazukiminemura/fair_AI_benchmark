"""Regression checks: python -m scripts.check_api."""

import json
import math
import subprocess
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from fairbench import report, server, worker, worker_process
from fairbench.stats import median, summarise, summarise_repeats


def check_delete_inputs() -> None:
    for value in (None, 123, 1.25, True, ["x"], {}, b"x", ".", ".."):
        try:
            report.safe_name(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid result name accepted: {value!r}")
    assert report.safe_name("normal.json") == "normal.json"

    with TemporaryDirectory() as directory, patch.object(report, "RESULTS_DIR", Path(directory)):
        root = Path(directory)
        saved = root / "keep.json"
        saved.write_text('{"ok": true}')
        fallback = root / "result"
        fallback.write_text('{"ok": true}')
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        def request(method, path, body=None, content_length=True):
            connection = HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
            try:
                if content_length:
                    connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
                else:
                    connection.putrequest(method, path)
                    connection.endheaders()
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        try:
            with patch.object(httpd, "handle_error") as handle_error:
                for payload in ({"file": None}, {"file": 123}, {"file": ["x"]}, {"file": {}},
                                {"file": True}, {"file": "."}, {"file": ".."},
                                [], {}, None, 123, "x", False):
                    status, body = request("POST", "/api/delete", json.dumps(payload))
                    assert status == 404 and body == {"ok": False, "error": "not found"}, payload
                for content, content_length in (("", True), (None, False), ("not JSON", True)):
                    status, body = request("POST", "/api/delete", content, content_length)
                    assert status == 404 and body == {"ok": False, "error": "not found"}
                assert saved.is_file() and fallback.is_file()
                for name in (".", "..", "missing.json"):
                    assert request("GET", f"/api/result/{name}") == (404, {"error": "not found"})
                assert request("GET", "/api/result/keep.json") == (200, {"ok": True})
                assert request("POST", "/api/delete", '{"file": "keep.json"}') == (200, {"ok": True})
                assert not saved.exists() and fallback.is_file()
                handle_error.assert_not_called()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join()
    print("API input checks passed: non-string names, non-object/empty bodies, missing Content-Length, GET errors and valid deletion.")


def check_numpy_json() -> None:
    def reject_constant(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")

    values = [np.longdouble("1.25"), np.longdouble("nan"), np.longdouble("inf"), np.longdouble("-inf")]
    for value, expected in zip(values, (1.25, None, None, None), strict=True):
        text = json.dumps({"x": value}, default=worker._json_default, allow_nan=False)
        assert json.loads(text, parse_constant=reject_constant) == {"x": expected}

    payload = {"ok": True, "longdouble": values,
               "array": np.array([values, values], dtype=np.longdouble),
               "zero_dim": np.array(np.longdouble("nan")),
               "zero_dim_finite": np.array(np.longdouble("1.25")),
               "empty": np.array([], dtype=np.longdouble),
               "nested": [{"array": np.array([[np.longdouble("nan")]])}]}
    expected = {"ok": True, "longdouble": [1.25, None, None, None],
                "array": [[1.25, None, None, None]] * 2,
                "zero_dim": None, "zero_dim_finite": 1.25, "empty": [],
                "nested": [{"array": [[None]]}]}
    for dtype in (np.float16, np.float32, np.float64):
        payload[dtype.__name__] = [dtype(1.25), dtype("nan"), dtype("inf"), dtype("-inf")]
        expected[dtype.__name__] = [1.25, None, None, None]
    for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
        payload[dtype.__name__] = dtype(np.iinfo(dtype).max)
        expected[dtype.__name__] = int(np.iinfo(dtype).max)
    payload["integer_array"] = np.array([[1, 2], [3, 4]], dtype=np.int64)
    payload["integer_zero_dim"] = np.array(7, dtype=np.int64)
    expected["integer_array"] = [[1, 2], [3, 4]]
    expected["integer_zero_dim"] = 7

    with TemporaryDirectory() as directory, patch.object(worker_process, "JOB_DIR", Path(directory)):
        def launch(cmd, **kwargs):
            with patch.object(worker, "run_cell", return_value=payload):
                assert worker.main(["worker", cmd[-2], cmd[-1]]) == 0
            parsed = json.loads(Path(cmd[-1]).read_text(), parse_constant=reject_constant)
            assert parsed == expected
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with patch.object(worker_process.subprocess, "run", side_effect=launch):
            result = worker_process.spawn_worker({"target": {"id": "test"}, "workload": "smoke"}, 1, "longdouble")
        assert result == {**expected, "worker_returncode": 0}
    print("NumPy JSON checks passed: longdouble scalars/arrays, strict null conversion, finite numbers, integer precision and parent readback.")


def check_nullable_median() -> None:
    for values in ([], [None], [None, None], [None, float("nan"), float("inf"), -float("inf")]):
        assert math.isnan(median(values)), values
    for values, expected in (([1.0, None, 2.0], 1.5), ([None, 3.0], 3.0),
                             ([3.0, None, 1.0, 2.0, float("nan"), float("inf")], 2.0),
                             ([0.0, None, -2.0], -1.0)):
        assert median(values) == expected, values
    assert summarise([1.0, None, 2.0]).to_dict() == summarise([1.0, 2.0]).to_dict()
    empty = summarise([None])
    assert empty.count == 0 and empty.outliers == 0 and math.isnan(empty.median_ms)
    repeats = summarise_repeats([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert repeats.repeats == 3 and repeats.median_units_per_s == 2.0 and repeats.spread_pct == 100.0
    print("Median checks passed: None/nonfinite exclusion, empty NaN, mixed samples and unchanged summary/repeat behavior.")


def main() -> None:
    check_delete_inputs()
    check_numpy_json()
    check_nullable_median()


if __name__ == "__main__":
    main()
