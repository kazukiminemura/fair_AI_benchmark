"""Local dashboard server.

A dependency-free HTTP server (stdlib only) that serves the single-page UI and
exposes a small JSON API so the dashboard can start runs, watch them live, and
browse every saved result. It binds to 127.0.0.1 only: benchmark results carry
machine fingerprints, and this is not meant to be exposed.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__, paths, system
from . import report as report_mod
from .config import RunConfig
from .devices import discover
from .runner import WORKLOAD_META, WORKLOAD_SETS, resolve_workloads, run_suite

WEB_ROOT = paths.resource_dir()


class RunState:
    """Single-slot run controller shared by all requests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.events: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.progress: dict[str, Any] = {"index": 0, "total": 0}
        self.result_file: str | None = None
        self.error: str | None = None
        self.report: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "active": self.active,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_s": (time.time() - self.started_at) if self.started_at and self.active else (
                    (self.finished_at - self.started_at) if self.started_at and self.finished_at else None
                ),
                "plan": self.plan,
                "progress": dict(self.progress),
                "events": self.events[-200:],
                "result_file": self.result_file,
                "error": self.error,
                "finished": bool(self.finished_at and not self.active),
            }

    def push(self, event: dict[str, Any]) -> None:
        with self.lock:
            if event.get("event") == "cell":
                self.progress = {"index": event.get("index", 0), "total": event.get("total", 0)}
            elif event.get("event") == "verify":
                self.progress.setdefault("verify", [])
            self.events.append({**event, "t": time.time()})
            if len(self.events) > 400:
                del self.events[:100]


STATE = RunState()
_CATALOG_CACHE: dict[str, Any] | None = None


def _catalog(force: bool = False) -> dict[str, Any]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None or force:
        fp = system.collect()
        disc = discover(fp)
        _CATALOG_CACHE = {
            "schema": "fairbench.catalog/1",
            "version": __version__,
            "system": fp.to_dict(),
            "discovery": disc.to_dict(),
            "workloads": [
                {"key": k, **v} for k, v in WORKLOAD_META.items()
            ],
            "workload_sets": WORKLOAD_SETS,
            "defaults": {
                "repeats": 3,
                "min_seconds": 4.0,
                "quick": {"repeats": 1, "min_seconds": 1.0},
            },
        }
    return _CATALOG_CACHE


# --------------------------------------------------------------------------
def _start_run(options: dict[str, Any]) -> dict[str, Any]:
    with STATE.lock:
        if STATE.active:
            return {"ok": False, "error": "a run is already in progress"}
        STATE.active = True
        STATE.started_at = time.time()
        STATE.finished_at = None
        STATE.error = None
        STATE.events = []
        STATE.report = None
        STATE.progress = {"index": 0, "total": 0}
        STATE.plan = options

    thread = threading.Thread(target=_run_thread, args=(options,), daemon=True)
    thread.start()
    return {"ok": True, "plan": options}


def _run_thread(options: dict[str, Any]) -> None:
    try:
        quick = bool(options.get("quick"))
        cfg = RunConfig(
            backend=str(options.get("backend", "all")),
            suite=str(options.get("suite", "custom" if options.get("workloads") else "standard")),
            mode=str(options.get("mode", "latency")),
            requests=int(options.get("requests", 0)),
            warmup=2 if quick else 5,
            min_iterations=5 if quick else 15,
            min_seconds=1.0 if quick else float(options.get("min_seconds", 4.0)),
            max_seconds=15.0 if quick else float(options.get("max_seconds", 45.0)),
            repeats=1 if quick else int(options.get("repeats", 3)),
            cooldown_seconds=0.5 if quick else 3.0,
            measure_power=bool(options.get("measure_power", True)),
            allow_battery=bool(options.get("allow_battery", False)),
            check_correctness=bool(options.get("check_correctness", True)),
            include_int8=bool(options.get("include_int8", True)),
            threads=options.get("threads"),
        )
        fp = system.collect()
        cfg.physical_cores = fp.cpu.physical_cores
        cfg.logical_cores = fp.cpu.logical_cores

        workloads = resolve_workloads(cfg, options.get("workloads") or None)
        targets = options.get("targets") or None
        report = run_suite(
            cfg,
            target_ids=targets,
            workloads=workloads,
            verify=bool(options.get("verify", True)),
            progress=STATE.push,
            fingerprint=fp,
        )
        path = report_mod.save(report)
        with STATE.lock:
            STATE.result_file = path.name
            STATE.report = {
                "headline": report_mod.headline(report),
                "grade": report.get("grade"),
                "results": [r for r in report.get("results", []) if r.get("ok")],
            }
    except Exception as exc:
        with STATE.lock:
            STATE.error = f"{type(exc).__name__}: {exc}"
            STATE.events.append({"event": "error", "error": str(exc), "traceback": traceback.format_exc()[-2000:]})
    finally:
        with STATE.lock:
            STATE.active = False
            STATE.finished_at = time.time()


# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = f"fairbench/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        if "/api/" in (self.path or ""):
            return
        super().log_message(fmt, *args)

    # -- helpers -------------------------------------------------------
    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(report_mod.json_safe(payload), default=str, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._json({"error": "not found", "path": str(path)}, 404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".html",):
            ctype = "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    # -- routing -------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self._json(
                {
                    "catalog": _catalog(),
                    "results": report_mod.list_results(),
                    "run": STATE.snapshot(),
                }
            )
            return
        if path == "/api/catalog":
            self._json(_catalog(force=True))
            return
        if path == "/api/results":
            self._json({"results": report_mod.list_results()})
            return
        if path.startswith("/api/result/"):
            name = report_mod.safe_name(path.split("/", 3)[3])
            self._json(report_mod.load(report_mod.RESULTS_DIR / name))
            return
        if path == "/api/run":
            self._json(STATE.snapshot())
            return
        if path == "/" or path == "":
            self._file(WEB_ROOT / "index.html")
            return
        rel = path.lstrip("/")
        candidate = (WEB_ROOT / rel).resolve()
        if WEB_ROOT.resolve() in candidate.parents and candidate.is_file():
            self._file(candidate)
            return
        self._json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            self._json(_start_run(self._body()))
            return
        if path == "/api/stop":
            with STATE.lock:
                STATE.active = False
                STATE.finished_at = time.time()
            self._json({"ok": True})
            return
        if path == "/api/delete":
            body = self._body()
            name = report_mod.safe_name(body.get("file", ""))
            target = report_mod.RESULTS_DIR / name
            if target.exists() and target.is_file():
                target.unlink()
                self._json({"ok": True})
            else:
                self._json({"ok": False, "error": "not found"}, 404)
            return
        self._json({"error": "not found"}, 404)


def serve(port: int = 8765, host: str = "127.0.0.1") -> None:
    if not WEB_ROOT.exists():
        raise SystemExit(f"dashboard assets are missing: {WEB_ROOT}")
    # Warm the catalogue (machine fingerprint + interpreter probing) before the
    # first browser request, so the page is not left waiting on a cold probe.
    threading.Thread(target=_catalog, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"  serving {WEB_ROOT}")
    print("  press Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        httpd.server_close()
