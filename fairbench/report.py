"""Report persistence and summarisation."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_name(prefix: str = "fairbench") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}.json"


def save(report: dict[str, Any], path: str | Path | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = RESULTS_DIR / default_name()
    path = Path(path)
    if path.is_dir():
        path = path / default_name()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def safe_name(name: str) -> str:
    cleaned = _SAFE.sub("-", name).strip("-")
    return cleaned or "result"


def list_results(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Summary of every result file, newest first, for the dashboard's sidebar."""
    directory = Path(directory or RESULTS_DIR)
    if not directory.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("schema") != "fairbench.result/1":
            continue
        out.append(
            {
                "file": path.name,
                "saved_at": path.stat().st_mtime,
                "title": headline(data),
                "duration_s": data.get("duration_s"),
                "grade": (data.get("grade") or {}).get("letter"),
                "score": (data.get("grade") or {}).get("score"),
                "device": (data.get("system") or {}).get("cpu", {}).get("name"),
                "host": (data.get("system") or {}).get("hostname"),
                "targets": len([r for r in data.get("results", []) if r.get("ok")]),
            }
        )
    return out


def headline(report: dict[str, Any]) -> str:
    sysinfo = report.get("system") or {}
    cpu = (sysinfo.get("cpu") or {}).get("name", "unknown CPU")
    host = sysinfo.get("hostname") or "unknown"
    return f"{host} - {cpu}"


def brief(report: dict[str, Any]) -> dict[str, Any]:
    """A compact view used for the dashboard header cards."""
    results = [r for r in report.get("results", []) if r.get("ok")]
    by_kind: dict[str, int] = {}
    for row in results:
        by_kind[row.get("kind", "?")] = by_kind.get(row.get("kind", "?"), 0) + 1
    scores = report.get("scores") or {}
    return {
        "headline": headline(report),
        "duration_s": report.get("duration_s"),
        "grade": report.get("grade"),
        "measured": len(results),
        "failed": len(report.get("results", [])) - len(results),
        "by_kind": by_kind,
        "ai_index": scores.get("ai_index", []),
        "config": report.get("config", {}),
    }
