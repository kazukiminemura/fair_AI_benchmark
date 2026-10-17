"""Report persistence and summarisation."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def json_safe(value: Any) -> Any:
    """JSON has no NaN/Infinity; represent unavailable numbers as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def format_compute_rate(row: dict[str, Any]) -> str:
    value = row.get("flops_per_s")
    if value is None or not math.isfinite(value):
        return "—"
    if str(row.get("precision", "")).lower().startswith("int"):
        return f"{value / 1e12:.3f} TOPS"
    return f"{value / 1e12:.3f} TFLOPS"


def default_name(prefix: str = "fairbench") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}.json"


def format_execution_precision(row: dict[str, Any]) -> str:
    gpu = row.get("gpu_precision", (row.get("session") or {}).get("gpu_precision", {}))
    gpu_assigned = gpu.get("available") and gpu.get("tensor_precisions")
    values = gpu["tensor_precisions"] if gpu_assigned else row.get("compute_precisions", (row.get("session") or {}).get("compute_precisions", []))
    names = {"f32": "FP32", "float": "FP32", "float32": "FP32", "f16": "FP16", "float16": "FP16",
             "i8": "INT8", "int8": "INT8", "u8": "UINT8", "uint8": "UINT8", "i32": "INT32", "int32": "INT32"}
    if not values:
        return "unverified"
    text = "/".join(sorted({names.get(value.lower(), value.upper()) for value in values}))
    if gpu_assigned:
        evidence = gpu.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if "profiled_fused_graph" in evidence:
            return text + " (GPU-assigned graph operands: types from pre-optimization ONNX, not optimized/runtime types; native arithmetic/accumulator unverified)"
        return text + " (GPU profile operand types; source " + ("profiled inputs" if "profiled_compute_inputs" in evidence else "not recorded") + "; native arithmetic/accumulator unverified)"
    return text if row.get("compute_precision_complete", True) else text + " (partially unverified)"


def save(report: dict[str, Any], path: str | Path | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = RESULTS_DIR / default_name()
    path = Path(path)
    if path.is_dir():
        path = path / default_name()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Older Windows builds wrote reports using the Japanese system codec.
        text = path.read_text(encoding="cp932")
    return json.loads(text, parse_constant=lambda _: None)


def safe_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Result name must be a string")
    cleaned = _SAFE.sub("-", name).strip("-")
    if cleaned and not cleaned.strip("."):
        raise ValueError("Result name must be a filename, not a directory")
    return cleaned or "result"


def list_results(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Summary of every result file, newest first, for the dashboard's sidebar."""
    directory = Path(directory or RESULTS_DIR)
    if not directory.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = load(path)
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
