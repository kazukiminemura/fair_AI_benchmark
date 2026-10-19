"""Windows ML EP Catalog registration and explicit hardware selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from importlib.metadata import version

_registered: set[str] = set()
_dll_directories: dict[str, object] = {}


def local_provider_library(library_path: str) -> Path:
    """Keep the complete provider payload together outside the Store package."""
    library = Path(library_path).resolve(strict=True)
    source = library.parent
    files = sorted(path for path in source.rglob('*') if path.is_file())
    identity = [(str(path.relative_to(source)), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    digest = hashlib.sha256((str(source) + json.dumps(identity)).encode()).hexdigest()[:20]
    root = Path('.cache/windowsml').resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / digest
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            staging = Path(temporary) / 'provider'
            shutil.copytree(source, staging)
            try:
                staging.rename(destination)
            except FileExistsError:
                pass  # Another worker finished copying the identical package.
    if hasattr(os, 'add_dll_directory'):
        for directory in {destination, *(path.parent for path in destination.rglob('*.dll'))}:
            key = str(directory)
            if key not in _dll_directories:
                _dll_directories[key] = os.add_dll_directory(key)
    return destination / library.name


def device_info(device) -> dict:
    hardware = device.device
    kind = hardware.type.name.lower()
    selector = {
        "provider": device.ep_name, "kind": kind,
        "vendor_id": hardware.vendor_id, "device_id": hardware.device_id,
        "metadata": dict(hardware.metadata), "ep_options": dict(device.ep_options),
    }
    key = hashlib.sha256(json.dumps(selector, sort_keys=True).encode()).hexdigest()[:16]
    return {**selector, "key": key, "vendor": hardware.vendor,
            "description": hardware.metadata.get("Description", device.ep_name),
            "ep_metadata": dict(device.ep_metadata)}


def register(*, download: bool = False, provider_name: str | None = None) -> list[str]:
    import onnxruntime as ort
    from windowsml import EpCatalog, EpCertification, EpReadyState

    notes = []
    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            if provider_name and provider.name != provider_name:
                continue
            if provider.certification != EpCertification.Certified:
                continue
            try:
                if provider.ready_state == EpReadyState.NotPresent and not download:
                    notes.append(f"{provider.name}: not installed; run 'uv run fairbench setup-winml'.")
                    continue
                provider.ensure_ready()
                if provider.name not in _registered:
                    library = (str(local_provider_library(provider.library_path))
                               if provider.name == 'VitisAIExecutionProvider' else provider.library_path)
                    ort.register_execution_provider_library(provider.name, library)
                    _registered.add(provider.name)
            except Exception as exc:
                notes.append(f"{provider.name}: {exc}")
    return notes


def probe(*, download: bool = False, provider_name: str | None = None) -> dict:
    import onnxruntime as ort

    notes = register(download=download, provider_name=provider_name)
    devices = [device_info(device) for device in ort.get_ep_devices()
               if not device.ep_name.endswith(".AUTO")]
    return {"winml_devices": devices, "winml_notes": notes,
            "winml_version": version("windowsml")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--provider")
    parser.add_argument("--require-npu", action="store_true")
    args = parser.parse_args()
    data = probe(download=args.setup, provider_name=args.provider)
    print(f"Windows ML {data['winml_version']}")
    for device in data["winml_devices"]:
        print(f"  {device['kind'].upper():3s} {device['provider']}: {device['description']}")
    for note in data["winml_notes"]:
        print(f"  {note}")
    matches = [device for device in data["winml_devices"]
               if (not args.provider or device["provider"] == args.provider)
               and (device["kind"] == "npu" if args.require_npu else device["kind"] in ("gpu", "npu"))]
    if not matches:
        print(f"Required device not found: {args.provider or 'Windows ML'} {'NPU' if args.require_npu else 'GPU/NPU'}. "
              "Check the AMD NPU driver and Windows Update; the EP catalog must support this hardware. "
              "https://ryzenai.docs.amd.com/projects/WinML/en/latest/installation.html")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
