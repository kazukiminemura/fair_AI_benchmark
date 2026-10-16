"""Windows ML EP Catalog registration and explicit hardware selection."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import version

_registered: set[str] = set()


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


def register(*, download: bool = False) -> list[str]:
    import onnxruntime as ort
    from windowsml import EpCatalog, EpCertification, EpReadyState

    notes = []
    with EpCatalog() as catalog:
        for provider in catalog.find_all_providers():
            if provider.certification != EpCertification.Certified:
                continue
            try:
                if provider.ready_state == EpReadyState.NotPresent and not download:
                    notes.append(f"{provider.name}: not installed; run 'uv run fairbench setup-winml'.")
                    continue
                provider.ensure_ready()
                if provider.name not in _registered:
                    ort.register_execution_provider_library(provider.name, provider.library_path)
                    _registered.add(provider.name)
            except Exception as exc:
                notes.append(f"{provider.name}: {exc}")
    return notes


def probe(*, download: bool = False) -> dict:
    import onnxruntime as ort

    notes = register(download=download)
    devices = [device_info(device) for device in ort.get_ep_devices()
               if not device.ep_name.endswith(".AUTO")]
    return {"winml_devices": devices, "winml_notes": notes,
            "winml_version": version("windowsml")}


def main() -> int:
    data = probe(download="--setup" in sys.argv)
    print(f"Windows ML {data['winml_version']}")
    for device in data["winml_devices"]:
        print(f"  {device['kind'].upper():3s} {device['provider']}: {device['description']}")
    for note in data["winml_notes"]:
        print(f"  {note}")
    return 0 if any(device["kind"] in ("gpu", "npu") for device in data["winml_devices"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
