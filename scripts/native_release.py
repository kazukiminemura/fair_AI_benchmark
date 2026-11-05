"""Pinned Windows native payloads and a fail-closed build provenance check."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DML_VERSION = "1.15.4"
DML_URL = f"https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/{DML_VERSION}/microsoft.ai.directml.{DML_VERSION}.nupkg"
DML_PACKAGE_SHA256 = "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9"
DML_SHA256 = "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1"
NATIVE_SUFFIXES = {".dll", ".pyd", ".exe"}
REVIEWED_VERSIONS = {
    "cffi": "2.1.1", "ml_dtypes": "0.6.0", "numpy": "2.5.4", "onnx": "1.23.2",
    "onnxruntime-directml": "1.24.4", "protobuf": "7.36.2", "Python": "3.12.10",
    "Microsoft.AI.DirectML": DML_VERSION, "fairbench + PyInstaller bootloader": "6.22.3",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def system_runtime(name):
    name = Path(name).name.lower()
    return name.startswith(("api-ms-", "ext-ms-")) or name == "ucrtbase.dll" or bool(
        re.fullmatch(r"(?:msvcp|vcruntime|concrt|vcomp)\d+[^/\\]*\.dll", name)
    )


def prepare_directml():
    cache = ROOT / ".cache" / f"directml-{DML_VERSION}"
    cache.mkdir(parents=True, exist_ok=True)
    package = cache / "package.nupkg"
    if not package.exists() or digest(package) != DML_PACKAGE_SHA256:
        with urllib.request.urlopen(DML_URL, timeout=30) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != DML_PACKAGE_SHA256:
            raise ValueError("DirectML NuGet package hash changed; review before building")
        package.write_bytes(payload)
    with zipfile.ZipFile(package) as archive:
        for name in ("LICENSE.txt", "LICENSE-CODE.txt", "ThirdPartyNotices.txt"):
            expected = ROOT / "packaging" / "licenses" / f"directml-{DML_VERSION}" / name
            if archive.read(name) != expected.read_bytes():
                raise ValueError(f"DirectML license changed: {name}")
        binary = cache / "DirectML.dll"
        binary.write_bytes(archive.read("bin/x64-win/DirectML.dll"))
    if digest(binary) != DML_SHA256:
        raise ValueError("Unreviewed DirectML binary")
    return binary


def filter_binaries(binaries):
    """Called by the spec before COLLECT; never redistribute PATH-found runtimes."""
    directml = prepare_directml()
    allowed = (Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve())
    selected = []
    for destination, source, kind in binaries:
        if system_runtime(destination):
            continue
        resolved = Path(source).resolve()
        if resolved.name.lower() == "directml.dll":
            source = str(directml)
        elif not any(resolved.is_relative_to(root) for root in allowed):
            raise ValueError(f"Native dependency came from an unapproved location: {destination}: {source}")
        selected.append((destination, source, kind))
    return selected


def audit(bundle):
    """Match every shipped native file to the pinned runtime or installed input."""
    origins = {}
    for dist in metadata.distributions():
        for entry in dist.files or []:
            if Path(str(entry)).suffix.lower() in NATIVE_SUFFIXES:
                path = Path(dist.locate_file(entry))
                if path.is_file() and not system_runtime(path.name):
                    origins[digest(path)] = {"component": dist.metadata["Name"], "version": dist.version,
                                             "source": str(entry).replace("\\", "/")}
    base = Path(sys.base_prefix)
    for directory in (base, base / "DLLs"):
        for path in directory.glob("*"):
            if path.is_file() and path.suffix.lower() in NATIVE_SUFFIXES and not system_runtime(path.name):
                origins[digest(path)] = {"component": "Python", "version": sys.version.split()[0],
                                         "source": path.relative_to(base).as_posix()}
    origins[DML_SHA256] = {"component": "Microsoft.AI.DirectML", "version": DML_VERSION,
                           "source": "NuGet:bin/x64-win/DirectML.dll"}
    records = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in NATIVE_SUFFIXES:
            continue
        if system_runtime(path.name):
            raise ValueError(f"Microsoft system/VC runtime must not be redistributed: {path}")
        sha = digest(path)
        if path == bundle / "fairbench.exe":
            origin = {"component": "fairbench + PyInstaller bootloader", "version": metadata.version("pyinstaller"),
                      "source": "packaging/fairbench.spec; PyInstaller bootloader exception"}
        else:
            origin = origins.get(sha)
            if not origin:
                raise ValueError(f"Unreviewed native binary: {path}")
        if REVIEWED_VERSIONS.get(origin["component"]) != origin["version"]:
            raise ValueError(f"Native component version needs a license review: {origin}")
        records.append({"path": path.relative_to(bundle).as_posix(), "sha256": sha, **origin})
    (bundle / "NATIVE_INVENTORY.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Native provenance verified: {len(records)} files; no bundled Microsoft system/VC runtime")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    audit(args.bundle)
