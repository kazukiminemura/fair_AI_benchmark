"""Collect installed distributions' unmodified license texts for a frozen build.

This is a notice collector, not a redistribution-permission checker. Run in
the isolated build interpreter; use a fresh output directory for every build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def license_file(path: Path) -> bool:
    return path.suffix.lower() not in {".py", ".pyi", ".pyc"} and any(
        marker in path.name.lower() for marker in ("license", "copying", "notice", "copyright", "authors")
    )


def collect(output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for dist in sorted(metadata.distributions(), key=lambda item: item.metadata["Name"].lower()):
        name, version = dist.metadata["Name"], dist.version
        if name.lower() == "fairbench":
            continue
        identifier = f"{name}-{version}"
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", identifier):
            raise ValueError(f"Unsafe distribution identifier: {identifier!r}")
        files = []
        for entry in dist.files or []:
            relative = Path(str(entry))
            if not license_file(relative):
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe license path in {identifier}: {entry}")
            files.append((Path(dist.locate_file(entry)), relative))
        if not files:
            # This exact wheel omits LICENSE; do not silently accept new versions.
            if (name.lower(), version) == ("flatbuffers", "25.12.19"):
                fallback = ROOT / "packaging" / "licenses"
                files = [(fallback / filename, Path(filename)) for filename in (
                    "flatbuffers-25.12.19-LICENSE.txt", "flatbuffers-25.12.19-NOTICE.txt",
                )]
            else:
                raise ValueError(f"No license text found for {identifier}; review the exact package before building")
        copied = []
        for source, relative in files:
            destination = output / identifier / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append({"path": destination.relative_to(output).as_posix(),
                           "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        declared = dist.metadata.get("License-Expression") or dist.metadata.get("License") or "See included license"
        records.append({"name": name, "version": version, "declared_license": declared.splitlines()[0],
                        "files": copied})
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise ValueError("Interpreter LICENSE.txt is missing; review Python's distribution before building")
    destination = output / "Python-LICENSE.txt"
    shutil.copyfile(python_license, destination)
    records.append({"name": "Python", "version": sys.version.split()[0], "declared_license": "See Python-LICENSE.txt",
                    "files": [{"path": destination.name,
                               "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}]})
    (output / "inventory.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "Unmodified license/notice texts from the isolated build environment.\n"
        "inventory.json records package versions and SHA256 hashes.\n"
        "Build tools are conservatively included even when not bundled.\n"
        "This inventory is not a native-binary redistribution approval.\n"
        "See THIRD_PARTY_NOTICES.md and LICENSING.md for release requirements.\n",
        encoding="utf-8",
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = collect(args.output)
    print(f"Collected license texts for {len(records)} components in {args.output}")


if __name__ == "__main__":
    main()
