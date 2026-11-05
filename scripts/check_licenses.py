"""Check license collection and optionally inspect built source/wheel archives."""

import argparse
import hashlib
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.collect_licenses import collect
from scripts.native_release import DML_SHA256, SOURCES, filter_binaries, system_runtime


def check_collection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "example.dist-info"
        source.mkdir()
        original = b"Copyright Example\r\nPermission notice\r\n"
        (source / "LICENSE").write_bytes(original)
        (root / "LICENSE.txt").write_text("Python license fixture", encoding="utf-8")
        dist = SimpleNamespace(metadata={"Name": "example", "License-Expression": "MIT"}, version="1.0",
                               files=["example.dist-info/LICENSE"], locate_file=lambda entry: root / entry)
        with patch("scripts.collect_licenses.metadata.distributions", return_value=[dist]), patch("sys.base_prefix", str(root)):
            output = root / "collected"
            records = collect(output)
            entry = records[0]["files"][0]
            assert (output / entry["path"]).read_bytes() == original
            assert entry["sha256"] == hashlib.sha256(original).hexdigest()
            assert json.loads((output / "inventory.json").read_text()) == records
            assert len(records) == 2 and records[1]["name"] == "Python"
            for folder, files in (("missing", []), ("unsafe", ["../LICENSE"])):
                dist.files = files
                try:
                    collect(root / folder)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Accepted {folder} license inventory")
            try:
                collect(output)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Existing inventory was silently reused")
    print("License collection: exact bytes, hashes, inventory, missing/unsafe paths and stale output checks passed.")


def check_archives(folder):
    wheels, sources = list(folder.glob("*.whl")), list(folder.glob("*.tar.gz"))
    assert wheels and sources, "Build both wheel and sdist before checking archives"
    for artifact in [*wheels, *sources]:
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
                metadata = archive.read(next(name for name in names if name.endswith(".dist-info/METADATA")))
                assert b"License-Expression: MIT" in metadata
                assert b"License-File: LICENSE" in metadata
        else:
            with tarfile.open(artifact) as archive:
                names = archive.getnames()
                for required in ("uv.lock", "CONTRIBUTING.md", "docs/LICENSING.md",
                                 "scripts/collect_licenses.py", "packaging/licenses/flatbuffers-25.12.19-LICENSE.txt"):
                    assert any(name.endswith("/" + required) for name in names), required
        for required in ("/LICENSE", "/THIRD_PARTY_NOTICES.md"):
            assert any(name.endswith(required) for name in names), required
        for name in names:
            assert "vitisai_int8_gemm.json" not in name, name
            assert not name.endswith(".local.json"), name
            assert not any(part.startswith((".venv", ".cache", ".git")) or part == "results"
                           for part in Path(name).parts), name
        print(f"Archive license and exclusion checks passed: {artifact.name}")


def check_frozen(artifact):
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        for required in ("LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSING.md",
                         "THIRD_PARTY_LICENSES/inventory.json", "NATIVE_INVENTORY.json",
                         "prepare-runtime.ps1", "LOCAL_RUNTIME_NAMES.json",
                         "THIRD_PARTY_LICENSES/native/directml-1.15.4/LICENSE.txt"):
            assert required in names, required
        assert not any("vitisai_int8_gemm.json" in name or name.endswith(".local.json") for name in names)
        records = json.loads(archive.read("THIRD_PARTY_LICENSES/inventory.json"))
        assert any(record["name"] == "Python" for record in records)
        assert any(record["name"] == "onnxruntime-directml" for record in records)
        for record in records:
            assert record["files"], record["name"]
            for entry in record["files"]:
                data = archive.read("THIRD_PARTY_LICENSES/" + entry["path"])
                assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
        natives = json.loads(archive.read("NATIVE_INVENTORY.json"))
        assert json.loads(archive.read("THIRD_PARTY_SOURCES/inventory.json")) == SOURCES
        for source in SOURCES:
            assert hashlib.sha256(archive.read("THIRD_PARTY_SOURCES/" + source["filename"])).hexdigest() == source["sha256"]
        native_names = {name for name in names if Path(name).suffix.lower() in {".dll", ".pyd", ".exe"}}
        assert native_names == {record["path"] for record in natives}
        for record in natives:
            assert not system_runtime(Path(record["path"]).name), record["path"]
            assert hashlib.sha256(archive.read(record["path"])).hexdigest() == record["sha256"]
            if Path(record["path"]).name.lower() == "directml.dll":
                assert record["sha256"] == DML_SHA256
        print(f"Frozen ZIP: {len(records)} components; notice hashes and exclusions passed.")


def check_native_policy():
    for name in ("vcruntime140.dll", "MSVCP140_1.dll", "api-ms-win-crt-private-l1-1-0.dll",
                 "ucrtbase.dll", "msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll"):
        assert system_runtime(name), name
    assert not system_runtime("DirectML.dll") and not system_runtime("libcrypto-3.dll")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with (patch("scripts.native_release.prepare_directml", return_value=root / "official.dll"),
              patch("sys.base_prefix", str(root)), patch("sys.prefix", str(root))):
            selected = filter_binaries([
                ("DirectML.dll", str(root / "wheel.dll"), "BINARY"),
                ("vcruntime140.dll", "C:/unrelated/PATH/vcruntime140.dll", "BINARY"),
            ])
            # Replacement is selected by actual source filename, as in the spec.
            assert len(selected) == 1
            selected = filter_binaries([("DirectML.dll", str(root / "DirectML.dll"), "BINARY")])
            assert selected[0][1] == str(root / "official.dll")
            try:
                filter_binaries([("unreviewed.dll", str(root.parent / "other/unreviewed.dll"), "BINARY")])
            except ValueError:
                pass
            else:
                raise AssertionError("Native DLL from an unrelated PATH location was accepted")
    print("Native policy: system DLL exclusion, pinned DirectML and foreign PATH rejection passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--frozen", type=Path, help="Windows frozen ZIP to check")
    args = parser.parse_args()
    check_collection()
    check_native_policy()
    if args.artifacts:
        check_archives(args.artifacts)
    if args.frozen:
        check_frozen(args.frozen)
