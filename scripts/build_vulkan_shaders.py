"""Rebuild shipped SPIR-V with glslang 16.6.0 (not required at runtime)."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="glslangValidator")
    args = parser.parse_args()
    version = subprocess.check_output([args.compiler, "--version"], text=True)
    if "16.6.0" not in version:
        raise SystemExit("Use glslang 16.6.0 to reproduce the bundled shaders.")
    root = Path(__file__).resolve().parents[1] / "fairbench" / "shaders"
    records = {}
    for source in sorted(root.glob("*.comp")):
        binary = source.with_suffix(".spv")
        subprocess.run([args.compiler, "-V", "--target-env", "vulkan1.3", str(source), "-o", str(binary)], check=True)
        records[source.stem] = {"source_sha256": hashlib.sha256(source.read_text().encode()).hexdigest(),
                               "spirv_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    (root / "manifest.json").write_text(json.dumps({"compiler": "glslang 16.6.0", "target": "vulkan1.3",
                                                   "shaders": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
