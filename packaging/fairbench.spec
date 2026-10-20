# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the fairbench distributable build.

Design choices, and why:

* **onedir, not onefile.** Onefile unpacks ~150 MB to a temp directory on every
  launch. Benchmark numbers are sensitive to what else the machine is doing, so
  paying a 2 s unpack (and a burst of disk I/O) immediately before a timed region
  is the last thing this tool should do. onedir also keeps the native ONNX
  Runtime DLLs in a stable location, which several Execution Providers require.
* **OpenVINO is excluded.** It is roughly a gigabyte of device plugins and it is
  optional. Users who need an Intel NPU
  run the packaged build with an OpenVINO interpreter next to it - the
  interpreter registry finds it automatically.
* **DirectML is bundled on Windows.** The isolated build environment contains
  one ORT wheel, with CPU and DirectX 12 GPU providers. Its native DLLs are
  collected below; CUDA and a separate GPU Python environment are unnecessary.
* **The source ships under ``src/``.** ``onnxruntime``, ``onnxruntime-directml``
  and ``onnxruntime-gpu`` all install into the same package, so at most one can
  be active per interpreter. Devices in another interpreter need the worker code,
  and ``src/`` is where they get it from. It is one level down so it can never
  shadow the modules the bootloader already provides.
"""

from pathlib import Path

import onnxruntime
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent

# The dashboard, extracted to sys._MEIPASS/web.
datas = [(str(ROOT / "fairbench" / "web"), "web")]
datas.append((str(ROOT / "fairbench" / "vitisai_int8_gemm.json"), "fairbench"))
datas.append((str(ROOT / "fairbench" / "shaders"), "fairbench/shaders"))

# ONNX Runtime's `transformers` helpers do bare top-level imports
# (`from fusion_utils import NumpyHelper`), so PyInstaller cannot resolve them
# and `import onnxruntime.quantization` fails at run time. Shipping the
# directory as real files plus the runtime hook below fixes it.
# See packaging/rthook_ort_transformers.py for the full explanation.
_ORT_DIR = Path(onnxruntime.__file__).resolve().parent
_ORT_TRANSFORMERS = _ORT_DIR / "transformers"
if _ORT_TRANSFORMERS.is_dir():
    datas.append((str(_ORT_TRANSFORMERS), "onnxruntime/transformers"))

binaries = collect_dynamic_libs("onnxruntime")

hiddenimports = [
    *collect_submodules("vulkan"),
    "onnxruntime.capi._pybind_state",
    "onnxruntime.quantization",
    # ORT 1.24's data-file transformer helpers import this by its bare name.
    "symbolic_shape_infer",
    *collect_submodules("onnxruntime.quantization"),
    "onnx",
    "onnx.helper",
    "onnx.numpy_helper",
    "numpy",
]

# Everything the packaged build must never drag in. OpenVINO is the big one;
# the rest are common transitive baggage that would triple the download.
excludes = [
    "openvino",
    "openvino_genai",
    "openvino_tokenizers",
    "torch",
    "torchvision",
    "transformers",
    "matplotlib",
    "pandas",
    "scipy",
    "PIL",
    "IPython",
    "tkinter",
    "test",
    "unittest",
    "pydoc_data",
    "PyInstaller",
    "ruff",
]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT), str(_ORT_DIR / "tools")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "rthook_ort_transformers.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fairbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Explicit so that `--worker` and `--probe-env` re-entry work regardless of
    # how Windows hands the process its command line.
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="fairbench",
)
