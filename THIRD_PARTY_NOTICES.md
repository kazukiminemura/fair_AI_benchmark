# Third-party software and assets

fairbench's own code, documentation, UI assets, and generated shaders are
covered by the root MIT LICENSE. Third-party software retains its own terms;
the project license does not relicense a dependency, driver, SDK, or model.

## Python dependencies

Versions are pinned in `uv.lock` and may vary with Python, OS, and architecture.
This is a license index, not a substitute for upstream license texts.
The frozen distribution ships the actual build environment's texts and version
inventory in `THIRD_PARTY_LICENSES/`. It conservatively includes build tools
as well as runtime packages; presence in the inventory does not imply that
the tool's executable is bundled.

| Component | License / upstream evidence |
| --- | --- |
| NumPy | BSD-3-Clause; wheels include additional bundled-code and library licenses. Preserve the entire wheel's license directory. [Upstream](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| ONNX | Apache-2.0, including NOTICE. [Upstream](https://github.com/onnx/onnx/blob/main/LICENSE) |
| ONNX Runtime / DirectML / GPU variants | MIT for ONNX Runtime; preserve LICENSE and ThirdPartyNotices.txt. Native provider DLLs can have additional terms. [Upstream](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| OpenVINO / openvino-telemetry | Apache-2.0 plus native runtime third-party notices. Installed by the Python package manager, excluded from the frozen ZIP. [Upstream](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE) |
| qrcode / colorama | BSD-3-Clause. [qrcode](https://github.com/lincolnloop/python-qrcode/blob/main/LICENSE), [colorama](https://github.com/tartley/colorama/blob/master/LICENSE.txt) |
| vulkan Python bindings | Apache-2.0, confirmed from the installed LICENSE even though metadata says UNKNOWN. [Upstream](https://github.com/realitix/vulkan/blob/master/LICENSE) |
| cffi / pycparser | MIT-0 / BSD-3-Clause for the currently locked versions; preserve each installed license. [cffi](https://github.com/python-cffi/cffi/blob/main/LICENSE), [pycparser](https://github.com/eliben/pycparser/blob/main/LICENSE) |
| flatbuffers | Apache-2.0. The locked wheel omits its license text; the build uses the upstream v25.12.19 copy in `packaging/licenses/`. [Exact version](https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE) |
| ml_dtypes | Apache-2.0; also preserve LICENSE.eigen and notices for Eigen code. [Upstream](https://github.com/jax-ml/ml_dtypes/blob/main/LICENSE) |
| protobuf | BSD-3-Clause and the installed bundled-code notices. [Upstream](https://github.com/protocolbuffers/protobuf/blob/main/LICENSE) |
| packaging | Apache-2.0 OR BSD-2-Clause. [Upstream](https://github.com/pypa/packaging/blob/main/LICENSE) |
| sympy / mpmath | BSD-3-Clause; retain AUTHORS and additional license files. [SymPy](https://github.com/sympy/sympy/blob/master/LICENSE), [mpmath](https://github.com/mpmath/mpmath/blob/master/LICENSE) |
| typing_extensions | PSF-2.0. [Upstream](https://github.com/python/typing_extensions/blob/main/LICENSE) |
| Python | PSF license and the additional component notices in the interpreter's LICENSE.txt. [Upstream](https://docs.python.org/3/license.html) |

## Build tools

PyInstaller is GPL-2.0-or-later with a bootloader exception; some files are
Apache-2.0. The exception permits distributing the combined executable under
the application's license, subject to dependency terms. Runtime hooks in
pyinstaller-hooks-contrib use Apache-2.0; other hooks use GPL-2.0-or-later.
This project does not modify or distribute PyInstaller as a standalone tool.
See [PyInstaller's license explanation](https://pyinstaller.org/en/stable/license.html)
and the license texts collected from the installed packages.

ruff and setuptools use MIT. PyInstaller's supporting tools altgraph, pefile,
macholib, and pywin32-ctypes use MIT or BSD licenses as specified by their
installed copies. A new platform or dependency version needs a fresh review.

## Vendor runtimes and native binaries

CUDA, TensorRT, TensorRT-RTX, Windows ML, Vitis AI / Ryzen AI, MIGraphX,
CoreML, GPU/NPU drivers, DirectML.dll, and Microsoft runtime DLLs are not
relicensed by fairbench's MIT license. Optional SDKs/providers are installed
separately on the user's machine and must retain their own agreements and
redistribution restrictions. Do not publish `.venv*`, `.cache*`, local EP
payloads, SDK installers, or compiled vendor engines.

The Windows x64 frozen build includes ONNX Runtime and the official
Microsoft.AI.DirectML 1.15.4 redistributable, under its own Microsoft DirectML
terms (not MIT). Those terms permit distribution in Windows ML applications.
The exact NuGet package and DLL hashes are pinned; LICENSE.txt,
LICENSE-CODE.txt, and ThirdPartyNotices.txt are shipped in
`THIRD_PARTY_LICENSES/native/directml-1.15.4/`.

Microsoft VC++ runtime DLLs (including NumPy's renamed MSVCP DLL), UCRT,
and API-set DLLs are excluded. Users install the official Microsoft x64
VC++ Redistributable themselves. The launcher makes NumPy's private-name
copy from that user's installed runtime locally, outside the distributed ZIP.
`NATIVE_INVENTORY.json` records the shipped native inputs and hashes.
See `docs/LICENSING.md` for the reviewed versions, sources, and release check.

For MPL-2.0-covered Eigen portions, the ZIP includes corresponding source
archives in `THIRD_PARTY_SOURCES/`, with original license notices. The
ml_dtypes 0.6.0 sdist includes its Eigen snapshot; a separate pinned archive
provides Eigen selected by ONNX Runtime build commit 2d924974ef. Recipients
may use and modify the covered source under MPL-2.0, without any restriction
from fairbench's MIT terms or DirectML's separate terms. No library binary
has been modified. See that directory's README.txt and inventory.json.

Python's additional native component license texts are supplemented in
`THIRD_PARTY_LICENSES/native/`: OpenSSL 3.0.16, Expat 2.7.1, zlib 1.3.1,
and the liblzma portion of xz 5.2.5. Python's own LICENSE.txt includes bzip2,
libffi, and additional notices. Python is embedded without source changes;
its standard modules are selected and packaged with fairbench.

## Project assets and model data

`fairbench/models.py` generates synthetic ONNX graphs, weights and inputs
locally with fixed seeds. No downloaded pretrained weights or external
datasets are tracked in this repository. Generated benchmark numbers are not
a grant of rights in an unrelated model or dataset a user might supply.

`fairbench/shaders/*.comp` are project MIT sources and the corresponding
`*.spv` files are their compiled forms. `manifest.json` records source/binary
hashes and compiler identity. glslang is a build tool, not a bundled runtime;
consult its [license](https://github.com/KhronosGroup/glslang/blob/main/LICENSE.txt)
if redistributing the compiler itself.

`docs/images/dashboard-*.jpg` are project dashboard screenshots. The owner
confirmed permission to publish project code and screenshots under MIT on
2026-11-05. UI SVGs and browser benchmark code are stored inline/in project
sources; no external JS, font, or image CDN is required by the dashboard.

The former `fairbench/vitisai_int8_gemm.json` was based on a vendor runtime's
output and has been removed because redistribution permission for that exact
material was not established. A user's optional local compiler profile is
outside the project license and must not be included in releases.

Vendor and product names identify compatible technologies. No endorsement,
affiliation, or trademark license is implied.
