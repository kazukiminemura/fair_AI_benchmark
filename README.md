# fair_AI_benchmark

A vendor-neutral AI benchmark for **laptop CPU / GPU / NPU**, built to compare
**Intel and AMD** machines without either side getting a free pass.

It measures the same ONNX graphs, with the same weights, the same inputs and the
same measurement policy, on every device it can find — and it writes down enough
about the run that a sceptical reader can tell whether the numbers are worth
anything.

```
   __       _         __                     _
  / _| __ _(_)_ __   / _| ___ _ __  ___  ___| |__
 | |_ / _` | | '__| | |_ / _ \ '_ \/ __|/ __| '_ \
 |  _| (_| | | |    |  _|  __/ | | \__ \ (__| | | |
 |_|  \__,_|_|_|    |_|  \___|_| |_|___/\___|_| |_|
```

## What it produces

* **Per-device throughput** for six workloads plus their INT8 variants:
  images/s, tokens/s, samples/s, and GFLOP/s derived from them.
* **AI index** — the geometric mean of "how much faster than this machine's own
  CPU". Answers *is the accelerator worth it here?*
* **Full latency statistics** — median (the headline), p90, p99, spread,
  coefficient of variation, per-iteration jitter.
* **Host CPU cost per inference** — how much CPU the offload still burns. An NPU
  that frees the CPU is worth more than one that does not.
* **Energy per inference** where a laptop exposes a power meter or battery
  counter.
* **An environment grade** that says whether the run is trustworthy at all.

## Quick start

Everything is driven by [uv](https://docs.astral.sh/uv/), so the toolchain is
pinned in `uv.lock` and reproducible.

```bash
uv sync --extra openvino     # core + the Intel CPU/GPU/NPU plugin (OpenVINO)
uv run fairbench devices     # what can this machine actually run?
uv run fairbench selftest    # prove this build works, end to end
uv run fairbench run         # measure everything (opens the dashboard)
uv run fairbench serve       # dashboard only
```

Prefer a globally installed command? Same thing, isolated:

```bash
uv tool install --from . --with openvino fairbench
fairbench run
```

`uv sync` alone installs the portable core (CPU path only); the `openvino` extra
adds Intel's CPU/iGPU/NPU plugin. GPU runtimes are handled separately — see
below. `pip install -r requirements.txt` still works if you would rather not
use uv.

The dashboard is a local single-page app at `http://127.0.0.1:8765`. Its overview
shows CPU/GPU/NPU speed ratios, measurement coverage and validity at a glance.
Links open separate views for measurements, device setup, validity checks,
history and methodology. Detail URLs keep the selected result after a reload.
It can start runs, watch them live, and browse saved results in `results/`.

## The device matrix

Everything is expressed as six cells, so a missing measurement is visible rather
than absent:

| | CPU | GPU | NPU |
| --- | --- | --- | --- |
| **Intel** | ONNX Runtime CPU EP, OpenVINO CPU plugin | OpenVINO GPU plugin, DirectML | OpenVINO NPU plugin (Core Ultra, AI Boost) |
| **AMD** | ONNX Runtime CPU EP, OpenVINO CPU plugin | DirectML (Windows), ROCm (Linux) | Vitis AI EP (Ryzen AI / XDNA) |

NVIDIA and Qualcomm are measured too, as extra columns — a laptop with a
discrete GPU is a legitimate comparison point, not a different benchmark.

### NVIDIA GPUs

Two paths, and the tool tells you which one is live:

* **DirectML** (`onnxruntime-directml`) — works on any DirectX 12 GPU with no
  CUDA toolkit, no cuDNN, no version matching. This is the path that makes an
  NVIDIA GPU directly comparable with an AMD one, because *both* go through the
  same runtime.
* **CUDA / TensorRT** (`onnxruntime-gpu` + CUDA 12 + cuDNN 9) — the fastest
  NVIDIA path, at the cost of a much more fragile install.

All ONNX Runtime wheels install into the *same* `onnxruntime` package, so adding
DirectML to an environment that already has the CPU build would displace it.
fairbench therefore keeps GPU runtimes in an isolated interpreter and launches
each cell in whichever interpreter owns the device:

```bash
fairbench setup-gpu            # creates .venv-gpu with onnxruntime-directml
fairbench setup-gpu --cuda     # creates .venv-cuda with onnxruntime-gpu + CUDA runtime
fairbench devices              # the new GPU target appears automatically
```

`setup-gpu` uses uv when it is on PATH and falls back to the stdlib `venv`.
Nothing in your main environment is touched. Extra interpreters can also be
pointed at explicitly with the `FAIRBENCH_PYTHON` environment variable
(`os.pathsep`-separated).

## Distribution

Two artefacts, two audiences:

```powershell
pwsh -File scripts\build.ps1
```

| artefact | for | notes |
| --- | --- | --- |
| `dist/fairbench-<ver>-win64.zip` | machines **without** Python | portable CPU + DirectML GPU build |
| `dist/fairbench-<ver>-py3-none-any.whl` | machines **with** Python | `uv tool install`, then add extras normally |

The **zip** is the right choice for handing the benchmark to a reviewer who
should not have to install Python, uv, CUDA or OpenVINO: unzip the entire folder
and double-click `start-fairbench.cmd` to open the dashboard. CPU and DirectML
GPU inference are bundled. GPU acceleration requires a DirectX 12 device and
its Windows graphics driver; it does not require a separate GPU environment.
`fairbench.exe selftest` checks the bundle from a terminal.
OpenVINO is excluded from the frozen build (it is about a
gigabyte of device plugins and is optional), so an Intel NPU number needs the
wheel instead — or an OpenVINO interpreter alongside the executable, which the
interpreter registry finds on its own.

The **wheel** is the right choice when the target has Python: it keeps the
optional pieces optional (`uv tool install --with openvino` for an Intel NPU,
`fairbench setup-gpu` for an NVIDIA or AMD GPU).

Things worth knowing about the frozen build:

* Each measurement cell runs in a **fresh child process**, and in a packaged app
  that child is `fairbench.exe --worker …` rather than `python -m fairbench.worker`.
  The hidden flag exists for exactly this.
* The `fairbench` source ships under `src/` for optional external runtimes.
  An external runtime needs its own working Python interpreter; CPU and
  DirectML GPU measurements use the bundled executable instead.
* The build uses `.venv-build` with only the DirectML ORT wheel on Windows,
  avoiding collisions with the CPU-only wheel in the development environment.
* `selftest` exists because a packaged build is hard to debug: it builds every
  workload, probes every device, spawns a real child process and checks the
  dashboard assets, then says precisely what is missing. It caught a real bug
  during development — ONNX Runtime's quantiser does bare top-level imports
  (`from fusion_utils import …`), which PyInstaller cannot resolve, so every
  INT8 workload was silently missing from the packaged build.

The portable build includes Python, NumPy, ONNX and ONNX Runtime internally.
These are still dependencies, but users do not install them separately. The
dashboard uses the standard browser and has no external JavaScript libraries
or online assets. Vendor NPU runtimes and drivers remain optional; their
availability cannot be assumed on a default Windows installation. Testing on
a development PC does not certify operation on a clean Windows installation.

## Workloads

Every graph is generated from a fixed seed inside fairbench — no downloads, no
hidden randomness, and the SHA-256 of each graph is written into every result
file. Two machines therefore run *byte-identical* work.

| workload | shape | size | measures |
| --- | --- | --- | --- |
| `dense_gemm` | MLP 256→3072→2048 | 6.4 GFLOP | raw matrix throughput; the NPU's home turf |
| `conv_vision` | CNN backbone, 1×3×224×224 | 1.5 GFLOP | classic vision inference |
| `depthwise_mobile` | MobileNet-style depthwise CNN | 0.11 GFLOP | what mobile NPUs are actually specified against |
| `transformer` | encoder ×2, S=192, H=768 | 5.7 GFLOP | NLP representation workloads |
| `llm_prefill` | decoder, 192 tokens | 10.4 GFLOP | time-to-first-token; compute bound |
| `llm_decode` | decoder, 1 token, 768-token KV cache | 0.06 GFLOP | tokens/s users actually feel; bandwidth bound |
| `*_int8` | INT8 weights (dynamic quantisation) | — | the format NPUs natively run |

The pair `llm_prefill` / `llm_decode` is deliberate: they stress opposite
resources, and the CPU/GPU/NPU ranking flips between them on most machines.

## Why this is fair

The full argument is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The short
version:

1. **Identical work.** One seeded ONNX graph per workload, hashed and printed.
2. **Identical inputs.** One seeded tensor set, shared by every device and repeat.
3. **Identical measurement.** Same warm-up, same time budget, median-first
   aggregation, never best-case.
4. **Physical cores, one stream.** Both vendors get physical cores, not SMT
   threads. (Measured: using all 12 logical threads on a 6-core i5-12400 raised
   the trimmed CV from 7.2 % to 21.7 %.)
5. **Correctness gate.** Every device's output is compared against a float32
   ONNX Runtime CPU reference. A silently wrong accelerator is flagged, not
   published.
6. **Isolated cells.** Each measurement runs in a fresh process, so a driver
   crash or leaked memory cannot distort the next one. (Measured: OpenVINO's GPU
   plugin aborted the interpreter outright on one machine — that cost one cell,
   not the suite.)
7. **Disclosed environment.** Threads, precision, providers, interpreter,
   drivers, power plan, power source and pre-run background load are all in the
   result file.
8. **Garbage collection parked.** Python's cyclic collector can pause for tens
   of milliseconds on a large heap; it is disabled for the timed region.

The **environment grade** turns those policies into a score, so a run taken on
battery, under a Power-saver plan, with 40 % background CPU load, is labelled
rather than quietly averaged in.

## Reading a result

```bash
python -m fairbench show results/fairbench-20260101-120000.json
```

```
  device                     workload               unit/s  median ms    GFLOPS    idx
  -------------------------- ------------------ ---------- ---------- --------- ------
  OpenVINO CPU               conv_vision            387.89      2.371     591.4  0.64x
  ORT DirectML GPU (.venv-gpu) conv_vision          2583.33      0.344    3938.7  3.34x
  ORT CPU                    conv_vision            267.33      4.038     407.6  1.00x
```

`unit/s` is the median-derived throughput across repeats; `idx` is the AI index
relative to the ONNX Runtime CPU baseline. The JSON that backs this contains
every repeat, every percentile, the provider list actually used, and the model
hashes.

## Commands

| command | purpose |
| --- | --- |
| `devices` | machine fingerprint, runtimes, the six-cell matrix, and why each cell is or is not measurable |
| `workloads` | list graphs, FLOPs, sizes |
| `verify` | execute a 1 ms probe on every device in an isolated process |
| `run` | run the suite. `--quick` for a fast low-confidence pass, `--full` for publication settings |
| `serve` | dashboard |
| `show <file>` | print a saved result |
| `setup-gpu` | create an isolated DirectML or CUDA interpreter |
| `info` | how this build is running, where it lives, which providers it sees |
| `selftest` | verify the build end to end and diagnose what is missing |

Useful `run` flags: `--targets`, `--workloads`, `--repeats`, `--min-seconds`,
`--threads`, `--no-int8`, `--no-check`, `--no-power`, `--allow-battery`.

## Project layout

```
fairbench/
  models.py        deterministic ONNX graph library (the "identical work" guarantee)
  interp.py        interpreter registry - finds GPUs living in other venvs
  paths.py         layout resolution for source checkouts and frozen builds
  devices.py       the six-cell matrix and concrete runnable targets
  backends/        ort_backend.py, openvino_backend.py + the Backend interface
  worker.py        one measurement cell, executed in its own process
  worker_process.py isolated worker launch, timeouts and crash reporting
  runner.py        scheduling, aggregation, scoring, grading
  power.py         power/energy and host-CPU-cost sampling
  stats.py         median-first aggregation, trimmed CV, outlier fences
  system.py        machine fingerprinting
  server.py        local dashboard server (stdlib only)
  web/             the dashboard (vanilla HTML/CSS/JS, no build step)
packaging/
  entry.py                    frozen-build entry point
  fairbench.spec              PyInstaller specification
  rthook_ort_transformers.py  runtime fix for ONNX Runtime's bare imports
docs/METHODOLOGY.md
scripts/build.ps1              wheel + frozen bundle
scripts/run-suite.ps1          pre-flight checks, then a suite
results/                       one self-describing JSON per run
```

## Verification

Run `python -m scripts.check_refactor` to check worker failure handling, scoring,
and a real isolated CPU inference. Run `python -m ruff check fairbench scripts/check_refactor.py`
for static checks.

## Adding a workload

Add a builder to `fairbench/models.py` returning a `BuiltModel`, register it in
`BASE_BUILDERS`, and add a blurb to `WORKLOAD_META` in `runner.py`. Keep shapes
static (NPUs require it), keep the opset at 17, and use deterministic seeds so
the graph hash stays reproducible.

## Requirements

* Python 3.10+ on Windows or Linux (not needed at all for the frozen build)
* [uv](https://docs.astral.sh/uv/) for development and building (`pip` works too)
* `numpy`, `onnx`, `onnxruntime` (required)
* `openvino` (required for any Intel NPU number; also gives an OpenVINO CPU and
  GPU path)
* optional: `onnxruntime-directml` / `onnxruntime-gpu` / `onnxruntime-qnn`, best
  installed in an isolated environment via `setup-gpu`
