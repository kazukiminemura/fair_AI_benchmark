# Methodology

Why a fairbench number means something, and exactly which decisions were made to
get there. Everything below is enforced in code and recorded in every result
file; nothing depends on the operator remembering to be careful.

---

## 1. Identical work

Each workload is an ONNX graph built at run time by `fairbench/models.py` from a
fixed seed:

* weights come from `numpy.random.default_rng(seed)` (PCG64, stable across
  platforms and NumPy versions);
* no name-hashing is used anywhere — `hash()` is randomised per process and
  would silently produce different weights on different machines;
* input tensors are derived from the graph's own input declarations with a fixed
  seed, so a shape change cannot desynchronise the two sides.

The serialised graph is hashed with SHA-256 and the hash is written into the
result file next to every measurement. A reader can regenerate the graph and
compare. This is stronger than shipping a model file: a stale or swapped model
is impossible to miss.

The INT8 variants are produced by ONNX Runtime's dynamic quantiser, cached on
disk keyed by the *source* graph hash. Dynamic quantisation needs no calibration
set, which is what makes it byte-reproducible — a static pipeline would depend
on the calibration data.

## 2. Identical measurement

| decision | value | why |
| --- | --- | --- |
| warm-up iterations | 5 (2 in `--quick`) | let clocks and caches settle; excluded from the statistics |
| minimum iterations | 15 | a median over 5 samples is not a median |
| minimum time per cell | 4 s | covers timer resolution and short bursts |
| maximum time per cell | 45 s | bounds the suite |
| repeats | 3, with a cooldown | catches thermal drift that a single window cannot see |
| aggregation | **median** of steady-state latencies | immune to one unlucky iteration; never best-case |
| reported statistics | median, p90, p99, min, max, stdev, MAD, IQR, raw CV, trimmed CV, outlier count | the whole distribution, not a single flattering number |

Repeats are **interleaved across targets**, not run target-by-target, so a
chassis that warms up over the suite penalises every device rather than only the
one measured last.

## 3. Thread policy

CPU comparisons use **physical cores**, not SMT threads, and a **single
inference stream** (`inter_op_num_threads = 1`). Anything else rewards a bigger
core count instead of better AI hardware, and the SMT penalty is not uniform
between Intel and AMD.

This was measured rather than assumed on a 6-core / 12-thread i5-12400, running
`llm_decode` for 6 seconds at a time (~1300 iterations each):

| threads | trimmed CV |
| --- | --- |
| 1 | 5.0 % |
| 4 | 9.7 % |
| **6 (physical cores)** | **7.2 %** |
| 12 (SMT) | 21.7 % |

Thread **affinity is off by default** for the same reason: pinning ONNX Runtime's
intra-op threads to fixed logical CPUs measured *worse* (9.4 % vs 7.2 %), because
a pinned pool must also share those CPUs with the main thread and the OS. The
option remains available, and never changes a number silently.

## 4. Correctness gate

After compiling, each device runs the graph once and its first output is compared
against a reference: the **float32 graph executed on ONNX Runtime's CPU
Execution Provider**. That reference is deliberate — the CPU EP behaves
identically on Intel and AMD silicon, so it cannot favour a vendor. It is cached
by graph hash and input seed.

The comparison is the relative L2 difference of the output. The tolerance is
5 % for float paths and 5× that for INT8 weights, which is generous enough for
quantisation noise and tight enough to catch a device returning zeros, NaNs, or
garbage from an unsupported operator. Failing cells are flagged in the report and
cost points in the environment grade.

## 5. Process isolation

Every cell — every (device, workload, repeat) triple — runs in a **fresh child
process** launched with the interpreter that owns the device.

Three reasons, all of them observed in practice:

* **Crashes.** OpenVINO's GPU plugin hard-aborted the interpreter (0xC0000005)
  when it enumerated an adapter it could not drive. Isolated, that cost one cell.
* **Memory state.** A GPU run leaves allocations and cache pressure behind. A
  later cell on a different device should not inherit that.
* **Timeouts.** A hung driver can be killed. A hung driver inside the parent
  process cannot.

The parent also uses this to route work across interpreters, which is how a
DirectML or CUDA wheel can live in its own virtual environment without
displacing the `onnxruntime` the user already has.

## 6. Noise control

* **Python GC is disabled** for the timed region. On a large heap the cyclic
  collector can pause for tens of milliseconds, which would appear as a fake
  latency spike on whichever device happened to be running.
* **Background load is sampled** once, before any timed region, via the total
  CPU utilisation counter. It cannot influence the numbers it explains, and it
  turns "this machine is noisy" from an excuse into a measurement.
* **Power sampling is fail-soft by design.** The sampler refuses to start when no
  instrument exists, because the sampling loop itself costs host CPU, and paying
  that cost on a CPU benchmark to measure nothing would be a self-inflicted
  unfairness. Which instruments exist is probed once, in the parent, never inside
  a timed region.
* **In-loop jitter is reported but does not decide pass/fail** when repeats
  agree. A 250 µs GPU kernel is dominated by dispatch overhead, so per-iteration
  spread is large even when throughput over the window is rock solid. Repeats are
  the decisive signal.

## 7. Interpreting the numbers

**Absolute throughput** is the honest cross-machine figure: identical work, so
images/s and tokens/s compare directly between an Intel laptop and an AMD one.

**AI index** is the geometric mean, over workloads, of `throughput / this
machine's ONNX Runtime CPU throughput`. It answers a different question — *does
the accelerator pay for itself on this machine?* — and being a geometric mean,
one enormous GEMM cannot drown out everything else. The **profile radar chart**
shows the same ratios split by workload family, because accelerators are
not uniformly faster: the ranking flips between `llm_prefill` (compute bound) and
`llm_decode` (bandwidth bound) on most machines.

**Energy per inference** comes from the Windows power meter counter, the battery
discharge-rate counter, or the battery charge-percentage slope, in that order of
preference. The last is only trustworthy over several minutes, because 1 % of a
60 Wh battery is 36 s at 60 W; the report says which method was used. A laptop's
battery is the one power instrument that is identical for Intel and AMD.

**Host CPU seconds per inference** comes from `process_time()` and is free and
exact. On a GPU or NPU cell it shows how much work stayed on the host, which is
the difference between an accelerator that helps and one that merely exists.

## 8. What this benchmark does not do

* It does not measure accuracy or model quality. Every graph is a fixed,
  untrained skeleton whose job is to be representative *work*, and the
  correctness gate only checks that the device computes the same function.
* It does not measure sustained multi-hour performance. Repeats with a cooldown
  catch a chassis that is already warm; they do not model a 90-minute export.
* It does not measure concurrent multi-client throughput. One stream is
  measured, deliberately, so that core counts cannot masquerade as AI
  performance.
* It does not claim that a missing cell means a device is bad. It means the
  device was not measurable on that machine with the runtimes installed, and the
  report says so in words.

## Comparable rankings and accelerator precision

The relative AI index uses one CPU target as its baseline and the intersection
of successful workloads across all targets with baseline-matched results.
A device cannot improve its rank by failing its slowest workload. Excluded
workloads remain visible in the per-workload results. Energy indices likewise
use a common workload set among targets with energy measurements.

OpenVINO GPU and NPU both receive an FP16 inference precision hint, including
for FP32 input graphs. The input model precision and inference hint are recorded
separately in session metadata. A hint does not certify every operator's actual
precision; quantized models can use mixed precision. ORT/DirectML follows the
input graph and its provider optimizations, so comparing it against OpenVINO
also compares runtime policies. This is single-request, host-to-host latency
(including transfers), not maximum asynchronous device throughput. GPU and NPU
rankings may legitimately differ by model; no device is guaranteed to win.

Results produced before this change retain their original scores and precision
settings. Run a new suite to measure the updated GPU inference policy.

## Sustained throughput mode

`--mode throughput` changes OpenVINO's performance hint to THROUGHPUT and runs
independent InferRequests concurrently. ORT uses one independent session per
lane. DirectML lanes use separate processes to isolate native device state;
other ORT providers use separate sessions on worker threads.
The ORT thread budget per session is max(1, configured threads / lane count).
All lanes are warmed before timing and retain host input/output transfers.
The graph, batch size and seed are unchanged. Threads submit synchronous native
inference calls concurrently; this is not a device-only kernel benchmark.

Auto search tests 1, 2, 4 and 8 requests for the configured time per candidate.
`--requests N` fixes concurrency (1..32). Throughput is the number of completed
requests divided by wall time from simultaneous release through the final
completion, including pipeline drain. Request latency is measured under load;
it is not the inverse of throughput. Raw candidate counts, durations, rates and
selected request counts are saved. Each repeat selects its best candidate and
repeat aggregation still uses the median; exploratory best-of-search values
can be optimistic, so use fixed concurrency for confirmation. Candidate failures
are recorded rather than silently counted as successful inference.

This bounded search measures an observed maximum, not an exhaustive maximum.
Compare machines using the same mode, search range, time budget and model set.
Default latency mode and previously saved results retain their original meaning.
