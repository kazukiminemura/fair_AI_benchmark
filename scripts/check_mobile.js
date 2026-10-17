'use strict';
const assert = require('node:assert/strict');
const bench = require('../fairbench/web/mobile-bench.js');
const a = new Float32Array([1, 2, 3, 4]), b = new Float32Array([5, 6, 7, 8]);
assert.deepEqual([...bench.matmul(a, b, 2)], [19, 22, 43, 50]);
assert.deepEqual(bench.inputs(16), bench.inputs(16));
assert.notDeepEqual(bench.inputs(16, 1), bench.inputs(16, 2));
assert.throws(() => bench.validate(new Float32Array([NaN]), new Float32Array([1])));
assert.throws(() => bench.validate(new Float32Array([1, 2]), new Float32Array([1])));
assert.throws(() => bench.validate(new Float32Array([100]), new Float32Array([1])));
assert(bench.validate(new Float32Array([3, 0, 0, 1]), new Float32Array([3]), 4).valid);
const measured = bench.sample(10, 2000, 128);
assert.equal(measured.gflops, 2 * 128 ** 3 * 10 / 2 / 1e9);
assert.equal(measured.mean_ms, 200);
const repeats = [];
const cpu = bench.cpu(16, (repeat) => repeats.push(repeat), 2);
assert.deepEqual(repeats, [1, 2, 3]);
assert(cpu.result.gflops > 0);
assert(cpu.result.samples.every((sample) => sample.iterations > 0 && sample.wall_ms >= 2));
const expected = bench.matmul(bench.inputs(16).a, bench.inputs(16).b, 16);
assert(bench.validate(cpu.reference, expected).valid);
const memory = bench.memoryCopy();
assert(memory.available && memory.gb_per_s > 0);
assert.equal(memory.buffer_bytes, 8 * 1024 ** 2);
assert.equal(memory.samples_gb_per_s.length, 5);
global.document = { createElement: () => ({ getContext: () => null }) };
assert.throws(() => bench.createGpu(128), /WebGL2/);
global.document = { createElement: () => ({ getContext: () => ({ getExtension: () => null }) }) };
assert.throws(() => bench.createGpu(128), /浮動小数点/);
delete global.document;
console.log('Browser CPU kernel, deterministic inputs, correctness rejection, timing and copy accounting passed.');

const gpuData = bench.gpuInputs(256);
const first = bench.matmul(gpuData.a, gpuData.b, 256);
assert(bench.validate(first, bench.gpuReference(256, 1)).valid);
const second = bench.matmul(first, gpuData.b, 256);
assert(bench.validate(second, bench.gpuReference(256, 2)).valid);
assert.throws(() => bench.gpuInputs(257));
assert.equal(bench.VERSION, 'fairbench.browser/2');

(async () => {
  let deleted = 0, polls = 0;
  const gl = { SYNC_GPU_COMMANDS_COMPLETE: 1, ALREADY_SIGNALED: 2, CONDITION_SATISFIED: 3, TIMEOUT_EXPIRED: 4,
    fenceSync: () => ({}), flush() {}, isContextLost: () => false,
    clientWaitSync: () => (++polls < 3 ? 4 : 3), deleteSync: () => deleted++ };
  await bench.waitForGpu(gl);
  assert.equal(polls, 3); assert.equal(deleted, 1);
  await assert.rejects(bench.waitForGpu({ ...gl, clientWaitSync: () => 5 }), /完了待ちに失敗/);
  assert.equal(deleted, 2);
  const aborted = new AbortController(); aborted.abort();
  await assert.rejects(bench.waitForGpu(gl, aborted.signal), { name: 'AbortError' });
  assert.equal(deleted, 3);
  let completed = 0;
  const window = await bench.gpuWindow({ batch: async (count) => {
    await new Promise((resolve) => setTimeout(resolve, 2)); completed += count;
  } }, 512, 8, 5);
  assert.equal(window.iterations, completed);
  assert.equal(window.gflops, 2 * 512 ** 3 * completed / (window.wall_ms / 1000) / 1e9);
  await assert.rejects(bench.gpuWindow({ batch: async () => { throw new Error('GPU lost'); } }, 512, 8, 5), /GPU lost/);
  console.log('Independent dense-reference parity, GPU completion polling, failure/abort cleanup and completed-work accounting passed.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
