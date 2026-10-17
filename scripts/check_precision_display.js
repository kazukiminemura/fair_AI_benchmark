'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = vm.createContext({ document: {
  createElement: (tag) => ({ tag, children: [], textContent: '',
    setAttribute() {}, appendChild(child) { this.children.push(child); } }),
} });
const source = fs.readFileSync(require.resolve('../fairbench/web/app.js'), 'utf8');
vm.runInContext(source.split('/* ------------------------------------------------------------------ wire */')[0], context);
const rows = [
  { kind: 'cpu', precision: 'fp32', flops_per_s: 99e12 },
  { kind: 'cpu', precision: 'fp16', flops_per_s: 2e12 },
  { kind: 'cpu', precision: 'fp16', flops_per_s: 3e12 },
  { kind: 'cpu', precision: 'fp8', flops_per_s: 4e12 },
  { kind: 'cpu', precision: 'int8', flops_per_s: 7e12 },
  { kind: 'gpu', precision: 'FP16', flops_per_s: 5e12 },
  { kind: 'gpu', precision: 'INT8', flops_per_s: 8e12 },
  { kind: 'gpu', precision: 'fp8', flops_per_s: NaN },
  { kind: 'gpu', precision: 'fp8', flops_per_s: -1 },
  { kind: 'npu', precision: 'int8', flops_per_s: 6e12 },
  { kind: 'npu', precision: 'fp16', flops_per_s: 9e12 },
  { kind: 'npu', precision: 'int4', flops_per_s: 0 },
].map((row) => ({ ...row, ok: true }));
rows.push({ kind: 'cpu', precision: 'fp16', flops_per_s: 100e12, ok: false });
rows.push({ kind: 'gpu', precision: 'fp32', flops_per_s: 100e12, ok: true,
  session: { inference_precision_hint: 'fp16' } });
for (const [kind, labels, values] of [
  ['cpu', ['FP16 · TFLOPS', 'INT8 · TOPS'], ['3.000', '7.000']],
  ['gpu', ['FP16 · TFLOPS', 'INT8 · TOPS'], ['5.000', '8.000']],
  ['npu', ['FP16 · TFLOPS', 'INT8 · TOPS'], ['9.000', '6.000']],
]) {
  const card = { children: [], appendChild(child) { this.children.push(child); } };
  context.renderPrecisionRates(card, rows, kind);
  assert.deepEqual(card.children.filter((child) => child.className === 'device-role').map((child) => child.textContent), labels);
  assert.deepEqual(card.children.filter((child) => child.className.includes('precision-index')).map((child) => child.textContent), values);
  assert.equal(context.formatComputeRate({ kind, precision: 'int8', flops_per_s: 2e12 }), '2.000 TOPS');
  assert.equal(context.formatComputeRate({ kind, precision: 'fp16', flops_per_s: 2e12 }), '2.000 TFLOPS');
}
const empty = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(empty, [], 'cpu');
assert.equal(empty.children.filter((child) => child.textContent === '未計測').length, 2);
assert.equal(context.formatComputeRate({ kind: 'cpu', flops_per_s: NaN }), '—');
const failedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(failedCard, [{ kind: 'gpu', precision: 'int8', ok: false, error: 'Unsupported INT8' }], 'gpu');
assert.equal(failedCard.children[4].textContent, '実行不可');
assert.equal(failedCard.children[5].textContent, 'INT8: Unsupported INT8');
const reportDirectory = require('node:path').join(__dirname, '../results');
for (const filename of (fs.existsSync(reportDirectory) ? fs.readdirSync(reportDirectory) : []).filter((name) => name.endsWith('.json'))) {
  const report = JSON.parse(fs.readFileSync(require('node:path').join(reportDirectory, filename), 'utf8').replace(/^\uFEFF/, ''));
  for (const kind of ['cpu', 'gpu']) {
    const card = { children: [], appendChild(child) { this.children.push(child); } };
    context.renderPrecisionRates(card, report.results, kind);
    assert.equal(card.children[0].textContent, 'FP16 · TFLOPS', filename);
    assert.deepEqual(card.children.filter((child) => child.className === 'device-role').map((child) => child.textContent),
      ['FP16 · TFLOPS', 'INT8 · TOPS'], filename);
  }
}
console.log('Precision display passed: FP16 TFLOPS and INT8 TOPS on every device; FP32 excluded.');
assert.equal(context.formatExecutionPrecision({ compute_precisions: ['f32'] }), 'FP32（ランタイム報告）');
assert.equal(context.formatExecutionPrecision({ precision: 'fp16' }), '未確認');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: true, tensor_precisions: ['float16'],
} }), 'FP16（GPU割当グラフ型・内部演算/蓄積は未確認）');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: false, tensor_precisions: ['int8'],
} }), '未確認');
const probedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(probedCard, [{ kind: 'gpu', precision: 'fp16', ok: true, flops_per_s: 2e12,
  gpu_precision: { available: true, tensor_precisions: ['float16'] },
}], 'gpu');
assert(probedCard.children.some((child) => child.textContent.includes('演算入力 FP16（GPU割当グラフ型')));
assert(!probedCard.children.some((child) => child.textContent.includes('実行 FP16')));
assert.equal(context.formatExecutionPrecision({ compute_precisions: ['i8', 'f32'], compute_precision_complete: false }), 'FP32/INT8（一部未確認）');
const timedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(timedCard, [{ kind: 'gpu', precision: 'fp16', ok: true, flops_per_s: 2e12,
  timing_breakdown: { available: true, h2d_ms: 1, compute_ms: 2, d2h_ms: 3, repeats_measured: 3 } }], 'gpu');
assert(timedCard.children.some((child) => child.textContent.includes('GPU常駐推論（同期込）2.000 ms')));
const oldGrade = { results: [{ ok: true, repeats_ok: 1 }], grade: { checks: [
  { name: 'Repeatability', status: 'pass', detail: 'spread 0%' },
  { name: 'Timing stability', status: 'warn', detail: 'jitter 20%, but repeats agree within 0%, so throughput is stable' },
] } };
assert.equal(context.gradeChecks(oldGrade)[0].status, 'skip');
assert(!context.gradeChecks(oldGrade)[1].detail.includes('repeats agree'));
const peakCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(peakCard, [{ kind: 'gpu', precision: 'fp16', ok: true, flops_per_s: 30e12,
  peak_settings: { gemm_size: 2048, requests: 1, threads: 6 }, peak_candidate_count: 4,
  measurement_scope: 'device_resident' }], 'gpu');
assert(peakCard.children.some((child) => child.textContent.includes('行列 2048×2048・1並列・6スレッド枠')));
assert(peakCard.children.some((child) => child.textContent.includes('GPU常駐（入出力転送を除外）')));
