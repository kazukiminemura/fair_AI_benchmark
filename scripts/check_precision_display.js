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
  { kind: 'gpu', precision: 'FP16', flops_per_s: 5e12 },
  { kind: 'gpu', precision: 'fp8', flops_per_s: NaN },
  { kind: 'gpu', precision: 'fp8', flops_per_s: -1 },
  { kind: 'npu', precision: 'int8', flops_per_s: 6e12 },
  { kind: 'npu', precision: 'int4', flops_per_s: 0 },
].map((row) => ({ ...row, ok: true }));
rows.push({ kind: 'cpu', precision: 'fp16', flops_per_s: 100e12, ok: false });
rows.push({ kind: 'gpu', precision: 'fp32', flops_per_s: 100e12, ok: true,
  session: { inference_precision_hint: 'fp16' } });
for (const [kind, labels, values] of [
  ['cpu', ['FP32 · TFLOPS', 'FP16 · TFLOPS', 'FP8 · TFLOPS'], ['99.000', '3.000', '4.000']],
  ['gpu', ['FP32 · TFLOPS', 'FP16 · TFLOPS', 'FP8 · TFLOPS'], ['100.000', '5.000', '未計測']],
  ['npu', ['INT8 · TOPS', 'INT4 · TOPS'], ['6.000', '0.000']],
]) {
  const card = { children: [], appendChild(child) { this.children.push(child); } };
  context.renderPrecisionRates(card, rows, kind);
  assert.deepEqual(card.children.filter((child) => child.className === 'device-role').map((child) => child.textContent), labels);
  assert.deepEqual(card.children.filter((child) => child.className.includes('precision-index')).map((child) => child.textContent), values);
  assert.equal(context.formatComputeRate({ kind, flops_per_s: 2e12 }), kind === 'npu' ? '2.000 TOPS' : '2.000 TFLOPS');
}
const empty = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(empty, [], 'cpu');
assert.equal(empty.children.filter((child) => child.textContent === '未計測').length, 2);
assert.equal(context.formatComputeRate({ kind: 'cpu', flops_per_s: NaN }), '—');
const failedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(failedCard, [{ kind: 'gpu', precision: 'fp8', ok: false, error: 'Unsupported FP8' }], 'gpu');
assert.equal(failedCard.children[4].textContent, '実行不可');
assert.equal(failedCard.children[5].textContent, 'FP8: Unsupported FP8');
const reportDirectory = require('node:path').join(__dirname, '../results');
for (const filename of (fs.existsSync(reportDirectory) ? fs.readdirSync(reportDirectory) : []).filter((name) => name.endsWith('.json'))) {
  const report = JSON.parse(fs.readFileSync(require('node:path').join(reportDirectory, filename), 'utf8').replace(/^\uFEFF/, ''));
  for (const kind of ['cpu', 'gpu']) {
    const recorded = (report.results || []).filter((row) => row.ok && row.kind === kind && row.precision === 'fp32'
      && Number.isFinite(row.flops_per_s) && row.flops_per_s >= 0);
    if (!recorded.length) continue;
    const card = { children: [], appendChild(child) { this.children.push(child); } };
    context.renderPrecisionRates(card, report.results, kind);
    assert.equal(card.children[0].textContent, 'FP32 · TFLOPS', filename);
    assert.equal(card.children[1].textContent, (Math.max(...recorded.map((row) => row.flops_per_s)) / 1e12).toFixed(3), filename);
  }
}
console.log('Precision display passed: separate rates, units, invalid/failed results, FP32 recovery in saved reports.');
