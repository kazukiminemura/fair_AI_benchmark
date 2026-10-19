'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = vm.createContext({ document: {
  createElement: (tag) => ({ tag, children: [], textContent: '',
    attributes: {}, handlers: {}, setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, handler) { this.handlers[name] = handler; },
    replaceChildren() { this.children = []; },
    appendChild(child) { this.children.push(child); } }),
} });
const source = fs.readFileSync(require.resolve('../fairbench/web/app.js'), 'utf8');
vm.runInContext(source.split('/* ------------------------------------------------------------------ wire */')[0], context);
const chartHosts = Object.fromEntries(['throughputChart', 'chartLegend']
  .map((id) => [id, { children: [], appendChild(child) { this.children.push(child); } }]));
context.document.getElementById = (id) => chartHosts[id];
context.document.createElementNS = (_, tag) => ({ ...context.document.createElement(tag), style: {} });
context.requestAnimationFrame = (handler) => handler();
context.renderThroughput({ results: [
  { ok: true, workload: 'fp16', target_id: 'gpu1', units_per_s: 56.53 },
  { ok: true, workload: 'fp16', target_id: 'gpu2', units_per_s: 30.15 },
  { ok: true, workload: 'int8', target_id: 'gpu1', units_per_s: 31.76 },
  { ok: true, workload: 'equal', target_id: 'gpu1', units_per_s: 30.15 },
  { ok: true, workload: 'tokens', target_id: 'gpu1', units_per_s: 10000, unit: 'token' },
  { ok: true, workload: 'invalid', target_id: 'gpu1', units_per_s: Infinity },
] });
const scaledBars = chartHosts.throughputChart.children[0].children.filter((node) => node.tag === 'rect');
const barWidths = scaledBars.map((node) => Number(node.attributes.width));
assert.equal(barWidths.length, 5);
assert(Math.abs(barWidths[2] / barWidths[0] - 31.76 / 56.53) < 1e-12);
assert.equal(barWidths[1], barWidths[3]);
assert.equal(barWidths[4], barWidths[0]); // Different units have independent scales.
assert(scaledBars[2].children.some((node) => node.tag === 'title' && node.textContent.includes('31.76 sample/s')));
console.log('Throughput scale passed: shared sample/s scale, equal values, separate units, finite values and hover labels.');
assert.equal(context.formatExecutionPrecision({ phase: 'precision', precision: 'int8' }),
  'INT8指定 · デバイス実行済み');
const unverifiedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(unverifiedCard, [{ kind: 'gpu', precision: 'int8', ok: false, phase: 'precision' }], 'gpu');
assert(unverifiedCard.children.some((child) => child.textContent === '再計測が必要'));
const referenceRow = { kind: 'gpu', precision: 'int8', ok: true, unit: 'sample', units_per_s: 20,
  flops: 1e11, flops_per_s: null, latency_ms: 50, measurement_validity: 'model_only', target_label: 'Windows ML TensorRT-RTX GPU',
  timing_breakdown: { available: true, compute_ms: 25, e2e_ms: 50 } };
const resultTable = context.document.createElement('table');
context.document.getElementById = () => resultTable;
context.renderTable({ results: [{ ...referenceRow, workload: 'precision_gemm_int8' },
  { ...referenceRow, workload: 'failed', ok: false }] });
const headers = resultTable.children[0].children[0].children.map((cell) => cell.textContent);
const tableRows = resultTable.children[1].children;
assert.equal(tableRows[1].children.length, headers.length);
assert.equal(tableRows[1].children[headers.indexOf('推論性能')].textContent, '4.000 TOPS');
assert.equal(tableRows[1].children[headers.indexOf('E2E性能')].textContent, '2.000 TOPS');
assert.equal(tableRows[0].children[5].colSpan, headers.length - 5);
context.document.getElementById = (id) => chartHosts[id];
const referenceCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(referenceCard, [referenceRow], 'gpu');
const referenceValue = referenceCard.children.find((child) => child.textContent === '4.000');
assert(referenceValue.children.some((child) => child.textContent === 'TOPS'));
assert(referenceValue.children.some((child) => child.textContent === '（E2E 2.000 TOPS）'));
assert.equal(context.formatComputeRate(referenceRow), '2.000 TOPS');
assert.equal(context.formatInferenceRate({ ...referenceRow,
  timing_breakdown: { available: true, compute_ms: 1 } }), '100.000 TOPS');
assert.equal(context.formatComputeRate({ ...referenceRow, model_operations_per_s: 3e12 }), '3.000 TOPS');
assert.equal(context.formatComputeRate({ precision: 'int8', flops: 137438953472, units_per_s: 61.87970122051206,
  session: { measurement_validity: 'model_only' } }), '8.505 TOPS');
assert.equal(context.inferenceRateValue(referenceRow), 4e12);
const referenceDetails = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(referenceDetails, [referenceRow], 'gpu', { details: true });
assert(referenceDetails.children.some((child) => child.textContent.includes('INT8指定')));
assert(!referenceDetails.children.some((child) => /内部INT8|モデル演算数×実測処理回数|遅延 50/.test(child.textContent)));
assert(!context.formatComputeRate({ precision: 'int8', flops_per_s: 1e12,
  session: { measurement_validity: 'model_only' } }).includes('TOPS'));
const gpuPaths = ['ort:DmlExecutionProvider', 'openvino:GPU',
  'winml:DmlExecutionProvider:gpu:device1', 'winml:OpenVINOExecutionProvider:gpu:device2'];
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: true, tensor_precisions: ['int8'], evidence: ['tensorrt_engine_inspector'],
  arithmetic_precision: 'int8', runtime_execution_verified: true,
} }), '入力型 INT8');
assert(!context.formatExecutionPrecision({ gpu_precision: {
  available: true, tensor_precisions: ['int8'], evidence: ['tensorrt_engine_inspector'],
  arithmetic_precision: 'unverified', runtime_execution_verified: false,
} }).includes('実行を確認'));
assert.equal(new Set(gpuPaths.map((target_id) => context.deviceColor({ kind: 'gpu', target_id }))).size, 4);
const cpuPaths = ['openvino:CPU', 'ort:CPUExecutionProvider', 'winml:CPUExecutionProvider:cpu:device1'];
assert.equal(new Set(cpuPaths.map((target_id) => context.deviceColor({ kind: 'cpu', target_id }))).size, 3);
for (const id of cpuPaths) {
  assert.equal(context.deviceColor({ kind: 'cpu', id }), context.deviceColor({ kind: 'cpu', target_id: id }));
  assert.equal(context.deviceColor({ kind: 'cpu', backend: id.split(':')[0] }), context.deviceColor({ kind: 'cpu', target_id: id }));
}
const inferenceWinner = { kind: 'gpu', precision: 'int8', ok: true, flops: 1e9,
  flops_per_s: 1e12, timing_breakdown: { available: true, compute_ms: 0.1 } };
const transferWinner = { ...inferenceWinner, flops_per_s: 2e12,
  timing_breakdown: { available: true, compute_ms: 1 } };
assert(context.compareInferenceRates(inferenceWinner, transferWinner) < 0);
const winnerCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(winnerCard, [transferWinner, inferenceWinner], 'gpu');
assert.equal(winnerCard.children[3].textContent, '10.000');
for (const id of gpuPaths) {
  assert.equal(context.deviceColor({ kind: 'gpu', id }), context.deviceColor({ kind: 'gpu', target_id: id }));
}
const diagram = { children: [], appendChild(child) { this.children.push(child); }, replaceChildren() { this.children = []; } };
context.renderMeasurementDiagram(diagram, [{ kind: 'gpu', precision: 'fp16', ok: true,
  flops_per_s: 2e12, gemm_shape: [2048, 2048, 2048],
  timing_breakdown: { available: true, h2d_ms: 2.001, compute_ms: 0.754, d2h_ms: 3.449 } }]);
assert(diagram.children[0].textContent.includes('2048×2048×2048'));
const steps = diagram.children.find((child) => child.tag === 'ol').children;
assert.deepEqual(steps.map((step) => step.children[0].textContent), ['入力転送', 'GPU常駐推論', '出力転送']);
assert.deepEqual(steps.map((step) => step.children[1].textContent), ['2.001 ms', '0.754 ms', '3.449 ms']);
assert(steps[1].children.some((child) => child.textContent.includes('主値')));
context.renderMeasurementDiagram(diagram, []);
assert(diagram.children.find((child) => child.tag === 'ol').children.every((step) => step.children[1].textContent === '未取得'));
for (const kind of ['cpu', 'gpu', 'npu']) {
  for (const precision of ['fp16', 'int8']) {
    context.renderMeasurementDiagram(diagram, [{ kind, precision, ok: true, flops_per_s: 1e12,
      timing_breakdown: { available: true, h2d_ms: 1, compute_ms: 2, d2h_ms: 3 } },
    { kind: 'other', precision, ok: true, flops_per_s: 99e12,
      timing_breakdown: { available: true, compute_ms: 99 } }], kind, precision);
    assert(diagram.children[0].textContent.includes(`${kind.toUpperCase()}・${precision.toUpperCase()}`));
    const flow = diagram.children.find((child) => child.tag === 'ol');
    assert.equal(flow.children[1].children[1].textContent, '2.000 ms');
    assert(flow.children[1].children[0].textContent.startsWith(kind.toUpperCase()));
    context.renderMeasurementDiagram(diagram, [{ kind, precision, ok: true, flops_per_s: 1e12 }], kind, precision);
    assert(diagram.children.some((child) => child.textContent.includes('全体の時間からは推定しません')));
    context.renderMeasurementDiagram(diagram, [{ kind, precision, ok: true, flops_per_s: 1e12,
      timing_breakdown: { available: true, transfer_scope: 'host_buffers', internal_transfers_in_compute: kind === 'npu',
        h2d_ms: 1, compute_ms: 2, d2h_ms: 3 } }], kind, precision);
    const hostFlow = diagram.children.find((child) => child.tag === 'ol');
    assert.equal(hostFlow.children[0].children[0].textContent, '入力準備・コピー');
    assert.equal(hostFlow.children[2].children[0].textContent, '出力取得・コピー');
    assert(diagram.children.some((child) => child.textContent.includes('物理転送を個別に測った値ではありません')));
  }
}
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
  ['cpu', ['FP16', 'INT8'], ['3.000', '7.000']],
  ['gpu', ['FP16', 'INT8'], ['5.000', '8.000']],
  ['npu', ['FP16', 'INT8'], ['9.000', '6.000']],
]) {
  const card = { children: [], appendChild(child) { this.children.push(child); } };
  context.renderPrecisionRates(card, rows, kind);
  assert.deepEqual(card.children.filter((child) => child.className === 'device-role').map((child) => child.textContent), labels);
  assert.deepEqual(card.children.filter((child) => child.className.includes('precision-index')).map((child) => child.textContent), ['分離不可', '分離不可']);
  assert.deepEqual(card.children.filter((child) => child.className.includes('precision-index')).map((child) => child.children.at(-1).textContent),
    values.map((value, index) => `（E2E ${value} ${index ? 'TOPS' : 'TFLOPS'}）`));
  assert.equal(card.children.length, labels.length * 2);
  assert(!card.children[1].children.some((child) => child.textContent === 'TFLOPS'));
  assert.equal(context.formatComputeRate({ kind, precision: 'int8', flops_per_s: 2e12 }), '2.000 TOPS');
  assert.equal(context.formatComputeRate({ kind, precision: 'fp16', flops_per_s: 2e12 }), '2.000 TFLOPS');
}
const empty = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(empty, [], 'cpu');
assert.equal(empty.children.filter((child) => child.textContent === '未計測').length, 2);
assert.equal(context.formatComputeRate({ kind: 'cpu', flops_per_s: NaN }), '—');
const failedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(failedCard, [{ kind: 'gpu', precision: 'int8', ok: false, error: 'Unsupported INT8' }], 'gpu');
assert.equal(failedCard.children[3].textContent, '実行不可');
const reportDirectory = require('node:path').join(__dirname, '../results');
for (const filename of (fs.existsSync(reportDirectory) ? fs.readdirSync(reportDirectory) : []).filter((name) => name.endsWith('.json'))) {
  // Use the same legacy NaN/encoding normalization as the dashboard API.
  const python = process.env.PYTHON || require('node:path').join(__dirname, '../.venv/Scripts/python.exe');
  const report = JSON.parse(require('node:child_process').execFileSync(python, ['-c',
    'import json, sys; from fairbench.report import load, json_safe; r=load(sys.argv[1]); print(json.dumps(json_safe({"schema": r.get("schema"), "results": [{k:v for k,v in row.items() if k in ("kind", "precision", "ok", "flops_per_s", "flops", "error")} for row in r.get("results", [])]})))',
    require('node:path').join(reportDirectory, filename)], { encoding: 'utf8', cwd: require('node:path').join(__dirname, '..') }));
  for (const kind of ['cpu', 'gpu']) {
    if (report.schema !== 'fairbench.result/1') continue;
    const card = { children: [], appendChild(child) { this.children.push(child); } };
    context.renderPrecisionRates(card, report.results, kind);
    assert.equal(card.children[0].textContent, 'FP16', filename);
    assert.deepEqual(card.children.filter((child) => child.className === 'device-role').map((child) => child.textContent),
      ['FP16', 'INT8'], filename);
    assert.equal(card.children.length, 4, filename);
  }
}
console.log('Precision display passed: FP16 TFLOPS and INT8 TOPS on every device; FP32 excluded.');
assert.equal(context.formatExecutionPrecision({ compute_precisions: ['f32'] }), '入力型 FP32');
assert.equal(context.formatExecutionPrecision({ precision: 'fp16' }), 'FP16指定');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: true, tensor_precisions: ['float16'], evidence: 'profiled_fused_graph',
} }), '入力型 FP16');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: true, tensor_precisions: ['float16'], evidence: ['profiled_compute_inputs'],
} }), '入力型 FP16');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: false, tensor_precisions: ['int8'],
} }), '記録なし');
const probedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(probedCard, [{ kind: 'gpu', precision: 'fp16', ok: true, flops_per_s: 2e12,
  gpu_precision: { available: true, tensor_precisions: ['float16'], evidence: 'profiled_fused_graph' },
}], 'gpu', { details: true });
assert(probedCard.children.some((child) => child.textContent.includes('指定精度 FP16')));
assert(!probedCard.children.some((child) => child.textContent.includes('実行 FP16')));
assert(probedCard.children.some((child) => child.textContent.includes('参考値：要素単位の正解検証がありません')));
const fallbackCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(fallbackCard, [{ kind: 'cpu', precision: 'fp16', ok: true, flops_per_s: 1e12,
  compute_precisions: ['f32'], correctness_verified: true,
}], 'cpu', { details: true });
assert(fallbackCard.children.some((child) => child.textContent.includes('指定精度 FP16')));
assert(!fallbackCard.children.some((child) => child.textContent.includes('参考値：')));
for (const evidence of ['profiled_fused_graph', 'profiled_compute_inputs', undefined]) {
  const int8Row = { kind: 'gpu', precision: 'int8', ok: true, flops_per_s: 2e12, correctness_verified: true,
    compute_precisions: ['int8'], gpu_precision: { available: true, tensor_precisions: ['float'], evidence } };
  const int8Card = { children: [], appendChild(child) { this.children.push(child); } };
  context.renderPrecisionRates(int8Card, [int8Row], 'gpu', { details: true });
  assert(int8Card.children.some((child) => child.textContent.includes('指定精度 INT8')));
  assert(!int8Card.children.some((child) => /内部INT8|内部演算\/蓄積|ネイティブINT8|FP32/.test(child.textContent)));
  assert(!int8Card.children.some((child) => child.textContent.includes('GPU割当グラフ型')));
  assert.equal(context.formatComputeRate(int8Row), '2.000 TOPS');
  const summaryCard = { children: [], appendChild(child) { this.children.push(child); } };
  context.renderPrecisionRates(summaryCard, [int8Row], 'gpu');
  assert(!summaryCard.children[3].attributes.title);
}
assert.equal(context.formatExecutionPrecision({ compute_precisions: ['i8', 'f32'], compute_precision_complete: false }), '入力型 FP32/INT8');
const timedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(timedCard, [{ kind: 'gpu', precision: 'fp16', ok: true, flops_per_s: 2e12,
  timing_breakdown: { available: true, h2d_ms: 1, compute_ms: 2, d2h_ms: 3, repeats_measured: 3 } }], 'gpu', { details: true });
assert(timedCard.children.some((child) => child.textContent.includes('GPU常駐推論（同期込）2.000 ms')));
for (const kind of ['cpu', 'gpu', 'npu']) {
  for (const precision of ['fp16', 'int8']) {
    const row = { kind, precision, ok: true, flops_per_s: 2e12, flops: 16e9,
      timing_breakdown: { available: true, compute_ms: 2 } };
    const unit = precision === 'int8' ? 'TOPS' : 'TFLOPS';
    assert.equal(context.formatInferenceRate(row), `8.000 ${unit}`);
    assert.equal(context.formatInferenceRate({ ...row, measurement_mode: 'throughput' }), `2.000 ${unit}`);
    assert.equal(context.formatE2ERate({ ...row, timing_breakdown: { available: true, compute_ms: 2, e2e_ms: 4 } }), `4.000 ${unit}`);
    const card = { children: [], appendChild(child) { this.children.push(child); } };
    context.renderPrecisionRates(card, [row], kind);
    const primary = card.children.find((child) => child.className?.includes('precision-index') && child.textContent === '8.000');
    assert(primary);
    assert(primary.children.some((child) => child.textContent === `（E2E 2.000 ${unit}）`));
    const residentCard = { children: [], appendChild(child) { this.children.push(child); } };
    context.renderPrecisionRates(residentCard, [{ ...row, measurement_scope: 'device_resident' }], kind);
    assert(residentCard.children.flatMap((child) => child.children).some((child) => child.textContent === '（E2E 未取得）'));
    assert.equal(context.formatInferenceRate({ ...row, measurement_scope: 'device_resident' }), `2.000 ${unit}`);
    for (const compute_ms of [0, -1, NaN, Infinity, null]) {
      assert.equal(context.formatInferenceRate({ ...row, timing_breakdown: { available: true, compute_ms } }), '分離不可');
    }
    assert.equal(context.formatInferenceRate({ ...row, flops: undefined }), '分離不可');
    assert.equal(context.formatInferenceRate({ ...row, timing_breakdown: { available: false, compute_ms: 2 } }), '分離不可');
  }
}
const oldGrade = { results: [{ ok: true, repeats_ok: 1 }], grade: { checks: [
  { name: 'Repeatability', status: 'pass', detail: 'spread 0%' },
  { name: 'Timing stability', status: 'warn', detail: 'jitter 20%, but repeats agree within 0%, so throughput is stable' },
] } };
assert.deepEqual(context.gradeChecks(oldGrade), oldGrade.grade.checks);
assert.equal(context.gradeChecks(oldGrade)[0].status, 'pass');
assert(context.gradeChecks(oldGrade)[1].detail.includes('repeats agree'));
const legacyGrade = { grade: { score: 100, letter: 'A', checks: [{ name: 'Correctness', status: 'pass', detail: 'saved check' }] } };
const gradeHost = { children: [], appendChild(child) { this.children.push(child); }, querySelector() { return null; } };
context.document.getElementById = () => gradeHost;
context.requestAnimationFrame = (handler) => handler();
context.renderGrade(legacyGrade);
assert(gradeHost.children[0].innerHTML.includes('100.0'));
assert(gradeHost.children.some((child) => child.textContent.includes('保存時のスコアとチェック')));
assert(gradeHost.children.at(-1).children[0].innerHTML.includes('mark pass'));
assert.equal(legacyGrade.grade.score, 100);
assert.equal(context.gradeChecks(legacyGrade)[0].status, 'pass');
gradeHost.children = [];
context.renderGrade({ ...legacyGrade, measurement_contract: 'element-checks-static-qdq-v2' });
assert(!gradeHost.children.some((child) => child.textContent.includes('保存時のスコアとチェック')));
const peakCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(peakCard, [{ kind: 'gpu', precision: 'fp16', ok: true, flops_per_s: 30e12,
  peak_settings: { gemm_size: 2048, requests: 1, threads: 6 }, peak_candidate_count: 4,
  measurement_scope: 'device_resident' }], 'gpu', { details: true });
assert(peakCard.children.some((child) => child.textContent.includes('行列 2048×2048・1並列・6スレッド枠')));
assert(peakCard.children.some((child) => child.textContent.includes('GPU常駐（入出力転送を除外）')));
assert(peakCard.children.some((child) => child.textContent.includes('直接比較はできません')));
assert(peakCard.children.some((child) => child.textContent.includes('GEMM M×K×N: 2048×2048×2048')));
assert.equal(context.gemmShape({ peak_settings: { gemm_size: 1024 } }).join('×'), '1024×1024×1024');
assert.equal(context.gemmShape({ gemm_shape: [1, 2, 3], peak_settings: { gemm_size: 1024 } }).join('×'), '1×2×3');
assert.equal(context.gemmShape({}), null);
const sizedCard = { children: [], appendChild(child) { this.children.push(child); } };
const sizedRow = { kind: 'cpu', precision: 'int8', ok: true, flops_per_s: 1e12,
  gemm_size: 2048, gemm_shape: [2048, 2048, 2048], operation_convention: { operations_per_mac: 2, gemm_formula: '2*M*K*N' } };
context.renderPrecisionRates(sizedCard, [sizedRow], 'cpu', { details: true });
assert(sizedCard.children.some((child) => child.textContent.includes('GEMM M×K×N: 2048×2048×2048')));
assert(sizedCard.children.some((child) => child.textContent.includes('1 MAC = 2演算')));
assert(probedCard.children.some((child) => child.textContent.includes('行列サイズは未記録')));
const minimalSizedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(minimalSizedCard, [sizedRow], 'cpu');
assert.equal(minimalSizedCard.children.length, 4);
const detailDiagram = { children: [], appendChild(child) { this.children.push(child); }, replaceChildren() { this.children = []; } };
const detailHeading = {};
context.document.getElementById = (id) => ({ deviceMeasurement: detailDiagram, measurementTitle: detailHeading })[id];
context.renderDeviceMeasurement([]);
assert.deepEqual(detailDiagram.children[0].children.map((button) => button.textContent), ['CPU', 'GPU', 'NPU']);
detailDiagram.children[0].children[0].handlers.click();
assert.equal(detailHeading.textContent, 'CPUの転送と推論');
assert.equal(detailDiagram.children[0].children[0].attributes['aria-pressed'], 'true');
detailDiagram.children[0].children[2].handlers.click();
assert.equal(detailHeading.textContent, 'NPUの転送と推論');
context.renderDeviceMeasurement([{ kind: 'cpu', precision: 'fp16', ok: true, flops_per_s: 1e12 }]);
assert.equal(detailHeading.textContent, 'CPUの転送と推論');
assert.equal(detailDiagram.children[0].children[0].attributes['aria-pressed'], 'true');
context.renderDeviceMeasurement([{ kind: 'npu', precision: 'int8', ok: true, flops_per_s: 1e12 }]);
assert.equal(detailHeading.textContent, 'NPUの転送と推論');
const heroHosts = Object.fromEntries(['heroEyebrow', 'heroTitle', 'heroMeta', 'heroStats'].map((id) =>
  [id, { children: [], appendChild(child) { this.children.push(child); } }]));
context.document.getElementById = (id) => heroHosts[id];
context.renderHero({ results: [{ ok: true }, { ok: true }], cells: Array(6).fill({}), duration_s: 1 });
assert(heroHosts.heroStats.children[0].innerHTML.includes('組み合わせ'));
assert(heroHosts.heroStats.children[0].innerHTML.includes('>2<'));
assert(source.includes('${r.targets || 0} 組み合わせ'));
console.log('Disclosure checks passed: INT8 conversion/source labels, stored grade pairs, combination labels and legacy/default views.');
// A newly installed WinML provider must appear without keeping the old modal.
function modalNode() {
  return { children: [], dataset: {}, value: '', hidden: false,
    set innerHTML(value) { this.children = []; },
    get childElementCount() { return this.children.length; },
    appendChild(child) { this.children.push(child); }, setAttribute() {},
    querySelector() { return { addEventListener() {} }; }, querySelectorAll() { return []; } };
}
const modalControls = Object.fromEntries(['targetChecks', 'workloadChecks', 'optBackend', 'optSuite', 'optMode',
  'modeField', 'requestsField', 'optGemmSize'].map((id) => [id, modalNode()]));
const modalContext = vm.createContext({ document: {
  createElement: modalNode, getElementById: (id) => modalControls[id],
} });
vm.runInContext(source.split('/* ------------------------------------------------------------------ wire */')[0], modalContext);
vm.runInContext("state.catalog = {discovery: {targets: [{id:'cpu',label:'CPU',backend:'ort'}]},workloads:[]}", modalContext);
modalContext.buildRunModal();
assert.equal(modalControls.targetChecks.childElementCount, 1);
const firstHandler = modalControls.optBackend.onchange;
modalContext.buildRunModal();
assert.equal(modalControls.optBackend.onchange, firstHandler);
vm.runInContext("state.catalog.discovery.targets.push({id:'winml:rtx',label:'TensorRT-RTX',backend:'winml'})", modalContext);
modalContext.buildRunModal();
assert.equal(modalControls.targetChecks.childElementCount, 2);
assert(modalControls.optBackend.children.some((child) => child.textContent === 'Windows ML'));
console.log('WinML modal discovery passed: rebuild on device changes, preserve unchanged modal, replace handlers.');
const controls = Object.fromEntries(['optSuite', 'optMode', 'workloadChecks', 'modeField', 'requestsField', 'optGemmSize',
  'optRequests', 'optRepeats', 'optThreads', 'optBackend', 'optPower', 'optBattery', 'targetChecks']
  .map((id) => [id, { value: '', checked: false }]));
context.document.getElementById = (id) => controls[id];
for (const [suite, mode, customVisible, modeVisible, requestsVisible, sizeDisabled] of [
  ['standard', 'latency', false, true, false, false],
  ['standard', 'throughput', false, true, true, false],
  ['custom', 'latency', true, true, false, false],
  ['npu_peak', 'throughput', false, false, true, false],
  ['peak', 'latency', false, false, false, true],
]) {
  controls.optSuite.value = suite;
  controls.optMode.value = mode;
  context.syncRunFields();
  assert.equal(!controls.workloadChecks.hidden, customVisible);
  assert.equal(!controls.modeField.hidden, modeVisible);
  assert.equal(!controls.requestsField.hidden, requestsVisible);
  assert.equal(controls.optGemmSize.disabled, sizeDisabled);
}
const markup = fs.readFileSync(require.resolve('../fairbench/web/index.html'), 'utf8');
assert(!markup.split('id="overview"')[1].split('</section>')[0].includes('measurement-diagram'));
assert(markup.includes('class="measurement-diagram" data-view="performance"'));
assert(markup.includes('<details class="run-advanced">'));
for (const removed of ['optQuick', 'optInt8', 'optCheck']) assert(!markup.includes(`id="${removed}"`));
assert(!markup.includes('<option value="peak">'));
assert(markup.includes('<option selected>4096</option>'));
(async () => {
  controls.targetChecks.querySelectorAll = () => [{ value: 'ort:CPUExecutionProvider' }];
  controls.workloadChecks.querySelectorAll = () => [{ value: 'precision_gemm_fp16' }, { value: 'precision_gemm_int8' }];
  Object.assign(controls.optGemmSize, { value: '4096' });
  Object.assign(controls.optRepeats, { value: '3' });
  Object.assign(controls.optBackend, { value: 'all' });
  context.toast = () => {};
  let sent;
  context.fetch = async (url, options) => { sent = JSON.parse(options.body); return { json: async () => ({ ok: false }) }; };
  for (const [suite, mode, expectedMode, requests] of [
    ['standard', 'latency', 'latency', 0], ['custom', 'throughput', 'throughput', 4],
    ['npu_peak', 'throughput', 'throughput', 4], ['peak', 'latency', 'peak', 0],
  ]) {
    controls.optSuite.value = suite;
    controls.optMode.value = mode;
    controls.optRequests.value = '4';
    controls.workloadChecks.querySelectorAll = () => [{ value: 'conv_vision' }];
    await context.startRun();
    assert.deepEqual(sent.workloads, ['standard', 'peak'].includes(suite)
      ? ['precision_gemm_fp16', 'precision_gemm_int8'] : ['conv_vision']);
    assert.equal(sent.mode, expectedMode);
    assert.equal(sent.requests, requests);
    assert.equal(sent.gemm_size, 4096);
    assert.equal(sent.check_correctness, true);
    assert.equal(sent.include_int8, true);
    assert.equal(sent.repeats, 3);
    assert(!Object.hasOwn(sent, 'min_seconds'));
    assert(!Object.hasOwn(sent, 'quick'));
  }
  console.log('Simplified settings passed: conditional fields, fixed correctness, explicit timing and mode payloads.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
