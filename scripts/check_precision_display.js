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
const diagram = { children: [], appendChild(child) { this.children.push(child); }, replaceChildren() { this.children = []; } };
context.renderMeasurementDiagram(diagram, [{ kind: 'gpu', precision: 'fp16', ok: true,
  flops_per_s: 2e12, gemm_shape: [2048, 2048, 2048],
  timing_breakdown: { available: true, h2d_ms: 2.001, compute_ms: 0.754, d2h_ms: 3.449 } }]);
assert(diagram.children[0].textContent.includes('2048×2048×2048'));
const steps = diagram.children.find((child) => child.tag === 'ol').children;
assert.deepEqual(steps.map((step) => step.children[0].textContent), ['入力転送', 'GPU常駐推論', '出力転送']);
assert.deepEqual(steps.map((step) => step.children[1].textContent), ['2.001 ms', '0.754 ms', '3.449 ms']);
assert(steps[1].children.some((child) => child.textContent.includes('括弧内')));
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
  assert.deepEqual(card.children.filter((child) => child.className.includes('precision-index')).map((child) => child.textContent), values);
  assert.equal(card.children.length, 4);
  assert(card.children[1].children.some((child) => child.textContent === '（推論 分離不可）'));
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
  const report = JSON.parse(fs.readFileSync(require('node:path').join(reportDirectory, filename), 'utf8').replace(/^\uFEFF/, ''));
  for (const kind of ['cpu', 'gpu']) {
    const card = { children: [], appendChild(child) { this.children.push(child); } };
    context.renderPrecisionRates(card, report.results, kind);
    assert.equal(card.children[0].textContent, 'FP16', filename);
    assert.deepEqual(card.children.filter((child) => child.className === 'device-role').map((child) => child.textContent),
      ['FP16', 'INT8'], filename);
    assert.equal(card.children.length, 4, filename);
  }
}
console.log('Precision display passed: FP16 TFLOPS and INT8 TOPS on every device; FP32 excluded.');
assert.equal(context.formatExecutionPrecision({ compute_precisions: ['f32'] }), 'FP32（ランタイム報告）');
assert.equal(context.formatExecutionPrecision({ precision: 'fp16' }), '未確認');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: true, tensor_precisions: ['float16'], evidence: 'profiled_fused_graph',
} }), 'FP16（最適化前ONNXの演算入力型（GPU実行を別途確認）・内部演算/蓄積は未確認）');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: true, tensor_precisions: ['float16'], evidence: ['profiled_compute_inputs'],
} }), 'FP16（GPUプロファイルの演算入力型・内部演算/蓄積は未確認）');
assert.equal(context.formatExecutionPrecision({ gpu_precision: {
  available: false, tensor_precisions: ['int8'],
} }), '未確認');
const probedCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(probedCard, [{ kind: 'gpu', precision: 'fp16', ok: true, flops_per_s: 2e12,
  gpu_precision: { available: true, tensor_precisions: ['float16'], evidence: 'profiled_fused_graph' },
}], 'gpu', { details: true });
assert(probedCard.children.some((child) => child.textContent.includes('演算入力 FP16（最適化前ONNX')));
assert(!probedCard.children.some((child) => child.textContent.includes('実行 FP16')));
assert(probedCard.children.some((child) => child.textContent.includes('参考値：要素単位の正解検証がありません')));
const fallbackCard = { children: [], appendChild(child) { this.children.push(child); } };
context.renderPrecisionRates(fallbackCard, [{ kind: 'cpu', precision: 'fp16', ok: true, flops_per_s: 1e12,
  compute_precisions: ['f32'], correctness_verified: true,
}], 'cpu', { details: true });
assert(fallbackCard.children.some((child) => child.textContent.includes('ネイティブFP16演算性能ではありません')));
assert(!fallbackCard.children.some((child) => child.textContent.includes('参考値：')));
assert(context.precisionRateNote({ compute_precisions: ['f32'],
  gpu_precision: { available: true, tensor_precisions: ['float16'], evidence: 'profiled_fused_graph' } }, 'fp16')
  .includes('ネイティブFP16演算性能ではありません'));
for (const evidence of ['profiled_fused_graph', 'profiled_compute_inputs', undefined]) {
  const int8Row = { kind: 'gpu', precision: 'int8', ok: true, flops_per_s: 2e12, correctness_verified: true,
    compute_precisions: ['int8'], gpu_precision: { available: true, tensor_precisions: ['float'], evidence } };
  const int8Card = { children: [], appendChild(child) { this.children.push(child); } };
  context.renderPrecisionRates(int8Card, [int8Row], 'gpu', { details: true });
  assert(int8Card.children.some((child) => child.textContent.includes('ネイティブINT8演算性能ではありません')));
  assert(int8Card.children.some((child) => child.textContent.includes('TOPS換算値')));
  assert(int8Card.children.some((child) => child.textContent.includes('FP32')));
  assert(!int8Card.children.some((child) => child.textContent.includes('GPU割当グラフ型')));
  assert.equal(context.formatComputeRate(int8Row), '2.000 TOPS');
  const summaryCard = { children: [], appendChild(child) { this.children.push(child); } };
  context.renderPrecisionRates(summaryCard, [int8Row], 'gpu');
  assert(summaryCard.children[3].attributes.title.includes('ネイティブINT8演算性能ではありません'));
}
assert(context.precisionRateNote({ compute_precisions: ['f32'] }, 'int8').includes('ネイティブINT8'));
assert.equal(context.precisionRateNote({ compute_precisions: ['int8', 'uint8'] }, 'int8'), '');
assert.equal(context.formatExecutionPrecision({ compute_precisions: ['i8', 'f32'], compute_precision_complete: false }), 'FP32/INT8（一部未確認）');
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
    const card = { children: [], appendChild(child) { this.children.push(child); } };
    context.renderPrecisionRates(card, [row], kind);
    assert(card.children.flatMap((child) => child.children).some((child) => child.textContent === `（推論 8.000 ${unit}）`));
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
const controls = Object.fromEntries(['optSuite', 'optMode', 'workloadChecks', 'modeField', 'requestsField', 'optGemmSize',
  'optRequests', 'optRepeats', 'optSeconds', 'optThreads', 'optBackend', 'optPower', 'optBattery', 'targetChecks']
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
  Object.assign(controls.optSeconds, { value: '4' });
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
    await context.startRun();
    assert.equal(sent.mode, expectedMode);
    assert.equal(sent.requests, requests);
    assert.equal(sent.gemm_size, 4096);
    assert.equal(sent.check_correctness, true);
    assert.equal(sent.include_int8, true);
    assert.equal(sent.repeats, 3);
    assert.equal(sent.min_seconds, 4);
    assert(!Object.hasOwn(sent, 'quick'));
  }
  console.log('Simplified settings passed: conditional fields, fixed correctness, explicit timing and mode payloads.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
