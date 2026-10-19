/* ==========================================================================
   fairbench dashboard - vanilla JS, no build step, no network.
   ========================================================================== */
'use strict';

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';

const KIND_COLOR = { cpu: '#2563eb', gpu: '#15803d', npu: '#7c3aed' };
const VENDOR_COLOR = { intel: '#4f9dff', amd: '#ff5a5f', nvidia: '#3f7a0a', qualcomm: '#7c3aed' };

const METHOD = [
  ['Identical work.', 'Every device runs byte-identical ONNX graphs generated from a fixed seed. Model hashes are printed below.'],
  ['Identical inputs.', 'One seeded tensor set per graph, shared by every device and every repeat.'],
  ['Identical measurement.', 'Same warm-up, same time budget, same aggregation: the median, never the best case.'],
  ['Physical cores, one stream.', 'CPU comparisons use physical cores and a single inference stream, so a bigger core count cannot buy a fake win.'],
  ['Correctness gate.', 'Each device\u2019s output is compared against the float32 CPU reference; a silently wrong accelerator is flagged, not published.'],
  ['Disclosed environment.', 'Threads, precision, driver, power plan and power source are recorded in every result file.'],
  ['Isolated cells.', 'Each measurement runs in a fresh process, so a driver crash or leaked memory cannot distort the next one.'],
  ['Energy where possible.', 'On a laptop, whole-system watts come from the battery, the one sensor that is identical for Intel and AMD.'],
];

let state = { catalog: null, results: [], report: null, activeFile: null, run: null, timer: null };
let measurementKind = 'gpu';

/* ------------------------------------------------------------------ utils */
function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  if (text !== undefined) node.textContent = text;
  return node;
}
function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const fmtInt = (v) => (v === null || v === undefined) ? '—' : Math.round(v).toLocaleString();
const isModelOnly = (row) => (row.measurement_validity || row.session?.measurement_validity) === 'model_only';
function modelOperationRate(row) {
  if (Number.isFinite(row.model_operations_per_s)) return row.model_operations_per_s;
  return Number.isFinite(row.flops) && row.flops > 0 && Number.isFinite(row.units_per_s)
    ? row.flops * row.units_per_s : null;
}
function formatComputeRate(row) {
  if (isModelOnly(row)) {
    const rate = modelOperationRate(row);
    return Number.isFinite(rate) ? `${fmt(rate / 1e12, 3)} TOPS` : '換算不可（演算数未記録）';
  }
  if (!Number.isFinite(row.flops_per_s)) return '—';
  return row.precision?.toLowerCase().startsWith('int')
    ? `${fmt(row.flops_per_s / 1e12, 3)} TOPS`
    : `${fmt(row.flops_per_s / 1e12, 3)} TFLOPS`;
}
function formatInferenceRate(row) {
  if (row.measurement_scope === 'device_resident' || ['throughput', 'peak'].includes(row.measurement_mode)) return formatComputeRate(row);
  const timing = row.timing_breakdown;
  if (!timing?.available || !Number.isFinite(timing.compute_ms) || timing.compute_ms <= 0
    || !Number.isFinite(row.flops) || row.flops <= 0) return '分離不可';
  const rate = row.flops / (timing.compute_ms / 1000);
  return formatComputeRate({ ...row, flops_per_s: rate, model_operations_per_s: rate });
}
function inferenceRateValue(row) {
  if (row.measurement_scope === 'device_resident' || ['throughput', 'peak'].includes(row.measurement_mode)) {
    const rate = isModelOnly(row) ? modelOperationRate(row) : row.flops_per_s;
    return Number.isFinite(rate) ? rate : -Infinity;
  }
  const timing = row.timing_breakdown;
  return timing?.available && Number.isFinite(timing.compute_ms) && timing.compute_ms > 0
    && Number.isFinite(row.flops) && row.flops > 0 ? row.flops / (timing.compute_ms / 1000) : -Infinity;
}
function compareInferenceRates(first, second) {
  const a = inferenceRateValue(first), b = inferenceRateValue(second);
  return a === b ? second.flops_per_s - first.flops_per_s : a > b ? -1 : 1;
}
function formatE2ERate(row) {
  const timing = row.timing_breakdown;
  if (timing?.available && Number.isFinite(timing.e2e_ms) && timing.e2e_ms > 0
    && Number.isFinite(row.flops) && row.flops > 0) {
    const rate = row.flops / (timing.e2e_ms / 1000);
    return formatComputeRate({ ...row, flops_per_s: rate, model_operations_per_s: rate });
  }
  return formatComputeRate(row);
}
function gemmShape(row) {
  if (Array.isArray(row?.gemm_shape) && row.gemm_shape.length === 3) return row.gemm_shape;
  const size = row?.peak_settings?.gemm_size;
  return Number.isFinite(size) && size > 0 ? [size, size, size] : null;
}
function renderPrecisionRates(card, rows, kind, { details = false } = {}) {
  const precisions = ['fp16', 'int8'];
  for (const precision of precisions) {
    const unit = precision === 'int8' ? 'TOPS' : 'TFLOPS';
    const matching = rows.filter((row) => row.kind === kind && row.precision?.toLowerCase() === precision);
    const best = matching.filter((row) => row.ok
      && Number.isFinite(isModelOnly(row) ? modelOperationRate(row) : row.flops_per_s)
      && (isModelOnly(row) ? modelOperationRate(row) : row.flops_per_s) >= 0)
      .sort(compareInferenceRates)[0];
    card.appendChild(el('p', { class: 'device-role' }, details ? `${precision.toUpperCase()}指定 · 実測性能` : precision.toUpperCase()));
    const failed = matching.find((row) => !row.ok);
    const failureReason = failed?.phase === 'precision' ? '旧結果では性能測定が省略されています。再計測すると実測性能とE2Eを表示します。'
      : failed && /Native .*GEMM|NOT_IMPLEMENTED|could not load/.test(failed.error || '')
      ? 'この機器・ランタイムでは対応する精度の行列演算を実行できません。詳細は測定データを確認してください。'
      : (failed?.error || 'この機器・ランタイムでは実行できません').slice(0, 140);
    const inference = best ? formatInferenceRate(best) : null;
    if (best && ['throughput', 'peak'].includes(best.measurement_mode)) {
      card.appendChild(el('p', { class: 'device-path' }, '並列処理の合計スループット（単発推論性能とは測定条件が異なります）'));
    }
    const value = el('div', { class: 'device-index precision-index' }, best ? inference.replace(` ${unit}`, '') : failed?.phase === 'precision' ? '再計測が必要' : failed ? '実行不可' : '未計測');
    if (best) {
      if (inference !== '分離不可') value.appendChild(el('small', {}, unit));
      value.appendChild(el('small', { class: 'inference-rate',
        title: ['device_resident', 'persistent_requests'].includes(best.measurement_scope)
          ? '固定バッファの並列測定のため、独立した転送込みE2Eは取得していません。'
          : best.timing_breakdown?.e2e_ms > 0 ? '推論と同じ試行で測定した入力準備・転送から出力取得までの全体時間です。'
            : '旧結果：推論とは別測定のE2Eです。再計測すると同じ試行の値になります。',
      }, ['device_resident', 'persistent_requests'].includes(best.measurement_scope) ? '（E2E 未取得）' : `（E2E ${formatE2ERate(best)}）`));
    }
    card.appendChild(value);
    if (!details) continue;
    if (failed) {
      const diagnostic = el('details', { class: 'device-path' });
      diagnostic.appendChild(el('summary', {}, `${precision.toUpperCase()} 失敗の詳細（コピーできます）`));
      diagnostic.appendChild(el('pre', { style: 'white-space:pre-wrap;overflow-wrap:anywhere' }, failed.error || '詳細は記録されていません'));
      card.appendChild(diagnostic);
    }
    card.appendChild(el('p', { class: 'device-definition', title: failed?.error || '' }, best
      ? `指定精度 ${precision.toUpperCase()} · ${best.target_label || best.target_id} · ${best.repeats_ok || 0}回`
      : failed ? `${precision.toUpperCase()}: ${failureReason}` : `${precision.toUpperCase()}の測定値がありません`));
    if (best) {
      const shape = gemmShape(best);
      const convention = best.operation_convention;
      card.appendChild(el('p', { class: 'device-path' }, Array.isArray(shape) && shape.length === 3
        ? `GEMM M×K×N: ${shape.join('×')}（A ${shape[0]}×${shape[1]} / B ${shape[1]}×${shape[2]}）`
        : '行列サイズは未記録です。サイズを確認するには再計測してください。'));
      card.appendChild(el('p', { class: 'device-path' }, convention?.operations_per_mac
        ? `演算規約: 1 MAC = ${convention.operations_per_mac}演算、GEMM演算数 = ${convention.gemm_formula}。変換処理は演算数に含めず、測定時間には含みます。`
        : 'この保存結果には演算数の規約が記録されていません。'));
    }
    if (best?.peak_settings) {
      const settings = best.peak_settings;
      const scope = best.measurement_scope === 'device_resident' ? 'GPU常駐（入出力転送を除外）'
        : best.measurement_scope === 'persistent_requests' ? '固定バッファ・リクエスト（ランタイム内部転送は含む）' : '転送込み';
      card.appendChild(el('p', { class: 'device-path' }, `探索${best.peak_candidate_count}候補 → 行列 ${settings.gemm_size}×${settings.gemm_size}・${settings.requests}並列・${settings.threads}スレッド枠 / ${scope}`));
      card.appendChild(el('p', { class: 'device-path' }, '条件別探索値：サイズ・並列数・測定範囲が異なるデバイス間の直接比較はできません。'));
    }
    if (best && !best.correctness_verified) {
      card.appendChild(el('p', { class: 'device-path' }, '参考値：要素単位の正解検証がありません。旧方式の結果は再計測してください。'));
    }
    if (best) {
      const timing = best.timing_breakdown || {};
      if (details && timing.available) {
        card.appendChild(el('p', { class: 'device-path' },
          `推論 ${timing.compute_samples || '未記録'}サンプル・試行ごとの中央値を集約`
          + (Number.isFinite(timing.compute_cv_max) ? ` / 最大CV ${fmt(timing.compute_cv_max * 100, 1)}%`
            + (timing.compute_cv_max > 0.05 ? '（ばらつきあり）' : '') : '')
          + (Number.isFinite(timing.compute_repeat_spread_pct) ? ` / 推論の試行間差 ${fmt(timing.compute_repeat_spread_pct, 1)}%`
            + (timing.compute_repeat_spread_pct > 5 ? '（不安定）' : '') : ' / 単一試行：再現性は未確認')));
      }
      const hostBuffers = timing.transfer_scope === 'host_buffers';
      card.appendChild(el('p', { class: 'device-path', title: timing.note || '' }, timing.available
        ? `分離時間（別測定）: ${hostBuffers ? '入力準備・コピー' : '入力転送'} ${fmt(timing.h2d_ms, 3)} ms / ${hostBuffers ? kind.toUpperCase() + '推論（内部転送を含む場合あり）' : 'GPU常駐推論（同期込）'}${fmt(timing.compute_ms, 3)} ms / ${hostBuffers ? '出力取得・コピー' : '出力転送'} ${fmt(timing.d2h_ms, 3)} ms · ${timing.repeats_measured}回の中央値`
        : '転送・計算の分離時間は未取得'));
    }
  }
}
function renderMeasurementDiagram(host, rows, kind = 'gpu', precision = 'fp16') {
  host.replaceChildren();
  const name = kind.toUpperCase();
  const row = rows.filter((row) => row.ok && row.kind === kind
    && row.precision?.toLowerCase() === precision && (Number.isFinite(row.flops_per_s) || isModelOnly(row) && Number.isFinite(row.units_per_s)))
    .sort(compareInferenceRates)[0];
  const timing = row?.timing_breakdown;
  const hostBuffers = timing?.transfer_scope === 'host_buffers';
  const shape = gemmShape(row);
  host.appendChild(el('p', { class: 'measurement-caption' }, row
    ? `表示中の${name}・${precision.toUpperCase()}${Array.isArray(shape) ? ` / GEMM ${shape.join('×')}` : ''} · 分離時間は別測定`
    : `${name}・${precision.toUpperCase()}は未計測です`));
  const flow = el('ol', { class: 'measurement-flow', 'aria-label': 'データの移動と推論の順序' });
  for (const [label, description, field] of [
    [hostBuffers ? '入力準備・コピー' : '入力転送', hostBuffers || kind === 'cpu' ? 'ホスト側の入力バッファへコピー' : `CPUメモリ → ${name}メモリ`, 'h2d_ms'],
    [kind === 'cpu' || hostBuffers ? `${name}推論` : `${name}常駐推論`, timing?.internal_transfers_in_compute ? '内部転送・行列演算・変換・同期' : '行列演算・内部変換・同期', 'compute_ms'],
    [hostBuffers ? '出力取得・コピー' : '出力転送', hostBuffers || kind === 'cpu' ? 'ホスト側の出力配列へコピー' : `${name}メモリ → CPUメモリ`, 'd2h_ms'],
  ]) {
    const step = el('li', { class: field === 'compute_ms' ? 'measurement-step inference-step' : 'measurement-step' });
    step.appendChild(el('strong', {}, label));
    const duration = timing?.available && Number.isFinite(timing[field]) && timing[field] >= 0
      ? `${fmt(timing[field], 3)} ms` : '未取得';
    step.appendChild(el('span', { class: 'measurement-duration' }, duration));
    step.appendChild(el('span', {}, description));
    if (field === 'compute_ms') step.appendChild(el('b', {}, '主値の「推論」はこの区間'));
    flow.appendChild(step);
  }
  host.appendChild(flow);
  host.appendChild(el('p', { class: 'measurement-scope' }, '主値：推論時間から換算した性能 / 括弧内のE2E：入力転送 → 推論 → 出力転送を含む全体の処理速度'));
  if (row && !timing?.available) host.appendChild(el('p', { class: 'measurement-caption' }, 'この測定結果には転送・推論の分離時間がありません。全体の時間からは推定しません。'));
  if (hostBuffers) host.appendChild(el('p', { class: 'measurement-caption' }, '入出力はホスト側の準備・コピー時間です。デバイス内部の物理転送を個別に測った値ではありません。ランタイム内部の転送は推論時間に含まれる場合があります。'));
  host.appendChild(el('p', { class: 'measurement-caption' }, '主値：演算数 ÷ 別測定の推論時間。純粋なカーネル時間ではありません。分離時間は各区間の中央値のため、合計がE2E測定時間と一致するとは限りません。'));
  host.appendChild(el('p', { class: 'measurement-caption' }, 'CPU・NPUも分離時間を取得できた場合だけ主値に換算値を表示します。分離できない場合は「分離不可」。デバイス常駐測定では転送込みE2Eは未取得です。'));
}
function renderDeviceMeasurement(rows, selectedKind) {
  if (selectedKind) measurementKind = selectedKind;
  else if (!rows.some((row) => row.ok && row.kind === measurementKind)) {
    measurementKind = ['gpu', 'cpu', 'npu'].find((kind) => rows.some((row) => row.ok && row.kind === kind)) || measurementKind;
  }
  const host = $('deviceMeasurement');
  host.replaceChildren();
  $('measurementTitle').textContent = `${measurementKind.toUpperCase()}の転送と推論`;
  const devices = el('div', { class: 'measurement-devices', role: 'group', 'aria-label': '時間内訳を表示するデバイス' });
  for (const kind of ['cpu', 'gpu', 'npu']) {
    const selector = el('button', { type: 'button', class: 'device-selector',
      'aria-pressed': String(measurementKind === kind), 'aria-controls': 'deviceMeasurement',
    }, kind.toUpperCase());
    selector.addEventListener('click', () => {
      renderDeviceMeasurement(state.report?.results || [], kind);
    });
    devices.appendChild(selector);
  }
  host.appendChild(devices);
  for (const precision of ['fp16', 'int8']) {
    const section = el('section', { class: 'measurement-precision' });
    section.appendChild(el('h4', {}, precision.toUpperCase()));
    const diagram = el('div');
    renderMeasurementDiagram(diagram, rows, measurementKind, precision);
    section.appendChild(diagram);
    host.appendChild(section);
  }
}
function renderPrecisionDetails(rep) {
  const host = $('precisionDetails');
  host.replaceChildren();
  for (const [kind, name] of [['cpu', 'CPU'], ['gpu', 'GPU'], ['npu', 'NPU']]) {
    const card = el('article', { class: 'device-card', style: `--accent:${KIND_COLOR[kind]}` });
    card.appendChild(el('h3', {}, name));
    renderPrecisionRates(card, rep.results || [], kind, { details: true });
    const discovery = rep.discovery || state.catalog?.discovery;
    if (kind === 'npu' && discovery && !discovery.targets?.some((target) => target.kind === 'npu' && target.available)) {
      card.appendChild(el('p', { class: 'device-path' }, '測定可能なNPUが検出されていません。機器情報で対応状況を確認できます。'));
    }
    host.appendChild(card);
  }
}
function formatExecutionPrecision(row) {
  if (row.precision) {
    const executed = row.ok || row.phase === 'precision';
    return `${row.precision.toUpperCase()}指定${executed ? ` · ${row.kind?.toUpperCase() || 'デバイス'}実行済み` : ''}`;
  }
  const gpu = row.gpu_precision || row.session?.gpu_precision;
  const gpuAssigned = gpu?.available && gpu.tensor_precisions?.length;
  const values = gpuAssigned ? gpu.tensor_precisions : row.compute_precisions || row.session?.compute_precisions || [];
  if (!values.length) return '記録なし';
  const names = { f32: 'FP32', float: 'FP32', float32: 'FP32', f16: 'FP16', float16: 'FP16',
    i8: 'INT8', int8: 'INT8', u8: 'UINT8', uint8: 'UINT8', i32: 'INT32', int32: 'INT32' };
  const text = [...new Set(values.map((value) => names[value.toLowerCase()] || value.toUpperCase()))].sort().join('/');
  return `入力型 ${text}`;
}
function gradeChecks(report) {
  // The stored score and its checks are one verdict; never mix grading policies.
  return report?.grade?.checks || [];
}
function fmtBytes(b) {
  if (!b) return '—';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0, n = b;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + ' ' + u[i];
}
function fmtDuration(s) {
  if (s === null || s === undefined) return '—';
  if (s < 60) return s.toFixed(1) + 's';
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return `${m}m ${r}s`;
}
function ago(ts) {
  if (!ts) return '';
  const d = Date.now() / 1000 - ts;
  if (d < 60) return 'just now';
  if (d < 3600) return Math.floor(d / 60) + ' min ago';
  if (d < 86400) return Math.floor(d / 3600) + ' h ago';
  return Math.floor(d / 86400) + ' d ago';
}
function toggles(sel, on) { document.querySelectorAll(sel).forEach((n) => n.classList.toggle('sel', on)); }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function deviceColor(row) {
  if (row.kind === 'cpu') {
    const id = row.target_id || row.id || '';
    const backend = row.backend || id.split(':')[0];
    return { openvino: '#1e3a8a', ort: '#2563eb', winml: '#38bdf8' }[backend] || KIND_COLOR.cpu;
  }
  if (row.kind === 'gpu') {
    const id = row.target_id || row.id || '';
    const backend = row.backend || id.split(':')[0];
    const provider = row.device || id.split(':')[1];
    const colors = {
      'openvino': '#16a34a',
      'ort:DmlExecutionProvider': '#14532d',
      'winml:DmlExecutionProvider': '#4ade80',
      'winml:OpenVINOExecutionProvider': '#86efac',
      'winml:NvTensorRTRTXExecutionProvider': '#0d9488',
      'ort:CUDAExecutionProvider': '#15803d',
      'ort:ROCMExecutionProvider': '#22c55e',
    };
    return colors[`${backend}:${provider}`] || colors[backend] || KIND_COLOR.gpu;
  }
  return KIND_COLOR[row.kind] || '#94a3b8';
}

function toast(msg, bad = false) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast' + (bad ? ' bad' : '');
  t.hidden = false;
  clearTimeout(t._h);
  t._h = setTimeout(() => { t.hidden = true; }, 4200);
}

/* ------------------------------------------------------------------ data */
async function loadState() {
  let data;
  try {
    const res = await fetch('/api/state');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    data = await res.json();
  } catch (err) {
    state.failures = (state.failures || 0) + 1;
    if (state.failures <= 2) toast('cannot reach the fairbench server', true);
    schedule(Math.min(15000, 1000 * 2 ** state.failures));
    return;
  }
  state.failures = 0;
  const firstCatalog = !state.catalog;
  const wasActive = state.run?.active;
  state.catalog = data.catalog;
  state.results = data.results || [];
  state.run = data.run;
  renderStatic();
  renderRunList();
  renderLive();
  if (firstCatalog || wasActive !== state.run?.active) renderOverview();

  const runFile = state.run && state.run.result_file;
  const newest = state.results[0] && state.results[0].file;

  if (!state.activeFile && !state.followRun) {
    // First paint: show what the last run produced, or the newest file.
    const first = new URLSearchParams(location.search).get('run') || runFile || newest;
    if (first) await openResult(first);
  } else if (state.followRun && state.run && !state.run.active && runFile) {
    // A run we started ourselves just finished - show its report once, then
    // stop following so the user's own selection is never overridden.
    if (await openResult(runFile)) state.followRun = false;
  }
  schedule(state.run && state.run.active ? 250 : 3000);
}

function schedule(ms) {
  clearTimeout(state.timer);
  state.timer = setTimeout(loadState, ms);
}

async function openResult(file) {
  try {
    const res = await fetch('/api/result/' + encodeURIComponent(file));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    state.report = await res.json();
    state.activeFile = file;
    const url = new URL(location.href);
    url.searchParams.set('run', file);
    history.replaceState(null, '', url);
  } catch (err) { toast('could not load ' + file, true); return false; }
  renderRunList();
  renderAll();
  return true;
}

/* ------------------------------------------------------------------ render */
function renderStatic() {
  const cat = state.catalog;
  if (!cat) return;
  const sys = cat.system || {};
  const chips = $('hostChips');
  chips.innerHTML = '';
  const cpu = sys.cpu || {};
  chips.appendChild(el('span', { class: 'chip' }, ''));
  chips.lastChild.innerHTML = `<span class="dot"></span><b>${esc(cpu.name || 'cpu')}</b> ${cpu.physical_cores || '?'}C/${cpu.logical_cores || '?'}T`;
  if (sys.memory && sys.memory.total_gib) chips.appendChild(el('span', { class: 'chip', html: `memory <b>${sys.memory.total_gib} GiB</b>${sys.memory.speed_mts ? ' @ ' + sys.memory.speed_mts : ''}` }));
  (sys.adapters || []).forEach((a) => {
    const c = el('span', { class: 'chip' });
    c.innerHTML = `<span class="dot" style="background:${KIND_COLOR[a.kind] || '#94a3b8'}"></span><b>${esc(a.name)}</b>`;
    chips.appendChild(c);
  });
  const runtimes = (cat.discovery && cat.discovery.runtimes) || {};
  const rt = Object.entries(runtimes).map(([k, v]) => `${k} ${String(v).split('-')[0]}`).join(' · ');
  if (rt) chips.appendChild(el('span', { class: 'chip', html: `runtime <b>${esc(rt)}</b>` }));

  renderMethod();
  renderMatrix();
  buildRunModal();
}

function renderMethod() {
  const list = $('methodList');
  if (list.childElementCount) return;
  METHOD.forEach(([b, t]) => {
    const li = el('li');
    li.innerHTML = `<b>${esc(b)}</b> ${esc(t)}`;
    list.appendChild(li);
  });
}

function renderMatrix() {
  const wrap = $('matrix');
  const cat = state.catalog;
  if (!cat) return;
  const caps = (cat.discovery && cat.discovery.capabilities) || {};
  const order = ['intel:cpu', 'intel:gpu', 'intel:npu', 'amd:cpu', 'amd:gpu', 'amd:npu', 'nvidia:gpu', 'qualcomm:npu'];
  const verified = {};
  if (state.report && state.report.targets) state.report.targets.forEach((t) => { verified[t.id] = t; });
  wrap.innerHTML = '';

  order.filter((k) => caps[k]).forEach((key) => {
    const cap = caps[key];
    const slot = cap.slot;
    const accent = VENDOR_COLOR[slot.vendor] || '#94a3b8';
    const on = !!cap.available;
    const card = el('div', { class: `slot ${on ? 'on' : 'off'}`, style: `--accent:${accent}` });
    const targets = (cap.targets || []).map((id) => verified[id]).filter(Boolean);
    const targetIds = cap.targets || [];
    let status = 'not present';
    if (on) status = targetIds.length + ' runtime path' + (targetIds.length === 1 ? '' : 's');
    card.innerHTML = `
      <div class="slot-head">
        <span class="slot-kind" style="color:${KIND_COLOR[slot.kind]};border-color:${KIND_COLOR[slot.kind]}55;background:${KIND_COLOR[slot.kind]}22">${slot.kind.toUpperCase()}</span>
        <span class="slot-vendor">${esc(slot.vendor)}</span>
      </div>
      <div class="slot-name">${esc(slot.label)}</div>
      <div class="slot-detail">${esc(cap.detail || '')}</div>
      <div class="slot-foot">
        <span class="badge ${on ? 'live' : 'none'}">${on ? 'MEASURABLE' : 'UNAVAILABLE'}</span>
        <span>${esc(status)}</span>
      </div>`;
    if (on) {
      const foot = card.querySelector('.slot-foot');
      targetIds.forEach((id) => {
        const v = verified[id];
        const b = el('span', { class: `badge ${v ? (v.verified ? 'live' : 'none') : 'idle'}`, style: 'margin-left:4px' },
          v ? (v.verified ? 'verified' : 'dropped') : 'unverified');
        foot.appendChild(b);
      });
    }
    wrap.appendChild(card);
  });

  const notes = $('matrixNotes');
  notes.innerHTML = '';
  const disc = (cat.discovery || {});
  (disc.notes || []).forEach((n) => {
    const li = el('div');
    li.textContent = '· ' + n;
    notes.appendChild(li);
  });
}

function renderRunList() {
  const list = $('runList');
  $('runCount').textContent = state.results.length;
  list.innerHTML = '';
  if (!state.results.length) {
    list.appendChild(el('p', { class: 'empty' }, 'no runs yet — press “new run”.'));
    return;
  }
  state.results.forEach((r) => {
    const btn = el('button', { class: 'run-item' + (r.file === state.activeFile ? ' active' : '') });
    btn.innerHTML = `
      <div class="r-top">
        <span class="r-name">${esc(r.host || 'unknown')}</span>
        ${r.grade ? `<span class="grade-pill grade-${esc(r.grade)}">${esc(r.grade)}</span>` : ''}
      </div>
      <div class="r-sub"><span>${esc((r.device || '').slice(0, 38))}</span></div>
      <div class="r-sub"><span>${r.targets || 0} 組み合わせ</span><span>${fmtDuration(r.duration_s)}</span><span>${ago(r.saved_at)}</span></div>`;
    btn.onclick = () => { state.hideLive = true; state.followRun = false; openResult(r.file); location.hash = 'overview'; };
    list.appendChild(btn);
  });
}

function renderAll() {
  const rep = state.report;
  if (!rep) return;
  renderHero(rep);
  renderIndexChart(rep);
  renderGrade(rep);
  renderThroughput(rep);
  renderRadar(rep);
  renderTable(rep);
  renderPrecisionDetails(rep);
  renderProvenance(rep);
  renderMatrix();
  renderOverview();
}

/* Keep the overview focused on indices, coverage and validity. */
function renderOverview() {
  const rep = state.report;
  const sys = (rep || state.catalog || {}).system || {};
  $('overviewMachine').textContent = (sys.cpu || {}).name || 'このPC';
  $('overviewDate').textContent = rep
    ? new Date(rep.started_at * 1000).toLocaleString('ja-JP') + ' の計測 · 履歴 →'
    : '過去の結果を見る →';
  const checks = gradeChecks(rep);
  const failed = checks.filter((c) => c.status === 'fail').length;
  const warned = checks.filter((c) => c.status === 'warn').length;
  const status = $('overviewStatus');
  status.className = 'overview-status ' + (!rep ? 'empty-state' : failed ? 'needs-check' : warned ? 'caution' : '');
  status.replaceChildren();
  const verdict = state.run?.active ? (rep ? '計測中 · 前回の結果を表示' : '計測中')
    : !rep ? 'まだ計測結果がありません' : failed ? '結果は要確認' : warned ? '注意点のある結果' : '計測完了';
  status.appendChild(el('strong', {}, verdict));
  status.appendChild(el('span', {}, !rep ? '右上の「計測を始める」から開始できます。'
    : `確認が必要な項目 ${failed}件 ・ 注意 ${warned}件`));
  if (rep) status.appendChild(el('a', { href: '#environment' }, '理由を見る →'));

  const rows = (rep?.results || []).filter((r) => r.ok);
  const host = $('overviewDevices');
  host.replaceChildren();
  for (const [kind, name] of [['cpu', 'CPU'], ['gpu', 'GPU'], ['npu', 'NPU']]) {
    const measured = rows.filter((r) => r.kind === kind);
    const card = el('article', { class: 'device-card' + (measured.length ? '' : ' unmeasured'), style: `--accent:${KIND_COLOR[kind]}` });
    const title = el('div', { class: 'device-card-heading' });
    title.appendChild(el('h3', {}, name));
    card.appendChild(title);
    renderPrecisionRates(card, rep?.results || [], kind);
    card.appendChild(el('a', { class: 'text-link', href: measured.length ? '#performance' : '#devices' },
      measured.length ? '測定データを見る →' : '対応状況を見る →'));
    host.appendChild(card);
  }
  const total = (rep?.results || []).length;
  renderDeviceMeasurement(rep?.results || []);
  $('overviewCoverage').textContent = rep
    ? `${rows.length}/${total}件成功 · 所要 ${fmtDuration(rep.duration_s)}。指定精度ごとの実測性能です。同じ行列サイズ・測定範囲で比較してください。`
    : 'CPU・GPU・NPUそれぞれでFP16（TFLOPS）とINT8（TOPS）を計測します。対応しない精度は実行不可の理由を表示します。';
  if (rep?.config?.mode === 'peak') {
    $('overviewCoverage').textContent += ' 最大探索ではデバイスごとにサイズ・並列数・測定範囲が異なります。探索内の最良条件を固定した確認測定の中央値で、理論ピークや絶対最大値ではありません。';
  }
}

const VIEWS = {
  overview: '概要', performance: '測定データ', devices: '機器情報',
  environment: '信頼性・計測条件', history: '実行履歴', method: '計測方法', activity: '実行ログ',
};
function renderRoute() {
  const requested = location.hash.slice(1);
  const view = Object.hasOwn(VIEWS, requested) ? requested : 'overview';
  document.querySelectorAll('[data-view]').forEach((node) => {
    node.hidden = !node.dataset.view.split(' ').includes(view);
  });
  document.querySelectorAll('.view-nav a').forEach((link) => {
    if (link.hash === '#' + view) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  $('viewTitle').textContent = VIEWS[view];
  document.title = `fairbench · ${VIEWS[view]}`;
}

function renderHero(rep) {
  const sys = rep.system || {};
  const good = (rep.results || []).filter((r) => r.ok);
  const scores = rep.scores || {};
  const best = (scores.ai_index || [])[0];
  $('heroEyebrow').textContent = `${sys.hostname || 'host'} · ${new Date((rep.started_at || 0) * 1000).toLocaleString()}`;
  $('heroTitle').textContent = (sys.cpu && sys.cpu.name) || 'unknown cpu';
  const meta = $('heroMeta');
  meta.innerHTML = '';
  [
    `${sys.os_name || ''} ${sys.os_release || ''} build ${sys.os_build || ''}`,
    `${(sys.cpu && sys.cpu.physical_cores) || '?'} cores / ${(sys.cpu && sys.cpu.logical_cores) || '?'} threads`,
    sys.memory ? `${sys.memory.total_gib} GiB` : null,
    rep.config ? `threads=${rep.config.threads || rep.config.physical_cores || 'auto'}` : null,
    rep.config ? `${rep.config.repeats} repeats` : null,
    rep.config?.mode === 'peak' ? '条件別探索（デバイス間比較不可）' : rep.config?.mode === 'throughput' ? '固定負荷の並列測定' : '単発の遅延',
  ].filter(Boolean).forEach((t) => {
    const s = el('span', {}, t);
    meta.appendChild(s);
  });

  const stats = $('heroStats');
  stats.innerHTML = '';
  const add = (k, v, sub) => {
    const d = el('div', { class: 'stat' });
    d.innerHTML = `<div class="k">${esc(k)}</div><div class="v">${v}${sub ? ` <small>${esc(sub)}</small>` : ''}</div>`;
    stats.appendChild(d);
  };
  add('組み合わせ', fmtInt(good.length), good.length !== (rep.results || []).length ? `of ${(rep.results || []).length}` : '');
  add('devices', fmtInt(new Set(good.map((r) => r.target_id)).size));
  add('duration', fmtDuration(rep.duration_s));
  add('grade', (rep.grade && rep.grade.letter) || '—', rep.grade ? `${rep.grade.score}/100` : '');
  if (best) add('fastest', `${fmt(best.ai_index)}×`, best.kind + ' · ' + (best.vendor || ''));
}

/* --------------------------------------------------------------- charts */
function svgSize(node) {
  const w = node.clientWidth || 720;
  return { w, h: 0 };
}

function renderIndexChart(rep) {
  const host = $('indexChart');
  const rows = ((rep.scores || {}).ai_index || []).filter((r) => r.ai_index);
  if (!rows.length) { host.innerHTML = '<p class="empty">run the suite to populate this chart</p>'; return; }
  const max = Math.max(...rows.map((r) => r.ai_index), 1.0001);
  const rowH = 34, padL = 168, padR = 74, top = 8;
  const h = top + rows.length * rowH + 8;
  const width = 900;
  const s = svg('svg', { viewBox: `0 0 ${width} ${h}`, preserveAspectRatio: 'xMinYMin meet' });

  [1, 2, 4, 8, 16, 32].filter((g) => g <= max * 1.02).forEach((g) => {
    const x = padL + (g / max) * (width - padL - padR);
    s.appendChild(svg('line', { x1: x, y1: top - 2, x2: x, y2: h - 6, class: 'gridline' }));
    s.appendChild(svg('text', { x, y: top - 6, class: 'axis-name', 'text-anchor': 'middle' }, g + '×'));
  });

  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const colour = deviceColor(r);
    const w = Math.max(3, (r.ai_index / max) * (width - padL - padR));
    s.appendChild(svg('text', { x: padL - 12, y: y + 17, class: 'bar-value', 'text-anchor': 'end' }, r.label || r.target_id));
    const defs = svg('defs');
    const id = 'grad-' + i;
    const lg = svg('linearGradient', { id, x1: '0', x2: '1', y1: '0', y2: '0' });
    lg.appendChild(svg('stop', { offset: '0', 'stop-color': colour, 'stop-opacity': '0.45' }));
    lg.appendChild(svg('stop', { offset: '1', 'stop-color': colour, 'stop-opacity': '1' }));
    defs.appendChild(lg);
    s.appendChild(defs);
    const bar = svg('rect', { x: padL, y: y + 5, width: 0, height: 17, rx: 5, fill: `url(#${id})`, class: 'bar' });
    const txt = svg('text', { x: padL + 9, y: y + 18, class: 'bar-value', fill: '#05060c' }, `${fmt(r.ai_index)}×`);
    if (w < 46) { txt.setAttribute('x', padL + w + 8); txt.setAttribute('fill', colour); }
    s.appendChild(bar); s.appendChild(txt);
    requestAnimationFrame(() => {
      bar.style.transition = 'width .8s cubic-bezier(.2,.8,.2,1)';
      requestAnimationFrame(() => bar.setAttribute('width', w));
    });
  });

  host.innerHTML = '';
  host.appendChild(s);
  const base = (rep.scores || {}).baseline;
  if (base) host.appendChild(el('p', { class: 'empty' }, `baseline: ${base.label} (${base.definition}), ${base.workloads} workloads`));
  const common = rep.scores?.comparison_workloads;
  if (common) host.appendChild(el('p', { class: 'empty' }, `総合点: 全比較対象に共通する${common.length}モデル。除外: ${(rep.scores.excluded_workloads || []).join(', ') || 'なし'}。入力グラフの精度と実行精度は異なる場合があります。`));
}

function renderGrade(rep) {
  const host = $('grade');
  const g = rep.grade || {};
  const score = g.score ?? 0;
  const letter = g.letter || '?';
  const colour = score >= 80 ? '#15803d' : score >= 60 ? '#92520b' : '#b42318';
  const r = 36, circ = 2 * Math.PI * r;
  host.innerHTML = '';
  const badge = el('div', { class: 'grade-badge' });
  badge.innerHTML = `
    <div class="grade-ring">
      <svg viewBox="0 0 92 92" width="92" height="92">
        <circle class="ring-bg" cx="46" cy="46" r="${r}" fill="none" stroke-width="8"></circle>
        <circle class="ring-fg" cx="46" cy="46" r="${r}" fill="none" stroke="${colour}" stroke-width="8"
          stroke-dasharray="${circ}" stroke-dashoffset="${circ}"></circle>
      </svg>
      <div class="ring-text" style="color:${colour}">${esc(letter)}</div>
    </div>
    <div class="grade-summary">
      <div style="font-family:var(--mono);font-size:20px;color:${colour}">${fmt(score, 1)}<span style="font-size:12px;color:var(--muted)">/100</span></div>
      <div>fairness and validity checks for this run</div>
    </div>`;
  host.appendChild(badge);
  if (!rep.measurement_contract) {
    host.appendChild(el('p', { class: 'device-path' }, '保存時のスコアとチェックを表示しています。旧方式の正解検証・再現性判定は現行と異なる場合があります。現行方式で再計測してください。'));
  }
  requestAnimationFrame(() => {
    const fg = host.querySelector('.ring-fg');
    if (fg) fg.style.strokeDashoffset = String(circ * (1 - Math.max(0, Math.min(1, score / 100))));
  });

  const checks = el('div', { class: 'checks' });
  gradeChecks(rep).forEach((c) => {
    const row = el('div', { class: 'check' });
    row.innerHTML = `<span class="mark ${esc(c.status)}">${esc(c.status)}</span><span class="name">${esc(c.name)}</span><span class="detail">${esc(c.detail)}</span>`;
    checks.appendChild(row);
  });
  host.appendChild(checks);
}

function renderThroughput(rep) {
  const host = $('throughputChart');
  const legend = $('chartLegend');
  const rows = (rep.results || []).filter((r) => r.ok && Number.isFinite(r.units_per_s) && r.units_per_s > 0);
  if (!rows.length) { host.innerHTML = '<p class="empty">run the suite to populate this chart</p>'; legend.innerHTML = ''; return; }

  const byWorkload = new Map();
  rows.forEach((r) => {
    if (!byWorkload.has(r.workload)) byWorkload.set(r.workload, []);
    byWorkload.get(r.workload).push(r);
  });
  const order = ((rep.workloads || []).map((w) => w.key)).filter((k) => byWorkload.has(k));
  byWorkload.forEach((_, k) => { if (!order.includes(k)) order.push(k); });

  const devices = [...new Set(rows.map((r) => r.target_id))];
  legend.innerHTML = '';
  devices.forEach((id) => {
    const sample = rows.find((r) => r.target_id === id);
    const sp = el('span');
    sp.innerHTML = `<i style="background:${deviceColor(sample)}"></i>${esc(sample.target_label || id)}`;
    legend.appendChild(sp);
  });

  const width = 900;
  const padL = 172, padR = 92;
  const barH = 11, gap = 4, groupPad = 18;
  const maxByUnit = new Map();
  rows.forEach((row) => {
    const unit = row.unit || 'sample';
    maxByUnit.set(unit, Math.max(maxByUnit.get(unit) || 0, row.units_per_s));
  });
  let y = 6;
  const layout = order.map((wl) => {
    const group = byWorkload.get(wl).slice().sort((a, b) => b.units_per_s - a.units_per_s);
    const max = maxByUnit.get(group[0].unit || 'sample');
    const item = { wl, group, max, y, height: group.length * (barH + gap) + groupPad };
    y += item.height;
    return item;
  });
  const height = y + 10;
  const s = svg('svg', { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: 'xMinYMin meet' });
  const unitOf = (wl) => {
    const meta = (rep.workloads || []).find((w) => w.key === wl);
    const u = rows.find((r) => r.workload === wl).unit || 'sample';
    return u + '/s';
  };

  layout.forEach((item) => {
    const labelY = item.y + 12;
    s.appendChild(svg('text', { x: 0, y: labelY, class: 'bar-value' }, item.wl));
    s.appendChild(svg('text', { x: 0, y: labelY + 13, class: 'axis-name' }, unitOf(item.wl)));
    s.appendChild(svg('line', { x1: 0, y1: item.y + item.height - 9, x2: width, y2: item.y + item.height - 9, class: 'gridline' }));

    item.group.forEach((r, i) => {
      const by = item.y + i * (barH + gap) + 4;
      const colour = deviceColor(r);
      const w = Math.max(3, (r.units_per_s / item.max) * (width - padL - padR));
      const bar = svg('rect', { x: padL, y: by, width: 0, height: barH, rx: 4, fill: colour, opacity: '0.92', class: 'bar' });
      const detail = `${r.target_label || r.target_id}\n${item.wl} · ${(r.precision || '').toUpperCase()}\n${fmt(r.units_per_s, 2)} ${unitOf(item.wl)}`;
      bar.appendChild(svg('title', {}, detail));
      bar.setAttribute('aria-label', detail);
      const val = svg('text', { x: padL + w + 8, y: by + barH - 1, class: 'bar-label' },
        `${r.units_per_s >= 1000 ? fmtInt(r.units_per_s) : fmt(r.units_per_s, 2)}`);
      val.appendChild(svg('title', {}, detail));
      const prec = svg('text', { x: width - 4, y: by + barH - 1, class: 'axis-name', 'text-anchor': 'end' }, r.precision || '');
      s.appendChild(bar); s.appendChild(val); s.appendChild(prec);
      requestAnimationFrame(() => {
        bar.style.transition = 'width .7s cubic-bezier(.2,.8,.2,1)';
        requestAnimationFrame(() => bar.setAttribute('width', w));
      });
    });
  });
  host.innerHTML = '';
  host.appendChild(s);
}

function renderRadar(rep) {
  const host = $('radarChart');
  const baseline = (rep.scores || {}).baseline;
  const rows = (rep.results || []).filter((r) => r.ok && r.units_per_s > 0);
  if (!rows.length || !baseline) { host.innerHTML = '<p class="empty">run the suite to populate this chart</p>'; return; }

  const baseRows = rows.filter((r) => r.target_id === baseline.target_id);
  if (!baseRows.length) { host.innerHTML = '<p class="empty">no CPU baseline in this run</p>'; return; }
  const baseByWl = Object.fromEntries(baseRows.map((r) => [r.workload, r.units_per_s]));

  const families = new Map();
  (rep.workloads || []).forEach((w) => {
    if (!baseByWl[w.key]) return;
    if (!families.has(w.category)) families.set(w.category, []);
    families.get(w.category).push(w.key);
  });
  const axes = [...families.keys()];
  if (axes.length < 3) { host.innerHTML = '<p class="empty">not enough workload families for a profile chart</p>'; return; }

  const targets = [...new Set(rows.map((r) => r.target_id))];
  const series = targets.map((id) => {
    const sample = rows.find((r) => r.target_id === id);
    const values = axes.map((fam) => {
      const ratios = families.get(fam)
        .map((wl) => {
          const row = rows.find((r) => r.target_id === id && r.workload === wl);
          return row && baseByWl[wl] ? row.units_per_s / baseByWl[wl] : null;
        })
        .filter((v) => v && v > 0);
      if (!ratios.length) return 0.001;
      return Math.exp(ratios.reduce((a, v) => a + Math.log(v), 0) / ratios.length);
    });
    return { id, label: sample.target_label || id, colour: deviceColor(sample), values };
  });

  const maxVal = Math.max(1.05, ...series.flatMap((s) => s.values));
  const size = 520, cx = size / 2, cy = size / 2 + 8, R = size * 0.34;
  const svgEl = svg('svg', { viewBox: `-30 -20 ${size + 60} ${size + 50}`, preserveAspectRatio: 'xMinYMin meet', style: 'max-width:640px;margin:0 auto;display:block' });
  const n = axes.length;
  const point = (i, v) => {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const rr = R * (v / maxVal);
    return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)];
  };
  for (let ring = 1; ring <= 4; ring++) {
    const pts = axes.map((_, i) => point(i, (maxVal * ring) / 4).join(',')).join(' ');
    svgEl.appendChild(svg('polygon', { points: pts, class: 'radar-grid' }));
  }
  axes.forEach((ax, i) => {
    const [px, py] = point(i, maxVal);
    svgEl.appendChild(svg('line', { x1: cx, y1: cy, x2: px, y2: py, class: 'radar-axis' }));
    const lx = cx + (px - cx) * 1.16, ly = cy + (py - cy) * 1.16;
    const t = svg('text', { x: lx, y: ly + 4, class: 'radar-label', 'text-anchor': lx < cx - 4 ? 'end' : lx > cx + 4 ? 'start' : 'middle' }, ax);
    svgEl.appendChild(t);
  });
  series.forEach((s) => {
    const pts = s.values.map((v, i) => point(i, v).join(',')).join(' ');
    const shape = svg('polygon', { points: pts, class: 'radar-shape', fill: s.colour, stroke: s.colour });
    shape.appendChild(svg('title', {}, `${s.label}: ${s.values.map((v, i) => `${axes[i]} ${v.toFixed(2)}×`).join(', ')}`));
    svgEl.appendChild(shape);
    s.values.forEach((v, i) => {
      const [px, py] = point(i, v);
      svgEl.appendChild(svg('circle', { cx: px, cy: py, r: 3, fill: s.colour }));
    });
  });
  host.innerHTML = '';
  host.appendChild(svgEl);
  const legend = el('div', { class: 'legend', style: 'justify-content:center' });
  series.forEach((s) => {
    const sp = el('span');
    sp.innerHTML = `<i style="background:${s.colour}"></i>${esc(s.label)}`;
    legend.appendChild(sp);
  });
  host.appendChild(legend);
  host.appendChild(el('p', { class: 'empty', style: 'text-align:center' }, 'values are acceleration vs this machine\u2019s CPU; 1.0× means parity'));
}

function renderTable(rep) {
  const table = $('resultTable');
  const rows = (rep.results || []).slice().sort((a, b) => (a.workload.localeCompare(b.workload)) || ((b.units_per_s || 0) - (a.units_per_s || 0)));
  const cols = ['device', 'workload', 'GEMM M×K×N', '指定精度', '実行情報', 'units/s', 'median ms', 'p90 ms', 'CV', '推論性能', 'E2E性能', '入力転送・準備 ms', '推論・同期 ms', '出力転送・取得 ms', '成功回数', 'host CPU s/inf', 'W', 'J/inf', 'compile ms', 'spread'];
  table.innerHTML = '';
  const thead = el('thead');
  const tr = el('tr');
  cols.forEach((c) => tr.appendChild(el('th', {}, c)));
  thead.appendChild(tr);
  table.appendChild(thead);
  const tb = el('tbody');

  rows.forEach((r) => {
    const tr = el('tr');
    const colour = deviceColor(r);
    const lat = r.latency || {};
    const rep0 = r.repeats || {};
    const warnings = [];
    if ((r.session || {}).fallback_used) warnings.push('provider fallback');
    if (lat.cv > 0.05) warnings.push('noisy');
    if ((rep0.spread_pct || 0) > 12) warnings.push('unstable');

    const cells = [];
    const dev = el('td');
    dev.innerHTML = `<span class="dev-tag"><i class="dev-dot" style="background:${colour}"></i>${esc(r.target_label || r.target_id)}${warnings.length ? ` <span class="flag" title="${esc(warnings.join(', '))}">▲</span>` : ''}</span>`;
    cells.push(dev);
    cells.push(el('td', {}, r.workload));
    cells.push(el('td', { title: `1 MAC = ${r.operation_convention?.operations_per_mac || '未記録'} ops` },
      gemmShape(r)?.join('×') || (r.workload?.startsWith('precision_gemm') ? '未記録' : '—')));
    cells.push(el('td', { title: `${r.session?.quantization || ''}; runtime precision: ${JSON.stringify(r.session?.execution_precisions || {})}; parallel requests: ${(r.concurrent_requests_per_repeat || [1]).join('/')}`, html: `<span class="prec">${esc(r.precision || '')}</span>` }));
    cells.push(el('td', { title: (r.compute_precision_sources || []).join('; ') }, formatExecutionPrecision(r)));
    if (!r.ok) {
      const td = el('td', { class: 'muted' }, r.phase === 'precision' ? '再計測が必要' : 'failed');
      td.colSpan = cols.length - 5;
      td.title = r.error || '';
      const diagnostic = el('details');
      diagnostic.appendChild(el('summary', {}, '失敗の詳細'));
      diagnostic.appendChild(el('pre', { style: 'white-space:pre-wrap;overflow-wrap:anywhere' }, r.error || '詳細は記録されていません'));
      td.appendChild(diagnostic);
      cells.push(td);
    } else {
      cells.push(el('td', { class: 'num' }, r.units_per_s >= 1000 ? fmtInt(r.units_per_s) : fmt(r.units_per_s)));
      cells.push(el('td', { class: 'num' }, fmt(r.latency_ms, 3)));
      cells.push(el('td', { class: 'muted' }, fmt(lat.p90_ms, 3)));
      cells.push(el('td', { class: (lat.cv_trimmed ?? lat.cv) > 0.05 ? 'flag' : 'muted', title: `trimmed CV ${lat.cv_trimmed !== undefined ? (lat.cv_trimmed * 100).toFixed(2) + '%' : 'n/a'}, ${lat.count || 0} iterations` }, lat.cv !== undefined ? (lat.cv * 100).toFixed(1) + '%' : '—'));
      cells.push(el('td', { class: 'num', title: '推論区間の実測時間に基づく性能。並列測定では合計スループット。' }, formatInferenceRate(r)));
      const e2eAvailable = r.measurement_scope !== 'device_resident'
        || r.timing_breakdown?.available && Number.isFinite(r.timing_breakdown.e2e_ms) && r.timing_breakdown.e2e_ms > 0;
      cells.push(el('td', { class: 'num', title: '入力から出力までの全体区間の実測時間に基づく性能。' }, e2eAvailable ? formatE2ERate(r) : '未取得'));
      const timing = r.timing_breakdown || {};
      for (const phase of ['h2d_ms', 'compute_ms', 'd2h_ms']) {
        cells.push(el('td', { class: 'num', title: timing.note || '分離時間は未取得' }, timing.available ? fmt(timing[phase], 3) : '—'));
      }
      cells.push(el('td', { class: 'num' }, `${r.repeats_ok || 0}/${r.repeats_requested || 0}`));
      cells.push(el('td', { class: 'muted' }, fmt(r.host_cpu_seconds_per_inference, 4)));
      cells.push(el('td', { class: 'muted' }, r.power && r.power.average_watts ? fmt(r.power.average_watts, 1) : '—'));
      cells.push(el('td', { class: 'muted' }, r.energy_per_inference_j ? fmt(r.energy_per_inference_j, 3) : '—'));
      cells.push(el('td', { class: 'muted' }, fmt(r.compile_ms, 0)));
      cells.push(el('td', { class: 'muted' }, r.repeats_ok >= 2 && Number.isFinite(rep0.spread_pct) ? fmt(rep0.spread_pct, 1) + '%' : '判定不可'));
    }
    cells.forEach((c) => tr.appendChild(c));
    tb.appendChild(tr);
  });
  table.appendChild(tb);
}

function renderProvenance(rep) {
  const host = $('provenance');
  host.innerHTML = '';
  const sys = rep.system || {};
  const cfg = rep.config || {};
  const disc = rep.discovery || {};

  const block = (title, pairs, wide) => {
    const d = el('div', { class: 'prov' + (wide ? ' wide' : '') });
    d.appendChild(el('h3', {}, title));
    const dl = el('dl');
    pairs.filter(Boolean).forEach(([k, v]) => {
      dl.appendChild(el('dt', {}, k));
      dl.appendChild(el('dd', {}, v === null || v === undefined ? '—' : String(v)));
    });
    if (!dl.childElementCount) dl.appendChild(el('dd', { class: 'muted' }, 'none'));
    d.appendChild(dl);
    host.appendChild(d);
    return d;
  };

  block('machine', [
    ['host', sys.hostname],
    ['cpu', (sys.cpu || {}).name],
    ['cores', `${(sys.cpu || {}).physical_cores}C / ${(sys.cpu || {}).logical_cores}T`],
    ['memory', sys.memory ? `${sys.memory.total_gib} GiB` + (sys.memory.speed_mts ? ` @ ${sys.memory.speed_mts} MT/s` : '') : null],
    ['RAMコピー帯域（読出＋書込）', sys.memory?.bandwidth?.available ? `${fmt(sys.memory.bandwidth.gb_per_s, 1)} GB/s（実測・NPU専用帯域ではありません）` : '未計測'],
    ['帯域測定条件', sys.memory?.bandwidth?.available ? `${fmt(sys.memory.bandwidth.buffer_bytes / 1024**2, 0)} MiB × 2バッファ・${sys.memory.bandwidth.samples}回の中央値` : null],
    ['メモリデータ幅', (sys.memory?.modules || []).map((m) => `${m.slot || 'module'}: ${m.data_width_bits ? m.data_width_bits + ' bit' : '不明'}`).join(' / ') || '不明'],
    ['理論メモリ帯域', 'チャネル構成を確定できないため算出しません'],
    ['os', `${sys.os_name} ${sys.os_release} (${sys.os_version})`],
    ['python', sys.python],
  ]).classList.add('machine-block');
  block('run policy', [
    ['warm-up', cfg.warmup],
    ['min iterations', cfg.min_iterations],
    ['min seconds', cfg.min_seconds],
    ['max seconds', cfg.max_seconds],
    ['backend', cfg.backend || 'all'],
    ['measurement set', cfg.suite || 'standard'],
    ['mode', cfg.mode || 'latency'],
    ['parallel requests', cfg.mode === 'latency' ? 1 : (cfg.requests || 'auto; 採用条件は各測定結果に記録')],
    ['repeats', cfg.repeats],
    ['cooldown', cfg.cooldown_seconds + 's'],
    ['cpu threads', cfg.threads || cfg.physical_cores],
    ['thread affinity', String(cfg.thread_affinity)],
    ['input seed', cfg.input_seed],
    ['correctness check', String(cfg.check_correctness)],
  ]);

  const runtimes = Object.entries(disc.runtimes || {}).map(([k, v]) => [k, String(v).split('-')[0]]);
  const interps = (disc.interpreters || []).filter((i) => i.available);
  interps.forEach((i) => {
    const provs = (i.providers || []).filter((p) => p !== 'AzureExecutionProvider').map((p) => p.replace('ExecutionProvider', ''));
    runtimes.push([i.primary ? 'main providers' : `${i.label || 'venv'} providers`, provs.join(', ') || 'none']);
  });
  block('runtimes', runtimes);

  const hashes = new Map();
  (rep.results || []).forEach((r) => { if (r.model_sha256) hashes.set(r.workload, r.model_sha256); });
  const hashBlock = block('model fingerprints', [...hashes.entries()].map(([k, v]) => [k, v.slice(0, 32) + '…']), true);
  hashBlock.classList.add('hash-block');

  const warns = (rep.warnings || []).filter(Boolean);
  const wd = el('div', { class: 'prov wide' });
  wd.appendChild(el('h3', {}, `warnings (${warns.length})`));
  if (!warns.length) {
    wd.appendChild(el('p', { class: 'empty' }, 'none — nothing was suppressed or worked around'));
  } else {
    const ul = el('ul', { class: 'warn-list' });
    warns.forEach((w) => ul.appendChild(el('li', {}, w)));
    wd.appendChild(ul);
  }
  host.appendChild(wd);
}

/* --------------------------------------------------------------- live run */
function renderLive() {
  const run = state.run;
  document.body.classList.toggle('running', !!run?.active);
  const panel = $('livePanel');
  // Keep the log on screen after the run ends so it can actually be read; it is
  // dismissed when a new run starts or the user opens a different result.
  if (!run || (!run.active && !run.finished) || (state.hideLive && !run.active)) { panel.hidden = true; return; }
  panel.hidden = false;
  panel.classList.toggle('done', !run.active);

  const head = panel.querySelector('h2');
  if (head) head.textContent = run.active ? '計測中' : '計測終了';
  const stop = $('stopBtn');
  if (stop) stop.hidden = !run.active;

  const p = run.progress || {};
  const pct = p.total ? Math.round((p.index / p.total) * 100) : 0;
  $('progressFill').style.width = (run.active ? pct : 100) + '%';
  $('progressText').textContent = run.active
    ? `${p.index || 0} / ${p.total || '?'} cells · ${fmtDuration(run.elapsed_s)}`
    : `${p.total || 0} / ${p.total || 0} cells · ${fmtDuration(run.elapsed_s)}${run.error ? ' · failed' : ''}`;
  $('liveElapsed').textContent = run.error ? 'error' : `${run.active ? pct : 100}%`;

  const log = $('liveLog');
  log.innerHTML = '';
  (run.events || []).slice(-160).forEach((e) => {
    let cls = 'info', txt = '';
    const t = new Date((e.t || 0) * 1000).toLocaleTimeString();
    if (e.event === 'verify') {
      cls = e.ok ? 'ok' : 'fail';
      txt = `verify ${e.target} — ${e.ok ? 'ok' : String(e.error || '')}`;
    } else if (e.event === 'tuning') {
      cls = e.ok ? 'ok' : 'fail';
      txt = `探索 ${e.target} · ${e.workload} · 行列 ${e.size}×${e.size} / ${e.requests}並列 / ${e.threads}スレッド枠 · ${fmt((e.flops_per_s || 0) / 1e12, 3)} T-op/s`;
    } else if (e.event === 'cell') {
      cls = e.ok ? 'ok' : 'fail';
      txt = e.ok
        ? `${e.target_label} · ${e.workload} · ${fmt(e.units_per_s)} units/s · ${fmt(e.median_ms, 2)} ms`
        : `${e.target_label} · ${e.workload} — ${String(e.error || '')}`;
    } else if (e.event === 'error') {
      cls = 'fail'; txt = e.error;
    } else { return; }
    const line = el('div', { class: 'log-line ' + cls });
    if (cls === 'fail') line.style.whiteSpace = 'pre-wrap';
    line.innerHTML = `<span class="t">${t}</span><span>${esc(txt)}</span>`;
    log.appendChild(line);
  });
  log.scrollTop = log.scrollHeight;
}

function renderFromLive() {
  const run = state.run;
  if (run && run.report) {
    renderHero({ system: (state.catalog || {}).system, results: run.report.results, grade: run.report.grade, scores: {}, config: {}, duration_s: run.elapsed_s });
  }
}

/* --------------------------------------------------------------- run modal */
function syncRunFields() {
  const suite = $('optSuite').value;
  $('workloadChecks').hidden = suite !== 'custom';
  $('modeField').hidden = ['peak', 'npu_peak'].includes(suite);
  $('requestsField').hidden = suite === 'peak' || $('optMode').value !== 'throughput';
  $('optGemmSize').disabled = suite === 'peak';
}
function buildRunModal() {
  const cat = state.catalog;
  if (!cat) return;
  const tWrap = $('targetChecks');
  const wWrap = $('workloadChecks');
  const targetSignature = JSON.stringify((cat.discovery?.targets || []).map((t) => [t.id, t.available, t.label, t.backend]));
  if (tWrap.childElementCount && tWrap.dataset.targets === targetSignature) return;
  tWrap.dataset.targets = targetSignature;
  const backendSelect = $('optBackend');
  backendSelect.innerHTML = '';
  backendSelect.appendChild(el('option', { value: 'all' }, 'すべてのバックエンド'));
  const backendNames = { ort: 'ONNX Runtime', openvino: 'OpenVINO', winml: 'Windows ML', tensorrt: 'NVIDIA TensorRT' };
  const availableBackends = new Set((cat.discovery?.targets || []).filter((t) => t.available !== false).map((t) => t.backend));
  Object.entries(backendNames).forEach(([key, name]) => {
    if (availableBackends.has(key)) backendSelect.appendChild(el('option', { value: key }, name));
  });
  backendSelect.onchange = () => {
    tWrap.querySelectorAll('label').forEach((label) => {
      const visible = backendSelect.value === 'all' || label.dataset.backend === backendSelect.value;
      label.hidden = !visible;
      label.querySelector('input').disabled = !visible;
    });
  };
  tWrap.innerHTML = '';
  ((cat.discovery || {}).targets || []).filter((t) => t.available !== false).forEach((t) => {
    const l = el('label', { class: 'sel', style: `--accent:${deviceColor(t)}` });
    l.dataset.backend = t.backend;
    l.innerHTML = `<input type="checkbox" value="${esc(t.id)}" checked /><span>${esc(t.label)}</span><span style="color:var(--dim);font-size:11px">${esc(t.kind)}</span>`;
    l.querySelector('input').addEventListener('change', (e) => l.classList.toggle('sel', e.target.checked));
    tWrap.appendChild(l);
  });
  if (!tWrap.childElementCount) tWrap.appendChild(el('p', { class: 'empty' }, 'no runnable targets were detected on this machine'));

  const suiteSelect = $('optSuite');
  suiteSelect.innerHTML = '';
  Object.entries(cat.workload_sets || {}).forEach(([key, suite]) => {
    if (key === 'precision') return;
    suiteSelect.appendChild(el('option', { value: key }, suite.label));
  });
  suiteSelect.appendChild(el('option', { value: 'custom' }, '個別指定'));
  suiteSelect.value = 'standard';
  $('optMode').value = 'latency';
  $('optMode').onchange = syncRunFields;
  suiteSelect.onchange = () => {
    const suite = cat.workload_sets?.[suiteSelect.value];
    $('optMode').value = suiteSelect.value === 'npu_peak' ? 'throughput' : 'latency';
    $('optRequests').value = '0';
    syncRunFields();
    if (!suite) return;
    wWrap.querySelectorAll('input').forEach((input) => {
      input.checked = suite.workloads.includes(input.value);
      input.parentElement.classList.toggle('sel', input.checked);
    });
  };
  wWrap.innerHTML = '';
  const defaults = cat.workload_sets?.standard?.workloads || ['precision_gemm_fp16', 'precision_gemm_int8'];
  (cat.workloads || []).forEach((w) => {
    if (w.key === 'dense_static' || (w.key.endsWith('_int8') && w.key !== 'dense_static_int8' && w.category !== 'precision')) return;
    const on = defaults.includes(w.key);
    const l = el('label', { class: on ? 'sel' : '', style: '--accent:#2563eb' });
    l.innerHTML = `<input type="checkbox" value="${esc(w.key)}" ${on ? 'checked' : ''} /><span>${esc(w.label || w.key)}</span><span style="color:var(--dim);font-size:11px">${esc(w.category)}</span>`;
    l.querySelector('input').addEventListener('change', (e) => {
      l.classList.toggle('sel', e.target.checked);
      suiteSelect.value = 'custom';
      syncRunFields();
    });
    l.title = w.blurb || '';
    wWrap.appendChild(l);
  });
  syncRunFields();
}

function openModal() {
  buildRunModal();
  const t = $('targetChecks');
  if (t.childElementCount && !t.querySelector('input:not(:disabled):checked')) {
    t.querySelectorAll('input:not(:disabled)').forEach((i) => { i.checked = true; i.parentElement.classList.add('sel'); });
  }
  $('runModal').hidden = false;
}

async function startRun() {
  const targets = [...$('targetChecks').querySelectorAll('input:not(:disabled):checked')].map((i) => i.value);
  const workloads = ['standard', 'precision', 'peak'].includes($('optSuite').value)
    ? ['precision_gemm_fp16', 'precision_gemm_int8']
    : [...$('workloadChecks').querySelectorAll('input:checked')].map((i) => i.value);
  if (!targets.length) { toast('pick at least one device', true); return; }
  if (!workloads.length) { toast('pick at least one workload', true); return; }
  const body = {
    targets,
    backend: $('optBackend').value,
    workloads,
    suite: $('optSuite').value,
    mode: $('optSuite').value === 'peak' ? 'peak' : $('optMode').value,
    requests: $('optSuite').value === 'peak' || $('optMode').value !== 'throughput' ? 0 : Number($('optRequests').value || 0),
    gemm_size: Number($('optGemmSize').value || 4096),
    repeats: Number($('optRepeats').value || 3),
    threads: $('optThreads').value ? Number($('optThreads').value) : null,
    include_int8: true,
    check_correctness: true,
    measure_power: $('optPower').checked,
    allow_battery: $('optBattery').checked,
    verify: true,
  };
  const res = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const out = await res.json();
  if (!out.ok) { toast(out.error || 'could not start', true); return; }
  $('runModal').hidden = true;
  state.hideLive = false;
  state.followRun = true;
  toast('run started — the log appears below');
  state.activeFile = null;
  schedule(300);
}

/* ------------------------------------------------------------------ wire */
async function copyLog() {
  const text = [...$('liveLog').children].map((line) => [...line.children]
    .map((part) => part.textContent).join(' ')).join('\n');
  if (!text) { toast('コピーするログがありません'); return; }
  try {
    await navigator.clipboard.writeText(text);
    toast('ログをコピーしました');
  } catch (err) {
    location.hash = 'activity';
    renderRoute();
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents($('liveLog'));
    selection.removeAllRanges();
    selection.addRange(range);
    toast('コピーできませんでした。実行ログを開いてCtrl+Cでコピーしてください', true);
  }
}
$('copyLogBtn').onclick = copyLog;
$('newRunBtn').onclick = openModal;
$('cancelRun').onclick = () => { $('runModal').hidden = true; };
$('confirmRun').onclick = startRun;
$('refreshBtn').onclick = async () => {
  await fetch('/api/catalog');
  state.activeFile = null;
  await loadState();
  toast('re-read this machine');
};
$('stopBtn').onclick = async () => {
  await fetch('/api/stop', { method: 'POST' });
  toast('stop requested — the current cell will finish first');
  schedule(500);
};
$('runModal').addEventListener('click', (e) => { if (e.target.id === 'runModal') $('runModal').hidden = true; });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('runModal').hidden = true; });

window.addEventListener('hashchange', () => {
  renderRoute();
  window.scrollTo(0, 0);
});
renderRoute();
renderOverview();
loadState();
