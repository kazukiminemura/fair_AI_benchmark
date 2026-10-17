/* UI and orchestration. Computation runs on the device that opened this page. */
'use strict';
const bench = FairbenchBrowser;
const $ = (id) => document.getElementById(id);
const STORAGE_KEY = 'fairbench.browser.last.v2';
let report = null, running = null;
const pause = () => new Promise((resolve) => setTimeout(resolve, 0));
const check = (signal) => { if (signal.aborted) throw new DOMException('中止しました', 'AbortError'); };
const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';

function render(data) {
  for (const kind of ['cpu', 'gpu']) {
    const result = data?.results?.[kind];
    $(kind + 'Value').textContent = result?.ok ? `${fmt(result.gflops / 1000, 5)} TFLOPS` : result ? '計測できません' : '未計測';
    $(kind + 'Detail').textContent = result?.ok
      ? `${fmt(result.mean_ms)} ms/回 · 3回の中央値${kind === 'gpu' ? ' · ' + result.size + '角・' + result.batch_size + '連続' : ''}${result.software_renderer ? ' · ソフトウェア描画' : ''}`
      : result?.error || (kind === 'cpu' ? 'FP32行列計算・1スレッド' : '対応状況は測定時に確認します');
  }
  const rows = data ? [
    ['測定方式', 'v2・GPU内の連続FP32行列計算（v1・PC用ONNX測定とは別方式）'],
    ['CPU行列サイズ', `${data.size} × ${data.size}`],
    ['GPU行列サイズ', data.results.gpu?.size ? `${data.results.gpu.size} × ${data.results.gpu.size}` : data.gpu_size],
    ['GPU連続実行数', data.results.gpu?.batch_size || '未計測'],
    ['GPU測定区間', 'GPU内の連続演算と完了待ち。準備・読戻し・正解照合は含めません'],
    ['GPU負荷の探索', (data.results.gpu?.trials || []).map((t) => `${t.size}角: ${t.ok ? fmt(t.gflops / 1000, 5) + ' TFLOPS' : '失敗'}`).join(' / ') || '未計測'],
    ['入力シード', data.seed],
    ['測定時間', '各デバイス3回・各1秒以上。準備・ウォームアップを除外'],
    ['CPU', 'JavaScript・Web Worker・1スレッド'],
    ['GPU', data.results.gpu?.renderer || data.results.gpu?.error || '未計測'],
    ['GPU正解照合', data.results.gpu?.correctness?.valid ? `CPU参照計算と一致（相対L2誤差 ${data.results.gpu.correctness.relative_l2_error.toExponential(2)}）` : '未確認'],
    ['メモリコピー帯域', data.memory?.available ? `${fmt(data.memory.gb_per_s)} GB/s（読出し＋書込み・キャッシュの影響を含む）` : '取得できません'],
    ['コピーの条件', data.memory?.available ? `${data.memory.buffer_bytes / 1024 ** 2} MiB × 2バッファ・${data.memory.samples_gb_per_s.length}回の中央値` : '未計測'],
    ['メモリデータ幅', 'ブラウザから取得できません'],
    ['ブラウザ', data.environment.user_agent],
    ['日時', new Date(data.started_at).toLocaleString()],
  ] : [];
  $('conditions').replaceChildren();
  for (const [key, value] of rows) {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = key; dd.textContent = value;
    $('conditions').append(dt, dd);
  }
  $('download').disabled = !data || data.status !== 'complete';
}

function measureCpu(size, signal) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('/mobile-bench.js');
    const finish = (error, result) => {
      clearTimeout(timeout); signal.removeEventListener('abort', abort); worker.terminate();
      error ? reject(error) : resolve(result);
    };
    const abort = () => finish(new DOMException('中止しました', 'AbortError'));
    const timeout = setTimeout(() => finish(new Error('CPU測定が制限時間を超えました')), 90000);
    signal.addEventListener('abort', abort, { once: true });
    worker.onerror = () => finish(new Error('CPU測定を開始できません。通常のブラウザでこのページを開いてください。'));
    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') {
        $('progress').value = data.repeat;
        $('status').textContent = `CPUを測定しています（${data.repeat}/3回完了）`;
      } else if (data.type === 'error') finish(new Error(data.error));
      else if (data.type === 'done') finish(null, data);
    };
    worker.postMessage({ size });
  });
}

$('start').onclick = async () => {
  if (running) return;
  if (document.hidden) { $('status').textContent = '画面を開いた状態で計測を開始してください。'; return; }
  const controller = new AbortController(), signal = controller.signal;
  running = controller;
  $('start').disabled = true; $('size').disabled = true; $('gpuSize').disabled = true; $('stop').disabled = false;
  $('progress').value = 0;
  report = { schema: bench.VERSION, status: 'running', started_at: new Date().toISOString(),
    size: Number($('size').value), gpu_size: $('gpuSize').value, seed: 1337, results: {},
    environment: { user_agent: navigator.userAgent, hardware_concurrency: navigator.hardwareConcurrency || null,
      device_memory_gib_approx: navigator.deviceMemory || null } };
  render(report);
  $('status').textContent = 'CPUを準備しています。この端末で計算します。';
  try {
    const measured = await measureCpu(report.size, signal);
    check(signal);
    report.results.cpu = { ok: true, ...measured.result };
    report.memory = measured.memory;
    render(report);
    $('status').textContent = 'GPUの対応状況と計算結果を確認しています。';
    await pause(); check(signal);
    try { report.results.gpu = await bench.measureGpu(report.gpu_size, signal, (message, progress) => {
      $('status').textContent = message; if (progress !== undefined) $('progress').value = progress;
    }); }
    catch (error) {
      if (error.name === 'AbortError') throw error;
      report.results.gpu = { ok: false, error: error.message };
    }
    check(signal);
    report.status = 'complete'; report.finished_at = new Date().toISOString();
    report.results.npu = { ok: false, error: 'This browser benchmark does not select or measure an NPU' };
    render(report); $('progress').value = 6;
    $('status').textContent = report.results.gpu.ok ? '測定が完了しました。結果はこの端末のものです。' : 'CPU測定が完了しました。GPUはこの環境では計測できませんでした。';
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(report)); }
    catch { $('status').textContent += ' ブラウザ内に保存できないため、JSONで保存してください。'; }
  } catch (error) {
    report.status = error.name === 'AbortError' ? 'cancelled' : 'error';
    render(report);
    $('status').textContent = error.name === 'AbortError' ? '測定を中止しました。画面を開いた状態で再実行してください。' : error.message;
  } finally {
    running = null; $('start').disabled = false; $('size').disabled = false; $('gpuSize').disabled = false; $('stop').disabled = true;
  }
};
$('stop').onclick = () => running?.abort();
document.addEventListener('visibilitychange', () => { if (document.hidden) running?.abort(); });
$('download').onclick = () => {
  if (report?.status !== 'complete') return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url;
  link.download = `fairbench-browser-${report.started_at.replace(/[:.]/g, '-')}.json`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
try {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
  if (saved?.schema === bench.VERSION && saved.status === 'complete' && [128, 256].includes(saved.size) && saved.results?.cpu?.ok && saved.environment) {
    report = saved; render(report); $('size').value = String(saved.size); $('gpuSize').value = String(saved.gpu_size || 'auto');
    $('status').textContent = 'このブラウザに保存した前回の結果です。再計測できます。';
  }
} catch { /* Storage can be disabled; live measurements and downloads still work. */ }
