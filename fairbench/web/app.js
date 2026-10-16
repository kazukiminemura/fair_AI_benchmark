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
function formatComputeRate(row) {
  if (!Number.isFinite(row.flops_per_s)) return '—';
  return row.kind === 'npu'
    ? `${fmt(row.flops_per_s / 1e12, 3)} TOPS`
    : `${fmt(row.flops_per_s / 1e9, 1)} GFLOPS`;
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

  if (!state.activeFile) {
    // First paint: show what the last run produced, or the newest file.
    const first = new URLSearchParams(location.search).get('run') || runFile || newest;
    if (first) await openResult(first);
  } else if (state.followRun && state.run && !state.run.active && runFile) {
    // A run we started ourselves just finished - show its report once, then
    // stop following so the user's own selection is never overridden.
    state.followRun = false;
    await openResult(runFile);
  }
  schedule(state.run && state.run.active ? 1000 : 3000);
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
  } catch (err) { toast('could not load ' + file, true); return; }
  renderRunList();
  renderAll();
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
      <div class="r-sub"><span>${r.targets || 0} cells</span><span>${fmtDuration(r.duration_s)}</span><span>${ago(r.saved_at)}</span></div>`;
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
  const checks = rep?.grade?.checks || [];
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
  const indices = (rep?.scores?.ai_index || []).filter((r) => Number.isFinite(r.ai_index) && r.ai_index > 0);
  const baselineId = rep?.scores?.baseline?.target_id;
  const maxIndex = Math.max(1, ...indices.filter((r) => r.kind !== 'cpu' || r.target_id === baselineId).map((r) => r.ai_index));
  const host = $('overviewDevices');
  host.replaceChildren();
  for (const [kind, name] of [['cpu', 'CPU'], ['gpu', 'GPU'], ['npu', 'NPU']]) {
    const measured = rows.filter((r) => r.kind === kind);
    const candidates = indices.filter((r) => r.kind === kind).sort((a, b) => b.ai_index - a.ai_index);
    const best = (kind === 'cpu' && candidates.find((r) => r.target_id === baselineId)) || candidates[0];
    const card = el('article', { class: 'device-card' + (measured.length ? '' : ' unmeasured'), style: `--accent:${KIND_COLOR[kind]}` });
    const title = el('div', { class: 'device-card-heading' });
    title.append(el('h3', {}, name), el('span', {}, measured.length ? '計測済み' : '未計測'));
    card.appendChild(title);
    card.appendChild(el('p', { class: 'device-role' }, { cpu: '基本の処理', gpu: '画像・並列処理', npu: 'AI専用の処理' }[kind]));
    const value = el('div', { class: 'device-index' }, best ? fmt(best.ai_index) : measured.length ? '比較なし' : '未計測');
    if (best) value.appendChild(el('small', {}, '倍'));
    card.appendChild(value);
    const meter = el('div', { class: 'device-meter', 'aria-hidden': 'true' });
    if (best) meter.appendChild(el('span', { style: `width:${100 * best.ai_index / maxIndex}%` }));
    card.appendChild(meter);
    const definition = best ? (best.target_id === baselineId ? '比較の基準となるCPUです' : 'CPUの基準値に対する総合倍率')
      : measured.length ? 'CPUの比較基準がありません' : '今回の結果には含まれていません';
    card.appendChild(el('p', { class: 'device-definition' }, definition));
    const target = (rep?.targets || []).find((t) => t.id === best?.target_id);
    card.appendChild(el('p', { class: 'device-path' }, best ? (target?.detail || best.label || best.target_id) : '機器の有無・設定は詳細で確認できます'));
    card.appendChild(el('a', { class: 'text-link', href: measured.length ? '#performance' : '#devices' },
      best ? `${best.workloads}項目の測定データを見る →` : measured.length ? '測定値を見る →' : '対応状況を見る →'));
    host.appendChild(card);
  }
  const total = (rep?.results || []).length;
  $('overviewCoverage').textContent = rep
    ? `${rows.length}/${total}件成功 · 所要 ${fmtDuration(rep.duration_s)}。CPUは基準値、GPU・NPUは最高倍率。同じテスト項目で比較してください。`
    : '機器ごとの倍率と、計測結果の注意点をここで確認できます。';
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
    rep.config?.mode === 'throughput' ? '最大スループット（並列実行）' : '単発の遅延',
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
  add('cells', fmtInt(good.length), good.length !== (rep.results || []).length ? `of ${(rep.results || []).length}` : '');
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
  requestAnimationFrame(() => {
    const fg = host.querySelector('.ring-fg');
    if (fg) fg.style.strokeDashoffset = String(circ * (1 - Math.max(0, Math.min(1, score / 100))));
  });

  const checks = el('div', { class: 'checks' });
  (g.checks || []).forEach((c) => {
    const row = el('div', { class: 'check' });
    row.innerHTML = `<span class="mark ${esc(c.status)}">${esc(c.status)}</span><span class="name">${esc(c.name)}</span><span class="detail">${esc(c.detail)}</span>`;
    checks.appendChild(row);
  });
  host.appendChild(checks);
}

function renderThroughput(rep) {
  const host = $('throughputChart');
  const legend = $('chartLegend');
  const rows = (rep.results || []).filter((r) => r.ok && r.units_per_s > 0);
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
  let y = 6;
  const layout = order.map((wl) => {
    const group = byWorkload.get(wl).slice().sort((a, b) => b.units_per_s - a.units_per_s);
    const max = group[0].units_per_s;
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
      const val = svg('text', { x: padL + w + 8, y: by + barH - 1, class: 'bar-label' },
        `${r.units_per_s >= 1000 ? fmtInt(r.units_per_s) : fmt(r.units_per_s, 2)}`);
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
  const cols = ['device', 'workload', 'prec', 'units/s', 'median ms', 'p90 ms', 'CV', '演算性能', 'host CPU s/inf', 'W', 'J/inf', 'compile ms', 'spread'];
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
    cells.push(el('td', { html: `<span class="prec">${esc(r.precision || '')}</span>` }));
    if (!r.ok) {
      const td = el('td', { class: 'muted' }, 'failed');
      td.colSpan = cols.length - 3;
      td.title = r.error || '';
      cells.push(td);
    } else {
      cells.push(el('td', { class: 'num' }, r.units_per_s >= 1000 ? fmtInt(r.units_per_s) : fmt(r.units_per_s)));
      cells.push(el('td', { class: 'num' }, fmt(r.latency_ms, 3)));
      cells.push(el('td', { class: 'muted' }, fmt(lat.p90_ms, 3)));
      cells.push(el('td', { class: (lat.cv_trimmed ?? lat.cv) > 0.05 ? 'flag' : 'muted', title: `trimmed CV ${lat.cv_trimmed !== undefined ? (lat.cv_trimmed * 100).toFixed(2) + '%' : 'n/a'}, ${lat.count || 0} iterations` }, lat.cv !== undefined ? (lat.cv * 100).toFixed(1) + '%' : '—'));
      cells.push(el('td', { class: 'num', title: 'モデルの演算数 × 実測推論数からの換算値。メーカーの理論ピーク性能とは異なります。' }, formatComputeRate(r)));
      cells.push(el('td', { class: 'muted' }, fmt(r.host_cpu_seconds_per_inference, 4)));
      cells.push(el('td', { class: 'muted' }, r.power && r.power.average_watts ? fmt(r.power.average_watts, 1) : '—'));
      cells.push(el('td', { class: 'muted' }, r.energy_per_inference_j ? fmt(r.energy_per_inference_j, 3) : '—'));
      cells.push(el('td', { class: 'muted' }, fmt(r.compile_ms, 0)));
      cells.push(el('td', { class: 'muted' }, rep0.spread_pct !== undefined ? fmt(rep0.spread_pct, 1) + '%' : '—'));
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
    ['memory', sys.memory ? `${sys.memory.total_gib} GiB` + (sys.memory.speed_mts ? ` @ ${sys.memory.speed_mts}` : '') : null],
    ['os', `${sys.os_name} ${sys.os_release} (${sys.os_version})`],
    ['python', sys.python],
  ]);
  block('run policy', [
    ['warm-up', cfg.warmup],
    ['min iterations', cfg.min_iterations],
    ['min seconds', cfg.min_seconds],
    ['max seconds', cfg.max_seconds],
    ['mode', cfg.mode || 'latency'],
    ['parallel requests', cfg.mode === 'throughput' ? (cfg.requests || 'auto: 1/2/4/8') : 1],
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
      txt = `verify ${e.target} — ${e.ok ? 'ok' : String(e.error || '').split('\n')[0]}`;
    } else if (e.event === 'cell') {
      cls = e.ok ? 'ok' : 'fail';
      txt = e.ok
        ? `${e.target_label} · ${e.workload} · ${fmt(e.units_per_s)} units/s · ${fmt(e.median_ms, 2)} ms`
        : `${e.target_label} · ${e.workload} — ${String(e.error || '').split('\n')[0]}`;
    } else if (e.event === 'error') {
      cls = 'fail'; txt = e.error;
    } else { return; }
    const line = el('div', { class: 'log-line ' + cls });
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
function buildRunModal() {
  const cat = state.catalog;
  if (!cat) return;
  const tWrap = $('targetChecks');
  const wWrap = $('workloadChecks');
  if (tWrap.childElementCount) return;
  tWrap.innerHTML = '';
  ((cat.discovery || {}).targets || []).filter((t) => t.available !== false).forEach((t) => {
    const l = el('label', { class: 'sel', style: `--accent:${deviceColor(t)}` });
    l.innerHTML = `<input type="checkbox" value="${esc(t.id)}" checked /><span>${esc(t.label)}</span><span style="color:var(--dim);font-size:11px">${esc(t.kind)}</span>`;
    l.querySelector('input').addEventListener('change', (e) => l.classList.toggle('sel', e.target.checked));
    tWrap.appendChild(l);
  });
  if (!tWrap.childElementCount) tWrap.appendChild(el('p', { class: 'empty' }, 'no runnable targets were detected on this machine'));

  wWrap.innerHTML = '';
  const defaults = ['dense_gemm', 'conv_vision', 'depthwise_mobile', 'transformer', 'llm_prefill', 'llm_decode'];
  (cat.workloads || []).forEach((w) => {
    if (w.key.endsWith('_int8')) return;
    const on = defaults.includes(w.key);
    const l = el('label', { class: on ? 'sel' : '', style: '--accent:#2563eb' });
    l.innerHTML = `<input type="checkbox" value="${esc(w.key)}" ${on ? 'checked' : ''} /><span>${esc(w.key)}</span><span style="color:var(--dim);font-size:11px">${esc(w.category)}</span>`;
    l.querySelector('input').addEventListener('change', (e) => l.classList.toggle('sel', e.target.checked));
    l.title = w.blurb || '';
    wWrap.appendChild(l);
  });
}

function openModal() {
  buildRunModal();
  const t = $('targetChecks');
  if (t.childElementCount && !t.querySelector('input:checked')) {
    t.querySelectorAll('input').forEach((i) => { i.checked = true; i.parentElement.classList.add('sel'); });
  }
  $('runModal').hidden = false;
}

async function startRun() {
  const targets = [...$('targetChecks').querySelectorAll('input:checked')].map((i) => i.value);
  const workloads = [...$('workloadChecks').querySelectorAll('input:checked')].map((i) => i.value);
  if (!targets.length) { toast('pick at least one device', true); return; }
  if (!workloads.length) { toast('pick at least one workload', true); return; }
  const body = {
    targets,
    workloads,
    mode: $('optMode').value,
    requests: Number($('optRequests').value || 0),
    repeats: Number($('optRepeats').value || 3),
    min_seconds: Number($('optSeconds').value || 4),
    threads: $('optThreads').value ? Number($('optThreads').value) : null,
    include_int8: $('optInt8').checked,
    check_correctness: $('optCheck').checked,
    measure_power: $('optPower').checked,
    allow_battery: $('optBattery').checked,
    quick: $('optQuick').checked,
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
