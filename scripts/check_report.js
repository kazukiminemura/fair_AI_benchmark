'use strict';
// Run with Playwright available in NODE_PATH; no benchmark hardware is required.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'fairbench-report-'));
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const row = { ok: true, kind: 'cpu', workload: 'precision_gemm_fp16', precision: 'fp16',
      target_id: 'cpu', target_label: 'Test CPU', units_per_s: 20, flops_per_s: 1e12,
      latency_ms: 50, repeats_ok: 3, repeats: 3 };
    const report = { started_at: 1793000000, system: { manufacturer: 'Test Maker', model: 'Test Model', computer_identity_recorded_at: 1793100000, cpu: { name: 'Report CPU <test>' } },
      config: { repeats: 3 }, results: [row], grade: { score: 90, letter: 'A', checks: [
        { name: 'Test warning', status: 'warn', detail: '<script>unsafe</script>' },
      ] }, scores: { ai_index: [{ target_id: 'cpu', label: 'Windows ML TensorRT-RTX GPU', ai_index: 1 }] } };
    let entries = [];
    await page.route('http://fairbench.test/**', async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/state') return route.fulfill({ json: {
        catalog: { system: { manufacturer: 'Other PC' }, discovery: {}, workloads: [], targets: [] }, results: entries, run: { active: false },
      } });
      if (url.pathname.startsWith('/api/result/')) return route.fulfill({ json: report });
      const file = path.join(__dirname, '../fairbench/web', url.pathname === '/' ? 'index.html' : url.pathname.slice(1));
      return route.fulfill({ path: file });
    });
    await page.goto('http://fairbench.test/#report');
    assert(await page.locator('#saveReportBtn').isDisabled());
    entries = [{ file: 'selected.json', host: 'Test CPU' }];
    await page.reload();
    await page.waitForFunction(() => !document.getElementById('saveReportBtn').disabled);
    assert.equal(await page.locator('#reportContent > section').count(), 8);
    assert((await page.locator('#reportContent').innerText()).includes('Report CPU <test>'));
    assert((await page.locator('#reportContent').innerText()).includes('PCメーカー: Test Maker / 機種: Test Model / CPU: Report CPU <test>'));
    assert((await page.locator('#reportContent').innerText()).includes('にこのPCから取得して追記しました。'));
    assert(!(await page.locator('#reportContent').innerText()).includes('Other PC'));
    assert.equal(await page.locator('#reportContent script').count(), 0);
    assert(await page.locator('#report-indexChart rect.bar').evaluateAll((bars) => bars.length > 0 && bars.every((bar) => +bar.getAttribute('width') > 0)));
    assert(await page.locator('#report-indexChart text[text-anchor="end"]').evaluateAll((labels) => labels.every((label) => label.getBBox().x >= 0)));
    assert.equal(await page.evaluate(() => {
      const ids = [...document.querySelectorAll('[id]')].map((node) => node.id);
      return ids.length === new Set(ids).size;
    }), true);
    const overview = page.locator('#reportContent > section').first();
    const pngDownloadPromise = page.waitForEvent('download');
    await overview.getByRole('button', { name: 'PNGで保存', exact: true }).click();
    const pngDownload = await pngDownloadPromise;
    assert.equal(pngDownload.suggestedFilename(), 'fairbench-Test-Maker-Report-CPU-test-overview-selected.png');
    const pngPath = path.join(folder, pngDownload.suggestedFilename());
    await pngDownload.saveAs(pngPath);
    const png = fs.readFileSync(pngPath);
    assert.equal(png.subarray(1, 4).toString(), 'PNG');
    assert.equal(png.readUInt32BE(16), 2880);
    assert(png.readUInt32BE(20) > 500);
    await page.evaluate(() => {
      // Test the browser-to-clipboard contract without changing the OS clipboard.
      window.ClipboardItem = class { constructor(data) { this.data = data; } };
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
        write: async (items) => { const blob = await items[0].data['image/png']; window.copiedPng = { type: blob.type, size: blob.size }; },
      } });
    });
    await overview.getByRole('button', { name: '画像をコピー', exact: true }).click();
    await page.waitForFunction(() => window.copiedPng?.size > 1000);
    assert.equal(await page.evaluate(() => window.copiedPng.type), 'image/png');
    await page.evaluate(() => { navigator.clipboard.write = async () => { throw new Error('denied'); }; });
    await overview.getByRole('button', { name: '画像をコピー', exact: true }).click();
    await page.waitForFunction(() => document.getElementById('toast').textContent.includes('コピーできませんでした'));
    assert(await overview.getByRole('button', { name: '画像をコピー', exact: true }).isEnabled());
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'HTMLで保存', exact: true }).click();
    const download = await downloadPromise;
    assert.equal(download.suggestedFilename(), 'fairbench-Test-Maker-Report-CPU-test-report-selected.html');
    const htmlPath = path.join(folder, download.suggestedFilename());
    await download.saveAs(htmlPath);
    const offline = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await offline.context().setOffline(true);
    await offline.goto(require('node:url').pathToFileURL(htmlPath).href);
    assert((await offline.locator('body').innerText()).includes('Report CPU <test>'));
    assert((await offline.locator('body').innerText()).includes('PCメーカー: Test Maker'));
    assert.equal(await offline.locator('script, link, button').count(), 0);
    assert(await offline.locator('rect.bar').evaluateAll((bars) => bars.every((bar) => +bar.getAttribute('width') > 0)));
    assert.equal(await offline.locator('rect[fill^="url"]').evaluateAll((bars) => bars.every((bar) =>
      document.getElementById(bar.getAttribute('fill').slice(5, -1)))), true);
    await page.evaluate(() => { window.print = () => { window.printCalled = document.title; }; });
    await page.getByRole('button', { name: '印刷・PDFで保存', exact: true }).click();
    assert.equal(await page.evaluate(() => window.printCalled), 'fairbench-Test-Maker-Report-CPU-test-report-selected');
    await page.emulateMedia({ media: 'print' });
    assert(await page.locator('.topbar').isHidden());
    assert.equal(await page.locator('#reportContent .table-wrap').evaluate((node) => getComputedStyle(node).maxHeight), 'none');
    await offline.pdf({ path: path.join(folder, 'report.pdf'), preferCSSPageSize: true });
    assert(fs.statSync(path.join(folder, 'report.pdf')).size > 1000);
    await page.emulateMedia({ media: 'screen' });
    await page.screenshot({ path: path.join(folder, 'report.png'), fullPage: true });
    await page.evaluate(() => { state.report.system = { cpu: { name: 'Legacy CPU' } }; renderAll(); });
    assert((await page.locator('#reportContent').innerText()).includes('PCメーカー: 不明'));
    assert(!(await page.locator('#reportContent').innerText()).includes('Other PC'));
    assert.equal(await page.evaluate(() => reportFilename('report', 'html')), 'fairbench-unknown-maker-Legacy-CPU-report-selected.html');
    assert.deepEqual(errors, []);
    console.log(`Report checks passed: PNG export/copy/fallback, empty state, snapshot, offline HTML, SVG references, escaping and PDF. Preview: ${folder}`);
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
