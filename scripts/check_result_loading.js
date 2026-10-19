'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({
  location: { search: '', href: 'http://localhost/' },
  history: { replaceState() {} }, URL, URLSearchParams,
  setTimeout() {}, clearTimeout() {},
});
vm.runInContext(fs.readFileSync(require.resolve('../fairbench/web/app.js'), 'utf8')
  .split('/* ------------------------------------------------------------------ wire */')[0], context);
vm.runInContext(`
  renderStatic = renderRunList = renderLive = renderOverview = toast = () => {};
  state.followRun = true;
`, context);
(async () => {
  const requests = [];
  let fail = false;
  let displayed;
  context.renderAll = () => { displayed = vm.runInContext('state.report', context); };
  context.fetch = async (url) => {
    requests.push(url);
    if (url === '/api/state') return { ok: true, json: async () => ({
      catalog: {}, results: [], run: { active: false, result_file: 'saved.json', report: { source: 'memory' } },
    }) };
    return { ok: !fail, status: 500, json: async () => ({ source: 'file' }) };
  };
  fail = true;
  await context.loadState();
  assert.equal(displayed, undefined);
  assert.equal(vm.runInContext('state.followRun', context), true);
  fail = false;
  await context.loadState();
  assert.equal(displayed.source, 'file');
  assert.equal(vm.runInContext('state.followRun', context), false);
  assert.deepEqual(requests, ['/api/state', '/api/result/saved.json', '/api/state', '/api/result/saved.json']);
  console.log('Completed runs render saved data after loading; failed loads retry.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
