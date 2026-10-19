'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../fairbench/web/app.js'), 'utf8');
let copied, notice, selected = false;
const log = { children: [{ children: [{ textContent: '16:00:00' }, { textContent: 'failed\nfull error' }] }] };
const context = vm.createContext({
  $: () => log, toast: (message) => { notice = message; },
  navigator: { clipboard: { writeText: async (text) => { copied = text; } } },
  location: {}, renderRoute() {},
  window: { getSelection: () => ({ removeAllRanges() {}, addRange() { selected = true; } }) },
  document: { createRange: () => ({ selectNodeContents() {} }) },
});
vm.runInContext(source.slice(source.indexOf('async function copyLog()'), source.indexOf("$('copyLogBtn').onclick")), context);
(async () => {
  await context.copyLog();
  assert.equal(copied, '16:00:00 failed\nfull error');
  assert.equal(notice, 'ログをコピーしました');
  context.navigator.clipboard.writeText = async () => { throw new Error('denied'); };
  await context.copyLog();
  assert(selected && context.location.hash === 'activity');
  log.children = [];
  await context.copyLog();
  assert.equal(notice, 'コピーするログがありません');
  console.log('Log copy preserves timestamps/multiline errors and handles empty/denied clipboard.');
})().catch((error) => { console.error(error); process.exitCode = 1; });
