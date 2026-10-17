'use strict';
// Regression gate for the grade the dashboard prints.
//
// `renderGrade` prints the score/letter stored in the result file, while the
// check list right below it is re-graded in the browser by `gradeChecks`
// (legacy reports downgrade Correctness to "warn").  The ring and the list
// describe the same run, so they must not disagree.
//
// Run with:  node scripts/check_dashboard_grade.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const makeElement = (tag) => ({
  tag, children: [], textContent: '', className: '', hidden: false, innerHTML: '',
  attributes: {}, handlers: {}, dataset: {}, style: {},
  setAttribute(name, value) { this.attributes[name] = value; },
  addEventListener(name, handler) { this.handlers[name] = handler; },
  replaceChildren() { this.children = []; },
  appendChild(child) { this.children.push(child); return child; },
  querySelector() { return makeElement('q'); },
  querySelectorAll() { return []; },
});
const context = vm.createContext({
  document: {
    createElement: (tag) => makeElement(tag),
    createElementNS: (namespace, tag) => makeElement(tag),
    getElementById: (id) => makeElement('#' + id),
    querySelectorAll: () => [],
    addEventListener: () => {},
  },
  window: { addEventListener: () => {}, scrollTo: () => {} },
  location: { hash: '', search: '', href: 'http://localhost/' },
  history: { replaceState: () => {} },
  requestAnimationFrame: (handler) => handler(),
  setTimeout: () => 0, clearTimeout: () => {}, console,
});
const source = fs.readFileSync(require.resolve('../fairbench/web/app.js'), 'utf8');
vm.runInContext(source.split('/* ------------------------------------------------------------------ wire */')[0], context);

// Same weights and skip rule as fairbench.runner._grade.
const WEIGHTS = { pass: 1.0, warn: 0.6, skip: 1.0, fail: 0.0 };
function scoreOfChecks(checks) {
  const scored = checks.filter((check) => check.status !== 'skip');
  if (!scored.length) return 100.0;
  return Math.round(1000 * scored.reduce((sum, check) => sum + WEIGHTS[check.status], 0) / scored.length) / 10;
}

function displayedScore(report) {
  return (report.grade || {}).score;
}
function recomputedScore(report) {
  return scoreOfChecks(context.gradeChecks(report));
}

const cases = [];
// Deterministic case: a legacy report (no measurement_contract) whose stored
// grade counted Correctness as a pass.  The browser shows it as "warn".
cases.push({
  name: 'legacy report without measurement_contract',
  report: {
    results: [{ ok: true, repeats_ok: 3 }],
    grade: { score: 100, letter: 'A', checks: [{ name: 'Correctness', status: 'pass', detail: '12 cell(s) matched' }] },
  },
});
// And every stored report, so the gate runs against real data too.
const directory = path.join(__dirname, '../results');
if (fs.existsSync(directory)) {
  for (const name of fs.readdirSync(directory).filter((file) => file.endsWith('.json'))) {
    try {
      cases.push({
        name,
        report: JSON.parse(fs.readFileSync(path.join(directory, name), 'utf8').replace(/^\uFEFF/, '')),
      });
    } catch (error) {
      console.warn(`  skipped ${name}: ${error.message}`);
    }
  }
}

const failures = [];
for (const { name, report } of cases) {
  if (!report.grade || !Array.isArray(report.grade.checks)) continue;
  const shown = displayedScore(report);
  const recomputed = recomputedScore(report);
  const changed = context.gradeChecks(report).filter((check, index) => check.status !== report.grade.checks[index].status);
  if (changed.length) {
    console.log(`  ${name}: displayed ${shown} vs recomputed ${recomputed} ` +
      `(${changed.map((check) => `${check.name}->${check.status}`).join(', ')})`);
  }
  if (shown !== recomputed) failures.push(`${name}: ring shows ${shown}/100 but the checks shown next to it score ${recomputed}/100`);
}
assert.deepEqual(failures, [], `the displayed grade contradicts the displayed checks:\n  ${failures.join('\n  ')}`);
console.log('Grade ring matches the checks the dashboard displays.');
