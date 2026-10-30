'use strict';
const { execFileSync, spawnSync } = require('node:child_process');
const { resolve } = require('node:path');
const { writeFileSync } = require('node:fs');
const [rootArg, parent, head, output] = process.argv.slice(2);
const root = resolve(rootArg), trials = [];
try {
  for (let i = 0; i < 5; i++) {
    for (const variant of i % 2 ? [head, parent] : [parent, head]) {
      execFileSync('git', ['switch', '--detach', variant], { cwd: root, stdio: 'ignore' });
      const start = performance.now();
      const r = spawnSync('node', ['scripts/probe-parser-errors.cjs', root, '--json'],
        { cwd: root, encoding: 'utf8', timeout: 30000, maxBuffer: 3000000 });
      if (r.status !== 0) throw Error(r.stderr || 'probe process failed');
      trials.push({ trial: i, revision: variant, ms: performance.now() - start,
        observed: JSON.parse(r.stdout) });
    }
  }
  const summary = Object.fromEntries([parent, head].map((revision) => {
    const rows = trials.filter((r) => r.revision === revision);
    return [revision, { medianMs: rows.map((r) => r.ms).sort((a, b) => a - b)[2],
      holdoutPasses: rows.flatMap((r) => r.observed.holdout).filter(Boolean).length,
      anchorPasses: rows.flatMap((r) => r.observed.anchor).filter(Boolean).length }];
  }));
  writeFileSync(output, JSON.stringify({ trials, summary, scope: 'Native socket contract trials; no speedup claim' }, null, 2) + '\n');
  console.log(JSON.stringify(summary));
} finally { execFileSync('git', ['switch', '--detach', head], { cwd: root }); }
