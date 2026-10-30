'use strict';
const assert = require('node:assert/strict');
const { execFileSync, spawnSync } = require('node:child_process');
const { readFileSync, writeFileSync } = require('node:fs');
const { createHash } = require('node:crypto');
const { join, resolve } = require('node:path');
const { pathToFileURL } = require('node:url');
const [rootArg, parent, head, toolsArg, outArg] = process.argv.slice(2);
const root = resolve(rootArg), tools = resolve(toolsArg), out = resolve(outArg);
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' });
const hash = (text) => createHash('sha256').update(text).digest('hex');
for (const file of ['package.json', 'package-lock.json', 'test.js', 'scripts/probe-parser-errors.cjs']) {
  assert.equal(git('show', parent + ':' + file), git('show', head + ':' + file));
}
const inverse = git('show', parent + ':index.js');
const measures = {};
const run = (cmd, args, seconds) => {
  const r = spawnSync('timeout', ['--kill-after=5', String(seconds), cmd, ...args], {
    cwd: root, encoding: 'utf8', timeout: (seconds + 10) * 1000, maxBuffer: 8000000,
    env: { ...process.env, CARGO_TARGET_DIR: '/tmp/simulator-oct30a-rust',
      CARGO_INCREMENTAL: '0', CARGO_PROFILE_RELEASE_DEBUG: '0', CARGO_PROFILE_DEV_DEBUG: '0' },
  });
  return { command: [cmd, ...args], exit: r.status, signal: r.signal,
    stdout: r.stdout || '', stderr: r.stderr || '' };
};
async function main() {
  const fw = await import(pathToFileURL(join(tools, 'node_modules/@metaharness/flywheel/dist/index.js')));
  assert.equal(JSON.parse(readFileSync(join(tools, 'node_modules/@metaharness/flywheel/package.json'))).version, '0.1.12');
  const evaluator = async (policy, suite) => {
    assert([parent, head].includes(policy.revision));
    if (!measures[policy.revision]) {
      git('switch', '--detach', policy.revision);
      const checks = {};
      checks.syntax = run('node', ['--check', 'index.js'], 20);
      const probe = run('node', ['scripts/probe-parser-errors.cjs', root, '--json'], 30);
      assert.equal(probe.exit, 0, probe.stderr);
      const observed = JSON.parse(probe.stdout);
      checks.test = run('npm', ['test'], 45);
      const names = checks.test.stdout.split('\n').filter((x) => x.trim().startsWith('PASS:')).map((x) => x.trim()).sort();
      const prior = measures[parent];
      const retained = !prior || JSON.stringify(prior.names) === JSON.stringify(names);
      checks.audit = run('npm', ['audit', '--json'], 30);
      checks.prodAudit = run('npm', ['audit', '--omit=dev', '--json'], 30);
      checks.rust = run('cargo', ['+1.88.0', 'test', '--locked', '--offline', '--release', '-j2'], 60);
      checks.rustBuild = run('cargo', ['+1.88.0', 'build', '--locked', '--offline', '--release', '-j2'], 45);
      checks.clippy = run('cargo', ['+1.88.0', 'clippy', '--locked', '--offline', '--all-targets', '--all-features', '-j2', '--', '-D', 'warnings'], 45);
      checks.format = run('cargo', ['+1.88.0', 'fmt', '--all', '--', '--check'], 30);
      measures[policy.revision] = { checks, observed, names, retained };
      writeFileSync(out + '.checkpoint', JSON.stringify(measures, null, 2));
    }
    const m = measures[policy.revision];
    const wins = suite.id === 'holdout' ? m.observed.holdout : m.observed.anchor;
    const primary = wins.filter(Boolean).length / wins.length;
    return { primary, noopRate: 1 - primary, costPerWin: 0, itemWins: wins,
      regressed: !m.retained || Object.values(m.checks).some((x) => x.exit !== 0) };
  };
  try {
    const result = await fw.runFlywheelGenerations({
      rootPolicy: { revision: parent },
      proposer: async () => ({ value: head, summary: 'Sanitize parser failures into stable JSON errors',
        inverse: { path: 'index.js', parentBytes: inverse, sha256: hash(inverse) } }),
      evaluator, holdout: { id: 'holdout', items: [0, 1, 2, 3, 4, 5, 6, 7] },
      anchor: { id: 'anchor', items: [0, 1, 2, 3, 4, 5] },
      maxGenerations: 1, signer: fw.makeSigner(), promotionRule: fw.meetsPromotionRule, dataSource: 'LIVE',
    });
    const replay = fw.verifyReplayBundle(result.replayBundle, {
      pinnedGateFingerprint: fw.gateFingerprint(fw.meetsPromotionRule),
    });
    assert.equal(replay.pass, true);
    writeFileSync(out, JSON.stringify({ parent, head, measures, replay,
      replayBundle: result.replayBundle, finalPolicy: result.finalPolicy }, null, 2));
    console.log(JSON.stringify({ replay, promotions: result.promotions.length }));
  } finally { git('switch', '--detach', head); }
}
main().catch((err) => { console.error(err); process.exitCode = 1; });
