#!/usr/bin/env node
// Gradient learning end to end on this laptop, in a temp save (about a minute):
//   node train/test-ppo-smoke.js
// 1. the CPU learner runs 3 iterations, stops, and resumes to 6 (a generation every 2, 20,000 decisions an iteration)
// 2. the save's files: all parse, the history is generations 0–3 once each, practice runs on across the resume
// 3. the evaluation replays exactly (train/test-replay.js), and every practice episode of the last iteration rebuilds
//    exactly from live.json's weights, progress.json's noise and the episode's seed
// 4. the gate report (train/ppo/tt-report.js --dir) runs and agrees with the learner's own evaluation
// 5. evolution refuses the save and writes nothing
const fs = require('fs'), os = require('os'), path = require('path'), { spawnSync } = require('child_process');
const { E, ttScore } = require('./lib');

const ROOT = path.join(__dirname, '..'), PYTHON = path.join(ROOT, '.venv', 'bin', 'python');
let failures = 0;
const check = (ok, text) => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${text}`);
  failures += !ok;
};
const save = fs.mkdtempSync(path.join(os.tmpdir(), 'ppo-smoke-')), read = file => JSON.parse(fs.readFileSync(path.join(save, file), 'utf8'));
fs.writeFileSync(path.join(save, 'meta.json'), JSON.stringify({ name: 'PPO smoke', tracks: 'nascar', cars: 'stock', start: 'ppo' }));
const learn = iterations => spawnSync(PYTHON, [path.join(__dirname, 'ppo', 'learner.py'), '--dir', path.join(save, 'ppo'), '--threads', '4', '--decisions', '20000',
  '--iterations', String(iterations), '--generation-every', '2', '--device', 'cpu'], { encoding: 'utf8' });
const node = (...args) => spawnSync(process.execPath, args, { encoding: 'utf8' });

console.log('1. learn, stop, resume');
const t0 = Date.now(), first = learn(3), second = learn(6);
check(first.status === 0 && second.status === 0 && second.stderr.includes('resuming at iteration 3'), `3 iterations, then resumed at 3 and ran to 6 (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
if (first.status || second.status) console.log(first.stderr.slice(-800), second.stderr.slice(-800));

console.log('\n2. the save');
const files = ['state.json', 'summary.json', 'live.json', 'progress.json', 'tournament.json', ...[0, 1, 2, 3].map(g => `generations/gen-000${g}.json`)];
const parsed = files.filter(f => { try { read(f); return true; } catch { return false; } });
check(parsed.length === files.length, `all ${files.length} files parse (${files.slice(0, 5).join(', ')}, generations 0–3)`);
const state = read('state.json'), progress = read('progress.json'), live = read('live.json');
check(JSON.stringify(state.history.map(h => [h.gen, h.ppo.iteration])) === '[[0,0],[1,2],[2,4],[3,6]]' && read('summary.json').generation === 3,
  'the history is generations 0, 1, 2, 3 (iterations 0, 2, 4, 6), once each; summary.json says 3');
check(state.practiceHistory.map(p => `${p.gen}.${p.round}`).join() === '1.1,1.2,2.1,2.2,3.1,3.2' && live.iteration === 5 && progress.ppo.iteration === 5,
  'practice: one point per iteration, 6 of them across the resume; live.json and progress.json both at iteration 5');

console.log('\n3. replays');
const replay = node(path.join(__dirname, 'test-replay.js'), save);
check(replay.status === 0 && replay.stdout.includes('32/32 time trials replay exactly'), replay.stdout.trim());
const policy = live.population[0], genes = Float32Array.from(policy.genes);
const rebuilt = progress.ppo.episodes.filter(ep => {
  const heat = new E.Heat(E.OvalTrack.get(ep.trackId), [new E.Brain(policy.layers, genes)], ep.laps, E.scenarioOptions({ kind: 'tt', trackId: ep.trackId, laps: ep.laps, cars: 'stock' }));
  if (ep.rolling) E.placeAt(heat.cars[0], heat.track, ep.rolling);
  heat.cars[0].explore = { sigma: progress.ppo.noise, seed: ep.seed, hold: progress.ppo.hold };
  while (!heat.over) heat.tick();
  return heat.cars[0].steps === ep.steps && Object.is(ttScore(heat.cars[0], heat), ep.score);
});
check(rebuilt.length === progress.ppo.episodes.length, `${rebuilt.length}/${progress.ppo.episodes.length} practice episodes of iteration 5 rebuilt exactly (steps and score)`);

console.log('\n4. the gate report');
const report = node(path.join(__dirname, 'ppo', 'tt-report.js'), '--dir', save), entry = fs.existsSync(path.join(save, 'tt-report.json')) && read('tt-report.json').at(-1);
check(report.status === 0 && entry?.gen === 3 && entry.agrees === true && fs.existsSync(path.join(save, 'tt-report-3.txt')),
  `tt-report.js --dir: generation 3, ${entry?.verdict}, agrees with the learner's own evaluation (score ${entry?.points?.toFixed(3)})`);

console.log('\n5. evolution keeps out');
const before = fs.readdirSync(save).sort().join(), evolve = node(path.join(__dirname, 'evolve.js'), '--dir', save);
check(evolve.status === 1 && evolve.stderr.includes('gradient-learning save') && fs.readdirSync(save).sort().join() === before, 'evolve.js refuses the save and writes nothing');

fs.rmSync(save, { recursive: true, force: true });
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
