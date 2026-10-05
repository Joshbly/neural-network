#!/usr/bin/env node
// Checks the driver's exploration mode for gradient learning (Car.tryAround in js/car.js).
//   node train/test-explore.js
// 1. noiseVector draws exactly as before (it now shares gaussianPair)
// 2. the same seed replays the same exploring time trial exactly: every try, its probability, and the result
// 3. every applied action is within the controls; each recorded probability recomputes from the decision, the
//    spread and the try
// 4. a different seed drives differently; the decision hook alone changes nothing
// 5. held noise: a fresh draw every hold decisions, the same in between; hold 1 is exactly the old every-decision noise
const fs = require('fs'), path = require('path');
const { E, oval } = require('./lib');

let failures = 0;
const check = (ok, text) => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${text}`);
  failures += !ok;
};

console.log('1. the practice-copy noise');
const dm = new Function(`${fs.readFileSync(path.join(__dirname, '..', 'js', 'dmath.js'), 'utf8')}\nreturn { dlog, dcos, dsin };`)();
const before = (seed, n) => {
  const rng = E.mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * dm.dlog(1 - rng())), a = 2 * Math.PI * rng();
    out[i] = r * dm.dcos(a);
    if (i + 1 < n) out[i + 1] = r * dm.dsin(a);
  }
  return out;
};
check([[1, 7], [77, 23554], [4_000_000_001, 21571]].every(([seed, n]) => { const a = before(seed, n), b = E.noiseVector(seed, n); return a.every((x, i) => Object.is(x, b[i])); }),
  'noiseVector draws exactly what it did before sharing gaussianPair');

// F's champion from the E vs F save, 6 km against the clock at Charlotte
const champ = require(path.join(__dirname, '..', 'models', 'slots', 'slot-5', 'generations', 'gen-0040.json')).population.find(b => b.name === 'F5');
const sigma = [0.3, 0.3];
function episode(explore, hook = true, trackId = 'charlotte') {
  const laps = E.lapsFor(trackId, 6000);
  const heat = new E.Heat(oval(trackId), [new E.Brain(champ.layers, Float32Array.from(champ.genes))], laps, E.scenarioOptions({ kind: 'tt', trackId, laps, cars: 'stock' }));
  const car = heat.cars[0], log = [];
  if (explore) car.explore = explore;
  if (hook) car.onDecide = c => log.push({ mean: Array.from(c.mean), raw: Array.from(c.raw), action: Array.from(c.action), logp: c.logp });
  while (!heat.over) heat.tick();
  return { log, result: JSON.stringify({ steps: car.steps, progress: car.progress, finished: car.finished, retired: car.retired, damage: car.damage, laps: car.laps }) };
}

console.log('\n2. a seeded episode replays exactly');
const seed = E.explorationSeed(3, 5, 7), a = episode({ sigma, seed }), b = episode({ sigma, seed });
check(a.log.length > 1000 && JSON.stringify(a.log) === JSON.stringify(b.log), `the same seed: the same ${a.log.length} tries and probabilities`);
check(a.result === b.result, `and the same result (${JSON.parse(a.result).finished ? 'finished' : 'did not finish'}, ${JSON.parse(a.result).steps} steps)`);

console.log('\n3. the tries');
check(a.log.every(d => d.action.every(v => v >= -1 && v <= 1)), 'every applied action is within [-1, 1]');
const clamped = a.log.filter(d => d.raw.some(v => v < -1 || v > 1));
check(clamped.length > 0 && clamped.every(d => d.raw.every((v, k) => v >= -1 && v <= 1 ? Math.fround(v) === d.action[k] : d.action[k] === Math.sign(v))), `tries past the controls are clamped (${clamped.length} of them)`);
const logp = d => d.raw.reduce((s, v, k) => s - 0.5 * ((v - d.mean[k]) / sigma[k]) ** 2 - Math.log(sigma[k]) - 0.5 * Math.log(2 * Math.PI), 0);
const worst = Math.max(...a.log.map(d => Math.abs(logp(d) - d.logp)));
check(worst < 1e-9, `every recorded probability recomputes from the decision, the spread and the try (largest difference ${worst.toExponential(1)})`);
check(a.log.filter(d => d.action.some((v, k) => v !== Math.fround(d.mean[k]))).length > 0.9 * a.log.length, 'the noise is actually applied to the decisions');

console.log('\n4. the rest');
const c = episode({ sigma, seed: E.explorationSeed(3, 5, 8) });
check(JSON.stringify(c.log.slice(0, 50)) !== JSON.stringify(a.log.slice(0, 50)), 'a different seed tries differently');
const plain = episode(null), quiet = episode(null, false);
check(plain.result === quiet.result && plain.log.every(d => d.logp === 0 && d.mean.every(v => v === 0)), 'with exploration off, the decision hook changes nothing and nothing is tried');

console.log('\n5. held noise');
const noiseOf = log => log.map(d => d.raw.map((v, k) => (v - d.mean[k]) / sigma[k]));
const rng = E.mulberry32(seed), fresh = a.log.map(() => Array.from(E.gaussianPair(rng)));
check(noiseOf(a.log).every((g, t) => g.every((v, k) => Math.abs(v - fresh[t][k]) < 1e-9)), 'hold 1 (the default): a fresh draw at every decision, the same stream as before');
const held = episode({ sigma, seed, hold: 8 }), again = episode({ sigma, seed, hold: 8 }), g8 = noiseOf(held.log);
const sameInBlock = g8.every((g, t) => t % 8 === 0 || g.every((v, k) => Math.abs(v - g8[t - 1][k]) < 1e-9));
const newEachBlock = g8.every((g, t) => t % 8 || t === 0 || g.some((v, k) => Math.abs(v - g8[t - 1][k]) > 1e-6));
check(sameInBlock && newEachBlock && g8.filter((_, t) => t % 8 === 0).every((g, b) => g.every((v, k) => Math.abs(v - fresh[b][k]) < 1e-9)),
  `hold 8: the noise changes every 8th decision and holds between (${held.log.length} decisions), drawing the same stream in blocks`);
check(JSON.stringify(held.log) === JSON.stringify(again.log) && held.result === again.result, 'and replays exactly');
const worstHeld = Math.max(...held.log.map(d => Math.abs(logp(d) - d.logp)));
check(worstHeld < 1e-9, `each held try's probability recomputes from its own decision (largest difference ${worstHeld.toExponential(1)})`);

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
