#!/usr/bin/env node
// Checks the traffic graft (train/graft.js) and the locked connections it relies on.
//   node train/test-graft.js [models/slots/slot-4] [--smoke]
// 1. every grafted copy decides every recorded moment exactly as its plain twin (SIMD kernel and plain JS)
// 2. every grafted copy runs a full race on every oval exactly as its plain twin
// 3. nudges, the engine's credit for them and Adam's steps leave locked connections exactly 0
// 4. a brain without a mask is nudged exactly as before (bit for bit)
// 5. a grafted brain that gains an input (train/lib.js widen) drives the same, the new input locked out of its block
// 6. --smoke: one real generation of train/evolve.js on a copy of the save (a few pairs, one round, selection forced
//    to fire), then: locked connections still 0, masks saved, the sparring tournament replays exactly, both designs rated
const fs = require('fs'), os = require('os'), path = require('path');
const { execFileSync } = require('child_process');
const { E, perturb, widen } = require('./lib');
const { TRAFFIC, recordFrames, decisionsMatch, racesMatch } = require('./graft');

const args = process.argv.slice(2), dir = path.resolve(args.find(a => !a.startsWith('--')) ?? 'models/slots/slot-4');
const st = JSON.parse(fs.readFileSync(path.join(dir, 'state.json'), 'utf8'));
let failures = 0;
const check = (ok, text) => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${text}`);
  failures += !ok;
};
const section = title => console.log(`\n${title}`);
const locked = b => E.geneScale(b.layers, b.mask);
const zeroWhereLocked = (genes, gs) => genes.every((g, j) => gs[j] !== 0 || Object.is(g, 0));

(async () => {
  const plain = st.population.filter(b => b.species === 'E'), grafted = st.population.filter(b => b.species === 'T');
  const twin = plain[0], rivals = st.hall.filter(h => h.name !== `${twin.parent}@${st.generation}`);
  console.log(`${path.basename(dir)}: ${plain.length} plain, ${grafted.length} grafted, generation ${st.generation}`);

  const frames = recordFrames(twin, rivals), trained = st.generation - JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8')).graft.generation;
  if (trained) console.log(`\n1-2. skipped: ${trained} generation(s) trained since the graft, so the copies have moved apart from their twin`);
  else {
    section('1. decisions');
    for (const g of grafted) check(decisionsMatch(twin, g, frames) === 0, `${g.name} decides all ${frames.length} recorded moments exactly as ${twin.name}`);

    section('2. whole races');
    const misses = await racesMatch(twin, grafted, rivals);
    grafted.forEach((g, k) => check(misses[k] === 0, `${g.name} runs all 12 ovals exactly as ${twin.name}`));
  }

  section('3. locked connections stay 0');
  const g = grafted[0], gs = locked(g), n = g.genes.length, theta = Float32Array.from(g.genes);
  const lockedCount = ins => 32 * (ins - g.mask.inputs.length);
  check(gs.filter(s => s === 0).length === lockedCount(g.layers[0]), `${lockedCount(g.layers[0])} locked connections, all 0 in the save: ${zeroWhereLocked(theta, gs)}`);
  check(zeroWhereLocked(perturb(theta, 12345, 0.04, gs), gs) && zeroWhereLocked(perturb(theta, 999, -0.04, gs), gs), 'nudged copies keep them 0');
  // the engine's update (train/evolve.js step): credit = noise x scale, then Adam with weight decay, 50 times
  const m = new Float32Array(n), v = new Float32Array(n), rng = E.mulberry32(7);
  for (let t = 1; t <= 50; t++) {
    const grad = new Float32Array(n);
    for (let pair = 0; pair < 8; pair++) {
      const w = rng() - 0.5, eps = E.noiseVector(1000 * t + pair, n);
      for (let j = 0; j < n; j++) grad[j] += w * (gs[j] * eps[j]);
    }
    for (let j = 0; j < n; j++) {
      m[j] = 0.9 * m[j] + 0.1 * grad[j];
      v[j] = 0.999 * v[j] + 0.001 * grad[j] * grad[j];
      theta[j] += 0.02 * (m[j] / (1 - 0.9 ** t)) / (Math.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8) - 0.02 * 0.005 * theta[j];
    }
  }
  check(zeroWhereLocked(theta, gs), '50 training steps leave them 0');
  check(theta.some((x, j) => gs[j] !== 0 && x !== g.genes[j]), 'while the trainable weights moved');
  // a replacement copy (train/evolve.js select): parent + sigma x scaled noise
  const eps = E.noiseVector(4242, n), copy = Float32Array.from(g.genes, (x, j) => x + 1 * g.sigma * (gs[j] * eps[j]));
  check(zeroWhereLocked(copy, gs), "a mutated copy keeps them 0");

  section('4. no mask, nudged as before');
  const old = (t, seed, scale) => { const e = E.noiseVector(seed, t.length), out = new Float32Array(t.length); for (let i = 0; i < t.length; i++) out[i] = t[i] + scale * e[i]; return out; };
  const base = Float32Array.from(twin.genes);
  check([1, 77, 4_000_000_001].every(seed => [0.04, -0.04, 0.0123].every(s => { const a = perturb(base, seed, s), b = old(base, seed, s); return a.every((x, i) => Object.is(x, b[i])); })), 'perturbGenes without a scale is bit-identical to before');
  check(E.geneScale(twin.layers, undefined) === null, 'a brain without a mask has no scale');

  section('5. new inputs');
  // the grafted brain as if it had been built before its last input existed, then widened the way the engine does
  // to every input it has today
  const [ins, h1] = g.layers, short = [ins - 1, ...g.layers.slice(1)], cut = new Float32Array(E.geneCount(short));
  let k = 0;
  for (let j = 0; j < h1; j++) for (let i = 0; i <= ins; i++) if (i !== ins) cut[k++] = g.genes[j * (ins + 1) + i];
  cut.set(Float32Array.from(g.genes).subarray(h1 * (ins + 1)), k);
  const wide = widen(short, cut), scale = E.geneScale(wide.layers, g.mask);
  check(wide.layers[0] === E.INPUT_COUNT && wide.layers.slice(1).join() === g.layers.slice(1).join(), `widened to ${wide.layers.join('-')}`);
  check(zeroWhereLocked(wide.genes, scale) && scale.filter(s => s === 0).length === lockedCount(E.INPUT_COUNT), 'the new inputs are locked out of the block');
  // it senses the closing speeds louder once it reads the motion inputs (js/car.js CLOSING_GAIN)
  const a = new E.Brain(short, cut), b = new E.Brain(wide.layers, wide.genes);
  const loud = x => { const y = Float32Array.from(x); for (const i of [E.IN.closing, E.IN.rearClosing]) y[i] *= E.CLOSING_GAIN; return y; };
  check(frames.every(({ x, m: mx }) => { const p = Array.from(a.thinkPair(mx, x)), q = Array.from(b.thinkPair(loud(mx), loud(x))); return p.every((y, i) => y === q[i]); }), 'and it drives exactly as before');

  if (args.includes('--smoke')) {
    section('6. one real generation (smoke run)');
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'graft-smoke-'));
    for (const file of ['meta.json', 'summary.json', 'ladder.json', 'rating.json']) fs.copyFileSync(path.join(dir, file), path.join(tmp, file));
    // selection forced to fire (no settling period, any gap enough), so a mutated copy is made too
    fs.writeFileSync(path.join(tmp, 'state.json'), JSON.stringify({ ...st, config: { ...st.config, settle: 0, margin: -1 } }));
    const out = execFileSync('node', [path.join(__dirname, 'evolve.js'), '--dir', tmp, '--pairs', '4', '--rounds', '1', '--until', String(st.generation + 1)], { encoding: 'utf8', timeout: 15 * 60e3, maxBuffer: 1 << 26 });
    console.log(out.trim().split('\n').map(l => `    ${l}`).join('\n'));
    const after = JSON.parse(fs.readFileSync(path.join(tmp, 'state.json'), 'utf8'));
    const tg = after.population.filter(x => x.species === 'T');
    check(after.generation === st.generation + 1, `generation ${after.generation} finished`);
    check(tg.length === grafted.length && tg.every(x => x.mask && zeroWhereLocked(x.genes, locked(x))), 'every grafted brain kept its mask and its locked connections at 0');
    const replaced = after.history.at(-1).replaced;
    check(replaced.length > 0, `selection made ${replaced.length} replacement(s): ${replaced.map(r => `${r.out} → ${r.by}`).join(', ')}`);
    const t = JSON.parse(fs.readFileSync(path.join(tmp, 'tournament.json'), 'utf8'));
    check(t.sparring?.length === st.hall.length && t.races.every(r => r.grid.length === 40), `tournament: ${t.races.length} races of ${t.races[0].grid.length} cars, ${t.sparring?.length} sparring partners published`);
    const replay = execFileSync('node', [path.join(__dirname, 'test-replay.js'), tmp], { encoding: 'utf8' });
    check(/tournament: 24\/24/.test(replay) && /rating races: (\d+)\/\1/.test(replay), `replays: ${replay.trim().split('\n').join('; ')}`);
    const rating = JSON.parse(fs.readFileSync(path.join(tmp, 'rating.json'), 'utf8'));
    const rated = rating.players.filter(p => p.gen === st.generation && p.kind === 'entrant');
    check(['T', 'E'].every(d => rated.some(p => p.design === d)), `rated at generation ${st.generation}: ${rated.map(p => `${p.name} ${p.r}±${p.sd}`).join(', ')}`);
    const live = JSON.parse(fs.readFileSync(path.join(tmp, 'live.json'), 'utf8'));
    check(live.population.filter(x => x.species === 'T').every(x => x.mask), 'the published brains carry their masks (the app rebuilds practice copies with them)');
    fs.rmSync(tmp, { recursive: true, force: true });
  }

  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})();
