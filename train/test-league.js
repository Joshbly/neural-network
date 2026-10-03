#!/usr/bin/env node
// Checks design F (train/lib.js SHAPES) and an E vs F save from scratch.
//   node train/test-league.js [--smoke]
// 1. F's shape: 24 traffic and 28 road and self senses, each mapping onto itself in the mirror image; 52-208-48-2
//    with shortcuts to the outputs; its lane block locked off the traffic senses and its traffic block off the road ones
// 2. a newborn F has every locked connection at 0, and nudges, the engine's Adam steps and mutation keep them there
// 3. --smoke: one real generation of train/evolve.js from scratch (2 E, 2 F, a few pairs, one round) in a temporary
//    save, then: F kept its blocks, shortcuts and locked zeros, and its tournament and rating races replay exactly
const fs = require('fs'), os = require('os'), path = require('path');
const { execFileSync } = require('child_process');
const { E, perturb, initParams, SHAPES } = require('./lib');

let failures = 0;
const check = (ok, text) => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${text}`);
  failures += !ok;
};
const section = title => console.log(`\n${title}`);
const zeroWhereLocked = (genes, gs) => genes.every((g, j) => gs[j] !== 0 || Object.is(g, 0));
const F = SHAPES.F, layers = [E.INPUT_COUNT, 208, 48, 2], T = E.TRAFFIC_SENSES, R = E.ROAD_SENSES;

section('1. the shape');
const closed = senses => senses.every(i => senses.includes(E.MIRROR_FROM[i]));
check(T.length === 24 && R.length === 28 && T.length + R.length === E.INPUT_COUNT && T.every(i => !R.includes(i)), `${T.length} traffic senses and ${R.length} road and self senses, every input in exactly one`);
check(closed(T) && closed(R), 'each group maps onto itself in the mirror image, so the split holds in both passes');
const genes = initParams(layers, 7, F), gs = E.geneScale(layers, F.mask, genes.length), locked = gs.filter(s => s === 0).length;
check(genes.length === E.geneCount(layers, true) && new E.Brain(layers, genes).shortcuts, `${genes.length} weights, shortcuts from the first hidden layer to the outputs`);
check(locked === 160 * T.length + 48 * R.length, `${locked} connections locked, ${genes.length - locked} trainable`);
const span = E.INPUT_COUNT + 1;
check(gs.every((s, k) => s !== 0 || (k < 208 * span && k % span > 0 && (Math.floor(k / span) < 160 ? T : R).includes(k % span - 1))),
  'every locked one is a lane neuron reading a traffic sense or a traffic neuron reading a road sense');

section('2. locked connections stay 0');
check(zeroWhereLocked(genes, gs), 'a newborn F has them all at 0');
check(zeroWhereLocked(perturb(genes, 12345, 0.04, gs), gs) && zeroWhereLocked(perturb(genes, 999, -0.04, gs), gs), 'nudged copies keep them 0');
// the engine's update (train/evolve.js step): credit = noise x scale, then Adam with weight decay, 50 times
const n = genes.length, theta = Float32Array.from(genes), m = new Float32Array(n), v = new Float32Array(n), rng = E.mulberry32(7);
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
check(zeroWhereLocked(theta, gs) && theta.some((x, j) => gs[j] !== 0 && x !== genes[j]), '50 training steps leave them 0 while the rest move');
const eps = E.noiseVector(4242, n);
check(zeroWhereLocked(Float32Array.from(genes, (x, j) => x + 0.04 * (gs[j] * eps[j])), gs), 'a mutated copy keeps them 0');

if (process.argv.includes('--smoke')) {
  section('3. one real generation from scratch (smoke run)');
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'league-smoke-')), config = { founding: 20, familySeats: 3, familySeatsUntil: 50 };
  fs.writeFileSync(path.join(tmp, 'meta.json'), JSON.stringify({ name: 'E vs F smoke', tracks: 'nascar', cars: 'stock', start: 'scratch', seed: 4242, seats: { E: 2, F: 2 }, config }));
  const out = execFileSync('node', [path.join(__dirname, 'evolve.js'), '--dir', tmp, '--pairs', '2', '--rounds', '1', '--until', '1'], { encoding: 'utf8', timeout: 20 * 60e3, maxBuffer: 1 << 26 });
  console.log(out.trim().split('\n').map(l => `    ${l}`).join('\n'));
  const read = file => JSON.parse(fs.readFileSync(path.join(tmp, file), 'utf8'));
  const st = read('state.json'), fs_ = st.population.filter(a => a.species === 'F'), es = st.population.filter(a => a.species === 'E');
  check(st.generation === 1 && JSON.stringify(st.config) === JSON.stringify(config), `generation ${st.generation} finished, the save's recipe founded in`);
  check(fs_.length === 2 && fs_.every(a => a.mask?.length === 2 && a.genes.length === E.geneCount(a.layers, true) && zeroWhereLocked(a.genes, E.geneScale(a.layers, a.mask, a.genes.length))),
    'both F brains kept their two blocks, their shortcuts and their locked zeros through a generation of training');
  check(es.length === 2 && es.every(a => !a.mask && a.genes.length === E.geneCount(a.layers)), 'the E brains are plain 52-128-128-2');
  check(['E', 'F'].every(d => st.history.at(-1).species[d]), 'the tournament scored both designs');
  check(read('live.json').population.filter(a => a.species === 'F').every(a => a.mask?.length === 2), 'the published brains carry F\'s blocks (the app rebuilds practice copies with them)');
  let replay = '';
  try {
    replay = execFileSync('node', [path.join(__dirname, 'test-replay.js'), tmp], { encoding: 'utf8' });
  } catch (e) {
    replay = e.stdout;
  }
  check(/tournament: 24\/24/.test(replay) && /rating races: (\d+)\/\1 /.test(replay), `replays: ${replay.trim().split('\n').join('; ')}`);
  fs.rmSync(tmp, { recursive: true, force: true });
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
