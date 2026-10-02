#!/usr/bin/env node
// Proves the WebAssembly brains (js/nn-wasm.js) compute exactly what the plain JS loop does, bit for bit:
//   1. every activation of every layer, for think() and thinkPair(), across all designs and edge inputs
//   2. whole races (practice, duel, time trial, tournament) give JSON-identical results either way
//   3. outputs stay exact while the kernel's memory fills and starts over mid-race
//   node train/test-nn.js
const path = require('path');
const { E, run } = require('./lib');

const { Brain, mulberry32 } = E;
if (!Brain.simd) {
  console.error('WebAssembly SIMD is not available here, nothing to compare');
  process.exit(1);
}
const st = require(path.join(__dirname, '..', 'models', 'slots', 'slot-1', 'state.json'));
const rng = mulberry32(2026);
const gaussian = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
const bits = a => Array.from(new Uint32Array(Float32Array.from(a).buffer));
const same = (a, b) => a.length === b.length && bits(a).every((v, i) => v === bits(b)[i]);
let failures = 0;
const fail = msg => {
  if (++failures <= 10) console.error(`  MISMATCH ${msg}`);
};

const withJS = fn => {
  Brain.simd = false;
  try {
    return fn();
  } finally {
    Brain.simd = true;
  }
};

// ---- 1. every activation, bit for bit ----
const inputCount = E.INPUT_COUNT;
const designs = [
  ...Object.values(Object.groupBy(st.population, a => a.species)).map(group => ({ name: `design ${group[0].species} (saved genes)`, layers: group[0].layers, genes: Float32Array.from(group[0].genes) })),
  { name: 'older brain reading fewer inputs', layers: [30, 16, 10, 2] },
  { name: 'odd widths (padding)', layers: [inputCount, 5, 13, 1] },
  { name: 'design E, wild random genes', layers: [inputCount, 128, 128, 2], scale: 3 },
];
const inputKinds = [
  () => rng() * 2 - 1,
  () => 0,
  () => (rng() * 2 - 1) * 8,
  () => (rng() < 0.5 ? -1 : 1) * (rng() < 0.5 ? 3 : 1e-30),
];
let vectors = 0;
for (const d of designs) {
  const genes = d.genes ?? Float32Array.from(Brain.random(d.layers), g => g * (d.scale ?? 1) + (d.scale ? gaussian() * 0.01 : 0));
  const js = new Brain(d.layers, genes), wasm = new Brain(d.layers, genes);
  for (let n = 0; n < 1500; n++) {
    const kind = inputKinds[n % inputKinds.length];
    const x = Float32Array.from({ length: inputCount }, kind), m = Float32Array.from({ length: inputCount }, kind);
    const want = withJS(() => js.think(x)).slice(), got = wasm.think(x);
    if (!same(want, got)) fail(`${d.name}: think output ${[...want]} vs ${[...got]}`);
    js.acts.forEach((a, l) => same(a, wasm.acts[l]) || fail(`${d.name}: think layer ${l}`));
    const wantPair = withJS(() => js.thinkPair(m, x)).slice(), gotPair = wasm.thinkPair(m, x);
    if (!same(wantPair, gotPair)) fail(`${d.name}: thinkPair ${[...wantPair]} vs ${[...gotPair]}`);
    js.acts.forEach((a, l) => same(a, wasm.acts[l]) || fail(`${d.name}: thinkPair layer ${l}`));
    vectors++;
  }
}
console.log(`1. activations: ${vectors} input vectors x ${designs.length} brains, think and thinkPair, every layer: ${failures ? 'FAILED' : 'bit-identical'}`);

// ---- 2. whole races ----
const before = failures, pick = (list, n) => list.slice().sort(() => rng() - 0.5).slice(0, n);
const driver = a => ({ layers: a.layers, genes: Float32Array.from(a.genes) });
const scenarios = [];
for (let k = 0; k < 6; k++) {
  const [me, ...rivals] = pick([...st.population, ...st.hall], 12);
  scenarios.push({ kind: 'race', ...driver(me), opponents: rivals.map(driver), trackSeed: 10_000_900 + k, slot: k * 2 % 12, laps: 3 });
  scenarios.push({ kind: 'race', ...driver(me), opponents: [driver(rivals[0])], trackSeed: 10_000_950 + k, slot: k % 2, laps: 2 });
  scenarios.push({ kind: 'tt', ...driver(me), trackSeed: 10_000_970 + k, laps: 3 });
}
for (let k = 0; k < 2; k++) scenarios.push({ kind: 'field', drivers: pick(st.population, 20).map(driver), trackSeed: 5_000_900 + k, laps: 10 });
for (const sc of scenarios) {
  const want = JSON.stringify(withJS(() => run(sc))), got = JSON.stringify(run(sc));
  if (want !== got) fail(`${sc.kind} race on track ${sc.trackSeed}`);
}
console.log(`2. whole races: ${scenarios.length} (12-car practice, duels, time trials, 20-car 10-lap tournament races): ${failures > before ? 'FAILED' : 'JSON-identical'}`);

// ---- 3. memory filling up and starting over mid-race ----
const before3 = failures, big = [inputCount, 128, 128, 2];
const brains = Array.from({ length: 400 }, () => {
  const genes = Brain.random(big);
  return { js: new Brain(big, genes), wasm: new Brain(big, genes) };
});
for (let n = 0; n < 6000; n++) {
  const b = brains[Math.floor(rng() * brains.length)], x = Float32Array.from({ length: inputCount }, () => rng() * 2 - 1);
  const want = withJS(() => b.js.think(x)).slice();
  if (!same(want, b.wasm.think(x))) fail(`stress call ${n}`);
  if (n % 500 === 0) b.js.acts.forEach((a, l) => same(a, b.wasm.acts[l]) || fail(`stress acts ${n} layer ${l}`));
}
// a brain whose memory was reused keeps showing its last activations
const kept = brains[0];
kept.wasm.think(Float32Array.from({ length: inputCount }, () => 0.5));
const shown = kept.wasm.acts.map(a => a.slice());
for (const b of brains.slice(1)) b.wasm.think(Float32Array.from({ length: inputCount }, () => -0.5));
if (kept.wasm.epoch !== -1) fail('expected the first brain to have been moved out of the kernel memory');
shown.forEach((a, l) => same(a, kept.wasm.acts[l]) || fail(`evicted brain lost layer ${l}`));
console.log(`3. memory reuse: 400 big brains (several times the kernel's memory), 6000 interleaved passes: ${failures > before3 ? 'FAILED' : 'exact, and evicted brains keep their activations'}`);

if (failures) {
  console.error(`\n${failures} mismatches`);
  process.exit(1);
}
console.log('\nall exact');
