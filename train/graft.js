#!/usr/bin/env node
// Grafts a traffic block onto a brain and founds an experiment save around it. The source save is only read.
//   node train/graft.js --from slot-3 --gen 72 --brain E7·8 [--copies 10] [--to slot-N]
//
// The graft: an E (44-128-128-2) gets 32 more first-layer neurons, after its own 128, that only see the 15 traffic
// senses (the car sensors, both closing speeds, the attacker's side, the draft, hit-in). Every other connection
// into them is locked at 0 for good (a mask: js/replay.js geneScale keeps training off them). Their connections
// into the second layer start at 0, so a grafted brain drives exactly like its source, bit for bit, until
// training gives the block a say. With no car near, every traffic sense is 0 and so is the block.
//
// The save: `copies` grafted copies (design T) and `copies` plain copies (design E) of the same brain, racing
// each other plus the source generation's whole field, frozen, as sparring partners in practice and in the
// tournaments. The rating ladder is the source's, so ratings carry on on the same scale. Nothing is written
// unless every grafted copy makes the source's decision on every recorded moment and runs every check race
// exactly as the source does.
const fs = require('fs'), os = require('os'), path = require('path');
const { Worker, isMainThread, parentPort } = require('worker_threads');
const { E, oval, run } = require('./lib');

const SLOTS = path.join(__dirname, '..', 'models', 'slots'), I = E.IN;
const BLOCK = 32;
// the traffic senses; they map onto each other in the mirrored pass (left and right car sensors swap, the
// attacker's side flips), so the block works in both passes of a decision
const TRAFFIC = [...Array.from({ length: E.CAR_RAY_DEG.length }, (_, r) => I.cars + r), I.closing, I.rearClosing, I.attacker, I.draft, I.ttc];
const PANEL = ['daytona', 'talladega', 'atlanta', 'charlotte', 'michigan', 'kansas', 'darlington', 'dover', 'phoenix', 'richmond', 'bristol', 'martinsville'];
const RACE_M = 24000, FRAME_M = 6000;

// a race worker only when this file is the worker's own script (other tools load it for its functions, in their workers too)
if (!isMainThread && require.main === module) parentPort.on('message', ({ id, job }) => parentPort.postMessage({ id, out: JSON.stringify(run(job)) }));

// The source's layout carried over to the grafted one, for weights or the optimizer's memory: the first layer's
// neurons, then the block (bias and traffic weights `block`, locked connections 0); each second-layer neuron's
// bias and weights from the source's neurons, then its weights from the block (`onward`); the outputs.
function remap(src, layers, block, onward) {
  const [n, h1, h2, outs] = layers, s1 = n + 1, s2 = h1 + 1, s2w = h1 + BLOCK + 1;
  const dst = new Float32Array((h1 + BLOCK) * s1 + h2 * s2w + outs * (h2 + 1));
  dst.set(src.subarray(0, h1 * s1));
  for (let j = h1; j < h1 + BLOCK; j++) {
    dst[j * s1] = block;
    for (const i of TRAFFIC) dst[j * s1 + 1 + i] = block;
  }
  const at = h1 * s1, atw = (h1 + BLOCK) * s1;
  for (let k = 0; k < h2; k++) {
    dst.set(src.subarray(at + k * s2, at + (k + 1) * s2), atw + k * s2w);
    dst.fill(onward, atw + k * s2w + s2, atw + (k + 1) * s2w);
  }
  dst.set(src.subarray(at + h2 * s2), atw + h2 * s2w);
  return dst;
}

// a grafted brain: the block's bias 0 (silent with no car near), its traffic weights random at `spread` from its
// own seed, everything onward from it 0
function graft(layers, genes, seed, spread) {
  if (layers.length !== 4 || layers[1] !== 128 || layers[2] !== 128) throw new Error(`grafting expects an E (128-128), not ${layers.join('-')}`);
  const [n, h1] = layers, wide = [n, h1 + BLOCK, layers[2], layers[3]], g = remap(genes, layers, 0, 0), w = E.noiseVector(seed, BLOCK * TRAFFIC.length);
  for (let j = 0; j < BLOCK; j++) TRAFFIC.forEach((i, k) => g[(h1 + j) * (n + 1) + 1 + i] = spread * w[j * TRAFFIC.length + k]);
  return { layers: wide, genes: E.quantizeGenes(g), mask: { layer: 1, from: h1, to: h1 + BLOCK, inputs: TRAFFIC, boost: 1 } };
}

// the optimizer's memory for a grafted copy: the source's, carried over; the block's new weights start with no
// momentum and a typical second moment for their layer (so their first steps are sized like everyone else's)
function graftMemory(layers, m, v) {
  const [n, h1, h2] = layers, mean = (a, from, to) => { let s = 0; for (let i = from; i < to; i++) s += a[i]; return s / (to - from); };
  const v1 = mean(v, 0, h1 * (n + 1)), v2 = mean(v, h1 * (n + 1), h1 * (n + 1) + h2 * (h1 + 1));
  return { m: remap(m, layers, 0, 0), v: remap(v, layers, v1, v2) };
}

// moments of the source racing: what it sensed (and the mirrored copy its second pass sees)
function recordFrames(source, others) {
  const frames = [];
  PANEL.forEach((trackId, t) => {
    const laps = E.lapsFor(trackId, FRAME_M), drivers = [source, ...others.slice(0, 19)];
    const heat = new E.Heat(oval(trackId), drivers.map(d => new E.Brain(d.layers, Float32Array.from(d.genes))), laps, E.scenarioOptions({ kind: 'field', trackId, laps, cars: 'stock' }));
    const me = heat.cars[0], decide = me.decide.bind(me);
    let k = 0;
    me.decide = () => {
      decide();
      if (k++ % 4 === 0) frames.push({ x: Float32Array.from(me.inputs), m: Float32Array.from(me.mirrored) });
    };
    while (!heat.over) heat.tick();
  });
  return frames;
}

// the block's starting spread: a traffic neuron's input sum about one standard deviation when traffic is present
function trafficSpread(frames) {
  let sum = 0, count = 0;
  for (const { x } of frames) {
    const sq = TRAFFIC.reduce((s, i) => s + x[i] * x[i], 0);
    if (sq > 0) { sum += sq; count++; }
  }
  return 1 / Math.sqrt(sum / count);
}

// every recorded moment, both passes, on the SIMD kernel and in plain JS: the copy's outputs must equal the source's
function decisionsMatch(source, copy, frames) {
  const a = new E.Brain(source.layers, Float32Array.from(source.genes)), b = new E.Brain(copy.layers, Float32Array.from(copy.genes));
  let mismatches = 0;
  for (const { x, m } of frames) {
    const pa = Array.from(a.thinkPair(m, x)), pb = Array.from(b.thinkPair(m, x));
    const ja = [...a.thinkJS(m), ...a.thinkJS(x)], jb = [...b.thinkJS(m), ...b.thinkJS(x)];
    if (pa.some((v, k) => v !== pb[k]) || ja.some((v, k) => v !== jb[k]) || pa.some((v, k) => v !== ja[k])) mismatches++;
  }
  return mismatches;
}

function pool() {
  const workers = Array.from({ length: Math.max(1, os.cpus().length - 2) }, () => new Worker(__filename));
  const runAll = jobs => new Promise(resolve => {
    const out = new Array(jobs.length);
    let next = 0, done = 0;
    if (!jobs.length) return resolve(out);
    const feed = w => {
      if (next >= jobs.length) return;
      const id = next++;
      w.once('message', msg => {
        out[id] = msg.out;
        if (++done === jobs.length) resolve(out);
        else feed(w);
      });
      w.postMessage({ id, job: jobs[id] });
    };
    workers.forEach(feed);
  });
  return { runAll, close: () => Promise.all(workers.map(w => w.terminate())) };
}

// a full race on every oval with the source in one seat, then the same race with each copy in that seat:
// everything about every car (places, laps, damage) must come out identical
async function racesMatch(source, copies, others) {
  const strip = b => ({ layers: b.layers, genes: Array.from(b.genes) });
  const grids = PANEL.map((trackId, t) => ({ trackId, seat: (t * 7) % 20, field: others.slice(0, 19).map(strip) }));
  const jobs = grids.flatMap(({ trackId, seat, field }) => [source, ...copies].map(driver => {
    const drivers = field.slice();
    drivers.splice(seat, 0, strip(driver));
    return { kind: 'field', trackId, laps: E.lapsFor(trackId, RACE_M), cars: 'stock', drivers };
  }));
  const workers = pool(), results = await workers.runAll(jobs);
  await workers.close();
  const per = copies.length + 1;
  return copies.map((_, c) => grids.filter((_, t) => results[t * per + 1 + c] !== results[t * per]).length);
}

const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const writeJson = (file, value) => {
  fs.writeFileSync(`${file}.tmp`, JSON.stringify(value));
  fs.renameSync(`${file}.tmp`, file);
};

async function main() {
  const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
  const from = args.from ?? 'slot-3', gen = +args.gen, name = args.brain, copies = +(args.copies ?? 10);
  if (!name || !Number.isInteger(gen)) throw new Error('usage: node train/graft.js --from slot-3 --gen 72 --brain E7·8 [--copies 10] [--to slot-N]');
  const src = path.join(SLOTS, from), meta = readJson(path.join(src, 'meta.json'));
  if (meta.tracks !== 'nascar') throw new Error('the graft experiment races the NASCAR ovals: pick a NASCAR save');
  const state = readJson(path.join(src, 'state.json'));
  // the generation's brains: the save itself if that's its latest (with the optimizer's memory), else its snapshot
  const latest = state.generation === gen, field = latest ? state.population : readJson(path.join(src, 'generations', `gen-${String(gen).padStart(4, '0')}.json`)).population;
  const source = field.find(b => b.name === name);
  if (!source) throw new Error(`no ${name} in ${from} generation ${gen}`);
  if (source.layers.join() !== [E.INPUT_COUNT, 128, 128, 2].join()) throw new Error(`${name} is ${source.layers.join('-')}, not a current E`);
  const taken = fs.readdirSync(SLOTS).filter(d => /^slot-\d+$/.test(d));
  const to = args.to ?? Array.from({ length: 10 }, (_, i) => `slot-${i + 1}`).find(s => !taken.includes(s));
  const dst = path.join(SLOTS, to);
  if (!to || fs.existsSync(dst)) throw new Error(`${to ?? 'no free slot'}: refusing to write over an existing save`);

  // the source exactly as the engine races it (weights published at 5 decimals), and its rivals
  const genes = E.quantizeGenes(Float32Array.from(source.genes)), base = { ...source, genes }, others = field.filter(b => b !== source);
  console.log(`grafting ${name} (${from}, generation ${gen}) into ${to}: ${copies} plain copies and ${copies} with a ${BLOCK}-neuron traffic block`);
  const frames = recordFrames(base, others), spread = trafficSpread(frames);
  const seeds = Array.from({ length: copies }, (_, k) => 7_200_000 + k + 1);
  const grafts = seeds.map(seed => graft(source.layers, genes, seed, spread));
  const locked = E.geneScale(grafts[0].layers, grafts[0].mask);
  console.log(`  ${frames.length} recorded moments; traffic weights start at spread ${spread.toFixed(3)}; ${locked.filter(s => s === 0).length} locked connections per copy`);

  // the proof: identical decisions on every recorded moment, identical races on every oval
  const decisionMisses = grafts.map(g => decisionsMatch(base, g, frames));
  const raceMisses = await racesMatch(base, grafts, others);
  const lockedNonZero = grafts.filter(g => g.genes.some((x, j) => locked[j] === 0 && x !== 0)).length;
  console.log(`  decisions differing from ${name}: ${decisionMisses.join(' ')} (of ${frames.length} each)`);
  console.log(`  races differing from ${name}: ${raceMisses.join(' ')} (of ${PANEL.length} each)`);
  if (decisionMisses.some(Boolean) || raceMisses.some(Boolean) || lockedNonZero) throw new Error('a grafted copy does not drive exactly like its source: nothing written');

  // written the way the engine saves (train/evolve.js save): weights at 5 decimals, the optimizer's memory at 6 figures
  const weights = g => Array.from(g, x => +x.toFixed(5)), figures = a => Array.from(a, x => +x.toPrecision(6));
  const memory = latest && source.m ? { m: Float32Array.from(source.m), v: Float32Array.from(source.v), t: source.t } : null;
  const born = { parent: name, born: gen, lr: source.lr, sigma: source.sigma, wins: 0, races: 0, recentPoints: [] };
  const plain = seeds.map((_, k) => ({
    layers: source.layers, genes: weights(genes), name: `${name}/${k + 1}`, founder: `${name}/${k + 1}`, label: `${name}, plain copy ${k + 1}`, species: 'E', ...born,
    ...memory && { m: figures(memory.m), v: figures(memory.v), t: memory.t },
  }));
  const grafted = grafts.map((g, k) => {
    const mv = memory && graftMemory(source.layers, memory.m, memory.v), tname = `T${name.slice(1)}/${k + 1}`;
    return {
      layers: g.layers, genes: weights(g.genes), mask: g.mask, name: tname, founder: tname, label: `${name} + traffic block, seed ${seeds[k]}`, species: 'T', ...born,
      ...mv && { m: figures(mv.m), v: figures(mv.v), t: memory.t },
    };
  });
  // the sparring partners: the source generation's whole field, frozen (never inducted over: hallEvery 0)
  const hall = field.map(b => ({ name: `${b.name}@${gen}`, species: b.species, layers: b.layers, genes: weights(E.quantizeGenes(Float32Array.from(b.genes))), gen }));
  const config = { pairs: 48, margin: 0.2, settle: 3, window: 3, pickFromTop: true, familyCap: true, mutate: 1, founding: 0, hallEvery: 0, hallSize: hall.length, hallRivals: 10, tournamentSparring: true };

  fs.mkdirSync(path.join(dst, 'generations'), { recursive: true });
  writeJson(path.join(dst, 'meta.json'), {
    name: `NASCAR · traffic graft (${name})`, created: new Date().toISOString(), tracks: 'nascar', cars: 'stock', start: 'graft', seats: { E: copies, T: copies },
    from: `${meta.name}, generation ${gen}: ${name} ×${copies} plain (design E) and ×${copies} with a grafted traffic block (design T)`,
    graft: { slot: from, generation: gen, brain: name, block: BLOCK, inputs: TRAFFIC, spread: +spread.toFixed(5), seeds, frames: frames.length },
  });
  writeJson(path.join(dst, 'state.json'), { started: new Date().toISOString(), generation: gen, clones: {}, history: [], population: [...plain, ...grafted], practiceRuns: 0, practiceHistory: [], config, hall });
  writeJson(path.join(dst, 'summary.json'), { generation: gen, champion: null, rating: null, updated: new Date().toISOString(), designs: {} });
  for (const file of ['ladder.json', 'rating.json']) if (fs.existsSync(path.join(src, file))) fs.copyFileSync(path.join(src, file), path.join(dst, file));
  console.log(`wrote ${to}: ${plain.length + grafted.length} brains, ${hall.length} sparring partners, the rating ladder from ${from}`);
}

if (isMainThread && require.main === module) main().catch(err => { console.error(err.message); process.exit(1); });

module.exports = { BLOCK, TRAFFIC, PANEL, remap, graft, graftMemory, recordFrames, trafficSpread, decisionsMatch, racesMatch };
