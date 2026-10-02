#!/usr/bin/env node
// Evolution Strategies trainer (OpenAI-ES): antithetic sampling, common random numbers, centred-rank
// fitness shaping, Adam. Stage "tt" learns pace on randomised tracks; stage "race" learns racecraft
// against a self-play league, starting from different grid positions.
//
//   node train/es.js --name pace-c --arch 64,64 --stage tt --iters 300
//   node train/es.js --name pro-c --init models/pace-c.json --stage race --iters 150
const os = require('os');
const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');
const { E, noise, initParams, widen } = require('./lib');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const opt = {
  name: args.name || 'run', stage: args.stage || 'tt', iters: +(args.iters || 300), pairs: +(args.pairs || 48),
  sigma: +(args.sigma || 0.04), lr: +(args.lr || 0.02), decay: +(args.decay || 0.005), seed: +(args.seed || 1),
  workers: +(args.workers || Math.max(1, os.cpus().length - 2)), evalEvery: +(args.evalEvery || 10),
  laps: +(args.laps || 2), field: +(args.field || 6), races: +(args.races || 2),
};
const TRAIN_TRACKS = Array.from({ length: 16 }, (_, i) => 101 + i);
const HELD_OUT = Array.from({ length: 8 }, (_, i) => 901 + i);
const MODELS = path.join(__dirname, '..', 'models');
fs.mkdirSync(MODELS, { recursive: true });

// --init models/x.json resumes a model; --init models/field.json#3 starts from that field's P3
const [initFile, initRank] = (args.init || '').split('#');
const init = initFile && (json => initRank ? json.drivers[initRank - 1] : json)(JSON.parse(fs.readFileSync(initFile, 'utf8')));
const upgrade = genes => widen(init.layers, Float32Array.from(genes)).genes;
const layers = [E.INPUT_COUNT, ...(init ? init.layers.slice(1) : [...(args.arch || '64,64').split(',').map(Number), 2])];
let theta = init ? upgrade(init.genes) : initParams(layers, opt.seed);
const n = theta.length, rng = E.mulberry32(opt.seed * 7919);
const pick = list => list[Math.floor(rng() * list.length)];

// ---- worker pool ----
const pool = Array.from({ length: opt.workers }, () => new Worker(path.join(__dirname, 'worker.js')));
let jobId = 0;
function dispatch(tasks, scenarios, genes = theta, sigma = opt.sigma) {
  const chunks = pool.map(() => []);
  tasks.forEach((t, i) => chunks[i % pool.length].push(t));
  return Promise.all(pool.map((w, wi) => new Promise(resolve => {
    if (!chunks[wi].length) return resolve([]);
    const id = jobId++;
    const onMessage = msg => { if (msg.id === id) { w.off('message', onMessage); resolve(msg.results); } };
    w.on('message', onMessage);
    w.postMessage({ id, layers, theta: genes, sigma, tasks: chunks[wi], scenarios });
  }))).then(parts => parts.flat());
}

// ---- self-play league (race stage) ----
// Fixed anchors come first: a built field (--rivals, ranked best first) or the starting policy and close
// variants (--bench). A resumed racer also brings its own past snapshots, so it trains against strong
// and varied rivals. Evaluating every lineage against the same --rivals field keeps win rates comparable.
const anchors = [], league = [];
const variants = (genes, label) => Array.from({ length: 5 }, (_, k) => {
  const eps = noise(424242 + k, genes.length), g = new Float32Array(genes.length);
  for (let i = 0; i < g.length; i++) g[i] = genes[i] + (k ? 0.02 * eps[i] : 0);
  return { layers, genes: g, label: k ? `${label}+${k}` : label };
});
if (opt.stage === 'race') {
  if (args.rivals) {
    anchors.push(...JSON.parse(fs.readFileSync(args.rivals, 'utf8')).drivers.map(d => ({ layers: d.layers, genes: Float32Array.from(d.genes), label: d.label })));
  } else {
    const bench = args.bench && JSON.parse(fs.readFileSync(args.bench, 'utf8'));
    anchors.push(...variants(bench ? widen(bench.layers, Float32Array.from(bench.genes)).genes : theta, bench ? bench.name : 'seed'));
  }
  league.push(...anchors);
  for (const l of init?.league || []) league.push({ layers, genes: upgrade(l.genes), label: l.label });
}
const benchmark = anchors.length ? Array.from({ length: opt.field - 1 }, (_, i) => anchors[i % anchors.length]) : [];
const sampleOpponents = () => {
  const pool = [...league], out = [];
  while (out.length < opt.field - 1) out.push(pool.length ? pool.splice(Math.floor(rng() * pool.length), 1)[0] : pick(league));
  return out;
};

function scenarios() {
  if (opt.stage === 'tt') return [0, 1].map(() => ({ kind: 'tt', trackSeed: pick(TRAIN_TRACKS), laps: opt.laps }));
  // one start from the front of the grid, one from the back: leading and attacking are both skills.
  // Plus a solo time trial as rehearsal, so racing a league of near-copies can't erode raw driving.
  const opponents = sampleOpponents();
  return [
    ...Array.from({ length: opt.races }, (_, k) => k % 2 ? opt.field - 1 - Math.floor(rng() * 3) : Math.floor(rng() * 2)).map(slot =>
      ({ kind: 'race', trackSeed: pick(TRAIN_TRACKS), slot, laps: opt.laps, opponents })),
    { kind: 'tt', trackSeed: pick(TRAIN_TRACKS), laps: opt.laps },
  ];
}

// one policy, many scenarios: spread the scenarios (not candidates) across workers
function across(scs, genes = theta) {
  const chunks = pool.map(() => []);
  scs.forEach((sc, i) => chunks[i % pool.length].push([i, sc]));
  return Promise.all(pool.map((w, wi) => new Promise(resolve => {
    if (!chunks[wi].length) return resolve([]);
    const id = jobId++;
    const onMessage = msg => { if (msg.id === id) { w.off('message', onMessage); resolve(msg.results[0].outcomes.map((o, k) => [chunks[wi][k][0], o])); } };
    w.on('message', onMessage);
    w.postMessage({ id, layers, theta: genes, sigma: 0, tasks: [{ key: 'eval', seed: null }], scenarios: chunks[wi].map(c => c[1]) });
  }))).then(parts => parts.flat().sort((a, b) => a[0] - b[0]).map(p => p[1]));
}

// ---- evaluation on tracks never trained on ----
async function evaluate() {
  const solo = await across(HELD_OUT.map(trackSeed => ({ kind: 'tt', trackSeed, laps: 2 })));
  const report = {
    completed: solo.filter(o => o.finished).length / solo.length,
    lap: mean(solo.filter(o => o.lap).map(o => o.lap)),
    walls: mean(solo.map(o => o.walls)),
    soloAero: mean(solo.map(o => o.aero)),
    score: mean(solo.map(o => o.score)),
  };
  if (opt.stage === 'race') {
    // 4 grid positions per unseen track: 32 races keeps the win-rate estimate's noise near ±8%
    const slots = [0, Math.floor(opt.field / 3), Math.floor(2 * opt.field / 3), opt.field - 1];
    const races = HELD_OUT.flatMap(trackSeed => slots.map(slot => ({ kind: 'race', trackSeed, slot, laps: opt.laps, opponents: benchmark })));
    const out = await across(races);
    Object.assign(report, {
      winRate: out.filter(o => o.won).length / out.length,
      avgPlace: mean(out.map(o => o.place + 1)),
      fromBackWins: out.filter((o, i) => races[i].slot === opt.field - 1 && o.won).length,
      passes: mean(out.map(o => o.passes)), rammed: mean(out.map(o => o.rammed)), aero: mean(out.map(o => o.aero)),
      raceScore: mean(out.map(o => o.score)),
    });
  }
  return report;
}
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;

// centred ranks in [-0.5, 0.5] within one scenario
function centredRanks(values) {
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), out = new Float32Array(values.length);
  order.forEach(([, i], r) => out[i] = r / (values.length - 1) - 0.5);
  return out;
}

function save(tag, extra) {
  const file = path.join(MODELS, `${opt.name}${tag}.json`);
  fs.writeFileSync(file, JSON.stringify({
    name: opt.name, layers, stage: opt.stage, genes: Array.from(theta, v => +v.toFixed(5)),
    league: opt.stage === 'race' ? league.slice(anchors.length).map(l => ({ label: l.label, genes: Array.from(l.genes, v => +v.toFixed(5)) })) : undefined,
    ...extra,
  }));
}

(async () => {
  console.log(`[${opt.name}] stage=${opt.stage} layers=${layers.join('-')} params=${n} pairs=${opt.pairs} sigma=${opt.sigma} lr=${opt.lr} workers=${opt.workers}`);
  const m = new Float32Array(n), v = new Float32Array(n), b1 = 0.9, b2 = 0.999;
  let best = -Infinity, bestReport = null;
  const t0 = Date.now();
  for (let it = 1; it <= opt.iters; it++) {
    const scs = scenarios(), seeds = Array.from({ length: opt.pairs }, (_, i) => opt.seed * 1e7 + it * 1000 + i);
    const tasks = seeds.flatMap((seed, i) => [{ key: `${i}+`, seed, sign: 1 }, { key: `${i}-`, seed, sign: -1 }]);
    const results = await dispatch(tasks, scs);
    const byKey = new Map(results.map(r => [r.key, r.outcomes]));

    // fitness shaping per scenario, then averaged, so a hard track can't dominate the update
    const shaped = new Map(tasks.map(t => [t.key, 0]));
    scs.forEach((_, k) => {
      const ranks = centredRanks(tasks.map(t => byKey.get(t.key)[k].score));
      tasks.forEach((t, i) => shaped.set(t.key, shaped.get(t.key) + ranks[i] / scs.length));
    });

    const grad = new Float32Array(n);
    seeds.forEach((seed, i) => {
      const w = (shaped.get(`${i}+`) - shaped.get(`${i}-`)) / (2 * opt.pairs * opt.sigma), eps = noise(seed, n);
      for (let j = 0; j < n; j++) grad[j] += w * eps[j];
    });
    for (let j = 0; j < n; j++) {
      m[j] = b1 * m[j] + (1 - b1) * grad[j];
      v[j] = b2 * v[j] + (1 - b2) * grad[j] * grad[j];
      const step = (m[j] / (1 - b1 ** it)) / (Math.sqrt(v[j] / (1 - b2 ** it)) + 1e-8);
      theta[j] += opt.lr * step - opt.lr * opt.decay * theta[j];
    }

    const scores = results.flatMap(r => r.outcomes.map(o => o.score));
    let line = `it ${String(it).padStart(3)} | mean ${mean(scores).toFixed(3)} max ${Math.max(...scores).toFixed(3)} | ${((Date.now() - t0) / 1000).toFixed(0)}s`;

    if (opt.stage === 'race' && it % 10 === 0) {
      league.push({ layers, genes: theta.slice(), label: `it${it}` });
      if (league.length > anchors.length + 7) league.splice(anchors.length, 1);
    }
    if (it % opt.evalEvery === 0 || it === opt.iters) {
      // a racer is only as good as its driving: checkpoints are chosen on racing and solo pace together
      const r = await evaluate(), target = opt.stage === 'race' ? r.raceScore + 0.5 * r.score : r.score;
      line += ` || held-out: done ${(r.completed * 100).toFixed(0)}% lap ${r.lap?.toFixed(2)} walls ${r.walls.toFixed(1)} aero-${(100 * r.soloAero).toFixed(0)}%`;
      if (opt.stage === 'race') line += ` | win ${(r.winRate * 100).toFixed(0)}% avgP ${r.avgPlace.toFixed(2)} backWins ${r.fromBackWins} passes ${r.passes.toFixed(1)} rammed ${r.rammed.toFixed(2)} aero-${(100 * r.aero).toFixed(0)}%`;
      if (target > best) {
        best = target;
        bestReport = { iteration: it, ...r };
        save('', { eval: bestReport });
        line += ' *';
      }
    }
    console.log(line);
  }
  save('-final', { eval: bestReport });
  console.log(`[${opt.name}] best held-out`, JSON.stringify(bestReport));
  await Promise.all(pool.map(w => w.terminate()));
})();
