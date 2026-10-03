#!/usr/bin/env node
// How the traffic graft experiment (train/graft.js) is going, and its verdict, judged by rules fixed in advance.
//   node train/graft-report.js [slot-4] [--apply-boost]
// 1. the tournaments, generation by generation: race points per race, wins and damage, T against E
// 2. the rating ladder: each design's best, on the same scale as the save it came from
// 3. a fixed evaluation: 2 independent sets of 96 full races, all 20 experiment brains plus the 20 sparring partners
//    in every race (40 cars, 12 ovals): points per race by design with a bootstrap standard error, the best T
//    against the best E head to head, finishes, wrecks and damage
// 4. the mechanism: how big the traffic block's connections onward have grown, and where hit-in, the closing speeds
//    and the car sensors rank among the brain's own 44 senses (a one-spread nudge to each, best T against best E);
//    inputs added to the engine after the graft don't reach these brains and don't count
// Rules (set before the run):
//   checkpoint, generation 77 to 81: if the block's connections onward average under 0.02, the block isn't being
//     used: --apply-boost doubles the training nudge on it (mask.boost 2), recorded in meta.json as a deviation
//   verdict, generation 92 on: the design works if T beats E by more than 2 standard errors in the evaluation, the
//     best T finishes ahead of the best E in at least 55% of shared races, and in the best T at least one of
//     hit-in or the two closing speeds ranks in the top half of the senses; promising if only that last part
//     holds; rejected at this budget if neither
// Writes graft-report.json in the save (one entry per generation reported).
const fs = require('fs'), os = require('os'), path = require('path');
const { Worker, isMainThread, parentPort } = require('worker_threads');
const { E, run, racePoints } = require('./lib');
const { PANEL, recordFrames } = require('./graft');

const SLOTS = path.join(__dirname, '..', 'models', 'slots'), I = E.IN, n = E.INPUT_COUNT;
const CHECKPOINT = [77, 82], VERDICT = 92, MIN_ONWARD = 0.02, RACE_M = 24000;

if (!isMainThread && require.main === module) parentPort.on('message', ({ id, job }) => parentPort.postMessage({ id, out: run(job) }));

function pool() {
  const workers = Array.from({ length: Math.max(1, os.cpus().length - 2) }, () => new Worker(__filename));
  const runAll = jobs => new Promise(resolve => {
    const out = new Array(jobs.length);
    let next = 0, done = 0;
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

const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const rms = xs => Math.sqrt(mean(xs.map(x => x * x)));
const pct = v => `${Math.round(100 * v)}%`;

// the block's connections into the next layer (from its neurons, in every next-layer neuron)
function onward(b) {
  const { layer, from, to } = b.mask, L = b.layers;
  let at = 0;
  for (let l = 1; l <= layer; l++) at += L[l] * (L[l - 1] + 1);
  const span = L[layer] + 1, w = [];
  for (let k = 0; k < L[layer + 1]; k++) for (let j = from; j < to; j++) w.push(b.genes[at + k * span + 1 + j]);
  return rms(w);
}

// the engine's own mirror tables, so a nudged moment's mirrored pass sees the nudge too
const decide = (brain, x) => { const m = new Float32Array(n); for (let i = 0; i < n; i++) m[i] = E.MIRROR_SIGN[i] * x[E.MIRROR_FROM[i]]; const p = brain.thinkPair(m, x); return [(p[2] - p[0]) / 2, (p[3] + p[1]) / 2]; };

// each of the brain's senses nudged by its own spread over recorded moments: how far the hands move; rank 1 = moves them most
function senseRanks(b, frames) {
  const brain = new E.Brain(b.layers, Float32Array.from(b.genes)), xs = frames.map(f => f.x), base = xs.map(x => decide(brain, x)), senses = b.layers[0];
  const sd = Array.from({ length: senses }, (_, i) => { const m = mean(xs.map(x => x[i])); return Math.sqrt(mean(xs.map(x => (x[i] - m) ** 2))); });
  const effect = Array.from({ length: senses }, (_, i) => {
    let d = 0;
    xs.forEach((x, f) => { const y = Float32Array.from(x); y[i] += sd[i]; const [s, g] = decide(brain, y); d += Math.abs(s - base[f][0]) + Math.abs(g - base[f][1]); });
    return d / xs.length;
  });
  const order = effect.map((e, i) => [e, i]).sort((a, c) => c[0] - a[0]).map(([, i]) => i), rank = i => order.indexOf(i) + 1;
  const cars = Array.from({ length: E.CAR_RAY_DEG.length }, (_, r) => I.cars + r);
  return { of: senses, hitIn: rank(I.ttc), closingAhead: rank(I.closing), closingBehind: rank(I.rearClosing), bestCarSensor: Math.min(...cars.map(rank)) };
}

async function main() {
  const args = process.argv.slice(2), slot = args.find(a => /^slot-\d+$/.test(a)) ?? 'slot-4', dir = path.join(SLOTS, slot);
  const st = JSON.parse(fs.readFileSync(path.join(dir, 'state.json'), 'utf8')), meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8'));
  if (!meta.graft) throw new Error(`${slot} is not a graft experiment (train/graft.js)`);
  const gen = st.generation, start = meta.graft.generation;
  const T = st.population.filter(b => b.species === 'T'), Ed = st.population.filter(b => b.species === 'E');
  console.log(`${meta.name}, generation ${gen} (${gen - start} trained since the graft)`);

  // 1. the tournaments
  console.log('\n1. tournaments: race points per race (wins), downforce lost');
  const history = st.history.filter(h => h.gen >= start).map(h => {
    const by = d => h.agents.filter(a => a.species === d);
    return { gen: h.gen, T: mean(by('T').map(a => a.points)), E: mean(by('E').map(a => a.points)), wT: by('T').reduce((s, a) => s + a.wins, 0), wE: by('E').reduce((s, a) => s + a.wins, 0), aT: mean(by('T').map(a => a.aero)), aE: mean(by('E').map(a => a.aero)) };
  });
  for (const h of history) console.log(`  gen ${h.gen}: T ${h.T.toFixed(3)} (${h.wT}) · E ${h.E.toFixed(3)} (${h.wE}) · T-E ${(h.T - h.E >= 0 ? '+' : '') + (h.T - h.E).toFixed(3)} · damage T ${pct(h.aT / 2)} E ${pct(h.aE / 2)}`);

  // 2. the ladder
  const rating = JSON.parse(fs.readFileSync(path.join(dir, 'rating.json'), 'utf8'));
  const rated = rating.players.filter(p => p.gen >= start && p.role === 'best' && ['T', 'E'].includes(p.design));
  console.log(`\n2. rating ladder (best of each design): ${rated.map(p => `${p.name} ${p.r}±${p.sd}`).join(', ') || 'none yet'}`);

  // 3. the fixed evaluation
  const drivers = [...st.population, ...st.hall].map(b => ({ layers: b.layers, genes: Array.from(b.genes) })), P = st.population.length;
  const plan = [];
  for (const set of [1, 2]) {
    const rng = E.mulberry32(9200 + set);
    for (let k = 0; k < 96; k++) {
      const order = drivers.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      const trackId = PANEL[k % PANEL.length];
      plan.push({ set, order, job: { kind: 'field', trackId, laps: E.lapsFor(trackId, RACE_M), cars: 'stock', drivers: order.map(i => drivers[i]) } });
    }
  }
  const workers = pool(), results = await workers.runAll(plan.map(p => p.job));
  await workers.close();
  // per race: each experiment brain's points, place and fate
  const races = plan.map((p, k) => {
    const of = i => results[k][p.order.indexOf(i)];
    return st.population.map((_, i) => ({ points: racePoints(of(i).place, p.order.length), place: of(i).place, finished: of(i).finished, out: !of(i).finished && (of(i).retired || !!of(i).parked), aero: of(i).aero }));
  });
  const idx = d => st.population.map((b, i) => b.species === d ? i : -1).filter(i => i >= 0);
  const armRace = (d, r) => mean(idx(d).map(i => races[r][i].points)), diffs = races.map((_, r) => armRace('T', r) - armRace('E', r));
  const boot = Array.from({ length: 2000 }, (_, t) => { const rng = E.mulberry32(t + 1); return mean(diffs.map(() => diffs[Math.floor(rng() * diffs.length)])); });
  const diff = mean(diffs), se = Math.sqrt(mean(boot.map(b => (b - mean(boot)) ** 2)));
  const brainPoints = st.population.map((_, i) => mean(races.map(r => r[i].points)));
  const best = d => idx(d).reduce((a, i) => brainPoints[i] > brainPoints[a] ? i : a, idx(d)[0]), bT = best('T'), bE = best('E');
  const ahead = mean(races.map(r => +(r[bT].place < r[bE].place)));
  const arm = d => ({ points: mean(races.map((_, r) => armRace(d, r))), finished: mean(races.flatMap(r => idx(d).map(i => +r[i].finished))), out: mean(races.flatMap(r => idx(d).map(i => +r[i].out))), aero: mean(races.flatMap(r => idx(d).map(i => r[i].aero))) / 2 });
  const sets = [1, 2].map(set => { const rs = races.map((_, r) => r).filter(r => plan[r].set === set); return { T: mean(rs.map(r => armRace('T', r))), E: mean(rs.map(r => armRace('E', r))) }; });
  const aT = arm('T'), aE = arm('E');
  console.log(`\n3. evaluation: 192 races of 40 cars (the experiment plus its sparring partners)`);
  console.log(`  points per race: T ${aT.points.toFixed(3)} · E ${aE.points.toFixed(3)} · T-E ${diff >= 0 ? '+' : ''}${diff.toFixed(3)} ± ${se.toFixed(3)} (${(diff / se).toFixed(1)} standard errors) · sets ${sets.map(s => `${s.T.toFixed(3)}/${s.E.toFixed(3)}`).join(' and ')}`);
  console.log(`  best T ${st.population[bT].name} ${brainPoints[bT].toFixed(3)} vs best E ${st.population[bE].name} ${brainPoints[bE].toFixed(3)}: T ahead in ${pct(ahead)} of the 192 races`);
  console.log(`  finished T ${pct(aT.finished)} E ${pct(aE.finished)} · wrecked or parked T ${pct(aT.out)} E ${pct(aE.out)} · downforce lost T ${pct(aT.aero)} E ${pct(aE.aero)}`);

  // 4. the mechanism
  const onwardSizes = T.map(onward);
  const rivals = st.hall.slice(0, 19);
  const ranks = { T: senseRanks(st.population[bT], recordFrames(st.population[bT], rivals)), E: senseRanks(st.population[bE], recordFrames(st.population[bE], rivals)) };
  console.log(`\n4. mechanism: the block's connections onward ${onwardSizes.map(x => x.toFixed(3)).join(' ')} (mean ${mean(onwardSizes).toFixed(3)}; started at 0)`);
  for (const d of ['T', 'E']) { const r = ranks[d]; console.log(`  ${d === 'T' ? 'best T' : 'best E'} sense ranks (of ${r.of}): hit-in #${r.hitIn} · closing ahead #${r.closingAhead} · closing behind #${r.closingBehind} · best car sensor #${r.bestCarSensor}`); }

  // the rules
  const mechanism = Math.min(ranks.T.hitIn, ranks.T.closingAhead, ranks.T.closingBehind) <= ranks.T.of / 2;
  let call = null;
  if (gen >= CHECKPOINT[0] && gen < CHECKPOINT[1]) {
    const fires = mean(onwardSizes) < MIN_ONWARD, boosted = T.some(b => (b.mask.boost ?? 1) !== 1);
    call = fires ? (boosted ? 'checkpoint: the boost is already on' : `checkpoint: the block's connections onward average ${mean(onwardSizes).toFixed(3)} < ${MIN_ONWARD}: the boost rule fires (run with --apply-boost while the save isn't training)`) : `checkpoint: the block is being used (connections onward ${mean(onwardSizes).toFixed(3)} ≥ ${MIN_ONWARD}), no change`;
    if (fires && !boosted && args.includes('--apply-boost')) {
      const cloud = fs.existsSync(path.join(SLOTS, 'cloud.json')) && JSON.parse(fs.readFileSync(path.join(SLOTS, 'cloud.json'), 'utf8'));
      if (cloud?.slot === slot) throw new Error(`${slot} is training on Modal: stop it before changing the save`);
      for (const b of st.population) if (b.mask) b.mask = { ...b.mask, boost: 2 };
      fs.writeFileSync(path.join(dir, 'state.json'), JSON.stringify(st));
      fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({ ...meta, deviations: [...meta.deviations ?? [], { generation: gen, change: 'mask.boost 2', reason: `traffic block connections onward averaged ${mean(onwardSizes).toFixed(4)} < ${MIN_ONWARD}` }] }));
      call += ' · applied: boost 2, recorded in meta.json';
    }
  }
  if (gen >= VERDICT) {
    const points = diff > 2 * se, headToHead = ahead >= 0.55;
    call = points && headToHead && mechanism ? 'VERDICT: the new design works (T beats E clearly, the best T beats the best E, and it uses the traffic senses)'
      : mechanism ? `VERDICT: promising, needs more training (it uses the traffic senses, but ${points ? '' : 'T does not beat E by 2 standard errors'}${!points && !headToHead ? ' and ' : ''}${headToHead ? '' : 'the best T does not beat the best E 55% of the time'})`
        : `VERDICT: rejected at this budget (${points && headToHead ? 'T wins, but not through the traffic senses' : 'no clear win and no sign it uses the traffic senses'})`;
  }
  if (call) console.log(`\n${call}`);

  const file = path.join(dir, 'graft-report.json'), reports = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : {};
  reports[gen] = { at: new Date().toISOString(), history, rated, evaluation: { T: aT, E: aE, diff, se, sets, bestT: st.population[bT].name, bestE: st.population[bE].name, ahead }, onward: onwardSizes, ranks, mechanism, call };
  fs.writeFileSync(file, JSON.stringify(reports, (k, v) => typeof v === 'number' ? +v.toFixed(4) : v));
  console.log(`\nwrote ${file}`);
}

if (isMainThread) main().catch(err => { console.error(err.message); process.exit(1); });
