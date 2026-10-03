#!/usr/bin/env node
// How a head-to-head of two designs from scratch is going (the E vs F save, meta.league), judged by rules fixed
// before the run.
//   node train/league-report.js slot-N
// 1. the tournaments, generation by generation: race points per race and wins by design
// 2. the rating ladder: each design's best at every rating
// 3. a fixed evaluation: 2 independent sets of 96 full races of the 20 brains on 12 ovals: points per race by design
//    with a bootstrap standard error, the best of each design head to head, finishes, wrecks and damage
// 4. the senses: where pace, drift, the closing speeds and hit-in rank in each design's best brain (a one-spread
//    nudge to each of its senses), i.e. whether the new senses get used
// 5. room: how many independent directions each hidden layer of the two best brains actually uses (90% of the
//    variation in its activity), so a loss can be told apart from running out of neurons
// Rules (set before the run): checkpoints at generations 20 and 30 report only; at generation 40 the verdict: a
// design wins if it leads the other by more than 2 standard errors in the evaluation and its best rates at least as
// high as the other's best; otherwise it's too close to call at 40 generations. The senses and the room are reported
// alongside, not counted.
// Writes league-report.json and league-report-<generation>.txt in the save.
const fs = require('fs'), os = require('os'), path = require('path');
const { Worker, isMainThread, parentPort } = require('worker_threads');
const { E, run, racePoints } = require('./lib');
const { PANEL, recordFrames } = require('./graft');

const SLOTS = path.join(__dirname, '..', 'models', 'slots'), I = E.IN, n = E.INPUT_COUNT, RACE_M = 24000;

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
const pct = v => `${Math.round(100 * v)}%`;
const signed = (v, d = 3) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;

// a decision as the car makes it: the real view and its mirror image, averaged
const decide = (brain, x) => {
  const m = new Float32Array(n);
  for (let i = 0; i < n; i++) m[i] = E.MIRROR_SIGN[i] * x[E.MIRROR_FROM[i]];
  const p = brain.thinkPair(m, x);
  return [(p[2] - p[0]) / 2, (p[3] + p[1]) / 2];
};

// each of the brain's senses nudged by its own spread over recorded moments: how far the hands move; rank 1 = most
function senseRanks(b, frames) {
  const brain = new E.Brain(b.layers, Float32Array.from(b.genes)), xs = frames.map(f => f.x), base = xs.map(x => decide(brain, x)), senses = b.layers[0];
  const effect = Array.from({ length: senses }, (_, i) => {
    const mu = mean(xs.map(x => x[i])), sd = Math.sqrt(mean(xs.map(x => (x[i] - mu) ** 2)));
    let d = 0;
    xs.forEach((x, f) => { const y = Float32Array.from(x); y[i] += sd; const [s, g] = decide(brain, y); d += Math.abs(s - base[f][0]) + Math.abs(g - base[f][1]); });
    return d / xs.length;
  });
  const order = effect.map((e, i) => [e, i]).sort((a, c) => c[0] - a[0]).map(([, i]) => i), rank = i => order.indexOf(i) + 1;
  const best = list => Math.min(...list.map(rank)), stretch = (start, count) => Array.from({ length: count }, (_, k) => start + k);
  return { of: senses, pace: best(stretch(I.pace, E.PACE_RAY_DEG.length)), drift: best(stretch(I.drift, E.DRIFT_RAY_DEG.length)), closingAhead: rank(I.closing), closingBehind: rank(I.rearClosing), hitIn: rank(I.ttc) };
}

// eigenvalues of a covariance (Jacobi rotations), and how many directions hold 90% of the variation
function directions(rows) {
  const k = rows[0].length, m = rows.length, mu = Array.from({ length: k }, (_, j) => mean(rows.map(r => r[j])));
  const C = Array.from({ length: k }, () => new Float64Array(k));
  for (const r of rows) for (let a = 0; a < k; a++) { const da = r[a] - mu[a]; if (da) for (let b = a; b < k; b++) C[a][b] += da * (r[b] - mu[b]); }
  for (let a = 0; a < k; a++) for (let b = a; b < k; b++) C[b][a] = C[a][b] /= m;
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) off += C[p][q] * C[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) {
      if (Math.abs(C[p][q]) < 1e-15) continue;
      const th = 0.5 * Math.atan2(2 * C[p][q], C[q][q] - C[p][p]), c = Math.cos(th), s = Math.sin(th);
      for (let r = 0; r < k; r++) { const a = C[r][p], b = C[r][q]; C[r][p] = c * a - s * b; C[r][q] = s * a + c * b; }
      for (let r = 0; r < k; r++) { const a = C[p][r], b = C[q][r]; C[p][r] = c * a - s * b; C[q][r] = s * a + c * b; }
    }
  }
  const ev = C.map((r, i) => Math.max(0, r[i])).sort((a, b) => b - a), total = ev.reduce((a, b) => a + b, 0);
  let used = 0, acc = 0;
  while (acc < 0.9 * total) acc += ev[used++];
  return used;
}

// the layers of a brain worth asking about: each hidden layer, with design F's lane and traffic blocks apart
function room(b, frames) {
  const brain = new E.Brain(b.layers, Float32Array.from(b.genes)), acts = frames.filter((_, i) => i % 2 === 0).map(f => { brain.thinkJS(f.x); return brain.acts.map(a => Array.from(a)); });
  const parts = [];
  for (let l = 1; l < b.layers.length - 1; l++) {
    const blocks = [b.mask ?? []].flat().filter(k => k.layer === l);
    const whole = b.mask && l === b.layers.length - 2 ? 'mixing layer' : `layer ${l}`;
    for (const { name, from, to } of blocks.length ? blocks : [{ name: whole, from: 0, to: b.layers[l] }])
      parts.push({ name: blocks.length ? `${name} block` : name, neurons: to - from, used: directions(acts.map(a => a[l].slice(from, to))) });
  }
  return parts;
}

async function main() {
  const where = process.argv[2], slot = where && path.basename(where), dir = where && (/^slot-\d+$/.test(where) ? path.join(SLOTS, where) : path.resolve(where));
  if (!dir) throw new Error('usage: node train/league-report.js slot-N (or a save directory)');
  const st = JSON.parse(fs.readFileSync(path.join(dir, 'state.json'), 'utf8')), meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json'), 'utf8'));
  if (!meta.league) throw new Error(`${slot} is not a head-to-head league (meta.league)`);
  const [A, B] = meta.league.designs, gen = st.generation, lines = [], say = (s = '') => { console.log(s); lines.push(s); };
  say(`${meta.name}, generation ${gen}: ${A} vs ${B}`);

  // 1. the tournaments
  say('\n1. tournaments: race points per race (wins)');
  const history = st.history.map(h => {
    const by = d => h.agents.filter(a => a.species === d);
    return { gen: h.gen, ...Object.fromEntries([A, B].flatMap(d => [[d, mean(by(d).map(a => a.points))], [`w${d}`, by(d).reduce((s, a) => s + a.wins, 0)]])) };
  });
  for (const h of history.filter(h => h.gen % 5 === 0 || h.gen > gen - 5)) say(`  gen ${h.gen}: ${A} ${h[A].toFixed(3)} (${h[`w${A}`]}) · ${B} ${h[B].toFixed(3)} (${h[`w${B}`]}) · ${B}-${A} ${signed(h[B] - h[A])}`);
  const recent = history.slice(-5);
  say(`  last ${recent.length}: ${A} ${mean(recent.map(h => h[A])).toFixed(3)} · ${B} ${mean(recent.map(h => h[B])).toFixed(3)} · wins ${A} ${recent.reduce((s, h) => s + h[`w${A}`], 0)} ${B} ${recent.reduce((s, h) => s + h[`w${B}`], 0)}`);

  // 2. the ladder
  const rating = JSON.parse(fs.readFileSync(path.join(dir, 'rating.json'), 'utf8'));
  const bests = rating.players.filter(p => p.role === 'best' && [A, B].includes(p.design) && p.gen <= gen);
  const ratedAt = Math.max(...bests.map(p => p.gen)), bestRated = d => bests.filter(p => p.gen === ratedAt && p.design === d).sort((x, y) => y.r - x.r)[0];
  say(`\n2. rating ladder (best of each design):`);
  for (const g of [...new Set(bests.map(p => p.gen))].sort((x, y) => x - y)) say(`  gen ${g}: ${bests.filter(p => p.gen === g).map(p => `${p.design} ${p.name} ${p.r}±${p.sd}`).join(' · ')}`);

  // 3. the fixed evaluation
  const drivers = st.population.map(b => ({ layers: b.layers, genes: Array.from(b.genes) })), plan = [];
  for (const set of [1, 2]) {
    const rng = E.mulberry32(9400 + set);
    for (let k = 0; k < 96; k++) {
      const order = drivers.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      const trackId = PANEL[k % PANEL.length];
      plan.push({ set, order, job: { kind: 'field', trackId, laps: E.lapsFor(trackId, RACE_M), cars: 'stock', drivers: order.map(i => drivers[i]) } });
    }
  }
  const workers = pool(), results = await workers.runAll(plan.map(p => p.job));
  await workers.close();
  const races = plan.map((p, k) => st.population.map((_, i) => {
    const r = results[k][p.order.indexOf(i)];
    return { points: racePoints(r.place, p.order.length), place: r.place, finished: r.finished, out: !r.finished && (r.retired || !!r.parked), aero: r.aero };
  }));
  const idx = d => st.population.map((b, i) => b.species === d ? i : -1).filter(i => i >= 0);
  const armRace = (d, r) => mean(idx(d).map(i => races[r][i].points)), diffs = races.map((_, r) => armRace(B, r) - armRace(A, r));
  const boot = Array.from({ length: 2000 }, (_, t) => { const rng = E.mulberry32(t + 1); return mean(diffs.map(() => diffs[Math.floor(rng() * diffs.length)])); });
  const diff = mean(diffs), se = Math.sqrt(mean(boot.map(b => (b - mean(boot)) ** 2)));
  const brainPoints = st.population.map((_, i) => mean(races.map(r => r[i].points)));
  const best = d => idx(d).reduce((a, i) => brainPoints[i] > brainPoints[a] ? i : a, idx(d)[0]), bA = best(A), bB = best(B);
  const ahead = mean(races.map(r => +(r[bB].place < r[bA].place)));
  const arm = d => ({ points: mean(races.map((_, r) => armRace(d, r))), finished: mean(races.flatMap(r => idx(d).map(i => +r[i].finished))), out: mean(races.flatMap(r => idx(d).map(i => +r[i].out))), aero: mean(races.flatMap(r => idx(d).map(i => r[i].aero))) / 2 });
  const sets = [1, 2].map(set => { const rs = races.map((_, r) => r).filter(r => plan[r].set === set); return { [A]: mean(rs.map(r => armRace(A, r))), [B]: mean(rs.map(r => armRace(B, r))) }; });
  const aA = arm(A), aB = arm(B);
  say(`\n3. evaluation: 192 races of the ${drivers.length} brains`);
  say(`  points per race: ${A} ${aA.points.toFixed(3)} · ${B} ${aB.points.toFixed(3)} · ${B}-${A} ${signed(diff)} ± ${se.toFixed(3)} (${(diff / se).toFixed(1)} standard errors) · sets ${sets.map(s => `${s[A].toFixed(3)}/${s[B].toFixed(3)}`).join(' and ')}`);
  say(`  best ${A} ${st.population[bA].name} ${brainPoints[bA].toFixed(3)} vs best ${B} ${st.population[bB].name} ${brainPoints[bB].toFixed(3)}: ${B} ahead in ${pct(ahead)} of the races`);
  say(`  finished ${A} ${pct(aA.finished)} ${B} ${pct(aB.finished)} · wrecked or parked ${A} ${pct(aA.out)} ${B} ${pct(aB.out)} · downforce lost ${A} ${pct(aA.aero)} ${B} ${pct(aB.aero)}`);

  // 4 and 5: the two best brains, recorded against the rest of the field
  const study = {};
  for (const [d, i] of [[A, bA], [B, bB]]) {
    const b = st.population[i], frames = recordFrames(b, st.population.filter(x => x !== b));
    study[d] = { name: b.name, senses: senseRanks(b, frames), room: room(b, frames) };
  }
  say(`\n4. senses (rank among the brain's ${n}; 1 moves the hands most):`);
  for (const d of [A, B]) { const s = study[d].senses; say(`  ${d} ${study[d].name}: pace #${s.pace} · drift #${s.drift} · closing ahead #${s.closingAhead} · closing behind #${s.closingBehind} · hit-in #${s.hitIn}`); }
  say('\n5. room: independent directions used (90% of the activity), of the neurons there');
  for (const d of [A, B]) say(`  ${d} ${study[d].name}: ${study[d].room.map(p => `${p.name} ${p.used} of ${p.neurons}`).join(' · ')}`);

  // the rules
  let call = `checkpoint at generation ${gen}: reported only (the verdict comes at generation ${meta.league.verdict})`;
  if (gen >= meta.league.verdict) {
    const rA = bestRated(A)?.r ?? -Infinity, rB = bestRated(B)?.r ?? -Infinity;
    call = diff > 2 * se && rB >= rA ? `VERDICT: ${B} wins (leads by ${(diff / se).toFixed(1)} standard errors, best rated ${rB} vs ${rA})`
      : diff < -2 * se && rA >= rB ? `VERDICT: ${A} wins (leads by ${(-diff / se).toFixed(1)} standard errors, best rated ${rA} vs ${rB})`
        : `VERDICT: too close to call at ${gen} generations (${B}-${A} ${signed(diff)} ± ${se.toFixed(3)}, best rated ${A} ${rA} vs ${B} ${rB})`;
  }
  say(`\n${call}`);

  const file = path.join(dir, 'league-report.json'), reports = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : {};
  reports[gen] = { at: new Date().toISOString(), history, rated: bests, evaluation: { [A]: aA, [B]: aB, diff, se, sets, best: { [A]: st.population[bA].name, [B]: st.population[bB].name }, ahead }, study, call };
  fs.writeFileSync(file, JSON.stringify(reports, (k, v) => typeof v === 'number' ? +v.toFixed(4) : v));
  fs.writeFileSync(path.join(dir, `league-report-${gen}.txt`), `${lines.join('\n')}\n`);
  console.log(`\nwrote ${file}`);
}

if (isMainThread) main().catch(err => { console.error(err.message); process.exit(1); });
