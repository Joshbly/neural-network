#!/usr/bin/env node
// The gate for gradient learning's first test, fixed before the first real run: the policy at a generation against
// the best evolved brains, every one driving the same 32 time trials (each oval once, 6 km, no exploration) with the
// time-trial score evolution trains on (train/lib.js timeTrial). Deterministic, so every number is exact.
//   node train/ppo/tt-report.js slot-N [--gen G]     the latest generation (or G)
//   node train/ppo/tt-report.js slot-N --watch       during a run: a report every 30 minutes as generations sync in,
//                                                    and a last one once the run has ended
//   node train/ppo/tt-report.js --dir PATH [--gen G] a save folder anywhere (a test's)
// PASS: the policy's mean score beats every evolved brain's, and it finishes at least as many ovals as the best of
// them. Writes baselines.json (the evolved brains on the panel, once), tt-report.json (an entry per report) and
// tt-report-GEN.txt into the save.
const fs = require('fs'), path = require('path');
const { run, oval } = require('../lib');
const { PANEL, RACE_PANEL } = require('./spec');

const ROOT = path.join(__dirname, '..', '..'), SLOTS = path.join(ROOT, 'models', 'slots');
const args = process.argv.slice(2), at = args.indexOf('--gen'), dirAt = args.indexOf('--dir');
const DIR = dirAt >= 0 ? path.resolve(args[dirAt + 1]) : path.join(SLOTS, args.find(a => /^slot-\d+$/.test(a)) ?? '');
if (DIR === SLOTS) throw new Error('which save? node train/ppo/tt-report.js slot-N (or --dir PATH)');
const slot = path.basename(DIR), read = (file, dir = DIR) => JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
const write = (file, value) => fs.writeFileSync(path.join(DIR, file), typeof value === 'string' ? value : JSON.stringify(value, null, 1));
const genFile = (dir, gen) => path.join(dir, 'generations', `gen-${String(gen).padStart(4, '0')}.json`);
if (read('meta.json').start !== 'ppo') throw new Error(`${slot} is not a gradient-learning save`);

// the best evolved brains: E vs F's best F and best E at generation 40 (rated 1300 and 1248), and D vs E's champion at 72
const BASELINES = [
  { name: 'F5@40', brain: 'F5', slot: 'slot-5', gen: 40, from: 'E vs F, gen 40: best F (evolution)' },
  { name: 'E9·3@40', brain: 'E9·3', slot: 'slot-5', gen: 40, from: 'E vs F, gen 40: best E' },
  { name: 'E7·8@72', brain: 'E7·8', slot: 'slot-3', gen: 72, from: 'D vs E, gen 72: champion' },
];
// for scale: E vs F (slot-5) evolved full racing for 40 generations on 3 machines at $3.21 an hour each
const EVOLVED_COST = { slot: 'slot-5', gens: 40, machines: 3, perHour: 3.21 }, PPO_PER_HOUR = 1.81;

const pole = trackId => { const t = oval(trackId); return t.length / t.refSpeed / 60; };
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
function panel({ layers, genes }, ovals = PANEL) {
  const trials = ovals.map(({ trackId, laps }) => {
    const r = run({ kind: 'tt', trackId, laps, cars: 'stock', layers, genes: Float32Array.from(genes) });
    return { trackId, score: r.score, finished: r.finished, pace: r.finished && r.lap ? r.lap / pole(trackId) : null, aero: r.aero };
  });
  const paces = trials.filter(t => t.pace != null).map(t => t.pace);
  return { points: mean(trials.map(t => t.score)), finished: trials.filter(t => t.finished).length, ovals: trials.length, pace: mean(paces), aero: mean(trials.map(t => t.aero)), trials };
}

// the evolved brains on the panel, and at race distance (race: score, finished), computed once
function baselines() {
  const file = path.join(DIR, 'baselines.json');
  if (fs.existsSync(file) && JSON.parse(fs.readFileSync(file, 'utf8')).brains.every(b => b.race)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const brains = BASELINES.map(b => {
    const a = JSON.parse(fs.readFileSync(genFile(path.join(SLOTS, b.slot), b.gen), 'utf8')).population.find(x => x.name === b.brain);
    const race = panel(a, RACE_PANEL);
    return { name: b.name, design: a.species, from: b.from, ...panel(a), race: { points: race.points, finished: race.finished } };
  });
  const value = { panel: `all ${PANEL.length} ovals, 6 km each, no exploration (train/lib.js timeTrial)`, made: new Date().toISOString(), brains };
  write('baselines.json', value);
  return value;
}

function report(gen) {
  const policy = JSON.parse(fs.readFileSync(genFile(DIR, gen), 'utf8')).population[0], me = panel(policy), base = baselines().brains;
  const state = read('state.json'), entry = state.history.find(h => h.gen === gen);
  // the learner scored this generation on Modal with the same 32 trials (train/ppo/rollouts.js evaluate): they agree
  const agrees = entry && Math.abs(entry.species.F.points - me.points) < 1e-4 && entry.species.F.finished === me.finished;
  const best = base.reduce((x, y) => y.points > x.points ? y : x);
  const pass = base.every(b => me.points > b.points) && me.finished >= best.finished;
  const vs = base.map(b => {
    const both = me.trials.map((t, k) => [t, b.trials[k]]).filter(([x, y]) => x.pace != null && y.pace != null);
    return { name: b.name, wins: me.trials.filter((t, k) => t.score > b.trials[k].score).length, shared: both.length,
      pace: both.length ? [mean(both.map(([x]) => x.pace)), mean(both.map(([, y]) => y.pace))] : null };
  });
  const hours = entry?.ppo?.hours ?? 0, evolved = read('state.json', path.join(SLOTS, EVOLVED_COST.slot)).history
    .filter(h => h.gen >= 1 && h.gen <= EVOLVED_COST.gens).reduce((s, h) => s + (h.minutes ?? 0), 0) / 60;
  const past = fs.existsSync(path.join(DIR, 'tt-report.json')) ? read('tt-report.json').filter(r => r.gen < gen) : [];
  const record = { gen, at: new Date().toISOString(), iteration: policy.iteration, decisions: entry?.ppo?.decisions ?? null, hours,
    points: me.points, finished: me.finished, pace: me.pace, aero: me.aero, verdict: pass ? 'PASS' : 'NOT YET',
    gap: { points: me.points - best.points, finished: me.finished - best.finished, against: best.name }, vs, agrees };
  // still improving: a new best score in this report or one of the two before it
  const series = [...past, record], bestBefore = Math.max(-Infinity, ...series.slice(0, -3).map(r => r.points));
  record.improving = series.length <= 3 || Math.max(...series.slice(-3).map(r => r.points)) > bestBefore;
  write('tt-report.json', series);

  const f3 = v => v == null ? '   —  ' : v.toFixed(3).padStart(6), pct = v => `${Math.round(v * 50)}%`.padStart(4), decisions = entry?.ppo?.decisions ?? 0;
  const row = (name, s, extra = '') => `  ${name.padEnd(10)} ${f3(s.points)}   ${String(s.finished).padStart(2)}/${s.ovals}   ${s.pace ? `${s.pace.toFixed(3)}×` : '   —  '}  ${pct(s.aero)}  ${extra}`;
  const lines = [
    `Gradient learning, time-trial gate: ${read('meta.json').name} (${slot}), generation ${gen}`,
    `${PANEL.length} ovals, 6 km each, no exploration · iteration ${policy.iteration} · ${(decisions / 1e6).toFixed(decisions < 1e7 ? 1 : 0)}M decisions · ${hours.toFixed(2)} h on one L4 ≈ $${(hours * PPO_PER_HOUR).toFixed(2)}`,
    `(for scale: E vs F evolved full racing for ${EVOLVED_COST.gens} generations in ${evolved.toFixed(1)} h on ${EVOLVED_COST.machines} machines ≈ $${(evolved * EVOLVED_COST.machines * EVOLVED_COST.perHour).toFixed(0)})`,
    '',
    '  brain       score   finished  pace     aero  against the policy',
    row(policy.name, me, agrees ? '(matches the learner\'s own evaluation)' : entry ? '(DIFFERS from the learner\'s evaluation)' : ''),
    ...base.map((b, k) => row(b.name, b, `policy better on ${vs[k].wins}/32${vs[k].pace ? `, pace ${vs[k].pace[0].toFixed(3)}× vs ${vs[k].pace[1].toFixed(3)}× on the ${vs[k].shared} both finished` : ''}`)),
    '',
    `Rule: PASS if the policy's mean score beats every evolved brain's and it finishes at least as many ovals as the best (${best.name}, ${best.finished}).`,
    pass ? `PASS: ${me.points.toFixed(3)} against ${best.name}'s ${best.points.toFixed(3)}, ${me.finished} ovals finished to ${best.finished}. Gradient learning drives solo better than evolution did.`
      : `NOT YET: score ${me.points.toFixed(3)} to ${best.name}'s ${best.points.toFixed(3)} (${record.gap.points >= 0 ? '+' : ''}${record.gap.points.toFixed(3)}), finished ${me.finished} to ${best.finished} (${record.gap.finished >= 0 ? '+' : ''}${record.gap.finished}).`,
    `Trend: ${series.length} report(s); ${record.improving ? 'still improving (a new best score in the last 3 reports)' : 'no new best score in the last 3 reports: worth stopping early'}.`,
    // the learner also drives the panel with the policy's own noise: a big positive gap means it has come to rely on it
    ...entry?.species.F.noisy != null ? [`With its own noise it scores ${entry.species.F.noisy.toFixed(3)} (${entry.species.F.noisyFinished} finished): a gap of ${(entry.species.F.noisy - entry.species.F.points >= 0 ? '+' : '') + (entry.species.F.noisy - entry.species.F.points).toFixed(3)}${entry.species.F.noisy - entry.species.F.points > 0.15 ? ', large: the policy is leaning on its exploration noise' : ''}.`] : [],
    // the distance it practises at, where leaning on the wall gets a car parked
    ...entry?.species.F.race != null ? [`At race distance (24 km): ${entry.species.F.race.toFixed(3)}, ${entry.species.F.raceFinished}/32 finished, ${entry.species.F.raceParked} parked for damage · `
      + base.map(b => `${b.name} ${b.race.points.toFixed(3)} (${b.race.finished}/32)`).join(', ') + '.'] : [],
  ];
  write(`tt-report-${gen}.txt`, `${lines.join('\n')}\n`);
  console.log(lines.join('\n'));
  return record;
}

const latest = () => fs.existsSync(path.join(DIR, 'summary.json')) ? read('summary.json').generation : null;
if (!args.includes('--watch')) {
  const gen = at >= 0 ? +args[at + 1] : latest();
  if (gen == null) throw new Error(`${slot} has no evaluated generation yet`);
  report(gen);
} else {
  // every 30 minutes while it learns; once the run is over (no cloud run on this save), the last generation and stop
  const EVERY = 30 * 60e3, learning = () => fs.existsSync(path.join(SLOTS, 'cloud.json')) && read('cloud.json', SLOTS).slot === slot;
  let lastAt = 0, lastGen = null;
  const tick = () => {
    const gen = latest(), over = !learning();
    if (gen != null && gen !== lastGen && (over || Date.now() - lastAt >= EVERY)) {
      report(gen);
      console.log('');
      [lastAt, lastGen] = [Date.now(), gen];
    }
    if (over) process.exit(0);
  };
  tick();
  setInterval(tick, 60e3);
}
