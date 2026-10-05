#!/usr/bin/env node
// The learning-speed pilot's verdict (train/modal_launch.py pilot): which settings the real gradient-learning run uses.
//   node train/ppo/pilot-report.js [models/ppo-pilot]
// The rule, fixed before the pilot ran: each setting is judged by the mean of its last two noise-free evaluations (all
// 32 ovals). A setting is out if, at the end, its own noise lifts its score by more than 0.05 (it's leaning on the noise
// again) or it has fallen more than 0.15 below its best (unstable). The highest end score wins; within 0.02 of it, the
// setting closest to today's defaults. Also: the slope over the last 50 iterations, and at that slope how long the
// winner would take to reach the gate (F5@40's 1.349).
// Writes pilot-report.txt and pilot-report.json there.
const fs = require('fs'), path = require('path');

const DIR = path.resolve(process.argv[2] ?? path.join(__dirname, '..', '..', 'models', 'ppo-pilot'));
const GATE = 1.349, ORDER = ['pilot-defaults', 'pilot-lr', 'pilot-looser', 'pilot-hold4'];
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const runs = fs.readdirSync(DIR).filter(d => d.startsWith('pilot-') && fs.existsSync(path.join(DIR, d, 'state.json'))).map(name => {
  const state = JSON.parse(fs.readFileSync(path.join(DIR, name, 'state.json'), 'utf8'));
  const metrics = fs.readFileSync(path.join(DIR, name, 'ppo', 'metrics.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
  const c = state.ppo.config, evals = state.history.map(h => ({ iteration: h.ppo.iteration, ...h.species.F }));
  const last = evals.slice(-2), final = evals.at(-1), best = Math.max(...evals.map(e => e.points));
  // least-squares slope of the evaluations over the last 50 iterations, per 100 iterations
  const recent = evals.filter(e => e.iteration >= final.iteration - 50), mx = mean(recent.map(e => e.iteration)), my = mean(recent.map(e => e.points));
  const slope = 100 * recent.reduce((s, e) => s + (e.iteration - mx) * (e.points - my), 0) / Math.max(1e-9, recent.reduce((s, e) => s + (e.iteration - mx) ** 2, 0));
  const out = final.noisy - final.points > 0.05 ? 'leans on its noise' : best - final.points > 0.15 ? 'unstable' : null;
  const seconds = mean(metrics.filter(m => !m.warmup).map(m => m.collect_s + m.update_s));
  return { name, c, evals, metrics, end: mean(last.map(e => e.points)), endAt: last.map(e => e.iteration), best, slope, out, seconds, gap: final.noisy - final.points };
}).sort((a, b) => ORDER.indexOf(a.name) - ORDER.indexOf(b.name));
if (!runs.length) throw new Error(`no pilot runs in ${DIR}`);

const describe = r => r.name === 'pilot-defaults' ? 'defaults' : r.name === 'pilot-looser' ? `KL ${r.c.kl_target}, clip ${r.c.clip}` : r.name === 'pilot-lr' ? `rate ${r.c.lr_driver}` : r.name === 'pilot-hold4' ? `hold ${r.c.hold}, steer ${[r.c.sigma0].flat()[0]}` : r.name;
const start = runs[0].evals[0].points, same = runs.every(r => r.evals[0].iteration === 0 && r.evals[0].points === start);
const qualified = runs.filter(r => !r.out), best = qualified.length ? Math.max(...qualified.map(r => r.end)) : null;
const pick = qualified.length ? qualified.filter(r => r.end >= best - 0.02).sort((a, b) => ORDER.indexOf(a.name) - ORDER.indexOf(b.name))[0] : null;

const f3 = v => v == null ? '     —' : v.toFixed(3).padStart(6), label = r => describe(r).padEnd(18);
const checkpoints = [...new Set(runs.flatMap(r => r.evals.map(e => e.iteration)))].sort((a, b) => a - b).filter((k, i, all) => k % 30 === 0 || i === all.length - 1);
const lines = [
  `Gradient learning speed pilot: ${runs.length} settings, the same starting weights, up to ${Math.max(...runs.map(r => r.metrics.length))} iterations of 600k+ decisions`,
  '',
  'Noise-free evaluation on all 32 ovals (mean score, ovals finished) by iteration:',
  `  setting           ${checkpoints.map(k => `it ${k}`.padStart(12)).join(' ')}   end score`,
  ...runs.map(r => `  ${label(r)}${checkpoints.map(k => { const e = r.evals.find(x => x.iteration === k); return (e ? `${f3(e.points)} ${String(e.finished).padStart(2)}/32` : '—').padStart(12); }).join(' ')}   ${f3(r.end)}${r.endAt.length === 2 ? ` (it ${r.endAt.join(', ')})` : ''}`),
  '',
  '  setting            pace at end   best    slope /100 it   noise gap   mean KL   clip    learning rate at end   s/iteration',
  ...runs.map(r => {
    const e = r.evals.at(-1), m = r.metrics.filter(x => !x.warmup);
    return `  ${label(r)} ${e.pace ? `${e.pace.toFixed(3)}×` : '   —  '}       ${f3(r.best)}   ${(r.slope >= 0 ? '+' : '') + r.slope.toFixed(3).padStart(6)}          ${(r.gap >= 0 ? '+' : '') + r.gap.toFixed(3)}      ${mean(m.map(x => x.kl)).toFixed(4)}    ${mean(m.map(x => x.clip_frac)).toFixed(3)}   ${m.at(-1).lr_driver.toExponential(1).padEnd(22)} ${r.seconds.toFixed(1)}${r.out ? `   OUT: ${r.out}` : ''}`;
  }),
  '',
  `Start: every setting began at ${start.toFixed(3)}${same ? ', identical across them, as it should be' : ': NOT identical across the runs, check the pilot'}.`,
  'Rule: the highest end score (mean of the last two evaluations), among settings not leaning on their noise (gap at most +0.05) and not unstable (at most 0.15 below their best); within 0.02, the one closest to the defaults.',
  pick ? `Pick: ${describe(pick)} (end score ${pick.end.toFixed(3)}).` : 'No pick: every setting is out.',
  ...pick ? [pick.slope > 0
    ? `At its last-50-iteration slope (+${pick.slope.toFixed(3)} per 100 iterations) it would reach the gate's ${GATE} in about ${Math.ceil((GATE - pick.evals.at(-1).points) / pick.slope * 100)} more iterations: ${((GATE - pick.evals.at(-1).points) / pick.slope * 100 * pick.seconds / 3600).toFixed(1)} h on one L4 at ${pick.seconds.toFixed(1)} s an iteration (a straight-line guess; learning usually slows).`
    : 'It was no longer improving over its last 50 iterations: the real run would not reach the gate as things stand.'] : [],
];
fs.writeFileSync(path.join(DIR, 'pilot-report.txt'), `${lines.join('\n')}\n`);
fs.writeFileSync(path.join(DIR, 'pilot-report.json'), JSON.stringify({ start, identicalStart: same, pick: pick && { name: pick.name, setting: describe(pick), config: pick.c, end: pick.end, slope: pick.slope },
  runs: runs.map(r => ({ name: r.name, setting: describe(r), end: r.end, best: r.best, slope: r.slope, gap: r.gap, out: r.out, evals: r.evals })) }, null, 1));
console.log(lines.join('\n'));
