#!/usr/bin/env node
// The contract between the race workers (Node) and the learner (Python) for gradient learning: what one recorded
// decision looks like, what the critic sees, the brain being trained and how it computes, and the ovals.
//   node train/ppo/spec.js > ppo-spec.json
// The learner reads only this, so the inputs, the brain's shape and the row layout have one source: the engine.
const { E, SHAPES } = require('../lib');

// Practice runs a race's distance (24 km, as evolution's practice and season do). Over 6 km leaning on the outside
// wall all the way round survives (0.7 of the 1.0 damage that gets a car parked) and finished 32 ovals of 32, so the
// from-scratch policy learned to do exactly that; over 24 km race control parks it on 14 of them.
const METRES = 24000;
// the gate (train/ppo/tt-report.js) and the evaluation every generation: every oval once, no exploration, 6 km; and
// the same at race distance
const GATE_METRES = 6000;
const PANEL = E.NASCAR_TRACKS.map(t => ({ trackId: t.id, laps: E.lapsFor(t.id, GATE_METRES) }));
const RACE_PANEL = E.NASCAR_TRACKS.map(t => ({ trackId: t.id, laps: E.lapsFor(t.id, METRES) }));
// design F: 160 lane neurons on the road and self senses, 48 traffic neurons on the traffic senses, 48 mixing
// neurons, and shortcuts from the first layer straight to steer and gas
const DESIGN = { name: 'F', layers: [E.INPUT_COUNT, 208, 48, 2], mask: SHAPES.F.mask, shortcuts: SHAPES.F.shortcuts };

// what only the critic sees, measured at the decision (time-trial version; races will add race state)
const EXTRAS = [
  { name: 'distance covered', scale: 'share of the run, 0 to 1' },
  { name: 'time left', scale: '1 - steps / maxSteps: the finish bonus if the car finished now' },
  { name: 'lap length', scale: 'metres / 4300' },
  { name: 'reference speed', scale: 'pole speed / the speed input scale (6.5 units a step)' },
  { name: 'plate package', scale: '0 or 1' },
  { name: 'laps', scale: 'laps in the run / 8' },
];
function extrasOf(car, heat, out, at) {
  const t = heat.track;
  out[at] = E.clamp((car.progress + car.bonus + car.gridOffset) / (heat.laps * t.length), 0, 1);
  out[at + 1] = 1 - car.steps / heat.maxSteps;
  out[at + 2] = t.length * 0.25 / 4300;
  out[at + 3] = t.refSpeed / car.spec.speedNorm;
  out[at + 4] = t.plate ? 1 : 0;
  out[at + 5] = heat.laps / 8;
}

// one decision: the senses, the critic's extras, the try before clamping, the decision before noise, the try's
// log-probability under the policy that raced, the reward that followed, and 1 on an episode's last row
const n = E.INPUT_COUNT, x = n + EXTRAS.length;
const ROW = { obs: [0, n], extras: [n, x], raw: [x, x + 2], mean: [x + 2, x + 4], logp: x + 4, reward: x + 5, done: x + 6, width: x + 7 };

// where each layer's neurons start in the weights: each neuron is [bias, a weight from each input], and with
// shortcuts each output reads the last hidden layer then the first
function geneLayout({ layers, shortcuts }) {
  const L = layers.length - 1, out = [];
  let at = 0;
  for (let l = 1; l <= L; l++) {
    const inputs = layers[l - 1] + (shortcuts && l === L ? layers[1] : 0);
    out.push({ layer: l, neurons: layers[l], inputs, reads: shortcuts && l === L ? [l - 1, 1] : [l - 1], at });
    at += layers[l] * (inputs + 1);
  }
  return out;
}

const spec = {
  version: 1,
  task: `one stock car alone, ${METRES} m against the clock on a NASCAR oval`,
  metres: METRES,
  gateMetres: GATE_METRES,
  inputs: n,
  // the mirror image of the inputs: mirrored[i] = sign[i] * inputs[from[i]]; a decision averages both views
  mirror: { from: Array.from(E.MIRROR_FROM), sign: Array.from(E.MIRROR_SIGN) },
  decision: 'steer = (real[0] - mirror[0]) / 2, gas = (real[1] + mirror[1]) / 2',
  // every neuron: squash(bias + sum of weight x input), squash(x) = -1 at or below -3, 1 at or above 3, else
  // x (27 + x^2) / (27 + 9 x^2); float32 weights and activations, float64 sums
  squash: { a: 27, b: 9, clip: 3 },
  design: { ...DESIGN, genes: E.geneCount(DESIGN.layers, DESIGN.shortcuts), layout: geneLayout(DESIGN) },
  extras: EXTRAS,
  row: ROW,
  // weights race as published: 5 decimals
  publishedDecimals: 5,
  tracks: E.NASCAR_TRACKS.map(t => t.id),
  panel: PANEL,
  racePanel: RACE_PANEL,
};

module.exports = { METRES, GATE_METRES, PANEL, RACE_PANEL, DESIGN, EXTRAS, ROW, extrasOf, spec };
if (require.main === module) process.stdout.write(`${JSON.stringify(spec, null, 1)}\n`);
