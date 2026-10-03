#!/usr/bin/env node
// Neuron lab: what each hidden neuron of a brain responds to, and what it does to the car.
//   node train/neurons.js [slot-N] [brain names, e.g. D7@30 E5·2 | --all] [--races 24] [--lesions 12]
// Each brain drives the same full-length 20-car races on a fixed panel of ovals while everything it sensed is
// recorded. Then, neuron by neuron: the inputs and racing situations it fires for, and what holding it at +1, -1
// or its average does to steering and throttle. A decision averages a pass over the real world with a pass over
// its mirror image, so every neuron works twice, once per pass, and is read in the view it sits in: "a car on my
// right" in the mirror pass is a car on the left. The neurons that move the hands most are then silenced for
// whole races. Each layer is also probed for whether the brain can tell a crash is coming.
// Writes models/slots/<slot>/neurons.json, which the app's brain view reads.
const fs = require('fs'), os = require('os'), path = require('path');
const { Worker, isMainThread, parentPort } = require('worker_threads');
const { E, oval } = require('./lib');

const SLOTS = path.join(__dirname, '..', 'models', 'slots');
const PANEL = ['daytona', 'talladega', 'atlanta', 'charlotte', 'michigan', 'kansas', 'darlington', 'dover', 'phoenix', 'richmond', 'bristol', 'martinsville'];
const RACE_M = 24000, FIELD = 20;
// every 8th decision (about 4 frames a second) keeps 24 races to roughly 45,000 frames
const EVERY = 8;
// "a crash is coming": within the next second an impact costs 2 points of downforce between two frames (a quarter
// of a second), or the race ends wrecked or parked. Scraping along a wall wears it away more slowly than that.
const CRASH_STEPS = 60, CRASH_JUMP = 0.02;
const n = E.INPUT_COUNT, I = E.IN, FR = 2 * n + 4;

const arrow = d => d === 180 ? '▼' : d < 0 ? `◀${-d}` : d > 0 ? `${d}▶` : '▲';
const INPUT_NAMES = [
  ...E.WALL_RAY_DEG.map(d => `wall ${arrow(d)}`), ...E.CAR_RAY_DEG.map(d => `car ${arrow(d)}`),
  'closing ▲', 'closing ▼', 'attack ◀▶', 'speed', 'slide', 'yaw', 'contact', 'track pos', 'heading', 'facing',
  ...E.LOOKAHEAD.map(d => `road +${d}`), 'draft', 'last steer', 'last gas', 'position', 'nose dmg', 'tail dmg',
  'track edge', 'paved edge', 'hit in ▲', 'banking', 'bank ahead', 'top steeper',
];
const car = d => I.cars + E.CAR_RAY_DEG.indexOf(d), wall = d => I.walls + E.WALL_RAY_DEG.indexOf(d);
// racing situations, judged on what the pass sees (◀ and ▶ swap in the mirror pass); the last one looks ahead
const CONCEPTS = [
  ['car alongside ◀', x => x[car(-90)] > 0.5],
  ['car alongside ▶', x => x[car(90)] > 0.5],
  ['car on my nose', x => Math.max(x[car(-12)], x[car(0)], x[car(12)]) > 0.5],
  ['closing on the car ahead', x => x[I.ttc] > 0.3],
  ['car on my bumper', x => x[car(180)] > 0.5],
  ['car closing from behind', x => x[I.rearClosing] > 0.05],
  ['in the draft', x => x[I.draft] > 0.4],
  ['road bends ◀', x => x[I.ahead + 3] < -0.08],
  ['road bends ▶', x => x[I.ahead + 3] > 0.08],
  ['wall close ◀', x => x[wall(-90)] > 0.7],
  ['wall close ▶', x => x[wall(90)] > 0.7],
  ['sliding', x => Math.abs(x[I.slide]) > 0.05],
  ['below the yellow line', x => x[I.edge] < 0],
  ['damaged', x => x[I.aero] + x[I.aero + 1] > 0.25],
  ['leading', x => x[I.position] < 0.01],
  ['near the back', x => x[I.position] > 0.8],
  ['slow', x => x[I.speed] < 0.45],
  ['crash within 1 s', null],
];

// ---------- worker: one race, optionally recording one car's every few decisions ----------

if (!isMainThread) {
  let roster = [];
  parentPort.on('message', msg => {
    if (msg.roster) return void (roster = msg.roster);
    parentPort.postMessage({ id: msg.id, out: race(msg.job, roster) });
  });
}

function race({ trackId, field, slot, subject, record }, roster) {
  const laps = E.lapsFor(trackId, RACE_M), drivers = field.map(i => roster[i]);
  drivers.splice(slot, 0, subject);
  const heat = new E.Heat(oval(trackId), drivers.map(d => new E.Brain(d.layers, Float32Array.from(d.genes))), laps,
    E.scenarioOptions({ kind: 'field', trackId, laps, cars: 'stock' }));
  const me = heat.cars[slot], frames = [];
  if (record) {
    const decide = me.decide.bind(me);
    let k = 0;
    me.decide = () => {
      decide();
      if (k++ % EVERY === 0) frames.push(...me.inputs, ...me.mirrored, me.action[0], me.action[1], me.condition.front + me.condition.rear, me.steps);
    };
  }
  while (!heat.over) heat.tick();
  const place = heat.standings().indexOf(me);
  return {
    frames: record ? Float32Array.from(frames) : null,
    place: place / (FIELD - 1), won: place === 0, finished: me.finished, out: !me.finished && (me.retired || me.wrecked || !!me.parked),
    worn: me.condition.front + me.condition.rear, lap: me.laps.length ? Math.min(...me.laps) / 60 : null, passes: me.overtakes, steps: me.steps,
  };
}

function pool(roster) {
  const workers = Array.from({ length: Math.max(1, os.cpus().length - 2) }, () => new Worker(__filename));
  for (const w of workers) w.postMessage({ roster });
  const run = jobs => new Promise(resolve => {
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
  return { run, close: () => Promise.all(workers.map(w => w.terminate())) };
}

// ---------- the network, re-run by hand so any neuron can be held at any value ----------

const squash = x => x <= -3 ? -1 : x >= 3 ? 1 : x * (27 + x * x) / (27 + 9 * x * x);

function net(layers, genes) {
  const L = layers.length - 1, offsets = [0, 0];
  for (let l = 2; l <= L; l++) offsets[l] = offsets[l - 1] + layers[l - 1] * (layers[l - 2] + 1);
  const acts = layers.map(k => new Float32Array(k)), pre = layers.map(k => new Float64Array(k)), tmp = layers.map(k => new Float32Array(k));
  // exactly Brain.thinkJS, keeping the pre-activations (brains from before newer inputs read the ones they know)
  function forward(x) {
    acts[0].set(x.length > layers[0] ? x.subarray(0, layers[0]) : x);
    let k = 0;
    for (let l = 1; l <= L; l++) {
      const src = acts[l - 1], dst = acts[l], p = pre[l];
      for (let j = 0; j < dst.length; j++) {
        let sum = genes[k++];
        for (let i = 0; i < src.length; i++) sum += src[i] * genes[k++];
        p[j] = sum;
        dst[j] = squash(sum);
      }
    }
    return acts[L];
  }
  // the outputs with neuron j of layer l held at v, after the last forward()
  function held(l, j, v) {
    const d = v - acts[l][j], span = layers[l] + 1;
    let src = tmp[l + 1];
    for (let k = 0; k < layers[l + 1]; k++) src[k] = squash(pre[l + 1][k] + genes[offsets[l + 1] + k * span + 1 + j] * d);
    for (let m = l + 2; m <= L; m++) {
      const dst = tmp[m];
      for (let k = 0; k < dst.length; k++) {
        let o = offsets[m] + k * (layers[m - 1] + 1), sum = genes[o++];
        for (let i = 0; i < src.length; i++) sum += src[i] * genes[o++];
        dst[k] = squash(sum);
      }
      src = dst;
    }
    return src;
  }
  // a brain whose neuron j of layer l always outputs v: its outgoing weights folded into the next layer's biases
  function lesioned(l, j, v) {
    const g = Float32Array.from(genes), span = layers[l] + 1;
    for (let k = 0; k < layers[l + 1]; k++) {
      const o = offsets[l + 1] + k * span;
      g[o] += g[o + 1 + j] * v;
      g[o + 1 + j] = 0;
    }
    return g;
  }
  return { forward, held, lesioned, acts, L };
}

// ---------- analysis ----------

function auc(scores, labels) {
  const idx = Array.from(scores.keys()).sort((a, b) => scores[a] - scores[b]);
  let pos = 0, rankSum = 0;
  idx.forEach((k, r) => { if (labels[k]) { pos++; rankSum += r + 1; } });
  const neg = labels.length - pos;
  return pos && neg ? (rankSum - pos * (pos + 1) / 2) / (pos * neg) : null;
}

// a linear readout trained on two thirds of the races and scored on the rest: can this layer tell what's coming?
// test: 0 trains, 1 tests, 2 sits out
function probe(features, dim, labels, test) {
  const rows = labels.length, mean = new Float64Array(dim), sd = new Float64Array(dim);
  let pos = 0, train = 0;
  for (let r = 0; r < rows; r++) if (!test[r]) {
    train++;
    pos += labels[r];
    for (let d = 0; d < dim; d++) mean[d] += features[r * dim + d];
  }
  for (let d = 0; d < dim; d++) mean[d] /= train;
  for (let r = 0; r < rows; r++) if (!test[r]) for (let d = 0; d < dim; d++) sd[d] += (features[r * dim + d] - mean[d]) ** 2;
  for (let d = 0; d < dim; d++) sd[d] = Math.sqrt(sd[d] / train) || 1;
  const Z = Float32Array.from(features, (v, k) => (v - mean[k % dim]) / sd[k % dim]);
  // crashes are rare: weigh them up so the readout can't win by always saying "no"
  const wPos = (train - pos) / Math.max(1, pos), w = new Float64Array(dim), grad = new Float64Array(dim);
  let b = 0;
  const score = r => { let s = b; for (let d = 0, o = r * dim; d < dim; d++) s += w[d] * Z[o + d]; return s; };
  for (let epoch = 0; epoch < 150; epoch++) {
    grad.fill(0);
    let gb = 0, total = 0;
    for (let r = 0; r < rows; r++) {
      if (test[r]) continue;
      const weight = labels[r] ? wPos : 1, err = weight * (1 / (1 + Math.exp(-score(r))) - labels[r]);
      for (let d = 0, o = r * dim; d < dim; d++) grad[d] += err * Z[o + d];
      gb += err;
      total += weight;
    }
    for (let d = 0; d < dim; d++) w[d] -= 0.5 * (grad[d] / total + 1e-3 * w[d]);
    b -= 0.5 * gb / total;
  }
  const scores = [], truth = [];
  for (let r = 0; r < rows; r++) if (test[r] === 1) {
    scores.push(score(r));
    truth.push(labels[r]);
  }
  return auc(scores, truth);
}

function analyse(brain, runs) {
  const { layers } = brain, genes = Float32Array.from(brain.genes), N = net(layers, genes), L = N.L;
  const hidden = layers.slice(1, L), H = hidden.reduce((a, b) => a + b, 0);
  const starts = hidden.map((_, h) => hidden.slice(0, h).reduce((a, b) => a + b, 0));
  const where = Array.from({ length: H }, (_, h) => { const l = starts.findLastIndex(s => s <= h); return [l + 1, h - starts[l]]; });
  const total = runs.reduce((s, r) => s + r.frames.length / FR, 0), S = 2 * total;
  // per sample (frame f, pass p: s = 2f + p): what the pass saw and every hidden activation
  const X = new Float32Array(S * n), A = new Float32Array(S * H), crash = new Uint8Array(total), raceOf = new Uint16Array(total);
  const gas = new Float32Array(total);
  let f = 0, matched = 0;
  runs.forEach((run, ri) => {
    const F = run.frames, m = F.length / FR;
    for (let k = 0; k < m; k++, f++) {
      const o = k * FR, steps = F[o + 2 * n + 3];
      let soon = run.out && run.steps - steps <= CRASH_STEPS;
      for (let g = k + 1; !soon && g < m && F[g * FR + 2 * n + 3] - steps <= CRASH_STEPS; g++) soon = F[g * FR + 2 * n + 2] - F[(g - 1) * FR + 2 * n + 2] >= CRASH_JUMP;
      crash[f] = soon;
      raceOf[f] = ri;
      gas[f] = F[o + 2 * n + 1];
      const outs = [];
      for (let p = 0; p < 2; p++) {
        const x = F.subarray(o + p * n, o + (p + 1) * n), s = 2 * f + p;
        X.set(x, s * n);
        outs.push(...N.forward(x));
        for (let h = 0; h < hidden.length; h++) A.set(N.acts[h + 1], s * H + starts[h]);
      }
      // the hand-run network has to reproduce the decision the car actually made, bit for bit
      matched += Math.fround((outs[0] - outs[2]) / 2) === F[o + 2 * n] && Math.fround((outs[1] + outs[3]) / 2) === F[o + 2 * n + 1];
    }
  });

  // activity
  const mean = new Float64Array(H), sd = new Float64Array(H), sat = new Float64Array(H);
  for (let s = 0; s < S; s++) for (let h = 0; h < H; h++) { const a = A[s * H + h]; mean[h] += a; sat[h] += Math.abs(a) > 0.95; }
  for (let h = 0; h < H; h++) { mean[h] /= S; sat[h] /= S; }
  for (let s = 0; s < S; s++) for (let h = 0; h < H; h++) sd[h] += (A[s * H + h] - mean[h]) ** 2;
  for (let h = 0; h < H; h++) sd[h] = Math.sqrt(sd[h] / S);

  // which inputs each neuron follows (correlation over every sample)
  const xm = new Float64Array(n), xs = new Float64Array(n), cov = new Float64Array(H * n);
  for (let s = 0; s < S; s++) for (let i = 0; i < n; i++) xm[i] += X[s * n + i];
  for (let i = 0; i < n; i++) xm[i] /= S;
  for (let s = 0; s < S; s++) for (let i = 0; i < n; i++) xs[i] += (X[s * n + i] - xm[i]) ** 2;
  for (let i = 0; i < n; i++) xs[i] = Math.sqrt(xs[i] / S);
  const dx = new Float64Array(n);
  for (let s = 0; s < S; s++) {
    for (let i = 0; i < n; i++) dx[i] = X[s * n + i] - xm[i];
    for (let h = 0; h < H; h++) {
      const a = A[s * H + h] - mean[h];
      if (!a) continue;
      for (let i = 0; i < n; i++) cov[h * n + i] += a * dx[i];
    }
  }
  const corr = (h, i) => xs[i] > 1e-6 && sd[h] > 1e-6 ? cov[h * n + i] / S / (xs[i] * sd[h]) : 0;

  // situations: how far (in standard deviations) each neuron moves when one is happening
  const C = CONCEPTS.length, inC = new Uint8Array(S * C), count = new Float64Array(C);
  for (let s = 0; s < S; s++) {
    const x = X.subarray(s * n, (s + 1) * n);
    CONCEPTS.forEach(([, test], c) => { const yes = test ? test(x) : crash[s >> 1]; inC[s * C + c] = yes; count[c] += yes; });
  }
  const sumIn = new Float64Array(H * C);
  for (let s = 0; s < S; s++) for (let c = 0; c < C; c++) if (inC[s * C + c]) for (let h = 0; h < H; h++) sumIn[h * C + c] += A[s * H + h];
  const shift = (h, c) => {
    const k = count[c];
    if (k < 0.005 * S || k > 0.995 * S || sd[h] < 1e-6) return 0;
    const meanIn = sumIn[h * C + c] / k, meanOut = (mean[h] * S - sumIn[h * C + c]) / (S - k);
    return (meanIn - meanOut) / sd[h];
  };

  // what each neuron does to the hands: held at +1, at -1 and at its average, on frames spread over every race;
  // and, situation by situation, what it adds to the hands over sitting at its average
  const picks = Math.min(total, 3000), push = new Float64Array(H * 2), plus = new Float64Array(H * 2), minus = new Float64Array(H * 2), imp = new Float64Array(H * 2);
  const dReal = new Float64Array(H * 2), adds = new Float64Array(H * C * 2), addsSeen = new Float64Array(C);
  for (let q = 0; q < picks; q++) {
    const fr = Math.floor(q * total / picks);
    for (let p = 0; p < 2; p++) {
      const s = 2 * fr + p, base = Array.from(N.forward(X.subarray(s * n, (s + 1) * n)));
      const happening = CONCEPTS.flatMap((_, c) => inC[s * C + c] ? [c] : []);
      for (const c of happening) addsSeen[c]++;
      for (let h = 0; h < H; h++) {
        const [l, j] = where[h];
        const up = Array.from(N.held(l, j, 1)), down = Array.from(N.held(l, j, -1)), avg = N.held(l, j, mean[h]);
        for (let o = 0; o < 2; o++) {
          // each pass makes half the decision
          push[h * 2 + o] += (up[o] - down[o]) / 4;
          plus[h * 2 + o] += (up[o] - base[o]) / 2;
          minus[h * 2 + o] += (down[o] - base[o]) / 2;
          for (const c of happening) adds[(h * C + c) * 2 + o] += (base[o] - avg[o]) / 2;
        }
        // silenced in both passes at once, the way a lesioned brain drives: the mirror pass's steering counts reversed
        if (p === 0) { dReal[h * 2] = avg[0] - base[0]; dReal[h * 2 + 1] = avg[1] - base[1]; }
        else {
          imp[h * 2] += Math.abs(dReal[h * 2] - (avg[0] - base[0])) / 2;
          imp[h * 2 + 1] += Math.abs(dReal[h * 2 + 1] + (avg[1] - base[1])) / 2;
        }
      }
    }
  }
  for (const v of [push, plus, minus]) for (let k = 0; k < v.length; k++) v[k] /= 2 * picks;
  for (let k = 0; k < imp.length; k++) imp[k] /= picks;

  const neurons = where.map(([l, j], h) => {
    const inputs = Array.from({ length: n }, (_, i) => [i, corr(h, i)]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 4).filter(([, r]) => Math.abs(r) > 0.15);
    // [situation, how far it moves then (sd), what it adds to steering and throttle then]
    const concepts = CONCEPTS.map(([name], c) => [name, shift(h, c), ...[0, 1].map(o => addsSeen[c] >= 20 ? adds[(h * C + c) * 2 + o] / addsSeen[c] : 0)])
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 4).filter(([, d]) => Math.abs(d) > 0.3);
    const nr = { l, j, mean: mean[h], sd: sd[h], sat: sat[h], inputs, concepts, push: [push[h * 2], push[h * 2 + 1]], plus: [plus[h * 2], plus[h * 2 + 1]], minus: [minus[h * 2], minus[h * 2 + 1]], imp: [imp[h * 2], imp[h * 2 + 1]] };
    nr.label = labelOf(nr);
    return nr;
  });

  // can the brain tell a crash is coming? A readout per layer, trained on two thirds of the races. Once a car is
  // sliding, pointing off line or against a wall the crash is obvious, so the fair test only uses moments it's
  // still under control.
  const real = (fr, i) => X[2 * fr * n + i];
  const control = Uint8Array.from({ length: total }, (_, fr) => Math.abs(real(fr, I.slide)) < 0.05 && Math.abs(real(fr, I.heading)) < 0.15 && real(fr, I.facing) > 0
    && !real(fr, I.contact) && real(fr, wall(-90)) < 0.7 && real(fr, wall(90)) < 0.7);
  const layerOf = (dim, get) => {
    const out = new Float32Array(total * dim);
    for (let fr = 0; fr < total; fr++) for (let d = 0; d < dim; d++) out[fr * dim + d] = get(fr, d);
    return out;
  };
  const features = [['inputs', n, (fr, d) => real(fr, d)], ...hidden.map((width, h) => [`layer ${h + 1}`, width, (fr, d) => A[2 * fr * H + starts[h] + d]])].map(([name, dim, get]) => [name, dim, layerOf(dim, get)]);
  const probes = {};
  for (const [kind, keep] of [['any moment', () => true], ['under control', fr => control[fr]]]) {
    const test = Uint8Array.from(raceOf, (r, fr) => !keep(fr) ? 2 : r % 3 === 2 ? 1 : 0), held = Array.from(crash.keys()).filter(fr => test[fr] === 1);
    probes[kind] = { 'hit in ▲ alone': auc(held.map(fr => real(fr, I.ttc)), held.map(fr => crash[fr])) };
    for (const [name, dim, Z] of features) probes[kind][name] = probe(Z, dim, crash, test);
  }

  const avgOf = keep => { let s = 0, k = 0; for (let fr = 0; fr < total; fr++) if (keep(fr)) { s += gas[fr]; k++; } return k ? s / k : null; };
  const behaviour = {
    gasBeforeCrash: avgOf(fr => crash[fr]), gasOtherwise: avgOf(fr => !crash[fr]),
    gasBeforeCrashInControl: avgOf(fr => crash[fr] && control[fr]), gasInControl: avgOf(fr => !crash[fr] && control[fr]),
    gasClosingFast: avgOf(fr => real(fr, I.ttc) > 0.5), gasClear: avgOf(fr => real(fr, I.ttc) === 0),
    crashShare: crash.reduce((a, b) => a + b, 0) / total,
  };
  let inControl = 0, crashesInControl = 0;
  for (let fr = 0; fr < total; fr++) if (control[fr]) { inControl++; crashesInControl += crash[fr]; }
  behaviour.crashInControl = crashesInControl / Math.max(1, inControl);
  behaviour.controlShare = inControl / total;
  const situations = Object.fromEntries(CONCEPTS.map(([name], c) => [name, count[c] / S]));
  return { frames: total, matched, neurons, probes, behaviour, situations, N };
}

// "in this situation, this neuron makes the car do that": the situation it reacts to most strongly, and what it
// adds to the hands while that's happening
function labelOf({ concepts, inputs, push }) {
  const [when, s, g] = concepts.length ? [concepts[0][0], concepts[0][2], concepts[0][3]]
    : inputs.length ? [`${INPUT_NAMES[inputs[0][0]]} ${inputs[0][1] > 0 ? 'high' : 'low'}`, ...push] : ['nothing clear', ...push];
  const big = Math.max(Math.abs(s), Math.abs(g)), hands = [];
  if (Math.abs(s) > 0.004 && Math.abs(s) > 0.3 * big) hands.push(`steer ${s > 0 ? '▶' : '◀'}`);
  if (Math.abs(g) > 0.004 && Math.abs(g) > 0.3 * big) hands.push(g > 0 ? 'more gas' : 'lift');
  return `${when} → ${hands.join(', ') || 'barely moves the hands'}`;
}

const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const tally = runs => ({
  place: mean(runs.map(r => r.place)), wins: runs.filter(r => r.won).length, finished: mean(runs.map(r => +r.finished)), out: mean(runs.map(r => +r.out)),
  worn: mean(runs.map(r => r.worn)), lap: mean(runs.filter(r => r.lap).map(r => r.lap)), passes: mean(runs.map(r => r.passes)),
});
const pct = v => `${Math.round(v * 100)}%`;
const line = t => `place ${t.place.toFixed(2)} · wins ${t.wins} · finished ${pct(t.finished)} · wrecked/parked ${pct(t.out)} · downforce lost ${pct(t.worn / 2)}`;

async function main() {
  const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? +args[i + 1] : d; };
  const words = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'));
  const slot = words.find(a => /^slot-\d+$/.test(a)) ?? JSON.parse(fs.readFileSync(path.join(SLOTS, 'active.json'))).id;
  const dir = path.join(SLOTS, slot), meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json')));
  if (meta.tracks !== 'nascar') throw new Error('the neuron lab races the NASCAR ovals: pick a NASCAR save');
  const state = JSON.parse(fs.readFileSync(path.join(dir, 'state.json'))), races = opt('--races', 24), lesionCount = opt('--lesions', 12);
  const known = [...state.population, ...(state.hall ?? [])], names = words.filter(w => w !== slot);
  const best = Object.values(Object.groupBy(state.population, b => b.species)).map(group => group.sort((a, b) => (a.last?.avgPlace ?? 1) - (b.last?.avgPlace ?? 1))[0]);
  // --all: everyone racing plus the hall of fame, except brains already analysed with lesions (a quick batch
  // shouldn't replace a full study)
  const studied = new Set(Object.entries(fs.existsSync(path.join(dir, 'neurons.json')) ? JSON.parse(fs.readFileSync(path.join(dir, 'neurons.json'))).brains : {})
    .filter(([, b]) => b.neurons.some(nr => nr.lesion)).map(([name]) => name));
  const subjects = args.includes('--all') ? known.filter(b => !studied.has(b.name))
    : names.length ? names.map(name => known.find(b => b.name === name) ?? (() => { throw new Error(`no brain called ${name} in ${slot}`); })()) : best;
  const roster = state.population.map(b => ({ layers: b.layers, genes: Array.from(b.genes) }));

  // the panel: every oval twice, once from the front half of the grid and once from the back, against the
  // current population (never against a copy of itself)
  const panel = subject => Array.from({ length: races }, (_, r) => {
    const rng = E.mulberry32(9000 + r), rivals = state.population.map((b, i) => [b, i]).filter(([b]) => b.name !== subject.name).map(([, i]) => i);
    for (let i = rivals.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [rivals[i], rivals[j]] = [rivals[j], rivals[i]]; }
    const front = Math.floor(r / PANEL.length) % 2 === 0;
    return { trackId: PANEL[r % PANEL.length], field: rivals.slice(0, FIELD - 1), slot: front ? 1 + (r * 7) % 9 : 10 + (r * 7) % 10 };
  });

  const workers = pool(roster), started = Date.now(), file = path.join(dir, 'neurons.json');
  // merged into what's there after every brain, so a long batch keeps what it finished
  const save = (name, entry) => {
    const atlas = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file)) : { brains: {} };
    fs.writeFileSync(file, JSON.stringify({ built: new Date().toISOString(), generation: state.generation, races, concepts: CONCEPTS.map(([c]) => c), inputs: INPUT_NAMES, brains: { ...atlas.brains, [name]: entry } }, (k, v) => typeof v === 'number' ? +v.toFixed(4) : v));
  };
  console.log(`${slot} (generation ${state.generation}): ${subjects.map(b => b.name).join(', ')} · ${races} races of ${RACE_M / 1000} km each`);
  for (const b of subjects) {
    const subject = { layers: b.layers, genes: Array.from(b.genes) }, jobs = panel(b);
    const runs = await workers.run(jobs.map(j => ({ ...j, subject, record: true })));
    const R = analyse(b, runs), intact = tally(runs);
    const H = R.neurons.length, dead = R.neurons.filter(nr => nr.sd < 0.03).length, stuck = R.neurons.filter(nr => nr.sat > 0.8).length;
    // steering dwarfs throttle, so the throttle movers get their own share of the lesions
    const by = o => R.neurons.slice().sort((a, b) => b.imp[o] - a.imp[o]);
    const chosen = new Set([...by(0).slice(0, Math.ceil(lesionCount * 0.6)), ...by(1)]);
    // placebo: a live neuron that barely touches the hands, so its races show how much is just chaos
    const placebo = R.neurons.filter(nr => nr.sd > 0.05).sort((a, b) => (a.imp[0] + a.imp[1]) - (b.imp[0] + b.imp[1]))[0];
    const lesions = lesionCount ? [...[...chosen].slice(0, lesionCount), ...placebo ? [placebo] : []] : [];
    const lesionRuns = await workers.run(lesions.flatMap(nr => jobs.map(j => ({ ...j, subject: { layers: b.layers, genes: Array.from(R.N.lesioned(nr.l, nr.j, nr.mean)) }, record: false }))));
    lesions.forEach((nr, k) => { nr.lesion = tally(lesionRuns.slice(k * races, (k + 1) * races)); });
    if (placebo) placebo.placebo = true;

    console.log(`\n${b.name} (${b.layers.slice(1, -1).join('-')}): ${R.frames} frames; the hand-run network reproduces ${pct(R.matched / R.frames)} of its decisions exactly`);
    console.log(`  intact: ${line(intact)}`);
    console.log(`  activity: ${dead} of ${H} neurons barely move, ${stuck} sit at ±1 more than 80% of the time`);
    console.log('  can it tell a crash is coming within a second? (AUC: 0.5 = no idea, 1 = always)');
    for (const [kind, scores] of Object.entries(R.probes)) console.log(`    ${kind.padEnd(14)} ${Object.entries(scores).map(([k, v]) => `${k} ${v?.toFixed(2)}`).join(' · ')}`);
    const bh = R.behaviour;
    console.log(`  a crash follows within a second in ${pct(bh.crashShare)} of moments (${pct(bh.crashInControl)} of the ${pct(bh.controlShare)} it's under control)`);
    console.log(`  gas in the second before a crash ${bh.gasBeforeCrash?.toFixed(2)} vs ${bh.gasOtherwise.toFixed(2)} otherwise; still under control ${bh.gasBeforeCrashInControl?.toFixed(2)} vs ${bh.gasInControl?.toFixed(2)}; closing fast (hit in > 0.5) ${bh.gasClosingFast?.toFixed(2)} vs clear road ${bh.gasClear?.toFixed(2)}`);
    console.log(`  situations: ${Object.entries(R.situations).map(([k, v]) => `${k} ${pct(v)}`).join(', ')}`);
    // the hit-in input's wiring: the neurons that follow it most and what they do when closing on the car ahead
    const ttcIdx = I.ttc, closing = 'closing on the car ahead';
    const ttcNeurons = R.neurons.filter(nr => nr.inputs.some(([i]) => i === ttcIdx)).sort((a, b) => Math.abs(b.inputs.find(([i]) => i === ttcIdx)[1]) - Math.abs(a.inputs.find(([i]) => i === ttcIdx)[1])).slice(0, 4);
    console.log(`  neurons that follow "hit in ▲": ${ttcNeurons.map(nr => { const c = nr.concepts.find(([name]) => name === closing); return `L${nr.l}·${nr.j + 1} (r ${nr.inputs.find(([i]) => i === ttcIdx)[1].toFixed(2)}${c ? `, adds steer ${c[2].toFixed(3)} gas ${c[3].toFixed(3)} when closing` : ''})`; }).join(', ') || 'none'}`);
    console.log('  the neurons that move its hands most, and races with each one silenced:');
    for (const nr of lesions) {
      const t = nr.lesion;
      console.log(`    ${nr.placebo ? 'placebo ' : ''}L${nr.l}·${String(nr.j + 1).padEnd(3)} ${nr.label.padEnd(46)} moves steer ${nr.imp[0].toFixed(3)} gas ${nr.imp[1].toFixed(3)} | silenced: ${line(t)}`);
    }
    const { N, ...rest } = R;
    save(b.name, { design: b.species, layers: b.layers, analysedAt: state.generation, intact, ...rest, neurons: R.neurons });
  }
  await workers.close();
  console.log(`\nwrote ${file} in ${((Date.now() - started) / 1000).toFixed(0)} s`);
}

if (isMainThread) main();
