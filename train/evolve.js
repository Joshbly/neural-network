#!/usr/bin/env node
// The evolution: the pros keep getting better on every core until you stop it.
//
// Population-based training. Twenty pros in five species, one per brain design. Every generation:
//   1. training: each pro gets the same number of evolution-strategies steps, racing rivals drawn from
//      the whole population (every design), so they learn racecraft from each other;
//   2. tournament: all twenty race 10-lap, 20-car races on tracks none of them train on;
//   3. selection: within each species, a pro that keeps finishing well behind its best sibling is
//      replaced by a mutated copy of that sibling (new learning rate and noise scale too).
// Species never mix (the weights don't fit across designs) and every species keeps four seats, so the
// standings answer "which brain design is best" with equal training for each.
//
// State lives in models/evolution/ and survives restarts. The founders are models/originals/tab-field.json,
// which is never written to.
//   node train/evolve.js [--pairs 48] [--rounds 5] [--workers N]
const os = require('os');
const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');
const { E, noise, widen, initParams, racePoints } = require('./lib');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const opt = {
  pairs: +(args.pairs || 48), rounds: +(args.rounds || 5), workers: +(args.workers || Math.max(1, os.cpus().length - 1)),
  field: 12, laps: 3, races: 2, decay: 0.005,
};
const ROOT = path.join(__dirname, '..');
const DIR = path.resolve(args.dir || path.join(ROOT, 'models', 'evolution')), GENS = path.join(DIR, 'generations');
const FOUNDERS = path.join(ROOT, 'models', 'originals', 'tab-field.json');
const range = (from, n) => Array.from({ length: n }, (_, i) => from + i);
// Practice tracks never repeat: every training round of every generation gets new ones, shared by all
// twenty pros that round. The tournament gets 8 new tracks each generation, raced 3 times each with
// different grids so neither grid luck nor one odd layout decides who survives. Only the yardstick
// against the originals stays on the same 8 tracks, so every generation sits the same test.
const practiceTrack = (gen, round, k) => 10_000_000 + (gen * 16 + round) * 8 + k;
const tourneyTrack = (gen, k) => 5_000_000 + gen * 8 + k;
const TOURNEY_TRACKS = 8, BENCH_TRACKS = range(931, 8);
const TOURNEY_RACES = 24, BENCH_RACES = 8, RACE_LAPS = 10, DUEL_LAPS = 2;
// A save races normal tracks or the real NASCAR ovals, in normal cars or stock cars (meta.json; any mix
// trains). On the ovals a race is a distance, not a lap count: practice about 5.5 km (two laps of Daytona,
// six of Martinsville), duels 3 km, the season's races 24 km, so a short track isn't over in seconds.
const META = (() => { try { return JSON.parse(fs.readFileSync(path.join(DIR, 'meta.json'), 'utf8')); } catch { return {}; } })();
const MODE = { tracks: META.tracks ?? 'normal', cars: META.cars ?? (META.tracks === 'nascar' ? 'stock' : 'normal') };
const NASCAR = MODE.tracks === 'nascar', CARS = MODE.cars === 'normal' && !NASCAR ? {} : { cars: MODE.cars };
const PRACTICE_M = 5500, DUEL_M = 3000, SEASON_M = 24000, NASCAR_FIELD = 40;
// where a race is: a generated track's seed, or a real oval and a lap count for the distance
const onTrack = (seedOrId, laps, metres) => NASCAR ? { trackId: seedOrId, laps: E.lapsFor(seedOrId, metres), ...CARS } : { trackSeed: seedOrId, laps, ...CARS };
// avg finish is a share of the field (0 = always first); 24 races put roughly ±0.06 of noise on it
const MARGIN = 0.1, SETTLE = 2;
const SPECIES = { '16-10': 'A', '32-24-16': 'B', '64-64': 'C', '64-64-64': 'D', '128-128': 'E' };
const speciesOf = layers => SPECIES[layers.slice(1, -1).join('-')] || layers.slice(1, -1).join('-');
fs.mkdirSync(GENS, { recursive: true });

const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const round = (v, d = 3) => v == null ? null : +v.toFixed(d);
const shuffle = (list, rng) => {
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
};
const writeJson = (file, value) => {
  fs.writeFileSync(`${file}.tmp`, JSON.stringify(value));
  fs.renameSync(`${file}.tmp`, file);
};

// ---- worker pool: one shared queue, idle slots pull the next job ----
// A slot is one racing thread: a local worker, or one thread of a remote train/worker-server.js
// (--remote host:port --token T), which together let one evolution use more than one machine.
const slots = [], idle = [], queue = [], pending = new Map(), remotes = [];
let nextId = 0;
const pump = () => {
  while (idle.length && queue.length) {
    const slot = idle.pop(), { job, resolve } = queue.shift(), id = nextId++;
    pending.set(id, { resolve, slot, job });
    slot.post({ type: 'job', id, job });
  }
};
const finished = (id, out) => {
  const p = pending.get(id);
  if (!p) return;
  pending.delete(id);
  idle.push(p.slot);
  p.resolve(out);
  pump();
};
const locals = Array.from({ length: opt.workers }, () => new Worker(path.join(__dirname, 'evolve-worker.js')));
for (const w of locals) {
  const slot = { post: msg => w.postMessage(msg) };
  slots.push(slot);
  idle.push(slot);
  w.on('message', ({ id, out }) => finished(id, out));
  w.on('error', err => {
    console.error(err);
    process.exit(1);
  });
}
const submit = job => new Promise(resolve => {
  queue.push({ job, resolve });
  pump();
});
// the latest roster and each pro's latest weights, so a machine that joins mid-generation can catch up
let lastRoster = null;
const thetas = new Map();
const broadcast = msg => {
  if (msg.type === 'roster') {
    lastRoster = msg;
    thetas.clear();
  }
  if (msg.type === 'theta') thetas.set(msg.agent, msg);
  locals.forEach(w => w.postMessage(msg));
  remotes.forEach(r => r.send(msg));
};

// A remote machine gets the current roster and weights before any job, so it can join at any time.
// --remote-file is re-read every 10 s: a helper that Modal restarts comes back under a new address.
const known = new Set();
function watchRemotes(file, token) {
  const check = () => {
    if (!fs.existsSync(file)) return;
    for (const address of fs.readFileSync(file, 'utf8').split('\n').map(s => s.trim()).filter(Boolean))
      if (!known.has(address)) {
        known.add(address);
        connectRemote(address, token);
      }
  };
  check();
  setInterval(check, 10e3).unref();
}

function connectRemote(address, token) {
  const net = require('net'), [host, port] = address.split(':');
  return new Promise(resolve => {
    const sock = net.connect(+port, host), remote = { slots: [] };
    remote.send = msg => sock.write(JSON.stringify(msg, (k, v) => ArrayBuffer.isView(v) ? Array.from(v) : v) + '\n');
    sock.setNoDelay(true);
    sock.setEncoding('utf8');
    let buf = '';
    sock.on('connect', () => remote.send({ type: 'hello', token }));
    sock.on('data', chunk => {
      buf += chunk;
      for (let i; (i = buf.indexOf('\n')) >= 0;) {
        const msg = JSON.parse(buf.slice(0, i));
        buf = buf.slice(i + 1);
        if (msg.type !== 'ready') {
          finished(msg.id, msg.out);
          continue;
        }
        if (lastRoster) remote.send(lastRoster);
        for (const theta of thetas.values()) remote.send(theta);
        for (let k = 0; k < msg.threads; k++) remote.slots.push({ post: remote.send, remote });
        slots.push(...remote.slots);
        idle.push(...remote.slots);
        remotes.push(remote);
        console.log(`[evolve] ${msg.threads} remote threads joined from ${address}`);
        resolve(remote);
      }
    });
    // if the machine drops out, its unfinished jobs go back in the queue and the local threads carry on
    const drop = () => {
      if (!remotes.includes(remote)) return resolve(null);
      remotes.splice(remotes.indexOf(remote), 1);
      for (const list of [slots, idle]) for (let i = list.length - 1; i >= 0; i--) if (list[i].remote === remote) list.splice(i, 1);
      for (const [id, p] of pending) if (p.slot.remote === remote) {
        pending.delete(id);
        queue.unshift({ job: p.job, resolve: p.resolve });
      }
      console.log(`[evolve] remote ${address} dropped; continuing with ${slots.length} threads`);
      pump();
    };
    sock.on('error', drop);
    sock.on('close', drop);
  });
}

// ---- state ----
// state.json: the population after the last finished generation (also kept per generation in generations/)
// live.json: everyone's weights right now, mid-generation, so the app can show the practice as it happens
// progress.json: phase, round, and exactly which practice races are being run this round
// summary.json: a few numbers for the save-slot list
const STATE = path.join(DIR, 'state.json'), PROGRESS = path.join(DIR, 'progress.json');
const LIVE = path.join(DIR, 'live.json'), SUMMARY = path.join(DIR, 'summary.json');
const founders = JSON.parse(fs.readFileSync(FOUNDERS, 'utf8')).drivers;
// the fixed yardstick every generation is measured against: the original P1-P10, two of each design
const originals = founders.slice(0, 10).map(d => ({ layers: d.layers, genes: Float32Array.from(d.genes) }));

// A save can carry its own recipe (train/upgrade.js writes one); these defaults are the original recipe.
//   window: tournaments averaged before judging a brain; settle: generations a new copy is safe for;
//   pickFromTop: copy a random brain from the top half instead of always the best;
//   hall*: past champions kept as practice rivals (hallRivals of the 11 in every practice race)
const RECIPE = { pairs: 48, margin: MARGIN, settle: SETTLE, window: 1, pickFromTop: false, hallEvery: 0, hallSize: 0, hallRivals: 0 };
let cfg = RECIPE;

function load() {
  if (!fs.existsSync(STATE)) return null;
  const st = JSON.parse(fs.readFileSync(STATE, 'utf8'));
  for (const list of [st.population, st.hall, st.yardstick]) for (const a of list || []) a.genes = Float32Array.from(a.genes);
  return st;
}

const DESIGNS = { A: [16, 10], B: [32, 24, 16], C: [64, 64], D: [64, 64, 64], E: [128, 128] };

function found() {
  const meta = META;
  const start = { started: new Date().toISOString(), generation: 0, clones: {}, history: [] };
  if (meta.start === 'scratch') {
    // Twenty random brains, four independent ones per design. All they start with is the standard
    // initialisation: output weights near zero and a slight throttle bias, so every car moves and can be
    // measured. Learning from zero gets a bigger step size than refining (selection keeps tuning it).
    return { ...start, population: Object.entries(DESIGNS).flatMap(([design, hidden], d) => range(1, 4).map(k => {
      const layers = [E.INPUT_COUNT, ...hidden, 2];
      return { layers, genes: initParams(layers, (meta.seed ?? 1) * 1000 + d * 10 + k), name: `${design}${k}`, founder: `${design}${k}`,
        label: `random ${design}${k}`, species: design, parent: null, born: 0, lr: 0.02, sigma: 0.04, wins: 0, races: 0 };
    })) };
  }
  return {
    ...start,
    population: founders.map(d => ({
      // the founders gain the two damage-feel inputs with zero weights: identical driving to start with
      ...widen(d.layers, Float32Array.from(d.genes)),
      name: `P${d.rank}`, founder: `P${d.rank}`, label: d.label, species: speciesOf(d.layers),
      parent: null, born: 0, lr: 0.004, sigma: 0.04, wins: 0, races: 0,
    })),
  };
}

const pack = list => list?.map(({ m, v, t, genes, ...a }) => ({ ...a, genes: Array.from(genes, g => +g.toFixed(5)) }));
const packed = st => pack(st.population);

function save(st) {
  const population = packed(st), last = st.history.at(-1);
  writeJson(path.join(GENS, `gen-${String(st.generation).padStart(4, '0')}.json`), { generation: st.generation, population });
  writeJson(STATE, { ...st, population, hall: pack(st.hall), yardstick: pack(st.yardstick) });
  writeJson(SUMMARY, { generation: st.generation, champion: last.champion, vsOriginals: last.vsOriginals, vsPast: last.vsPast, updated: last.at,
    designs: Object.fromEntries(Object.entries(last.species).map(([d, s]) => [d, round(1 - s.avgPlace, 2)])) });
}

let progress = {};
const report = fields => writeJson(PROGRESS, progress = { ...progress, ...fields, workers: slots.length, machines: 1 + remotes.length, updated: new Date().toISOString() });

// ---- training: OpenAI-ES with Adam, one update per pro per round ----
function centredRanks(values) {
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), out = new Float32Array(values.length);
  order.forEach(([, i], r) => out[i] = r / (values.length - 1) - 0.5);
  return out;
}

// roster layout (shared with the workers): population, the original pros, hall of fame, past-champion yardstick
const rosterBase = st => {
  const hall = st.population.length + originals.length;
  return { originals: st.population.length, hall, yardstick: hall + (st.hall?.length ?? 0) };
};
const rosterName = (st, i) => {
  const base = rosterBase(st);
  return i < base.originals ? st.population[i].name : i >= base.hall && i < base.yardstick ? st.hall[i - base.hall].name : `#${i}`;
};

function scenarios(st, ai, rng, gen, r) {
  // rivals from the current field, plus a few past champions so nobody just learns to beat this crop
  const hall = range(rosterBase(st).hall, st.hall?.length ?? 0), fromHall = Math.min(cfg.hallRivals, hall.length);
  const others = shuffle([
    ...shuffle(st.population.map((_, i) => i).filter(i => i !== ai), rng).slice(0, opt.field - 1 - fromHall),
    ...shuffle(hall, rng).slice(0, fromHall),
  ], rng);
  // one start from the front, one from the back, plus a solo run so raw pace never erodes, plus a two-car
  // duel where only the winner scores: starting behind, the only way to score is to get past (out-brake it
  // or spin it round); starting in front, the only way is to hold it off. Rounds alternate attack and defence.
  if (NASCAR) {
    // on the ovals: one race in a superspeedway pack, one on another oval, a solo run and a duel elsewhere
    const [pack, other] = E.practiceOvals(gen, r), solo = E.pickFrom(E.OTHER_OVALS(), `tt${gen}:${r}`), duel = E.pickFrom(E.OTHER_OVALS(), `duel${gen}:${r}`);
    return [
      { kind: 'race', ...onTrack(pack, 0, PRACTICE_M), rivals: others, slot: Math.floor(rng() * 2) },
      { kind: 'race', ...onTrack(other, 0, PRACTICE_M), rivals: others, slot: opt.field - 1 - Math.floor(rng() * 3) },
      { kind: 'tt', ...onTrack(solo, 0, PRACTICE_M) },
      { kind: 'race', duel: true, ...onTrack(duel, 0, DUEL_M), rivals: [others[0]], slot: (gen + r) % 2 },
    ];
  }
  return [
    ...range(0, opt.races).map(k => ({ kind: 'race', ...onTrack(practiceTrack(gen, r, k), opt.laps), rivals: others,
      slot: k % 2 ? opt.field - 1 - Math.floor(rng() * 3) : Math.floor(rng() * 2) })),
    { kind: 'tt', ...onTrack(practiceTrack(gen, r, opt.races), opt.laps) },
    { kind: 'race', duel: true, ...onTrack(practiceTrack(gen, r, opt.races + 1), DUEL_LAPS), rivals: [others[0]], slot: (gen + r) % 2 },
  ];
}
const where = sc => sc.trackId ?? sc.trackSeed;

function step(a, seeds, outs, scenarioCount) {
  const n = a.genes.length, shaped = new Float32Array(outs.length);
  for (let k = 0; k < scenarioCount; k++) centredRanks(outs.map(o => o[k])).forEach((r, j) => shaped[j] += r / scenarioCount);
  const grad = new Float32Array(n);
  seeds.forEach((seed, i) => {
    const w = (shaped[2 * i] - shaped[2 * i + 1]) / (2 * seeds.length * a.sigma), eps = noise(seed, n);
    for (let j = 0; j < n; j++) grad[j] += w * eps[j];
  });
  a.m ??= new Float32Array(n);
  a.v ??= new Float32Array(n);
  a.t = (a.t || 0) + 1;
  const b1 = 0.9, b2 = 0.999, { m, v, genes, lr, t } = a;
  for (let j = 0; j < n; j++) {
    m[j] = b1 * m[j] + (1 - b1) * grad[j];
    v[j] = b2 * v[j] + (1 - b2) * grad[j] * grad[j];
    genes[j] += lr * (m[j] / (1 - b1 ** t)) / (Math.sqrt(v[j] / (1 - b2 ** t)) + 1e-8) - lr * opt.decay * genes[j];
  }
}

const roster = st => [...st.population.map(a => ({ layers: a.layers, genes: a.genes.slice() })), ...originals,
  ...[...(st.hall ?? []), ...(st.yardstick ?? [])].map(a => ({ layers: a.layers, genes: a.genes }))];

async function train(st, gen) {
  broadcast({ type: 'roster', drivers: roster(st) });
  st.practiceRuns ??= 0;
  st.practiceHistory ??= [];
  for (let r = 1; r <= opt.rounds; r++) {
    const rng = E.mulberry32(gen * 1009 + r), plans = st.population.map((_, ai) => scenarios(st, ai, rng, gen, r));
    writeJson(LIVE, { generation: gen, round: r, population: packed(st), hall: pack(st.hall) });
    // the practice board: for each brain, how many of its copies have raced this round and how they scored
    const copies = 2 * opt.pairs, perCopy = plans[0].length, n = st.population.length;
    const done = new Array(n).fill(0), sum = new Array(n).fill(0), best = new Array(n).fill(null);
    const board = () => ({
      copies, runsPerCopy: perCopy, done: [...done],
      mean: sum.map((s, i) => done[i] ? round(s / done[i]) : null), best: best.map(b => b == null ? null : round(b)),
    });
    const started = Date.now(), plan = {
      laps: NASCAR ? plans[0].map(sc => sc.laps) : opt.laps, field: opt.field, pairs: opt.pairs, tracks: plans[0].map(where), mode: MODE,
      pros: plans.map((scs, ai) => ({ name: st.population[ai].name, rivals: scs[0].rivals.map(i => rosterName(st, i)),
        slots: scs.filter(sc => sc.kind === 'race' && !sc.duel).map(sc => sc.slot),
        duels: scs.filter(sc => sc.duel).map(sc => ({ rival: rosterName(st, sc.rivals[0]), slot: sc.slot, track: where(sc), laps: sc.laps })) })),
    };
    const update = () => {
      const runs = done.reduce((a, b) => a + b, 0) * perCopy;
      report({ generation: gen, phase: 'training', round: r, rounds: opt.rounds, tournament: null, practice: { ...plan, board: board() },
        runs: { round: runs, roundTotal: n * copies * perCopy, total: st.practiceRuns + runs, perSecond: Math.round(runs / Math.max(1, (Date.now() - started) / 1000)) },
        practiceHistory: st.practiceHistory });
    };
    update();
    const ticker = setInterval(update, 2000);
    await Promise.all(st.population.map((a, ai) => {
      broadcast({ type: 'theta', agent: ai, layers: a.layers, theta: a.genes });
      const scs = plans[ai];
      const seeds = range(0, opt.pairs).map(i => (Math.imul(gen, 2654435761) ^ Math.imul(r * 64 + ai, 40503) ^ Math.imul(i + 1, 97531)) >>> 0);
      const jobs = seeds.flatMap(seed => [1, -1].map(sign => submit({ kind: 'es', agent: ai, seed, sign, sigma: a.sigma, scenarios: scs }).then(scores => {
        const reward = mean(scores);
        done[ai]++;
        sum[ai] += reward;
        if (best[ai] == null || reward > best[ai]) best[ai] = reward;
        return scores;
      })));
      return Promise.all(jobs).then(outs => step(a, seeds, outs, scs.length));
    }));
    clearInterval(ticker);
    st.practiceRuns += n * copies * perCopy;
    // one point per round on the learning curve: each brain's average practice reward
    st.practiceHistory.push({ gen, round: r, mean: board().mean });
    if (st.practiceHistory.length > 400) st.practiceHistory.shift();
    update();
  }
}

// ---- tournament: everyone races everyone, full distance ----
async function tournament(st, gen) {
  writeJson(LIVE, { generation: gen, round: null, population: packed(st), hall: pack(st.hall) });
  // the tournament's tracks: 8 new generated tracks, or on the ovals an 8-race season (two superspeedways)
  const venues = NASCAR ? E.seasonOvals(gen) : range(0, TOURNEY_TRACKS).map(k => tourneyTrack(gen, k));
  const raceAt = k => ({ ...onTrack(venues[k % venues.length], RACE_LAPS, SEASON_M), ...NASCAR && { stages: true } });
  report({ generation: gen, phase: 'tournament', practice: null, tournament: { laps: NASCAR ? venues.map((_, k) => raceAt(k).laps) : RACE_LAPS, tracks: venues, mode: MODE } });
  broadcast({ type: 'roster', drivers: roster(st) });
  // a grid holds 20 cars (40 on the ovals): a bigger population races in random fields, enough races for ~24 each
  const rng = E.mulberry32(gen * 7919 + 17), size = st.population.length, grid = Math.min(size, NASCAR ? NASCAR_FIELD : 20);
  const raceCount = Math.ceil(TOURNEY_RACES * size / grid);
  const races = range(0, raceCount).map(r => ({ kind: 'field', ...raceAt(r), entrants: shuffle(range(0, size), rng).slice(0, grid) }));
  const results = await Promise.all(races.map(submit));
  const tally = st.population.map(() => ({ places: [], points: [], wins: 0, podiums: 0, aero: [], laps: [], rammed: [], passes: [], walls: [], tail: [], led: [],
    stage: [], cautions: [], penalties: [], below: [] }));
  races.forEach((race, k) => results[k].forEach((res, slot) => {
    const t = tally[race.entrants[slot]];
    t.places.push(res.place / (race.entrants.length - 1));
    // stage points count a little: 10 for winning a stage is worth a tenth of a race win
    t.points.push(racePoints(res.place, race.entrants.length) + (res.stagePoints ? res.stagePoints / 100 : 0));
    if (res.stagePoints !== undefined) {
      t.stage.push(res.stagePoints);
      t.cautions.push(res.cautionsCaused);
      t.penalties.push(res.penalties);
      t.below.push(res.belowLine);
    }
    t.led.push(res.led);
    t.wins += res.place === 0;
    t.podiums += res.place < 3;
    t.aero.push(res.aero);
    t.rammed.push(res.rammed);
    t.passes.push(res.passes);
    t.walls.push(res.walls);
    t.tail.push(res.tail);
    if (res.lap) t.laps.push(res.lap);
  }));
  const agents = st.population.map((a, i) => ({
    name: a.name, species: a.species, avgPlace: round(mean(tally[i].places)), wins: tally[i].wins, podiums: tally[i].podiums,
    aero: round(mean(tally[i].aero)), lap: round(mean(tally[i].laps), 2), rammed: round(mean(tally[i].rammed), 2),
    passes: round(mean(tally[i].passes), 1), walls: round(mean(tally[i].walls), 1), tail: round(mean(tally[i].tail)),
    points: round(mean(tally[i].points)), winRate: round(tally[i].wins / tally[i].places.length), led: round(mean(tally[i].led)),
    ...NASCAR && { stagePoints: round(mean(tally[i].stage), 1), cautionsCaused: round(mean(tally[i].cautions), 2),
      penalties: round(mean(tally[i].penalties), 2), belowLine: round(mean(tally[i].below)) },
  }));
  st.population.forEach((a, i) => {
    a.wins += tally[i].wins;
    a.races += tally[i].places.length;
    a.last = agents[i];
    // what selection judges: race points (wins count most), averaged over a few tournaments so one unlucky
    // tournament doesn't end a lineage
    a.recentPoints = [...(a.recentPoints ?? []), agents[i].points].slice(-cfg.window);
  });

  // The yardstick, on 8 fixed tracks. With a past-champion panel (st.yardstick), this generation's best 10
  // race it; otherwise the best two of each design race the original P1-P10.
  const base = rosterBase(st), panel = st.yardstick?.length ? range(base.yardstick, st.yardstick.length) : range(base.originals, originals.length);
  const ours = st.yardstick?.length
    ? agents.map((g, i) => ({ ...g, i })).sort((x, y) => y.points - x.points).slice(0, panel.length).map(g => g.i)
    : Object.values(Object.groupBy(agents.map((g, i) => ({ ...g, i })), g => g.species)).flatMap(group => group.sort((x, y) => y.points - x.points).slice(0, 2).map(g => g.i));
  const benchVenues = NASCAR ? E.YARDSTICK_OVALS : BENCH_TRACKS;
  const benchRaces = range(0, BENCH_RACES).map(r => ({ kind: 'field', ...onTrack(benchVenues[r], RACE_LAPS, SEASON_M), ...NASCAR && { stages: true },
    entrants: shuffle([...ours, ...panel], rng) }));
  const bench = await Promise.all(benchRaces.map(submit));
  const isNew = i => i < size, field = ours.length + panel.length;
  const benchPlaces = side => mean(benchRaces.flatMap((race, k) => bench[k].filter((_, s) => isNew(race.entrants[s]) === side).map(res => res.place / (field - 1))));
  const yardstick = {
    races: BENCH_RACES,
    wins: benchRaces.filter((race, k) => isNew(race.entrants[bench[k].findIndex(res => res.place === 0)])).length,
    newAvg: round(benchPlaces(true)), oldAvg: round(benchPlaces(false)),
  };

  const species = Object.fromEntries(Object.entries(Object.groupBy(agents, g => g.species)).map(([sp, group]) => [sp, {
    avgPlace: round(mean(group.map(g => g.avgPlace))), best: Math.min(...group.map(g => g.avgPlace)),
    wins: group.reduce((n, g) => n + g.wins, 0), aero: round(mean(group.map(g => g.aero))),
    lap: round(Math.min(...group.map(g => g.lap ?? Infinity)), 2), rammed: round(mean(group.map(g => g.rammed)), 2),
    tail: round(mean(group.map(g => g.tail))), points: round(mean(group.map(g => g.points))), led: round(mean(group.map(g => g.led))),
  }]));
  // the champion is whoever scored the most race points, which mostly means whoever won the most
  const champion = agents.reduce((x, y) => y.points > x.points ? y : x).name;
  return { gen, at: new Date().toISOString(), races: raceCount, species, agents, champion, [st.yardstick?.length ? 'vsPast' : 'vsOriginals']: yardstick, replaced: [] };
}

// ---- selection: within a species, a clear laggard is replaced by a mutated copy of a strong sibling ----
function select(st, entry, rng) {
  const score = a => mean(a.recentPoints?.length ? a.recentPoints : [a.last.points]);
  for (const sp of new Set(st.population.map(a => a.species))) {
    const members = st.population.filter(a => a.species === sp).sort((x, y) => score(y) - score(x));
    const best = members[0], worst = members.at(-1);
    if (score(best) - score(worst) < cfg.margin || entry.gen - worst.born < cfg.settle) continue;
    // always copying the single best collapses a design onto one lineage; the top half keeps several alive
    const pool = cfg.pickFromTop ? members.slice(0, Math.ceil(members.length / 2)) : [best], parent = pool[Math.floor(rng() * pool.length)];
    const n = st.clones[parent.founder] = (st.clones[parent.founder] || 1) + 1, name = `${parent.founder}·${n}`;
    entry.replaced.push({ out: worst.name, by: name, parent: parent.name, species: sp });
    const jitter = () => rng() < 0.5 ? 0.8 : 1.25;
    Object.assign(worst, {
      name, founder: parent.founder, label: parent.label, parent: parent.name, born: entry.gen, genes: parent.genes.slice(),
      lr: clamp(parent.lr * jitter(), 0.001, 0.02), sigma: clamp(parent.sigma * jitter(), 0.01, 0.1), wins: 0, races: 0, recentPoints: [],
      m: null, v: null, t: 0,
    });
  }
}

// every hallEvery generations the champion joins the hall of fame, frozen, as a practice rival
function induct(st, entry) {
  if (!cfg.hallEvery || entry.gen % cfg.hallEvery) return;
  const champ = st.population.find(a => a.name === entry.champion);
  st.hall = [...(st.hall ?? []), { name: `${champ.name}@${entry.gen}`, species: champ.species, layers: champ.layers, genes: champ.genes.slice(), gen: entry.gen }];
  if (st.hall.length > cfg.hallSize) st.hall.shift();
  entry.inducted = st.hall.at(-1).name;
}

function log(entry, minutes) {
  const sp = Object.entries(entry.species).sort((x, y) => y[1].points - x[1].points).map(([k, s]) => `${k} ${s.points.toFixed(2)} (${s.wins} wins)`).join('  ');
  const vs = entry.vsPast ?? entry.vsOriginals, against = entry.vsPast ? 'past champions' : 'originals';
  const swaps = entry.replaced.map(r => `${r.out} → ${r.by}`).join(', ');
  console.log(`gen ${entry.gen} | ${minutes.toFixed(1)} min | champion ${entry.champion} | race points by design: ${sp} | vs ${against}: won ${vs.wins}/${vs.races}, avg ${vs.newAvg} vs ${vs.oldAvg}${swaps ? ` | replaced ${swaps}` : ''}${entry.inducted ? ` | hall of fame: ${entry.inducted}` : ''}`);
}

(async () => {
  const st = load() || found();
  cfg = { ...RECIPE, ...st.config };
  if (!args.pairs) opt.pairs = cfg.pairs;
  for (const address of (args.remote || '').split(',').filter(Boolean)) {
    known.add(address);
    await connectRemote(address, args.token);
  }
  if (args['remote-file']) watchRemotes(args['remote-file'], args.token);
  console.log(`[evolve] ${st.population.length} pros, generation ${st.generation}, ${slots.length} threads on ${1 + remotes.length} machine(s), ${opt.pairs} pairs × ${opt.rounds} rounds per generation`);
  if (!st.history.length) {
    const t0 = Date.now(), entry = await tournament(st, 0);
    st.history.push(entry);
    save(st);
    log(entry, (Date.now() - t0) / 60000);
  }
  for (;;) {
    const gen = st.generation + 1, t0 = Date.now();
    report({ started: new Date(t0).toISOString(), lastMinutes: progress.lastMinutes });
    await train(st, gen);
    const entry = await tournament(st, gen);
    select(st, entry, E.mulberry32(gen * 31 + 7));
    induct(st, entry);
    entry.minutes = round((Date.now() - t0) / 60000, 2);
    st.generation = gen;
    st.history.push(entry);
    save(st);
    report({ generation: gen + 1, phase: 'training', round: 0, lastMinutes: entry.minutes });
    log(entry, entry.minutes);
  }
})();
