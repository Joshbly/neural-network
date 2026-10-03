#!/usr/bin/env node
// The evolution: the pros keep getting better on every core until you stop it.
//
// Population-based training. Twenty pros in five species, one per brain design. Every generation:
//   1. training: each pro gets the same number of evolution-strategies steps, racing rivals drawn from
//      the whole population (every design), so they learn racecraft from each other;
//   2. tournament: all twenty race 10-lap, 20-car races on tracks none of them train on;
//   3. selection: within each species, a pro that keeps finishing well behind a strong sibling is
//      replaced by a mutated copy of that sibling (new learning rate and noise scale too); no family line
//      may hold more than half a species' seats, so every design keeps exploring more than one way to drive;
//   4. every few generations, the rating ladder (train/ladder.js): each design's best races frozen past
//      champions on fixed tracks, giving a rating that means the same thing across the whole run.
// Species never mix (the weights don't fit across designs) and every species keeps four seats, so the
// standings answer "which brain design is best" with equal training for each.
//
// State lives in models/evolution/ and survives restarts. The founders are models/originals/tab-field.json,
// which is never written to.
//   node train/evolve.js [--pairs 48] [--rounds 5] [--workers N] [--until GEN]
const os = require('os');
const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');
const { E, noise, widen, initParams, racePoints, SHAPES } = require('./lib');
const { fitRatings, pickOpponents, pickParent } = require('./ladder');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const opt = {
  pairs: +(args.pairs || 48), rounds: +(args.rounds || 5), workers: +(args.workers || Math.max(1, os.cpus().length - 1)),
  field: 12, laps: 4, races: 2, decay: 0.005,
};
const ROOT = path.join(__dirname, '..');
const DIR = path.resolve(args.dir || path.join(ROOT, 'models', 'evolution')), GENS = path.join(DIR, 'generations');
const FOUNDERS = path.join(ROOT, 'models', 'originals', 'tab-field.json');
const range = (from, n) => Array.from({ length: n }, (_, i) => from + i);
// Practice tracks never repeat: every training round of every generation gets new ones, shared by all
// twenty pros that round. The tournament gets 8 new tracks each generation, raced 3 times each with
// different grids so neither grid luck nor one odd layout decides who survives. Only the rating ladder
// stays on the same tracks, so every generation sits the same test.
const practiceTrack = (gen, round, k) => 10_000_000 + (gen * 16 + round) * 8 + k;
const tourneyTrack = (gen, k) => 5_000_000 + gen * 8 + k;
const TOURNEY_TRACKS = 8, TOURNEY_RACES = 24, RACE_LAPS = 10, DUEL_LAPS = 2;
// The ladder, every RATE_EVERY generations: 96 ten-car races, each design's best five against five frozen
// past champions, on 12 fixed tracks (every kind of oval in NASCAR mode). Fixed for good: changing any of
// this would make new ratings incomparable with old ones.
const RATE_EVERY = 5, RATE_RACES = 96, RATE_FIELD = 10, POOL_ACTIVE = 20;
const RATE_TRACKS = range(7_000_001, 12), RATE_LAPS = 5, RATE_M = 24000;
const RATE_OVALS = ['daytona', 'talladega', 'atlanta', 'charlotte', 'michigan', 'kansas', 'darlington', 'dover', 'phoenix', 'richmond', 'bristol', 'martinsville'];
// A save races normal tracks or the real NASCAR ovals, in normal cars or stock cars (meta.json; any mix
// trains). On the ovals a race is a distance, not a lap count. Practice races, the season and the rating are
// 24 km (six laps of Daytona, 28 of Martinsville): long enough that a car has to live with its damage, a pack
// has time to form and a driver can wait for the move instead of going all in at once. Duels and time trials
// are 6 km.
const META = (() => { try { return JSON.parse(fs.readFileSync(path.join(DIR, 'meta.json'), 'utf8')); } catch { return {}; } })();
const MODE = { tracks: META.tracks ?? 'normal', cars: META.cars ?? (META.tracks === 'nascar' ? 'stock' : 'normal') };
const NASCAR = MODE.tracks === 'nascar', CARS = MODE.cars === 'normal' && !NASCAR ? {} : { cars: MODE.cars };
const SEASON_M = 24000, PRACTICE_M = SEASON_M, DUEL_M = 6000, SOLO_M = 6000, NASCAR_FIELD = 40;
// practice fields on the ovals: the superspeedway pack is full, the other race the size of the tournament's
const PACK_FIELD = 40, OVAL_FIELD = 20;
// where a race is: a generated track's seed, or a real oval and a lap count for the distance
const onTrack = (seedOrId, laps, metres) => NASCAR ? { trackId: seedOrId, laps: E.lapsFor(seedOrId, metres), ...CARS } : { trackSeed: seedOrId, laps, ...CARS };
// race points per tournament race; judged over several tournaments, so a margin of 0.2 (about two places in a
// 20-car field, a third of the gap between a win and second) is well clear of luck
const MARGIN = 0.2;
// T: an E with a grafted traffic block (train/graft.js), 32 first-layer neurons that only see the traffic senses;
// F: lane and traffic blocks, a narrow mixing layer and shortcuts to the hands (SHAPES in train/lib.js)
const SPECIES = { '16-10': 'A', '32-24-16': 'B', '64-64': 'C', '64-64-64': 'D', '128-128': 'E', '160-128': 'T', '208-48': 'F' };
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
// ladder.json: every rating race ever run and the frozen players in it; rating.json: the fitted ratings
const STATE = path.join(DIR, 'state.json'), PROGRESS = path.join(DIR, 'progress.json');
const LIVE = path.join(DIR, 'live.json'), SUMMARY = path.join(DIR, 'summary.json');
const LADDER = path.join(DIR, 'ladder.json'), RATING = path.join(DIR, 'rating.json');
// tournament.json / rating-races.json: the latest tournament's and rating round's races, set up exactly as
// they ran with everyone's weights, plus the official finishing orders (the app replays them)
const TOURNAMENT = path.join(DIR, 'tournament.json'), RATING_RACES = path.join(DIR, 'rating-races.json');
const founders = JSON.parse(fs.readFileSync(FOUNDERS, 'utf8')).drivers;

// A save can carry its own recipe (train/upgrade.js writes one); anything it leaves out comes from here.
//   window: tournaments averaged before judging a brain; settle: generations a new copy is safe for;
//   pickFromTop: copy any of the strong half instead of always the single best;
//   hall*: past champions kept as practice rivals (hallRivals of the 11 in every practice race), so the
//     field never forgets how to beat an older style;
//   familyCap: no family line holds more than half a species' seats;
//   mutate: a copy starts this many noise-scales (sigma) away from its parent instead of identical to it;
//   tournamentSparring: the hall races the tournaments too, filling every grid after the whole population
//     (an experiment save, train/graft.js, keeps a fixed hall of sparring partners: hallEvery 0)
const RECIPE = { pairs: 48, margin: MARGIN, settle: 3, window: 3, pickFromTop: true, hallEvery: 10, hallSize: 16, hallRivals: 3, familyCap: true, mutate: 1 };
let cfg = RECIPE;

function load() {
  if (!fs.existsSync(STATE)) return null;
  const st = JSON.parse(fs.readFileSync(STATE, 'utf8'));
  for (const list of [st.population, st.hall, st.yardstick]) for (const a of list || []) a.genes = Float32Array.from(a.genes);
  // inputs added since a brain was born reach it with zero weights: it drives exactly as before until training
  // finds a use for them (frozen past champions stay as they were). The optimizer's memory carries on across a
  // restart, so the first step after one isn't a full-size jolt to every weight; a widened brain starts it afresh.
  for (const a of st.population) {
    const before = a.genes.length;
    Object.assign(a, widen(a.layers, a.genes));
    if (a.m && a.genes.length === before) {
      a.m = Float32Array.from(a.m);
      a.v = Float32Array.from(a.v);
    } else Object.assign(a, { m: null, v: null, t: 0 });
    // a grafted brain's locked connections are exactly 0, or something upstream is broken: don't train on it
    const gs = scaleOf(a);
    if (gs && a.genes.some((g, j) => gs[j] === 0 && g !== 0)) throw new Error(`${a.name}: a locked connection isn't 0`);
  }
  return st;
}

// players: everyone who has run a rating race, frozen; pool ones keep their weights and can be raced again
function loadLadder() {
  if (!fs.existsSync(LADDER)) return { players: [], races: [] };
  const ladder = JSON.parse(fs.readFileSync(LADDER, 'utf8'));
  for (const p of ladder.players) if (p.genes) p.genes = Float32Array.from(p.genes);
  return ladder;
}
let ladder = { players: [], races: [] }, ratings = [];
const poolOf = () => ladder.players.filter(p => p.genes);

// F's lane and traffic blocks and shortcuts are in SHAPES (train/lib.js)
const DESIGNS = { A: [16, 10], B: [32, 24, 16], C: [64, 64], D: [64, 64, 64], E: [128, 128], F: [208, 48] };

function found() {
  const meta = META;
  // meta.config: the save's own recipe changes (the founding period and family caps of a head-to-head)
  const start = { started: new Date().toISOString(), generation: 0, clones: {}, history: [], ...meta.config && { config: meta.config } };
  if (meta.start === 'scratch') {
    // Random brains, independent ones per design: four of each of the five classic designs unless the save names
    // its own seats (meta.seats, up to 10 a design). All they start with is the standard initialisation: output
    // weights near zero and a slight throttle bias, so every car moves and can be measured. Learning from zero gets
    // a bigger step size than refining (selection keeps tuning it).
    const seats = meta.seats ?? { A: 4, B: 4, C: 4, D: 4, E: 4 };
    return { ...start, population: Object.entries(DESIGNS).flatMap(([design, hidden], d) => range(1, seats[design] ?? 0).map(k => {
      const layers = [E.INPUT_COUNT, ...hidden, 2], shape = SHAPES[design];
      return { layers, genes: initParams(layers, (meta.seed ?? 1) * 1000 + d * 10 + k, shape), ...shape && { mask: shape.mask },
        name: `${design}${k}`, founder: `${design}${k}`, label: `random ${design}${k}`, species: design, parent: null, born: 0, lr: 0.02, sigma: 0.04, wins: 0, races: 0 };
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
  // the save keeps each brain's optimizer memory too (6 significant figures is plenty); the published files don't
  const memory = xs => Array.from(xs, x => +x.toPrecision(6));
  const withMemory = population.map((p, i) => st.population[i].m ? { ...p, m: memory(st.population[i].m), v: memory(st.population[i].v), t: st.population[i].t } : p);
  writeJson(STATE, { ...st, population: withMemory, hall: pack(st.hall), yardstick: pack(st.yardstick) });
  const rated = st.history.findLast(e => e.rating)?.rating;
  writeJson(SUMMARY, { generation: st.generation, champion: last.champion, rating: rated?.champion, updated: last.at,
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

// roster layout (shared with the workers): population, hall of fame, the ladder's pool of frozen champions
const rosterBase = st => {
  const hall = st.population.length;
  return { hall, pool: hall + (st.hall?.length ?? 0) };
};
const rosterName = (st, i) => {
  const base = rosterBase(st);
  return i < base.hall ? st.population[i].name : i < base.pool ? st.hall[i - base.hall].name : `#${i}`;
};

function scenarios(st, ai, rng, gen, r) {
  // rivals from the current field, plus a few past champions so nobody just learns to beat this crop
  const hall = range(rosterBase(st).hall, st.hall?.length ?? 0), fromHall = Math.min(cfg.hallRivals, hall.length);
  const field = NASCAR ? OVAL_FIELD : opt.field, rest = st.population.map((_, i) => i).filter(i => i !== ai);
  const others = shuffle([
    ...shuffle(rest.slice(), rng).slice(0, field - 1 - fromHall),
    ...shuffle(hall, rng).slice(0, fromHall),
  ], rng);
  // one start from the front, one from the back, plus a solo run so raw pace never erodes, plus a two-car
  // duel where only the winner scores: starting behind, the only way to score is to get past (out-brake it
  // or spin it round); starting in front, the only way is to hold it off. Rounds alternate attack and defence.
  if (NASCAR) {
    // on the ovals: a full 40-car superspeedway pack, a 20-car race on another oval, a solo run and a duel
    // elsewhere. The pack is everyone else, every past champion, then the field again (some brains drive two
    // cars) until it's full, so there's always a drafting partner. One race starts somewhere in the front half
    // and the other in the back half, swapping each round, so both kinds of oval get practised from every part
    // of the grid, as the tournament's shuffled grids demand.
    const [pack, other] = E.practiceOvals(gen, r), solo = E.pickFrom(E.OTHER_OVALS(), `tt${gen}:${r}`), duel = E.pickFrom(E.OTHER_OVALS(), `duel${gen}:${r}`);
    const crowd = shuffle([...rest, ...hall, ...shuffle(rest.slice(), rng), ...shuffle(rest.slice(), rng)].slice(0, PACK_FIELD - 1), rng);
    const start = (rivals, front) => {
      const half = (rivals.length + 1) >> 1;
      return front ? Math.floor(rng() * half) : half + Math.floor(rng() * (rivals.length + 1 - half));
    };
    const packFront = (gen + r) % 2 === 0;
    return [
      { kind: 'race', ...onTrack(pack, 0, PRACTICE_M), rivals: crowd, slot: start(crowd, packFront) },
      { kind: 'race', ...onTrack(other, 0, PRACTICE_M), rivals: others, slot: start(others, !packFront) },
      { kind: 'tt', ...onTrack(solo, 0, SOLO_M) },
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

// a grafted brain's locked connections and nudge scales (js/replay.js geneScale), worked out once per design
const scales = new Map();
const scaleOf = a => {
  if (!a.mask) return null;
  const key = `${a.layers.join('-')}:${JSON.stringify(a.mask)}`;
  if (!scales.has(key)) scales.set(key, E.geneScale(a.layers, a.mask, a.genes.length));
  return scales.get(key);
};

function step(a, seeds, outs, scenarioCount) {
  const n = a.genes.length, shaped = new Float32Array(outs.length), gs = scaleOf(a);
  for (let k = 0; k < scenarioCount; k++) centredRanks(outs.map(o => o[k])).forEach((r, j) => shaped[j] += r / scenarioCount);
  const grad = new Float32Array(n);
  // the copies were nudged by the scaled noise, so that's what each weight is credited with: a locked connection
  // gets no gradient, so Adam never moves it off 0 (and decay of 0 is 0)
  seeds.forEach((seed, i) => {
    const w = (shaped[2 * i] - shaped[2 * i + 1]) / (2 * seeds.length * a.sigma), eps = noise(seed, n);
    if (gs) for (let j = 0; j < n; j++) grad[j] += w * (gs[j] * eps[j]);
    else for (let j = 0; j < n; j++) grad[j] += w * eps[j];
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

const roster = st => [...st.population.map(a => ({ layers: a.layers, genes: a.genes.slice() })),
  ...[...(st.hall ?? []), ...poolOf()].map(a => ({ layers: a.layers, genes: a.genes }))];

async function train(st, gen) {
  st.practiceRuns ??= 0;
  st.practiceHistory ??= [];
  for (let r = 1; r <= opt.rounds; r++) {
    // everyone races with exactly the weights live.json publishes (5 decimals), rivals as they are this round,
    // so the app can rebuild any practice race the engine runs, copy for copy (js/replay.js)
    for (const a of st.population) a.genes = E.quantizeGenes(a.genes);
    broadcast({ type: 'roster', drivers: roster(st) });
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
        duels: scs.filter(sc => sc.duel).map(sc => ({ rival: rosterName(st, sc.rivals[0]), slot: sc.slot, track: where(sc), laps: sc.laps })),
        // every copy of this brain races exactly these (rivals by name): the app rebuilds any copy's race from them
        scenarios: scs.map(({ rivals, ...sc }) => ({ ...sc, ...rivals && { rivals: rivals.map(i => rosterName(st, i)) } })) })),
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
      broadcast({ type: 'theta', agent: ai, layers: a.layers, theta: a.genes, mask: a.mask });
      const scs = plans[ai];
      const seeds = range(0, opt.pairs).map(i => E.esSeed(gen, r, ai, i));
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
  for (const a of st.population) a.genes = E.quantizeGenes(a.genes);
  writeJson(LIVE, { generation: gen, round: null, population: packed(st), hall: pack(st.hall) });
  // the tournament's tracks: 8 new generated tracks, or on the ovals an 8-race season (two superspeedways)
  const venues = NASCAR ? E.seasonOvals(gen) : range(0, TOURNEY_TRACKS).map(k => tourneyTrack(gen, k));
  // full flag rules once the brains are old enough to race clean (RC.cautionsFrom); green racing before that
  const raceAt = k => ({ ...onTrack(venues[k % venues.length], RACE_LAPS, SEASON_M), ...NASCAR && { stages: true, cautions: gen >= E.RC.cautionsFrom } });
  broadcast({ type: 'roster', drivers: roster(st) });
  const rng = E.mulberry32(gen * 7919 + 17), size = st.population.length, cap = NASCAR ? NASCAR_FIELD : 20;
  // sparring partners (cfg.tournamentSparring): the frozen hall fills every grid after the whole population, so an
  // experiment is judged against a real field; only the population is scored
  const sparring = cfg.tournamentSparring ? range(rosterBase(st).hall, st.hall?.length ?? 0) : [];
  let races;
  if (sparring.length) {
    races = range(0, TOURNEY_RACES).map(r => ({ kind: 'field', ...raceAt(r),
      entrants: shuffle([...range(0, size), ...shuffle(sparring.slice(), rng).slice(0, Math.max(0, cap - size))], rng) }));
  } else {
    // a grid holds 20 cars (40 on the ovals): a bigger population races in random fields, enough races for ~24 each
    const grid = Math.min(size, cap);
    races = range(0, Math.ceil(TOURNEY_RACES * size / grid)).map(r => ({ kind: 'field', ...raceAt(r), entrants: shuffle(range(0, size), rng).slice(0, grid) }));
  }
  // every race exactly as it will run (the grid in starting order), so the app can show the real ones
  const setups = races.map(({ entrants, ...scenario }) => ({ scenario, grid: entrants.map(i => rosterName(st, i)) }));
  report({ generation: gen, phase: 'tournament', practice: null, tournament: { laps: NASCAR ? venues.map((_, k) => raceAt(k).laps) : RACE_LAPS, tracks: venues, mode: MODE, races: setups } });
  const results = await Promise.all(races.map(submit));
  // the official results, with the brains exactly as they raced: the app's tournament channel replays these
  writeJson(TOURNAMENT, {
    gen, mode: MODE, population: packed(st), ...sparring.length && { sparring: pack(st.hall) },
    races: setups.map((setup, k) => ({ ...setup, order: results[k].map((res, slot) => [res.place, setup.grid[slot]]).sort((x, y) => x[0] - y[0]).map(x => x[1]) })),
  });
  const tally = st.population.map(() => ({ places: [], points: [], wins: 0, podiums: 0, aero: [], laps: [], rammed: [], passes: [], walls: [], tail: [], rub: [], led: [],
    stage: [], cautions: [], penalties: [], below: [] }));
  races.forEach((race, k) => results[k].forEach((res, slot) => {
    if (race.entrants[slot] >= size) return;
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
    t.rub.push(res.rub);
    if (res.lap) t.laps.push(res.lap);
  }));
  const agents = st.population.map((a, i) => ({
    name: a.name, species: a.species, avgPlace: round(mean(tally[i].places)), wins: tally[i].wins, podiums: tally[i].podiums,
    aero: round(mean(tally[i].aero)), lap: round(mean(tally[i].laps), 2), rammed: round(mean(tally[i].rammed), 2),
    passes: round(mean(tally[i].passes), 1), walls: round(mean(tally[i].walls), 1), tail: round(mean(tally[i].tail)), rub: round(mean(tally[i].rub)),
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

  const species = Object.fromEntries(Object.entries(Object.groupBy(agents, g => g.species)).map(([sp, group]) => [sp, {
    avgPlace: round(mean(group.map(g => g.avgPlace))), best: Math.min(...group.map(g => g.avgPlace)),
    wins: group.reduce((n, g) => n + g.wins, 0), aero: round(mean(group.map(g => g.aero))),
    lap: round(Math.min(...group.map(g => g.lap ?? Infinity)), 2), rammed: round(mean(group.map(g => g.rammed)), 2),
    tail: round(mean(group.map(g => g.tail))), rub: round(mean(group.map(g => g.rub))), points: round(mean(group.map(g => g.points))), led: round(mean(group.map(g => g.led))),
  }]));
  // the champion is whoever scored the most race points, which mostly means whoever won the most
  const champion = agents.reduce((x, y) => y.points > x.points ? y : x).name;
  return { gen, at: new Date().toISOString(), races: races.length, species, agents, champion, replaced: [] };
}

// ---- the rating ladder ----
// Each design's best (by this tournament) is frozen as a new player and races RATE_RACES fixed-track races
// against the strongest frozen champions in the pool; then every result ever recorded is refitted. The
// generation's champion joins the pool, which keeps the weights of its POOL_ACTIVE strongest players.
async function rate(st, gen, entry) {
  if (!ladder.players.length) {
    // a save with past champions already frozen (hall of fame, an old past-champion panel) starts its pool with them
    for (const a of [...(st.yardstick ?? []), ...(st.hall ?? [])])
      if (!ladder.players.some(p => p.name === a.name)) ladder.players.push({ id: ladder.players.length, name: a.name, gen: a.gen ?? null, design: a.species, kind: 'seed', layers: a.layers, genes: a.genes });
  }
  const bySpecies = Object.values(Object.groupBy(st.population.map((a, i) => ({ a, i, points: entry.agents[i].points })), x => x.a.species))
    .map(group => group.sort((x, y) => y.points - x.points));
  // while the pool is small (a new save), the next-best brains fill the field
  const pool = poolOf(), extra = Math.max(0, RATE_FIELD / 2 - pool.length);
  const picks = [...bySpecies.map(g => g[0]), ...bySpecies.flatMap(g => g.slice(1)).sort((x, y) => y.points - x.points).slice(0, extra)];
  const entrants = picks.map(({ a, i }, k) => {
    // role 'best': its design's best this generation, the one the rating chart follows
    const player = { id: ladder.players.length, name: `${a.name}@${gen}`, gen, design: a.species, kind: 'entrant', role: k < bySpecies.length ? 'best' : 'filler', layers: a.layers, genes: a.genes.slice() };
    ladder.players.push(player);
    return { player, roster: i };
  });
  broadcast({ type: 'roster', drivers: roster(st) });
  const base = rosterBase(st), ratingOf = p => ratings[p.id]?.r ?? 1000, rng = E.mulberry32(gen * 4217 + 3);
  const venues = NASCAR ? RATE_OVALS : RATE_TRACKS;
  const races = range(0, RATE_RACES).map(r => {
    const rivals = pickOpponents(pool, ratingOf, Math.min(RATE_FIELD - entrants.length, pool.length), rng);
    const field = shuffle([...entrants.map(e => ({ player: e.player, roster: e.roster })), ...rivals.map(p => ({ player: p, roster: base.pool + pool.indexOf(p) }))], rng);
    return { field, job: { kind: 'field', ...onTrack(venues[r % venues.length], RATE_LAPS, RATE_M), entrants: field.map(f => f.roster) } };
  });
  const results = await Promise.all(races.map(r => submit(r.job)));
  const official = races.map(({ field }, k) => {
    const order = results[k].map((res, s) => ({ id: field[s].player.id, place: res.place })).sort((x, y) => x.place - y.place).map(x => x.id);
    ladder.races.push({ gen, order });
    return order;
  });
  // this round's races exactly as they ran, with everyone in them (before most entrants' weights are dropped)
  const used = new Map(races.flatMap(({ field }) => field.map(f => [f.player.id, f.player])));
  writeJson(RATING_RACES, {
    gen, mode: MODE,
    players: [...used.values()].map(p => ({ id: p.id, name: p.name, gen: p.gen, design: p.design, kind: p.kind, layers: p.layers, genes: Array.from(p.genes, g => +g.toFixed(5)) })),
    races: races.map(({ field, job: { entrants, ...scenario } }, k) => ({ scenario, grid: field.map(f => f.player.id), order: official[k] })),
  });
  // the champion is kept to race future generations; only the strongest keep their weights
  const champ = entrants.find(e => e.player.name === `${entry.champion}@${gen}`)?.player;
  for (const e of entrants) if (e.player !== champ) delete e.player.genes;
  ratings = fitRatings(ladder.players.length, ladder.races.map(r => r.order));
  const strongest = new Set(poolOf().sort((a, b) => ratings[b.id].r - ratings[a.id].r).slice(0, POOL_ACTIVE));
  for (const p of poolOf()) if (!strongest.has(p)) delete p.genes;
  writeJson(LADDER, { ...ladder, players: ladder.players.map(({ genes, ...p }) => genes ? { ...p, genes: Array.from(genes, g => +g.toFixed(5)) } : p) });
  writeJson(RATING, {
    gen, every: RATE_EVERY, races: ladder.races.length,
    players: ladder.players.map(({ genes, layers, ...p }) => ({ ...p, pool: !!genes, r: Math.round(ratings[p.id].r), sd: Math.round(ratings[p.id].sd) })),
  });
  const of = p => ({ name: p.name, r: Math.round(ratings[p.id].r), sd: Math.round(ratings[p.id].sd) });
  entry.rating = { champion: champ && of(champ), designs: Object.fromEntries(entrants.slice(0, bySpecies.length).map(e => [e.player.design, of(e.player)])) };
}

// ---- selection: within a species, a clear laggard is replaced by a mutated copy of a strong sibling ----
function select(st, entry, rng) {
  // a founding period (cfg.founding generations): early results are mostly luck, so every line gets time to learn
  // to drive before anyone is judged; after it, cfg.familySeats caps each family until cfg.familySeatsUntil
  if (entry.gen < (cfg.founding ?? 0)) return;
  const seats = cfg.familySeats && entry.gen < (cfg.familySeatsUntil ?? Infinity) ? cfg.familySeats : undefined;
  const score = a => mean(a.recentPoints?.length ? a.recentPoints : [a.last.points]);
  for (const sp of new Set(st.population.map(a => a.species))) {
    const members = st.population.filter(a => a.species === sp).sort((x, y) => score(y) - score(x));
    const worst = members.at(-1);
    if (entry.gen - worst.born < cfg.settle) continue;
    const parent = cfg.familyCap
      ? pickParent(members, worst, score, { margin: cfg.margin, fromTop: cfg.pickFromTop, seats }, rng)
      : score(members[0]) - score(worst) >= cfg.margin ? members[0] : null;
    if (!parent) continue;
    const n = st.clones[parent.founder] = (st.clones[parent.founder] || 1) + 1, name = `${parent.founder}·${n}`;
    entry.replaced.push({ out: worst.name, by: name, parent: parent.name, species: sp });
    const jitter = () => rng() < 0.5 ? 0.8 : 1.25;
    // not an exact clone: starting a little way off, the copy explores its own direction from the first round
    const eps = cfg.mutate ? noise((Math.imul(entry.gen, 2654435761) ^ Math.imul(n, 40503) ^ sp.charCodeAt(0)) >>> 0, parent.genes.length) : null;
    const gs = scaleOf(parent);
    const genes = eps ? parent.genes.map((g, j) => g + cfg.mutate * parent.sigma * (gs ? gs[j] * eps[j] : eps[j])) : parent.genes.slice();
    Object.assign(worst, {
      name, founder: parent.founder, label: parent.label, parent: parent.name, born: entry.gen, genes, ...parent.mask && { mask: parent.mask },
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
  const swaps = entry.replaced.map(r => `${r.out} → ${r.by}`).join(', '), rt = entry.rating;
  const rating = rt ? ` | rating: ${Object.entries(rt.designs).map(([d, x]) => `${d} ${x.r}±${Math.round(1.96 * x.sd)}`).join('  ')}` : '';
  console.log(`gen ${entry.gen} | ${minutes.toFixed(1)} min | champion ${entry.champion} | race points by design: ${sp}${rating}${swaps ? ` | replaced ${swaps}` : ''}${entry.inducted ? ` | hall of fame: ${entry.inducted}` : ''}`);
}

(async () => {
  const st = load() || found();
  cfg = { ...RECIPE, ...st.config };
  if (!args.pairs) opt.pairs = cfg.pairs;
  ladder = loadLadder();
  if (ladder.races.length) ratings = fitRatings(ladder.players.length, ladder.races.map(r => r.order));
  // rated at generation 0, every RATE_EVERY generations, and straight away for a save that has never been
  const rateDue = gen => gen % RATE_EVERY === 0 || !ladder.races.length;
  for (const address of (args.remote || '').split(',').filter(Boolean)) {
    known.add(address);
    await connectRemote(address, args.token);
  }
  if (args['remote-file']) watchRemotes(args['remote-file'], args.token);
  console.log(`[evolve] ${st.population.length} pros, generation ${st.generation}, ${slots.length} threads on ${1 + remotes.length} machine(s), ${opt.pairs} pairs × ${opt.rounds} rounds per generation`);
  // a new save's opening tournament and rating: generation 0, or the generation a save was founded from
  if (!st.history.length) {
    const t0 = Date.now(), entry = await tournament(st, st.generation);
    await rate(st, st.generation, entry);
    st.history.push(entry);
    save(st);
    log(entry, (Date.now() - t0) / 60000);
  }
  for (;;) {
    // --until N: stop once generation N is saved (a capped run, a test); a restarted run that's already there stops at once
    if (args.until && st.generation >= +args.until) process.exit(0);
    const gen = st.generation + 1, t0 = Date.now();
    report({ started: new Date(t0).toISOString(), lastMinutes: progress.lastMinutes });
    await train(st, gen);
    const entry = await tournament(st, gen);
    if (rateDue(gen)) {
      report({ phase: 'rating' });
      await rate(st, gen, entry);
    }
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
