// Shared by the trainer and its workers: the browser engine plus episode runners.
const fs = require('fs');
const path = require('path');

const files = ['dmath.js', 'nn-wasm.js', 'nn.js', 'nascar-tracks.js', 'track.js', 'car.js', 'heat.js', 'racecontrol.js', 'nascar.js', 'replay.js'];
const source = files.map(f => fs.readFileSync(path.join(__dirname, '..', 'js', f), 'utf8')).join('\n');
const E = new Function(`${source}
return { Brain, Track, OvalTrack, NASCAR_TRACKS, SURFACE, MPH_PER_SPEED, Heat, Car, RaceControl, IN, mulberry32, geneCount, INPUT_COUNT,
  HALF_WIDTH, REAR_CAR_RAYS, clamp, G, NORMAL, NASCAR_670, NASCAR_PLATE, specFor, stockSpec, steerLock,
  SUPERSPEEDWAYS, YARDSTICK_OVALS, practiceOvals, seasonOvals, lapsFor, pickFrom, OTHER_OVALS, RC,
  noiseVector, perturbGenes, geneScale, esSeed, quantizeGenes, scenarioOptions, WALL_RAY_DEG, CAR_RAY_DEG, LOOKAHEAD, DECIDE_EVERY,
  MIRROR_FROM, MIRROR_SIGN, PACE_RAY_DEG, DRIFT_RAY_DEG, CLOSING_GAIN, TRAFFIC_SENSES, ROAD_SENSES };`)();

// deterministic Gaussian noise from a seed (js/replay.js, shared with the app so it can rebuild any copy)
const noise = E.noiseVector, perturb = E.perturbGenes;

// Designs shaped beyond stacked layers. F (52-208-48-2): the first layer is 160 lane neurons that read only the road
// and self senses and 48 traffic neurons that read only the traffic senses, then 48 mixing neurons, and both blocks
// also wire straight to steer and gas (shortcuts).
const SHAPES = {
  F: {
    mask: [{ name: 'lane', layer: 1, from: 0, to: 160, inputs: E.ROAD_SENSES }, { name: 'traffic', layer: 1, from: 160, to: 208, inputs: E.TRAFFIC_SENSES }],
    shortcuts: true,
  },
};

// Xavier-style hidden layers: weights ~ N(0, 1/fan_in), zero biases. The output layer starts near zero
// with a positive throttle bias, so every fresh policy is "drive straight, steady throttle" whatever
// its random hidden weights: something measurable from iteration one, never a dead start.
// Upgrade a brain trained before newer inputs were appended: they get zero weights, so it drives
// exactly as it did and training can then learn to use them.
// A brain gaining the motion inputs reads the closing speeds louder (js/car.js CLOSING_GAIN): its weights on them
// shrink to match.
function widen(layers, genes) {
  const known = layers[0], extra = E.INPUT_COUNT - known;
  if (extra <= 0) return { layers, genes };
  const wide = [E.INPUT_COUNT, ...layers.slice(1)], out = new Float32Array(genes.length + layers[1] * extra), louder = known <= E.IN.pace;
  for (let j = 0; j < layers[1]; j++) {
    const at = j * (known + extra + 1);
    out.set(genes.subarray(j * (known + 1), (j + 1) * (known + 1)), at);
    if (louder) for (const i of [E.IN.closing, E.IN.rearClosing]) out[at + 1 + i] /= E.CLOSING_GAIN;
  }
  out.set(genes.subarray(layers[1] * (known + 1)), layers[1] * (known + extra + 1));
  return { layers: wide, genes: out };
}

// A shaped design (SHAPES): a locked block's neurons are scaled to the inputs they may read and start at 0 on the rest,
// and with shortcuts the outputs read the first hidden layer too.
function initParams(layers, seed, { mask, shortcuts = false } = {}) {
  const rng = noise(seed, E.geneCount(layers, shortcuts)), genes = new Float32Array(rng.length), L = layers.length - 1;
  const blocks = [mask ?? []].flat().map(b => ({ ...b, inputs: new Set(b.inputs) }));
  let k = 0;
  for (let l = 1; l <= L; l++) {
    const output = l === L, fanIn = layers[l - 1] + (output && shortcuts ? layers[1] : 0);
    for (let j = 0; j < layers[l]; j++) {
      const only = blocks.find(b => b.layer === l && j >= b.from && j < b.to)?.inputs;
      const scale = (output ? 0.01 : 1) / Math.sqrt(only ? only.size : fanIn);
      genes[k++] = output && j === 1 ? 1 : 0;
      for (let i = 0; i < fanIn; i++, k++) genes[k] = only && !only.has(i) ? 0 : rng[k] * scale;
    }
  }
  return genes;
}

// the evolution races on fresh tracks all the time (about 2 MB each), so only the most recent are kept
// 24 covers a generation (3 practice tracks a round, 8 tournament, 8 yardstick) while keeping 80 threads
// a machine well inside its memory
const tracks = new Map(), TRACK_CACHE = 24;
function track(seed) {
  const t = tracks.get(seed) ?? E.Track.random(E.mulberry32(seed));
  tracks.delete(seed);
  tracks.set(seed, t);
  if (tracks.size > TRACK_CACHE) tracks.delete(tracks.keys().next().value);
  return t;
}
// the real ovals are bigger (about 4 MB each, all 32 would be 116 MB a thread) and a generation only visits a
// dozen, so the 8 most recent stay built
const ovals = new Map(), OVAL_CACHE = 8;
function oval(id) {
  const t = ovals.get(id) ?? new E.OvalTrack(E.NASCAR_TRACKS.find(d => d.id === id));
  ovals.delete(id);
  ovals.set(id, t);
  if (ovals.size > OVAL_CACHE) ovals.delete(ovals.keys().next().value);
  return t;
}
// a scenario names a generated track by seed or a real oval by id, and which cars race: normal saves leave
// both out, so their races are exactly what they always were
const trackOf = sc => sc.trackId ? oval(sc.trackId) : track(sc.trackSeed);
const heatOf = (sc, t, brains) => new E.Heat(t, brains, sc.laps, E.scenarioOptions(sc));

// Share of downforce lost by the flag. Training races are a few laps but the real ones run ten, so the
// car you bring home is scored as if you still had to race it, at half weight: it already cost places here.
const worn = car => car.condition.front + car.condition.rear;
// Share of the race spent glued to the bumper of the car ahead (within two lengths, weighted by closeness),
// for the stats only: pushing someone just helps them win, and the physics shows that without a cost here.
const tail = car => car.tailSteps / Math.max(1, car.steps);
// Share of the race spent leaning door to door on another car, for the stats only: staying glued to someone's
// flank is its own punishment on track (two-wide drag, side drafts, a loose car), not a cost added here.
const rub = car => car.sideSteps / Math.max(1, car.steps);
// Arcade cars: every wall contact costs reward on top of the speed and downforce it costs on track, about a
// place in a 12-car practice race, so braking for the corner always beats bouncing off the barrier. Stock
// cars pay nothing here: a hard hit can bend them, cripple their aero or end their race (STOCK_BASE.damage).
const WALL_PENALTY = { race: 0.04, solo: 0.04 };
const walls = (car, perHit) => car.spec.stock ? 0 : perHit * car.wallHits;
// Winning is the marker of success: P1 is worth more than twice P2, so a win with a scrape beats a clean second
// and sitting in P2 is never good enough. Every place behind is worth fighting for too: from P2 the points fall
// in equal steps to -0.45 for last, 0.09 a place in a 12-car race, more than a scrape or a lap spent leading.
// Practice and the tournament use the same table.
const racePoints = (place, n) => place === 0 ? 1 : place === n - 1 ? -0.45 : 0.45 - 0.9 * (place - 1) / (n - 2);
// Leading only breaks ties: a whole race of it is about a place, so leading and losing is still losing.
const led = car => car.ledSteps / Math.max(1, car.steps), LEAD_BONUS = 0.05;
// On the ovals, breaking the rules costs reward on top of what it costs on track: a black flag (passing below
// the yellow line, jumping a restart) about a place and a half, bringing out a caution about one, and every
// moment spent below the line at the plate tracks a little, so the bottom of the track stays a no-go zone.
// Zero in normal races, where race control doesn't exist.
const YELLOW_LINE = 1;
const ruleCost = car => car.penalties === undefined ? 0 : 0.15 * car.penalties + 0.1 * car.cautionsCaused + YELLOW_LINE * car.belowLine / Math.max(1, car.steps);
// distance covered, counting a lap given back by the lucky dog
const covered = car => car.progress + car.bonus + car.gridOffset;

// Solo laps against the clock: pace and clean driving, no traffic noise.
function timeTrial(sc) {
  const { layers, genes, laps } = sc, t = trackOf(sc), heat = heatOf(sc, t, [new E.Brain(layers, genes)]);
  while (!heat.over) heat.tick();
  const car = heat.cars[0], share = E.clamp(covered(car) / (laps * t.length), 0, 1);
  // a DNF (stalled or wrong way) must always score below crashing forward, or "never move" becomes
  // a local optimum that the cost of crashing alone would make attractive
  return {
    score: (car.finished ? 1 + (1 - car.steps / heat.maxSteps) : share) - walls(car, WALL_PENALTY.solo) - (car.spec.stock ? 0 : 0.001 * car.impact) - 0.25 * worn(car) - (car.retired ? 0.5 : 0) - ruleCost(car),
    finished: car.finished, lap: car.laps.length ? Math.min(...car.laps) / 60 : null, walls: car.wallHits, aero: worn(car),
  };
}

// A race: the candidate starts from `slot` among opponents. Winning is what counts; leading, distance
// and finishing time only break ties; ramming and damage cost you, and walls too in arcade cars.
function race(sc) {
  const { layers, genes, opponents, slot, laps, blind } = sc, t = trackOf(sc), brains = opponents.map(o => new E.Brain(o.layers, o.genes));
  brains.splice(slot, 0, new E.Brain(layers, genes));
  const heat = heatOf(sc, t, brains), me = heat.cars[slot];
  if (blind) {
    // the "mirrors blacked out" control for checking whether a driver actually uses rear awareness
    const sense = me.sense.bind(me);
    me.sense = (tr, rivals) => {
      sense(tr, rivals);
      for (const r of E.REAR_CAR_RAYS) me.inputs[E.IN.cars + r] = 0;
      E.PACE_RAY_DEG.forEach((d, k) => { if (Math.abs(d) >= 150) me.inputs[E.IN.pace + k] = 0; });
      me.inputs[E.IN.rearClosing] = me.inputs[E.IN.attacker] = 0;
    };
  }
  while (!heat.over) heat.tick();
  const n = heat.cars.length, place = heat.standings().indexOf(me);
  const share = E.clamp(covered(me) / (laps * t.length), 0, 1);
  return {
    score: racePoints(place, n) + LEAD_BONUS * led(me) + 0.15 * share + (me.finished ? 0.05 * (1 - me.steps / heat.maxSteps) : 0)
      - walls(me, WALL_PENALTY.race) - 0.1 * me.rammed - 0.5 * worn(me) - (me.retired ? 0.5 : 0) - ruleCost(me),
    place, won: place === 0, finished: me.finished, walls: me.wallHits, passes: me.overtakes, passedBy: me.passedBy,
    rammed: me.rammed, aero: worn(me), tail: tail(me), rub: rub(me), led: led(me), lap: me.laps.length ? Math.min(...me.laps) / 60 : null,
  };
}

// A full field of independent drivers; returns every car's result in grid order.
function fieldRace(sc) {
  const t = trackOf(sc), heat = heatOf(sc, t, sc.drivers.map(d => new E.Brain(d.layers, Float32Array.from(d.genes))));
  while (!heat.over) heat.tick();
  const standings = heat.standings();
  return heat.cars.map(car => ({
    place: standings.indexOf(car), finished: car.finished, retired: car.retired, walls: car.wallHits,
    passes: car.overtakes, passedBy: car.passedBy, rammed: car.rammed, aero: worn(car), tail: tail(car), rub: rub(car), led: led(car), draft: car.draftSteps / Math.max(1, car.steps),
    lap: car.laps.length ? Math.min(...car.laps) / 60 : null,
    ...heat.control && { stagePoints: car.stagePoints, cautionsCaused: car.cautionsCaused, penalties: car.penalties, belowLine: car.belowLine / Math.max(1, car.steps), parked: car.parked },
  }));
}

const RUNNERS = { tt: timeTrial, race, field: fieldRace };
const run = scenario => RUNNERS[scenario.kind](scenario);

module.exports = { E, noise, perturb, initParams, widen, run, track, oval, racePoints, SHAPES };
