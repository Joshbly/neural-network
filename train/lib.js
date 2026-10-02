// Shared by the trainer and its workers: the browser engine plus episode runners.
const fs = require('fs');
const path = require('path');

const files = ['nn.js', 'track.js', 'car.js', 'heat.js'];
const source = files.map(f => fs.readFileSync(path.join(__dirname, '..', 'js', f), 'utf8')).join('\n');
const E = new Function(`${source}
return { Brain, Track, Heat, Car, IN, mulberry32, geneCount, INPUT_COUNT, HALF_WIDTH, REAR_CAR_RAYS, clamp };`)();

// deterministic Gaussian noise from a seed, so perturbations never need to be shipped between threads
function noise(seed, n) {
  const rng = E.mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * Math.log(1 - rng())), a = 2 * Math.PI * rng();
    out[i] = r * Math.cos(a);
    if (i + 1 < n) out[i + 1] = r * Math.sin(a);
  }
  return out;
}

function perturb(theta, seed, scale) {
  const eps = noise(seed, theta.length), genes = new Float32Array(theta.length);
  for (let i = 0; i < theta.length; i++) genes[i] = theta[i] + scale * eps[i];
  return genes;
}

// Xavier-style hidden layers: weights ~ N(0, 1/fan_in), zero biases. The output layer starts near zero
// with a positive throttle bias, so every fresh policy is "drive straight, steady throttle" whatever
// its random hidden weights: something measurable from iteration one, never a dead start.
// Upgrade a brain trained before newer inputs were appended: they get zero weights, so it drives
// exactly as it did and training can then learn to use them.
function widen(layers, genes) {
  const known = layers[0], extra = E.INPUT_COUNT - known;
  if (extra <= 0) return { layers, genes };
  const wide = [E.INPUT_COUNT, ...layers.slice(1)], out = new Float32Array(E.geneCount(wide));
  for (let j = 0; j < layers[1]; j++) out.set(genes.subarray(j * (known + 1), (j + 1) * (known + 1)), j * (known + extra + 1));
  out.set(genes.subarray(layers[1] * (known + 1)), layers[1] * (known + extra + 1));
  return { layers: wide, genes: out };
}

function initParams(layers, seed) {
  const rng = noise(seed, E.geneCount(layers)), genes = new Float32Array(rng.length);
  let k = 0;
  for (let l = 1; l < layers.length; l++) {
    const output = l === layers.length - 1, scale = (output ? 0.01 : 1) / Math.sqrt(layers[l - 1]);
    for (let j = 0; j < layers[l]; j++) {
      genes[k++] = output && j === 1 ? 1 : 0;
      for (let i = 0; i < layers[l - 1]; i++, k++) genes[k] = rng[k] * scale;
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

// Share of downforce lost by the flag. Training races are a few laps but the real ones run ten, so the
// car you bring home is scored as if you still had to race it.
const worn = car => car.condition.front + car.condition.rear;
// Share of the race spent glued to the bumper of the car ahead (within two lengths, weighted by closeness).
// Racing from further back in the slipstream is free; sitting on a bumper instead of passing is not:
// a whole race of it costs about what dropping half a field would.
const tail = car => car.tailSteps / Math.max(1, car.steps), TAIL_PENALTY = 0.5;
// Every wall contact costs reward on top of the speed and downforce it costs on track: about a place in a
// 12-car practice race for each one, so braking for the corner always beats bouncing off the barrier.
const WALL_PENALTY = { race: 0.12, solo: 0.04 };
// Winning is the marker of success. P1 is worth more than twice P2 and the rest slide down to nothing, so a
// win with a scrape beats a clean second and sitting in P2 is never good enough; moving up still counts a
// little. Practice and the tournament use the same table.
const racePoints = (place, n) => place === 0 ? 1 : 0.45 * (n - 1 - place) / Math.max(1, n - 2);
// Leading is rewarded for as long as you hold it: take the lead and keep everyone else from winning.
const led = car => car.ledSteps / Math.max(1, car.steps), LEAD_BONUS = 0.3;

// Solo laps against the clock: pace and clean driving, no traffic noise.
function timeTrial({ layers, genes, trackSeed, laps }) {
  const t = track(trackSeed), heat = new E.Heat(t, [new E.Brain(layers, genes)], laps);
  while (!heat.over) heat.tick();
  const car = heat.cars[0], share = E.clamp((car.progress + car.gridOffset) / (laps * t.length), 0, 1);
  // a DNF (stalled or wrong way) must always score below crashing forward, or "never move" becomes
  // a local optimum that wall penalties alone would make attractive
  return {
    score: (car.finished ? 1 + (1 - car.steps / heat.maxSteps) : share) - WALL_PENALTY.solo * car.wallHits - 0.001 * car.impact - 0.5 * worn(car) - (car.retired ? 0.5 : 0),
    finished: car.finished, lap: car.laps.length ? Math.min(...car.laps) / 60 : null, walls: car.wallHits, aero: worn(car),
  };
}

// A race: the candidate starts from `slot` among opponents. Winning and leading are what count; distance
// and finishing time only break ties; walls, ramming, damage and riding bumpers cost you.
function race({ layers, genes, opponents, trackSeed, slot, laps, blind }) {
  const t = track(trackSeed), brains = opponents.map(o => new E.Brain(o.layers, o.genes));
  brains.splice(slot, 0, new E.Brain(layers, genes));
  const heat = new E.Heat(t, brains, laps), me = heat.cars[slot];
  if (blind) {
    // the "mirrors blacked out" control for checking whether a driver actually uses rear awareness
    const sense = me.sense.bind(me);
    me.sense = (tr, rivals) => {
      sense(tr, rivals);
      for (const r of E.REAR_CAR_RAYS) me.inputs[E.IN.cars + r] = 0;
      me.inputs[E.IN.rearClosing] = me.inputs[E.IN.attacker] = 0;
    };
  }
  while (!heat.over) heat.tick();
  const n = heat.cars.length, place = heat.standings().indexOf(me);
  const share = E.clamp((me.progress + me.gridOffset) / (laps * t.length), 0, 1);
  return {
    score: racePoints(place, n) + LEAD_BONUS * led(me) + 0.15 * share + (me.finished ? 0.05 * (1 - me.steps / heat.maxSteps) : 0)
      - WALL_PENALTY.race * me.wallHits - 0.1 * me.rammed - worn(me) - TAIL_PENALTY * tail(me) - (me.retired ? 0.5 : 0),
    place, won: place === 0, finished: me.finished, walls: me.wallHits, passes: me.overtakes, passedBy: me.passedBy,
    rammed: me.rammed, aero: worn(me), tail: tail(me), led: led(me), lap: me.laps.length ? Math.min(...me.laps) / 60 : null,
  };
}

// A full field of independent drivers; returns every car's result in grid order.
function fieldRace({ drivers, trackSeed, laps }) {
  const t = track(trackSeed), heat = new E.Heat(t, drivers.map(d => new E.Brain(d.layers, Float32Array.from(d.genes))), laps);
  while (!heat.over) heat.tick();
  const standings = heat.standings();
  return heat.cars.map(car => ({
    place: standings.indexOf(car), finished: car.finished, retired: car.retired, walls: car.wallHits,
    passes: car.overtakes, passedBy: car.passedBy, rammed: car.rammed, aero: worn(car), tail: tail(car), led: led(car), draft: car.draftSteps / Math.max(1, car.steps),
    lap: car.laps.length ? Math.min(...car.laps) / 60 : null,
  }));
}

const RUNNERS = { tt: timeTrial, race, field: fieldRace };
const run = scenario => RUNNERS[scenario.kind](scenario);

module.exports = { E, noise, perturb, initParams, widen, run, track, racePoints };
