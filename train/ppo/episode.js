// One time trial for gradient learning: a stock car alone on a NASCAR oval, trying actions around its decisions
// (Car.tryAround), every decision recorded as a row (train/ppo/spec.js ROW). The reward for a decision is how much
// the time-trial score (train/lib.js ttScore) changed until the next one, so an episode's rewards add up to the
// score evolution would give it.
const { E, oval, ttScore } = require('../lib');
const { METRES, ROW, extrasOf } = require('./spec');

// which oval and which noise episode e of iteration k gets, so any episode can be raced again exactly
const episodeOf = (iteration, e) => {
  const trackId = E.pickFrom(E.NASCAR_TRACKS.map(t => t.id), `tt${iteration}:${e}`);
  return { trackId, laps: E.lapsFor(trackId, METRES), seed: E.explorationSeed(iteration, 0, e) };
};

// the shimmy gauges: how far the steering decisions jump about their half-second average (jitter) and how often they
// change direction (reversals a second; generation 77 hit 27 at Bristol, where it crashed); and the slow swing, the
// steering's rms about its 4-second average (generation 77 swung ±0.32 every 0.55 s at Daytona, F5@40 ±0.22 as dither)
function jitterOf(steer) {
  let sq = 0, n = 0, flips = 0, prev = 0, sw = 0, ns = 0;
  for (let i = 7; i < steer.length - 7; i++) {
    let m = 0;
    for (let j = i - 7; j <= i + 7; j++) m += steer[j];
    sq += (steer[i] - m / 15) ** 2;
    n++;
    const d = steer[i] - steer[i - 1];
    if (d * prev < 0 && Math.abs(d) > 0.02) flips++;
    prev = d;
  }
  for (let i = 60; i < steer.length - 60; i++) {
    let m = 0;
    for (let j = i - 60; j <= i + 60; j++) m += steer[j];
    sw += (steer[i] - m / 121) ** 2;
    ns++;
  }
  return { jitter: n ? Math.sqrt(sq / n) : 0, reversals: n ? flips / n * 30 : 0, swing: ns ? Math.sqrt(sw / ns) : 0 };
}

// the rows of the episode just run, reused (copy or write them out before the next one)
let buffer = new Float32Array(ROW.width * 4096);

// sigma null: no exploration (the car drives its decisions, as in evolution); hold: decisions each noise draw lasts;
// rolling: the share of exploring episodes that start somewhere round the oval at speed (js/replay.js rollingStart);
// timed: the clock charged as it runs (below)
function runEpisode({ layers, genes, sigma, trackId, laps, seed, hold = 1, rolling = 0, timed = false }) {
  const heat = new E.Heat(oval(trackId), [new E.Brain(layers, genes)], laps, E.scenarioOptions({ kind: 'tt', trackId, laps, cars: 'stock' }));
  const car = heat.cars[0], W = ROW.width, start = sigma ? E.rollingStart(seed, heat.track, rolling) : null;
  if (start) E.placeAt(car, heat.track, start);
  if (sigma) car.explore = { sigma, seed, hold };
  // timed: the time-trial score pays for speed only through its bonus at the flag, minutes away at race distance and
  // discounted to nearly nothing, so crawling (31 mph at Bristol, lifting at Daytona) cost almost nothing. Charged per
  // decision instead, the clock costs as it runs: a finisher's rewards still add up to its score exactly; below about
  // half pole speed every decision loses; a car that doesn't finish also pays for the time it used
  const running = c => ttScore(c, heat) - (timed && !c.finished ? c.steps / heat.maxSteps : 0);
  let count = 0, before = 0, sum = 0, first = null, onWall = 0;
  const steer = [];
  car.onDecide = c => {
    onWall += c.touchingWall ? 1 : 0;
    steer.push(sigma ? c.mean[0] : c.action[0]);
    const now = running(c);
    if (count) {
      buffer[(count - 1) * W + ROW.reward] = now - before;
      sum += now - before;
    } else first = now;
    before = now;
    if ((count + 1) * W > buffer.length) {
      const bigger = new Float32Array(buffer.length * 2);
      bigger.set(buffer);
      buffer = bigger;
    }
    const at = count * W;
    buffer.set(c.inputs, at + ROW.obs[0]);
    extrasOf(c, heat, buffer, at + ROW.extras[0]);
    const tried = sigma ? c.raw : c.action;
    buffer[at + ROW.raw[0]] = tried[0];
    buffer[at + ROW.raw[0] + 1] = tried[1];
    buffer[at + ROW.mean[0]] = sigma ? c.mean[0] : c.action[0];
    buffer[at + ROW.mean[0] + 1] = sigma ? c.mean[1] : c.action[1];
    buffer[at + ROW.logp] = c.logp;
    buffer[at + ROW.done] = 0;
    count++;
  };
  while (!heat.over) heat.tick();
  const score = ttScore(car, heat), end = running(car);
  if (count) {
    buffer[(count - 1) * W + ROW.reward] = end - before;
    buffer[(count - 1) * W + ROW.done] = 1;
    sum += end - before;
  }
  return {
    rows: buffer.subarray(0, count * W), count,
    summary: {
      trackId, laps, seed, hold, ...start && { rolling: start }, steps: car.steps, decisions: count, finished: car.finished, retired: car.retired,
      parked: !!car.parked && car.condition.front + car.condition.rear > E.RC.parkDamage, score, return: sum, start: first,
      // share of decisions with the car against the wall: the wall-riding gauge
      wall: count ? onWall / count : 0,
      ...jitterOf(steer),
      bestLap: car.laps.length ? Math.min(...car.laps) / 60 : null, worn: car.condition.front + car.condition.rear,
      // a lap at the oval's real pole speed, in seconds: bestLap / pole compares pace across ovals
      pole: heat.track.length / heat.track.refSpeed / 60,
    },
  };
}

module.exports = { episodeOf, runEpisode };
