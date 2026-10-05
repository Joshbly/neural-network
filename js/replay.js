// Everything it takes to run one of the engine's races again, exactly: the engine (train/lib.js, evolve.js)
// and the app both use these, so a race replayed in the app is the race the engine ran, car for car.

// two standard Gaussians from a seeded stream (Box-Muller in the deterministic math, so every machine draws the
// same), into a reused pair
const PAIR = new Float64Array(2);
function gaussianPair(rng, out = PAIR) {
  const r = Math.sqrt(-2 * dlog(1 - rng())), a = 2 * Math.PI * rng();
  out[0] = r * dcos(a);
  out[1] = r * dsin(a);
  return out;
}

// the noise that makes each practice copy of a brain: deterministic from a seed, so it never has to be shipped
function noiseVector(seed, n) {
  const rng = mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const g = gaussianPair(rng);
    out[i] = g[0];
    if (i + 1 < n) out[i + 1] = g[1];
  }
  return out;
}

// the seed of a car's exploration noise (Car.tryAround) in episode `episode` of rollout worker `worker` in learning
// iteration `iteration` (gradient learning)
const explorationSeed = (iteration, worker, episode) =>
  (Math.imul(iteration + 1, 2654435761) ^ Math.imul(worker + 1, 40503) ^ Math.imul(episode + 1, 97531) ^ 0x9e3779b9) >>> 0;

// Gradient learning's rolling starts: a share of practice time trials begin somewhere round the oval, already at
// speed and off the wall, so the brain learns to take every corner from many states instead of polishing one line
// from the grid (from the grid, a random brain slides along the outside wall and learning starts there). The start
// comes from the episode's seed, so the app can race the episode again exactly. null: from the grid.
function rollingStart(seed, track, share) {
  if (!share || !track.nascar) return null;
  const rng = mulberry32((seed ^ 0x5bd1e995) >>> 0);
  if (rng() >= share) return null;
  return { arc: rng() * track.length, lateral: (rng() * 1.2 - 0.6) * track.halfWidth, speed: (0.3 + 0.45 * rng()) * track.refSpeed };
}
// put a car at a start: heading down the track at that speed, the distance it covers counted from there
function placeAt(car, track, { arc, lateral, speed }) {
  const [x, y] = track.pointAt(arc, lateral), angle = track.headingAt(((arc % track.length) + track.length) % track.length);
  Object.assign(car, { x, y, angle, vx: speed * dcos(angle), vy: speed * dsin(angle) });
  car.lastArc = track.progressAt(x, y);
  car.progress = car.lastArc > track.length / 2 ? car.lastArc - track.length : car.lastArc;
  car.gridOffset = -car.progress;
  car.checkpoint = car.progress;
}

// gs: per-weight nudge scale from geneScale (0 keeps a locked connection exactly 0); null nudges every weight alike
function perturbGenes(theta, seed, scale, gs = null) {
  const eps = noiseVector(seed, theta.length), genes = new Float32Array(theta.length);
  if (gs) for (let i = 0; i < theta.length; i++) genes[i] = theta[i] + scale * (gs[i] * eps[i]);
  else for (let i = 0; i < theta.length; i++) genes[i] = theta[i] + scale * eps[i];
  return genes;
}

// A brain's locked connections. mask { layer, from, to, inputs, boost }, or a list of them (design F's lane and traffic
// blocks): neurons from..to-1 of `layer` take only `inputs` from the layer below, and every other connection into them
// stays exactly 0 for good; boost scales the nudge on the block's own weights and on its connections into the next
// layer. Per weight: 0 locked, boost for the block, 1 everywhere else. null for a brain without a mask. count: the
// brain's weight count (more than geneCount(layers) with shortcuts).
function geneScale(layers, mask, count = geneCount(layers)) {
  if (!mask) return null;
  const scale = new Float32Array(count).fill(1);
  for (const block of [mask].flat()) {
    const boost = block.boost ?? 1, allowed = new Set(block.inputs);
    let at = 0;
    for (let l = 1; l < block.layer; l++) at += layers[l] * (layers[l - 1] + 1);
    const span = layers[block.layer - 1] + 1, next = at + layers[block.layer] * span, nextSpan = layers[block.layer] + 1;
    for (let j = block.from; j < block.to; j++) {
      scale[at + j * span] = boost;
      for (let i = 0; i < span - 1; i++) scale[at + j * span + 1 + i] = allowed.has(i) ? boost : 0;
    }
    for (let k = 0; k < layers[block.layer + 1]; k++)
      for (let j = block.from; j < block.to; j++) scale[next + k * nextSpan + 1 + j] = boost;
  }
  return scale;
}

// the seed of practice copy pair i of brain ai in round `round` of generation `gen` (+sigma and -sigma share it)
const esSeed = (gen, round, ai, i) => (Math.imul(gen, 2654435761) ^ Math.imul(round * 64 + ai, 40503) ^ Math.imul(i + 1, 97531)) >>> 0;

// weights as published (5 decimals): the engine races exactly these, so anyone loading them races the same car
const quantizeGenes = genes => Float32Array.from(genes, g => +g.toFixed(5));

// a race's options from its scenario: NASCAR races get race control (training cautions restart at once);
// normal races leave everything out, so they're exactly what they always were
const scenarioOptions = sc => sc.cars || sc.trackId ? { cars: sc.cars, stages: sc.stages, practice: !sc.stages, cautions: sc.cautions, fastCaution: true } : {};
