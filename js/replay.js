// Everything it takes to run one of the engine's races again, exactly: the engine (train/lib.js, evolve.js)
// and the app both use these, so a race replayed in the app is the race the engine ran, car for car.

// the noise that makes each practice copy of a brain: deterministic from a seed, so it never has to be shipped
function noiseVector(seed, n) {
  const rng = mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const r = Math.sqrt(-2 * dlog(1 - rng())), a = 2 * Math.PI * rng();
    out[i] = r * dcos(a);
    if (i + 1 < n) out[i + 1] = r * dsin(a);
  }
  return out;
}

function perturbGenes(theta, seed, scale) {
  const eps = noiseVector(seed, theta.length), genes = new Float32Array(theta.length);
  for (let i = 0; i < theta.length; i++) genes[i] = theta[i] + scale * eps[i];
  return genes;
}

// the seed of practice copy pair i of brain ai in round `round` of generation `gen` (+sigma and -sigma share it)
const esSeed = (gen, round, ai, i) => (Math.imul(gen, 2654435761) ^ Math.imul(round * 64 + ai, 40503) ^ Math.imul(i + 1, 97531)) >>> 0;

// weights as published (5 decimals): the engine races exactly these, so anyone loading them races the same car
const quantizeGenes = genes => Float32Array.from(genes, g => +g.toFixed(5));

// a race's options from its scenario: NASCAR races get race control (training cautions restart at once);
// normal races leave everything out, so they're exactly what they always were
const scenarioOptions = sc => sc.cars || sc.trackId ? { cars: sc.cars, stages: sc.stages, practice: !sc.stages, cautions: sc.cautions, fastCaution: true } : {};
