// genes are laid out per neuron: [bias, w_from_0, w_from_1, ...]
const geneCount = layers => layers.slice(1).reduce((n, size, l) => n + size * (layers[l] + 1), 0);

// rational tanh approximation; plenty accurate for evolved weights and much cheaper than Math.tanh
const squash = x => x <= -3 ? -1 : x >= 3 ? 1 : x * (27 + x * x) / (27 + 9 * x * x);

function gaussian() {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

class Brain {
  constructor(layers, genes = Brain.random(layers)) {
    this.layers = layers;
    this.genes = genes;
    this.acts = layers.map(n => new Float32Array(n));
    this.offsets = [0, 0];
    for (let l = 2; l < layers.length; l++) this.offsets[l] = this.offsets[l - 1] + layers[l - 1] * (layers[l - 2] + 1);
  }

  static random(layers) {
    return Float32Array.from({ length: geneCount(layers) }, gaussian);
  }

  think(inputs) {
    const g = this.genes, acts = this.acts;
    // brains from before newer inputs were appended read just the inputs they know
    acts[0].set(inputs.length > acts[0].length ? inputs.subarray(0, acts[0].length) : inputs);
    let k = 0;
    for (let l = 1; l < acts.length; l++) {
      const src = acts[l - 1], dst = acts[l];
      for (let j = 0; j < dst.length; j++) {
        let sum = g[k++];
        for (let i = 0; i < src.length; i++) sum += src[i] * g[k++];
        dst[j] = squash(sum);
      }
    }
    return acts[acts.length - 1];
  }

  weight(l, i, j) {
    return this.genes[this.offsets[l] + j * (this.layers[l - 1] + 1) + 1 + i];
  }
}

// swaps whole neurons so useful feature detectors survive breeding intact
function crossover(mom, dad, layers) {
  const child = new Float32Array(mom.length);
  let k = 0;
  for (let l = 1; l < layers.length; l++) {
    const span = layers[l - 1] + 1;
    for (let j = 0; j < layers[l]; j++, k += span)
      child.set((Math.random() < 0.5 ? mom : dad).subarray(k, k + span), k);
  }
  return child;
}

function mutate(genes, rate) {
  for (let i = 0; i < genes.length; i++)
    if (Math.random() < rate) genes[i] += gaussian() * (Math.random() < 0.1 ? 1 : 0.2);
  return genes;
}

class Evolution {
  constructor(size, layers, mutationRate) {
    this.size = size;
    this.layers = layers;
    this.mutationRate = mutationRate;
    this.reset();
  }

  reset() {
    this.generation = 1;
    this.eliteCount = 0;
    this.genomes = Array.from({ length: this.size }, () => Brain.random(this.layers));
  }

  // ranked: genomes sorted best-first
  breed(ranked, elites) {
    const pool = ranked.slice(0, Math.round(this.size * 0.3));
    const pick = () => pool[Math.min(...[0, 1, 2].map(() => Math.random() * pool.length | 0))];

    const next = ranked.slice(0, elites);
    while (next.length < this.size * 0.15) next.push(mutate(ranked[0].slice(), this.mutationRate));
    while (next.length < this.size) next.push(mutate(crossover(pick(), pick(), this.layers), this.mutationRate));

    this.genomes = next;
    this.eliteCount = elites;
    this.generation++;
  }
}
