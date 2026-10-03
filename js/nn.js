// genes are laid out per neuron: [bias, w_from_0, w_from_1, ...]. A brain with shortcuts (design F) has outputs that
// also read the first hidden layer directly: each output neuron is [bias, w from the last hidden layer, w from the first].
const geneCount = (layers, shortcuts = false) =>
  layers.slice(1).reduce((n, size, l) => n + size * (layers[l] + 1), 0) + (shortcuts ? layers[1] * layers.at(-1) : 0);

// rational tanh approximation; plenty accurate for evolved weights and much cheaper than Math.tanh
const squash = x => x <= -3 ? -1 : x >= 3 ? 1 : x * (27 + x * x) / (27 + 9 * x * x);

function gaussian() {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

// The WebAssembly SIMD kernel (train/nn.wat, inlined as js/nn-wasm.js) does thinkJS's arithmetic in the
// same order and precision, so its results are bit-identical; it just runs 8 neurons side by side. Brains
// keep their weights and activations in its memory, handed out by a bump allocator that starts over when
// full (first giving every brain still in there its activations back as plain arrays). Without SIMD, or
// without nn-wasm.js, everything runs on thinkJS.
const NN = (() => {
  try {
    const bytes = Uint8Array.from(atob(NN_WASM), c => c.charCodeAt(0));
    const { memory, forward, forward2 } = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports;
    const { buffer } = memory, start = 33792;  // below: the kernel's scratch for one layer's inputs
    return { forward, forward2, f32: new Float32Array(buffer), f64: new Float64Array(buffer), i32: new Int32Array(buffer), start, next: start, end: buffer.byteLength, epoch: 0, resident: [] };
  } catch {
    return null;
  }
})();

function claim(bytes) {
  if (NN.next + bytes > NN.end) {
    for (const brain of NN.resident) brain.evict();
    NN.resident.length = 0;
    NN.epoch++;
    NN.next = NN.start;
  }
  const at = NN.next;
  NN.next += bytes;
  return at;
}

class Brain {
  static simd = !!NN;

  constructor(layers, genes = Brain.random(layers)) {
    this.layers = layers;
    this.genes = genes;
    // whether it has shortcuts is in its weight count, so every saved brain builds the same way as before
    this.shortcuts = genes.length !== geneCount(layers);
    if (this.shortcuts && genes.length !== geneCount(layers, true))
      throw new Error(`${genes.length} weights fit neither a ${layers.join('-')} brain nor one with shortcuts`);
    this.acts = layers.map(n => new Float32Array(n));
    this.offsets = [0, 0];
    for (let l = 2; l < layers.length; l++) this.offsets[l] = this.offsets[l - 1] + layers[l - 1] * (layers[l - 2] + 1);
    this.pair = new Float32Array(4);
    this.epoch = -1;
    // the neuron lab: [layer, neuron, value] held fixed in the real-world pass (the mirror pass runs as usual)
    this.held = null;
  }

  static random(layers, shortcuts = false) {
    return Float32Array.from({ length: geneCount(layers, shortcuts) }, gaussian);
  }

  think(inputs) {
    if (!Brain.simd || this.held) return this.thinkJS(inputs, this.held);
    if (this.epoch !== NN.epoch) this.upload();
    const n = this.layers[0];
    this.acts[0].set(inputs.length > n ? inputs.subarray(0, n) : inputs);
    NN.forward(this.desc);
    return this.acts[this.acts.length - 1];
  }

  // a decision's two passes in one: [mirrorSteer, mirrorThrottle, steer, throttle], leaving acts exactly as
  // think(mirrored) followed by think(inputs) would
  thinkPair(mirrored, inputs) {
    const pair = this.pair;
    if (!Brain.simd || this.held) {
      const flipped = this.thinkJS(mirrored);
      pair[0] = flipped[0];
      pair[1] = flipped[1];
      const out = this.thinkJS(inputs, this.held);
      pair[2] = out[0];
      pair[3] = out[1];
      return pair;
    }
    if (this.epoch !== NN.epoch) this.upload();
    const n = this.layers[0];
    this.acts[0].set(inputs.length > n ? inputs.subarray(0, n) : inputs);
    this.mirror[0].set(mirrored.length > n ? mirrored.subarray(0, n) : mirrored);
    NN.forward2(this.desc);
    const flipped = this.mirror[this.mirror.length - 1], out = this.acts[this.acts.length - 1];
    pair[0] = flipped[0];
    pair[1] = flipped[1];
    pair[2] = out[0];
    pair[3] = out[1];
    return pair;
  }

  // weights into the kernel's memory as float64, in blocks of 8 neurons (biases, then the 8 weights from
  // each input, zero neurons padding the last block), with room for both passes' activations. The kernel reads a
  // layer's inputs as one run of memory, so a brain with shortcuts keeps its first hidden layer's activations right
  // after its last one's: the outputs read both as a single run, in their weights' order.
  upload() {
    const { layers, genes, offsets, shortcuts } = this, L = layers.length - 1;
    if (Math.max(...layers) > 1024) throw new Error('layers wider than 1024 neurons overflow the kernel scratch');
    if (shortcuts && (layers[1] % 8 || layers[L - 1] % 8)) throw new Error('a brain with shortcuts needs hidden layers in whole blocks of 8');
    const blocks = layers.map(n => Math.ceil(n / 8));
    const width = layers.map((n, l) => l ? blocks[l] * 8 : Math.ceil(n / 4) * 4);
    const ins = layers.map((n, l) => l === L && shortcuts ? layers[L - 1] + layers[1] : layers[l - 1]);
    let weights = 0;
    for (let l = 1; l <= L; l++) weights += blocks[l] * 8 * (ins[l] + 1);
    const actFloats = width.reduce((a, b) => a + b, 0), descBytes = 16 * Math.ceil((1 + 7 * L) / 4);
    const { f32, f64, i32 } = NN, base = claim(descBytes + 8 * weights + 8 * actFloats);
    const desc = base / 4, actsAt = [], mirrorAt = [];
    const order = shortcuts ? [0, ...Array.from({ length: L - 2 }, (_, k) => k + 2), 1, L] : layers.map((_, l) => l);
    let w = (base + descBytes) / 8, p = (base + descBytes) / 4 + 2 * weights;
    for (const l of order) [actsAt[l], p] = [p, p + width[l]];
    for (const l of order) [mirrorAt[l], p] = [p, p + width[l]];

    i32[desc] = L;
    for (let l = 1; l <= L; l++) {
      const nIn = ins[l], nOut = layers[l], span = nIn + 1, d = desc + 1 + 7 * (l - 1);
      i32[d] = 8 * w;
      i32[d + 1] = nIn;
      i32[d + 2] = blocks[l];
      i32[d + 3] = 4 * actsAt[l - 1];
      i32[d + 4] = 4 * actsAt[l];
      i32[d + 5] = 4 * mirrorAt[l - 1];
      i32[d + 6] = 4 * mirrorAt[l];
      for (let b = 0; b < blocks[l]; b++)
        for (let r = 0; r < span; r++)
          for (let j = b * 8; j < b * 8 + 8; j++) f64[w++] = j < nOut ? genes[offsets[l] + j * span + r] : 0;
    }
    const shown = this.acts;
    this.acts = layers.map((n, l) => f32.subarray(actsAt[l], actsAt[l] + n));
    this.mirror = layers.map((n, l) => f32.subarray(mirrorAt[l], mirrorAt[l] + n));
    shown.forEach((a, l) => this.acts[l].set(a));
    this.desc = 4 * desc;
    this.epoch = NN.epoch;
    NN.resident.push(this);
  }

  // the kernel's memory is being reused: keep the latest activations (the brain view draws them) as plain arrays
  evict() {
    this.acts = this.acts.map(a => a.slice());
    this.mirror = null;
    this.epoch = -1;
  }

  thinkJS(inputs, held) {
    const g = this.genes, acts = this.acts, L = acts.length - 1;
    // brains from before newer inputs were appended read just the inputs they know
    acts[0].set(inputs.length > acts[0].length ? inputs.subarray(0, acts[0].length) : inputs);
    let k = 0;
    for (let l = 1; l <= L; l++) {
      const src = acts[l - 1], dst = acts[l], shortcut = this.shortcuts && l === L ? acts[1] : null;
      for (let j = 0; j < dst.length; j++) {
        let sum = g[k++];
        for (let i = 0; i < src.length; i++) sum += src[i] * g[k++];
        if (shortcut) for (let i = 0; i < shortcut.length; i++) sum += shortcut[i] * g[k++];
        dst[j] = squash(sum);
      }
      if (held) for (const [hl, j, v] of held) if (hl === l) dst[j] = v;
    }
    return acts[L];
  }

  // input i of layer l's neuron j; for the outputs of a brain with shortcuts, i past the last hidden layer reaches
  // into the first
  weight(l, i, j) {
    const { layers } = this, span = layers[l - 1] + 1 + (this.shortcuts && l === layers.length - 1 ? layers[1] : 0);
    return this.genes[this.offsets[l] + j * span + 1 + i];
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
