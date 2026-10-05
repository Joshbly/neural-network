#!/usr/bin/env node
// The race workers for gradient learning: a pool of threads that turns the policy's weights into experience.
//   node train/ppo/rollouts.js [--threads N] [--dir /dev/shm/ppo]
// Spoken to by the learner in JSON lines on stdin; replies on stdout (logs go to stderr):
//   {"type":"init","seed":7,"out":".../genes-0.bin"}     design F's starting weights (initParams, as evolution starts
//                                                        it), published to 5 decimals → {"type":"inited","genes":N}
//   {"type":"collect","iteration":k,"genes":".../genes-k.bin","sigma":[s0,s1],"hold":H,"decisions":D}
//       episodes 0, 1, 2, ... of iteration k (oval and noise from k and the index: train/ppo/episode.js episodeOf)
//       until D decisions are in; each thread appends its episodes' rows to iter-k-thread-j.bin (train/ppo/spec.js
//       ROW, float32) → {"type":"collected","iteration":k,"decisions":n,"seconds":s,"perSecond":r,"episodes":[...]}
//       each episode: { index, file, row (offset in rows), rows, summary }
//   {"type":"evaluate","genes":".../genes.bin","episodes":[{"trackId":"bristol","laps":13}, ...]}
//       those time trials with no exploration (the brain drives its decisions, as evolution's timeTrial scores
//       them), nothing written → {"type":"evaluated","seconds":s,"episodes":[summary, ...]} in the order asked;
//       an episode with sigma and seed races exploring instead (how the policy does with its own noise)
//   {"type":"quit"}
// Weights files are raw little-endian float32 in the gene layout (spec.design.layout).
const fs = require('fs'), os = require('os'), path = require('path'), readline = require('readline');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { E, initParams, keepOvals } = require('../lib');
const { DESIGN, ROW } = require('./spec');
const { episodeOf, runEpisode } = require('./episode');

const GENES = E.geneCount(DESIGN.layers, DESIGN.shortcuts);
const readGenes = file => {
  const bytes = fs.readFileSync(file);
  if (bytes.length !== GENES * 4) throw new Error(`${file}: ${bytes.length / 4} weights, design F has ${GENES}`);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
};

if (!isMainThread) {
  // a thread: load the iteration's weights once, then race the episodes it's handed, appending their rows
  keepOvals(E.NASCAR_TRACKS.length);
  let genes = null, sigma = null, hold = 1, rolling = 0, timed = false, iteration = -1, fd = null, file = null, written = 0;
  parentPort.on('message', msg => {
    if (msg.type === 'trial') {
      // read every time (86 KB): a weights file can be rewritten under the same name
      const { summary } = runEpisode({ layers: DESIGN.layers, genes: readGenes(msg.genes), sigma: msg.sigma ?? null, seed: msg.seed, hold: msg.hold, trackId: msg.trackId, laps: msg.laps });
      return parentPort.postMessage({ type: 'judged', index: msg.index, summary });
    }
    if (msg.type === 'load') {
      if (fd !== null) fs.closeSync(fd);
      ({ iteration, sigma, hold = 1, rolling = 0, timed = false } = msg);
      genes = readGenes(msg.genes);
      file = path.join(workerData.dir, `iter-${iteration}-thread-${workerData.index}.bin`);
      fd = fs.openSync(file, 'w');
      written = 0;
      return parentPort.postMessage({ type: 'loaded' });
    }
    if (msg.type === 'episode') {
      const { rows, count, summary } = runEpisode({ layers: DESIGN.layers, genes, sigma, hold, rolling, timed, ...episodeOf(iteration, msg.index) });
      fs.writeSync(fd, rows);
      parentPort.postMessage({ type: 'done', index: msg.index, file, row: written, rows: count, summary });
      written += count;
    }
  });
  return;
}

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const threads = +(args.threads ?? Math.max(1, os.cpus().length - 1)), dir = args.dir ?? path.join(os.tmpdir(), `ppo-${process.pid}`);
fs.mkdirSync(dir, { recursive: true });
const workers = Array.from({ length: threads }, (_, index) => new Worker(__filename, { workerData: { dir, index } }));
const say = msg => process.stdout.write(`${JSON.stringify(msg)}\n`);
const ask = (w, msg) => new Promise(resolve => { w.once('message', resolve); w.postMessage(msg); });
for (const w of workers) w.on('error', err => { console.error(err.stack ?? err.message); say({ type: 'error', message: err.message }); process.exit(1); });

async function collect({ iteration, genes, sigma, hold, rolling, timed, decisions }) {
  const t0 = Date.now();
  await Promise.all(workers.map(w => ask(w, { type: 'load', iteration, genes, sigma, hold, rolling, timed })));
  const episodes = [];
  let next = 0, total = 0;
  // episodes handed out in index order; no new ones once the budget is met, those under way finish and count
  await Promise.all(workers.map(async w => {
    while (total < decisions) {
      const done = await ask(w, { type: 'episode', index: next++ });
      episodes.push(done);
      total += done.rows;
    }
  }));
  episodes.sort((a, b) => a.index - b.index);
  const seconds = (Date.now() - t0) / 1000;
  say({ type: 'collected', iteration, decisions: total, seconds, perSecond: Math.round(total / seconds), episodes: episodes.map(({ type, ...e }) => e) });
}

async function evaluate({ genes, episodes }) {
  const t0 = Date.now(), summaries = [];
  let next = 0;
  await Promise.all(workers.map(async w => {
    while (next < episodes.length) {
      const index = next++;
      summaries[index] = (await ask(w, { type: 'trial', index, genes, ...episodes[index] })).summary;
    }
  }));
  say({ type: 'evaluated', seconds: (Date.now() - t0) / 1000, episodes: summaries });
}

say({ type: 'ready', threads, dir, genes: GENES, row: ROW.width });
const lines = readline.createInterface({ input: process.stdin });
let queue = Promise.resolve();
lines.on('line', line => {
  if (!line.trim()) return;
  const msg = JSON.parse(line);
  queue = queue.then(async () => {
    if (msg.type === 'init') {
      const genes = E.quantizeGenes(initParams(DESIGN.layers, msg.seed, DESIGN));
      fs.writeFileSync(msg.out, Buffer.from(genes.buffer));
      say({ type: 'inited', genes: genes.length });
    } else if (msg.type === 'collect') await collect(msg);
    else if (msg.type === 'evaluate') await evaluate(msg);
    else if (msg.type === 'quit') {
      await Promise.all(workers.map(w => w.terminate()));
      process.exit(0);
    } else say({ type: 'error', message: `unknown message ${msg.type}` });
  }).catch(err => { console.error(err.stack ?? err.message); say({ type: 'error', message: err.message }); });
});
lines.on('close', () => queue.then(() => Promise.all(workers.map(w => w.terminate()))).then(() => process.exit(0)));
