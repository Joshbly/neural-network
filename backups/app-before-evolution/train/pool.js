// Runs standalone jobs (time trials, races, full-field races) across all cores.
const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');

function createPool(size = Math.max(1, os.cpus().length - 2)) {
  const workers = Array.from({ length: size }, () => new Worker(path.join(__dirname, 'worker.js')));
  let next = 0;
  const runAll = jobs => {
    const chunks = workers.map(() => []);
    jobs.forEach((job, i) => chunks[i % workers.length].push([i, job]));
    return Promise.all(workers.map((w, wi) => new Promise(resolve => {
      if (!chunks[wi].length) return resolve([]);
      const id = next++;
      const onMessage = msg => {
        if (msg.id !== id) return;
        w.off('message', onMessage);
        resolve(msg.results.map((r, k) => [chunks[wi][k][0], r]));
      };
      w.on('message', onMessage);
      w.postMessage({ id, jobs: chunks[wi].map(c => c[1]) });
    }))).then(parts => parts.flat().sort((a, b) => a[0] - b[0]).map(p => p[1]));
  };
  return { runAll, close: () => Promise.all(workers.map(w => w.terminate())) };
}

const loadDrivers = file => {
  const model = require(path.resolve(file)), base = path.basename(file, '.json');
  if (model.drivers) return model.drivers;
  return [{ label: base, layers: model.layers, genes: model.genes },
    ...(model.league || []).map(l => ({ label: `${base}·${l.label}`, layers: model.layers, genes: l.genes }))];
};

// a lineage's best and final checkpoints share league snapshots; keep one copy of each driver
const uniqueDrivers = drivers => {
  const seen = new Set();
  return drivers.filter(d => {
    const key = `${d.layers.join('-')}:${d.genes.slice(0, 64).join(',')}`;
    return !seen.has(key) && seen.add(key);
  });
};

module.exports = { createPool, loadDrivers, uniqueDrivers };
