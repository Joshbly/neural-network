// Evaluates perturbed policies. Receives the current mean parameters and noise seeds, never vectors.
const { parentPort } = require('worker_threads');
const { perturb, run } = require('./lib');

parentPort.on('message', ({ id, layers, theta, sigma, tasks, scenarios, jobs }) => {
  // standalone jobs (field races, benchmarks) carry their own drivers
  if (jobs) return parentPort.postMessage({ id, results: jobs.map(run) });
  const results = tasks.map(({ key, seed, sign }) => {
    const genes = seed == null ? theta : perturb(theta, seed, sign * sigma);
    return { key, outcomes: scenarios.map(sc => run({ ...sc, layers, genes })) };
  });
  parentPort.postMessage({ id, results });
});
