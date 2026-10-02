// Worker for the evolution engine. The population is sent once per generation ("roster") and each
// pro's current weights once per training round ("theta"); jobs then refer to drivers by index, so a
// thousand jobs a round never re-ship megabytes of weights.
const { parentPort } = require('worker_threads');
const { perturb, run } = require('./lib');

let roster = [];
const thetas = new Map();

parentPort.on('message', msg => {
  if (msg.type === 'roster') return void (roster = msg.drivers);
  if (msg.type === 'theta') return void thetas.set(msg.agent, msg);
  const { job } = msg;
  if (job.kind === 'es') {
    const { layers, theta } = thetas.get(job.agent), genes = perturb(theta, job.seed, job.sign * job.sigma);
    const scores = job.scenarios.map(sc => run({ ...sc, layers, genes, opponents: sc.rivals?.map(i => roster[i]) }).score);
    return parentPort.postMessage({ id: msg.id, out: scores });
  }
  // a full-field race between roster drivers: tournaments and the originals benchmark
  parentPort.postMessage({ id: msg.id, out: run({ ...job, kind: 'field', drivers: job.entrants.map(i => roster[i]) }) });
});
