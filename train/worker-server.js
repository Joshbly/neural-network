#!/usr/bin/env node
// Extra racing power for the evolution engine, on another machine: a pool of evolve-workers taking jobs
// from train/evolve.js over TCP. Messages are newline-delimited JSON (weights travel as plain arrays);
// the first message must carry the run's token or the connection is dropped.
//   node train/worker-server.js --port 9000 --threads 80 --token SECRET
const net = require('net');
const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const port = +(args.port || 9000), threads = +(args.threads || os.cpus().length), token = args.token, lonelySeconds = +(args.lonely || 360);

const pool = Array.from({ length: threads }, () => new Worker(path.join(__dirname, 'evolve-worker.js')));
const idle = [...pool], queue = [];
let client = null;
const send = msg => client?.write(JSON.stringify(msg) + '\n');
const pump = () => {
  while (idle.length && queue.length) idle.pop().postMessage(queue.shift());
};
for (const w of pool) {
  w.on('message', ({ id, out }) => {
    send({ id, out });
    idle.push(w);
    pump();
  });
  w.on('error', err => {
    console.error(err);
    process.exit(1);
  });
}

// roster and weights arrive as arrays; the workers expect typed arrays
const revive = msg => {
  if (msg.type === 'roster') msg.drivers = msg.drivers.map(d => ({ layers: d.layers, genes: Float32Array.from(d.genes) }));
  if (msg.type === 'theta') msg.theta = Float32Array.from(msg.theta);
  return msg;
};

// an engine that never connects (or never comes back) shouldn't leave this machine billing
let lonelySince = Date.now();
setInterval(() => {
  if (!client && Date.now() - lonelySince > lonelySeconds * 1000) {
    console.log(`[worker-server] no engine connected for ${lonelySeconds} s; shutting down`);
    process.exit(0);
  }
}, 5e3);

net.createServer(sock => {
  sock.setNoDelay(true);
  sock.setEncoding('utf8');
  let buf = '', authed = false;
  sock.on('data', chunk => {
    buf += chunk;
    for (let i; (i = buf.indexOf('\n')) >= 0;) {
      const msg = JSON.parse(buf.slice(0, i));
      buf = buf.slice(i + 1);
      if (!authed) {
        if (msg.type !== 'hello' || msg.token !== token) return sock.destroy();
        authed = true;
        client = sock;
        sock.write(JSON.stringify({ type: 'ready', threads }) + '\n');
      } else if (msg.type === 'job') {
        queue.push(msg);
        pump();
      } else {
        const shared = revive(msg);
        pool.forEach(w => w.postMessage(shared));
      }
    }
  });
  sock.on('error', () => sock.destroy());
  sock.on('close', () => {
    if (client === sock) {
      client = null;
      queue.length = 0;
      lonelySince = Date.now();
    }
  });
}).listen(port, '0.0.0.0', () => console.log(`[worker-server] ${threads} threads on port ${port}`));
