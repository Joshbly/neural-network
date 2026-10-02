#!/usr/bin/env node
// Serves the app and runs the evolution engine (train/evolve.js) on every core but one.
// Up to 10 save slots live in models/slots/slot-N; exactly one trains at a time and saves itself
// every generation.
//   node server.js              → http://127.0.0.1:8777, the active slot starts training immediately
//   node server.js --no-engine  → just the app
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = __dirname, PORT = +process.env.PORT || 8777;
const SLOTS = path.join(ROOT, 'models', 'slots'), ACTIVE = path.join(SLOTS, 'active.json'), MAX_SLOTS = 10;
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml' };

const readJson = file => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
};
const writeJson = (file, value) => fs.writeFileSync(file, JSON.stringify(value));
const slotDir = id => path.join(SLOTS, id);
const validId = id => /^slot-([1-9]|10)$/.test(id || '') && fs.existsSync(slotDir(id));
const slotIds = () => fs.readdirSync(SLOTS).filter(d => /^slot-\d+$/.test(d)).sort((a, b) => a.slice(5) - b.slice(5));
const cleanName = name => String(name || '').replace(/[\u0000-\u001f]/g, '').trim().slice(0, 40);

fs.mkdirSync(SLOTS, { recursive: true });
let active = readJson(ACTIVE)?.id;
if (!validId(active)) active = slotIds()[0] || null;

// ---- the engine: one child process, training the active slot ----
let engine = null;
function startEngine() {
  if (engine || !active) return;
  const dir = slotDir(active), out = fs.openSync(path.join(dir, 'evolve.log'), 'a');
  engine = spawn(process.execPath, [path.join(ROOT, 'train', 'evolve.js'), '--dir', dir], { cwd: ROOT, stdio: ['ignore', out, out] });
  const me = engine;
  engine.on('exit', code => {
    console.log(`evolution engine stopped${code ? ` (exit ${code})` : ''}`);
    if (engine === me) engine = null;
  });
  console.log(`evolution engine training ${active} on ${os.cpus().length - 1} cores`);
}
const stopEngine = () => new Promise(resolve => {
  if (!engine) return resolve();
  engine.once('exit', () => resolve());
  engine.kill('SIGTERM');
});

// ---- cloud training on Modal (train/modal_train.py): the save trains remotely, files are pulled back ----
// Runs are started through train/modal_launch.py: deployed and spawned, so they don't depend on this laptop.
const MODAL = path.join(ROOT, '.venv', 'bin', 'modal'), CLOUD = path.join(SLOTS, 'cloud.json'), VOLUME = 'neural-racers';
const PYTHON = path.join(ROOT, '.venv', 'bin', 'python'), LAUNCH = path.join(ROOT, 'train', 'modal_launch.py');
let cloud = readJson(CLOUD);
const launch = args => new Promise(resolve => {
  const child = spawn(PYTHON, [LAUNCH, ...args], { cwd: ROOT });
  let out = '';
  child.stdout.on('data', d => out += d);
  child.stderr.on('data', d => out += d);
  child.on('close', code => resolve({ code, out: out.trim() }));
});
const modal = args => new Promise(resolve => {
  const child = spawn(MODAL, args, { cwd: ROOT });
  let out = '';
  child.stdout.on('data', d => out += d);
  child.stderr.on('data', d => out += d);
  child.on('close', code => resolve({ code, out }));
});
const pull = (slot, file) => modal(['volume', 'get', '--force', VOLUME, `/slots/${slot}/${file}`, path.join(slotDir(slot), file)]);

let syncing = false;
async function syncCloud() {
  if (!cloud || syncing) return;
  syncing = true;
  const { slot } = cloud, before = readJson(path.join(slotDir(slot), 'summary.json'))?.generation;
  const round = p => p && `${p.generation}:${p.round}:${p.phase}`, was = round(readJson(path.join(slotDir(slot), 'progress.json')));
  await pull(slot, 'progress.json');
  // each round's starting weights, so the app can show the practice that's running on Modal
  if (round(readJson(path.join(slotDir(slot), 'progress.json'))) !== was) await pull(slot, 'live.json');
  await pull(slot, 'summary.json');
  const now = readJson(path.join(slotDir(slot), 'summary.json'))?.generation;
  if (now != null && now !== before) {
    await pull(slot, 'state.json');
    fs.mkdirSync(path.join(slotDir(slot), 'generations'), { recursive: true });
    await pull(slot, `generations/gen-${String(now).padStart(4, '0')}.json`);
  }
  syncing = false;
}

// one 64-core machine with 24 GiB of memory
const MACHINE_PER_HOUR = 3.21;
async function startCloud(slot, hours, helpers) {
  cloud = { slot, started: new Date().toISOString(), until: Date.now() + hours * 3600e3, hours, helpers, perHour: MACHINE_PER_HOUR * (1 + helpers) };
  writeJson(CLOUD, cloud);
  const { out } = await launch(['start', '--slot', slot, '--hours', String(hours), '--helpers', String(helpers)]);
  fs.appendFileSync(path.join(slotDir(slot), 'cloud.log'), `${new Date().toISOString()} start: ${out}\n`);
  const callId = out.match(/fc-[A-Za-z0-9]+/)?.[0];
  if (!callId) {
    console.log(`Modal launch failed: ${out.slice(-300)}`);
    cloud = null;
    fs.rmSync(CLOUD, { force: true });
    return;
  }
  writeJson(CLOUD, cloud = { ...cloud, callId });
  console.log(`training ${slot} on Modal for ${hours} h with ${1 + helpers} machines (${callId})`);
}

// ask Modal whether the run is still going; "unknown" (e.g. no network) never ends it, and neither does
// "done"/"failed" while any of the run's machines is still up (a glitchy status check once ended a sync early)
const runningMachines = async () => {
  const { out } = await modal(['container', 'list', '--json']);
  try {
    return JSON.parse(out.slice(out.indexOf('['))).filter(c => c.app_name === 'neural-racers').length;
  } catch {
    return 1;
  }
};
async function checkCloud() {
  if (!cloud?.callId || cloud.ending) return;
  const overdue = Date.now() > cloud.until + 15 * 60e3, { out } = await launch(['status', '--call', cloud.callId]);
  if (!overdue && !out.startsWith('done') && !out.startsWith('failed')) return;
  if (!overdue && await runningMachines() > 0) return;
  endCloud();
}

async function endCloud() {
  if (!cloud || cloud.ending) return;
  cloud.ending = true;
  syncing = false;
  await syncCloud();
  // bring home every generation the run saved, including any that finished between syncs
  await modal(['volume', 'get', '--force', VOLUME, `/slots/${cloud.slot}/generations`, slotDir(cloud.slot)]);
  console.log(`Modal training of ${cloud.slot} finished`);
  cloud = null;
  fs.rmSync(CLOUD, { force: true });
}

// stopping the deployed app terminates every machine at once
async function stopCloud() {
  if (!cloud) return;
  await launch(['stop']);
  await endCloud();
}

setInterval(syncCloud, 20e3);
setInterval(checkCloud, 60e3);

function slotList() {
  return slotIds().map(id => ({ id, ...readJson(path.join(slotDir(id), 'meta.json')), ...readJson(path.join(slotDir(id), 'summary.json')) }));
}

// ---- slot actions ----
async function slotAction(params) {
  const action = params.get('action'), id = params.get('id');
  if (action === 'new') {
    const ids = slotIds();
    if (ids.length >= MAX_SLOTS) return [409, { error: `All ${MAX_SLOTS} slots are full. Delete one first.` }];
    const from = params.get('from'), source = from === 'originals' || from === 'scratch' ? null : from;
    if (source && !validId(source)) return [400, { error: 'No such slot to copy.' }];
    const free = Array.from({ length: MAX_SLOTS }, (_, i) => `slot-${i + 1}`).find(s => !ids.includes(s)), dir = slotDir(free);
    fs.mkdirSync(path.join(dir, 'generations'), { recursive: true });
    // a save races normal tracks or the NASCAR ovals, in normal cars or stock cars; a copy keeps its source's
    // unless told otherwise
    const sourceMeta = source ? readJson(path.join(slotDir(source), 'meta.json')) || {} : {};
    const tracks = params.get('tracks') ?? sourceMeta.tracks ?? 'normal';
    if (!['normal', 'nascar'].includes(tracks)) return [400, { error: 'Unknown track set.' }];
    const cars = params.get('cars') ?? (tracks === (sourceMeta.tracks ?? 'normal') ? sourceMeta.cars : null) ?? (tracks === 'nascar' ? 'stock' : 'normal');
    if (!['normal', 'stock'].includes(cars)) return [400, { error: 'Unknown cars.' }];
    const label = tracks === 'nascar' ? 'NASCAR' : 'Save';
    const meta = { name: cleanName(params.get('name')) || `${label} ${free.slice(5)}`, created: new Date().toISOString(), from: 'the originals', tracks, cars };
    // random brains: the engine founds the slot from this seed instead of the originals
    if (from === 'scratch') Object.assign(meta, { from: 'scratch (random brains)', start: 'scratch', seed: 1 + Math.floor(Math.random() * 1e6) });
    if (source) {
      // a copy starts from the source's latest generation and carries its history forward
      const state = readJson(path.join(slotDir(source), 'state.json'));
      if (!state) return [409, { error: 'That slot has no finished generation to copy yet.' }];
      meta.from = `${sourceMeta.name || source}, generation ${state.generation}`;
      if (tracks !== (sourceMeta.tracks ?? 'normal') || cars !== (sourceMeta.cars ?? (tracks === 'nascar' ? 'stock' : 'normal'))) {
        // into another mode: the same brains, but their old results mean nothing here, so it starts again at
        // generation 0 with a fresh tournament in the new mode (and nobody counts as newly born)
        state.population.forEach(a => Object.assign(a, { born: 0, recentPoints: [], last: undefined }));
        writeJson(path.join(dir, 'state.json'), { ...state, generation: 0, history: [], practiceHistory: [], practiceRuns: 0, convertedFrom: meta.from });
        meta.from += ` (was ${sourceMeta.tracks === 'nascar' ? 'NASCAR' : 'normal'} tracks, ${sourceMeta.cars ?? 'normal'} cars)`;
      } else {
        fs.copyFileSync(path.join(slotDir(source), 'state.json'), path.join(dir, 'state.json'));
        fs.copyFileSync(path.join(slotDir(source), 'summary.json'), path.join(dir, 'summary.json'));
        const snapshot = `gen-${String(state.generation).padStart(4, '0')}.json`;
        if (fs.existsSync(path.join(slotDir(source), 'generations', snapshot))) fs.copyFileSync(path.join(slotDir(source), 'generations', snapshot), path.join(dir, 'generations', snapshot));
      }
    }
    writeJson(path.join(dir, 'meta.json'), meta);
    return [200, { id: free }];
  }
  if (!validId(id)) return [400, { error: 'No such slot.' }];
  if (cloud && (action === 'train' || action === 'cloud' || (action === 'delete' && id === cloud.slot)))
    return [409, { error: `"${readJson(path.join(slotDir(cloud.slot), 'meta.json'))?.name}" is training on Modal. Stop it first.` }];
  if (action === 'cloud') {
    const hours = Math.min(4.5, Math.max(0.1, +params.get('hours') || 1)), helpers = Math.min(3, Math.max(0, Math.round(+params.get('helpers') || 0)));
    await stopEngine();
    active = id;
    writeJson(ACTIVE, { id });
    startCloud(id, hours, helpers);
  } else if (action === 'train') {
    if (id !== active) {
      await stopEngine();
      active = id;
      writeJson(ACTIVE, { id });
    }
    startEngine();
  } else if (action === 'rename') {
    const file = path.join(slotDir(id), 'meta.json'), meta = readJson(file) || {};
    writeJson(file, { ...meta, name: cleanName(params.get('name')) || meta.name });
  } else if (action === 'delete') {
    if (id === active) return [409, { error: 'That slot is the one training. Train another slot first.' }];
    fs.rmSync(slotDir(id), { recursive: true, force: true });
  } else return [400, { error: 'Unknown action.' }];
  return [200, { ok: true }];
}

const send = (res, status, body, type = 'application/json') => {
  res.writeHead(status, { 'Content-Type': type, 'Cache-Control': 'no-store' });
  res.end(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body));
};
const status = () => ({
  active, running: !!engine || !!cloud, cores: os.cpus().length, workers: os.cpus().length - 1, maxSlots: MAX_SLOTS, slots: slotList(),
  cloud: cloud && { slot: cloud.slot, until: cloud.until, hours: cloud.hours, helpers: cloud.helpers, perHour: cloud.perHour },
});

http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/api/engine') {
    if (req.method === 'POST') {
      const action = url.searchParams.get('action');
      if (action === 'start' && !cloud) startEngine();
      if (action === 'stop') await (cloud ? stopCloud() : stopEngine());
    }
    return send(res, 200, status());
  }
  if (url.pathname === '/api/slots') {
    if (req.method !== 'POST') return send(res, 200, status());
    const [code, body] = await slotAction(url.searchParams);
    return send(res, code, { ...body, ...status() });
  }
  const file = path.join(ROOT, path.normalize(decodeURIComponent(url.pathname)));
  if (!file.startsWith(ROOT)) return send(res, 403, 'forbidden', 'text/plain');
  const target = url.pathname === '/' ? path.join(ROOT, 'index.html') : file;
  fs.readFile(target, (err, body) => err ? send(res, 404, 'not found', 'text/plain') : send(res, 200, body, TYPES[path.extname(target)] || 'application/octet-stream'));
}).listen(PORT, '127.0.0.1', () => {
  console.log(`Neural Racers on http://127.0.0.1:${PORT}`);
  if (!process.argv.includes('--no-engine') && !cloud) startEngine();
});

// a Modal run keeps going if this server stops (it was started detached); only the local engine stops
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, async () => {
  await stopEngine();
  process.exit(0);
});
