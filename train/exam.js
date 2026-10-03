#!/usr/bin/env node
// How good are a NASCAR save's brains, measured against things outside the population?
//   node train/exam.js [slot-N]
// 1. Solo pace: every brain (the population and the hall of fame) laps all 32 ovals alone. Each best lap is set
//    against the perfect lap in this physics (train/nascar/line.js: the minimum-curvature line at the grip limit,
//    the same calculation the car constants were fitted to real poles with), a scripted driver actually driving
//    that line in the sim, and the real Cup pole speed.
// 2. Races: the population races every oval, 24 km, 20 cars, plus 40-car packs at the superspeedways, on the
//    same grids twice: green-only (as training runs until generation 100) and under the full rulebook (cautions,
//    restarts, stages, red and black flags). Counted: who finishes, wrecks, cautions, lead changes at the line.
// Writes models/slots/<slot>/exam.json.
const fs = require('fs'), os = require('os'), path = require('path');
const { Worker, isMainThread, parentPort } = require('worker_threads');
const { E, oval, run } = require('./lib');
const { racingLine, lapTime, scripted } = require('./nascar/line');

const SLOTS = path.join(__dirname, '..', 'models', 'slots');
const SOLO_M = 6000, RACE_M = 24000;
const mphOf = (id, seconds) => E.NASCAR_TRACKS.find(d => d.id === id).miles * 3600 / seconds;

// ---------- worker ----------

if (!isMainThread) {
  let roster = [];
  parentPort.on('message', msg => {
    if (msg.roster) return void (roster = msg.roster);
    const { job } = msg;
    const out = job.kind === 'solo' ? solo(job, roster) : job.kind === 'script' ? script(job) : race(job, roster);
    parentPort.postMessage({ id: msg.id, out });
  });
}

function solo({ brain, trackId }, roster) {
  const b = roster[brain], r = run({ kind: 'tt', layers: b.layers, genes: Float32Array.from(b.genes), trackId, laps: E.lapsFor(trackId, SOLO_M), cars: 'stock' });
  return { finished: r.finished, mph: r.lap && mphOf(trackId, r.lap), walls: r.walls, aero: r.aero };
}

// the scripted driver: the perfect line at a share of the perfect speed, blind to everything else
function script({ trackId, pace }) {
  const t = oval(trackId), heat = new E.Heat(t, [null], E.lapsFor(trackId, SOLO_M), { cars: 'stock', fastCaution: true }), car = heat.cars[0], drive = scripted(t, E.stockSpec(t));
  while (!heat.over) {
    drive(heat, car, { pace });
    heat.tick();
  }
  return { finished: car.finished, mph: car.laps.length ? mphOf(trackId, Math.min(...car.laps) / 60) : null };
}

// cautions: the full rulebook, or green-only racing the way training runs it before generation 100
function race({ trackId, field, cautions }, roster) {
  const laps = E.lapsFor(trackId, RACE_M), t = oval(trackId);
  const heat = new E.Heat(t, field.map(i => new E.Brain(roster[i].layers, Float32Array.from(roster[i].genes))), laps, { cars: 'stock', stages: cautions, cautions, fastCaution: true });
  const rc = heat.control;
  // lead changes counted the way NASCAR does, at the line: who leads each time the lead lap count ticks over
  let lineLeader = null, leadLap = 0, leadChanges = 0, reds = 0, wasRed = false;
  while (!heat.over) {
    heat.tick();
    if (rc.phase === 'red' && !wasRed) reds++;
    wasRed = rc.phase === 'red';
    let lead = null;
    for (const car of heat.cars) if (car.running && (!lead || car.progress + car.bonus > lead.progress + lead.bonus)) lead = car;
    if (!lead || lead.laps.length <= leadLap) continue;
    leadLap = lead.laps.length;
    if (lineLeader && lead !== lineLeader) leadChanges++;
    lineLeader = lead;
  }
  const standings = heat.standings();
  return {
    laps, steps: heat.step, cautions: rc.cautions, reds, leadChanges, checkered: heat.finishers > 0,
    cars: heat.cars.map((car, k) => ({
      brain: field[k], place: standings.indexOf(car), finished: car.finished, wrecked: car.wrecked, parked: !!car.parked,
      out: !car.finished && (car.retired || car.wrecked || !!car.parked), passes: car.overtakes, cautionsCaused: car.cautionsCaused,
      penalties: car.penalties, aero: (car.condition.front + car.condition.rear) / 2, draft: car.draftSteps / Math.max(1, car.steps),
      lap: car.laps.length ? mphOf(trackId, Math.min(...car.laps) / 60) : null,
    })),
  };
}

function pool(roster) {
  const workers = Array.from({ length: Math.max(1, os.cpus().length - 2) }, () => new Worker(__filename));
  for (const w of workers) w.postMessage({ roster });
  const runAll = jobs => new Promise(resolve => {
    const out = new Array(jobs.length);
    let next = 0, done = 0;
    if (!jobs.length) return resolve(out);
    const feed = w => {
      if (next >= jobs.length) return;
      const id = next++;
      w.once('message', msg => {
        out[id] = msg.out;
        if (++done === jobs.length) resolve(out);
        else feed(w);
      });
      w.postMessage({ id, job: jobs[id] });
    };
    workers.forEach(feed);
  });
  return { runAll, close: () => Promise.all(workers.map(w => w.terminate())) };
}

// ---------- main ----------

const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const median = xs => { const s = xs.slice().sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : null; };
const pct = v => v == null ? '—' : `${Math.round(v * 100)}%`;

async function main() {
  const slot = process.argv.slice(2).find(a => /^slot-\d+$/.test(a)) ?? JSON.parse(fs.readFileSync(path.join(SLOTS, 'active.json'))).id;
  const dir = path.join(SLOTS, slot), state = JSON.parse(fs.readFileSync(path.join(dir, 'state.json')));
  const brains = [...state.population, ...(state.hall ?? [])], P = state.population.length;
  const roster = brains.map(b => ({ layers: b.layers, genes: Array.from(b.genes) }));
  const tracks = E.NASCAR_TRACKS.map(d => d.id), started = Date.now();
  const workers = pool(roster);
  console.log(`${slot}, generation ${state.generation}: ${P} brains racing, ${brains.length - P} in the hall of fame, on ${os.cpus().length - 2} threads`);

  // ---- 1. solo pace ----
  const perfect = Object.fromEntries(tracks.map(id => { const t = oval(id); return [id, lapTime(t, racingLine(t), E.stockSpec(t)).mph]; }));
  const pole = Object.fromEntries(E.NASCAR_TRACKS.map(d => [d.id, d.pole.mph]));
  const soloRuns = await workers.runAll(brains.flatMap((_, brain) => tracks.map(trackId => ({ kind: 'solo', brain, trackId }))));
  const scriptRuns = await workers.runAll([0.95, 1].flatMap(pace => tracks.map(trackId => ({ kind: 'script', trackId, pace }))));
  const soloOf = brain => soloRuns.slice(brain * tracks.length, (brain + 1) * tracks.length).map((r, k) => ({ ...r, trackId: tracks[k] }));
  const paceRow = runs => {
    const clean = runs.filter(r => r.finished && r.mph);
    return {
      finished: runs.filter(r => r.finished).length / runs.length,
      ofPerfect: median(clean.map(r => r.mph / perfect[r.trackId])), ofPole: median(clean.map(r => r.mph / pole[r.trackId])),
      walls: mean(runs.map(r => r.walls ?? 0)), aero: mean(runs.map(r => r.aero ?? 0)),
    };
  };
  const scripts = [0.95, 1].map((pace, k) => ({ pace, ...paceRow(scriptRuns.slice(k * tracks.length, (k + 1) * tracks.length).map((r, j) => ({ ...r, trackId: tracks[j] }))) }));
  const soloTable = brains.map((b, i) => ({ name: b.name, design: b.species, hall: i >= P, ...paceRow(soloOf(i)) }));
  // the best any brain managed on each track
  const perTrack = tracks.map((id, k) => {
    const laps = brains.map((b, i) => ({ name: b.name, ...soloRuns[i * tracks.length + k] })).filter(r => r.finished && r.mph).sort((a, b) => b.mph - a.mph);
    return { id, pole: pole[id], perfect: perfect[id], script: scriptRuns[tracks.length + k].mph, best: laps[0] ? { name: laps[0].name, mph: laps[0].mph } : null, finishers: laps.length / brains.length };
  });

  console.log('\n1. SOLO PACE: every oval alone, best lap (median over the tracks it finished)');
  console.log(`   perfect lap in this physics = 100%; real pole is ${pct(median(tracks.map(id => pole[id] / perfect[id])))} of it (median)`);
  for (const s of scripts) console.log(`   scripted driver at ${s.pace * 100}% of perfect speed: finished ${pct(s.finished)} of ovals, ${pct(s.ofPerfect)} of perfect, ${pct(s.ofPole)} of pole`);
  for (const r of soloTable.slice().sort((a, b) => (b.ofPerfect ?? 0) - (a.ofPerfect ?? 0)))
    console.log(`   ${r.hall ? '(hall) ' : ''}${r.name.padEnd(9)} finished ${pct(r.finished).padStart(4)} · ${pct(r.ofPerfect).padStart(4)} of perfect · ${pct(r.ofPole).padStart(4)} of pole · wall hits ${r.walls.toFixed(1)} · downforce lost ${pct(r.aero / 2)}`);
  console.log('   hardest tracks (share of brains that could finish 6 km alone):');
  for (const t of perTrack.slice().sort((a, b) => a.finishers - b.finishers).slice(0, 6))
    console.log(`     ${t.id.padEnd(14)} ${pct(t.finishers)} finish; best ${t.best ? `${t.best.name} ${t.best.mph.toFixed(1)} mph` : '—'} vs pole ${t.pole} (perfect ${t.perfect.toFixed(1)})`);

  // ---- 2. real races ----
  const rng = E.mulberry32(2026), shuffled = list => { const a = list.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const field = Array.from({ length: P }, (_, i) => i);
  // the same grids twice: green-only as in training, then the full rulebook
  const grids = [
    ...tracks.map(trackId => ({ trackId, field: shuffled(field) })),
    ...E.SUPERSPEEDWAYS.flatMap(trackId => [0, 1].map(() => ({ trackId, field: shuffled([...field, ...field]) }))),
  ];
  const plan = [false, true].flatMap(cautions => grids.map(g => ({ kind: 'race', ...g, cautions })));
  const races = await workers.runAll(plan);
  await workers.close();
  const milesOf = id => E.NASCAR_TRACKS.find(d => d.id === id).miles;
  const summary = rs => {
    const cars = rs.flatMap(r => r.cars);
    return {
      races: rs.length, miles: mean(rs.map(r => r.laps * milesOf(r.trackId))), checkered: mean(rs.map(r => +r.checkered)),
      finished: mean(cars.map(c => +c.finished)), out: mean(cars.map(c => +c.out)),
      cautions: mean(rs.map(r => r.cautions)), reds: mean(rs.map(r => r.reds)), leadChanges: mean(rs.map(r => r.leadChanges)),
      blackFlags: mean(rs.map(r => r.cars.reduce((s, c) => s + c.penalties, 0))), draft: mean(cars.map(c => c.draft)), aero: mean(cars.map(c => c.aero)),
    };
  };
  races.forEach((r, k) => r.trackId = plan[k].trackId);
  const N = grids.length, sets = { green: races.slice(0, N), full: races.slice(N) };
  const ss = id => E.SUPERSPEEDWAYS.includes(id), short = id => milesOf(id) < 1;
  const kinds = rules => [
    ['every oval, 20 cars', sets[rules].slice(0, tracks.length)],
    ['superspeedways, 20 cars', sets[rules].slice(0, tracks.length).filter(r => ss(r.trackId))],
    ['superspeedways, 40 cars', sets[rules].slice(tracks.length)],
    ['intermediates (1-2 miles), 20 cars', sets[rules].slice(0, tracks.length).filter(r => !ss(r.trackId) && !short(r.trackId))],
    ['short tracks (under 1 mile), 20 cars', sets[rules].slice(0, tracks.length).filter(r => short(r.trackId))],
  ].map(([label, rs]) => ({ label, ...summary(rs) }));
  const raceTable = { green: kinds('green'), full: kinds('full') };
  for (const [rules, title] of [['green', 'GREEN-ONLY (as in training)'], ['full', 'FULL RULEBOOK (cautions, restarts, stages)']]) {
    console.log(`\n2. RACES, ${title}: 24 km, about 15 miles`);
    for (const r of raceTable[rules])
      console.log(`   ${r.label.padEnd(36)} ${String(r.races).padStart(2)} races · reached the checkered ${pct(r.checkered).padStart(4)} · cars finished ${pct(r.finished).padStart(4)} · wrecked/parked ${pct(r.out).padStart(4)} · cautions ${r.cautions.toFixed(1)} · lead changes at the line ${r.leadChanges.toFixed(1)} · black flags ${r.blackFlags.toFixed(1)} · in the draft ${pct(r.draft)} · downforce lost ${pct(r.aero)}`);
  }
  const perBrain = field.map(i => {
    const mine = sets.green.slice(0, tracks.length).flatMap(r => r.cars.filter(c => c.brain === i));
    return { name: brains[i].name, place: mean(mine.map(c => c.place / (P - 1))), wins: mine.filter(c => c.place === 0).length, finished: mean(mine.map(c => +c.finished)), cautionsCaused: mean(mine.map(c => c.cautionsCaused)), passes: mean(mine.map(c => c.passes)) };
  }).sort((a, b) => a.place - b.place);
  console.log('   by brain over the 32 ovals, green-only:');
  for (const b of perBrain) console.log(`     ${b.name.padEnd(9)} place ${b.place.toFixed(2)} · wins ${b.wins} · finished ${pct(b.finished)} · cautions caused/race ${b.cautionsCaused.toFixed(2)} · passes ${b.passes.toFixed(1)}`);

  const out = { built: new Date().toISOString(), slot, generation: state.generation, solo: { scripts, brains: soloTable, tracks: perTrack }, races: { kinds: raceTable, brains: perBrain } };
  fs.writeFileSync(path.join(dir, 'exam.json'), JSON.stringify(out, (k, v) => typeof v === 'number' ? +v.toFixed(4) : v));
  console.log(`\nwrote ${path.join(dir, 'exam.json')} in ${((Date.now() - started) / 1000).toFixed(0)} s`);
}

if (isMainThread) main();
