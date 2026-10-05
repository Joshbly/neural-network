#!/usr/bin/env node
// Checks the race workers for gradient learning (train/ppo/episode.js, train/ppo/rollouts.js).
//   node train/test-rollouts.js
// 1. without exploration, an episode's score is exactly evolution's time-trial score (train/lib.js ttScore)
// 2. an episode's rewards add up to its score: exactly in float64, closely from the stored float32 rows
// 3. the rows are what the car saw (checked against an independent run), the critic's extras are in range, and
//    there's one done per episode, on its last row
// 4. the same episode gives the same bytes, and a 1-thread pool writes exactly what a 4-thread pool does
// 5. the protocol: ready, init, collect, evaluate, quit, with the files matching the manifest; the pool (which
//    keeps all 32 ovals built) writes the bytes this process (8 kept, the rest rebuilt) does
// 6. throughput
const fs = require('fs'), os = require('os'), path = require('path'), { spawn } = require('child_process');
const { E, run, initParams } = require('./lib');
const { DESIGN, ROW, EXTRAS } = require('./ppo/spec');
const { episodeOf, runEpisode } = require('./ppo/episode');

let failures = 0;
const check = (ok, text) => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${text}`);
  failures += !ok;
};
const champ = require(path.join(__dirname, '..', 'models', 'slots', 'slot-5', 'generations', 'gen-0040.json')).population.find(b => b.name === 'F5');
const trained = Float32Array.from(champ.genes), random = E.quantizeGenes(initParams(DESIGN.layers, 3, DESIGN)), layers = DESIGN.layers, sigma = [0.3, 0.3];

(async () => {
  console.log('1. the score without exploration');
  const ovals = ['bristol', 'daytona', 'charlotte', 'martinsville', 'phoenix', 'pocono'];
  check(ovals.every(trackId => {
    const laps = E.lapsFor(trackId, 6000), mine = runEpisode({ layers, genes: trained, sigma: null, trackId, laps, seed: 0 });
    return Object.is(mine.summary.score, run({ kind: 'tt', trackId, laps, cars: 'stock', layers, genes: trained }).score);
  }), `an episode without exploration scores exactly what evolution's time trial does (${ovals.length} ovals)`);

  console.log('\n2. the rewards add up');
  const runs = [];
  for (const [genes, label] of [[trained, 'trained'], [random, 'random']])
    for (let e = 0; e < 12; e++) {
      const r = runEpisode({ layers, genes, sigma, ...episodeOf(5, e) });
      runs.push({ label, e, rows: r.rows.slice(), count: r.count, summary: r.summary });
    }
  const exact = Math.max(...runs.map(r => Math.abs(r.summary.return - r.summary.score)));
  check(runs.every(r => r.summary.start === 0) && exact < 1e-12, `in float64 the rewards sum to the score (24 episodes, largest difference ${exact.toExponential(1)}), from 0 at the first decision`);
  const stored = Math.max(...runs.map(r => { let s = 0; for (let k = 0; k < r.count; k++) s += r.rows[k * ROW.width + ROW.reward]; return Math.abs(s - r.summary.score); }));
  check(stored < 1e-4, `summed from the stored float32 rows they're within ${stored.toExponential(1)} of it`);
  console.log(`    (trained: ${runs.filter(r => r.label === 'trained' && r.summary.finished).length} of 12 finished · random: ${runs.filter(r => r.label === 'random' && r.summary.finished).length} of 12)`);

  console.log('\n3. the rows');
  const r0 = runs[0], seen = [], ep = episodeOf(5, 0);
  {
    const heat = new E.Heat(new E.OvalTrack(E.NASCAR_TRACKS.find(t => t.id === ep.trackId)), [new E.Brain(layers, trained)], ep.laps, E.scenarioOptions({ kind: 'tt', trackId: ep.trackId, laps: ep.laps, cars: 'stock' }));
    const car = heat.cars[0];
    car.explore = { sigma, seed: ep.seed };
    car.onDecide = c => seen.push(Float32Array.from(c.inputs));
    while (!heat.over) heat.tick();
  }
  check(seen.length === r0.count && seen.every((x, k) => x.every((v, i) => Object.is(v, r0.rows[k * ROW.width + ROW.obs[0] + i]))), `every row's senses are exactly what the car saw (${seen.length} decisions, checked against a separate run)`);
  const extras = runs.flatMap(r => Array.from({ length: r.count }, (_, k) => Array.from(r.rows.subarray(k * ROW.width + ROW.extras[0], k * ROW.width + ROW.extras[1]))));
  const ranges = EXTRAS.map((_, j) => extras.reduce(([lo, hi], x) => [Math.min(lo, x[j]), Math.max(hi, x[j])], [Infinity, -Infinity]));
  // shares stay within 0..1; laps / 8 goes past 1 at race distance (60 laps of Bowman Gray is 7.5), and the critic
  // normalises its inputs anyway
  check(ranges.every(([lo, hi], j) => lo >= 0 && hi <= (EXTRAS[j].name === 'laps' ? 8 : 1.01)), `the critic's extras stay in range: ${EXTRAS.map((x, j) => `${x.name} ${ranges[j][0].toFixed(2)}–${ranges[j][1].toFixed(2)}`).join(', ')}`);
  check(runs.every(r => Array.from({ length: r.count }, (_, k) => r.rows[k * ROW.width + ROW.done]).every((d, k) => d === (k === r.count - 1 ? 1 : 0))), 'one done per episode, on its last row');

  console.log('\n4. determinism');
  const again = runEpisode({ layers, genes: trained, sigma, ...episodeOf(5, 0) });
  check(Buffer.compare(Buffer.from(again.rows.buffer, again.rows.byteOffset, again.rows.byteLength), Buffer.from(r0.rows.buffer)) === 0, 'the same episode gives the same bytes');

  console.log('\n5. the protocol (and 1 thread against 4)');
  const pool = async threads => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rollouts-test-')), child = spawn('node', [path.join(__dirname, 'ppo', 'rollouts.js'), '--threads', String(threads), '--dir', dir], { stdio: ['pipe', 'pipe', 'inherit'] });
    const replies = [];
    let buffered = '', waiting = null;
    child.stdout.on('data', d => {
      buffered += d;
      let nl;
      while ((nl = buffered.indexOf('\n')) >= 0) { replies.push(JSON.parse(buffered.slice(0, nl))); buffered = buffered.slice(nl + 1); waiting?.(); }
    });
    const next = () => new Promise(resolve => { if (replies.length) return resolve(replies.shift()); waiting = () => { waiting = null; resolve(replies.shift()); }; });
    const send = msg => child.stdin.write(`${JSON.stringify(msg)}\n`);
    const ready = await next();
    send({ type: 'init', seed: 3, out: path.join(dir, 'genes-0.bin') });
    const inited = await next();
    const t0 = Date.now();
    send({ type: 'collect', iteration: 5, genes: path.join(dir, 'genes-0.bin'), sigma, decisions: 50000 });
    const collected = await next(), wall = (Date.now() - t0) / 1000;
    const rowsOf = e => { const bytes = fs.readFileSync(e.file); return bytes.subarray(e.row * ROW.width * 4, (e.row + e.rows) * ROW.width * 4); };
    const byIndex = new Map(collected.episodes.map(e => [e.index, Buffer.from(rowsOf(e))]));
    const sizes = [...new Set(collected.episodes.map(e => e.file))].every(f => fs.statSync(f).size === collected.episodes.filter(e => e.file === f).reduce((s, e) => s + e.rows, 0) * ROW.width * 4);
    fs.writeFileSync(path.join(dir, 'trained.bin'), Buffer.from(trained.buffer));
    send({ type: 'evaluate', genes: path.join(dir, 'trained.bin'), episodes: ovals.map(trackId => ({ trackId, laps: E.lapsFor(trackId, 6000) })) });
    const evaluated = await next();
    // the same file rewritten with other weights must be raced with the new ones
    fs.writeFileSync(path.join(dir, 'trained.bin'), Buffer.from(random.buffer));
    send({ type: 'evaluate', genes: path.join(dir, 'trained.bin'), episodes: ovals.map(trackId => ({ trackId, laps: E.lapsFor(trackId, 6000) })) });
    const rewritten = await next();
    send({ type: 'quit' });
    const code = await new Promise(resolve => child.on('exit', resolve));
    fs.rmSync(dir, { recursive: true, force: true });
    return { ready, inited, collected, evaluated, rewritten, byIndex, sizes, code, wall };
  };
  const one = await pool(1), four = await pool(4);
  check(one.ready.type === 'ready' && one.ready.genes === E.geneCount(layers, true) && one.inited.genes === one.ready.genes, `ready and init: design F's ${one.ready.genes} starting weights`);
  check(four.collected.decisions >= 50000 && four.collected.decisions === four.collected.episodes.reduce((s, e) => s + e.rows, 0) && four.sizes, `collect: ${four.collected.decisions} decisions in ${four.collected.episodes.length} episodes, the files exactly the manifest's rows`);
  check(four.collected.episodes.every(e => four.byIndex.get(e.index).readFloatLE((e.rows - 1) * ROW.width * 4 + ROW.done * 4) === 1), 'each episode in the files ends on its done row');
  const shared = [...one.byIndex.keys()].filter(k => four.byIndex.has(k));
  check(shared.length > 5 && shared.every(k => Buffer.compare(one.byIndex.get(k), four.byIndex.get(k)) === 0), `1 thread and 4 threads write byte-identical rows for the ${shared.length} episodes both ran`);
  const here = runs.filter(r => r.label === 'random' && four.byIndex.has(r.e));
  check(here.length === 12 && here.every(r => Buffer.compare(Buffer.from(r.rows.buffer), four.byIndex.get(r.e)) === 0), `the pool's rows (every oval kept built) equal this process's (8 kept) for the ${here.length} episodes both raced`);
  const judged = four.evaluated.episodes, official = ovals.map(trackId => run({ kind: 'tt', trackId, laps: E.lapsFor(trackId, 6000), cars: 'stock', layers, genes: trained }));
  check(four.evaluated.type === 'evaluated' && judged.length === ovals.length && judged.every((s, k) => Object.is(s.score, official[k].score) && s.finished === official[k].finished && s.bestLap === official[k].lap && s.worn === official[k].aero),
    `evaluate: ${ovals.length} time trials with no exploration, in order, exactly evolution's scores, finishes, best laps and damage (${four.evaluated.seconds.toFixed(2)} s)`);
  const other = ovals.map(trackId => run({ kind: 'tt', trackId, laps: E.lapsFor(trackId, 6000), cars: 'stock', layers, genes: random }).score);
  check(four.rewritten.episodes.every((s, k) => Object.is(s.score, other[k])) && four.rewritten.episodes.some((s, k) => s.score !== judged[k].score),
    'a weights file rewritten under the same name is raced with its new weights');
  check(one.code === 0 && four.code === 0, 'quit exits cleanly');

  console.log('\n6. throughput (this machine)');
  let n = 0;
  const t0 = process.hrtime.bigint();
  for (let e = 0; e < 16; e++) n += runEpisode({ layers, genes: trained, sigma, ...episodeOf(9, e) }).count;
  const single = n / (Number(process.hrtime.bigint() - t0) / 1e9);
  console.log(`    one thread: ${Math.round(single / 1000)}k decisions a second · the 4-thread pool: ${Math.round(four.collected.perSecond / 1000)}k a second · a 600,000-decision batch on ${os.cpus().length - 1} threads: about ${(600000 / (single * (os.cpus().length - 1))).toFixed(1)} s`);

  console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})();
