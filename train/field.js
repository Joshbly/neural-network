#!/usr/bin/env node
// Picks the 20 best drivers for the showcase: qualify on held-out tracks, then race the survivors
// against each other in full 20-car fields and rank by average finishing position.
//   node train/field.js models/pro-c-s1.json models/pro-c-s2.json ... --out models/field.json
const fs = require('fs');
const path = require('path');
const { createPool, loadDrivers, uniqueDrivers } = require('./pool');
const { E } = require('./lib');

const argv = process.argv.slice(2), outIdx = argv.indexOf('--out');
const out = outIdx >= 0 ? argv[outIdx + 1] : path.join(__dirname, '..', 'models', 'field.json');
const files = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--out');
const QUALI_TRACKS = [911, 912, 913, 914, 915, 916], RACE_TRACKS = [921, 922, 923, 924, 925, 926, 927, 928];
// selected on the same 10-lap distance the showcase runs, where looking after the car decides races
const RACES = 80, LAPS = 10, SIZE = 20;
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;

(async () => {
  const pool = createPool(), rng = E.mulberry32(2026);
  const drivers = uniqueDrivers(files.flatMap(loadDrivers));
  console.log(`${drivers.length} candidates from ${files.length} models`);

  const quali = await pool.runAll(drivers.flatMap(d => QUALI_TRACKS.map(trackSeed => ({ kind: 'tt', layers: d.layers, genes: Float32Array.from(d.genes), trackSeed, laps: 2 }))));
  drivers.forEach((d, i) => {
    const runs = quali.slice(i * QUALI_TRACKS.length, (i + 1) * QUALI_TRACKS.length);
    d.qualified = runs.filter(r => r.finished).length;
    d.qualiLap = mean(runs.filter(r => r.lap).map(r => r.lap));
  });
  const entrants = drivers.filter(d => d.qualified >= QUALI_TRACKS.length - 1);
  console.log(`${entrants.length} qualified (finished at least ${QUALI_TRACKS.length - 1} of ${QUALI_TRACKS.length} unseen tracks)`);

  // every race: a random 20 of the entrants on a random unseen track, random grid
  const races = Array.from({ length: RACES }, () => {
    const ids = entrants.map((_, i) => i).sort(() => rng() - 0.5).slice(0, SIZE);
    return { ids, job: { kind: 'field', drivers: ids.map(i => entrants[i]), trackSeed: RACE_TRACKS[Math.floor(rng() * RACE_TRACKS.length)], laps: LAPS } };
  });
  const results = await pool.runAll(races.map(r => r.job));
  const tally = entrants.map(() => ({ places: [], wins: 0, walls: [], passes: [], rammed: [], aero: [] }));
  races.forEach((r, k) => results[k].forEach((res, slot) => {
    const t = tally[r.ids[slot]], field = r.ids.length;
    t.places.push(res.place / (field - 1));
    t.wins += res.place === 0;
    t.walls.push(res.walls);
    t.passes.push(res.passes);
    t.rammed.push(res.rammed);
    t.aero.push(res.aero);
  }));
  const ranked = entrants.map((d, i) => ({ ...d, avgPlace: mean(tally[i].places), wins: tally[i].wins, races: tally[i].places.length,
    walls: mean(tally[i].walls), passes: mean(tally[i].passes), rammed: mean(tally[i].rammed), aero: mean(tally[i].aero) }))
    .filter(d => d.races)
    .sort((a, b) => a.avgPlace - b.avgPlace)
    .slice(0, SIZE);
  ranked.forEach((d, i) => console.log(`P${String(i + 1).padStart(2)} ${d.label.padEnd(22)} avg finish ${(d.avgPlace * 100).toFixed(0).padStart(3)}% of field | wins ${d.wins}/${d.races} | quali lap ${d.qualiLap?.toFixed(2)} | walls ${d.walls.toFixed(1)} passes ${d.passes.toFixed(1)} rammed ${d.rammed.toFixed(2)} aero lost ${(100 * d.aero).toFixed(0)}%`));
  fs.writeFileSync(out, JSON.stringify({ built: new Date().toISOString(), drivers: ranked.map((d, i) => ({
    rank: i + 1, label: d.label, layers: d.layers, genes: d.genes,
    stats: { avgPlace: +d.avgPlace.toFixed(3), wins: d.wins, races: d.races, qualiLap: d.qualiLap && +d.qualiLap.toFixed(2), walls: +d.walls.toFixed(2), passes: +d.passes.toFixed(2), aero: +d.aero.toFixed(3) },
  })) }));
  console.log(`wrote ${ranked.length} drivers to ${out}`);
  await pool.close();
})();
