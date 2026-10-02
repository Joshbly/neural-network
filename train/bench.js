#!/usr/bin/env node
// Final evaluation: the ES-trained field against the in-browser genetic algorithm's best drivers.
//   node train/bench.js
const path = require('path');
const { createPool, loadDrivers } = require('./pool');
const { E } = require('./lib');

const MODELS = path.join(__dirname, '..', 'models');
const UNSEEN = Array.from({ length: 12 }, (_, i) => 931 + i);
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
const f32 = d => ({ ...d, genes: Float32Array.from(d.genes) });

(async () => {
  const pool = createPool(), rng = E.mulberry32(77);
  const pros = loadDrivers(path.join(MODELS, 'field.json')).map(f32);
  const ga = require(path.join(MODELS, 'ga-ref.json'));
  const gaDrivers = ga.drivers.map(f32);

  // 1. time trials: best pro vs best GA brain, on the GA's home track and on tracks neither trained on
  const trial = (d, tracks) => pool.runAll(tracks.map(trackSeed => ({ kind: 'tt', layers: d.layers, genes: d.genes, trackSeed, laps: 2 })));
  const summary = runs => ({ finished: `${runs.filter(r => r.finished).length}/${runs.length}`, lap: +mean(runs.filter(r => r.lap).map(r => r.lap)).toFixed(2), walls: +mean(runs.map(r => r.walls)).toFixed(1) });
  console.log('TIME TRIAL (2 laps, solo)');
  for (const [label, tracks] of [[`GA home track (seed ${ga.trackSeed})`, [ga.trackSeed]], ['12 unseen tracks', UNSEEN]])
    console.log(`  ${label.padEnd(26)} pro P1 ${JSON.stringify(summary(await trial(pros[0], tracks)))}   GA #1 ${JSON.stringify(summary(await trial(gaDrivers[0], tracks)))}`);

  // 2. showdown: top 10 pros vs top 10 GA brains in 20-car races, random grids
  const races = [...UNSEEN.slice(0, 12), ga.trackSeed, ga.trackSeed, ga.trackSeed, ga.trackSeed].map(trackSeed => {
    const entrants = [...pros.slice(0, 10).map(d => ({ ...d, group: 'pro' })), ...gaDrivers.slice(0, 10).map(d => ({ ...d, group: 'ga' }))].sort(() => rng() - 0.5);
    return { entrants, job: { kind: 'field', drivers: entrants, trackSeed, laps: 10 } };
  });
  const results = await pool.runAll(races.map(r => r.job));
  const groups = { pro: { places: [], wins: 0, walls: [], passes: [], rammed: [], podiums: 0 }, ga: { places: [], wins: 0, walls: [], passes: [], rammed: [], podiums: 0 } };
  races.forEach((r, k) => results[k].forEach((res, i) => {
    const g = groups[r.entrants[i].group];
    g.places.push(res.place + 1);
    g.wins += res.place === 0;
    g.podiums += res.place < 3;
    g.walls.push(res.walls);
    g.passes.push(res.passes);
    g.rammed.push(res.rammed);
  }));
  console.log(`\nSHOWDOWN: 10 pros vs 10 GA brains, ${races.length} 10-lap races of 20 cars (12 unseen tracks + 4 on the GA's home track)`);
  for (const [name, g] of Object.entries(groups))
    console.log(`  ${name.padEnd(4)} wins ${g.wins}/${races.length} | podiums ${g.podiums} | avg finish ${mean(g.places).toFixed(1)} | walls ${mean(g.walls).toFixed(1)} | passes ${mean(g.passes).toFixed(1)} | rammed ${mean(g.rammed).toFixed(2)}`);

  // 3. does the best pro actually use its mirrors? race it against fellow pros with and without them
  const rivals = pros.slice(1, 8);
  const mirrorRaces = blind => pool.runAll(UNSEEN.flatMap(trackSeed => [0, 7].map(slot =>
    ({ kind: 'race', layers: pros[0].layers, genes: pros[0].genes, opponents: rivals, trackSeed, slot, laps: 3, blind }))));
  console.log('\nMIRROR TEST: P1 vs P2-P8, from pole and from last, 12 unseen tracks');
  for (const blind of [false, true]) {
    const out = await mirrorRaces(blind);
    const front = out.filter((_, i) => i % 2 === 0), back = out.filter((_, i) => i % 2 === 1);
    console.log(`  mirrors ${blind ? 'BLACKED OUT' : 'on         '} | from pole: wins ${front.filter(o => o.won).length}/12, avg finish ${mean(front.map(o => o.place + 1)).toFixed(2)}, passed ${mean(front.map(o => o.passedBy)).toFixed(1)}x | from last: wins ${back.filter(o => o.won).length}/12, avg finish ${mean(back.map(o => o.place + 1)).toFixed(2)}, passes ${mean(back.map(o => o.passes)).toFixed(1)}`);
  }
  await pool.close();
})();
