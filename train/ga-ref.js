#!/usr/bin/env node
// Trains the in-browser genetic algorithm headless, as a baseline for the ES-trained drivers.
//   node train/ga-ref.js --gens 100 --laps 2
const fs = require('fs');
const path = require('path');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const gens = +(args.gens || 100), laps = +(args.laps || 2), seed = +(args.track || 2);
const files = ['nn-wasm.js', 'nn.js', 'track.js', 'car.js', 'heat.js', 'sim.js'];
const source = files.map(f => fs.readFileSync(path.join(__dirname, '..', 'js', f), 'utf8')).join('\n');
const { Sim, Track, mulberry32 } = new Function(`${source}\nreturn { Sim, Track, mulberry32 };`)();

const sim = new Sim(Track.random(mulberry32(seed)));
sim.laps = laps;
sim.spawn();
const t0 = Date.now();
while (sim.generation <= gens) {
  sim.tick();
  if (sim.history.length && sim.history.length % 20 === 0 && sim.history.at(-1).gen === sim.history.length && !sim.logged?.[sim.history.length]) {
    (sim.logged ??= {})[sim.history.length] = true;
    const h = sim.history.at(-1);
    console.log(`gen ${h.gen} lap ${h.lap?.toFixed(2)} finishers ${h.finishers} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
}
const top = sim.ranked.slice(0, 20).map((d, i) => ({ label: `GA ${d.species.name}${i + 1}`, layers: d.species.layers, genes: Array.from(d.genes, v => +v.toFixed(5)) }));
fs.writeFileSync(path.join(__dirname, '..', 'models', 'ga-ref.json'), JSON.stringify({ trackSeed: seed, gens, laps, drivers: top }));
console.log(`saved top ${top.length} GA drivers after ${gens} generations`);
