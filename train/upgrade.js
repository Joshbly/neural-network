#!/usr/bin/env node
// Continues a save with the upgraded recipe, as a new save (the source is only read):
//   - more seats, weighted toward the strongest designs, filled for diversity: the source's current brains,
//     then each design's best from earlier eras of the run, then lines that died out, then mutated copies
//   - a hall of fame of past champions who join practice races
//   - a frozen panel of past champions every generation is measured against (disjoint from the hall)
//   - fine-tuning: twice the copies per training step, smaller tweaks, gentler selection
//   node train/upgrade.js --from slot-2 --name "Big run · upgraded"
const fs = require('fs');
const path = require('path');
const { noise } = require('./lib');

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc, []));
const SLOTS = path.join(__dirname, '..', 'models', 'slots');
const SEATS = { A: 10, B: 10, C: 6, D: 10, E: 6 };
const RECIPE = { seats: SEATS, pairs: 96, margin: 0.15, settle: 5, window: 3, pickFromTop: true, hallEvery: 10, hallSize: 16, hallRivals: 3 };
const ERAS = [100, 75, 50, 25], YARDSTICK_GENS = [25, 50, 100], HALL_GENS = [35, 60, 85, 110, 125];

const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const src = path.join(SLOTS, args.from), st = read(path.join(src, 'state.json')), last = st.generation;
const snapshot = g => fs.existsSync(path.join(src, 'generations', `gen-${String(g).padStart(4, '0')}.json`)) ? read(path.join(src, 'generations', `gen-${String(g).padStart(4, '0')}.json`)).population : [];
const placeAt = (g, name) => st.history.find(e => e.gen === g)?.agents.find(a => a.name === name)?.avgPlace ?? Infinity;
const ranked = (g, design) => snapshot(g).filter(a => !design || a.species === design).sort((x, y) => placeAt(g, x.name) - placeAt(g, y.name));
const frozen = (a, g) => ({ name: `${a.name}@${g}`, species: a.species, layers: a.layers, genes: a.genes, gen: g });
// fine-tuning: smaller tweaks, and a learning rate no faster than the careful end of what the run used
const tune = a => ({ ...a, sigma: Math.max(0.01, Math.min(a.sigma, 0.02)), lr: Math.max(0.001, Math.min(a.lr, 0.005)) });

const population = [];
for (const [design, seats] of Object.entries(SEATS)) {
  const mine = [], have = name => mine.some(a => a.name === name), add = (a, label) => mine.length < seats && !have(a.name) && mine.push({ ...a, label });
  const current = st.population.filter(a => a.species === design);
  current.forEach(a => add(a, a.label));
  // the same lineage from earlier eras drives differently: real variety that can still compete
  for (const g of ERAS) {
    const best = ranked(g, design)[0];
    if (best) add({ ...best, born: last, parent: `${best.name} (generation ${g})` }, `${design} from generation ${g}`);
  }
  // lines that died out, from the last generation they appeared in
  const living = new Set(current.map(a => a.founder));
  for (const founder of [1, 2, 3, 4].map(k => `${design}${k}`).filter(f => !living.has(f))) {
    for (let g = last; g >= 0; g--) {
      const line = ranked(g, design).filter(a => a.founder === founder);
      if (!line.length) continue;
      add({ ...line[0], born: last, parent: `${line[0].name} (generation ${g})` }, `revived line ${founder} from generation ${g}`);
      break;
    }
  }
  // any seats left: slightly mutated copies of the current brains
  for (let k = 0; mine.length < seats; k++) {
    const parent = current[k % current.length], n = st.clones[parent.founder] = (st.clones[parent.founder] || 1) + 1, eps = noise(9000 + population.length + mine.length, parent.genes.length);
    mine.push({ ...parent, name: `${parent.founder}·${n}`, parent: parent.name, born: last, label: `copy of ${parent.name}`, genes: parent.genes.map((g, j) => +(g + 0.01 * eps[j]).toFixed(5)) });
  }
  population.push(...mine.map(a => tune({ ...a, recent: [], wins: a.wins ?? 0, races: a.races ?? 0 })));
}

const champion = g => snapshot(g).find(a => a.name === st.history.find(e => e.gen === g)?.champion);
const hall = HALL_GENS.map(g => champion(g) && frozen(champion(g), g)).filter(Boolean);
const yardstick = [...YARDSTICK_GENS.flatMap(g => ranked(g).slice(0, 3).map(a => frozen(a, g))), frozen(champion(last), last)];

const free = Array.from({ length: 10 }, (_, i) => `slot-${i + 1}`).find(s => !fs.existsSync(path.join(SLOTS, s)));
if (!free) throw new Error('all 10 save slots are in use');
const out = path.join(SLOTS, free), meta = read(path.join(src, 'meta.json'));
fs.mkdirSync(path.join(out, 'generations'), { recursive: true });
const next = { ...st, population, hall, yardstick, config: RECIPE, upgradedFrom: { save: meta.name, generation: last } };
fs.writeFileSync(path.join(out, 'state.json'), JSON.stringify(next));
fs.writeFileSync(path.join(out, 'generations', `gen-${String(last).padStart(4, '0')}.json`), JSON.stringify({ generation: last, population }));
fs.writeFileSync(path.join(out, 'summary.json'), fs.readFileSync(path.join(src, 'summary.json')));
// the copy races the same tracks in the same cars (a NASCAR save stays NASCAR) and keeps its rating ladder
fs.writeFileSync(path.join(out, 'meta.json'), JSON.stringify({ name: args.name || `${meta.name} · upgraded`, created: new Date().toISOString(), from: `${meta.name}, generation ${last}, upgraded recipe`,
  ...meta.tracks && { tracks: meta.tracks }, ...meta.cars && { cars: meta.cars } }));
for (const file of ['ladder.json', 'rating.json'])
  if (fs.existsSync(path.join(src, file))) fs.copyFileSync(path.join(src, file), path.join(out, file));

console.log(`new save ${free}: ${population.length} brains`);
for (const d of Object.keys(SEATS)) console.log(`  ${d}: ${population.filter(a => a.species === d).map(a => `${a.name} (${a.label})`).join(', ')}`);
console.log(`hall of fame: ${hall.map(a => a.name).join(', ')}`);
console.log(`yardstick: ${yardstick.map(a => a.name).join(', ')}`);
