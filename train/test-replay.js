#!/usr/bin/env node
// Replays a save's recorded tournament and rating races the way the app does (js/replay.js: the brains as
// published, the grid in starting order, the scenario's own track and options) and checks every finishing
// order against the official one the engine wrote.
//   node train/test-replay.js models/slots/slot-6
const fs = require('fs');
const path = require('path');
const { E } = require('./lib');

const dir = path.resolve(process.argv[2] || 'models/slots/slot-1');
const read = file => fs.existsSync(path.join(dir, file)) ? JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8')) : null;
const trackOf = sc => sc.trackId ? E.OvalTrack.get(sc.trackId) : E.Track.random(E.mulberry32(sc.trackSeed));
function replay(scenario, drivers) {
  const heat = new E.Heat(trackOf(scenario), drivers.map(d => new E.Brain(d.layers, Float32Array.from(d.genes))), scenario.laps, E.scenarioOptions(scenario));
  while (!heat.over) heat.tick();
  const standings = heat.standings();
  return heat.cars.map((car, slot) => [standings.indexOf(car), slot]).sort((x, y) => x[0] - y[0]).map(x => x[1]);
}

let checked = 0, matched = 0;
const tourney = read('tournament.json');
if (tourney) {
  // an experiment save's tournaments seat its frozen sparring partners too (train/evolve.js tournamentSparring)
  const byName = new Map([...tourney.population, ...tourney.sparring ?? []].map(a => [a.name, a]));
  for (const [k, race] of tourney.races.entries()) {
    const order = replay(race.scenario, race.grid.map(name => byName.get(name))).map(slot => race.grid[slot]);
    const same = order.join() === race.order.join();
    checked++;
    matched += same;
    if (!same) console.log(`  tournament race ${k + 1}: replay ${order.slice(0, 5).join(' ')} … official ${race.order.slice(0, 5).join(' ')} …`);
  }
  console.log(`generation ${tourney.gen} tournament: ${matched}/${checked} races replay exactly`);
}
const rating = read('rating-races.json');
if (rating) {
  const byId = new Map(rating.players.map(p => [p.id, p]));
  let ok = 0;
  for (const [k, race] of rating.races.entries()) {
    const order = replay(race.scenario, race.grid.map(id => byId.get(id))).map(slot => race.grid[slot]);
    const same = order.join() === race.order.join();
    ok += same;
    if (!same) console.log(`  rating race ${k + 1} differs`);
  }
  checked += rating.races.length;
  matched += ok;
  console.log(`generation ${rating.gen} rating races: ${ok}/${rating.races.length} replay exactly`);
}
process.exit(checked && matched === checked ? 0 : 1);
