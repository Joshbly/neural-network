#!/usr/bin/env node
// Replays a save's recorded tournament and rating races the way the app does (js/replay.js: the brains as
// published, the grid in starting order, the scenario's own track and options) and checks every finishing
// order against the official one the engine wrote. A recorded time trial (gradient learning's evaluation) has
// one car, so its time is checked instead: the steps to the flag, whether it finished, and its score.
//   node train/test-replay.js models/slots/slot-6
const fs = require('fs');
const path = require('path');
const { E, ttScore } = require('./lib');

const dir = path.resolve(process.argv[2] || 'models/slots/slot-1');
const read = file => fs.existsSync(path.join(dir, file)) ? JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8')) : null;
const trackOf = sc => sc.trackId ? E.OvalTrack.get(sc.trackId) : E.Track.random(E.mulberry32(sc.trackSeed));
let timed = null;
function replay(scenario, drivers) {
  const heat = new E.Heat(trackOf(scenario), drivers.map(d => new E.Brain(d.layers, Float32Array.from(d.genes))), scenario.laps, E.scenarioOptions(scenario));
  while (!heat.over) heat.tick();
  const standings = heat.standings(), car = heat.cars[0];
  timed = { steps: car.steps, finished: car.finished, score: ttScore(car, heat) };
  return heat.cars.map((car, slot) => [standings.indexOf(car), slot]).sort((x, y) => x[0] - y[0]).map(x => x[1]);
}

let checked = 0, matched = 0;
const tourney = read('tournament.json');
if (tourney) {
  // an experiment save's tournaments seat its frozen sparring partners too (train/evolve.js tournamentSparring)
  const byName = new Map([...tourney.population, ...tourney.sparring ?? []].map(a => [a.name, a]));
  for (const [k, race] of tourney.races.entries()) {
    const order = replay(race.scenario, race.grid.map(name => byName.get(name))).map(slot => race.grid[slot]);
    const same = order.join() === race.order.join() && (!race.time || (timed.steps === race.time.steps && timed.finished === race.time.finished && Object.is(timed.score, race.time.score)));
    checked++;
    matched += same;
    if (!same && race.time) console.log(`  time trial ${k + 1} (${race.scenario.trackId}): replay ${timed.steps} steps, score ${timed.score} … official ${race.time.steps}, ${race.time.score}`);
    else if (!same) console.log(`  tournament race ${k + 1}: replay ${order.slice(0, 5).join(' ')} … official ${race.order.slice(0, 5).join(' ')} …`);
  }
  console.log(`generation ${tourney.gen} ${tourney.races.every(r => r.time) ? 'evaluation' : 'tournament'}: ${matched}/${checked} ${tourney.races.every(r => r.time) ? 'time trials' : 'races'} replay exactly`);
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
