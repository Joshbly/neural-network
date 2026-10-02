// Two brain architectures race head-to-head in every heat and evolve separately: an A/B test
// under identical tracks, traffic and physics.
const SPECIES = [
  { name: 'A', layers: [INPUT_COUNT, 16, 10, 2], hue: 200, color: '#4fc3ff' },
  { name: 'B', layers: [INPUT_COUNT, 32, 24, 16, 2], hue: 285, color: '#c08cff' },
];
const HEATS = 6, HEAT_SIZE = GRID_SLOTS, PER_HEAT = HEAT_SIZE / SPECIES.length;
const RACE_LAPS = 10, LINE_LAPS = 2;
const SQUAD = HEATS * PER_HEAT, POPULATION = SQUAD * SPECIES.length;
// same per-weight rate for both designs; in A/B runs this was B's best setting (a size-scaled rate starved it)
const MUTATION_RATE = 0.06;
// fitness weights in track units (a second of race time is worth roughly 130):
// wall impulse, each separate wall hit, impulse delivered with your nose, door-to-door impulse, and
// each net position gained by passing. A wall brush must cost more than the time it buys, or
// evolution learns to ride the barriers.
const FITNESS = { wall: 15, hit: 150, ramming: 8, rubbing: 1, pass: 120 };

function shuffle(list) {
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.random() * (i + 1) | 0;
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}

const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;

class Sim {
  constructor(track) {
    this.species = SPECIES.map(spec => {
      const genes = geneCount(spec.layers);
      return { ...spec, genes, evo: new Evolution(SQUAD, spec.layers, MUTATION_RATE) };
    });
    this.history = [];
    this.champion = null;
    this.ranked = null;
    this.featured = 0;
    this.laps = RACE_LAPS;
    this.onGeneration = null;
    this.setTrack(track);
  }

  get generation() {
    return this.species[0].evo.generation;
  }

  setTrack(track) {
    this.track = track;
    // the racing line only needs the opening laps, however long the race
    const lineSteps = Math.ceil(LINE_LAPS * track.length / 2.2) + 300;
    this.paths = this.species.map(() => Array.from({ length: SQUAD }, () => new Float32Array((Math.ceil(lineSteps / 2) + 1) * 3)));
    if (this.champion) this.champion.path = null;
    this.spawn();
  }

  restart() {
    for (const sp of this.species) sp.evo.reset();
    this.history = [];
    this.champion = this.ranked = null;
    this.spawn();
  }

  // genome h of each species (a returning elite after gen 1) seeds heat h, so every race has a
  // proven driver from both sides
  spawn() {
    const rest = this.species.map(() => shuffle(Array.from({ length: SQUAD - HEATS }, (_, i) => i + HEATS)));
    this.heats = Array.from({ length: HEATS }, (_, h) => {
      const entrants = shuffle(this.species.flatMap((sp, s) =>
        [h, ...rest[s].slice(h * (PER_HEAT - 1), (h + 1) * (PER_HEAT - 1))].map(id => ({ sp, s, id }))));
      const heat = new Heat(this.track, entrants.map(({ sp, id }) => new Brain(sp.layers, sp.evo.genomes[id])), this.laps);
      heat.cars.forEach((car, k) => {
        const { sp, s, id } = entrants[k];
        Object.assign(car, { species: sp, genome: id, elite: id < sp.evo.eliteCount, path: this.paths[s][id] });
      });
      return heat;
    });
    this.heats[this.featured].events = [];
  }

  feature(h) {
    this.heats[this.featured].events = null;
    this.featured = h;
    this.heats[h].events = [];
  }

  tick() {
    let running = false;
    for (const heat of this.heats) {
      if (heat.over) continue;
      heat.tick();
      running = true;
    }
    if (!running) this.evolve();
  }

  evolve() {
    const byScore = (a, b) => b.score - a.score, raceLength = car => car.raceLaps * this.track.length;
    for (const heat of this.heats)
      for (const car of heat.cars) car.score = car.fitness(raceLength(car), heat.maxSteps, FITNESS);
    const cars = this.heats.flatMap(heat => heat.cars), winners = this.heats.map(heat => heat.standings()[0]);
    const share = car => clamp((car.progress + car.gridOffset) / raceLength(car), 0, 1);
    const fastest = group => Math.min(...group.flatMap(car => car.laps)) / 60;
    const summary = group => {
      const finishers = group.filter(car => car.finished), lap = fastest(group);
      return {
        lap: lap < Infinity ? lap : null,
        avg: mean(group.map(share)),
        finishers: finishers.length,
        wallHits: mean(finishers.map(car => car.wallHits)),
        overtakes: mean(group.map(car => car.overtakes)),
      };
    };

    const ranked = cars.slice().sort(byScore), best = ranked[0];
    const squads = this.species.map(sp => cars.filter(car => car.species === sp).sort(byScore));
    const entry = {
      gen: this.generation,
      trackId: this.track.id,
      laps: this.heats[0].laps,
      ...summary(cars),
      best: share(best),
      championHits: best.wallHits,
      species: Object.fromEntries(this.species.map((sp, s) => [sp.name, {
        ...summary(squads[s]),
        wins: winners.filter(car => car.species === sp).length,
      }])),
    };
    this.history.push(entry);
    this.champion = { genes: best.brain.genes, species: best.species, lap: entry.lap, wallHits: best.wallHits, path: best.path.slice(0, best.pathLen * 3) };
    this.ranked = ranked.map(car => ({ species: car.species, genes: car.brain.genes }));
    this.species.forEach((sp, s) => sp.evo.breed(squads[s].map(car => car.brain.genes), HEATS));
    this.spawn();
    this.onGeneration?.(entry, best);
  }
}
