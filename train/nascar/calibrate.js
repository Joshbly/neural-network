#!/usr/bin/env node
// Calibrates the stock-car packages against real pole speeds, then has a scripted car drive every oval in the
// actual simulation.
//   node train/nascar/calibrate.js          fit + table + scripted laps
//   node train/nascar/calibrate.js --check  table + scripted laps with the constants in js/car.js
//
// 1. Racing line: a pole lap uses the whole width (wide entry, low apex, wide exit). The line is the
//    minimum-curvature path between the walls, found by relaxing lateral offsets at several scales.
// 2. Lap-time simulation along it, with the sim's own physics: cornering limit from grip, downforce and
//    banking (the bank's inward push and extra tyre load, as in Car.drive), then acceleration (power, the
//    rear tyres' share of grip, drag) and braking passes. This is what a perfect driver could do.
// 3. Fit: tyre grip, downforce and the 670 package's drag to the Next Gen Cup poles at the 670 tracks; the
//    plate package's drag to Daytona, Talladega and Atlanta. Power stays at the engines' real output.
const { E } = require('../lib');
const { racingLine, lapTime, scripted } = require('./line');

const check = process.argv.includes('--check');
const MPH = E.MPH_PER_SPEED, G = E.G;

const tracks = E.NASCAR_TRACKS.map(def => {
  const track = E.OvalTrack.get(def.id);
  return { def, track, line: racingLine(track) };
});
const pkgOf = t => t.def.pkg === 'plate' ? 'plate' : '670';
const fitSet = pkg => tracks.filter(t => t.def.pole.fit && pkgOf(t) === pkg);
const err = (P, set) => Math.sqrt(set.reduce((s, t) => s + (lapTime(t.track, t.line, P).mph / t.def.pole.mph - 1) ** 2, 0) / set.length);

let P670 = { ...E.NASCAR_670 }, PPLATE = { ...E.NASCAR_PLATE };
if (!check) {
  // coordinate descent in log space: tyre grip, downforce, load sensitivity, 670 drag on the 670 poles
  const keys = ['grip', 'downforce', 'loadSens', 'aero'];
  let step = 0.2;
  for (let round = 0; round < 40 && step > 0.002; round++) {
    let improved = false;
    for (const key of keys)
      for (const dir of [1, -1]) {
        const trial = { ...P670, [key]: P670[key] * Math.exp(dir * step) };
        // keep it physical: drag within 25% of the engineering estimate, load sensitivity in 0.5..1
        if (trial.aero < E.NASCAR_670.aero * 0.75 || trial.aero > E.NASCAR_670.aero * 1.25 || trial.loadSens < 0.5 || trial.loadSens > 1) continue;
        if (err(trial, fitSet('670')) < err(P670, fitSet('670'))) {
          P670 = trial;
          improved = true;
        }
      }
    if (!improved) step /= 2;
  }
  // the plate package shares the tyres; its big spoiler sets its drag
  PPLATE = { ...PPLATE, grip: P670.grip, downforce: P670.downforce, loadSens: P670.loadSens };
  step = 0.2;
  for (let round = 0; round < 40 && step > 0.002; round++) {
    let improved = false;
    for (const dir of [1, -1]) {
      const trial = { ...PPLATE, aero: PPLATE.aero * Math.exp(dir * step) };
      if (err(trial, fitSet('plate')) < err(PPLATE, fitSet('plate'))) {
        PPLATE = trial;
        improved = true;
      }
    }
    if (!improved) step /= 2;
  }
  console.log('fitted constants for js/car.js:');
  console.log(`  tyre grip ${(P670.grip / G).toFixed(3)} g  (grip: ${(P670.grip / G).toFixed(3)} * G)`);
  console.log(`  downforce ${P670.downforce.toExponential(3)}   load sensitivity ${P670.loadSens.toFixed(3)}`);
  console.log(`  670 drag  ${P670.aero.toExponential(3)}   plate drag ${PPLATE.aero.toExponential(3)}`);
  console.log(`  RMS error: 670 tracks ${(100 * err(P670, fitSet('670'))).toFixed(1)}%, plate tracks ${(100 * err(PPLATE, fitSet('plate'))).toFixed(1)}%\n`);
}

// 4. a scripted car drives the racing line at 95% of the perfect lap's speeds in the real simulation
function drive(t, P) {
  const { track } = t, heat = new E.Heat(track, [null], 3, { cars: 'stock', fastCaution: true });
  const car = heat.cars[0], driver = scripted(track, P);
  car.spec = Object.assign(Object.create(null), car.spec, P);
  let spun = false;
  while (!heat.over) {
    driver(heat, car);
    heat.tick();
    const c = Math.cos(car.angle), s = Math.sin(car.angle);
    if (Math.abs(Math.atan2(-car.vx * s + car.vy * c, car.vx * c + car.vy * s)) > 1.2) spun = true;
  }
  const best = car.laps.length ? Math.min(...car.laps) : Infinity;
  return { mph: track.length / best * MPH, spun, walls: car.wallHits, finished: car.finished };
}

console.log('track            real pole               perfect lap   error   scripted car in the sim');
for (const t of tracks) {
  const P = pkgOf(t) === 'plate' ? PPLATE : P670, perfect = lapTime(t.track, t.line, P), sim = drive(t, P), d = t.def;
  const e = d.pole.mph ? (perfect.mph / d.pole.mph - 1) * 100 : 0;
  const pole = `${d.pole.mph.toFixed(1)} (${d.pole.year}${d.pole.series !== 'Cup' ? ' ' + d.pole.series : ''})`;
  console.log(`${d.short.padEnd(14)} ${pole.padEnd(24)} ${perfect.mph.toFixed(1).padStart(7)} mph ${(e >= 0 ? '+' : '') + e.toFixed(1)}%${d.pole.fit ? ' ' : '*'}  ${sim.mph.toFixed(1)} mph${sim.spun ? ' SPUN' : ''}${sim.walls ? ` walls ${sim.walls}` : ''}${sim.finished ? '' : ' DNF'}`);
}
console.log('\n* not in the fit: older car generations or other series (shown for reference)');
