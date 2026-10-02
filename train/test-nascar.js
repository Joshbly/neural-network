#!/usr/bin/env node
// NASCAR mode, checked end to end:
//   1. all 32 ovals build with their real length, width and banking
//   2. a scripted car laps every one of them in the real simulation without spinning
//   3. physics: a slow car slides down the banking; grass and sand slow a car; a plate-track pack is faster than a lone car
//   4. every flag rule, in scripted races (the cars are driven by race control's autopilot on fixed lanes)
//   5. brains cross over: all four mixes of tracks and cars run through the training runner with sane scores
//   node train/test-nascar.js
const fs = require('fs');
const path = require('path');
const { E, run, perturb } = require('./lib');
const { SPECS } = require('./nascar/specs');
const { scripted } = require('./nascar/line');

let failures = 0, checks = 0;
const ok = (cond, msg) => {
  checks++;
  if (!cond) {
    failures++;
    console.log(`  FAIL ${msg}`);
  }
};
const section = title => console.log(`\n${title}`);
const deg = r => r * 180 / Math.PI, MILE = 1609.344, M = 4;

// ---- 1. geometry ----
section('1. the 32 ovals');
ok(E.NASCAR_TRACKS.length === 32, `32 tracks (got ${E.NASCAR_TRACKS.length})`);
const built = new Map();
for (const spec of SPECS) {
  const t = E.OvalTrack.get(spec.id), miles = t.length / M / MILE, tol = spec.lengthTolerance ?? 0.07;
  built.set(spec.id, t);
  ok(Math.abs(miles / spec.miles - 1) <= tol, `${spec.id}: length ${miles.toFixed(3)} mi vs ${spec.miles} (±${tol * 100}%)`);
  ok(Math.abs(t.halfWidth * 2 / M - spec.widthFt * 0.3048) < 0.01, `${spec.id}: width`);
  const turns = [spec.banking.turns].flat(2), want = Math.max(...turns), got = deg(Math.max(...t.bankHi));
  ok(Math.abs(got - want) < 1.5, `${spec.id}: steepest banking ${got.toFixed(1)}° vs ${want}°`);
  ok(t.plate === (spec.pkg === 'plate'), `${spec.id}: package`);
  ok(t.turnRegions.length >= 2, `${spec.id}: found ${t.turnRegions.length} corners`);
}
console.log(`  built ${built.size}; plates at ${E.NASCAR_TRACKS.filter(d => d.pkg === 'plate').map(d => d.id).join(', ')}`);

// ---- 2. scripted laps ----
section('2. a scripted car laps every oval');
for (const t of built.values()) {
  const P = E.stockSpec(t), heat = new E.Heat(t, [null], 3, { cars: 'stock', fastCaution: true }), car = heat.cars[0], drive = scripted(t, P);
  let spun = false;
  while (!heat.over) {
    drive(heat, car);
    heat.tick();
    const c = Math.cos(car.angle), s = Math.sin(car.angle);
    spun ||= Math.abs(Math.atan2(-car.vx * s + car.vy * c, car.vx * c + car.vy * s)) > 1.2;
  }
  const mph = t.length / Math.min(...car.laps) * E.MPH_PER_SPEED;
  ok(car.finished && !spun, `${t.key}: finished ${car.finished}, spun ${spun}`);
  ok(mph > t.def.pole.mph * 0.75 && mph < t.def.pole.mph * 1.05, `${t.key}: scripted lap ${mph.toFixed(1)} mph vs pole ${t.def.pole.mph}`);
}

// ---- 3. physics ----
section('3. banking, surfaces, the draft');
// 200 mph through Talladega's 33° turns in the middle lane (beyond what tyres and downforce alone can hold):
// the banking holds the car up; flatten it and the same car at the same speed washes up the track
function corner(flat) {
  const t = built.get('talladega'), saved = [t.bankLo.slice(), t.bankHi.slice()];
  if (flat) t.bankLo.fill(0), t.bankHi.fill(0);
  const heat = new E.Heat(t, [null], 3, { cars: 'stock' }), car = heat.cars[0], v = 200 / E.MPH_PER_SPEED;
  const region = t.turnRegions[0], arc = t.arc[region.start + Math.floor((region.end - region.start) * 0.15)], [x, y] = t.pointAt(arc, 0), angle = t.headingAt(arc);
  Object.assign(car, { x, y, angle, vx: Math.cos(angle) * v, vy: Math.sin(angle) * v, spin: 0, lastArc: arc, paced: false });
  let wide = 0;
  for (let k = 0; k < 150 && car.running; k++) {
    heat.control.autodrive(car, 0, v);
    car.drive(car.auto[0], car.auto[1]);
    car.hitWalls(t, null);
    car.advance(t, k);
    wide = Math.max(wide, t.lateralAt(car.x, car.y));
  }
  t.bankLo.set(saved[0]);
  t.bankHi.set(saved[1]);
  return { wide: wide / M, walls: car.wallHits };
}
const banked = corner(false), flat = corner(true);
console.log(`  200 mph through Talladega turn 1, middle lane: drifts ${banked.wide.toFixed(1)} m wide with the banking (${banked.walls} wall hits), ${flat.wide.toFixed(1)} m flat (${flat.walls})`);
ok(banked.walls === 0 && banked.wide < 3, 'the banking holds a car through the turn at race speed');
ok(flat.wide > banked.wide + 3 || flat.walls > 0, 'without it, the same car washes up the track');

// coasting from 120 mph down the backstretch: on asphalt, on the infield grass, in the sand
function coast(surface) {
  const t = built.get('charlotte'), heat = new E.Heat(t, [null], 3, { cars: 'stock' }), car = heat.cars[0];
  const arc = t.length * 0.5, hw = t.halfWidth, lat = { asphalt: 0, grass: -hw - t.apron - 12 * M }[surface];
  const [x, y] = t.pointAt(arc, lat), angle = t.headingAt(arc), v = 120 / E.MPH_PER_SPEED;
  Object.assign(car, { x, y, angle, vx: Math.cos(angle) * v, vy: Math.sin(angle) * v, spin: 0, lastArc: arc });
  const got = t.surfaceAt(x, y);
  for (let k = 0; k < 60; k++) car.drive(0, 0);
  return { mph: Math.hypot(car.vx, car.vy) * E.MPH_PER_SPEED, surface: got };
}
const onAsphalt = coast('asphalt'), onGrass = coast('grass');
console.log(`  coasting 1 s from 120 mph: ${onAsphalt.mph.toFixed(1)} mph on asphalt, ${onGrass.mph.toFixed(1)} mph on the grass`);
ok(onAsphalt.surface === E.SURFACE.racing && onGrass.surface === E.SURFACE.grass, `surfaces under the car (${onAsphalt.surface}, ${onGrass.surface})`);
ok(onGrass.mph < onAsphalt.mph - 3, 'grass slows a car');
{
  const t = built.get('daytona'), sand = t.sandAt[0], [x, y] = t.pointAt(sand, -t.halfWidth - t.apron - 14 * M);
  ok(t.surfaceAt(x, y) === E.SURFACE.sand, 'sand where spinning cars end up (Daytona turn exits)');
}

// a single-file train of six at Daytona, flat out, against one car alone
function daytonaLap(n) {
  const t = built.get('daytona'), heat = new E.Heat(t, Array(n).fill(null), 3, { cars: 'stock', fastCaution: true }), drive = scripted(t, E.stockSpec(t));
  // one lane: everyone nose to tail on the racing line
  heat.cars.forEach((car, i) => {
    const arc = t.length - 40 - i * 34, [x, y] = t.pointAt(arc, t.halfWidth * 0.3), angle = t.headingAt(arc);
    Object.assign(car, { x, y, angle, lastArc: arc, progress: arc - t.length });
    car.gridOffset = -car.progress;
    car.checkpoint = car.progress;
  });
  while (!heat.over) {
    for (const car of heat.cars) if (car.running && !car.paced) drive(heat, car, { pace: 2, lane: t.halfWidth * 0.3 });
    heat.tick();
  }
  const laps = heat.cars.filter(car => car.laps.length >= 2).map(car => car.laps[1]);
  return laps.reduce((a, b) => a + b, 0) / laps.length / 60;
}
const alone = daytonaLap(1), pack = daytonaLap(6);
console.log(`  Daytona, second lap flat out: alone ${alone.toFixed(2)} s, in a six-car train ${pack.toFixed(2)} s`);
ok(pack < alone * 0.99, 'the pack is faster than a lone car at a plate track');

// racing in traffic: two wide is slow for both, a push through a corner gets the leader loose, a side draft
// slows the car being side-drafted
function formation(id, setup, laps = 3) {
  const t = built.get(id), heat = new E.Heat(t, Array(setup.length).fill(null), laps, { cars: 'stock', practice: true }), drive = scripted(t, E.stockSpec(t));
  heat.cars.forEach((car, i) => {
    const arc = t.length - 40 - setup[i].back, [x, y] = t.pointAt(arc, setup[i].lane), angle = t.headingAt(arc);
    Object.assign(car, { x, y, angle, lastArc: arc, progress: arc - t.length });
    car.gridOffset = -car.progress;
    car.checkpoint = car.progress;
  });
  const loose = heat.cars.map(() => 0);
  let steps = 0;
  while (!heat.over) {
    heat.cars.forEach((car, i) => car.running && !car.paced && drive(heat, car, { pace: setup[i].pace ?? 2, lane: setup[i].lane }));
    heat.tick();
    steps++;
    heat.cars.forEach((car, i) => loose[i] += car.rearLoose);
  }
  return heat.cars.map((car, i) => ({ lap: car.laps[1] / 60, loose: loose[i] / steps }));
}
{
  const [solo] = formation('daytona', [{ back: 0, lane: 0 }]), wide = formation('daytona', [{ back: 0, lane: -5 }, { back: 0, lane: 5 }]);
  const cornerSolo = formation('kansas', [{ back: 0, lane: 0, pace: 0.97 }]), cornerPushed = formation('kansas', [{ back: 0, lane: 0, pace: 0.97 }, { back: 21, lane: 0, pace: 1.3 }]);
  console.log(`  Daytona lap 2: alone ${solo.lap.toFixed(2)} s, two wide ${wide.map(c => c.lap.toFixed(2)).join(' / ')} s; Kansas leader loose ${(100 * cornerSolo[0].loose).toFixed(1)}% alone, ${(100 * cornerPushed[0].loose).toFixed(1)}% pushed`);
  ok(wide.every(c => c.lap > solo.lap + 0.3), 'running two wide is slower for both cars');
  ok(cornerPushed[0].loose > cornerSolo[0].loose + 0.1, 'a push through a corner takes the air off the leader\'s spoiler: it gets loose');
  const t = built.get('daytona'), heat = new E.Heat(t, [null, null], 3, { cars: 'stock', practice: true }), drive = scripted(t, E.stockSpec(t));
  heat.tick();
  const [a, b] = heat.cars, arc = t.length * 0.35, h = t.headingAt(arc), v = 170 / E.MPH_PER_SPEED;
  const put = (car, back, lane) => { const [x, y] = t.pointAt(arc - back, lane); Object.assign(car, { x, y, angle: h, vx: Math.cos(h) * v, vy: Math.sin(h) * v, spin: 0, lastArc: arc - back, paced: false }); };
  put(a, 0, -5);
  put(b, 12, 5.6);
  for (let k = 0; k < 120; k++) {
    drive(heat, a, { pace: 2, lane: -5 });
    drive(heat, b, { pace: 2, lane: 5.6 });
    heat.tick();
  }
  ok(Math.hypot(b.vx, b.vy) > Math.hypot(a.vx, a.vy), 'a nose on the rear quarter side-drafts the car ahead: it slows');
}

// bodies are their outlines: a bump to the bumper, even half a metre off-centre, pushes the car ahead straight on
function bump(offsetUnits) {
  const t = built.get('michigan'), heat = new E.Heat(t, [null, null], 3, { cars: 'stock', practice: true });
  heat.tick();
  const [lead, chase] = heat.cars, arc = t.length * 0.5, h = t.headingAt(arc);
  const place = (car, back, lat, speed) => {
    const [x, y] = t.pointAt(arc - back, lat);
    Object.assign(car, { x, y, angle: h, vx: Math.cos(h) * speed, vy: Math.sin(h) * speed, spin: 0, lastArc: arc - back, paced: false, manualSteer: 0, manualThrottle: 0, steer: 0 });
  };
  place(lead, 0, 0, 4);
  place(chase, 21, offsetUnits, 4.6);
  let sideways = 0;
  for (let k = 0; k < 60; k++) {
    heat.tick();
    sideways = Math.max(sideways, Math.abs(-lead.vx * Math.sin(lead.angle) + lead.vy * Math.cos(lead.angle)));
  }
  return { sideways, gained: lead.vx * Math.cos(lead.angle) + lead.vy * Math.sin(lead.angle) - 4 };
}
{
  const square = bump(0), offset = bump(2);
  console.log(`  bump from behind: car ahead gains ${square.gained.toFixed(2)}, sideways ${square.sideways.toFixed(3)} square on, ${offset.sideways.toFixed(3)} half a metre off-centre`);
  ok(square.gained > 0.1 && square.sideways < 0.01 && offset.sideways < 0.05, 'a bump pushes the car ahead straight on, not aside');
  const spec = E.NASCAR_670;
  ok(spec.box.hx === spec.len / 2 && spec.box.hy === spec.wid / 2, 'the collision body is the drawn body: 4.97 m by 1.99 m');
}

// ---- 4. flags ----
section('4. flag rules');
// n scripted cars on their grid lanes; hooks run before each step
function scriptedRace(id, n, laps, opts = {}, pace = 0.8) {
  const t = built.get(id), drive = scripted(t, E.stockSpec(t));
  const heat = new E.Heat(t, Array(n).fill(null), laps, { cars: 'stock', ...opts });
  const lane = car => (car.slot & 1 ? 1 : -1) * t.halfWidth * 0.42;
  const step = () => {
    for (const car of heat.cars) if (car.running && !car.paced) drive(heat, car, { pace, lane: lane(car) });
    heat.tick();
  };
  const until = (cond, limit = 200000) => {
    for (let k = 0; k < limit && !heat.over; k++) {
      if (cond()) return true;
      step();
    }
    return cond();
  };
  return { t, heat, control: heat.control, step, until, said: text => heat.control.log.some(l => l.text.includes(text)) };
}
const spinOut = (car, t) => {
  car.angle += Math.PI / 2;
  car.vx *= 0.2;
  car.vy *= 0.2;
};
const leaderOf = heat => heat.standings().find(car => car.running) ?? heat.standings()[0];

// green, white, checkered: a clean 3-lap race at Bristol
{
  const r = scriptedRace('bristol', 8, 3);
  ok(r.control.log[0].flag === 'green' && r.heat.cars.every(car => car.paced), 'rolling start: green, everyone paced');
  r.step();
  ok(r.heat.cars.every(car => !car.paced), 'green: everyone racing after the first decision');
  r.until(() => r.heat.done);
  ok(r.said('White flag') && r.said('Checkered flag'), `white then checkered (${r.control.log.map(l => l.flag).join(' ')})`);
  ok(r.heat.cars.every(car => car.finished), 'everyone takes the checkered flag');
}

// yellow: a spin brings out the caution; frozen order with the spinner at the back; pace car; lucky dog; one to go; double-file restart
{
  const r = scriptedRace('bristol', 10, 12);
  r.until(() => r.heat.step > 1500);
  // car 9 a lap down, so there's a lucky dog to give
  const lapped = r.heat.cars[9];
  lapped.progress -= r.t.length;
  const culprit = r.heat.standings()[2];
  spinOut(culprit, r.t);
  r.until(() => r.control.flag === 'yellow', 200);
  ok(r.control.flag === 'yellow' && r.said('spun'), `a spin brings out the yellow (${r.control.log.at(-1).text})`);
  ok(culprit.cautionsCaused === 1, 'the spinner is charged with the caution');
  const leadLap = r.control.order.filter(car => car !== lapped);
  ok(leadLap.at(-1) === culprit, 'the spinner restarts at the back of the lead lap');
  ok(r.said('Lucky dog') && lapped.freeLaps === 1, 'lucky dog: the first lapped car gets its lap back');
  r.step();
  ok(!!r.control.paceCar && r.heat.cars.filter(car => car.running).every(car => car.paced), 'pace car out, the field on autopilot');
  const lead = leaderOf(r.heat), counted = r.control.lapsOf(lead), crossed = lead.laps.length;
  r.until(() => r.control.phase === 'onetogo', 30000);
  ok(r.said('One to go'), 'one to go');
  r.until(() => r.control.phase === 'green', 30000);
  ok(r.control.phase === 'green' && r.control.log.at(-1).flag === 'green', 'green again: the restart');
  const [p1, p2] = r.control.order.filter(car => car.running);
  ok(Math.sign(r.t.lateralAt(p1.x, p1.y)) !== Math.sign(r.t.lateralAt(p2.x, p2.y)), 'double-file restart: the front row side by side');
  ok(lead.laps.length > crossed && r.control.lapsOf(lead) === counted, `caution laps don't count (crossed the line ${lead.laps.length - crossed} times under yellow, still on lap ${counted + 1})`);
}

// no cautions in short races: a spin in a 2-lap practice race at Daytona stays green
{
  const r = scriptedRace('daytona', 8, 2, { practice: true });
  r.until(() => r.heat.step > 900);
  spinOut(r.heat.standings()[2], r.t);
  r.until(() => r.heat.done);
  ok(r.control.cautions === 0 && !r.control.log.some(l => l.flag === 'yellow'), 'a practice race stays green through a spin');
  const short = scriptedRace('bristol', 8, 4);
  short.until(() => short.heat.step > 900);
  spinOut(short.heat.standings()[2], short.t);
  short.until(() => short.heat.done);
  ok(short.control.cautions === 0, 'so does any race under 5 laps');
  // a young save's tournament races: a long race with stages, cautions off
  const young = scriptedRace('martinsville', 12, 9, { stages: true, cautions: false });
  young.until(() => young.heat.step > 1500);
  spinOut(young.heat.standings()[2], young.t);
  young.until(() => young.heat.done);
  ok(young.control.cautions === 0 && young.said('Stage 1 ends') && young.heat.cars.reduce((s, car) => s + car.stagePoints, 0) === 110,
    'with cautions off, a long race stays green and its stage breaks still pay points');
}

// overtime: a caution on the final lap means a green-white-checkered (the race still runs its full green
// distance); after two attempts, a caution on the final lap ends it under yellow
{
  const r = scriptedRace('bristol', 8, 6);
  // the leader a third of the way round its last lap, under green
  const finalLap = () => r.control.phase === 'green' && !leaderOf(r.heat).underYellow && r.control.lapsOf(leaderOf(r.heat)) === 5 && r.control.arcOf(leaderOf(r.heat)) > r.t.length * 0.3;
  r.until(finalLap);
  spinOut(r.heat.standings()[3], r.t);
  r.until(() => r.said('attempt 1'), 400);
  ok(r.said('Overtime: green-white-checkered, attempt 1'), 'a caution on the final lap means overtime');
  r.until(() => r.control.phase === 'green', 40000);
  const lead = leaderOf(r.heat);
  ok(r.control.lapsOf(lead) === 5, 'the restart is on the final lap: nothing ran out under yellow');
  r.until(() => !lead.underYellow, 2000);
  ok(r.control.lapsOf(lead) === 5 && !r.heat.cars.some(car => car.finished), 'crossing the line at the restart starts a full green lap, it doesn\'t finish the race');
  for (const attempt of [2, 3]) {
    r.until(finalLap, 100000);
    spinOut(r.heat.standings()[3], r.t);
    r.until(() => r.control.phase !== 'green', 400);
    if (attempt === 2) r.until(() => r.control.phase === 'green', 40000);
  }
  ok(r.said('attempt 2') && r.said('ends under yellow'), 'after two attempts, a caution on the final lap ends it under yellow');
}

// red: four cars out within a few seconds stops the race; it resumes under caution
{
  const r = scriptedRace('martinsville', 10, 10);
  r.until(() => r.heat.step > 1200);
  for (const k of [2, 4, 6, 8]) {
    r.heat.cars[k].wrecked = true;
    r.step();
  }
  r.until(() => r.control.phase === 'red', 200);
  ok(r.control.phase === 'red' && r.said('Red flag'), 'a big wreck brings out the red flag');
  const at = r.heat.cars.find(car => car.running), x = at.x;
  r.until(() => false, 100);
  ok(at.x === x, 'red flag: everything stops');
  r.until(() => r.control.phase !== 'red', 400);
  ok(r.control.phase === 'caution' && r.said('resumes under caution'), 'then the race resumes under yellow');
}

// black: passing below the yellow line at Daytona
{
  const r = scriptedRace('daytona', 6, 4);
  r.until(() => r.heat.step > 600);
  const order = r.heat.standings(), cheat = order[4], ahead = order[2];
  const arc = r.control.arcOf(ahead) + 30, [x, y] = r.t.pointAt(arc, -r.t.halfWidth - r.t.apron * 0.7), angle = r.t.headingAt(arc);
  Object.assign(cheat, { x, y, angle, vx: ahead.vx, vy: ahead.vy, progress: ahead.progress + 30, lastArc: ((arc % r.t.length) + r.t.length) % r.t.length });
  r.until(() => cheat.penalty > 0, 120);
  ok(cheat.penalty > 0 && r.said('yellow line'), 'passing below the yellow line: black flag');
  ok(cheat.paced, 'stop-and-go: race control holds it to pit-road speed');
  r.until(() => !cheat.penalty, 20000);
  ok(!cheat.penalty && !cheat.paced, 'penalty served, back to racing');
  ok(cheat.belowLine > 0, 'time below the line is counted');
}

// black: jumping a restart
{
  const r = scriptedRace('bristol', 10, 12);
  r.until(() => r.heat.step > 1500);
  spinOut(r.heat.standings()[8], r.t);
  r.until(() => r.control.phase === 'green' && r.heat.step > 1600, 40000);
  const order = r.control.order.filter(car => car.running), jumper = order[6], target = order[1];
  const arc = r.control.arcOf(target) + 25, [x, y] = r.t.pointAt(arc, 0);
  Object.assign(jumper, { x, y, vx: target.vx, vy: target.vy, progress: target.progress + 25, lastArc: ((arc % r.t.length) + r.t.length) % r.t.length });
  r.until(() => jumper.penalty > 0, 30);
  ok(jumper.penalty > 0 && r.said('jumped the restart'), 'passing before the line on a restart: black flag');
}

// red-and-black: too damaged to go on, parked
{
  const r = scriptedRace('richmond', 6, 5);
  r.until(() => r.heat.step > 600);
  const car = r.heat.cars[3];
  car.damage.front = car.damage.rear = 1e3;
  r.until(() => !car.running, 30);
  ok(car.parked && car.retired && r.said('parked'), 'damaged vehicle policy: parked');
}

// blue-and-yellow: a lapped car with the leaders closing in
{
  const r = scriptedRace('martinsville', 6, 10);
  r.until(() => leaderOf(r.heat).laps.length >= 2);
  const lead = leaderOf(r.heat), slow = r.heat.standings().at(-1);
  const arc = r.control.arcOf(lead) + 150, [x, y] = r.t.pointAt(arc, r.t.halfWidth * 0.42), angle = r.t.headingAt(arc);
  slow.laps.length = lead.laps.length - 1;
  Object.assign(slow, { x, y, angle, vx: lead.vx, vy: lead.vy, progress: lead.progress + 150 - r.t.length, lastArc: ((arc % r.t.length) + r.t.length) % r.t.length });
  r.until(() => slow.blueFlag, 30);
  ok(slow.blueFlag, 'blue-and-yellow flag for the lapped car');
}

// stages: two stage breaks, points to the top ten
{
  const r = scriptedRace('martinsville', 12, 9, { stages: true });
  r.until(() => r.said('Stage 1 ends'), 200000);
  ok(r.said('Stage 1 ends'), `stage 1 ends at lap ${r.control.stages.length ? 3 : '?'}`);
  ok(r.heat.cars.reduce((s, car) => s + car.stagePoints, 0) === 55, 'stage points: 10 for the winner down to 1 for tenth');
  r.until(() => r.said('Stage 2 ends'), 200000);
  ok(r.said('Stage 2 ends'), 'stage 2');
}

// training's compressed caution: the field closes up behind the leader and goes green; nobody gains distance
{
  const r = scriptedRace('bristol', 10, 12, { fastCaution: true });
  r.until(() => r.heat.step > 1500);
  const lead = leaderOf(r.heat), before = lead.progress, culprit = r.heat.standings()[1];
  spinOut(culprit, r.t);
  r.until(() => r.control.cautions > 0, 200);
  ok(r.control.cautions === 1 && r.control.phase === 'green', 'training caution: straight back to green');
  ok(Math.abs(lead.progress - before) < 4 * 30, `the leader stays put (moved ${(lead.progress - before).toFixed(0)} units)`);
  ok(r.control.order.at(-1) === culprit || r.control.order.indexOf(culprit) === r.control.order.filter(c => c.freeLaps === 0 && c.running).length - 1, 'the spinner restarts at the back');
  const crossers = r.heat.cars.filter(car => car.laps.includes(Infinity) || car.lapVoid);
  ok(crossers.length === r.control.order.length, 'every lap the reform touched is void');
  // solo runs have no cautions: a spin only costs your own time
  const solo = scriptedRace('bristol', 1, 3, { fastCaution: true });
  solo.until(() => solo.heat.step > 600);
  spinOut(solo.heat.cars[0], solo.t);
  solo.until(() => false, 300);
  ok(solo.control.cautions === 0, 'no cautions in a time trial');
}

// ---- 5. brains cross over ----
section('5. brains in both modes, both kinds of car');
const slot = fs.readdirSync(path.join(__dirname, '..', 'models', 'slots')).filter(d => d.startsWith('slot-')).map(d => path.join(__dirname, '..', 'models', 'slots', d, 'state.json')).find(f => fs.existsSync(f));
const st = JSON.parse(fs.readFileSync(slot, 'utf8')), drivers = st.population.map(a => ({ layers: a.layers, genes: Float32Array.from(a.genes) }));
const combos = [
  ['generated tracks, normal cars', { trackSeed: 10_000_900, laps: 3 }],
  ['generated tracks, stock cars', { trackSeed: 10_000_900, laps: 3, cars: 'stock' }],
  ['NASCAR ovals, normal cars', { trackId: 'kansas', laps: E.lapsFor('kansas', 5500), cars: 'normal' }],
  ['NASCAR ovals, stock cars', { trackId: 'kansas', laps: E.lapsFor('kansas', 5500), cars: 'stock' }],
];
for (const [label, where] of combos) {
  const me = drivers[0], theta = perturb(me.genes, 7, 0.04);
  const race = run({ kind: 'race', ...where, layers: me.layers, genes: theta, opponents: drivers.slice(1, 12), slot: 3 });
  const tt = run({ kind: 'tt', ...where, layers: me.layers, genes: me.genes });
  const field = run({ kind: 'field', ...where, laps: where.laps, drivers: drivers.slice(0, 20), ...where.trackId && { stages: true } });
  const fine = [race.score, tt.score, ...field.map(f => f.place)].every(Number.isFinite);
  console.log(`  ${label.padEnd(32)} race score ${race.score.toFixed(2)} (P${race.place + 1}), time trial ${tt.score.toFixed(2)}, field of 20 ok`);
  ok(fine && new Set(field.map(f => f.place)).size === 20, `${label}: runs and scores`);
}

console.log(`\n${checks - failures}/${checks} checks passed${failures ? `, ${failures} FAILED` : ''}`);
process.exit(failures ? 1 : 0);
