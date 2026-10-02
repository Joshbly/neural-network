const $ = sel => document.querySelector(sel);
const stage = $('#stage'), canvas = $('#view'), ctx = canvas.getContext('2d');

const SPEEDS = [1, 3, 10, 30, Infinity];
const RACE_FIELD = 10, HUMAN_SLOT = 4, HUMAN_RACE_LAPS = 3, YOU_COLOR = '#ffd166';
const RACE_LENGTHS = [2, 5, 10];

const seed = +new URLSearchParams(location.search).get('seed') || 2;
let track = Track.random(mulberry32(seed));
const sim = new Sim(track);

const state = { mode: 'train', speed: 3, paused: true, showLine: true, showRays: true, focus: null };
const keys = {}, humanBest = {}, milestones = {};
let panels, line = null, race = null, sketch = null, lastFocus = null, cameraBeforeSketch = 'follow';
// the top-20 field trained offline (train/es.js + train/field.js), raced back to back on the current track
let field = null, pros = null;

function fitCanvas(el) {
  const dpr = devicePixelRatio || 1, { width, height } = el.getBoundingClientRect();
  el.width = Math.round(width * dpr);
  el.height = Math.round(height * dpr);
  const c = el.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx: c, w: width, h: height };
}

function layout() {
  const box = stage.getBoundingClientRect();
  Object.assign(view, { w: box.width, h: box.height, dpr: devicePixelRatio || 1 });
  canvas.width = Math.round(view.w * view.dpr);
  canvas.height = Math.round(view.h * view.dpr);
  track.minimap = null;
  panels = { brain: fitCanvas($('#brain')), progress: fitCanvas($('#chart-progress')), lap: fitCanvas($('#chart-lap')), walls: fitCanvas($('#chart-walls')) };
  drawCharts();
}

const activeHeat = () => race ? race.heat : pros ? pros.heat : sim.heats[sim.featured];
// each species gets its own family of shades, so A and B are tellable apart at a glance
const colorOf = car => car.human ? YOU_COLOR : car.pro ? car.species.color : `hsl(${car.species.hue + (car.slot % 4 - 1.5) * 9} 85% ${54 + (car.slot % 3) * 9}%)`;
const nameOf = car => car.human ? 'YOU' : car.pro ? car.species.name : race ? `AI ${car.rank}·${car.species.name}` : `${car.species.name}${car.slot + 1}`;
const describe = sp => `${sp.layers.length - 2} hidden layers (${sp.layers.slice(1, -1).join('-')})`;
const damageText = car => {
  const { front, rear, drag } = car.condition, pct = v => Math.round(v * 100);
  return car.wrecked ? 'WRECKED' : `${pct(front)}·${pct(rear)}% +${pct(drag)}%`;
};

function leaderOf(heat) {
  let lead = null;
  for (const car of heat.cars) if (car.running && (!lead || car.progress > lead.progress)) lead = car;
  return lead;
}

function pickFocus(heat) {
  if (race) return race.you;
  if (state.focus?.running && heat.cars.includes(state.focus)) return state.focus;
  state.focus = null;
  return leaderOf(heat) || lastFocus;
}

// the AI whose brain is on screen during a race: whoever is closest to you
function nearestRival(heat, you) {
  let best = null, bestDist = Infinity;
  for (const car of heat.cars) {
    if (car === you || !car.running) continue;
    const d = Math.hypot(car.x - you.x, car.y - you.y);
    if (d < bestDist) [best, bestDist] = [car, d];
  }
  return best;
}

// ---------- drawing ----------

function drawSketch() {
  const pts = sketch.points;
  if (pts.length < 2) return;
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.beginPath();
  pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.strokeStyle = 'rgba(38, 42, 50, 0.9)';
  ctx.lineWidth = HALF_WIDTH * 2;
  ctx.stroke();
  ctx.strokeStyle = '#38e1ff';
  ctx.lineWidth = 2 / view.zoom;
  ctx.stroke();
}

function render() {
  const heat = sketch ? null : activeHeat(), focus = heat && pickFocus(heat);
  if (focus) lastFocus = focus;
  moveCamera(focus, false);
  drawGround(ctx);
  if (sketch) return drawSketch();

  drawTrack(ctx, track);
  if (state.showLine && line && state.mode === 'train') drawRacingLine(ctx, line);
  if (heat.events) spawnImpacts(heat.events);
  for (const car of heat.cars) if (car.running) drawWake(ctx, car);
  for (const car of heat.cars) {
    if (car.running) {
      drawCar(ctx, car, colorOf(car), { glow: car === focus });
      if (view.mode === 'follow') spawnSmoke(car);
    } else {
      const fade = 1 - (heat.step - car.doneAt) / 45;
      if (fade > 0) drawCar(ctx, car, colorOf(car), { alpha: fade * 0.7 });
    }
  }
  drawParticles(ctx);
  const thinker = race ? nearestRival(heat, race.you) : focus;
  if (thinker && state.showRays) drawRays(ctx, thinker, track);
  for (const car of heat.cars)
    if (car.running && (view.mode === 'follow' || car === focus))
      drawLabel(ctx, car, nameOf(car), car === focus || car.human ? '#fff' : 'rgba(220, 230, 245, 0.5)');
  if (view.mode === 'follow') drawMinimap(ctx, track, heat.cars, focus, colorOf);

  if (thinker) {
    drawBrain(panels.brain.ctx, panels.brain.w, panels.brain.h, thinker);
    $('#brain-title').textContent = `${race ? `${nameOf(thinker)}, nearest rival` : `Car ${nameOf(thinker)}`} · ${describe(thinker.species)}`;
  }
}

// one line per species for a given per-generation stat
const perSpecies = (rows, stat) => sim.species.map(sp => ({
  points: rows.filter(h => h.species[sp.name][stat] != null).map(h => [h.gen, h.species[sp.name][stat]]),
  color: sp.color,
}));

// share of heat wins over the trailing `span` generations, so the chart shows current form, not history
const winShare = (rows, sp, span) => rows.map((h, i) => {
  const recent = rows.slice(Math.max(0, i - span + 1), i + 1);
  return [h.gen, recent.reduce((n, r) => n + r.species[sp.name].wins, 0) / (recent.length * HEATS)];
});

function drawCharts() {
  const { history } = sim, here = history.filter(h => h.trackId === track.id), human = humanBest[track.id];
  drawChart(panels.progress.ctx, panels.progress.w, panels.progress.h, {
    series: sim.species.map(sp => ({ points: winShare(here, sp, 5), color: sp.color })),
    yMin: 0, yMax: 1,
    ref: here.length && { value: 0.5, label: 'even' },
    format: v => `${Math.round(v * 100)}%`,
    empty: 'waiting for generation 1 to finish…',
  });

  const laps = perSpecies(here, 'lap'), values = laps.flatMap(s => s.points.map(p => p[1])).concat(human ? [human] : []);
  drawChart(panels.lap.ctx, panels.lap.w, panels.lap.h, {
    series: laps,
    yMin: Math.floor(Math.min(...values) * 0.95), yMax: Math.ceil(Math.max(...values) * 1.05) || 1,
    ref: human && { value: human, label: `you ${human.toFixed(2)}s` },
    format: v => `${v.toFixed(0)}s`,
    empty: 'no car has finished a lap yet',
  });

  const walls = perSpecies(here, 'wallHits');
  drawChart(panels.walls.ctx, panels.walls.w, panels.walls.h, {
    series: walls,
    yMin: 0, yMax: Math.max(4, ...walls.flatMap(s => s.points.map(p => p[1]))),
    format: v => v.toFixed(0),
    empty: 'nobody has finished yet',
  });

  const record = milestones[track.id]?.record, tally = rows => sim.species.map(sp => rows.reduce((n, h) => n + h.species[sp.name].wins, 0));
  const recent = tally(here.slice(-10)), total = tally(here);
  $('#stat-lap').innerHTML = record ? `${record.toFixed(2)}s <small style="color:${milestones[track.id].holder.color}">${milestones[track.id].holder.name}</small>` : '—';
  $('#stat-wins').innerHTML = here.length
    ? sim.species.map((sp, s) => `<span style="color:${sp.color}">${recent[s]}</span>`).join(' : ') + `<small>total ${total.join(' : ')}</small>`
    : '—';
  $('#stat-human').textContent = human ? `${human.toFixed(2)}s` : '—';
}

function towerGap(heat, car, standings) {
  const first = standings[0];
  if (car.retired) return 'OUT';
  if (car.finished) return car === first ? 'WIN' : `+${((car.steps - first.steps) / 60).toFixed(1)}`;
  if (car === first) return 'LEAD';
  return `+${heat.gap(car).toFixed(2)}`;
}

function updateHud() {
  const heat = activeHeat(), standings = heat.standings(), focus = race ? race.you : pickFocus(heat);
  const lead = standings[0];
  $('#tower-head').textContent = `${race ? 'RACE' : pros ? `PRO RACE ${pros.count}` : `HEAT ${sim.featured + 1}`} · LAP ${Math.min(heat.laps, lead.laps.length + 1)}/${heat.laps}`;
  const shown = standings.slice(0, 12);
  if (focus && !shown.includes(focus)) shown[shown.length - 1] = focus;
  $('#tower-list').innerHTML = shown.map(car => `
    <li data-slot="${car.slot}" class="${car === focus ? 'focus' : ''} ${car.retired ? 'out' : ''}">
      <b>${standings.indexOf(car) + 1}</b><i style="background:${colorOf(car)}"></i>
      <span${car.species ? ` style="color:${car.species.color}"` : ''}>${nameOf(car)}${car.elite ? ' <span class="star">★</span>' : ''}</span><em>${towerGap(heat, car, standings)}</em>
    </li>`).join('');

  if (race) {
    const you = race.you;
    $('#race-pos').textContent = `P${standings.indexOf(you) + 1}/${heat.cars.length}`;
    $('#race-lap').textContent = `${Math.min(heat.laps, you.laps.length + 1)}/${heat.laps}`;
    $('#race-time').textContent = (you.steps / 60).toFixed(2);
    $('#race-dmg').textContent = damageText(you);
  } else {
    $('#hud-gen').textContent = pros ? pros.count : sim.generation;
    $('#hud-heat-n').textContent = `${sim.featured + 1}/${HEATS}`;
    $('#hud-alive').textContent = heat.cars.reduce((n, car) => n + car.running, 0);
  }

  const car = race ? race.you : focus;
  if (car) $('#car-stats').innerHTML = [
    ['speed', `${Math.round(Math.max(0, car.forwardSpeed) * KMH)} km/h`],
    ['slipstream', `${Math.round(car.draft * 100)}%`],
    ['aero lost F·R · drag', damageText(car)],
    ['wall hits', car.wallHits],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
}

// ---------- UI helpers ----------

function toast(html, tone = '') {
  const box = $('#toasts'), el = document.createElement('div');
  el.className = `toast ${tone}`;
  el.innerHTML = html;
  box.append(el);
  while (box.children.length > 3) box.firstChild.remove();
  setTimeout(() => el.classList.add('out'), 5500);
  setTimeout(() => el.remove(), 6100);
}

function banner(big, small = '', compact = false) {
  const el = $('#banner');
  el.hidden = !big;
  el.classList.toggle('compact', compact);
  el.querySelector('b').textContent = big || '';
  el.querySelector('small').textContent = small;
}

function setPaused(paused) {
  state.paused = paused;
  $('#btn-pause').textContent = paused ? 'Resume' : 'Pause';
  $('#btn-pause').classList.toggle('active', paused);
}

function setSpeed(i) {
  state.speed = SPEEDS[i];
  document.querySelectorAll('#speed button').forEach((b, k) => b.classList.toggle('on', k === i));
  $('#hud-speed').textContent = state.speed === Infinity ? 'MAX' : `${state.speed}×`;
}

function setCamera(mode) {
  view.mode = mode;
  $('#btn-camera').textContent = `Camera: ${mode === 'follow' ? 'follow' : 'full track'}`;
}

function begin() {
  $('#intro').hidden = true;
  setPaused(false);
}

function setTrack(next, message) {
  track = next;
  sim.setTrack(next);
  line = null;
  lastFocus = state.focus = null;
  layout();
  toast(message);
  if (pros) startPros();
}

// ---------- evolution milestones ----------

sim.onGeneration = (entry, best) => {
  line = sim.champion.path.length > 6 ? racingLine(sim.champion.path) : null;
  const m = milestones[entry.trackId] ??= {};
  const now = performance.now(), improved = entry.lap && !(entry.lap >= m.record);
  const holder = entry.lap && sim.species.find(sp => entry.species[sp.name].lap === entry.lap);
  const takeover = improved && m.holder && m.holder !== holder;
  if (!m.seen && sim.history.length > 1) {
    m.firstLap = !!entry.lap;
    toast(entry.finishers
      ? `First try on a track they've never seen: <b>${entry.finishers} of ${POPULATION}</b> brains finished it.`
      : 'First try on the new track: nobody finished. Watch them adapt.', 'gold');
  } else if (entry.lap && !m.firstLap) {
    m.firstLap = true;
    toast(`<b>Generation ${entry.gen}:</b> ${entry.finishers === 1 ? 'a car' : `${entry.finishers} cars`} made it to the flag, mostly by bouncing off walls. <span style="color:#6b7689">Press 5 for MAX speed.</span>`, 'gold');
  } else if (takeover) {
    toast(`<b style="color:${holder.color}">${holder.name}</b>, ${describe(holder)}, took the lap record from <b style="color:${m.holder.color}">${m.holder.name}</b>: <b>${entry.lap.toFixed(2)}s</b>`, 'gold');
  } else if (improved && entry.lap < m.record - 0.01 && now - (m.toastAt || 0) > 8000) {
    m.toastAt = now;
    toast(`New lap record: <b>${entry.lap.toFixed(2)}s</b> by <b style="color:${holder.color}">${holder.name}</b> <span style="color:#6b7689">(gen ${entry.gen})</span>`);
  }
  m.seen = true;
  if (improved) [m.record, m.holder] = [entry.lap, holder];

  if (best.finished && best.wallHits === 0 && !milestones.clean) {
    milestones.clean = true;
    toast(`<b>Emergent behavior, gen ${entry.gen}:</b> the champion finished the whole race <b>without touching a wall</b>.`, 'pink');
  } else if (best.finished && best.braked && !milestones.brakes) {
    milestones.brakes = true;
    toast('<b>Emergent behavior:</b> the champion <b>brakes before hairpins</b> instead of bouncing off them. Watch for red brake lights.', 'pink');
  }
  if (best.finished && best.draftSteps / best.steps > 0.08 && !milestones.draft) {
    milestones.draft = true;
    toast(`<b>Slipstream:</b> the champion spent <b>${Math.round(best.draftSteps / best.steps * 100)}%</b> of its race tucked in behind rivals.`, 'pink');
  }
  const human = humanBest[entry.trackId];
  if (human && entry.lap && entry.lap < human && !m.beatHuman) {
    m.beatHuman = true;
    toast(`The AI's best lap (<b>${entry.lap.toFixed(2)}s</b>) is now faster than yours (${human.toFixed(2)}s).`, 'pink');
  }
  drawCharts();
};

// ---------- race mode ----------

function startRace() {
  if (!sim.ranked) return toast('Let it train for at least one generation first.');
  $('#intro').hidden = $('#results').hidden = true;
  const drivers = sim.ranked.slice(0, RACE_FIELD - 1);
  drivers.splice(HUMAN_SLOT, 0, null);
  const heat = new Heat(track, drivers.map(d => d && new Brain(d.species.layers, d.genes)), HUMAN_RACE_LAPS);
  heat.events = [];
  heat.cars.forEach((car, slot) => Object.assign(car, { rank: slot < HUMAN_SLOT ? slot + 1 : slot, species: drivers[slot]?.species }));
  race = { heat, you: heat.cars[HUMAN_SLOT], t: -180, steer: 0, count: null, endAt: Infinity };
  race.you.human = true;
  state.mode = 'race';
  $('#hud').hidden = true;
  $('#race-hud').hidden = false;
  moveCamera(race.you, true);
  updateHud();
}

function raceTick() {
  race.t++;
  if (race.t <= 0) {
    const n = Math.ceil(-race.t / 60);
    if (n !== race.count) banner((race.count = n) || '', n ? 'arrow keys or WASD · ↓ brakes, then reverses' : '');
    return;
  }
  if (race.t === 1) {
    banner('GO!');
    setTimeout(() => race?.t > 0 && banner(''), 700);
  }
  const target = (keys.ArrowRight || keys.KeyD ? 1 : 0) - (keys.ArrowLeft || keys.KeyA ? 1 : 0);
  race.steer += (target - race.steer) * 0.25;
  race.you.manualSteer = race.steer;
  race.you.manualThrottle = keys.ArrowUp || keys.KeyW ? 1 : keys.ArrowDown || keys.KeyS ? -1 : 0;
  race.heat.tick();
  if (race.you.finished && race.endAt === Infinity) race.endAt = race.heat.step + 480;
  if (race.heat.done || race.heat.step >= race.endAt || race.heat.step > race.heat.maxSteps * 1.5) finishRace();
}

function finishRace() {
  state.mode = 'results';
  banner('');
  const { heat, you } = race, standings = heat.standings(), first = standings[0];
  const best = car => car.laps.length ? Math.min(...car.laps) / 60 : null;
  const youLap = best(you), place = standings.indexOf(you) + 1;
  if (youLap) humanBest[track.id] = Math.min(humanBest[track.id] ?? Infinity, youLap);

  $('#res-title').textContent = place === 1 ? 'You won!' : `You finished P${place} of ${standings.length}`;
  $('#res-table').innerHTML = standings.map((car, i) => {
    const result = !car.finished ? (car.retired ? 'OUT' : `${Math.round((car.progress + car.gridOffset) / (car.raceLaps * track.length) * 100)}%`)
      : car === first ? `${(car.steps / 60).toFixed(2)}s` : `+${((car.steps - first.steps) / 60).toFixed(2)}s`;
    return `<tr class="${car.human ? 'you' : ''}"><td>${i + 1}</td><td><i style="background:${colorOf(car)}"></i>${nameOf(car)}</td>
      <td>${result}</td><td>best ${best(car)?.toFixed(2) ?? '—'}</td><td>${car.wallHits} wall hits</td></tr>`;
  }).join('');
  const ais = heat.cars.filter(car => !car.human), aiLap = Math.min(...ais.map(car => best(car) ?? Infinity));
  const quickest = ais.find(car => best(car) === aiLap);
  $('#res-note').innerHTML = youLap && aiLap < Infinity
    ? aiLap < youLap
      ? `The fastest AI lap (${nameOf(quickest)}, a ${quickest.species.genes.toLocaleString()}-weight network) was <b>${Math.round((youLap / aiLap - 1) * 100)}% quicker</b> than your best, found by trial and error over ${sim.history.length} generations.`
      : 'You out-lapped every AI. Give evolution a few more generations and try again.'
    : 'Finish a lap to get a lap time on the board.';
  $('#results').hidden = false;
  drawCharts();
}

function exitRace() {
  race = null;
  state.mode = 'train';
  banner('');
  $('#results').hidden = $('#race-hud').hidden = true;
  $('#hud').hidden = false;
  lastFocus = null;
}

// ---------- pro race: the trained top-20 field ----------

const proSpecies = d => ({ name: `P${d.rank}`, label: d.label, layers: d.layers, color: `hsl(${(d.rank * 137.5 + 20) % 360} 85% 62%)` });

function startPros() {
  if (!field) return toast('No trained pro field yet. Run <b>node train/field.js</b> to build models/field.json.');
  if (race) exitRace();
  if (sketch) stopDrawing();
  $('#intro').hidden = true;
  const grid = shuffle(field.slice()), heat = new Heat(track, grid.map(d => new Brain(d.layers, d.weights)), sim.laps);
  heat.events = [];
  heat.cars.forEach((car, i) => Object.assign(car, { pro: true, species: grid[i].species }));
  pros = { heat, count: (pros?.count || 0) + 1, nextAt: null };
  state.mode = 'pros';
  state.focus = lastFocus = null;
  setPaused(false);
  $('#btn-pros').textContent = 'Back to evolution';
  $('#hud-gen-label').textContent = 'Pro race';
  $('#hud-heat').hidden = true;
}

function stopPros() {
  pros = null;
  state.mode = 'train';
  state.focus = lastFocus = null;
  $('#btn-pros').textContent = 'Watch the pros';
  $('#hud-gen-label').textContent = 'Generation';
  $('#hud-heat').hidden = false;
}

function prosTick() {
  if (pros.nextAt) return;
  pros.heat.tick();
  if (!pros.heat.over) return;
  const [winner, second] = pros.heat.standings();
  const margin = second?.finished ? ` by ${((second.steps - winner.steps) / 60).toFixed(2)}s` : '';
  toast(`<b style="color:${winner.species.color}">${winner.species.name}</b> wins pro race ${pros.count}${margin}`, 'gold');
  pros.nextAt = performance.now() + 4000;
}

fetch('models/field.json').then(r => r.ok ? r.json() : null).then(data => {
  if (!data) return;
  field = data.drivers.map(d => ({ ...d, weights: Float32Array.from(d.genes), species: proSpecies(d) }));
  $('#btn-pros').disabled = false;
  $('#btn-pros').title = `The ${field.length} best drivers trained offline with evolution strategies (P)`;
}).catch(() => {});

// ---------- draw-your-own-track mode ----------

function startDrawing() {
  if (race) exitRace();
  $('#intro').hidden = true;
  state.mode = 'draw';
  sketch = { points: [], active: false };
  cameraBeforeSketch = view.mode;
  setCamera('full');
  stage.classList.add('drawing');
  $('#hud').hidden = $('#tower').hidden = true;
  banner('Draw a loop', 'one continuous stroke · long straights and tight corners make the best racing · Esc to cancel', true);
}

function stopDrawing() {
  sketch = null;
  state.mode = pros ? 'pros' : 'train';
  setCamera(cameraBeforeSketch);
  stage.classList.remove('drawing');
  $('#hud').hidden = $('#tower').hidden = false;
  banner('');
}

const eventToWorld = e => {
  const r = canvas.getBoundingClientRect();
  return toWorld(e.clientX - r.left, e.clientY - r.top);
};

canvas.addEventListener('pointerdown', e => {
  if (state.mode === 'train' || state.mode === 'pros') {
    const [x, y] = eventToWorld(e);
    let pick = null, bestDist = 20 + 12 / view.zoom;
    for (const car of activeHeat().cars) {
      const d = Math.hypot(car.x - x, car.y - y);
      if (car.running && d < bestDist) [pick, bestDist] = [car, d];
    }
    state.focus = pick;
    return;
  }
  if (state.mode !== 'draw') return;
  sketch = { points: [eventToWorld(e)], active: true };
  canvas.setPointerCapture(e.pointerId);
  banner('');
});
canvas.addEventListener('pointermove', e => {
  if (!sketch?.active) return;
  const p = eventToWorld(e), last = sketch.points.at(-1);
  if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 8) sketch.points.push(p);
});
canvas.addEventListener('pointerup', () => {
  if (!sketch?.active) return;
  sketch.active = false;
  const { track: drawn, error } = Track.fromSketch(sketch.points);
  if (error) {
    sketch.points = [];
    banner('Try again', error, true);
    return;
  }
  stopDrawing();
  setTrack(drawn, 'Your track. <b>None of these brains have ever seen it.</b> Watch what they do.');
  if (state.paused) begin();
});

$('#tower-list').addEventListener('click', e => {
  const row = e.target.closest('li');
  if (row && !race) state.focus = activeHeat().cars[+row.dataset.slot];
});

// ---------- controls ----------

RACE_LENGTHS.forEach(laps => {
  const b = document.createElement('button');
  b.textContent = `${laps} laps`;
  b.classList.toggle('on', laps === sim.laps);
  b.onclick = () => {
    sim.laps = laps;
    document.querySelectorAll('#race-length button').forEach(x => x.classList.toggle('on', x === b));
    toast(`Races will be <b>${laps} laps</b> from the next generation.`);
  };
  $('#race-length').append(b);
});

SPEEDS.forEach((s, i) => {
  const b = document.createElement('button');
  b.textContent = s === Infinity ? 'MAX' : `${s}×`;
  b.onclick = () => setSpeed(i);
  $('#speed').append(b);
});

const nextHeat = () => {
  if (race || pros) return;
  sim.feature((sim.featured + 1) % HEATS);
  state.focus = lastFocus = null;
};

$('#btn-begin').onclick = begin;
$('#btn-pause').onclick = () => $('#intro').hidden ? setPaused(!state.paused) : begin();
$('#btn-camera').onclick = () => setCamera(view.mode === 'follow' ? 'full' : 'follow');
$('#hud-heat').onclick = nextHeat;
$('#btn-track').onclick = () => {
  if (sketch) stopDrawing();
  if (race) exitRace();
  setTrack(Track.random(), 'New track. <b>None of these brains have ever seen it.</b>');
  if (!$('#intro').hidden) begin();
};
$('#btn-draw').onclick = startDrawing;
$('#btn-race').onclick = () => {
  if (sketch) stopDrawing();
  if (pros) stopPros();
  startRace();
};
$('#btn-pros').onclick = () => pros ? stopPros() : startPros();
$('#res-again').onclick = startRace;
$('#res-back').onclick = exitRace;
$('#btn-restart').onclick = () => {
  sim.restart();
  for (const key of Object.keys(milestones)) delete milestones[key];
  line = null;
  lastFocus = state.focus = null;
  drawCharts();
  toast('All brains wiped. Back to generation 1.');
};
$('#mutation').oninput = e => {
  for (const sp of sim.species) sp.evo.mutationRate = e.target.value / 100;
  $('#mutation-val').textContent = `${e.target.value}%`;
};
$('#species-legend').innerHTML = sim.species.map(sp =>
  `<div style="color:${sp.color}"><span><b>${sp.name}</b> · ${describe(sp)} · ${sp.layers.slice(1, -1).reduce((a, b) => a + b)} neurons · ${sp.genes.toLocaleString()} weights</span></div>`).join('');
$('#tog-line').onchange = e => state.showLine = e.target.checked;
$('#tog-rays').onchange = e => state.showRays = e.target.checked;

addEventListener('keydown', e => {
  if (e.code.startsWith('Arrow') || e.code === 'Space' || e.code === 'Enter') e.preventDefault();
  keys[e.code] = true;
  if (e.repeat || e.metaKey || e.ctrlKey) return;
  if (state.mode === 'race') return e.code === 'Escape' && exitRace();
  if (state.mode === 'results') {
    if (e.code === 'Enter') startRace();
    if (e.code === 'Escape') exitRace();
    return;
  }
  if (state.mode === 'draw') return e.code === 'Escape' && stopDrawing();
  if (e.code === 'Space') $('#btn-pause').click();
  if (e.code === 'KeyC') $('#btn-camera').click();
  if (e.code === 'KeyH') nextHeat();
  if (e.code === 'KeyN') $('#btn-track').click();
  if (e.code === 'KeyD') startDrawing();
  if (e.code === 'KeyR') $('#btn-race').click();
  if (e.code === 'KeyP') $('#btn-pros').click();
  if (/^Digit[1-5]$/.test(e.code)) setSpeed(+e.code.slice(5) - 1);
});
addEventListener('keyup', e => keys[e.code] = false);
addEventListener('blur', () => Object.keys(keys).forEach(k => keys[k] = false));
addEventListener('resize', layout);

// ---------- main loop ----------

let last = performance.now(), acc = 0, frameNo = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000), deadline = now + 14;
  last = now;
  const atSpeed = tick => {
    if (state.speed === Infinity) do tick(); while (performance.now() < deadline);
    else {
      acc += dt * 60 * state.speed;
      while (acc >= 1 && performance.now() < deadline) tick(), acc--;
      acc = Math.min(acc, 2);
    }
  };
  if (state.mode === 'train' && !state.paused) atSpeed(() => sim.tick());
  else if (state.mode === 'pros' && !state.paused) {
    if (pros.nextAt && now > pros.nextAt) startPros();
    else if (!pros.nextAt) atSpeed(prosTick);
  } else if (state.mode === 'race') {
    acc += dt * 60;
    while (acc >= 1 && state.mode === 'race') raceTick(), acc--;
  }
  render();
  if (frameNo++ % 8 === 0 && !sketch) updateHud();
  requestAnimationFrame(loop);
}

setSpeed(1);
setCamera('follow');
setPaused(true);
layout();
moveCamera(leaderOf(activeHeat()), true);
requestAnimationFrame(loop);
