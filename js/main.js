const $ = sel => document.querySelector(sel);
const stage = $('#stage'), canvas = $('#view'), ctx = canvas.getContext('2d');

const SPEEDS = [1, 3, 10, 30, Infinity];
const RACE_FIELD = 10, HUMAN_SLOT = 4, HUMAN_RACE_LAPS = 3, YOU_COLOR = '#ffd166';
const RACE_LENGTHS = [2, 5, 10];

const seed = +new URLSearchParams(location.search).get('seed') || 2;
let track = Track.random(mulberry32(seed));
const sim = new Sim(track);

// playing: your own track (new, drawn, or a race against them); otherwise evolving, on the training tracks
// focus: the car you clicked this race (otherwise the camera follows the leader)
// picked: the brain you last clicked, whose practice is shown next
// away: watching the save's brains on the other kind of track ('nascar' or 'normal'), in their own cars
// (awayCars 'own') or the cars that belong there ('host')
// channel: which kind of real race the evolving screen shows (see CHANNELS)
const state = { mode: 'train', speed: 3, paused: true, showRays: true, playing: false, focus: null, picked: null, away: null, awayCars: 'own', oval: -1, channel: 'live' };
const keys = {};
let panels, race = null, sketch = null, lastFocus = null, cameraBeforeSketch = 'follow';
// The races you watch: a save's latest generation (models/slots/<id>/state.json, written by train/evolve.js).
let league = null, pros = null, engine = null, progress = null, xrayShown = null, rating = null;
// everyone's weights at the start of the engine's current round, plus the practice plan it's running
let live = null;
// the neuron lab: what train/neurons.js found about each analysed brain's neurons, and the neuron being inspected
let neuronAtlas = null, pickedNeuron = null;
// the tracks the engine raced generation g's tournament on (train/evolve.js tourneyTrack)
const TOURNEY_TRACKS = 8, tourneyTrack = (gen, k) => 5_000_000 + gen * 8 + k;
// races in a generation's tournament (older saves didn't record it: 24, or fields of 20 for ~24 each)
const tourneyRaces = h => h.races ?? (h.agents.length > 20 ? Math.ceil(24 * h.agents.length / 20) : 24);
// the strongest brains first: race points (wins count most), or field beaten for saves from before points
const strength = a => a.last?.points ?? 1 - (a.last?.avgPlace ?? 1), byStrength = (x, y) => strength(y) - strength(x);
let armedDelete = null;
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);
const slotUrl = (id, file) => `models/slots/${id}/${file}`;
const slotName = id => engine?.slots?.find(s => s.id === id)?.name || id;
const DESIGN_HUE = { A: 205, B: 268, C: 150, D: 30, E: 330 };
const DESIGN_OF = { '16-10': 'A', '32-24-16': 'B', '64-64': 'C', '64-64-64': 'D', '128-128': 'E' };
const designOf = layers => DESIGN_OF[layers.slice(1, -1).join('-')] || '?';
const designColor = d => `hsl(${DESIGN_HUE[d] ?? 0} 85% 64%)`;

// A save races generated tracks or the real NASCAR ovals, in normal cars or stock cars (any mix). Its brains
// can be watched on the other kind of track too, in their own cars or the ones that race there.
const homeCars = tracks => tracks === 'nascar' ? 'stock' : 'normal';
const modeOf = id => {
  const s = engine?.slots?.find(x => x.id === id), tracks = s?.tracks ?? 'normal';
  return { tracks, cars: s?.cars ?? homeCars(tracks) };
};
const homeMode = () => modeOf(league?.slot ?? engine?.active);
const watchedTracks = () => state.away ?? homeMode().tracks;
const carsFor = t => {
  const home = homeMode(), tracks = t.nascar ? 'nascar' : 'normal';
  return tracks === home.tracks || state.awayCars === 'own' ? home.cars : homeCars(tracks);
};
// a race's track: a generated track's seed or a real oval's id, on whichever kind is being watched
function trackFor(ref, tracks) {
  if (tracks !== 'nascar') return Track.random(mulberry32(typeof ref === 'number' ? ref : nameHash(ref)));
  // the browser keeps a few ovals built (a few MB each), not all 32
  if (OvalTrack.byId.size > 6) OvalTrack.byId.delete(OvalTrack.byId.keys().next().value);
  return OvalTrack.get(typeof ref === 'string' ? ref : NASCAR_TRACKS[ref % NASCAR_TRACKS.length].id);
}
// on an oval a race is a distance; on generated tracks a lap count
const RACE_METRES = { practice: 24000, tournament: 24000, saved: 24000, duel: 6000, solo: 6000, race: 6000 };
const lapsOn = (t, kind, laps) => t.nascar ? lapsFor(t.key, RACE_METRES[kind]) : laps;
// numbers from the brains' names (A2·73 runs #73), unique in the field and the same from race to race
function dress(heat) {
  if (!heat.cars.some(car => car.spec.stock)) return;
  const names = [...new Set([...(league?.population.map(a => a.name) ?? []).sort(), ...heat.cars.map(car => car.human ? 'YOU' : car.species.name)])];
  const numbers = assignNumbers(names);
  for (const car of heat.cars) {
    const name = car.human ? 'YOU' : car.species.name;
    car.number = numbers.get(name);
    car.livery = car.human ? { primary: YOU_COLOR, secondary: '#141414', accent: '#141414', pattern: 'checks', sponsor: 'Your Name Here', team: 'You' } : liveryFor(name, car.species.design);
  }
}
// four shades per design so siblings are tellable apart
const proStyle = (name, layers, k, extra = {}) => {
  const design = designOf(layers), hue = DESIGN_HUE[design] ?? 0;
  return { name, layers, design, genes: geneCount(layers), color: `hsl(${hue + (k % 4 - 1.5) * 9} 85% ${56 + (k % 2) * 12}%)`, ...extra };
};

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
  panels = {
    brain: fitCanvas($('#brain')), practice: fitCanvas($('#chart-practice')), vs: fitCanvas($('#chart-vs')), laps: fitCanvas($('#chart-laps')),
    designs: fitCanvas($('#chart-designs')), aero: fitCanvas($('#chart-aero')),
  };
  drawLeague();
  drawBoard();
}

const activeHeat = () => race ? race.heat : pros ? pros.heat : sim.heats[sim.featured];
// each species gets its own family of shades, so A and B are tellable apart at a glance
const colorOf = car => car.human ? YOU_COLOR : car.pro ? car.species.color : `hsl(${car.species.hue + (car.slot % 4 - 1.5) * 9} 85% ${54 + (car.slot % 3) * 9}%)`;
const nameOf = car => car.human ? 'YOU' : `${car.number != null ? `#${car.number} ` : ''}${car.pro ? car.species.name : race ? `AI ${car.rank}·${car.species.name}` : `${car.species.name}${car.slot + 1}`}`;
const describe = sp => `${sp.design ? `design ${sp.design}, ` : ''}${sp.layers.length - 2} hidden layers (${sp.layers.slice(1, -1).join('-')})`;
const damageText = car => {
  const { front, rear, drag } = car.condition, pct = v => Math.round(v * 100);
  return car.wrecked ? 'WRECKED' : `${pct(front)}·${pct(rear)}% +${pct(drag)}%`;
};

function leaderOf(heat) {
  let lead = null;
  for (const car of heat.cars) if (car.running && (!lead || car.progress + car.bonus > lead.progress + lead.bonus)) lead = car;
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

// a car that jumped further than this in one step was moved (a new race, a tow, a restart lining up): no blending
const BLEND_JUMP = 60;
function render() {
  const heat = sketch ? null : activeHeat();
  if (!heat) return drawScene(null);
  const rc = heat.control, pace = rc?.paceCar, mix = (a, b) => a + (b - a) * blend;
  for (const car of heat.cars) {
    car.simX = car.x;
    car.simY = car.y;
    car.simAngle = car.angle;
    if (car.wasX === undefined || Math.abs(car.x - car.wasX) + Math.abs(car.y - car.wasY) > BLEND_JUMP) continue;
    car.x = mix(car.wasX, car.x);
    car.y = mix(car.wasY, car.y);
    car.angle = car.wasAngle + Math.atan2(Math.sin(car.angle - car.wasAngle), Math.cos(car.angle - car.wasAngle)) * blend;
  }
  const was = rc?.wasPace;
  if (pace && was && Math.abs(pace.x - was.x) + Math.abs(pace.y - was.y) < BLEND_JUMP)
    rc.paceCar = { x: mix(was.x, pace.x), y: mix(was.y, pace.y), angle: was.angle + Math.atan2(Math.sin(pace.angle - was.angle), Math.cos(pace.angle - was.angle)) * blend };
  try {
    drawScene(heat);
  } finally {
    for (const car of heat.cars) {
      car.x = car.simX;
      car.y = car.simY;
      car.angle = car.simAngle;
    }
    if (rc) rc.paceCar = pace;
  }
}

// the brain view changes with each decision (30 times a second at 1x), so it's redrawn at about that rate
let brainDrawn = { at: 0, car: null }, brainTitle = '';
function drawScene(heat) {
  const focus = heat && pickFocus(heat);
  if (focus) lastFocus = focus;
  moveCamera(focus, false);
  drawGround(ctx);
  if (sketch) return drawSketch();

  drawTrack(ctx, track);
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
  if (heat.control?.paceCar) drawPaceCar(ctx, heat.control.paceCar);
  drawParticles(ctx);
  const thinker = race ? nearestRival(heat, race.you) : focus;
  if (thinker && state.showRays) drawRays(ctx, thinker, track);
  for (const car of heat.cars)
    if (car.running && (view.mode === 'follow' || car === focus))
      drawLabel(ctx, car, nameOf(car), car === focus || car.human ? '#fff' : 'rgba(220, 230, 245, 0.5)');
  if (view.mode === 'follow') drawMinimap(ctx, track, heat.cars, focus, colorOf);

  // a held neuron stays with the car it was picked on: let go when the brain view moves to another
  if (pickedNeuron && pickedNeuron.brain !== thinker?.brain) dropNeuron();
  const now = performance.now();
  if (thinker && (now - brainDrawn.at >= 30 || brainDrawn.car !== thinker)) {
    brainDrawn = { at: now, car: thinker };
    const marks = new Map();
    if (xrayShown?.car === thinker) {
      xrayShown.steerBy.forEach(e => marks.set(e.i, '#38e1ff'));
      xrayShown.gasBy.forEach(e => marks.has(e.i) || marks.set(e.i, e.gas > 0 ? '#4dff9a' : '#ff4d4d'));
    }
    drawBrain(panels.brain.ctx, panels.brain.w, panels.brain.h, thinker, marks, pickedNeuron);
    const sp = thinker.species, lineage = sp.parent ? ` · from ${sp.parent}, gen ${sp.born}` : sp.frozen ? ' · original, frozen' : '';
    const title = `${race ? `${nameOf(thinker)}, nearest rival` : `Car ${nameOf(thinker)}`} · ${describe(sp)}${lineage}`;
    if (title !== brainTitle) $('#brain-title').textContent = brainTitle = title;
  }
}

// ---------- the evolution panel ----------

function drawLeague() {
  if (!panels || !league) return;
  const { history } = league, designs = Object.keys(DESIGN_HUE).filter(d => history.some(h => h.species[d]));
  const series = fn => designs.map(d => ({ color: designColor(d), points: history.filter(h => h.species[d]).map(h => [h.gen, fn(h.species[d], h)]) }));
  const pct = v => `${Math.round(v * 100)}%`;
  // winning is what counts: each design's share of the generation's tournament races
  const winShare = (s, h) => s.wins / tourneyRaces(h);
  drawChart(panels.designs.ctx, panels.designs.w, panels.designs.h, {
    series: series(winShare), yMin: 0, yMax: Math.max(0.5, ...series(winShare).flatMap(s => s.points.map(p => p[1]))), format: pct, empty: 'waiting for the first tournament…',
  });
  // the rating ladder: each design's best, every few generations, against frozen champions on fixed tracks;
  // the shading is the 95% range
  const best = rating?.slot === league.slot ? rating.players.filter(p => p.role === 'best') : [];
  const rated = designs.map(d => {
    const pts = best.filter(p => p.design === d).sort((a, b) => a.gen - b.gen);
    return { color: designColor(d), bandColor: `hsla(${DESIGN_HUE[d]}, 85%, 64%, 0.14)`, points: pts.map(p => [p.gen, p.r]), band: pts.map(p => [p.gen, p.r - 1.96 * p.sd, p.r + 1.96 * p.sd]) };
  });
  const values = best.flatMap(p => [p.r - 1.96 * p.sd, p.r + 1.96 * p.sd]);
  drawChart(panels.vs.ctx, panels.vs.w, panels.vs.h, {
    series: rated, yMin: Math.floor(Math.min(900, ...values) / 100) * 100, yMax: Math.ceil(Math.max(1100, ...values) / 100) * 100,
    format: v => `${Math.round(v)}`, ref: { value: 1000, label: 'generation 0' }, empty: 'the first rating comes with the next rating generation…',
  });

  // a design's lap is its best member's average fastest lap; null until one of them completes a lap
  const laps = designs.map(d => ({ color: designColor(d), points: history.filter(h => h.species[d]?.lap != null).map(h => [h.gen, h.species[d].lap]) }));
  const lapValues = laps.flatMap(s => s.points.map(p => p[1]));
  drawChart(panels.laps.ctx, panels.laps.w, panels.laps.h, {
    series: laps, yMin: Math.floor(Math.min(...lapValues) * 0.95) || 0, yMax: Math.ceil(Math.max(...lapValues) * 1.05) || 1,
    format: v => `${v.toFixed(0)}s`, empty: 'nobody has finished a lap yet',
  });
  const aero = series(s => s.aero / 2);
  drawChart(panels.aero.ctx, panels.aero.w, panels.aero.h, {
    series: aero, yMin: 0, yMax: Math.max(0.2, ...aero.flatMap(s => s.points.map(p => p[1]))), format: pct, empty: 'waiting for the first tournament…',
  });

  const last = history.at(-1), champ = league.population.find(a => a.name === last.champion);
  const top = best.length ? best.reduce((x, y) => y.gen > x.gen || (y.gen === x.gen && y.r > x.r) ? y : x) : null;
  $('#stat-champion').innerHTML = champ ? `<span style="color:${champ.style.color}">${champ.name}</span><small>design ${champ.species}, gen ${last.gen}${top ? ` · rated ${top.r} ±${Math.round(1.96 * top.sd)}` : ''}</small>` : '—';
  const fastest = designs.filter(d => last.species[d].lap != null).sort((x, y) => last.species[x].lap - last.species[y].lap)[0];
  $('#stat-lap').innerHTML = fastest ? `${last.species[fastest].lap.toFixed(2)}s<small style="color:${designColor(fastest)}">design ${fastest}, gen ${last.gen}</small>` : '—<small>no laps yet</small>';
  $('#stat-gen-time').textContent = last.minutes ? `${last.minutes.toFixed(1)} min` : '—';
  const rows = designs.map(d => ({ d, s: last.species[d], members: league.population.filter(a => a.species === d) })).sort((x, y) => y.s.wins - x.s.wins);
  $('#design-table').innerHTML = `<tr><th>Design</th><th>Brain</th><th title="Share of this generation's tournament races won">Wins</th><th title="Share of the field beaten">Beats</th><th>Best</th><th>Aero lost</th><th title="Share of the race spent glued to the bumper of the car ahead">Tail</th><th title="Share of the race spent door to door in contact with another car">Rub</th></tr>` + rows.map(({ d, s, members }) => {
    const best = members.slice().sort(byStrength)[0];
    return `<tr><td><b style="color:${designColor(d)}">${d}</b></td><td>${members[0].layers.slice(1, -1).join('-')} <small>${(geneCount(members[0].layers) / 1000).toFixed(1)}k</small></td>
      <td>${Math.round(100 * s.wins / tourneyRaces(last))}%</td><td>${Math.round((1 - s.avgPlace) * 100)}%</td><td style="color:${best.style.color}">${best.name}</td><td>${Math.round(s.aero * 50)}%</td><td>${s.tail != null ? `${Math.round(s.tail * 100)}%` : '—'}</td><td>${s.rub != null ? `${Math.round(s.rub * 100)}%` : '—'}</td></tr>`;
  }).join('');
}

// ---------- the practice board: every brain's practice this round, and the learning curve ----------

const count = v => Math.round(v).toLocaleString();
function drawBoard() {
  if (!panels) return;
  const p = engine?.running ? progress : null, b = p?.practice?.board, runs = p?.runs;
  $('#pb-summary').textContent = !p ? 'learning is paused'
    : p.phase === 'tournament' || p.phase === 'rating' ? `generation ${p.generation} · ${p.phase === 'rating' ? 'rating ladder: the best of each design against frozen champions' : 'tournament'}${runs ? ` · ${count(runs.total)} practice runs so far` : ''}`
    : runs ? `gen ${p.generation} · round ${p.round}/${p.rounds} · ${count(runs.round)} of ${count(runs.roundTotal)} runs · ${count(runs.perSecond)}/s · ${count(runs.total)} total` : 'starting…';

  const rounds = p?.rounds ?? 5, now = !p ? 0 : p.phase === 'tournament' || p.phase === 'rating' ? rounds + 1 : p.round;
  $('#pb-rounds').innerHTML = [...Array(rounds + 1)].map((_, i) =>
    `<i class="${i + 1 < now ? 'done' : i + 1 === now ? 'now' : ''}">${i < rounds ? `R${i + 1}` : 'T'}</i>`).join('');

  // the seats never move (a replaced brain's copy takes its seat), so seat i lines up across rounds
  const seats = league?.population ?? [], hist = p?.practiceHistory ?? [];
  const last = hist.at(-1), prev = last && last.gen === p?.generation && last.round === p?.round ? hist.at(-2) : last;
  const watching = pros?.plan?.learner ?? state.picked;
  const byDesign = Object.keys(DESIGN_HUE).map(d => seats.map((a, i) => ({ a, i })).filter(x => x.a.species === d));
  const rows = Math.max(0, ...byDesign.map(col => col.length));
  const cells = [];
  for (let r = 0; r < rows; r++) for (const col of byDesign) {
    const seat = col[r];
    if (!seat) { cells.push('<div></div>'); continue; }
    const name = p?.practice?.pros?.[seat.i]?.name ?? seat.a.name, done = b?.done[seat.i] ?? 0, mean = b?.mean[seat.i], before = prev?.mean[seat.i];
    const delta = mean != null && before != null ? mean - before : null;
    const trend = delta == null || Math.abs(delta) < 0.005 ? '' : `<span class="${delta > 0 ? 'up' : 'down'}">${delta > 0 ? '▲' : '▼'}${Math.abs(delta).toFixed(2)}</span>`;
    cells.push(`<div class="pb-cell ${name === watching ? 'watching' : ''}" data-name="${esc(name)}" title="${esc(name)}: ${done} of ${b?.copies ?? '—'} copies raced this round${mean != null ? `, average reward ${mean.toFixed(2)}` : ''}. Click to watch it practise.">
      <b style="color:${seat.a.style.color}">${esc(name)}</b><div class="bar"><i style="width:${b ? Math.round(done / b.copies * 100) : 0}%"></i></div>${mean != null ? mean.toFixed(2) : '—'} ${trend}</div>`);
  }
  $('#pb-grid').innerHTML = cells.join('');

  // the learning curve: each design's average practice reward, one point per round
  const designs = Object.keys(DESIGN_HUE).filter(d => seats.some(a => a.species === d));
  const series = designs.map(d => ({ color: designColor(d), points: hist.flatMap(h => {
    const vals = seats.map((a, i) => a.species === d ? h.mean[i] : null).filter(v => v != null);
    return vals.length ? [[h.gen + (h.round - 1) / rounds, vals.reduce((x, y) => x + y, 0) / vals.length]] : [];
  }) }));
  const values = series.flatMap(s => s.points.map(pt => pt[1]));
  drawChart(panels.practice.ctx, panels.practice.w, panels.practice.h, {
    series, yMin: Math.min(0, ...values), yMax: Math.max(0.5, ...values) * 1.05, format: v => v.toFixed(1), empty: 'the first practice round is still running…',
  });
}

function showEngine() {
  const running = engine?.running, training = engine?.active;
  $('#engine-dot').classList.toggle('on', !!running);
  const phase = progress?.phase === 'tournament' ? 'tournament' : progress?.phase === 'rating' ? 'rating ladder' : progress?.round ? `training round ${progress.round}/${progress.rounds}` : 'starting';
  const where = engine?.cloud
    ? ` on <b>Modal</b> (${progress?.workers ?? '…'} threads on ${progress?.machines ?? 1 + (engine.cloud.helpers ?? 0)} machines, ≈$${(engine.cloud.perHour ?? 3.15).toFixed(2)}/h, stops ${new Date(engine.cloud.until).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })})`
    : ` on <b>${progress?.workers ?? engine?.workers} cores</b>`;
  $('#engine-text').innerHTML = running
    ? `Training <b>${esc(slotName(training))}</b>${where} · generation ${progress?.generation ?? '…'} · ${phase}`
    : engine ? `Engine stopped${training ? ` · ${esc(slotName(training))} is saved at generation ${engine.slots.find(s => s.id === training)?.generation ?? 0}` : ''}` : `Engine offline · run <b>node server.js</b>`;
  const btn = $('#btn-engine');
  btn.hidden = !engine?.active;
  btn.textContent = running ? 'Stop' : 'Start';
  const done = !running || !progress?.rounds ? 0 : progress.phase === 'tournament' || progress.phase === 'rating' ? 0.97 : ((progress.round - 1) / progress.rounds) * 0.95;
  $('#engine-bar').style.width = `${Math.round(done * 100)}%`;
}

const ago = iso => {
  const min = (Date.now() - new Date(iso)) / 60000;
  return min < 1 ? 'just now' : min < 60 ? `${Math.round(min)} min ago` : min < 1440 ? `${Math.round(min / 60)} h ago` : `${Math.round(min / 1440)} d ago`;
};

let slotsHtml = '';
function drawSlots() {
  if (!engine?.slots) return;
  const { slots, active, running, maxSlots, cloud } = engine;
  $('#slot-count').textContent = `${slots.length} of ${maxSlots} saves used`;
  $('#btn-slot-scratch').disabled = slots.length >= maxSlots;
  $('#watch-name').textContent = active ? slotName(active) : '';
  if (armedDelete && Date.now() > armedDelete.until) armedDelete = null;
  const html = slots.map(s => {
    const loaded = s.id === active, armed = armedDelete?.id === s.id, { tracks, cars } = modeOf(s.id), other = tracks === 'nascar' ? 'normal' : 'NASCAR';
    const gen = s.generation != null ? `generation ${s.generation} · champion ${esc(s.champion)}` : 'starting…';
    const status = !loaded ? `saved ${ago(s.updated)}` : cloud ? 'loaded · learning on Modal' : running ? 'loaded · learning' : 'loaded · paused';
    const badge = cars === homeCars(tracks) ? `<span class="mode ${tracks === 'nascar' ? 'nascar' : ''}">${tracks === 'nascar' ? 'NASCAR' : 'normal'}</span>`
      : `<span class="mode mixed">${tracks === 'nascar' ? 'NASCAR ovals · normal cars' : 'normal tracks · stock cars'}</span>`;
    return `<div class="slot ${loaded ? 'loaded' : ''}" data-id="${s.id}" title="${loaded ? 'This save is loaded' : 'Click to load this save'}">
      <i class="${loaded && running ? 'on' : ''}"></i>
      <div><b>${esc(s.name)}${badge}</b><small>${gen} · ${status}</small></div>
      <button class="small icon" data-act="convert" ${s.generation == null ? 'disabled title="Wait for its first generation"' : `title="Copy these brains into ${other} mode"`}>⇄</button>
      <button class="small icon ${armed ? 'armed' : ''}" data-act="delete" ${loaded ? 'disabled title="Load another save first"' : 'title="Delete this save"'}>${armed ? 'Delete?' : '✕'}</button>
    </div>`;
  }).join('');
  // only rebuild when something changed, so a click never lands on a row that's being replaced
  if (html !== slotsHtml) $('#slot-list').innerHTML = slotsHtml = html;
  const home = homeMode(), otherTracks = home.tracks === 'nascar' ? 'normal' : 'nascar';
  $('#watch-other').textContent = otherTracks === 'nascar' ? 'NASCAR ovals' : 'Normal tracks';
  $('#watch-host').textContent = `Drive the ${homeCars(otherTracks) === 'stock' ? 'stock' : 'normal'} cars`;
  $('#watch-cars').hidden = !state.away;
  document.querySelectorAll('#watch-tracks button').forEach(b => b.classList.toggle('on', !!b.dataset.tracks === !!state.away));
  document.querySelectorAll('#watch-cars button').forEach(b => b.classList.toggle('on', b.dataset.cars === state.awayCars));
}

function adoptLeague(data, slot) {
  const switched = league && league.slot !== slot;
  data.population.forEach((a, k) => {
    a.weights = Float32Array.from(a.genes);
    a.style = proStyle(a.name, a.layers, k, { parent: a.parent, born: a.born, label: a.label });
  });
  const before = league?.generation;
  league = Object.assign(data, { slot });
  drawLeague();
  showMode();
  refreshOfficial(slot, league.generation);
  // the ladder's latest fit (rating.json is rewritten every rating generation)
  getJson(slotUrl(slot, 'rating.json')).then(r => {
    rating = r && { ...r, slot };
    drawLeague();
  });
  getJson(slotUrl(slot, 'neurons.json')).then(atlas => neuronAtlas = atlas);
  if (switched) {
    state.away = null;
    toast(`Loaded <b>${esc(slotName(slot))}</b>, generation ${league.generation}.`);
    if (state.mode === 'pros' && !state.playing) startPros();
    return;
  }
  if (before == null || before === league.generation) return;
  const last = league.history.at(-1), champ = league.population.find(a => a.name === last.champion);
  const swaps = last.replaced.map(r => `<b>${r.out}</b> → <b>${r.by}</b>`).join(', ');
  toast(`<b>Generation ${last.gen}</b> is on the grid next race. Champion <b style="color:${champ.style.color}">${champ.name}</b> (design ${champ.species}).${swaps ? ` Replaced: ${swaps}.` : ''}`, 'gold');
}

const getJson = url => fetch(url, { cache: 'no-store' }).then(r => r.ok ? r.json() : null).catch(() => null);

// refreshed once per round, when the engine moves on
async function refreshLive() {
  if (!engine?.running || !progress?.phase) return;
  const key = `${engine.active}:${progress.generation}:${progress.round}:${progress.phase}`;
  if (live?.key === key) return;
  const data = await getJson(slotUrl(engine.active, 'live.json'));
  if (!data || data.generation !== progress.generation) return;
  // past champions from the hall of fame race in practice too
  const everyone = [...data.population, ...(data.hall ?? [])];
  everyone.forEach((a, k) => {
    a.weights = Float32Array.from(a.genes);
    a.style = proStyle(a.name, a.layers, k, { parent: a.parent, born: a.born, label: a.label, frozen: k >= data.population.length });
  });
  live = { ...data, key, progress, byName: new Map(everyone.map(a => [a.name, a])) };
}

// the engine writes progress.json every round and state.json + summary.json every generation
async function poll() {
  engine = await getJson('api/engine');
  if (!engine) return showEngine();
  progress = engine.active && await getJson(slotUrl(engine.active, 'progress.json'));
  await refreshLive();
  const loaded = engine.slots.find(s => s.id === engine.active);
  if (loaded?.generation != null && (league?.slot !== loaded.id || loaded.generation > league.generation)) {
    const data = await getJson(slotUrl(loaded.id, 'state.json'));
    if (data) {
      adoptLeague(data, loaded.id);
      if (!$('#intro').hidden && !pros) startPros(false);
    }
  }
  showEngine();
  drawBoard();
  drawSlots();
  showMode();
}

// NASCAR: no drawing your own track (they're all real ovals); New track goes round them in turn
function showMode() {
  const nascar = watchedTracks() === 'nascar';
  $('#btn-draw').hidden = nascar;
  $('#btn-track').textContent = nascar ? 'Next oval' : 'New track';
  $('#btn-track').title = nascar ? 'The 32 real ovals in turn (N)' : 'N';
}

async function slotAction(params) {
  const res = await fetch(`api/slots?${new URLSearchParams(params)}`, { method: 'POST' }).then(r => r.json()).catch(() => ({ error: 'The server is not responding.' }));
  if (res.error) toast(esc(res.error));
  await poll();
  return res;
}

function towerGap(heat, car, standings) {
  const first = standings[0];
  if (car.retired) return 'OUT';
  if (car.finished) return car === first ? 'WIN' : `+${((car.steps - first.steps) / 60).toFixed(1)}`;
  if (car === first) return 'LEAD';
  return `+${heat.gap(car).toFixed(2)}`;
}

// Tower rows are built once and updated in place: rebuilding them several times a second swallowed
// clicks whenever the press and release landed on different copies of the row.
const towerRows = [];
function towerRow(i) {
  if (!towerRows[i]) {
    const row = document.createElement('li');
    row.innerHTML = '<b></b><i></i><span></span><em></em>';
    $('#tower-list').append(row);
    towerRows[i] = row;
  }
  return towerRows[i];
}

function updateHud() {
  const heat = activeHeat(), standings = heat.standings(), focus = race ? race.you : pickFocus(heat);
  const lead = standings[0];
  const plan = pros?.plan;
  const round = `GEN ${plan?.gen} · ROUND ${plan?.round}/${plan?.rounds}`, of = `GEN ${plan?.gen} · RACE ${plan?.k + 1}/${plan?.count}`;
  const head = race ? 'RACE' : !pros ? 'WAITING FOR GENERATION 0' : state.playing ? `PLAYING · GEN ${pros.gen}` : {
    practice: `PRACTICE · ${round}`, duel: `DUEL · ${round}`, solo: `TIME TRIAL · ${round}`,
    tournament: `TOURNAMENT · ${of}`, rating: `RATING · ${of}`, saved: `GEN ${plan?.gen} · PAUSED`,
  }[plan?.kind] ?? `GEN ${pros.gen}`;
  showPhase();
  const control = heat.control, total = lead.raceLaps, done = lead.laps.length + lead.freeLaps - lead.yellowLaps;
  $('#tower-head').textContent = `${head} · LAP ${Math.min(total, done + 1)}/${total}${control?.overtime ? ' · OT' : ''}${control?.flag === 'yellow' ? ' · CAUTION' : ''}`;
  showFlags(heat, focus);
  $('#tower').classList.toggle('dense', standings.length > 24);
  standings.forEach((car, i) => {
    const row = towerRow(i), [pos, chip, name, gap] = row.children;
    row.dataset.slot = car.slot;
    row.className = `${car === focus ? 'focus' : ''} ${car.retired ? 'out' : ''}`;
    pos.textContent = i + 1;
    chip.style.background = colorOf(car);
    name.style.color = car.species?.color ?? '';
    name.innerHTML = `${nameOf(car)}${car.elite ? ' <span class="star">★</span>' : ''}`;
    gap.textContent = towerGap(heat, car, standings);
  });
  towerRows.forEach((row, i) => row.hidden = i >= standings.length);

  if (race) {
    const you = race.you;
    $('#race-pos').textContent = `P${standings.indexOf(you) + 1}/${heat.cars.length}`;
    $('#race-lap').textContent = `${Math.min(heat.laps, you.laps.length + you.freeLaps - you.yellowLaps + 1)}/${heat.laps}`;
    $('#race-time').textContent = (you.steps / 60).toFixed(2);
    $('#race-dmg').textContent = damageText(you);
  }

  const thinker = race ? nearestRival(heat, race.you) : focus;
  if (thinker?.brain) {
    xrayShown = { car: thinker, ...xray(thinker) };
    $('#xray').innerHTML = xrayHtml(xrayShown);
  }
  if (pickedNeuron) $('#neuron-live').textContent = pickedNeuron.brain.acts[pickedNeuron.l][pickedNeuron.j].toFixed(2);

  const car = race ? race.you : focus;
  if (car) $('#car-stats').innerHTML = [
    ['speed', car.spec.stock ? `${Math.round(Math.max(0, car.forwardSpeed) * MPH_PER_SPEED)} mph` : `${Math.round(Math.max(0, car.forwardSpeed) * KMH)} km/h`],
    ['slipstream', `${Math.round(car.draft * 100)}%`],
    ['aero lost F·R · drag', damageText(car)],
    ['wall hits', car.wallHits],
    ...track.nascar ? [
      ['banking · surface', `${Math.round(track.bankAt(car.lastArc, track.lateralAt(car.x, car.y)) * 180 / Math.PI)}° · ${SURFACE_NAME[car.surface]}`],
      ['flags', car.parked ? 'parked' : car.penalty ? 'black: stop-and-go' : car.blueFlag ? 'blue-yellow' : car.paced ? 'following the pace car' : car.puncture ? 'slow puncture' : '—'],
    ] : [],
  ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('');
}

const SURFACE_NAME = ['asphalt', 'apron', 'grass', 'sand', 'pit road', 'off track'];
// the flag stand: the race's flag and the latest word from race control; the big calls also pop up as toasts
let lastCall = null;
function showFlags(heat, focus) {
  const control = heat.control, stand = $('#flagstand');
  stand.hidden = !control || !!sketch;
  if (stand.hidden) return;
  const flag = focus?.penalty ? 'black' : control.flag, [color, label] = FLAG_STYLE[flag] ?? FLAG_STYLE.green;
  const said = control.log.at(-1), swatch = stand.querySelector('i');
  swatch.className = color === 'checkered' ? 'checkered' : '';
  swatch.style.background = color === 'checkered' ? '' : color;
  stand.querySelector('b').textContent = control.overtime && flag !== 'checkered' ? `${label} · OVERTIME` : label;
  stand.querySelector('b').style.color = flag === 'black' ? '#f5f5f5' : color === 'checkered' ? '#f5f5f5' : color;
  stand.querySelector('small').textContent = focus?.penalty ? `${nameOf(focus)}: stop-and-go penalty` : said?.text ?? '';
  if (said && said !== lastCall) {
    const first = lastCall === null || !control.log.includes(lastCall);
    lastCall = said;
    if (!first && said.flag !== 'green' && said.flag !== 'white') toast(`<b>${esc(said.text)}</b>`, said.flag === 'checkered' ? 'gold' : '');
  }
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
}

function setCamera(mode) {
  view.mode = mode;
  $('#btn-camera').textContent = `Camera: ${mode === 'follow' ? 'follow' : 'full track'}`;
}

function begin() {
  if (!league) return toast('The first generation is still being set up. Give it a few seconds.');
  // a fresh race, so it shows what the engine is doing now rather than the intro's backdrop
  if (!state.playing || !pros) return startPros();
  $('#intro').hidden = true;
  setPaused(false);
}

function setTrack(next, message) {
  track = next;
  if (state.mode === 'train') sim.setTrack(next);
  lastFocus = state.focus = null;
  layout();
  toast(message);
  if (pros) startPros(false);
}

// ---------- race mode ----------

// your rivals: the top 9 of the watched save's latest tournament (19 on an oval: it's pack racing)
const raceField = () => league?.population.slice().sort(byStrength).slice(0, (track.nascar ? 2 * RACE_FIELD : RACE_FIELD) - 1).map(a => ({ species: a.style, genes: a.weights }));

function startRace() {
  const drivers = raceField();
  if (!drivers) return toast('The first generation is still being set up. Give it a few seconds.');
  $('#intro').hidden = $('#results').hidden = true;
  const you = track.nascar ? 2 * HUMAN_SLOT : HUMAN_SLOT;
  drivers.splice(you, 0, null);
  const heat = new Heat(track, drivers.map(d => d && new Brain(d.species.layers, d.genes)), lapsOn(track, 'race', HUMAN_RACE_LAPS),
    { cars: carsFor(track), cautions: league.generation >= RC.cautionsFrom });
  heat.events = [];
  heat.cars.forEach((car, slot) => Object.assign(car, { rank: slot < you ? slot + 1 : slot, species: drivers[slot]?.species, pro: !!drivers[slot]?.species.design }));
  race = { heat, you: heat.cars[you], t: -180, steer: 0, count: null, endAt: Infinity };
  race.you.human = true;
  dress(heat);
  state.mode = 'race';
  stage.classList.add('racing');
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
      ? `The fastest AI lap (${nameOf(quickest)}, a ${quickest.species.genes.toLocaleString()}-weight network) was <b>${Math.round((youLap / aiLap - 1) * 100)}% quicker</b> than your best, learned from nothing in ${league.generation} generations of evolution.`
      : 'You out-lapped every AI. Give the evolution a few more generations and try again.'
    : 'Finish a lap to get a lap time on the board.';
  $('#results').hidden = false;
}

function exitRace() {
  race = null;
  state.mode = pros ? 'pros' : 'train';
  stage.classList.remove('racing');
  banner('');
  $('#results').hidden = $('#race-hud').hidden = true;
  lastFocus = null;
}

// ---------- the races you watch: the watched save's latest generation, back to back ----------

function useTrack(next) {
  track = next;
  lastFocus = null;
  layout();
}

// ---------- channels: what to watch. Every race is one the engine really ran, car for car ----------
// live: whatever the engine is doing right now (practice, then the tournament); practice / duels: this round's
// practice races, each a real copy of the learning brain (its seed rebuilds the exact weights it raced);
// tournament: the latest tournament's official races; rating: the latest rating round's races
const CHANNELS = [
  ['live', 'Live', 'Whatever the engine is running right now'],
  ['practice', 'Practice', "This round's practice races and time trials: real copies of the brain that's learning"],
  ['duels', 'Duels', "This round's two-car duels: attack or defend, only the winner scores"],
  ['tournament', 'Tournament', "The latest tournament's official races, the ones that decide who gets replaced"],
  ['rating', 'Rating', "The latest rating round: each design's best against frozen past champions"],
];
// the latest recorded tournament and rating round (tournament.json, rating-races.json), with the official results
let official = { slot: null, gen: null, tournament: null, rating: null };
async function refreshOfficial(slot, gen) {
  if (official.slot === slot && official.gen === gen) return;
  official = { slot, gen, tournament: null, rating: null };
  const [t, r] = await Promise.all([getJson(slotUrl(slot, 'tournament.json')), getJson(slotUrl(slot, 'rating-races.json'))]);
  if (official.slot !== slot) return;
  const asDriver = (a, k) => ({ weights: Float32Array.from(a.genes), species: proStyle(a.name, a.layers, k, { parent: a.parent, born: a.born, label: a.label }) });
  if (t) official.tournament = { gen: t.gen, races: t.races, byName: new Map(t.population.map((a, k) => [a.name, asDriver(a, k)])) };
  if (r) official.rating = { gen: r.gen, races: r.races, byId: new Map(r.players.map((p, k) => [p.id, asDriver(p, k)])) };
}

// a practice race exactly as one of the learning brain's copies ran it: weights from its seed, rivals by name
function practicePlan(prev, p, which) {
  const roster = p.practice.pros;
  if (!roster[0]?.scenarios) return null;
  const shows = sc => which === 'duels' ? sc.duel : which === 'practice' ? !sc.duel : sc.kind === 'race';
  const lineup = pro => pro.scenarios.map((sc, k) => ({ sc, k })).filter(({ sc }) => shows(sc));
  // the brain you clicked, or each brain in turn through all of its races. A watched race often outlasts a
  // training round (about a minute on Modal), so the turn carries on into the new round rather than starting over.
  const picked = roster.find(x => x.name === state.picked), last = roster.findIndex(x => x.name === prev?.learner);
  const stay = last >= 0 && prev.j + 1 < lineup(roster[last]).length;
  const pro = picked ?? (stay ? roster[last] : roster[last >= 0 ? (last + 1) % roster.length : 0]);
  const races = lineup(pro), j = prev?.learner === pro.name ? (prev.j + 1) % races.length : 0, { sc } = races[j];
  const ai = roster.indexOf(pro), learner = live.byName.get(pro.name), copies = 2 * p.practice.pairs;
  const copy = (pros?.count ?? 0) % copies, sign = copy & 1 ? -1 : 1;
  const me = { weights: perturbGenes(learner.weights, esSeed(p.generation, p.round, ai, copy >> 1), sign * learner.sigma), species: learner.style };
  // a full superspeedway pack seats some brains twice: the second car gets a ′ so it has its own number and paint
  const seated = new Set(), grid = (sc.rivals ?? []).map(name => {
    const { weights, style } = live.byName.get(name), again = seated.has(name);
    seated.add(name);
    return { weights, species: again ? { ...style, name: `${style.name}′` } : style };
  });
  grid.splice(sc.slot ?? 0, 0, me);
  const { rivals, ...scenario } = sc;
  return { kind: sc.kind === 'tt' ? 'solo' : sc.duel ? 'duel' : 'practice', canonical: true, scenario, grid, learner: pro.name, rival: sc.duel ? rivals[0] : null,
    slot: sc.slot ?? 0, j, copy, copies, gen: p.generation, round: p.round, rounds: p.rounds };
}
// one of a recorded set of field races (a tournament or a rating round), the next each time
function fieldPlan(kind, prev, set, driverOf, nameOf) {
  const k = prev?.kind === kind && prev.gen === set.gen ? (prev.k + 1) % set.races.length : 0, race = set.races[k];
  return { kind, canonical: true, scenario: race.scenario, grid: race.grid.map(driverOf), official: race.order?.map(nameOf), k, count: set.races.length, gen: set.gen };
}
const tournamentPlan = (prev, set) => fieldPlan('tournament', prev, set, name => set.byName.get(name), name => name);
const ratingPlan = (prev, set) => fieldPlan('rating', prev, set, id => set.byId.get(id), id => set.byId.get(id).species.name);

function evolvingPlan(next) {
  const p = engine?.running && live?.progress, prev = next ? pros?.plan : null, ch = state.channel, training = p?.phase === 'training' && p.practice;
  // the tournament the engine is running right now: its real grids, with the weights live.json published for it
  const running = p?.phase === 'tournament' && p.tournament?.races && live?.generation === p.generation
    ? { gen: p.generation, races: p.tournament.races, byName: new Map([...live.byName].map(([name, a]) => [name, { weights: a.weights, species: a.style }])) } : null;
  const latest = official.tournament, plan =
    ch === 'live' ? (training && practicePlan(prev, p, 'live')) || (running && tournamentPlan(prev, running)) || (latest && tournamentPlan(prev, latest))
    : ch === 'practice' || ch === 'duels' ? training && practicePlan(prev, p, ch)
    : ch === 'tournament' ? latest && tournamentPlan(prev, latest)
    : official.rating && ratingPlan(prev, official.rating);
  if (plan) return plan;
  // nothing on this channel right now (practice while the engine is judging or paused, no rating round yet)
  const fallback = latest ? tournamentPlan(prev, latest) : savedPlan(prev);
  return { ...fallback, fallback: ch };
}

// a save from before races were recorded: its best on the tracks its tournament was raced on (not a real race)
function savedPlan(prev) {
  const field = homeMode().tracks === 'nascar' ? 40 : 20, entry = a => ({ weights: a.weights, species: a.style });
  const k = prev?.kind === 'saved' && prev.gen === league.generation ? (prev.k + 1) % TOURNEY_TRACKS : 0;
  const top = league.population.slice().sort(byStrength).slice(0, field);
  const trackRef = homeMode().tracks === 'nascar' ? seasonOvals(league.generation)[k] : tourneyTrack(league.generation, k);
  return { kind: 'saved', k, trackRef, laps: homeMode().tracks === 'nascar' ? lapsFor(trackRef, RACE_METRES.saved) : 10, grid: shuffle(top.map(entry)), gen: league.generation };
}

function showPhase() {
  const plan = pros?.plan, chip = $('#phase');
  chip.hidden = !plan || state.playing || !!race || !!sketch;
  if (chip.hidden) return;
  const color = name => pros.heat.cars.find(car => car.species.name === name)?.species.color;
  const at = track.nascar ? ` at ${esc(track.name)}${track.plate ? ', restrictor plates' : ''}` : '';
  const who = name => `<b style="color:${color(name)}">${esc(name)}</b>`, laps = plan.scenario?.laps ?? plan.laps;
  const copy = () => `copy ${(plan.copy >> 1) + 1}${plan.copy & 1 ? '−' : '+'} of ${plan.copies}`;
  const nothing = plan.fallback && { practice: 'No practice is running right now', duels: 'No duels are running right now', rating: 'No rating round recorded yet', live: 'Learning is paused', tournament: 'No tournament recorded yet' }[plan.fallback];
  const main = {
    practice: () => `<b>Practice</b> · generation ${plan.gen}, round ${plan.round} of ${plan.rounds}<small>${who(plan.learner)}, ${copy()}: one of the versions of it the engine raced in this exact race, starting P${plan.slot + 1}. Their results nudge it toward whatever did better. Click any car to follow another brain's practice.</small>`,
    duel: () => `<b>Duel</b> · generation ${plan.gen}, round ${plan.round} of ${plan.rounds}<small>${who(plan.learner)}, ${copy()}, learning to ${plan.slot ? 'attack' : 'defend'}: ${laps} laps against ${who(plan.rival)}, starting ${plan.slot ? 'behind' : 'in front'}. Only the winner scores, so ${plan.slot ? 'it has to get past, by out-braking it or spinning it round' : 'it has to hold on and keep its rear corners covered'}.</small>`,
    solo: () => `<b>Time trial</b> · generation ${plan.gen}, round ${plan.round} of ${plan.rounds}<small>${who(plan.learner)}, ${copy()}, alone against the clock: raw pace and clean laps, no traffic to blame.</small>`,
    tournament: () => `<b>Tournament</b> · generation ${plan.gen}, race ${plan.k + 1} of ${plan.count}<small>${plan.official ? 'The official race' : 'Running on the engine right now'}: the real grid, car for car. No learning here: these races rank everyone, and in each design a brain that's clearly behind is replaced.</small>`,
    rating: () => `<b>Rating</b> · generation ${plan.gen}, race ${plan.k + 1} of ${plan.count}<small>Each design's best against frozen past champions on a fixed track: one of the races the rating chart comes from.</small>`,
    saved: () => `<b>Generation ${plan.gen}</b> · learning is paused<small>The latest saved cars on the tracks their tournament was raced on.</small>`,
  }[plan.kind]().replace('</b>', `</b>${at}`);
  const html = nothing ? main.replace('<small>', `<small>${nothing}, so here's ${plan.kind === 'saved' ? 'an exhibition instead' : 'the latest tournament'}. `) : main;
  if (chip.dataset.html !== html) chip.innerHTML = chip.dataset.html = html;
}

// go: hide the intro and start racing; otherwise just put the field on the grid (intro backdrop, new track)
// next: the following race in a series (evolving moves on to the next practice race or tournament track;
// playing stays on your track at your race length)
function startPros(go = true, next = false) {
  if (!league) return;
  if (race) exitRace();
  if (sketch) stopDrawing();
  const plan = state.playing ? null : evolvingPlan(next);
  // a real race runs exactly as the engine ran it: its track, its options, its grid; watching on the other kind of
  // track (Settings) turns it into an exhibition on that track instead
  const canonical = plan?.canonical && !state.away, sc = plan?.scenario;
  if (canonical) useTrack(sc.trackId ? trackFor(sc.trackId, 'nascar') : Track.random(mulberry32(sc.trackSeed)));
  else if (plan) useTrack(trackFor(plan.trackRef ?? sc.trackId ?? sc.trackSeed, watchedTracks()));
  const best = () => shuffle(league.population.slice().sort(byStrength).slice(0, track.nascar ? 40 : 20).map(a => ({ weights: a.weights, species: a.style })));
  const grid = plan ? plan.grid : best(), brains = grid.map(d => new Brain(d.species.layers, d.weights));
  let heat;
  if (canonical) heat = new Heat(track, brains, sc.laps, scenarioOptions(sc));
  else {
    // on the save's own tracks the engine's lap count; elsewhere the same kind of race at that track's length
    const kind = !plan ? 'tournament' : { solo: 'practice', rating: 'tournament' }[plan.kind] ?? plan.kind;
    const away = !!track.nascar !== (homeMode().tracks === 'nascar');
    const laps = !plan ? lapsOn(track, 'tournament', sim.laps) * (track.nascar ? sim.laps / 10 : 1) : away ? lapsOn(track, plan.kind in RACE_METRES ? plan.kind : kind, { practice: 4, duel: 2 }[kind] ?? 10) : plan.laps ?? sc.laps;
    heat = new Heat(track, brains, Math.max(2, Math.round(laps)), {
      cars: carsFor(track), stages: track.nascar && kind !== 'practice' && kind !== 'duel', practice: kind === 'practice' || kind === 'duel',
      // the same as the engine: green racing until the save's brains are RC.cautionsFrom generations old
      cautions: (plan?.gen ?? league.generation) >= RC.cautionsFrom,
    });
  }
  heat.events = [];
  heat.cars.forEach((car, i) => Object.assign(car, { pro: true, species: grid[i].species }));
  dress(heat);
  pros = { heat, count: (pros?.count ?? 0) + 1, nextAt: null, gen: plan?.gen ?? league.generation, plan, canonical };
  state.mode = 'pros';
  // the camera follows the leader; a car clicked during the race takes over until it drops out or the race ends
  state.focus = lastFocus = null;
  showPhase();
  if (!go) return;
  $('#intro').hidden = true;
  setPaused(false);
}

// playing: the loaded save's cars on your track; the engine keeps learning on its own tracks
function play() {
  state.playing = true;
  $('#play-save').textContent = slotName(engine?.active);
  $('#playbar').hidden = false;
  showPhase();
}

function backToEvolving() {
  state.playing = false;
  $('#playbar').hidden = true;
  if (sketch) stopDrawing();
  startPros();
}

function prosTick() {
  if (pros.nextAt) return;
  pros.heat.tick();
  if (!pros.heat.over) return;
  const [winner, second] = pros.heat.standings();
  const margin = second?.finished ? ` by ${((second.steps - winner.steps) / 60).toFixed(2)}s` : '';
  // early on nobody reaches the flag; the car that got furthest still deserves a mention
  const plan = pros.plan, which = state.playing ? `your race` : { practice: 'this practice race', duel: 'this duel', solo: 'this time trial', tournament: `tournament race ${plan.k + 1}`, rating: `rating race ${plan.k + 1}` }[plan.kind] ?? `generation ${pros.gen}, race ${pros.count}`;
  // a recorded race replayed exactly finishes exactly as it did on the engine
  const order = pros.heat.standings().map(car => car.species.name);
  const check = pros.canonical && plan.official ? (order.join() === plan.official.join() ? ' · ✓ the official result' : ' · ✗ not the official result') : '';
  toast(`<b style="color:${winner.species.color}">${winner.species.name}</b> (design ${winner.species.design}) ${winner.finished ? 'wins' : 'gets furthest in'} ${which}${margin}${check}`, 'gold');
  pros.nextAt = performance.now() + 4000;
}

function setChannel(id) {
  state.channel = id;
  document.querySelectorAll('#channels button').forEach(b => b.classList.toggle('on', b.dataset.ch === id));
  if (!league || race || sketch) return;
  if (state.playing) return backToEvolving();
  startPros(!state.paused);
}

// ---------- draw-your-own-track mode ----------

function startDrawing() {
  if (race) exitRace();
  $('#intro').hidden = true;
  state.mode = 'draw';
  sketch = { points: [], active: false };
  cameraBeforeSketch = view.mode;
  setCamera('full');
  stage.classList.add('drawing');
  $('#tower').hidden = true;
  banner('Draw a loop', 'one continuous stroke · long straights and tight corners make the best racing · Esc to cancel', true);
}

function stopDrawing() {
  sketch = null;
  state.mode = pros ? 'pros' : 'train';
  setCamera(cameraBeforeSketch);
  stage.classList.remove('drawing');
  $('#tower').hidden = false;
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
    state.picked = pick?.species?.name ?? null;
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
  play();
  setTrack(drawn, 'Your track. <b>None of these brains have ever seen it.</b>');
  if (state.paused) begin();
});

// a brain on the practice board: watch its practice next
$('#pb-grid').addEventListener('pointerdown', e => {
  const name = e.target.closest('.pb-cell')?.dataset.name;
  if (!name) return;
  state.picked = name;
  if (state.playing) return toast(`<b>${esc(name)}</b> is picked; press <b>Back to evolving</b> to watch it practise.`);
  if (pros?.plan?.learner) startPros();
  drawBoard();
});

// on press, not click: the standings reshuffle constantly, so take whichever car is under the pointer now
$('#tower-list').addEventListener('pointerdown', e => {
  const row = e.target.closest('li');
  if (!row || race) return;
  state.focus = activeHeat().cars[+row.dataset.slot];
  state.picked = state.focus?.species?.name ?? null;
  updateHud();
});

// ---------- the neuron lab: click a neuron in the brain view to see what it does, and force it ----------

const thinkerNow = () => { const heat = activeHeat(); return race ? nearestRival(heat, race.you) : pickFocus(heat); };
// what train/neurons.js found about this neuron, if it analysed this brain (same name, same shape)
function neuronFacts(name, brain, l, j) {
  const entry = neuronAtlas?.brains?.[name];
  if (entry?.layers.join() !== brain.layers.join()) return null;
  return { entry, nr: entry.neurons.find(x => x.l === l && x.j === j) };
}

function pickNeuron(car, l, j) {
  dropNeuron();
  pickedNeuron = { brain: car.brain, name: car.species?.name, l, j };
  showNeuron();
}

function dropNeuron() {
  if (pickedNeuron) pickedNeuron.brain.held = null;
  pickedNeuron = null;
  $('#neuron').hidden = true;
}

function showNeuron() {
  const { brain, name, l, j } = pickedNeuron, facts = neuronFacts(name, brain, l, j), held = brain.held?.[0]?.[2];
  const pct = v => `${Math.round(v * 100)}%`;
  const hands = ([s, g]) => `steer ${s > 0 ? '▶' : '◀'}${Math.abs(s).toFixed(2)} gas ${g >= 0 ? '+' : '−'}${Math.abs(g).toFixed(2)}`;
  $('#neuron').hidden = false;
  $('#neuron-name').textContent = `${name} · layer ${l} of ${brain.layers.length - 2} · neuron ${j + 1}`;
  if (facts?.nr) {
    const { entry, nr } = facts, base = entry.intact, cut = nr.lesion;
    $('#neuron-label').textContent = nr.label;
    $('#neuron-facts').innerHTML = [
      nr.concepts.length && `<em>reacts to</em> ${nr.concepts.map(([c, d]) => `${esc(c)} ${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(1)}σ`).join(', ')}`,
      nr.inputs.length && `<em>follows</em> ${nr.inputs.map(([i, r]) => `${esc(neuronAtlas.inputs[i])} ${r.toFixed(2)}`).join(', ')}`,
      `<em>forced on</em> ${hands(nr.plus)} <em>· off</em> ${hands(nr.minus)}`,
      cut && `<em>silenced for ${neuronAtlas.races} races</em> place ${base.place.toFixed(2)}→${cut.place.toFixed(2)} · wins ${base.wins}→${cut.wins} · finished ${pct(base.finished)}→${pct(cut.finished)}${nr.placebo ? ' (placebo)' : ''}`,
      `<em>analysed at generation ${entry.analysedAt}</em>`,
    ].filter(Boolean).join('<br>');
  } else {
    $('#neuron-label').textContent = 'Not analysed yet';
    $('#neuron-facts').innerHTML = `<em>node train/neurons.js ${esc(league?.slot ?? '')} ${esc(name)}</em> finds out what it does. You can still force it and watch.`;
    // the analysis may have finished since the save loaded
    const asked = pickedNeuron;
    if (league && !asked.refetched) getJson(slotUrl(league.slot, 'neurons.json')).then(atlas => {
      asked.refetched = true;
      if (!atlas) return;
      neuronAtlas = atlas;
      if (pickedNeuron === asked && neuronFacts(name, brain, l, j)) showNeuron();
    });
  }
  document.querySelectorAll('.neuron-hold button').forEach(b => b.classList.toggle('on', held !== undefined && b.dataset.hold === (held === 1 ? '1' : held === -1 ? '-1' : 'avg')));
}

$('#brain').addEventListener('pointerdown', e => {
  const thinker = thinkerNow(), r = e.currentTarget.getBoundingClientRect(), hit = neuronAt(e.clientX - r.left, e.clientY - r.top);
  if (thinker?.brain && hit) pickNeuron(thinker, hit.l, hit.j);
});
$('#brain').addEventListener('pointermove', e => {
  const thinker = thinkerNow(), r = e.currentTarget.getBoundingClientRect(), hit = neuronAt(e.clientX - r.left, e.clientY - r.top);
  const label = hit && thinker?.brain && neuronFacts(thinker.species?.name, thinker.brain, hit.l, hit.j)?.nr?.label;
  e.currentTarget.title = hit ? `Layer ${hit.l} · neuron ${hit.j + 1}${label ? `: ${label}` : ''}` : '';
});
document.querySelectorAll('.neuron-hold button').forEach(b => b.onclick = () => {
  const { brain, name, l, j } = pickedNeuron, hold = b.dataset.hold;
  brain.held = hold === '' ? null : [[l, j, hold === 'avg' ? neuronFacts(name, brain, l, j)?.nr?.mean ?? 0 : +hold]];
  showNeuron();
});
$('#neuron-close').onclick = dropNeuron;

// ---------- controls ----------

RACE_LENGTHS.forEach(laps => {
  const b = document.createElement('button');
  b.textContent = `${laps} laps`;
  b.classList.toggle('on', laps === sim.laps);
  b.onclick = () => {
    sim.laps = laps;
    document.querySelectorAll('#race-length button').forEach(x => x.classList.toggle('on', x === b));
    toast(`Races will be <b>${laps} laps</b> from the next one.`);
  };
  $('#race-length').append(b);
});

SPEEDS.forEach((s, i) => {
  const b = document.createElement('button');
  b.textContent = s === Infinity ? 'MAX' : `${s}×`;
  b.onclick = () => setSpeed(i);
  $('#speed').append(b);
});

CHANNELS.forEach(([id, label, help]) => {
  const b = document.createElement('button');
  Object.assign(b, { textContent: label, title: `${help}. Every race shown is one the engine really ran, car for car.` });
  b.dataset.ch = id;
  b.classList.toggle('on', id === state.channel);
  b.onclick = () => setChannel(id);
  $('#channels').append(b);
});

$('#btn-begin').onclick = begin;
$('#btn-pause').onclick = () => $('#intro').hidden ? setPaused(!state.paused) : begin();
$('#btn-camera').onclick = () => setCamera(view.mode === 'follow' ? 'full' : 'follow');
$('#btn-track').onclick = () => {
  if (sketch) stopDrawing();
  if (race) exitRace();
  play();
  if (watchedTracks() === 'nascar') {
    // the real ovals in turn, Daytona to Pikes Peak
    const def = NASCAR_TRACKS[state.oval = (state.oval + 1) % NASCAR_TRACKS.length], next = trackFor(def.id, 'nascar');
    const bank = Math.round(Math.max(...next.bankHi) * 180 / Math.PI);
    setTrack(next, `<b>${esc(def.name)}</b>: ${(next.length / UNITS_PER_M / 1609.344).toFixed(3)} mi, ${bank}° banking, ${def.pkg === 'plate' ? 'restrictor plates: pack racing' : '670 hp'} · pole ${Math.round(next.refSpeed * MPH_PER_SPEED)} mph`);
  } else setTrack(Track.random(), 'New track. <b>None of these brains have ever seen it.</b>');
  if (!$('#intro').hidden) begin();
};
$('#btn-draw').onclick = startDrawing;
$('#btn-race').onclick = () => {
  if (sketch) stopDrawing();
  play();
  startRace();
};
$('#btn-back').onclick = backToEvolving;
$('#btn-engine').onclick = () => fetch(`api/engine?action=${engine?.running ? 'stop' : 'start'}`, { method: 'POST' }).then(poll);
$('#tog-rays').onchange = e => state.showRays = e.target.checked;
$('#res-again').onclick = startRace;
$('#res-back').onclick = exitRace;

// ---------- settings: saves and options ----------

const toggleSettings = (open = $('#settings').hidden) => {
  $('#settings').hidden = !open;
  $('#btn-settings').classList.toggle('on', open);
  if (open) drawSlots();
};
$('#btn-settings').onclick = () => toggleSettings();
$('#settings-close').onclick = () => toggleSettings(false);

// no browser dialogs here: the embedded browser may block them, which silently cancelled every action
$('#btn-slot-scratch').onclick = async () => {
  const res = await slotAction({ action: 'new', from: 'scratch', tracks: 'normal' });
  if (res.id) await slotAction({ action: 'train', id: res.id });
};
$('#btn-slot-nascar').onclick = async () => {
  const res = await slotAction({ action: 'new', from: 'scratch', tracks: 'nascar', cars: 'stock' });
  if (res.id) await slotAction({ action: 'train', id: res.id });
};

// copying a save into the other mode: its brains, in their own cars or the other mode's
let converting = null;
function showConvert(id) {
  const from = modeOf(id), tracks = from.tracks === 'nascar' ? 'normal' : 'nascar';
  converting = { id, tracks, from, cars: 'own' };
  $('#convert').hidden = false;
  $('#convert-text').innerHTML = `Copy <b>${esc(slotName(id))}</b> into <b>${tracks === 'nascar' ? 'NASCAR' : 'normal'} mode</b>, where it starts a new evolution from these brains:`;
  $('#convert-host').textContent = `Drive the ${homeCars(tracks)} cars`;
  document.querySelectorAll('#convert [data-cars]').forEach(b => b.classList.toggle('on', b.dataset.cars === 'own'));
}
$('#convert').addEventListener('click', async e => {
  const choice = e.target.closest('[data-cars]')?.dataset.cars;
  if (choice) {
    converting.cars = choice;
    document.querySelectorAll('#convert [data-cars]').forEach(b => b.classList.toggle('on', b.dataset.cars === choice));
  }
  if (e.target.id === 'convert-cancel') $('#convert').hidden = true;
  if (e.target.id !== 'convert-go') return;
  const { id, tracks, from, cars } = converting;
  $('#convert').hidden = true;
  const res = await slotAction({ action: 'new', from: id, tracks, cars: cars === 'own' ? from.cars : homeCars(tracks), name: `${slotName(id)} ${tracks === 'nascar' ? 'NASCAR' : 'normal'}` });
  if (res.id) {
    await slotAction({ action: 'train', id: res.id });
    toast(`Copied into ${tracks === 'nascar' ? 'NASCAR' : 'normal'} mode as <b>${esc(slotName(res.id))}</b>. Its first tournament in the new mode is running.`);
  }
});

$('#watch-tracks').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  const home = homeMode().tracks;
  state.away = b.dataset.tracks ? (home === 'nascar' ? 'normal' : 'nascar') : null;
  drawSlots();
  showMode();
  if (!race && !sketch && league) state.playing ? backToEvolving() : startPros(!state.paused);
});
$('#watch-cars').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  state.awayCars = b.dataset.cars;
  drawSlots();
  if (!race && !sketch && league && state.away) startPros(!state.paused);
});
$('#slot-list').addEventListener('click', async e => {
  const id = e.target.closest('.slot')?.dataset.id;
  if (!id) return;
  if (e.target.closest('[data-act="convert"]')) return showConvert(id);
  if (e.target.closest('[data-act="delete"]')) {
    // two clicks on the same button: the first arms it for a few seconds, the second deletes
    if (armedDelete?.id === id && Date.now() < armedDelete.until) {
      armedDelete = null;
      await slotAction({ action: 'delete', id });
    } else {
      armedDelete = { id, until: Date.now() + 4000 };
      drawSlots();
      setTimeout(drawSlots, 4100);
    }
  } else if (id !== engine.active) await slotAction({ action: 'train', id });
});

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
  if (e.code === 'Escape' && !$('#settings').hidden) return toggleSettings(false);
  if (e.code === 'Space') $('#btn-pause').click();
  if (e.code === 'KeyC') $('#btn-camera').click();
  if (e.code === 'KeyN') $('#btn-track').click();
  if (e.code === 'KeyD' && !$('#btn-draw').hidden) $('#btn-draw').click();
  if (e.code === 'KeyR') $('#btn-race').click();
  if (e.code === 'KeyS') toggleSettings();
  if (e.code === 'KeyE' && state.playing) backToEvolving();
  if (/^Digit[1-5]$/.test(e.code)) setSpeed(+e.code.slice(5) - 1);
});
addEventListener('keyup', e => keys[e.code] = false);
addEventListener('blur', () => Object.keys(keys).forEach(k => keys[k] = false));
addEventListener('resize', layout);

// ---------- main loop ----------

// The screen runs between simulation steps: every car is drawn part of the way from where it was a step ago to
// where it is now, by how far the clock has got toward the next step, so motion is even at any refresh rate and
// any speed (a 120 Hz screen at 1x would otherwise move the cars every other frame). Only the drawing blends;
// the simulation never sees it.
let blend = 1;
function remember(heat) {
  for (const car of heat.cars) {
    car.wasX = car.x;
    car.wasY = car.y;
    car.wasAngle = car.angle;
  }
  // race control builds a new pace car every step, so the last one is where it was
  if (heat.control) heat.control.wasPace = heat.control.paceCar;
}

let last = performance.now(), acc = 0, frameNo = 0;
function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000), deadline = now + 14;
  last = now;
  view.dt = dt;
  const atSpeed = tick => {
    if (state.speed === Infinity) do remember(activeHeat()), tick(); while (performance.now() < deadline);
    else {
      acc += dt * 60 * state.speed;
      while (acc >= 1 && performance.now() < deadline) remember(activeHeat()), tick(), acc--;
      acc = Math.min(acc, 2);
    }
  };
  if (state.mode === 'train' && !state.paused) atSpeed(() => sim.tick());
  else if (state.mode === 'pros' && !state.paused) {
    if (pros.nextAt && now > pros.nextAt) startPros(true, true);
    else if (!pros.nextAt) atSpeed(prosTick);
  } else if (state.mode === 'race') {
    acc += dt * 60;
    while (acc >= 1 && state.mode === 'race') remember(race.heat), raceTick(), acc--;
  }
  blend = state.speed === Infinity && state.mode !== 'race' ? 1 : Math.min(1, acc);
  render();
  if (frameNo++ % 8 === 0 && !sketch) updateHud();
  requestAnimationFrame(loop);
}

setSpeed(1);
setCamera('follow');
setPaused(true);
layout();
poll();
setInterval(poll, 3000);
moveCamera(leaderOf(activeHeat()), true);
requestAnimationFrame(loop);
