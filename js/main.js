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
const state = { mode: 'train', speed: 3, paused: true, showRays: true, playing: false, focus: null, picked: null, away: null, awayCars: 'own', oval: -1 };
const keys = {};
let panels, race = null, sketch = null, lastFocus = null, cameraBeforeSketch = 'follow';
// The races you watch: a save's latest generation (models/slots/<id>/state.json, written by train/evolve.js).
let league = null, pros = null, engine = null, progress = null, xrayShown = null;
// everyone's weights at the start of the engine's current round, plus the practice plan it's running
let live = null;
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
const RACE_METRES = { practice: 5500, duel: 3000, tournament: 24000, saved: 24000, race: 8000 };
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

function render() {
  const heat = sketch ? null : activeHeat(), focus = heat && pickFocus(heat);
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

  if (thinker) {
    const marks = new Map();
    if (xrayShown?.car === thinker) {
      xrayShown.steerBy.forEach(e => marks.set(e.i, '#38e1ff'));
      xrayShown.gasBy.forEach(e => marks.has(e.i) || marks.set(e.i, e.gas > 0 ? '#4dff9a' : '#ff4d4d'));
    }
    drawBrain(panels.brain.ctx, panels.brain.w, panels.brain.h, thinker, marks);
    const sp = thinker.species, lineage = sp.parent ? ` · from ${sp.parent}, gen ${sp.born}` : sp.frozen ? ' · original, frozen' : '';
    $('#brain-title').textContent = `${race ? `${nameOf(thinker)}, nearest rival` : `Car ${nameOf(thinker)}`} · ${describe(sp)}${lineage}`;
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
  // the yardstick: share of the field beaten in races against a frozen panel of the run's past champions
  const past = history.filter(h => h.vsPast);
  drawChart(panels.vs.ctx, panels.vs.w, panels.vs.h, {
    series: past.length ? [
      { color: '#5b6578', points: past.map(h => [h.gen, 1 - h.vsPast.oldAvg]) },
      { color: '#ffd166', fill: 'rgba(255,209,102,0.08)', points: past.map(h => [h.gen, 1 - h.vsPast.newAvg]) },
    ] : [],
    yMin: 0, yMax: 1, ref: { value: 0.5, label: 'even' }, format: pct,
    empty: league.yardstick?.length || league.config ? 'waiting for the first generation with the panel…' : 'this save has no past-champion panel',
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
  $('#stat-champion').innerHTML = champ ? `<span style="color:${champ.style.color}">${champ.name}</span><small>design ${champ.species}, gen ${last.gen}</small>` : '—';
  const fastest = designs.filter(d => last.species[d].lap != null).sort((x, y) => last.species[x].lap - last.species[y].lap)[0];
  $('#stat-lap').innerHTML = fastest ? `${last.species[fastest].lap.toFixed(2)}s<small style="color:${designColor(fastest)}">design ${fastest}, gen ${last.gen}</small>` : '—<small>no laps yet</small>';
  $('#stat-gen-time').textContent = last.minutes ? `${last.minutes.toFixed(1)} min` : '—';
  const rows = designs.map(d => ({ d, s: last.species[d], members: league.population.filter(a => a.species === d) })).sort((x, y) => y.s.wins - x.s.wins);
  $('#design-table').innerHTML = `<tr><th>Design</th><th>Brain</th><th title="Share of this generation's tournament races won">Wins</th><th title="Share of the field beaten">Beats</th><th>Best</th><th>Aero lost</th><th title="Share of the race spent glued to the bumper of the car ahead">Tail</th></tr>` + rows.map(({ d, s, members }) => {
    const best = members.slice().sort(byStrength)[0];
    return `<tr><td><b style="color:${designColor(d)}">${d}</b></td><td>${members[0].layers.slice(1, -1).join('-')} <small>${(geneCount(members[0].layers) / 1000).toFixed(1)}k</small></td>
      <td>${Math.round(100 * s.wins / tourneyRaces(last))}%</td><td>${Math.round((1 - s.avgPlace) * 100)}%</td><td style="color:${best.style.color}">${best.name}</td><td>${Math.round(s.aero * 50)}%</td><td>${s.tail != null ? `${Math.round(s.tail * 100)}%` : '—'}</td></tr>`;
  }).join('');
}

// ---------- the practice board: every brain's practice this round, and the learning curve ----------

const count = v => Math.round(v).toLocaleString();
function drawBoard() {
  if (!panels) return;
  const p = engine?.running ? progress : null, b = p?.practice?.board, runs = p?.runs;
  $('#pb-summary').textContent = !p ? 'learning is paused'
    : p.phase === 'tournament' ? `generation ${p.generation} · tournament${runs ? ` · ${count(runs.total)} practice runs so far` : ''}`
    : runs ? `gen ${p.generation} · round ${p.round}/${p.rounds} · ${count(runs.round)} of ${count(runs.roundTotal)} runs · ${count(runs.perSecond)}/s · ${count(runs.total)} total` : 'starting…';

  const rounds = p?.rounds ?? 5, now = !p ? 0 : p.phase === 'tournament' ? rounds + 1 : p.round;
  $('#pb-rounds').innerHTML = [...Array(rounds + 1)].map((_, i) =>
    `<i class="${i + 1 < now ? 'done' : i + 1 === now ? 'now' : ''}">${i < rounds ? `R${i + 1}` : 'T'}</i>`).join('');

  // the seats never move (a replaced brain's copy takes its seat), so seat i lines up across rounds
  const seats = league?.population ?? [], hist = p?.practiceHistory ?? [];
  const last = hist.at(-1), prev = last && last.gen === p?.generation && last.round === p?.round ? hist.at(-2) : last;
  const watching = pros?.plan?.kind === 'practice' ? pros.plan.learner : state.picked;
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
  const phase = progress?.phase === 'tournament' ? 'tournament' : progress?.round ? `training round ${progress.round}/${progress.rounds}` : 'starting';
  const where = engine?.cloud
    ? ` on <b>Modal</b> (${progress?.workers ?? '…'} threads on ${progress?.machines ?? 1 + (engine.cloud.helpers ?? 0)} machines, ≈$${(engine.cloud.perHour ?? 3.15).toFixed(2)}/h, stops ${new Date(engine.cloud.until).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })})`
    : ` on <b>${progress?.workers ?? engine?.workers} cores</b>`;
  $('#engine-text').innerHTML = running
    ? `Training <b>${esc(slotName(training))}</b>${where} · generation ${progress?.generation ?? '…'} · ${phase}`
    : engine ? `Engine stopped${training ? ` · ${esc(slotName(training))} is saved at generation ${engine.slots.find(s => s.id === training)?.generation ?? 0}` : ''}` : `Engine offline · run <b>node server.js</b>`;
  const btn = $('#btn-engine');
  btn.hidden = !engine?.active;
  btn.textContent = running ? 'Stop' : 'Start';
  const done = !running || !progress?.rounds ? 0 : progress.phase === 'tournament' ? 0.97 : ((progress.round - 1) / progress.rounds) * 0.95;
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
  const head = race ? 'RACE' : !pros ? 'WAITING FOR GENERATION 0' : state.playing ? `PLAYING · GEN ${pros.gen}` : {
    practice: `${plan?.duel ? 'DUEL' : 'PRACTICE'} · GEN ${plan?.gen} · ROUND ${plan?.round}/${plan?.rounds}`,
    tournament: `TOURNAMENT · GEN ${plan?.gen} · TRACK ${plan?.k + 1}/${TOURNEY_TRACKS}`,
    saved: `GEN ${plan?.gen} · PAUSED`,
  }[plan?.kind];
  showPhase();
  const control = heat.control, total = lead.raceLaps, done = lead.laps.length + lead.freeLaps - lead.yellowLaps;
  $('#tower-head').textContent = `${head} · LAP ${Math.min(total, done + 1)}/${total}${control?.overtime ? ' · OT' : ''}${control?.flag === 'yellow' ? ' · CAUTION' : ''}`;
  showFlags(heat, focus);
  const shown = standings.slice(0, 12);
  if (focus && !shown.includes(focus)) shown[shown.length - 1] = focus;
  shown.forEach((car, i) => {
    const row = towerRow(i), [pos, chip, name, gap] = row.children;
    row.dataset.slot = car.slot;
    row.className = `${car === focus ? 'focus' : ''} ${car.retired ? 'out' : ''}`;
    pos.textContent = standings.indexOf(car) + 1;
    chip.style.background = colorOf(car);
    name.style.color = car.species?.color ?? '';
    name.innerHTML = `${nameOf(car)}${car.elite ? ' <span class="star">★</span>' : ''}`;
    gap.textContent = towerGap(heat, car, standings);
  });
  towerRows.forEach((row, i) => row.hidden = i >= shown.length);

  if (race) {
    const you = race.you;
    $('#race-pos').textContent = `P${standings.indexOf(you) + 1}/${heat.cars.length}`;
    $('#race-lap').textContent = `${Math.min(heat.laps, you.laps.length + you.freeLaps - you.yellowLaps + 1)}/${heat.laps}`;
    $('#race-time').textContent = (you.steps / 60).toFixed(2);
    $('#race-dmg').textContent = damageText(you);
  } else {
    $('#hud-gen').textContent = pros ? pros.gen : '—';
    $('#hud-heat-n').textContent = pros ? pros.count : '—';
    $('#hud-alive').textContent = heat.cars.reduce((n, car) => n + car.running, 0);
  }

  const thinker = race ? nearestRival(heat, race.you) : focus;
  if (thinker?.brain) {
    xrayShown = { car: thinker, ...xray(thinker) };
    $('#xray').innerHTML = xrayHtml(xrayShown);
  }

  const car = race ? race.you : focus;
  if (car) $('#car-stats').innerHTML = [
    ['speed', car.spec.stock ? `${Math.round(Math.max(0, car.forwardSpeed) * MPH_PER_SPEED)} mph` : `${Math.round(Math.max(0, car.forwardSpeed) * KMH)} km/h`],
    ['slipstream', `${Math.round(car.draft * 100)}%`],
    ['aero lost F·R · drag', damageText(car)],
    ['wall hits', car.wallHits],
    ...track.nascar ? [
      ['banking · surface', `${Math.round(track.bankAt(car.lastArc, track.lateralAt(car.x, car.y)) * 180 / Math.PI)}° · ${SURFACE_NAME[car.surface]}`],
      ['flags', car.parked ? 'parked' : car.penalty ? 'black: stop-and-go' : car.blueFlag ? 'blue-yellow' : car.paced ? 'following the pace car' : '—'],
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
  $('#hud-speed').textContent = state.speed === Infinity ? 'MAX' : `${state.speed}×`;
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
  const heat = new Heat(track, drivers.map(d => d && new Brain(d.species.layers, d.genes)), lapsOn(track, 'race', HUMAN_RACE_LAPS), { cars: carsFor(track) });
  heat.events = [];
  heat.cars.forEach((car, slot) => Object.assign(car, { rank: slot < you ? slot + 1 : slot, species: drivers[slot]?.species, pro: !!drivers[slot]?.species.design }));
  race = { heat, you: heat.cars[you], t: -180, steer: 0, count: null, endAt: Infinity };
  race.you.human = true;
  dress(heat);
  state.mode = 'race';
  stage.classList.add('racing');
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
  $('#hud').hidden = false;
  lastFocus = null;
}

// ---------- the races you watch: the watched save's latest generation, back to back ----------

function useTrack(next) {
  track = next;
  lastFocus = null;
  layout();
}

// What the evolving screen shows: exactly what the engine is doing right now.
//   practice: one of this round's practice races, same track, rivals and grid slot, with everyone's
//             weights as of the start of the round (live.json + progress.json)
//   tournament: this generation's tournament tracks, the races that decide who gets replaced
//   saved: learning is paused, so the latest saved generation on the tracks it was judged on
function evolvingPlan(next) {
  const p = engine?.running && live?.progress, prev = next ? pros?.plan : null, entry = a => ({ weights: a.weights, species: a.style });
  if (p?.phase === 'training' && p.practice) {
    const roster = p.practice.pros;
    // the brain you clicked, or each brain in turn
    const pro = roster.find(x => x.name === state.picked) ?? roster[prev?.kind === 'practice' ? (roster.findIndex(x => x.name === prev.learner) + 1) % roster.length : 0];
    const duels = pro.duels ?? [], k = prev?.learner === pro.name ? (prev.k + 1) % (pro.slots.length + duels.length) : 0;
    const base = { kind: 'practice', learner: pro.name, k, gen: p.generation, round: p.round, rounds: p.rounds, copies: p.practice.pairs * 2 };
    const duel = duels[k - pro.slots.length];
    if (duel) {
      const pair = [live.byName.get(pro.name), live.byName.get(duel.rival)];
      return { ...base, duel, slot: duel.slot, trackRef: duel.track, laps: duel.laps, grid: (duel.slot ? pair.reverse() : pair).map(entry) };
    }
    const grid = pro.rivals.map(name => live.byName.get(name)), laps = p.practice.laps;
    grid.splice(pro.slots[k], 0, live.byName.get(pro.name));
    return { ...base, slot: pro.slots[k], trackRef: p.practice.tracks[k], laps: Array.isArray(laps) ? laps[k] : laps, grid: grid.map(entry) };
  }
  // a grid holds 20 (40 on the ovals): bigger populations race in random fields, like the engine's tournament
  const field = homeMode().tracks === 'nascar' ? 40 : 20;
  if (p?.phase === 'tournament' && p.tournament) {
    const k = prev?.kind === 'tournament' && prev.gen === p.generation ? (prev.k + 1) % p.tournament.tracks.length : 0, laps = p.tournament.laps;
    return { kind: 'tournament', k, trackRef: p.tournament.tracks[k], laps: Array.isArray(laps) ? laps[k] : laps, grid: shuffle(live.population.map(entry)).slice(0, field), gen: p.generation };
  }
  // paused: the best of the latest saved generation, on its tournament's tracks
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
  const html = {
    practice: () => plan.duel ? `<b>Duel</b> · generation ${plan.gen}, round ${plan.round} of ${plan.rounds}<small><b style="color:${color(plan.learner)}">${esc(plan.learner)}</b> is learning to ${plan.slot ? 'attack' : 'defend'}: ${plan.laps} laps against <b style="color:${color(plan.duel.rival)}">${esc(plan.duel.rival)}</b>, starting ${plan.slot ? 'behind' : 'in front'}. Only the winner scores, so ${plan.slot ? 'it has to get past, by out-braking it or spinning it round' : 'it has to hold on and keep its rear corners covered'}. ${plan.copies} versions of it are racing this right now.</small>`
      : `<b>Practice</b> · generation ${plan.gen}, round ${plan.round} of ${plan.rounds}<small><b style="color:${color(plan.learner)}">${esc(plan.learner)}</b> is learning, starting P${plan.slot + 1}. The engine is racing ${plan.copies} slightly different versions of it in this exact race right now, then nudging it toward the ones that did better. Click any car to watch it learn next.</small>`,
    tournament: () => `<b>Tournament</b> · judging generation ${plan.gen}, track ${plan.k + 1} of ${TOURNEY_TRACKS}<small>No learning here: these races rank everyone, and in each design the slowest brain is replaced by a copy of the best.</small>`,
    saved: () => `<b>Generation ${plan.gen}</b> · learning is paused<small>The latest saved cars on the tracks their tournament was raced on.</small>`,
  }[plan.kind]().replace('</b>', `</b>${at}`);
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
  if (plan) useTrack(trackFor(plan.trackRef, watchedTracks()));
  const best = () => shuffle(league.population.slice().sort(byStrength).slice(0, track.nascar ? 40 : 20).map(a => ({ weights: a.weights, species: a.style })));
  const grid = plan ? plan.grid : best();
  // on the save's own tracks the engine's lap count; elsewhere the same kind of race at that track's length
  const away = !!track.nascar !== (homeMode().tracks === 'nascar'), kind = !plan ? 'tournament' : plan.duel ? 'duel' : plan.kind;
  const laps = !plan ? lapsOn(track, 'tournament', sim.laps) * (track.nascar ? sim.laps / 10 : 1) : away ? lapsOn(track, kind, { practice: 3, duel: 2 }[kind] ?? 10) : plan.laps;
  const heat = new Heat(track, grid.map(d => new Brain(d.species.layers, d.weights)), Math.max(2, Math.round(laps)), {
    cars: carsFor(track), stages: track.nascar && kind !== 'practice' && kind !== 'duel', practice: kind === 'practice' || kind === 'duel',
  });
  heat.events = [];
  heat.cars.forEach((car, i) => Object.assign(car, { pro: true, species: grid[i].species }));
  dress(heat);
  pros = { heat, count: (pros?.count ?? 0) + 1, nextAt: null, gen: plan?.gen ?? league.generation, plan };
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
  const plan = pros.plan, which = state.playing ? `your race` : plan.kind === 'practice' ? 'this practice race' : plan.kind === 'tournament' ? `tournament race ${plan.k + 1}` : `generation ${pros.gen}, race ${pros.count}`;
  toast(`<b style="color:${winner.species.color}">${winner.species.name}</b> (design ${winner.species.design}) ${winner.finished ? 'wins' : 'gets furthest in'} ${which}${margin}`, 'gold');
  pros.nextAt = performance.now() + 4000;
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
  if (pros?.plan?.kind === 'practice') startPros();
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
    if (pros.nextAt && now > pros.nextAt) startPros(true, true);
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
poll();
setInterval(poll, 3000);
moveCamera(leaderOf(activeHeat()), true);
requestAnimationFrame(loop);
