// NASCAR mode: the teams, each car's number and livery, the season's tracks, and how the ovals, stock cars and
// pace car are drawn. Teams and sponsors are made up; the tracks are real places.

// the five brain designs race as five teams
const TEAMS = {
  A: { name: 'Velocity Motorsports', colors: ['#1f6feb', '#ffffff', '#ffcc00'] },
  B: { name: 'Thunder Valley Racing', colors: ['#7b2ff7', '#ffd400', '#141414'] },
  C: { name: 'Coastline Speed Group', colors: ['#00a86b', '#ffffff', '#0b3d2e'] },
  D: { name: 'Iron Horse Racing', colors: ['#ff6b00', '#141414', '#ffffff'] },
  E: { name: 'Summit Performance', colors: ['#ff2d87', '#c9ced4', '#1a1a1a'] },
};
const SPONSORS = [
  'Turbo Cola', 'Apex Lube', 'Blue Ridge Tires', 'Bolt Energy', 'Lone Star Lumber', 'Peak Mutual', 'Rocket Burger',
  'Copperline Wireless', 'Big Sky Insurance', 'Redline Batteries', 'Northstar Freight', 'Prairie Seed Co.', 'Hammerhead Tools',
  'Sunset Motor Oil', 'Iron Ridge Steel', 'Crescent Bank', 'Gulf Breeze Snacks', 'Thunder Root Beer', 'Silverline Paint',
  'Maverick Jerky', 'Pinecrest Pharmacy', 'Bluewater Boats', 'Highway Hauling', 'Cobalt Cloud', 'Wildcat Wireless',
  'Granite Mutual', 'Rio Grande Rentals', 'Velvet Car Wash', 'Catfish Kitchen', 'Atlas Generators', 'Comet Coffee',
  'Bayou Boots', 'Jetstream Pest Control', 'Ozark Outdoors', 'Kingpin Bowling', 'Starlite Diner', 'Glacier Water',
  'Desert Sun Solar', 'Cardinal Plumbing', 'Riverboat Casino', 'Mustang Mattress', 'Lighthouse Realty',
];
const LIVERY_PATTERNS = ['stripes', 'chevron', 'twotone', 'flames', 'sash', 'fade', 'checks'];

function nameHash(s) {
  let h = 2166136261;
  for (const ch of String(s)) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}

// a number for every car in a save, from its name where it can (A2·73 runs #73), each one unique
function assignNumbers(names) {
  const taken = new Set(), out = new Map();
  for (const name of names) {
    let n = +(String(name).match(/(\d+)\D*$/)?.[1] ?? 0) % 100 || 1 + nameHash(name) % 99;
    while (taken.has(n)) n = n % 99 + 1;
    taken.add(n);
    out.set(name, n);
  }
  return out;
}

// the team's colours, a pattern and a sponsor of its own
function liveryFor(name, design) {
  const team = TEAMS[design] ?? TEAMS.A, h = nameHash(name);
  const [primary, secondary, accent] = h & 1 ? team.colors : [team.colors[0], team.colors[2], team.colors[1]];
  return { primary, secondary, accent, pattern: LIVERY_PATTERNS[(h >>> 3) % LIVERY_PATTERNS.length], sponsor: SPONSORS[(h >>> 7) % SPONSORS.length], team: team.name };
}

// ---- the season: which ovals a NASCAR save practises and races on ----
const SUPERSPEEDWAYS = ['daytona', 'talladega', 'atlanta'];
const OTHER_OVALS = () => NASCAR_TRACKS.map(t => t.id).filter(id => !SUPERSPEEDWAYS.includes(id));
// a fixed panel for the yardstick, every kind of oval
const YARDSTICK_OVALS = ['daytona', 'talladega', 'charlotte', 'michigan', 'darlington', 'phoenix', 'bristol', 'martinsville'];
const pickFrom = (list, seed) => list[nameHash(seed) % list.length];
// every practice round: one race at a superspeedway (pack racing), one at another oval
const practiceOvals = (gen, round) => [pickFrom(SUPERSPEEDWAYS, `ss${gen}:${round}`), pickFrom(OTHER_OVALS(), `ov${gen}:${round}`)];
// each generation's tournament is an 8-race season: two superspeedways and six other ovals
function seasonOvals(gen) {
  const others = OTHER_OVALS(), out = [pickFrom(SUPERSPEEDWAYS, `season${gen}a`), pickFrom(SUPERSPEEDWAYS, `season${gen}b`)];
  for (let k = 0; out.length < 8; k++) {
    const id = pickFrom(others, `season${gen}:${k}`);
    if (!out.includes(id)) out.push(id);
  }
  return out;
}
const lapLength = id => NASCAR_TRACKS.find(t => t.id === id).miles * 1609.344;
// race distance in laps for a given distance, at least a few laps
const lapsFor = (id, metres, min = 2, max = 60) => Math.max(min, Math.min(max, Math.round(metres / lapLength(id))));

// ---- drawing (browser only) ----

// a lateral line round the oval, every few points, as a flat [x, y, ...] list
function ovalLine(track, lateral, from = 0, to = track.length, step = 16) {
  const out = [];
  for (let s = from; s <= to; s += step) out.push(...track.pointAt(s, lateral));
  return out;
}
function bandPath(track, inner, outer, from = 0, to = track.length, step = 16) {
  const path = new Path2D(), a = ovalLine(track, outer, from, to, step), b = ovalLine(track, inner, from, to, step);
  for (let i = 0; i < a.length; i += 2) i ? path.lineTo(a[i], a[i + 1]) : path.moveTo(a[i], a[i + 1]);
  for (let i = b.length - 2; i >= 0; i -= 2) path.lineTo(b[i], b[i + 1]);
  path.closePath();
  return path;
}
function linePath(track, lateral, from = 0, to = track.length, step = 16, closed = true) {
  const path = new Path2D(), a = ovalLine(track, lateral, from, to, step);
  for (let i = 0; i < a.length; i += 2) i ? path.lineTo(a[i], a[i + 1]) : path.moveTo(a[i], a[i + 1]);
  if (closed) path.closePath();
  return path;
}

// everything that doesn't move, built once per track
function ovalScenery(track) {
  const m = 4, hw = track.halfWidth, apron = track.apron, L = track.length, def = track.def;
  const innerEdge = -hw - apron, pitWall = innerEdge - 7 * m, pitZone = track.pitZone;
  // banking as shading: the steeper the lane, the lighter the asphalt
  const n = track.bankHi.length, buckets = Array.from({ length: 8 }, () => new Path2D());
  for (let i = 0; i < n; i += 2) {
    const s0 = track.arc[i], s1 = track.arc[Math.min(n, i + 2)] + 1;
    const bank = (track.bankLo[i] + track.bankHi[i]) / 2, b = Math.min(7, Math.floor(bank / (34 * Math.PI / 180) * 8));
    const p = bandPath(track, -hw, hw, s0, s1, 8);
    buckets[b].addPath(p);
  }
  // the infield: everything inside the apron's edge
  const infield = linePath(track, innerEdge, 0, L, 16);
  const sand = track.sandAt.map(at => bandPath(track, innerEdge - 22 * m, innerEdge - 6 * m, at - 30 * m, at + 30 * m, 8));
  const stands = [];
  for (let s = -pitZone; s < pitZone; s += 60 * m) stands.push(bandPath(track, hw + 12 * m, hw + 40 * m, s, s + 55 * m, 10));
  // infield lake (Daytona's Lake Lloyd) or a big logo, around the infield's middle
  let cx = 0, cy = 0;
  for (const [x, y] of track.points) cx += x, cy += y;
  cx /= track.points.length;
  cy /= track.points.length;
  return {
    outside: linePath(track, hw + 60 * m, 0, L, 24),
    wall: bandPath(track, hw, hw + 1.2 * m),
    fence: linePath(track, hw + 2.6 * m),
    surface: bandPath(track, -hw, hw), buckets,
    apron: bandPath(track, innerEdge, -hw),
    infield, sand, stands,
    yellow: track.plate ? [linePath(track, -hw + 0.25 * m), linePath(track, -hw + 0.75 * m)] : [],
    white: [linePath(track, hw - 0.6 * m), linePath(track, innerEdge + 0.2 * m)],
    pitWall: def.insideWall ? null : bandPath(track, pitWall - 0.6 * m, pitWall, L - pitZone, L + pitZone, 8),
    pitRoad: def.insideWall ? null : bandPath(track, pitWall - 15 * m, pitWall - 0.6 * m, L - pitZone, L + pitZone, 8),
    innerWall: def.insideWall ? bandPath(track, innerEdge - 1 * m, innerEdge) : null,
    centre: [cx, cy],
  };
}

function drawOval(ctx, track) {
  const sc = track.scenery ??= ovalScenery(track), m = 4, hw = track.halfWidth;
  // outside the wall: the property, stands and parking
  ctx.fillStyle = '#20251f';
  ctx.fill(sc.outside);
  ctx.fillStyle = '#2f3b2c';
  ctx.fill(sc.infield);
  // mowed stripes across the infield
  ctx.save();
  ctx.clip(sc.infield);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.035)';
  ctx.lineWidth = 12 * m;
  ctx.beginPath();
  const [cx, cy] = sc.centre, reach = Math.hypot(track.worldW, track.worldH) / 2;
  for (let k = -reach; k <= reach; k += 24 * m) ctx.moveTo(cx - reach, cy + k - reach), ctx.lineTo(cx + reach, cy + k + reach);
  ctx.stroke();
  if (track.def.infield === 'lake') {
    ctx.fillStyle = '#1d4f73';
    ctx.beginPath();
    ctx.ellipse(cx, cy, 90 * m, 45 * m, 0.4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  ctx.fillStyle = '#c8b27a';
  for (const s of sc.sand) ctx.fill(s);
  ctx.fillStyle = '#3a4048';
  for (const s of sc.stands) ctx.fill(s);
  if (sc.pitRoad) {
    ctx.fillStyle = '#3b3f45';
    ctx.fill(sc.pitRoad);
  }
  ctx.fillStyle = '#5a5f66';
  ctx.fill(sc.apron);
  ctx.fillStyle = '#26292e';
  ctx.fill(sc.surface);
  sc.buckets.forEach((path, b) => {
    if (!b) return;
    ctx.fillStyle = `rgba(255, 255, 255, ${b * 0.012})`;
    ctx.fill(path);
  });
  ctx.lineWidth = 0.35 * m;
  ctx.strokeStyle = '#f2c94c';
  for (const p of sc.yellow) ctx.stroke(p);
  ctx.strokeStyle = 'rgba(240, 244, 248, 0.75)';
  for (const p of sc.white) ctx.stroke(p);
  // SAFER barrier, with its blue band, and the catch fence
  ctx.fillStyle = '#e9edf1';
  ctx.fill(sc.wall);
  ctx.strokeStyle = 'rgba(40, 90, 200, 0.8)';
  ctx.lineWidth = 0.3 * m;
  ctx.stroke(linePathCached(track, hw + 0.6 * m));
  ctx.strokeStyle = 'rgba(200, 205, 210, 0.35)';
  ctx.lineWidth = 0.2 * m;
  ctx.stroke(sc.fence);
  if (sc.pitWall) {
    ctx.fillStyle = '#d9dde2';
    ctx.fill(sc.pitWall);
  }
  if (sc.innerWall) {
    ctx.fillStyle = '#d9dde2';
    ctx.fill(sc.innerWall);
  }
  // start/finish: a checkered band across the racing surface
  const [sx, sy] = track.points[0], sq = 1.2 * m, rows = Math.ceil((2 * hw) / sq);
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(track.heading[0]);
  for (let row = 0; row < 2; row++)
    for (let col = 0; col < rows; col++) {
      ctx.fillStyle = (row + col) % 2 ? '#111' : '#f5f5f5';
      ctx.fillRect(row * sq - sq, -hw + col * sq, sq, sq);
    }
  ctx.restore();
}
function linePathCached(track, lateral) {
  track.lines ??= new Map();
  if (!track.lines.has(lateral)) track.lines.set(lateral, linePath(track, lateral));
  return track.lines.get(lateral);
}

// a stock car seen from above: 4.97 m x 1.99 m, livery, roof number, windshield, spoiler, hood sponsor
function drawStockCar(ctx, car, livery, { glow = false, alpha = 1, pace = false } = {}) {
  const L = car.spec.len / 2, W = car.spec.wid / 2;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(car.x, car.y);
  ctx.rotate(car.angle);
  if (car.draft > 0.15) {
    ctx.strokeStyle = `rgba(150, 240, 255, ${car.draft * 0.9})`;
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (const y of [-W - 1, W + 1]) ctx.moveTo(L - 4, y), ctx.lineTo(-L - 8, y);
    ctx.stroke();
  }
  const braking = car.running && car.throttle < -0.1;
  const body = new Path2D();
  body.moveTo(L - 1.6, -W);
  body.quadraticCurveTo(L, -W, L, -W + 1.6);
  body.lineTo(L, W - 1.6);
  body.quadraticCurveTo(L, W, L - 1.6, W);
  body.lineTo(-L + 0.6, W);
  body.lineTo(-L, W - 0.6);
  body.lineTo(-L, -W + 0.6);
  body.lineTo(-L + 0.6, -W);
  body.closePath();
  // tyres peeking out
  ctx.fillStyle = '#0b0c0e';
  for (const [x, y] of [[L - 4.2, -W - 0.3], [L - 4.2, W - 1.1], [-L + 3, -W - 0.3], [-L + 3, W - 1.1]]) ctx.fillRect(x - 1.6, y, 3.2, 1.4);
  if (glow) {
    ctx.shadowColor = livery.primary;
    ctx.shadowBlur = 16 * view.dpr;
  }
  ctx.fillStyle = livery.primary;
  ctx.fill(body);
  ctx.shadowBlur = 0;
  ctx.save();
  ctx.clip(body);
  ctx.fillStyle = livery.secondary;
  switch (livery.pattern) {
    case 'stripes': ctx.fillRect(-L, -1.6, 2 * L, 1.1); ctx.fillRect(-L, 0.5, 2 * L, 1.1); break;
    case 'chevron': ctx.beginPath(); ctx.moveTo(L, -W); ctx.lineTo(L * 0.1, 0); ctx.lineTo(L, W); ctx.lineTo(L, W - 2.2); ctx.lineTo(L * 0.1 + 2.2, 0); ctx.lineTo(L, -W + 2.2); ctx.fill(); break;
    case 'twotone': ctx.fillRect(-L, -W, L * 0.9, 2 * W); break;
    case 'flames': ctx.beginPath(); for (let k = -2; k <= 2; k++) { ctx.moveTo(L, k * 1.6); ctx.quadraticCurveTo(L * 0.3, k * 1.6 + 1.2, -L * 0.2, k * 1.4); ctx.quadraticCurveTo(L * 0.3, k * 1.6 - 0.4, L, k * 1.6 + 0.8); } ctx.fill(); break;
    case 'sash': ctx.beginPath(); ctx.moveTo(L * 0.6, -W); ctx.lineTo(L * 0.6 + 3, -W); ctx.lineTo(-L * 0.4 + 3, W); ctx.lineTo(-L * 0.4, W); ctx.fill(); break;
    case 'fade': { const g = ctx.createLinearGradient(-L, 0, L, 0); g.addColorStop(0, livery.secondary); g.addColorStop(1, 'rgba(0,0,0,0)'); ctx.fillStyle = g; ctx.fillRect(-L, -W, 2 * L, 2 * W); break; }
    case 'checks': for (let x = -L; x < -L * 0.3; x += 1.4) for (let y = -W; y < W; y += 1.4) if (((x + L) / 1.4 + (y + W) / 1.4 | 0) % 2) ctx.fillRect(x, y, 1.4, 1.4); break;
  }
  ctx.fillStyle = livery.accent;
  ctx.fillRect(L - 0.5, -W, 0.5, 2 * W);
  ctx.restore();
  // glass: windshield, side windows, rear window; then the roof
  ctx.fillStyle = 'rgba(12, 16, 22, 0.92)';
  ctx.beginPath();
  ctx.moveTo(3.6, -W + 0.9);
  ctx.lineTo(1.3, -W + 1.2);
  ctx.lineTo(1.3, W - 1.2);
  ctx.lineTo(3.6, W - 0.9);
  ctx.closePath();
  ctx.fill();
  ctx.fillRect(-5.6, -W + 1.1, 1.4, 2 * W - 2.2);
  ctx.fillStyle = pace ? '#f4f4f4' : livery.primary;
  ctx.fillRect(-4.2, -W + 1, 5.5, 2 * W - 2);
  // roof number, the way the TV cameras read it
  if (!pace && car.number != null) {
    ctx.fillStyle = '#f7f7f7';
    ctx.fillRect(-3.6, -2.4, 4.2, 4.8);
    ctx.save();
    ctx.rotate(Math.PI / 2);
    ctx.fillStyle = '#111';
    ctx.font = `900 3.6px ui-sans-serif, system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(car.number, 0, 1.5);
    ctx.restore();
  }
  // decklid spoiler
  ctx.fillStyle = '#0d0f12';
  ctx.fillRect(-L, -W + 0.3, 0.7, 2 * W - 0.6);
  // hood sponsor, legible up close
  if (!pace && view.zoom > 2.2) {
    ctx.save();
    ctx.translate(6.4, 0);
    ctx.rotate(Math.PI / 2);
    ctx.fillStyle = livery.accent;
    ctx.font = `800 1.25px ui-sans-serif, system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(livery.sponsor.toUpperCase(), 0, 0);
    ctx.restore();
  }
  if (pace) {
    // light bar
    const on = (performance.now() / 250 | 0) % 2;
    ctx.fillStyle = on ? '#ffb000' : '#ff3b30';
    ctx.fillRect(-2.6, -W + 1.2, 1.2, 2 * W - 2.4);
  }
  if (braking) {
    ctx.fillStyle = '#ff2a2a';
    ctx.fillRect(-L - 0.2, -W + 0.6, 0.5, 1.4);
    ctx.fillRect(-L - 0.2, W - 2, 0.5, 1.4);
  }
  ctx.restore();
}

const PACE_LIVERY = { primary: '#f2f2f2', secondary: '#ffcc00', accent: '#111', pattern: 'stripes', sponsor: 'PACE CAR' };
function drawPaceCar(ctx, pace) {
  drawStockCar(ctx, { ...pace, vx: 0, vy: 0, draft: 0, running: true, throttle: 0, spec: { len: 19, wid: 7.6 } }, PACE_LIVERY, { pace: true });
}

// the flag stand: the colour of the current flag and what it means
const FLAG_STYLE = {
  green: ['#2ecc71', 'GREEN'], yellow: ['#f1c40f', 'CAUTION'], red: ['#e74c3c', 'RED FLAG'], white: ['#f5f5f5', 'WHITE FLAG · FINAL LAP'],
  checkered: ['checkered', 'CHECKERED'], black: ['#111', 'BLACK FLAG'], redblack: ['#e74c3c', 'PARKED'], stage: ['checkered', 'STAGE END'],
};
