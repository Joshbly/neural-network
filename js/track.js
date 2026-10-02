const WORLD_W = 2400, WORLD_H = 1560;
const CELL = 4, HALF_WIDTH = 36, SPACING = 8;
const REACH = HALF_WIDTH + 40, OUTSIDE = -45;
const R_MIN = 46, R_MAX = 260;
const GRID_SLOTS = 20;

function mulberry32(seed) {
  return () => {
    seed = seed + 0x6d2b79f5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function resampleLoop(pts, spacing) {
  const segs = pts.map((p, i) => {
    const q = pts[(i + 1) % pts.length];
    return Math.hypot(q[0] - p[0], q[1] - p[1]);
  });
  const total = segs.reduce((a, b) => a + b, 0);
  const n = Math.max(8, Math.round(total / spacing)), step = total / n;
  const out = [];
  let seg = 0, along = 0;
  for (let k = 0; k < n; k++) {
    const target = k * step;
    while (along + segs[seg] < target) along += segs[seg++];
    const p = pts[seg], q = pts[(seg + 1) % pts.length], t = segs[seg] ? (target - along) / segs[seg] : 0;
    out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
  }
  return out;
}

function smoothLoop(pts) {
  const n = pts.length;
  return pts.map((b, i) => {
    const a = pts[(i - 1 + n) % n], c = pts[(i + 1) % n];
    return [(a[0] + 2 * b[0] + c[0]) / 4, (a[1] + 2 * b[1] + c[1]) / 4];
  });
}

function smoothUntilDrivable(pts) {
  for (let pass = 0; pass < 80 && trackProblem(pts) === 'sharp'; pass++) pts = resampleLoop(smoothLoop(pts), SPACING);
  return pts;
}

function convexHull(pts) {
  pts = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const chain = list => list.reduce((acc, p) => {
    while (acc.length > 1 && cross(acc.at(-2), acc.at(-1), p) <= 0) acc.pop();
    acc.push(p);
    return acc;
  }, []).slice(0, -1);
  return chain(pts).concat(chain(pts.slice().reverse()));
}

const wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));
const bend = (a, p, b) => Math.abs(wrapAngle(Math.atan2(b[1] - p[1], b[0] - p[0]) - Math.atan2(p[1] - a[1], p[0] - a[0])));

function turnAt(pts, i) {
  const n = pts.length;
  return bend(pts[(i - 2 + n) % n], pts[i], pts[(i + 2) % n]);
}

// drops vertices that are nearly collinear or crowd their neighbour
function simplifyPolygon(poly) {
  for (let i = 0; poly.length > 4 && i < poly.length;) {
    const n = poly.length, a = poly[(i - 1 + n) % n], p = poly[i];
    if (bend(a, p, poly[(i + 1) % n]) < 0.12 || Math.hypot(p[0] - a[0], p[1] - a[1]) < 90) poly.splice(i, 1), i = 0;
    else i++;
  }
  return poly;
}

// rounds every polygon corner into a constant-radius arc, leaving true straights between them
function filletPolygon(poly, rng) {
  const n = poly.length;
  const corners = poly.map((p, i) => {
    const a = poly[(i - 1 + n) % n], b = poly[(i + 1) % n];
    const inLen = Math.hypot(p[0] - a[0], p[1] - a[1]), outLen = Math.hypot(b[0] - p[0], b[1] - p[1]);
    const din = [(p[0] - a[0]) / inLen, (p[1] - a[1]) / inLen], dout = [(b[0] - p[0]) / outLen, (b[1] - p[1]) / outLen];
    return {
      p, din,
      turn: Math.acos(Math.max(-1, Math.min(1, din[0] * dout[0] + din[1] * dout[1]))),
      side: Math.sign(din[0] * dout[1] - din[1] * dout[0]),
      r: R_MIN + (R_MAX - R_MIN) * rng() ** 2,
    };
  });
  for (let pass = 0; pass < 3; pass++)
    corners.forEach((c, i) => {
      const next = corners[(i + 1) % n], edge = Math.hypot(next.p[0] - c.p[0], next.p[1] - c.p[1]);
      const need = c.r * Math.tan(c.turn / 2) + next.r * Math.tan(next.turn / 2);
      if (need > edge * 0.92) {
        const k = edge * 0.92 / need;
        c.r *= k;
        next.r *= k;
      }
    });
  if (corners.some(c => c.r < R_MIN)) return null;

  const out = [];
  for (const { p, din, turn, side, r } of corners) {
    const t = r * Math.tan(turn / 2), sx = p[0] - din[0] * t, sy = p[1] - din[1] * t;
    const cx = sx - din[1] * r * side, cy = sy + din[0] * r * side;
    const a0 = Math.atan2(sy - cy, sx - cx), steps = Math.max(1, Math.ceil(turn * r / SPACING));
    for (let k = 0; k <= steps; k++) {
      const a = a0 + side * turn * k / steps;
      out.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
  }
  return out;
}

// expects evenly spaced points; returns a human-readable reason or null
function trackProblem(pts) {
  const n = pts.length, margin = HALF_WIDTH + 10;
  if (n * SPACING < 2000) return 'That loop is too small — draw a bigger one.';
  if (pts.some(([x, y]) => x < margin || y < margin || x > WORLD_W - margin || y > WORLD_H - margin))
    return 'Keep the loop away from the edges.';
  for (let i = 0; i < n; i++)
    if (turnAt(pts, i) > 2 * SPACING / (HALF_WIDTH * 1.15)) return 'sharp';
  // wide grass gaps between neighbouring sections so a hard hit can't push a car onto the wrong stretch
  const neighbours = Math.ceil(HALF_WIDTH * 5 / SPACING), clearance = (2 * HALF_WIDTH + 40) ** 2;
  for (let i = 0; i < n; i++)
    for (let j = i + neighbours; j < Math.min(n, n - neighbours + i + 1); j++) {
      const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];
      if (dx * dx + dy * dy < clearance) return 'The track crosses or touches itself — try a simpler loop.';
    }
  return null;
}

class Track {
  static nextId = 1;

  constructor(points) {
    this.id = Track.nextId++;
    const pts = resampleLoop(points, SPACING), n = pts.length;

    // start/finish on the straightest stretch, with room behind it for the grid
    const turns = pts.map((_, i) => turnAt(pts, i)), behind = Math.round(340 / SPACING), ahead = Math.round(380 / SPACING);
    let start = 0, calmest = Infinity;
    for (let i = 0; i < n; i++) {
      let curviness = 0;
      for (let k = -behind; k < ahead; k++) curviness += turns[(i + k + n) % n];
      if (curviness < calmest) [start, calmest] = [i, curviness];
    }
    this.points = pts.slice(start).concat(pts.slice(0, start));

    this.arc = new Float32Array(n + 1);
    this.heading = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const [ax, ay] = this.points[i], [bx, by] = this.points[(i + 1) % n];
      this.arc[i + 1] = this.arc[i] + Math.hypot(bx - ax, by - ay);
      this.heading[i] = Math.atan2(by - ay, bx - ax);
    }
    this.length = this.arc[n];
    this.spacing = this.length / n;
    this.buildField();
  }

  // convex hull stretched over the world, long edges pushed in to make hairpins, then corners filleted
  static random(rng = Math.random) {
    const m = HALF_WIDTH + 60;
    for (let attempt = 0; attempt < 400; attempt++) {
      let poly = convexHull(Array.from({ length: 10 + (rng() * 12 | 0) }, () => [rng(), rng()]));
      const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      poly = poly.map(([x, y]) => [m + (x - x0) / (x1 - x0) * (WORLD_W - 2 * m), m + (y - y0) / (y1 - y0) * (WORLD_H - 2 * m)]);
      poly = poly.flatMap((p, i) => {
        const q = poly[(i + 1) % poly.length], len = Math.hypot(q[0] - p[0], q[1] - p[1]);
        if (len < 380 || rng() < 0.15) return [p];
        let nx = (p[1] - q[1]) / len, ny = (q[0] - p[0]) / len;
        if (nx * (WORLD_W / 2 - p[0]) + ny * (WORLD_H / 2 - p[1]) < 0) nx = -nx, ny = -ny;
        const dents = len > 800 ? 2 : 1, out = [p];
        for (let k = 0; k < dents; k++) {
          const t = (k + 0.3 + rng() * 0.4) / dents, push = (rng() * 1.3 - 0.15) * len / dents * 0.62;
          out.push([p[0] + (q[0] - p[0]) * t + nx * push, p[1] + (q[1] - p[1]) * t + ny * push]);
        }
        return out;
      });
      const raw = filletPolygon(simplifyPolygon(poly), rng);
      if (!raw) continue;
      const pts = resampleLoop(raw, SPACING);
      if (rng() < 0.5) pts.reverse();
      if (!trackProblem(pts)) return new Track(pts);
    }
    return new Track(Array.from({ length: 64 }, (_, i) => [
      WORLD_W / 2 + Math.cos(i / 64 * Math.PI * 2) * (WORLD_W / 2 - m),
      WORLD_H / 2 + Math.sin(i / 64 * Math.PI * 2) * (WORLD_H / 2 - m),
    ]));
  }

  static fromSketch(raw) {
    if (raw.length < 10) return { error: 'Draw a closed loop.' };
    let pts = resampleLoop(raw, SPACING);
    for (let pass = 0; pass < 10; pass++) pts = smoothLoop(pts);
    pts = smoothUntilDrivable(resampleLoop(pts, SPACING));
    const problem = trackProblem(pts);
    if (problem === 'sharp') return { error: 'Some corners are too tight — draw smoother curves.' };
    return problem ? { error: problem } : { track: new Track(pts) };
  }

  // grid of signed distance to the nearest wall (+ inside) and arc position, so sensors and contacts are O(1)
  buildField() {
    const cols = this.cols = WORLD_W / CELL + 1, rows = this.rows = WORLD_H / CELL + 1;
    const nearest = new Float32Array(cols * rows).fill(Infinity);
    this.field = new Float32Array(cols * rows);
    this.arcAt = new Float32Array(cols * rows);
    const pts = this.points, n = pts.length;

    for (let i = 0; i < n; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[(i + 1) % n];
      const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy || 1, segLen = this.arc[i + 1] - this.arc[i];
      const c0 = Math.max(0, Math.floor((Math.min(ax, bx) - REACH) / CELL));
      const c1 = Math.min(cols - 1, Math.ceil((Math.max(ax, bx) + REACH) / CELL));
      const r0 = Math.max(0, Math.floor((Math.min(ay, by) - REACH) / CELL));
      const r1 = Math.min(rows - 1, Math.ceil((Math.max(ay, by) + REACH) / CELL));
      for (let r = r0; r <= r1; r++)
        for (let c = c0; c <= c1; c++) {
          const px = c * CELL, py = r * CELL;
          const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
          const d = Math.hypot(px - ax - dx * t, py - ay - dy * t), idx = r * cols + c;
          if (d < nearest[idx]) {
            nearest[idx] = d;
            this.arcAt[idx] = this.arc[i] + segLen * t;
          }
        }
    }
    for (let i = 0; i < nearest.length; i++) this.field[i] = Math.max(OUTSIDE, HALF_WIDTH - nearest[i]);
  }

  distAt(x, y) {
    if (x < 0 || y < 0) return OUTSIDE;
    const gx = x / CELL, gy = y / CELL, ix = gx | 0, iy = gy | 0;
    if (ix >= this.cols - 1 || iy >= this.rows - 1) return OUTSIDE;
    const fx = gx - ix, fy = gy - iy, i = iy * this.cols + ix, f = this.field, below = i + this.cols;
    const top = f[i] + (f[i + 1] - f[i]) * fx;
    const bottom = f[below] + (f[below + 1] - f[below]) * fx;
    return top + (bottom - top) * fy;
  }

  progressAt(x, y) {
    return this.arcAt[Math.round(y / CELL) * this.cols + Math.round(x / CELL)];
  }

  headingAt(arc) {
    return this.heading[Math.floor(arc / this.length * this.heading.length) % this.heading.length];
  }

  // F1-style staggered two-column grid behind the start line
  gridSlot(slot) {
    const n = this.points.length, arc = this.length - (24 + (slot >> 1) * 32 + (slot & 1) * 16);
    const i = Math.floor(arc / this.length * n) % n, [x, y] = this.points[i], angle = this.heading[i];
    const lateral = slot & 1 ? 15 : -15;
    return { x: x - Math.sin(angle) * lateral, y: y + Math.cos(angle) * lateral, angle };
  }
}
// the generated tracks have no separate walls, surfaces or banking: the edge of the road is the wall
Object.assign(Track.prototype, { cell: CELL, outside: OUTSIDE, halfWidth: HALF_WIDTH, worldW: WORLD_W, worldH: WORLD_H, nascar: false });
Track.prototype.wallAt = Track.prototype.distAt;

// ---- NASCAR ovals (js/nascar-tracks.js), at real size: 1 unit = 0.25 m, so a stock car is 20 x 8 ----
const UNITS_PER_M = 4, FT = 0.3048, OVAL_CELL = 8, OVAL_SPACING = 8;
const MPH_PER_SPEED = 0.25 * 60 * 3600 / 1609.344;      // a speed of 1 unit/step in mph
// what lies across the track, from the infield out
const SURFACE = { racing: 0, apron: 1, grass: 2, sand: 3, pit: 4, beyond: 5 };
// infield grass reach where there's no inside wall, the grass strip before the pit wall, pit road width
const GRASS_M = 30, PIT_GAP_M = 7, PIT_ROAD_M = 14;

function decodeTrackPoints(b64) {
  const bytes = typeof Buffer !== 'undefined' ? Uint8Array.from(Buffer.from(b64, 'base64')) : Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const d = new Int16Array(bytes.buffer), pts = [];
  let x = 0, y = 0;
  for (let i = 0; i < d.length; i += 2) pts.push([(x += d[i]) / 10, (y += d[i + 1]) / 10]);
  return pts;
}

class OvalTrack {
  static byId = new Map();

  static get(id) {
    if (!OvalTrack.byId.has(id)) OvalTrack.byId.set(id, new OvalTrack(NASCAR_TRACKS.find(t => t.id === id)));
    return OvalTrack.byId.get(id);
  }

  constructor(def) {
    this.id = Track.nextId++;
    this.def = def;
    this.key = def.id;
    this.name = def.name;
    this.nascar = true;
    this.cell = OVAL_CELL;
    this.outside = OUTSIDE;
    this.plate = def.pkg === 'plate';
    const m = UNITS_PER_M;
    this.halfWidth = def.widthFt * FT * m / 2;
    this.apron = def.apronFt * FT * m;
    this.grip = def.surface === 'concrete' ? 0.97 : 1;
    // single-car qualifying pace, the yardstick for timing and the pace car
    this.refSpeed = def.pole.mph / MPH_PER_SPEED;
    this.paceSpeed = this.refSpeed * 0.33;
    const innerReach = def.insideWall ? this.halfWidth + this.apron : this.halfWidth + this.apron + GRASS_M * m;
    this.reachIn = innerReach + PIT_ROAD_M * m;
    this.reachOut = this.halfWidth + 6 * m;

    // centreline: metres -> units, resampled evenly, framed with room for the infield, walls and stands
    let pts = resampleLoop(decodeTrackPoints(def.points).map(([x, y]) => [x * m, y * m]), OVAL_SPACING);
    const margin = Math.max(this.reachIn, this.reachOut) + 60 * m;
    const x0 = Math.min(...pts.map(p => p[0])), y0 = Math.min(...pts.map(p => p[1]));
    pts = pts.map(([x, y]) => [x - x0 + margin, y - y0 + margin]);
    this.worldW = Math.ceil((Math.max(...pts.map(p => p[0])) + margin) / OVAL_CELL) * OVAL_CELL;
    this.worldH = Math.ceil((Math.max(...pts.map(p => p[1])) + margin) / OVAL_CELL) * OVAL_CELL;
    const n = pts.length;
    this.points = pts;
    this.arc = new Float32Array(n + 1);
    this.heading = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[(i + 1) % n];
      this.arc[i + 1] = this.arc[i] + Math.hypot(bx - ax, by - ay);
      this.heading[i] = Math.atan2(by - ay, bx - ax);
    }
    this.length = this.arc[n];
    this.spacing = this.length / n;
    this.buildBanking();
    this.buildField();
  }

  // Banking per centreline point: each corner gets its published angle (progressive corners a low and high
  // lane), blended in by curvature so turn-ins and exits are gradual; the frontstretch (where the line is) and
  // the backstretch keep their own. Corners are found as the stretches curving hardest.
  buildBanking() {
    const pts = this.points, n = pts.length, m = UNITS_PER_M, reach = Math.max(2, Math.round(60 * m / this.spacing));
    const raw = pts.map((_, i) => {
      const a = pts[(i - reach + n) % n], p = pts[i], b = pts[(i + reach) % n];
      const d = wrapAngle(Math.atan2(b[1] - p[1], b[0] - p[0]) - Math.atan2(p[1] - a[1], p[0] - a[0]));
      return Math.abs(d) / (Math.hypot(p[0] - a[0], p[1] - a[1]) + Math.hypot(b[0] - p[0], b[1] - p[1])) * 2;
    });
    // curvature averaged over 60 m so mapping wiggles don't make corners
    const span = Math.max(1, Math.round(30 * m / this.spacing)), kappa = raw.map((_, i) => {
      let sum = 0;
      for (let k = -span; k <= span; k++) sum += raw[(i + k + n) % n];
      return sum / (2 * span + 1);
    });
    this.curvature = Float32Array.from(kappa);
    const kmax = Math.max(...kappa), inTurn = kappa.map(k => k > kmax * 0.5);
    // contiguous corners, starting the scan on a straight so none is split at the wrap
    let regions = [];
    let s0 = inTurn.findIndex(t => !t);
    if (s0 < 0) s0 = 0;
    for (let k = 0; k < n; k++) {
      const i = (s0 + k) % n;
      if (inTurn[i] && !inTurn[(i - 1 + n) % n]) regions.push({ start: i, end: i });
      if (inTurn[i] && regions.length) regions.at(-1).end = i;
    }
    // one corner, not several: merge across short gaps, drop slivers
    const lenOf = (a, b) => ((b - a + n) % n) * this.spacing, gap = Math.min(100 * m, this.length * 0.06);
    for (let k = 0; k < regions.length - 1;) {
      if (lenOf(regions[k].end, regions[k + 1].start) < gap) regions.splice(k, 2, { start: regions[k].start, end: regions[k + 1].end });
      else k++;
    }
    if (regions.length > 1 && lenOf(regions.at(-1).end, regions[0].start) < gap) {
      regions[0] = { start: regions.at(-1).start, end: regions[0].end };
      regions.pop();
    }
    // a corner turns the car a long way; a tri-oval dogleg or a kink in a straight doesn't
    const turnAngle = r => {
      let sum = 0;
      for (let i = r.start; i !== (r.end + 1) % n; i = (i + 1) % n) sum += kappa[i] * this.spacing;
      return sum;
    };
    regions = regions.filter(r => lenOf(r.start, r.end) > Math.min(40 * m, this.length * 0.03) && turnAngle(r) > 1.05);
    const spec = this.def.banking, deg = Math.PI / 180, turns = spec.turns.map(t => Array.isArray(t) ? t : [t, t]);
    // corners in the order a car meets them after the line
    regions.sort((a, b) => a.start - b.start);
    const turnOf = r => turns[Math.min(turns.length - 1, Math.floor(r * turns.length / Math.max(1, regions.length)))];
    // the straight holding the start/finish line (tri-oval and all) is the frontstretch
    const inside = (r, i) => r.start <= r.end ? i >= r.start && i <= r.end : i >= r.start || i <= r.end;
    const inCorner = i => regions.some(r => inside(r, i)), front = new Uint8Array(n);
    for (let i = 0; i < n && !inCorner(i); i++) front[i] = 1;
    for (let i = n - 1; i > 0 && !inCorner(i); i--) front[i] = 1;
    this.bankLo = new Float32Array(n);
    this.bankHi = new Float32Array(n);
    // full corner banking inside a corner, ramping to the straight's over the transition beyond its ends
    const ramp = Math.min(80 * m, this.length * 0.06) / this.spacing;
    const beyond = (r, i) => Math.min((r.start - i + n) % n, (i - r.end + n) % n);
    for (let i = 0; i < n; i++) {
      let near = 0, best = Infinity;
      regions.forEach((r, j) => {
        const d = inside(r, i) ? 0 : beyond(r, i);
        if (d < best) [near, best] = [j, d];
      });
      const [lo, hi] = regions.length ? turnOf(near) : [spec.back, spec.back];
      const straight = front[i] ? spec.front : spec.back;
      const w = regions.length ? Math.max(0, 1 - best / ramp) : 0, t = w * w * (3 - 2 * w);
      this.bankLo[i] = (straight + (lo - straight) * t) * deg;
      this.bankHi[i] = (straight + (hi - straight) * t) * deg;
    }
    this.turnRegions = regions;
    // the frontstretch, where pit road runs along the inside behind the pit wall
    const frontLen = front.reduce((a, b) => a + b, 0) * this.spacing;
    this.pitZone = Math.min(frontLen * 0.42, this.length * 0.18);
    // run-off: sand traps in the infield grass at corner exits, where spinning cars end up
    this.sandAt = regions.map(r => this.arc[(r.end + Math.round(25 * UNITS_PER_M / this.spacing)) % n]);
  }

  // Grids, every cell: signed sideways offset from the centreline (+ toward the outside wall), distance to the
  // paved edge (what the sensors see), distance to a real wall (what cars hit), surface, and progress round the lap.
  buildField() {
    const cell = OVAL_CELL, cols = this.cols = Math.round(this.worldW / cell) + 1, rows = this.rows = Math.round(this.worldH / cell) + 1;
    const size = cols * rows, nearest = new Float32Array(size).fill(Infinity), lat = new Float32Array(size).fill(-Infinity);
    this.arcAt = new Float32Array(size);
    const pts = this.points, n = pts.length, reach = Math.max(this.reachIn, this.reachOut);
    for (let i = 0; i < n; i++) {
      const [ax, ay] = pts[i], [bx, by] = pts[(i + 1) % n];
      const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy || 1, segLen = this.arc[i + 1] - this.arc[i];
      const c0 = Math.max(0, Math.floor((Math.min(ax, bx) - reach) / cell)), c1 = Math.min(cols - 1, Math.ceil((Math.max(ax, bx) + reach) / cell));
      const r0 = Math.max(0, Math.floor((Math.min(ay, by) - reach) / cell)), r1 = Math.min(rows - 1, Math.ceil((Math.max(ay, by) + reach) / cell));
      for (let r = r0; r <= r1; r++)
        for (let c = c0; c <= c1; c++) {
          const px = c * cell, py = r * cell, t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
          const ex = px - ax - dx * t, ey = py - ay - dy * t, d = Math.sqrt(ex * ex + ey * ey), idx = r * cols + c;
          if (d < nearest[idx]) {
            nearest[idx] = d;
            // right of travel is the outside: (-dy, dx) . e
            lat[idx] = (ex * -dy + ey * dx) >= 0 ? d : -d;
            this.arcAt[idx] = this.arc[i] + segLen * t;
          }
        }
    }
    const hw = this.halfWidth, apron = this.apron, m = UNITS_PER_M, L = this.length;
    this.lat = lat;
    this.field = new Float32Array(size);
    this.wall = new Float32Array(size);
    this.surface = new Uint8Array(size);
    for (let idx = 0; idx < size; idx++) {
      const s = lat[idx], arc = this.arcAt[idx];
      if (s === -Infinity || s > this.reachOut || s < -this.reachIn) {
        this.field[idx] = this.wall[idx] = OUTSIDE;
        this.surface[idx] = SURFACE.beyond;
        continue;
      }
      const fromLine = Math.min(arc, L - arc), pits = fromLine < this.pitZone;
      const inner = this.def.insideWall ? hw + apron : pits ? hw + apron + PIT_GAP_M * m : hw + apron + GRASS_M * m;
      this.field[idx] = Math.max(OUTSIDE, Math.min(hw - s, hw + apron + s));
      this.wall[idx] = Math.max(OUTSIDE, Math.min(hw - s, inner + s));
      let surface = s > hw ? SURFACE.beyond : s >= -hw ? SURFACE.racing : s >= -hw - apron ? SURFACE.apron : s >= -inner ? SURFACE.grass : pits ? SURFACE.pit : SURFACE.beyond;
      if (surface === SURFACE.grass && !this.def.insideWall && !pits) {
        for (const at of this.sandAt) {
          const along = Math.abs(((arc - at) % L + L * 1.5) % L - L / 2);
          if (along < 30 * m && s < -hw - apron - 6 * m && s > -hw - apron - 22 * m) surface = SURFACE.sand;
        }
      }
      this.surface[idx] = surface;
    }
  }

  distAt(x, y) {
    if (x < 0 || y < 0) return OUTSIDE;
    const gx = x / OVAL_CELL, gy = y / OVAL_CELL, ix = gx | 0, iy = gy | 0;
    if (ix >= this.cols - 1 || iy >= this.rows - 1) return OUTSIDE;
    const fx = gx - ix, fy = gy - iy, i = iy * this.cols + ix, f = this.field, below = i + this.cols;
    const top = f[i] + (f[i + 1] - f[i]) * fx, bottom = f[below] + (f[below + 1] - f[below]) * fx;
    return top + (bottom - top) * fy;
  }

  wallAt(x, y) {
    if (x < 0 || y < 0) return OUTSIDE;
    const gx = x / OVAL_CELL, gy = y / OVAL_CELL, ix = gx | 0, iy = gy | 0;
    if (ix >= this.cols - 1 || iy >= this.rows - 1) return OUTSIDE;
    const fx = gx - ix, fy = gy - iy, i = iy * this.cols + ix, f = this.wall, below = i + this.cols;
    const top = f[i] + (f[i + 1] - f[i]) * fx, bottom = f[below] + (f[below + 1] - f[below]) * fx;
    return top + (bottom - top) * fy;
  }

  cellAt(x, y) {
    const c = Math.round(x / OVAL_CELL), r = Math.round(y / OVAL_CELL);
    return c < 0 || r < 0 || c >= this.cols || r >= this.rows ? -1 : r * this.cols + c;
  }

  surfaceAt(x, y) {
    const i = this.cellAt(x, y);
    return i < 0 ? SURFACE.beyond : this.surface[i];
  }

  lateralAt(x, y) {
    const i = this.cellAt(x, y);
    return i < 0 ? -Infinity : this.lat[i];
  }

  progressAt(x, y) {
    const i = this.cellAt(x, y);
    return i < 0 ? 0 : this.arcAt[i];
  }

  headingAt(arc) {
    return this.heading[Math.floor(arc / this.length * this.heading.length) % this.heading.length];
  }

  // banking under a car: the corner's angle at that lane (progressive corners steepen toward the wall); the
  // apron is nearly flat and the grass flat
  bankAt(arc, lateral) {
    const n = this.heading.length, i = ((Math.floor(arc / this.length * n) % n) + n) % n, hw = this.halfWidth;
    if (lateral < -hw - this.apron) return 0;
    const lane = Math.min(1, Math.max(0, (lateral + hw) / (2 * hw))), bank = this.bankLo[i] + (this.bankHi[i] - this.bankLo[i]) * lane;
    return lateral < -hw ? bank * 0.25 : bank;
  }

  // a point at a given distance round the lap and sideways offset (+ toward the outside wall)
  pointAt(arc, lateral) {
    const n = this.points.length, f = ((arc % this.length) + this.length) % this.length / this.spacing, i = Math.floor(f) % n, t = f - Math.floor(f);
    const [ax, ay] = this.points[i], [bx, by] = this.points[(i + 1) % n], h = this.heading[i];
    return [ax + (bx - ax) * t - Math.sin(h) * lateral, ay + (by - ay) * t + Math.cos(h) * lateral];
  }

  // NASCAR double-file rolling start: two lanes behind the line, leader on the inside
  gridSlot(slot) {
    const arc = this.length - (40 + (slot >> 1) * 52), lane = (slot & 1 ? 1 : -1) * this.halfWidth * 0.42;
    const [x, y] = this.pointAt(arc, lane);
    return { x, y, angle: this.headingAt(((arc % this.length) + this.length) % this.length) };
  }
}
