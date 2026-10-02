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
