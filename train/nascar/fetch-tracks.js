#!/usr/bin/env node
// Bakes js/nascar-tracks.js: every oval's real centreline from OpenStreetMap, or a parametric layout from its
// published dimensions when OSM has no clean loop. Run once (cached under train/nascar/osm-cache):
//   node train/nascar/fetch-tracks.js [--refresh] [--only daytona,bristol]
//
// OSM maps racing surfaces as highway=raceway ways, split wherever pit lanes and road courses join. The ways
// near a track become a graph; the simple cycle closest to the official length (and enclosing the most area,
// so the racing surface wins over pit road) is the track. It is smoothed, scaled to the exact official length,
// oriented so the infield is on the driver's left (NASCAR turns left), and started at the start/finish line.
// Track geometry (c) OpenStreetMap contributors, available under the Open Database License.
const fs = require('fs');
const path = require('path');
const { SPECS } = require('./specs');

const args = process.argv.slice(2), refresh = args.includes('--refresh');
const only = args.includes('--only') ? args[args.indexOf('--only') + 1].split(',') : null;
const CACHE = path.join(__dirname, 'osm-cache'), OUT = path.join(__dirname, '..', '..', 'js', 'nascar-tracks.js');
const ENDPOINTS = ['https://maps.mail.ru/osm/tools/overpass/api/interpreter', 'https://overpass-api.de/api/interpreter'];
const MILE = 1609.344, BAKE_SPACING = 3;
fs.mkdirSync(CACHE, { recursive: true });

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function overpass(query) {
  for (let attempt = 0; attempt < 6; attempt++) {
    const url = ENDPOINTS[attempt % ENDPOINTS.length];
    try {
      const res = await fetch(url, {
        method: 'POST', body: new URLSearchParams({ data: query }),
        headers: { 'User-Agent': 'neural-racers-track-builder/1.0 (personal project)', Accept: 'application/json' },
        signal: AbortSignal.timeout(90_000),
      });
      if (res.ok) return await res.json();
      console.log(`  ${url.split('/')[2]} answered ${res.status}, retrying`);
    } catch (e) {
      console.log(`  ${url.split('/')[2]} failed (${e.message}), retrying`);
    }
    await sleep(4000 * (attempt + 1));
  }
  throw new Error('Overpass unavailable');
}

async function osmFor(spec) {
  const file = path.join(CACHE, `${spec.id}.json`);
  if (!refresh && fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const radius = Math.round(Math.max(2500, spec.miles * MILE * 0.6));
  const data = await overpass(`[out:json][timeout:60];way["highway"="raceway"](around:${radius},${spec.lat},${spec.lon});out body;>;out skel qt;`);
  fs.writeFileSync(file, JSON.stringify(data));
  await sleep(2000);
  return data;
}

// ---- geometry helpers (metres, y down like the canvas, north up) ----
const project = (lat0, lon0) => ([lat, lon]) => [(lon - lon0) * Math.cos(lat0 * Math.PI / 180) * 111320, -(lat - lat0) * 110574];
const perimeter = pts => pts.reduce((s, p, i) => s + Math.hypot(pts[(i + 1) % pts.length][0] - p[0], pts[(i + 1) % pts.length][1] - p[1]), 0);
const signedArea = pts => pts.reduce((s, p, i) => s + p[0] * pts[(i + 1) % pts.length][1] - pts[(i + 1) % pts.length][0] * p[1], 0) / 2;
function resample(pts, spacing) {
  const n = pts.length, segs = pts.map((p, i) => Math.hypot(pts[(i + 1) % n][0] - p[0], pts[(i + 1) % n][1] - p[1]));
  const total = segs.reduce((a, b) => a + b, 0), count = Math.max(16, Math.round(total / spacing)), step = total / count, out = [];
  let seg = 0, along = 0;
  for (let k = 0; k < count; k++) {
    const target = k * step;
    while (along + segs[seg] < target) along += segs[seg++];
    const p = pts[seg], q = pts[(seg + 1) % n], t = segs[seg] ? (target - along) / segs[seg] : 0;
    out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
  }
  return out;
}
const smooth = (pts, passes) => {
  for (let k = 0; k < passes; k++) {
    const n = pts.length;
    pts = pts.map((b, i) => {
      const a = pts[(i - 1 + n) % n], c = pts[(i + 1) % n];
      return [(a[0] + 2 * b[0] + c[0]) / 4, (a[1] + 2 * b[1] + c[1]) / 4];
    });
  }
  return pts;
};
function fitLength(pts, metres) {
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const k = metres / perimeter(pts);
  return pts.map(([x, y]) => [cx + (x - cx) * k, cy + (y - cy) * k]);
}

// the loop of raceway ways closest to the official length, preferring the one enclosing the most area
function stitch(data, spec) {
  // only the oval: not pit road, nor the infield road courses that share its straights; where the oval's own
  // pieces carry road-course names (Indianapolis), fall back to everything but pit road
  const strict = stitchWays(data, spec, /pit|garage|paddock|access|service|road course|infield|roval|kart|drag/i);
  return strict && !strict.failed ? strict : stitchWays(data, spec, /pit|garage|paddock|access|service/i) || strict;
}

function stitchWays(data, spec, skip) {
  const nodes = new Map(data.elements.filter(e => e.type === 'node').map(e => [e.id, [e.lat, e.lon]]));
  const ways = data.elements.filter(e => e.type === 'way' && e.nodes?.length > 1).filter(w => {
    const t = w.tags || {};
    return !skip.test(`${t.name || ''} ${t.raceway || ''} ${t.service || ''} ${t.description || ''}`) && t.area !== 'yes';
  });
  if (!ways.length) return null;
  const uses = new Map();
  for (const w of ways) for (const id of new Set(w.nodes)) uses.set(id, (uses.get(id) || 0) + 1);
  const isVertex = (w, k) => k === 0 || k === w.nodes.length - 1 || uses.get(w.nodes[k]) > 1;
  const lat0 = spec.lat, lon0 = spec.lon, toXY = project(lat0, lon0);
  const edges = [];
  for (const w of ways) {
    let start = 0;
    for (let k = 1; k < w.nodes.length; k++) {
      if (!isVertex(w, k)) continue;
      const ids = w.nodes.slice(start, k + 1);
      if (ids.every(id => nodes.has(id))) {
        const xy = ids.map(id => toXY(nodes.get(id)));
        edges.push({ a: ids[0], b: ids.at(-1), ids, xy, len: perimeter(xy) - Math.hypot(xy.at(-1)[0] - xy[0][0], xy.at(-1)[1] - xy[0][1]) });
      }
      start = k;
    }
  }
  const adj = new Map();
  edges.forEach((e, i) => {
    for (const v of [e.a, e.b]) adj.set(v, [...(adj.get(v) || []), i]);
  });
  const target = spec.miles * MILE, cycles = [];
  let budget = 3_000_000;
  // every simple cycle through edges, each found once from its lowest-numbered edge
  for (let first = 0; first < edges.length && budget > 0; first++) {
    const e0 = edges[first];
    if (e0.a === e0.b) {
      cycles.push([{ e: first, fwd: true }]);
      continue;
    }
    const seen = new Set([e0.a, e0.b]);
    const walk = (v, len, route) => {
      if (--budget < 0 || len > target * 1.2) return;
      for (const i of adj.get(v) || []) {
        if (i <= first) {
          if (i === first) continue;
          else continue;
        }
        const e = edges[i], next = e.a === v ? e.b : e.a;
        if (next === e0.a) {
          if (len + e.len > target * 0.75) cycles.push([...route, { e: i, fwd: e.a === v }]);
          continue;
        }
        if (seen.has(next)) continue;
        seen.add(next);
        walk(next, len + e.len, [...route, { e: i, fwd: e.a === v }]);
        seen.delete(next);
      }
    };
    walk(e0.b, e0.len, [{ e: first, fwd: true }]);
  }
  const loops = cycles.map(route => {
    const pts = [];
    for (const { e, fwd } of route) {
      const xy = fwd ? edges[e].xy : edges[e].xy.slice().reverse();
      pts.push(...xy.slice(0, -1));
    }
    return { pts, len: perimeter(pts), area: Math.abs(signedArea(pts)), ways: route.length };
  }).filter(l => l.pts.length > 8);
  // official lengths are usually measured along a line near the inside edge, OSM traces the centre: allow 7%
  const close = loops.filter(l => Math.abs(l.len - target) / target < (spec.lengthTolerance ?? 0.07));
  if (!close.length) {
    const nearest = loops.sort((x, y) => Math.abs(x.len - target) - Math.abs(y.len - target))[0];
    return { failed: nearest ? `closest loop ${(nearest.len / MILE).toFixed(3)} mi` : `${ways.length} ways, no loop` };
  }
  // an oval turns through one full circle; a loop that detours down a spur (a pit-lane or road-course
  // connector and back) turns much more. Of the most oval-like loops, the one enclosing the most area.
  const turning = l => {
    const p = smooth(resample(l.pts, 5), 6), n = p.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const a = p[(i - 1 + n) % n], b = p[i], c = p[(i + 1) % n];
      const d = Math.atan2(c[1] - b[1], c[0] - b[0]) - Math.atan2(b[1] - a[1], b[0] - a[0]);
      sum += Math.abs(Math.atan2(Math.sin(d), Math.cos(d)));
    }
    return sum;
  };
  close.forEach(l => l.turning = turning(l));
  const calmest = Math.min(...close.map(l => l.turning));
  return close.filter(l => l.turning < calmest + 0.5).sort((x, y) => y.area - x.area)[0];
}

// ---- parametric fallback: a polygon of the layout's corners, filleted, scaled to the official length ----
function fillet(poly, radii) {
  const n = poly.length, out = [];
  for (let i = 0; i < n; i++) {
    const a = poly[(i - 1 + n) % n], p = poly[i], b = poly[(i + 1) % n];
    const inLen = Math.hypot(p[0] - a[0], p[1] - a[1]), outLen = Math.hypot(b[0] - p[0], b[1] - p[1]);
    const din = [(p[0] - a[0]) / inLen, (p[1] - a[1]) / inLen], dout = [(b[0] - p[0]) / outLen, (b[1] - p[1]) / outLen];
    const turn = Math.acos(Math.max(-1, Math.min(1, din[0] * dout[0] + din[1] * dout[1]))), side = Math.sign(din[0] * dout[1] - din[1] * dout[0]) || 1;
    const r = Math.min(radii[i], 0.45 * Math.min(inLen, outLen) / Math.max(1e-6, Math.tan(turn / 2)));
    const t = r * Math.tan(turn / 2), sx = p[0] - din[0] * t, sy = p[1] - din[1] * t;
    const cx = sx - din[1] * r * side, cy = sy + din[0] * r * side, a0 = Math.atan2(sy - cy, sx - cx), steps = Math.max(2, Math.ceil(turn * r / 4));
    for (let k = 0; k <= steps; k++) out.push([cx + Math.cos(a0 + side * turn * k / steps) * r, cy + Math.sin(a0 + side * turn * k / steps) * r]);
  }
  return out;
}
function parametric(spec) {
  // corner polygons laid out anticlockwise-on-screen-with-infield-left after orientation; units are arbitrary
  const big = 1000, R = r => Array(16).fill(r);
  const layouts = {
    oval: [[[0, 0], [big, 0], [big, 400], [0, 400]], R(200)],
    paperclip: [[[0, 0], [big, 0], [big, 300], [0, 300]], R(150)],
    tri: [[[0, 0], [big, 0], [big, 420], [big / 2, 520], [0, 420]], [200, 200, 200, 900, 200]],
    quad: [[[0, 0], [big, 0], [big, 420], [big * 0.7, 480], [big * 0.3, 480], [0, 420]], [200, 200, 200, 700, 700, 200]],
    d: [[[0, 0], [big, 0], [big + 60, 250], [big, 500], [0, 500], [-60, 250]], [220, 220, 400, 220, 220, 400]],
    triangle: [[[0, 0], [big, 0], [big * 0.42, 700]], [260, 260, 260]],
    rectangle: [[[0, 0], [big * 1.6, 0], [big * 1.6, 360], [0, 360]], R(90)],
    egg: [[[0, 0], [big, 0], [big, 380], [0, 330]], [170, 230, 230, 170]],
    dogleg: [[[0, 0], [big, 0], [big, 420], [big * 0.45, 420], [big * 0.3, 360], [0, 360]], [180, 220, 220, 300, 300, 180]],
  };
  const [poly, radii] = layouts[spec.shape] || layouts.oval;
  return fitLength(resample(fillet(poly, radii), 2), spec.miles * MILE);
}

// ---- start/finish: middle of the frontstretch ----
function curvatures(pts, reach = 12) {
  const n = pts.length;
  return pts.map((_, i) => {
    const a = pts[(i - reach + n) % n], p = pts[i], b = pts[(i + reach) % n];
    const h1 = Math.atan2(p[1] - a[1], p[0] - a[0]), h2 = Math.atan2(b[1] - p[1], b[0] - p[0]);
    const d = Math.atan2(Math.sin(h2 - h1), Math.cos(h2 - h1));
    return d / (Math.hypot(p[0] - a[0], p[1] - a[1]) + Math.hypot(b[0] - p[0], b[1] - p[1])) * 2;
  });
}
function startFinish(pts, sf) {
  const n = pts.length, kappa = curvatures(pts).map(Math.abs), kmax = Math.max(...kappa);
  const straight = kappa.map(k => k < kmax * 0.35);
  if (sf === 'bulge') {
    // principal axis of the loop; the frontstretch bulge sticks out furthest across it
    const cx = pts.reduce((s, p) => s + p[0], 0) / n, cy = pts.reduce((s, p) => s + p[1], 0) / n;
    let sxx = 0, syy = 0, sxy = 0;
    for (const [x, y] of pts) sxx += (x - cx) ** 2, syy += (y - cy) ** 2, sxy += (x - cx) * (y - cy);
    const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy), mx = -Math.sin(ang), my = Math.cos(ang);
    let best = 0, far = -Infinity;
    for (let i = 0; i < n; i++) if (straight[i]) {
      const m = Math.abs((pts[i][0] - cx) * mx + (pts[i][1] - cy) * my);
      if (m > far) [best, far] = [i, m];
    }
    return best;
  }
  // longest run of straight points, from its middle
  let bestStart = 0, bestLen = 0;
  for (let i = 0; i < n; i++) {
    if (!straight[i] || straight[(i - 1 + n) % n]) continue;
    let len = 0;
    while (len < n && straight[(i + len) % n]) len++;
    if (len > bestLen) [bestStart, bestLen] = [i, len];
  }
  return (bestStart + Math.floor(bestLen / 2)) % n;
}

function finish(pts, spec, source) {
  pts = fitLength(resample(pts, BAKE_SPACING), spec.miles * MILE);
  if (signedArea(pts) > 0) pts.reverse();
  const s = startFinish(pts, spec.sf);
  pts = pts.slice(s).concat(pts.slice(0, s));
  const x0 = Math.min(...pts.map(p => p[0])), y0 = Math.min(...pts.map(p => p[1]));
  // decimetre deltas, base64 Int16: small enough to ship to the browser
  const d = new Int16Array(pts.length * 2);
  let px = 0, py = 0;
  pts.forEach(([x, y], i) => {
    const qx = Math.round((x - x0) * 10), qy = Math.round((y - y0) * 10);
    d[2 * i] = qx - px;
    d[2 * i + 1] = qy - py;
    px = qx;
    py = qy;
  });
  return { ...spec, source, points: Buffer.from(d.buffer).toString('base64') };
}

(async () => {
  const baked = fs.existsSync(OUT) ? new Function(`${fs.readFileSync(OUT, 'utf8')}; return NASCAR_TRACKS;`)() : [];
  const out = [];
  for (const spec of SPECS) {
    if (only && !only.includes(spec.id)) {
      const kept = baked.find(t => t.id === spec.id);
      if (kept) {
        out.push({ ...spec, source: kept.source, points: kept.points });
        continue;
      }
    }
    let line = `${spec.short.padEnd(14)} ${spec.miles.toFixed(3)} mi: `;
    try {
      const data = await osmFor(spec), loop = stitch(data, spec);
      if (loop && !loop.failed) {
        // OSM ways are polylines with nodes tens of metres apart, and doglegs are often drawn as a sharp
        // kink: smooth over ~2% of a lap (sigma), which keeps the long corners' radii and turns kinks into the
        // gentle arcs they really are. n passes of [1,2,1]/4 at 2 m spacing is sigma = 2 * sqrt(n / 2).
        const sigma = spec.miles * MILE * 0.022, passes = Math.max(60, Math.round(2 * (sigma / 2) ** 2));
        const pts = smooth(resample(loop.pts, 2), passes);
        out.push(finish(pts, spec, 'osm'));
        line += `OpenStreetMap loop of ${loop.ways} way piece(s), ${(loop.len / MILE).toFixed(3)} mi before scaling`;
      } else {
        out.push(finish(parametric(spec), spec, 'parametric'));
        line += `parametric ${spec.shape} (OSM: ${loop?.failed || 'nothing'})`;
      }
    } catch (e) {
      out.push(finish(parametric(spec), spec, 'parametric'));
      line += `parametric ${spec.shape} (${e.message})`;
    }
    console.log(line);
  }
  fs.writeFileSync(OUT, `// generated by train/nascar/fetch-tracks.js from train/nascar/specs.js; don't edit.\n` +
    `// Track centrelines (c) OpenStreetMap contributors, Open Database License (openstreetmap.org/copyright).\n` +
    `const NASCAR_TRACKS = ${JSON.stringify(out)};\n`);
  console.log(`\n${out.filter(t => t.source === 'osm').length} of ${out.length} tracks from OpenStreetMap -> js/nascar-tracks.js (${(fs.statSync(OUT).size / 1024).toFixed(0)} KB)`);
})();
