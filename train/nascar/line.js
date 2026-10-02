// The racing line round an oval and what a perfect driver does on it: used to fit the stock-car constants to
// real pole speeds (calibrate.js) and to drive scripted cars in the tests (test-nascar.js).
const { E } = require('../lib');
const G = E.G;

// minimum curvature inside the lanes
function racingLine(track) {
  const pts = track.points, n = pts.length, hw = track.halfWidth, lo = -hw + 5, hi = hw - 6;
  const off = new Float32Array(n), nx = new Float32Array(n), ny = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    nx[i] = -Math.sin(track.heading[i]);
    ny[i] = Math.cos(track.heading[i]);
  }
  const at = i => [pts[i][0] + nx[i] * off[i], pts[i][1] + ny[i] * off[i]];
  // long scales first: straightening a dogleg means swinging out hundreds of metres before it
  for (const k of [256, 128, 64, 32, 16, 8, 4, 2, 1].filter(k => k < n / 4))
    for (let it = 0; it < 150; it++)
      for (let i = 0; i < n; i++) {
        const a = at((i - k + n) % n), b = at((i + k) % n), p = at(i);
        const mx = (a[0] + b[0]) / 2 - p[0], my = (a[1] + b[1]) / 2 - p[1];
        off[i] = Math.max(lo, Math.min(hi, off[i] + 0.5 * (mx * nx[i] + my * ny[i])));
      }
  const line = Array.from({ length: n }, (_, i) => at(i));
  // signed curvature, positive turning left (toward the infield), over +-10 m
  const reach = 5, kappa = new Float32Array(n), ds = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = line[(i - reach + n) % n], p = line[i], b = line[(i + reach) % n], q = line[(i + 1) % n];
    const d = Math.atan2(Math.sin(Math.atan2(b[1] - p[1], b[0] - p[0]) - Math.atan2(p[1] - a[1], p[0] - a[0])), Math.cos(Math.atan2(b[1] - p[1], b[0] - p[0]) - Math.atan2(p[1] - a[1], p[0] - a[0])));
    kappa[i] = -d / (Math.hypot(p[0] - a[0], p[1] - a[1]) + Math.hypot(b[0] - p[0], b[1] - p[1])) * 2;
    ds[i] = Math.hypot(q[0] - p[0], q[1] - p[1]);
  }
  const theta = Float32Array.from(off, (o, i) => track.bankAt(track.arc[i], o));
  return { off, kappa, ds, line, theta };
}

// the cornering limit and lap time a perfect driver gets out of a spec on a track
function lapTime(track, line, P) {
  const n = line.kappa.length, mech = P.grip * track.grip, vTop = Math.cbrt(P.power / P.aero) * 1.02;
  // as in Car.drive: the bank's inward push, and tyre load (with load sensitivity) plus downforce as grip
  const vmax = new Float64Array(n), latUse = (v, i) => {
    const th = line.theta[i], k = line.kappa[i], ac = v * v * k, sb = Math.sin(th), cb = Math.cos(th);
    const push = (G * cb + ac * sb) * sb, cap = mech * Math.max(0.3, cb * (cb + ac / G * sb)) ** P.loadSens + P.downforce * v * v;
    return { need: Math.abs(ac - push), cap };
  };
  for (let i = 0; i < n; i++) {
    let lo = 0, hi = vTop;
    if (latUse(hi, i).need <= latUse(hi, i).cap) vmax[i] = vTop;
    else {
      for (let it = 0; it < 40; it++) {
        const mid = (lo + hi) / 2, u = latUse(mid, i);
        if (u.need <= u.cap) lo = mid;
        else hi = mid;
      }
      vmax[i] = lo;
    }
  }
  const v = Float64Array.from(vmax);
  for (let lap = 0; lap < 2; lap++)
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, { need, cap } = latUse(v[i], i), room = Math.sqrt(Math.max(0, 1 - (need / cap) ** 2));
      const drive = Math.min(P.traction, P.power / Math.max(v[i], 1), cap * 0.5 * room);
      const a = drive - P.aero * v[i] * v[i] - P.roll * v[i];
      v[j] = Math.min(vmax[j], Math.sqrt(Math.max(0.01, v[i] * v[i] + 2 * a * line.ds[i])));
    }
  for (let lap = 0; lap < 2; lap++)
    for (let i = n - 1; i >= 0; i--) {
      const j = (i + 1) % n, { need, cap } = latUse(v[i], i), room = Math.sqrt(Math.max(0, 1 - (need / cap) ** 2));
      const b = Math.min(P.brake, cap * room) + P.aero * v[j] * v[j] + P.roll * v[j];
      v[i] = Math.min(v[i], Math.sqrt(v[j] * v[j] + 2 * b * line.ds[i]));
    }
  let steps = 0;
  for (let i = 0; i < n; i++) steps += line.ds[i] / ((v[i] + v[(i + 1) % n]) / 2);
  return { steps, mph: track.length / steps * E.MPH_PER_SPEED, v };
}

// a scripted driver for any car in a heat: the racing line (or a lane offset from it) at a share of the
// perfect speed, steered by race control's autopilot
function scripted(track, P) {
  const line = racingLine(track), perfect = lapTime(track, line, P), n = line.kappa.length;
  return (heat, car, { pace = 0.95, lane = null } = {}) => {
    const i = Math.floor(((car.lastArc % track.length) + track.length) % track.length / track.length * n) % n;
    const ahead = (i + Math.round(60 / track.spacing)) % n, speed = Math.hypot(car.vx, car.vy);
    // brake for the slowest point in the next half second, not just the one in front of the bumper
    let target = Infinity;
    for (let k = 0, reach = Math.round((60 + speed * 30) / track.spacing); k <= reach; k++) target = Math.min(target, perfect.v[(i + k) % n]);
    heat.control.autodrive(car, lane ?? line.off[ahead], target * pace);
    car.manualSteer = car.auto[0];
    car.manualThrottle = car.auto[1];
  };
}

module.exports = { racingLine, lapTime, scripted };
