const CAR_BOUNCE = 0.3, CAR_FRICTION = 0.25;
const GAP_MARK = 40, GAP_OFFSET = 12;

// aerodynamic effects of `follow` running behind or alongside `lead` (the lead car's size and wake)
function airflow(lead, follow) {
  const dx = follow.x - lead.x, dy = follow.y - lead.y, c = lead.c, s = lead.s;
  if (c * follow.c + s * follow.s < 0.7) return;
  const behind = -(dx * c + dy * s), lateral = Math.abs(-dx * s + dy * c);
  const L = lead.spec.len, W = lead.spec.wid, wake = lead.spec.draftLen;

  // side draft: a nose alongside the leader's rear quarter pulls air off its flank and slows it, and takes the
  // air off its spoiler too, so the car being side-drafted goes loose
  if (behind > L * 0.2 && behind < L && lateral > W * 0.9 && lateral < W * 2.3) {
    const strength = 1 - (lateral - W * 0.9) / (W * 1.4);
    if (strength > lead.sideDrafted) lead.sideDrafted = strength;
    if (strength > lead.airOff) lead.airOff = strength;
  }
  // two wide: overlapping lengthwise, door to door, neither car is in clean air and both push more of it
  // (this runs for both orders of every pair, so each of the two picks it up)
  if (Math.abs(behind) < L * 0.9 && lateral > W * 0.8 && lateral < W * 2.4) {
    const strength = (1 - Math.abs(behind) / L) * (1 - (lateral - W * 0.8) / (W * 1.6));
    if (strength > follow.twoWide) follow.twoWide = strength;
  }

  // slipstream: the wake cuts the follower's drag, strongest close behind and on the centreline;
  // only nose-to-tail: once the cars overlap lengthwise they're alongside, not in each other's wake
  if (behind < L * 0.95 || behind > wake) return;
  const width = W * 0.8 + behind * 0.07;
  if (lateral > width) return;
  const leadSpeed = Math.sqrt(lead.vx * lead.vx + lead.vy * lead.vy), off = lateral / width, centred = 1 - off * off;
  const strength = (1 - behind / wake) * centred * Math.min(1, leadSpeed / lead.spec.wakeSpeed);
  if (strength > follow.draft) follow.draft = strength;
  if (strength > lead.tow) lead.tow = strength;
  // tandem push: right on the bumper, the follower fills the leader's low-pressure wake and speeds it up
  // too (and a bump hands it the pusher's momentum outright), so the pusher is towing the leader to the
  // line: to win from there it has to pull out, losing the draft into the side draft and the two-wide
  // drag above. Tucked that close it also takes the air off the leader's spoiler: a push in a corner gets
  // the leader loose.
  const bumper = clamp(2 - behind / L, 0, 1) * centred;
  if (bumper > lead.pushed) lead.pushed = bumper;
  if (bumper > follow.tailing) follow.tailing = bumper;
  if (bumper > lead.airOff) lead.airOff = bumper;
}

// ---- contact between two bodies that are exactly their outlines (stock cars) ----
// Each is a core rectangle grown by its corner radius, so the gap between two cars is the gap between their
// cores less both radii: the closest vertex-to-edge pair while the cores are apart, the shallowest separating
// axis once they overlap (a heavy hit). Faces that meet flat (nose to tail, door to door) touch along a strip,
// so the contact point is the middle of it: a straight bump pushes straight, it doesn't swing the car ahead
// aside the way two round ends glancing off each other did.
const CORE_A = new Float64Array(8), CORE_B = new Float64Array(8), CORNER_X = [1, -1, -1, 1], CORNER_Y = [1, 1, -1, -1];
const hit = { depth: 0, nx: 0, ny: 0, px: 0, py: 0 };
function coreOf(car, out) {
  const { cx, cy } = car.spec.box;
  for (let k = 0; k < 4; k++) {
    const lx = CORNER_X[k] * cx, ly = CORNER_Y[k] * cy;
    out[2 * k] = car.x + car.c * lx - car.s * ly;
    out[2 * k + 1] = car.y + car.s * lx + car.c * ly;
  }
}
// middle of a polygon's extreme vertices along (ux, uy): one vertex, or the middle of an edge lying flat
function support(core, ux, uy, sign) {
  let best = -Infinity, x = 0, y = 0, n = 0;
  for (let k = 0; k < 8; k += 2) {
    const p = sign * (core[k] * ux + core[k + 1] * uy);
    if (p > best + 1e-3) [best, x, y, n] = [p, core[k], core[k + 1], 1];
    else if (p > best - 1e-3) [x, y, n] = [x + core[k], y + core[k + 1], n + 1];
  }
  return [x / n, y / n];
}
function boxContact(a, b) {
  coreOf(a, CORE_A);
  coreOf(b, CORE_B);
  const reach = a.spec.box.r + b.spec.box.r;
  // the shallowest overlap over each car's two axes: positive on all of them means the cores interpenetrate
  let least = Infinity, ax = 0, ay = 0;
  for (const [ux, uy] of [[a.c, a.s], [-a.s, a.c], [b.c, b.s], [-b.s, b.c]]) {
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (let k = 0; k < 8; k += 2) {
      const pa = CORE_A[k] * ux + CORE_A[k + 1] * uy, pb = CORE_B[k] * ux + CORE_B[k + 1] * uy;
      if (pa < a0) a0 = pa;
      if (pa > a1) a1 = pa;
      if (pb < b0) b0 = pb;
      if (pb > b1) b1 = pb;
    }
    const overlap = Math.min(a1, b1) - Math.max(a0, b0);
    if (overlap < least) [least, ax, ay] = [overlap, ux, uy];
  }
  if (least > 0) {
    if ((b.x - a.x) * ax + (b.y - a.y) * ay < 0) [ax, ay] = [-ax, -ay];
    const [qx, qy] = support(CORE_A, ax, ay, 1), [rx, ry] = support(CORE_B, ax, ay, -1);
    return Object.assign(hit, { depth: least + reach, nx: ax, ny: ay, px: (qx + rx) / 2, py: (qy + ry) / 2 });
  }
  // cores apart: every vertex of each against every edge of the other. The nearest pair gives the gap. The
  // push is square to the face when a corner meets the flat of an edge (the closest point inside the edge),
  // and along the line between them only corner to corner. Pairs within 0.25 units (6 cm) of the nearest
  // are the same contact strip (bumpers and doors give a little), and its middle is where the push lands.
  let gap = Infinity, face = null;
  const pairs = [];
  for (const [from, to, flip] of [[CORE_A, CORE_B, 1], [CORE_B, CORE_A, -1]])
    for (let v = 0; v < 8; v += 2)
      for (let e = 0; e < 8; e += 2) {
        const x1 = to[e], y1 = to[e + 1], x2 = to[(e + 2) % 8], y2 = to[(e + 3) % 8], ex = x2 - x1, ey = y2 - y1;
        const along = ((from[v] - x1) * ex + (from[v + 1] - y1) * ey) / (ex * ex + ey * ey), t = Math.max(0, Math.min(1, along));
        const qx = x1 + ex * t, qy = y1 + ey * t, d = dhypot(qx - from[v], qy - from[v + 1]);
        if (d > reach) continue;
        // the points on a's surface and b's, and the face's square-on direction from a toward b
        const ax1 = flip > 0 ? from[v] : qx, ay1 = flip > 0 ? from[v + 1] : qy, bx1 = flip > 0 ? qx : from[v], by1 = flip > 0 ? qy : from[v + 1];
        const el = dhypot(ex, ey);
        let fx = -ey / el, fy = ex / el;
        if (fx * (b.x - a.x) + fy * (b.y - a.y) < 0) [fx, fy] = [-fx, -fy];
        const pair = { d, ax: ax1, ay: ay1, bx: bx1, by: by1, flat: along > 0 && along < 1, fx, fy };
        pairs.push(pair);
        if (d < gap) gap = d;
      }
  if (gap >= reach) return null;
  let px = 0, py = 0, n = 0, corner = null;
  for (const p of pairs) {
    if (p.d > gap + 0.25) continue;
    px += (p.ax + p.bx) / 2;
    py += (p.ay + p.by) / 2;
    n++;
    if (p.flat && (!face || p.d < face.d)) face = p;
    if (!corner || p.d < corner.d) corner = p;
  }
  let nx, ny;
  if (face) [nx, ny] = [face.fx, face.fy];
  else {
    const len = dhypot(corner.bx - corner.ax, corner.by - corner.ay);
    [nx, ny] = len > 1e-9 ? [(corner.bx - corner.ax) / len, (corner.by - corner.ay) / len] : [corner.fx, corner.fy];
  }
  return Object.assign(hit, { depth: reach - gap, nx, ny, px: px / n, py: py / n });
}

function collideCars(a, b, events) {
  const dx = b.x - a.x, dy = b.y - a.y, reach = (a.spec.len + b.spec.len) / 2 + 2;
  if (dx * dx + dy * dy > reach * reach) return;
  let depth = 0, nx = 0, ny = 0, px = 0, py = 0;
  if (a.spec.box && b.spec.box) {
    const contact = boxContact(a, b);
    if (!contact) return;
    ({ depth, nx, ny, px, py } = contact);
  } else {
    // arcade cars: a chain of circles along each car
    const touch = a.spec.bodyR + b.spec.bodyR;
    for (const oa of a.spec.bodyX)
      for (const ob of b.spec.bodyX) {
        const ax = a.x + a.c * oa, ay = a.y + a.s * oa, bx = b.x + b.c * ob, by = b.y + b.s * ob;
        const ex = bx - ax, ey = by - ay, dist = Math.sqrt(ex * ex + ey * ey), overlap = touch - dist;
        if (overlap > depth && dist > 0) {
          depth = overlap;
          nx = ex / dist;
          ny = ey / dist;
          px = (ax + bx) / 2;
          py = (ay + by) / 2;
        }
      }
  }
  if (!depth) return;
  a.x -= nx * depth / 2;
  a.y -= ny * depth / 2;
  b.x += nx * depth / 2;
  b.y += ny * depth / 2;

  const rax = px - a.x, ray = py - a.y, rbx = px - b.x, rby = py - b.y;
  const rvx = b.vx - b.spin * rby - (a.vx - a.spin * ray), rvy = b.vy + b.spin * rbx - (a.vy + a.spin * rax);
  const vn = rvx * nx + rvy * ny;
  if (vn >= 0) return;
  const ran = rax * ny - ray * nx, rbn = rbx * ny - rby * nx, bounce = vn < -RESTING ? CAR_BOUNCE : 0;
  const Ia = a.spec.inertia, Ib = b.spec.inertia, same = Ia === Ib;
  const j = -(1 + bounce) * vn / (2 + (same ? (ran * ran + rbn * rbn) / Ia : ran * ran / Ia + rbn * rbn / Ib));
  const tx = -ny, ty = nx, rat = rax * ty - ray * tx, rbt = rbx * ty - rby * tx;
  const jt = clamp(-(rvx * tx + rvy * ty) / (2 + (same ? (rat * rat + rbt * rbt) / Ia : rat * rat / Ia + rbt * rbt / Ib)), -CAR_FRICTION * j, CAR_FRICTION * j);
  a.vx -= j * nx + jt * tx;
  a.vy -= j * ny + jt * ty;
  a.spin -= (ran * j + rat * jt) / Ia;
  b.vx += j * nx + jt * tx;
  b.vy += j * ny + jt * ty;
  b.spin += (rbn * j + rbt * jt) / Ib;
  // b lies in direction n from a, and a in direction -n from b
  const alongA = rax * a.c + ray * a.s, alongB = rbx * b.c + rby * b.s;
  const zoneA = a.takeHit(nx, ny, j, Math.abs(jt), 'car', alongA), zoneB = b.takeHit(-nx, -ny, j, Math.abs(jt), 'car', alongB);
  a.contactAt[zoneA] += j;
  b.contactAt[zoneB] += j;
  // door to door: both flanks (a nose into someone's rear quarter is an attack, not leaning on them)
  if (zoneA === 'side' && zoneB === 'side') a.rubbing = b.rubbing = true;
  if (events && j > HIT_THRESHOLD.car) events.push({ x: px, y: py, power: j, cars: true });
}

const PASS_CHECK = 30;

class Heat {
  // brains in grid order, null for a human. opts.cars: 'normal' or 'stock' for everyone, or one per car.
  // On a NASCAR oval, race control runs the flags; opts.stages adds stage breaks, opts.practice short cautions.
  constructor(track, brains, laps, opts = {}) {
    this.track = track;
    this.laps = laps;
    const carsOf = slot => Array.isArray(opts.cars) ? opts.cars[slot] : opts.cars;
    this.cars = brains.map((brain, slot) => new Car(brain, track, slot, laps, specFor(carsOf(slot), track)));
    this.order = [];
    this.step = 0;
    this.finishers = 0;
    this.marks = [];
    this.events = null;
    this.maxSteps = this.deadline = Math.ceil(laps * track.length / 2.2) + 300;
    this.control = null;
    if (track.nascar) {
      // timed from real qualifying pace, with room for cautions (race control adds more when they come)
      this.maxSteps = this.deadline = Math.ceil(laps * track.length / (track.refSpeed * 0.5)) + 1200;
      this.control = new RaceControl(this, opts);
    }
  }

  get done() {
    return this.cars.every(car => !car.running);
  }

  get over() {
    return this.step >= this.deadline || this.done;
  }

  tick() {
    const { track, cars, step, control } = this, live = cars.filter(car => car.running);
    // red flag: everything stops where it is
    if (control && control.hold(live)) {
      this.step++;
      return;
    }
    for (const car of live) {
      car.draft = car.tow = car.pushed = car.sideDrafted = car.tailing = car.airOff = car.twoWide = 0;
      car.c = dcos(car.angle);
      car.s = dsin(car.angle);
    }
    for (const a of live) for (const b of live) if (a !== b) airflow(a, b);
    if (step % DECIDE_EVERY === 0) {
      const order = live.slice().sort((a, b) => (b.progress + b.bonus) - (a.progress + a.bonus));
      order.forEach((car, i) => car.position = i / Math.max(1, order.length - 1));
      // time spent leading the race (until the winner takes the flag)
      if (order.length > 1 && !this.finishers) order[0].ledSteps += DECIDE_EVERY;
      if (control) control.update(order, step);
      for (const car of live) if (car.brain && !car.paced) {
        car.sense(track, live);
        car.decide();
      }
    }
    if (control) control.steer(live);
    for (const car of live) {
      if (car.paced) car.drive(car.auto[0], car.auto[1]);
      else car.brain ? car.drive(car.action[0], car.action[1]) : car.drive(car.manualSteer, car.manualThrottle);
    }
    for (const car of live) car.hitWalls(track, this.events);
    for (let i = 0; i < live.length; i++)
      for (let j = i + 1; j < live.length; j++) collideCars(live[i], live[j], this.events);
    for (const car of live) if (car.rubbing) {
      car.sideSteps++;
      car.rubbing = false;
    }
    for (const car of live) {
      car.advance(track, step);
      if (car.finished) car.place = ++this.finishers;
      // when the race leader passed each point, so every car's gap to it is a lookup
      while (car.progress + car.bonus >= (this.marks.length - GAP_OFFSET) * GAP_MARK) this.marks.push(step);
    }
    if (control) control.after(live, step);
    // like a real chequered flag: once the winner is home, everyone gets about one more lap to finish
    else if (this.finishers && this.deadline === this.maxSteps) this.deadline = Math.round(step * (1 + 1 / this.laps));
    if (step % PASS_CHECK === 0) this.countPasses(live);
    this.step++;
  }

  // a pass: a car that was behind a still-running rival at the last check is now ahead of it
  countPasses(live) {
    const order = live.slice().sort((a, b) => (b.progress + b.bonus) - (a.progress + a.bonus)), before = new Map(this.order.map((car, i) => [car, i]));
    for (let i = 0; i < order.length; i++)
      for (let j = i + 1; j < order.length; j++)
        if (before.get(order[i]) > before.get(order[j])) {
          order[i].overtakes++;
          order[j].passedBy++;
        }
    this.order = order;
  }

  gap(car) {
    const mark = Math.floor((car.progress + car.bonus) / GAP_MARK) + GAP_OFFSET;
    return mark >= 0 && mark < this.marks.length ? (this.step - this.marks[mark]) / 60 : 0;
  }

  // running order: finishers by place, then everyone else by distance covered
  standings() {
    return this.cars.slice().sort((a, b) =>
      (a.finished && b.finished) ? a.place - b.place
        : a.finished !== b.finished ? b.finished - a.finished
          : a.retired !== b.retired ? a.retired - b.retired
            : (b.progress + b.bonus) - (a.progress + a.bonus));
  }
}
