const CAR_BOUNCE = 0.3, CAR_FRICTION = 0.25;
const GAP_MARK = 40, GAP_OFFSET = 12;

// aerodynamic effects of `follow` running behind or alongside `lead` (the lead car's size and wake)
function airflow(lead, follow) {
  const dx = follow.x - lead.x, dy = follow.y - lead.y, c = lead.c, s = lead.s;
  if (c * follow.c + s * follow.s < 0.7) return;
  const behind = -(dx * c + dy * s), lateral = Math.abs(-dx * s + dy * c);
  const L = lead.spec.len, W = lead.spec.wid, wake = lead.spec.draftLen;

  // side draft: a nose alongside the leader's rear quarter pulls air off its flank and slows it
  if (behind > L * 0.2 && behind < L && lateral > W * 0.9 && lateral < W * 2.3) {
    const strength = 1 - (lateral - W * 0.9) / (W * 1.4);
    if (strength > lead.sideDrafted) lead.sideDrafted = strength;
  }

  // slipstream: the wake cuts the follower's drag, strongest close behind and on the centreline;
  // only nose-to-tail: once the cars overlap lengthwise they're alongside, not in each other's wake
  if (behind < L * 0.95 || behind > wake) return;
  const width = W * 0.8 + behind * 0.07;
  if (lateral > width) return;
  const leadSpeed = Math.sqrt(lead.vx * lead.vx + lead.vy * lead.vy), centred = 1 - (lateral / width) ** 2;
  const strength = (1 - behind / wake) * centred * Math.min(1, leadSpeed / lead.spec.wakeSpeed);
  if (strength > follow.draft) follow.draft = strength;
  if (strength > lead.tow) lead.tow = strength;
  // tandem push: right on the bumper, the follower fills the leader's low-pressure wake and speeds it
  // up too, so glueing yourself to a car's bumper helps it as much as it helps you
  const bumper = clamp(2 - behind / L, 0, 1) * centred;
  if (bumper > lead.pushed) lead.pushed = bumper;
  if (bumper > follow.tailing) follow.tailing = bumper;
}

function collideCars(a, b, events) {
  const dx = b.x - a.x, dy = b.y - a.y, reach = (a.spec.len + b.spec.len) / 2 + 2;
  if (dx * dx + dy * dy > reach ** 2) return;
  let depth = 0, nx = 0, ny = 0, px = 0, py = 0;
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
      car.draft = car.tow = car.pushed = car.sideDrafted = car.tailing = 0;
      car.c = Math.cos(car.angle);
      car.s = Math.sin(car.angle);
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
