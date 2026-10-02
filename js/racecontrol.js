// Race control for every race on a NASCAR oval: the flags and the rules that go with them.
//   green: racing. Starts and restarts are double-file at pace speed; the leader goes in the restart zone,
//          and nobody may pass before the line (jumping the restart is a black flag).
//   yellow: a spin, a stopped car or a wreck. The field is frozen as it ran, the pace car comes out, and the
//          field lines up single file behind it (cars on autopilot, like drivers just following). The first
//          lapped car gets its lap back (lucky dog). After a caution lap, a double-file restart. Laps run
//          under yellow don't count (as at many short tracks), so a caution bunches the field without eating
//          a sprint race's green distance. Short races (practice, duels, under 5 laps) have no cautions at
//          all: a spin costs only the car that spun, and a car that stops is towed.
//   white: the leader's final lap. checkered: the finish; everyone else completes the lap they're on.
//   overtime: a race never ends under yellow; a caution on the final lap means a green-white-checkered,
//          and after two of those attempts the next caution ends it under yellow, order frozen.
//   red: a big wreck stops the race; then it resumes under yellow.
//   black: a stop-and-go (held to pit-road speed for a pass-through's length) for passing below the double
//          yellow line at the plate tracks, or jumping a restart. red-and-black: a car too damaged to go on
//          is parked. blue-and-yellow: a lapped car about to be passed by the leaders.
//   stages: in tournament races, stage breaks end on a green-checkered and a caution; stage points shown.
// Sprint format: no pits, fuel or tyre wear, so there is no wave-around (it exists for lapped cars that stay
// out while the leaders pit).
const RC = {
  check: 10,           // steps between hazard checks
  gapRows: 52,         // single-file spacing under caution (units, about 13 m)
  stoppedFor: 120,     // a car stopped this long brings out the yellow
  towAfter: 600,       // and one stopped this long is towed off, out of the race
  redWindow: 300, redCars: 4, redHold: 240,
  parkDamage: 1.25,    // downforce lost front + rear (each 0..1) that parks a car
  zone: 0.06,          // restart zone: this share of the lap before the line
  cautionLaps: 1, maxCautionLaps: 3,
  minCautionLaps: 5,   // shorter races stay green
  overtimeAttempts: 2,
};

class RaceControl {
  constructor(heat, opts = {}) {
    this.heat = heat;
    this.track = heat.track;
    const laps = heat.laps;
    this.cautionsOn = !opts.practice && laps >= RC.minCautionLaps;
    // training skips the pace laps: the field closes up two-wide behind the leader and restarts at once (same
    // order, lucky dog and overtime rules), because a caution lap behind the pace car teaches nothing and costs minutes
    this.fast = !!opts.fastCaution;
    this.stages = opts.stages && laps >= 6 ? [Math.round(laps / 3), Math.round(laps * 2 / 3)] : [];
    this.stagePoints = new Map();
    this.flag = 'green';
    this.phase = 'start';          // start | green | caution | onetogo | red | checkered
    this.paceArc = 0;
    this.paceRun = 0;
    this.paceCar = null;
    this.order = [];               // the frozen running order under caution
    this.restartOrder = null;      // standings at the green, for the jumped-restart check
    this.leaderLine = 0;           // the leader's completed laps when the green came out
    this.overtime = 0;             // green-white-checkered attempts so far
    this.log = [];                 // { step, flag, text } for the app's banners
    this.cautions = 0;
    this.cautionAt = -Infinity;
    this.wrecks = [];
    this.redLeft = 0;
    // rolling start: everyone already rolling at pace speed in their double-file slot
    for (const car of heat.cars) {
      car.vx = Math.cos(car.angle) * this.track.paceSpeed;
      car.vy = Math.sin(car.angle) * this.track.paceSpeed;
      car.paced = true;
      car.auto = new Float32Array(2);
      car.stopped = 0;
      car.penalty = 0;
      car.penalties = 0;
      car.cautionsCaused = 0;
      car.belowLine = 0;
      car.stagePoints = 0;
      car.blueFlag = false;
      car.parked = false;
    }
    this.paceArc = this.arcOf(heat.cars[0]) + 4 * RC.gapRows;
    this.say(0, 'green', 'Green flag: rolling start, two-wide');
  }

  say(step, flag, text) {
    this.log.push({ step, flag, text });
    if (this.log.length > 40) this.log.shift();
  }

  arcOf(car) {
    return ((car.lastArc % this.track.length) + this.track.length) % this.track.length;
  }

  // signed distance round the lap from b to a, the short way
  ahead(a, b) {
    const L = this.track.length;
    return ((a - b) % L + L * 1.5) % L - L / 2;
  }

  // laps that count: run under green, plus any handed back
  lapsOf(car) {
    return car.laps.length + car.freeLaps - car.yellowLaps;
  }

  hold(live) {
    if (this.phase !== 'red') return false;
    for (const car of live) car.vx = car.vy = car.spin = 0;
    if (--this.redLeft <= 0) {
      this.phase = 'caution';
      this.flag = 'yellow';
      this.say(this.heat.step, 'yellow', 'Race resumes under caution');
    }
    return true;
  }

  // ---- every decision step: flags, cautions, penalties ----
  update(order, step) {
    const track = this.track, hw = track.halfWidth;
    const racing = order.filter(car => !car.finished);
    if (!racing.length) return;
    const leader = racing[0];

    // damaged-vehicle policy
    for (const car of racing) {
      const { front, rear } = car.condition;
      if (!car.parked && front + rear > RC.parkDamage) {
        car.parked = car.counted = true;
        car.running = false;
        car.retired = true;
        car.doneAt = step;
        this.say(step, 'redblack', `#${car.number ?? car.slot + 1} parked: too damaged to continue`);
        this.wreck(car, step);
      }
    }
    // a car that can't get going under its own power needs a tow, and a tow means it's out
    if (step % RC.check === 0 && this.phase !== 'start')
      for (const car of racing) {
        car.stopped = Math.hypot(car.vx, car.vy) < track.refSpeed * 0.08 && !car.penalty ? car.stopped + RC.check : 0;
        if (car.stopped >= RC.towAfter && car.running) {
          car.parked = car.counted = true;
          car.running = false;
          car.retired = true;
          car.doneAt = step;
          this.say(step, 'redblack', `#${car.number ?? car.slot + 1} towed off: out of the race`);
          this.wreck(car, step);
        }
      }
    if (this.cautionAt === step) return;

    if (this.phase === 'start' || this.phase === 'onetogo') {
      // green in the restart zone, when the leader reaches it
      const toLine = track.length - this.arcOf(leader);
      if (this.phase === 'start' || (toLine < track.length * RC.zone && toLine > 0)) this.green(order, step);
    }

    if (this.phase === 'green') {
      // white flag, stages
      const done = this.lapsOf(leader), last = this.heat.cars[0].raceLaps;
      if (done === last - 1 && this.flag !== 'white') {
        this.flag = 'white';
        this.say(step, 'white', 'White flag: final lap');
      }
      if (this.stages.length && done >= this.stages[0]) {
        const stage = this.stages.shift();
        racing.slice(0, 10).forEach((car, i) => car.stagePoints += 10 - i);
        this.say(step, 'stage', `Stage ${this.stages.length ? 1 : 2} ends at lap ${stage}: green-checkered`);
        this.caution(order, step, null, 'Stage break');
      }
      // hazards: a spin, a stopped car (alone on track, or in a short race, a spin only costs its own time)
      if (this.cautionsOn && step % RC.check === 0 && racing.length > 1) {
        for (const car of racing) {
          if (car.paced) continue;
          const speed = Math.hypot(car.vx, car.vy), c = Math.cos(car.angle), s = Math.sin(car.angle);
          const sideways = Math.abs(Math.atan2(-car.vx * s + car.vy * c, car.vx * c + car.vy * s)) > 1.1;
          const lateral = track.lateralAt(car.x, car.y), onTrack = lateral > -hw - track.apron - 4;
          if ((sideways && speed < track.refSpeed * 0.45 && onTrack && speed > 0.2) || car.stopped >= RC.stoppedFor) {
            const where = this.where(car);
            this.caution(order, step, car, `#${car.number ?? car.slot + 1} ${car.stopped >= RC.stoppedFor ? 'stopped' : 'spun'} ${where}`);
            break;
          }
        }
      }
      // a training caution just reformed the field: this step's running order is out of date
      if (this.cautionAt === step) return;
      // yellow line at the plate tracks: passing with all four wheels below it is a black flag
      if (track.plate && step % 30 === 0) {
        const rank = new Map(racing.map((car, i) => [car, i]));
        for (const car of racing) {
          const below = track.lateralAt(car.x, car.y) < -hw - car.spec.wid / 2;
          if (below) car.belowLine += 30;
          const before = this.lastRank?.get(car);
          if (below && before !== undefined && rank.get(car) < before && !car.penalty) this.blackFlag(car, step, 'passed below the yellow line');
        }
        this.lastRank = rank;
      }
      // jumping the restart: nobody but the leader gains a place before the leader reaches the line
      if (this.restartOrder) {
        if (leader.progress + leader.bonus >= this.restartUntil) this.restartOrder = null;
        else {
          // beating the leader to the line, or getting by anyone but the car alongside; passing a car that's
          // spun or slowing is fine
          const pace = Math.hypot(leader.vx, leader.vy), healthy = car => Math.hypot(car.vx, car.vy) > pace * 0.8;
          const dist = car => car.progress + car.bonus;
          racing.forEach((car, i) => {
            const was = this.restartOrder.get(car);
            if (!(was > 0) || car.penalty) return;
            // clearly ahead (half a car length), not a nose in front of the car alongside
            let passed = 0, passedLeader = false;
            for (let j = i + 1; j < racing.length; j++) {
              const other = racing[j], before = this.restartOrder.get(other);
              if (before < was && healthy(other) && dist(car) - dist(other) > car.spec.len / 2) passed++, passedLeader ||= before === 0;
            }
            if (passed >= 2 || (was === 1 && passedLeader)) this.blackFlag(car, step, 'jumped the restart');
          });
        }
      }
      // blue-and-yellow: lapped cars with the leaders closing
      for (const car of racing) car.blueFlag = this.lapsOf(car) < this.lapsOf(leader) && racing.some(other =>
        this.lapsOf(other) === this.lapsOf(leader) && other !== car && this.ahead(this.arcOf(car), this.arcOf(other)) > 0 && this.ahead(this.arcOf(car), this.arcOf(other)) < 400);
    }

    if (this.phase === 'caution') {
      // the field lines up; after the caution lap, one to go and a double-file restart
      const laps = this.paceRun / track.length, formed = this.order.every((car, i) => !car.running || car.finished || Math.abs(this.slotError(car, i)) < RC.gapRows * 2);
      if ((laps >= RC.cautionLaps && formed) || laps >= RC.maxCautionLaps) {
        if (this.arcOf(leader) > track.length * 0.45) {
          this.phase = 'onetogo';
          this.say(step, 'yellow', 'One to go: double-file restart next time by');
        }
      }
    }
  }

  where(car) {
    const track = this.track, i = Math.floor(this.arcOf(car) / track.length * track.heading.length) % track.heading.length;
    const regions = track.turnRegions;
    for (let k = 0; k < regions.length; k++) {
      const r = regions[k], inside = r.start <= r.end ? i >= r.start && i <= r.end : i >= r.start || i <= r.end;
      if (inside) return `in turn${regions.length === 2 ? `s ${2 * k + 1}-${2 * k + 2}` : ` ${k + 1}`}`;
    }
    return this.arcOf(car) < track.length * 0.25 || this.arcOf(car) > track.length * 0.75 ? 'on the frontstretch' : 'on the backstretch';
  }

  green(order, step) {
    this.phase = 'green';
    this.flag = 'green';
    this.paceCar = null;
    const racing = order.filter(car => !car.finished);
    for (const car of this.heat.cars) if (!car.penalty) car.paced = false;
    this.restartOrder = new Map(racing.map((car, i) => [car, i]));
    this.leaderLine = racing.length ? this.lapsOf(racing[0]) : 0;
    // the restart is over at the line (a training restart mid-lap: once the leader has run a zone's length)
    const L = this.track.length, at = racing.length ? racing[0].progress + racing[0].bonus : 0;
    this.restartUntil = Math.min(Math.ceil(at / L + 1e-9) * L, at + L * RC.zone);
    this.lastRank = null;
    // the lap the caution came out on doesn't count either: each car's caution ends when it next crosses the line
    for (const car of this.heat.cars) car.greenAtLine = car.underYellow;
    if (step > 0) this.say(step, 'green', this.overtime ? `Green: overtime attempt ${this.overtime}` : 'Green flag');
  }

  caution(order, step, culprit, text) {
    // training restarts at once, so a pile-up would throw a caution (and a lucky dog) per car: one per second at most
    if (this.phase !== 'green' || (this.fast && step - this.cautionAt < 60)) return;
    this.cautionAt = step;
    const racing = order.filter(car => !car.finished && car.running), leader = racing[0];
    if (!leader) return;
    // overtime: caution laps don't count, so a yellow on the final lap sets up a green-white-checkered; once
    // those attempts are used up, the next caution ends the race under yellow
    const finalLap = this.lapsOf(leader) >= this.heat.cars[0].raceLaps - 1;
    if (finalLap && this.overtime >= RC.overtimeAttempts) return this.endUnderYellow(racing, step, text);
    this.cautions++;
    if (culprit) culprit.cautionsCaused++;
    this.phase = 'caution';
    this.flag = 'yellow';
    this.paceRun = 0;
    this.paceArc = this.arcOf(leader) + 3 * RC.gapRows;
    for (const car of this.heat.cars) {
      car.underYellow = true;
      car.greenAtLine = false;
    }
    this.say(step, 'yellow', `Caution: ${text}`);
    if (finalLap) this.say(step, 'yellow', `Overtime: green-white-checkered, attempt ${++this.overtime} of ${RC.overtimeAttempts}`);
    // the field frozen as it ran; the first lapped car gets its lap back
    const lapsDown = car => Math.floor((leader.progress + leader.bonus - car.progress - car.bonus) / this.track.length);
    const lucky = racing.find(car => lapsDown(car) >= 1);
    if (lucky) {
      lucky.freeLaps++;
      lucky.bonus = lucky.freeLaps * this.track.length;
      this.say(step, 'yellow', `Lucky dog: #${lucky.number ?? lucky.slot + 1} gets a lap back`);
    }
    const leadLap = racing.filter(car => lapsDown(car) < 1).sort((a, b) => (b.progress + b.bonus) - (a.progress + a.bonus));
    const lapped = racing.filter(car => lapsDown(car) >= 1);
    // a spin in traffic costs the places of everyone who got by before the yellow flew: the car that brought it
    // out restarts at the back of its group (otherwise spinning would be a free way to bunch the field up)
    for (const group of [leadLap, lapped]) {
      const at = group.indexOf(culprit);
      if (at >= 0) group.push(...group.splice(at, 1));
    }
    this.order = [...leadLap, ...lapped];
    for (const car of racing) car.paced = true;
    if (this.fast) return this.reform(step, lapsDown);
    // room in the race clock for the laps behind the pace car, which don't count
    this.heat.deadline += Math.ceil((RC.cautionLaps + 2) * this.track.length / this.track.paceSpeed);
  }

  // the field closes up behind the leader, double-file in the frozen order, and goes green right there: the
  // bunching and the restart a caution brings, without the laps behind the pace car. Nobody gains distance
  // (the leader stays put); lapped cars keep their deficit; the laps it touches get no lap time.
  reform(step, lapsDown) {
    const track = this.track, L = track.length, leader = this.order[0], D = leader.progress + leader.bonus;
    this.order.forEach((car, i) => {
      const row = i >> 1, lane = (i & 1 ? 1 : -1) * track.halfWidth * 0.42;
      car.progress = D - RC.gapRows * row - lapsDown(car) * L - car.bonus;
      car.lastArc = ((car.progress % L) + L) % L;
      car.checkpoint = car.progress;
      // crossing the line in the jump is crossing it under yellow: no lap time, and the lap doesn't count
      while (car.progress >= (car.laps.length + 1) * L) car.laps.push(Infinity), car.yellowLaps++;
      car.lapVoid = true;
      const [x, y] = track.pointAt(car.lastArc, lane), angle = track.headingAt(car.lastArc);
      Object.assign(car, { x, y, angle, spin: 0, steer: 0, stopped: 0, c: Math.cos(angle), s: Math.sin(angle) });
      car.vx = car.c * track.paceSpeed;
      car.vy = car.s * track.paceSpeed;
    });
    this.green(this.order, step);
  }

  endUnderYellow(racing, step, text) {
    this.phase = 'checkered';
    this.flag = 'checkered';
    this.say(step, 'checkered', `Caution on the last overtime attempt (${text}): the race ends under yellow`);
    for (const car of racing) {
      car.finished = true;
      car.running = false;
      car.doneAt = step;
      car.place = ++this.heat.finishers;
    }
  }

  wreck(car, step) {
    if (this.phase === 'checkered' || !this.cautionsOn) return;
    this.wrecks.push(step);
    this.wrecks = this.wrecks.filter(s => step - s < RC.redWindow);
    const standing = () => this.heat.standings().filter(c => c.running);
    if (this.phase === 'green') this.caution(standing(), step, car, `#${car.number ?? car.slot + 1} out of the race`);
    // (training has no red flags: stopping the clock teaches nothing)
    if (!this.fast && this.wrecks.length >= RC.redCars && this.phase !== 'red' && this.phase !== 'checkered') {
      // the caution is set up (frozen order, pace car); then everything stops until the track is cleared
      this.phase = 'red';
      this.flag = 'red';
      this.redLeft = RC.redHold;
      this.heat.deadline += RC.redHold;
      this.wrecks = [];
      this.say(step, 'red', 'Red flag: big wreck, the race is stopped');
    }
  }

  blackFlag(car, step, why) {
    car.penalty = 2 * this.track.pitZone + this.track.length * 0.05;
    car.penalties++;
    car.paced = true;
    this.say(step, 'black', `Black flag: #${car.number ?? car.slot + 1} ${why}, stop-and-go`);
  }

  // where car i should be under caution: in line behind the pace car (double-file for the restart)
  slot(i) {
    if (this.phase === 'onetogo') {
      const row = i >> 1;
      return { arc: this.paceArc - RC.gapRows * (row + 1), lane: (i & 1 ? 1 : -1) * this.track.halfWidth * 0.42 };
    }
    return { arc: this.paceArc - RC.gapRows * (i + 1), lane: -this.track.halfWidth * 0.15 };
  }

  slotError(car, i) {
    return this.ahead(this.slot(i).arc, this.arcOf(car));
  }

  // ---- every step: autopilot for cars under race control ----
  steer(live) {
    const track = this.track, pace = track.paceSpeed;
    if (this.phase === 'caution' || this.phase === 'onetogo') {
      this.paceArc += pace;
      this.paceRun += pace;
      const [x, y] = track.pointAt(this.paceArc, -track.halfWidth * 0.15);
      this.paceCar = { x, y, angle: track.headingAt(((this.paceArc % track.length) + track.length) % track.length) };
    } else if (this.phase === 'start') {
      this.paceArc += pace;
    }
    const order = this.phase === 'start' ? this.heat.cars : this.order;
    order.forEach((car, i) => {
      if (!car.running || !car.paced || car.penalty) return;
      if (this.phase === 'green') {
        car.paced = false;
        return;
      }
      const { arc, lane } = this.phase === 'start' ? { arc: this.arcOf(car), lane: (car.slot & 1 ? 1 : -1) * track.halfWidth * 0.42 } : this.slot(i);
      const err = this.phase === 'start' ? 0 : this.ahead(arc, this.arcOf(car));
      // catch up from behind in the outside lane, settle into the slot, never stop
      const target = Math.max(pace * 0.4, Math.min(track.refSpeed * 0.7, pace + err * 0.02));
      this.autodrive(car, err > RC.gapRows * 3 ? track.halfWidth * 0.55 : lane, target, true);
    });
    // stop-and-go: down on the apron at pit-road speed for the length of a pass-through
    for (const car of live) {
      if (!car.penalty) continue;
      const speed = Math.hypot(car.vx, car.vy);
      car.penalty -= speed;
      this.autodrive(car, -track.halfWidth - track.apron * 0.5, pace, true);
      if (car.penalty <= 0) {
        car.penalty = 0;
        car.paced = this.phase !== 'green';
      }
    }
  }

  // drive toward a lane at a target speed, whatever car it is. gentle: how a driver follows race control's
  // instructions (lift and brake progressively, smooth steering), not the limit-of-grip driving of a pole lap;
  // slamming on the brakes mid-corner when the yellow flies would spin a rear-drive car
  autodrive(car, lane, target, gentle = false) {
    const track = this.track, speed = car.vx * Math.cos(car.angle) + car.vy * Math.sin(car.angle);
    const [tx, ty] = track.pointAt(car.lastArc + Math.max(60, speed * (gentle ? 30 : 18)), lane);
    const err = Math.atan2(Math.sin(Math.atan2(ty - car.y, tx - car.x) - car.angle), Math.cos(Math.atan2(ty - car.y, tx - car.x) - car.angle));
    const sideways = car.vy * Math.cos(car.angle) - car.vx * Math.sin(car.angle);
    car.auto[0] = Math.max(-1, Math.min(1, err / steerLock(car.spec, speed, sideways) * (gentle ? 1.2 : 1.6)));
    car.auto[1] = Math.max(gentle ? -0.3 : -1, Math.min(1, (target - speed) / (0.08 * target + 0.05)));
  }

  // ---- after the cars move: the finish ----
  after(live, step) {
    const heat = this.heat;
    for (const car of live) {
      if (car.finished && this.phase !== 'checkered') {
        // the leader took the checkered flag: everyone else finishes the lap they're on
        this.phase = 'checkered';
        this.flag = 'checkered';
        this.say(step, 'checkered', `Checkered flag: #${car.number ?? car.slot + 1} wins`);
        for (const other of heat.cars) if (other.running && !other.finished) other.raceLaps = this.lapsOf(other) + 1;
        heat.deadline = Math.min(heat.deadline, step + Math.ceil(1.5 * this.track.length / (this.track.refSpeed * 0.4)));
      }
    }
    for (const car of heat.cars) if (car.retired && !car.counted) {
      car.counted = true;
      this.wreck(car, step);
    }
  }

  get stageResults() {
    return this.heat.cars.map(car => car.stagePoints);
  }
}
