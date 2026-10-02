const deg = d => d * Math.PI / 180;
const WALL_RAY_DEG = [-90, -50, -20, 0, 20, 50, 90];
// the rear three are mirrors: without them a leader can't see an attack coming, let alone block it
const CAR_RAY_DEG = [-150, -90, -40, -12, 0, 12, 40, 90, 150, 180];
const WALL_RAYS = WALL_RAY_DEG.map(deg), CAR_RAYS = CAR_RAY_DEG.map(deg);
const FRONT_CAR_RAYS = CAR_RAY_DEG.flatMap((d, i) => Math.abs(d) <= 12 ? [i] : []);
const REAR_CAR_RAYS = CAR_RAY_DEG.flatMap((d, i) => Math.abs(d) >= 150 ? [i] : []);
const WALL_RAY_LEN = 260, CAR_RAY_LEN = 200, ATTACK_RANGE = 90;
// centreline points the driver "looks at" ahead, in track units
const LOOKAHEAD = [40, 100, 180, 290, 440];

// Input layout. Raycasts only see walls already in line of sight, so the brain also gets what
// serious racing agents get: where the road goes next, where it sits across the track, how it is
// rotating and sliding, whether it is touching a wall, and how fast it is closing on the car ahead.
const IN = {};
IN.walls = 0;
IN.cars = IN.walls + WALL_RAYS.length;
IN.closing = IN.cars + CAR_RAYS.length;
IN.rearClosing = IN.closing + 1;
IN.attacker = IN.rearClosing + 1;
IN.speed = IN.attacker + 1;
IN.slide = IN.speed + 1;
IN.yaw = IN.slide + 1;
IN.contact = IN.yaw + 1;
IN.trackPos = IN.contact + 1;
IN.heading = IN.trackPos + 1;
IN.facing = IN.heading + 1;
IN.ahead = IN.facing + 1;
IN.draft = IN.ahead + LOOKAHEAD.length;
// last applied controls (smooth driving needs to know what the hands are already doing) and race position
IN.prevSteer = IN.draft + 1;
IN.prevThrottle = IN.prevSteer + 1;
IN.position = IN.prevThrottle + 1;
// how the car feels: share of front and rear downforce lost, so a damaged car can brake for the grip it
// has left. Appended last so brains trained before it simply never read it.
IN.aero = IN.position + 1;
const INPUT_COUNT = IN.aero + 2;

// mirroring the world left/right: rays swap sides, anything signed left/right flips sign
const MIRROR_FROM = Array.from({ length: INPUT_COUNT }, (_, i) => i);
const MIRROR_SIGN = new Float32Array(INPUT_COUNT).fill(1);
for (const [start, angles] of [[IN.walls, WALL_RAY_DEG], [IN.cars, CAR_RAY_DEG]])
  angles.forEach((a, r) => MIRROR_FROM[start + r] = start + angles.findIndex(b => ((a + b) % 360 + 360) % 360 === 0));
for (const i of [IN.attacker, IN.slide, IN.yaw, IN.trackPos, IN.heading, IN.prevSteer]) MIRROR_SIGN[i] = -1;
for (let k = 0; k < LOOKAHEAD.length; k++) MIRROR_SIGN[IN.ahead + k] = -1;

const CAR_LEN = 20, CAR_WID = 10;
const BODY_X = [-5.5, 0, 5.5], BODY_R = 5;            // three circles approximate the chassis
const INERTIA = (CAR_LEN ** 2 + CAR_WID ** 2) / 12;   // unit mass

// Drag is low relative to mass, as on a real stock car: excess speed bleeds off over about a second
// rather than instantly, so a run built in someone's draft carries far enough to complete a pass.
// Top speed is set by POWER = AERO·v³ + ROLL·v² (about 9 here).
const POWER = 0.36, TRACTION = 0.16, BRAKE = 0.5, REVERSE = 0.06, REVERSE_MAX = 1.6;
const AERO = 0.0004, ROLL = 0.0008;
const GRIP = 0.22, DOWNFORCE = 0.0025;
const MAX_STEER = 0.6, WHEELBASE = 13;
// Two-axle tyre model, rear-wheel drive. Each axle cancels its own sideways slip up to its grip; the
// rear also carries the drive force, so power uses up rear grip (power oversteer); braking moves weight
// forward and off the rear (brake and lift oversteer); a sliding tyre grips a little less than a rolling
// one, so once the rear lets go the car keeps rotating unless the driver catches it (drift or spin).
const AXLE = WHEELBASE / 2, SLIDE_GRIP = 0.85, LOAD_TRANSFER = 0.24, BRAKE_FRONT = 0.68, WHEELSPIN = 4, SPIN_UP = 0.12;
const MAX_SPEED = 9;
const WALL_BOUNCE = 0.3, WALL_FRICTION = 0.7, RESTING = 0.6;
// follower in the wake, leader with a car on its bumper (tandem push), car being side-drafted
const DRAFT_DRAG = 0.5, DIRTY_AIR = 0.4, PUSH_DRAG = 0.15, SIDE_DRAG = 0.08;
// how fast the wheel can be turned per step, and how often (in steps) the brain makes a decision
const STEER_RATE = 0.15, DECIDE_EVERY = 2;
const STALL_WINDOW = 180, STALL_MIN = 150;
const KMH = 36;

// Damage tracks impact energy, so it grows with the square of the closing speed: a bump-draft tap
// bends nothing, a hard shunt wrecks the aero, a big hit ends the race. Contact below the threshold
// impulse is free (rubbing is racing); above it, what breaks depends on where the hit lands.
const HIT_THRESHOLD = { car: 0.4, wall: 0.25 };
const HIT_DAMAGE = {
  car: { front: 1.2, side: 1.0, rear: 0.7 },
  wall: { front: 2.5, side: 1.7, rear: 2.0 },
};
// A real wall hit crumples the car and bleeds speed in proportion to how hard it was, so bouncing off the
// barrier into a corner is slower than braking for it (otherwise the wall is a free cushion)
const WALL_CRUSH = 0.25, WALL_KEEP_MIN = 0.35;
// grinding along a wall or another car (sliding friction impulse) shreds bodywork too
const SCRAPE_DAMAGE = 0.6;
// a single impulse this big bends the suspension; this big wrecks the car outright
const BENDS = 2.5, WRECKS = 6.5;
// Aero: a smashed nose/splitter loses front downforce (the car pushes in fast corners), adds drag and
// overheats the engine; a smashed spoiler/decklid loses rear downforce (the car goes loose); bent
// fenders and quarter panels drag and rub the tyres. Bent suspension costs mechanical grip.
const AERO_LOSS = 0.15, NOSE_DRAG = 0.03, TAIL_DRAG = 0.015, FLANK_DRAG = 0.03, OVERHEAT = 0.02, BENT_GRIP = 0.1;

const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;
const WALL_COS = WALL_RAYS.map(Math.cos), WALL_SIN = WALL_RAYS.map(Math.sin);
const CAR_COS = CAR_RAYS.map(Math.cos), CAR_SIN = CAR_RAYS.map(Math.sin);
const rayX = new Float32Array(CAR_RAYS.length), rayY = new Float32Array(CAR_RAYS.length);

// impulse against an immovable surface at offset (rx, ry) from the car's centre, normal pointing at the car
function wallImpulse(car, rx, ry, nx, ny) {
  const vx = car.vx - car.spin * ry, vy = car.vy + car.spin * rx, vn = vx * nx + vy * ny;
  if (vn >= 0) return 0;
  // a car leaning on the barrier shouldn't chatter; only real impacts rebound
  const bounce = vn < -RESTING ? WALL_BOUNCE : 0;
  const rn = rx * ny - ry * nx, j = -(1 + bounce) * vn / (1 + rn * rn / INERTIA);
  const tx = -ny, ty = nx, rt = rx * ty - ry * tx;
  const jt = clamp(-(vx * tx + vy * ty) / (1 + rt * rt / INERTIA), -WALL_FRICTION * j, WALL_FRICTION * j);
  car.vx += j * nx + jt * tx;
  car.vy += j * ny + jt * ty;
  car.spin += (rn * j + rt * jt) / INERTIA;
  car.scrape = Math.abs(jt);
  return j;
}

class Car {
  constructor(brain, track, slot, raceLaps) {
    this.brain = brain;
    this.slot = slot;
    this.raceLaps = raceLaps;
    this.inputs = new Float32Array(INPUT_COUNT);
    this.mirrored = new Float32Array(INPUT_COUNT);
    this.wallSight = new Float32Array(WALL_RAYS.length);
    this.carSight = new Float32Array(CAR_RAYS.length);

    const { x, y, angle } = track.gridSlot(slot);
    Object.assign(this, { x, y, angle, vx: 0, vy: 0, spin: 0, slip: 0, frontLoose: false, rearLoose: false, wheelspin: 0, steer: 0, throttle: 0, draft: 0, tow: 0, pushed: 0, sideDrafted: 0, tailing: 0, position: 0 });
    this.action = new Float32Array(2);
    this.running = true;
    this.finished = this.retired = this.braked = this.touchingWall = this.elite = this.human = false;
    // tailSteps: time glued to the bumper of the car ahead (within two lengths, weighted by how close)
    this.draftSteps = this.tailSteps = this.ledSteps = this.overtakes = this.passedBy = 0;
    this.lastArc = track.progressAt(x, y);
    this.progress = this.lastArc > track.length / 2 ? this.lastArc - track.length : this.lastArc;
    this.gridOffset = -this.progress;
    this.checkpoint = this.progress;
    this.laps = [];
    this.lapStart = this.steps = 0;
    this.impact = this.wallHits = 0;
    this.damage = { front: 0, side: 0, rear: 0, bent: 0 };
    // impact delivered with this car's nose beyond a gentle bump: what a steward would penalise
    this.pull = this.scrape = this.rammed = 0;
    this.wrecked = false;
    // car-to-car impulse by where it landed; front means you did the hitting
    this.contactAt = { front: 0, side: 0, rear: 0 };
    this.c = Math.cos(angle);
    this.s = Math.sin(angle);
    this.manualSteer = this.manualThrottle = 0;
    this.path = null;
    this.pathLen = 0;
  }

  get forwardSpeed() {
    return this.vx * Math.cos(this.angle) + this.vy * Math.sin(this.angle);
  }

  // for display: share of front and rear downforce lost, and extra drag
  get condition() {
    const { front, side, rear } = this.damage;
    return { front: 1 - 1 / (1 + AERO_LOSS * front), rear: 1 - 1 / (1 + AERO_LOSS * rear), drag: NOSE_DRAG * front + TAIL_DRAG * rear + FLANK_DRAG * side };
  }

  // which part of the car faces an obstacle lying in direction (dx, dy)
  zoneFacing(dx, dy) {
    const along = dx * this.c + dy * this.s;
    return along > 0.7 ? 'front' : along < -0.7 ? 'rear' : 'side';
  }

  // j: impulse into the car from direction (dx, dy); scrape: sliding impulse; kind: 'car' | 'wall';
  // along: where on the body the contact is, front positive
  takeHit(dx, dy, j, scrape, kind, along) {
    const zone = this.zoneFacing(dx, dy), over = j - HIT_THRESHOLD[kind];
    if (kind === 'car' && zone === 'front' && over > 0) this.rammed += over;
    // a side-on hit at a front or rear corner takes out that end's splitter or spoiler too
    const end = zone === 'side' && Math.abs(along) > CAR_LEN / 4 ? (along > 0 ? 'front' : 'rear') : null;
    // door-to-door rubbing only dents sheet metal; grinding along the wall tears up whatever touches it
    const rub = SCRAPE_DAMAGE * scrape, impact = over > 0 ? HIT_DAMAGE[kind][zone] * over ** 1.5 : 0;
    const energy = kind === 'car' ? impact : impact + rub;
    if (kind === 'car') this.damage.side += rub;
    if (end) {
      this.damage[end] += energy / 2;
      this.damage.side += energy / 2;
    } else this.damage[zone] += energy;
    if (j > BENDS) {
      // a bent toe link drags the car toward the side that was hit
      this.damage.bent++;
      this.pull += 0.04 * Math.sign(-dx * this.s + dy * this.c);
    }
    if (j > WRECKS) this.wrecked = true;
    return zone;
  }

  sense(track, rivals) {
    const c = Math.cos(this.angle), s = Math.sin(this.angle), inputs = this.inputs;

    // walls: sphere-trace each ray through the distance field
    for (let r = 0; r < WALL_RAYS.length; r++) {
      const dx = c * WALL_COS[r] - s * WALL_SIN[r], dy = s * WALL_COS[r] + c * WALL_SIN[r];
      let t = 0;
      for (let it = 0; it < 20 && t < WALL_RAY_LEN; it++) {
        const d = track.distAt(this.x + dx * t, this.y + dy * t);
        if (d < 1) break;
        t += d > 1.5 ? d : 1.5;
      }
      this.wallSight[r] = Math.min(t, WALL_RAY_LEN);
      // inverse scaling: the last few car-lengths before a wall get most of the signal range
      inputs[IN.walls + r] = 1 / (1 + this.wallSight[r] / 30);
    }

    // rivals: nearest chassis circle along each car ray
    this.carSight.fill(CAR_RAY_LEN);
    let ahead = null, aheadDist = CAR_RAY_LEN, behind = null, behindDist = ATTACK_RANGE, behindSide = 0;
    for (let r = 0; r < CAR_RAYS.length; r++) {
      rayX[r] = c * CAR_COS[r] - s * CAR_SIN[r];
      rayY[r] = s * CAR_COS[r] + c * CAR_SIN[r];
    }
    for (const other of rivals) {
      if (other === this) continue;
      const dx = other.x - this.x, dy = other.y - this.y;
      if (dx * dx + dy * dy > (CAR_RAY_LEN + 12) ** 2) continue;
      // ignore cars on a different stretch of track behind a wall
      const apart = Math.abs(other.lastArc - this.lastArc);
      if (Math.min(apart, track.length - apart) > CAR_RAY_LEN + 80) continue;
      // the would-be attacker: nearest car behind, anywhere across the track, not just on a ray
      const fore = dx * c + dy * s, across = -dx * s + dy * c, d = Math.sqrt(dx * dx + dy * dy);
      if (fore < -CAR_LEN / 2 && d < behindDist && Math.abs(across) < HALF_WIDTH * 2) [behind, behindDist, behindSide] = [other, d, across];
      for (const ox of BODY_X) {
        const cx = dx + other.c * ox, cy = dy + other.s * ox, dist2 = cx * cx + cy * cy;
        for (let r = 0; r < CAR_RAYS.length; r++) {
          const along = cx * rayX[r] + cy * rayY[r];
          if (along <= 0) continue;
          const miss2 = dist2 - along * along;
          if (miss2 > BODY_R * BODY_R) continue;
          const t = Math.max(0, along - Math.sqrt(BODY_R * BODY_R - miss2));
          if (t < this.carSight[r]) this.carSight[r] = t;
          if (t < aheadDist && FRONT_CAR_RAYS.includes(r)) [ahead, aheadDist] = [other, t];
        }
      }
    }
    for (let r = 0; r < CAR_RAYS.length; r++)
      inputs[IN.cars + r] = this.carSight[r] < CAR_RAY_LEN ? 1 / (1 + this.carSight[r] / 25) : 0;
    // a feedforward brain has no memory, so it can't infer closing speed from shrinking distances;
    // positive means the gap is shrinking, whether I'm catching the car ahead or the car behind is catching me
    const closing = other => {
      if (!other) return 0;
      const dx = other.x - this.x, dy = other.y - this.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
      return clamp(((this.vx - other.vx) * dx + (this.vy - other.vy) * dy) / d / 3, -1, 1);
    };
    inputs[IN.closing] = closing(ahead);
    inputs[IN.rearClosing] = closing(behind);
    // which side the attack is coming from: makes "move over to cover it" a one-weight behaviour
    inputs[IN.attacker] = behind ? clamp(behindSide / HALF_WIDTH, -1, 1) : 0;

    // what the driver feels in the seat
    inputs[IN.speed] = (this.vx * c + this.vy * s) / MAX_SPEED;
    inputs[IN.slide] = clamp((-this.vx * s + this.vy * c) / 3, -1, 1);
    inputs[IN.yaw] = clamp(this.spin / 0.08, -1, 1);
    inputs[IN.contact] = this.touchingWall ? 1 : 0;

    // where it sits on the track, measured across the track rather than along the car's body
    const n = track.points.length, i = Math.floor(this.lastArc / track.length * n) % n;
    const [px, py] = track.points[i], h = track.heading[i], rel = this.angle - h;
    inputs[IN.trackPos] = ((this.x - px) * -Math.sin(h) + (this.y - py) * Math.cos(h)) / HALF_WIDTH;
    inputs[IN.heading] = Math.sin(rel);
    inputs[IN.facing] = Math.cos(rel);

    // the road ahead: bearing to centreline points further down the track, in the car's frame
    for (let k = 0; k < LOOKAHEAD.length; k++) {
      const [ax, ay] = track.points[(i + Math.round(LOOKAHEAD[k] / track.spacing)) % n];
      const dx = ax - this.x, dy = ay - this.y;
      inputs[IN.ahead + k] = clamp(Math.atan2(-dx * s + dy * c, dx * c + dy * s) / (Math.PI / 2), -1, 1);
    }
    inputs[IN.draft] = this.draft;
    inputs[IN.prevSteer] = this.steer;
    inputs[IN.prevThrottle] = this.throttle;
    inputs[IN.position] = this.position;
    const { front, rear } = this.condition;
    inputs[IN.aero] = front;
    inputs[IN.aero + 1] = rear;
  }

  // The brain also judges a left/right-mirrored copy of the world and the two opinions are
  // averaged, so a brain evolved on a mostly-clockwise track still handles anticlockwise ones.
  decide() {
    const { brain, inputs, mirrored } = this;
    for (let i = 0; i < INPUT_COUNT; i++) mirrored[i] = MIRROR_SIGN[i] * inputs[MIRROR_FROM[i]];
    const flipped = brain.think(mirrored), mirrorSteer = flipped[0], mirrorThrottle = flipped[1];
    const out = brain.think(inputs);
    this.action[0] = (out[0] - mirrorSteer) / 2;
    this.action[1] = (out[1] + mirrorThrottle) / 2;
  }

  drive(steer, throttle) {
    steer = clamp(steer, this.steer - STEER_RATE, this.steer + STEER_RATE);
    this.steer = steer;
    this.throttle = throttle;
    steer = clamp(steer + this.pull, -1, 1);
    const c = Math.cos(this.angle), s = Math.sin(this.angle);
    let forward = this.vx * c + this.vy * s, side = -this.vx * s + this.vy * c;
    const rolling = forward;

    // Tyres have one grip budget for braking, accelerating and cornering. Grip is mechanical (bent
    // suspension costs it) plus downforce, which grows with speed; turbulent air from a car ahead
    // steals some, and so does crash damage: a smashed nose loses front downforce, a smashed tail rear.
    const { front, side: sideDamage, rear, bent } = this.damage;
    const mechanical = GRIP / (1 + BENT_GRIP * bent), downforce = DOWNFORCE * forward * forward * (1 - DIRTY_AIR * this.draft);
    const frontGrip = mechanical + downforce / (1 + AERO_LOSS * front);
    const hold = mechanical + downforce / (1 + AERO_LOSS * rear);
    let push;
    if (throttle >= 0) push = throttle * Math.min(TRACTION, POWER / (1 + OVERHEAT * front) / Math.max(forward, 1));
    else if (forward > 0.3) push = throttle * BRAKE;
    else push = forward > -REVERSE_MAX ? throttle * REVERSE : 0;

    // weight moves forward under braking and back under power; each axle's grip follows its load, and a
    // tyre that's already sliding grips less than one that isn't, so slides and spins carry on
    const toFront = clamp(0.5 - LOAD_TRANSFER * push, 0.3, 0.7);
    const capFront = frontGrip * toFront * (this.frontLoose ? SLIDE_GRIP : 1);
    const capRear = hold * (1 - toFront) * (this.rearLoose ? SLIDE_GRIP : 1);
    // drive goes through the rear only; brakes are biased to the front so a hard stop pushes wide, not round
    const braking = throttle < 0 && forward > 0.3;
    let wantRear = braking ? push * (1 - BRAKE_FRONT) : push, wantFront = braking ? push * BRAKE_FRONT : 0;
    // a human on a keyboard gets traction control and ABS; the AIs drive the raw car
    if (this.human) {
      wantRear = clamp(wantRear, -capRear * 0.7, capRear * 0.7);
      wantFront = clamp(wantFront, -capFront * 0.9, capFront * 0.9);
    }

    // Each axle's grip is one budget shared between driving/braking along the wheel and stopping its own
    // sideways slide; asking for more scales both back (wheelspin, lock-up). The front works along the
    // steered wheel. Solved as impulses, a few passes so the two axles settle together.
    const delta = steer * MAX_STEER, sd = Math.sin(delta), cd = Math.cos(delta);
    const massFront = 1 / (1 + (AXLE * cd) ** 2 / INERTIA), massRear = 1 / (1 + AXLE * AXLE / INERTIA);
    let r = this.spin, alongF = 0, acrossF = 0, alongR = 0, acrossR = 0, overF = false, overR = false, slideRear = 0;
    for (let pass = 0; pass < 3; pass++) {
      const slideFront = -sd * forward + cd * (side + AXLE * r);
      let across = acrossF - slideFront * massFront, need = Math.hypot(wantFront, across);
      overF = need > capFront;
      let k = overF ? capFront / need : 1;
      const da = wantFront * k - alongF, db = across * k - acrossF, fy = da * sd + db * cd;
      alongF += da;
      acrossF += db;
      forward += da * cd - db * sd;
      side += fy;
      r += AXLE * fy / INERTIA;

      // A sliding tyre pushes back along its slip. Wheels spinning under power slip mostly lengthways, so
      // they have little left to hold the tail (power oversteer); the wider the tail swings, the more of
      // the grip turns sideways again, which is what lets a driver balance a drift on the throttle.
      slideRear = side - AXLE * r;
      across = acrossR - slideRear * massRear;
      const lengthways = braking ? wantRear : wantRear * (1 + WHEELSPIN * this.wheelspin);
      need = Math.hypot(lengthways, across);
      overR = need > capRear;
      k = overR ? capRear / need : 1;
      const along = Math.sign(wantRear) * Math.min(Math.abs(wantRear), Math.abs(lengthways * k));
      across *= k;
      const dr = across - acrossR;
      forward += along - alongR;
      alongR = along;
      acrossR += dr;
      side += dr;
      r -= AXLE * dr / INERTIA;
    }
    if (throttle < 0 && rolling > 0.3 && forward < 0) forward = 0;
    this.frontLoose = overF;
    this.rearLoose = overR;
    // stay on the power past the grip and the wheels spin up; lift and they catch again
    this.wheelspin = !braking && throttle > 0 && overR && !this.human ? Math.min(1, this.wheelspin + SPIN_UP * throttle) : this.wheelspin * 0.6;
    this.slip = overR ? Math.min(1, Math.abs(slideRear) / 2) : 0;
    this.spin = r;
    this.angle += this.spin;

    this.vx = forward * c - side * s;
    this.vy = forward * s + side * c;
    const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
    const bodywork = 1 + NOSE_DRAG * front + TAIL_DRAG * rear + FLANK_DRAG * sideDamage;
    const air = 1 - DRAFT_DRAG * this.draft - PUSH_DRAG * this.pushed + SIDE_DRAG * this.sideDrafted;
    const drag = AERO * bodywork * air * speed + ROLL;
    this.vx *= 1 - drag;
    this.vy *= 1 - drag;
    this.x += this.vx;
    this.y += this.vy;
    if (throttle < -0.3 && forward > 4) this.braked = true;
    if (this.draft > 0.2) this.draftSteps++;
    this.tailSteps += this.tailing;
  }

  hitWalls(track, events) {
    const c = this.c = Math.cos(this.angle), s = this.s = Math.sin(this.angle);
    let touching = false;
    for (const ox of BODY_X) {
      const px = this.x + c * ox, py = this.y + s * ox, d = track.distAt(px, py);
      if (d >= BODY_R) continue;
      let nx = track.distAt(px + 1, py) - track.distAt(px - 1, py), ny = track.distAt(px, py + 1) - track.distAt(px, py - 1);
      const len = Math.sqrt(nx * nx + ny * ny);
      if (!len) continue;
      nx /= len;
      ny /= len;
      touching = true;
      this.x += nx * (BODY_R - d);
      this.y += ny * (BODY_R - d);
      this.scrape = 0;
      const rx = c * ox - nx * BODY_R, ry = s * ox - ny * BODY_R, j = wallImpulse(this, rx, ry, nx, ny);
      this.impact += j;
      if (j > 0) this.takeHit(-nx, -ny, j, this.scrape, 'wall', ox);
      if (j > HIT_THRESHOLD.wall) {
        const keep = Math.max(WALL_KEEP_MIN, 1 - WALL_CRUSH * (j - HIT_THRESHOLD.wall));
        this.vx *= keep;
        this.vy *= keep;
      }
      if (events && j > 0.6) events.push({ x: this.x + rx, y: this.y + ry, power: j });
    }
    if (touching && !this.touchingWall) this.wallHits++;
    this.touchingWall = touching;
  }

  advance(track, step) {
    this.steps++;
    if (this.wrecked) {
      this.running = false;
      this.retired = true;
      this.doneAt = step;
      return;
    }
    const arc = track.progressAt(this.x, this.y);
    let ds = arc - this.lastArc;
    if (ds < -track.length / 2) ds += track.length;
    else if (ds > track.length / 2) ds -= track.length;
    this.progress += ds;
    this.lastArc = arc;

    if (this.progress >= (this.laps.length + 1) * track.length) {
      this.laps.push(this.steps - this.lapStart);
      this.lapStart = this.steps;
      if (this.laps.length >= this.raceLaps) {
        this.finished = true;
        this.running = false;
        this.doneAt = step;
        return;
      }
    }
    const at = (step >> 1) * 3;
    if (this.path && !(step & 1) && at < this.path.length) {
      this.path[at] = this.x;
      this.path[at + 1] = this.y;
      this.path[at + 2] = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
      this.pathLen = (step >> 1) + 1;
    }
    if (this.human) return;
    const stalled = this.steps % STALL_WINDOW === 0 && this.progress - this.checkpoint < STALL_MIN;
    if (this.steps % STALL_WINDOW === 0) this.checkpoint = this.progress;
    if (stalled || this.progress < -this.gridOffset - 150) {
      this.running = false;
      this.retired = true;
      this.doneAt = step;
    }
  }

  // in race lengths, so heats on tracks of different lengths compare fairly
  // Positions are zero-sum: a pass earns what the passed car loses, so holding position pays. Contact
  // is judged like a steward would: hitting someone with your nose costs a lot, rubbing a little,
  // getting hit from behind nothing.
  fitness(race, maxSteps, w) {
    const base = this.finished ? race * (2 - this.steps / maxSteps) : Math.max(0, this.progress + this.gridOffset);
    return (base - this.impact * w.wall - this.wallHits * w.hit
      - this.contactAt.front * w.ramming - this.contactAt.side * w.rubbing
      + (this.overtakes - this.passedBy) * w.pass) / race;
  }
}
