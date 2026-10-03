const deg = d => d * Math.PI / 180;
const WALL_RAY_DEG = [-90, -50, -20, 0, 20, 50, 90];
// the rear three are mirrors: without them a leader can't see an attack coming, let alone block it
const CAR_RAY_DEG = [-150, -90, -40, -12, 0, 12, 40, 90, 150, 180];
const WALL_RAYS = WALL_RAY_DEG.map(deg), CAR_RAYS = CAR_RAY_DEG.map(deg);
const FRONT_CAR_RAY = CAR_RAY_DEG.map(d => Math.abs(d) <= 12);
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
// where the racing surface ends: positive on it, 0 at its edge, negative once a car is down on the apron;
// and where the pavement ends, negative on the grass (on a road course both are just the road's edge)
IN.edge = IN.aero + 2;
IN.paved = IN.edge + 1;
const INPUT_COUNT = IN.paved + 1;

// mirroring the world left/right: rays swap sides, anything signed left/right flips sign
const MIRROR_FROM = Array.from({ length: INPUT_COUNT }, (_, i) => i);
const MIRROR_SIGN = new Float32Array(INPUT_COUNT).fill(1);
for (const [start, angles] of [[IN.walls, WALL_RAY_DEG], [IN.cars, CAR_RAY_DEG]])
  angles.forEach((a, r) => MIRROR_FROM[start + r] = start + angles.findIndex(b => ((a + b) % 360 + 360) % 360 === 0));
for (const i of [IN.attacker, IN.slide, IN.yaw, IN.trackPos, IN.heading, IN.prevSteer]) MIRROR_SIGN[i] = -1;
for (let k = 0; k < LOOKAHEAD.length; k++) MIRROR_SIGN[IN.ahead + k] = -1;

const CAR_LEN = 20, CAR_WID = 10;
const BODY_X = [-5.5, 0, 5.5], BODY_R = 5;            // three circles approximate the chassis
const INERTIA = (CAR_LEN * CAR_LEN + CAR_WID * CAR_WID) / 12;   // unit mass

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
// the arcade cars' damage: these constants, no luck. Stock cars carry their own (STOCK_BASE.damage).
const NORMAL_DAMAGE = { threshold: HIT_THRESHOLD, hit: HIT_DAMAGE, scrape: { car: SCRAPE_DAMAGE, wall: SCRAPE_DAMAGE }, sideToEnds: 0,
  bend: [BENDS, BENDS], wreck: [WRECKS, WRECKS], luck: false };
// A crash's luck, drawn from the hit itself (which car, when, how hard), so a race still replays exactly.
function luck(car, j, salt) {
  let h = Math.imul(car.slot + 1, 0x9e3779b1) ^ Math.imul(car.steps + 7, 0x85ebca77) ^ Math.imul(Math.round(j * 1e6), 0xc2b2ae3d) ^ salt;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
// certain above the top of the range, and with luck a chance rising across it
const chance = (car, j, [lo, hi], lucky, salt) => j > hi || (lucky && j > lo && luck(car, j, salt) < (j - lo) / (hi - lo));
// Aero: a smashed nose/splitter loses front downforce (the car pushes in fast corners), adds drag and
// overheats the engine; a smashed spoiler/decklid loses rear downforce (the car goes loose); bent
// fenders and quarter panels drag and rub the tyres. Bent suspension costs mechanical grip.
const AERO_LOSS = 0.15, NOSE_DRAG = 0.03, TAIL_DRAG = 0.015, FLANK_DRAG = 0.03, OVERHEAT = 0.02, BENT_GRIP = 0.1;

// Every car carries a physics spec. NORMAL is the original arcade car (the constants above). The stock cars
// are Next Gen Cup cars at the same scale (1 unit = 0.25 m, 60 steps a second, forces per unit mass), tuned
// by train/nascar/calibrate.js to real pole speeds: the 670 hp package everywhere, the 510 hp tapered-spacer
// ("restrictor plate") package with its big spoiler at the superspeedways, where the draft decides everything.
const G = 9.81 / 0.25 / 3600;   // gravity, units/step^2
// selfExtent: from the middle of the car to its own bodywork along each car ray
const withDerived = spec => ({ ...spec, inertia: (spec.len * spec.len + spec.wid * spec.wid) / 12, axle: spec.wheelbase / 2,
  selfExtent: CAR_RAY_DEG.map(d => Math.min(spec.len / 2 / Math.max(1e-9, Math.abs(dcos(deg(d)))), spec.wid / 2 / Math.max(1e-9, Math.abs(dsin(deg(d)))))),
  ...spec.box && { box: boxGeometry(spec.box) } });
// A body that is exactly its drawn outline: a rectangle hx by hy (half sizes) with corners rounded to r, i.e. a
// core rectangle grown by r. Contacts are measured against it (heat.js boxContact, hitWalls, the car rays);
// the wall check samples the core's outline closely enough that no corner or flank can slip into a barrier.
function boxGeometry({ hx, hy, r }) {
  const cx = hx - r, cy = hy - r, samples = [[cx, 0], [-cx, 0]];
  for (const x of [-cx, -cx / 2, 0, cx / 2, cx]) samples.push([x, -cy], [x, cy]);
  return { hx, hy, r, cx, cy, samples };
}
const NORMAL = withDerived({
  name: 'normal', stock: false, len: CAR_LEN, wid: CAR_WID, bodyX: BODY_X, bodyR: BODY_R, wheelbase: WHEELBASE,
  power: POWER, traction: TRACTION, brake: BRAKE, reverse: REVERSE, reverseMax: REVERSE_MAX, aero: AERO, roll: ROLL,
  grip: GRIP, downforce: DOWNFORCE, maxSteer: MAX_STEER, steerRate: STEER_RATE, loadTransfer: LOAD_TRANSFER,
  draftDrag: DRAFT_DRAG, dirtyAir: DIRTY_AIR, pushDrag: PUSH_DRAG, sideDrag: SIDE_DRAG, draftLen: 220,
  // input scales: top speed, a big slide, a fast rotation; road-ahead reach
  speedNorm: MAX_SPEED, slideNorm: 3, yawNorm: 0.08, slipNorm: 2, lookScale: 1, wakeSpeed: 6, edgeNorm: HALF_WIDTH,
  // a car that covers less than this in STALL_WINDOW steps has stalled
  stallMin: STALL_MIN,
  // tyre grip grows with load to this power (1: in proportion)
  loadSens: 1,
});
// 4.97 m x 1.99 m, 110 in wheelbase; ~1590 kg with driver; weight shifts about 14% per g of braking.
// box: the body for every contact, drawn and collided alike (a chain of circles bulged at the corners and
// pinched between them, so a bump to the bumper slid the car ahead sideways); the circles remain only for
// the rare contact with an arcade car.
// steerSpeed: the wheels' reach shrinks with speed, full lock at walking pace and about 2.5 degrees at 180 mph.
// At that speed a stock car needs under a degree, so with full lock on tap at any speed a brain's whole
// useful steering range was the first 1% of its output; this gives it about what the arcade cars have.
// catchSlide: once the car is sliding, the reach grows by the slide angle, so a tail that steps out can
// still be caught with opposite lock.
const STOCK_BASE = {
  stock: true, len: 20, wid: 8, box: { hx: 10, hy: 4, r: 1 }, bodyX: [-6, -2, 2, 6], bodyR: 4, wheelbase: 11.2,
  traction: 0.009, brake: 0.016, reverse: 0.002, reverseMax: 0.3, roll: 2.6e-5, grip: 1.663 * G, downforce: 1.781e-4,
  maxSteer: 0.45, steerSpeed: 1.8, catchSlide: 1, steerRate: 0.04, loadTransfer: 0.14 / G, edgeNorm: 20,
  // car rays read from the car's own bodywork, on a 3 m scale: touching reads 1, a metre's gap 0.75, 3 m 0.5.
  // Measured from the middle on the arcade cars' scale, a car rubbing your door and one a metre away looked
  // almost the same, and two-wide pack racing is all about that metre.
  carNear: 12, speedNorm: 6.5, slideNorm: 1.5, yawNorm: 0.012, slipNorm: 1, lookScale: 1.5, wakeSpeed: 4,
  stallMin: 60,
  // real tyres: twice the load gives less than twice the grip, so steep banking helps less than the textbook
  loadSens: 0.5,
  // Crashes, by the hit's delta-v (the change in speed straight into what was hit). Rubbing and bump drafting
  // (under 8 mph) are free; a 16 mph hit costs about a fifth of that end's downforce, 30 mph about half. From
  // 12 to 25 mph the suspension may bend and from 18 to 35 mph the car may be wrecked outright, likelier the
  // harder the hit, and each hit lands somewhere between half and one and a half times as hard, so the same
  // crash sometimes limps on and sometimes ends the race. A hard hit to the door bends both ends too. Leaning
  // on the wall for 3 s costs about a tenth of the downforce. Door to door with another car, sheet metal
  // dents for the first second; after a full second without a break both cars lose a point of downforce
  // a second at each end until they part.
  damage: {
    lean: { after: 60, grace: 6, perStep: 0.01 / 60 },
    threshold: { car: 8 / MPH_PER_SPEED, wall: 8 / MPH_PER_SPEED },
    hit: { car: { front: 6.5, side: 5.4, rear: 3.8 }, wall: { front: 13.5, side: 9.2, rear: 10.8 } },
    scrape: { car: SCRAPE_DAMAGE, wall: 0.9 }, sideToEnds: 0.25,
    bend: [12 / MPH_PER_SPEED, 25 / MPH_PER_SPEED], wreck: [18 / MPH_PER_SPEED, 35 / MPH_PER_SPEED], luck: true,
  },
};
// power from the engines' real output at the wheels; grip, downforce, load sensitivity and drag fitted to the
// Next Gen Cup poles (train/nascar/calibrate.js: 2.7% RMS over 18 tracks on 670 hp, 0.9% over the plate tracks)
// Racing in traffic: draftDrag (tucked in a wake), pushDrag (a car filling your wake from your bumper),
// sideDrag (a nose on your rear quarter), twoWideDrag (door to door, nobody in clean air), spoilerLoss (rear
// downforce gone when someone is on your bumper or your rear quarter, which is why a push in a corner or a
// side draft gets a car loose). The plate package is all draft, so its draft and push matter more.
const NASCAR_670 = withDerived({ ...STOCK_BASE, name: '670', power: 0.0205, aero: 6.886e-5, draftDrag: 0.2, dirtyAir: 0.35, pushDrag: 0.08, sideDrag: 0.06,
  twoWideDrag: 0.04, spoilerLoss: 0.4, draftLen: 260 });
const NASCAR_PLATE = withDerived({ ...STOCK_BASE, name: 'plate', power: 0.0156, aero: 9.714e-5, draftDrag: 0.3, dirtyAir: 0.2, pushDrag: 0.12, sideDrag: 0.06,
  twoWideDrag: 0.05, spoilerLoss: 0.35, draftLen: 300 });
// a stock car runs the plate package only where the rules require it
const stockSpec = track => track.plate ? NASCAR_PLATE : NASCAR_670;
// front-wheel angle at full steering, at this forward speed
const steerLock = (P, forward, sideways = 0) => P.steerSpeed
  ? Math.min(P.maxSteer, P.maxSteer / (1 + dpow(forward / P.steerSpeed, 2)) + P.catchSlide * Math.abs(datan2(sideways, Math.abs(forward) + 1e-6)))
  : P.maxSteer;
const specFor = (cars, track) => cars === 'stock' ? stockSpec(track) : NORMAL;

// off the racing surface: grip left, and how hard grass and sand drag (units/step^2)
const SURFACE_GRIP = [1, 0.96, 0.45, 0.35, 0.9, 0.5];
const SURFACE_DRAG = [0, 0, 0.3 * G, 0.65 * G, 0, 0.5 * G];

const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;
const WALL_COS = WALL_RAYS.map(a => dcos(a)), WALL_SIN = WALL_RAYS.map(a => dsin(a));
const CAR_COS = CAR_RAYS.map(a => dcos(a)), CAR_SIN = CAR_RAYS.map(a => dsin(a));
const rayX = new Float32Array(CAR_RAYS.length), rayY = new Float32Array(CAR_RAYS.length);

// impulse against an immovable surface at offset (rx, ry) from the car's centre, normal pointing at the car
function wallImpulse(car, rx, ry, nx, ny) {
  const vx = car.vx - car.spin * ry, vy = car.vy + car.spin * rx, vn = vx * nx + vy * ny;
  if (vn >= 0) return 0;
  // a car leaning on the barrier shouldn't chatter; only real impacts rebound
  const bounce = vn < -RESTING ? WALL_BOUNCE : 0;
  const inertia = car.spec.inertia, rn = rx * ny - ry * nx, j = -(1 + bounce) * vn / (1 + rn * rn / inertia);
  const tx = -ny, ty = nx, rt = rx * ty - ry * tx;
  const jt = clamp(-(vx * tx + vy * ty) / (1 + rt * rt / inertia), -WALL_FRICTION * j, WALL_FRICTION * j);
  car.vx += j * nx + jt * tx;
  car.vy += j * ny + jt * ty;
  car.spin += (rn * j + rt * jt) / inertia;
  car.scrape = Math.abs(jt);
  return j;
}

// a feedforward brain has no memory, so it can't infer closing speed from shrinking distances;
// positive means the gap is shrinking, whether I'm catching the car ahead or the car behind is catching me
function closingSpeed(car, other) {
  if (!other) return 0;
  const dx = other.x - car.x, dy = other.y - car.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
  return clamp(((car.vx - other.vx) * dx + (car.vy - other.vy) * dy) / d / 3, -1, 1);
}

class Car {
  constructor(brain, track, slot, raceLaps, spec = NORMAL) {
    this.brain = brain;
    this.spec = spec;
    this.track = track;
    this.surface = 0;
    this.slot = slot;
    this.raceLaps = raceLaps;
    this.inputs = new Float32Array(INPUT_COUNT);
    this.mirrored = new Float32Array(INPUT_COUNT);
    this.wallSight = new Float32Array(WALL_RAYS.length);
    this.carSight = new Float32Array(CAR_RAYS.length);

    const { x, y, angle } = track.gridSlot(slot);
    Object.assign(this, { x, y, angle, vx: 0, vy: 0, spin: 0, slip: 0, frontLoose: false, rearLoose: false, wheelspin: 0, steer: 0, throttle: 0, draft: 0, tow: 0, pushed: 0, sideDrafted: 0, tailing: 0, airOff: 0, twoWide: 0, position: 0 });
    this.action = new Float32Array(2);
    this.running = true;
    this.finished = this.retired = this.braked = this.touchingWall = this.elite = this.human = false;
    // tailSteps: time glued to the bumper of the car ahead (within two lengths, weighted by how close)
    this.draftSteps = this.tailSteps = this.ledSteps = this.overtakes = this.passedBy = 0;
    // sideSteps: time spent leaning door to door on another car (rubbing: in contact flank-on this step)
    this.sideSteps = 0;
    this.rubbing = false;
    // doorSteps: how long it's been touching another car flank to flank without a break; doorGap: since it last did
    this.doorSteps = 0;
    this.doorGap = Infinity;
    this.doorTouch = false;
    // race control (NASCAR ovals): laps handed back (lucky dog), laps run under yellow (they don't count), and
    // whether the car is under its command
    this.freeLaps = this.bonus = this.yellowLaps = 0;
    this.paced = this.underYellow = this.greenAtLine = false;
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
    this.c = dcos(angle);
    this.s = dsin(angle);
    this.manualSteer = this.manualThrottle = 0;
    this.path = null;
    this.pathLen = 0;
  }

  get forwardSpeed() {
    return this.vx * dcos(this.angle) + this.vy * dsin(this.angle);
  }

  // for display: share of front and rear downforce lost, and extra drag
  get condition() {
    const { front, side, rear } = this.damage;
    return { front: 1 - 1 / (1 + AERO_LOSS * front), rear: 1 - 1 / (1 + AERO_LOSS * rear), drag: NOSE_DRAG * front + TAIL_DRAG * rear + FLANK_DRAG * side };
  }

  // takes `points` more of each end's downforce away (0.01: one percentage point), whatever is already gone
  wearAero(points) {
    for (const end of ['front', 'rear']) {
      const lost = 1 - 1 / (1 + AERO_LOSS * this.damage[end]);
      this.damage[end] = (1 / (1 - Math.min(0.99, lost + points)) - 1) / AERO_LOSS;
    }
  }

  // which part of the car faces an obstacle lying in direction (dx, dy)
  zoneFacing(dx, dy) {
    const along = dx * this.c + dy * this.s;
    return along > 0.7 ? 'front' : along < -0.7 ? 'rear' : 'side';
  }

  // j: impulse into the car from direction (dx, dy); scrape: sliding impulse; kind: 'car' | 'wall';
  // along: where on the body the contact is, front positive
  takeHit(dx, dy, j, scrape, kind, along) {
    const D = this.spec.damage ?? NORMAL_DAMAGE, zone = this.zoneFacing(dx, dy), over = j - D.threshold[kind];
    if (kind === 'car' && zone === 'front' && over > 0) this.rammed += over;
    // a side-on hit at a front or rear corner takes out that end's splitter or spoiler too
    const end = zone === 'side' && Math.abs(along) > this.spec.len / 4 ? (along > 0 ? 'front' : 'rear') : null;
    // door-to-door rubbing only dents sheet metal; grinding along the wall tears up whatever touches it
    const rub = D.scrape[kind] * scrape, impact = over > 0 ? D.hit[kind][zone] * over * Math.sqrt(over) * (D.luck ? 0.5 + luck(this, j, 1) : 1) : 0;
    const energy = kind === 'car' ? impact : impact + rub;
    if (kind === 'car') this.damage.side += rub;
    if (end) {
      this.damage[end] += energy / 2;
      this.damage.side += energy / 2;
    } else if (zone === 'side' && D.sideToEnds) {
      this.damage.front += energy * D.sideToEnds;
      this.damage.rear += energy * D.sideToEnds;
      this.damage.side += energy * (1 - 2 * D.sideToEnds);
    } else this.damage[zone] += energy;
    if (chance(this, j, D.bend, D.luck, 2)) {
      // a bent toe link drags the car toward the side that was hit
      this.damage.bent++;
      this.pull += 0.04 * Math.sign(-dx * this.s + dy * this.c);
    }
    if (chance(this, j, D.wreck, D.luck, 3)) this.wrecked = true;
    return zone;
  }

  sense(track, rivals) {
    const c = dcos(this.angle), s = dsin(this.angle), inputs = this.inputs, P = this.spec;

    // walls: sphere-trace each ray through the distance field (track.distAt written out inline: this loop
    // is the hottest code outside the brains). On an oval the "wall" a sensor sees is the paved edge: the
    // outside wall, or the inside of the apron where the grass starts.
    const field = track.field, cols = track.cols, rows = track.rows, cell = track.cell, outside = track.outside;
    for (let r = 0; r < WALL_RAYS.length; r++) {
      const dx = c * WALL_COS[r] - s * WALL_SIN[r], dy = s * WALL_COS[r] + c * WALL_SIN[r];
      let t = 0;
      for (let it = 0; it < 20 && t < WALL_RAY_LEN; it++) {
        const x = this.x + dx * t, y = this.y + dy * t;
        let d;
        if (x < 0 || y < 0) d = outside;
        else {
          const gx = x / cell, gy = y / cell, ix = gx | 0, iy = gy | 0;
          if (ix >= cols - 1 || iy >= rows - 1) d = outside;
          else {
            const fx = gx - ix, fy = gy - iy, i = iy * cols + ix, below = i + cols;
            const top = field[i] + (field[i + 1] - field[i]) * fx;
            const bottom = field[below] + (field[below + 1] - field[below]) * fx;
            d = top + (bottom - top) * fy;
          }
        }
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
      if (dx * dx + dy * dy > (CAR_RAY_LEN + 12) * (CAR_RAY_LEN + 12)) continue;
      // ignore cars on a different stretch of track behind a wall
      const apart = Math.abs(other.lastArc - this.lastArc);
      if (Math.min(apart, track.length - apart) > CAR_RAY_LEN + 80) continue;
      // the would-be attacker: nearest car behind, anywhere across the track, not just on a ray
      const fore = dx * c + dy * s, across = -dx * s + dy * c, d = Math.sqrt(dx * dx + dy * dy);
      if (fore < -P.len / 2 && d < behindDist && Math.abs(across) < track.halfWidth * 2) {
        behind = other;
        behindDist = d;
        behindSide = across;
      }
      if (other.spec.box) {
        // the ray against the other body's outline, in its own frame (slab test)
        const { hx, hy } = other.spec.box, oc = other.c, os = other.s, x0 = -dx * oc - dy * os, y0 = dx * os - dy * oc;
        for (let r = 0; r < CAR_RAYS.length; r++) {
          const ux = rayX[r] * oc + rayY[r] * os, uy = -rayX[r] * os + rayY[r] * oc;
          let near = -Infinity, far = Infinity;
          if (Math.abs(ux) < 1e-9) { if (Math.abs(x0) > hx) continue; }
          else {
            const t1 = (-hx - x0) / ux, t2 = (hx - x0) / ux;
            near = Math.max(near, Math.min(t1, t2));
            far = Math.min(far, Math.max(t1, t2));
          }
          if (Math.abs(uy) < 1e-9) { if (Math.abs(y0) > hy) continue; }
          else {
            const t1 = (-hy - y0) / uy, t2 = (hy - y0) / uy;
            near = Math.max(near, Math.min(t1, t2));
            far = Math.min(far, Math.max(t1, t2));
          }
          if (far < Math.max(near, 0)) continue;
          const t = Math.max(0, near);
          if (t < this.carSight[r]) this.carSight[r] = t;
          if (t < aheadDist && FRONT_CAR_RAY[r]) {
            ahead = other;
            aheadDist = t;
          }
        }
        continue;
      }
      const bodyR = other.spec.bodyR;
      for (const ox of other.spec.bodyX) {
        const cx = dx + other.c * ox, cy = dy + other.s * ox, dist2 = cx * cx + cy * cy;
        for (let r = 0; r < CAR_RAYS.length; r++) {
          const along = cx * rayX[r] + cy * rayY[r];
          if (along <= 0) continue;
          const miss2 = dist2 - along * along;
          if (miss2 > bodyR * bodyR) continue;
          const t = Math.max(0, along - Math.sqrt(bodyR * bodyR - miss2));
          if (t < this.carSight[r]) this.carSight[r] = t;
          if (t < aheadDist && FRONT_CAR_RAY[r]) {
            ahead = other;
            aheadDist = t;
          }
        }
      }
    }
    for (let r = 0; r < CAR_RAYS.length; r++)
      inputs[IN.cars + r] = this.carSight[r] >= CAR_RAY_LEN ? 0
        : P.carNear ? 1 / (1 + Math.max(0, this.carSight[r] - P.selfExtent[r]) / P.carNear) : 1 / (1 + this.carSight[r] / 25);
    inputs[IN.closing] = closingSpeed(this, ahead);
    inputs[IN.rearClosing] = closingSpeed(this, behind);
    // which side the attack is coming from: makes "move over to cover it" a one-weight behaviour
    inputs[IN.attacker] = behind ? clamp(behindSide / track.halfWidth, -1, 1) : 0;

    // what the driver feels in the seat, scaled to this car (a stock car's slides and rotation are gentler)
    inputs[IN.speed] = (this.vx * c + this.vy * s) / P.speedNorm;
    inputs[IN.slide] = clamp((-this.vx * s + this.vy * c) / P.slideNorm, -1, 1);
    inputs[IN.yaw] = clamp(this.spin / P.yawNorm, -1, 1);
    inputs[IN.contact] = this.touchingWall ? 1 : 0;

    // where it sits on the track, measured across the track rather than along the car's body
    const n = track.points.length, i = Math.floor(this.lastArc / track.length * n) % n;
    const px = track.points[i][0], py = track.points[i][1], h = track.heading[i], rel = this.angle - h;
    const lateral = (this.x - px) * -dsin(h) + (this.y - py) * dcos(h);
    inputs[IN.trackPos] = lateral / track.halfWidth;
    inputs[IN.heading] = dsin(rel);
    inputs[IN.facing] = dcos(rel);
    const paved = track.distAt(this.x, this.y), edge = track.nascar ? track.halfWidth - Math.abs(lateral) : paved;
    inputs[IN.edge] = clamp(edge / P.edgeNorm, -1, 1);
    inputs[IN.paved] = clamp(paved / P.edgeNorm, -1, 1);

    // the road ahead: bearing to centreline points further down the track, in the car's frame
    for (let k = 0; k < LOOKAHEAD.length; k++) {
      const point = track.points[(i + Math.round(LOOKAHEAD[k] * P.lookScale / track.spacing)) % n];
      const dx = point[0] - this.x, dy = point[1] - this.y;
      inputs[IN.ahead + k] = clamp(datan2(-dx * s + dy * c, dx * c + dy * s) / (Math.PI / 2), -1, 1);
    }
    inputs[IN.draft] = this.draft;
    inputs[IN.prevSteer] = this.steer;
    inputs[IN.prevThrottle] = this.throttle;
    inputs[IN.position] = this.position;
    // share of front and rear downforce lost, as in `condition`
    inputs[IN.aero] = 1 - 1 / (1 + AERO_LOSS * this.damage.front);
    inputs[IN.aero + 1] = 1 - 1 / (1 + AERO_LOSS * this.damage.rear);
  }

  // The brain also judges a left/right-mirrored copy of the world and the two opinions are
  // averaged, so a brain evolved on a mostly-clockwise track still handles anticlockwise ones.
  decide() {
    const { brain, inputs, mirrored } = this;
    for (let i = 0; i < INPUT_COUNT; i++) mirrored[i] = MIRROR_SIGN[i] * inputs[MIRROR_FROM[i]];
    const pair = brain.thinkPair(mirrored, inputs);
    this.action[0] = (pair[2] - pair[0]) / 2;
    this.action[1] = (pair[3] + pair[1]) / 2;
  }

  drive(steer, throttle) {
    const P = this.spec, track = this.track;
    // the wheels turn at the same rate whatever the lock, so a smaller reach at speed lets the hands move further
    const ca = dcos(this.angle), sa = dsin(this.angle);
    const rate = P.steerSpeed ? P.steerRate * P.maxSteer / steerLock(P, this.vx * ca + this.vy * sa, this.vy * ca - this.vx * sa) : P.steerRate;
    steer = clamp(steer, this.steer - rate, this.steer + rate);
    this.steer = steer;
    this.throttle = throttle;
    steer = clamp(steer + this.pull, -1, 1);

    // On an oval the road itself acts on the car. A banked turn pushes it up and in: the sim works in the
    // flat top-down plane, so the bank shows up as an inward push (gravity down the slope plus the tilt of the
    // road's support) and as extra tyre load when cornering hard toward the infield (ac). A stopped car on
    // 33 degrees just about holds; a fast one gets the banked-turn grip that makes Talladega flat out.
    let loadGrip = 1, surfaceGrip = 1;
    if (track.nascar) {
      const lateral = track.lateralAt(this.x, this.y), bank = track.bankAt(this.lastArc, lateral);
      this.surface = track.surfaceAt(this.x, this.y);
      if (bank) {
        const h = track.headingAt(this.lastArc), sb = dsin(bank), cb = dcos(bank);
        const speedNow = Math.sqrt(this.vx * this.vx + this.vy * this.vy), ac = Math.max(-G, -this.spin * speedNow);
        const inward = (G * cb + ac * sb) * sb;
        this.vx += inward * dsin(h);
        this.vy -= inward * dcos(h);
        loadGrip = dpow(Math.max(0.3, cb * (cb + ac / G * sb)), P.loadSens);
      }
      surfaceGrip = SURFACE_GRIP[this.surface] * track.grip;
    }
    const c = dcos(this.angle), s = dsin(this.angle);
    let forward = this.vx * c + this.vy * s, side = -this.vx * s + this.vy * c;
    const rolling = forward;

    // Tyres have one grip budget for braking, accelerating and cornering. Grip is mechanical (bent
    // suspension costs it) plus downforce, which grows with speed; turbulent air from a car ahead
    // steals some, and so does crash damage: a smashed nose loses front downforce, a smashed tail rear.
    const { front, side: sideDamage, rear, bent } = this.damage;
    let mechanical = P.grip / (1 + BENT_GRIP * bent), downforce = P.downforce * forward * forward * (1 - P.dirtyAir * this.draft);
    if (track.nascar) {
      mechanical *= loadGrip * surfaceGrip;
      downforce *= surfaceGrip;
    }
    const frontGrip = mechanical + downforce / (1 + AERO_LOSS * front);
    // a car on your bumper or your rear quarter takes the air off your spoiler (stock cars: arcade cars have none)
    const hold = mechanical + (P.spoilerLoss ? downforce * (1 - P.spoilerLoss * this.airOff) : downforce) / (1 + AERO_LOSS * rear);
    let push;
    if (throttle >= 0) push = throttle * Math.min(P.traction, P.power / (1 + OVERHEAT * front) / Math.max(forward, 1));
    else if (forward > 0.3) push = throttle * P.brake;
    else push = forward > -P.reverseMax ? throttle * P.reverse : 0;

    // weight moves forward under braking and back under power; each axle's grip follows its load, and a
    // tyre that's already sliding grips less than one that isn't, so slides and spins carry on
    const toFront = clamp(0.5 - P.loadTransfer * push, 0.3, 0.7);
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
    const AXLE = P.axle, INERTIA = P.inertia;
    const delta = steer * steerLock(P, forward, side), sd = dsin(delta), cd = dcos(delta);
    const massFront = 1 / (1 + AXLE * cd * (AXLE * cd) / INERTIA), massRear = 1 / (1 + AXLE * AXLE / INERTIA);
    let r = this.spin, alongF = 0, acrossF = 0, alongR = 0, acrossR = 0, overF = false, overR = false, slideRear = 0;
    for (let pass = 0; pass < 3; pass++) {
      const slideFront = -sd * forward + cd * (side + AXLE * r);
      let across = acrossF - slideFront * massFront, need = dhypot(wantFront, across);
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
      need = dhypot(lengthways, across);
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
    this.slip = overR ? Math.min(1, Math.abs(slideRear) / P.slipNorm) : 0;
    this.spin = r;
    this.angle += this.spin;

    this.vx = forward * c - side * s;
    this.vy = forward * s + side * c;
    const speed = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
    const bodywork = 1 + NOSE_DRAG * front + TAIL_DRAG * rear + FLANK_DRAG * sideDamage;
    const air = 1 - P.draftDrag * this.draft - P.pushDrag * this.pushed + P.sideDrag * this.sideDrafted + (P.twoWideDrag ? P.twoWideDrag * this.twoWide : 0);
    const drag = P.aero * bodywork * air * speed + P.roll;
    this.vx *= 1 - drag;
    this.vy *= 1 - drag;
    // grass and sand bog the car down
    if (track.nascar && SURFACE_DRAG[this.surface] && speed > 0) {
      const keep = Math.max(0, 1 - SURFACE_DRAG[this.surface] / speed);
      this.vx *= keep;
      this.vy *= keep;
    }
    this.x += this.vx;
    this.y += this.vy;
    if (throttle < -0.3 && forward > 4) this.braked = true;
    if (this.draft > 0.2) this.draftSteps++;
    this.tailSteps += this.tailing;
  }

  hitWalls(track, events) {
    const c = this.c = dcos(this.angle), s = this.s = dsin(this.angle);
    let touching = false;
    // real walls only: on an oval the grass and the apron are drivable
    if (this.spec.box) return this.hitWallsBox(track, events, c, s);
    const BODY_R = this.spec.bodyR;
    for (const ox of this.spec.bodyX) {
      const px = this.x + c * ox, py = this.y + s * ox, d = track.wallAt(px, py);
      if (d >= BODY_R) continue;
      let nx = track.wallAt(px + 1, py) - track.wallAt(px - 1, py), ny = track.wallAt(px, py + 1) - track.wallAt(px, py - 1);
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

  // the same for a body that is exactly its outline: points round the core, each grown by the corner radius
  hitWallsBox(track, events, c, s) {
    const { samples, r } = this.spec.box;
    let touching = false;
    for (const [ox, oy] of samples) {
      const lx = c * ox - s * oy, ly = s * ox + c * oy, px = this.x + lx, py = this.y + ly, d = track.wallAt(px, py);
      if (d >= r) continue;
      let nx = track.wallAt(px + 1, py) - track.wallAt(px - 1, py), ny = track.wallAt(px, py + 1) - track.wallAt(px, py - 1);
      const len = Math.sqrt(nx * nx + ny * ny);
      if (!len) continue;
      nx /= len;
      ny /= len;
      touching = true;
      this.x += nx * (r - d);
      this.y += ny * (r - d);
      this.scrape = 0;
      const rx = lx - nx * r, ry = ly - ny * r, j = wallImpulse(this, rx, ry, nx, ny);
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
      // a lap the field was reformed during (a compressed caution) isn't a real lap time
      this.laps.push(this.lapVoid ? Infinity : this.steps - this.lapStart);
      this.lapVoid = false;
      this.lapStart = this.steps;
      if (this.underYellow) {
        this.yellowLaps++;
        // the green has waved: this crossing ends the caution, the next lap counts
        if (this.greenAtLine) this.underYellow = this.greenAtLine = false;
      }
      if (this.laps.length + this.freeLaps - this.yellowLaps >= this.raceLaps) {
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
    if (this.human || this.paced) return;
    const stalled = this.steps % STALL_WINDOW === 0 && this.progress - this.checkpoint < this.spec.stallMin;
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
