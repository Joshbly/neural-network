const view = { x: WORLD_W / 2, y: WORLD_H / 2, zoom: 0.4, mode: 'follow', w: 1, h: 1, dpr: 1 };
const FOLLOW_SPAN = 560;
const PALETTE = Array.from({ length: GRID_SLOTS }, (_, i) => `hsl(${(i * 137.5 + 190) % 360} 80% 62%)`);
const particles = [];

// the world is the current track's: generated tracks share one size, each real oval has its own
const worldW = () => track?.worldW ?? WORLD_W, worldH = () => track?.worldH ?? WORLD_H;
const fullZoom = () => Math.min(view.w / worldW(), view.h / worldH()) * 0.97;
const toWorld = (sx, sy) => [(sx - view.w / 2) / view.zoom + view.x, (sy - view.h / 2) / view.zoom + view.y];

function moveCamera(target, snap) {
  // closer in on the ovals: real-size stock cars are smaller than the arcade cars' world
  const follow = view.mode === 'follow' && target, span = track?.nascar ? 320 : FOLLOW_SPAN;
  const zoom = follow ? Math.min(view.w, view.h) / span : fullZoom();
  const x = follow ? target.x + target.vx * 14 : worldW() / 2, y = follow ? target.y + target.vy * 14 : worldH() / 2;
  const far = Math.hypot(x - view.x, y - view.y) > span;
  const k = snap || far ? 1 : 0.12;
  view.x += (x - view.x) * k;
  view.y += (y - view.y) * k;
  view.zoom += (zoom - view.zoom) * (snap ? 1 : 0.1);
}

function worldTransform(ctx) {
  const z = view.zoom * view.dpr;
  ctx.setTransform(z, 0, 0, z, view.w * view.dpr / 2 - view.x * z, view.h * view.dpr / 2 - view.y * z);
}

function screenTransform(ctx) {
  ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
}

function drawGround(ctx) {
  screenTransform(ctx);
  ctx.fillStyle = '#08110c';
  ctx.fillRect(0, 0, view.w, view.h);
  worldTransform(ctx);
  const [x0, y0] = toWorld(0, 0), [x1, y1] = toWorld(view.w, view.h);
  ctx.strokeStyle = 'rgba(120, 255, 190, 0.035)';
  ctx.lineWidth = 1 / view.zoom;
  ctx.beginPath();
  if (track?.nascar) return;
  for (let x = Math.max(0, Math.ceil(x0 / 80) * 80); x <= Math.min(worldW(), x1); x += 80) ctx.moveTo(x, Math.max(0, y0)), ctx.lineTo(x, Math.min(worldH(), y1));
  for (let y = Math.max(0, Math.ceil(y0 / 80) * 80); y <= Math.min(worldH(), y1); y += 80) ctx.moveTo(Math.max(0, x0), y), ctx.lineTo(Math.min(worldW(), x1), y);
  ctx.stroke();
}

function outline(points) {
  const path = new Path2D();
  points.forEach(([x, y], i) => i ? path.lineTo(x, y) : path.moveTo(x, y));
  path.closePath();
  return path;
}

function drawTrack(ctx, track) {
  if (track.nascar) return drawOval(ctx, track);
  const path = track.outline ??= outline(track.points);
  ctx.lineJoin = ctx.lineCap = 'round';
  const stroke = (style, width, dash = []) => {
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke(path);
  };
  stroke('rgba(0, 0, 0, 0.5)', HALF_WIDTH * 2 + 34);
  stroke('#e8ecf1', HALF_WIDTH * 2 + 10);
  stroke('#e5484d', HALF_WIDTH * 2 + 10, [16, 16]);
  stroke('#23272e', HALF_WIDTH * 2);
  stroke('rgba(255, 255, 255, 0.07)', 1.5, [12, 22]);
  ctx.setLineDash([]);

  const [sx, sy] = track.points[0], sq = HALF_WIDTH / 6;
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(track.heading[0]);
  for (let row = 0; row < 2; row++)
    for (let col = 0; col < 12; col++) {
      ctx.fillStyle = (row + col) % 2 ? '#111' : '#f5f5f5';
      ctx.fillRect(row * sq - sq, -HALF_WIDTH + col * sq, sq, sq);
    }
  ctx.restore();

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
  ctx.lineWidth = 1.2;
  for (let slot = 0; slot < GRID_SLOTS; slot++) {
    const { x, y, angle } = track.gridSlot(slot);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(4, -8);
    ctx.lineTo(13, -8);
    ctx.lineTo(13, 8);
    ctx.lineTo(4, 8);
    ctx.stroke();
    ctx.restore();
  }
}

// champion's path split into speed buckets so it's a handful of strokes per frame
function racingLine(path) {
  const buckets = Array.from({ length: 12 }, () => new Path2D());
  for (let k = 3; k < path.length; k += 3) {
    const fast = clamp((path[k + 2] / MAX_SPEED - 0.3) / 0.7, 0, 1), b = Math.min(11, fast * 12 | 0);
    buckets[b].moveTo(path[k - 3], path[k - 2]);
    buckets[b].lineTo(path[k], path[k + 1]);
  }
  return buckets.map((path, b) => ({ path, color: `hsla(${b / 11 * 140}, 95%, 55%, 0.8)` }));
}

function drawRacingLine(ctx, line) {
  ctx.lineWidth = 2.5 / view.zoom;
  ctx.lineCap = 'round';
  for (const { path, color } of line) {
    ctx.strokeStyle = color;
    ctx.stroke(path);
  }
}

function drawWake(ctx, car) {
  if (car.tow < 0.04) return;
  const tail = car.spec.len / 2, half = car.spec.wid / 2, len = car.spec.draftLen * 0.85, spread = car.spec.wid * 0.8 + len * 0.07;
  ctx.save();
  ctx.translate(car.x, car.y);
  ctx.rotate(car.angle);
  const fade = ctx.createLinearGradient(-tail, 0, -tail - len, 0);
  fade.addColorStop(0, `rgba(56, 225, 255, ${0.32 * car.tow})`);
  fade.addColorStop(1, 'rgba(56, 225, 255, 0)');
  ctx.fillStyle = fade;
  ctx.beginPath();
  ctx.moveTo(-tail, -half);
  ctx.lineTo(-tail - len, -spread);
  ctx.lineTo(-tail - len, spread);
  ctx.lineTo(-tail, half);
  ctx.fill();
  ctx.restore();
}

function drawCar(ctx, car, color, opts = {}) {
  if (car.spec.stock) return drawStockCar(ctx, car, car.livery ?? { primary: color, secondary: '#fff', accent: '#111', pattern: 'stripes', sponsor: '' }, opts);
  const { glow = false, alpha = 1 } = opts;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(car.x, car.y);
  ctx.rotate(car.angle);

  if (car.draft > 0.15) {
    ctx.strokeStyle = `rgba(150, 240, 255, ${car.draft * 0.9})`;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (const y of [-7, 7]) ctx.moveTo(6, y), ctx.lineTo(-18, y);
    ctx.stroke();
  }
  const braking = car.running && car.throttle < -0.1;
  if (braking) {
    ctx.fillStyle = 'rgba(255, 40, 40, 0.35)';
    ctx.beginPath();
    ctx.arc(-11, 0, glow ? 8 : 5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = '#0c0e11';
  ctx.fillRect(-8.6, -5.5, 4.6, 2.6);
  ctx.fillRect(-8.6, 2.9, 4.6, 2.6);
  ctx.fillRect(3.4, -5.2, 3.8, 2.3);
  ctx.fillRect(3.4, 2.9, 3.8, 2.3);

  if (glow) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 14 * view.dpr;
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(10.2, -0.9);
  ctx.lineTo(10.2, 0.9);
  ctx.lineTo(3, 2.1);
  ctx.lineTo(-1, 3.7);
  ctx.lineTo(-7, 3.3);
  ctx.lineTo(-9, 1.6);
  ctx.lineTo(-9, -1.6);
  ctx.lineTo(-7, -3.3);
  ctx.lineTo(-1, -3.7);
  ctx.lineTo(3, -2.1);
  ctx.closePath();
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillRect(8.5, -4.8, 2, 9.6);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(-10.4, -4.4, 1.9, 8.8);
  ctx.fillStyle = '#0b0d10';
  ctx.beginPath();
  ctx.ellipse(-0.6, 0, 2.3, 1.5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#f4f4f4';
  ctx.beginPath();
  ctx.arc(-0.9, 0, 0.95, 0, Math.PI * 2);
  ctx.fill();
  if (braking) {
    ctx.fillStyle = '#ff2a2a';
    ctx.fillRect(-10.9, -1, 1.3, 2);
  }
  ctx.restore();
}

function drawRays(ctx, car, track) {
  const c = Math.cos(car.angle), s = Math.sin(car.angle), n = track.points.length;

  // the road ahead the brain is reading, as orange beads down the track
  const i = Math.floor(car.lastArc / track.length * n) % n;
  ctx.strokeStyle = 'rgba(255, 179, 107, 0.35)';
  ctx.lineWidth = 1.2 / view.zoom;
  ctx.setLineDash([4 / view.zoom, 5 / view.zoom]);
  ctx.beginPath();
  ctx.moveTo(car.x, car.y);
  for (const d of LOOKAHEAD) ctx.lineTo(...track.points[(i + Math.round(d * car.spec.lookScale / track.spacing)) % n]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = '#ffb36b';
  ctx.lineWidth = 2 / view.zoom;
  for (const d of LOOKAHEAD) {
    const [x, y] = track.points[(i + Math.round(d * car.spec.lookScale / track.spacing)) % n];
    ctx.beginPath();
    ctx.arc(x, y, 5 / view.zoom, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.lineWidth = 1.2 / view.zoom;
  WALL_RAYS.forEach((_, r) => {
    const dx = c * WALL_COS[r] - s * WALL_SIN[r], dy = s * WALL_COS[r] + c * WALL_SIN[r], len = car.wallSight[r];
    const hue = 120 * len / WALL_RAY_LEN;
    ctx.strokeStyle = `hsla(${hue}, 90%, 60%, 0.45)`;
    ctx.beginPath();
    ctx.moveTo(car.x, car.y);
    ctx.lineTo(car.x + dx * len, car.y + dy * len);
    ctx.stroke();
    ctx.fillStyle = `hsl(${hue}, 95%, 62%)`;
    ctx.beginPath();
    ctx.arc(car.x + dx * len, car.y + dy * len, 2.6 / view.zoom, 0, Math.PI * 2);
    ctx.fill();
  });
  CAR_RAYS.forEach((_, r) => {
    const len = car.carSight[r];
    if (len >= CAR_RAY_LEN) return;
    const dx = c * CAR_COS[r] - s * CAR_SIN[r], dy = s * CAR_COS[r] + c * CAR_SIN[r];
    ctx.strokeStyle = 'rgba(255, 79, 163, 0.85)';
    ctx.lineWidth = 1.6 / view.zoom;
    ctx.beginPath();
    ctx.moveTo(car.x, car.y);
    ctx.lineTo(car.x + dx * len, car.y + dy * len);
    ctx.stroke();
    ctx.fillStyle = '#ff4fa3';
    ctx.beginPath();
    ctx.arc(car.x + dx * len, car.y + dy * len, 3.2 / view.zoom, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawLabel(ctx, car, text, color) {
  ctx.font = `700 ${11 / view.zoom}px ui-sans-serif, system-ui`;
  ctx.textAlign = 'center';
  ctx.fillStyle = color;
  ctx.fillText(text, car.x, car.y - car.spec.len * 0.55 - 6 / view.zoom);
}

// ---------- sparks & smoke ----------

function spawnImpacts(events) {
  for (const { x, y, power, cars } of events.splice(0, 40)) {
    const n = Math.min(14, 3 + power * 5 | 0);
    for (let i = 0; i < n && particles.length < 500; i++) {
      const a = Math.random() * Math.PI * 2, v = 0.8 + Math.random() * 3.5 * Math.min(2, power);
      particles.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 14 + Math.random() * 14, max: 28, spark: true, hue: cars ? 50 : 30 });
    }
  }
  events.length = 0;
}

function spawnSmoke(car) {
  // tyre smoke off the rear wheels when they slide or spin
  const scrub = Math.max(car.slip, car.wheelspin * 0.6);
  if (scrub > 0.15 && particles.length < 500)
    for (const side of [-5, 5])
      particles.push({ x: car.x - car.c * 8 - car.s * side, y: car.y - car.s * 8 + car.c * side, vx: car.vx * 0.3, vy: car.vy * 0.3, life: 30 + scrub * 30, max: 60, spark: false });
  const hurt = car.damage.front + car.damage.rear * 0.6 + car.damage.side * 0.3;
  if (hurt < 6 || Math.random() > hurt / 300 || particles.length > 500) return;
  particles.push({ x: car.x - car.c * 9, y: car.y - car.s * 9, vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4, life: 50, max: 50, spark: false });
}

function drawParticles(ctx) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vx *= 0.92;
    p.vy *= 0.92;
    if (--p.life <= 0) particles.splice(i, 1);
  }
  ctx.save();
  for (const p of particles) {
    const t = p.life / p.max;
    if (p.spark) {
      ctx.globalCompositeOperation = 'lighter';
      ctx.strokeStyle = `hsla(${p.hue}, 100%, ${55 + t * 30}%, ${t})`;
      ctx.lineWidth = 1.4 / view.zoom;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x - p.vx * 2.5, p.y - p.vy * 2.5);
      ctx.stroke();
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = `rgba(160, 165, 175, ${t * 0.16})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3 + (1 - t) * 9, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

// ---------- minimap ----------

function drawMinimap(ctx, track, cars, focus, colorOf) {
  const W = track.worldW, H = track.worldH, w = Math.min(240, view.w * 0.22, 240 * W / H), h = w * H / W, x = view.w - w - 12, y = 12, k = w / W;
  if (!track.minimap || track.minimap.width !== Math.round(w * view.dpr)) {
    const map = track.minimap = document.createElement('canvas');
    map.width = Math.round(w * view.dpr);
    map.height = Math.round(h * view.dpr);
    const m = map.getContext('2d');
    m.scale(k * view.dpr, k * view.dpr);
    m.lineJoin = 'round';
    m.strokeStyle = 'rgba(220, 230, 245, 0.55)';
    m.lineWidth = track.halfWidth * (track.nascar ? 4 : 1.2);
    m.stroke(track.outline ??= outline(track.points));
  }
  screenTransform(ctx);
  ctx.fillStyle = 'rgba(6, 10, 16, 0.75)';
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.beginPath();
  ctx.roundRect(x - 6, y - 6, w + 12, h + 12, 10);
  ctx.fill();
  ctx.stroke();
  ctx.drawImage(track.minimap, x, y, w, h);
  for (const car of cars) {
    if (!car.running && car !== focus) continue;
    ctx.fillStyle = colorOf(car);
    ctx.beginPath();
    ctx.arc(x + car.x * k, y + car.y * k, car === focus ? 4 : 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
  const [vx0, vy0] = toWorld(0, 0), [vx1, vy1] = toWorld(view.w, view.h);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.lineWidth = 1;
  ctx.strokeRect(x + vx0 * k, y + vy0 * k, (vx1 - vx0) * k, (vy1 - vy0) * k);
}
