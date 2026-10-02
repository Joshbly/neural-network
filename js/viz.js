const arrowLabel = d => d === 180 ? '▼' : d < 0 ? `◀${-d}` : d > 0 ? `${d}▶` : '▲';
const INPUTS = [
  ...WALL_RAY_DEG.map(d => [`wall ${arrowLabel(d)}`, '#8fe8b4']),
  ...CAR_RAY_DEG.map(d => [`car ${arrowLabel(d)}`, '#ff9fd0']),
  ['closing ▲', '#ff9fd0'], ['closing ▼', '#ff9fd0'], ['attack ◀▶', '#ff9fd0'],
  ['speed', '#9ecbff'], ['slide', '#9ecbff'], ['yaw', '#9ecbff'], ['contact', '#9ecbff'],
  ['track pos', '#ffd98a'], ['heading', '#ffd98a'], ['facing', '#ffd98a'],
  ...LOOKAHEAD.map(d => [`road +${d}`, '#ffb36b']),
  ['draft', '#9ecbff'], ['last steer', '#c9d4e5'], ['last gas', '#c9d4e5'], ['position', '#ffd166'],
  ['nose dmg', '#ff8a7a'], ['tail dmg', '#ff8a7a'],
];
const CYAN = [56, 225, 255], PINK = [255, 79, 163], IDLE = [28, 34, 48];

function activationColor(v) {
  const t = Math.min(1, Math.abs(v)), hot = v >= 0 ? CYAN : PINK;
  return `rgb(${IDLE.map((c, k) => c + (hot[k] - c) * t | 0)})`;
}

// Brain X-ray: blank out one input at a time (as if the car couldn't sense it), re-run the same
// mirror-averaged decision, and see how far the hands move. The biggest movers are what it's reacting to.
const xrayInputs = new Float32Array(INPUT_COUNT), xrayMirror = new Float32Array(INPUT_COUNT);
function decideFrom(brain, x) {
  for (let i = 0; i < INPUT_COUNT; i++) xrayMirror[i] = MIRROR_SIGN[i] * x[MIRROR_FROM[i]];
  const flipped = brain.think(xrayMirror), mirrorSteer = flipped[0], mirrorGas = flipped[1];
  const out = brain.think(x);
  return [(out[0] - mirrorSteer) / 2, (out[1] + mirrorGas) / 2];
}

function xray(car) {
  const { brain, inputs } = car, effects = [];
  const [steer, gas] = decideFrom(brain, inputs);
  for (let i = 0; i < brain.layers[0]; i++) {
    if (!inputs[i]) continue;
    xrayInputs.set(inputs);
    xrayInputs[i] = 0;
    const [s, g] = decideFrom(brain, xrayInputs);
    effects.push({ i, steer: steer - s, gas: gas - g });
  }
  // leave the activations on screen showing the real decision, not the last blanked-out one
  decideFrom(brain, inputs);
  const top = key => effects.filter(e => Math.abs(e[key]) > 0.03).sort((a, b) => Math.abs(b[key]) - Math.abs(a[key])).slice(0, 3);
  return { steer, gas, steerBy: top('steer'), gasBy: top('gas') };
}

function xrayHtml({ steer, gas, steerBy, gasBy }) {
  const chip = (e, key, up, down) => `<span><i style="color:${INPUTS[e.i][1]}">${INPUTS[e.i][0]}</i> ${e[key] > 0 ? up : down}${Math.abs(e[key]).toFixed(2)}</span>`;
  const row = (head, list, key, up, down) => `<div><em>${head}</em>${list.length ? list.map(e => chip(e, key, up, down)).join('') : '<span class="quiet">nothing stands out</span>'}</div>`;
  return row(`steer ${steer > 0.05 ? '▶' : steer < -0.05 ? '◀' : '·'} ${Math.abs(steer).toFixed(2)}`, steerBy, 'steer', '▶', '◀')
    + row(`${gas < -0.05 ? 'brake' : 'gas'} ${Math.abs(gas).toFixed(2)}`, gasBy, 'gas', '▲', '▼');
}

// marks: input index → colour, ringed in the drawing (the X-ray's top inputs)
function drawBrain(ctx, w, h, car, marks = new Map()) {
  const { brain } = car;
  ctx.clearRect(0, 0, w, h);
  if (!brain) return;
  const { layers } = brain, edges = layers.slice(1).reduce((n, size, l) => n + size * layers[l], 0);
  const dense = edges > 1000, huge = edges > 3000;
  const padL = 66, padR = 96, padY = 10;
  // each layer spreads over the full height, so a 128-wide hidden layer can't squash the labelled inputs
  const gaps = layers.map(n => Math.min(16, (h - 2 * padY) / Math.max(1, n - 1)));
  const radii = gaps.map(g => Math.max(1.2, Math.min(7, g * 0.36)));
  const nodes = layers.map((n, l) => Array.from({ length: n }, (_, i) =>
    [padL + l * (w - padL - padR) / (layers.length - 1), h / 2 + (i - (n - 1) / 2) * gaps[l]]));

  // edges glow with the signal actually flowing through them right now
  ctx.lineCap = 'round';
  for (let l = 1; l < layers.length; l++)
    for (let j = 0; j < layers[l]; j++)
      for (let i = 0; i < layers[l - 1]; i++) {
        const weight = brain.weight(l, i, j), signal = weight * brain.acts[l - 1][i];
        // the pros' wide nets have thousands of edges; only the ones carrying real signal get drawn
        if (huge && Math.abs(signal) < 0.12) continue;
        const [r, g, b] = signal >= 0 ? CYAN : PINK;
        // the wide net has ~2,000 edges; thinner lines keep its active pathways readable
        ctx.strokeStyle = `rgba(${r},${g},${b},${Math.min(0.9, (dense ? 0.012 : 0.025) + Math.abs(signal) * (dense ? 0.2 : 0.32))})`;
        ctx.lineWidth = Math.min(dense ? 2 : 3.2, (dense ? 0.2 : 0.3) + Math.abs(weight) * (dense ? 0.4 : 0.7));
        ctx.beginPath();
        ctx.moveTo(...nodes[l - 1][i]);
        ctx.lineTo(...nodes[l][j]);
        ctx.stroke();
      }

  nodes.forEach((layer, l) => layer.forEach(([x, y], i) => {
    const v = brain.acts[l][i];
    ctx.beginPath();
    ctx.arc(x, y, radii[l], 0, Math.PI * 2);
    ctx.fillStyle = activationColor(v);
    ctx.shadowColor = activationColor(v);
    ctx.shadowBlur = layer.length > 40 ? 0 : Math.abs(v) * 12;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }));

  ctx.font = '500 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  nodes[0].forEach(([x, y], i) => {
    ctx.globalAlpha = marks.has(i) ? 1 : 0.4 + Math.min(1, Math.abs(brain.acts[0][i])) * 0.6;
    ctx.fillStyle = INPUTS[i][1];
    ctx.fillText(INPUTS[i][0], x - radii[0] - 5, y);
    if (!marks.has(i)) return;
    ctx.strokeStyle = marks.get(i);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, radii[0] + 3, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  const { steer, throttle } = car, outR = radii.at(-1);
  const out = nodes.at(-1), barX = out[0][0] + outR + 10, barW = padR - outR - 18;
  const bar = (y, v, label, color) => {
    ctx.fillStyle = 'rgba(255,255,255,0.06)';
    ctx.fillRect(barX, y + 2, barW, 6);
    ctx.fillStyle = color;
    const mid = barX + barW / 2;
    ctx.fillRect(Math.min(mid, mid + v * barW / 2), y + 2, Math.abs(v) * barW / 2, 6);
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.fillRect(mid - 0.5, y, 1, 10);
    ctx.textAlign = 'left';
    ctx.fillStyle = '#c9d4e5';
    ctx.fillText(label, barX, y - 8);
  };
  bar(out[0][1], steer, steer < -0.15 ? '◀ LEFT' : steer > 0.15 ? 'RIGHT ▶' : 'STRAIGHT', '#38e1ff');
  bar(out[1][1], throttle, throttle < -0.05 ? 'BRAKE' : 'GAS', throttle < -0.05 ? '#ff4d4d' : '#4dff9a');
}

function drawChart(ctx, w, h, { series, yMin, yMax, format, markers = [], ref, empty }) {
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 38, r: 8, t: 8, b: 16 };
  const points = series.flatMap(s => s.points);
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  if (!points.length) {
    ctx.fillStyle = '#4c5668';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(empty, w / 2, h / 2);
    return;
  }
  const x0 = Math.min(...points.map(p => p[0])), x1 = Math.max(x0 + 1, ...points.map(p => p[0]));
  const X = x => pad.l + (x - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = y => h - pad.b - (Math.min(yMax, Math.max(yMin, y)) - yMin) / (yMax - yMin) * (h - pad.t - pad.b);

  ctx.textBaseline = 'middle';
  ctx.textAlign = 'right';
  for (let k = 0; k <= 2; k++) {
    const v = yMin + (yMax - yMin) * k / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(v));
    ctx.lineTo(w - pad.r, Y(v));
    ctx.stroke();
    ctx.fillStyle = '#5b6578';
    ctx.fillText(format(v), pad.l - 6, Y(v));
  }
  ctx.textAlign = 'left';
  ctx.fillText(`gen ${Math.floor(x0)}`, pad.l, h - 6);
  ctx.textAlign = 'right';
  ctx.fillText(`gen ${Math.floor(x1)}`, w - pad.r, h - 6);

  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = 'rgba(255,209,102,0.35)';
  for (const gen of markers) {
    ctx.beginPath();
    ctx.moveTo(X(gen), pad.t);
    ctx.lineTo(X(gen), h - pad.b);
    ctx.stroke();
  }
  if (ref) {
    ctx.strokeStyle = '#ffd166';
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(ref.value));
    ctx.lineTo(w - pad.r, Y(ref.value));
    ctx.stroke();
    ctx.fillStyle = '#ffd166';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(ref.label, w - pad.r, Y(ref.value) - 2);
  }
  ctx.setLineDash([]);

  for (const { points: pts, color, fill } of series) {
    if (!pts.length) continue;
    ctx.beginPath();
    pts.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)));
    if (fill) {
      ctx.save();
      ctx.lineTo(X(pts.at(-1)[0]), Y(yMin));
      ctx.lineTo(X(pts[0][0]), Y(yMin));
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.restore();
      ctx.beginPath();
      pts.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    const [lx, ly] = pts.at(-1);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(X(lx), Y(ly), 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}
