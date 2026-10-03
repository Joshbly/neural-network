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
  ['track edge', '#ffd98a'], ['paved edge', '#ffd98a'], ['hit in ▲', '#ff9fd0'],
  ['banking', '#ffd98a'], ['bank ahead', '#ffd98a'], ['top steeper', '#ffd98a'],
  ...PACE_RAY_DEG.map(d => [`pace ${arrowLabel(d)}`, '#ffc4e4']),
  ...DRIFT_RAY_DEG.map(d => [`drift ${arrowLabel(d)}`, '#ffc4e4']),
];
const CYAN = [56, 225, 255], PINK = [255, 79, 163], IDLE = [28, 34, 48];
const BLOCK_RGB = { traffic: '64, 224, 208', lane: '255, 209, 102' };

function activationColor(v) {
  const t = Math.min(1, Math.abs(v)), hot = v >= 0 ? CYAN : PINK;
  return `rgb(${IDLE.map((c, k) => c + (hot[k] - c) * t | 0)})`;
}

// Brain X-ray: blank out one input at a time (as if the car couldn't sense it), re-run the same
// mirror-averaged decision, and see how far the hands move. The biggest movers are what it's reacting to.
const xrayInputs = new Float32Array(INPUT_COUNT), xrayMirror = new Float32Array(INPUT_COUNT);
function decideFrom(brain, x) {
  for (let i = 0; i < INPUT_COUNT; i++) xrayMirror[i] = MIRROR_SIGN[i] * x[MIRROR_FROM[i]];
  const pair = brain.thinkPair(xrayMirror, x);
  return [(pair[2] - pair[0]) / 2, (pair[3] + pair[1]) / 2];
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

// where the last drawing put every neuron, for clicking on them
let brainLayout = null;
function neuronAt(x, y) {
  if (!brainLayout) return null;
  const { nodes, gaps } = brainLayout;
  let best = null, bestDist = Infinity;
  for (let l = 1; l < nodes.length - 1; l++) {
    if (Math.abs(x - nodes[l][0][0]) > 10) continue;
    nodes[l].forEach(([, ny], j) => {
      const d = Math.abs(y - ny);
      if (d < Math.max(5, gaps[l] / 2) && d < bestDist) [best, bestDist] = [{ l, j }, d];
    });
  }
  return best;
}

// each input's three strongest connections into the first layer (input * 4096 + neuron), worked out once per brain
function strongestWiring(brain) {
  if (brain.wiring) return brain.wiring;
  const [inputs, first] = brain.layers, wiring = new Set();
  for (let i = 0; i < inputs; i++)
    Array.from({ length: first }, (_, j) => j).sort((a, b) => Math.abs(brain.weight(1, i, b)) - Math.abs(brain.weight(1, i, a)))
      .slice(0, 3).forEach(j => wiring.add(i * 4096 + j));
  return brain.wiring = wiring;
}

// The brain view's layout. Every layer is split into groups (the inputs by kind, a design's locked blocks) that sit a
// little apart; each neuron gets at least a minimum spacing (the inputs enough for their labels); the canvas is as
// tall as the busiest layer needs, never shorter than the panel's usual height, and each layer spreads to fill it.
const BRAIN_MIN_H = 480, BRAIN_PAD_Y = 10, NODE_GAP = { inputs: 9, neurons: 3.5, most: 16 }, GROUP_GAP = { inputs: 5, neurons: 12 };
// each layer's groups as sizes: runs of one input colour, then a design's blocks (whole layers otherwise)
function brainGroups(layers, mask) {
  return layers.map((n, l) => {
    if (l === 0) return Array.from({ length: n }, (_, i) => INPUTS[i][1]).reduce((runs, kind, i, all) => (i && kind === all[i - 1] ? runs[runs.length - 1]++ : runs.push(1), runs), []);
    const blocks = [mask ?? []].flat().filter(b => b.layer === l).sort((a, b) => a.from - b.from), sizes = [];
    let at = 0;
    for (const { from, to } of blocks) {
      if (from > at) sizes.push(from - at);
      sizes.push(to - from);
      at = to;
    }
    if (at < n) sizes.push(n - at);
    return sizes;
  });
}
const layerSpan = (sizes, gap, groupGap) => (sizes.reduce((a, b) => a + b, 0) - 1) * gap + (sizes.length - 1) * groupGap;
const brainHeight = (layers, mask) => Math.ceil(Math.max(BRAIN_MIN_H, ...brainGroups(layers, mask).map((sizes, l) =>
  layerSpan(sizes, l ? NODE_GAP.neurons : NODE_GAP.inputs, l ? GROUP_GAP.neurons : GROUP_GAP.inputs) + 2 * BRAIN_PAD_Y)));

// marks: input index → colour, ringed in the drawing (the X-ray's top inputs); picked: the neuron being inspected
function drawBrain(ctx, w, h, car, marks = new Map(), picked = null) {
  const { brain } = car;
  ctx.clearRect(0, 0, w, h);
  if (!brain) return;
  const { layers } = brain, edges = layers.slice(1).reduce((n, size, l) => n + size * layers[l], 0);
  const dense = edges > 1000, huge = edges > 3000;
  const padL = 66, padR = 96, padY = BRAIN_PAD_Y;
  // each layer spreads over the full height, groups a little apart, so a 208-wide hidden layer can't squash the
  // labelled inputs or run its blocks together
  const clusters = brainGroups(layers, car.species?.mask);
  const gaps = clusters.map((sizes, l) => {
    const n = sizes.reduce((a, b) => a + b, 0), groupGap = l ? GROUP_GAP.neurons : GROUP_GAP.inputs;
    return Math.min(NODE_GAP.most, (h - 2 * padY - (sizes.length - 1) * groupGap) / Math.max(1, n - 1));
  });
  const radii = gaps.map(g => Math.max(1.2, Math.min(7, g * 0.36)));
  const nodes = clusters.map((sizes, l) => {
    const x = padL + l * (w - padL - padR) / (layers.length - 1), groupGap = l ? GROUP_GAP.neurons : GROUP_GAP.inputs, column = [];
    let y = h / 2 - layerSpan(sizes, gaps[l], groupGap) / 2 - gaps[l];
    sizes.forEach((size, g) => {
      for (let k = 0; k < size; k++) column.push([x, y += gaps[l] + (g && !k ? groupGap : 0)]);
    });
    return column;
  });
  brainLayout = { nodes, gaps };

  // locked blocks (a graft's traffic block, design F's lane and traffic blocks): a tint behind their neurons and a
  // label, so you can watch the traffic one wake up when a car is near (with no car around every traffic sense is 0)
  for (const block of [car.species?.mask ?? []].flat()) {
    if (!nodes[block.layer]?.[block.to - 1]) continue;
    const name = block.name ?? 'traffic', rgb = BLOCK_RGB[name];
    const [x, top] = nodes[block.layer][block.from], bottom = nodes[block.layer][block.to - 1][1], r = radii[block.layer] + 4;
    ctx.fillStyle = `rgba(${rgb}, 0.08)`;
    ctx.strokeStyle = `rgba(${rgb}, 0.35)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x - r, top - r, 2 * r, bottom - top + 2 * r, r);
    ctx.fill();
    ctx.stroke();
    ctx.save();
    ctx.translate(x - r - 5, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '500 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.fillStyle = `rgba(${rgb}, 0.8)`;
    ctx.fillText(name, 0, 0);
    ctx.restore();
  }

  // edges glow with the signal actually flowing through them right now. Thousands of them: each is filed under
  // its colour, opacity and width (rounded to steps too fine to see) and every group is stroked as one path
  ctx.lineCap = 'round';
  const groups = new Map(), wiring = huge ? strongestWiring(brain) : null, resting = new Path2D();
  const edge = (from, to, weight, signal) => {
    // the wide net has ~2,000 edges; thinner lines keep its active pathways readable
    const alpha = Math.round(Math.min(0.9, (dense ? 0.012 : 0.025) + Math.abs(signal) * (dense ? 0.2 : 0.32)) * 50);
    const width = Math.round(Math.min(dense ? 2 : 3.2, (dense ? 0.2 : 0.3) + Math.abs(weight) * (dense ? 0.4 : 0.7)) * 5);
    const key = (signal >= 0 ? 0 : 100000) + alpha * 100 + width;
    let path = groups.get(key);
    if (!path) groups.set(key, path = new Path2D());
    path.moveTo(...from);
    path.lineTo(...to);
  };
  for (let l = 1; l < layers.length; l++)
    for (let j = 0; j < layers[l]; j++)
      for (let i = 0; i < layers[l - 1]; i++) {
        const weight = brain.weight(l, i, j), signal = weight * brain.acts[l - 1][i];
        // the pros' wide nets have thousands of edges; only the ones carrying real signal get drawn, plus each input's
        // strongest wiring in grey, so an input that's quiet right now (nothing ahead, a flat corner) still shows
        if (huge && Math.abs(signal) < 0.12) {
          if (l === 1 && wiring.has(i * 4096 + j)) {
            resting.moveTo(...nodes[0][i]);
            resting.lineTo(...nodes[1][j]);
          }
          continue;
        }
        edge(nodes[l - 1][i], nodes[l][j], weight, signal);
      }
  // design F's shortcuts: the first hidden layer straight to steer and gas
  if (brain.shortcuts) {
    const L = layers.length - 1;
    for (let j = 0; j < layers[L]; j++)
      for (let i = 0; i < layers[1]; i++) {
        const weight = brain.weight(L, layers[L - 1] + i, j), signal = weight * brain.acts[1][i];
        if (!huge || Math.abs(signal) >= 0.12) edge(nodes[1][i], nodes[L][j], weight, signal);
      }
  }
  if (wiring) {
    ctx.strokeStyle = 'rgba(160, 172, 196, 0.16)';
    ctx.lineWidth = 0.6;
    ctx.stroke(resting);
  }
  for (const [key, path] of groups) {
    const [r, g, b] = key >= 100000 ? PINK : CYAN, rest = key % 100000;
    ctx.strokeStyle = `rgba(${r},${g},${b},${Math.max(1, Math.floor(rest / 100)) / 50})`;
    ctx.lineWidth = Math.max(1, rest % 100) / 5;
    ctx.stroke(path);
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
  const ring = ({ l, j }, color, gap) => {
    const [x, y] = nodes[l][j];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, radii[l] + gap, 0, Math.PI * 2);
    ctx.stroke();
  };
  if (picked && picked.l < layers.length - 1 && picked.j < layers[picked.l]) ring(picked, '#fff', 3);
  for (const [l, j] of brain.held ?? []) ring({ l, j }, '#ffd166', 6);

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

  // uncertainty bands, under every line: [x, low, high]
  for (const { band, bandColor } of series) {
    if (!band?.length) continue;
    ctx.beginPath();
    band.forEach(([x, lo], i) => i ? ctx.lineTo(X(x), Y(lo)) : ctx.moveTo(X(x), Y(lo)));
    for (let i = band.length - 1; i >= 0; i--) ctx.lineTo(X(band[i][0]), Y(band[i][2]));
    ctx.closePath();
    ctx.fillStyle = bandColor;
    ctx.fill();
  }

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
