#!/usr/bin/env node
// The engine's decisions for given senses, to check the learner's brain against (train/ppo/test_brain.py).
//   node train/ppo/decide.js GENES.bin INPUTS.bin OUT.bin [--js]
// GENES: design F's weights, float32. INPUTS: float32 rows of the senses. OUT: float32 [steer, gas] for each row,
// through Car.decide itself (mirror the senses, think both views, average). --js: the plain-JS forward instead of the
// SIMD kernel.
const fs = require('fs');
const { E } = require('../lib');
const { DESIGN } = require('./spec');

const [genesFile, inputsFile, outFile] = process.argv.slice(2);
if (process.argv.includes('--js')) E.Brain.simd = false;
const f32 = file => { const b = fs.readFileSync(file); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const inputs = f32(inputsFile), n = E.INPUT_COUNT, rows = inputs.length / n, out = new Float32Array(rows * 2);
const car = { brain: new E.Brain(DESIGN.layers, f32(genesFile)), inputs: new Float32Array(n), mirrored: new Float32Array(n), action: new Float32Array(2) };
for (let r = 0; r < rows; r++) {
  car.inputs.set(inputs.subarray(r * n, (r + 1) * n));
  E.Car.prototype.decide.call(car);
  out.set(car.action, r * 2);
}
fs.writeFileSync(outFile, Buffer.from(out.buffer));
