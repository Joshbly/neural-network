// Deterministic maths for the simulation. Math.sin, Math.cos, Math.atan2, Math.log and friends aren't pinned
// down to the last bit by the JavaScript standard: Node and the browser (different versions of V8) disagree
// on Math.cos for some inputs by one unit in the last place, and in a chaotic race that one bit decides the
// finishing order a few minutes later. These are fdlibm's algorithms written with only +, -, *, / and sqrt,
// which IEEE 754 fixes exactly, so every engine gets the same bits and a race replays identically anywhere.
// (Accurate to about one unit in the last place; drawing code keeps using Math.)

// --- sine and cosine: reduce to [-pi/4, pi/4] by multiples of pi/2, then fdlibm's kernels ---
const PIO2_1 = 1.57079632673412561417e+00, PIO2_2 = 6.07710050630396597660e-11, PIO2_3 = 2.02226624871116645580e-21;
const INV_PIO2 = 6.36619772367581382433e-01;
const S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03, S3 = -1.98412698298579493134e-04,
  S4 = 2.75573137070700676789e-06, S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
const C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03, C3 = 2.48015872894767294178e-05,
  C4 = -2.75573143513906633035e-07, C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
function kernelSin(x) {
  const z = x * x, v = z * x, r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
  return x + v * (S1 + z * r);
}
function kernelCos(x) {
  const z = x * x, r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6))))), hz = 0.5 * z, w = 1 - hz;
  return w + (((1 - w) - hz) + z * r);
}
// x = n * pi/2 + reduced, in three parts so the subtraction stays exact for any angle a race produces
let reduced = 0;
function quadrant(x) {
  const n = Math.floor(x * INV_PIO2 + 0.5);
  reduced = ((x - n * PIO2_1) - n * PIO2_2) - n * PIO2_3;
  return n & 3;
}
function dsin(x) {
  const q = quadrant(x), y = reduced;
  return q === 0 ? kernelSin(y) : q === 1 ? kernelCos(y) : q === 2 ? -kernelSin(y) : -kernelCos(y);
}
function dcos(x) {
  const q = quadrant(x), y = reduced;
  return q === 0 ? kernelCos(y) : q === 1 ? -kernelSin(y) : q === 2 ? -kernelCos(y) : kernelSin(y);
}
const dtan = x => dsin(x) / dcos(x);

// --- arctangent: fdlibm's argument reduction to |x| < 7/16 and its odd polynomial ---
const ATAN_HI = [4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01, 1.57079632679489655800e+00];
const ATAN_LO = [2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17];
const AT = [3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01, -1.11111104054623557880e-01,
  9.09088713343650656196e-02, -7.69187620504482999495e-02, 6.66107313738753120669e-02, -5.83357013379057348645e-02,
  4.97687799461593236017e-02, -3.65315727442169155270e-02, 1.62858201153657823623e-02];
function datan(x) {
  const negative = x < 0;
  let a = negative ? -x : x, id;
  if (a < 0.4375) id = -1;
  else if (a < 0.6875) { id = 0; a = (2 * a - 1) / (2 + a); }
  else if (a < 1.1875) { id = 1; a = (a - 1) / (a + 1); }
  else if (a < 2.4375) { id = 2; a = (a - 1.5) / (1 + 1.5 * a); }
  else { id = 3; a = -1 / a; }
  const z = a * a, w = z * z;
  const s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10]))))), s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
  const r = id < 0 ? a - a * (s1 + s2) : ATAN_HI[id] - ((a * (s1 + s2) - ATAN_LO[id]) - a);
  return negative ? -r : r;
}
const PI = 3.14159265358979311600e+00, PI_O2 = 1.57079632679489655800e+00;
function datan2(y, x) {
  if (x === 0) return y > 0 ? PI_O2 : y < 0 ? -PI_O2 : 0;
  if (y === 0) return x > 0 ? 0 : PI;
  const a = datan(Math.abs(y / x));
  return x > 0 ? (y < 0 ? -a : a) : (y < 0 ? a - PI : PI - a);
}
const dacos = x => datan2(Math.sqrt((1 - x) * (1 + x)), x);
const dhypot = (x, y) => Math.sqrt(x * x + y * y);

// --- natural log: fdlibm's, splitting x into 2^k * (1 + f) through its bits ---
const LG1 = 6.666666666666735130e-01, LG2 = 3.999999999940941908e-01, LG3 = 2.857142874366239149e-01, LG4 = 2.222219843214978396e-01,
  LG5 = 1.818357216161805012e-01, LG6 = 1.531383769920937332e-01, LG7 = 1.479819860511658591e-01;
const LN2_HI = 6.93147180369123816490e-01, LN2_LO = 1.90821492927058770002e-10;
const bits = new DataView(new ArrayBuffer(8));
function dlog(x) {
  if (!(x > 0)) return x === 0 ? -Infinity : NaN;
  if (x === Infinity) return x;
  let k = 0;
  if (x < 2.2250738585072014e-308) { k = -54; x *= 18014398509481984; }
  bits.setFloat64(0, x);
  let hx = bits.getInt32(0);
  k += (hx >> 20) - 1023;
  hx &= 0x000fffff;
  const i = (hx + 0x95f64) & 0x100000;
  bits.setInt32(0, hx | (i ^ 0x3ff00000));
  k += i >> 20;
  const f = bits.getFloat64(0) - 1, s = f / (2 + f), dk = k, z = s * s, w = z * z;
  const t1 = w * (LG2 + w * (LG4 + w * LG6)), t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7))), R = t2 + t1;
  if (((hx - 0x6147a) | (0x6b851 - hx)) > 0) {
    const hfsq = 0.5 * f * f;
    return dk * LN2_HI - ((hfsq - (s * (hfsq + R) + dk * LN2_LO)) - f);
  }
  return dk * LN2_HI - ((s * (f - R) - dk * LN2_LO) - f);
}

// --- exp: fdlibm's, x = k ln2 + r, a rational approximation for e^r, then 2^k exactly through the bits ---
const INV_LN2 = 1.44269504088896338700e+00, P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03,
  P3 = 6.61375632143793436117e-05, P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08;
function dexp(x) {
  if (x !== x) return x;
  if (x > 709.78) return Infinity;
  if (x < -745.13) return 0;
  const k = Math.floor(x * INV_LN2 + 0.5), hi = x - k * LN2_HI, lo = k * LN2_LO, r = hi - lo, t = r * r;
  const c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5)))), y = 1 - ((lo - (r * c) / (2 - c)) - hi);
  // 2^k in two exact halves, so k down to the subnormal range still works
  const half = k >> 1;
  bits.setInt32(0, (half + 1023) << 20);
  bits.setInt32(4, 0);
  const a = bits.getFloat64(0);
  bits.setInt32(0, (k - half + 1023) << 20);
  return y * a * bits.getFloat64(0);
}
// x^y for x > 0 (powers of 1 and 1/2 exactly)
const dpow = (x, y) => y === 1 ? x : y === 0.5 ? Math.sqrt(x) : y === 2 ? x * x : dexp(y * dlog(x));
