// The rating ladder: how good the brains really are, on a scale that means the same thing at generation 5 and
// generation 500.
//
// Practice rewards can't show it (points come from beating rivals who improve too), so every few generations
// each design's best brain races a pool of frozen past champions on fixed tracks, and every result ever
// recorded is fitted to one strength per player. Players never change once frozen, so old results stay valid
// and every new race sharpens the whole ladder.
//
// Model: Plackett-Luce, the standard model for finishing orders. A player with strength e^theta wins a race
// with probability e^theta / sum of everyone's, then second place is the same among those left, and so on,
// so a 10-car result counts as one result, not 45 separate head-to-heads. Fitted by Newton's method with a
// weak prior (keeps a player who has only ever won finite); the range is from the curvature of the fit.
// Shown Elo-style: 400 points = 10x stronger, the first player anchored at 1000.
const ELO = 400 / Math.LN10, PRIOR_SD = 1000 / ELO;

function fitRatings(count, races, anchor = 0) {
  const theta = new Float64Array(count), grad = new Float64Array(count), hess = new Float64Array(count);
  for (let iter = 0; iter < 200; iter++) {
    grad.fill(0);
    hess.fill(0);
    for (const order of races) {
      const n = order.length, e = order.map(i => Math.exp(theta[i]));
      // the field still racing for place j is everyone from j on
      let rest = e.reduce((a, b) => a + b, 0);
      for (let j = 0; j < n - 1; j++) {
        grad[order[j]] += 1;
        for (let k = j; k < n; k++) {
          const p = e[k] / rest;
          grad[order[k]] -= p;
          hess[order[k]] -= p * (1 - p);
        }
        rest -= e[j];
      }
    }
    let moved = 0;
    for (let i = 0; i < count; i++) {
      const g = grad[i] - theta[i] / PRIOR_SD ** 2, h = hess[i] - 1 / PRIOR_SD ** 2;
      const step = Math.max(-1, Math.min(1, g / h));
      theta[i] -= step;
      moved = Math.max(moved, Math.abs(step));
    }
    if (moved < 1e-6) break;
  }
  // the range: every rating is measured against the anchor, through chains of shared races, so it needs the
  // whole picture: the inverse of the fit's full curvature (the Fisher information), not each player alone
  const info = Array.from({ length: count }, () => new Float64Array(count));
  for (const order of races) {
    const n = order.length, e = order.map(i => Math.exp(theta[i]));
    let rest = e.reduce((a, b) => a + b, 0);
    for (let j = 0; j < n - 1; j++) {
      for (let k = j; k < n; k++) {
        const pk = e[k] / rest;
        info[order[k]][order[k]] += pk;
        for (let l = j; l < n; l++) info[order[k]][order[l]] -= pk * e[l] / rest;
      }
      rest -= e[j];
    }
  }
  for (let i = 0; i < count; i++) info[i][i] += 1 / PRIOR_SD ** 2;
  const cov = invertSPD(info), base = theta[anchor];
  return Array.from(theta, (t, i) => ({
    r: 1000 + (t - base) * ELO,
    sd: Math.sqrt(Math.max(0, cov[i][i] + cov[anchor][anchor] - 2 * cov[i][anchor])) * ELO,
  }));
}

// inverse of a symmetric positive-definite matrix, by Cholesky
function invertSPD(a) {
  const n = a.length, L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
  // L^-1, then (L^-1)^T L^-1
  const M = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    M[i][i] = 1 / L[i][i];
    for (let j = 0; j < i; j++) {
      let s = 0;
      for (let k = j; k < i; k++) s -= L[i][k] * M[k][j];
      M[i][j] = s / L[i][i];
    }
  }
  const inv = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = i; k < n; k++) s += M[k][i] * M[k][j];
      inv[i][j] = inv[j][i] = s;
    }
  return inv;
}

// opponents for a rating race: the strongest of the pool, where results say the most about a strong brain
function pickOpponents(pool, ratingOf, count, rng) {
  const strongest = pool.slice().sort((a, b) => ratingOf(b) - ratingOf(a)).slice(0, Math.max(count, Math.round(count * 1.6)));
  for (let i = strongest.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [strongest[i], strongest[j]] = [strongest[j], strongest[i]];
  }
  return strongest.slice(0, count);
}

// Who a laggard is replaced by a copy of. The candidates are its design's members, best first. No family
// line may hold more than half the design's seats, so a copy can't come from a family at the cap unless the
// laggard is from that family itself (then the count doesn't change); without that, one line takes every
// seat within a few dozen generations and the design stops exploring. The parent must clearly beat the
// laggard. fromTop: any of the eligible top half, which keeps several strong branches going, not just the best.
function pickParent(members, laggard, score, { margin, fromTop }, rng) {
  const cap = Math.ceil(members.length / 2), count = founder => members.filter(a => a.founder === founder).length;
  const eligible = members.filter(a => a !== laggard && score(a) - score(laggard) >= margin && (a.founder === laggard.founder || count(a.founder) < cap));
  if (!eligible.length) return null;
  const pool = fromTop ? eligible.slice(0, Math.ceil(eligible.length / 2)) : eligible.slice(0, 1);
  return pool[Math.floor(rng() * pool.length)];
}

module.exports = { fitRatings, pickOpponents, pickParent, ELO };
