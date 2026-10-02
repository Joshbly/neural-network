#!/usr/bin/env node
// The rating ladder and the family cap, on made-up data where the right answer is known.
//   node train/test-ladder.js
const { fitRatings, pickOpponents, pickParent, ELO } = require('./ladder');
const { E } = require('./lib');

let failures = 0, checks = 0;
const ok = (cond, msg) => {
  checks++;
  if (!cond) {
    failures++;
    console.log(`  FAIL ${msg}`);
  }
};

// players with known Elo strengths race 10-car fields; the fit should recover them, with honest ranges
const rng = E.mulberry32(7), truth = [1000, 1100, 1200, 1350, 1500, 1500, 1650, 1800, 1900, 2100, 2200, 2400];
function race(field) {
  // draw a Plackett-Luce finishing order
  const left = field.slice(), order = [];
  while (left.length) {
    const w = left.map(i => 10 ** (truth[i] / 400)), total = w.reduce((a, b) => a + b, 0);
    let u = rng() * total, k = 0;
    while (u > w[k]) u -= w[k++];
    order.push(left.splice(Math.min(k, left.length - 1), 1)[0]);
  }
  return order;
}
const fieldOf = () => {
  // ladder-like fields: neighbours in strength, as the opponent picker makes them
  const centre = Math.floor(rng() * truth.length);
  return truth.map((_, i) => i).sort((a, b) => Math.abs(a - centre) - Math.abs(b - centre) + (rng() - 0.5) * 3).slice(0, 8);
};
const races = Array.from({ length: 1500 }, () => race(fieldOf()));
const fit = fitRatings(truth.length, races);
console.log(`  recovered: ${fit.map((f, i) => `${truth[i]}→${Math.round(f.r)}±${Math.round(1.96 * f.sd)}`).join('  ')}`);
ok(fit.every((f, i) => Math.abs(f.r - truth[i]) < 60), 'every rating within 60 points of the truth');
ok(fit.slice(1).every((f, i) => f.r >= fit[i].r - 40), 'the order comes out right');
// honest ranges: over many repeats, the 95% range should hold the truth about 95% of the time
let inside = 0, total = 0;
for (let rep = 0; rep < 30; rep++) {
  const f = fitRatings(truth.length, Array.from({ length: 300 }, () => race(fieldOf())));
  f.forEach((x, i) => {
    if (i === 0) return;
    total++;
    if (Math.abs(x.r - truth[i]) <= 1.96 * x.sd) inside++;
  });
}
console.log(`  95% ranges held the truth ${(100 * inside / total).toFixed(1)}% of the time over 30 repeats`);
ok(inside / total > 0.9 && inside / total < 0.99, 'the ranges are honest: they hold the truth about 95% of the time');
const few = fitRatings(truth.length, races.slice(0, 40));
ok(few[6].sd > fit[6].sd * 3, `fewer races, wider range (${Math.round(few[6].sd)} vs ${Math.round(fit[6].sd)})`);
// a player who has only ever won stays finite
const winner = fitRatings(3, Array.from({ length: 30 }, () => [2, 0, 1]));
ok(Number.isFinite(winner[2].r) && winner[2].r > 1300, `an unbeaten player gets a finite, high rating (${Math.round(winner[2].r)})`);

// opponents come from the strongest of the pool
const pool = truth.map((t, i) => ({ id: i, t }));
const picked = pickOpponents(pool, p => p.t, 5, rng);
ok(picked.length === 5 && picked.every(p => p.t >= 1500), 'opponents are drawn from the 8 strongest of the pool');

// family cap: 4 seats, at most 2 from one family
const brain = (name, founder, s) => ({ name, founder, s });
const score = a => a.s, cfg = { margin: 0.1, fromTop: false };
{
  const members = [brain('A2·5', 'A2', 0.9), brain('A2·6', 'A2', 0.8), brain('A1·3', 'A1', 0.5), brain('A3', 'A3', 0.2)];
  const parent = pickParent(members, members[3], score, cfg, rng);
  ok(parent?.founder === 'A1', `A2 holds 2 of 4 seats, so A3's replacement comes from A1 (got ${parent?.name})`);
}
{
  const members = [brain('A2·5', 'A2', 0.9), brain('A2·6', 'A2', 0.8), brain('A1·3', 'A1', 0.6), brain('A1·2', 'A1', 0.45)];
  const parent = pickParent(members, members[3], score, cfg, rng);
  ok(parent?.name === 'A1·3', `with A2 at the cap, A1's laggard is replaced from its own family (got ${parent?.name})`);
}
{
  const members = [brain('A2·5', 'A2', 0.9), brain('A2·6', 'A2', 0.8), brain('A1·3', 'A1', 0.25), brain('A3', 'A3', 0.2)];
  ok(pickParent(members, members[3], score, cfg, rng) === null, 'no replacement when the only eligible parent isn\'t clearly better');
}
{
  // a design that has already collapsed onto one family keeps working as before
  const members = [brain('A2·5', 'A2', 0.9), brain('A2·6', 'A2', 0.8), brain('A2·7', 'A2', 0.5), brain('A2·8', 'A2', 0.2)];
  ok(pickParent(members, members[3], score, cfg, rng)?.name === 'A2·5', 'a one-family design still copies its best');
}

console.log(`\n${checks - failures}/${checks} checks passed${failures ? `, ${failures} FAILED` : ''}`);
process.exit(failures ? 1 : 0);
