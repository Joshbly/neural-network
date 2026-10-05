"""Checks the learner's brain against the engine's (train/ppo/model.py against js/nn.js and Car.decide), and its gradients.

    .venv/bin/python train/ppo/test_brain.py

1. decisions: a float64 forward with the engine's precision (float32 weights and senses, float64 sums, each activation
   rounded to float32, the mirror average rounded to float32) decides exactly what the JS car did, on recorded states
   (F5@40 and random weights, both JS kernels) and on extreme ones; the float32 forward PPO trains is within 1e-5
2. what it trains is what races: after 20 PPO updates the exported weights decide, in the engine, what the learner predicts
3. gradients: autograd equals finite differences (float64) for the try's log-probability, PPO's clipped objective (tries
   past the clip contribute nothing) and the critic's loss; locked weights get exactly 0
4. the sign: an update toward the better tries moves the decision toward them, for gas and for steer
"""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learner import CONFIG, ROOT, Rollouts, clipped_loss, ppo_update, read_rows, read_spec  # noqa: E402
from model import Critic, Driver, squash  # noqa: E402

failures = 0


def check(ok, text):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {text}")
    failures += not ok


spec, tmp = read_spec(), pathlib.Path(tempfile.mkdtemp(prefix='brain-test-'))
row, n = spec['row'], spec['inputs']
f5 = np.array(next(b for b in __import__('json').load(open(ROOT / 'models/slots/slot-5/generations/gen-0040.json'))['population'] if b['name'] == 'F5')['genes'], dtype=np.float32)


def decide64(driver, x):
    """The engine's arithmetic, in float64 where it uses float64 and rounded to float32 where it stores float32."""
    W1, b1, W2, b2, W3, b3 = [p.detach().double() for p in driver.weights()]
    W1 = W1 * driver.mask.double()

    def view(v):
        a1 = squash(v @ W1.T + b1).float().double()
        a2 = squash(a1 @ W2.T + b2).float().double()
        return squash(torch.cat([a2, a1], dim=1) @ W3.T + b3).float().double()

    x = torch.as_tensor(x).double()
    real, flip = view(x), view(x[:, driver.mirror_from] * driver.mirror_sign.double())
    return torch.stack([(real[:, 0] - flip[:, 0]) / 2, (real[:, 1] + flip[:, 1]) / 2], dim=1).float().numpy()


def engine(genes, states, plain=False):
    """Car.decide in the engine, SIMD kernel or plain JS (train/ppo/decide.js)."""
    g, x, out = tmp / 'genes.bin', tmp / 'states.bin', tmp / 'decisions.bin'
    np.asarray(genes, dtype=np.float32).tofile(g)
    np.ascontiguousarray(states, dtype=np.float32).tofile(x)
    subprocess.check_call(['node', str(ROOT / 'train' / 'ppo' / 'decide.js'), str(g), str(x), str(out), *(['--js'] if plain else [])])
    return np.fromfile(out, dtype='<f4').reshape(-1, 2)


def ulps(a, b):
    """How many float32 steps apart, per row (the worse of steer and gas)."""
    ia, ib = [np.asarray(v, dtype=np.float32).view(np.int32).astype(np.int64) for v in (a, b)]
    ia, ib = [np.where(i < 0, -(i & 0x7FFFFFFF), i) for i in (ia, ib)]
    return np.abs(ia - ib).max(axis=1)


def driver_of(genes, sigma=0.3):
    d = Driver(spec, sigma)
    d.load_genes(genes)
    return d


def judge(name, mine, theirs):
    off = ulps(mine, theirs)
    check(int((off > 1).sum()) == 0 and (off > 0).mean() <= 1e-4,
          f'{name}: {int((off == 0).sum())}/{len(off)} decisions identical' + (f', {int((off > 0).sum())} one float32 step apart' if off.any() else ''))


print('1. decisions')
rollouts = Rollouts(4, tmp)
try:
    (tmp / 'f5.bin').write_bytes(f5.tobytes())
    rows_f5, _ = read_rows(rollouts.collect(3, tmp / 'f5.bin', [0.3, 0.3], 60_000), row['width'])
    unlearned = rollouts.init(3, tmp / 'random.bin')
    rows_rand, _ = read_rows(rollouts.collect(4, tmp / 'random.bin', [0.3, 0.3], 40_000), row['width'])
finally:
    rollouts.quit()
obs_of = lambda rows: rows[:, row['obs'][0]:row['obs'][1]]
decided_of = lambda rows: rows[:, row['mean'][0]:row['mean'][1]]
for label, genes, rows in [('F5@40', f5, rows_f5), ('random weights', unlearned, rows_rand)]:
    judge(f'{label}, {len(rows)} recorded states', decide64(driver_of(genes), obs_of(rows)), decided_of(rows))
    plain = engine(genes, obs_of(rows), plain=True)
    check(np.array_equal(plain.view(np.int32), decided_of(rows).view(np.int32)) and np.array_equal(engine(genes, obs_of(rows)).view(np.int32), plain.view(np.int32)),
          f'{label}: the plain-JS forward and the SIMD kernel decide exactly what the car recorded')
rng = np.random.default_rng(5)
extreme = np.concatenate([rng.normal(0, 3, (15_000, n)), rng.choice([-50.0, -3.0, -1.0, 0.0, 1.0, 3.0, 50.0], (5_000, n))]).astype(np.float32)
d = driver_of(f5)
with torch.no_grad():
    pre = torch.from_numpy(extreme) @ (d.W1 * d.mask).T + d.b1
judge(f'extreme states ({float((pre.abs() > 3).float().mean()):.0%} of first-layer sums past ±3), {len(extreme)} states', decide64(d, extreme), engine(f5, extreme))
judge('extreme states, plain JS', decide64(d, extreme), engine(f5, extreme, plain=True))
with torch.no_grad():
    gap = max(float((driver_of(g).mean(torch.from_numpy(obs_of(r))) - torch.from_numpy(decided_of(r))).abs().max()) for g, r in [(f5, rows_f5), (unlearned, rows_rand)])
check(gap < 1e-5, f'the float32 forward PPO trains with is within {gap:.1e} of the engine on every recorded state')

print('\n2. what it trains is what races')
t = torch.from_numpy(rows_f5)
obs, extras, raw, logp = t[:, :n], t[:, row['extras'][0]:row['extras'][1]], t[:, row['raw'][0]:row['raw'][1]], t[:, row['logp']]
z = torch.cat([obs, extras], dim=1)
driver, critic = driver_of(f5), Critic(z.shape[1])
critic.inputs.update(z)
adv = torch.from_numpy(rows_f5[:, row['reward']]).float()
opt_d, opt_c = torch.optim.Adam(driver.parameters(), lr=3e-4, eps=1e-5), torch.optim.Adam(critic.parameters(), lr=1e-3, eps=1e-5)
cfg = {**CONFIG, 'minibatch': 9_000, 'epochs': 3, 'kl_target': 1.0}
result = ppo_update(driver, critic, opt_d, opt_c, dict(obs=obs, z=z, raw=raw, logp=logp, adv=adv, ret=adv), cfg, torch.Generator().manual_seed(0))
driver.quantize_()
trained = driver.genes()
moved = ulps(decide64(driver, obs.numpy()), decided_of(rows_f5))
print(f"    ({result['epochs'] * -(-len(obs) // cfg['minibatch'])} updates; {int((moved > 0).sum())} of {len(obs)} decisions changed)")
judge('the exported weights in the engine', decide64(driver, obs.numpy()), engine(trained, obs.numpy()))


class LogProb(nn.Module):
    """Driver.log_prob as a module, so its weights can be handed in for the gradient check."""

    def __init__(self, driver):
        super().__init__()
        self.d = driver

    def forward(self, x, raw):
        return self.d.log_prob(x, raw)


print('\n3. gradients (float64, autograd against finite differences)')
d64 = Driver(spec, 0.3).double()
d64.load_genes(f5)
with torch.no_grad():
    # states away from the squash's kinks at ±3, where finite differences would straddle the corner
    x_all = obs[:4000].double()

    def margin(v):
        W1, W2, W3 = d64.W1 * d64.mask, d64.W2, d64.W3
        p1 = v @ W1.T + d64.b1
        p2 = squash(p1) @ W2.T + d64.b2
        p3 = torch.cat([squash(p2), squash(p1)], dim=1) @ W3.T + d64.b3
        return torch.cat([p1, p2, p3], dim=1).abs().sub(3).abs().min(dim=1).values

    ok = (margin(x_all) > 1e-3) & (margin(x_all[:, d64.mirror_from] * d64.mirror_sign) > 1e-3)
    x = x_all[ok][:8]
    raw64 = raw[:4000][ok][:8].double()
wrapper, names = LogProb(d64), [f'd.{k}' for k, _ in d64.named_parameters()]
params = tuple(p.detach().clone().requires_grad_() for p in d64.parameters())
logp_of = lambda *ps: functional_call(wrapper, dict(zip(names, ps)), (x, raw64))
check(torch.autograd.gradcheck(logp_of, params, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=False),
      f"a try's log-probability: every one of the {sum(p.numel() for p in params)} weights and both noise levels, 8 states")
with torch.no_grad():
    now = logp_of(*params)
# tries 0 and 1 are past the clip on the side that stops their gradient; the rest are inside it or past it on the side
# that keeps theirs
shift = torch.tensor([0.5, -0.5, 0.05, -0.05, 0.4, -0.4, 0.1, -0.1], dtype=torch.float64)
a = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
old = now - shift
check(torch.autograd.gradcheck(lambda *ps: clipped_loss(logp_of(*ps), old, a, 0.2)[0], params, eps=1e-6, atol=1e-6, rtol=1e-4, fast_mode=True, raise_exception=False), "PPO's clipped objective")
past = clipped_loss(functional_call(wrapper, dict(zip(names, params)), (x[:2], raw64[:2])), old[:2], a[:2], 0.2)[0]
grads = torch.autograd.grad(past, params)
inside = clipped_loss(functional_call(wrapper, dict(zip(names, params)), (x[2:], raw64[2:])), old[2:], a[2:], 0.2)[0]
check(all(bool((g == 0).all()) for g in grads) and any(bool((g != 0).any()) for g in torch.autograd.grad(inside, params)),
      'tries past the clip contribute exactly 0 gradient; the others do contribute')
c64 = Critic(z.shape[1]).double()
c64.inputs.update(z.double())
zz, target = z[:64].double(), torch.from_numpy(np.random.default_rng(2).normal(size=64))
cnames = [k for k, _ in c64.named_parameters()]
check(torch.autograd.gradcheck(lambda *ps: ((functional_call(c64, dict(zip(cnames, ps)), (zz,)) - target) ** 2).mean(),
                               tuple(p.detach().clone().requires_grad_() for p in c64.parameters()), eps=1e-6, atol=1e-6, rtol=1e-4, fast_mode=True, raise_exception=False),
      "the critic's loss (the critic predicts in normalised units)")
d = driver_of(f5)
d.log_prob(obs[:2000], raw[:2000]).sum().backward()
check(bool((d.W1.grad[d.mask == 0] == 0).all()) and int((d.W1.grad[d.mask == 1] != 0).sum()) > 1000,
      f'locked weights get exactly 0 gradient ({int((d.mask == 0).sum())} of them); {int((d.W1.grad[d.mask == 1] != 0).sum())} open ones get some')

print('\n4. the sign')
with torch.no_grad():
    mean0 = driver_of(f5).mean(obs)
for k, name in [(1, 'gas'), (0, 'steer')]:
    # where the decision isn't pinned at ±1 (a saturated output can't move), tries above the decision did better
    free = mean0[:, k].abs() < 0.9
    o, e = obs[free], extras[free]
    tries = mean0[free] + 0.3 * torch.randn(len(o), 2, generator=torch.Generator().manual_seed(k))
    d, c = driver_of(f5), Critic(z.shape[1])
    c.inputs.update(torch.cat([o, e], dim=1))
    with torch.no_grad():
        lp = d.log_prob(o, tries)
    better = torch.sign(tries[:, k] - mean0[free][:, k])
    ppo_update(d, c, torch.optim.Adam(d.parameters(), lr=3e-4, eps=1e-5), torch.optim.Adam(c.parameters(), lr=1e-3, eps=1e-5),
               dict(obs=o, z=torch.cat([o, e], dim=1), raw=tries, logp=lp, adv=better, ret=torch.zeros(len(o))),
               {**CONFIG, 'epochs': 1, 'minibatch': len(o), 'kl_target': 1.0})
    with torch.no_grad():
        delta = d.mean(o)[:, k] - mean0[free][:, k]
    up = float((delta > 0).float().mean())
    check(float(delta.mean()) > 0 and up > 0.7, f'{name}: told higher {name} did better, one update raises it on {up:.0%} of {len(o)} states (by {float(delta.mean()):.4f} on average)')

print(f"\n{failures} check(s) failed" if failures else '\nall checks passed')
sys.exit(1 if failures else 0)
