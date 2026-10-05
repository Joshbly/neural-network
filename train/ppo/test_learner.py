"""Checks the PyTorch learner (train/ppo/model.py, train/ppo/learner.py).

    .venv/bin/python train/ppo/test_learner.py

1. gene round trip: F's champion imported and exported gives the same bytes, locked weights exactly 0
2. quantisation: the learner's 5-decimal rounding equals the engine's quantizeGenes, bit for bit
3. locked blocks: after 20 PPO updates on a real batch, every locked weight is still exactly 0, the rest have moved
4. GAE: the per-episode filter equals a plain loop
5. PPO starts where the race left off: the learner's log-probabilities of the tries equal the ones the JS car
   recorded, so the first minibatch's ratio is 1
6. a dry run: 3 iterations against the real race workers, metrics and a checkpoint written, then resumed for a 4th
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learner import CONFIG, ROOT, Rollouts, advantages, ppo_update, read_rows, read_spec  # noqa: E402
from model import Critic, Driver, quantize  # noqa: E402

failures = 0


def check(ok, text):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'} {text}")
    failures += not ok


spec = read_spec()
row = spec['row']
champ = next(b for b in json.load(open(ROOT / 'models/slots/slot-5/generations/gen-0040.json'))['population'] if b['name'] == 'F5')
genes = np.array(champ['genes'], dtype=np.float32)
tmp = pathlib.Path(tempfile.mkdtemp(prefix='learner-test-'))

print('1. gene round trip')
driver = Driver(spec)
driver.load_genes(genes)
check(driver.genes().tobytes() == genes.tobytes() and driver.locked_at_zero(), f"F's champion ({len(genes)} weights) in and out: identical bytes, locked weights exactly 0")

print('\n2. quantisation')
rng = np.random.default_rng(1)
values = np.concatenate([rng.normal(0, s, 33_334).astype(np.float32) for s in (0.01, 0.3, 3.0)])
values.tofile(tmp / 'raw.bin')
subprocess.check_call(['node', '-e', f"""
const fs = require('fs'), {{ E }} = require('{ROOT}/train/lib');
const raw = fs.readFileSync('{tmp}/raw.bin'), g = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
fs.writeFileSync('{tmp}/js.bin', Buffer.from(E.quantizeGenes(g).buffer));"""])
js = np.fromfile(tmp / 'js.bin', dtype='<f4')
check(quantize(values).tobytes() == js.tobytes(), f'{len(values)} random weights rounded to 5 decimals: identical to the engine, bit for bit')

print('\n3 and 5. a real batch')
rollouts = Rollouts(4, tmp)
try:
    (tmp / 'f5.bin').write_bytes(genes.tobytes())
    manifest = rollouts.collect(3, tmp / 'f5.bin', [0.3, 0.3], 40_000)
    rows, lengths = read_rows(manifest, row['width'])
finally:
    rollouts.quit()
t = torch.from_numpy(rows)
obs, extras, raw, logp_js = t[:, :spec['inputs']], t[:, row['extras'][0]:row['extras'][1]], t[:, row['raw'][0]:row['raw'][1]], t[:, row['logp']]
driver = Driver(spec, 0.3)
driver.load_genes(genes)
with torch.no_grad():
    gap = float((driver.log_prob(obs, raw) - logp_js).abs().max())
check(gap < 1e-4, f'the learner gives the {len(rows)} recorded tries the probabilities the JS car did (largest difference {gap:.1e})')
critic = Critic(spec['inputs'] + len(spec['extras']))
z = torch.cat([obs, extras], dim=1)
with torch.no_grad():
    values = critic.value(z).numpy()
adv = advantages(rows[:, row['reward']], values, lengths, CONFIG['gamma'], CONFIG['lam'])
critic.inputs.update(z)
batch = dict(obs=obs, z=z, raw=raw, logp=logp_js, adv=torch.from_numpy(adv).float(), ret=torch.from_numpy(adv + values).float())
opt_d, opt_c = torch.optim.Adam(driver.parameters(), lr=3e-4, eps=1e-5), torch.optim.Adam(critic.parameters(), lr=1e-3, eps=1e-5)
cfg = {**CONFIG, 'minibatch': 6_000, 'epochs': 3, 'kl_target': 1.0}
before = driver.genes()
result = ppo_update(driver, critic, opt_d, opt_c, batch, cfg, torch.Generator().manual_seed(0))
check(result['first_ratio'] < 1e-4, f"the first minibatch's ratio is 1 to within {result['first_ratio']:.1e}: PPO starts from the policy that raced")
steps = result['epochs'] * -(-len(rows) // cfg['minibatch'])
moved = int((driver.genes() != before).sum())
check(steps >= 20 and driver.locked_at_zero() and moved > 10_000, f'after {steps} PPO updates every locked weight is still exactly 0, while {moved} weights moved')

print('\n4. GAE')
lengths = [5, 1, 17, 40, 3]
r, v = rng.normal(size=sum(lengths)), rng.normal(size=sum(lengths))
plain, at = np.empty(sum(lengths)), 0
for n in lengths:
    a = 0.0
    for i in reversed(range(n)):
        nxt = v[at + i + 1] if i + 1 < n else 0.0
        a = r[at + i] + CONFIG['gamma'] * nxt - v[at + i] + CONFIG['gamma'] * CONFIG['lam'] * a
        plain[at + i] = a
    at += n
check(float(np.abs(advantages(r, v, lengths, CONFIG['gamma'], CONFIG['lam']) - plain).max()) < 1e-12, 'per-episode filtered advantages equal a plain loop (5 episodes, to 1e-12)')

print('\n6. a dry run')
run = tmp / 'run'
learner = str(ROOT / 'train' / 'ppo' / 'learner.py')
common = ['--dir', str(run), '--threads', '4', '--decisions', '30000', '--device', 'cpu']
first = subprocess.run([sys.executable, learner, *common, '--iterations', '3'], capture_output=True, text=True)
lines = [json.loads(l) for l in open(run / 'metrics.jsonl')] if (run / 'metrics.jsonl').exists() else []
state = torch.load(run / 'checkpoint.pt', weights_only=False) if (run / 'checkpoint.pt').exists() else {}
check(first.returncode == 0 and len(lines) == 3 and state.get('iteration') == 3, f'3 iterations: exit {first.returncode}, {len(lines)} metric lines, checkpoint at iteration {state.get("iteration")}')
second = subprocess.run([sys.executable, learner, *common, '--iterations', '4'], capture_output=True, text=True)
lines = [json.loads(l) for l in open(run / 'metrics.jsonl')]
check(second.returncode == 0 and 'resuming at iteration 3' in second.stderr and [l['iteration'] for l in lines] == [0, 1, 2, 3],
      'resumed from the checkpoint and ran iteration 3, then quit cleanly')

print(f"\n{failures} check(s) failed" if failures else '\nall checks passed')
sys.exit(1 if failures else 0)
