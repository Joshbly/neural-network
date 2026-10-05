"""Gradient learning: trains design F with PPO on single-car NASCAR time trials.

    python train/ppo/learner.py --dir RUN [--shm DIR] [--threads N] [--decisions 600000] [--iterations N] [--hours H]
                                [--seed 7] [--device cuda|cpu] [--generation-every 25]

Each iteration: publish the driver's weights, have the race workers (train/ppo/rollouts.js) race episodes with
them, read every decision back, work out how much better than expected each try was (GAE against a critic that
also sees race state), and move the driver toward the better tries (PPO). The driver is design F exactly as the
JS brain computes it (train/ppo/model.py); everything about the data comes from train/ppo/spec.js.
RUN (a save's ppo/ folder) gets metrics.jsonl (a line an iteration) and checkpoint.pt (resumed if present). The save
around it gets the files evolution writes (Save), so the server syncs the run and the app shows it like any other.
"""
import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from scipy.signal import lfilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Critic, Driver  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = dict(
    # foresight: about 5,000 decisions (2.8 minutes). At 0.999 (33 s) it couldn't see the wall-rider being parked minutes
    # after the scraping that did it
    gamma=0.9998,
    lam=0.95,
    epochs=5,
    minibatch=32768,
    clip=0.2,
    lr_driver=3e-4,
    lr_critic=1e-3,
    kl_target=0.01,       # the driver's learning rate comes down when an update moves the policy more than this
    lr_min=1e-5,
    # and never goes above where it started: the KL is measured against the noise, so a small KL doesn't mean a small
    # change to how the car drives, and letting the rate climb to 1e-2 on small KLs wrecked the driver within 5 iterations
    lr_max=3e-4,
    grad_clip=1.0,
    # the starting noise, steer then gas. Too much and the exploring car rarely finishes, so the best it can learn is to
    # go fast before the crash (at 0.5 it finished 2%, and the noise-free driver fell from 27 of 32 ovals to 1). Held
    # (below), steering noise goes much further than gas: this is the most steering noise with which both starting
    # brains tried (seeds 7, 8) still finish 85% as many ovals as without noise (77% and 78%, against 84% and 91%)
    # Racing from scratch, twice that steering noise is what got the car off the wall (the foresight pilot,
    # models/ppo-pilot/foresight: 76% of the time against it down to 37% by iteration 180, the other two still at 60%)
    sigma0=(0.03, 0.1),
    # the noise never shrinks below half where it started: left free, the from-scratch run's went to the floor (steer
    # 0.002 by iteration 500, gas 0.12 to 0.013), its learning rate came down with it, and its pace stopped improving
    sigma_floor=0.5,
    # each noise draw lasts 4 decisions (0.13 s). Drawn fresh every decision, the noise was dither the car's wheel and
    # tyres turn into damping: the policy learned to need it (the same noise held made it worse, none made it worse
    # still), and the noise-free driver that races fell behind the one practising. Held 4 with steer 0.015 beat held 8
    # with 0.01 in the speed pilot (models/ppo-pilot/pilot-report.txt: 1.040 against 0.996 after 150 iterations)
    hold=4,
    # a fresh run trains only the critic for its first iterations (Monte Carlo returns): advantages from an untrained
    # critic point the wrong way (one update with them made the starting brain worse, one with a fitted critic better)
    warmup=3,
    # the share of practice episodes that start somewhere round the oval at speed instead of from the grid
    # (js/replay.js rollingStart). Off: tried at 50% when the race-distance run stalled, it made the brain worse
    # from the grid (6 km 1.416 at generation 77 down to 1.239 by 192)
    rolling=0.0,
    # weight on deciding alike from one decision to the next (ppo_update): without it the brain learned a twitchy
    # controller that locked into flip-flopping its steering every decision at Bristol and crashed out there
    smooth=0.0,
    # the clock charged per decision instead of only at the flag (train/ppo/episode.js): the speed incentive arrives
    # where the speed is decided
    timed=False,
)
NAME = 'F·PPO'
PRACTICE_KEPT = 4000      # iterations on the app's learning curve (about 4 hours)


def read_spec():
    return json.loads(subprocess.check_output(['node', str(ROOT / 'train' / 'ppo' / 'spec.js')]))


class Rollouts:
    """The race workers, spoken to in JSON lines (train/ppo/rollouts.js)."""

    def __init__(self, threads, shm):
        self.proc = subprocess.Popen(['node', str(ROOT / 'train' / 'ppo' / 'rollouts.js'), '--threads', str(threads), '--dir', str(shm)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ready = self._reply('ready')

    def _reply(self, kind):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('the race workers stopped')
        reply = json.loads(line)
        if reply['type'] != kind:
            raise RuntimeError(f"race workers: {reply.get('message', reply)}")
        return reply

    def ask(self, msg, kind):
        self.proc.stdin.write(json.dumps(msg) + '\n')
        self.proc.stdin.flush()
        return self._reply(kind)

    def init(self, seed, out):
        self.ask({'type': 'init', 'seed': seed, 'out': str(out)}, 'inited')
        return np.fromfile(out, dtype='<f4')

    def collect(self, iteration, genes_file, sigma, decisions, hold=1, rolling=0.0, timed=False):
        return self.ask({'type': 'collect', 'iteration': iteration, 'genes': str(genes_file), 'sigma': sigma, 'hold': hold, 'rolling': rolling,
                         'timed': timed, 'decisions': decisions}, 'collected')

    def evaluate(self, genes_file, episodes):
        return self.ask({'type': 'evaluate', 'genes': str(genes_file), 'episodes': episodes}, 'evaluated')['episodes']

    def quit(self):
        if self.proc.poll() is None:
            self.proc.stdin.write(json.dumps({'type': 'quit'}) + '\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)


def read_rows(manifest, width, remove=True):
    """Every recorded decision of an iteration, episodes in index order, and each episode's length."""
    files = {}
    for e in manifest['episodes']:
        if e['file'] not in files:
            files[e['file']] = np.fromfile(e['file'], dtype='<f4').reshape(-1, width)
    rows = np.concatenate([files[e['file']][e['row']:e['row'] + e['rows']] for e in manifest['episodes']])
    if remove:
        for f in files:
            os.remove(f)
    return rows, [e['rows'] for e in manifest['episodes']]


def advantages(rewards, values, lengths, gamma, lam):
    """GAE per episode. Every episode ending is terminal (the critic sees time left), so nothing follows the last row."""
    out, at = np.empty(len(rewards)), 0
    for n in lengths:
        r, v = rewards[at:at + n].astype(np.float64), values[at:at + n].astype(np.float64)
        delta = r + gamma * np.append(v[1:], 0.0) - v
        out[at:at + n] = lfilter([1.0], [1.0, -gamma * lam], delta[::-1])[::-1]
        at += n
    return out


def clipped_loss(logp, logp_old, a, clip):
    """PPO's objective, to minimise: the tries weighted by advantage, a try's weight capped once it is clip away from
    the policy that raced (past the cap it contributes no gradient). Returns the loss and log-ratios."""
    log_ratio = logp - logp_old
    ratio = log_ratio.exp()
    return -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean(), log_ratio


def compiled(fn, label):
    """fn through torch.compile (on the GPU the update is mostly passes over memory, one per elementwise op of the
    activation; compiled, each chain is one kernel), falling back to fn as it is if compiling fails."""
    fast, state = torch.compile(fn, dynamic=False), {'ok': True}

    def run(*args):
        if state['ok']:
            try:
                return fast(*args)
            except Exception as error:  # a missing compiler or an unsupported op: the update still runs, just slower
                print(f'torch.compile of {label} failed ({type(error).__name__}: {str(error)[:120]}); running it uncompiled', file=sys.stderr)
                state['ok'] = False
        return fn(*args)
    return run


def ppo_update(driver, critic, opt_d, opt_c, batch, cfg, generator=None):
    """PPO epochs over a batch: the clipped policy objective for the driver, regression on normalised returns for the
    critic, the driver's learning rate adapted to the measured KL. Returns what happened."""
    obs, z, raw, logp_old, adv, ret = batch['obs'], batch['z'], batch['raw'], batch['logp'], batch['adv'], batch['ret']
    # next: each decision's next one in the same episode (-1 on an episode's last), for the smoothness term. Kept a
    # fixed size on the device (an episode's last decision paired with itself, weighted 0): picking rows by a mask
    # would make the CPU ask the GPU how many there are, a wait every step
    nxt, smooth = batch.get('next'), cfg.get('smooth', 0.0)
    if nxt is not None:
        has_next = (nxt >= 0).float()
        nxt = torch.where(nxt >= 0, nxt, torch.arange(len(nxt), device=nxt.device))
    critic.returns.update(ret)
    target = critic.returns.normalize(ret.unsqueeze(1)).squeeze(1)
    n, stats, first_ratio = len(obs), [], None
    mean_of, value_of = getattr(driver, 'fast_mean', driver.mean), getattr(critic, 'fast', critic)
    mb = min(cfg['minibatch'], n)
    for epoch in range(cfg['epochs']):
        perm = torch.randperm(n, device=obs.device, generator=generator)
        # the step's numbers stay on the device until the epoch ends: reading one back makes the CPU wait for the GPU,
        # and five of those a step were most of the update's time
        kls, clips, plosses, vlosses, slosses = [], [], [], [], []
        # whole minibatches only, so a compiled step always sees the same shapes (the rows left over differ each epoch)
        for i in range(0, n - mb + 1, mb):
            idx = perm[i:i + mb]
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            smooth_loss = None
            if smooth and nxt is not None:
                # steer alike from one moment to the next unless it pays: a controller that flip-flops its steering
                # every decision (generation 77 at Bristol, 27 reversals a second) costs here, not in the race score.
                # Steering only: the throttle swinging from flat out to braking into a corner is driving, not a shimmy.
                # The tries and their next decisions go through the network in one pass
                both = mean_of(torch.cat([obs[idx], obs[nxt[idx]]]))
                mu, w = both[:len(idx)], has_next[idx]
                smooth_loss = (w * (mu[:, 0] - both[len(idx):, 0]) ** 2).sum() / w.sum().clamp(min=1.0)
            else:
                mu = mean_of(obs[idx])
            policy_loss, log_ratio = clipped_loss(driver.log_prob_of(mu, raw[idx]), logp_old[idx], a, cfg['clip'])
            loss = policy_loss if smooth_loss is None else policy_loss + smooth * smooth_loss
            opt_d.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(driver.parameters(), cfg['grad_clip'])
            opt_d.step()
            value_loss = ((value_of(z[idx]) - target[idx]) ** 2).mean()
            opt_c.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg['grad_clip'])
            opt_c.step()
            with torch.no_grad():
                ratio = log_ratio.exp()
                if first_ratio is None:
                    first_ratio = (ratio - 1).abs().max()
                kls.append(((ratio - 1) - log_ratio).mean())
                clips.append(((ratio - 1).abs() > cfg['clip']).float().mean())
                plosses.append(policy_loss.detach())
                vlosses.append(value_loss.detach())
                if smooth_loss is not None:
                    slosses.append(smooth_loss.detach())
        # the epoch's numbers read back in one go
        kl, clip_frac, ploss, vloss, *sloss = torch.stack([torch.stack(xs).mean() for xs in (kls, clips, plosses, vlosses, slosses) if xs]).tolist()
        stats.append(dict(kl=kl, clip_frac=clip_frac, policy_loss=ploss, value_loss=vloss, smooth_loss=sloss[0] if sloss else None))
        lr = opt_d.param_groups[0]['lr']
        lr = lr / 1.5 if kl > 2 * cfg['kl_target'] else lr * 1.5 if kl < cfg['kl_target'] / 2 else lr
        for g in opt_d.param_groups:
            g['lr'] = min(cfg['lr_max'], max(cfg['lr_min'], lr))
        if kl > 4 * cfg['kl_target']:
            break
    return dict(epochs=len(stats), first_ratio=float(first_ratio), **stats[-1])


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, separators=(',', ':'), ensure_ascii=False))
    os.replace(tmp, path)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Save:
    """The save folder around the run, written the way evolution writes one: a generation is `every` iterations then an
    evaluation (a time trial on every oval, no exploration, recorded like a tournament so it replays exactly); the
    population is the one policy, F·PPO, as it races. Every iteration: live.json (the weights it raced with) and
    progress.json (its episodes, so the app can race any of them again). Every generation: generations/gen-NNNN.json,
    tournament.json, state.json, then summary.json, which tells the server there's a new generation to pull."""

    def __init__(self, folder, spec, every, threads, carried, config):
        self.folder, self.spec, self.every, self.threads, self.config = folder, spec, every, threads, config
        (folder / 'generations').mkdir(parents=True, exist_ok=True)
        self.history, self.practice = carried.get('history', []), carried.get('practice', [])
        self.decisions, self.episodes = carried.get('decisions', 0), carried.get('episodes', 0)
        self.hours_before, self.started = carried.get('hours', 0.0), carried.get('started', now())
        self.clock = self.since = time.time()

    def hours(self):
        return self.hours_before + (time.time() - self.clock) / 3600

    def carry(self):
        return dict(history=self.history, practice=self.practice, decisions=self.decisions, episodes=self.episodes, hours=self.hours(), started=self.started)

    def has(self, gen):
        return any(h['gen'] == gen for h in self.history)

    label = 'gradient learning (PPO) from random weights'

    def policy(self, genes, noise, iteration):
        d = self.spec['design']
        return dict(name=NAME, founder=NAME, label=self.label, species=d['name'], parent=None, born=0,
                    layers=d['layers'], mask=d['mask'], noise=[round(s, 4) for s in noise], iteration=iteration,
                    genes=genes.astype(np.float64).round(5).tolist())

    def live(self, k, genes, noise):
        write_json(self.folder / 'live.json', dict(generation=k // self.every + 1, round=k % self.every + 1, iteration=k,
                                                   population=[self.policy(genes, noise, k)]))

    def progress(self, k, manifest, noise, asked):
        eps = manifest['episodes']
        self.decisions += manifest['decisions']
        self.episodes += len(eps)
        gen, step = k // self.every + 1, k % self.every + 1
        self.practice.append(dict(gen=gen, round=step, mean=[round(float(np.mean([e['summary']['score'] for e in eps])), 4)]))
        del self.practice[:-PRACTICE_KEPT]
        write_json(self.folder / 'progress.json', dict(
            started=self.started, updated=now(), workers=self.threads, machines=1, generation=gen, phase='learning', round=step, rounds=self.every,
            runs=dict(round=manifest['decisions'], roundTotal=asked, total=self.decisions, perSecond=manifest['perSecond']),
            practiceHistory=self.practice, lastMinutes=self.history[-1]['minutes'] if self.history else None,
            # the iteration's episodes, each raced again exactly from live.json's weights, the noise, its hold and the seed
            ppo=dict(iteration=k, noise=noise, hold=self.config['hold'], hours=round(self.hours(), 3), episodes=[
                dict(index=e['index'], trackId=e['summary']['trackId'], laps=e['summary']['laps'], seed=e['summary']['seed'],
                     steps=e['summary']['steps'], score=e['summary']['score'], **({'rolling': e['summary']['rolling']} if 'rolling' in e['summary'] else {}))
                for e in eps])))

    def generation(self, gen, k, genes, noise, trials, lr, noisy, race):
        policy = self.policy(genes, noise, k)
        finished = [t for t in trials if t['finished']]
        paces = [t['bestLap'] / t['pole'] for t in finished if t['bestLap']]
        # noisy: the same ovals driven with the policy's own noise (fixed seeds); far above the noise-free score means
        # the policy has come to rely on the noise, and the noise-free driver that races is worse than training shows
        stats = dict(points=round(float(np.mean([t['score'] for t in trials])), 4), finished=len(finished), ovals=len(trials),
                     pace=round(float(np.mean(paces)), 4) if paces else None, aero=round(float(np.mean([t['worn'] for t in trials])), 4),
                     retired=sum(bool(t['retired']) for t in trials),
                     noisy=round(float(np.mean([t['score'] for t in noisy])), 4), noisyFinished=sum(bool(t['finished']) for t in noisy),
                     # the same ovals at race distance, the distance it practises at
                     race=round(float(np.mean([t['score'] for t in race])), 4), raceFinished=sum(bool(t['finished']) for t in race),
                     raceParked=sum(bool(t.get('parked')) for t in race), raceAero=round(float(np.mean([t['worn'] for t in race])), 4),
                     raceWall=round(float(np.mean([t.get('wall', 0) for t in race])), 4),
                     # the shimmy gauge on the 6 km panel: mean steering jitter, and the worst oval's reversals a second
                     jitter=round(float(np.mean([t.get('jitter', 0) for t in trials])), 4),
                     shimmy=max(({'trackId': t['trackId'], 'reversals': round(t.get('reversals', 0), 1)} for t in trials), key=lambda x: x['reversals']),
                     swing=round(float(np.mean([t.get('swing', 0) for t in trials])), 4),
                     swingWorst=max(({'trackId': t['trackId'], 'swing': round(t.get('swing', 0), 3)} for t in trials), key=lambda x: x['swing']))
        entry = dict(gen=gen, at=now(), races=len(trials), species={policy['species']: stats}, agents=[dict(name=NAME, species=policy['species'], **stats)],
                     champion=NAME, replaced=[], minutes=round((time.time() - self.since) / 60, 2),
                     ppo=dict(iteration=k, decisions=self.decisions, episodes=self.episodes, noise=policy['noise'], lr=lr, hours=round(self.hours(), 3)))
        self.since = time.time()
        self.history = [h for h in self.history if h['gen'] < gen] + [entry]
        write_json(self.folder / 'generations' / f'gen-{gen:04d}.json', dict(generation=gen, population=[policy]))
        write_json(self.folder / 'tournament.json', dict(gen=gen, mode=dict(tracks='nascar', cars='stock'), population=[policy], races=[
            dict(scenario=dict(kind='tt', trackId=t['trackId'], laps=t['laps'], cars='stock'), grid=[NAME], order=[NAME],
                 time=dict(steps=t['steps'], finished=t['finished'], score=t['score'])) for t in trials]))
        write_json(self.folder / 'state.json', dict(started=self.started, generation=gen, history=self.history, population=[policy],
                                                    practiceRuns=self.episodes, practiceHistory=self.practice,
                                                    ppo=dict(every=self.every, decisions=self.decisions, hours=round(self.hours(), 3), config=self.config)))
        write_json(self.folder / 'summary.json', dict(generation=gen, champion=NAME, updated=entry['at'], designs={policy['species']: stats['points']}))
        return entry


def fit_critic(critic, opt_c, z, ret, cfg, generator=None):
    """The critic alone, for the warm-up: regression on normalised returns, cfg epochs of minibatches."""
    critic.returns.update(ret)
    target = critic.returns.normalize(ret.unsqueeze(1)).squeeze(1)
    losses = []
    for epoch in range(cfg['epochs']):
        perm = torch.randperm(len(z), device=z.device, generator=generator)
        for i in range(0, len(z), cfg['minibatch']):
            idx = perm[i:i + cfg['minibatch']]
            value_loss = ((critic(z[idx]) - target[idx]) ** 2).mean()
            opt_c.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg['grad_clip'])
            opt_c.step()
            losses.append(value_loss.detach())
    return dict(epochs=cfg['epochs'], first_ratio=0.0, kl=0.0, clip_frac=0.0, policy_loss=None,
                value_loss=float(torch.stack(losses[-(len(losses) // cfg['epochs']):]).mean()))


def save_checkpoint(path, iteration, driver, critic, opt_d, opt_c, save):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(iteration=iteration, genes=driver.genes(), driver=driver.state_dict(), critic=critic.state_dict(),
                    opt_d=opt_d.state_dict(), opt_c=opt_c.state_dict(), save=save.carry()), tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    p.add_argument('--shm', default=None)
    p.add_argument('--threads', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--decisions', type=int, default=600_000)
    p.add_argument('--iterations', type=int, default=10**9)
    p.add_argument('--hours', type=float, default=1e9)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--generation-every', type=int, default=10, help='learning iterations per generation (an evaluation and the save files)')
    p.add_argument('--sigma0', help="the starting noise: one level, or steer,gas (default CONFIG['sigma0']; a resumed run keeps its own)")
    p.add_argument('--lr-max', type=float, help="the driver's learning-rate ceiling (default CONFIG['lr_max'])")
    p.add_argument('--lr', type=float, help="the driver's learning rate, where it starts and its ceiling")
    p.add_argument('--kl-target', type=float, help="how far an update may move the policy before the rate comes down (default CONFIG['kl_target'])")
    p.add_argument('--clip', type=float, help="PPO's clip on each try's weight (default CONFIG['clip'])")
    p.add_argument('--hold', type=int, help="decisions each noise draw lasts (default CONFIG['hold'])")
    p.add_argument('--gamma', type=float, help="how far ahead it looks, as a discount per decision (default CONFIG['gamma'])")
    p.add_argument('--lam', type=float, help="GAE lambda (default CONFIG['lam'])")
    p.add_argument('--rolling', type=float, help="share of practice episodes started round the oval at speed (default CONFIG['rolling'])")
    p.add_argument('--profile', action='store_true', help='profile the first learning update and print where its time goes')
    args = p.parse_args()

    # plain float32 everywhere, so the learner's driver computes what the JS brain does
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    sigma0 = [float(s) for s in args.sigma0.split(',')] if args.sigma0 else CONFIG['sigma0']
    given = {'lr_max': args.lr_max, 'lr_driver': args.lr, 'kl_target': args.kl_target, 'clip': args.clip, 'hold': args.hold,
             'gamma': args.gamma, 'lam': args.lam, 'rolling': args.rolling}
    cfg = {**CONFIG, 'sigma0': sigma0[0] if isinstance(sigma0, list) and len(sigma0) == 1 else sigma0,
           **{k: v for k, v in given.items() if v is not None}, **({'lr_max': args.lr} if args.lr else {})}
    deadline, device = time.time() + args.hours * 3600, torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    print(f'learning on {device_name}', file=sys.stderr)
    run = pathlib.Path(args.dir)
    run.mkdir(parents=True, exist_ok=True)
    shm = pathlib.Path(args.shm or tempfile.mkdtemp(prefix='ppo-rows-'))
    shm.mkdir(parents=True, exist_ok=True)
    spec, ckpt, metrics = read_spec(), run / 'checkpoint.pt', run / 'metrics.jsonl'
    row, obs_n = spec['row'], spec['inputs']
    # a save's config.json overrides settings for its run (the watcher turns on the next lever when a run stalls)
    if (run.parent / 'config.json').exists():
        cfg.update({k: v for k, v in json.loads((run.parent / 'config.json').read_text()).items() if k in CONFIG})
        print(f"config from the save: {json.loads((run.parent / 'config.json').read_text())}", file=sys.stderr)
    # a save can start from a trained brain of design F instead of random weights: its start.json (a brain record)
    begin = json.loads((run.parent / 'start.json').read_text()) if (run.parent / 'start.json').exists() else None

    driver, critic = Driver(spec, cfg['sigma0'], cfg['sigma_floor']).to(device), Critic(obs_n + len(spec['extras'])).to(device)
    opt_d = torch.optim.Adam(driver.parameters(), lr=cfg['lr_driver'], eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg['lr_critic'], eps=1e-5)
    if device.type == 'cuda':
        driver.fast_mean, critic.fast = compiled(driver.mean, "the driver"), compiled(critic, "the critic")
    rollouts = Rollouts(args.threads, shm)
    try:
        start, carried = 0, {}
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=False)
            driver.load_state_dict(state['driver'])
            critic.load_state_dict(state['critic'])
            opt_d.load_state_dict(state['opt_d'])
            opt_c.load_state_dict(state['opt_c'])
            start, carried = state['iteration'], state.get('save', {})
            print(f'resuming at iteration {start}', file=sys.stderr)
        elif begin:
            driver.load_genes(np.array(begin['genes'], dtype=np.float32))
            print(f"starting from {begin['name']} ({begin.get('from', 'start.json')})", file=sys.stderr)
        else:
            driver.load_genes(rollouts.init(args.seed, shm / 'genes-init.bin'))
        assert driver.locked_at_zero()
        save, every = Save(run.parent, spec, args.generation_every, args.threads, carried, cfg), args.generation_every
        if begin:
            save.label = f"gradient learning (PPO) from {begin['name']}"
        noise = lambda: [float(s) for s in driver.sigma().detach().cpu()]

        def evaluate(k):
            genes_file = shm / f'genes-judged-{k}.bin'
            driver.genes().tofile(genes_file)
            trials = rollouts.evaluate(genes_file, spec['panel'])
            noisy = rollouts.evaluate(genes_file, [dict(p, sigma=noise(), seed=9_000_001 + i, hold=cfg['hold']) for i, p in enumerate(spec['panel'])])
            race = rollouts.evaluate(genes_file, spec['racePanel'])
            os.remove(genes_file)
            s = save.generation(k // every, k, driver.genes(), noise(), trials, opt_d.param_groups[0]['lr'], noisy, race)['species'][spec['design']['name']]
            pace = f"{s['pace']:.3f}" if s['pace'] else '—'
            print(f"generation {k // every} (iteration {k}): time trials {s['points']:.3f}, {s['finished']}/{s['ovals']} ovals finished, pace {pace} of pole"
                  f" | with its noise {s['noisy']:.3f} (gap {s['noisy'] - s['points']:+.3f}) | 24 km {s['race']:.3f}, {s['raceFinished']} finished,"
                  f" {s['raceParked']} parked, damage {s['raceAero']:.2f}, on the wall {s['raceWall']:.0%} | steering jitter {s['jitter']:.3f},"
                  f" worst {s['shimmy']['trackId']} {s['shimmy']['reversals']}/s | steering swing {s['swing']:.3f},"
                  f" worst {s['swingWorst']['trackId']} {s['swingWorst']['swing']}", file=sys.stderr)

        done = start
        for k in range(start, args.iterations):
            if time.time() > deadline:
                break
            if k % every == 0 and not save.has(k // every):
                evaluate(k)
            t0 = time.time()
            genes_file = shm / f'genes-{k}.bin'
            driver.genes().tofile(genes_file)
            sigma = noise()
            manifest = rollouts.collect(k, genes_file, sigma, args.decisions, cfg['hold'], cfg.get('rolling', 0.0), cfg.get('timed', False))
            rows, lengths = read_rows(manifest, row['width'])
            os.remove(genes_file)
            save.live(k, driver.genes(), sigma)
            t1 = time.time()

            t = torch.from_numpy(rows).to(device)
            obs, extras = t[:, row['obs'][0]:row['obs'][1]], t[:, row['extras'][0]:row['extras'][1]]
            z = torch.cat([obs, extras], dim=1)
            with torch.no_grad():
                values = torch.cat([critic.value(z[i:i + 65536]) for i in range(0, len(z), 65536)])
            warming = k < cfg['warmup']
            # warming up: lambda 1, so the targets are the plain discounted returns, whatever the critic says yet
            adv = advantages(rows[:, row['reward']], values.cpu().numpy(), lengths, cfg['gamma'], 1.0 if warming else cfg['lam'])
            ret = adv + values.cpu().numpy()
            critic.inputs.update(z)
            if warming:
                update = fit_critic(critic, opt_c, z, torch.from_numpy(ret).float().to(device), cfg)
            else:
                nxt = np.arange(1, len(rows) + 1)
                nxt[np.cumsum(lengths) - 1] = -1
                batch = dict(obs=obs, z=z, raw=t[:, row['raw'][0]:row['raw'][1]], logp=t[:, row['logp']],
                             adv=torch.from_numpy(adv).float().to(device), ret=torch.from_numpy(ret).float().to(device),
                             next=torch.from_numpy(nxt).to(device))
                if args.profile and not getattr(args, 'profiled', False):
                    from torch.profiler import ProfilerActivity, profile
                    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == 'cuda' else [])
                    with profile(activities=activities) as prof:
                        update = ppo_update(driver, critic, opt_d, opt_c, batch, cfg)
                    sort = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
                    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=18), file=sys.stderr)
                    print(prof.key_averages().table(sort_by=sort, row_limit=12), file=sys.stderr)
                    args.profiled = True
                else:
                    update = ppo_update(driver, critic, opt_d, opt_c, batch, cfg)
            driver.quantize_()
            assert driver.locked_at_zero(), 'a locked connection moved'
            explained = 1 - float(np.var(ret - values.cpu().numpy()) / max(np.var(ret), 1e-12))
            t2 = time.time()

            eps = [e['summary'] for e in manifest['episodes']]
            laps = [e['bestLap'] for e in eps if e['finished'] and e['bestLap']]
            line = dict(iteration=k, at=time.strftime('%Y-%m-%dT%H:%M:%S'), device=device_name, decisions=manifest['decisions'], episodes=len(eps),
                        collect_s=round(t1 - t0, 2), update_s=round(t2 - t1, 2), per_second=manifest['perSecond'],
                        score=float(np.mean([e['score'] for e in eps])), finished=float(np.mean([e['finished'] for e in eps])),
                        best_lap=float(np.mean(laps)) if laps else None, worn=float(np.mean([e['worn'] for e in eps])),
                        sigma=sigma, lr_driver=opt_d.param_groups[0]['lr'], lr_critic=opt_c.param_groups[0]['lr'],
                        explained_variance=explained, entropy=float(driver.entropy().detach()), warmup=warming, **update)
            with open(metrics, 'a') as f:
                f.write(json.dumps(line) + '\n')
            save.progress(k, manifest, sigma, args.decisions)
            print(f"iter {k}{' (critic warm-up)' if warming else ''}: score {line['score']:.3f} finished {line['finished']:.0%} sigma {sigma[0]:.2f}/{sigma[1]:.2f} "
                  f"kl {line['kl']:.4f} ev {explained:.2f} ({line['decisions']} decisions, {t1 - t0:.1f}+{t2 - t1:.1f} s)", file=sys.stderr)
            done = k + 1
            if done % 10 == 0:
                save_checkpoint(ckpt, done, driver, critic, opt_d, opt_c, save)
        # a run that stops on a generation's last iteration gets that generation's evaluation
        if done % every == 0 and not save.has(done // every):
            evaluate(done)
        save_checkpoint(ckpt, done, driver, critic, opt_d, opt_c, save)
    finally:
        rollouts.quit()


if __name__ == '__main__':
    main()
