"""Design F in PyTorch, computed the way the JS brain computes it (js/nn.js), and the critic that trains it.

The driver's weights travel in the engine's gene layout (train/ppo/spec.js design.layout): every neuron is
[bias, a weight from each input], and the outputs read the mixing layer, then the first layer (the shortcuts).
What the learner holds is always rounded to the 5 decimals the engine publishes, so it is exactly what races.
"""
import math

import numpy as np
import torch
from torch import nn


def squash(x):
    # js/nn.js: -1 at or below -3, 1 at or above 3, else x (27 + x^2) / (27 + 9 x^2); flat beyond +-3
    xc = x.clamp(-3.0, 3.0)
    return xc * (27 + xc * xc) / (27 + 9 * xc * xc)


def quantize(weights):
    # the engine's quantizeGenes: +g.toFixed(5) on each float32, back to float32. A float32 is never exactly halfway
    # between two 5-decimal numbers, so correctly rounded decimal formatting agrees with toFixed on every value
    return np.array([float(f'{w:.5f}') for w in np.asarray(weights, dtype=np.float32).astype(np.float64).ravel()], dtype=np.float32)


class Driver(nn.Module):
    """Design F: 160 lane neurons on the road and self senses, 48 traffic neurons on the traffic senses (the rest of
    the first layer locked at 0), 48 mixing neurons, and both blocks wired straight to steer and gas."""

    def __init__(self, spec, sigma=0.5, floor=None):
        super().__init__()
        design = spec['design']
        n, h1, h2, outs = design['layers']
        assert design['shortcuts'] and [d['reads'] for d in design['layout']] == [[0], [1], [2, 1]], 'expects design F'
        self.layout = design['layout']
        self.W1, self.b1 = nn.Parameter(torch.zeros(h1, n)), nn.Parameter(torch.zeros(h1))
        self.W2, self.b2 = nn.Parameter(torch.zeros(h2, h1)), nn.Parameter(torch.zeros(h2))
        self.W3, self.b3 = nn.Parameter(torch.zeros(outs, h2 + h1)), nn.Parameter(torch.zeros(outs))
        mask = torch.ones(h1, n)
        for block in design['mask']:
            mask[block['from']:block['to']] = 0
            mask[block['from']:block['to'], block['inputs']] = 1
        self.register_buffer('mask', mask)
        self.register_buffer('mirror_from', torch.tensor(spec['mirror']['from'], dtype=torch.long))
        self.register_buffer('mirror_sign', torch.tensor(spec['mirror']['sign'], dtype=torch.float32))
        # the starting noise: one level for both hands, or [steer, gas]
        sigma = list(sigma) if isinstance(sigma, (list, tuple)) else [sigma] * outs
        self.log_sigma = nn.Parameter(torch.tensor([math.log(s) for s in sigma]))
        # the noise never goes below floor x where it started (left to itself it shrank to nothing, and with it the
        # exploring: the from-scratch run's pace stopped improving once it had); not saved, it comes from the config
        self.register_buffer('log_floor', torch.tensor([math.log(max(0.002, (floor or 0) * s)) for s in sigma]), persistent=False)

    def weights(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def outputs(self, x):
        a1 = squash(x @ (self.W1 * self.mask).T + self.b1)
        a2 = squash(a1 @ self.W2.T + self.b2)
        return squash(torch.cat([a2, a1], dim=1) @ self.W3.T + self.b3)

    def mean(self, x):
        # the car's decision: the real view and the mirror image averaged (steer flips with the world), both views
        # through the network in one pass
        both = self.outputs(torch.cat([x, x[:, self.mirror_from] * self.mirror_sign]))
        real, flip = both[:len(x)], both[len(x):]
        return torch.stack([(real[:, 0] - flip[:, 0]) / 2, (real[:, 1] + flip[:, 1]) / 2], dim=1)

    def sigma(self):
        return torch.maximum(self.log_sigma, self.log_floor).clamp(max=0.0).exp()

    def log_prob(self, x, raw):
        return self.log_prob_of(self.mean(x), raw)

    def log_prob_of(self, mu, raw):
        sd = self.sigma()
        return (-0.5 * ((raw - mu) / sd) ** 2 - torch.log(sd) - 0.5 * math.log(2 * math.pi)).sum(1)

    def entropy(self):
        return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma())).sum()

    @torch.no_grad()
    def load_genes(self, genes):
        genes = np.asarray(genes, dtype=np.float32)
        for (W, b), part in zip([(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3)], self.layout):
            block = genes[part['at']:part['at'] + part['neurons'] * (part['inputs'] + 1)].reshape(part['neurons'], part['inputs'] + 1)
            b.copy_(torch.from_numpy(block[:, 0].copy()))
            W.copy_(torch.from_numpy(block[:, 1:].copy()))

    @torch.no_grad()
    def genes(self):
        return np.concatenate([torch.cat([b[:, None], W], dim=1).cpu().numpy().ravel()
                               for W, b in [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3)]]).astype(np.float32)

    @torch.no_grad()
    def quantize_(self):
        for p in self.weights():
            p.copy_(torch.from_numpy(quantize(p.detach().cpu().numpy()).reshape(p.shape)).to(p.device))

    @torch.no_grad()
    def locked_at_zero(self):
        return bool((self.W1[self.mask == 0] == 0).all())


class RunningNorm(nn.Module):
    """A running mean and variance (merged batch by batch), for the critic's inputs and for returns."""

    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size, dtype=torch.float64))
        self.register_buffer('var', torch.ones(size, dtype=torch.float64))
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.double().reshape(len(x), -1)
        m, v, n = x.mean(0), x.var(0, unbiased=False), len(x)
        delta, total = m - self.mean, self.count + n
        self.var.copy_((self.var * self.count + v * n + delta ** 2 * self.count * n / total) / total)
        self.mean.copy_(self.mean + delta * n / total)
        self.count.copy_(total)

    def normalize(self, x):
        return ((x - self.mean.float()) / (self.var.float().sqrt() + 1e-8)).clamp(-10, 10)

    def denormalize(self, x):
        return x * (self.var.float().sqrt() + 1e-8) + self.mean.float()


class Critic(nn.Module):
    """How much more score a moment is worth. Sees the senses plus race state the driver doesn't (spec extras);
    shares nothing with the driver and is never exported. Predicts returns in normalised units."""

    def __init__(self, inputs, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0 if layer.out_features == 1 else math.sqrt(2))
                nn.init.zeros_(layer.bias)
        self.inputs = RunningNorm(inputs)
        self.returns = RunningNorm(1)

    def forward(self, z):
        return self.net(self.inputs.normalize(z)).squeeze(1)

    def value(self, z):
        return self.returns.denormalize(self(z).unsqueeze(1)).squeeze(1)
