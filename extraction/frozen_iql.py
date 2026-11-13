"""The paper's fixed-sigma AWR / DDPG+BC comparison.

Numerical operations follow the archived November 3/4 extraction workers.
The only changes are explicit inputs and portable I/O. In particular, this
uses the frozen *target* twin minimum, not the online IQL critic. DDPG+BC
optimises the distribution mean; both objectives retain Normal likelihood BC
and sigma=1. This is not a new stochastic action-selection rule.
"""
import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from algorithms.corl_vendor.iql import GaussianPolicy, TwinQ, ValueFunction


def module_hash(modules):
    digest = hashlib.sha256()
    for name, module in sorted(modules.items()):
        for key, value in sorted(module.state_dict().items()):
            digest.update((name + '/' + key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ExtractionConfig:
    seed: int = 123
    kind: str = 'awr'
    coefficient: float = 5.0  # AWR beta, or DDPG+BC behavioural weight
    batch_size: int = 256
    learning_rate: float = 3e-4
    cap: float = 20.0
    q_scale_samples: int = 65536

    def __post_init__(self):
        if self.kind not in ('awr', 'ddpg_bc'):
            raise ValueError('kind must be awr or ddpg_bc')
        if self.coefficient <= 0 or self.batch_size <= 0:
            raise ValueError('coefficient and batch_size must be positive')


class FrozenIQLExtractor:
    def __init__(self, source, config, device='cpu'):
        if source['step'] != 500000:
            raise ValueError('The paper comparison freezes the 500k IQL checkpoint.')
        self.config, self.device = config, device
        self.mean = torch.as_tensor(source['mean'], dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(source['scale'], dtype=torch.float32, device=device)
        if self.mean.shape != (25,) or self.scale.shape != (25,) or not (self.scale > 0).all():
            raise ValueError('Expected the frozen training-only 25-dimensional normalizer.')
        self.q = TwinQ(25, 3).to(device)
        self.q.load_state_dict(source['modules']['q_target'])
        self.v = ValueFunction(25).to(device)
        self.v.load_state_dict(source['modules']['vf'])
        self.original = GaussianPolicy(25, 3, 1.).to(device)
        self.original.load_state_dict(source['modules']['actor'])
        self.frozen = {'q': self.q, 'v': self.v, 'original': self.original}
        for module in self.frozen.values():
            module.eval().requires_grad_(False)
        # Preserve the archived fresh-actor initialization and sampler streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 91823)
            self.actor = GaussianPolicy(25, 3, 1.).to(device)
        self.actor.log_std.data.zero_()
        self.actor.log_std.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(config.seed + 78001)
        self.step, self.qscale = 0, 1.
        self.initial_frozen = module_hash(self.frozen)

    def norm(self, observations):
        return (observations - self.mean) / self.scale

    def batch(self, observations, actions):
        indices = torch.randint(len(observations), (self.config.batch_size,), generator=self.generator).numpy()
        return (torch.tensor(np.array(observations[indices]), device=self.device),
                torch.tensor(np.array(actions[indices]), device=self.device))

    def estimate_q_scale(self, observations, actions):
        generator = torch.Generator().manual_seed(self.config.seed + 123456)
        indices = torch.randint(len(observations), (self.config.q_scale_samples,), generator=generator).numpy()
        total = 0.
        with torch.no_grad():
            for ids in np.array_split(indices, 64):
                z = self.norm(torch.tensor(np.array(observations[ids]), device=self.device))
                a = torch.tensor(np.array(actions[ids]), device=self.device)
                total += float(self.q(z, a).abs().double().sum())
        self.qscale = max(total / len(indices), 1e-6)
        return self.qscale

    def update(self, observations, actions):
        z = self.norm(observations)
        distribution = self.actor(z)
        nll = -distribution.log_prob(actions).sum(-1)
        if self.config.kind == 'awr':
            with torch.no_grad():
                advantage = self.q(z, actions) - self.v(z)
                log_weights = self.config.coefficient * advantage
                weights = log_weights.clamp(max=float(np.log(self.config.cap))).exp()
            loss = (weights * nll).mean()
            extra = dict(ess=float(weights.sum().square() / weights.square().sum().clamp_min(1e-30)),
                         clipped=float((log_weights >= np.log(self.config.cap)).float().mean()))
        else:
            values = self.q(z, distribution.mean)
            loss = -values.mean() / self.qscale + self.config.coefficient * nll.mean()
            extra = dict(actor_q=float(values.detach().mean()))
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite extraction loss')
        self.optimizer.zero_grad()
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.actor.parameters()):
            raise FloatingPointError('Non-finite actor gradient')
        self.optimizer.step()
        self.step += 1
        assert torch.equal(self.actor.log_std, torch.zeros_like(self.actor.log_std))
        assert self.actor.log_std.grad is None
        assert all(p.grad is None for module in self.frozen.values() for p in module.parameters())
        return dict(loss=float(loss.detach()), nll=float(nll.detach().mean()),
                    policy_std=distribution.stddev.detach().mean(0).cpu().tolist(),
                    mean_action_abs=float(distribution.mean.detach().abs().mean()),
                    action_near_bound=float((distribution.mean.detach().abs() > .95).float().mean()),
                    qscale=self.qscale, **extra)

    def snapshot(self):
        return dict(config=self.config.__dict__, step=self.step, qscale=self.qscale,
                    actor=copy.deepcopy(self.actor.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    rng=self.generator.get_state(), frozen=self.initial_frozen)

    def restore(self, state):
        if state['config'] != self.config.__dict__ or state['frozen'] != self.initial_frozen:
            raise ValueError('Extraction configuration or frozen source changed')
        self.actor.load_state_dict(state['actor'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.generator.set_state(state['rng'])
        self.step, self.qscale = state['step'], state['qscale']

    def assert_frozen(self):
        assert module_hash(self.frozen) == self.initial_frozen
