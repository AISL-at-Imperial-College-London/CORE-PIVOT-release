"""Truncated-normal behavior MLE for Brandfonbrener et al., NeurIPS 2021.

Paper: https://arxiv.org/abs/2106.08909, Appendix C.1. This independently
implemented distribution is not clipping or a tanh-transformed Gaussian.
Explicit numerical choices: tanh-bounded Gaussian location, state-dependent
log scale bounded to [-5, 2], float64 CDF/inverse-CDF arithmetic.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from algorithms.networks import Normalizer, mlp, optimize
from algorithms.offline import OfflineLearner


class TruncatedNormal:
    """Independent normal coordinates conditioned on normalized [-1, 1]."""
    def __init__(self, location, scale):
        if not torch.isfinite(location).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Expected finite location and positive scale.')
        if (location.abs() > 1).any():
            raise ValueError('This stable implementation requires a bounded location.')
        self.dtype = location.dtype
        self.location, self.scale = location.double(), scale.double()
        self.alpha = (-1 - self.location) / self.scale
        self.beta = (1 - self.location) / self.scale
        self.lower = torch.special.ndtr(self.alpha)
        self.upper = torch.special.ndtr(self.beta)
        self.mass = self.upper - self.lower
        if (self.mass <= 0).any():
            raise ValueError('Degenerate truncation normalizer.')

    def log_prob(self, actions):
        actions = actions.double()
        z = (actions - self.location) / self.scale
        value = -.5 * z.square() - self.scale.log() - .5 * math.log(2 * math.pi) - self.mass.log()
        return torch.where(actions.abs() <= 1, value, torch.full_like(value, -torch.inf))

    def sample(self, generator):
        # CPU uniforms give checkpoint-portable private RNG state.
        u = torch.rand(self.location.shape, dtype=torch.float64, generator=generator).to(self.location.device)
        p = self.lower + self.mass * u
        eps = torch.finfo(torch.float64).eps
        z = torch.special.ndtri(p.clamp(eps, 1 - eps))
        action = (self.location + self.scale * z).to(self.dtype)
        if not torch.isfinite(action).all() or (action.abs() > 1).any():
            raise ValueError('Inverse-CDF sample outside truncation support.')
        return action

    @property
    def mean(self):
        pdf = lambda z: torch.exp(-.5 * z.square()) / math.sqrt(2 * math.pi)
        return (self.location + self.scale * (pdf(self.alpha) - pdf(self.beta)) / self.mass).to(self.dtype)


class TruncatedPolicy(nn.Module):
    def __init__(self, obs_mean, obs_std, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.normalize = Normalizer(obs_mean, obs_std)
        self.network = mlp(25, hidden_size, 6)

    def distribution(self, observations):
        location, log_std = self.network(self.normalize(observations)).chunk(2, dim=-1)
        return TruncatedNormal(location.tanh(), log_std.clamp(-5, 2).exp())


@dataclass(frozen=True)
class TruncatedBCConfig:
    hidden_size: int = 1024
    batch_size: int = 512
    learning_rate: float = 1e-4


class TruncatedBC(OfflineLearner):
    config_type = TruncatedBCConfig

    def __init__(self, observations, config=TruncatedBCConfig(), seed=123, device='cpu'):
        super().__init__(observations, config, seed, device)

    def _build(self, proposal):
        if proposal is not None:
            raise ValueError('Behavior cloning has no pretrained proposal.')
        self.generator = torch.Generator().manual_seed(self.generator.initial_seed())
        self.policy = TruncatedPolicy(self.mean, self.scale, self.config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.modules = dict(policy=self.policy)
        self.optimizers = dict(policy=self.optimizer)

    def update(self, batch):
        log_prob = self.policy.distribution(batch['observations']).log_prob(batch['actions'])
        loss = -log_prob.sum(-1).mean()
        optimize(self.optimizer, loss)
        self.step += 1
        return dict(step=self.step, neglogp=float(loss.detach()))

    @torch.no_grad()
    def predict(self, observations):
        return self.policy.distribution(self._tensor(observations)).mean.cpu().numpy()
