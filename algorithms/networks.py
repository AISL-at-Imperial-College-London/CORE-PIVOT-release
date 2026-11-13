"""Small networks shared by the offline learners; actions are normalized."""
import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def training_statistics(observations, chunk_size=100_000):
    """Parallel-variance merge avoids a full float64 copy of the 24m-row cache."""
    count = 0
    mean = np.zeros(observations.shape[1], dtype=np.float64)
    squared_deviation = np.zeros_like(mean)
    for start in range(0, len(observations), chunk_size):
        chunk = np.asarray(observations[start:start + chunk_size], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError('Non-finite training observations.')
        size = len(chunk)
        chunk_mean = chunk.mean(0)
        delta = chunk_mean - mean
        total = count + size
        squared_deviation += chunk.var(0) * size + np.square(delta) * count * size / total
        mean += delta * size / total
        count = total
    return mean, np.maximum(np.sqrt(squared_deviation / count), 1e-6)


class Normalizer(nn.Module):
    """Statistics fitted on training observations only, saved with every network."""
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())

    def forward(self, observations):
        return (observations - self.mean) / self.scale


class TwinQ(nn.Module):
    def __init__(self, mean, scale, hidden=256):
        super().__init__()
        self.normalize = Normalizer(mean, scale)
        self.q1 = mlp(len(mean) + 3, hidden, 1)
        self.q2 = mlp(len(mean) + 3, hidden, 1)

    def forward(self, observations, actions):
        inputs = torch.cat((self.normalize(observations), actions), dim=-1)
        return self.q1(inputs), self.q2(inputs)

    def minimum(self, observations, actions):
        first, second = self(observations, actions)
        return torch.minimum(first, second)


class DeterministicActor(nn.Module):
    def __init__(self, mean, scale, hidden=256):
        super().__init__()
        self.normalize = Normalizer(mean, scale)
        self.network = mlp(len(mean), hidden, 3)

    def forward(self, observations):
        return self.network(self.normalize(observations)).tanh()


class SquashedActor(nn.Module):
    """Bounded Gaussian with the tanh change-of-variables density."""
    def __init__(self, mean, scale, hidden=256):
        super().__init__()
        self.normalize = Normalizer(mean, scale)
        self.network = mlp(len(mean), hidden, 6)

    def distribution(self, observations):
        mean, log_std = self.network(self.normalize(observations)).chunk(2, dim=-1)
        return Normal(mean, log_std.clamp(-5, 2).exp())

    def forward(self, observations):
        return self.distribution(observations).mean.tanh()

    def sample(self, observations, generator):
        distribution = self.distribution(observations)
        noise = torch.randn(distribution.mean.shape, device=observations.device,
                            generator=generator)
        latent = distribution.mean + distribution.stddev * noise
        actions = latent.tanh()
        # Stable log(1 - tanh(u)^2); avoid bias from a clamped action Jacobian.
        correction = 2 * (np.log(2.) - latent - nn.functional.softplus(-2 * latent))
        log_prob = (distribution.log_prob(latent) - correction).sum(-1, keepdim=True)
        return actions, log_prob

    def log_prob(self, observations, actions):
        bounded = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        latent = torch.atanh(bounded)
        correction = torch.log1p(-bounded.square())
        return (self.distribution(observations).log_prob(latent) - correction).sum(-1, keepdim=True)


def frozen_copy(module):
    return copy.deepcopy(module).eval().requires_grad_(False)


@torch.no_grad()
def soft_update(source, target, tau):
    for parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.lerp_(parameter, tau)


def optimize(optimizer, loss):
    """Fail loudly on numerical errors; do not silently skip bad updates."""
    if not torch.isfinite(loss):
        raise ValueError('Non-finite loss.')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError('Non-finite gradient.')
    optimizer.step()
