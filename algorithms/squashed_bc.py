"""Density-bearing bounded BC proposal for CORE-PIVOT, not clipped Gaussian BC."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from algorithms.networks import SquashedActor, optimize, training_statistics


@dataclass(frozen=True)
class SquashedBCConfig:
    hidden_size: int = 256
    learning_rate: float = 3e-4
    batch_size: int = 256


class SquashedPolicy(SquashedActor):
    loss_names = ('neglogp', 'entropy')

    def __init__(self, config, mean, scale):
        super().__init__(mean, scale, config.hidden_size)
        self.config = config

    def loss(self, observations, actions, generator):
        # At the fixed action limits, evaluate the finite interior approximation
        # in log_prob. It is explicit, not the density of a clipped Gaussian.
        neglogp = -self.log_prob(observations, actions).mean()
        with torch.no_grad():
            _, log_prob = self.sample(observations, generator)
        return neglogp, neglogp, -log_prob.mean()

    @torch.no_grad()
    def predict(self, observations, deterministic=True, generator=None):
        observations = torch.as_tensor(np.asarray(observations), dtype=torch.float32,
                                       device=self.normalize.mean.device)
        if deterministic:
            actions = self(observations)
        else:
            if generator is None:
                raise ValueError('Stochastic prediction requires a seeded generator.')
            actions, _ = self.sample(observations, generator)
        return actions.cpu().numpy()


class SquashedBC:
    """Supervised CPU trainer with the same readable API as the other BC policies."""
    def __init__(self, observations, config=SquashedBCConfig(), seed=123):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != 25 or not len(observations):
            raise ValueError('Expected nonempty N×25 training observations.')
        if min(config.hidden_size, config.learning_rate, config.batch_size) <= 0:
            raise ValueError('Invalid bounded BC configuration.')
        torch.manual_seed(seed)
        mean, scale = training_statistics(observations)
        self.policy = SquashedPolicy(config, mean, scale)
        self.actor = self.policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(seed)
        self.step = 0

    def update(self, observations, actions, indices=None):
        if indices is None:
            indices = torch.randint(len(observations), (self.policy.config.batch_size,),
                                    generator=self.generator)
        loss, neglogp, entropy = self.policy.loss(observations[indices], actions[indices], self.generator)
        optimize(self.optimizer, loss)
        self.step += 1
        return dict(step=self.step, loss=float(loss.detach()),
                    neglogp=float(neglogp.detach()), entropy=float(entropy.detach()))

    def save(self, path):
        with Path(path).open('xb') as stream:
            torch.save(dict(config=asdict(self.policy.config), model=self.policy.state_dict(),
                            optimizer=self.optimizer.state_dict(), rng=self.generator.get_state(),
                            step=self.step), stream)

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location=device, weights_only=True)
        trainer = cls.__new__(cls)
        config = SquashedBCConfig(**state['config'])
        devices = [] if torch.device(device).type == 'cpu' else [torch.device(device).index or 0]
        with torch.random.fork_rng(devices=devices):
            trainer.policy = SquashedPolicy(config, state['model']['normalize.mean'],
                                            state['model']['normalize.scale']).to(device)
        trainer.policy.load_state_dict(state['model'])
        trainer.actor = trainer.policy
        trainer.optimizer = torch.optim.Adam(trainer.policy.parameters(), lr=config.learning_rate)
        trainer.optimizer.load_state_dict(state['optimizer'])
        trainer.generator = torch.Generator(device=device)
        trainer.generator.set_state(state['rng'].cpu())
        trainer.step = state['step']
        return trainer
