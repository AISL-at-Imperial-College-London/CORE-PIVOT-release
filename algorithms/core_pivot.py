"""CORE-PIVOT: frozen bounded BC proposal, SARSA targets, fixed CQL penalty.

Only critics learn. Deployment searches the action box, without HOFLON's ball.
New fixed-bound implementation; not interchangeable with historical scalers.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch

from algorithms.action_search import refine_action
from algorithms.conservative import conservative_penalty
from algorithms.networks import TwinQ, frozen_copy, optimize, soft_update
from algorithms.offline import OfflineLearner
from algorithms.squashed_bc import SquashedBCConfig, SquashedPolicy


@dataclass(frozen=True)
class COREPIVOTConfig:
    batch_size: int = 256
    hidden_size: int = 256
    learning_rate: float = 3e-5
    discount: float = .99
    tau: float = .005
    conservative_weight: float = .05
    conservative_temperature: float = 1.
    samples: int = 10
    bc_candidates: int = 10
    uniform_candidates: int = 40
    search_iterations: int = 40


class COREPIVOT(OfflineLearner):
    config_type = COREPIVOTConfig

    def __init__(self, observations, config=COREPIVOTConfig(), seed=123,
                 device='cpu', proposal=None):
        super().__init__(observations, config, seed, device, proposal)

    def _build(self, proposal):
        if not isinstance(proposal, SquashedPolicy):
            raise TypeError('CORE-PIVOT requires pretrained SquashedBC, with a bounded density.')
        if min(self.config.samples, self.config.bc_candidates, self.config.uniform_candidates) < 1:
            raise ValueError('Candidate/sample counts must be positive.')
        if self.config.conservative_temperature <= 0:
            raise ValueError('Conservative temperature must be positive.')
        self.proposal = frozen_copy(proposal).to(self.device)
        self.critic = TwinQ(self.mean, self.scale, self.config.hidden_size).to(self.device)
        self.target = frozen_copy(self.critic)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.modules = dict(critic=self.critic, target=self.target, proposal=self.proposal)
        self.optimizers = dict(critic=self.optimizer)

    def _proposal_state(self):
        return dict(config=asdict(self.proposal.config), model=self.proposal.state_dict())

    @classmethod
    def _load_proposal(cls, state):
        proposal = SquashedPolicy(SquashedBCConfig(**state['config']),
                                  state['model']['normalize.mean'], state['model']['normalize.scale'])
        proposal.load_state_dict(state['model'])
        return proposal

    def update(self, batch):
        if torch.any(batch['timeouts'] & ~batch['terminals']):
            raise ValueError('SARSA needs recorded successor actions at timeout boundaries.')
        observations, actions = batch['observations'], batch['actions']
        with torch.no_grad():
            next_q = self.target.minimum(batch['next_observations'], batch['next_actions'])
            target = batch['rewards'] + self.config.discount * batch['bootstrap_mask'] * next_q
        first, second = self.critic(observations, actions)
        bellman_loss = (first - target).square().mean() + (second - target).square().mean()
        conservative_loss = conservative_penalty(self.critic, observations, batch['next_observations'],
            actions, self.proposal, self.generator, self.config.samples, self.config.conservative_temperature)
        loss = bellman_loss + self.config.conservative_weight * conservative_loss
        optimize(self.optimizer, loss)
        soft_update(self.critic, self.target, self.config.tau)
        self.step += 1
        return dict(step=self.step, critic_loss=float(loss.detach()),
                    bellman_loss=float(bellman_loss.detach()), conservative_loss=float(conservative_loss.detach()))

    @torch.no_grad()
    def _anchor(self, observation):
        repeated = observation[None, :].expand(self.config.bc_candidates, -1)
        bc_actions, _ = self.proposal.sample(repeated, self.generator)
        uniform = 2 * torch.rand((self.config.uniform_candidates, 3),
                                device=self.device, generator=self.generator) - 1
        # Observation ax_prev..az_prev retains physical units; normalize ONCE.
        previous = (observation[12:15] / 5).clamp(-1, 1)[None, :]
        actions = torch.cat((bc_actions, uniform, previous, torch.zeros_like(previous)))
        values = self.critic.minimum(observation[None, :].expand(len(actions), -1), actions)
        return actions[values.argmax()].cpu().numpy()

    def predict(self, observations):
        tensor = self._tensor(observations)
        def action(row):
            return refine_action(self.critic, row, self._anchor(row),
                                 max_iterations=self.config.search_iterations)
        return action(tensor) if tensor.ndim == 1 else np.stack([action(row) for row in tensor])
