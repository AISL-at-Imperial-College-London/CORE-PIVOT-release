"""Single-critic one-step Easy BCQ; Brandfonbrener et al., NeurIPS 2021.

https://arxiv.org/abs/2106.08909, Sections 4.1--4.3, Eq. (2), Appendix C.1.
Fit Q^behavior using recorded same-episode successor actions; deploy the best
of M draws from a frozen truncated-normal behavior policy. No twin-min backup,
entropy term, perturbation actor, or evaluation of the improved policy.
"""
from dataclasses import dataclass
import torch
from torch import nn
from algorithms.networks import Normalizer, frozen_copy, mlp, optimize, soft_update
from algorithms.offline import OfflineLearner
from algorithms.truncated_bc import TruncatedPolicy


class SingleQ(nn.Module): # (s,a) -> Q(s,a)
    def __init__(self, mean, scale, hidden):
        super().__init__()
        self.normalize = Normalizer(mean, scale)
        self.network = mlp(28, hidden, 1)

    def forward(self, observations, actions):
        return self.network(torch.cat((self.normalize(observations), actions), -1)) # actions already normalised


@dataclass(frozen=True)
class OneStepEasyBCQConfig:
    hidden_size: int = 1024
    batch_size: int = 512
    learning_rate: float = 1e-4
    discount: float = .99
    tau: float = .005
    candidates: int = 50


class OneStepEasyBCQ(OfflineLearner):
    config_type = OneStepEasyBCQConfig

    def __init__(self, observations, config=OneStepEasyBCQConfig(), seed=123, device='cpu', proposal=None):
        super().__init__(observations, config, seed, device, proposal)

    def _build(self, proposal):
        if not isinstance(proposal, TruncatedPolicy):
            raise TypeError('Requires an explicitly fitted truncated-normal behavior policy.')
        self.generator = torch.Generator().manual_seed(self.generator.initial_seed()) # random actions sampled during deployment
        self.proposal = frozen_copy(proposal).to(self.device) # Frozen truncated BC copied
        self.critic = SingleQ(self.mean, self.scale, self.config.hidden_size).to(self.device) # Initialise Q network
        self.target = frozen_copy(self.critic) # same weights initially
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.modules = dict(proposal=self.proposal, critic=self.critic, target=self.target)
        self.optimizers = dict(critic=self.optimizer)

    def _proposal_state(self):
        return dict(hidden_size=self.proposal.hidden_size, model=self.proposal.state_dict())

    @classmethod
    def _load_proposal(cls, state):
        proposal = TruncatedPolicy(state['model']['normalize.mean'], state['model']['normalize.scale'], state['hidden_size'])
        proposal.load_state_dict(state['model'])
        return proposal

    def update(self, batch):
        if torch.any(batch['timeouts'] & ~batch['terminals']):
            raise ValueError('Recorded-action FQE needs successor actions at timeout boundaries.')
        with torch.no_grad():
            target = batch['rewards'] + self.config.discount * batch['bootstrap_mask'] * self.target(batch['next_observations'], batch['next_actions'])
        values = self.critic(batch['observations'], batch['actions'])
        loss = (values - target).square().mean()
        optimize(self.optimizer, loss)
        soft_update(self.critic, self.target, self.config.tau)
        self.step += 1
        return dict(step=self.step, critic_loss=float(loss.detach()), q_mean=float(values.detach().mean()))

    @torch.no_grad()
    def _action(self, observation):
        repeated = observation[None].expand(self.config.candidates, -1)
        actions = self.proposal.distribution(repeated).sample(self.generator)
        return actions[self.critic(repeated, actions).argmax()].cpu().numpy()

    def predict(self, observations):
        tensor = self._tensor(observations)
        if tensor.ndim != 1:
            raise ValueError('Deployment expects one observation at a time.')
        return self._action(tensor)
