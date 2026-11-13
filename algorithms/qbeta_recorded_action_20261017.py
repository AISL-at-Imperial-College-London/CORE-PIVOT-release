"""Critic-only native OneStepEasyBCQ training; no proposal is fitted or fabricated.

Q_beta's native update depends on recorded successor actions, not the proposal.
This adapter omits ONLY the frozen deployment proposal from the training bundle.
"""
import torch
from algorithms.offline import OfflineLearner
from algorithms.one_step_easy_bcq import OneStepEasyBCQ, OneStepEasyBCQConfig, SingleQ
from algorithms.networks import frozen_copy
from algorithms.truncated_bc import TruncatedPolicy
from training.rl import check_proposal_scalers


class RecordedActionQBeta(OfflineLearner):
    config_type = OneStepEasyBCQConfig

    def _build(self, proposal):
        if proposal is not None:
            raise ValueError('Critic-only training does not accept or fit a proposal.')
        self.generator = torch.Generator().manual_seed(self.generator.initial_seed())
        self.critic = SingleQ(self.mean, self.scale, self.config.hidden_size).to(self.device)
        self.target = frozen_copy(self.critic)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.modules = dict(critic=self.critic, target=self.target)
        self.optimizers = dict(critic=self.optimizer)

    # Exact SAME function object/equations/step order as the repo's EasyBCQ.
    update = OneStepEasyBCQ.update

    def predict(self, observations):
        raise RuntimeError('Q_beta is not a policy; attach an audited fitted truncated BC proposal first.')

    def attach_truncated_proposal(self, proposal):
        """Low-level conversion to native EasyBCQ; caller must certify BC provenance.

        Does not train, select or certify a proposal. Use a fitted same-data,
        same-seed, audited 500k BC distribution, not a random/selected substitute.
        All critic/target/Adam/counter/private RNG state is retained exactly.
        """
        if not isinstance(proposal, TruncatedPolicy):
            raise TypeError('Native EasyBCQ requires a truncated-normal proposal.')
        check_proposal_scalers(self, proposal)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        result = OneStepEasyBCQ.__new__(OneStepEasyBCQ)
        with torch.random.fork_rng(devices=devices):
            result._initialize(self.config, self.mean, self.scale, None, str(self.device), proposal)
        result.critic.load_state_dict(self.critic.state_dict())
        result.target.load_state_dict(self.target.state_dict())
        result.optimizer.load_state_dict(self.optimizer.state_dict())
        result.generator.set_state(self.generator.get_state())
        result.step = self.step
        return result
