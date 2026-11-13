"""CORL learning kernels with the selected d3rlpy-era numerical settings.

New fits only: historical .d3 checkpoints are not interchangeable. Inputs are
the repo's unstandardized 25D observations and fixed-bound normalized actions.
Observation statistics are fitted by OfflineLearner, not by a second scaler.
See corl_vendor/provenance.json and docs/corl_baselines.md for known differences.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from algorithms.offline import OfflineLearner
from algorithms.corl_vendor import td3_bc, iql, cql


@dataclass(frozen=True)
class CORLTD3BCConfig:
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    discount: float = .99
    tau: float = .005
    alpha: float = 2.
    target_noise: float = .2
    noise_clip: float = .5
    actor_interval: int = 2


@dataclass(frozen=True)
class CORLIQLConfig:
    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    discount: float = .99
    tau: float = .005
    expectile: float = .98
    weight_temperature: float = 30.
    max_weight: float = 20.


@dataclass(frozen=True)
class CORLCQLConfig:
    batch_size: int = 256
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 3e-5
    multiplier_learning_rate: float = 1e-4
    discount: float = .99
    tau: float = .005
    entropy_temperature: float = .1
    conservative_weight: float = 1.
    lagrange: bool = True
    initial_multiplier: float = .5
    target_gap: float = 5.
    samples: int = 10


def adam(module, learning_rate):
    return torch.optim.Adam(module.parameters(), lr=learning_rate)


class ConstantActorSchedule:
    """Keep the actor LR unchanged; record calls for complete restoration."""
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def state_dict(self):
        return dict(steps=self.steps)

    def load_state_dict(self, state):
        self.steps = state['steps']


class _CORLLearner(OfflineLearner):
    """Adapt only preprocessing, the public API and full-state checkpointing."""

    def _initialize(self, config, mean, scale, seed, device, proposal):
        from dataclasses import asdict
        if not all(np.isfinite(value) for value in asdict(config).values()):
            raise ValueError('All CORL numerical settings must be finite.')
        if type(config.batch_size) is not int or config.batch_size < 1:
            raise ValueError('Batch size must be a positive integer.')
        super()._initialize(config, mean, scale, seed, device, proposal)
        # CORL kernels use global Torch CPU/CUDA RNGs, saved separately in the
        # checkpoint bundle. This unused API generator stays on CPU so its
        # serialized state is portable between CPU and CUDA learners.
        self.generator = torch.Generator(device='cpu').manual_seed(0 if seed is None else seed)

    def _standardize(self, observations):
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(self.scale, dtype=torch.float32, device=self.device)
        return (observations - mean) / scale

    def update(self, batch):
        # CORL receives normalized actions unchanged. No reward scaling.
        ordered = [self._standardize(batch['observations']), batch['actions'],
                   batch['rewards'], self._standardize(batch['next_observations']),
                   batch['terminals'].float()]
        metrics = self.trainer.train(ordered)
        self.step = self.trainer.total_it
        return dict(metrics, step=self.step)

    @torch.no_grad()
    def predict(self, observations):
        tensor = self._tensor(observations)
        single = tensor.ndim == 1
        if single:
            tensor = tensor[None]
        if tensor.ndim != 2 or tensor.shape[1] != 25 or not torch.isfinite(tensor).all():
            raise ValueError('Expected finite 25-feature observations.')
        tensor = self._standardize(tensor)
        if isinstance(self, CORLIQL):
            action = self.trainer.actor(tensor).mean
        elif isinstance(self, CORLCQL):
            action, _ = self.trainer.actor(tensor, deterministic=True)
        else:
            action = self.trainer.actor(tensor)
        result = action.cpu().numpy().astype(np.float32)
        return result[0] if single else result

    def state_dict(self):
        state = super().state_dict()
        scheduler = getattr(self.trainer, 'actor_lr_schedule', None)
        state['scheduler'] = None if scheduler is None else scheduler.state_dict()
        return state

    @classmethod
    def load(cls, path, device='cpu'):
        learner = super().load(path, device=device)
        state = torch.load(path, map_location='cpu', weights_only=True)
        learner.trainer.total_it = learner.step
        if state['scheduler'] is not None:
            learner.trainer.actor_lr_schedule.load_state_dict(state['scheduler'])
        return learner

    def _reject_proposal(self, proposal):
        if proposal is not None:
            raise ValueError('CORL actor baselines do not use an external proposal.')


class CORLTD3BC(_CORLLearner):
    config_type = CORLTD3BCConfig

    def __init__(self, observations, config=CORLTD3BCConfig(), seed=123, device='cpu', proposal=None):
        super().__init__(observations, config, seed, device, proposal)

    def _build(self, proposal):
        self._reject_proposal(proposal)
        cfg = self.config
        if type(cfg.actor_interval) is not int or cfg.actor_interval < 1 or cfg.alpha < 0 or min(cfg.target_noise, cfg.noise_clip) < 0:
            raise ValueError('Invalid TD3+BC settings.')
        actor = td3_bc.Actor(25, 3, 1.).to(self.device)
        first = td3_bc.Critic(25, 3).to(self.device)
        second = td3_bc.Critic(25, 3).to(self.device)
        self.trainer = td3_bc.TD3_BC(
            max_action=1., actor=actor, actor_optimizer=adam(actor, cfg.actor_learning_rate),
            critic_1=first, critic_1_optimizer=adam(first, cfg.critic_learning_rate),
            critic_2=second, critic_2_optimizer=adam(second, cfg.critic_learning_rate),
            discount=cfg.discount, tau=cfg.tau, policy_noise=cfg.target_noise,
            noise_clip=cfg.noise_clip, policy_freq=cfg.actor_interval,
            alpha=cfg.alpha, device=str(self.device))
        self.modules = {name: getattr(self.trainer, name) for name in (
            'actor', 'actor_target', 'critic_1', 'critic_2', 'critic_1_target', 'critic_2_target')}
        self.optimizers = {name: getattr(self.trainer, name) for name in (
            'actor_optimizer', 'critic_1_optimizer', 'critic_2_optimizer')}


class CORLIQL(_CORLLearner):
    config_type = CORLIQLConfig

    def __init__(self, observations, config=CORLIQLConfig(), seed=123, device='cpu', proposal=None):
        super().__init__(observations, config, seed, device, proposal)

    def _build(self, proposal):
        self._reject_proposal(proposal)
        cfg = self.config
        if not 0 < cfg.expectile < 1 or min(cfg.weight_temperature, cfg.max_weight) <= 0:
            raise ValueError('Invalid expectile or advantage weights.')
        q = iql.TwinQ(25, 3).to(self.device)
        value = iql.ValueFunction(25).to(self.device)
        actor = iql.GaussianPolicy(25, 3, 1.).to(self.device)
        self.trainer = iql.ImplicitQLearning(
            max_action=1., actor=actor, actor_optimizer=adam(actor, cfg.actor_learning_rate),
            q_network=q, q_optimizer=adam(q, cfg.critic_learning_rate),
            v_network=value, v_optimizer=adam(value, cfg.value_learning_rate),
            iql_tau=cfg.expectile, beta=cfg.weight_temperature,
            discount=cfg.discount, tau=cfg.tau, device=str(self.device))
        # Deliberate adaptations: selected d3rlpy run used cap20 and no cosine decay.
        self.trainer.max_weight = cfg.max_weight
        self.trainer.actor_lr_schedule = ConstantActorSchedule()
        self.modules = {name: getattr(self.trainer, name) for name in ('actor', 'qf', 'q_target', 'vf')}
        self.optimizers = {name: getattr(self.trainer, name) for name in (
            'actor_optimizer', 'q_optimizer', 'v_optimizer')}


class CORLCQL(_CORLLearner):
    config_type = CORLCQLConfig

    def __init__(self, observations, config=CORLCQLConfig(), seed=123, device='cpu', proposal=None):
        super().__init__(observations, config, seed, device, proposal)

    def _build(self, proposal):
        self._reject_proposal(proposal)
        cfg = self.config
        if (min(cfg.entropy_temperature, cfg.initial_multiplier, cfg.conservative_weight) <= 0
                or type(cfg.samples) is not int or cfg.samples < 1):
            raise ValueError('CQL temperatures and multipliers must be positive.')
        actor = cql.TanhGaussianPolicy(25, 3, 1.).to(self.device)
        first = cql.FullyConnectedQFunction(25, 3, orthogonal_init=True, n_hidden_layers=2).to(self.device)
        second = cql.FullyConnectedQFunction(25, 3, orthogonal_init=True, n_hidden_layers=2).to(self.device)
        self.trainer = cql.ContinuousCQL(
            first, adam(first, cfg.critic_learning_rate), second, adam(second, cfg.critic_learning_rate),
            actor, adam(actor, cfg.actor_learning_rate), target_entropy=-3.,
            discount=cfg.discount, soft_target_update_rate=cfg.tau,
            policy_lr=cfg.actor_learning_rate, qf_lr=cfg.critic_learning_rate,
            use_automatic_entropy_tuning=False, alpha_multiplier=cfg.entropy_temperature,
            backup_entropy=False, bc_steps=0, cql_n_actions=cfg.samples,
            cql_importance_sample=True, cql_lagrange=cfg.lagrange,
            cql_target_action_gap=cfg.target_gap, cql_temp=1.,
            cql_alpha=cfg.conservative_weight, cql_max_target_backup=False,
            device=str(self.device))
        multiplier = self.trainer.log_alpha_prime.to(self.device)
        with torch.no_grad():
            multiplier.constant.fill_(math.log(cfg.initial_multiplier))
        for group in self.trainer.alpha_prime_optimizer.param_groups:
            group['lr'] = cfg.multiplier_learning_rate
        self.modules = {name: getattr(self.trainer, name) for name in (
            'actor', 'critic_1', 'critic_2', 'target_critic_1', 'target_critic_2', 'log_alpha_prime')}
        self.optimizers = {name: getattr(self.trainer, name) for name in (
            'actor_optimizer', 'critic_1_optimizer', 'critic_2_optimizer', 'alpha_prime_optimizer')}


CORL_ALGORITHMS = {
    'td3-bc': (CORLTD3BC, CORLTD3BCConfig),
    'iql': (CORLIQL, CORLIQLConfig),
    'cql': (CORLCQL, CORLCQLConfig),
}
