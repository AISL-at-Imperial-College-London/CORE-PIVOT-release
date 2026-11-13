"""Load native release checkpoints and original paper checkpoint bundles."""
import copy
import numpy as np
import torch

from algorithms.truncated_bc import TruncatedPolicy, TruncatedBC
from algorithms.qbeta_recorded_action_20261017 import RecordedActionQBeta
from algorithms.corl_baselines import CORLTD3BC, CORLIQL, CORLCQL
from core_pivot_truncated_20261019 import TruncatedCOREPIVOT
from evaluation.search import refine_action, install_core_fallback


def native_model(cls, state):
    """Evaluation-only equivalent of OfflineLearner.load, including RNG state."""
    if state.get('version') != 1 or state.get('algorithm') != cls.__name__:
        raise ValueError(f'Expected {cls.__name__} checkpoint')
    learner = cls.__new__(cls)
    with torch.random.fork_rng(devices=[]):
        learner._initialize(cls.config_type(**state['config']), state['mean'], state['scale'],
                            None, 'cpu', cls._load_proposal(state['proposal']))
    if learner.modules.keys() != state['modules'].keys():
        raise ValueError('Checkpoint network set mismatch')
    for name, module in learner.modules.items():
        module.load_state_dict(state['modules'][name])
        module.eval().requires_grad_(False)
    learner.generator.set_state(state['rng'].cpu())
    learner.step = state['step']
    return learner


class MeanBC:
    def __init__(self, policy, step):
        self.policy, self.step = policy, step
        self.policy.eval().requires_grad_(False)

    @torch.no_grad()
    def predict(self, observation):
        return self.policy.distribution(torch.as_tensor(observation, dtype=torch.float32)).mean.numpy()


def load_bc(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('algorithm') == 'TruncatedBC':
        learner = native_model(TruncatedBC, state)
        return MeanBC(learner.policy, learner.step)
    record = state.get('trainer', state)
    weights = record.get('model')
    if weights is None:
        raise ValueError('Unrecognised BC checkpoint')
    width = weights['network.0.weight'].shape[0]
    policy = TruncatedPolicy(weights['normalize.mean'], weights['normalize.scale'], width)
    policy.load_state_dict(weights)
    return MeanBC(policy, record.get('step'))


class RankedRefinement:
    """Paper OneStepHOFLON prediction with optional box-only constraint."""
    def __init__(self, learner, candidates=100, radius=.1, iterations=40):
        self.proposal = copy.deepcopy(learner.proposal)
        self.critic = copy.deepcopy(learner.critic)
        self.generator = torch.Generator()
        self.generator.set_state(learner.generator.get_state())
        self.candidates, self.radius, self.iterations = candidates, radius, iterations
        for module in (self.proposal, self.critic):
            module.eval().requires_grad_(False)

    def predict(self, observation):
        observation = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        if observation.shape != (25,) or not torch.isfinite(observation).all():
            raise ValueError('Expected one finite 25-feature observation')
        with torch.no_grad():
            repeated = observation[None].expand(self.candidates, -1)
            actions = self.proposal.distribution(repeated).sample(self.generator)
            anchor = actions[self.critic(repeated, actions).argmax()].cpu().numpy()
        return refine_action(self.critic, observation, anchor, self.radius, self.iterations)


class ExtractedMean:
    def __init__(self, fit):
        self.fit, self.step = fit, fit.step
        fit.actor.eval().requires_grad_(False)

    @torch.no_grad()
    def predict(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)[None]
        return self.fit.actor(self.fit.norm(observation)).mean[0].numpy()


def load_model(method, checkpoint, bc_checkpoint=None, expected_step=None, source_critic=None):
    if method == 'bc':
        model = load_bc(checkpoint)
    elif method in ('awr', 'ddpg-bc'):
        if source_critic is None:
            raise ValueError('Frozen extraction requires --source-critic exported500k IQL bundle')
        from extraction.frozen_iql import FrozenIQLExtractor, ExtractionConfig
        source = torch.load(source_critic, map_location='cpu', weights_only=True)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = ExtractionConfig(**state['config'])
        if config.kind != ('ddpg_bc' if method == 'ddpg-bc' else 'awr'):
            raise ValueError('Extraction objective does not match --method')
        fit = FrozenIQLExtractor(source, config)
        fit.restore(state)
        fit.assert_frozen()
        model = ExtractedMean(fit)
    else:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        # The SARSA-CQL experiment wraps the unchanged CQL network state.
        if method == 'sarsa-cql' and 'learner' in state:
            if state.get('variant') != 'cql_sarsa':
                raise ValueError('Not a recorded-action CQL checkpoint')
            state = state['learner']
        classes = {'easybcq': RecordedActionQBeta, 'hoflon': RecordedActionQBeta,
                   'td3-bc': CORLTD3BC, 'iql': CORLIQL, 'cql': CORLCQL,
                   'sarsa-cql': CORLCQL, 'core-pivot': TruncatedCOREPIVOT}
        if state.get('algorithm') == 'BackupCQL' and method in ('cql', 'sarsa-cql'):
            from ablations.cql_backup import BackupCQL
            required = 'sarsa' if method == 'sarsa-cql' else 'actor'
            if state.get('backup') != required:
                raise ValueError('Backup-CQL arm does not match the requested method')
            model = BackupCQL.load(checkpoint, device='cpu')
        else:
            model = native_model(classes[method], state)
        if method in ('easybcq', 'hoflon'):
            if bc_checkpoint is None:
                raise ValueError('A frozen BC checkpoint is required')
            bc = load_bc(bc_checkpoint)
            if bc.step != 500000:
                raise ValueError('Main-paper proposals require BC frozen at 500k')
            model = model.attach_truncated_proposal(bc.policy)
    if expected_step is not None and model.step != expected_step:
        raise ValueError(f'Checkpoint step {model.step} != expected {expected_step}')
    for module in getattr(model, 'modules', {}).values():
        module.eval().requires_grad_(False)
    return model


def episode_policy(model, method, wind_seed, candidates=100, radius=.1, iterations=40):
    if method in ('easybcq', 'hoflon'):
        return RankedRefinement(model, candidates, 0 if method == 'easybcq' else radius, iterations)
    policy = copy.deepcopy(model)
    if method == 'core-pivot':
        install_core_fallback()
        policy.generator.manual_seed(9001 + wind_seed)
    return policy
