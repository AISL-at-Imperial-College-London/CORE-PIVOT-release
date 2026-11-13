"""Construction and checkpoint conversion, without changing learning equations."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from algorithms.corl_baselines import (
    CORLCQL, CORLCQLConfig, CORLIQL, CORLIQLConfig, CORLTD3BC, CORLTD3BCConfig,
)
from algorithms.core_pivot import COREPIVOTConfig
from algorithms.one_step_easy_bcq import OneStepEasyBCQConfig
from algorithms.qbeta_recorded_action_20261017 import RecordedActionQBeta
from algorithms.truncated_bc import TruncatedBC, TruncatedBCConfig, TruncatedPolicy
from core_pivot_truncated_20261019 import DensityProposal, TruncatedCOREPIVOT

LEARNERS = {
    'bc': (TruncatedBC, TruncatedBCConfig),
    'qbeta': (RecordedActionQBeta, OneStepEasyBCQConfig),
    'core-pivot': (TruncatedCOREPIVOT, COREPIVOTConfig),
    'td3-bc': (CORLTD3BC, CORLTD3BCConfig),
    'iql': (CORLIQL, CORLIQLConfig),
    'cql': (CORLCQL, CORLCQLConfig),
}


def configure(device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback.')


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def global_rng():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=(n[0], n[1].tolist(), *n[2:]),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_global_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG restoration requires CUDA.')
        torch.cuda.set_rng_state_all(state['cuda'])


def proposal_from_checkpoint(path, *, density=False, require_500k=True):
    """Read native BC or historical truncated-BC bundles, without refitting."""
    s = torch.load(path, map_location='cpu', weights_only=True)
    if 'trainer' in s:
        s = s['trainer']
    if 'modules' in s:
        model = s['modules']['policy']
        hidden = s['config']['hidden_size']
    else:
        model = s['model']
        hidden = s['config']['hidden_size']
    if require_500k and s['step'] != 500000:
        raise ValueError('Paper proposals use BC frozen at 500000 updates.')
    cls = DensityProposal if density else TruncatedPolicy
    result = cls(model['normalize.mean'], model['normalize.scale'], hidden)
    result.load_state_dict(model)
    result.eval().requires_grad_(False)
    return result


def fresh(algorithm, config, mean, scale, seed, device='cpu', proposal=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    cls, cfg = LEARNERS[algorithm]
    learner = cls.__new__(cls)
    learner._initialize(cfg(**config), mean, scale, seed, device, proposal)
    # Historical Q_beta used a distinct replay stream. Other families use +10000.
    sampler = torch.Generator().manual_seed(seed + (20000 if algorithm == 'qbeta' else 10000))
    return learner, sampler


def restore_native(learner, state):
    """Restore all learner state without consuming the caller's random stream."""
    if state['config'] != vars(learner.config):
        raise ValueError('Checkpoint and requested configuration differ.')
    for name, module in learner.modules.items():
        module.load_state_dict(state['modules'][name])
    for name, optimizer in learner.optimizers.items():
        optimizer.load_state_dict(state['optimizers'][name])
    learner.generator.set_state(state['rng'])
    learner.step = state['step']
    if hasattr(learner, 'trainer'):
        learner.trainer.total_it = learner.step
        if state.get('scheduler') is not None:
            learner.trainer.actor_lr_schedule.load_state_dict(state['scheduler'])


def load_initial(learner, sampler, path):
    """Optional archival initial state. No inference from a training seed alone."""
    path = Path(path)
    s = torch.load(path, map_location='cpu', weights_only=True)
    aux = s.get('auxiliary', {})
    state = s.get('learner', s)
    if 'model' in state and isinstance(learner, TruncatedBC):
        if state['step'] != 0:
            raise ValueError('An initial state must have zero updates.')
        learner.policy.load_state_dict(state['model'])
        learner.optimizer.load_state_dict(state['optimizer'])
        learner.generator.set_state(state['rng'])
        aux = dict(global_rng=state.get('global_rng'))
        if state.get('initial_batches'):
            aux['replay_rng'] = state['initial_batches']['rng']
    else:
        if state['step'] != 0:
            raise ValueError('An initial state must have zero updates; use --resume instead.')
        restore_native(learner, state)
    sidecar = path.with_suffix('.state.pt')
    if sidecar.exists():
        aux = torch.load(sidecar, map_location='cpu', weights_only=True)
    if aux.get('replay_rng') is not None:
        sampler.set_state(aux['replay_rng'])
    if aux.get('global_rng') is not None:
        restore_global_rng(aux['global_rng'])
    return aux
