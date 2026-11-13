"""Checkpoint plumbing, not a training framework; equations live in each learner."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from algorithms.networks import training_statistics


class OfflineLearner:
    config_type = None

    def __init__(self, observations, config, seed=123, device='cpu', proposal=None):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != 25 or not len(observations):
            raise ValueError('Expected nonempty N×25 training observations.')
        mean, scale = training_statistics(observations)
        self._initialize(config, mean, scale, seed, device, proposal)

    def _initialize(self, config, mean, scale, seed, device, proposal):
        for name, value in asdict(config).items():
            if 'learning_rate' in name or name in ('hidden_size', 'batch_size', 'samples', 'candidates', 'search_iterations'):
                if value <= 0:
                    raise ValueError(f'{name} must be positive.')
            if name in ('discount', 'tau') and not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0, 1].')
            if name.endswith('_weight') and value < 0:
                raise ValueError(f'{name} must be nonnegative.')
        self.config, self.device = config, torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Choose cpu or cuda.')
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.step = 0
        if seed is not None:
            torch.manual_seed(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(0 if seed is None else seed)
        self.modules, self.optimizers = {}, {}
        self._build(proposal)

    def _tensor(self, observations):
        return torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)

    def state_dict(self):
        return dict(version=1, algorithm=type(self).__name__, config=asdict(self.config),
                    mean=self.mean.tolist(), scale=self.scale.tolist(), step=self.step,
                    modules={name: module.state_dict() for name, module in self.modules.items()},
                    optimizers={name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
                    rng=self.generator.get_state(), proposal=self._proposal_state())

    def _proposal_state(self):
        return None

    @classmethod
    def _load_proposal(cls, state):
        if state is not None:
            raise ValueError('Unexpected proposal in checkpoint.')
        return None

    def save(self, path):
        # Exclusive creation prevents an experiment from overwriting a checkpoint.
        with Path(path).open('xb') as stream:
            torch.save(self.state_dict(), stream)

    @classmethod
    def load(cls, path, device='cpu'):
        # Deserialize on CPU first: Adam's non-capturable step counters belong
        # on CPU, even when parameters and moment tensors are restored to CUDA.
        # optimizer.load_state_dict performs the parameter-aware device transfer.
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['version'] != 1 or state['algorithm'] != cls.__name__:
            raise ValueError('Checkpoint algorithm/version mismatch.')
        learner = cls.__new__(cls)
        # Restoring a model must not change the caller's global Torch RNG.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            learner._initialize(cls.config_type(**state['config']), state['mean'], state['scale'],
                                None, device, cls._load_proposal(state['proposal']))
        if learner.modules.keys() != state['modules'].keys():
            raise ValueError('Checkpoint network set mismatch.')
        for name, module in learner.modules.items():
            module.load_state_dict(state['modules'][name])
        for name, optimizer in learner.optimizers.items():
            optimizer.load_state_dict(state['optimizers'][name])
        learner.generator.set_state(state['rng'].cpu())
        learner.step = state['step']
        return learner
