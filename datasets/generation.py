"""Regenerate the paper's training replay without historical workspace paths.

The release retains the original controller catalogue and split, rather than
redrawing either. Numerical integration is float64; network arrays are float32.
Rows are ordered by wind, controller, then transition. No validation or test
trajectories enter generation, augmentation statistics, or normalization.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from environments.reference import reference
from environments.wind import wind_trace
from environments.drone_env import OBSERVATION_NAMES

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
FIELDS = ('observations', 'next_observations', 'actions', 'next_actions',
          'rewards', 'terminals', 'timeouts')
DT, STEPS = .02, 750
PROTOCOL = 'state_consistent_previous_action_v1'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def split_plan():
    return json.loads((CONFIGS / 'dataset_split.json').read_text())['splits']


def controller_catalog():
    # pandas parsing is retained because this is how the original generator
    # loaded these serialized gains. Nominal gains were (2, 1), not PD-11.
    return pd.read_csv(CONFIGS / 'pd_generation_catalog.csv').sort_values(
        'controller_id')[['controller_id', 'kp', 'kd', 'a_max']]


def simulate_wind(seed, controller_ids=None, *, stronger=False):
    """Return clean float32 replay for complete 750-step controller episodes."""
    catalog = controller_catalog()
    if controller_ids is not None:
        catalog = catalog[catalog.controller_id.isin(controller_ids)]
        if set(catalog.controller_id) != set(controller_ids):
            raise ValueError('Unknown controller ID.')
    count = len(catalog)
    time = np.arange(STEPS + 1) * DT
    refs = np.asarray([reference(float(t)) for t in time])
    preview10 = np.asarray([reference(min(float(t) + 10 * DT, 15.))[1] for t in time])
    preview20 = np.asarray([reference(min(float(t) + 20 * DT, 15.))[1] for t in time])
    wind = wind_trace(np.random.default_rng(seed), stronger)
    states = np.zeros((count, STEPS + 1, 6), dtype=np.float64)
    actions = np.empty((count, STEPS, 3), dtype=np.float64)
    kp = catalog.kp.to_numpy()[:, None]
    kd = catalog.kd.to_numpy()[:, None]
    limit = catalog.a_max.to_numpy()[:, None]
    for k in range(STEPS):
        pos, vel = states[:, k, :3], states[:, k, 3:]
        ep, ev = refs[k, 0] - pos, refs[k, 1] - vel
        action = np.clip(kp * ep + kd * ev, -limit, limit)
        states[:, k + 1, :3] = pos + vel * DT
        states[:, k + 1, 3:] = vel + (action + wind[k]) * DT
        actions[:, k] = action
    errors = refs[None, :, 0] - states[:, :, :3]
    velocity_errors = refs[None, :, 1] - states[:, :, 3:]
    residual = ((states[:, 1:, 3:] - states[:, :-1, 3:]) /
                np.diff(time)[None, :, None] - actions)
    previous = np.concatenate((np.zeros((count, 1, 3)), actions), axis=1)
    estimate = np.concatenate((np.zeros((count, 1, 3)), residual), axis=1)
    broadcast = lambda array: np.broadcast_to(array, (count,) + array.shape)
    observations = np.concatenate((errors, velocity_errors, broadcast(preview10),
        broadcast(preview20), previous, broadcast((time / 15.)[:, None]),
        estimate, broadcast(refs[:, 1]), broadcast(refs[:, 2])), axis=2).astype(np.float32)
    rewards = -(np.sum(errors[:, 1:]**2, axis=2) +
                .2 * np.sum(velocity_errors[:, 1:]**2, axis=2))
    normalized = (actions / 5).astype(np.float32)
    next_actions = np.zeros_like(normalized)
    next_actions[:, :-1] = normalized[:, 1:]
    terminals = np.zeros((count, STEPS, 1), dtype=bool)
    terminals[:, -1] = True
    return dict(observations=observations[:, :-1].reshape(-1, 25),
        next_observations=observations[:, 1:].reshape(-1, 25),
        actions=normalized.reshape(-1, 3), next_actions=next_actions.reshape(-1, 3),
        rewards=rewards.astype(np.float32).reshape(-1, 1),
        terminals=terminals.reshape(-1, 1), timeouts=np.zeros((count * STEPS, 1), bool))


def augment_in_place(arrays, std=.01):
    """One fixed cached noise draw, never fresh noise during learner updates.

    Current and successor observations share the same noisy previous-action
    channels for interior states. Terminal successors retain the historical
    independently indexed draw. Rewards and recorded actions are unchanged.
    """
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    count = len(arrays['actions'])
    for start in range(0, count, 100_000):
        act = np.asarray(arrays['actions'][start:start + 100_000], dtype=np.float64) * 5
        lower, upper = np.minimum(lower, act.min(0)), np.maximum(upper, act.max(0))
    physical_std = std * (upper - lower) / 2
    clipping = {}
    for field in ('observations', 'next_observations'):
        obs = arrays[field]
        prev_low, prev_high = np.full(3, np.inf), np.full(3, -np.inf)
        for start in range(0, count, 100_000):
            previous = obs[start:start + 100_000, 12:15]
            prev_low = np.minimum(prev_low, previous.min(0))
            prev_high = np.maximum(prev_high, previous.max(0))
        lo, hi = np.minimum(lower, prev_low), np.maximum(upper, prev_high)
        clipping[field] = dict(min=lo.tolist(), max=hi.tolist())
        rng = np.random.default_rng(104852)
        for start in range(0, count, 100_000):
            stop = min(count, start + 100_000)
            noise = rng.normal(0, physical_std, size=(stop - start, 3))
            obs[start:stop, 12:15] = np.clip(
                obs[start:stop, 12:15].astype(np.float64) + noise, lo, hi)
    boundary = (arrays['terminals'] | arrays['timeouts']).reshape(-1)
    for start in range(0, count, 100_000):
        stop = min(count, start + 100_000)
        interior = start + np.flatnonzero(~boundary[start:stop])
        arrays['next_observations'][interior, 12:15] = arrays['observations'][interior + 1, 12:15]
    return dict(protocol=PROTOCOL, std_data_scaled=std, seed=104852,
        std_physical=physical_std.tolist(), physical_action_min=lower.tolist(),
        physical_action_max=upper.tolist(), clipping=clipping,
        targets_unchanged=True, rollout_noise=False, fixed_cached_view=True)


def training_statistics(observations):
    """Exact original float64 chunked population statistics (100k rows/chunk)."""
    count = 0
    mean = np.zeros(observations.shape[1], np.float64)
    squared_deviation = np.zeros_like(mean)
    for start in range(0, len(observations), 100_000):
        chunk = np.asarray(observations[start:start + 100_000], dtype=np.float64)
        size = len(chunk)
        delta = chunk.mean(0) - mean
        total = count + size
        squared_deviation += chunk.var(0) * size + np.square(delta) * count * size / total
        mean += delta * size / total
        count = total
    return mean, np.maximum(np.sqrt(squared_deviation / count), 1e-6)


def generate(directory, *, smoke=False, clean=False):
    """Write a fresh replay directory; incomplete output is never reusable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    seeds = split_plan()['train']['ordinary']
    controllers = list(range(200))
    if smoke:
        seeds, controllers = seeds[:2], [0, 11, 199]
    rows = len(seeds) * len(controllers) * STEPS
    dims = dict(observations=25, next_observations=25, actions=3, next_actions=3,
                rewards=1, terminals=1, timeouts=1)
    arrays = {name: np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+',
        dtype=bool if name in ('terminals', 'timeouts') else np.float32,
        shape=(rows, dims[name])) for name in FIELDS}
    offset = 0
    for index, seed in enumerate(seeds):
        data = simulate_wind(seed, controllers)
        length = len(data['actions'])
        for name in FIELDS:
            arrays[name][offset:offset + length] = data[name]
        offset += length
        print(f'Generated wind {index + 1}/{len(seeds)} ({seed})', flush=True)
    augmentation = dict(protocol='none', std_data_scaled=0.) if clean else augment_in_place(arrays)
    mean, scale = training_statistics(arrays['observations'])
    norm = dict(mean=mean.tolist(), scale=scale.tolist(), method='chunked_population_std',
                train_rows=rows, inputs=25)
    (directory / 'normalization.json').write_text(json.dumps(norm, indent=2) + '\n')
    files = {}
    for name, array in arrays.items():
        array.flush()
        files[f'{name}.npy'] = file_hash(directory / f'{name}.npy')
    files['normalization.json'] = file_hash(directory / 'normalization.json')
    historical_match = None
    if not smoke and not clean:
        expected = json.loads((CONFIGS / 'historical_replay_sha256.json').read_text())
        historical_match = {name: files[name] == digest for name, digest in expected['files'].items()}
        if not all(historical_match.values()):
            raise RuntimeError(f'Historical replay mismatch; incomplete output preserved: {historical_match}')
    metadata = dict(schema=1, complete=True, rows=rows, transitions=rows,
        episodes=rows // STEPS, controllers=len(controllers), training_winds=len(seeds),
        inputs=25, observation_names=list(OBSERVATION_NAMES), split='train',
        wind_seeds=seeds, controller_ids=controllers, order='wind/controller/time',
        action_coordinates='normalized', physical_action_bounds=[-5, 5],
        reward_timing='next_state', horizon_terminal=True, previous_action_augmentation=augmentation,
        normalization_training_only=True, test_accessed=False, smoke=smoke,
        source_files={p.name: file_hash(p) for p in (CONFIGS / 'dataset_split.json',
                                                   CONFIGS / 'pd_generation_catalog.csv')},
        historical_replay_sha256_match=historical_match, files=files)
    (directory / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
