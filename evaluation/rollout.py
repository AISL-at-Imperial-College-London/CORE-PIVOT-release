"""One complete 750-transition episode and optional recorded trajectory."""
import hashlib
import numpy as np
from environments.drone_env import DroneEnv, NormalizedActionWrapper
from environments.reference import reference
from evaluation.protocol import BEST_PD, DisturbedEnv, disturbance


def rollout(policy, condition, wind_seed, record=False):
    env = NormalizedActionWrapper(
        DroneEnv(stronger_wind=condition.get('stronger', False))
        if condition['family'] == 'split' else DisturbedEnv(disturbance(condition, wind_seed)))
    obs, info = env.reset(seed=wind_seed)
    data = dict(state=[info['state'].copy()], observation=[obs.copy()],
                reference=[info['reference_position'].copy()], action=[], wind=[], cost=[])
    squared_p, squared_v, max_action, total = [], [], 0., 0.
    for tick in range(750):
        if policy is None:
            pr, vr, _ = reference(tick * .02)
            physical = np.clip(BEST_PD['kp'] * (pr - info['state'][:3])
                               + BEST_PD['kd'] * (vr - info['state'][3:]), -5., 5.)
            action = (physical / 5.).astype(np.float32)
        else:
            action = np.asarray(policy.predict(obs), dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Invalid normalised action')
        obs, reward, done, truncated, info = env.step(action)
        if truncated or done != (tick == 749) or info['action_clipped']:
            raise ValueError('Unexpected episode termination or action clipping')
        total += reward
        squared_p.append(np.sum((info['reference_position'] - info['state'][:3]) ** 2))
        squared_v.append(np.sum((info['reference_velocity'] - info['state'][3:]) ** 2))
        max_action = max(max_action, float(np.abs(action).max()))
        data['wind'].append(info['disturbance'].copy())
        if record:
            data['state'].append(info['state'].copy())
            data['observation'].append(obs.copy())
            data['reference'].append(info['reference_position'].copy())
            data['action'].append(action.copy())
            data['cost'].append(-reward)
    env.close()
    if not np.isfinite([total, *squared_p, *squared_v]).all():
        raise ValueError('Non-finite rollout')
    wind = np.ascontiguousarray(data['wind'], dtype=np.float64)
    result = dict(wind_seed=wind_seed, condition=condition['id'],
                  episode_cost=-total, position_rmse=float(np.sqrt(np.mean(squared_p))),
                  velocity_rmse=float(np.sqrt(np.mean(squared_v))),
                  max_normalized_action=max_action,
                  wind_sha256=hashlib.sha256(wind.tobytes()).hexdigest())
    return result, {k: np.asarray(v) for k, v in data.items()} if record else None
