"""Complete, seeded drone rollouts for normalized-action baseline policies."""
import numpy as np

from environments.drone_env import DroneEnv, NormalizedActionWrapper


def evaluate(policy, winds):
    rows = []
    for wind in winds:
        env = NormalizedActionWrapper(DroneEnv(stronger_wind=wind['stress']))
        observation, _ = env.reset(seed=wind['seed'])
        total, position_error, velocity_error, max_action = 0., [], [], 0.
        for step in range(750):
            action = np.zeros(3, dtype=np.float32) if policy is None else policy.predict(observation)
            if not env.action_space.contains(action):
                raise ValueError('Policy action violates normalized bounds.')
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            position_error.append(np.sum((info['reference_position'] - info['state'][:3])**2))
            velocity_error.append(np.sum((info['reference_velocity'] - info['state'][3:])**2))
            max_action = max(max_action, float(np.abs(action).max()))
            if terminated != (step == 749) or truncated or info['action_clipped']:
                raise ValueError('Unexpected terminal, truncation, or action clipping.')
        env.close()
        if not np.isfinite([total, *position_error, *velocity_error]).all():
            raise ValueError('Non-finite validation rollout.')
        rows.append(dict(wind_seed=wind['seed'], stronger=wind['stress'],
                         episode_return=total, episode_cost=-total,
                         position_rmse=float(np.sqrt(np.mean(position_error))),
                         velocity_rmse=float(np.sqrt(np.mean(velocity_error))),
                         max_normalized_action=max_action))
    return rows
