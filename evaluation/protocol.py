"""Original split conditions and the 17 reported robustness conditions."""
import numpy as np
from environments.wind import wind_trace
from environments.drone_env import DroneEnv

BEST_PD = dict(kp=2.9413960487898065, kd=1.3931211213221977, a_max=5.)


def robustness_conditions():
    return [dict(id=f'{family}_{a:g}', family=family, amplitude=a,
                 onset=4., reversal=9.)
            for family, values in [('severity', [1.5, 2., 2.5, 3., 4.]),
                                   ('step', [1.5, 2., 2.5, 3., 3.5, 4.]),
                                   ('reversing', [1.5, 2., 2.5, 3., 3.5, 4.])]
            for a in values]


def disturbance(condition, seed):
    rng = np.random.default_rng(seed)
    if condition['family'] == 'severity':
        z = wind_trace(rng, False)
        z = z / np.abs(z).max()
    else:
        direction = rng.normal(size=3)
        direction /= np.abs(direction).max()
        t = np.arange(750) * .02
        signal = (t >= condition['onset']).astype(float)
        if condition['family'] == 'reversing':
            signal[t >= condition['reversal']] = -1
        z = signal[:, None] * direction
    return np.ascontiguousarray(z * condition['amplitude'], dtype=np.float64)


class DisturbedEnv(DroneEnv):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self._wind = self.trace.copy()
        return observation, info
