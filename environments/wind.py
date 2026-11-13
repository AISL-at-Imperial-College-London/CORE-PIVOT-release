"""Seeded acceleration disturbances used by the drone benchmark."""

import numpy as np


def wind_trace(rng, stronger=False):
    """Generate 750 bounded random-walk samples in m/s².

    Stronger winds rescale the complete trace to peak absolute component 1.5,
    matching the historical stress protocol. This is not a new online process.
    """
    disturbance = np.zeros(3)
    trace = []
    for _ in range(750):
        disturbance = np.clip(
            disturbance + 0.025 * rng.uniform(-1, 1, 3), -1, 1
        )
        trace.append(disturbance.copy())
    trace = np.asarray(trace)
    if stronger:
        peak = np.abs(trace).max()
        if peak == 0:
            raise ValueError("Cannot rescale an identically zero wind trace.")
        trace = trace * (1.5 / peak)
    return trace
