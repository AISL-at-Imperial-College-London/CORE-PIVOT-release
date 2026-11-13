"""Known smooth three-segment position, velocity, and acceleration reference."""

import numpy as np

INITIAL_VELOCITY = np.array([1.0, 0.0, 0.2])
VELOCITY_JUMPS = (
    np.array([-1.0, 1.0, 0.0]),
    np.array([-1.0, -1.0, -0.4]),
)


def reference(time):
    """Return position [m], velocity [m/s], and acceleration [m/s²].

    Turns at 5 and 10 seconds blend velocity over a one-second window.
    The reference does not stop at the finite-horizon endpoint.
    """
    if not np.isfinite(time) or not 0 <= time <= 15:
        raise ValueError("Reference time must lie in [0, 15] seconds.")
    position = INITIAL_VELOCITY * time
    velocity = INITIAL_VELOCITY.copy()
    acceleration = np.zeros(3)
    for centre, jump in zip((5.0, 10.0), VELOCITY_JUMPS):
        u = float(np.clip(time - (centre - 0.5), 0, 1))
        blend = u * u * (3 - 2 * u)
        if time <= centre - 0.5:
            integral = 0.0
        elif time >= centre + 0.5:
            integral = time - centre
        else:
            integral = u**3 - 0.5 * u**4
        position = position + jump * integral
        velocity = velocity + jump * blend
        acceleration = acceleration + jump * (6 * u * (1 - u))
    return position, velocity, acceleration
