"""Finite-horizon drone tracking with the historical 25 observation features.

This is a commanded-acceleration point-mass model, not a motor/attitude simulator.
State integration uses float64; observations use float32 for network inputs.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.action_space import (
    PHYSICAL_HIGH,
    PHYSICAL_LOW,
    denormalize_action,
)
from environments.reference import reference
from environments.wind import wind_trace

OBSERVATION_NAMES = (
    "ex", "ey", "ez", "evx", "evy", "evz",
    "vx_ref_h10", "vy_ref_h10", "vz_ref_h10",
    "vx_ref_h20", "vy_ref_h20", "vz_ref_h20",
    "ax_prev", "ay_prev", "az_prev", "t_norm",
    "dhat_x", "dhat_y", "dhat_z",
    "vx_ref", "vy_ref", "vz_ref", "ax_ref", "ay_ref", "az_ref",
)
DT = 0.02
HORIZON = 15.0
EPISODE_STEPS = 750


def integrate(state, action, disturbance):
    """Explicit Euler: p_next=p+dt*v; v_next=v+dt*(command+wind)."""
    return np.concatenate((
        state[:3] + DT * state[3:],
        state[3:] + DT * (action + disturbance),
    ))


def tracking_cost(state, time):
    """Squared position error + 0.2 squared velocity error at the given state/time."""
    position, velocity, _ = reference(time)
    position_error = position - state[:3]
    velocity_error = velocity - state[3:]
    return float(
        np.dot(position_error, position_error)
        + 0.2 * np.dot(velocity_error, velocity_error)
    )


class DroneEnv(gym.Env):
    """Physical three-axis acceleration actions in [-5, 5] m/s².

    The 15-second endpoint is a task terminal, not an external time-limit cutoff.
    Global time is observable. Wind is hidden; only a previous-transition wind estimate
    is exposed. Ground contact, attitude, gravity, and motor dynamics are absent.
    """

    metadata = {"render_modes": []}

    def __init__(self, stronger_wind=False):
        self.stronger_wind = bool(stronger_wind)
        self.action_space = spaces.Box(
            np.array(PHYSICAL_LOW, dtype=np.float32),
            np.array(PHYSICAL_HIGH, dtype=np.float32), dtype=np.float32,
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)
        self._active = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("No reset options are supported by this benchmark.")
        self.state = np.zeros(6)
        self.tick = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_velocity = None
        self.previous_time = None
        self._wind = wind_trace(self.np_random, self.stronger_wind)
        self._active = True
        return self._observation(), self._info()

    def _observation(self):
        time = self.tick * DT
        position, velocity, acceleration = reference(time)
        state = self.state.astype(np.float32)
        # Preserve the original deployment observation's float32 arithmetic.
        position_error = position.astype(np.float32) - state[:3]
        velocity_error = velocity.astype(np.float32) - state[3:]
        wind_estimate = np.zeros(3, dtype=np.float32)
        if self.previous_velocity is not None:
            wind_estimate = ((state[3:] - self.previous_velocity)
                        / (time - self.previous_time) - self.previous_action).astype(np.float32)
        preview10 = reference(min(time + 10 * DT, HORIZON))[1]
        preview20 = reference(min(time + 20 * DT, HORIZON))[1]
        return np.concatenate((
            position_error, velocity_error, preview10, preview20,
            self.previous_action, [time / HORIZON], wind_estimate,
            velocity, acceleration,
        )).astype(np.float32)

    def _info(self):
        position, velocity, _ = reference(self.tick * DT)
        return dict(time=self.tick * DT, state=self.state.copy(),
                    reference_position=position, reference_velocity=velocity)

    def step(self, action):
        if not self._active:
            raise RuntimeError("Call reset before stepping or after an episode ends.")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("An action must be a finite three-component vector.")
        applied = np.clip(action, PHYSICAL_LOW, PHYSICAL_HIGH)
        time = self.tick * DT
        disturbance = self._wind[self.tick]
        self.previous_velocity = self.state[3:].astype(np.float32)
        self.previous_time = time
        self.previous_action = applied.astype(np.float32)
        self.state = integrate(self.state, applied, disturbance)
        self.tick += 1
        # Match the offline datasets: score the successor state and reference.
        cost = tracking_cost(self.state, self.tick * DT)
        terminated = self.tick == EPISODE_STEPS
        self._active = not terminated
        info = self._info()
        info.update(cost=cost, applied_action=applied.copy(),
                    disturbance=disturbance.copy(), action_clipped=bool(np.any(applied != action)))
        return self._observation(), -cost, terminated, False, info


class NormalizedActionWrapper(gym.ActionWrapper):
    """Expose [-1, 1] commands while leaving observation features unchanged."""

    def __init__(self, env):
        if not isinstance(env.unwrapped, DroneEnv):
            raise TypeError("This wrapper requires the fixed-bounds DroneEnv.")
        super().__init__(env)
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

    def action(self, action):
        return denormalize_action(action)
