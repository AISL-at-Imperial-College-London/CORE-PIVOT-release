"""Convert three-axis acceleration commands between physical and model units.

Physical commands are bounded independently on each axis, not by vector norm.
Networks and critics use normalized coordinates; simulation uses physical units.
"""

import numpy as np

ACTION_DIM = 3
ACTION_LIMIT = 5.0
PHYSICAL_LOW = (-ACTION_LIMIT,) * ACTION_DIM
PHYSICAL_HIGH = (ACTION_LIMIT,) * ACTION_DIM


def _action_array(action):
    """Accept one action or a batch with three components on the last axis."""
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != ACTION_DIM:
        raise ValueError("Actions must have shape (..., 3).")
    if not np.isfinite(array).all():
        raise ValueError("Actions must contain only finite values.")
    return array


def normalize_action(physical_action):
    """Map physical acceleration [-5, 5] to [-1, 1], without clipping data.

Reject out-of-range recorded commands rather than silently changing the dataset.
"""
    action = _action_array(physical_action)
    if np.any(np.abs(action) > ACTION_LIMIT):
        raise ValueError("Physical actions must lie within [-5, 5] on each axis.")
    return action / ACTION_LIMIT


def denormalize_action(normalized_action):
    """Map model actions to physical units, clipping proposals to [-1, 1]."""
    action = _action_array(normalized_action)
    return ACTION_LIMIT * np.clip(action, -1.0, 1.0)


def clip_physical_action(physical_action):
    """Enforce the environment's per-axis physical command limits."""
    return np.clip(_action_array(physical_action), -ACTION_LIMIT, ACTION_LIMIT)
