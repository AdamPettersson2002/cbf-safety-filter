"""
Utility functions for the simulation.
"""

import numpy as np


def clamp_magnitude(v: np.ndarray, max_mag: float) -> np.ndarray:
    """
    Clamp vector magnitude to max_mag.

    Args:
        v: input vector
        max_mag: maximum magnitude

    Returns:
        clamped vector
    """
    mag = np.linalg.norm(v)
    if mag > max_mag:
        return v * (max_mag / mag)
    return v


def clamp_componentwise(v: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
    """
    Clamp each component of vector.

    Args:
        v: input vector
        v_min: minimum value
        v_max: maximum value

    Returns:
        clamped vector
    """
    return np.clip(v, v_min, v_max)