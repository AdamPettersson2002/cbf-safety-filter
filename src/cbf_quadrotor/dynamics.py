"""
Double integrator dynamics for 2D quadrotor simulation.

State: x = [px, py, vx, vy]
Control: u = [ax, ay]

Dynamics:
  p_{k+1} = p_k + dt * v_k
  v_{k+1} = v_k + dt * u_k
"""

import numpy as np


def step(x: np.ndarray, u: np.ndarray, dt: float, v_max: float = None) -> np.ndarray:
    """
    Integrate dynamics one timestep using Euler method.

    Args:
        x: current state [px, py, vx, vy]
        u: control input [ax, ay]
        dt: timestep
        v_max: optional speed limit

    Returns:
        x_next: next state [px, py, vx, vy]
    """
    p = x[:2]
    v = x[2:]

    # Update velocity
    v_next = v + dt * u

    # Optional speed limiting
    if v_max is not None:
        speed = np.linalg.norm(v_next)
        if speed > v_max:
            v_next = v_next * (v_max / speed)

    # Update position
    p_next = p + dt * v_next

    return np.concatenate([p_next, v_next])