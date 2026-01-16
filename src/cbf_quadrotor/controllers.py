"""
Nominal PD controller for goal reaching.
"""

import numpy as np


def pd_controller(x: np.ndarray, goal: np.ndarray, kp: float, kd: float) -> np.ndarray:
    """
    PD control law for goal reaching.

    u_nom = kp * (p_goal - p) - kd * v

    Args:
        x: current state [px, py, vx, vy]
        goal: goal position [px_goal, py_goal]
        kp: proportional gain
        kd: derivative gain

    Returns:
        u_nom: nominal control [ax, ay]
    """
    p = x[:2]
    v = x[2:]

    error = goal - p
    u_nom = kp * error - kd * v

    return u_nom


def is_goal_reached(x: np.ndarray, goal: np.ndarray, eps: float = 0.1) -> bool:
    """
    Check if goal is reached within tolerance.

    Args:
        x: current state [px, py, vx, vy]
        goal: goal position [px_goal, py_goal]
        eps: position tolerance

    Returns:
        True if ||p - goal|| < eps
    """
    p = x[:2]
    return np.linalg.norm(p - goal) < eps
