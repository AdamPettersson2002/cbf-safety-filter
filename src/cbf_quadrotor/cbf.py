"""
Control Barrier Function (CBF) safety filter for 2D quadrotor simulation.

For a no-fly zone represented as a circle:
  Safe set: h(p) = ||p - c||^2 - R^2 >= 0  (R = r + d_safe)

For double integrator dynamics (relative degree 2):
  p_dot = v
  v_dot = u

We need h_ddot + k1*h_dot + k0*h >= 0 (ECBF constraint)

Derivation:
  h(p) = (px-cx)^2 + (py-cy)^2 - R^2
  h_dot = 2*(px-cx)*vx + 2*(py-cy)*vy = 2*(p-c)^T * v
  h_ddot = 2*vx^2 + 2*(px-cx)*ax + 2*vy^2 + 2*(py-cy)*ay
         = 2*||v||^2 + 2*(p-c)^T * u

Linear constraint in u:
  h_ddot + k1*h_dot + k0*h >= 0
  2*(p-c)^T * u >= -2*||v||^2 - k1*h_dot - k0*h

Matrix form: A*u >= b where
  A = 2*(p-c)^T  (1x2 row vector)
  b = -2*||v||^2 - k1*h_dot - k0*h
"""

import numpy as np
from typing import List, Dict, Tuple
import osqp
from scipy import sparse


def compute_barrier_constraint(x: np.ndarray, zone: Dict, k0: float, k1: float) -> Tuple[np.ndarray, float]:
    """
    Compute CBF constraint A*u >= b for a single circular no-fly zone.

    Args:
        x: state [px, py, vx, vy]
        zone: dict with 'center' (x,y) and 'radius' (with margin already added)
        k0, k1: ECBF parameters

    Returns:
        A: (1,2) array - constraint gradient
        b: scalar - constraint bound
    """
    p = x[:2]  # position
    v = x[2:]  # velocity
    c = zone['center']
    R = zone['radius']

    # Barrier function h(p) = ||p-c||^2 - R^2
    dp = p - c
    h = np.dot(dp, dp) - R ** 2

    # h_dot = 2*(p-c)^T * v
    h_dot = 2 * np.dot(dp, v)

    # Constraint: h_ddot + k1*h_dot + k0*h >= 0
    # where h_ddot = 2*||v||^2 + 2*(p-c)^T * u

    # A*u >= b form:
    A = 2 * dp  # (2,) array
    b = -2 * np.dot(v, v) - k1 * h_dot - k0 * h

    return A, b


def cbf_filter(x: np.ndarray, u_nom: np.ndarray, zones: List[Dict],
               params: Dict) -> np.ndarray:
    """
    QP-based CBF safety filter.

    Solves:
        minimize ||u - u_nom||^2
        subject to: A_i * u >= b_i for all zones
                    ||u|| <= a_max (box constraints)

    Args:
        x: state [px, py, vx, vy]
        u_nom: nominal control [ax, ay]
        zones: list of no-fly zones
        params: dict with 'k0', 'k1', 'a_max'

    Returns:
        u_safe: filtered control [ax, ay]
    """
    k0 = params['k0']
    k1 = params['k1']
    a_max = params['a_max']

    # Build constraint matrices
    A_list = []
    b_list = []

    for zone in zones:
        A_i, b_i = compute_barrier_constraint(x, zone, k0, k1)
        A_list.append(A_i)
        b_list.append(b_i)

    if len(A_list) == 0:
        # No constraints, just clamp
        return np.clip(u_nom, -a_max, a_max)

    # Stack constraints
    A = np.vstack(A_list)  # (n_zones, 2)
    b = np.array(b_list)  # (n_zones,)

    # Setup QP: min 0.5*u^T*P*u + q^T*u  s.t. l <= A*u <= u
    # For ||u - u_nom||^2 = u^T*u - 2*u_nom^T*u + const
    P = sparse.csc_matrix(2 * np.eye(2))  # 2*I for the 0.5 factor
    q = -2 * u_nom

    # Lower bounds: A*u >= b => lower = b
    l = b
    u_upper = np.full(len(b), np.inf)

    # Box constraints: -a_max <= u <= a_max
    # Add as additional constraints
    A_box = np.vstack([np.eye(2), -np.eye(2)])  # (4, 2)
    l_box = np.array([-a_max, -a_max, -a_max, -a_max])
    u_box = np.full(4, np.inf)

    # Combine all constraints
    A_full = np.vstack([A, A_box])
    l_full = np.concatenate([l, l_box])
    u_full = np.concatenate([u_upper, u_box])

    A_sparse = sparse.csc_matrix(A_full)

    # Solve QP
    try:
        prob = osqp.OSQP()
        prob.setup(P, q, A_sparse, l_full, u_full, verbose=False, polish=True)
        result = prob.solve()

        if result.info.status == 'solved' or result.info.status == 'solved inaccurate':
            return result.x
        else:
            # Fallback: use nominal control clamped
            print(f"Warning: QP solver status: {result.info.status}")
            return np.clip(u_nom, -a_max, a_max)
    except Exception as e:
        print(f"Warning: QP solver failed: {e}")
        return np.clip(u_nom, -a_max, a_max)


def check_safety(x: np.ndarray, zones: List[Dict]) -> Tuple[bool, float]:
    """
    Check if current state is safe and return minimum margin.

    Args:
        x: state [px, py, vx, vy]
        zones: list of no-fly zones

    Returns:
        is_safe: True if safe
        min_margin: minimum distance margin (positive = safe)
    """
    p = x[:2]
    min_margin = np.inf

    for zone in zones:
        c = zone['center']
        R = zone['radius']
        dist = np.linalg.norm(p - c)
        margin = dist - R
        min_margin = min(min_margin, margin)

    return min_margin >= 0, min_margin