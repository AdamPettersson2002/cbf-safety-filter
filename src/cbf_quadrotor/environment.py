"""
Environment setup: no-fly zones and test scenarios.
"""

import numpy as np
from typing import List, Dict


def create_default_scenario() -> Dict:
    """
    Create the default test scenario.

    Returns:
        scenario: dict with 'start', 'goal', 'zones'
    """
    # Initial state
    start = np.array([0.0, 0.0, 0.0, 0.0])  # [px, py, vx, vy]

    # Goal position
    goal = np.array([10.0, 8.0])

    # No-fly zones: circles with center and radius
    # Note: radius here includes the safety margin d_safe
    d_safe = 0.5  # safety margin
    zones = [
        {'center': np.array([4.0, 3.0]), 'radius': 1.0 + d_safe},
        {'center': np.array([7.0, 6.0]), 'radius': 1.2 + d_safe},
    ]

    return {
        'start': start,
        'goal': goal,
        'zones': zones,
        'd_safe': d_safe  # for visualization
    }


def get_simulation_params() -> Dict:
    """
    Get default simulation parameters.

    Returns:
        params: dict with simulation and control parameters
    """
    # Acceleration limit based on tilt angle
    # For quadrotor: a_max ≈ g * tan(theta_max)
    g = 9.81  # m/s^2
    theta_max_deg = 25.0
    a_max = g * np.tan(np.deg2rad(theta_max_deg))

    params = {
        # Simulation
        'dt': 0.02,  # timestep (s)
        'T': 20.0,  # total time (s)

        # Controller gains
        'kp': 1.2,
        'kd': 1.8,

        # Dynamics limits
        'a_max': a_max,
        'v_max': 5.0,  # optional speed limit (m/s)

        # CBF parameters
        'k0': 1.0,  # ECBF parameter
        'k1': 2.0,  # ECBF parameter

        # Goal tolerance
        'goal_eps': 0.1,
    }

    return params