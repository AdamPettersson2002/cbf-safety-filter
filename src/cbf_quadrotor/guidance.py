import numpy as np
import scipy.linalg


class PPNGuidance:
    def __init__(self, gain=4.0):
        self.N = gain

    def compute_u_nom(self, drone_pos, drone_vel, target_pos, target_vel):
        rel_pos = target_pos - drone_pos
        rel_vel = target_vel - drone_vel
        dist = np.linalg.norm(rel_pos)

        if dist < 0.01:
            return np.zeros(3)

        # Calculate rotational velocity of LOS vector
        # Omega = (r x v) / (r . r)
        omega = np.cross(rel_pos, rel_vel) / (dist**2)

        # Accelerate perpendicular to the LOS
        closing_vel = -rel_vel
        accel_cmd = self.N * np.cross(omega, closing_vel)

        push_gain = 2.0
        push_cmd = (rel_pos / dist) * push_gain
        noise = np.random.normal(0, 0.01, size=3)
        return accel_cmd + push_cmd + noise
