import numpy as np
from dataclasses import dataclass


@dataclass
class Obstacle:
    """
    Represents a spherical obstacle.
    """
    center: np.ndarray  # [x, y, z] position
    velocity: np.ndarray  # [vx, vy, vz] velocity
    radius: float  # Radius of the obstacle

    def update(self, dt: float):
        """Default behavior: Linear movement based on constant velocity."""
        self.center += self.velocity * dt
        

    def get_cbf_constraints(self, drone_pos, drone_vel, k0=5.0, k1=5.0):
        """
        Calculates the linear constraint on control input u:
        A * u <= b
        Derived from the High-Order CBF condition:
        h_ddot + k1 * h_dot + k0 * h >= 0
        """
        rel_pos = drone_pos - self.center
        rel_vel = drone_vel - self.velocity

        # Barrier Function h(x)
        # h = ||dp||^2 - R^2
        dist_sq = np.dot(rel_pos, rel_pos)
        h = dist_sq - self.radius ** 2

        # First Derivative h_dot
        # h_dot = 2 * dp^T * dv
        h_dot = 2 * np.dot(rel_pos, rel_vel)

        # Second Derivative h_ddot terms
        # h_ddot = 2*||dv||^2 + 2*dp^T * (u_drone - u_obs)
        # We assume obstacle acceleration (u_obs) is 0 for prediction.

        # The constraint is:
        # 2*dp^T * u_drone >= -2*||dv||^2 - k1*h_dot - k0*h

        # Convert to form: A * u <= b
        A = -2 * rel_pos
        b = 2 * np.dot(rel_vel, rel_vel) + k1 * h_dot + k0 * h
        return A, b


class PatrollingObstacle(Obstacle):
    """Starts at Waypoint A, start moving toward Waypoint B."""
    def __init__(self, waypoint_a, waypoint_b, speed=2.0, radius=1.0):
        super().__init__(center=np.array(waypoint_a, dtype=float), velocity=np.zeros(3), radius=radius)
        self.waypoint_a = np.array(waypoint_a, dtype=float)
        self.waypoint_b = np.array(waypoint_b, dtype=float)
        self.speed = speed
        self.target = self.waypoint_b

    def update(self, dt: float):
        direction = self.target - self.center
        dist = np.linalg.norm(direction)

        if dist < 0.1:
            if np.array_equal(self.target, self.waypoint_b):
                self.target = self.waypoint_a
            else:
                self.target = self.waypoint_b
            
            direction = self.target - self.center
            dist = np.linalg.norm(direction)

        if dist > 0:
            norm_dir = direction / dist
            self.velocity = norm_dir * self.speed
            self.center += self.velocity * dt
        else:
            self.velocity = np.zeros(3)
