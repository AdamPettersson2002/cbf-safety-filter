import numpy as np
from dataclasses import dataclass


@dataclass
class State:
    """
    Represents the full 6D state of the drone.
    We separate position and velocity for clarity.
    """
    pos: np.ndarray  # [x, y, z] (Meters)
    vel: np.ndarray  # [vx, vy, vz] (Meters/second)

    @property
    def vector(self):
        """Returns the full 6x1 state vector used for matrix math."""
        return np.concatenate([self.pos, self.vel])


class DroneDynamics:
    def __init__(self, dt: float = 0.05):
        self.dt = dt

        # --- Double Integrator Matrices ---
        # State Transition Matrix A (6x6)
        # p_next = p + v*dt
        # v_next = v
        self.A = np.eye(6)
        self.A[0:3, 3:6] = np.eye(3) * dt

        # Control Input Matrix B (6x3)
        # p_next += 0
        # v_next += u*dt
        self.B = np.zeros((6, 3))
        self.B[3:6, :] = np.eye(3) * dt

    def step(self, x_k: np.ndarray, u: np.ndarray) -> State:
        """
        Advances the simulation by one time step dt.
        x_{k+1} = A * x_k + B * u_k
        """
        # x_k = state.vector
        x_next = self.A @ x_k + self.B @ u
        return x_next
        # return State(pos=x_next[:3], vel=x_next[3:])
