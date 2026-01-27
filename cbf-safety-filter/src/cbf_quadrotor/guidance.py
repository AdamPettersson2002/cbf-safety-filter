import numpy as np
import scipy.linalg


class NominalGuidance:
    """
    Generates a nominal control input u_nom to track a target.
    Uses LQR (derived from CLF theory) to find the optimal guidance.
    """

    def __init__(self):
        # Define the System (Double Integrator Error Dynamics)
        # e_dot = A*e + B*u
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)

        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)

        # Define Weights (Tuning Knobs)
        # Q: Penalty on position/velocity error
        # R: Penalty on control effort
        Q = np.eye(6) * 10.0
        R = np.eye(3) * 1.0

        # Solve ARE for P
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # Compute LQR Gain Matrix K = R^-1 * B^T * P
        # This gives the optimal control law: u = -K * error
        self.K = np.linalg.inv(R) @ B.T @ P

    def compute_u_nom(self, drone_pos, drone_vel, target_pos, target_vel):
        """
        Calculates u_nom based on current error state.
        """
        err_pos = drone_pos - target_pos
        err_vel = drone_vel - target_vel
        error_state = np.concatenate([err_pos, err_vel])

        # u = -K * e
        u_nom = -self.K @ error_state
        noise = np.random.normal(0, 0.01, size=3)
        return u_nom + noise