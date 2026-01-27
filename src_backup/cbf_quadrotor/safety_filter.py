import numpy as np
import osqp
from scipy import sparse


class SafetyFilter:
    def __init__(self, u_max=10.0):
        self.u_max = u_max

    def filter(self, u_nom, drone_pos, drone_vel, obstacles):
        """
        Solves QP:
        we use the identity matrix for Hessian matrix
        minimize || u - u_nom ||^2
        subject to CBF Constraints (A_cbf * u <= b_cbf)
        """

        # --- SETUP QP ---
        # Minimize (1/2)u^T P u + q^T u
        # To match ||u - u_nom||^2, we expand to: u^T u - 2*u_nom^T u
        # So P (Hessian) is Identity, q (linear) is -u_nom

        P_qp = sparse.csc_matrix(np.eye(3))
        q_qp = -u_nom

        # --- OBSTACLE CONSTRAINTS ---
        A_cbf_list = []
        b_cbf_list = []

        for obs in obstacles:
            A_i, b_i = obs.get_cbf_constraints(drone_pos, drone_vel)
            A_cbf_list.append(A_i)
            b_cbf_list.append(b_i)

        # --- BUILD CONSTRAINTS ---
        if not A_cbf_list:
            # No obstacles
            return np.clip(u_nom, -self.u_max, self.u_max)

        A_cons = np.vstack(A_cbf_list)
        b_cons = np.hstack(b_cbf_list)

        # Add Actuator Box Constraints (-u_max <= u <= u_max)
        A_box = np.eye(3)
        l_box = np.full(3, -self.u_max)
        u_box = np.full(3, self.u_max)

        # Combine for OSQP (l <= Ax <= u)
        # CBF is one-sided (Ax <= b), so l = -inf
        l_cons = np.full_like(b_cons, -np.inf)

        A_final = sparse.csc_matrix(np.vstack([A_cons, A_box]))
        l_final = np.hstack([l_cons, l_box])
        u_final = np.hstack([b_cons, u_box])

        # --- SOLVE ---
        prob = osqp.OSQP()
        prob.setup(P_qp, q_qp, A_final, l_final, u_final, verbose=False, polish=False)
        res = prob.solve()

        # --- STATUS ---
        if res.info.status not in ['solved', 'solved inaccurate']:
            print("Safety Filter Infeasible! Braking.")
            norm_v = np.linalg.norm(drone_vel)
            if norm_v > 0.01:
                # Direction opposing velocity
                u_brake = -drone_vel / norm_v * self.u_max
                return u_brake
            else:
                return np.zeros(3)

        return res.x
