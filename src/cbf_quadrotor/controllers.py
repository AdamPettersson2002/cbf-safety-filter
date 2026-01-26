import numpy as np
import scipy.linalg
from scipy import sparse
import osqp


class CLF_CBF_QP:
    def __init__(self, u_max=10.0, c_clf=2.0):
        self.u_max = u_max      # Max acceleration
        self.c_clf = c_clf      # How aggressive the tracking is (decay rate)

        # --- LYAPUNOV FUNCTION ---
        # We need P for V(e) = e^T * P * e
        # We solve the Continuous Algebraic Riccati Equation (CARE)
        # for the system e_dot = A*e + B*u

        # System matrices for Double Integrator (6x6 and 6x3)
        # These represent how error evolves: p_err_dot = v_err, v_err_dot = u
        A_sys = np.zeros((6, 6))
        A_sys[0:3, 3:6] = np.eye(3)  # derivative of pos error is vel error
        
        B_sys = np.zeros((6, 3))
        B_sys[3:6, :] = np.eye(3)    # derivative of vel error is control u

        # Weights for the "optimal" behavior
        Q = np.eye(6) * 10.0  # High penalty on position/velocity error
        R = np.eye(3) * 1.0   # Moderate penalty on using fuel/acceleration

        # Solves: A^T P + P A - P B R^-1 B^T P + Q = 0
        self.P = scipy.linalg.solve_continuous_are(A_sys, B_sys, Q, R)

    def solve(self, drone_pos, drone_vel, target_pos, target_vel, obstacles):
        """
        Solves the QP:
        minimize   0.5 * ||u - u_ref||^2 + weight * delta^2
        subject to A_cbf * u <= b_cbf  (Hard Safety)
                   A_clf * u <= b_clf + delta (Soft Tracking)
                   |u| <= u_max
        """
        
        # --- 2. CALCULATE CLF (GUIDANCE) CONSTRAINTS ---
        # Error State
        e_pos = drone_pos - target_pos
        e_vel = drone_vel - target_vel
        e = np.concatenate([e_pos, e_vel]) # 6x1 vector

        # Calculate V(e) current energy
        V = e.T @ self.P @ e

        # Calculate Lie Derivatives
        # LfV = gradient(V) * f(x) = 2 * e^T * P * (Ax)
        # LgV = gradient(V) * g(x) = 2 * e^T * P * B
        
        # f(e) part of dynamics (A * e)
        f_e = np.zeros(6)
        f_e[:3] = e_vel # pos_dot = vel
        
        # g(e) part of dynamics (B) - rows 3,4,5 are identity
        
        LfV = 2 * e.T @ self.P @ f_e
        
        # LgV is a 1x3 vector (row vector)
        # B matrix is 0s on top, Identity on bottom. 
        # So P @ B is just the bottom 3 columns of P.
        PB = self.P @ np.zeros((6,3)) # Helper to visualize
        PB[3:6, :] = self.P[3:6, 3:6] # This logic depends on B structure...
        # Let's do it the safe numpy way:
        B_mat = np.zeros((6, 3))
        B_mat[3:6, :] = np.eye(3)
        LgV = 2 * e.T @ self.P @ B_mat

        # CLF Condition: LfV + LgV*u <= -c*V + delta
        # Form: LgV*u - delta <= -c*V - LfV
        # Vector variables for QP: [ux, uy, uz, delta]
        
        A_clf = np.hstack([LgV, -1.0])  # [LgV_x, LgV_y, LgV_z, -1]
        b_clf = -self.c_clf * V - LfV

        # --- 3. CALCULATE CBF (SAFETY) CONSTRAINTS ---
        A_cbf_list = []
        b_cbf_list = []
        
        for obs in obstacles:
            # Your Obstacle class handles the math here!
            A_i, b_i = obs.get_cbf_constraints(drone_pos, drone_vel)
            
            # Add 0 coefficient for delta (safety is hard constraint, no slack)
            # A_i is shape (3,), we need (4,) -> [Ax, Ay, Az, 0]
            A_cbf_list.append(np.append(A_i, 0.0)) 
            b_cbf_list.append(b_i)

        # --- 4. BUILD AND SOLVE QP ---
        # Variables: [ux, uy, uz, delta]
        
        # Cost: 0.5 * u^2 + large_cost * delta^2
        H = sparse.csc_matrix(np.diag([1.0, 1.0, 1.0, 10000.0]))
        q = np.zeros(4) # Linear cost terms (none)

        # Combine Constraints
        # Stack CLF and CBF rows
        if A_cbf_list:
            A_cons = np.vstack([A_clf] + A_cbf_list)
            b_cons = np.hstack([b_clf] + b_cbf_list)
        else:
            A_cons = np.atleast_2d(A_clf)
            b_cons = np.atleast_1d(b_clf)

        # Box constraints for u (-u_max to u_max) and delta (0 to inf)
        A_box = np.eye(4)
        l_box = np.array([-self.u_max, -self.u_max, -self.u_max, 0.0])
        u_box = np.array([self.u_max, self.u_max, self.u_max, np.inf])

        # OSQP Format: l <= Ax <= u
        # Our constraints are Ax <= b, so l = -infinity
        l_cons = np.full_like(b_cons, -np.inf)
        
        # Full Stack
        A_final = sparse.csc_matrix(np.vstack([A_cons, A_box]))
        l_final = np.hstack([l_cons, l_box])
        u_final = np.hstack([b_cons, u_box])

        # Solve
        prob = osqp.OSQP()
        prob.setup(H, q, A_final, l_final, u_final, verbose=False, polish=True)
        res = prob.solve()

        if res.info.status not in ['solved', 'solved inaccurate']:
            # Failsafe: return 0 acceleration
            return np.zeros(3)
            
        return res.x[:3] # Return just acceleration [ux, uy, uz]