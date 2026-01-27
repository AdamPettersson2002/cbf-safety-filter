import numpy as np
import scipy.linalg

class NominalGuidance:
    """
    Generates a nominal control input u_nom.
    
    MODES:
    - 'rendezvous': Smoothly match target position AND velocity (Docking)
    - 'intercept':  Aggressively hit target position, ignore velocity matching (Missile)
    """

    def __init__(self, mode='intercept'):
        self.mode = mode
        
        # System Dynamics (Double Integrator)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3)

        # --- TUNING FOR INTERCEPTION ---
        if mode == 'intercept':
            # Q: State Penalty
            Q = np.eye(6)
            Q[0:3, 0:3] *= 500  # Huge penalty on POSITION error (Hit it!)
            Q[3:6, 3:6] *= 0.1  # Zero penalty on VELOCITY error (Don't brake!)
            
            # R: Control Penalty
            # Low R = "Cheap" fuel. Use maximum acceleration to get there fast.
            R = np.eye(3) * 20   
            
        else: # Rendezvous (Original)
            Q = np.eye(6) * 10.0
            R = np.eye(3) * 1.0

        # Solve LQR
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P


    def compute_u_nom(self, drone_pos, drone_vel, target_pos, target_vel):
        """
        General Interceptor Logic:
        Always aims for the target's current position with zero intent to slow down.
        """
        # 1. Position Error: Where we are vs Where the target is right now
        err_pos = drone_pos - target_pos
        
        # 2. Velocity Error: 
        # By comparing drone_vel to ZERO (instead of target_vel), 
        # the LQR stops trying to match speeds. 
        # Combined with a low Q[3:6] penalty, the drone will just keep 
        # accelerating until it hits the arrival_threshold.
        err_vel = drone_vel - np.zeros(3) 

        error_state = np.concatenate([err_pos, err_vel])
        
        # 3. Apply Gain
        u_nom = -self.K @ error_state
        noise = np.random.normal(0, 0.01, size=3)
        return u_nom + noise