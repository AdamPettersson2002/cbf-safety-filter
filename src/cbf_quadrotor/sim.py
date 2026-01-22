import numpy as np
import matplotlib.pyplot as plt
from dynamics import DroneDynamics, State


def run_simulation():
    dt = 0.05
    drone_physics = DroneDynamics(dt=dt)

    # Initial Conditions
    # Start at origin (0,0,0) with no velocity
    current_state = State(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0])
    )

    # Storage for plotting
    path_history = []

    # Simulation Loop
    print("Starting Simulation...")
    for t_step in range(100):

        # --- CONTROLLER PLACEHOLDER ---
        # Ideally, CLF-CBF-QP goes here.
        u_control = np.array([1.0, 0.2, 0.5])
        # -----------------------------

        # Step the Physics
        current_state = drone_physics.step(current_state, u_control)

        # Save for plotting
        path_history.append(current_state.pos)

    # 5. Visualization (Simple 3D Plot)
    path_history = np.array(path_history)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path_history[:,0], path_history[:,1], path_history[:,2], label='Drone Path')
    ax.set_xlabel('X (East)')
    ax.set_ylabel('Y (North)')
    ax.set_zlabel('Z (Altitude)')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()