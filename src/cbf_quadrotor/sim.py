import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from dynamics import DroneDynamics, State
from guidance import NominalGuidance
from safety_filter import SafetyFilter
import scenarios


def run_simulation(scenario_name="head_on"):
    # 1. LOAD SCENARIO
    if scenario_name not in scenarios.SCENARIOS:
        print(f"Error: Scenario '{scenario_name}' not found.")
        return

    # Instantiate the scenario object
    mission = scenarios.SCENARIOS[scenario_name]()
    print(f"Loaded Scenario: {mission.name}")

    # 2. Initialize Engines
    dt = 0.01
    drone_physics = DroneDynamics(dt=dt)
    guidance = NominalGuidance()
    safety_filter = SafetyFilter(u_max=10.0)

    # Load Initial State from Scenario
    current_state = mission.start_state

    # Storage
    path_history = []
    target_history = []

    # 3. RUN LOOP
    total_steps = int(mission.duration / dt)

    for t_step in range(total_steps):
        time = t_step * dt

        # --- A. UPDATE SCENARIO OBJECTS ---
        # 1. Update Target (Generic method works for Static OR Moving!)
        target_pos, target_vel = mission.target.update(time)

        # 2. Update Obstacles (Simple Physics)
        for obs in mission.obstacles:
            # Simple bounce logic for patrolling obstacles
            if np.linalg.norm(obs.velocity) > 0:
                obs.center += obs.velocity * dt
                # Bounce bounds (simple example)
                if abs(obs.center[1]) > 4.0:
                    obs.velocity[1] *= -1

        # --- B. CONTROL PIPELINE ---
        u_nom = guidance.compute_u_nom(
            current_state.pos, current_state.vel,
            target_pos, target_vel
        )

        u_safe = safety_filter.filter(
            u_nom,
            current_state.pos, current_state.vel,
            mission.obstacles
        )

        current_state = drone_physics.step(current_state, u_safe)

        # Logging
        path_history.append(current_state.pos)
        target_history.append(target_pos)

    # --- VISUALIZATION ---
    plot_results(np.array(path_history), np.array(target_history), mission)


def plot_results(path, target_path, mission):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Drone', color='blue')
    ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], label='Target', color='green', linestyle='--')

    # Plot obstacles
    u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:10j]
    for obs in mission.obstacles:
        x = obs.radius * np.cos(u) * np.sin(v) + obs.center[0]
        y = obs.radius * np.sin(u) * np.sin(v) + obs.center[1]
        z = obs.radius * np.cos(v) + obs.center[2]
        ax.plot_wireframe(x, y, z, color="red", alpha=0.3)

    ax.set_title(f"Scenario: {mission.name}")
    ax.set_xlabel('X');
    ax.set_ylabel('Y');
    ax.set_zlabel('Z')
    ax.set_xlim(-12, 12);
    ax.set_ylim(-12, 12);
    ax.set_zlim(-5, 5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Running Scenario test...")
    #run_simulation("head_on")
    #run_simulation("chase")
    run_simulation("clutter")
