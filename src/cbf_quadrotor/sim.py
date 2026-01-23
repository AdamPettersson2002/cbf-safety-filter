import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Custom modules
from dynamics import DroneDynamics
from guidance import NominalGuidance
from safety_filter import SafetyFilter
import scenarios


def run_simulation(scenario_name="chase", animate=True):
    # 1. Load Scenario
    if scenario_name not in scenarios.SCENARIOS:
        print(f"Error: Scenario '{scenario_name}' not found.")
        return
    mission = scenarios.SCENARIOS[scenario_name]()
    print(f"Loaded: {mission.name} | Animate: {animate}")

    # 2. Setup
    dt = 0.01
    drone_physics = DroneDynamics(dt=dt)
    guidance = NominalGuidance()
    safety_filter = SafetyFilter(u_max=10.0)
    current_state = mission.start_state

    # 3. Data Storage
    path_history = []
    target_history = []
    obs_history = []

    total_steps = int(mission.duration / dt)

    # 4. Simulation Loop
    print("Simulating physics...")
    for t_step in range(total_steps):
        time = t_step * dt

        # A. Update Scenario (Target + Obstacles)
        target_pos, target_vel = mission.target.update(time)

        current_obs_snapshot = []
        for obs in mission.obstacles:
            if np.linalg.norm(obs.velocity) > 0:
                obs.center += obs.velocity * dt
                if abs(obs.center[1]) > 4.0:
                    obs.velocity[1] *= -1
            current_obs_snapshot.append(obs.center.copy())

        obs_history.append(current_obs_snapshot)

        # B. Control & Physics
        u_nom = guidance.compute_u_nom(current_state.pos, current_state.vel, target_pos, target_vel)
        u_safe = safety_filter.filter(u_nom, current_state.pos, current_state.vel, mission.obstacles)
        current_state = drone_physics.step(current_state, u_safe)

        path_history.append(current_state.pos)
        target_history.append(target_pos)

    # 5. Visualization Selector
    path_history = np.array(path_history)
    target_history = np.array(target_history)

    if animate:
        print("Starting Animation...")
        animate_results(path_history, target_history, obs_history, mission, dt)
    else:
        print("Plotting Static Graph...")
        plot_static(path_history, target_history, mission)


# --- VISUALIZER 1: STATIC PLOT ---
def plot_static(path, target_path, mission):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Trajectories
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Drone', color='blue')
    ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], label='Target', color='green', linestyle='--')

    # Plot Obstacles (At their FINAL positions)
    u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:10j]
    for obs in mission.obstacles:
        x = obs.radius * np.cos(u) * np.sin(v) + obs.center[0]
        y = obs.radius * np.sin(u) * np.sin(v) + obs.center[1]
        z = obs.radius * np.cos(v) + obs.center[2]
        ax.plot_wireframe(x, y, z, color="red", alpha=0.3)

    ax.set_title(f"Static Result: {mission.name}")
    ax.set_xlabel('X');
    ax.set_ylabel('Y');
    ax.set_zlabel('Z')
    ax.set_xlim(-12, 12);
    ax.set_ylim(-12, 12);
    ax.set_zlim(-5, 5)
    plt.legend()
    plt.show()


# --- VISUALIZER 2: ANIMATION ---
def animate_results(path, target_path, obs_history, mission, dt):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    line_drone, = ax.plot([], [], [], color='blue', linewidth=1, label='Drone')
    point_drone, = ax.plot([], [], [], color='blue', marker='o')
    point_target, = ax.plot([], [], [], color='green', marker='*', markersize=10, label='Target')
    obs_plots = []

    ax.set_xlim(-12, 12);
    ax.set_ylim(-12, 12);
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X');
    ax.set_ylabel('Y');
    ax.set_zlabel('Z')
    ax.set_title(f"Animation: {mission.name}")
    ax.legend()

    # Pre-compute sphere geometry
    u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:8j]  # Lower res for speed
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)

    def update(frame):
        # Update Drone
        line_drone.set_data(path[:frame, 0], path[:frame, 1])
        line_drone.set_3d_properties(path[:frame, 2])
        point_drone.set_data([path[frame, 0]], [path[frame, 1]])
        point_drone.set_3d_properties([path[frame, 2]])

        # Update Target
        point_target.set_data([target_path[frame, 0]], [target_path[frame, 1]])
        point_target.set_3d_properties([target_path[frame, 2]])

        # Update Obstacles
        for p in obs_plots: p.remove()
        obs_plots.clear()

        current_obs = obs_history[frame]
        for i, obs_def in enumerate(mission.obstacles):
            # Scale and translate
            x = x_sphere * obs_def.radius + current_obs[i][0]
            y = y_sphere * obs_def.radius + current_obs[i][1]
            z = z_sphere * obs_def.radius + current_obs[i][2]
            obs_plots.append(ax.plot_wireframe(x, y, z, color="red", alpha=0.2))

        return line_drone, point_drone, point_target

    # Faster animation: interval=20ms means 50fps
    anim = FuncAnimation(fig, update, frames=len(path), interval=20, blit=False)
    plt.show()


if __name__ == "__main__":
    #run_simulation("head_on", animate=True)
    #run_simulation("chase", animate=True)
    run_simulation("clutter", animate=True)
