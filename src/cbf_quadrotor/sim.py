import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from dynamics import DroneDynamics
from guidance import PPNGuidance
from safety_filter import SafetyFilter
from scenarios import MissionFactory


def plot_static_results(drone_path, target_path, obstacles, u_nom_hist, u_safe_hist, time_hist):
    # --- FIGURE 1: 3D TRAJECTORY ---
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    path = np.array(drone_path)
    t_path = np.array(target_path)

    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Drone', color='blue', linewidth=2)
    ax.plot(t_path[:, 0], t_path[:, 1], t_path[:, 2], label='Target', color='green', linestyle='--')

    u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:10j]
    for obs in obstacles:
        x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
        y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
        z = obs.center[2] + obs.radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color='red', alpha=0.3)

    ax.set_title("3D Flight Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # --- FIGURE 2: CONTROL INPUTS ---
    u_nom = np.array(u_nom_hist)
    u_safe = np.array(u_safe_hist)
    t = np.array(time_hist)

    fig2, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['$a_x$', '$a_y$', '$a_z$']

    for i in range(3):
        axs[i].plot(t, u_nom[:, i], 'g--', label='Nominal (LQR)', alpha=0.6)
        axs[i].plot(t, u_safe[:, i], 'b-', label='Safe (CBF)')

        diff = np.abs(u_nom[:, i] - u_safe[:, i])
        mask = diff > 0.05
        if np.any(mask):
            axs[i].fill_between(t, -15, 15, where=mask, color='red', alpha=0.1, label='Intervention')

        axs[i].set_ylabel(f"{labels[i]} ($m/s^2$)")
        axs[i].grid(True)

    axs[0].set_title("Control Inputs: PPN (Green) vs CBF (Blue)")
    axs[0].legend(loc='upper right')
    axs[2].set_xlabel("Time (s)")


def animate_sim(path, target_path, obs_history, mission, dt):
    path = np.array(path)
    target_path = np.array(target_path)
    total_frames = len(path)

    stride = max(1, total_frames // 200)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Background elements
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', alpha=0.2, linewidth=1, label='Path')
    ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], 'g--', alpha=0.2, linewidth=1)

    # 2. Dynamic elements
    # NOTE: We use lists [x], [y] inside update() to avoid the "sequence" error
    drone_dot, = ax.plot([], [], [], 'bo', markersize=8, label='Drone')
    target_dot, = ax.plot([], [], [], 'gD', markersize=8, label='Target')

    obs_actors = []

    # Axis limits
    all_x = np.concatenate([path[:, 0], target_path[:, 0]])
    all_y = np.concatenate([path[:, 1], target_path[:, 1]])
    all_z = np.concatenate([path[:, 2], target_path[:, 2]])
    pad = 2.0

    ax.set_xlim(np.min(all_x) - pad, np.max(all_x) + pad)
    ax.set_ylim(np.min(all_y) - pad, np.max(all_y) + pad)
    ax.set_zlim(np.min(all_z) - pad, np.max(all_z) + pad)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Simulation: {mission.name}")
    ax.legend()

    # Pre-calculate sphere geometry
    u_grid, v_grid = np.mgrid[0:2 * np.pi:10j, 0:np.pi:8j]
    x_unit = np.cos(u_grid) * np.sin(v_grid)
    y_unit = np.sin(u_grid) * np.sin(v_grid)
    z_unit = np.cos(v_grid)

    def update(frame_idx):
        nonlocal obs_actors

        # A. Update Drone/Target
        # Fix for "x must be a sequence": Wrap scalars in lists []
        drone_dot.set_data([path[frame_idx, 0]], [path[frame_idx, 1]])
        drone_dot.set_3d_properties([path[frame_idx, 2]])

        target_dot.set_data([target_path[frame_idx, 0]], [target_path[frame_idx, 1]])
        target_dot.set_3d_properties([target_path[frame_idx, 2]])

        # B. Update Obstacles (Remove old, draw new)
        for actor in obs_actors:
            actor.remove()
        obs_actors.clear()

        current_obs_centers = obs_history[frame_idx]
        for i, center in enumerate(current_obs_centers):
            radius = mission.obstacles[i].radius
            X = center[0] + radius * x_unit
            Y = center[1] + radius * y_unit
            Z = center[2] + radius * z_unit

            actor = ax.plot_wireframe(X, Y, Z, color='red', alpha=0.5, rcount=10, ccount=10)
            obs_actors.append(actor)

        return drone_dot, target_dot, *obs_actors

    ani = animation.FuncAnimation(
        fig, update, frames=range(0, total_frames, stride),
        interval=30, blit=False
    )
    return ani


def run_simulation(mission, animate=True):
    try:
        print(f"Loaded: {mission.name} | Animate: {animate}")
        dt = 0.01
        drone_physics = DroneDynamics(dt=dt)
        guidance = PPNGuidance()
        safety_filter = SafetyFilter()
        current_state = mission.start_state

        arrival_threshold = 0.2

        path_history = []
        target_history = []
        obs_history = []

        u_nom_history = []
        u_safe_history = []
        time_history = []

        total_steps = int(mission.duration / dt)
        print(f"Simulating {total_steps} steps...")

        for t_step in range(total_steps):
            time = t_step * dt

            target_pos, target_vel = mission.target.update(time)

            dist_to_target = np.linalg.norm(current_state.pos - target_pos)
            if dist_to_target < arrival_threshold:
                print(f"Target Reached! Stopping at T={time:.2f}s")
                break

            current_obs_snapshot = []
            for obs in mission.obstacles:
                obs.update(dt)
                current_obs_snapshot.append(obs.center.copy())
            obs_history.append(current_obs_snapshot)

            u_nom = guidance.compute_u_nom(current_state.pos, current_state.vel, target_pos, target_vel)
            u_safe = safety_filter.filter(u_nom, current_state.pos, current_state.vel, mission.obstacles)
            x_k = current_state.vector
            next_state = drone_physics.step(x_k, u_safe)
            current_state.pos = next_state[:3]
            current_state.vel = next_state[3:]

            path_history.append(current_state.pos)
            target_history.append(target_pos)
            u_nom_history.append(u_nom.copy())
            u_safe_history.append(u_safe.copy())
            time_history.append(time)

        path_history = np.array(path_history)
        target_history = np.array(target_history)

        if len(path_history) == 0:
            return

        plot_static_results(path_history, target_history, mission.obstacles,
                            u_nom_history, u_safe_history, time_history)

        anim_object = None
        if animate:
            anim_object = animate_sim(path_history, target_history, obs_history, mission, dt)
            # TODO: Save animation rather than render it. If we wish to run all the missions in one go, 
            # it's better if we save them and look and them in MP4's or something afterwards.

        plt.show()
    except Exception as e:
        print(f"Exception: {e}")


def main():
    mf = MissionFactory(scenarios={"head_on"})
    missions = mf.generate_missions()
    for m in missions:

        print("Starting Simulation...")
        run_simulation(m, animate=True)


if __name__ == "__main__":
    main()
