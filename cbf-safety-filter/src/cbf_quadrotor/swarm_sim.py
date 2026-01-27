import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dynamics import DroneDynamics
from guidance import NominalGuidance
from safety_filter import SafetyFilter
from constraints import Obstacle
import swarm_scenarios


def run_swarm_simulation(scenario_name="formation"):
    # 1. LOAD SCENARIO
    if scenario_name not in swarm_scenarios.SWARM_SCENARIOS:
        print("Scenario not found.")
        return
    mission = swarm_scenarios.SWARM_SCENARIOS[scenario_name]()
    print(f"Loaded Swarm Mission: {mission.name}")

    # 2. SETUP
    dt = 0.01
    num_drones = mission.num_drones

    drones = [DroneDynamics(dt=dt) for _ in range(num_drones)]
    guidance_sys = [NominalGuidance() for _ in range(num_drones)]
    safety_filters = [SafetyFilter(u_max=15.0) for _ in range(num_drones)]

    states = [s for s in mission.start_states]  # Copy starting states

    # Initialize targets (will update dynamically for formation)
    current_targets = [np.zeros(3) for _ in range(num_drones)]
    if mission.type == "independent":
        current_targets = mission.fixed_targets

    pos_history = [[] for _ in range(num_drones)]

    print("Simulating...")
    for t_step in range(1000):
        time = t_step * dt

        # --- A. UPDATE TARGETS ---
        if mission.type == "leader_follower":
            # 1. Update Global Target (Ghost)
            global_tgt_pos, global_tgt_vel = mission.global_target.update(time)

            # 2. Assign Leader Target (Drone 0)
            current_targets[0] = global_tgt_pos

            # 3. Assign Follower Targets (Drone i -> Leader Pos + Offset)
            leader_pos = states[0].pos
            for i in range(1, num_drones):
                # Simple formation: Go to where leader is + offset
                # (Ideally, you rotate offset by leader's yaw, but simple addition works for now)
                current_targets[i] = leader_pos + mission.formation_offsets[i]

        # --- B. PLAN & CONTROL ---
        safe_controls = []

        for i in range(num_drones):
            # 1. Nominal Guidance
            # Note: For followers, target_vel is technically leader_vel,
            # but we assume 0 for simplicity here (PD control will catch up)
            target_vel = np.zeros(3)
            u_nom = guidance_sys[i].compute_u_nom(states[i].pos, states[i].vel, current_targets[i], target_vel)

            # 2. Sense Neighbors (CRITICAL STEP)
            local_obstacles = []
            if mission.static_obstacles:
                local_obstacles.extend(mission.static_obstacles)

            for j in range(num_drones):
                if i == j: continue

                # CRITICAL: Inter-agent radius must be LARGE to prevent high-speed crashes
                # dist = 2.0 ensures they start braking when 2m apart
                neighbor = Obstacle(
                    center=states[j].pos,
                    velocity=states[j].vel,
                    radius=2.0
                )
                local_obstacles.append(neighbor)

            # 3. Safety Filter
            u_safe = safety_filters[i].filter(u_nom, states[i].pos, states[i].vel, local_obstacles)
            safe_controls.append(u_safe)

            pos_history[i].append(states[i].pos)

        # --- C. STEP PHYSICS ---
        for i in range(num_drones):
            states[i] = drones[i].step(states[i], safe_controls[i])

    # --- D. ANIMATE ---
    animate_swarm(pos_history, mission, dt)


def animate_swarm(pos_history, mission, dt):
    pos_history = [np.array(h) for h in pos_history]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm']
    lines = []
    points = []

    # Initialize plots
    for i in range(mission.num_drones):
        ln, = ax.plot([], [], [], color=colors[i % 5], label=f'Drone {i}')
        pt, = ax.plot([], [], [], marker='o', color=colors[i % 5])
        lines.append(ln)
        points.append(pt)

    # Draw Obstacles
    if mission.static_obstacles:
        u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
        for obs in mission.static_obstacles:
            x = obs.radius * np.cos(u) * np.sin(v) + obs.center[0]
            y = obs.radius * np.sin(u) * np.sin(v) + obs.center[1]
            z = obs.radius * np.cos(v) + obs.center[2]
            ax.plot_wireframe(x, y, z, color="black", alpha=0.3)

    ax.set_xlim(-15, 15);
    ax.set_ylim(-15, 15);
    ax.set_zlim(-5, 5)
    ax.set_title(f"Swarm: {mission.name}")
    ax.legend()

    def update(frame):
        for i in range(mission.num_drones):
            # Trail length = 50 frames
            start = max(0, frame - 50)
            lines[i].set_data(pos_history[i][start:frame, 0], pos_history[i][start:frame, 1])
            lines[i].set_3d_properties(pos_history[i][start:frame, 2])

            points[i].set_data([pos_history[i][frame, 0]], [pos_history[i][frame, 1]])
            points[i].set_3d_properties([pos_history[i][frame, 2]])
        return lines + points

    anim = FuncAnimation(fig, update, frames=len(pos_history[0]), interval=50, blit=False)
    plt.show()


if __name__ == "__main__":
    run_swarm_simulation("crossover")
    #run_swarm_simulation("formation")