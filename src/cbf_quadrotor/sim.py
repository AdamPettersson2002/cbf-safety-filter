"""
Main simulation loop and visualization for CBF-protected quadrotor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple
import os
from datetime import datetime

from cbf_quadrotor import dynamics, controllers, cbf, environment


def save_simulation_data(trajectory, controls_nom, controls_safe,
                         safety_margins, violations, params, results_dir):
    """
    Save simulation data to a numpy file.

    Args:
        trajectory: state trajectory
        controls_nom: nominal controls
        controls_safe: safe controls
        safety_margins: safety margins over time
        violations: violation flags
        params: simulation parameters
        results_dir: directory to save data
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cbf_data_{timestamp}.npz"
    filepath = os.path.join(results_dir, filename)

    np.savez(filepath,
             trajectory=trajectory,
             controls_nom=controls_nom,
             controls_safe=controls_safe,
             safety_margins=safety_margins,
             violations=violations,
             params=params)

    print(f"Data saved to: {filepath}")


def run_simulation():
    """Run the main simulation and generate plots."""

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Setup scenario
    scenario = environment.create_default_scenario()
    params = environment.get_simulation_params()

    x = scenario['start'].copy()
    goal = scenario['goal']
    zones = scenario['zones']

    # Simulation parameters
    dt = params['dt']
    T = params['T']
    n_steps = int(T / dt)

    # Storage for logging
    trajectory = np.zeros((n_steps, 4))
    controls_nom = np.zeros((n_steps, 2))
    controls_safe = np.zeros((n_steps, 2))
    safety_margins = np.zeros(n_steps)
    violations = np.zeros(n_steps, dtype=bool)

    # Main simulation loop
    print("Running simulation...")
    for i in range(n_steps):
        # Store current state
        trajectory[i] = x

        # Check safety
        is_safe, margin = cbf.check_safety(x, zones)
        safety_margins[i] = margin
        violations[i] = not is_safe

        # Nominal controller
        u_nom = controllers.pd_controller(x, goal, params['kp'], params['kd'])
        controls_nom[i] = u_nom

        # CBF safety filter
        u_safe = cbf.cbf_filter(x, u_nom, zones, params)
        controls_safe[i] = u_safe

        # Check if goal reached
        if controllers.is_goal_reached(x, goal, params['goal_eps']):
            print(f"Goal reached at t={i*dt:.2f}s")
            trajectory = trajectory[:i+1]
            controls_nom = controls_nom[:i+1]
            controls_safe = controls_safe[:i+1]
            safety_margins = safety_margins[:i+1]
            violations = violations[:i+1]
            break

        # Step dynamics
        x = dynamics.step(x, u_safe, dt, params.get('v_max'))

    # Results summary
    print("\n=== Simulation Results ===")
    print(f"Final position: ({x[0]:.2f}, {x[1]:.2f})")
    print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"Distance to goal: {np.linalg.norm(x[:2] - goal):.3f} m")
    print(f"Minimum safety margin: {np.min(safety_margins):.3f} m")
    print(f"Violations detected: {np.sum(violations)}")

    # Save simulation data
    save_simulation_data(trajectory, controls_nom, controls_safe,
                        safety_margins, violations, params, results_dir)

    # Visualization
    plot_results(trajectory, goal, zones, safety_margins,
                 controls_nom, controls_safe, scenario['d_safe'], params['dt'], results_dir)


def plot_results(trajectory: np.ndarray, goal: np.ndarray, zones: List,
                 safety_margins: np.ndarray, controls_nom: np.ndarray,
                 controls_safe: np.ndarray, d_safe: float, dt: float, results_dir: str):
    """
    Create visualization plots.

    Args:
        trajectory: (N, 4) array of states
        goal: (2,) goal position
        zones: list of no-fly zones
        safety_margins: (N,) array of minimum margins
        controls_nom: (N, 2) nominal controls
        controls_safe: (N, 2) safe controls
        d_safe: safety margin for visualization
        dt: timestep
        results_dir: directory to save plots
    """
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Trajectory and zones
    ax1 = plt.subplot(2, 2, 1)

    # Plot trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')

    # Plot no-fly zones
    for zone in zones:
        center = zone['center']
        radius = zone['radius']

        # Inner circle (actual no-fly zone)
        circle_inner = Circle(center, radius - d_safe, color='red', alpha=0.3)
        ax1.add_patch(circle_inner)

        # Outer circle (with safety margin)
        circle_outer = Circle(center, radius, color='orange', fill=False,
                             linestyle='--', linewidth=2)
        ax1.add_patch(circle_outer)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Trajectory with No-Fly Zones')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot 2: Safety margin over time
    ax2 = plt.subplot(2, 2, 2)
    time = np.arange(len(safety_margins)) * dt
    ax2.plot(time, safety_margins, 'b-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', label='Safety boundary')
    ax2.fill_between(time, safety_margins, 0, where=(safety_margins < 0),
                     color='red', alpha=0.3, label='Violations')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Safety Margin (m)')
    ax2.set_title('Safety Margin Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control inputs
    ax3 = plt.subplot(2, 2, 3)
    time = np.arange(len(controls_nom)) * dt
    ax3.plot(time, controls_nom[:, 0], 'r--', label='u_nom_x', alpha=0.7)
    ax3.plot(time, controls_nom[:, 1], 'b--', label='u_nom_y', alpha=0.7)
    ax3.plot(time, controls_safe[:, 0], 'r-', label='u_safe_x', linewidth=2)
    ax3.plot(time, controls_safe[:, 1], 'b-', label='u_safe_y', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Control Inputs (Nominal vs Safe)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Velocity profile
    ax4 = plt.subplot(2, 2, 4)
    time = np.arange(len(trajectory)) * dt
    speed = np.linalg.norm(trajectory[:, 2:], axis=1)
    ax4.plot(time, trajectory[:, 2], 'r-', label='vx')
    ax4.plot(time, trajectory[:, 3], 'b-', label='vy')
    ax4.plot(time, speed, 'k-', linewidth=2, label='||v||')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cbf_simulation_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    plt.show()


if __name__ == '__main__':
    run_simulation()