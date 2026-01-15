import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
dt = 0.02              # 50 Hz
T = 20.0               # simtid (s)
steps = int(T / dt)

g = 9.81
tilt_max_deg = 25.0
tilt_max = np.deg2rad(tilt_max_deg)
a_xy_max = g * np.tan(tilt_max)   # "quadrotor-ish" horisontell accelbegränsning

v_max = 6.0            # m/s (valfritt för stabilitet)
drag = 0.2             # enkel linjär drag för att dämpa

# Controller gains (börja smått)
kp = 1.2
kd = 1.8

# ----------------------------
# No-fly zones: circles in XY
# zone = (center_xy, radius)
# ----------------------------
no_fly_zones = [
    (np.array([4.0, 3.0]), 1.0),
    (np.array([7.0, 6.0]), 1.2),
]

# Repulsion parameters
d_safe = 0.6   # "buffert" utanför zonkanten (m)
k_rep = 3.0    # styrka på repulsion

# ----------------------------
# Helper functions
# ----------------------------
def clamp_norm(vec, max_norm):
    n = np.linalg.norm(vec)
    if n <= 1e-9:
        return vec
    return vec if n <= max_norm else vec * (max_norm / n)

def repulsive_accel(p_xy):
    """
    Mjuk repulsion utanför no-fly zone.
    Vi vill hålla minst d_safe utanför zonkanten.
    """
    a = np.zeros(2)
    for c, r in no_fly_zones:
        dist_to_center = np.linalg.norm(p_xy - c)
        dist_to_boundary = dist_to_center - r

        # om vi är nära (inom d_safe), knuffa utåt
        if dist_to_boundary < d_safe:
            # riktning utåt
            if dist_to_center < 1e-6:
                direction = np.array([1.0, 0.0])
            else:
                direction = (p_xy - c) / dist_to_center

            # "mjuk" styrka som växer när vi närmar oss gränsen
            # (undviker oändligheter)
            x = max(dist_to_boundary, 1e-3)
            strength = k_rep * (1.0 / x - 1.0 / d_safe)
            strength = max(strength, 0.0)

            a += strength * direction
    return a

# ----------------------------
# Simulation state (2D + constant height)
# ----------------------------
p = np.array([0.0, 0.0])   # position xy
v = np.array([0.0, 0.0])   # velocity xy

goal = np.array([10.0, 8.0])

traj = np.zeros((steps, 2))

# ----------------------------
# Main loop
# ----------------------------
for k in range(steps):
    traj[k] = p

    # Nominal go-to-goal PD (v_goal = 0)
    a_goal = kp * (goal - p) - kd * v

    # No-fly repulsion
    a_avoid = repulsive_accel(p)

    # Combine
    a_cmd = a_goal + a_avoid

    # Tilt constraint: limit horizontal accel
    a_cmd = clamp_norm(a_cmd, a_xy_max)

    # Optional: speed limit (for stability / realism)
    if np.linalg.norm(v) > v_max:
        v = v * (v_max / np.linalg.norm(v))

    # Integrate with simple drag
    v = v + dt * (a_cmd - drag * v)
    p = p + dt * v

    # stop if reached
    if np.linalg.norm(goal - p) < 0.2 and np.linalg.norm(v) < 0.2:
        traj = traj[:k+1]
        break

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots()
ax.plot(traj[:, 0], traj[:, 1], label="trajectory")
ax.scatter([0], [0], marker="o", label="start")
ax.scatter([goal[0]], [goal[1]], marker="x", label="goal")

# draw no-fly zones and safety buffer
for c, r in no_fly_zones:
    zone = plt.Circle(c, r, fill=False, linewidth=2)
    buf = plt.Circle(c, r + d_safe, fill=False, linestyle="--")
    ax.add_patch(zone)
    ax.add_patch(buf)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()
ax.grid(True)
plt.show()
