import numpy as np
from dataclasses import dataclass, field
from typing import List, Protocol
from dynamics import State
from constraints import Obstacle


# --- 1. Abstracting the Target ---
# This allows you to swap a "Static Point" for a "Moving Circle" easily
class Target(Protocol):
    def update(self, time: float):
        """Returns pos, vel at given time"""
        ...


class StaticTarget:
    def __init__(self, pos: np.ndarray):
        self.pos = pos
        self.vel = np.zeros(3)

    def update(self, time: float):
        return self.pos, self.vel


class CircularTarget:
    def __init__(self, radius=10.0, speed=0.5, height=0.0):
        self.radius = radius
        self.speed = speed
        self.height = height

    def update(self, time: float):
        pos = np.array([
            self.radius * np.cos(self.speed * time),
            self.radius * np.sin(self.speed * time),
            self.height
        ])
        vel = np.array([
            -self.radius * self.speed * np.sin(self.speed * time),
            self.radius * self.speed * np.cos(self.speed * time),
            0.0
        ])
        return pos, vel


class LinearTarget:
    def __init__(self, start_pos: np.ndarray, vel: np.ndarray):
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)

    def update(self, time: float):
        pos = self.start_pos + self.vel * time
        return pos, self.vel


# --- 2. The Scenario Container ---
@dataclass
class Scenario:
    name: str
    start_state: State
    target: Target
    obstacles: List[Obstacle] = field(default_factory=list)
    duration: float = 10.0



# --- 3. The Scenario Definitions ---

def get_scenario_1_head_on():
    """Simple static obstacle blocking the path."""
    return Scenario(
        name="Head On Collision Test",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=StaticTarget(pos=np.array([10., 0., 0.])),
        obstacles=[
            Obstacle(np.array([5.0, 0.0, 0.0]), np.array([0.0, 0., 0.]), radius=1.0)
        ],
        duration=15.0
    )


def get_scenario_2_head_on():
    """Simple static obstacle blocking the path and moving toward us."""
    return Scenario(
        name="Head On Collision Test (Moving Obstacle)",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=StaticTarget(pos=np.array([10., 0., 0.])),
        obstacles=[
            Obstacle(np.array([5.0, 0.0, 0.0]), np.array([-1.0, 0., 0.]), radius=1.0)
        ],
        duration=15.0
    )


def get_scenario_3_chase():
    """Chasing a moving target through a patrol."""
    obs_patrol = Obstacle(center=np.array([8.0, 0.0, 0.0]),velocity= np.array([0.0, 2.0, 0.0]), radius=1.0)

    return Scenario(
        name="Circular Chase",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=CircularTarget(radius=8.0, speed=0.),
        obstacles=[obs_patrol],
        duration=50.0
    )


def get_scenario_4_blockers():
    """Target moves in a straight line; 5 circular obstacles move back/forth in y blocking the way."""
    # Drone starts at origin
    start = State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.]))

    # Target moves straight along +x
    target = LinearTarget(
        start_pos=np.array([12., 0., 0.]),
        vel=np.array([0.25, 0.0, 0.0])
    )

    # 5 obstacles placed along the corridor (x direction), oscillating in y due to your loop's bounce logic
    xs = [3.0, 6.0, 9.0, 12.0, 15.0]
    y0s = [-3.0, -1.5, 0.0, 1.5, 3.0]          # stagger initial positions
    speeds = [1.2, -1.0, 1.4, -1.1, 1.3]        # stagger directions/speeds
    radius = 1.0

    obstacles = []
    for x, y0, vy in zip(xs, y0s, speeds):
        obstacles.append(
            Obstacle(
                center=np.array([x, y0, 0.0]),
                velocity=np.array([0.0, vy, 0.0]),   # IMPORTANT: nonzero y-velocity triggers your motion + bounce
                radius=radius
            )
        )

    return Scenario(
        name="Straight Target + 5 Blockers (Back/Forth)",
        start_state=start,
        target=target,
        obstacles=obstacles,
        duration=40.0
    )



# Dictionary for easy loading
SCENARIOS = {
    "head_on": get_scenario_1_head_on,
    "head_on_2": get_scenario_2_head_on,
    "clutter": get_scenario_3_chase,
    "many_blockers": get_scenario_4_blockers,
}