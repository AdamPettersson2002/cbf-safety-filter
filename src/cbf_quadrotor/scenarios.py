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
            # Offset slightly Y=0.1 to break symmetry
            Obstacle(np.array([5.0, 0.0, 0.0]), np.array([0., 0., 0.]), radius=1.0)
        ],
        duration=15.0
    )


def get_scenario_2_chase():
    """Chasing a moving target through a patrol."""
    obs_patrol = Obstacle(center=np.array([8.0, 0.0, 0.0]),velocity= np.array([0.0, 2.0, 0.0]), radius=1.0)

    return Scenario(
        name="Circular Chase",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=CircularTarget(radius=8.0, speed=0.5),
        obstacles=[obs_patrol],
        duration=50.0
    )


def get_scenario_3_clutter():
    """A field of random static obstacles."""
    # Generate 5 random obstacles
    obs_list = []
    np.random.seed(42)  # Fixed seed for reproducibility
    for _ in range(20):
        center = np.random.uniform(low=[-5, -5, -5], high=[5, 5, 5])
        obs_list.append(Obstacle(center, np.zeros(3), radius=0.8))

    return Scenario(
        name="Cluttered Field",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=StaticTarget(pos=np.array([12., -6., 0.])),
        obstacles=obs_list,
        duration=20.0
    )


# Dictionary for easy loading
SCENARIOS = {
    "head_on": get_scenario_1_head_on,
    "chase": get_scenario_2_chase,
    "clutter": get_scenario_3_clutter
}