import numpy as np
from dataclasses import dataclass, field
from typing import List, Protocol
from dynamics import State
from constraints import Obstacle, PatrollingObstacle


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


# --- Different Scenarios ---
@dataclass
class Scenario:
    name: str
    start_state: State
    target: Target
    obstacles: List[Obstacle] = field(default_factory=list)
    duration: float = 10.0


def get_scenario_1_head_on():
    """Simple static obstacle blocking the path."""
    return Scenario(
        name="Head On Collision",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=StaticTarget(pos=np.array([10., 0., 0.])),
        obstacles=[Obstacle(np.array([5.0, 0.0, 0.0]), np.array([0.0, 0., 0.]), radius=1.0)],
        duration=20.0
    )


def get_scenario_2_head_on():
    """Simple static obstacle blocking the path and moving toward the drone."""
    return Scenario(
        name="Head On Collision With Moving Obstacle)",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=StaticTarget(pos=np.array([10., 0., 0.])),
        obstacles=[Obstacle(np.array([5.0, 0.0, 0.0]), np.array([-1.0, 0., 0.]), radius=1.0)],
        duration=20.0
    )


def get_scenario_3_chase():
    """Chasing a moving target in a circle through moving obstacles."""
    obs_patrol1 = PatrollingObstacle(waypoint_a=[8.0, -4.0, 0.0],waypoint_b=[8.0, 4.0, 0.0], speed=1.0, radius=1.0)
    obs_patrol2 = PatrollingObstacle(waypoint_a=[4.0, 6.0, 0.0], waypoint_b=[4.0, 2.0, 0.0], speed=1.0, radius=1.0)
    obs_patrol3 = PatrollingObstacle(waypoint_a=[-8.0, 0.0, 0.0],waypoint_b=[0.5, 8.0, 0.0], speed=1.0, radius=1.0)
    return Scenario(
        name="Circular Chase with Patrols",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=CircularTarget(radius=8.0, speed=0.7),
        obstacles=[obs_patrol1, obs_patrol2, obs_patrol3],
        duration=30.0
    )


def get_scenario_4_blockers():
    """Target moves in a straight line while patrols move back/forth and block the path."""
    start = State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.]))
    target = LinearTarget(start_pos=np.array([12., 0., 0.]), vel=np.array([2, 0.0, 0.0]))

    xs = [3.0, 6.0, 9.0, 12.0, 15.0]
    y0s = [-3.0, -1.5, 0.0, 1.5, 3.0]
    speeds = [1.2, -1.0, 1.4, -1.1, 1.3]
    radius = 1.0

    obstacles = []
    for x, y0, vy in zip(xs, y0s, speeds):
        obstacles.append(
            Obstacle(
                center=np.array([x, y0, 0.0]),
                velocity=np.array([0.0, vy, 0.0]),
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


def get_scenario_5_los():
    """
    The target moves in a straight line far away. A wall of static pillars periodically blocks the LOS.
    Crucial for testing what happens when the drone 'loses' the target when using sensor modeling.
    """
    target = LinearTarget(start_pos=np.array([40.0, -20.0, 0.0]), vel=np.array([0.0, 4.0, 0.0]))
    obstacles = []

    # Pillar 1 (Low Y)
    obstacles.append(Obstacle(center=np.array([20.0, -10.0, 0.0]), velocity=np.zeros(3), radius=4.0))

    # Pillar 2 (Center)
    obstacles.append(Obstacle(center=np.array([20.0, 0.0, 0.0]), velocity=np.zeros(3), radius=4.0))

    # Pillar 3 (High Y)
    obstacles.append(Obstacle(center=np.array([20.0, 10.0, 0.0]), velocity=np.zeros(3), radius=4.0
    ))

    return Scenario(
        name="LOS Problem",
        start_state=State(pos=np.array([0., 0., 0.]), vel=np.array([0., 0., 0.])),
        target=target,
        obstacles=obstacles,
        duration=20.0
    )


SCENARIOS = {
    "head_on": get_scenario_1_head_on,
    "head_on_2": get_scenario_2_head_on,
    "chase": get_scenario_3_chase,
    "many_blockers": get_scenario_4_blockers,
    "los_problem": get_scenario_5_los,
}
