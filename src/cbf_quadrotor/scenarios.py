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


class MissionFactory:
    def __init__(self,
                 scenarios=None,
                 **kwargs):
        if scenarios is None:
            self.scenarios = {"head_on", "head_on_2", "chase", "many_blockers", "los_problem"}
        else:
            self.scenarios = scenarios

        # Default configurations for each scenario
        self.configs = {
            "head_on": {
                "start_pos": [0., 0., 0.], "start_vel": [0., 0., 0.],
                "target_pos": [10., 0., 0.],
                "obs_pos": [5.0, 0.0, 0.0], "obs_vel": [0.0, 0., 0.], "obs_radius": 1.0,
                "duration": 20.0
            },
            "head_on_2": {
                "start_pos": [0., 0., 0.], "start_vel": [0., 0., 0.],
                "target_pos": [10., 0., 0.],
                "obs_pos": [5.0, 0.0, 0.0], "obs_vel": [-1.0, 0., 0.], "obs_radius": 1.0,
                "duration": 20.0
            },
            "chase": {
                "start_pos": [0., 0., 0.], "start_vel": [0., 0., 0.],
                "target_radius": 8.0, "target_speed": 0.7,
                "duration": 30.0
            },
            "many_blockers": {
                "start_pos": [0., 0., 0.], "start_vel": [0., 0., 0.],
                "target_start_pos": [12., 0., 0.], "target_vel": [2.0, 0.0, 0.0],
                "duration": 40.0
            },
            "los_problem": {
                "start_pos": [0., 0., 0.], "start_vel": [0., 0., 0.],
                "target_start_pos": [40.0, -20.0, 0.0], "target_vel": [0.0, 4.0, 0.0],
                "duration": 20.0
            }
        }

        # Override defaults with any provided kwargs
        for scenario_name, config_overrides in kwargs.items():
            if scenario_name in self.configs:
                self.configs[scenario_name].update(config_overrides)

    def generate_missions(self) -> List[Scenario]:
        missions = []
        try:
            for scenario in list(self.scenarios):
                method_name = f"_create_{scenario}"
                if hasattr(self, method_name):
                    missions.append(getattr(self, method_name)())
                else:
                    print(f"Warning: Unknown scenario {scenario}")
            return missions
        except ValueError as e:
            print(f"Error: {e}")
            return []

    def _create_head_on(self) -> Scenario:
        cfg = self.configs["head_on"]
        return Scenario(
            name="Head On Collision",
            start_state=State(pos=np.array(cfg["start_pos"]), vel=np.array(cfg["start_vel"])),
            target=StaticTarget(pos=np.array(cfg["target_pos"])),
            obstacles=[Obstacle(np.array(cfg["obs_pos"]), np.array(cfg["obs_vel"]), radius=cfg["obs_radius"])],
            duration=cfg["duration"])

    def _create_head_on_2(self) -> Scenario:
        cfg = self.configs["head_on_2"]
        return Scenario(
            name="Head On Collision With Moving Obstacle",
            start_state=State(pos=np.array(cfg["start_pos"]), vel=np.array(cfg["start_vel"])),
            target=StaticTarget(pos=np.array(cfg["target_pos"])),
            obstacles=[Obstacle(np.array(cfg["obs_pos"]), np.array(cfg["obs_vel"]), radius=cfg["obs_radius"])],
            duration=cfg["duration"])

    def _create_chase(self) -> Scenario:
        cfg = self.configs["chase"]
        obs_patrol1 = PatrollingObstacle(waypoint_a=[8.0, -4.0, 0.0], waypoint_b=[8.0, 4.0, 0.0], speed=1.0, radius=1.0)
        obs_patrol2 = PatrollingObstacle(waypoint_a=[4.0, 6.0, 0.0], waypoint_b=[4.0, 2.0, 0.0], speed=1.0, radius=1.0)
        obs_patrol3 = PatrollingObstacle(waypoint_a=[-8.0, 0.0, 0.0], waypoint_b=[0.5, 8.0, 0.0], speed=1.0, radius=1.0)
        return Scenario(
            name="Circular Chase with Patrols",
            start_state=State(pos=np.array(cfg["start_pos"]), vel=np.array(cfg["start_vel"])),
            target=CircularTarget(radius=cfg["target_radius"], speed=cfg["target_speed"]),
            obstacles=[obs_patrol1, obs_patrol2, obs_patrol3],
            duration=cfg["duration"]
        )

    def _create_many_blockers(self) -> Scenario:
        cfg = self.configs["many_blockers"]
        target = LinearTarget(start_pos=np.array(cfg["target_start_pos"]), vel=np.array(cfg["target_vel"]))

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
            start_state=State(pos=np.array(cfg["start_pos"]), vel=np.array(cfg["start_vel"])),
            target=target,
            obstacles=obstacles,
            duration=cfg["duration"])

    def _create_los_problem(self) -> Scenario:
        cfg = self.configs["los_problem"]
        target = LinearTarget(start_pos=np.array(cfg["target_start_pos"]), vel=np.array(cfg["target_vel"]))
        obstacles = [
            Obstacle(center=np.array([20.0, -10.0, 0.0]), velocity=np.zeros(3), radius=4.0),
            Obstacle(center=np.array([20.0, 0.0, 0.0]), velocity=np.zeros(3), radius=4.0),
            Obstacle(center=np.array([20.0, 10.0, 0.0]), velocity=np.zeros(3), radius=4.0),
        ]

        return Scenario(
            name="LOS Problem",
            start_state=State(pos=np.array(cfg["start_pos"]), vel=np.array(cfg["start_vel"])),
            target=target,
            obstacles=obstacles,
            duration=cfg["duration"])
