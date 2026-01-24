import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from constraints import Obstacle
from dynamics import State
from scenarios import Target, CircularTarget, StaticTarget


@dataclass
class SwarmScenario:
    name: str
    num_drones: int
    start_states: List[State]
    # If "Formation", targets are calculated relative to Leader.
    # If "Independent", targets are fixed points.
    type: str  # 'independent' or 'leader_follower'
    global_target: Optional[Target] = None # For leader
    formation_offsets: Optional[List[np.ndarray]] = None # For followers
    fixed_targets: Optional[List[np.ndarray]] = None # For independent
    static_obstacles: List[Obstacle] = None

def get_scenario_crossover():
    """ 3 Drones flying head-on through a choke point. """
    return SwarmScenario(
        name="Cross-Over",
        type="independent",
        num_drones=3,
        start_states=[
            State(pos=np.array([0.0, 0.0, 0.0]), vel=np.zeros(3)), # Center
            State(pos=np.array([0.0, 4.0, 0.0]), vel=np.zeros(3)), # Left
            State(pos=np.array([0.0, -4.0, 0.0]), vel=np.zeros(3)) # Right
        ],
        fixed_targets=[
            np.array([15.0, 0.0, 0.0]),  # Center -> Center
            np.array([15.0, -4.0, 0.0]), # Left -> Right (Cross!)
            np.array([15.0, 4.0, 0.0])   # Right -> Left (Cross!)
        ],
        static_obstacles=[
            # A "Pillar" in the middle they must also dodge
            Obstacle(np.array([7.5, 0.0, 0.0]), np.zeros(3), radius=3.0)
        ]
    )

def get_scenario_leader_follower():
    """ Drone 0 chases a moving target; Drones 1 & 2 maintain V-formation. """
    return SwarmScenario(
        name="V-Formation Chase",
        type="leader_follower",
        num_drones=5,
        start_states=[
            State(pos=np.array([0.0, 0.0, 0.0]), vel=np.zeros(3)), # Leader
            State(pos=np.array([-2.0, 2.0, 0.0]), vel=np.zeros(3)), # Wingman 1
            State(pos=np.array([-2.0, -2.0, 0.0]), vel=np.zeros(3)), # Wingman 2
            State(pos=np.array([-4.0, 4.0, 0.0]), vel=np.zeros(3)), # Wingman 3
            State(pos=np.array([-4.0, -4.0, 0.0]), vel=np.zeros(3)) # Wingman 4
        ],
        global_target=CircularTarget(radius=10.0, speed=0.5),
        formation_offsets=[
            np.array([0.0, 0.0, 0.0]),  # Leader (No offset)
            np.array([-2.0, 2.0, 0.0]), # Back-Left
            np.array([-2.0, -2.0, 0.0]), # Back-Right
            np.array([-4.0, 4.0, 0.0]),  # Back-Nack-Left
            np.array([-4.0, -4.0, 0.0])  # Back-Back-Right
        ],
        static_obstacles=[]
    )

SWARM_SCENARIOS = {
    "crossover": get_scenario_crossover,
    "formation": get_scenario_leader_follower
}