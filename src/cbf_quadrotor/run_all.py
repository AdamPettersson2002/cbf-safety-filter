import os
from scenarios import MissionFactory
from sim import run_simulation
from swarm_scenarios import SWARM_SCENARIOS
from swarm_sim import run_swarm_simulation


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Results will be saved to: {results_dir}")

    print("\n=== Running Single Drone Scenarios ===")
    mf = MissionFactory(scenarios={"head_on", "head_on_2", "chase", "many_blockers", "los_problem"})
    missions = mf.generate_missions()
    for m in missions:
        print(f"-> Running single scenario: {m.name}")
        run_simulation(m, animate=True, save_dir=results_dir)

    print("\n=== Running Swarm Scenarios ===")
    for scenario_name in SWARM_SCENARIOS.keys():
        print(f"-> Running swarm scenario: {scenario_name}")
        run_swarm_simulation(scenario_name, save_dir=results_dir)
        
    print(f"\nAll scenarios completed. Results saved in {results_dir}")

if __name__ == "__main__":
    main()
