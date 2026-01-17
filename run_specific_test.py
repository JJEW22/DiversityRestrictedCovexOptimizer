"""
Specific test case for p-MON attack where we expect the attacker to benefit.

Setup:
- 2 resources (items)
- n agents (default 10)
- n-1 agents have preferences [1, 0] (only want item 0)
- 1 attacker has preferences [1-1/n, 1/n] (slight preference for item 1)
- No initial constraints

In the initial Nash welfare allocation, everyone competes for item 0.
The attacker should be able to issue a proportionality constraint that 
forces them to get some of item 1, which no one else wants.
"""

import numpy as np
import argparse

from nash_welfare_optimizer import (
    compute_optimal_self_benefit_constraint,
    compute_optimal_harm_constraint,
    NashWelfareOptimizer,
)
from debug_output_helper import save_debug_xlsx


def run_specific_test(n_agents: int = 10, debug: bool = False, verbose: bool = False, debug_dir: str = "."):
    """
    Run the specific test case.
    
    Setup:
    - 2 resources
    - n agents
    - Agents 0 to n-2 have preferences [1, 0]
    - Agent n-1 (attacker) has preferences [1-1/n, 1/n]
    """
    n_resources = 2
    
    # Build utility matrix
    utilities = np.zeros((n_agents, n_resources))
    
    # Non-attackers: prefer only item 0
    for i in range(n_agents - 1):
        utilities[i, 0] = 1.0
        utilities[i, 1] = 0.0
    
    # Attacker (last agent): slight preference for item 1
    attacker_id = n_agents - 1
    utilities[attacker_id, 0] = 1.0 - 1.0/n_agents
    utilities[attacker_id, 1] = 1.0/n_agents
    
    # Supply: 1 unit of each resource
    supply = np.ones(n_resources)
    
    # Groups
    attacking_group = {attacker_id}
    victim_group = set(range(n_agents - 1))
    
    print("=" * 60)
    print("SPECIFIC TEST CASE")
    print("=" * 60)
    print(f"n_agents: {n_agents}")
    print(f"n_resources: {n_resources}")
    print(f"Attacker (Agent {attacker_id}): preferences = [{utilities[attacker_id, 0]:.4f}, {utilities[attacker_id, 1]:.4f}]")
    print(f"Others (Agents 0-{n_agents-2}): preferences = [1.0, 0.0]")
    print()
    
    # First, compute the initial allocation to see what happens
    print("Computing initial Nash welfare allocation...")
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    initial_result = optimizer.solve(verbose=verbose)
    
    print(f"Initial allocation:")
    for i in range(n_agents):
        role = "attacker" if i == attacker_id else "victim"
        print(f"  Agent {i} ({role}): {initial_result.allocation[i].round(6)} -> utility = {initial_result.agent_utilities[i]:.6f}")
    print(f"Initial Nash welfare: {initial_result.nash_welfare:.6f}")
    print()
    
    # Expected analysis:
    # Since all non-attackers only want item 0, and there's only 1 unit of item 0,
    # they have to share it. Item 1 is unwanted by them (utility 0).
    # The attacker wants mostly item 0 but also some item 1.
    # In Nash welfare, the attacker should get some of item 0 (competing with others)
    # and potentially all of item 1 (no competition).
    
    # Run p-MON attack
    print("Running p-MON (self-benefit) attack...")
    print("-" * 40)
    
    constraints, info = compute_optimal_self_benefit_constraint(
        utilities=utilities,
        attacking_group=attacking_group,
        victim_group=victim_group,
        supply=supply,
        verbose=verbose,
        debug=debug,
        timing=True
    )
    
    # Add constraint ratios to info for debug output
    if constraints:
        info['constraints_found'] = True
        info['constraint_ratios'] = {
            agent_id: constraint.get_ratios() 
            for agent_id, constraint in constraints.items()
        }
    else:
        info['constraints_found'] = False
        info['constraint_ratios'] = {}
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {info['status']}")
    print(f"p-MON (group): {info['p_mon_group']:.6f}")
    print(f"  (p-MON < 1 means attacker benefited)")
    print()
    
    print("Initial vs Final utilities:")
    for i in range(n_agents):
        role = "attacker" if i == attacker_id else "victim"
        init_u = info['initial_utilities'][i]
        final_u = info['final_utilities'][i]
        change_pct = (final_u / init_u - 1) * 100 if init_u > 1e-10 else 0
        print(f"  Agent {i} ({role}): {init_u:.6f} -> {final_u:.6f} ({change_pct:+.2f}%)")
    print()
    
    if constraints:
        print("Optimal constraint for attacker:")
        for agent_id, constraint in constraints.items():
            ratios = constraint.get_ratios()
            print(f"  Agent {agent_id}: ratios = {ratios.round(6)}")
            print(f"    (Must receive resources in ratio {ratios[0]:.4f} : {ratios[1]:.4f})")
    else:
        print("No beneficial constraint found.")
    print()
    
    print("Final allocation:")
    for i in range(n_agents):
        role = "attacker" if i == attacker_id else "victim"
        print(f"  Agent {i} ({role}): {info['final_allocation'][i].round(6)}")
    print()
    
    # Boundary check info
    print("Boundary check:")
    print(f"  Initial on boundary: {info.get('initial_on_boundary')}, dir_deriv = {info.get('initial_directional_deriv'):.6e}")
    print(f"  CP on boundary: {info.get('cp_on_boundary')}, dir_deriv = {info.get('cp_directional_deriv'):.6e}")
    print(f"  Final on boundary: {info.get('final_on_boundary')}, dir_deriv = {info.get('final_directional_deriv'):.6e}")
    print()
    
    # Save debug info
    if debug:
        filename = f"{debug_dir}/debug_specific_pmon_n{n_agents}.xlsx"
        save_debug_xlsx(
            filename=filename,
            info=info,
            attacking_group=attacking_group,
            victim_group=victim_group,
            utilities=utilities,
            n_agents=n_agents,
            n_resources=n_resources,
            run_id=0,
            attack_type="pmon"
        )
    
    return constraints, info


def main():
    parser = argparse.ArgumentParser(description='Run specific p-MON test case')
    parser.add_argument('--n', type=int, default=10, help='Number of agents (default: 10)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output and save xlsx')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory for output files')
    
    args = parser.parse_args()
    
    run_specific_test(
        n_agents=args.n,
        debug=args.debug,
        verbose=args.verbose,
        debug_dir=args.output_dir
    )


if __name__ == "__main__":
    main()