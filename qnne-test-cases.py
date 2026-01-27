"""
Test script for q-NEE worst-case computation with known analytical results.

This script tests the projected gradient descent method against cases where
we know the analytical worst-case q-NEE value.
"""

import numpy as np
from typing import Dict, Set, Tuple, List
import sys

from nash_welfare_optimizer import (
    NashWelfareOptimizer,
    solve_optimal_harm_constraint_pgd,
    compute_optimal_self_benefit_constraint,
)


def run_test_case(
    name: str,
    utilities: np.ndarray,
    attacking_group: Set[int],
    defending_group: Set[int],
    expected_q_nee_worst: float = None,
    supply: np.ndarray = None,
    verbose: bool = True,
    debug: bool = False,
    pgd_max_iterations: int = 100,
    pgd_step_size: float = 0.1,
) -> Dict:
    """
    Run a single test case and compare against expected results.
    
    Args:
        name: Name of the test case
        utilities: Utility matrix (n_agents, n_resources)
        attacking_group: Set of attacker indices
        defending_group: Set of defender indices
        expected_q_nee_worst: Expected worst-case q-NEE (if known)
        supply: Supply vector (defaults to n_agents per resource)
        verbose: Print detailed output
        debug: Print debug output from PGD
        pgd_max_iterations: Max iterations for PGD
        pgd_step_size: Step size for PGD
        
    Returns:
        Dict with test results
    """
    n_agents, n_resources = utilities.shape
    supply = supply if supply is not None else np.ones(n_resources) * n_agents
    
    print("=" * 70)
    print(f"TEST CASE: {name}")
    print("=" * 70)
    print(f"n_agents: {n_agents}, n_resources: {n_resources}")
    print(f"attacking_group: {attacking_group}")
    print(f"defending_group: {defending_group}")
    print(f"supply: {supply}")
    print()
    print("Utilities:")
    for i in range(n_agents):
        role = "ATK" if i in attacking_group else ("DEF" if i in defending_group else "OTH")
        print(f"  Agent {i} ({role}): {utilities[i]}")
    print()
    
    # Step 1: Compute initial Nash welfare allocation
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    initial_result = optimizer.solve(verbose=False)
    V_initial = initial_result.agent_utilities
    
    print("Initial Nash Welfare Allocation:")
    print(f"  Nash welfare: {initial_result.nash_welfare:.6f}")
    print(f"  Utilities: {V_initial.round(6)}")
    print(f"  Allocation:")
    for i in range(n_agents):
        role = "ATK" if i in attacking_group else ("DEF" if i in defending_group else "OTH")
        print(f"    Agent {i} ({role}): {initial_result.allocation[i].round(6)}")
    print()
    
    # Step 2: Compute worst-case q-NEE using PGD
    print("Running Projected Gradient Descent for worst-case q-NEE...")
    constraints, info = solve_optimal_harm_constraint_pgd(
        utilities=utilities,
        attacking_group=attacking_group,
        defending_group=defending_group,
        supply=supply,
        max_iterations=pgd_max_iterations,
        step_size=pgd_step_size,
        verbose=verbose,
        debug=debug,
    )
    
    print()
    print("PGD Results:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Initial defender welfare: {info['initial_defender_welfare']:.6e}")
    print(f"  Final defender welfare: {info['final_defender_welfare']:.6e}")
    print(f"  Welfare reduction: {info['initial_defender_welfare'] / max(info['final_defender_welfare'], 1e-20):.4f}x")
    print()
    print(f"  q-NEE (worst) for defending group: {info['q_nee_group']:.6f}")
    print(f"  Individual q-NEE (worst):")
    for d in sorted(defending_group):
        print(f"    Agent {d}: {info['q_nee_individual'].get(d, float('nan')):.6f}")
    print()
    print(f"  Final utilities: {info['final_utilities'].round(6)}")
    print(f"  Final allocation:")
    for i in range(n_agents):
        role = "ATK" if i in attacking_group else ("DEF" if i in defending_group else "OTH")
        print(f"    Agent {i} ({role}): {info['final_allocation'][i].round(6)}")
    
    if constraints:
        print()
        print("  Optimal constraint (attacker allocation direction):")
        for a in sorted(attacking_group):
            if a in constraints:
                print(f"    Agent {a}: {constraints[a].get_ratios().round(6)}")
    
    # Step 3: Re-solve Nash welfare with the constraint applied
    print()
    print("-" * 50)
    print("Re-solving Nash Welfare with PGD constraint applied...")
    print("-" * 50)
    
    if constraints:
        resolve_optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
        for agent_id, constraint in constraints.items():
            resolve_optimizer.add_agent_constraint(agent_id, constraint)
        
        resolved_result = resolve_optimizer.solve(verbose=False)
        V_resolved = resolved_result.agent_utilities
        
        print(f"  Nash welfare (resolved): {resolved_result.nash_welfare:.6f}")
        print(f"  Utilities (resolved): {V_resolved.round(6)}")
        print(f"  Allocation (resolved):")
        for i in range(n_agents):
            role = "ATK" if i in attacking_group else ("DEF" if i in defending_group else "OTH")
            print(f"    Agent {i} ({role}): {resolved_result.allocation[i].round(6)}")
        
        # Compute q-NEE from resolved utilities
        q_nee_resolved_individual = {}
        for d in defending_group:
            if V_initial[d] > 1e-10:
                q_nee_resolved_individual[d] = V_resolved[d] / V_initial[d]
            else:
                q_nee_resolved_individual[d] = float('inf') if V_resolved[d] > 1e-10 else 1.0
        
        # Geometric mean for group q-NEE
        if defending_group:
            k = len(defending_group)
            log_sum = sum(np.log(max(q_nee_resolved_individual[d], 1e-20)) for d in defending_group)
            q_nee_resolved_group = np.exp(log_sum / k)
        else:
            q_nee_resolved_group = 1.0
        
        print()
        print(f"  q-NEE (resolved) for defending group: {q_nee_resolved_group:.6f}")
        print(f"  Individual q-NEE (resolved):")
        for d in sorted(defending_group):
            print(f"    Agent {d}: {q_nee_resolved_individual.get(d, float('nan')):.6f}")
        
        # Compare CP vs Resolved
        print()
        print("  Comparison (CP vs Resolved):")
        print(f"    q-NEE group - CP: {info['q_nee_group']:.6f}, Resolved: {q_nee_resolved_group:.6f}")
        cp_resolved_diff = abs(info['q_nee_group'] - q_nee_resolved_group)
        print(f"    Difference: {cp_resolved_diff:.6f}")
    else:
        print("  No constraints found - skipping re-solve")
        q_nee_resolved_group = info['q_nee_group']
        q_nee_resolved_individual = info['q_nee_individual']
        V_resolved = info['final_utilities']
    
    # Step 4: Compare with expected result
    print()
    if expected_q_nee_worst is not None:
        actual = q_nee_resolved_group  # Use resolved value for comparison
        error = abs(actual - expected_q_nee_worst)
        rel_error = error / max(abs(expected_q_nee_worst), 1e-10)
        passed = rel_error < 0.01  # 1% tolerance
        
        status = "PASS" if passed else "FAIL"
        print(f"EXPECTED q-NEE (worst): {expected_q_nee_worst:.6f}")
        print(f"ACTUAL q-NEE (worst, resolved):   {actual:.6f}")
        print(f"Error: {error:.6f} (relative: {rel_error:.4f})")
        print(f"Status: {status}")
    else:
        print("No expected value provided - results shown above for manual verification")
        passed = None
    
    print()
    
    return {
        'name': name,
        'passed': passed,
        'expected': expected_q_nee_worst,
        'actual_cp': info['q_nee_group'],
        'actual_resolved': q_nee_resolved_group,
        'info': info,
        'constraints': constraints,
        'resolved_utilities': V_resolved,
        'q_nee_resolved_individual': q_nee_resolved_individual,
    }


def test_case_1_homogeneous_defenders(verbose=False, debug=False, pgd_max_iterations=200, pgd_step_size=0.1):
    """
    Test case 1: 10 agents, 2 items
    - 9 defenders with preferences [1, 0]
    - 1 attacker with preferences [1 - 1/10, 1/10] = [0.9, 0.1]
    
    Expected: TBD (will compute analytically)
    """
    n_agents = 10
    n_resources = 2
    
    utilities = np.zeros((n_agents, n_resources))
    
    # Defenders (agents 0-8): all prefer item 0
    for i in range(9):
        utilities[i] = [1.0, 0.0]
    
    # Attacker (agent 9): prefers item 0 slightly, but also values item 1
    utilities[9] = [1.0 - 1.0/10, 1.0/10]  # [0.9, 0.1]
    
    attacking_group = {9}
    defending_group = set(range(9))
    
    return run_test_case(
        name="Homogeneous Defenders (10 agents, 2 items)",
        utilities=utilities,
        attacking_group=attacking_group,
        defending_group=defending_group,
        expected_q_nee_worst=None,  # Will fill in once we know
        verbose=verbose,
        debug=debug,
        pgd_max_iterations=pgd_max_iterations,
        pgd_step_size=pgd_step_size,
    )


def test_case_2_unique_preferences_9_defenders(verbose=False, debug=False, pgd_max_iterations=200, pgd_step_size=0.1):
    """
    Test case 2: 10 agents, 10 items
    - Each agent i has preference [0, ..., 1, ..., 0] (1 in position i)
    - Attacker: agent 9
    - Defenders: agents 0-8
    """
    n_agents = 10
    n_resources = 10
    
    utilities = np.zeros((n_agents, n_resources))
    
    # Each agent only values their own item
    for i in range(n_agents):
        utilities[i, i] = 1.0
    
    attacking_group = {9}
    defending_group = set(range(9))
    
    return run_test_case(
        name="Unique Preferences, 9 Defenders (10 agents, 10 items)",
        utilities=utilities,
        attacking_group=attacking_group,
        defending_group=defending_group,
        expected_q_nee_worst=None,
        verbose=verbose,
        debug=debug,
        pgd_max_iterations=pgd_max_iterations,
        pgd_step_size=pgd_step_size,
    )


def test_case_3_unique_preferences_5_defenders(verbose=False, debug=False, pgd_max_iterations=200, pgd_step_size=0.1):
    """
    Test case 3: 10 agents, 10 items
    - Each agent i has preference [0, ..., 1, ..., 0] (1 in position i)
    - Attacker: agent 9
    - Defenders: agents 0-4 (first 5 only)
    """
    n_agents = 10
    n_resources = 10
    
    utilities = np.zeros((n_agents, n_resources))
    
    # Each agent only values their own item
    for i in range(n_agents):
        utilities[i, i] = 1.0
    
    attacking_group = {9}
    defending_group = set(range(5))  # First 5 agents
    
    return run_test_case(
        name="Unique Preferences, 5 Defenders (10 agents, 10 items)",
        utilities=utilities,
        attacking_group=attacking_group,
        defending_group=defending_group,
        expected_q_nee_worst=None,
        verbose=verbose,
        debug=debug,
        pgd_max_iterations=pgd_max_iterations,
        pgd_step_size=pgd_step_size,
    )


def test_case_4_unique_preferences_3_attackers_5_defenders(verbose=False, debug=False, pgd_max_iterations=200, pgd_step_size=0.1):
    """
    Test case 4: 10 agents, 10 items
    - Each agent i has preference [0, ..., 1, ..., 0] (1 in position i)
    - Attackers: agents 7, 8, 9 (last 3)
    - Defenders: agents 0-4 (first 5)
    """
    n_agents = 10
    n_resources = 10
    
    utilities = np.zeros((n_agents, n_resources))
    
    # Each agent only values their own item
    for i in range(n_agents):
        utilities[i, i] = 1.0
    
    attacking_group = {7, 8, 9}  # Last 3 agents
    defending_group = set(range(5))  # First 5 agents
    
    return run_test_case(
        name="Unique Preferences, 3 Attackers vs 5 Defenders (10 agents, 10 items)",
        utilities=utilities,
        attacking_group=attacking_group,
        defending_group=defending_group,
        expected_q_nee_worst=None,
        verbose=verbose,
        debug=debug,
        pgd_max_iterations=pgd_max_iterations,
        pgd_step_size=pgd_step_size,
    )


# Add more test cases here as needed
TEST_CASES = [
    test_case_1_homogeneous_defenders,
    test_case_2_unique_preferences_9_defenders,
    test_case_3_unique_preferences_5_defenders,
    test_case_4_unique_preferences_3_attackers_5_defenders,
]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test q-NEE worst-case computation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output from PGD")
    parser.add_argument("--test", type=int, default=None, help="Run specific test case (1-indexed)")
    parser.add_argument("--pgd-max-iter", type=int, default=200, dest="pgd_max_iterations",
                        help="Max iterations for PGD")
    parser.add_argument("--pgd-step-size", type=float, default=0.1, dest="pgd_step_size",
                        help="Step size for PGD")
    
    args = parser.parse_args()
    
    # Common kwargs to pass to test functions
    test_kwargs = {
        'verbose': args.verbose,
        'debug': args.debug,
        'pgd_max_iterations': args.pgd_max_iterations,
        'pgd_step_size': args.pgd_step_size,
    }
    
    if args.test is not None:
        # Run specific test
        if args.test < 1 or args.test > len(TEST_CASES):
            print(f"Invalid test number. Available: 1-{len(TEST_CASES)}")
            sys.exit(1)
        test_func = TEST_CASES[args.test - 1]
        result = test_func(**test_kwargs)
    else:
        # Run all tests
        results = []
        for i, test_func in enumerate(TEST_CASES):
            print(f"\n{'#' * 70}")
            print(f"# Running test {i+1}/{len(TEST_CASES)}")
            print(f"{'#' * 70}\n")
            result = test_func(**test_kwargs)
            results.append(result)
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        n_passed = sum(1 for r in results if r['passed'] is True)
        n_failed = sum(1 for r in results if r['passed'] is False)
        n_no_expected = sum(1 for r in results if r['passed'] is None)
        
        for r in results:
            if r['passed'] is True:
                status = "PASS"
            elif r['passed'] is False:
                status = "FAIL"
            else:
                status = "NO EXPECTED VALUE"
            print(f"  {r['name']}: {status}")
        
        print()
        print(f"Passed: {n_passed}, Failed: {n_failed}, No expected value: {n_no_expected}")


if __name__ == "__main__":
    main()