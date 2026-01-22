"""
Compare the boundary points (x*) produced by binary search vs integral method.

This script runs both methods on the same test cases and compares:
- The x* boundary points
- The directional derivatives at those points
- The number of iterations/Nash solves
- The time taken
"""

import numpy as np
import time
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass

# Import from nash_welfare_optimizer
from nash_welfare_optimizer import (
    DirectionalDerivativeOracle,
    NashWelfareOptimizer,
)


@dataclass
class BoundaryResult:
    """Result from finding a boundary point."""
    x_star: np.ndarray  # The boundary point (flattened)
    t_final: float  # Final scale factor (if applicable)
    D_final: float  # Directional derivative at boundary
    n_nash_solves: int  # Number of Nash solves
    time_seconds: float  # Time taken
    method: str  # 'binary' or 'integral'
    converged: bool  # Whether it converged properly


def generate_random_utilities(n_agents: int, n_resources: int, seed: int) -> np.ndarray:
    """Generate random utility matrix using rod-cutting approach."""
    np.random.seed(seed)
    utilities = np.zeros((n_agents, n_resources))
    for i in range(n_agents):
        cuts = np.sort(np.random.uniform(0, 1, n_resources - 1))
        cuts = np.concatenate([[0], cuts, [1]])
        utilities[i] = np.diff(cuts)
    return utilities


def find_violating_point(n_agents: int, n_resources: int, utilities: np.ndarray,
                         attacking_group: Set[int], supply: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Find a point that violates the directional derivative constraint.
    
    Returns:
        x_full: The violating allocation matrix
        V: Utilities at violating point
        D_0: Directional derivative at violating point (should be negative)
    """
    # Start with Nash welfare solution
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    result = optimizer.solve(verbose=False)
    x_nash = result.allocation
    
    # Create oracle to check/find violations
    oracle = DirectionalDerivativeOracle(
        n_agents, n_resources, utilities, attacking_group, supply,
        verbose=False, track_timing=False, use_integral_method=False
    )
    
    # Scale up attacker allocation to create a violation
    x_full = x_nash.copy()
    
    for scale in [1.5, 2.0, 3.0, 5.0, 10.0]:
        # Scale attacker allocation
        x_test = x_nash.copy()
        for a in attacking_group:
            x_test[a] = x_nash[a] * scale
        
        # Recompute non-attacker allocation with remaining supply
        remaining_supply = supply.copy()
        for a in attacking_group:
            remaining_supply -= x_test[a]
        
        if np.any(remaining_supply < 0):
            continue
        
        # Solve Nash for non-attackers
        non_attacker_alloc = oracle._solve_nash_for_non_attackers(remaining_supply)
        if non_attacker_alloc is None:
            continue
        
        for i in oracle.non_attacking_group:
            x_test[i] = non_attacker_alloc[i]
        
        # Check if this violates
        V_test = oracle._compute_utilities_from_allocation(x_test)
        
        if np.any(V_test <= 1e-10):
            # Zero utility - definitely violated
            return x_test, V_test, -1.0
        
        direction, valid = oracle._compute_direction(x_test, V_test, backward=False)
        if valid:
            D = oracle._compute_directional_derivative_with_direction(x_test, V_test, direction)
            if D < -1e-8:
                return x_test, V_test, D
    
    # If we couldn't find a violation, return the Nash solution (not violated)
    V_nash = oracle._compute_utilities_from_allocation(x_nash)
    return x_nash, V_nash, 0.0


def run_binary_search_method(x_full: np.ndarray, V: np.ndarray, D_0: float,
                              oracle: DirectionalDerivativeOracle) -> BoundaryResult:
    """Run binary search method and return the boundary point."""
    x_free = x_full.flatten()
    
    start_time = time.perf_counter()
    
    # Reset timing stats to count Nash solves
    oracle.reset_timing_stats()
    oracle.track_timing = True
    
    # Call the binary search method directly
    normal, rhs = oracle._find_separating_hyperplane(x_free, x_full, V)
    
    elapsed = time.perf_counter() - start_time
    
    # Get the boundary point
    x_star = oracle.last_boundary_point
    
    # Compute directional derivative at boundary
    x_star_full = x_star.reshape((oracle.n_agents, oracle.n_resources))
    V_star = oracle._compute_utilities_from_allocation(x_star_full)
    
    direction, valid = oracle._compute_direction(x_star_full, V_star, backward=False)
    if valid:
        D_star = oracle._compute_directional_derivative_with_direction(x_star_full, V_star, direction)
    else:
        D_star = float('nan')
    
    # Count Nash solves from timing stats
    stats = oracle.get_timing_stats()
    n_nash = stats['binary_search_nash_solves']['count']
    
    return BoundaryResult(
        x_star=x_star,
        t_final=0.0,  # Not tracked in binary search
        D_final=D_star,
        n_nash_solves=n_nash,
        time_seconds=elapsed,
        method='binary',
        converged=abs(D_star) < 1e-6 if not np.isnan(D_star) else False
    )


def run_integral_method(x_full: np.ndarray, V: np.ndarray, D_0: float,
                        oracle: DirectionalDerivativeOracle, verbose: bool = False) -> BoundaryResult:
    """Run integral method and return the boundary point."""
    x_free = x_full.flatten()
    
    # Set oracle verbose for debugging
    oracle.verbose = verbose
    
    start_time = time.perf_counter()
    
    # Reset timing stats to count Nash solves
    oracle.reset_timing_stats()
    oracle.track_timing = True
    
    # Call the integral method directly
    normal, rhs = oracle._find_separating_hyperplane_integral(x_free, x_full, V, D_0)
    
    elapsed = time.perf_counter() - start_time
    
    # Reset verbose
    oracle.verbose = False
    
    # Get the boundary point
    x_star = oracle.last_boundary_point
    
    # Compute directional derivative at boundary
    x_star_full = x_star.reshape((oracle.n_agents, oracle.n_resources))
    V_star = oracle._compute_utilities_from_allocation(x_star_full)
    
    direction, valid = oracle._compute_direction(x_star_full, V_star, backward=False)
    if valid:
        D_star = oracle._compute_directional_derivative_with_direction(x_star_full, V_star, direction)
    else:
        D_star = float('nan')
    
    # Count Nash solves from timing stats
    stats = oracle.get_timing_stats()
    n_nash = stats['binary_search_nash_solves']['count']
    
    return BoundaryResult(
        x_star=x_star,
        t_final=0.0,
        D_final=D_star,
        n_nash_solves=n_nash,
        time_seconds=elapsed,
        method='integral',
        converged=abs(D_star) < 1e-6 if not np.isnan(D_star) else False
    )


def compare_results(binary_result: BoundaryResult, integral_result: BoundaryResult,
                    n_agents: int, n_resources: int) -> Dict:
    """Compare the two results and return metrics."""
    # Compute difference in x*
    x_diff = binary_result.x_star - integral_result.x_star
    x_diff_norm = np.linalg.norm(x_diff)
    x_diff_max = np.max(np.abs(x_diff))
    
    # Relative difference
    binary_norm = np.linalg.norm(binary_result.x_star)
    rel_diff = x_diff_norm / binary_norm if binary_norm > 1e-10 else float('inf')
    
    return {
        'x_diff_norm': x_diff_norm,
        'x_diff_max': x_diff_max,
        'x_diff_relative': rel_diff,
        'D_binary': binary_result.D_final,
        'D_integral': integral_result.D_final,
        'D_diff': abs(binary_result.D_final - integral_result.D_final),
        'nash_solves_binary': binary_result.n_nash_solves,
        'nash_solves_integral': integral_result.n_nash_solves,
        'time_binary': binary_result.time_seconds,
        'time_integral': integral_result.time_seconds,
        'speedup': binary_result.time_seconds / integral_result.time_seconds if integral_result.time_seconds > 0 else float('inf'),
        'converged_binary': binary_result.converged,
        'converged_integral': integral_result.converged,
    }


def run_comparison_test(n_agents: int, n_resources: int, n_attackers: int, 
                        seed: int, verbose: bool = False) -> Optional[Dict]:
    """Run a single comparison test."""
    # Generate utilities
    utilities = generate_random_utilities(n_agents, n_resources, seed)
    supply = np.ones(n_resources) * n_agents
    
    # Random attackers
    np.random.seed(seed + 1000)
    attacking_group = set(np.random.choice(n_agents, n_attackers, replace=False))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Test: n_agents={n_agents}, n_resources={n_resources}, n_attackers={n_attackers}, seed={seed}")
        print(f"Attacking group: {attacking_group}")
    
    # Find a violating point
    x_full, V, D_0 = find_violating_point(n_agents, n_resources, utilities, attacking_group, supply)
    
    if D_0 >= -1e-8:
        if verbose:
            print(f"  Could not find violating point, skipping")
        return None
    
    if verbose:
        print(f"  Found violating point with D_0 = {D_0:.6e}")
    
    # Create oracles for each method
    oracle_binary = DirectionalDerivativeOracle(
        n_agents, n_resources, utilities, attacking_group, supply,
        verbose=False, track_timing=True, use_integral_method=False
    )
    
    oracle_integral = DirectionalDerivativeOracle(
        n_agents, n_resources, utilities, attacking_group, supply,
        verbose=False, track_timing=True, use_integral_method=True
    )
    
    # Run both methods
    binary_result = run_binary_search_method(x_full, V, D_0, oracle_binary)
    
    if verbose:
        print(f"\n  Running integral method with debug output:")
    integral_result = run_integral_method(x_full, V, D_0, oracle_integral, verbose=verbose)
    
    # Compare results
    comparison = compare_results(binary_result, integral_result, n_agents, n_resources)
    comparison['n_agents'] = n_agents
    comparison['n_resources'] = n_resources
    comparison['n_attackers'] = n_attackers
    comparison['seed'] = seed
    comparison['D_0'] = D_0
    
    if verbose:
        print(f"\n  Binary Search:")
        print(f"    Time: {binary_result.time_seconds:.4f}s")
        print(f"    Nash solves: {binary_result.n_nash_solves}")
        print(f"    D at x*: {binary_result.D_final:.6e}")
        print(f"    Converged: {binary_result.converged}")
        
        print(f"\n  Integral Method:")
        print(f"    Time: {integral_result.time_seconds:.4f}s")
        print(f"    Nash solves: {integral_result.n_nash_solves}")
        print(f"    D at x*: {integral_result.D_final:.6e}")
        print(f"    Converged: {integral_result.converged}")
        
        print(f"\n  Comparison:")
        print(f"    x* diff (L2 norm): {comparison['x_diff_norm']:.6e}")
        print(f"    x* diff (max):     {comparison['x_diff_max']:.6e}")
        print(f"    x* diff (relative): {comparison['x_diff_relative']:.6e}")
        print(f"    D diff:            {comparison['D_diff']:.6e}")
        print(f"    Speedup:           {comparison['speedup']:.2f}x")
        
        # Print x* values side by side (first few elements)
        print(f"\n  x* comparison (first 10 elements):")
        print(f"    {'Binary':>12} {'Integral':>12} {'Diff':>12}")
        for i in range(min(10, len(binary_result.x_star))):
            diff = binary_result.x_star[i] - integral_result.x_star[i]
            print(f"    {binary_result.x_star[i]:>12.6f} {integral_result.x_star[i]:>12.6f} {diff:>12.6e}")
    
    return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare binary search vs integral boundary finding")
    parser.add_argument("-n", "--n-tests", type=int, default=5, dest="n_tests",
                        help="Number of test cases (default: 5)")
    parser.add_argument("--min-agents", type=int, default=5, dest="min_agents")
    parser.add_argument("--max-agents", type=int, default=10, dest="max_agents")
    parser.add_argument("--min-resources", type=int, default=3, dest="min_resources")
    parser.add_argument("--max-resources", type=int, default=8, dest="max_resources")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Comparing Binary Search vs Integral Method for Boundary Finding")
    print("="*70)
    
    np.random.seed(args.seed)
    
    results = []
    
    for i in range(args.n_tests):
        test_seed = args.seed + i * 100
        n_agents = np.random.randint(args.min_agents, args.max_agents + 1)
        n_resources = np.random.randint(args.min_resources, args.max_resources + 1)
        n_attackers = np.random.randint(1, min(n_agents - 1, 5) + 1)
        
        result = run_comparison_test(n_agents, n_resources, n_attackers, test_seed, verbose=args.verbose)
        
        if result is not None:
            results.append(result)
            if not args.verbose:
                print(f"Test {i+1}: x_diff={result['x_diff_norm']:.2e}, "
                      f"D_binary={result['D_binary']:.2e}, D_integral={result['D_integral']:.2e}, "
                      f"speedup={result['speedup']:.2f}x")
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Tests completed: {len(results)}")
        
        x_diffs = [r['x_diff_norm'] for r in results]
        d_diffs = [r['D_diff'] for r in results]
        speedups = [r['speedup'] for r in results]
        
        print(f"\nx* difference (L2 norm):")
        print(f"  Mean: {np.mean(x_diffs):.6e}")
        print(f"  Max:  {np.max(x_diffs):.6e}")
        print(f"  Min:  {np.min(x_diffs):.6e}")
        
        print(f"\nD difference:")
        print(f"  Mean: {np.mean(d_diffs):.6e}")
        print(f"  Max:  {np.max(d_diffs):.6e}")
        
        print(f"\nSpeedup (binary time / integral time):")
        print(f"  Mean: {np.mean(speedups):.2f}x")
        print(f"  Max:  {np.max(speedups):.2f}x")
        print(f"  Min:  {np.min(speedups):.2f}x")
        
        print(f"\nConvergence:")
        binary_converged = sum(1 for r in results if r['converged_binary'])
        integral_converged = sum(1 for r in results if r['converged_integral'])
        print(f"  Binary:   {binary_converged}/{len(results)}")
        print(f"  Integral: {integral_converged}/{len(results)}")
        
        print(f"\nNash solves:")
        binary_nash = [r['nash_solves_binary'] for r in results]
        integral_nash = [r['nash_solves_integral'] for r in results]
        print(f"  Binary mean:   {np.mean(binary_nash):.1f}")
        print(f"  Integral mean: {np.mean(integral_nash):.1f}")


if __name__ == "__main__":
    main()