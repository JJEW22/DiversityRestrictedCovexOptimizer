"""
Probe the boundary surface in 3D attacker allocation space with custom preferences.

This script allows you to specify:
- Attacker preference vector
- Defender preference vectors (can specify multiple defenders with same or different preferences)
- Number of agents

Example usage:
    python boundary_3d_custom.py --attacker-prefs 0.9,0.1,0.0 --defender-prefs 1.0,0.0,0.0 --n-defenders 9
    python boundary_3d_custom.py --attacker-prefs 0.5,0.3,0.2 --defender-prefs "1,0,0;0,1,0;0,0,1" --n-defenders 9
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Set
from openpyxl import Workbook
from openpyxl.styles import Font
import json
import argparse

from nash_welfare_optimizer import (
    DirectionalDerivativeOracle,
    NashWelfareOptimizer,
    compute_normal_at_boundary,
)


def parse_preference_vector(pref_str: str, normalize: bool = True) -> np.ndarray:
    """Parse a comma-separated preference string into a numpy array.
    
    Args:
        pref_str: Comma-separated string of values
        normalize: If True, normalize so values sum to 1
    """
    values = [float(x.strip()) for x in pref_str.split(',')]
    arr = np.array(values)
    
    if normalize:
        total = np.sum(arr)
        if total > 1e-10:
            arr = arr / total
    
    return arr


def parse_defender_prefs(pref_str: str, n_resources: int, normalize: bool = True) -> List[np.ndarray]:
    """
    Parse defender preferences. Can be:
    - Single preference: "1.0,0.0,0.0" (all defenders have same preference)
    - Multiple preferences separated by semicolons: "1,0,0;0,1,0;0,0,1" (cycle through)
    
    Args:
        pref_str: Preference string
        n_resources: Expected number of resources
        normalize: If True, normalize each preference vector to sum to 1
    """
    if ';' in pref_str:
        # Multiple preferences
        prefs = []
        for part in pref_str.split(';'):
            pref = parse_preference_vector(part.strip(), normalize=normalize)
            if len(pref) != n_resources:
                raise ValueError(f"Preference vector {pref} has wrong length (expected {n_resources})")
            prefs.append(pref)
        return prefs
    else:
        # Single preference
        pref = parse_preference_vector(pref_str, normalize=normalize)
        if len(pref) != n_resources:
            raise ValueError(f"Preference vector {pref} has wrong length (expected {n_resources})")
        return [pref]


def create_custom_setup(
    attacker_prefs: np.ndarray,
    defender_prefs: List[np.ndarray],
    n_defenders: int,
    n_resources: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Set[int], Set[int], int, int, int]:
    """
    Create a test setup with custom preferences.
    
    Args:
        attacker_prefs: Preference vector for the attacker
        defender_prefs: List of preference vectors for defenders (cycled if fewer than n_defenders)
        n_defenders: Number of defender agents
        n_resources: Number of resources (must match preference vector lengths)
    
    Returns:
        utilities, supply, attacking_group, defending_group, attacker_id, n_agents, n_resources
    """
    n_agents = n_defenders + 1  # defenders + 1 attacker
    
    utilities = np.zeros((n_agents, n_resources))
    
    # Set defender preferences (cycle through defender_prefs if needed)
    for i in range(n_defenders):
        pref_idx = i % len(defender_prefs)
        utilities[i] = defender_prefs[pref_idx]
    
    # Set attacker preferences (last agent)
    attacker_id = n_agents - 1
    utilities[attacker_id] = attacker_prefs
    
    # Supply: n_agents units of each resource
    supply = np.ones(n_resources) * n_agents
    
    # Groups
    attacking_group = {attacker_id}
    defending_group = set(range(n_defenders))
    
    return utilities, supply, attacking_group, defending_group, attacker_id, n_agents, n_resources


def generate_sphere_directions(n_phi: int = 10, n_theta: int = 20, skip_zero_components: bool = False) -> List[Tuple[float, float, np.ndarray]]:
    """Generate directions on the positive octant of a sphere."""
    directions = []
    seen_directions = set()
    
    for i in range(n_phi + 1):
        phi = (np.pi / 2) * i / n_phi
        phi_deg = np.degrees(phi)
        
        if i == 0:
            theta_range = [0]
        else:
            theta_range = range(n_theta + 1)
        
        for j in theta_range:
            theta = (np.pi / 2) * j / n_theta
            theta_deg = np.degrees(theta)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            direction = np.array([x, y, z])
            
            if np.linalg.norm(direction) < 1e-10:
                continue
            
            direction = direction / np.linalg.norm(direction)
            
            if skip_zero_components:
                if any(abs(direction[k]) < 1e-6 for k in range(3)):
                    continue
            
            dir_key = (round(direction[0], 6), round(direction[1], 6), round(direction[2], 6))
            
            if dir_key in seen_directions:
                continue
            seen_directions.add(dir_key)
            
            directions.append((phi_deg, theta_deg, direction))
    
    return directions


def solve_nash_for_non_attackers(
    utilities: np.ndarray,
    non_attacking_group: Set[int],
    remaining_supply: np.ndarray,
    n_agents: int,
    n_resources: int
) -> Optional[Dict[int, np.ndarray]]:
    """Solve Nash welfare maximization for non-attackers given remaining supply."""
    non_attackers = list(non_attacking_group)
    n_non_attackers = len(non_attackers)
    
    if n_non_attackers == 0:
        return {}
    
    if np.all(remaining_supply < 1e-10):
        return {i: np.zeros(n_resources) for i in non_attackers}
    
    try:
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
        
        n_vars = n_non_attackers * n_resources
        
        def var_idx(local_i, j):
            return local_i * n_resources + j
        
        def F(x=None, z=None):
            if x is None:
                x0 = matrix(0.0, (n_vars, 1))
                for local_i, agent in enumerate(non_attackers):
                    for j in range(n_resources):
                        x0[var_idx(local_i, j)] = remaining_supply[j] / n_non_attackers
                return (0, x0)
            
            V = []
            for local_i, agent in enumerate(non_attackers):
                v_i = sum(utilities[agent, j] * x[var_idx(local_i, j)] for j in range(n_resources))
                V.append(max(v_i, 1e-12))
            
            f = sum(-np.log(v) for v in V)
            
            Df = matrix(0.0, (1, n_vars))
            for local_i, agent in enumerate(non_attackers):
                for j in range(n_resources):
                    Df[var_idx(local_i, j)] = -utilities[agent, j] / V[local_i]
            
            if z is None:
                return (f, Df)
            
            H = matrix(0.0, (n_vars, n_vars))
            for local_i, agent in enumerate(non_attackers):
                for j1 in range(n_resources):
                    for j2 in range(n_resources):
                        idx1 = var_idx(local_i, j1)
                        idx2 = var_idx(local_i, j2)
                        H[idx1, idx2] = z[0] * utilities[agent, j1] * utilities[agent, j2] / (V[local_i] ** 2)
            
            return (f, Df, H)
        
        G = matrix(-np.eye(n_vars))
        h = matrix(np.zeros(n_vars))
        
        A = matrix(0.0, (n_resources, n_vars))
        b = matrix(remaining_supply)
        for j in range(n_resources):
            for local_i in range(n_non_attackers):
                A[j, var_idx(local_i, j)] = 1.0
        
        sol = solvers.cp(F, G, h, A=A, b=b)
        
        if sol['status'] != 'optimal':
            return None
        
        x_sol = np.array(sol['x']).flatten()
        
        result = {}
        for i in range(n_agents):
            if i in non_attacking_group:
                local_i = non_attackers.index(i)
                alloc = np.array([max(0, x_sol[var_idx(local_i, j)]) for j in range(n_resources)])
                result[i] = alloc
            else:
                result[i] = np.zeros(n_resources)
        
        return result
        
    except Exception as e:
        print(f"  Error in Nash welfare solver: {e}")
        return None


def build_full_allocation(
    attacker_alloc: np.ndarray,
    non_attacker_alloc: Dict[int, np.ndarray],
    attacker_id: int,
    n_agents: int,
    n_resources: int
) -> np.ndarray:
    """Build full allocation matrix from attacker and non-attacker allocations."""
    x_full = np.zeros((n_agents, n_resources))
    x_full[attacker_id] = attacker_alloc
    for i, alloc in non_attacker_alloc.items():
        x_full[i] = alloc
    return x_full


def probe_direction(
    direction: np.ndarray,
    utilities: np.ndarray,
    supply: np.ndarray,
    attacking_group: Set[int],
    non_attacking_group: Set[int],
    attacker_id: int,
    n_agents: int,
    n_resources: int,
    oracle: DirectionalDerivativeOracle,
    verbose: bool = False
) -> Dict:
    """Probe a single direction to find the boundary point."""
    
    # Scale direction to max allocation
    scale_factors = []
    for j in range(n_resources):
        if direction[j] > 1e-10:
            scale_factors.append(supply[j] / direction[j])
    
    if not scale_factors:
        return {
            'violated_at_max': False,
            'boundary_attacker_alloc': None,
            'max_attacker_alloc': None,
            'boundary_non_attacker_utils': None,
            'normal_vector': None,
        }
    
    max_scale = min(scale_factors)
    max_attacker_alloc = direction * max_scale
    
    # Compute non-attacker allocation at max
    remaining_supply = supply - max_attacker_alloc
    remaining_supply = np.maximum(remaining_supply, 0)
    non_attacker_alloc = solve_nash_for_non_attackers(
        utilities, non_attacking_group, remaining_supply, n_agents, n_resources
    )
    
    if non_attacker_alloc is None:
        return {
            'violated_at_max': False,
            'boundary_attacker_alloc': None,
            'max_attacker_alloc': max_attacker_alloc,
            'boundary_non_attacker_utils': None,
            'normal_vector': None,
        }
    
    # Build full allocation and check if violated
    x_full = build_full_allocation(max_attacker_alloc, non_attacker_alloc, attacker_id, n_agents, n_resources)
    
    violated, _, _ = oracle.is_violated(x_full.flatten())
    
    result = {
        'violated_at_max': violated,
        'max_attacker_alloc': max_attacker_alloc,
    }
    
    if violated and oracle.last_boundary_point is not None:
        # Get the boundary point
        boundary_full = oracle.last_boundary_point.reshape((n_agents, n_resources))
        boundary_attacker_alloc = boundary_full[attacker_id].copy()
        
        # Compute non-attacker utilities at boundary
        non_attacker_utils = {}
        for i in non_attacking_group:
            non_attacker_utils[i] = np.dot(utilities[i], boundary_full[i])
        
        result['boundary_attacker_alloc'] = boundary_attacker_alloc
        result['boundary_non_attacker_utils'] = non_attacker_utils
        
        # Compute normal vector at boundary
        try:
            normal = compute_normal_at_boundary(
                boundary_full.flatten(), utilities, attacking_group, supply
            )
            # Extract attacker portion of normal
            attacker_normal = normal.reshape((n_agents, n_resources))[attacker_id]
            result['normal_vector'] = attacker_normal
        except:
            result['normal_vector'] = None
    else:
        result['boundary_attacker_alloc'] = None
        result['boundary_non_attacker_utils'] = None
        result['normal_vector'] = None
    
    return result


def compute_global_optimum_direction(
    utilities: np.ndarray,
    supply: np.ndarray,
    attacking_group: Set[int],
    n_agents: int,
    n_resources: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the global optimum and extract the attacker's direction."""
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    result = optimizer.solve()
    
    attacker_id = list(attacking_group)[0]
    attacker_allocation = result.allocation[attacker_id].copy()
    
    norm = np.linalg.norm(attacker_allocation)
    if norm > 1e-10:
        attacker_direction = attacker_allocation / norm
    else:
        attacker_direction = attacker_allocation.copy()
    
    return attacker_direction, attacker_allocation


def main():
    parser = argparse.ArgumentParser(
        description='Probe boundary surface in 3D with custom preferences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 9 defenders all preferring resource 0, attacker prefers resources 0 and 1
  python boundary_3d_custom.py --attacker-prefs 0.9,0.1,0.0 --defender-prefs 1.0,0.0,0.0 --n-defenders 9

  # 9 defenders with cycling preferences (3 prefer R0, 3 prefer R1, 3 prefer R2)
  python boundary_3d_custom.py --attacker-prefs 0.5,0.3,0.2 --defender-prefs "1,0,0;0,1,0;0,0,1" --n-defenders 9

  # Each defender has unique preference for their own resource (10 defenders, attacker is 11th)
  python boundary_3d_custom.py --attacker-prefs 0.33,0.33,0.34 --defender-prefs "1,0,0;0,1,0;0,0,1" --n-defenders 3
        """
    )
    parser.add_argument('--attacker-prefs', type=str, required=True,
                        help='Attacker preference vector (comma-separated, e.g., "0.9,0.1,0.0")')
    parser.add_argument('--defender-prefs', type=str, required=True,
                        help='Defender preference(s). Single: "1,0,0". Multiple (cycled): "1,0,0;0,1,0;0,0,1"')
    parser.add_argument('--n-defenders', type=int, required=True,
                        help='Number of defender agents')
    parser.add_argument('--n-phi', type=int, default=8, help='Number of phi divisions (default: 8)')
    parser.add_argument('--n-theta', type=int, default=16, help='Number of theta divisions (default: 16)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse preferences (with normalization)
    n_resources = 3  # Fixed for 3D visualization
    
    # Parse raw first to show what was input
    attacker_prefs_raw = parse_preference_vector(args.attacker_prefs, normalize=False)
    attacker_prefs = parse_preference_vector(args.attacker_prefs, normalize=True)
    if len(attacker_prefs) != n_resources:
        print(f"Error: Attacker preferences must have {n_resources} values, got {len(attacker_prefs)}")
        return
    
    defender_prefs_raw = parse_defender_prefs(args.defender_prefs, n_resources, normalize=False)
    defender_prefs = parse_defender_prefs(args.defender_prefs, n_resources, normalize=True)
    
    print("=" * 60)
    print("3D BOUNDARY SURFACE PROBE (CUSTOM PREFERENCES)")
    print("=" * 60)
    
    print()
    print("Input preferences (raw):")
    print(f"  Attacker: {attacker_prefs_raw.round(6)}")
    for i, pref in enumerate(defender_prefs_raw):
        print(f"  Defender pattern {i}: {pref.round(6)}")
    
    print()
    print("Normalized preferences (sum to 1):")
    print(f"  Attacker: {attacker_prefs.round(6)}")
    for i, pref in enumerate(defender_prefs):
        print(f"  Defender pattern {i}: {pref.round(6)}")
    print()
    
    # Create setup
    utilities, supply, attacking_group, defending_group, attacker_id, n_agents, n_resources = \
        create_custom_setup(attacker_prefs, defender_prefs, args.n_defenders, n_resources)
    
    non_attacking_group = set(range(n_agents)) - attacking_group
    
    print(f"n_agents: {n_agents}")
    print(f"n_resources: {n_resources}")
    print(f"n_defenders: {args.n_defenders}")
    print(f"supply: {supply}")
    print()
    print(f"Attacker (Agent {attacker_id}): preferences = {utilities[attacker_id].round(6)}")
    print()
    print(f"Defender preferences ({len(defender_prefs)} unique pattern(s)):")
    for i, pref in enumerate(defender_prefs):
        print(f"  Pattern {i}: {pref.round(6)}")
    print()
    print(f"Defender utilities:")
    for i in range(min(10, args.n_defenders)):
        print(f"  Agent {i}: {utilities[i].round(6)}")
    if args.n_defenders > 10:
        print(f"  ... ({args.n_defenders - 10} more defenders)")
    print()
    
    # Compute global optimum direction
    print("Computing global optimum (unconstrained Nash welfare)...")
    optimum_direction, optimum_allocation = compute_global_optimum_direction(
        utilities, supply, attacking_group, n_agents, n_resources
    )
    print(f"  Global optimum attacker allocation: {optimum_allocation.round(6)}")
    print(f"  Global optimum direction: {optimum_direction.round(6)}")
    print()
    
    # Create oracle
    oracle = DirectionalDerivativeOracle(
        n_agents=n_agents,
        n_resources=n_resources,
        utilities=utilities,
        attacking_group=attacking_group,
        supply=supply,
        verbose=False
    )
    
    # Generate directions
    directions = generate_sphere_directions(args.n_phi, args.n_theta)
    
    # Add optimum direction if not already present
    opt_dir_key = (round(optimum_direction[0], 6), round(optimum_direction[1], 6), round(optimum_direction[2], 6))
    existing_keys = set()
    for _, _, d in directions:
        key = (round(d[0], 6), round(d[1], 6), round(d[2], 6))
        existing_keys.add(key)
    
    if opt_dir_key not in existing_keys and np.linalg.norm(optimum_direction) > 1e-10:
        directions.append((999.0, 999.0, optimum_direction))
        print(f"Added global optimum direction to test set")
    
    print(f"Probing {len(directions)} directions...")
    print()
    
    # Probe each direction
    results = []
    
    for idx, (phi, theta, direction) in enumerate(directions):
        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(directions)}")
        
        result = probe_direction(
            direction=direction,
            utilities=utilities,
            supply=supply,
            attacking_group=attacking_group,
            non_attacking_group=non_attacking_group,
            attacker_id=attacker_id,
            n_agents=n_agents,
            n_resources=n_resources,
            oracle=oracle,
            verbose=args.verbose
        )
        
        is_optimum = (phi == 999.0 and theta == 999.0)
        
        if is_optimum:
            print(f"\n*** GLOBAL OPTIMUM DIRECTION RESULT ***")
            print(f"  Direction: {direction.round(6)}")
            print(f"  Boundary point: {result['boundary_attacker_alloc']}")
            print(f"  Violated at max: {result['violated_at_max']}")
            print()
        
        results.append((phi, theta, result))
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_violated = sum(1 for _, _, r in results if r['violated_at_max'])
    n_boundary = sum(1 for _, _, r in results if r['boundary_attacker_alloc'] is not None)
    print(f"Total directions probed: {len(results)}")
    print(f"Violated at max: {n_violated}")
    print(f"Boundary points found: {n_boundary}")
    
    # Print boundary points
    print()
    print("Boundary points:")
    for phi, theta, r in results:
        if r['boundary_attacker_alloc'] is not None:
            bp = r['boundary_attacker_alloc']
            print(f"  phi={phi:.1f}, theta={theta:.1f}: [{bp[0]:.4f}, {bp[1]:.4f}, {bp[2]:.4f}]")
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot boundary points
    boundary_points = []
    for phi, theta, r in results:
        if r['boundary_attacker_alloc'] is not None:
            boundary_points.append(r['boundary_attacker_alloc'])
    
    if boundary_points:
        boundary_points = np.array(boundary_points)
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                   c='blue', s=50, alpha=0.6, label='Boundary points')
    
    # Plot global optimum
    ax.scatter([optimum_allocation[0]], [optimum_allocation[1]], [optimum_allocation[2]],
               c='green', s=200, marker='*', label='Global optimum')
    
    ax.set_xlabel('Resource 0')
    ax.set_ylabel('Resource 1')
    ax.set_zlabel('Resource 2')
    ax.set_title(f'Boundary Surface\nAttacker: {attacker_prefs.round(2)}, Defenders: {args.n_defenders}')
    ax.legend()
    
    # Save plot
    output_file = f"{args.output_dir}/boundary_3d_custom.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    main()