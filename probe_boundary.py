"""
Probe the boundary surface from different radial directions in the attacker's allocation space.

This script:
1. Defines radial directions in the attacker's 2D allocation space (R0, R1)
2. For each direction, scales to the maximum (where one component = 1)
3. Solves Nash welfare for non-attackers given remaining supply
4. Checks if directional derivative is violated
5. If violated, uses binary search to find x* (boundary point)
6. Outputs results to spreadsheet and displays a graph
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from openpyxl import Workbook
from openpyxl.styles import Font

from nash_welfare_optimizer import (
    DirectionalDerivativeOracle,
    NashWelfareOptimizer,
)


def create_test_setup(n_agents: int = 10):
    """Create the standard test setup."""
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
    
    return utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources


def generate_directions(n_between: int = 5) -> List[Tuple[float, np.ndarray]]:
    """
    Generate radial directions in 2D space.
    
    Returns list of (angle_degrees, direction_vector) tuples.
    Includes [1,0], [0,1], n_between evenly spaced directions between them,
    and special directions [1, 1/10] and [1/10, 1].
    """
    directions = []
    
    # Total angles: 0°, then n_between evenly spaced, then 90°
    # That's n_between + 2 total directions in the first quadrant
    n_total = n_between + 2
    
    for i in range(n_total):
        angle_rad = (np.pi / 2) * i / (n_total - 1)  # 0 to pi/2
        angle_deg = np.degrees(angle_rad)
        
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # Normalize to unit vector
        direction = direction / np.linalg.norm(direction)
        
        directions.append((angle_deg, direction))
    
    # Add special directions [1, 1/10] and [1/10, 1]
    special_dir_1 = np.array([1.0, 0.1])
    special_dir_1 = special_dir_1 / np.linalg.norm(special_dir_1)
    angle_1 = np.degrees(np.arctan2(0.1, 1.0))
    directions.append((angle_1, special_dir_1))
    
    special_dir_2 = np.array([0.1, 1.0])
    special_dir_2 = special_dir_2 / np.linalg.norm(special_dir_2)
    angle_2 = np.degrees(np.arctan2(1.0, 0.1))
    directions.append((angle_2, special_dir_2))
    
    # Sort by angle
    directions.sort(key=lambda x: x[0])
    
    return directions


def scale_to_maximum(direction: np.ndarray, supply: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Scale the direction until one component equals its supply (max allocation).
    
    Returns (scaled_allocation, scale_factor).
    """
    # Find the scale factor where the first component hits its limit
    scale_factors = []
    for j in range(len(direction)):
        if direction[j] > 1e-10:
            scale_factors.append(supply[j] / direction[j])
    
    if not scale_factors:
        return direction.copy(), 1.0
    
    scale = min(scale_factors)
    return direction * scale, scale


def solve_nash_for_non_attackers_standalone(
    utilities: np.ndarray,
    non_attacking_group: Set[int],
    remaining_supply: np.ndarray,
    n_agents: int,
    n_resources: int
) -> Optional[Dict[int, np.ndarray]]:
    """
    Solve Nash welfare maximization for non-attackers given remaining supply.
    Uses the same approach as in DirectionalDerivativeOracle.
    """
    non_attackers = list(non_attacking_group)
    n_non_attackers = len(non_attackers)
    
    if n_non_attackers == 0:
        return {}
    
    # Check if there's any supply to allocate
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
                v_i = 0.0
                for j in range(n_resources):
                    v_i += utilities[agent, j] * x[var_idx(local_i, j)]
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
        for local_i, agent in enumerate(non_attackers):
            alloc = np.zeros(n_resources)
            for j in range(n_resources):
                alloc[j] = max(0, x_sol[var_idx(local_i, j)])
            result[agent] = alloc
        
        return result
        
    except Exception as e:
        print(f"Nash solve exception: {e}")
        return None


def build_full_allocation(
    attacker_allocation: np.ndarray,
    non_attacker_allocation: Dict[int, np.ndarray],
    attacker_id: int,
    n_agents: int,
    n_resources: int
) -> np.ndarray:
    """Build full allocation matrix from attacker and non-attacker allocations."""
    x_full = np.zeros((n_agents, n_resources))
    x_full[attacker_id] = attacker_allocation
    for agent_id, alloc in non_attacker_allocation.items():
        x_full[agent_id] = alloc
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
    """
    Probe a single direction to find the boundary point.
    
    Returns dict with:
    - direction: the input direction
    - max_attacker_alloc: attacker allocation at maximum scale
    - max_scale: scale factor to reach maximum
    - violated_at_max: whether directional derivative is violated at max
    - boundary_attacker_alloc: attacker allocation at boundary (x*)
    - boundary_scale: scale factor at boundary
    - boundary_full_alloc: full allocation matrix at boundary
    - dir_deriv_at_max: directional derivative value at max
    - dir_deriv_at_boundary: directional derivative value at boundary
    """
    result = {
        'direction': direction.copy(),
        'max_attacker_alloc': None,
        'max_scale': None,
        'violated_at_max': None,
        'boundary_attacker_alloc': None,
        'boundary_scale': None,
        'boundary_full_alloc': None,
        'dir_deriv_at_max': None,
        'dir_deriv_at_boundary': None,
    }
    
    # Scale direction to maximum
    max_attacker_alloc, max_scale = scale_to_maximum(direction, supply)
    result['max_attacker_alloc'] = max_attacker_alloc
    result['max_scale'] = max_scale
    
    if verbose:
        print(f"  Direction: {direction.round(4)}")
        print(f"  Max attacker allocation: {max_attacker_alloc.round(6)}")
    
    # Solve Nash for non-attackers at maximum
    remaining_supply_max = supply - max_attacker_alloc
    remaining_supply_max = np.maximum(remaining_supply_max, 0)
    
    non_attacker_alloc_max = solve_nash_for_non_attackers_standalone(
        utilities, non_attacking_group, remaining_supply_max, n_agents, n_resources
    )
    
    if non_attacker_alloc_max is None:
        print(f"  Failed to solve Nash for non-attackers at max")
        return result
    
    # Build full allocation at max
    x_full_max = build_full_allocation(
        max_attacker_alloc, non_attacker_alloc_max, attacker_id, n_agents, n_resources
    )
    x_free_max = x_full_max.flatten()
    
    # Check if violated at max
    violated, normal, rhs = oracle.is_violated(x_free_max)
    result['violated_at_max'] = violated
    
    # Compute directional derivative at max for logging
    V_max = np.array([np.dot(utilities[i], x_full_max[i]) for i in range(n_agents)])
    
    # Try to compute dir deriv if possible
    if not any(V_max[i] <= 1e-10 for i in non_attacking_group):
        fwd_dir, fwd_valid = oracle._compute_direction(x_full_max, V_max, backward=False)
        if fwd_valid:
            result['dir_deriv_at_max'] = oracle._compute_directional_derivative_with_direction(x_full_max, V_max, fwd_dir)
    
    if verbose:
        print(f"  Violated at max: {violated}")
        print(f"  Dir deriv at max: {result['dir_deriv_at_max']}")
    
    if not violated:
        # Boundary is at or beyond maximum
        result['boundary_attacker_alloc'] = max_attacker_alloc
        result['boundary_scale'] = max_scale
        result['boundary_full_alloc'] = x_full_max
        result['dir_deriv_at_boundary'] = result['dir_deriv_at_max']
        if verbose:
            print(f"  Not violated - boundary at max or beyond")
        return result
    
    # Binary search to find boundary
    if verbose:
        print(f"  Violated - performing binary search...")
    
    t_low, t_high = 0.0, 1.0
    
    for iteration in range(50):
        t_mid = (t_low + t_high) / 2
        
        # Scale attacker allocation
        attacker_alloc_mid = max_attacker_alloc * t_mid
        
        # Remaining supply
        remaining_supply_mid = supply - attacker_alloc_mid
        remaining_supply_mid = np.maximum(remaining_supply_mid, 0)
        
        # Solve Nash for non-attackers
        non_attacker_alloc_mid = solve_nash_for_non_attackers_standalone(
            utilities, non_attacking_group, remaining_supply_mid, n_agents, n_resources
        )
        
        if non_attacker_alloc_mid is None:
            t_low = t_mid
            continue
        
        # Build full allocation
        x_full_mid = build_full_allocation(
            attacker_alloc_mid, non_attacker_alloc_mid, attacker_id, n_agents, n_resources
        )
        
        V_mid = np.array([np.dot(utilities[i], x_full_mid[i]) for i in range(n_agents)])
        
        # Check for zero utilities
        if any(V_mid[i] <= 1e-10 for i in non_attacking_group):
            t_low = t_mid
            continue
        
        # Compute directional derivative
        fwd_dir, fwd_valid = oracle._compute_direction(x_full_mid, V_mid, backward=False)
        
        if not fwd_valid:
            t_high = t_mid
            continue
        
        dir_deriv = oracle._compute_directional_derivative_with_direction(x_full_mid, V_mid, fwd_dir)
        
        if abs(dir_deriv) < 1e-8:
            # Converged
            break
        elif dir_deriv > 0:
            t_low = t_mid
        else:
            t_high = t_mid
    
    # Final boundary point
    boundary_scale = t_mid * max_scale
    boundary_attacker_alloc = max_attacker_alloc * t_mid
    
    remaining_supply_boundary = supply - boundary_attacker_alloc
    remaining_supply_boundary = np.maximum(remaining_supply_boundary, 0)
    
    non_attacker_alloc_boundary = solve_nash_for_non_attackers_standalone(
        utilities, non_attacking_group, remaining_supply_boundary, n_agents, n_resources
    )
    
    if non_attacker_alloc_boundary is not None:
        x_full_boundary = build_full_allocation(
            boundary_attacker_alloc, non_attacker_alloc_boundary, attacker_id, n_agents, n_resources
        )
        
        V_boundary = np.array([np.dot(utilities[i], x_full_boundary[i]) for i in range(n_agents)])
        
        if not any(V_boundary[i] <= 1e-10 for i in non_attacking_group):
            fwd_dir, fwd_valid = oracle._compute_direction(x_full_boundary, V_boundary, backward=False)
            if fwd_valid:
                result['dir_deriv_at_boundary'] = oracle._compute_directional_derivative_with_direction(
                    x_full_boundary, V_boundary, fwd_dir
                )
        
        result['boundary_full_alloc'] = x_full_boundary
    
    result['boundary_attacker_alloc'] = boundary_attacker_alloc
    result['boundary_scale'] = boundary_scale
    
    if verbose:
        print(f"  Boundary attacker allocation: {boundary_attacker_alloc.round(6)}")
        print(f"  Boundary scale: {boundary_scale:.6f}")
        print(f"  Dir deriv at boundary: {result['dir_deriv_at_boundary']}")
    
    return result


def save_results_to_xlsx(
    results: List[Tuple[float, Dict]],
    filename: str,
    n_agents: int,
    n_resources: int
):
    """Save probe results to an Excel file."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Boundary Probe Results"
    
    # Header
    headers = [
        "Angle (deg)", 
        "Direction R0", "Direction R1",
        "Max Scale",
        "Max Attacker R0", "Max Attacker R1",
        "Violated at Max",
        "Dir Deriv at Max",
        "Boundary Scale",
        "Boundary Attacker R0", "Boundary Attacker R1",
        "Dir Deriv at Boundary"
    ]
    ws.append(headers)
    
    for col in range(1, len(headers) + 1):
        ws.cell(row=1, column=col).font = Font(bold=True)
    
    # Data rows
    for angle, result in results:
        row = [
            angle,
            result['direction'][0],
            result['direction'][1],
            result['max_scale'],
            result['max_attacker_alloc'][0] if result['max_attacker_alloc'] is not None else None,
            result['max_attacker_alloc'][1] if result['max_attacker_alloc'] is not None else None,
            result['violated_at_max'],
            result['dir_deriv_at_max'],
            result['boundary_scale'],
            result['boundary_attacker_alloc'][0] if result['boundary_attacker_alloc'] is not None else None,
            result['boundary_attacker_alloc'][1] if result['boundary_attacker_alloc'] is not None else None,
            result['dir_deriv_at_boundary'],
        ]
        ws.append(row)
    
    # Adjust column widths
    for col in ws.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        ws.column_dimensions[column].width = max(max_length + 2, 12)
    
    wb.save(filename)
    print(f"Results saved to {filename}")


def plot_boundary(results: List[Tuple[float, Dict]], filename: str):
    """Plot the boundary points."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Extract boundary points
    boundary_r0 = []
    boundary_r1 = []
    max_r0 = []
    max_r1 = []
    
    for angle, result in results:
        if result['boundary_attacker_alloc'] is not None:
            boundary_r0.append(result['boundary_attacker_alloc'][0])
            boundary_r1.append(result['boundary_attacker_alloc'][1])
        if result['max_attacker_alloc'] is not None:
            max_r0.append(result['max_attacker_alloc'][0])
            max_r1.append(result['max_attacker_alloc'][1])
    
    # Plot maximum points (outer boundary of search)
    ax.scatter(max_r0, max_r1, c='lightgray', s=100, marker='x', label='Max (search limit)', zorder=2)
    
    # Plot boundary points
    ax.scatter(boundary_r0, boundary_r1, c='blue', s=100, marker='o', label='Boundary (x*)', zorder=3)
    
    # Connect boundary points
    if len(boundary_r0) > 1:
        ax.plot(boundary_r0 + [boundary_r0[0]], boundary_r1 + [boundary_r1[0]], 
                'b-', alpha=0.5, linewidth=2, zorder=1)
    
    # Draw lines from origin to max points
    for angle, result in results:
        if result['max_attacker_alloc'] is not None:
            ax.plot([0, result['max_attacker_alloc'][0]], 
                   [0, result['max_attacker_alloc'][1]], 
                   'gray', alpha=0.3, linestyle='--', zorder=0)
    
    # Labels
    ax.set_xlabel('Attacker R0 Allocation', fontsize=12)
    ax.set_ylabel('Attacker R1 Allocation', fontsize=12)
    ax.set_title('Boundary Surface Probe\n(Attacker Allocation Space)', fontsize=14)
    ax.legend()
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add origin
    ax.scatter([0], [0], c='green', s=100, marker='s', label='Origin', zorder=4)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Probe boundary surface from different directions')
    parser.add_argument('--n-agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--n-between', type=int, default=5, help='Number of directions between [1,0] and [0,1]')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BOUNDARY SURFACE PROBE")
    print("=" * 60)
    
    # Setup
    utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources = \
        create_test_setup(args.n_agents)
    
    non_attacking_group = set(range(n_agents)) - attacking_group
    
    print(f"n_agents: {n_agents}")
    print(f"n_resources: {n_resources}")
    print(f"Attacker (Agent {attacker_id}): utilities = {utilities[attacker_id]}")
    print(f"Others: utilities = {utilities[0]}")
    print()
    
    # Create oracle
    oracle = DirectionalDerivativeOracle(
        n_agents=n_agents,
        n_resources=n_resources,
        utilities=utilities,
        attacking_group=attacking_group,
        supply=supply,
        verbose=False  # We'll handle our own verbose output
    )
    
    # Generate directions
    directions = generate_directions(args.n_between)
    print(f"Probing {len(directions)} directions...")
    print()
    
    # Probe each direction
    results = []
    for angle, direction in directions:
        print(f"Direction {angle:.1f}°:")
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
        results.append((angle, result))
        
        # Print summary
        if result['boundary_attacker_alloc'] is not None:
            print(f"  -> Boundary: [{result['boundary_attacker_alloc'][0]:.4f}, {result['boundary_attacker_alloc'][1]:.4f}]")
        print()
    
    # Save results
    xlsx_filename = f"{args.output_dir}/boundary_probe_n{args.n_agents}.xlsx"
    save_results_to_xlsx(results, xlsx_filename, n_agents, n_resources)
    
    # Plot
    plot_filename = f"{args.output_dir}/boundary_probe_n{args.n_agents}.png"
    plot_boundary(results, plot_filename)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for angle, result in results:
        violated = "YES" if result['violated_at_max'] else "NO"
        if result['boundary_attacker_alloc'] is not None:
            boundary = f"[{result['boundary_attacker_alloc'][0]:.4f}, {result['boundary_attacker_alloc'][1]:.4f}]"
        else:
            boundary = "N/A"
        print(f"  {angle:5.1f}°: violated={violated}, boundary={boundary}")


if __name__ == "__main__":
    main()