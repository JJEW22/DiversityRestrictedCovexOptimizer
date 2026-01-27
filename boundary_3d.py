"""
Probe the boundary surface in 3D attacker allocation space.

This script:
1. Creates a test case with 3 resources
2. Defines radial directions on a sphere/hemisphere in the attacker's 3D allocation space
3. For each direction, finds the boundary point x* using binary search
4. Outputs results to spreadsheet and creates Desmos 3D compatible output
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Set
from openpyxl import Workbook
from openpyxl.styles import Font
import json

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
    """
    if ';' in pref_str:
        prefs = []
        for part in pref_str.split(';'):
            pref = parse_preference_vector(part.strip(), normalize=normalize)
            if len(pref) != n_resources:
                raise ValueError(f"Preference vector {pref} has wrong length (expected {n_resources})")
            prefs.append(pref)
        return prefs
    else:
        pref = parse_preference_vector(pref_str, normalize=normalize)
        if len(pref) != n_resources:
            raise ValueError(f"Preference vector {pref} has wrong length (expected {n_resources})")
        return [pref]


def create_custom_setup_3d(
    attacker_prefs: np.ndarray,
    defender_prefs: List[np.ndarray],
    n_defenders: int,
    n_resources: int = 3,
):
    """
    Create a test setup with custom preferences.
    
    Args:
        attacker_prefs: Preference vector for the attacker (should be normalized)
        defender_prefs: List of preference vectors for defenders (cycled if fewer than n_defenders)
        n_defenders: Number of defender agents
        n_resources: Number of resources (must match preference vector lengths)
    
    Returns:
        utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources
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
    
    # Supply: 1 unit of each resource (same as original)
    supply = np.ones(n_resources)
    
    # Groups
    attacking_group = {attacker_id}
    victim_group = set(range(n_defenders))
    
    return utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources


def create_test_setup_3d(n_agents: int = 10, random_utilities: bool = False, seed: Optional[int] = None):
    """Create a test setup with 3 resources.
    
    Args:
        n_agents: Number of agents
        random_utilities: If True, use random utilities instead of hardcoded
        seed: Random seed for reproducibility (only used if random_utilities=True)
    """
    n_resources = 3
    
    if random_utilities:
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random utilities using rod-cutting approach
        utilities = np.zeros((n_agents, n_resources))
        for i in range(n_agents):
            cuts = np.sort(np.random.uniform(0, 1, n_resources - 1))
            cuts = np.concatenate([[0], cuts, [1]])
            utilities[i] = np.diff(cuts)
    else:
        # Build utility matrix with hardcoded values
        utilities = np.zeros((n_agents, n_resources))
        
        # Non-attackers: different preferences
        # Split them into groups with different preferences
        for i in range(n_agents - 1):
            if i % 3 == 0:
                utilities[i] = [1.0, 0.0, 0.0]  # Only want R0
            elif i % 3 == 1:
                utilities[i] = [0.0, 1.0, 0.0]  # Only want R1
            else:
                utilities[i] = [0.0, 0.0, 1.0]  # Only want R2
        
        # Attacker (last agent): wants a mix
        attacker_id = n_agents - 1
        utilities[attacker_id] = [0.5, 0.3, 0.2]
    
    # Supply: 1 unit of each resource
    supply = np.ones(n_resources)
    
    # Groups - attacker is always the last agent
    attacker_id = n_agents - 1
    attacking_group = {attacker_id}
    victim_group = set(range(n_agents - 1))
    
    return utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources


def compute_global_optimum_direction(
    utilities: np.ndarray,
    supply: np.ndarray,
    attacking_group: Set[int],
    n_agents: int,
    n_resources: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the global optimum (unconstrained Nash welfare) and extract the attacker's direction.
    
    Returns:
        attacker_direction: Normalized direction vector for the attacker's allocation
        attacker_allocation: The actual allocation for the attacker at the optimum
    """
    from nash_welfare_optimizer import NashWelfareOptimizer
    
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    result = optimizer.solve()
    
    attacker_id = list(attacking_group)[0]
    attacker_allocation = result.allocation[attacker_id].copy()
    
    # Normalize to get direction
    norm = np.linalg.norm(attacker_allocation)
    if norm > 1e-10:
        attacker_direction = attacker_allocation / norm
    else:
        attacker_direction = attacker_allocation.copy()
    
    return attacker_direction, attacker_allocation


def generate_sphere_directions(n_phi: int = 10, n_theta: int = 20, skip_zero_components: bool = False) -> List[Tuple[float, float, np.ndarray]]:
    """
    Generate directions on the positive octant of a sphere.
    
    Uses spherical coordinates:
    - phi: angle from z-axis (0 to pi/2 for positive z)
    - theta: angle in xy-plane from x-axis (0 to pi/2 for positive x and y)
    
    Args:
        n_phi: Number of divisions for phi angle
        n_theta: Number of divisions for theta angle
        skip_zero_components: If True, skip directions where any component is near zero
    
    Returns list of (phi_degrees, theta_degrees, direction_vector) tuples.
    """
    directions = []
    seen_directions = set()  # Track unique directions to avoid duplicates
    
    for i in range(n_phi + 1):
        phi = (np.pi / 2) * i / n_phi  # 0 to pi/2
        phi_deg = np.degrees(phi)
        
        # At phi=0 (z-axis), all theta values give the same point [0,0,1]
        # So only use theta=0 when phi=0
        if i == 0:
            theta_range = [0]
        else:
            theta_range = range(n_theta + 1)
        
        for j in theta_range:
            theta = (np.pi / 2) * j / n_theta  # 0 to pi/2
            theta_deg = np.degrees(theta)
            
            # Spherical to Cartesian (positive octant)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            direction = np.array([x, y, z])
            
            # Skip near-zero directions
            if np.linalg.norm(direction) < 1e-10:
                continue
            
            # Normalize
            direction = direction / np.linalg.norm(direction)
            
            # Skip directions where any component is near zero (on edges/faces of octant)
            if skip_zero_components:
                if any(abs(direction[k]) < 1e-6 for k in range(3)):
                    continue
            
            # Round to avoid floating point duplicates
            dir_key = (round(direction[0], 6), round(direction[1], 6), round(direction[2], 6))
            
            if dir_key in seen_directions:
                continue
            seen_directions.add(dir_key)
            
            directions.append((phi_deg, theta_deg, direction))
    
    return directions


def scale_to_maximum_3d(direction: np.ndarray, supply: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Scale the direction until one component equals its supply (max allocation).
    """
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


def compute_ratio_differences(
    x_full_boundary: np.ndarray,
    utilities: np.ndarray,
    attacking_group: Set[int],
    non_attacking_group: Set[int],
    n_agents: int,
    n_resources: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Optional[int]]]:
    """
    Compute the ratio differences (v_aj/V_a - v_dj/V_d) at a boundary point.
    
    Returns:
        ratio_diffs: Array of (v_aj/V_a - v_dj/V_d) for each resource (using first attacker)
        attacker_ratios: Array of v_aj/V_a for each resource
        dependent_ratios: Array of v_dj/V_d for each resource
        dependent_agents: List of dependent agent for each resource
    """
    # Compute V at boundary
    V_boundary = np.array([np.dot(utilities[i], x_full_boundary[i]) for i in range(n_agents)])
    
    # Compute v_ij/V_i for all agents at boundary
    ratios = np.zeros((n_agents, n_resources))
    for i in range(n_agents):
        V_i = max(V_boundary[i], 1e-10)
        for j in range(n_resources):
            ratios[i, j] = utilities[i, j] / V_i
    
    # Get attacker ratios (use first attacker for now)
    attacker_id = list(attacking_group)[0]
    attacker_ratios = ratios[attacker_id].copy()
    
    # Find dependent agent for each resource
    dependent_agents = [None] * n_resources
    dependent_ratios = np.zeros(n_resources)
    for j in range(n_resources):
        min_ratio = float('inf')
        min_agent = None
        for d in non_attacking_group:
            if x_full_boundary[d, j] > 1e-10:  # Must have allocation
                if ratios[d, j] < min_ratio:
                    min_ratio = ratios[d, j]
                    min_agent = d
        dependent_agents[j] = min_agent
        dependent_ratios[j] = min_ratio if min_ratio < float('inf') else 0.0
    
    # Compute ratio differences
    ratio_diffs = attacker_ratios - dependent_ratios
    
    return ratio_diffs, attacker_ratios, dependent_ratios, dependent_agents


def compute_directional_derivative_at_point(
    x_full: np.ndarray,
    V: np.ndarray,
    utilities: np.ndarray,
    attacking_group: Set[int],
    non_attacking_group: Set[int],
    n_agents: int,
    n_resources: int,
    oracle: DirectionalDerivativeOracle
) -> Optional[float]:
    """
    Compute the directional derivative at a given point using the oracle.
    
    Returns the directional derivative value, or None if computation fails.
    """
    # Use the oracle's method to compute direction
    forward_direction, forward_valid = oracle._compute_direction(x_full, V, backward=False)
    
    if not forward_valid:
        # Try backward
        backward_direction, backward_valid = oracle._compute_direction(x_full, V, backward=True)
        if backward_valid:
            backward_deriv = oracle._compute_directional_derivative_with_direction(x_full, V, backward_direction)
            return -backward_deriv  # Negate because it's backward
        return None
    
    # Compute directional derivative using the oracle
    dir_deriv = oracle._compute_directional_derivative_with_direction(x_full, V, forward_direction)
    return dir_deriv


def probe_direction_3d(
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
    """Probe a single direction to find the boundary point in 3D."""
    result = {
        'direction': direction.copy(),
        'max_attacker_alloc': None,
        'max_scale': None,
        'violated_at_max': None,
        'boundary_attacker_alloc': None,
        'boundary_scale': None,
        'normal_vector': None,
        'ratio_diffs': None,
        'attacker_ratios': None,
        'dependent_ratios': None,
        'dependent_agents': None,
    }
    
    # Scale direction to maximum
    max_attacker_alloc, max_scale = scale_to_maximum_3d(direction, supply)
    result['max_attacker_alloc'] = max_attacker_alloc
    result['max_scale'] = max_scale
    
    # Solve Nash for non-attackers at maximum
    remaining_supply_max = supply - max_attacker_alloc
    remaining_supply_max = np.maximum(remaining_supply_max, 0)
    
    non_attacker_alloc_max = solve_nash_for_non_attackers_standalone(
        utilities, non_attacking_group, remaining_supply_max, n_agents, n_resources
    )
    
    if non_attacker_alloc_max is None:
        return result
    
    # Build full allocation at max
    x_full_max = build_full_allocation(
        max_attacker_alloc, non_attacker_alloc_max, attacker_id, n_agents, n_resources
    )
    x_free_max = x_full_max.flatten()
    
    # Check if violated at max
    violated, normal, rhs = oracle.is_violated(x_free_max)
    result['violated_at_max'] = violated
    
    if not violated:
        result['boundary_attacker_alloc'] = max_attacker_alloc
        result['boundary_scale'] = max_scale
        # Compute ratio differences at max (which is the boundary)
        ratio_diffs, attacker_ratios, dependent_ratios, dependent_agents = compute_ratio_differences(
            x_full_max, utilities, attacking_group, non_attacking_group, n_agents, n_resources
        )
        result['ratio_diffs'] = ratio_diffs
        result['attacker_ratios'] = attacker_ratios
        result['dependent_ratios'] = dependent_ratios
        result['dependent_agents'] = dependent_agents
        
        # Compute non-attacker utilities at max (which is the boundary in this case)
        V_max = np.array([np.dot(utilities[i], x_full_max[i]) for i in range(n_agents)])
        non_attacker_utilities = {i: V_max[i] for i in non_attacking_group}
        result['non_attacker_utilities'] = non_attacker_utilities
        result['total_non_attacker_utility'] = sum(non_attacker_utilities.values())
        result['min_non_attacker_utility'] = min(non_attacker_utilities.values()) if non_attacker_utilities else 0.0
        result['non_attacker_nash_welfare'] = np.prod([V_max[i] for i in non_attacking_group])
        
        return result
    
    # Binary search to find boundary
    t_low, t_high = 0.0, 1.0
    
    if verbose:
        print(f"  Binary search for direction {direction.round(4)}:")
    
    for iteration in range(50):
        t_mid = (t_low + t_high) / 2
        
        attacker_alloc_mid = max_attacker_alloc * t_mid
        remaining_supply_mid = supply - attacker_alloc_mid
        remaining_supply_mid = np.maximum(remaining_supply_mid, 0)
        
        non_attacker_alloc_mid = solve_nash_for_non_attackers_standalone(
            utilities, non_attacking_group, remaining_supply_mid, n_agents, n_resources
        )
        
        if non_attacker_alloc_mid is None:
            if verbose:
                print(f"    iter {iteration}: t={t_mid:.6f} -> Nash solve failed, t_low <- t_mid")
            t_low = t_mid
            continue
        
        x_full_mid = build_full_allocation(
            attacker_alloc_mid, non_attacker_alloc_mid, attacker_id, n_agents, n_resources
        )
        
        V_mid = np.array([np.dot(utilities[i], x_full_mid[i]) for i in range(n_agents)])
        
        if any(V_mid[i] <= 1e-10 for i in non_attacking_group):
            if verbose:
                print(f"    iter {iteration}: t={t_mid:.6f} -> zero utility, t_low <- t_mid")
            t_low = t_mid
            continue
        
        fwd_dir, fwd_valid = oracle._compute_direction(x_full_mid, V_mid, backward=False)
        
        if not fwd_valid:
            if verbose:
                print(f"    iter {iteration}: t={t_mid:.6f} -> fwd invalid, t_high <- t_mid")
            t_high = t_mid
            continue
        
        dir_deriv = oracle._compute_directional_derivative_with_direction(x_full_mid, V_mid, fwd_dir)
        
        # Find dependent agents (agents with negative direction in fwd_dir for each resource)
        dep_agents = []
        for j in range(n_resources):
            deps_for_j = []
            for i in non_attacking_group:
                if fwd_dir[i, j] < -1e-10:
                    deps_for_j.append(i)
            dep_agents.append(deps_for_j if deps_for_j else None)
        
        # Compute ratios for all non-attackers for logging
        if verbose:
            print(f"    iter {iteration}: t={t_mid:.6f}, dir_deriv={dir_deriv:.6e}, dep_agents={dep_agents}, t_low={t_low:.6f}, t_high={t_high:.6f}")
            print(f"      Attacker alloc: {attacker_alloc_mid.round(6)}")
            print(f"      fwd_dir (non-zero entries):")
            for i in range(n_agents):
                if np.any(np.abs(fwd_dir[i]) > 1e-10):
                    role = "ATK" if i == attacker_id else "VIC"
                    print(f"        Agent {i} ({role}): {fwd_dir[i].round(8)}")
            print(f"      Non-attacker ratios (v_ij/V_i) for each resource:")
            for j in range(n_resources):
                ratios_for_resource = []
                for i in non_attacking_group:
                    V_i = max(V_mid[i], 1e-10)
                    ratio = utilities[i, j] / V_i
                    has_alloc = x_full_mid[i, j] > 1e-6  # Match threshold in _compute_direction
                    ratios_for_resource.append((i, ratio, has_alloc, x_full_mid[i, j]))
                # Sort by ratio
                ratios_for_resource.sort(key=lambda x: x[1])
                print(f"        R{j}: ", end="")
                for agent, ratio, has_alloc, alloc in ratios_for_resource:
                    marker = "*" if has_alloc else " "
                    print(f"A{agent}:{ratio:.10f}{marker}({alloc:.6f}) ", end="")
                print()
        
        if abs(dir_deriv) < 1e-8:
            if verbose:
                print(f"    -> converged at t={t_mid:.6f}")
            break
        elif dir_deriv > 0:
            t_low = t_mid
        else:
            t_high = t_mid
    
    boundary_scale = t_mid * max_scale
    boundary_attacker_alloc = max_attacker_alloc * t_mid
    
    result['boundary_attacker_alloc'] = boundary_attacker_alloc
    result['boundary_scale'] = boundary_scale
    
    # Compute normal vector and ratio differences at boundary
    remaining_supply_boundary = supply - boundary_attacker_alloc
    remaining_supply_boundary = np.maximum(remaining_supply_boundary, 0)
    
    non_attacker_alloc_boundary = solve_nash_for_non_attackers_standalone(
        utilities, non_attacking_group, remaining_supply_boundary, n_agents, n_resources
    )
    
    if non_attacker_alloc_boundary is not None:
        x_full_boundary = build_full_allocation(
            boundary_attacker_alloc, non_attacker_alloc_boundary, attacker_id, n_agents, n_resources
        )
        
        result['normal_vector'] = compute_normal_at_boundary(
            x_full_boundary, utilities, attacking_group,
            non_attacking_group, n_agents, n_resources
        )
        
        # Compute ratio differences
        ratio_diffs, attacker_ratios, dependent_ratios, dependent_agents = compute_ratio_differences(
            x_full_boundary, utilities, attacking_group, non_attacking_group, n_agents, n_resources
        )
        result['ratio_diffs'] = ratio_diffs
        result['attacker_ratios'] = attacker_ratios
        result['dependent_ratios'] = dependent_ratios
        result['dependent_agents'] = dependent_agents
        
        # Compute non-attacker utilities at boundary
        V_boundary = np.array([np.dot(utilities[i], x_full_boundary[i]) for i in range(n_agents)])
        non_attacker_utilities = {i: V_boundary[i] for i in non_attacking_group}
        result['non_attacker_utilities'] = non_attacker_utilities
        result['total_non_attacker_utility'] = sum(non_attacker_utilities.values())
        result['min_non_attacker_utility'] = min(non_attacker_utilities.values()) if non_attacker_utilities else 0.0
        result['non_attacker_nash_welfare'] = np.prod([V_boundary[i] for i in non_attacking_group])
    
    return result


def save_results_to_xlsx_3d(
    results: List[Tuple[float, float, Dict]],
    filename: str
):
    """Save 3D probe results to an Excel file."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Boundary Probe 3D"
    
    headers = [
        "Phi (deg)", "Theta (deg)",
        "Dir R0", "Dir R1", "Dir R2",
        "Max Scale",
        "Max R0", "Max R1", "Max R2",
        "Violated",
        "Boundary Scale",
        "Boundary R0", "Boundary R1", "Boundary R2",
        "Normal R0", "Normal R1", "Normal R2",
        "v_a/V_a R0", "v_a/V_a R1", "v_a/V_a R2",
        "v_d/V_d R0", "v_d/V_d R1", "v_d/V_d R2",
        "Ratio Diff R0", "Ratio Diff R1", "Ratio Diff R2",
        "Dep Agent R0", "Dep Agent R1", "Dep Agent R2",
    ]
    ws.append(headers)
    
    for col in range(1, len(headers) + 1):
        ws.cell(row=1, column=col).font = Font(bold=True)
    
    for phi, theta, result in results:
        row = [
            phi, theta,
            result['direction'][0], result['direction'][1], result['direction'][2],
            result['max_scale'],
            result['max_attacker_alloc'][0] if result['max_attacker_alloc'] is not None else None,
            result['max_attacker_alloc'][1] if result['max_attacker_alloc'] is not None else None,
            result['max_attacker_alloc'][2] if result['max_attacker_alloc'] is not None else None,
            result['violated_at_max'],
            result['boundary_scale'],
            result['boundary_attacker_alloc'][0] if result['boundary_attacker_alloc'] is not None else None,
            result['boundary_attacker_alloc'][1] if result['boundary_attacker_alloc'] is not None else None,
            result['boundary_attacker_alloc'][2] if result['boundary_attacker_alloc'] is not None else None,
            result['normal_vector'][0] if result['normal_vector'] is not None else None,
            result['normal_vector'][1] if result['normal_vector'] is not None else None,
            result['normal_vector'][2] if result['normal_vector'] is not None else None,
            result['attacker_ratios'][0] if result['attacker_ratios'] is not None else None,
            result['attacker_ratios'][1] if result['attacker_ratios'] is not None else None,
            result['attacker_ratios'][2] if result['attacker_ratios'] is not None else None,
            result['dependent_ratios'][0] if result['dependent_ratios'] is not None else None,
            result['dependent_ratios'][1] if result['dependent_ratios'] is not None else None,
            result['dependent_ratios'][2] if result['dependent_ratios'] is not None else None,
            result['ratio_diffs'][0] if result['ratio_diffs'] is not None else None,
            result['ratio_diffs'][1] if result['ratio_diffs'] is not None else None,
            result['ratio_diffs'][2] if result['ratio_diffs'] is not None else None,
            result['dependent_agents'][0] if result['dependent_agents'] is not None else None,
            result['dependent_agents'][1] if result['dependent_agents'] is not None else None,
            result['dependent_agents'][2] if result['dependent_agents'] is not None else None,
        ]
        ws.append(row)
    
    wb.save(filename)
    print(f"Results saved to {filename}")


def save_desmos_format(results: List[Tuple[float, float, Dict]], filename: str):
    """
    Save boundary points in a format suitable for Desmos 3D.
    
    Desmos 3D can accept:
    - Points as (x, y, z)
    - Lists for parametric plotting
    """
    boundary_points = []
    max_points = []
    normal_vectors = []  # Normal vectors at each boundary point
    
    for phi, theta, result in results:
        if result['boundary_attacker_alloc'] is not None:
            bp = result['boundary_attacker_alloc']
            boundary_points.append((float(bp[0]), float(bp[1]), float(bp[2])))
            
            # Add normal vector if available
            if result['normal_vector'] is not None:
                nv = result['normal_vector']
                normal_vectors.append((float(nv[0]), float(nv[1]), float(nv[2])))
            else:
                normal_vectors.append((0.0, 0.0, 0.0))
                
        if result['max_attacker_alloc'] is not None:
            mp = result['max_attacker_alloc']
            max_points.append((float(mp[0]), float(mp[1]), float(mp[2])))
    
    with open(filename, 'w') as f:
        f.write("# Desmos 3D Format - Boundary Surface Probe\n")
        f.write("# Copy and paste the lists below into Desmos 3D\n\n")
        
        # Boundary points as separate lists for x, y, z
        f.write("# Boundary Points (x*)\n")
        f.write("# Paste these as three separate lists, then use (B_x, B_y, B_z) to plot\n\n")
        
        bx = [round(p[0], 6) for p in boundary_points]
        by = [round(p[1], 6) for p in boundary_points]
        bz = [round(p[2], 6) for p in boundary_points]
        
        f.write(f"B_x = {bx}\n\n")
        f.write(f"B_y = {by}\n\n")
        f.write(f"B_z = {bz}\n\n")
        
        # Normal vectors at boundary points
        f.write("# Normal Vectors at Boundary Points\n")
        f.write("# These point toward the excluded zone\n")
        f.write("# To visualize: plot vectors from (B_x, B_y, B_z) to (B_x + s*N_x, B_y + s*N_y, B_z + s*N_z)\n")
        f.write("# where s is a scale factor (e.g., 0.1)\n\n")
        
        nx = [round(p[0], 6) for p in normal_vectors]
        ny = [round(p[1], 6) for p in normal_vectors]
        nz = [round(p[2], 6) for p in normal_vectors]
        
        f.write(f"N_x = {nx}\n\n")
        f.write(f"N_y = {ny}\n\n")
        f.write(f"N_z = {nz}\n\n")
        
        # Endpoints of normal vectors (for visualization)
        f.write("# Normal vector endpoints (B + 0.1*N) for visualization\n")
        scale = 0.1
        ex = [round(boundary_points[i][0] + scale * normal_vectors[i][0], 6) for i in range(len(boundary_points))]
        ey = [round(boundary_points[i][1] + scale * normal_vectors[i][1], 6) for i in range(len(boundary_points))]
        ez = [round(boundary_points[i][2] + scale * normal_vectors[i][2], 6) for i in range(len(boundary_points))]
        
        f.write(f"E_x = {ex}\n\n")
        f.write(f"E_y = {ey}\n\n")
        f.write(f"E_z = {ez}\n\n")
        
        # Also output as individual points for easier copy-paste
        f.write("# Individual boundary points with normals (for manual entry):\n")
        for i, (bp, nv) in enumerate(zip(boundary_points, normal_vectors)):
            f.write(f"# P_{i} = ({bp[0]:.6f}, {bp[1]:.6f}, {bp[2]:.6f}), normal = ({nv[0]:.6f}, {nv[1]:.6f}, {nv[2]:.6f})\n")
        
        f.write("\n# Max points (search limits)\n")
        f.write("# M_x, M_y, M_z lists:\n\n")
        
        mx = [round(float(p[0]), 6) for p in max_points]
        my = [round(float(p[1]), 6) for p in max_points]
        mz = [round(float(p[2]), 6) for p in max_points]
        
        f.write(f"M_x = {mx}\n\n")
        f.write(f"M_y = {my}\n\n")
        f.write(f"M_z = {mz}\n\n")
    
    print(f"Desmos format saved to {filename}")


def check_point_invalidated(point: np.ndarray, all_boundary_points: List[np.ndarray], 
                            all_normal_vectors: List[np.ndarray], tolerance: float = 1e-6) -> bool:
    """
    Check if a boundary point is invalidated by any other cutting plane.
    
    A point is invalidated if normal · point > normal · boundary_point for some other plane.
    (i.e., the constraint normal · x <= rhs is violated)
    
    Returns True if the point is invalidated by at least one plane.
    """
    point = np.array(point)
    
    for i, (bp, nv) in enumerate(zip(all_boundary_points, all_normal_vectors)):
        bp = np.array(bp)
        nv = np.array(nv)
        
        # Skip if this is the same point or normal is zero
        if np.linalg.norm(point - bp) < tolerance:
            continue
        if np.linalg.norm(nv) < tolerance:
            continue
        
        # Check if point violates the constraint: normal · x <= normal · bp
        rhs = np.dot(nv, bp)
        lhs = np.dot(nv, point)
        
        if lhs > rhs + tolerance:
            return True  # Point is invalidated by this plane
    
    return False


def validate_all_boundary_points(boundary_points: List[np.ndarray], 
                                  normal_vectors: List[np.ndarray]) -> List[bool]:
    """
    Check each boundary point against all cutting planes.
    
    Returns a list of booleans where True means the point was invalidated.
    """
    invalidated = []
    
    for i, point in enumerate(boundary_points):
        is_invalid = check_point_invalidated(point, boundary_points, normal_vectors)
        invalidated.append(is_invalid)
    
    return invalidated


def plot_boundary_3d(results: List[Tuple[float, float, Dict]], filename: str, 
                     optimum_result: Optional[Tuple[np.ndarray, np.ndarray, bool]] = None,
                     supply: Optional[np.ndarray] = None):
    """Plot the boundary points in 3D with normal vectors, colored by validation status."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use fixed max_range of 1.0 (unit supply)
    max_range = 1.0
    
    boundary_points = []
    normal_vectors = []
    max_points = []
    
    for phi, theta, result in results:
        if result['boundary_attacker_alloc'] is not None:
            boundary_points.append(result['boundary_attacker_alloc'])
            if result['normal_vector'] is not None:
                normal_vectors.append(result['normal_vector'])
            else:
                normal_vectors.append(np.zeros(3))
        if result['max_attacker_alloc'] is not None:
            max_points.append(result['max_attacker_alloc'])
    
    # Validate all boundary points against all cutting planes
    print("Validating boundary points against all cutting planes...")
    invalidated = validate_all_boundary_points(boundary_points, normal_vectors)
    n_invalid = sum(invalidated)
    n_valid = len(invalidated) - n_invalid
    print(f"  Valid (blue): {n_valid}, Invalidated (red): {n_invalid}")
    
    # Separate valid and invalid points
    valid_points = [bp for bp, inv in zip(boundary_points, invalidated) if not inv]
    invalid_points = [bp for bp, inv in zip(boundary_points, invalidated) if inv]
    
    # Plot valid points in blue
    if valid_points:
        vp = np.array(valid_points)
        ax.scatter(vp[:, 0], vp[:, 1], vp[:, 2], c='blue', s=50, marker='o', label=f'Valid boundary ({n_valid})')
    
    # Plot invalid points in red
    if invalid_points:
        ip = np.array(invalid_points)
        ax.scatter(ip[:, 0], ip[:, 1], ip[:, 2], c='red', s=50, marker='o', label=f'Invalidated ({n_invalid})')
    
    if max_points:
        mp = np.array(max_points)
        ax.scatter(mp[:, 0], mp[:, 1], mp[:, 2], c='gray', s=30, marker='x', alpha=0.5, label='Max (search limit)')
    
    # Draw normal vectors at boundary points (every Nth point to avoid clutter)
    if boundary_points and normal_vectors:
        scale = 0.1  # Scale factor for normal vectors
        step = max(1, len(boundary_points) // 30)  # Show ~30 normal vectors
        for i in range(0, len(boundary_points), step):
            bp = boundary_points[i]
            nv = normal_vectors[i]
            if np.linalg.norm(nv) > 1e-10:  # Only draw non-zero normals
                # Color normal vector based on whether its point is valid
                nv_color = 'darkred' if invalidated[i] else 'darkgreen'
                ax.quiver(bp[0], bp[1], bp[2], 
                         nv[0]*scale, nv[1]*scale, nv[2]*scale,
                         color=nv_color, alpha=0.7, arrow_length_ratio=0.3)
    
    # Draw lines from origin to some boundary points
    if boundary_points:
        for i in range(0, len(boundary_points), max(1, len(boundary_points) // 20)):
            bp = boundary_points[i]
            line_color = 'red' if invalidated[i] else 'blue'
            ax.plot([0, bp[0]], [0, bp[1]], [0, bp[2]], color=line_color, alpha=0.2)
    
    # Plot global optimum point (yellow if valid, orange if invalidated)
    if optimum_result is not None:
        opt_boundary, opt_allocation, opt_invalidated = optimum_result
        if opt_boundary is not None:
            opt_color = 'orange' if opt_invalidated else 'yellow'
            opt_label = 'Global optimum (INVALID)' if opt_invalidated else 'Global optimum (valid)'
            ax.scatter([opt_boundary[0]], [opt_boundary[1]], [opt_boundary[2]], 
                      c=opt_color, s=200, marker='*', edgecolors='black', linewidths=1,
                      label=opt_label, zorder=10)
            # Draw line from origin to optimum
            ax.plot([0, opt_boundary[0]], [0, opt_boundary[1]], [0, opt_boundary[2]], 
                   color=opt_color, linewidth=2, alpha=0.8)
            print(f"  Global optimum boundary point: {opt_boundary.round(6)}")
            print(f"  Global optimum invalidated: {opt_invalidated}")
    
    # Origin
    ax.scatter([0], [0], [0], c='green', s=100, marker='s', label='Origin')
    
    ax.set_xlabel('Attacker R0 Allocation')
    ax.set_ylabel('Attacker R1 Allocation')
    ax.set_zlabel('Attacker R2 Allocation')
    ax.set_title('Boundary Surface Probe (3D)\n(Attacker Allocation Space)')
    ax.legend()
    
    # Set equal aspect ratio (max_range already set at top of function from supply)
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.show()


def plot_non_attacker_utility_surfaces(results: List[Tuple[float, float, Dict]], filename_prefix: str):
    """
    Plot 3D surfaces showing non-attacker utilities as a function of normalized attacker direction.
    
    Creates two plots:
    1. Total non-attacker utility (sum) vs normalized direction
    2. Minimum non-attacker utility vs normalized direction
    
    X, Y are the first two components of the normalized boundary allocation (sums to 1, on simplex)
    Z is the utility metric at the boundary point x*
    """
    # Collect data points
    simplex_x = []  # First component of normalized boundary allocation
    simplex_y = []  # Second component of normalized boundary allocation
    total_utilities = []
    min_utilities = []
    nash_welfares = []
    
    for phi, theta, result in results:
        if result['boundary_attacker_alloc'] is not None and result.get('total_non_attacker_utility') is not None:
            # Get the boundary allocation and normalize to sum to 1 (simplex)
            boundary_alloc = result['boundary_attacker_alloc']
            alloc_sum = np.sum(boundary_alloc)
            if alloc_sum > 1e-10:
                normalized_alloc = boundary_alloc / alloc_sum
                simplex_x.append(normalized_alloc[0])
                simplex_y.append(normalized_alloc[1])
                total_utilities.append(result['total_non_attacker_utility'])
                min_utilities.append(result['min_non_attacker_utility'])
                nash_welfares.append(result.get('non_attacker_nash_welfare', 0.0))
    
    if not simplex_x:
        print("No valid data points for non-attacker utility plots")
        return
    
    simplex_x = np.array(simplex_x)
    simplex_y = np.array(simplex_y)
    total_utilities = np.array(total_utilities)
    min_utilities = np.array(min_utilities)
    nash_welfares = np.array(nash_welfares)
    
    # Plot 1: Total non-attacker utility
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    scatter1 = ax1.scatter(simplex_x, simplex_y, total_utilities, 
                           c=total_utilities, cmap='viridis', s=50)
    
    ax1.set_xlabel('Normalized Boundary Alloc[0]')
    ax1.set_ylabel('Normalized Boundary Alloc[1]')
    ax1.set_zlabel('Total Non-Attacker Utility')
    ax1.set_title('Total Non-Attacker Utility vs Attacker Boundary Allocation\n(Allocation normalized to sum to 1)')
    fig1.colorbar(scatter1, ax=ax1, label='Total Utility')
    
    plt.tight_layout()
    filename1 = f"{filename_prefix}_total_utility.png"
    plt.savefig(filename1, dpi=150)
    print(f"Total utility plot saved to {filename1}")
    plt.show()
    
    # Plot 2: Minimum non-attacker utility
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    scatter2 = ax2.scatter(simplex_x, simplex_y, min_utilities, 
                           c=min_utilities, cmap='plasma', s=50)
    
    ax2.set_xlabel('Normalized Boundary Alloc[0]')
    ax2.set_ylabel('Normalized Boundary Alloc[1]')
    ax2.set_zlabel('Min Non-Attacker Utility')
    ax2.set_title('Minimum Non-Attacker Utility vs Attacker Boundary Allocation\n(Allocation normalized to sum to 1)')
    fig2.colorbar(scatter2, ax=ax2, label='Min Utility')
    
    plt.tight_layout()
    filename2 = f"{filename_prefix}_min_utility.png"
    plt.savefig(filename2, dpi=150)
    print(f"Min utility plot saved to {filename2}")
    plt.show()
    
    # Plot 3: Non-attacker Nash welfare (product of utilities)
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Use log scale for Nash welfare since it can vary widely
    log_nash = np.log(nash_welfares + 1e-20)
    
    scatter3 = ax3.scatter(simplex_x, simplex_y, log_nash, 
                           c=log_nash, cmap='coolwarm', s=50)
    
    ax3.set_xlabel('Normalized Boundary Alloc[0]')
    ax3.set_ylabel('Normalized Boundary Alloc[1]')
    ax3.set_zlabel('Log(Non-Attacker Nash Welfare)')
    ax3.set_title('Non-Attacker Nash Welfare vs Attacker Boundary Allocation\n(Allocation normalized to sum to 1)')
    fig3.colorbar(scatter3, ax=ax3, label='Log(Nash Welfare)')
    
    plt.tight_layout()
    filename3 = f"{filename_prefix}_nash_welfare.png"
    plt.savefig(filename3, dpi=150)
    print(f"Nash welfare plot saved to {filename3}")
    plt.show()
    
    # Print summary statistics
    print()
    print("=" * 60)
    print("NON-ATTACKER UTILITY STATISTICS")
    print("=" * 60)
    print(f"Number of data points: {len(total_utilities)}")
    print()
    print("Total Non-Attacker Utility:")
    print(f"  Min: {np.min(total_utilities):.6f}")
    print(f"  Max: {np.max(total_utilities):.6f}")
    print(f"  Mean: {np.mean(total_utilities):.6f}")
    print()
    print("Minimum Non-Attacker Utility:")
    print(f"  Min: {np.min(min_utilities):.6f}")
    print(f"  Max: {np.max(min_utilities):.6f}")
    print(f"  Mean: {np.mean(min_utilities):.6f}")
    print()
    print("Non-Attacker Nash Welfare:")
    print(f"  Min: {np.min(nash_welfares):.6e}")
    print(f"  Max: {np.max(nash_welfares):.6e}")
    print(f"  Mean: {np.mean(nash_welfares):.6e}")
    
    # Find the boundary allocation that minimizes non-attacker welfare (worst for defenders)
    worst_idx = np.argmin(nash_welfares)
    print()
    print("Boundary allocation that MINIMIZES non-attacker Nash welfare (worst for defenders):")
    print(f"  Normalized allocation: [{simplex_x[worst_idx]:.4f}, {simplex_y[worst_idx]:.4f}, {1-simplex_x[worst_idx]-simplex_y[worst_idx]:.4f}]")
    print(f"  Nash welfare: {nash_welfares[worst_idx]:.6e}")
    print(f"  Min utility: {min_utilities[worst_idx]:.6f}")
    print(f"  Total utility: {total_utilities[worst_idx]:.6f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Probe boundary surface in 3D')
    parser.add_argument('--n-agents', type=int, default=10, help='Number of agents (ignored if using custom prefs)')
    parser.add_argument('--n-phi', type=int, default=8, help='Number of phi divisions')
    parser.add_argument('--n-theta', type=int, default=16, help='Number of theta divisions')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--random-utilities', action='store_true', help='Use random utilities instead of hardcoded')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (only used with --random-utilities)')
    # Custom preference arguments
    parser.add_argument('--attacker-prefs', type=str, default=None,
                        help='Custom attacker preferences (comma-separated, e.g., "0.9,0.1,0.0")')
    parser.add_argument('--defender-prefs', type=str, default=None,
                        help='Custom defender preferences. Single: "1,0,0". Multiple (cycled): "1,0,0;0,1,0;0,0,1"')
    parser.add_argument('--n-defenders', type=int, default=None,
                        help='Number of defenders (required if using custom prefs)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("3D BOUNDARY SURFACE PROBE")
    print("=" * 60)
    
    # Check if using custom preferences
    use_custom = args.attacker_prefs is not None or args.defender_prefs is not None
    
    if use_custom:
        if args.attacker_prefs is None or args.defender_prefs is None or args.n_defenders is None:
            print("Error: When using custom preferences, must specify --attacker-prefs, --defender-prefs, and --n-defenders")
            return
        
        n_resources = 3
        
        # Parse raw first to show what was input
        attacker_prefs_raw = parse_preference_vector(args.attacker_prefs, normalize=False)
        attacker_prefs = parse_preference_vector(args.attacker_prefs, normalize=True)
        if len(attacker_prefs) != n_resources:
            print(f"Error: Attacker preferences must have {n_resources} values, got {len(attacker_prefs)}")
            return
        
        defender_prefs_raw = parse_defender_prefs(args.defender_prefs, n_resources, normalize=False)
        defender_prefs = parse_defender_prefs(args.defender_prefs, n_resources, normalize=True)
        
        print()
        print("CUSTOM PREFERENCES MODE")
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
        
        utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources = \
            create_custom_setup_3d(attacker_prefs, defender_prefs, args.n_defenders, n_resources)
    else:
        # Setup using original method
        utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources = \
            create_test_setup_3d(args.n_agents, random_utilities=args.random_utilities, seed=args.seed)
    
    non_attacking_group = set(range(n_agents)) - attacking_group
    
    print(f"n_agents: {n_agents}")
    print(f"n_resources: {n_resources}")
    if not use_custom:
        print(f"random_utilities: {args.random_utilities}")
        if args.seed is not None:
            print(f"seed: {args.seed}")
    print(f"supply: {supply}")
    print(f"Attacker (Agent {attacker_id}): utilities = {utilities[attacker_id].round(6)}")
    print(f"Non-attacker utilities:")
    for i in range(min(5, n_agents - 1)):
        print(f"  Agent {i}: {utilities[i].round(6)}")
    if n_agents - 1 > 5:
        print(f"  ... ({n_agents - 1 - 5} more agents)")
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
    
    # Check if the global optimum direction is already in the list
    opt_dir_key = (round(optimum_direction[0], 6), round(optimum_direction[1], 6), round(optimum_direction[2], 6))
    existing_keys = {}
    for idx, (_, _, d) in enumerate(directions):
        key = (round(d[0], 6), round(d[1], 6), round(d[2], 6))
        existing_keys[key] = idx
    
    optimum_direction_idx = None  # Index of optimum direction in results (if it exists)
    
    if opt_dir_key not in existing_keys and np.linalg.norm(optimum_direction) > 1e-10:
        # Add optimum direction with dummy phi/theta
        directions.append((999.0, 999.0, optimum_direction))  # 999 marks it as the optimum
        optimum_direction_idx = len(directions) - 1
        print(f"Added global optimum direction to test set")
    elif opt_dir_key in existing_keys:
        optimum_direction_idx = existing_keys[opt_dir_key]
        print(f"Global optimum direction already in test set at index {optimum_direction_idx}")
    else:
        print(f"Global optimum direction has zero norm, skipping")
    
    print(f"Probing {len(directions)} directions...")
    print()
    
    # Probe each direction
    results = []
    optimum_result_data = None
    results_idx_for_optimum = None  # Track which index in results corresponds to optimum
    
    for idx, (phi, theta, direction) in enumerate(directions):
        if idx % 20 == 0:
            print(f"Progress: {idx}/{len(directions)}")
        
        result = probe_direction_3d(
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
        
        # Compute directional derivative at the boundary point
        if result['boundary_attacker_alloc'] is not None:
            boundary_alloc = result['boundary_attacker_alloc']
            remaining_supply = supply - boundary_alloc
            remaining_supply = np.maximum(remaining_supply, 0)
            non_attacker_alloc = solve_nash_for_non_attackers_standalone(
                utilities, non_attacking_group, remaining_supply, n_agents, n_resources
            )
            if non_attacker_alloc is not None:
                x_full = build_full_allocation(boundary_alloc, non_attacker_alloc, attacker_id, n_agents, n_resources)
                V = np.array([np.dot(utilities[i], x_full[i]) for i in range(n_agents)])
                
                dir_deriv = compute_directional_derivative_at_point(
                    x_full, V, utilities, attacking_group, non_attacking_group, n_agents, n_resources, oracle
                )
                result['directional_derivative'] = dir_deriv
            else:
                result['directional_derivative'] = None
        else:
            result['directional_derivative'] = None
        
        # Check if this is the optimum direction (either added with 999.0 or matched existing)
        is_optimum = (phi == 999.0 and theta == 999.0) or (idx == optimum_direction_idx)
        
        if is_optimum:
            optimum_result_data = result
            results_idx_for_optimum = len(results) if phi != 999.0 else None
            print(f"\n*** GLOBAL OPTIMUM DIRECTION RESULT ***")
            print(f"  Direction: {direction.round(6)}")
            print(f"  Boundary point: {result['boundary_attacker_alloc']}")
            print(f"  Violated at max: {result['violated_at_max']}")
            print(f"  Directional derivative: {result.get('directional_derivative', 'N/A')}")
            print()
        
        # Add to results (even if it's the optimum, so we get its cutting plane)
        if phi != 999.0:  # Only add regular directions to results
            results.append((phi, theta, result))
        else:
            # For the added optimum direction, still add it to results
            results.append((phi, theta, result))
    
    print(f"Progress: {len(directions)}/{len(directions)}")
    print()
    
    # Check if optimum boundary point is invalidated by other cutting planes
    optimum_plot_data = None
    if optimum_result_data is not None and optimum_result_data['boundary_attacker_alloc'] is not None:
        opt_boundary = optimum_result_data['boundary_attacker_alloc']
        
        # Collect all normal vectors from other results
        all_normals = []
        all_boundary_points = []
        for _, _, r in results:
            if r['boundary_attacker_alloc'] is not None and r['normal_vector'] is not None:
                all_boundary_points.append(r['boundary_attacker_alloc'])
                all_normals.append(r['normal_vector'])
        
        # Check if optimum is invalidated
        opt_invalidated = False
        for bp, nv in zip(all_boundary_points, all_normals):
            if np.linalg.norm(nv) > 1e-10:
                # Check if opt_boundary violates the constraint defined by this cutting plane
                # Constraint: nv · x <= nv · bp
                rhs = np.dot(nv, bp)
                lhs = np.dot(nv, opt_boundary)
                if lhs > rhs + 1e-6:
                    opt_invalidated = True
                    break
        
        optimum_plot_data = (opt_boundary, optimum_allocation, opt_invalidated)
        print(f"Global optimum boundary point invalidated by other planes: {opt_invalidated}")
    
    # Save results
    xlsx_filename = f"{args.output_dir}/boundary_probe_3d_n{args.n_agents}.xlsx"
    save_results_to_xlsx_3d(results, xlsx_filename)
    
    desmos_filename = f"{args.output_dir}/boundary_probe_3d_desmos.txt"
    save_desmos_format(results, desmos_filename)
    
    # Plot
    plot_filename = f"{args.output_dir}/boundary_probe_3d_n{args.n_agents}.png"
    plot_boundary_3d(results, plot_filename, optimum_result=optimum_plot_data, supply=supply)
    
    # Plot non-attacker utility surfaces
    utility_plot_prefix = f"{args.output_dir}/boundary_probe_3d_n{args.n_agents}_non_attacker"
    plot_non_attacker_utility_surfaces(results, utility_plot_prefix)
    
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
    if optimum_plot_data is not None:
        print(f"Global optimum invalidated: {optimum_plot_data[2]}")
    
    # Directional derivative statistics
    dir_derivs = [r.get('directional_derivative') for _, _, r in results if r.get('directional_derivative') is not None]
    if dir_derivs:
        print()
        print("Directional derivative at boundary points:")
        print(f"  Count: {len(dir_derivs)}")
        print(f"  Min: {min(dir_derivs):.6f}")
        print(f"  Max: {max(dir_derivs):.6f}")
        print(f"  Mean: {np.mean(dir_derivs):.6f}")
        print(f"  Std: {np.std(dir_derivs):.6f}")
        
        # Count how many are close to zero (should all be ~0 at boundary)
        near_zero = sum(1 for d in dir_derivs if abs(d) < 1e-4)
        print(f"  Near zero (|d| < 1e-4): {near_zero}/{len(dir_derivs)}")
        
        # Show the worst offenders
        if any(abs(d) > 1e-4 for d in dir_derivs):
            print()
            print("  WARNING: Some boundary points have non-zero directional derivative!")
            print("  Top 5 largest |dir_deriv|:")
            sorted_results = sorted([(phi, theta, r) for phi, theta, r in results if r.get('directional_derivative') is not None],
                                   key=lambda x: abs(x[2].get('directional_derivative', 0)), reverse=True)
            for phi, theta, r in sorted_results[:5]:
                dd = r.get('directional_derivative', 0)
                bp = r.get('boundary_attacker_alloc')
                print(f"    phi={phi:.1f}, theta={theta:.1f}: dir_deriv={dd:.6f}, boundary={bp.round(4) if bp is not None else None}")
    
    # Convexity check
    print()
    print("=" * 60)
    print("CONVEXITY CHECK")
    print("=" * 60)
    check_boundary_convexity(results)


def check_boundary_convexity(results: List[Tuple[float, float, Dict]]):
    """
    Check if the boundary surface is convex.
    
    For a convex feasible region, every boundary point should satisfy all cutting planes
    defined by other boundary points. That is, for each pair of boundary points (p1, p2)
    with normal n1 at p1, we should have: n1 · p2 <= n1 · p1
    
    Also checks that the origin is inside all cutting planes (since origin should be feasible).
    """
    # Collect boundary points and their normals
    boundary_data = []
    for phi, theta, r in results:
        if r['boundary_attacker_alloc'] is not None and r['normal_vector'] is not None:
            bp = r['boundary_attacker_alloc']
            nv = r['normal_vector']
            if np.linalg.norm(nv) > 1e-10:
                boundary_data.append((bp, nv, phi, theta))
    
    if len(boundary_data) < 2:
        print("Not enough boundary points with normals to check convexity")
        return
    
    print(f"Checking {len(boundary_data)} boundary points...")
    
    # Check 1: Origin should be inside all cutting planes
    # Constraint is: n · x <= n · bp, so for origin (x=0): 0 <= n · bp
    origin_violations = []
    for bp, nv, phi, theta in boundary_data:
        rhs = np.dot(nv, bp)
        if rhs < -1e-6:  # Origin violates this constraint
            origin_violations.append((phi, theta, bp, nv, rhs))
    
    if origin_violations:
        print(f"\n  WARNING: Origin violates {len(origin_violations)} cutting planes!")
        print("  This suggests normals may be pointing the wrong direction.")
        for phi, theta, bp, nv, rhs in origin_violations[:5]:
            print(f"    phi={phi:.1f}, theta={theta:.1f}: n·bp = {rhs:.6f} < 0")
            print(f"      boundary={bp.round(4)}, normal={nv.round(4)}")
    else:
        print(f"  Origin is inside all {len(boundary_data)} cutting planes: OK")
    
    # Check 2: Each boundary point should be inside all OTHER cutting planes
    # (or on the boundary, i.e., n · p2 <= n · p1 + epsilon)
    convexity_violations = []
    for i, (bp1, nv1, phi1, theta1) in enumerate(boundary_data):
        rhs1 = np.dot(nv1, bp1)
        for j, (bp2, nv2, phi2, theta2) in enumerate(boundary_data):
            if i == j:
                continue
            lhs = np.dot(nv1, bp2)
            if lhs > rhs1 + 1e-4:  # bp2 violates the cutting plane from bp1
                violation = lhs - rhs1
                convexity_violations.append((i, j, phi1, theta1, phi2, theta2, violation, bp1, bp2, nv1))
    
    if convexity_violations:
        print(f"\n  WARNING: {len(convexity_violations)} convexity violations detected!")
        print("  This means the boundary is NOT convex (some boundary points cut off others).")
        print("\n  Top 10 worst violations:")
        convexity_violations.sort(key=lambda x: x[6], reverse=True)
        for i, j, phi1, theta1, phi2, theta2, violation, bp1, bp2, nv1 in convexity_violations[:10]:
            print(f"    Plane at ({phi1:.1f}, {theta1:.1f}) cuts off point at ({phi2:.1f}, {theta2:.1f})")
            print(f"      violation = {violation:.6f}")
            print(f"      bp1={bp1.round(4)}, bp2={bp2.round(4)}")
            print(f"      normal={nv1.round(4)}")
    else:
        print(f"  All {len(boundary_data)} boundary points are mutually consistent: CONVEX")
    
    # Check 3: Normals should point "outward" - away from origin
    # n · bp should be positive (normal points away from origin toward the boundary)
    inward_normals = []
    for bp, nv, phi, theta in boundary_data:
        dot = np.dot(nv, bp)
        if dot < 1e-6:
            inward_normals.append((phi, theta, bp, nv, dot))
    
    if inward_normals:
        print(f"\n  WARNING: {len(inward_normals)} normals may be pointing inward!")
        for phi, theta, bp, nv, dot in inward_normals[:5]:
            print(f"    phi={phi:.1f}, theta={theta:.1f}: n·bp = {dot:.6f}")
            print(f"      boundary={bp.round(4)}, normal={nv.round(4)}")


if __name__ == "__main__":
    main()