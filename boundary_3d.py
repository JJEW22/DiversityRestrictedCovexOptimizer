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


def create_test_setup_3d(n_agents: int = 10):
    """Create a test setup with 3 resources."""
    n_resources = 3
    
    # Build utility matrix
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
    
    # Groups
    attacking_group = {attacker_id}
    victim_group = set(range(n_agents - 1))
    
    return utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources


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
        return result
    
    # Binary search to find boundary
    t_low, t_high = 0.0, 1.0
    
    for iteration in range(50):
        t_mid = (t_low + t_high) / 2
        
        attacker_alloc_mid = max_attacker_alloc * t_mid
        remaining_supply_mid = supply - attacker_alloc_mid
        remaining_supply_mid = np.maximum(remaining_supply_mid, 0)
        
        non_attacker_alloc_mid = solve_nash_for_non_attackers_standalone(
            utilities, non_attacking_group, remaining_supply_mid, n_agents, n_resources
        )
        
        if non_attacker_alloc_mid is None:
            t_low = t_mid
            continue
        
        x_full_mid = build_full_allocation(
            attacker_alloc_mid, non_attacker_alloc_mid, attacker_id, n_agents, n_resources
        )
        
        V_mid = np.array([np.dot(utilities[i], x_full_mid[i]) for i in range(n_agents)])
        
        if any(V_mid[i] <= 1e-10 for i in non_attacking_group):
            t_low = t_mid
            continue
        
        fwd_dir, fwd_valid = oracle._compute_direction(x_full_mid, V_mid, backward=False)
        
        if not fwd_valid:
            t_high = t_mid
            continue
        
        dir_deriv = oracle._compute_directional_derivative_with_direction(x_full_mid, V_mid, fwd_dir)
        
        if abs(dir_deriv) < 1e-8:
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


def plot_boundary_3d(results: List[Tuple[float, float, Dict]], filename: str):
    """Plot the boundary points in 3D with normal vectors, colored by validation status."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
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
    
    # Origin
    ax.scatter([0], [0], [0], c='green', s=100, marker='s', label='Origin')
    
    ax.set_xlabel('Attacker R0 Allocation')
    ax.set_ylabel('Attacker R1 Allocation')
    ax.set_zlabel('Attacker R2 Allocation')
    ax.set_title('Boundary Surface Probe (3D)\n(Attacker Allocation Space)')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = 1.0
    ax.set_xlim(0, max_range)
    ax.set_ylim(0, max_range)
    ax.set_zlim(0, max_range)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Probe boundary surface in 3D')
    parser.add_argument('--n-agents', type=int, default=10, help='Number of agents')
    parser.add_argument('--n-phi', type=int, default=8, help='Number of phi divisions')
    parser.add_argument('--n-theta', type=int, default=16, help='Number of theta divisions')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("3D BOUNDARY SURFACE PROBE")
    print("=" * 60)
    
    # Setup
    utilities, supply, attacking_group, victim_group, attacker_id, n_agents, n_resources = \
        create_test_setup_3d(args.n_agents)
    
    non_attacking_group = set(range(n_agents)) - attacking_group
    
    print(f"n_agents: {n_agents}")
    print(f"n_resources: {n_resources}")
    print(f"Attacker (Agent {attacker_id}): utilities = {utilities[attacker_id]}")
    print(f"Non-attacker utilities:")
    for i in range(min(3, n_agents - 1)):
        print(f"  Agent {i}: {utilities[i]}")
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
    print(f"Probing {len(directions)} directions...")
    print()
    
    # Probe each direction
    results = []
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
        results.append((phi, theta, result))
    
    print(f"Progress: {len(directions)}/{len(directions)}")
    print()
    
    # Save results
    xlsx_filename = f"{args.output_dir}/boundary_probe_3d_n{args.n_agents}.xlsx"
    save_results_to_xlsx_3d(results, xlsx_filename)
    
    desmos_filename = f"{args.output_dir}/boundary_probe_3d_desmos.txt"
    save_desmos_format(results, desmos_filename)
    
    # Plot
    plot_filename = f"{args.output_dir}/boundary_probe_3d_n{args.n_agents}.png"
    plot_boundary_3d(results, plot_filename)
    
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


if __name__ == "__main__":
    main()