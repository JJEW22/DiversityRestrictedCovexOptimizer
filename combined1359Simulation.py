"""
Simulation script for p-MON and q-NEE experiments with worst-case q-NEE.

Runs simulations with:
- 10 agents
- Attacker sizes: {1, 3, 5, 9}
- Defender sizes: {1, 3, 5, 9}
- Valid pairs: attacker_size + defender_size <= 10
- Resource counts: {2, 5, 10}
- 2 random seeds per configuration

Computes:
- p-MON for attackers (best-case constraint for attackers)
- q-NEE for defenders (using the p-MON optimal constraint)
- Worst-case q-NEE for defenders (using PGD to find the worst constraint)

Logs results to spreadsheet every 6 simulations.
"""

import numpy as np
import time
import itertools
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import os
from datetime import datetime

# Import from nash_welfare_optimizer
from nash_welfare_optimizer import (
    NashWelfareOptimizer,
    compute_optimal_self_benefit_constraint,
    solve_optimal_harm_constraint_pgd,
)


@dataclass
class SimulationResult:
    """Result from a single simulation."""
    attacker_size: int
    defender_size: int
    n_resources: int
    seed: int
    attacking_group: Set[int]
    defending_group: Set[int]
    
    # p-MON optimization results (best for attackers)
    pmon_converged: bool
    pmon_time_seconds: float
    pmon_n_iterations: int
    p_mon_attacking_group: float
    q_nee_defending_group: float  # q-NEE under p-MON optimal constraint
    q_nee_non_attackers: float
    p_mon_individual: Dict[int, float]
    q_nee_individual: Dict[int, float]  # Under p-MON constraint
    V_initial: np.ndarray
    V_pmon_final: np.ndarray
    nash_welfare_initial: float
    nash_welfare_pmon_final: float
    
    # Worst-case q-NEE optimization results (worst for defenders)
    qnee_worst_converged: bool
    qnee_worst_time_seconds: float
    qnee_worst_iterations: int
    q_nee_worst_defending_group: float  # The worst-case q-NEE
    q_nee_worst_individual: Dict[int, float]  # Under worst-case constraint
    p_mon_under_qnee_worst: float  # p-MON for attackers under worst-case q-NEE constraint
    p_mon_under_qnee_worst_individual: Dict[int, float]  # Individual p-MON under worst-case constraint
    V_qnee_worst_final: np.ndarray
    nash_welfare_qnee_worst_final: float
    defender_welfare_initial: float
    defender_welfare_qnee_worst_final: float


def generate_utilities_rod_cutting(n_agents: int, n_resources: int, seed: int) -> np.ndarray:
    """Generate random utility matrix using rod-cutting approach."""
    np.random.seed(seed)
    utilities = np.zeros((n_agents, n_resources))
    for i in range(n_agents):
        cuts = np.sort(np.random.uniform(0, 1, n_resources - 1))
        cuts = np.concatenate([[0], cuts, [1]])
        utilities[i] = np.diff(cuts)
    return utilities


def select_groups(n_agents: int, attacker_size: int, defender_size: int, seed: int) -> Tuple[Set[int], Set[int]]:
    """Randomly select attacking and defending groups."""
    np.random.seed(seed + 10000)  # Different seed for group selection
    all_agents = list(range(n_agents))
    np.random.shuffle(all_agents)
    
    attacking_group = set(all_agents[:attacker_size])
    defending_group = set(all_agents[attacker_size:attacker_size + defender_size])
    
    return attacking_group, defending_group


def geometric_mean(values: List[float]) -> float:
    """Compute geometric mean of a list of positive values."""
    if not values:
        return float('nan')
    product = 1.0
    for v in values:
        if v <= 0:
            return float('nan')
        product *= v
    return product ** (1.0 / len(values))


def run_simulation(attacker_size: int, defender_size: int, n_resources: int, 
                   seed: int, n_agents: int = 10, verbose: bool = False,
                   use_projection: bool = True, use_integral_method: bool = False,
                   debug: bool = False,
                   pgd_max_iterations: int = 100,
                   pgd_step_size: float = 0.1) -> SimulationResult:
    """Run a single simulation with both p-MON and worst-case q-NEE optimization.
    
    Args:
        attacker_size: Number of attackers
        defender_size: Number of defenders
        n_resources: Number of resources
        seed: Random seed
        n_agents: Total number of agents (default 10)
        verbose: Print verbose output
        use_projection: Use projection method (default True)
        use_integral_method: Use integral-based boundary finding (default False, uses binary search)
        debug: Print detailed debug output (default False)
        pgd_max_iterations: Max iterations for PGD (default 100)
        pgd_step_size: Step size for PGD (default 0.1)
    """
    
    start_time = time.perf_counter()
    
    # Generate utilities
    utilities = generate_utilities_rod_cutting(n_agents, n_resources, seed)
    supply = np.ones(n_resources) * n_agents
    
    # Select groups
    attacking_group, defending_group = select_groups(n_agents, attacker_size, defender_size, seed)
    non_attacking_group = set(range(n_agents)) - attacking_group
    
    if verbose:
        print(f"  Attackers: {attacking_group}")
        print(f"  Defenders: {defending_group}")
        print(f"  Non-attackers: {non_attacking_group}")
    
    # =========================================================================
    # Part 1: p-MON optimization (best constraint for attackers)
    # =========================================================================
    pmon_start = time.perf_counter()
    
    try:
        constraints, info = compute_optimal_self_benefit_constraint(
            utilities=utilities,
            attacking_group=attacking_group,
            victim_group=defending_group if defending_group else None,
            supply=supply,
            verbose=verbose or debug,
            debug=debug,
            use_projection=use_projection,
            use_integral_method=use_integral_method,
        )
        
        V_initial = info['initial_utilities']
        V_pmon_final = info['final_utilities']
        pmon_converged = info.get('status') == 'optimal' or info.get('status') == 'no_benefit'
        pmon_n_iterations = info.get('cp_iterations', 0)
        
        if verbose:
            print(f"  [p-MON] V_initial: {V_initial.round(6)}")
            print(f"  [p-MON] V_final: {V_pmon_final.round(6)}")
            print(f"  [p-MON] Status: {info.get('status')}")
        
    except Exception as e:
        if verbose:
            print(f"  [p-MON] ERROR: {e}")
            import traceback
            traceback.print_exc()
        # Compute initial allocation ourselves
        optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
        initial_result = optimizer.solve(verbose=False)
        V_initial = initial_result.agent_utilities
        V_pmon_final = V_initial.copy()
        pmon_converged = False
        pmon_n_iterations = 0
    
    pmon_time = time.perf_counter() - pmon_start
    
    # Convert to numpy arrays
    V_initial = np.array(V_initial)
    V_pmon_final = np.array(V_pmon_final)
    
    # Compute p-MON metrics
    p_mon_individual = {}
    for i in attacking_group:
        if V_pmon_final[i] > 1e-10:
            p_mon_individual[i] = V_initial[i] / V_pmon_final[i]
        else:
            p_mon_individual[i] = float('inf')
    
    q_nee_individual = {}
    for i in non_attacking_group:
        if V_initial[i] > 1e-10:
            q_nee_individual[i] = V_pmon_final[i] / V_initial[i]
        else:
            q_nee_individual[i] = float('inf')
    
    p_mon_attacking_group = geometric_mean(list(p_mon_individual.values()))
    defender_q_nee_values = [q_nee_individual[i] for i in defending_group]
    q_nee_defending_group = geometric_mean(defender_q_nee_values)
    q_nee_non_attackers = geometric_mean(list(q_nee_individual.values()))
    
    nash_welfare_initial = float(np.prod(V_initial))
    nash_welfare_pmon_final = float(np.prod(V_pmon_final))
    
    # Compute defender welfare (product of defender utilities)
    defender_welfare_initial = float(np.prod([V_initial[d] for d in defending_group]))
    
    if verbose:
        print(f"  [p-MON] p-MON attacking group: {p_mon_attacking_group:.6f}")
        print(f"  [p-MON] q-NEE defending group: {q_nee_defending_group:.6f}")
        print(f"  [p-MON] Time: {pmon_time:.2f}s")
    
    # =========================================================================
    # Part 2: Worst-case q-NEE optimization (worst constraint for defenders)
    # =========================================================================
    qnee_start = time.perf_counter()
    
    try:
        qnee_constraints, qnee_info = solve_optimal_harm_constraint_pgd(
            utilities=utilities,
            attacking_group=attacking_group,
            defending_group=defending_group,
            supply=supply,
            max_iterations=pgd_max_iterations,
            step_size=pgd_step_size,
            verbose=verbose or debug,
            debug=debug,
        )
        
        V_qnee_worst_final = qnee_info['final_utilities']
        qnee_worst_converged = qnee_info['converged']
        qnee_worst_iterations = qnee_info['iterations']
        defender_welfare_qnee_worst_final = qnee_info['final_defender_welfare']
        
        if verbose:
            print(f"  [q-NEE worst] V_final: {V_qnee_worst_final.round(6)}")
            print(f"  [q-NEE worst] Converged: {qnee_worst_converged}")
            print(f"  [q-NEE worst] Iterations: {qnee_worst_iterations}")
        
    except Exception as e:
        if verbose:
            print(f"  [q-NEE worst] ERROR: {e}")
            import traceback
            traceback.print_exc()
        V_qnee_worst_final = V_initial.copy()
        qnee_worst_converged = False
        qnee_worst_iterations = 0
        defender_welfare_qnee_worst_final = defender_welfare_initial
    
    qnee_time = time.perf_counter() - qnee_start
    
    # Convert to numpy array
    V_qnee_worst_final = np.array(V_qnee_worst_final)
    
    # Compute worst-case q-NEE metrics
    q_nee_worst_individual = {}
    for i in defending_group:
        if V_initial[i] > 1e-10:
            q_nee_worst_individual[i] = V_qnee_worst_final[i] / V_initial[i]
        else:
            q_nee_worst_individual[i] = float('inf')
    
    q_nee_worst_defending_group = geometric_mean(list(q_nee_worst_individual.values()))
    nash_welfare_qnee_worst_final = float(np.prod(V_qnee_worst_final))
    
    # Compute p-MON for attackers under worst-case q-NEE constraint
    p_mon_under_qnee_worst_individual = {}
    for i in attacking_group:
        if V_qnee_worst_final[i] > 1e-10:
            p_mon_under_qnee_worst_individual[i] = V_initial[i] / V_qnee_worst_final[i]
        else:
            p_mon_under_qnee_worst_individual[i] = float('inf')
    
    p_mon_under_qnee_worst = geometric_mean(list(p_mon_under_qnee_worst_individual.values()))
    
    if verbose:
        print(f"  [q-NEE worst] q-NEE defending group (worst): {q_nee_worst_defending_group:.6f}")
        print(f"  [q-NEE worst] p-MON attacking group (under worst q-NEE): {p_mon_under_qnee_worst:.6f}")
        print(f"  [q-NEE worst] Defender welfare reduction: {defender_welfare_initial / max(defender_welfare_qnee_worst_final, 1e-20):.4f}x")
        print(f"  [q-NEE worst] Time: {qnee_time:.2f}s")
    
    total_time = time.perf_counter() - start_time
    
    if verbose:
        print(f"  Total time: {total_time:.2f}s")
        print(f"  q-NEE comparison: p-MON optimal={q_nee_defending_group:.6f}, worst-case={q_nee_worst_defending_group:.6f}")
    
    return SimulationResult(
        attacker_size=attacker_size,
        defender_size=defender_size,
        n_resources=n_resources,
        seed=seed,
        attacking_group=attacking_group,
        defending_group=defending_group,
        # p-MON results
        pmon_converged=pmon_converged,
        pmon_time_seconds=pmon_time,
        pmon_n_iterations=pmon_n_iterations,
        p_mon_attacking_group=p_mon_attacking_group,
        q_nee_defending_group=q_nee_defending_group,
        q_nee_non_attackers=q_nee_non_attackers,
        p_mon_individual=p_mon_individual,
        q_nee_individual=q_nee_individual,
        V_initial=V_initial,
        V_pmon_final=V_pmon_final,
        nash_welfare_initial=nash_welfare_initial,
        nash_welfare_pmon_final=nash_welfare_pmon_final,
        # Worst-case q-NEE results
        qnee_worst_converged=qnee_worst_converged,
        qnee_worst_time_seconds=qnee_time,
        qnee_worst_iterations=qnee_worst_iterations,
        q_nee_worst_defending_group=q_nee_worst_defending_group,
        q_nee_worst_individual=q_nee_worst_individual,
        p_mon_under_qnee_worst=p_mon_under_qnee_worst,
        p_mon_under_qnee_worst_individual=p_mon_under_qnee_worst_individual,
        V_qnee_worst_final=V_qnee_worst_final,
        nash_welfare_qnee_worst_final=nash_welfare_qnee_worst_final,
        defender_welfare_initial=defender_welfare_initial,
        defender_welfare_qnee_worst_final=defender_welfare_qnee_worst_final,
    )


def results_to_rows(results: List[SimulationResult], n_agents: int = 10) -> List[Dict]:
    """Convert simulation results to spreadsheet rows."""
    rows = []
    for r in results:
        # Format groups with semicolons to avoid CSV delimiter issues
        attacking_group_str = ";".join(str(x) for x in sorted(r.attacking_group))
        defending_group_str = ";".join(str(x) for x in sorted(r.defending_group))
        
        row = {
            'attacker_size': r.attacker_size,
            'defender_size': r.defender_size,
            'n_resources': r.n_resources,
            'seed': r.seed,
            'attacking_group': attacking_group_str,
            'defending_group': defending_group_str,
            
            # p-MON results
            'pmon_converged': r.pmon_converged,
            'pmon_time_seconds': r.pmon_time_seconds,
            'pmon_n_iterations': r.pmon_n_iterations,
            'p_mon_attacking_group': r.p_mon_attacking_group,
            'q_nee_defending_group': r.q_nee_defending_group,
            'q_nee_non_attackers': r.q_nee_non_attackers,
            'nash_welfare_initial': r.nash_welfare_initial,
            'nash_welfare_pmon_final': r.nash_welfare_pmon_final,
            
            # Worst-case q-NEE results
            'qnee_worst_converged': r.qnee_worst_converged,
            'qnee_worst_time_seconds': r.qnee_worst_time_seconds,
            'qnee_worst_iterations': r.qnee_worst_iterations,
            'q_nee_worst_defending_group': r.q_nee_worst_defending_group,
            'p_mon_under_qnee_worst': r.p_mon_under_qnee_worst,
            'nash_welfare_qnee_worst_final': r.nash_welfare_qnee_worst_final,
            'defender_welfare_initial': r.defender_welfare_initial,
            'defender_welfare_qnee_worst_final': r.defender_welfare_qnee_worst_final,
        }
        
        # Add individual V_initial values
        for i in range(n_agents):
            row[f'V_initial_{i}'] = r.V_initial[i] if i < len(r.V_initial) else float('nan')
        
        # Add individual V_pmon_final values
        for i in range(n_agents):
            row[f'V_pmon_final_{i}'] = r.V_pmon_final[i] if i < len(r.V_pmon_final) else float('nan')
        
        # Add individual V_qnee_worst_final values
        for i in range(n_agents):
            row[f'V_qnee_worst_final_{i}'] = r.V_qnee_worst_final[i] if i < len(r.V_qnee_worst_final) else float('nan')
        
        # Add individual p-MON values (attackers only)
        for i in range(n_agents):
            if i in r.attacking_group:
                row[f'p_mon_{i}'] = r.p_mon_individual.get(i, float('nan'))
            else:
                row[f'p_mon_{i}'] = ''
        
        # Add individual q-NEE values under p-MON constraint (non-attackers only)
        for i in range(n_agents):
            if i not in r.attacking_group:
                row[f'q_nee_pmon_{i}'] = r.q_nee_individual.get(i, float('nan'))
            else:
                row[f'q_nee_pmon_{i}'] = ''
        
        # Add individual q-NEE values under worst-case constraint (defenders only)
        for i in range(n_agents):
            if i in r.defending_group:
                row[f'q_nee_worst_{i}'] = r.q_nee_worst_individual.get(i, float('nan'))
            else:
                row[f'q_nee_worst_{i}'] = ''
        
        # Add individual p-MON values under worst-case q-NEE constraint (attackers only)
        for i in range(n_agents):
            if i in r.attacking_group:
                row[f'p_mon_under_qnee_worst_{i}'] = r.p_mon_under_qnee_worst_individual.get(i, float('nan'))
            else:
                row[f'p_mon_under_qnee_worst_{i}'] = ''
        
        rows.append(row)
    
    return rows


def save_to_csv(rows: List[Dict], output_folder: str, file_prefix: str, checkpoint_num: int):
    """Save results to CSV file in output folder with unique filename."""
    if not rows:
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate unique filename with timestamp and checkpoint number
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_folder, f"{file_prefix}_checkpoint{checkpoint_num:03d}_{timestamp}.csv")
    
    # Get all column names
    columns = list(rows[0].keys())
    
    with open(filename, 'w') as f:
        # Header
        f.write(','.join(columns) + '\n')
        
        # Data rows
        for row in rows:
            values = []
            for col in columns:
                val = row.get(col, '')
                if isinstance(val, float):
                    if np.isnan(val):
                        values.append('NaN')
                    elif np.isinf(val):
                        values.append('Inf')
                    else:
                        values.append(f'{val:.8f}')
                elif isinstance(val, bool):
                    values.append(str(val))
                else:
                    values.append(str(val))
            f.write(','.join(values) + '\n')
    
    print(f"  Saved {len(rows)} rows to {filename}")
    return filename


def generate_configurations() -> List[Tuple[int, int, int, int]]:
    """Generate all (attacker_size, defender_size, n_resources, seed) configurations."""
    attacker_sizes = [1, 3, 5, 9]
    defender_sizes = [1, 3, 5, 9]
    resource_counts = [2, 5, 10]
    seeds_per_config = 2
    n_agents = 10
    
    configs = []
    
    for att_size in attacker_sizes:
        for def_size in defender_sizes:
            if att_size + def_size <= n_agents:
                for n_resources in resource_counts:
                    for seed_offset in range(seeds_per_config):
                        seed = att_size * 10000 + def_size * 1000 + n_resources * 100 + seed_offset
                        configs.append((att_size, def_size, n_resources, seed))
    
    return configs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run p-MON/q-NEE simulations with worst-case q-NEE")
    parser.add_argument("-o", "--output-folder", type=str, default="simulation_results",
                        dest="output_folder",
                        help="Output folder for CSV files (default: simulation_results)")
    parser.add_argument("-p", "--prefix", type=str, default="pmon_qnee_worstcase",
                        help="Filename prefix (default: pmon_qnee_worstcase)")
    parser.add_argument("--save-interval", type=int, default=6, dest="save_interval",
                        help="Save results every N simulations (default: 6)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print verbose output for each simulation")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed debug output (implies verbose)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Just print configurations without running")
    
    # Method selection arguments
    parser.add_argument("--use-projection", type=str, choices=['true', 'false'], 
                        default='false', dest="use_projection",
                        help="Use projection method (default: false)")
    parser.add_argument("--use-integral", type=str, choices=['true', 'false'],
                        default='false', dest="use_integral",
                        help="Use integral-based boundary finding instead of binary search (default: false)")
    
    # PGD parameters
    parser.add_argument("--pgd-max-iter", type=int, default=100, dest="pgd_max_iterations",
                        help="Max iterations for PGD (default: 100)")
    parser.add_argument("--pgd-step-size", type=float, default=0.1, dest="pgd_step_size",
                        help="Step size for PGD (default: 0.1)")
    
    args = parser.parse_args()
    
    # Parse boolean arguments
    use_projection = args.use_projection.lower() == 'true'
    use_integral_method = args.use_integral.lower() == 'true'
    
    # Generate configurations
    configs = generate_configurations()
    
    print("=" * 70)
    print("p-MON / q-NEE Simulation with Worst-Case q-NEE")
    print("=" * 70)
    print(f"Total configurations: {len(configs)}")
    print(f"Output folder: {args.output_folder}")
    print(f"File prefix: {args.prefix}")
    print(f"Save interval: every {args.save_interval} simulations")
    print(f"Method settings:")
    print(f"  use_projection: {use_projection}")
    print(f"  use_integral_method: {use_integral_method}")
    print(f"PGD settings:")
    print(f"  max_iterations: {args.pgd_max_iterations}")
    print(f"  step_size: {args.pgd_step_size}")
    print()
    
    if args.dry_run:
        print("Configurations:")
        for i, (att, def_, res, seed) in enumerate(configs):
            print(f"  {i+1}. attackers={att}, defenders={def_}, resources={res}, seed={seed}")
        return
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Run simulations
    results = []
    checkpoint_num = 0
    total_start_time = time.perf_counter()
    sim_times = []  # Track individual simulation times for weighted average
    
    for i, (att_size, def_size, n_resources, seed) in enumerate(configs):
        # Progress info
        completed = i
        total = len(configs)
        percent = (completed / total) * 100
        
        elapsed_total = time.perf_counter() - total_start_time
        if completed > 0 and len(sim_times) > 0:
            # Weighted average: sum(t^2) / sum(t)
            sum_t = sum(sim_times)
            sum_t_squared = sum(t * t for t in sim_times)
            weighted_avg_time = sum_t_squared / sum_t if sum_t > 0 else 0
            remaining_sims = total - completed
            eta_seconds = weighted_avg_time * remaining_sims
            eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds >= 60 else f"{eta_seconds:.0f}s"
        else:
            eta_str = "calculating..."
        
        elapsed_str = f"{elapsed_total/60:.1f}min" if elapsed_total >= 60 else f"{elapsed_total:.0f}s"
        
        print(f"[{i+1}/{total}] ({percent:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}")
        print(f"  Config: attackers={att_size}, defenders={def_size}, "
              f"resources={n_resources}, seed={seed}")
        
        result = run_simulation(
            attacker_size=att_size,
            defender_size=def_size,
            n_resources=n_resources,
            seed=seed,
            verbose=args.verbose or args.debug,
            use_projection=use_projection,
            use_integral_method=use_integral_method,
            debug=args.debug,
            pgd_max_iterations=args.pgd_max_iterations,
            pgd_step_size=args.pgd_step_size,
        )
        
        results.append(result)
        total_sim_time = result.pmon_time_seconds + result.qnee_worst_time_seconds
        sim_times.append(total_sim_time)
        
        pmon_status = "✓" if result.pmon_converged else "✗"
        qnee_status = "✓" if result.qnee_worst_converged else "✗"
        print(f"  p-MON: {pmon_status} p-MON={result.p_mon_attacking_group:.4f}, "
              f"q-NEE(def)={result.q_nee_defending_group:.4f}, "
              f"time={result.pmon_time_seconds:.2f}s")
        print(f"  q-NEE worst: {qnee_status} q-NEE(def)={result.q_nee_worst_defending_group:.4f}, "
              f"iter={result.qnee_worst_iterations}, "
              f"time={result.qnee_worst_time_seconds:.2f}s")
        
        # Periodic save
        if (i + 1) % args.save_interval == 0:
            checkpoint_num += 1
            print(f"\n  --- Saving checkpoint {checkpoint_num} ({i+1} simulations completed) ---")
            rows = results_to_rows(results)
            save_to_csv(rows, args.output_folder, args.prefix, checkpoint_num)
            print()
    
    # Final save
    checkpoint_num += 1
    print(f"\n--- Final save (checkpoint {checkpoint_num}, all {len(results)} simulations) ---")
    rows = results_to_rows(results)
    final_file = save_to_csv(rows, args.output_folder, args.prefix, checkpoint_num)
    
    # Summary
    total_time = time.perf_counter() - total_start_time
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Total simulations: {len(results)}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per simulation: {total_time/len(results):.2f}s")
    print()
    
    # Statistics
    pmon_converged = sum(1 for r in results if r.pmon_converged)
    qnee_converged = sum(1 for r in results if r.qnee_worst_converged)
    print(f"p-MON convergence: {pmon_converged}/{len(results)} ({100*pmon_converged/len(results):.1f}%)")
    print(f"q-NEE worst convergence: {qnee_converged}/{len(results)} ({100*qnee_converged/len(results):.1f}%)")
    
    # Compare q-NEE under p-MON vs worst-case
    qnee_pmon_values = [r.q_nee_defending_group for r in results if not np.isnan(r.q_nee_defending_group)]
    qnee_worst_values = [r.q_nee_worst_defending_group for r in results if not np.isnan(r.q_nee_worst_defending_group)]
    
    if qnee_pmon_values and qnee_worst_values:
        print()
        print("q-NEE comparison (defending group):")
        print(f"  Under p-MON constraint:")
        print(f"    Mean: {np.mean(qnee_pmon_values):.4f}")
        print(f"    Min:  {np.min(qnee_pmon_values):.4f}")
        print(f"    Max:  {np.max(qnee_pmon_values):.4f}")
        print(f"  Under worst-case constraint:")
        print(f"    Mean: {np.mean(qnee_worst_values):.4f}")
        print(f"    Min:  {np.min(qnee_worst_values):.4f}")
        print(f"    Max:  {np.max(qnee_worst_values):.4f}")
    
    print()
    print(f"Results saved to: {final_file}")


if __name__ == "__main__":
    main()