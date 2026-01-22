"""
Simulation script for p-MON and q-NEE experiments.

Runs simulations with:
- 10 agents
- Attacker sizes: {1, 3, 5, 9}
- Defender sizes: {1, 3, 5, 9}
- Valid pairs: attacker_size + defender_size <= 10
- Resource counts: {2, 5, 10}
- 2 random seeds per configuration

Computes p-MON for attackers and q-NEE for non-attackers.
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
    _solve_optimal_constraint_convex_program,
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
    converged: bool
    time_seconds: float
    n_iterations: int
    p_mon_attacking_group: float
    q_nee_defending_group: float
    q_nee_non_attackers: float
    p_mon_individual: Dict[int, float]  # agent_id -> p_mon (attackers only)
    q_nee_individual: Dict[int, float]  # agent_id -> q_nee (non-attackers only)
    V_initial: np.ndarray
    V_final: np.ndarray


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
                   use_projection: bool = True, use_integral_method: bool = False) -> SimulationResult:
    """Run a single simulation.
    
    Args:
        attacker_size: Number of attackers
        defender_size: Number of defenders
        n_resources: Number of resources
        seed: Random seed
        n_agents: Total number of agents (default 10)
        verbose: Print verbose output
        use_projection: Use projection method (default True)
        use_integral_method: Use integral-based boundary finding (default False, uses binary search)
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
    
    # Step 1: Initial Nash welfare solution
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    initial_result = optimizer.solve(verbose=False)
    x_initial = initial_result.allocation
    V_initial = np.array([np.dot(x_initial[i], utilities[i]) for i in range(n_agents)])
    
    if verbose:
        print(f"  Initial utilities: {V_initial.round(4)}")
    
    # Steps 2-3: Find p-MON constraint using convex program solver
    # For p-MON, attacking_group = target_group, maximize_harm = False
    try:
        optimal_directions, x_final, status, cp_debug_info = _solve_optimal_constraint_convex_program(
            utilities=utilities,
            attacking_group=attacking_group,
            target_group=attacking_group,  # For p-MON, target is the attackers themselves
            supply=supply,
            maximize_harm=False,  # Minimize p-MON (maximize attacker utility)
            verbose=False,
            use_projection=use_projection,
            use_integral_method=use_integral_method,
        )
        
        converged = cp_debug_info.get('converged', True)
        n_iterations = cp_debug_info.get('iterations', 0)
        
        # x_final is already the final allocation after applying the constraint
        
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        # Return a failed result
        elapsed = time.perf_counter() - start_time
        return SimulationResult(
            attacker_size=attacker_size,
            defender_size=defender_size,
            n_resources=n_resources,
            seed=seed,
            attacking_group=attacking_group,
            defending_group=defending_group,
            converged=False,
            time_seconds=elapsed,
            n_iterations=0,
            p_mon_attacking_group=float('nan'),
            q_nee_defending_group=float('nan'),
            q_nee_non_attackers=float('nan'),
            p_mon_individual={},
            q_nee_individual={},
            V_initial=V_initial,
            V_final=np.full(n_agents, np.nan),
        )
    
    # Compute final utilities
    V_final = np.array([np.dot(x_final[i], utilities[i]) for i in range(n_agents)])
    
    if verbose:
        print(f"  Final utilities: {V_final.round(4)}")
    
    # Compute individual p-MON for attackers: V_initial / V_final
    p_mon_individual = {}
    for i in attacking_group:
        if V_final[i] > 1e-10:
            p_mon_individual[i] = V_initial[i] / V_final[i]
        else:
            p_mon_individual[i] = float('inf')
    
    # Compute individual q-NEE for non-attackers: V_final / V_initial
    q_nee_individual = {}
    for i in non_attacking_group:
        if V_initial[i] > 1e-10:
            q_nee_individual[i] = V_final[i] / V_initial[i]
        else:
            q_nee_individual[i] = float('inf')
    
    # Compute group p-MON (geometric mean for attackers)
    p_mon_attacking_group = geometric_mean(list(p_mon_individual.values()))
    
    # Compute group q-NEE for defenders
    defender_q_nee_values = [q_nee_individual[i] for i in defending_group]
    q_nee_defending_group = geometric_mean(defender_q_nee_values)
    
    # Compute group q-NEE for all non-attackers
    q_nee_non_attackers = geometric_mean(list(q_nee_individual.values()))
    
    elapsed = time.perf_counter() - start_time
    
    if verbose:
        print(f"  p-MON attacking group: {p_mon_attacking_group:.6f}")
        print(f"  q-NEE defending group: {q_nee_defending_group:.6f}")
        print(f"  q-NEE non-attackers: {q_nee_non_attackers:.6f}")
        print(f"  Time: {elapsed:.2f}s")
    
    return SimulationResult(
        attacker_size=attacker_size,
        defender_size=defender_size,
        n_resources=n_resources,
        seed=seed,
        attacking_group=attacking_group,
        defending_group=defending_group,
        converged=converged,
        time_seconds=elapsed,
        n_iterations=n_iterations,
        p_mon_attacking_group=p_mon_attacking_group,
        q_nee_defending_group=q_nee_defending_group,
        q_nee_non_attackers=q_nee_non_attackers,
        p_mon_individual=p_mon_individual,
        q_nee_individual=q_nee_individual,
        V_initial=V_initial,
        V_final=V_final,
    )


def results_to_rows(results: List[SimulationResult], n_agents: int = 10) -> List[Dict]:
    """Convert simulation results to spreadsheet rows."""
    rows = []
    for r in results:
        row = {
            'attacker_size': r.attacker_size,
            'defender_size': r.defender_size,
            'n_resources': r.n_resources,
            'seed': r.seed,
            'attacking_group': str(sorted(r.attacking_group)),
            'defending_group': str(sorted(r.defending_group)),
            'converged': r.converged,
            'time_seconds': r.time_seconds,
            'n_iterations': r.n_iterations,
            'p_mon_attacking_group': r.p_mon_attacking_group,
            'q_nee_defending_group': r.q_nee_defending_group,
            'q_nee_non_attackers': r.q_nee_non_attackers,
        }
        
        # Add individual p-MON values (attackers only)
        for i in range(n_agents):
            if i in r.attacking_group:
                row[f'p_mon_{i}'] = r.p_mon_individual.get(i, float('nan'))
            else:
                row[f'p_mon_{i}'] = ''  # NA for non-attackers
        
        # Add individual q-NEE values (non-attackers only)
        for i in range(n_agents):
            if i not in r.attacking_group:
                row[f'q_nee_{i}'] = r.q_nee_individual.get(i, float('nan'))
            else:
                row[f'q_nee_{i}'] = ''  # NA for attackers
        
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
    
    parser = argparse.ArgumentParser(description="Run p-MON/q-NEE simulations")
    parser.add_argument("-o", "--output-folder", type=str, default="simulation_results",
                        dest="output_folder",
                        help="Output folder for CSV files (default: simulation_results)")
    parser.add_argument("-p", "--prefix", type=str, default="pmon_qnee",
                        help="Filename prefix (default: pmon_qnee)")
    parser.add_argument("--save-interval", type=int, default=6, dest="save_interval",
                        help="Save results every N simulations (default: 6)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print verbose output for each simulation")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Just print configurations without running")
    
    # Method selection arguments
    parser.add_argument("--use-projection", type=str, choices=['true', 'false'], 
                        default='true', dest="use_projection",
                        help="Use projection method (default: true)")
    parser.add_argument("--use-integral", type=str, choices=['true', 'false'],
                        default='false', dest="use_integral",
                        help="Use integral-based boundary finding instead of binary search (default: false)")
    
    args = parser.parse_args()
    
    # Parse boolean arguments
    use_projection = args.use_projection.lower() == 'true'
    use_integral_method = args.use_integral.lower() == 'true'
    
    # Generate configurations
    configs = generate_configurations()
    
    print("=" * 70)
    print("p-MON / q-NEE Simulation")
    print("=" * 70)
    print(f"Total configurations: {len(configs)}")
    print(f"Output folder: {args.output_folder}")
    print(f"File prefix: {args.prefix}")
    print(f"Save interval: every {args.save_interval} simulations")
    print(f"Method settings:")
    print(f"  use_projection: {use_projection}")
    print(f"  use_integral_method: {use_integral_method}")
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
    
    for i, (att_size, def_size, n_resources, seed) in enumerate(configs):
        # Progress info
        completed = i
        total = len(configs)
        percent = (completed / total) * 100
        
        elapsed_total = time.perf_counter() - total_start_time
        if completed > 0:
            avg_time_per_sim = elapsed_total / completed
            remaining_sims = total - completed
            eta_seconds = avg_time_per_sim * remaining_sims
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
            verbose=args.verbose,
            use_projection=use_projection,
            use_integral_method=use_integral_method,
        )
        
        results.append(result)
        
        status = "✓" if result.converged else "✗"
        print(f"  {status} p-MON={result.p_mon_attacking_group:.4f}, "
              f"q-NEE(def)={result.q_nee_defending_group:.4f}, "
              f"time={result.time_seconds:.2f}s")
        
        # Periodic save
        if (i + 1) % args.save_interval == 0:
            checkpoint_num += 1
            print(f"\n  --- Saving checkpoint {checkpoint_num} ({i+1} simulations completed) ---")
            rows = results_to_rows(results)
            save_to_csv(rows, args.output_folder, args.prefix, checkpoint_num)
            print()
    
    # Final save
    print(f"\n{'=' * 70}")
    print("Simulation complete!")
    print(f"{'=' * 70}")
    
    checkpoint_num += 1
    rows = results_to_rows(results)
    save_to_csv(rows, args.output_folder, args.prefix, checkpoint_num)
    
    # Summary statistics
    converged_count = sum(1 for r in results if r.converged)
    total_time = sum(r.time_seconds for r in results)
    
    print(f"\nSummary:")
    print(f"  Total simulations: {len(results)}")
    print(f"  Converged: {converged_count}/{len(results)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average time per simulation: {total_time/len(results):.2f}s")


if __name__ == "__main__":
    main()

V_initial and V_final are not logged to the output so idk those values. the total utility before and after would be a good thing to add to the simulation log. the p-MON constraint should be benifitting the attackers because it is finding the constraint that maximized their utility. the weird part is that every elses utility is going up after. To me this seems like it may be a bug in accessing the before allocation or after allocation correctly so lets start by accessing those.