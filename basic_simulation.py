"""
Simulation script for optimal harm constraint experiments.

Runs a 3x3 grid of scenarios varying:
- Attacking group size: 1, 3, 5
- Victim group size: 1, 3, 5

For each combination, runs 3 simulations with different random seeds.
Total: 27 simulation runs.
"""

import numpy as np
import pandas as pd
from itertools import product

from nash_welfare_optimizer import (
    generate_normalized_utilities,
    compute_optimal_harm_constraint,
    compute_optimal_self_benefit_constraint,
)


def run_harm_simulation(
    n_agents: int = 10,
    n_resources: int = 10,
    attacking_sizes: list = [1, 3, 5],
    victim_sizes: list = [1, 3, 5],
    runs_per_combo: int = 3,
    base_seed: int = 42,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run the harm optimization simulation.
    
    Args:
        n_agents: Number of agents (default 10)
        n_resources: Number of resources (default 10)
        attacking_sizes: List of attacking group sizes to test
        victim_sizes: List of victim group sizes to test
        runs_per_combo: Number of runs per (attacking_size, victim_size) combination
        base_seed: Base random seed for reproducibility
        verbose: Print progress
    
    Returns:
        DataFrame with results for all runs
    """
    results = []
    run_id = 0
    
    total_runs = len(attacking_sizes) * len(victim_sizes) * runs_per_combo
    
    for attack_size, victim_size in product(attacking_sizes, victim_sizes):
        for run_num in range(runs_per_combo):
            seed = base_seed + run_id
            
            if verbose:
                print(f"\nRun {run_id + 1}/{total_runs}: "
                      f"attack_size={attack_size}, victim_size={victim_size}, "
                      f"run_num={run_num + 1}, seed={seed}")
            
            # Generate random utilities
            rng = np.random.default_rng(seed)
            utilities = generate_normalized_utilities(n_agents, n_resources, rng)
            
            # Randomly select attacking and victim groups (non-overlapping)
            all_agents = list(range(n_agents))
            rng.shuffle(all_agents)
            
            attacking_group = set(all_agents[:attack_size])
            victim_group = set(all_agents[attack_size:attack_size + victim_size])
            
            if verbose:
                print(f"  Attacking group: {attacking_group}")
                print(f"  Victim group: {victim_group}")
            
            # Run the optimal harm constraint computation
            try:
                constraints, info = compute_optimal_harm_constraint(
                    utilities=utilities,
                    attacking_group=attacking_group,
                    victim_group=victim_group,
                    verbose=verbose
                )
                
                status = info['status']
                p_mon_group = info['p_mon_group']
                q_nee_group = info['q_nee_group']
                welfare_loss = info['welfare_loss_ratio']
                
                # Individual metrics
                p_mon_individual = info['p_mon_individual']
                q_nee_individual = info['q_nee_individual']
                
                # Debug: Check if constraints were found
                constraints_found = constraints is not None and len(constraints) > 0
                
                if verbose:
                    print(f"  Status: {status}")
                    print(f"  Constraints found: {constraints_found}")
                    if constraints_found:
                        for aid, c in constraints.items():
                            print(f"    Agent {aid} ratios: {c.get_ratios()[:3]}...")  # Show first 3
                    print(f"  p-MON (group): {p_mon_group:.6f}")
                    print(f"  q-NEE (group): {q_nee_group:.6f}")
                    print(f"  Welfare loss ratio: {welfare_loss:.6f}")
                
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                status = f"error: {e}"
                p_mon_group = None
                q_nee_group = None
                welfare_loss = None
                p_mon_individual = {}
                q_nee_individual = {}
            
            # Record results
            result = {
                'run_id': run_id,
                'seed': seed,
                'n_agents': n_agents,
                'n_resources': n_resources,
                'attack_size': attack_size,
                'victim_size': victim_size,
                'run_num': run_num,
                'attacking_group': ','.join(map(str, sorted(attacking_group))),
                'victim_group': ','.join(map(str, sorted(victim_group))),
                'status': status,
                'p_mon_group': p_mon_group,
                'q_nee_group': q_nee_group,
                'welfare_loss_ratio': welfare_loss,
            }
            
            # Add individual p-MON values (sorted by agent number)
            for agent_id in range(n_agents):
                if agent_id in attacking_group:
                    result[f'p_mon_agent_{agent_id}'] = p_mon_individual.get(agent_id)
                else:
                    result[f'p_mon_agent_{agent_id}'] = None
            
            # Add individual q-NEE values (sorted by agent number)
            for agent_id in range(n_agents):
                if agent_id in victim_group:
                    result[f'q_nee_agent_{agent_id}'] = q_nee_individual.get(agent_id)
                else:
                    result[f'q_nee_agent_{agent_id}'] = None
            
            results.append(result)
            run_id += 1
    
    df = pd.DataFrame(results)
    
    # Reorder columns: metadata, then p_mon_agent_0 through p_mon_agent_N, then q_nee_agent_0 through q_nee_agent_N
    priority_cols = [
        'run_id', 'seed', 'n_agents', 'n_resources',
        'attack_size', 'victim_size', 'run_num',
        'attacking_group', 'victim_group',
        'status', 'p_mon_group', 'q_nee_group', 'welfare_loss_ratio'
    ]
    
    # p_mon columns sorted by agent number
    p_mon_cols = [f'p_mon_agent_{i}' for i in range(n_agents)]
    
    # q_nee columns sorted by agent number  
    q_nee_cols = [f'q_nee_agent_{i}' for i in range(n_agents)]
    
    # Combine all columns in order
    all_cols = priority_cols + p_mon_cols + q_nee_cols
    
    # Only include columns that exist in the dataframe
    final_cols = [c for c in all_cols if c in df.columns]
    df = df[final_cols]
    
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics grouped by attack_size and victim_size.
    """
    summary = df.groupby(['attack_size', 'victim_size']).agg({
        'p_mon_group': ['mean', 'std', 'min', 'max'],
        'q_nee_group': ['mean', 'std', 'min', 'max'],
        'welfare_loss_ratio': ['mean', 'std', 'min', 'max'],
    }).round(6)
    
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Running Optimal Harm Constraint Simulation")
    print("=" * 60)
    print("\nConfiguration:")
    print("  - 10 agents, 10 resources")
    print("  - Attacking group sizes: 1, 3, 5")
    print("  - Victim group sizes: 1, 3, 5")
    print("  - 3 runs per combination")
    print("  - Total: 27 runs")
    print("=" * 60)
    
    # Run simulation
    df = run_harm_simulation(
        n_agents=10,
        n_resources=10,
        attacking_sizes=[1, 3, 5],
        victim_sizes=[1, 3, 5],
        runs_per_combo=3,
        base_seed=42,
        verbose=True
    )
    
    # Save detailed results
    output_file = "harm_simulation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    summary = summarize_results(df)
    print(summary.to_string())
    
    # Save summary
    summary_file = "harm_simulation_summary.csv"
    summary.to_csv(summary_file)
    print(f"\nSummary saved to: {summary_file}")
    
    # Quick analysis
    print("\n" + "=" * 60)
    print("QUICK ANALYSIS")
    print("=" * 60)
    
    # Check for different status values
    status_counts = df['status'].value_counts()
    print(f"\nStatus distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    successful = df[df['status'] == 'optimal']
    print(f"\nSuccessful runs (status='optimal'): {len(successful)}/{len(df)}")
    
    # Also check runs where metrics changed (even if status is different)
    changed = df[(df['p_mon_group'] != 1.0) | (df['q_nee_group'] != 1.0)]
    print(f"Runs with metric changes: {len(changed)}/{len(df)}")
    
    if len(successful) > 0:
        print(f"\nOverall p-MON (attackers):")
        print(f"  Mean: {successful['p_mon_group'].mean():.6f}")
        print(f"  < 1 (benefited): {(successful['p_mon_group'] < 1).sum()}/{len(successful)}")
        
        print(f"\nOverall q-NEE (victims):")
        print(f"  Mean: {successful['q_nee_group'].mean():.6f}")
        print(f"  < 1 (harmed): {(successful['q_nee_group'] < 1).sum()}/{len(successful)}")
        
        print(f"\nWelfare loss ratio:")
        print(f"  Mean: {successful['welfare_loss_ratio'].mean():.6f}")