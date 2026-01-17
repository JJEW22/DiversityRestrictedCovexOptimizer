"""
Small test simulation for optimal harm constraint.

Quick test with:
- 6 agents, 6 resources
- 3 attacking agents, 3 victim agents
- 3 iterations
"""

import numpy as np
import pandas as pd
import time

from nash_welfare_optimizer import (
    generate_normalized_utilities,
    compute_optimal_harm_constraint,
    compute_optimal_self_benefit_constraint,
)
from debug_output_helper import save_debug_xlsx




def save_debug_info(run_id: int, attack_type: str, info: dict, attacking_group: set, 
                    victim_group: set, utilities: np.ndarray, n_agents: int, n_resources: int,
                    debug_dir: str = "."):
    """Wrapper for save_debug_xlsx for backward compatibility."""
    filename = f"{debug_dir}/debug_run{run_id}_{attack_type}.xlsx"
    save_debug_xlsx(
        filename=filename,
        info=info,
        attacking_group=attacking_group,
        victim_group=victim_group,
        utilities=utilities,
        n_agents=n_agents,
        n_resources=n_resources,
        run_id=run_id,
        attack_type=attack_type
    )


def print_debug_summary(attack_type: str, info: dict, attacking_group: set, victim_group: set, n_agents: int):
    """Print a detailed debug summary."""
    print(f"\n  === DEBUG: {attack_type} ===")
    print(f"  Status: {info.get('status')}")
    print(f"  Constraints found: {info.get('constraints_found')}")
    
    if info.get('constraint_ratios'):
        print(f"  Constraint ratios:")
        for agent_id, ratios in info['constraint_ratios'].items():
            print(f"    Agent {agent_id}: {np.array(ratios).round(4)}")
    
    print(f"\n  Initial utilities: {info['initial_utilities'].round(6)}")
    print(f"  Final utilities:   {info['final_utilities'].round(6)}")
    
    print(f"\n  Utility changes:")
    for i in range(n_agents):
        init_u = info['initial_utilities'][i]
        final_u = info['final_utilities'][i]
        change = final_u - init_u
        pct = (final_u / init_u - 1) * 100 if init_u > 1e-10 else 0
        
        if i in attacking_group:
            role = "ATTACKER"
        elif i in victim_group:
            role = "VICTIM"
        else:
            role = "neither"
        
        print(f"    Agent {i} ({role:8}): {init_u:.6f} -> {final_u:.6f} (Δ={change:+.6f}, {pct:+.2f}%)")
    
    print(f"\n  p-MON (group): {info['p_mon_group']:.6f} {'<-- SHOULD BE <= 1 for self-benefit!' if attack_type == 'pmon' and info['p_mon_group'] > 1.001 else ''}")
    print(f"  q-NEE (group): {info['q_nee_group']:.6f}")
    print(f"  Welfare loss:  {info['welfare_loss_ratio']:.6f}")
    
    # Check for anomalies
    if attack_type == 'pmon' and info['p_mon_group'] > 1.001:
        print(f"\n  *** BUG DETECTED: p-MON > 1 in self-benefit attack! ***")
        print(f"  *** Attackers got WORSE, not better. This should not happen. ***")
    
    if attack_type == 'qnee' and info['q_nee_group'] > 1.001:
        print(f"\n  *** NOTE: q-NEE > 1 means victims actually benefited (unexpected for harm attack) ***")


def run_small_test(
    n_agents: int = 6,
    n_resources: int = 6,
    attack_size: int = 3,
    victim_size: int = 3,
    n_runs: int = 3,
    base_seed: int = 42,
    verbose: bool = False,
    timing: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    """
    Run a small test simulation.
    
    For each run, computes BOTH:
    - q-NEE attack (maximize harm to victims)
    - p-MON attack (maximize self-benefit to attackers)
    
    Args:
        debug: If True, enables verbose mode and saves detailed debug output
    """
    results = []
    
    # Debug mode implies verbose
    if debug:
        verbose = True
    
    for run_id in range(n_runs):
        seed = base_seed + run_id
        
        print(f"\n{'='*60}")
        print(f"Run {run_id + 1}/{n_runs}: seed={seed}")
        print(f"{'='*60}")
        
        # Time: Generate random groups and matrices
        t_start = time.time()
        rng = np.random.default_rng(seed)
        utilities = generate_normalized_utilities(n_agents, n_resources, rng)
        
        # Select attacking and victim groups (non-overlapping)
        all_agents = list(range(n_agents))
        rng.shuffle(all_agents)
        
        attacking_group = set(all_agents[:attack_size])
        victim_group = set(all_agents[attack_size:attack_size + victim_size])
        generation_time = time.time() - t_start
        
        print(f"Attacking group: {attacking_group}")
        print(f"Victim group: {victim_group}")
        if debug:
            print(f"Utilities matrix:\n{utilities.round(4)}")
        
        result = {
            'run_id': run_id,
            'seed': seed,
            'n_agents': n_agents,
            'n_resources': n_resources,
            'attack_size': attack_size,
            'victim_size': victim_size,
            'attacking_group': ','.join(map(str, sorted(attacking_group))),
            'victim_group': ','.join(map(str, sorted(victim_group))),
        }
        
        if timing:
            result['time_generation'] = generation_time
        
        # ============================================================
        # Q-NEE Attack (maximize harm to victims)
        # ============================================================
        print(f"\n--- Q-NEE Attack (maximize harm) ---")
        try:
            t_start = time.time()
            qnee_constraints, qnee_info = compute_optimal_harm_constraint(
                utilities=utilities,
                attacking_group=attacking_group,
                victim_group=victim_group,
                verbose=verbose,
                timing=timing,
                debug=debug
            )
            qnee_total_time = time.time() - t_start
            
            result['qnee_attack_status'] = qnee_info['status']
            result['qnee_attack_q_nee_group'] = qnee_info['q_nee_group']
            result['qnee_attack_p_mon_group'] = qnee_info['p_mon_group']
            result['qnee_attack_welfare_loss'] = qnee_info['welfare_loss_ratio']
            
            # Individual metrics for ALL agents during q-NEE attack
            # With role labels: attacking, victim, or neither
            for i in range(n_agents):
                if i in attacking_group:
                    role = 'attacking'
                elif i in victim_group:
                    role = 'victim'
                else:
                    role = 'neither'
                
                result[f'qnee_attack_p_mon_{role}_agent_{i}'] = qnee_info['p_mon_individual_all'].get(i)
                result[f'qnee_attack_q_nee_{role}_agent_{i}'] = qnee_info['q_nee_individual_all'].get(i)
            
            if timing:
                result['time_qnee_attack'] = qnee_total_time
                if qnee_info.get('timing'):
                    result['time_qnee_initial_solve'] = qnee_info['timing'].get('initial_solve_time')
                    result['time_qnee_optimization'] = qnee_info['timing'].get('optimization_time')
                    result['time_qnee_final_solve'] = qnee_info['timing'].get('final_solve_time')
                    result['time_qnee_metrics'] = qnee_info['timing'].get('metrics_time')
            
            print(f"Status: {qnee_info['status']}")
            print(f"q-NEE (group): {qnee_info['q_nee_group']:.6f}")
            print(f"p-MON (group): {qnee_info['p_mon_group']:.6f}")
            
            if debug:
                print_debug_summary('qnee', qnee_info, attacking_group, victim_group, n_agents)
                save_debug_info(run_id, 'qnee', qnee_info, attacking_group, victim_group, 
                               utilities, n_agents, n_resources)
            
        except Exception as e:
            print(f"ERROR in q-NEE attack: {e}")
            import traceback
            traceback.print_exc()
            result['qnee_attack_status'] = f'error: {e}'
        
        # ============================================================
        # P-MON Attack (maximize self-benefit)
        # ============================================================
        print(f"\n--- P-MON Attack (maximize self-benefit) ---")
        try:
            t_start = time.time()
            pmon_constraints, pmon_info = compute_optimal_self_benefit_constraint(
                utilities=utilities,
                attacking_group=attacking_group,
                victim_group=victim_group,
                verbose=verbose,
                timing=timing,
                debug=debug
            )
            pmon_total_time = time.time() - t_start
            
            result['pmon_attack_status'] = pmon_info['status']
            result['pmon_attack_p_mon_group'] = pmon_info['p_mon_group']
            result['pmon_attack_q_nee_group'] = pmon_info['q_nee_group']
            result['pmon_attack_welfare_loss'] = pmon_info['welfare_loss_ratio']
            
            # Individual metrics for ALL agents during p-MON attack
            # With role labels: attacking, victim, or neither
            for i in range(n_agents):
                if i in attacking_group:
                    role = 'attacking'
                elif i in victim_group:
                    role = 'victim'
                else:
                    role = 'neither'
                
                result[f'pmon_attack_p_mon_{role}_agent_{i}'] = pmon_info['p_mon_individual_all'].get(i)
                result[f'pmon_attack_q_nee_{role}_agent_{i}'] = pmon_info['q_nee_individual_all'].get(i)
            
            if timing:
                result['time_pmon_attack'] = pmon_total_time
                if pmon_info.get('timing'):
                    result['time_pmon_initial_solve'] = pmon_info['timing'].get('initial_solve_time')
                    result['time_pmon_optimization'] = pmon_info['timing'].get('optimization_time')
                    result['time_pmon_final_solve'] = pmon_info['timing'].get('final_solve_time')
                    result['time_pmon_metrics'] = pmon_info['timing'].get('metrics_time')
            
            print(f"Status: {pmon_info['status']}")
            print(f"p-MON (group): {pmon_info['p_mon_group']:.6f}")
            print(f"q-NEE (group): {pmon_info['q_nee_group']:.6f}")
            
            if debug:
                print_debug_summary('pmon', pmon_info, attacking_group, victim_group, n_agents)
                save_debug_info(run_id, 'pmon', pmon_info, attacking_group, victim_group,
                               utilities, n_agents, n_resources)
                
                # Extra check for the bug
                if pmon_info['p_mon_group'] > 1.001:
                    print(f"\n  !!! BUG: p-MON > 1 detected in run {run_id} !!!")
                    print(f"  This means the 'optimal' constraint made attackers WORSE OFF.")
                    print(f"  The optimization should have at least returned the trivial solution (no constraint).")
            
        except Exception as e:
            print(f"ERROR in p-MON attack: {e}")
            import traceback
            traceback.print_exc()
            result['pmon_attack_status'] = f'error: {e}'
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    meta_cols = ['run_id', 'seed', 'n_agents', 'n_resources', 'attack_size', 'victim_size',
                 'attacking_group', 'victim_group']
    
    qnee_group_cols = ['qnee_attack_status', 'qnee_attack_q_nee_group', 'qnee_attack_p_mon_group', 
                       'qnee_attack_welfare_loss']
    
    # Q-NEE attack individual columns: p_mon by role, then q_nee by role
    qnee_pmon_attacking = [f'qnee_attack_p_mon_attacking_agent_{i}' for i in range(n_agents)]
    qnee_pmon_victim = [f'qnee_attack_p_mon_victim_agent_{i}' for i in range(n_agents)]
    qnee_pmon_neither = [f'qnee_attack_p_mon_neither_agent_{i}' for i in range(n_agents)]
    qnee_qnee_attacking = [f'qnee_attack_q_nee_attacking_agent_{i}' for i in range(n_agents)]
    qnee_qnee_victim = [f'qnee_attack_q_nee_victim_agent_{i}' for i in range(n_agents)]
    qnee_qnee_neither = [f'qnee_attack_q_nee_neither_agent_{i}' for i in range(n_agents)]
    
    pmon_group_cols = ['pmon_attack_status', 'pmon_attack_p_mon_group', 'pmon_attack_q_nee_group',
                       'pmon_attack_welfare_loss']
    
    # P-MON attack individual columns: p_mon by role, then q_nee by role
    pmon_pmon_attacking = [f'pmon_attack_p_mon_attacking_agent_{i}' for i in range(n_agents)]
    pmon_pmon_victim = [f'pmon_attack_p_mon_victim_agent_{i}' for i in range(n_agents)]
    pmon_pmon_neither = [f'pmon_attack_p_mon_neither_agent_{i}' for i in range(n_agents)]
    pmon_qnee_attacking = [f'pmon_attack_q_nee_attacking_agent_{i}' for i in range(n_agents)]
    pmon_qnee_victim = [f'pmon_attack_q_nee_victim_agent_{i}' for i in range(n_agents)]
    pmon_qnee_neither = [f'pmon_attack_q_nee_neither_agent_{i}' for i in range(n_agents)]
    
    timing_cols = [c for c in df.columns if c.startswith('time_')]
    
    ordered_cols = (meta_cols + 
                    qnee_group_cols + 
                    qnee_pmon_attacking + qnee_pmon_victim + qnee_pmon_neither +
                    qnee_qnee_attacking + qnee_qnee_victim + qnee_qnee_neither +
                    pmon_group_cols + 
                    pmon_pmon_attacking + pmon_pmon_victim + pmon_pmon_neither +
                    pmon_qnee_attacking + pmon_qnee_victim + pmon_qnee_neither +
                    timing_cols)
    
    # Only include columns that exist
    final_cols = [c for c in ordered_cols if c in df.columns]
    df = df[final_cols]
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Small test simulation for Nash welfare optimization")
    parser.add_argument('--n-agents', type=int, default=6, help='Number of agents (default: 6)')
    parser.add_argument('--n-resources', type=int, default=6, help='Number of resources (default: 6)')
    parser.add_argument('--attack-size', type=int, default=3, help='Number of attacking agents (default: 3)')
    parser.add_argument('--victim-size', type=int, default=3, help='Number of victim agents (default: 3)')
    parser.add_argument('--n-runs', type=int, default=3, help='Number of simulation runs (default: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('--timing', action='store_true', help='Include timing information')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (verbose + debug outputs)')
    parser.add_argument('--output', type=str, default='small_test_results.csv', help='Output CSV file (default: small_test_results.csv)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Small Test Simulation")
    print("=" * 60)
    print(f"Config: {args.n_agents} agents, {args.n_resources} resources, "
          f"{args.attack_size} attackers, {args.victim_size} victims, {args.n_runs} runs")
    print(f"Timing: {args.timing}, Verbose: {args.verbose}, Debug: {args.debug}")
    print("=" * 60)
    
    df = run_small_test(
        n_agents=args.n_agents,
        n_resources=args.n_resources,
        attack_size=args.attack_size,
        victim_size=args.victim_size,
        n_runs=args.n_runs,
        base_seed=args.seed,
        verbose=args.verbose,
        timing=args.timing,
        debug=args.debug
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Print key columns
    summary_cols = ['run_id', 'qnee_attack_status', 'qnee_attack_q_nee_group', 'qnee_attack_p_mon_group',
                    'pmon_attack_status', 'pmon_attack_p_mon_group', 'pmon_attack_q_nee_group']
    available_cols = [c for c in summary_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))
    
    # Check for bugs
    if 'pmon_attack_p_mon_group' in df.columns:
        bad_runs = df[df['pmon_attack_p_mon_group'] > 1.001]
        if len(bad_runs) > 0:
            print(f"\n*** WARNING: {len(bad_runs)} runs have p-MON > 1 in self-benefit attack! ***")
            print("This indicates a bug - see debug output for details.")
    
    # Timing summary
    if args.timing:
        timing_cols = [c for c in df.columns if c.startswith('time_')]
        if timing_cols:
            print("\nTiming Summary:")
            for col in timing_cols:
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}: mean={vals.mean():.4f}s, total={vals.sum():.4f}s")
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    if args.debug:
        print(f"Debug xlsx files saved as debug_run*_*.xlsx (with multiple sheets)")