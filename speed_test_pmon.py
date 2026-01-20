"""
Speed test script for p-mon computation.
Instruments the actual nash_welfare_optimizer.py to track timing of each operation.

Tracks the following steps:
- Step 1: CP without p-mon constraint (initial Nash welfare solve)
- Step 2: Find optimal constraint (cutting plane method)
- Step 3: Create constraint objects from directions
- Step 4: CP with p-mon constraint (final Nash welfare solve)

Also tracks sub-operations within Step 2:
- solve_inner_optimization (the inner CP solves)
- swap_oracle_check
- dirderiv_oracle_check
"""

import numpy as np
import time
import csv
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools


@dataclass
class TimingStats:
    """Accumulates timing statistics for an operation."""
    total_time: float = 0.0
    call_count: int = 0
    times: List[float] = field(default_factory=list)
    
    def add(self, elapsed: float):
        self.total_time += elapsed
        self.call_count += 1
        self.times.append(elapsed)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0.0
    
    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0.0
    
    @property
    def q25(self) -> float:
        return float(np.percentile(self.times, 25)) if self.times else 0.0
    
    @property
    def q50(self) -> float:
        return float(np.percentile(self.times, 50)) if self.times else 0.0
    
    @property
    def q75(self) -> float:
        return float(np.percentile(self.times, 75)) if self.times else 0.0
    
    @property
    def five_number_summary(self) -> str:
        """Returns (min; Q25; Q50; Q75; max) as a formatted string."""
        if not self.times:
            return "(0; 0; 0; 0; 0)"
        return f"({self.min_time:.4f}; {self.q25:.4f}; {self.q50:.4f}; {self.q75:.4f}; {self.max_time:.4f})"
    
    @property
    def time_bins(self) -> str:
        """Returns percentage of calls in each time bin as a formatted string.
        Bins: >=10, [1,10), [0.1,1), [0.01,0.1), <0.01
        """
        if not self.times:
            return "(0%; 0%; 0%; 0%; 0%)"
        
        n = len(self.times)
        bin_gte_10 = sum(1 for t in self.times if t >= 10) / n * 100
        bin_1_10 = sum(1 for t in self.times if 1 <= t < 10) / n * 100
        bin_01_1 = sum(1 for t in self.times if 0.1 <= t < 1) / n * 100
        bin_001_01 = sum(1 for t in self.times if 0.01 <= t < 0.1) / n * 100
        bin_lt_001 = sum(1 for t in self.times if t < 0.01) / n * 100
        
        return f"({bin_gte_10:.0f}%; {bin_1_10:.0f}%; {bin_01_1:.0f}%; {bin_001_01:.0f}%; {bin_lt_001:.0f}%)"


@dataclass 
class TestCaseResult:
    """Results from a single test case."""
    n_agents: int
    n_resources: int
    n_attackers: int
    total_time: float
    cutting_planes_added: int
    converged: bool
    pmon_value: float
    timings: Dict[str, TimingStats] = field(default_factory=dict)
    projection_iterations: int = 0
    inner_solves: int = 0
    cuts_from_projection: int = 0
    cuts_from_inner_solve: int = 0
    cut_details: Optional[List[Dict]] = None  # Details of each cutting plane added


class TimingTracker:
    """Tracks timing for various operations."""
    
    def __init__(self):
        self.stats: Dict[str, TimingStats] = {}
        self.current_test_stats: Dict[str, TimingStats] = {}
    
    def reset_current_test(self):
        """Reset stats for current test case."""
        self.current_test_stats = {}
    
    @contextmanager
    def track(self, operation_name: str):
        """Context manager to track time for an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            
            # Add to current test stats
            if operation_name not in self.current_test_stats:
                self.current_test_stats[operation_name] = TimingStats()
            self.current_test_stats[operation_name].add(elapsed)
            
            # Add to global stats
            if operation_name not in self.stats:
                self.stats[operation_name] = TimingStats()
            self.stats[operation_name].add(elapsed)
    
    def get_current_test_stats(self) -> Dict[str, TimingStats]:
        return self.current_test_stats.copy()


def generate_random_utilities(n_agents: int, n_resources: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random utility matrix using rod-cutting approach.
    Each agent's utilities sum to 1 (uniform random on simplex).
    """
    if seed is not None:
        np.random.seed(seed)
    
    utilities = np.zeros((n_agents, n_resources))
    
    for i in range(n_agents):
        # Rod cutting: generate n_resources-1 cut points, sort, take differences
        cuts = np.sort(np.random.uniform(0, 1, n_resources - 1))
        cuts = np.concatenate([[0], cuts, [1]])
        utilities[i] = np.diff(cuts)
    
    return utilities


def generate_test_case(seed: Optional[int] = None, config: Optional[Dict] = None) -> Tuple[int, int, int, np.ndarray, np.ndarray, Set[int]]:
    """
    Generate a random test case.
    
    Args:
        seed: Random seed
        config: Optional dict with keys:
            - min_agents, max_agents (default: 5, 15)
            - min_resources, max_resources (default: 3, 20)
            - min_attackers, max_attackers (default: 1, 10)
    
    Returns:
        n_agents, n_resources, n_attackers, utilities, supply, attacking_group
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default config
    if config is None:
        config = {}
    
    min_agents = config.get('min_agents', 5)
    max_agents = config.get('max_agents', 15)
    min_resources = config.get('min_resources', 3)
    max_resources = config.get('max_resources', 20)
    min_attackers = config.get('min_attackers', 1)
    max_attackers = config.get('max_attackers', 10)
    
    n_agents = np.random.randint(min_agents, max_agents + 1)
    n_resources = np.random.randint(min_resources, max_resources + 1)
    # Cap attackers at n_agents - 1 (need at least one non-attacker)
    actual_max_attackers = min(max_attackers, n_agents - 1)
    actual_min_attackers = min(min_attackers, actual_max_attackers)
    n_attackers = np.random.randint(actual_min_attackers, actual_max_attackers + 1)
    
    utilities = generate_random_utilities(n_agents, n_resources)
    supply = np.ones(n_resources)  # Unit supply for simplicity
    
    # Randomly select attackers
    attacking_group = set(np.random.choice(n_agents, n_attackers, replace=False))
    
    return n_agents, n_resources, n_attackers, utilities, supply, attacking_group


def run_single_test(n_agents: int, n_resources: int, n_attackers: int,
                    utilities: np.ndarray, supply: np.ndarray, 
                    attacking_group: Set[int], tracker: TimingTracker,
                    verbose: bool = False, use_projection: bool = True,
                    use_two_phase: bool = True) -> TestCaseResult:
    """
    Run a single test case with the actual optimizer.
    
    Instruments the high-level steps:
    - Step 1: CP without p-mon constraint (initial solve)
    - Step 2: Find optimal constraint (cutting plane iterations)
    - Step 3: Create constraint objects
    - Step 4: CP with p-mon constraint (final solve)
    
    Also instruments sub-operations within Step 2.
    """
    import sys
    import nash_welfare_optimizer as nwo
    from nash_welfare_optimizer import (
        NashWelfareOptimizer, 
        ProportionalityConstraint,
        _solve_optimal_constraint_convex_program,
        _solve_inner_optimization,
        SwapOptimalityOracle,
        DirectionalDerivativeOracle
    )
    
    def print_step(step_num: int, step_name: str):
        """Print current step, overwriting the previous line."""
        sys.stdout.write(f"\r  Currently on step {step_num}/4: {step_name}...".ljust(60))
        sys.stdout.flush()
    
    def clear_step():
        """Clear the step line."""
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
    
    tracker.reset_current_test()
    
    start_time = time.perf_counter()
    
    try:
        # ====== STEP 1: CP without p-mon constraint (initial Nash solve) ======
        print_step(1, "Initial Nash solve")
        with tracker.track("step1_initial_nash_solve"):
            optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
            initial_result = optimizer.solve(verbose=verbose)
            initial_utilities = initial_result.agent_utilities
        
        # ====== STEP 2: Find optimal constraint ======
        print_step(2, "Find optimal constraint")
        
        # We need to instrument the internals of _solve_optimal_constraint_convex_program
        
        # Store original functions to wrap
        original_solve_inner = nwo._solve_inner_optimization
        original_project = nwo._project_to_feasible_region
        
        # Wrap _solve_inner_optimization
        def timed_solve_inner(*args, **kwargs):
            with tracker.track("step2_solve_inner_optimization"):
                return original_solve_inner(*args, **kwargs)
        nwo._solve_inner_optimization = timed_solve_inner
        
        # Wrap _project_to_feasible_region
        def timed_project(*args, **kwargs):
            with tracker.track("step2_projection"):
                return original_project(*args, **kwargs)
        nwo._project_to_feasible_region = timed_project
        
        # Also instrument the oracles if we can access them
        # We'll wrap the is_violated methods
        original_swap_is_violated = SwapOptimalityOracle.is_violated
        original_dirderiv_is_violated = DirectionalDerivativeOracle.is_violated
        
        def timed_swap_is_violated(self, x_free):
            with tracker.track("step2_swap_oracle_check"):
                return original_swap_is_violated(self, x_free)
        
        def timed_dirderiv_is_violated(self, x_free):
            with tracker.track("step2_dirderiv_oracle_check"):
                return original_dirderiv_is_violated(self, x_free)
        
        SwapOptimalityOracle.is_violated = timed_swap_is_violated
        DirectionalDerivativeOracle.is_violated = timed_dirderiv_is_violated
        
        try:
            with tracker.track("step2_find_optimal_constraint"):
                optimal_directions, convex_program_allocation, opt_status, cp_debug_info = _solve_optimal_constraint_convex_program(
                    utilities=utilities,
                    attacking_group=attacking_group,
                    target_group=attacking_group,
                    supply=supply,
                    initial_constraints=None,
                    maximize_harm=False,
                    verbose=verbose,
                    debug=False,
                    use_projection=use_projection,
                    use_two_phase=use_two_phase
                )
        finally:
            # Restore original functions
            nwo._solve_inner_optimization = original_solve_inner
            nwo._project_to_feasible_region = original_project
            SwapOptimalityOracle.is_violated = original_swap_is_violated
            DirectionalDerivativeOracle.is_violated = original_dirderiv_is_violated
        
        # ====== STEP 3: Create constraint objects ======
        print_step(3, "Create constraints")
        with tracker.track("step3_create_constraints"):
            optimal_constraints = {}
            if optimal_directions is not None and opt_status == 'optimal':
                for agent_id, direction in optimal_directions.items():
                    if direction is not None:
                        optimal_constraints[agent_id] = ProportionalityConstraint(n_resources, direction)
        
        # ====== STEP 4: CP with p-mon constraint (final Nash solve) ======
        print_step(4, "Final Nash solve")
        with tracker.track("step4_final_nash_solve"):
            if optimal_constraints:
                final_optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
                for agent_id, constraint in optimal_constraints.items():
                    final_optimizer.add_agent_constraint(agent_id, constraint)
                final_result = final_optimizer.solve(verbose=verbose)
                final_utilities = final_result.agent_utilities
            else:
                final_result = initial_result
                final_utilities = initial_utilities.copy()
        
        clear_step()
        
        total_time = time.perf_counter() - start_time
        
        # Compute p-MON value
        pmon_individual = {}
        for agent_id in attacking_group:
            if final_utilities[agent_id] > 1e-10:
                pmon_individual[agent_id] = initial_utilities[agent_id] / final_utilities[agent_id]
            else:
                pmon_individual[agent_id] = float('inf')
        
        # Group p-MON is geometric mean
        pmon_values = [pmon_individual[a] for a in attacking_group if pmon_individual[a] < float('inf')]
        if pmon_values:
            pmon_group = np.exp(np.mean(np.log(pmon_values)))
        else:
            pmon_group = float('inf')
        
        # Get cutting planes info
        cutting_planes = cp_debug_info.get('total_cuts', 0) if cp_debug_info else 0
        converged = cp_debug_info.get('converged', False) if cp_debug_info else False
        projection_iters = cp_debug_info.get('projection_iterations', 0) if cp_debug_info else 0
        inner_solves = cp_debug_info.get('inner_solves', 0) if cp_debug_info else 0
        cuts_from_projection = cp_debug_info.get('cuts_from_projection', 0) if cp_debug_info else 0
        cuts_from_inner_solve = cp_debug_info.get('cuts_from_inner_solve', 0) if cp_debug_info else 0
        cut_details = cp_debug_info.get('cut_details', []) if cp_debug_info else []
        
        return TestCaseResult(
            n_agents=n_agents,
            n_resources=n_resources,
            n_attackers=n_attackers,
            total_time=total_time,
            cutting_planes_added=cutting_planes,
            converged=converged,
            pmon_value=pmon_group if pmon_group < float('inf') else 0.0,
            timings=tracker.get_current_test_stats(),
            projection_iterations=projection_iters,
            inner_solves=inner_solves,
            cuts_from_projection=cuts_from_projection,
            cuts_from_inner_solve=cuts_from_inner_solve,
            cut_details=cut_details
        )
        
    except Exception as e:
        clear_step()
        total_time = time.perf_counter() - start_time
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        return TestCaseResult(
            n_agents=n_agents,
            n_resources=n_resources,
            n_attackers=n_attackers,
            total_time=total_time,
            cutting_planes_added=0,
            converged=False,
            pmon_value=0.0,
            timings=tracker.get_current_test_stats()
        )


def run_speed_test(n_test_cases: int = 10, seed: Optional[int] = None, 
                   verbose: bool = False, config: Optional[Dict] = None,
                   use_projection: bool = True, use_two_phase: bool = True) -> Tuple[List[TestCaseResult], Dict[str, TimingStats]]:
    """
    Run speed test with random test cases.
    
    Args:
        n_test_cases: Number of test cases to run
        seed: Random seed for reproducibility
        verbose: Enable verbose output
        config: Configuration dict for test case generation
    
    Returns:
        (list of test case results, global timing stats)
    """
    tracker = TimingTracker()
    
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    
    for i in range(n_test_cases):
        print(f"\nRunning test case {i+1}/{n_test_cases}...")
        
        # Generate random test case
        case_seed = seed + i if seed is not None else None
        n_agents, n_resources, n_attackers, utilities, supply, attacking_group = generate_test_case(case_seed, config)
        
        print(f"  n_agents={n_agents}, n_resources={n_resources}, n_attackers={n_attackers}")
        
        result = run_single_test(
            n_agents, n_resources, n_attackers,
            utilities, supply, attacking_group,
            tracker,
            verbose=verbose,
            use_projection=use_projection,
            use_two_phase=use_two_phase
        )
        
        print(f"  Completed in {result.total_time:.3f}s, cuts={result.cutting_planes_added}, "
              f"proj_iters={result.projection_iterations}, inner_solves={result.inner_solves}, "
              f"converged={result.converged}, pmon={result.pmon_value:.4f}")
        
        results.append(result)
    
    return results, tracker.stats


def generate_timing_report(results: List[TestCaseResult], global_stats: Dict[str, TimingStats]) -> Tuple[str, str]:
    """
    Generate timing report as CSV strings.
    
    Returns:
        (timing_csv, summary_csv)
    """
    # Timing breakdown table
    timing_rows = []
    
    total_time = sum(r.total_time for r in results)
    
    # Operations to report (in hierarchical order)
    operations = [
        # High-level steps
        "step1_initial_nash_solve",
        "step2_find_optimal_constraint",
        "step3_create_constraints",
        "step4_final_nash_solve",
        # Sub-operations within step 2
        "step2_solve_inner_optimization",
        "step2_projection",
        "step2_swap_oracle_check",
        "step2_dirderiv_oracle_check",
    ]
    
    timing_rows.append(["Operation", "Total Time (s)", "Call Count", "Avg Time (s)", "% of Total", "(Min; Q25; Q50; Q75; Max)", "(>=10; [1-10); [.1-1); [.01-.1); <.01)"])
    
    for op in operations:
        if op in global_stats:
            stats = global_stats[op]
            pct = (stats.total_time / total_time * 100) if total_time > 0 else 0
            timing_rows.append([
                op,
                f"{stats.total_time:.4f}",
                str(stats.call_count),
                f"{stats.avg_time:.6f}",
                f"{pct:.1f}%",
                stats.five_number_summary,
                stats.time_bins
            ])
    
    # Add any other operations that were tracked but not in our list
    for op, stats in global_stats.items():
        if op not in operations:
            pct = (stats.total_time / total_time * 100) if total_time > 0 else 0
            timing_rows.append([
                op,
                f"{stats.total_time:.4f}",
                str(stats.call_count),
                f"{stats.avg_time:.6f}",
                f"{pct:.1f}%",
                stats.five_number_summary,
                stats.time_bins
            ])
    
    timing_rows.append([])
    timing_rows.append(["TOTAL", f"{total_time:.4f}", "", "", "100%"])
    
    # Summary table
    summary_rows = []
    summary_rows.append(["Test Case", "n_agents", "n_resources", "n_attackers", 
                         "Total Time (s)", "Cuts", "Cuts(proj)", "Cuts(inner)", "Proj Iters", "Inner Solves", "Converged", "P-MON"])
    
    for i, r in enumerate(results):
        summary_rows.append([
            str(i + 1),
            str(r.n_agents),
            str(r.n_resources),
            str(r.n_attackers),
            f"{r.total_time:.4f}",
            str(r.cutting_planes_added),
            str(r.cuts_from_projection),
            str(r.cuts_from_inner_solve),
            str(r.projection_iterations),
            str(r.inner_solves),
            str(r.converged),
            f"{r.pmon_value:.4f}"
        ])
    
    # Add summary statistics
    summary_rows.append([])
    summary_rows.append(["SUMMARY STATISTICS", "", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Total test cases", str(len(results)), "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Converged", str(sum(1 for r in results if r.converged)), "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Total time", f"{sum(r.total_time for r in results):.4f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg time per case", f"{np.mean([r.total_time for r in results]):.4f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg cutting planes", f"{np.mean([r.cutting_planes_added for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg cuts from proj", f"{np.mean([r.cuts_from_projection for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg cuts from inner", f"{np.mean([r.cuts_from_inner_solve for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg proj iterations", f"{np.mean([r.projection_iterations for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg inner solves", f"{np.mean([r.inner_solves for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg n_agents", f"{np.mean([r.n_agents for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg n_resources", f"{np.mean([r.n_resources for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    summary_rows.append(["Avg n_attackers", f"{np.mean([r.n_attackers for r in results]):.1f}", "", "", "", "", "", "", "", "", "", ""])
    
    converged_results = [r for r in results if r.converged]
    if converged_results:
        summary_rows.append(["Avg pmon value (converged)", f"{np.mean([r.pmon_value for r in converged_results]):.4f}", "", "", "", "", "", "", "", "", "", ""])
    
    # Convert to CSV strings
    def rows_to_csv(rows):
        lines = []
        for row in rows:
            lines.append(",".join(str(cell) for cell in row))
        return "\n".join(lines)
    
    return rows_to_csv(timing_rows), rows_to_csv(summary_rows)


def print_report(timing_csv: str, summary_csv: str):
    """Print formatted report to console."""
    print("\n" + "="*200)
    print("TIMING BREAKDOWN")
    print("="*200)
    
    for line in timing_csv.split("\n"):
        if line:
            parts = line.split(",")
            if len(parts) >= 7:
                print(f"{parts[0]:<35} {parts[1]:>14} {parts[2]:>12} {parts[3]:>14} {parts[4]:>12} {parts[5]:>45} {parts[6]:>35}")
            elif len(parts) >= 6:
                print(f"{parts[0]:<35} {parts[1]:>14} {parts[2]:>12} {parts[3]:>14} {parts[4]:>12} {parts[5]:>45}")
            elif len(parts) >= 5:
                print(f"{parts[0]:<35} {parts[1]:>14} {parts[2]:>12} {parts[3]:>14} {parts[4]:>12}")
            else:
                print(line)
    
    print("\n" + "="*200)
    print("TEST CASE SUMMARY")
    print("="*200)
    
    for line in summary_csv.split("\n"):
        if line:
            parts = line.split(",")
            if len(parts) >= 12:
                print(f"{parts[0]:<15} {parts[1]:>8} {parts[2]:>11} {parts[3]:>11} {parts[4]:>13} {parts[5]:>6} {parts[6]:>11} {parts[7]:>12} {parts[8]:>11} {parts[9]:>13} {parts[10]:>10} {parts[11]:>8}")
            elif len(parts) >= 2:
                print(f"{parts[0]:<25} {parts[1]:>8}")
            else:
                print(line)


def save_reports(timing_csv: str, summary_csv: str, prefix: str = "speed_test"):
    """Save reports to CSV files in results/speed_tests/ directory."""
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "results/speed_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    timing_file = os.path.join(output_dir, f"{prefix}_timing.csv")
    summary_file = os.path.join(output_dir, f"{prefix}_summary.csv")
    
    with open(timing_file, "w") as f:
        f.write(timing_csv)
    
    with open(summary_file, "w") as f:
        f.write(summary_csv)
    
    print(f"\nReports saved to: {timing_file}, {summary_file}")
    
    return timing_file, summary_file


def save_cut_details(result: 'TestCaseResult', prefix: str = "speed_test"):
    """
    Save cut details from the first simulation to a CSV file.
    
    For each cutting plane, outputs:
    - cut_number: Sequential number of the cut
    - cut_type: 'swap' or 'dir_deriv'
    - source: 'projection', 'inner_solve', or 'inner_solve_after_projection'
    - iteration: Which iteration the cut was added
    - x: The violating point (flattened array as semicolon-separated string)
    - x_star: The boundary point (flattened array as semicolon-separated string)
    - normal: The normal vector (flattened array as semicolon-separated string)
    - rhs: The right-hand side of the constraint
    """
    import os
    
    if not result.cut_details:
        print("No cut details to save.")
        return None
    
    output_dir = "results/speed_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    cut_file = os.path.join(output_dir, f"{prefix}_cut_details.csv")
    
    rows = []
    # Header
    rows.append([
        "cut_number", "cut_type", "source", "iteration", "rhs",
        "x (violating point)", "x_star (boundary point)", "normal"
    ])
    
    for cut in result.cut_details:
        # Convert numpy arrays to semicolon-separated strings
        x_str = ";".join(f"{v:.6f}" for v in cut['violating_point'])
        x_star_str = ";".join(f"{v:.6f}" for v in cut['boundary_point'])
        normal_str = ";".join(f"{v:.6f}" for v in cut['normal'])
        
        rows.append([
            str(cut['cut_number']),
            cut['cut_type'],
            cut.get('source', 'unknown'),
            str(cut['iteration']),
            f"{cut['rhs']:.6f}",
            x_str,
            x_star_str,
            normal_str
        ])
    
    # Write to CSV
    with open(cut_file, "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
    
    print(f"Cut details saved to: {cut_file}")
    print(f"  Total cuts: {len(result.cut_details)}")
    print(f"  Cuts from projection: {result.cuts_from_projection}")
    print(f"  Cuts from inner solve: {result.cuts_from_inner_solve}")
    print(f"  n_agents={result.n_agents}, n_resources={result.n_resources}")
    print(f"  Vector length: {result.n_agents * result.n_resources}")
    
    return cut_file


def main():
    """Run the speed test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speed test for p-MON computation")
    parser.add_argument("-n", "--n-test-cases", type=int, default=10, dest="n_test_cases",
                        help="Number of test cases to run (default: 10)")
    parser.add_argument("--min-agents", type=int, default=5, dest="min_agents",
                        help="Minimum number of agents (default: 5)")
    parser.add_argument("--max-agents", type=int, default=15, dest="max_agents",
                        help="Maximum number of agents (default: 15)")
    parser.add_argument("--min-resources", type=int, default=3, dest="min_resources",
                        help="Minimum number of resources (default: 3)")
    parser.add_argument("--max-resources", type=int, default=20, dest="max_resources",
                        help="Maximum number of resources (default: 20)")
    parser.add_argument("--min-attackers", type=int, default=1, dest="min_attackers",
                        help="Minimum number of attackers (default: 1)")
    parser.add_argument("--max-attackers", type=int, default=10, dest="max_attackers",
                        help="Maximum number of attackers (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output during optimization")
    parser.add_argument("-o", "--output-prefix", type=str, default="speed_test", dest="output_prefix",
                        help="Prefix for output CSV files (default: speed_test)")
    parser.add_argument("--solver", type=str, default="gurobi", choices=["gurobi", "cvxopt"],
                        help="Solver to use for inner optimization (default: gurobi, falls back to cvxopt if unavailable)")
    parser.add_argument("--project", type=str, default="true", choices=["true", "false"],
                        help="Enable projection-based optimization (default: true)")
    parser.add_argument("--two-phase", type=str, default="true", choices=["true", "false"], dest="two_phase",
                        help="Enable two-phase optimization (attackers then non-attackers) (default: true)")
    
    args = parser.parse_args()
    
    # Store ranges in a config dict for generate_test_case
    config = {
        'min_agents': args.min_agents,
        'max_agents': args.max_agents,
        'min_resources': args.min_resources,
        'max_resources': args.max_resources,
        'min_attackers': args.min_attackers,
        'max_attackers': args.max_attackers,
    }
    
    print("P-MON Speed Test")
    print("="*110)
    
    print(f"Running {args.n_test_cases} random test cases...")
    print(f"Parameters:")
    print(f"  n_agents: {args.min_agents}-{args.max_agents}")
    print(f"  n_resources: {args.min_resources}-{args.max_resources}")
    print(f"  n_attackers: {args.min_attackers}-{args.max_attackers} (capped at n_agents)")
    print(f"  utilities: uniform random (rod cutting)")
    print(f"  seed: {args.seed}")
    print(f"  solver: {args.solver}")
    print(f"  projection: {args.project}")
    print(f"  two-phase: {args.two_phase}")
    print(f"\nSteps tracked:")
    print(f"  Step 1: Initial Nash solve (CP without p-mon constraint)")
    print(f"  Step 2: Find optimal constraint (cutting plane method)")
    print(f"  Step 3: Create constraint objects")
    print(f"  Step 4: Final Nash solve (CP with p-mon constraint)")
    
    # Set the solver
    import nash_welfare_optimizer as nwo
    nwo.set_default_solver(args.solver)
    print(f"\nUsing solver: {args.solver}" + (" (will fall back to cvxopt if gurobi unavailable)" if args.solver == "gurobi" else ""))
    
    # Parse use_projection and use_two_phase
    use_projection = args.project.lower() == "true"
    use_two_phase = args.two_phase.lower() == "true"
    print(f"Using projection: {use_projection}")
    print(f"Using two-phase: {use_two_phase}")
    
    # Run tests
    results, global_stats = run_speed_test(
        n_test_cases=args.n_test_cases, 
        seed=args.seed,
        verbose=args.verbose,
        config=config,
        use_projection=use_projection,
        use_two_phase=use_two_phase
    )
    
    # Generate reports
    timing_csv, summary_csv = generate_timing_report(results, global_stats)
    
    # Print reports
    print_report(timing_csv, summary_csv)
    
    # Save reports
    timing_file, summary_file = save_reports(timing_csv, summary_csv, prefix=args.output_prefix)
    
    # Save cut details from first simulation
    if results and results[0].cut_details:
        save_cut_details(results[0], prefix=args.output_prefix)
    
    return timing_file, summary_file


if __name__ == "__main__":
    main()