import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from copy import deepcopy

solvers.options['show_progress'] = False


# =============================================================================
# Shared Constants and Helper Functions for Dependent Agent Selection
# =============================================================================

# Threshold for considering an allocation as "meaningful" (not numerical noise)
ALLOCATION_THRESHOLD = 1e-3

# Tolerance for considering ratios as "tied"
TIE_TOLERANCE = 1e-4


def has_meaningful_allocation(allocation: float) -> bool:
    """Check if an allocation value is meaningful (not just numerical noise)."""
    return allocation > ALLOCATION_THRESHOLD


def find_dependent_agents_for_resource(
    resource_idx: int,
    x_full: np.ndarray,
    ratios: np.ndarray,
    non_attacking_group: Set[int],
    find_min: bool = True
) -> Tuple[List[int], float]:
    """
    Find the dependent agent(s) for a resource.
    
    For forward direction (find_min=True): finds non-attackers with SMALLEST ratio who have allocation.
    For backward direction (find_min=False): finds non-attackers with LARGEST ratio.
    
    Handles ties by returning all agents within TIE_TOLERANCE of the extreme ratio.
    
    Args:
        resource_idx: Index of the resource (j)
        x_full: Full allocation matrix (n_agents x n_resources)
        ratios: Ratio matrix v_ij/V_i (n_agents x n_resources)
        non_attacking_group: Set of non-attacker agent IDs
        find_min: If True, find minimum ratio (forward). If False, find maximum (backward).
    
    Returns:
        (list of tied agent IDs, extreme ratio value)
        Returns ([], 0.0) if no valid agents found.
    """
    # Collect candidates
    candidates = []
    for agent in non_attacking_group:
        # For forward direction, agent must have allocation
        # For backward direction, any agent can receive resources
        if find_min:
            if not has_meaningful_allocation(x_full[agent, resource_idx]):
                continue
        candidates.append((agent, ratios[agent, resource_idx]))
    
    if not candidates:
        return [], 0.0
    
    # Find extreme ratio
    if find_min:
        extreme_ratio = min(ratio for _, ratio in candidates)
    else:
        extreme_ratio = max(ratio for _, ratio in candidates)
    
    # Find all agents tied at the extreme ratio
    tied_agents = [agent for agent, ratio in candidates if abs(ratio - extreme_ratio) < TIE_TOLERANCE]
    
    return tied_agents, extreme_ratio


def compute_dependent_ratios_for_all_resources(
    x_full: np.ndarray,
    ratios: np.ndarray,
    non_attacking_group: Set[int],
    n_resources: int,
    find_min: bool = True
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Compute dependent ratios for all resources, handling ties by averaging.
    
    Args:
        x_full: Full allocation matrix (n_agents x n_resources)
        ratios: Ratio matrix v_ij/V_i (n_agents x n_resources)
        non_attacking_group: Set of non-attacker agent IDs
        n_resources: Number of resources
        find_min: If True, find minimum ratio (forward). If False, find maximum (backward).
    
    Returns:
        (dependent_ratios array, list of dependent agents for each resource)
    """
    dependent_ratios = np.zeros(n_resources)
    dependent_agents_list = []
    
    for j in range(n_resources):
        tied_agents, extreme_ratio = find_dependent_agents_for_resource(
            j, x_full, ratios, non_attacking_group, find_min=find_min
        )
        
        if tied_agents:
            # Average the ratios of tied agents
            avg_ratio = sum(ratios[agent, j] for agent in tied_agents) / len(tied_agents)
            dependent_ratios[j] = avg_ratio
        else:
            dependent_ratios[j] = 0.0
        
        dependent_agents_list.append(tied_agents)
    
    return dependent_ratios, dependent_agents_list


# =============================================================================
# Standalone Normal Vector Computation
# =============================================================================

def compute_normal_at_boundary(
    x_full_boundary: np.ndarray,
    utilities: np.ndarray,
    attacking_group: Set[int],
    non_attacking_group: Set[int],
    n_agents: int,
    n_resources: int
) -> np.ndarray:
    """
    Compute the normal vector at a boundary point.
    
    The directional derivative is:
        D(x) = Σ_a Σ_j (v_aj/V_a - v_dj/V_d) * x_aj
    
    At the boundary, D(x) = 0. The gradient with respect to x_{a'j'} is:
        ∂D/∂x_{a'j'} = -(r_{j'} + Σ_{a∈A} Σ_{j∈I: t_j≠∅} (x_{aj} / (|t_j| * |t_{j'}|)) * r_{j'} * (Σ_{t∈t_j∩t_{j'}} v_{tj}/V_t))
    
    Where:
        r_{j'} = min ratio for resource j' (v_dj'/V_d for any tied agent d)
        t_j = set of tied dependent agents for resource j
        t_j ∩ t_{j'} = intersection of tied agents for resources j and j'
    
    The normal pointing toward the excluded zone is the negative of this gradient.
    
    Args:
        x_full_boundary: Full allocation matrix at boundary point (n_agents x n_resources)
        utilities: Utility matrix (n_agents x n_resources)
        attacking_group: Set of attacker agent IDs
        non_attacking_group: Set of non-attacker agent IDs
        n_agents: Number of agents
        n_resources: Number of resources
    
    Returns:
        Normal vector for the attacker's allocation space (n_resources dimensions), normalized.
        Points toward the excluded zone (where attackers would want to go but can't).
    """
    # Compute V at boundary
    V_boundary = np.array([np.dot(utilities[i], x_full_boundary[i]) for i in range(n_agents)])
    
    # Compute v_ij/V_i for all agents at boundary
    ratios = np.zeros((n_agents, n_resources))
    for i in range(n_agents):
        V_i = max(V_boundary[i], 1e-10)
        for j in range(n_resources):
            ratios[i, j] = utilities[i, j] / V_i
    
    # For each resource j, find the set of tied dependent agents (non-attackers with 
    # smallest v_dj/V_d who have allocation) and the minimum ratio r_j
    tied_agents = [set() for _ in range(n_resources)]  # t_j for each resource
    min_ratio = [None] * n_resources  # r_j for each resource
    
    for j in range(n_resources):
        # Find the minimum ratio among non-attackers with allocation
        min_r = float('inf')
        for d in non_attacking_group:
            if x_full_boundary[d, j] > ALLOCATION_THRESHOLD:  # Must have allocation
                if ratios[d, j] < min_r:
                    min_r = ratios[d, j]
        
        if min_r < float('inf'):
            min_ratio[j] = min_r
            # Find all agents tied at this minimum ratio
            for d in non_attacking_group:
                if x_full_boundary[d, j] > ALLOCATION_THRESHOLD:
                    if abs(ratios[d, j] - min_r) < TIE_TOLERANCE:
                        tied_agents[j].add(d)
    
    # Compute the gradient for each resource j' using the corrected formula:
    # ∂D/∂x_{a'j'} = -(r_{j'} + Σ_{a∈A} Σ_{j∈I: t_j≠∅} (x_{aj} / (|t_j| * |t_{j'}|)) * r_{j'} * (Σ_{t∈t_j∩t_{j'}} v_{tj}/V_t))
    # 
    # Normal points toward excluded zone, so we negate: normal = -gradient
    
    normal_attacker = np.zeros(n_resources)
    
    for j_prime in range(n_resources):
        # If t_{j'} is empty, the normal component for this resource is 0
        if len(tied_agents[j_prime]) == 0:
            normal_attacker[j_prime] = 0.0
            continue
        
        r_j_prime = min_ratio[j_prime]
        t_j_prime_size = len(tied_agents[j_prime])
        
        # Compute the sum term: Σ_{a∈A} Σ_{j∈I: t_j≠∅} (x_{aj} / (|t_j| * |t_{j'}|)) * r_{j'} * (Σ_{t∈t_j∩t_{j'}} v_{tj}/V_t)
        sum_term = 0.0
        for a in attacking_group:
            for j in range(n_resources):
                # Skip if t_j is empty
                if len(tied_agents[j]) == 0:
                    continue
                
                x_aj = x_full_boundary[a, j]
                t_j_size = len(tied_agents[j])
                
                # Compute intersection t_j ∩ t_{j'}
                intersection = tied_agents[j] & tied_agents[j_prime]
                
                # Compute inner sum: Σ_{t∈t_j∩t_{j'}} v_{tj}/V_t
                inner_sum = 0.0
                for t in intersection:
                    V_t = max(V_boundary[t], 1e-10)
                    v_tj = utilities[t, j]
                    inner_sum += v_tj / V_t
                
                # Add contribution to sum term
                sum_term += (x_aj / (t_j_size * t_j_prime_size)) * r_j_prime * inner_sum
        
        # Gradient = -(r_{j'} + sum_term)
        # Everything inside parentheses is positive, so gradient is negative
        gradient_j_prime = -(r_j_prime + sum_term)
        
        # Normal is negative of gradient (points toward excluded zone)
        # Since gradient is negative, normal will be positive
        normal_attacker[j_prime] = -gradient_j_prime
    
    # Normalize
    norm = np.linalg.norm(normal_attacker)
    if norm > 1e-10:
        normal_attacker = normal_attacker / norm
    
    return normal_attacker


# =============================================================================
# Diversity Constraints
# =============================================================================

class AgentDiversityConstraint(ABC):
    """
    Base class for agent-specific diversity constraints.
    Operates on a single agent's allocation x_i ∈ R^m.
    Must be convex and contain the origin.
    """
    
    def __init__(self, n_resources: int):
        self.n_resources = n_resources
    
    @abstractmethod
    def is_violated(self, x_i: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        pass
    
    def contains_origin(self) -> bool:
        violated, _, _ = self.is_violated(np.zeros(self.n_resources))
        return not violated


class CategoryBalanceConstraint(AgentDiversityConstraint):
    """Minimum fraction of allocation from specified categories."""
    
    def __init__(self, n_resources: int, category_masks: Dict[str, np.ndarray], 
                 min_fractions: Dict[str, float]):
        super().__init__(n_resources)
        self.category_masks = category_masks
        self.min_fractions = min_fractions
    
    def is_violated(self, x_i: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        total = x_i.sum()
        if total < 1e-8:
            return False, None, None
        
        for cat, mask in self.category_masks.items():
            if cat not in self.min_fractions:
                continue
            
            cat_alloc = x_i[mask].sum()
            required = self.min_fractions[cat] * total
            
            if cat_alloc < required - 1e-6:
                normal = np.ones(self.n_resources) * self.min_fractions[cat]
                normal[mask] -= 1
                return True, normal, 0.0
        
        return False, None, None


class MaxCategoryConstraint(AgentDiversityConstraint):
    """Maximum fraction of allocation from a category."""
    
    def __init__(self, n_resources: int, category_mask: np.ndarray, max_fraction: float):
        super().__init__(n_resources)
        self.category_mask = category_mask
        self.max_fraction = max_fraction
    
    def is_violated(self, x_i: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        total = x_i.sum()
        if total < 1e-8:
            return False, None, None
        
        cat_alloc = x_i[self.category_mask].sum()
        
        if cat_alloc > self.max_fraction * total + 1e-6:
            normal = -np.ones(self.n_resources) * self.max_fraction
            normal[self.category_mask] += 1
            return True, normal, 0.0
        
        return False, None, None


class LinearConstraint(AgentDiversityConstraint):
    """Linear constraint a'x_i <= b with b >= 0."""
    
    def __init__(self, n_resources: int, a: np.ndarray, b: float):
        super().__init__(n_resources)
        assert b >= 0, "b must be >= 0 for origin feasibility"
        self.a = np.array(a)
        self.b = b
    
    def is_violated(self, x_i: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        if self.a @ x_i > self.b + 1e-6:
            return True, self.a.copy(), self.b
        return False, None, None


class ProportionalityConstraint(AgentDiversityConstraint):
    """
    Constraint requiring allocation to follow specified ratios.
    
    Given a ratio vector r = [r_0, r_1, ..., r_{m-1}], the agent's allocation
    x_i must satisfy x_i[j] / x_i[k] = r[j] / r[k] for all pairs j, k where
    both r[j] and r[k] are non-zero.
    
    Equivalently, x_i must be proportional to r: x_i = t * r for some t >= 0.
    
    Example:
        r = [1, 2, 3] means:
        - For every 1 unit of item 0, must have 2 units of item 1 and 3 units of item 2
        - Valid allocations: [0,0,0], [1,2,3], [0.5,1,1.5], etc.
    
    This constraint always contains the origin (t=0 is valid).
    
    The constraint is enforced via separation oracle: if the current allocation
    violates the proportionality, we return a hyperplane that separates it from
    the feasible set.
    """
    
    def __init__(self, n_resources: int, ratios: np.ndarray):
        """
        Args:
            n_resources: Number of resources
            ratios: Ratio vector of shape (n_resources,). The allocation must
                   be proportional to this vector. Zero entries mean that
                   resource must not be allocated (x_i[j] = 0 if ratios[j] = 0).
        """
        super().__init__(n_resources)
        self.ratios = np.array(ratios, dtype=float)
        
        # Normalize ratios for numerical stability
        norm = np.linalg.norm(self.ratios)
        if norm > 1e-10:
            self.ratios_normalized = self.ratios / norm
        else:
            self.ratios_normalized = self.ratios
    
    def is_violated(self, x_i: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Check if allocation violates proportionality constraint.
        
        The allocation x_i is feasible iff x_i = t * ratios for some t >= 0.
        We check this by projecting x_i onto the ratios direction and seeing
        if the residual is zero.
        """
        # Handle zero allocation (always feasible - origin is in constraint set)
        x_norm = np.linalg.norm(x_i)
        if x_norm < 1e-10:
            return False, None, None
        
        # Handle zero ratios vector (only zero allocation is feasible)
        r_norm = np.linalg.norm(self.ratios)
        if r_norm < 1e-10:
            # Ratios are all zero, so x_i must be zero
            if x_norm > 1e-6:
                # Return hyperplane that pushes toward zero
                return True, x_i / x_norm, 0.0
            return False, None, None
        
        # Project x_i onto ratios direction
        # t = (x_i · r) / (r · r)
        t = np.dot(x_i, self.ratios) / np.dot(self.ratios, self.ratios)
        
        # The projection is t * ratios
        projection = t * self.ratios
        
        # Residual: component of x_i not in ratios direction
        residual = x_i - projection
        residual_norm = np.linalg.norm(residual)
        
        # Check if residual is significant
        if residual_norm < 1e-6 * max(x_norm, 1.0):
            # Also check that t >= 0 (allocation in positive direction)
            if t >= -1e-6:
                return False, None, None
            else:
                # t < 0 means allocation is in opposite direction
                # Hyperplane: ratios · x_i >= 0, i.e., -ratios · x_i <= 0
                return True, -self.ratios, 0.0
        
        # Violated: return separating hyperplane
        # The hyperplane is perpendicular to the residual and passes through projection
        # residual · (x - projection) <= 0
        # residual · x <= residual · projection = 0 (since residual ⊥ projection)
        normal = residual / residual_norm
        rhs = 0.0
        
        return True, normal, rhs
    
    def get_ratios(self) -> np.ndarray:
        """Return the ratio vector."""
        return self.ratios.copy()
    
    def __repr__(self) -> str:
        return f"ProportionalityConstraint(ratios={self.ratios})"


class CompositeConstraint(AgentDiversityConstraint):
    """Combines multiple constraints (intersection)."""
    
    def __init__(self, n_resources: int, constraints: List[AgentDiversityConstraint]):
        super().__init__(n_resources)
        self.constraints = constraints
    
    def is_violated(self, x_i: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        for constraint in self.constraints:
            violated, normal, rhs = constraint.is_violated(x_i)
            if violated:
                return violated, normal, rhs
        return False, None, None


# =============================================================================
# Allocation Result
# =============================================================================

@dataclass
class AllocationResult:
    """Result of solving the allocation problem."""
    allocation: np.ndarray
    status: str
    nash_welfare: float
    agent_utilities: np.ndarray
    iterations: int
    cuts_by_agent: Dict[int, int]


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class SimulationMetrics:
    """
    Metrics computed from comparing initial and final allocations.
    
    Metric interpretations:
    - p_mon < 1: issuers BENEFITED from their constraint
    - p_mon > 1: issuers LOST value from their constraint
    - q_nee < 1: non-issuers were HARMED
    - q_nee > 1: non-issuers BENEFITED
    - welfare_loss_ratio < 1: total welfare decreased
    """
    
    # Welfare
    initial_nash_welfare: float
    final_nash_welfare: float
    welfare_loss_ratio: float
    
    # p-MON: Did issuers benefit? (< 1 means yes)
    issuer_ids: Set[int]
    p_mon_group: float
    p_mon_individual: Dict[int, float]
    
    # q-NEE: Were non-issuers harmed? (< 1 means yes)
    non_issuer_ids: Set[int]
    q_nee_group: float
    q_nee_individual: Dict[int, float]
    
    # Raw utilities
    initial_utilities: np.ndarray
    final_utilities: np.ndarray
    
    def summary(self, agent_names: Optional[List[str]] = None) -> str:
        n = len(self.initial_utilities)
        names = agent_names or [f"Agent {i}" for i in range(n)]
        
        lines = [
            "=" * 60,
            "SIMULATION METRICS",
            "=" * 60,
            "",
            f"Welfare Loss Ratio: {self.welfare_loss_ratio:.6f}",
            f"  Initial Nash Welfare: {self.initial_nash_welfare:.6f}",
            f"  Final Nash Welfare:   {self.final_nash_welfare:.6f}",
            "",
            f"p-MON (issuer benefit, <1 = benefited): {self.p_mon_group:.6f}",
            f"  Issuers: {[names[i] for i in sorted(self.issuer_ids)]}",
            "  Individual p-MON:"
        ]
        for i in sorted(self.issuer_ids):
            status = "benefited" if self.p_mon_individual[i] < 1 else "lost"
            lines.append(f"    {names[i]}: {self.p_mon_individual[i]:.6f} ({status})")
        
        lines.extend([
            "",
            f"q-NEE (non-issuer harm, <1 = harmed): {self.q_nee_group:.6f}",
            f"  Non-issuers: {[names[i] for i in sorted(self.non_issuer_ids)]}",
            "  Individual q-NEE:"
        ])
        for i in sorted(self.non_issuer_ids):
            status = "harmed" if self.q_nee_individual[i] < 1 else "benefited"
            lines.append(f"    {names[i]}: {self.q_nee_individual[i]:.6f} ({status})")
        
        lines.extend([
            "",
            "Utility Changes:"
        ])
        for i in range(n):
            pct = (self.final_utilities[i] / self.initial_utilities[i] - 1) * 100
            role = "issuer" if i in self.issuer_ids else "non-issuer"
            lines.append(f"  {names[i]} ({role}): {self.initial_utilities[i]:.4f} -> "
                        f"{self.final_utilities[i]:.4f} ({pct:+.1f}%)")
        
        return "\n".join(lines)
    
    def to_dict(self, run_id: Optional[int] = None) -> Dict:
        """Convert metrics to a flat dictionary for DataFrame export."""
        d = {
            'initial_nash_welfare': self.initial_nash_welfare,
            'final_nash_welfare': self.final_nash_welfare,
            'welfare_loss_ratio': self.welfare_loss_ratio,
            'p_mon_group': self.p_mon_group,
            'q_nee_group': self.q_nee_group,
            'n_issuers': len(self.issuer_ids),
            'n_non_issuers': len(self.non_issuer_ids),
            'issuer_ids': ','.join(map(str, sorted(self.issuer_ids))),
            'non_issuer_ids': ','.join(map(str, sorted(self.non_issuer_ids))),
        }
        
        if run_id is not None:
            d['run_id'] = run_id
        
        # Add individual metrics
        for i, val in self.p_mon_individual.items():
            d[f'p_mon_agent_{i}'] = val
        for i, val in self.q_nee_individual.items():
            d[f'q_nee_agent_{i}'] = val
        
        # Add utilities
        for i, val in enumerate(self.initial_utilities):
            d[f'initial_utility_agent_{i}'] = val
        for i, val in enumerate(self.final_utilities):
            d[f'final_utility_agent_{i}'] = val
        
        return d


def compute_metrics(
    initial_result: AllocationResult,
    final_result: AllocationResult,
    issuer_ids: Set[int],
    n_agents: int
) -> SimulationMetrics:
    """
    Compute all metrics comparing initial and final allocations.
    
    p-MON is the RECIPROCAL ratio: initial/final
    - < 1 means issuers benefited (final > initial)
    - > 1 means issuers lost value (final < initial)
    """
    initial_u = initial_result.agent_utilities
    final_u = final_result.agent_utilities
    
    non_issuer_ids = set(range(n_agents)) - issuer_ids
    
    # Welfare loss ratio (final / initial)
    welfare_loss_ratio = final_result.nash_welfare / initial_result.nash_welfare
    
    # p-MON: RECIPROCAL ratio for issuers (initial / final)
    # < 1 means they benefited
    p_mon_individual = {}
    for i in issuer_ids:
        if final_u[i] > 1e-10:
            p_mon_individual[i] = initial_u[i] / final_u[i]
        else:
            p_mon_individual[i] = float('inf') if initial_u[i] > 1e-10 else 1.0
    
    if issuer_ids:
        k = len(issuer_ids)
        log_sum = sum(np.log(max(r, 1e-10)) for r in p_mon_individual.values())
        p_mon_group = np.exp(log_sum / k)
    else:
        p_mon_group = 1.0
    
    # q-NEE: ratio for non-issuers (final / initial)
    # < 1 means they were harmed
    q_nee_individual = {}
    for i in non_issuer_ids:
        if initial_u[i] > 1e-10:
            q_nee_individual[i] = final_u[i] / initial_u[i]
        else:
            q_nee_individual[i] = float('inf') if final_u[i] > 1e-10 else 1.0
    
    if non_issuer_ids:
        k = len(non_issuer_ids)
        log_sum = sum(np.log(max(r, 1e-10)) for r in q_nee_individual.values())
        q_nee_group = np.exp(log_sum / k)
    else:
        q_nee_group = 1.0
    
    return SimulationMetrics(
        initial_nash_welfare=initial_result.nash_welfare,
        final_nash_welfare=final_result.nash_welfare,
        welfare_loss_ratio=welfare_loss_ratio,
        issuer_ids=issuer_ids,
        p_mon_group=p_mon_group,
        p_mon_individual=p_mon_individual,
        non_issuer_ids=non_issuer_ids,
        q_nee_group=q_nee_group,
        q_nee_individual=q_nee_individual,
        initial_utilities=initial_u.copy(),
        final_utilities=final_u.copy()
    )


# =============================================================================
# Core Optimizer
# =============================================================================

class NashWelfareOptimizer:
    """Solves Nash welfare resource allocation with per-agent diversity constraints."""
    
    def __init__(self, n_agents: int, n_resources: int, utilities: np.ndarray, 
                 supply: Optional[np.ndarray] = None):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.n_vars = n_agents * n_resources
        self.utilities = np.array(utilities)
        self.supply = supply if supply is not None else np.ones(n_resources)
        self.agent_constraints: Dict[int, Optional[AgentDiversityConstraint]] = {
            i: None for i in range(n_agents)
        }
    
    def set_agent_constraint(self, agent_id: int, 
                             constraint: Optional[AgentDiversityConstraint]) -> 'NashWelfareOptimizer':
        if constraint is not None and not constraint.contains_origin():
            raise ValueError(f"Constraint for agent {agent_id} doesn't contain origin")
        self.agent_constraints[agent_id] = constraint
        return self
    
    def add_agent_constraint(self, agent_id: int, 
                             constraint: AgentDiversityConstraint) -> 'NashWelfareOptimizer':
        """Add constraint, combining with existing if present."""
        existing = self.agent_constraints.get(agent_id)
        if existing is not None:
            combined = CompositeConstraint(self.n_resources, [existing, constraint])
            self.agent_constraints[agent_id] = combined
        else:
            self.agent_constraints[agent_id] = constraint
        return self
    
    def clear_all_constraints(self) -> 'NashWelfareOptimizer':
        self.agent_constraints = {i: None for i in range(self.n_agents)}
        return self
    
    def _get_agent_allocation(self, x: np.ndarray, agent_id: int) -> np.ndarray:
        start = agent_id * self.n_resources
        return x[start:start + self.n_resources]
    
    def _compute_utilities(self, x: np.ndarray) -> np.ndarray:
        utilities = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            x_i = self._get_agent_allocation(x, i)
            utilities[i] = np.dot(self.utilities[i], x_i)
        return utilities
    
    def _nash_welfare(self, x: np.ndarray) -> float:
        utilities = self._compute_utilities(x)
        utilities = np.maximum(utilities, 1e-10)
        return np.sum(np.log(utilities))
    
    def _gradient_nash(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.n_vars)
        for i in range(self.n_agents):
            u_i = np.dot(self.utilities[i], self._get_agent_allocation(x, i))
            if u_i > 1e-10:
                start = i * self.n_resources
                grad[start:start + self.n_resources] = self.utilities[i] / u_i
        return grad
    
    def _build_base_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        G_list, h_list = [], []
        
        # x >= 0
        G_list.append(-np.eye(self.n_vars))
        h_list.append(np.zeros(self.n_vars))
        
        # x <= supply (per resource, per agent)
        G_list.append(np.eye(self.n_vars))
        h_list.append(np.tile(self.supply, self.n_agents))
        
        # sum_i x_ij <= supply[j] for each resource j
        for j in range(self.n_resources):
            row = np.zeros(self.n_vars)
            for i in range(self.n_agents):
                row[i * self.n_resources + j] = 1
            G_list.append(row.reshape(1, -1))
            h_list.append(np.array([self.supply[j]]))
        
        return np.vstack(G_list), np.concatenate(h_list)
    
    def _check_agent_constraints(self, x: np.ndarray) -> List[Tuple[int, np.ndarray, float]]:
        violations = []
        for agent_id, constraint in self.agent_constraints.items():
            if constraint is None:
                continue
            
            x_i = self._get_agent_allocation(x, agent_id)
            violated, normal_local, rhs = constraint.is_violated(x_i)
            
            if violated:
                normal_full = np.zeros(self.n_vars)
                start = agent_id * self.n_resources
                normal_full[start:start + self.n_resources] = normal_local
                violations.append((agent_id, normal_full, rhs))
        
        return violations
    
    def solve(self, max_outer: int = 50, max_inner: int = 100, 
              tol: float = 1e-6, verbose: bool = False) -> AllocationResult:
        """
        Solve Nash welfare maximization using CVXOPT's convex programming solver.
        
        Uses solvers.cp which can directly handle the concave log objective.
        Cutting planes are added for diversity constraints via separation oracle.
        """
        
        G_base, h_base = self._build_base_constraints()
        cuts_G, cuts_h = [], []
        cuts_by_agent = {i: 0 for i in range(self.n_agents)}
        
        violations = []
        x = None
        
        for outer in range(max_outer):
            # Combine base constraints with cutting planes
            if cuts_G:
                G = np.vstack([G_base] + cuts_G)
                h = np.concatenate([h_base] + cuts_h)
            else:
                G = G_base
                h = h_base
            
            # Solve inner problem: maximize Nash welfare subject to current constraints
            x, status = self._solve_nash_welfare_cp(G, h, verbose)
            
            if status != 'optimal':
                return AllocationResult(
                    allocation=x.reshape(self.n_agents, self.n_resources),
                    status=f"Solver failed: {status}",
                    nash_welfare=self._nash_welfare(x),
                    agent_utilities=self._compute_utilities(x),
                    iterations=outer + 1,
                    cuts_by_agent=cuts_by_agent
                )
            
            # Check diversity constraints
            violations = self._check_agent_constraints(x)
            
            if not violations:
                if verbose:
                    print(f"Converged after {outer + 1} outer iterations")
                break
            
            # Add cutting planes for violated constraints
            for agent_id, normal, rhs in violations:
                cuts_G.append(normal.reshape(1, -1))
                cuts_h.append(np.array([rhs]))
                cuts_by_agent[agent_id] += 1
                if verbose:
                    print(f"Iteration {outer + 1}: Cut for agent {agent_id}")
        
        status = "optimal" if not violations else "max_iterations"
        
        return AllocationResult(
            allocation=x.reshape(self.n_agents, self.n_resources),
            status=status,
            nash_welfare=self._nash_welfare(x),
            agent_utilities=self._compute_utilities(x),
            iterations=outer + 1,
            cuts_by_agent=cuts_by_agent
        )
    
    def _solve_nash_welfare_cp(self, G: np.ndarray, h: np.ndarray, 
                                verbose: bool = False) -> Tuple[np.ndarray, str]:
        """
        Solve max sum_i log(u_i) using CVXOPT's convex programming solver.
        
        We minimize the negative: min -sum_i log(u_i)
        where u_i = sum_j v_ij * x_ij
        
        The solver requires a function F that returns:
        - F(): number of nonlinear constraints, initial point
        - F(x): (f, Df) objective value and gradient
        - F(x, z): (f, Df, H) objective, gradient, and Hessian
        """
        n_vars = self.n_vars
        n_agents = self.n_agents
        n_resources = self.n_resources
        utilities_matrix = self.utilities
        
        def F(x=None, z=None):
            """
            Objective function for solvers.cp
            
            We minimize: -sum_i log(u_i) where u_i = v_i' @ x_i
            """
            if x is None:
                # Return (number of nonlinear constraints, initial point)
                # We have 0 nonlinear constraints (just the objective)
                # Initial point: equal allocation
                x0 = matrix(0.0, (n_vars, 1))
                for j in range(n_resources):
                    for i in range(n_agents):
                        x0[i * n_resources + j] = 1.0 / n_agents
                return (0, x0)
            
            # Convert to numpy for easier indexing
            x_np = np.array(x).flatten()
            
            # Compute utilities for each agent
            u = np.zeros(n_agents)
            for i in range(n_agents):
                start = i * n_resources
                x_i = x_np[start:start + n_resources]
                u[i] = np.dot(utilities_matrix[i], x_i)
                
                # Ensure utility is positive (required for log)
                if u[i] <= 0:
                    return (None, None, None) if z is not None else (None, None)
            
            # Objective: f = -sum log(u_i)
            f_val = -np.sum(np.log(u))
            f = matrix(f_val)
            
            # Gradient: df/dx_ij = -v_ij / u_i
            Df = matrix(0.0, (1, n_vars))
            for i in range(n_agents):
                start = i * n_resources
                for j in range(n_resources):
                    Df[start + j] = -utilities_matrix[i, j] / u[i]
            
            if z is None:
                return (f, Df)
            
            # Hessian: d^2f / dx_ij dx_ik = v_ij * v_ik / u_i^2
            # This is a block diagonal matrix (agents don't interact in Hessian)
            H = matrix(0.0, (n_vars, n_vars))
            for i in range(n_agents):
                start = i * n_resources
                for j1 in range(n_resources):
                    for j2 in range(n_resources):
                        H[start + j1, start + j2] = (
                            z[0] * utilities_matrix[i, j1] * utilities_matrix[i, j2] / (u[i] ** 2)
                        )
            
            return (f, Df, H)
        
        # Convert constraints to CVXOPT matrices
        G_cvx = matrix(G.astype(float))
        h_cvx = matrix(h.astype(float))
        
        # Solve using CVXOPT's convex programming solver
        old_show_progress = solvers.options.get('show_progress', False)
        solvers.options['show_progress'] = verbose
        
        try:
            sol = solvers.cp(F, G=G_cvx, h=h_cvx)
        finally:
            solvers.options['show_progress'] = old_show_progress
        
        if sol['status'] == 'optimal' or sol['status'] == 'unknown':
            x = np.array(sol['x']).flatten()
            # Ensure non-negative (numerical cleanup)
            x = np.maximum(x, 0)
            return x, 'optimal' if sol['status'] == 'optimal' else sol['status']
        else:
            # Return a fallback equal allocation
            x = np.zeros(n_vars)
            for j in range(n_resources):
                for i in range(n_agents):
                    x[i * n_resources + j] = 1.0 / n_agents
            return x, sol['status']


# =============================================================================
# Optimal Constraint Generation
# =============================================================================

class DirectionalDerivativeOracle:
    """
    Separation oracle for the directional derivative constraint.
    
    This oracle checks whether a given allocation is on the optimal boundary
    by examining the directional derivative of the Nash welfare in the direction
    of scaling up the attacker's allocation.
    
    For attackers: direction is their current allocation
    For non-attackers: use greedy assignment to absorb negative of attacker direction
    
    If the directional derivative is negative, we've overshot the boundary and
    return a separating hyperplane.
    """
    
    def __init__(self, n_agents: int, n_resources: int, utilities: np.ndarray,
                 attacking_group: Set[int], supply: Optional[np.ndarray] = None,
                 verbose: bool = False, track_timing: bool = False,
                 use_integral_method: bool = False):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.utilities = utilities
        self.attacking_group = attacking_group
        self.non_attacking_group = set(range(n_agents)) - attacking_group
        self.supply = supply if supply is not None else np.ones(n_resources)
        self.verbose = verbose
        self.n_free_vars = n_agents * n_resources
        self.use_integral_method = use_integral_method
        
        # Counters for debugging
        self.n_backward_checks = 0
        self.n_backward_violations = 0
        
        # Store the last computed boundary point (set by _find_separating_hyperplane)
        self.last_boundary_point = None
        
        # Timing tracking (enabled via track_timing=True)
        self.track_timing = track_timing
        self.timing_stats = {
            'check_validity': {'total': 0.0, 'count': 0, 'times': []},
            'binary_search': {'total': 0.0, 'count': 0, 'times': []},
            'binary_search_nash_solves': {'total': 0.0, 'count': 0, 'times': []},
            'compute_normal': {'total': 0.0, 'count': 0, 'times': []},
        }
    
    def reset_timing_stats(self):
        """Reset all timing statistics."""
        for key in self.timing_stats:
            self.timing_stats[key] = {'total': 0.0, 'count': 0, 'times': []}
    
    def get_timing_stats(self) -> Dict:
        """Get timing statistics with computed averages."""
        result = {}
        for key, stats in self.timing_stats.items():
            result[key] = {
                'total': stats['total'],
                'count': stats['count'],
                'avg': stats['total'] / stats['count'] if stats['count'] > 0 else 0.0,
                'times': stats['times'].copy()
            }
        return result
    
    def _record_timing(self, operation: str, elapsed: float):
        """Record timing for an operation."""
        if self.track_timing:
            self.timing_stats[operation]['total'] += elapsed
            self.timing_stats[operation]['count'] += 1
            self.timing_stats[operation]['times'].append(elapsed)
    
    def _get_full_allocation(self, x_free: np.ndarray) -> np.ndarray:
        return x_free.reshape((self.n_agents, self.n_resources))
    
    def _compute_utilities_from_allocation(self, x_full: np.ndarray) -> np.ndarray:
        return np.array([
            np.dot(self.utilities[i], x_full[i]) 
            for i in range(self.n_agents)
        ])
    
    def _compute_direction(self, x_full: np.ndarray, V: np.ndarray, backward: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Compute the direction vector for the directional derivative test.
        
        Forward direction (backward=False):
            For attackers: direction is their current allocation (scaling up)
            For non-attackers: use greedy assignment to absorb negative of attacker direction
                              (assign to non-attacker with SMALLEST v_{i,j}/V_i who has allocation)
        
        Backward direction (backward=True):
            For attackers: direction is NEGATIVE of their current allocation (scaling down)
            For non-attackers: use greedy assignment giving POSITIVE direction
                              (assign to non-attacker with LARGEST v_{i,j}/V_i)
        
        Returns:
            (direction, valid): direction matrix and whether it's valid 
                               (False if we couldn't assign direction for some resource)
        """
        direction = np.zeros((self.n_agents, self.n_resources))
        valid = True
        
        if self.verbose:
            dir_type = "BACKWARD" if backward else "FORWARD"
            print(f"    [DirDeriv._compute_direction] Computing {dir_type} direction...")
            print(f"    [DirDeriv._compute_direction] Current allocations:")
            for i in range(self.n_agents):
                role = "ATK" if i in self.attacking_group else "VIC"
                print(f"      Agent {i} ({role}): {x_full[i].round(6)}, V={V[i]:.6f}")
        
        # Compute ratios for all agents
        ratios = np.zeros((self.n_agents, self.n_resources))
        for i in range(self.n_agents):
            V_i = max(V[i], 1e-10)
            for j in range(self.n_resources):
                ratios[i, j] = self.utilities[i, j] / V_i
        
        # Attackers: direction is their allocation (positive for forward, negative for backward)
        sign = -1.0 if backward else 1.0
        for i in self.attacking_group:
            direction[i] = sign * x_full[i]
        
        # For each resource, compute total attacker direction and assign to non-attackers
        for j in range(self.n_resources):
            attacker_total = sum(x_full[i, j] for i in self.attacking_group)
            
            if self.verbose:
                print(f"    [DirDeriv._compute_direction] Resource {j}: attacker_total={attacker_total:.6f}")
            
            if attacker_total < 1e-10:
                if self.verbose:
                    print(f"      -> No attacker allocation, skipping")
                continue
            
            # Use shared helper to find tied agents
            # backward=True means find max ratio, backward=False means find min ratio
            tied_agents, extreme_ratio = find_dependent_agents_for_resource(
                j, x_full, ratios, self.non_attacking_group, find_min=(not backward)
            )
            
            if self.verbose:
                ratio_type = "LARGEST" if backward else "SMALLEST with allocation"
                print(f"      Non-attacker ratios for resource {j} (looking for {ratio_type}):")
                for i in self.non_attacking_group:
                    has_alloc = has_meaningful_allocation(x_full[i, j])
                    alloc_str = f"x={x_full[i,j]:.6f}" if has_alloc else f"x={x_full[i,j]:.6f} (skipped)"
                    print(f"        Agent {i}: {alloc_str}, v={self.utilities[i,j]:.4f}, V={max(V[i],1e-10):.6f}, ratio={ratios[i,j]:.6f}")
            
            if tied_agents:
                # Divide the direction equally among tied agents
                if backward:
                    direction_per_agent = attacker_total / len(tied_agents)  # Positive (gaining)
                else:
                    direction_per_agent = -attacker_total / len(tied_agents)  # Negative (losing)
                
                for agent in tied_agents:
                    direction[agent, j] = direction_per_agent
                
                if self.verbose:
                    print(f"      -> {len(tied_agents)} agents tied at ratio {extreme_ratio:.6f}: {tied_agents}")
                    print(f"      -> Assigned direction {direction_per_agent:.6f} to each")
            else:
                valid = False
                if self.verbose:
                    if backward:
                        print(f"      -> WARNING: No non-attacker found for resource {j}")
                    else:
                        print(f"      -> No non-attacker has resource {j}, forward direction INVALID")
        
        if self.verbose:
            print(f"    [DirDeriv._compute_direction] Final direction matrix (valid={valid}):")
            for i in range(self.n_agents):
                role = "ATK" if i in self.attacking_group else "VIC"
                print(f"      Agent {i} ({role}): {direction[i].round(6)}")
        
        return direction, valid
    
    def _compute_directional_derivative_with_direction(self, x_full: np.ndarray, V: np.ndarray, 
                                                        direction: np.ndarray) -> float:
        """Compute the directional derivative of log Nash welfare given a direction."""
        dir_deriv = 0.0
        if self.verbose:
            print(f"    [DirDeriv._compute_directional_derivative] Computing contributions:")
        for i in range(self.n_agents):
            V_i = max(V[i], 1e-10)
            for j in range(self.n_resources):
                contrib = (self.utilities[i, j] / V_i) * direction[i, j]
                if self.verbose and abs(contrib) > 1e-10:
                    role = "ATK" if i in self.attacking_group else "VIC"
                    print(f"      Agent {i} ({role}), Resource {j}: (v={self.utilities[i,j]:.4f}/V={V_i:.6f}) * dir={direction[i,j]:.6f} = {contrib:.6f}")
                dir_deriv += contrib
        
        if self.verbose:
            print(f"    [DirDeriv._compute_directional_derivative] Total: {dir_deriv:.6e}")
        
        return dir_deriv
    
    def is_violated(self, x_free: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Check if the directional derivative constraint is violated."""
        import time as _time
        
        if self.verbose:
            print(f"  [DirDerivOracle.is_violated] Checking directional derivative...")
        
        # Start timing for check_validity
        _t_check_start = _time.perf_counter()
        
        x_full = self._get_full_allocation(x_free)
        V = self._compute_utilities_from_allocation(x_full)
        
        if self.verbose:
            print(f"    Allocation matrix:")
            for i in range(self.n_agents):
                role = "ATK" if i in self.attacking_group else "VIC"
                print(f"      Agent {i} ({role}): {x_full[i].round(6)}, V={V[i]:.6f}")
        
        # Check for non-positive utilities among non-attackers
        non_attacker_zero_utility = any(V[i] <= 1e-10 for i in self.non_attacking_group)
        
        if non_attacker_zero_utility:
            # Record check_validity time
            self._record_timing('check_validity', _time.perf_counter() - _t_check_start)
            
            if self.verbose:
                print(f"    Non-attacker has zero utility - this means we've definitely overshot")
                print(f"    (taking from them would be -∞ change, giving to them would be +∞ benefit)")
                print(f"    Proceeding directly to boundary search...")
            
            # We've definitely overshot - go directly to boundary search
            # For integral method, we need a D_0 value - use a large negative number
            if self.use_integral_method:
                normal, rhs = self._find_separating_hyperplane_integral(x_free, x_full, V, -1.0)
            else:
                normal, rhs = self._find_separating_hyperplane(x_free, x_full, V)
            
            if self.verbose:
                print(f"    Cutting plane: normal · x <= {rhs:.6f}")
            
            return True, normal, rhs
        
        # Try forward direction first
        forward_direction, forward_valid = self._compute_direction(x_full, V, backward=False)
        
        if forward_valid:
            # Forward direction is valid, compute directional derivative
            forward_deriv = self._compute_directional_derivative_with_direction(x_full, V, forward_direction)
            
            if self.verbose:
                print(f"    Forward directional derivative: {forward_deriv:.6e}")
            
            if forward_deriv >= -1e-8:
                # Record check_validity time (not violated case)
                self._record_timing('check_validity', _time.perf_counter() - _t_check_start)
                
                if self.verbose:
                    print(f"    -> NOT VIOLATED (forward dir_deriv >= 0)")
                return False, None, None
            
            # Record check_validity time (violated case - before binary search)
            self._record_timing('check_validity', _time.perf_counter() - _t_check_start)
            
            # Forward derivative is negative - we've overshot
            if self.verbose:
                print(f"    -> VIOLATED (forward dir_deriv < 0), finding separating hyperplane...")
            
            if self.use_integral_method:
                normal, rhs = self._find_separating_hyperplane_integral(x_free, x_full, V, forward_deriv)
            else:
                normal, rhs = self._find_separating_hyperplane(x_free, x_full, V)
            
            if self.verbose:
                print(f"    Cutting plane: normal · x <= {rhs:.6f}")
            
            return True, normal, rhs
        
        else:
            # Forward direction is invalid (no non-attacker has some resource)
            # This means we're at the maximum in forward direction for that resource
            # Check backward direction to see if we've overshot
            if self.verbose:
                print(f"    Forward direction invalid, checking backward direction...")
            
            self.n_backward_checks += 1
            
            backward_direction, backward_valid = self._compute_direction(x_full, V, backward=True)
            
            if not backward_valid:
                # Record check_validity time (not violated case)
                self._record_timing('check_validity', _time.perf_counter() - _t_check_start)
                
                if self.verbose:
                    print(f"    Backward direction also invalid, returning not violated")
                return False, None, None
            
            backward_deriv = self._compute_directional_derivative_with_direction(x_full, V, backward_direction)
            
            if self.verbose:
                print(f"    Backward directional derivative: {backward_deriv:.6e}")
            
            if backward_deriv <= 1e-8:
                # Record check_validity time (not violated case)
                self._record_timing('check_validity', _time.perf_counter() - _t_check_start)
                
                # Backward derivative is non-positive, meaning going backward wouldn't help
                # We haven't overshot
                if self.verbose:
                    print(f"    -> NOT VIOLATED (backward dir_deriv <= 0, we haven't overshot)")
                return False, None, None
            
            # Record check_validity time (violated case - before binary search)
            self._record_timing('check_validity', _time.perf_counter() - _t_check_start)
            
            # Backward derivative is positive - going backward would help
            # This means we've overshot in the forward direction
            # Use the same binary search - as we scale down, forward direction becomes valid
            self.n_backward_violations += 1
            
            if self.verbose:
                print(f"    -> VIOLATED (backward dir_deriv > 0, we've overshot)")
                print(f"    Finding separating hyperplane...")
            
            # For integral method with backward derivative, we use negative of backward_deriv
            # since backward_deriv > 0 means we're past the boundary
            if self.use_integral_method:
                normal, rhs = self._find_separating_hyperplane_integral(x_free, x_full, V, -backward_deriv)
            else:
                normal, rhs = self._find_separating_hyperplane(x_free, x_full, V)
            
            if self.verbose:
                print(f"    Cutting plane: normal · x <= {rhs:.6f}")
            
            return True, normal, rhs
    
    def _find_separating_hyperplane(self, x_free: np.ndarray, x_full: np.ndarray,
                                     V: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find separating hyperplane by binary search for boundary.
        
        At each step:
        1. Scale down only the attacker's allocation by t_mid
        2. Solve Nash welfare for non-attackers with remaining supply
        3. Compute directional derivative at this combined allocation
        """
        import time as _time
        
        # Start timing for binary search
        _t_bs_start = _time.perf_counter()
        _t_nash_total = 0.0  # Accumulate Nash solve time
        
        t_low, t_high = 0.0, 1.0
        
        # Get current attacker allocation
        attacker_allocation_current = {}
        for i in self.attacking_group:
            attacker_allocation_current[i] = x_full[i].copy()
        
        if self.verbose:
            print(f"")
            print(f"    ╔══════════════════════════════════════════════════════════════════╗")
            print(f"    ║          BINARY SEARCH FOR BOUNDARY - START                      ║")
            print(f"    ╚══════════════════════════════════════════════════════════════════╝")
            print(f"      Attacker allocation at t=1:")
            for i in self.attacking_group:
                print(f"        Agent {i}: {attacker_allocation_current[i].round(6)}")
        
        # Temporarily disable verbose for internal _compute_direction calls
        saved_verbose = self.verbose
        
        for iteration in range(50):
            t_mid = (t_low + t_high) / 2
            
            # Scale attacker allocation
            attacker_allocation_scaled = {}
            for i in self.attacking_group:
                attacker_allocation_scaled[i] = attacker_allocation_current[i] * t_mid
            
            # Compute remaining supply for non-attackers
            remaining_supply = self.supply.copy()
            for i in self.attacking_group:
                remaining_supply -= attacker_allocation_scaled[i]
            
            # Check if remaining supply is valid
            if np.any(remaining_supply < -1e-10):
                t_high = t_mid
                if saved_verbose:
                    print(f"      [BS iter {iteration:2d}] t={t_mid:.6f} -> negative supply, t_high <- t_mid")
                continue
            
            remaining_supply = np.maximum(remaining_supply, 0)  # Clamp to non-negative
            
            # Solve Nash welfare for non-attackers with remaining supply (timed)
            _t_nash_start = _time.perf_counter()
            non_attacker_allocation = self._solve_nash_for_non_attackers(remaining_supply)
            _t_nash_total += _time.perf_counter() - _t_nash_start
            
            if non_attacker_allocation is None:
                t_low = t_mid
                if saved_verbose:
                    print(f"      [BS iter {iteration:2d}] t={t_mid:.6f} -> Nash solve failed, t_low <- t_mid")
                continue
            
            # Build full allocation
            x_full_scaled = np.zeros((self.n_agents, self.n_resources))
            for i in self.attacking_group:
                x_full_scaled[i] = attacker_allocation_scaled[i]
            for i in self.non_attacking_group:
                x_full_scaled[i] = non_attacker_allocation[i]
            
            V_scaled = self._compute_utilities_from_allocation(x_full_scaled)
            
            if np.any(V_scaled <= 1e-10):
                t_low = t_mid
                if saved_verbose:
                    print(f"      [BS iter {iteration:2d}] t={t_mid:.6f} -> V has zeros, t_low <- t_mid")
                continue
            
            # Compute forward direction at scaled point (with verbose OFF)
            self.verbose = False
            scaled_direction, scaled_valid = self._compute_direction(x_full_scaled, V_scaled, backward=False)
            self.verbose = saved_verbose
            
            if not scaled_valid:
                # Forward still invalid at this scale, need to scale down more
                t_high = t_mid
                if saved_verbose:
                    print(f"      [BS iter {iteration:2d}] t={t_mid:.6f} -> forward invalid, t_high <- t_mid")
                continue
            
            # Compute directional derivative (with verbose OFF)
            self.verbose = False
            dir_deriv = self._compute_directional_derivative_with_direction(x_full_scaled, V_scaled, scaled_direction)
            self.verbose = saved_verbose
            
            if saved_verbose:
                action = "t_low <- t_mid" if dir_deriv > 0 else "t_high <- t_mid"
                if abs(dir_deriv) < 1e-8:
                    action = "CONVERGED"
                print(f"      [BS iter {iteration:2d}] t={t_mid:.6f}, dir_deriv={dir_deriv:+.6e} -> {action}")
            
            if abs(dir_deriv) < 1e-8:
                if saved_verbose:
                    print(f"      Binary search converged at iteration {iteration}")
                break
            elif dir_deriv > 0:
                t_low = t_mid
            else:
                t_high = t_mid
        
        if saved_verbose:
            print(f"      Final t_mid = {t_mid:.6f}")
        
        # Compute final boundary point
        attacker_allocation_boundary = {}
        for i in self.attacking_group:
            attacker_allocation_boundary[i] = attacker_allocation_current[i] * t_mid
        
        remaining_supply_boundary = self.supply.copy()
        for i in self.attacking_group:
            remaining_supply_boundary -= attacker_allocation_boundary[i]
        remaining_supply_boundary = np.maximum(remaining_supply_boundary, 0)
        
        # Final Nash solve (timed)
        _t_nash_start = _time.perf_counter()
        non_attacker_allocation_boundary = self._solve_nash_for_non_attackers(remaining_supply_boundary)
        _t_nash_total += _time.perf_counter() - _t_nash_start
        
        x_full_boundary = np.zeros((self.n_agents, self.n_resources))
        for i in self.attacking_group:
            x_full_boundary[i] = attacker_allocation_boundary[i]
        if non_attacker_allocation_boundary is not None:
            for i in self.non_attacking_group:
                x_full_boundary[i] = non_attacker_allocation_boundary[i]
        
        # Store the boundary point for later retrieval
        self.last_boundary_point = x_full_boundary.flatten().copy()
        
        # Record binary search timing (excluding compute_normal)
        _t_bs_end = _time.perf_counter()
        self._record_timing('binary_search', _t_bs_end - _t_bs_start)
        self._record_timing('binary_search_nash_solves', _t_nash_total)
        
        # Start timing for compute_normal
        _t_normal_start = _time.perf_counter()
        
        # Use the standalone function to compute the normal vector (single point of control)
        normal_attacker = compute_normal_at_boundary(
            x_full_boundary, self.utilities, self.attacking_group,
            self.non_attacking_group, self.n_agents, self.n_resources
        )
        
        # Expand to full space: normal is only for attacker dimensions
        # All non-attacker components are 0
        normal_full = np.zeros((self.n_agents, self.n_resources))
        for a in self.attacking_group:
            normal_full[a] = normal_attacker
        
        normal = normal_full.flatten()
        
        # Handle zero normal case
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            # Fallback: use direction from x* to x_violating directly
            if saved_verbose:
                print(f"      WARNING: Normal is zero, falling back to direction vector")
            x_violating_full = self._get_full_allocation(x_free)
            direction = x_violating_full - x_full_boundary
            normal = direction.flatten()
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
        
        # Record compute_normal timing
        self._record_timing('compute_normal', _time.perf_counter() - _t_normal_start)
        
        x_boundary_flat = x_full_boundary.flatten()
        
        # Compute dot products
        boundary_dot = np.dot(normal, x_boundary_flat)
        violating_dot = np.dot(normal, x_free)
        
        # Flip normal if needed so that violating point has higher dot product than boundary
        # This ensures the constraint normal · x <= rhs excludes the violating point
        if violating_dot < boundary_dot:
            normal = -normal
            boundary_dot = -boundary_dot
            violating_dot = -violating_dot
        
        rhs = boundary_dot
        
        if saved_verbose:
            V_boundary = self._compute_utilities_from_allocation(x_full_boundary)
            print(f"      Boundary point found:")
            print(f"        V at boundary: {V_boundary.round(6)}")
            print(f"        Normal (attacker components): {normal_attacker.round(6)}")
            print(f"        rhs = normal · x* = {rhs:.6f}")
            print(f"        Violating point: normal · x_free = {violating_dot:.6f}")
            violation = violating_dot - rhs
            print(f"        Violation (lhs - rhs): {violation:.6f} (should be > 0)")
            print(f"    ╔══════════════════════════════════════════════════════════════════╗")
            print(f"    ║          BINARY SEARCH FOR BOUNDARY - END                        ║")
            print(f"    ╚══════════════════════════════════════════════════════════════════╝")
            print(f"")
        
        return normal, rhs
    
    def _solve_integral_step(self, V: np.ndarray, direction: np.ndarray, D_current: float, debug: bool = False) -> float:
        """
        Solve for step size x using the integral formula.
        
        We want to find x such that:
            integral_0^x sum_i (delta_V_i / (V_i + delta_V_i * t)) dt = -D_current
        
        Which gives:
            sum_i ln((V_i + delta_V_i * x) / V_i) = -D_current
            prod_i ((V_i + delta_V_i * x) / V_i) = exp(-D_current)
        
        This is a polynomial in x. We solve it numerically using Newton's method
        or bisection.
        
        Args:
            V: Current utilities for all agents
            direction: Direction matrix (n_agents, n_resources)
            D_current: Current directional derivative value
            debug: If True, print debug information
            
        Returns:
            x: Step size to reach the boundary (estimated)
        """
        # Compute delta_V_i = d_i · v_i for each agent
        delta_V = np.array([
            np.dot(direction[i], self.utilities[i]) 
            for i in range(self.n_agents)
        ])
        
        target = np.exp(-D_current)
        
        if debug:
            print(f"\n      [_solve_integral_step DEBUG]")
            print(f"        D_current = {D_current:.6e}")
            print(f"        target = exp(-D_current) = {target:.6f}")
            print(f"\n        Per-agent details:")
            for i in range(self.n_agents):
                role = "ATK" if i in self.attacking_group else "DEF"
                print(f"          Agent {i} ({role}):")
                print(f"            v[{i}] (utilities) = {self.utilities[i].round(6)}")
                print(f"            d[{i}] (direction) = {direction[i].round(6)}")
                print(f"            V[{i}] (total utility) = {V[i]:.6f}")
                print(f"            delta_V[{i}] = d·v = {delta_V[i]:.6f}")
                if V[i] > 1e-10:
                    print(f"            delta_V[{i}]/V[{i}] = {delta_V[i]/V[i]:.6f}")
            print(f"\n        delta_V = {delta_V.round(6)}")
            print(f"        V = {V.round(6)}")
            # Verify: sum(delta_V / V) should equal D_current
            D_check = sum(delta_V[i] / V[i] for i in range(self.n_agents) if V[i] > 1e-10)
            print(f"        Verification: sum(delta_V/V) = {D_check:.6e} (should equal D_current)")
        
        # Define the function f(x) = prod_i ((V_i + delta_V_i * x) / V_i) - target
        # We want to find x where f(x) = 0
        
        def f(x):
            product = 1.0
            for i in range(self.n_agents):
                if abs(V[i]) < 1e-10:
                    continue
                ratio = (V[i] + delta_V[i] * x) / V[i]
                if ratio <= 0:
                    return float('inf') if x > 0 else float('-inf')
                product *= ratio
            return product - target
        
        def f_derivative(x):
            """Derivative of f(x) for Newton's method."""
            # d/dx prod_i (1 + delta_V_i * x / V_i) 
            # = sum_j (delta_V_j / V_j) * prod_{i != j} (1 + delta_V_i * x / V_i)
            # = prod_i (1 + delta_V_i * x / V_i) * sum_j (delta_V_j / (V_j + delta_V_j * x))
            product = 1.0
            sum_term = 0.0
            for i in range(self.n_agents):
                if abs(V[i]) < 1e-10:
                    continue
                denom = V[i] + delta_V[i] * x
                if abs(denom) < 1e-10:
                    return float('inf')
                ratio = (V[i] + delta_V[i] * x) / V[i]
                if ratio <= 0:
                    return float('inf')
                product *= ratio
                sum_term += delta_V[i] / denom
            return product * sum_term
        
        if debug:
            print(f"        f(0) = {f(0):.6f} (should be 1 - target = {1 - target:.6f})")
            # Test positive x values in [0, 1] range (direction vector handles the sign)
            for test_x in [0.1, 0.2, 0.5, 0.8, 1.0]:
                fx = f(test_x)
                if not np.isinf(fx):
                    print(f"        f({test_x:.1f}) = {fx:.6f}")
                else:
                    print(f"        f({test_x:.1f}) = inf (invalid - ratio went negative)")
        
        # Use Newton's method with bisection fallback
        # x should always be positive in [0, 1] since direction vector handles the sign
        # - For backward direction: direction[a] = -x_current[a], so x in [0,1] scales down
        # - For forward direction: direction[a] = +x_current[a], so x in [0,1] scales up (but not beyond 2x)
        
        f_0 = f(0)
        f_1 = f(1.0)
        
        if debug:
            print(f"        f(0) = {f_0:.6f}, f(1) = {f_1:.6f}")
        
        # Check if root is in [0, 1]
        if f_0 * f_1 < 0:
            # Root is in [0, 1]
            x_lo, x_hi = 0.0, 1.0
        elif abs(f_0) < 1e-10:
            # Already at root
            if debug:
                print(f"        Already at root (f(0) ≈ 0)")
            return 0.0
        elif abs(f_1) < 1e-10:
            # Root at x=1
            if debug:
                print(f"        Root at x=1")
            return 1.0
        else:
            # Root might be beyond x=1, search further
            x_lo, x_hi = 0.0, 1.0
            if f_0 * f_1 > 0:
                # Same sign, need to extend search
                if debug:
                    print(f"        No sign change in [0,1], extending search...")
                # Try extending to [0, 2] or beyond
                for x_test in [1.5, 2.0, 3.0, 5.0]:
                    f_test = f(x_test)
                    if debug:
                        print(f"        f({x_test}) = {f_test:.6f}")
                    if not np.isinf(f_test) and f_0 * f_test < 0:
                        x_hi = x_test
                        break
                else:
                    if debug:
                        print(f"        WARNING: Could not find bracket, using x=0.5 as fallback")
                    return 0.5
        
        if debug:
            print(f"        Bracket: [{x_lo:.6f}, {x_hi:.6f}]")
            print(f"        f(x_lo) = {f(x_lo):.6f}, f(x_hi) = {f(x_hi):.6f}")
        
        # Newton's method with bisection fallback
        x = (x_lo + x_hi) / 2
        for iteration in range(50):
            fx = f(x)
            
            if abs(fx) < 1e-10:
                break
                
            # Try Newton step
            fpx = f_derivative(x)
            if abs(fpx) > 1e-10:
                x_newton = x - fx / fpx
                # Accept Newton step if it's within bracket and makes progress
                if x_lo < x_newton < x_hi:
                    x_new = x_newton
                else:
                    # Bisection step
                    x_new = (x_lo + x_hi) / 2
            else:
                # Bisection step
                x_new = (x_lo + x_hi) / 2
            
            # Update bracket
            if f(x_new) * f(x_lo) < 0:
                x_hi = x_new
            else:
                x_lo = x_new
            
            x = x_new
            
            if abs(x_hi - x_lo) < 1e-12:
                break
        
        if debug:
            print(f"        Solution: x = {x:.6f}")
            print(f"        Verification: f(x) = {f(x):.6e}")
        
        return x
    
    def _find_separating_hyperplane_integral(self, x_free: np.ndarray, x_full: np.ndarray,
                                              V: np.ndarray, D_0: float) -> Tuple[np.ndarray, float]:
        """
        Find separating hyperplane using integral-based stepping.
        
        Instead of binary search, we use the integral formula to estimate
        the step size to reach the boundary, then refine with additional steps.
        
        Args:
            x_free: Original free variables (flattened)
            x_full: Original allocation matrix
            V: Original utilities
            D_0: Original directional derivative (should be negative)
        """
        import time as _time
        
        # Start timing
        _t_bs_start = _time.perf_counter()
        _t_nash_total = 0.0
        n_iterations = 0
        
        saved_verbose = self.verbose
        
        # Current state
        x_current = x_full.copy()
        V_current = V.copy()
        D_current = D_0
        
        # Track cumulative scaling factor (starts at 1.0 = original point)
        t_cumulative = 1.0
        
        # Original attacker allocation (we scale from this)
        attacker_allocation_original = {}
        for i in self.attacking_group:
            attacker_allocation_original[i] = x_full[i].copy()
        
        if saved_verbose:
            print(f"")
            print(f"    ╔══════════════════════════════════════════════════════════════════╗")
            print(f"    ║       INTEGRAL-BASED BOUNDARY SEARCH - START                     ║")
            print(f"    ╚══════════════════════════════════════════════════════════════════╝")
            print(f"      Initial D_0 = {D_0:.6e}")
        
        max_iterations = 20
        
        for iteration in range(max_iterations):
            n_iterations += 1
            
            # Choose direction based on D_current:
            # D < 0: attackers have too much, need to scale DOWN → use BACKWARD direction
            # D > 0: attackers have too little, need to scale UP → use FORWARD direction
            
            use_backward = (D_current < 0)
            
            self.verbose = False
            direction, direction_valid = self._compute_direction(x_current, V_current, backward=use_backward)
            self.verbose = saved_verbose
            
            if not direction_valid:
                # Try the other direction as fallback
                self.verbose = False
                direction, direction_valid = self._compute_direction(x_current, V_current, backward=(not use_backward))
                use_backward = not use_backward
                self.verbose = saved_verbose
                
                if not direction_valid:
                    if saved_verbose:
                        print(f"      [Iter {iteration}] No valid direction, stopping")
                    break
            
            if saved_verbose:
                print(f"      [Iter {iteration}] D={D_current:+.6e}, using {'BACKWARD' if use_backward else 'FORWARD'} direction")
            
            # Solve for step size using integral formula
            x_step = self._solve_integral_step(V_current, direction, D_current, debug=saved_verbose)
            
            if saved_verbose:
                print(f"      [Iter {iteration}] integral step x={x_step:.6f}")
            
            # The step x_step is in "direction units"
            # For FORWARD direction: direction[a] = +x_current[a]
            #   new_alloc = x_current + x_step * direction = x_current * (1 + x_step)
            #   t_new = t_cumulative * (1 + x_step)
            #
            # For BACKWARD direction: direction[a] = -x_current[a]  
            #   new_alloc = x_current + x_step * direction = x_current * (1 - x_step)
            #   t_new = t_cumulative * (1 - x_step)
            
            if use_backward:
                t_new = t_cumulative * (1 - x_step)
            else:
                t_new = t_cumulative * (1 + x_step)
            
            # Clamp to valid range
            t_new = max(0.0, min(1.0, t_new))
            
            if saved_verbose:
                print(f"      [Iter {iteration}] t_cumulative: {t_cumulative:.6f} -> {t_new:.6f}")
            
            # Check for convergence (no progress)
            if abs(t_new - t_cumulative) < 1e-10:
                if saved_verbose:
                    print(f"      [Iter {iteration}] No progress, stopping")
                break
            
            t_cumulative = t_new
            
            # Compute new attacker allocation
            attacker_allocation_new = {}
            for i in self.attacking_group:
                attacker_allocation_new[i] = attacker_allocation_original[i] * t_cumulative
            
            # Compute remaining supply
            remaining_supply = self.supply.copy()
            for i in self.attacking_group:
                remaining_supply -= attacker_allocation_new[i]
            
            if np.any(remaining_supply < -1e-10):
                if saved_verbose:
                    print(f"      [Iter {iteration}] Negative supply, adjusting t_cumulative")
                # Scale back to valid range
                t_cumulative = t_cumulative * 0.9
                continue
            
            remaining_supply = np.maximum(remaining_supply, 0)
            
            # Nash solve for non-attackers
            _t_nash_start = _time.perf_counter()
            non_attacker_allocation = self._solve_nash_for_non_attackers(remaining_supply)
            _t_nash_total += _time.perf_counter() - _t_nash_start
            
            if non_attacker_allocation is None:
                if saved_verbose:
                    print(f"      [Iter {iteration}] Nash solve failed")
                t_cumulative = t_cumulative * 0.9
                continue
            
            # Build new full allocation
            x_current = np.zeros((self.n_agents, self.n_resources))
            for i in self.attacking_group:
                x_current[i] = attacker_allocation_new[i]
            for i in self.non_attacking_group:
                x_current[i] = non_attacker_allocation[i]
            
            V_current = self._compute_utilities_from_allocation(x_current)
            
            if np.any(V_current <= 1e-10):
                if saved_verbose:
                    print(f"      [Iter {iteration}] Zero utility, adjusting")
                t_cumulative = t_cumulative * 0.9
                continue
            
            # Compute new directional derivative
            self.verbose = False
            new_direction, new_valid = self._compute_direction(x_current, V_current, backward=False)
            self.verbose = saved_verbose
            
            if new_valid:
                self.verbose = False
                D_current = self._compute_directional_derivative_with_direction(x_current, V_current, new_direction)
                self.verbose = saved_verbose
            else:
                # Try backward
                self.verbose = False
                new_direction, new_valid = self._compute_direction(x_current, V_current, backward=True)
                if new_valid:
                    D_current = self._compute_directional_derivative_with_direction(x_current, V_current, new_direction)
                self.verbose = saved_verbose
            
            if saved_verbose:
                print(f"      [Iter {iteration}] After step: t={t_cumulative:.6f}, D={D_current:+.6e}")
            
            # Check convergence
            if abs(D_current) < 1e-8:
                if saved_verbose:
                    print(f"      Converged at iteration {iteration}")
                break
        
        if saved_verbose:
            print(f"      Final: t_cumulative={t_cumulative:.6f}, D={D_current:+.6e}, iterations={n_iterations}")
        
        # Final boundary point
        x_full_boundary = x_current.copy()
        
        # Store the boundary point
        self.last_boundary_point = x_full_boundary.flatten().copy()
        
        # Record timing
        _t_bs_end = _time.perf_counter()
        self._record_timing('binary_search', _t_bs_end - _t_bs_start)
        self._record_timing('binary_search_nash_solves', _t_nash_total)
        
        # Compute normal
        _t_normal_start = _time.perf_counter()
        
        normal_attacker = compute_normal_at_boundary(
            x_full_boundary, self.utilities, self.attacking_group,
            self.non_attacking_group, self.n_agents, self.n_resources
        )
        
        normal_full = np.zeros((self.n_agents, self.n_resources))
        for a in self.attacking_group:
            normal_full[a] = normal_attacker
        
        normal = normal_full.flatten()
        
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            if saved_verbose:
                print(f"      WARNING: Normal is zero, falling back to direction vector")
            x_violating_full = self._get_full_allocation(x_free)
            direction = x_violating_full - x_full_boundary
            normal = direction.flatten()
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
        
        self._record_timing('compute_normal', _time.perf_counter() - _t_normal_start)
        
        x_boundary_flat = x_full_boundary.flatten()
        
        boundary_dot = np.dot(normal, x_boundary_flat)
        violating_dot = np.dot(normal, x_free)
        
        if violating_dot < boundary_dot:
            normal = -normal
            boundary_dot = -boundary_dot
            violating_dot = -violating_dot
        
        rhs = boundary_dot
        
        if saved_verbose:
            V_boundary = self._compute_utilities_from_allocation(x_full_boundary)
            print(f"      Boundary point found:")
            print(f"        V at boundary: {V_boundary.round(6)}")
            print(f"        Normal (attacker components): {normal_attacker.round(6)}")
            print(f"        rhs = normal · x* = {rhs:.6f}")
            print(f"        Violating point: normal · x_free = {violating_dot:.6f}")
            violation = violating_dot - rhs
            print(f"        Violation (lhs - rhs): {violation:.6f} (should be > 0)")
            print(f"    ╔══════════════════════════════════════════════════════════════════╗")
            print(f"    ║       INTEGRAL-BASED BOUNDARY SEARCH - END                       ║")
            print(f"    ╚══════════════════════════════════════════════════════════════════╝")
            print(f"")
        
        return normal, rhs
    
    def _solve_nash_for_non_attackers(self, remaining_supply: np.ndarray) -> Optional[Dict[int, np.ndarray]]:
        """
        Solve Nash welfare maximization for non-attackers given remaining supply.
        
        Returns dict mapping agent_id -> allocation array, or None if solve fails.
        """
        non_attackers = list(self.non_attacking_group)
        n_non_attackers = len(non_attackers)
        
        if n_non_attackers == 0:
            return {}
        
        # Check if there's any supply to allocate
        if np.all(remaining_supply < 1e-10):
            # No supply left, give everyone zeros
            return {i: np.zeros(self.n_resources) for i in non_attackers}
        
        # Build optimization problem for non-attackers
        # Variables: x_ij for each non-attacker i and resource j
        n_vars = n_non_attackers * self.n_resources
        
        # Map from (local_idx, resource) to variable index
        def var_idx(local_i, j):
            return local_i * self.n_resources + j
        
        # Objective: maximize sum of log utilities
        # We'll use CVXOPT's GP solver or convert to standard form
        # For simplicity, use the conelp solver with log barrier
        
        try:
            from cvxopt import matrix, solvers
            solvers.options['show_progress'] = False
            
            # We want to maximize sum_i log(V_i) where V_i = sum_j v_ij * x_ij
            # This is equivalent to maximizing prod_i V_i
            # Use geometric programming or reformulate
            
            # Actually, let's use a simpler approach: solve the KKT conditions
            # For Nash welfare with linear utilities, the optimal allocation gives
            # each agent equal value per unit of "bang per buck"
            
            # Even simpler: use cvxopt's cp solver for convex programming
            # Minimize -sum_i log(sum_j v_ij * x_ij)
            # Subject to: sum_i x_ij = remaining_supply_j, x_ij >= 0
            
            # Define the objective and gradient for cvxopt cp solver
            def F(x=None, z=None):
                if x is None:
                    # Return (m, x0) where m is number of nonlinear constraints (0 here)
                    # and x0 is initial point
                    x0 = matrix(0.0, (n_vars, 1))
                    for local_i, agent in enumerate(non_attackers):
                        for j in range(self.n_resources):
                            # Initial: equal share
                            x0[var_idx(local_i, j)] = remaining_supply[j] / n_non_attackers
                    return (0, x0)
                
                # Compute utilities
                V = []
                for local_i, agent in enumerate(non_attackers):
                    v_i = 0.0
                    for j in range(self.n_resources):
                        v_i += self.utilities[agent, j] * x[var_idx(local_i, j)]
                    V.append(max(v_i, 1e-12))
                
                # Objective: -sum log(V_i)
                f = sum(-np.log(v) for v in V)
                
                # Gradient
                Df = matrix(0.0, (1, n_vars))
                for local_i, agent in enumerate(non_attackers):
                    for j in range(self.n_resources):
                        Df[var_idx(local_i, j)] = -self.utilities[agent, j] / V[local_i]
                
                if z is None:
                    return (f, Df)
                
                # Hessian
                H = matrix(0.0, (n_vars, n_vars))
                for local_i, agent in enumerate(non_attackers):
                    for j1 in range(self.n_resources):
                        for j2 in range(self.n_resources):
                            idx1 = var_idx(local_i, j1)
                            idx2 = var_idx(local_i, j2)
                            H[idx1, idx2] = z[0] * self.utilities[agent, j1] * self.utilities[agent, j2] / (V[local_i] ** 2)
                
                return (f, Df, H)
            
            # Inequality constraints: -x <= 0 (i.e., x >= 0)
            G = matrix(-np.eye(n_vars))
            h = matrix(np.zeros(n_vars))
            
            # Equality constraints: sum_i x_ij = remaining_supply_j
            A = matrix(0.0, (self.n_resources, n_vars))
            b = matrix(remaining_supply)
            for j in range(self.n_resources):
                for local_i in range(n_non_attackers):
                    A[j, var_idx(local_i, j)] = 1.0
            
            sol = solvers.cp(F, G, h, A=A, b=b)
            
            if sol['status'] != 'optimal':
                return None
            
            x_sol = np.array(sol['x']).flatten()
            
            result = {}
            for local_i, agent in enumerate(non_attackers):
                alloc = np.zeros(self.n_resources)
                for j in range(self.n_resources):
                    alloc[j] = max(0, x_sol[var_idx(local_i, j)])
                result[agent] = alloc
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"      Nash solve exception: {e}")
            return None


class SwapOptimalityOracle:
    """
    Separation oracle for the swap-optimality constraint among non-attackers.
    
    This oracle checks whether the non-attacking group's allocation is swap-optimal
    among themselves. For swap-optimality, no beneficial swap should exist between
    any two non-attackers.
    """
    
    def __init__(self, n_agents: int, n_resources: int, utilities: np.ndarray,
                 attacking_group: Set[int], supply: Optional[np.ndarray] = None,
                 verbose: bool = False):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.utilities = utilities
        self.attacking_group = attacking_group
        self.non_attacking_group = set(range(n_agents)) - attacking_group
        self.supply = supply if supply is not None else np.ones(n_resources)
        self.verbose = verbose
        self.n_free_vars = n_agents * n_resources
        
        # Store the last computed boundary point (set by _find_separating_hyperplane)
        self.last_boundary_point = None
    
    def _get_full_allocation(self, x_free: np.ndarray) -> np.ndarray:
        return x_free.reshape((self.n_agents, self.n_resources))
    
    def _compute_utilities_from_allocation(self, x_full: np.ndarray) -> np.ndarray:
        return np.array([
            np.dot(self.utilities[i], x_full[i]) 
            for i in range(self.n_agents)
        ])
    
    def is_violated(self, x_free: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Check if the swap-optimality constraint is violated."""
        if self.verbose:
            print(f"  [SwapOracle.is_violated] Checking swap-optimality among non-attackers...")
        
        x_full = self._get_full_allocation(x_free)
        V = self._compute_utilities_from_allocation(x_full)
        
        # Compute total allocation per resource for non-attackers
        # Skip resources where non-attackers have essentially nothing (attacker took all)
        non_attacker_total_per_resource = np.zeros(self.n_resources)
        for i in self.non_attacking_group:
            non_attacker_total_per_resource += x_full[i, :]
        
        # Find the best swap among non-attackers
        non_attackers = list(self.non_attacking_group)
        best_swap = None
        best_improvement = 0.0
        
        def round_to_sig_figs(x, sig_figs=3):
            """Round to significant figures."""
            if x == 0:
                return 0
            return round(x, sig_figs - 1 - int(np.floor(np.log10(abs(x)))))
        
        for r in range(self.n_resources):
            # Skip resources where non-attackers have essentially no allocation
            # (attacker took all or nearly all of it)
            if non_attacker_total_per_resource[r] < 1e-6:
                continue
            
            for i in non_attackers:
                # Skip if agent i has essentially no allocation of this resource
                # (nothing meaningful to transfer)
                if x_full[i, r] < 1e-6:
                    continue
                
                V_i = max(V[i], 1e-10)
                ratio_i = self.utilities[i, r] / V_i
                
                for j in non_attackers:
                    if i == j:
                        continue
                    
                    V_j = max(V[j], 1e-10)
                    ratio_j = self.utilities[j, r] / V_j
                    
                    # Compare ratios rounded to 3 significant figures
                    # Only consider it a violation if ratio_j is meaningfully larger
                    ratio_i_rounded = round_to_sig_figs(ratio_i, 3)
                    ratio_j_rounded = round_to_sig_figs(ratio_j, 3)
                    
                    improvement = ratio_j - ratio_i
                    
                    # Weight the improvement by the giver's allocation
                    # This measures the actual utility benefit, not just the rate
                    weighted_improvement = improvement * x_full[i, r]
                    
                    # Require weighted improvement >= 1e-4 and rounded ratios to differ
                    if ratio_j_rounded > ratio_i_rounded and weighted_improvement >= 1e-4:
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (i, j, r)
        
        if best_swap is None:
            if self.verbose:
                print(f"    -> NOT VIOLATED (swap-optimal)")
            return False, None, None
        
        agent_from, agent_to, resource = best_swap
        
        # Always print swap violation details for debugging
        V_from, V_to = max(V[agent_from], 1e-10), max(V[agent_to], 1e-10)
        print(f"    -> SWAP VIOLATED: Agent {agent_from} -> Agent {agent_to} for resource {resource}")
        print(f"      Agent {agent_from}: x={x_full[agent_from, resource]:.10f}, V={V_from:.10f}, v_r={self.utilities[agent_from, resource]:.10f}, v/V={self.utilities[agent_from, resource]/V_from:.10f}")
        print(f"      Agent {agent_to}: x={x_full[agent_to, resource]:.10f}, V={V_to:.10f}, v_r={self.utilities[agent_to, resource]:.10f}, v/V={self.utilities[agent_to, resource]/V_to:.10f}")
        print(f"      Improvement: {best_improvement:.10f}")
        
        normal, rhs = self._find_separating_hyperplane(x_full, V, agent_from, agent_to, resource)
        
        if self.verbose:
            print(f"    Cutting plane: normal · x <= {rhs:.6f}")
        
        return True, normal, rhs
    
    def _find_separating_hyperplane(self, x_full: np.ndarray, V: np.ndarray,
                                     agent_from: int, agent_to: int, 
                                     resource: int) -> Tuple[np.ndarray, float]:
        """Find separating hyperplane that cuts to where ratios are equal."""
        v_from = self.utilities[agent_from, resource]
        v_to = self.utilities[agent_to, resource]
        V_from = V[agent_from]
        V_to = V[agent_to]
        
        # Compute δ that equalizes ratios
        numerator = v_to * V_from - v_from * V_to
        denominator = 2 * v_from * v_to
        
        if abs(denominator) < 1e-12:
            delta = 1e-6
        else:
            delta = numerator / denominator
        
        delta = max(0, min(delta, x_full[agent_from, resource]))
        
        if self.verbose:
            print(f"    [Swap Hyperplane] delta (transfer amount) = {delta:.6f}")
        
        # Compute and store the boundary point (after the swap)
        x_boundary = x_full.copy()
        x_boundary[agent_from, resource] -= delta
        x_boundary[agent_to, resource] += delta
        self.last_boundary_point = x_boundary.flatten().copy()
        
        # Normal: +1 at (agent_from, resource), -1 at (agent_to, resource)
        normal = np.zeros(self.n_free_vars)
        idx_from = agent_from * self.n_resources + resource
        idx_to = agent_to * self.n_resources + resource
        
        normal[idx_from] = 1.0
        normal[idx_to] = -1.0
        
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        
        x_from_new = x_full[agent_from, resource] - delta
        x_to_new = x_full[agent_to, resource] + delta
        rhs = normal[idx_from] * x_from_new + normal[idx_to] * x_to_new
        
        return normal, rhs


# Keep OptimalBoundaryOracle as an alias that combines both for backward compatibility
class OptimalBoundaryOracle:
    """
    Combined oracle that checks both directional derivative and swap-optimality.
    This is kept for backward compatibility but the two constraints are now separate.
    """
    
    def __init__(self, n_agents: int, n_resources: int, utilities: np.ndarray,
                 attacking_group: Set[int], supply: Optional[np.ndarray] = None,
                 verbose: bool = False):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.utilities = utilities
        self.attacking_group = attacking_group
        self.non_attacking_group = set(range(n_agents)) - attacking_group
        self.supply = supply if supply is not None else np.ones(n_resources)
        self.verbose = verbose
        self.n_free_vars = n_agents * n_resources
        
        # Create both oracles
        self.dir_deriv_oracle = DirectionalDerivativeOracle(
            n_agents, n_resources, utilities, attacking_group, supply, verbose)
        self.swap_oracle = SwapOptimalityOracle(
            n_agents, n_resources, utilities, attacking_group, supply, verbose)
    
    def _get_full_allocation(self, x_free: np.ndarray) -> np.ndarray:
        return x_free.reshape((self.n_agents, self.n_resources))
    
    def _compute_utilities_from_allocation(self, x_full: np.ndarray) -> np.ndarray:
        return np.array([
            np.dot(self.utilities[i], x_full[i]) 
            for i in range(self.n_agents)
        ])
    
    def _compute_directional_derivative(self, x_full: np.ndarray, V: np.ndarray) -> float:
        # Try forward direction first
        forward_direction, forward_valid = self.dir_deriv_oracle._compute_direction(x_full, V, backward=False)
        if forward_valid:
            return self.dir_deriv_oracle._compute_directional_derivative_with_direction(x_full, V, forward_direction)
        else:
            # If forward is invalid, compute backward and return its negative
            backward_direction, _ = self.dir_deriv_oracle._compute_direction(x_full, V, backward=True)
            backward_deriv = self.dir_deriv_oracle._compute_directional_derivative_with_direction(x_full, V, backward_direction)
            return -backward_deriv  # Negate since backward is opposite of forward
    
    def is_violated(self, x_free: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Check both oracles and return violation from either."""
        # Check swap-optimality first
        violated, normal, rhs = self.swap_oracle.is_violated(x_free)
        if violated:
            return violated, normal, rhs
        
        # Then check directional derivative
        return self.dir_deriv_oracle.is_violated(x_free)


def _project_to_feasible_region(
    x_target: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    n_agents: int,
    n_resources: int,
    supply: np.ndarray
) -> Tuple[np.ndarray, str]:
    """
    Project x_target onto the feasible region defined by cutting planes.
    
    Solves: min ||x - x_target||² 
            subject to: G @ x <= h, A_eq @ x = b_eq
    
    This is a QP, much faster than the full log-sum CP.
    
    Args:
        x_target: Point to project (the unconstrained optimum)
        G: Inequality constraint matrix
        h: Inequality constraint RHS
        A_eq: Equality constraint matrix
        b_eq: Equality constraint RHS
        n_agents: Number of agents
        n_resources: Number of resources
        supply: Resource supply vector
    
    Returns:
        (projected_point, status)
    """
    n_free_vars = len(x_target)
    
    # QP: min (1/2) x^T P x + q^T x
    # For ||x - x_target||², we have:
    # P = 2 * I
    # q = -2 * x_target
    
    P = matrix(2.0 * np.eye(n_free_vars))
    q = matrix(-2.0 * x_target)
    
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    A_cvx = matrix(A_eq)
    b_cvx = matrix(b_eq)
    
    old_show_progress = solvers.options.get('show_progress', False)
    solvers.options['show_progress'] = False
    
    try:
        sol = solvers.qp(P, q, G_cvx, h_cvx, A_cvx, b_cvx)
    except Exception as e:
        solvers.options['show_progress'] = old_show_progress
        return x_target, f"projection_error: {e}"
    finally:
        solvers.options['show_progress'] = old_show_progress
    
    if sol['status'] in ['optimal', 'unknown']:
        x_proj = np.array(sol['x']).flatten()
        x_proj = np.maximum(x_proj, 0)  # Ensure non-negative
        return x_proj, 'optimal'
    else:
        return x_target, sol['status']


def _solve_optimal_constraint_convex_program(
    utilities: np.ndarray,
    attacking_group: Set[int],
    target_group: Set[int],
    supply: Optional[np.ndarray] = None,
    initial_constraints: Optional[Dict[int, AgentDiversityConstraint]] = None,
    maximize_harm: bool = False,
    verbose: bool = False,
    debug: bool = False,
    use_projection: bool = False,
    use_two_phase: bool = True,
    track_timing: bool = False,
    use_integral_method: bool = False
) -> Tuple[Optional[Dict[int, np.ndarray]], np.ndarray, str]:
    """
    Solve the convex program to find optimal proportionality constraints.
    
    This finds the directions (ratio vectors) for the attacking group that either:
    - Maximize their own benefit (minimize p-MON) if maximize_harm=False
    - Maximize harm to target_group (minimize q-NEE) if maximize_harm=True
    
    Uses cutting planes with two separate oracles:
    1. SwapOptimalityOracle - ensures non-attackers are swap-optimal among themselves
    2. DirectionalDerivativeOracle - ensures we're on the optimal boundary
    
    Optimization: Uses projection of unconstrained optimum as a fast filter before
    solving the full inner CP. Only solves the expensive CP when projection is feasible.
    
    Args:
        utilities: Preference matrix (n_agents, n_resources)
        attacking_group: Set of attacker agent indices
        target_group: Set of target agent indices (attackers for p-MON, victims for q-NEE)
        supply: Resource supply vector
        initial_constraints: Pre-existing constraints
        maximize_harm: If True, minimize target utility; if False, maximize target utility
        verbose: Print progress
        debug: Print detailed debug information
        use_projection: If True, use projection-based optimization; if False, always use inner solve
        track_timing: If True, enable detailed timing in DirectionalDerivativeOracle
        use_integral_method: If True, use integral-based boundary finding instead of binary search
    
    Returns:
        Tuple of:
        - optimal_directions: Dict mapping attacker agent_id -> ratio vector, or None
        - final_allocation: The resulting allocation matrix
        - status: 'optimal', 'infeasible', or error message
        - cp_debug_info: Dict with debug info about the convex program (includes dirderiv_timing_stats if track_timing=True)
    """
    # Debug implies verbose
    if debug:
        verbose = True
    
    n_agents, n_resources = utilities.shape
    supply = supply if supply is not None else np.ones(n_resources)
    
    if verbose:
        print(f"  [ConvexProgram] n_agents={n_agents}, n_resources={n_resources}")
        print(f"  [ConvexProgram] attacking_group={attacking_group}")
        print(f"  [ConvexProgram] target_group={target_group}")
        print(f"  [ConvexProgram] maximize_harm={maximize_harm}")
    
    # Create both oracles separately
    swap_oracle = SwapOptimalityOracle(n_agents, n_resources, utilities, attacking_group, supply, verbose=debug)
    dir_deriv_oracle = DirectionalDerivativeOracle(n_agents, n_resources, utilities, attacking_group, supply, verbose=debug, track_timing=track_timing, use_integral_method=use_integral_method)
    
    # All agents are free variables now
    n_free_vars = n_agents * n_resources
    
    if verbose:
        print(f"  [ConvexProgram] n_free_vars={n_free_vars}")
    
    # Build base constraints for the convex program
    # x_ij >= 0 for all agents
    # sum_i x_ij = supply_j for each resource j (equality constraint)
    
    G_list = []
    h_list = []
    
    # x >= 0 for all agents
    G_list.append(-np.eye(n_free_vars))
    h_list.append(np.zeros(n_free_vars))
    
    # x <= supply (per resource, per agent) - upper bound
    G_list.append(np.eye(n_free_vars))
    h_list.append(np.tile(supply, n_agents))
    
    G_base = np.vstack(G_list)
    h_base = np.concatenate(h_list)
    
    # Equality constraints: sum_i x_ij = supply_j for each resource j
    A_eq_list = []
    b_eq_list = []
    for j in range(n_resources):
        row = np.zeros(n_free_vars)
        for i in range(n_agents):
            row[i * n_resources + j] = 1.0
        A_eq_list.append(row)
        b_eq_list.append(supply[j])
    
    A_eq = np.array(A_eq_list)
    b_eq = np.array(b_eq_list)
    
    # Cutting planes from both oracles
    cuts_G = []
    cuts_h = []
    
    # Track cut details for debugging
    cut_details = []  # List of dicts with cut info
    
    # Initial point: equal allocation
    x = np.zeros(n_free_vars)
    for i in range(n_agents):
        start = i * n_resources
        x[start:start + n_resources] = supply / n_agents
    
    # Iterative optimization with cutting planes
    max_iterations = 150
    max_projections_per_inner_solve = 7  # Max projections between inner CP solves
    n_swap_cuts = 0
    n_dir_deriv_cuts = 0
    n_projection_iterations = 0
    n_inner_solves = 0
    n_cuts_from_projection = 0  # Cuts added after projection
    n_cuts_from_inner_solve = 0  # Cuts added after inner solve
    final_iteration = 0
    
    # First, solve the unconstrained problem to get x* (global optimum)
    if verbose:
        print(f"  [ConvexProgram] Solving unconstrained problem to get global optimum...")
    
    x_star, status = _solve_inner_optimization(
        utilities, n_agents, n_resources, attacking_group, target_group,
        G_base, h_base, A_eq, b_eq, supply, maximize_harm, verbose,
        use_two_phase=use_two_phase
    )
    n_inner_solves += 1
    
    if status != 'optimal':
        if verbose:
            print(f"  [ConvexProgram] Unconstrained optimization failed: {status}")
        x_full = x_star.reshape((n_agents, n_resources))
        return None, x_full, status, {
            'iterations': 1,
            'swap_cuts': 0,
            'dir_deriv_cuts': 0,
            'total_cuts': 0,
            'backward_checks': dir_deriv_oracle.n_backward_checks,
            'backward_violations': dir_deriv_oracle.n_backward_violations,
            'converged': False,
            'failure_reason': status,
            'cut_details': cut_details,
            'projection_iterations': 0,
            'inner_solves': n_inner_solves
        }
    
    if verbose:
        print(f"  [ConvexProgram] Got global optimum x*")
    
    # Check if x* is already feasible (satisfies both oracles)
    x = x_star.copy()
    projections_since_last_inner_solve = 0
    
    for iteration in range(max_iterations):
        final_iteration = iteration
        
        # Combine constraints
        if cuts_G:
            G = np.vstack([G_base] + cuts_G)
            h = np.concatenate([h_base] + cuts_h)
        else:
            G = G_base
            h = h_base
        
        # Decide whether to use projection or full inner solve
        # Use projection if enabled, we have cuts, and haven't exhausted projection budget since last inner solve
        do_projection = use_projection and (projections_since_last_inner_solve < max_projections_per_inner_solve) and (len(cuts_G) > 0)
        
        if do_projection:
            # Project x* onto current feasible region
            x_proj, proj_status = _project_to_feasible_region(
                x_star, G, h, A_eq, b_eq, n_agents, n_resources, supply
            )
            n_projection_iterations += 1
            projections_since_last_inner_solve += 1
            
            if proj_status != 'optimal':
                if verbose:
                    print(f"  [ConvexProgram] Iteration {iteration}: projection failed ({proj_status}), falling back to inner solve")
                do_projection = False
            else:
                x = x_proj
                if verbose:
                    print(f"  [ConvexProgram] Iteration {iteration}: using projection ({projections_since_last_inner_solve}/{max_projections_per_inner_solve})")
        
        if not do_projection:
            # Solve full inner optimization
            x, status = _solve_inner_optimization(
                utilities, n_agents, n_resources, attacking_group, target_group,
                G, h, A_eq, b_eq, supply, maximize_harm, verbose,
                use_two_phase=use_two_phase
            )
            n_inner_solves += 1
            projections_since_last_inner_solve = 0  # Reset projection counter
            
            if verbose:
                print(f"  [ConvexProgram] Iteration {iteration}: inner solve status={status}")
            
            if status != 'optimal':
                if verbose:
                    print(f"  [ConvexProgram] Inner optimization failed: {status}")
                x_full = x.reshape((n_agents, n_resources))
                return None, x_full, status, {
                    'iterations': iteration + 1,
                    'swap_cuts': n_swap_cuts,
                    'dir_deriv_cuts': n_dir_deriv_cuts,
                    'total_cuts': n_swap_cuts + n_dir_deriv_cuts,
                    'backward_checks': dir_deriv_oracle.n_backward_checks,
                    'backward_violations': dir_deriv_oracle.n_backward_violations,
                    'converged': False,
                    'failure_reason': status,
                    'cut_details': cut_details,
                    'projection_iterations': n_projection_iterations,
                    'inner_solves': n_inner_solves
                }
        
        # Check both oracles independently
        cuts_added_this_iteration = 0
        cut_source = 'projection' if do_projection else 'inner_solve'
        
        # Check swap-optimality oracle
        swap_violated, swap_normal, swap_rhs = swap_oracle.is_violated(x)
        if swap_violated:
            cuts_G.append(swap_normal.reshape(1, -1))
            cuts_h.append(np.array([swap_rhs]))
            n_swap_cuts += 1
            cuts_added_this_iteration += 1
            if do_projection:
                n_cuts_from_projection += 1
            else:
                n_cuts_from_inner_solve += 1
            
            # Store cut details
            cut_details.append({
                'cut_number': len(cut_details) + 1,
                'cut_type': 'swap',
                'iteration': iteration,
                'source': cut_source,
                'violating_point': x.copy(),  # Point that was outside the boundary
                'boundary_point': swap_oracle.last_boundary_point.copy() if swap_oracle.last_boundary_point is not None else x.copy(),
                'normal': swap_normal.copy(),
                'rhs': swap_rhs
            })
            
            if verbose:
                print(f"  [ConvexProgram] Iteration {iteration}: swap oracle violated, added cut {n_swap_cuts} (from {cut_source})")
        
        # Check directional derivative oracle
        dir_violated, dir_normal, dir_rhs = dir_deriv_oracle.is_violated(x)
        if dir_violated:
            cuts_G.append(dir_normal.reshape(1, -1))
            cuts_h.append(np.array([dir_rhs]))
            n_dir_deriv_cuts += 1
            cuts_added_this_iteration += 1
            if do_projection:
                n_cuts_from_projection += 1
            else:
                n_cuts_from_inner_solve += 1
            
            # Store cut details
            cut_details.append({
                'cut_number': len(cut_details) + 1,
                'cut_type': 'dir_deriv',
                'iteration': iteration,
                'source': cut_source,
                'violating_point': x.copy(),  # Point that was outside the boundary
                'boundary_point': dir_deriv_oracle.last_boundary_point.copy() if dir_deriv_oracle.last_boundary_point is not None else x.copy(),
                'normal': dir_normal.copy(),
                'rhs': dir_rhs
            })
            
            if verbose:
                print(f"  [ConvexProgram] Iteration {iteration}: dir_deriv oracle violated, added cut {n_dir_deriv_cuts} (from {cut_source})")
        
        if verbose:
            print(f"  [ConvexProgram] Iteration {iteration}: swap_violated={swap_violated}, dir_violated={dir_violated}")
        
        # If projection was used and oracles are satisfied, do one inner solve to get exact optimum
        if do_projection and not swap_violated and not dir_violated:
            if verbose:
                print(f"  [ConvexProgram] Projection feasible, solving inner CP for exact optimum...")
            x, status = _solve_inner_optimization(
                utilities, n_agents, n_resources, attacking_group, target_group,
                G, h, A_eq, b_eq, supply, maximize_harm, verbose,
                use_two_phase=use_two_phase
            )
            n_inner_solves += 1
            projections_since_last_inner_solve = 0  # Reset projection counter
            
            if status != 'optimal':
                if verbose:
                    print(f"  [ConvexProgram] Final inner optimization failed: {status}")
                # Use the projected point as fallback
                x = x_proj
            else:
                # Re-check oracles on the inner solve result
                swap_violated, swap_normal, swap_rhs = swap_oracle.is_violated(x)
                dir_violated, dir_normal, dir_rhs = dir_deriv_oracle.is_violated(x)
                
                if swap_violated:
                    cuts_G.append(swap_normal.reshape(1, -1))
                    cuts_h.append(np.array([swap_rhs]))
                    n_swap_cuts += 1
                    n_cuts_from_inner_solve += 1
                    cut_details.append({
                        'cut_number': len(cut_details) + 1,
                        'cut_type': 'swap',
                        'iteration': iteration,
                        'source': 'inner_solve_after_projection',
                        'violating_point': x.copy(),
                        'boundary_point': swap_oracle.last_boundary_point.copy() if swap_oracle.last_boundary_point is not None else x.copy(),
                        'normal': swap_normal.copy(),
                        'rhs': swap_rhs
                    })
                    if verbose:
                        print(f"  [ConvexProgram] Inner solve violated swap oracle, added cut {n_swap_cuts}")
                
                if dir_violated:
                    cuts_G.append(dir_normal.reshape(1, -1))
                    cuts_h.append(np.array([dir_rhs]))
                    n_dir_deriv_cuts += 1
                    n_cuts_from_inner_solve += 1
                    cut_details.append({
                        'cut_number': len(cut_details) + 1,
                        'cut_type': 'dir_deriv',
                        'iteration': iteration,
                        'source': 'inner_solve_after_projection',
                        'violating_point': x.copy(),
                        'boundary_point': dir_deriv_oracle.last_boundary_point.copy() if dir_deriv_oracle.last_boundary_point is not None else x.copy(),
                        'normal': dir_normal.copy(),
                        'rhs': dir_rhs
                    })
                    if verbose:
                        print(f"  [ConvexProgram] Inner solve violated dir_deriv oracle, added cut {n_dir_deriv_cuts}")
                
                if swap_violated or dir_violated:
                    if verbose:
                        print(f"  [ConvexProgram] Inner solve result violated oracles, continuing...")
                    continue
        
        # Converge only when BOTH oracles are satisfied (and we used inner solve, not just projection)
        if not swap_violated and not dir_violated and not do_projection:
            if verbose:
                print(f"  [ConvexProgram] Converged after {iteration + 1} iterations")
                print(f"  [ConvexProgram] Stats: {n_projection_iterations} projection iters, {n_inner_solves} inner solves")
            converged = True
            break
    else:
        # Loop completed without breaking - did not converge
        converged = False
        if verbose:
            print(f"  [ConvexProgram] Did not converge after {max_iterations} iterations")
            print(f"  [ConvexProgram] Stats: {n_projection_iterations} projection iters, {n_inner_solves} inner solves")
    
    # Extract directions for attacking group
    x_full = x.reshape((n_agents, n_resources))
    
    if debug:
        print(f"  [DEBUG] Final allocation from convex program:")
        print(f"  [DEBUG] x_full:\n{x_full.round(6)}")
        # Compute utilities from this allocation
        final_V = np.array([np.dot(utilities[i], x_full[i]) for i in range(n_agents)])
        print(f"  [DEBUG] Utilities from final allocation: {final_V.round(6)}")
    
    optimal_directions = {}
    for agent_id in attacking_group:
        direction = x_full[agent_id].copy()
        # Normalize to get ratio vector
        if np.linalg.norm(direction) > 1e-10:
            optimal_directions[agent_id] = direction
            if debug:
                print(f"  [DEBUG] Agent {agent_id} direction: {direction.round(6)} (norm={np.linalg.norm(direction):.6f})")
        else:
            optimal_directions[agent_id] = None
            if debug:
                print(f"  [DEBUG] Agent {agent_id} direction: None (zero norm)")
    
    cp_debug_info = {
        'iterations': final_iteration + 1,
        'swap_cuts': n_swap_cuts,
        'dir_deriv_cuts': n_dir_deriv_cuts,
        'total_cuts': n_swap_cuts + n_dir_deriv_cuts,
        'cuts_from_projection': n_cuts_from_projection,
        'cuts_from_inner_solve': n_cuts_from_inner_solve,
        'backward_checks': dir_deriv_oracle.n_backward_checks,
        'backward_violations': dir_deriv_oracle.n_backward_violations,
        'converged': converged,
        'failure_reason': None if converged else 'max_iterations_reached',
        'cut_details': cut_details,
        'projection_iterations': n_projection_iterations,
        'inner_solves': n_inner_solves,
        'dirderiv_timing_stats': dir_deriv_oracle.get_timing_stats() if track_timing else None
    }
    
    return optimal_directions, x_full, 'optimal', cp_debug_info


def _solve_inner_optimization_cvxopt(
    utilities: np.ndarray,
    n_agents: int,
    n_resources: int,
    attacking_group: Set[int],
    target_group: Set[int],
    G: np.ndarray,
    h: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    supply: np.ndarray,
    maximize_harm: bool,
    verbose: bool
) -> Tuple[np.ndarray, str]:
    """
    Solve the inner optimization problem using CVXOPT.
    
    The objective is: sum_{i in target} log(V_i)
    
    This maximizes/minimizes the Nash welfare of the target group.
    
    All agents are free variables with equality constraints sum_i x_ij = supply_j.
    
    For maximize_harm=False: Maximize the objective using CVXOPT's cp solver.
    For maximize_harm=True: Minimize using projected gradient descent (non-convex).
    
    Returns:
        (x_optimal, status): Optimal allocation (flattened) and status
    """
    n_free_vars = n_agents * n_resources
    
    if verbose:
        print(f"  [InnerOpt-CVXOPT] target_group={target_group}, maximize_harm={maximize_harm}")
        print(f"  [InnerOpt-CVXOPT] n_free_vars={n_free_vars}")
    
    def get_full_allocation(x_np):
        """Convert free variables to full allocation matrix."""
        return x_np.reshape((n_agents, n_resources))
    
    def compute_utilities_inner(x_np):
        """Compute utilities for all agents."""
        x_full = get_full_allocation(x_np)
        return np.array([np.dot(utilities[i], x_full[i]) for i in range(n_agents)])
    
    def compute_agent_gradient(agent_id, V_i):
        """Compute gradient of log(V_i) w.r.t. all free variables."""
        grad = np.zeros(n_free_vars)
        start = agent_id * n_resources
        grad[start:start + n_resources] = utilities[agent_id] / V_i
        return grad
    
    def project_to_feasible(x_np):
        """Project x to satisfy constraints."""
        # Clip to non-negative
        x_np = np.maximum(x_np, 0)
        
        # Clip each agent's allocation to supply
        x_full = x_np.reshape((n_agents, n_resources))
        x_full = np.minimum(x_full, supply)
        
        # Enforce supply constraint: sum_i x_ij = supply_j
        for j in range(n_resources):
            total = x_full[:, j].sum()
            if total > 1e-10:
                x_full[:, j] = x_full[:, j] * supply[j] / total
            else:
                # If all zero, distribute equally
                x_full[:, j] = supply[j] / n_agents
        
        return x_full.flatten()
    
    # Initial point: equal allocation
    x0 = np.zeros(n_free_vars)
    for i in range(n_agents):
        start = i * n_resources
        x0[start:start + n_resources] = supply / n_agents
    
    if not maximize_harm:
        # Use CVXOPT's cp solver for convex case (maximize = minimize negative)
        # With equality constraints A_eq @ x = b_eq
        
        def F(x=None, z=None):
            if x is None:
                return (0, matrix(x0))
            
            x_np = np.array(x).flatten()
            V = compute_utilities_inner(x_np)
            
            # Check for non-positive utilities for target group
            for i in target_group:
                if V[i] <= 1e-10:
                    return (None, None, None) if z is not None else (None, None)
            
            # Objective: sum log(V_i) for target group
            obj_sum = sum(np.log(V[i]) for i in target_group)
            
            # We minimize negative
            f_val = -obj_sum
            f = matrix(f_val)
            
            # Gradient: -sum_{i in target} (1/V_i) * dV_i/dx
            grad = np.zeros(n_free_vars)
            for i in target_group:
                grad -= compute_agent_gradient(i, V[i])
            
            Df = matrix(grad, (1, n_free_vars))
            
            if z is None:
                return (f, Df)
            
            # Hessian computation: sum (1/V_i^2) * grad_Vi * grad_Vi^T
            H = matrix(0.0, (n_free_vars, n_free_vars))
            
            for i in target_group:
                grad_Vi = np.zeros(n_free_vars)
                start = i * n_resources
                grad_Vi[start:start + n_resources] = utilities[i]
                
                for j1 in range(n_free_vars):
                    for j2 in range(n_free_vars):
                        H[j1, j2] += z[0] * (1.0 / (V[i] ** 2)) * grad_Vi[j1] * grad_Vi[j2]
            
            # Regularization
            for j in range(n_free_vars):
                H[j, j] += z[0] * 1e-8
            
            return (f, Df, H)
        
        old_show_progress = solvers.options.get('show_progress', False)
        # Only show CVXOPT internal progress in debug mode, not just verbose
        solvers.options['show_progress'] = False
        
        try:
            # Use cp solver with equality constraints
            sol = solvers.cp(F, G=matrix(G), h=matrix(h), 
                           A=matrix(A_eq), b=matrix(b_eq))
        except Exception as e:
            solvers.options['show_progress'] = old_show_progress
            if verbose:
                print(f"  [InnerOpt-CVXOPT] Exception: {e}")
            return x0, f"solver_error: {e}"
        finally:
            solvers.options['show_progress'] = old_show_progress
        
        if verbose:
            print(f"  [InnerOpt-CVXOPT] Solver status: {sol['status']}")
        
        if sol['status'] in ['optimal', 'unknown']:
            x = np.array(sol['x']).flatten()
            x = np.maximum(x, 0)
            return x, 'optimal'
        else:
            return x0, sol['status']
    
    else:
        # Projected gradient descent for non-convex case (minimize target utility)
        # Objective: minimize sum log(V_target)
        
        def compute_objective(x_np):
            """Compute objective."""
            V = compute_utilities_inner(x_np)
            return sum(np.log(max(V[i], 1e-10)) for i in target_group)
        
        def compute_gradient(x_np):
            """Compute gradient of objective."""
            V = compute_utilities_inner(x_np)
            grad = np.zeros(n_free_vars)
            
            for i in target_group:
                V_i = max(V[i], 1e-10)
                grad += compute_agent_gradient(i, V_i)
            
            return grad
        
        x = x0.copy()
        x = project_to_feasible(x)
        
        best_x = x.copy()
        best_obj = compute_objective(x)
        
        # Gradient descent parameters
        max_iters = 500
        initial_step = 0.1
        min_step = 1e-8
        decay = 0.999
        
        step_size = initial_step
        
        for iteration in range(max_iters):
            # Compute gradient
            grad = compute_gradient(x)
            
            # We want to MINIMIZE, so move in NEGATIVE gradient direction
            direction = -grad
            
            # Normalize direction
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                if verbose:
                    print(f"  [InnerOpt-CVXOPT] Converged at iteration {iteration} (zero gradient)")
                break
            direction = direction / dir_norm
            
            # Line search with backtracking
            current_obj = compute_objective(x)
            
            step = step_size
            for _ in range(20):
                x_new = x + step * direction
                x_new = project_to_feasible(x_new)
                
                # Check if target utilities are positive
                V_new = compute_utilities_inner(x_new)
                if all(V_new[i] > 1e-10 for i in target_group):
                    new_obj = compute_objective(x_new)
                    if new_obj < current_obj:
                        break
                step *= 0.5
                if step < min_step:
                    break
            
            x = x_new
            obj = compute_objective(x)
            
            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
            
            # Decay step size
            step_size *= decay
            
            # Check convergence
            if step < min_step:
                if verbose:
                    print(f"  [InnerOpt-CVXOPT] Converged at iteration {iteration} (small step)")
                break
        
        if verbose:
            print(f"  [InnerOpt-CVXOPT] PGD finished after {iteration + 1} iterations")
            print(f"  [InnerOpt-CVXOPT] Final objective: {best_obj:.6f}")
        
        return best_x, 'optimal'


# Global flag to track if Gurobi is available
_GUROBI_AVAILABLE = None

def _check_gurobi_available():
    """Check if Gurobi is available."""
    global _GUROBI_AVAILABLE
    if _GUROBI_AVAILABLE is None:
        try:
            import gurobipy as gp
            # Try to create a model to verify license
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.start()
            model = gp.Model(env=env)
            model.dispose()
            env.dispose()
            _GUROBI_AVAILABLE = True
        except Exception:
            _GUROBI_AVAILABLE = False
    return _GUROBI_AVAILABLE


def _solve_inner_optimization_gurobi(
    utilities: np.ndarray,
    n_agents: int,
    n_resources: int,
    attacking_group: Set[int],
    target_group: Set[int],
    G: np.ndarray,
    h: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    supply: np.ndarray,
    maximize_harm: bool,
    verbose: bool
) -> Tuple[np.ndarray, str]:
    """
    Solve the inner optimization problem using Gurobi.
    
    The objective is: sum_{i in target} log(V_i)
    
    This maximizes/minimizes the Nash welfare of the target group.
    
    Gurobi supports log() natively in nonlinear objectives.
    
    Returns:
        (x_optimal, status): Optimal allocation (flattened) and status
    """
    import gurobipy as gp
    from gurobipy import GRB
    
    n_free_vars = n_agents * n_resources
    
    if verbose:
        print(f"  [InnerOpt-Gurobi] target_group={target_group}, maximize_harm={maximize_harm}")
        print(f"  [InnerOpt-Gurobi] n_free_vars={n_free_vars}")
    
    # Initial point for fallback
    x0 = np.zeros(n_free_vars)
    for i in range(n_agents):
        start = i * n_resources
        x0[start:start + n_resources] = supply / n_agents
    
    try:
        # Create environment with output suppressed
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 2  # Allow non-convex quadratic objectives
        
        # Create variables: x[i, j] = allocation of resource j to agent i
        x = {}
        for i in range(n_agents):
            for j in range(n_resources):
                x[i, j] = model.addVar(lb=0, ub=supply[j], name=f"x_{i}_{j}")
        
        # Equality constraints: sum_i x[i,j] = supply[j]
        for j in range(n_resources):
            model.addConstr(
                gp.quicksum(x[i, j] for i in range(n_agents)) == supply[j],
                name=f"supply_{j}"
            )
        
        # Add cutting plane constraints from G @ x <= h
        n_cuts = G.shape[0] - 2 * n_free_vars  # Exclude base non-negativity and upper bound constraints
        for cut_idx in range(G.shape[0]):
            # Convert row to constraint
            lhs = gp.quicksum(
                G[cut_idx, i * n_resources + j] * x[i, j]
                for i in range(n_agents)
                for j in range(n_resources)
            )
            model.addConstr(lhs <= h[cut_idx], name=f"cut_{cut_idx}")
        
        # Create utility variables V[i] = sum_j utilities[i,j] * x[i,j]
        V = {}
        for i in range(n_agents):
            V[i] = model.addVar(lb=1e-8, name=f"V_{i}")  # Small positive lower bound
            model.addConstr(
                V[i] == gp.quicksum(utilities[i, j] * x[i, j] for j in range(n_resources)),
                name=f"utility_{i}"
            )
        
        # Objective: sum_{i in target} log(V[i])
        # Gurobi doesn't support log() directly in the objective, so we use a piecewise linear approximation
        # or auxiliary variables with general constraints
        
        # Use auxiliary variables for log
        log_V = {}
        for i in target_group:
            log_V[i] = model.addVar(lb=-GRB.INFINITY, name=f"log_V_{i}")
            # Add general constraint: log_V[i] = log(V[i])
            model.addGenConstrLog(V[i], log_V[i], name=f"log_constr_{i}")
        
        # Set objective
        if maximize_harm:
            # Minimize sum of log utilities
            model.setObjective(
                gp.quicksum(log_V[i] for i in target_group),
                GRB.MINIMIZE
            )
        else:
            # Maximize sum of log utilities
            model.setObjective(
                gp.quicksum(log_V[i] for i in target_group),
                GRB.MAXIMIZE
            )
        
        # Solve
        model.optimize()
        
        if verbose:
            print(f"  [InnerOpt-Gurobi] Solver status: {model.Status}")
        
        if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            # Extract solution
            x_sol = np.zeros(n_free_vars)
            for i in range(n_agents):
                for j in range(n_resources):
                    x_sol[i * n_resources + j] = x[i, j].X
            
            x_sol = np.maximum(x_sol, 0)
            
            model.dispose()
            env.dispose()
            
            return x_sol, 'optimal'
        else:
            model.dispose()
            env.dispose()
            
            status_map = {
                GRB.INFEASIBLE: 'infeasible',
                GRB.INF_OR_UNBD: 'infeasible_or_unbounded',
                GRB.UNBOUNDED: 'unbounded',
                GRB.NUMERIC: 'numeric_error',
                GRB.TIME_LIMIT: 'time_limit',
            }
            status_str = status_map.get(model.Status, f'gurobi_status_{model.Status}')
            return x0, status_str
            
    except Exception as e:
        if verbose:
            print(f"  [InnerOpt-Gurobi] Exception: {e}")
        return x0, f"solver_error: {e}"


# Default solver selection
_DEFAULT_SOLVER = 'gurobi'  # Will fall back to cvxopt if gurobi not available

def set_default_solver(solver: str):
    """Set the default solver for inner optimization.
    
    Args:
        solver: 'gurobi' or 'cvxopt'
    """
    global _DEFAULT_SOLVER
    if solver not in ['gurobi', 'cvxopt']:
        raise ValueError(f"Unknown solver: {solver}. Must be 'gurobi' or 'cvxopt'.")
    _DEFAULT_SOLVER = solver


def get_default_solver() -> str:
    """Get the current default solver."""
    return _DEFAULT_SOLVER


def _solve_inner_optimization_two_phase(
    utilities: np.ndarray,
    n_agents: int,
    n_resources: int,
    attacking_group: Set[int],
    target_group: Set[int],
    G_full: np.ndarray,
    h_full: np.ndarray,
    A_eq_full: np.ndarray,
    b_eq_full: np.ndarray,
    supply: np.ndarray,
    maximize_harm: bool,
    verbose: bool,
    solver: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    """
    Solve the inner optimization using a two-phase approach:
    
    Phase 1: Optimize attackers only
        max sum_{a in attackers} log(V_a)
        s.t. sum_{a in attackers} x_aj <= supply_j
             x_aj >= 0
             cutting plane constraints (projected to attacker space)
    
    Phase 2: Optimize non-attackers only
        max sum_{i in non-attackers} log(V_i)
        s.t. sum_{i in non-attackers} x_ij = supply_j - (attacker allocation for j)
             x_ij >= 0
    
    This removes the need for swap-optimality constraints since non-attackers
    are automatically Nash-optimal among themselves in Phase 2.
    
    Returns:
        (x_optimal, status): Full allocation (flattened, n_agents * n_resources) and status
    """
    if solver is None:
        solver = _DEFAULT_SOLVER
    
    attackers = sorted(list(attacking_group))
    non_attackers = sorted(list(set(range(n_agents)) - attacking_group))
    n_attackers = len(attackers)
    n_non_attackers = len(non_attackers)
    
    if verbose:
        print(f"  [TwoPhase] Phase 1: Optimizing {n_attackers} attackers")
        print(f"  [TwoPhase] Phase 2: Optimizing {n_non_attackers} non-attackers")
    
    # Map from attacker index to local index
    attacker_to_local = {a: i for i, a in enumerate(attackers)}
    non_attacker_to_local = {a: i for i, a in enumerate(non_attackers)}
    
    n_attacker_vars = n_attackers * n_resources
    n_full_vars = n_agents * n_resources
    
    # ========== PHASE 1: Attackers only ==========
    
    # Project cutting plane constraints to attacker space
    # The full G matrix has rows of the form: [g_0, g_1, ..., g_{n_agents-1}] for each resource
    # We need to extract only the attacker components
    
    # First, identify which rows in G_full are cutting planes (not base constraints)
    # Base constraints are: x >= 0 (n_full_vars rows) and x <= supply (n_full_vars rows)
    n_base_constraints = 2 * n_full_vars
    
    # Build attacker-only constraints
    G_attacker_list = []
    h_attacker_list = []
    
    # x >= 0 for attackers
    G_attacker_list.append(-np.eye(n_attacker_vars))
    h_attacker_list.append(np.zeros(n_attacker_vars))
    
    # x <= supply for each attacker variable
    G_attacker_list.append(np.eye(n_attacker_vars))
    h_attacker_list.append(np.tile(supply, n_attackers))
    
    # sum_a x_aj <= supply_j (inequality, not equality!)
    G_supply = np.zeros((n_resources, n_attacker_vars))
    for j in range(n_resources):
        for local_i, a in enumerate(attackers):
            G_supply[j, local_i * n_resources + j] = 1.0
    G_attacker_list.append(G_supply)
    h_attacker_list.append(supply.copy())
    
    # Project cutting plane constraints from full space to attacker space
    # Cutting planes are the rows after the base constraints
    if G_full.shape[0] > n_base_constraints:
        for row_idx in range(n_base_constraints, G_full.shape[0]):
            # Extract attacker components from this row
            g_projected = np.zeros(n_attacker_vars)
            for local_i, a in enumerate(attackers):
                for j in range(n_resources):
                    full_idx = a * n_resources + j
                    local_idx = local_i * n_resources + j
                    g_projected[local_idx] = G_full[row_idx, full_idx]
            
            # Only add if the projected constraint is non-trivial
            if np.linalg.norm(g_projected) > 1e-10:
                G_attacker_list.append(g_projected.reshape(1, -1))
                h_attacker_list.append(np.array([h_full[row_idx]]))
    
    G_attacker = np.vstack(G_attacker_list)
    h_attacker = np.concatenate(h_attacker_list)
    
    # No equality constraints for attackers (supply is inequality)
    # But we need a dummy equality constraint for the solver interface
    # Actually, let's just not pass equality constraints
    
    # Initial point for attackers: equal share
    x0_attacker = np.zeros(n_attacker_vars)
    for local_i in range(n_attackers):
        x0_attacker[local_i * n_resources:(local_i + 1) * n_resources] = supply / n_agents
    
    # Solve Phase 1 using CVXOPT
    try:
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
        
        # Utilities for attackers only
        utilities_attacker = utilities[attackers, :]
        
        def F_attacker(x=None, z=None):
            if x is None:
                return (0, matrix(x0_attacker))
            
            x_np = np.array(x).flatten()
            
            # Compute utilities for attackers
            V = []
            for local_i in range(n_attackers):
                v_i = 0.0
                for j in range(n_resources):
                    v_i += utilities_attacker[local_i, j] * x_np[local_i * n_resources + j]
                V.append(max(v_i, 1e-12))
            
            # Objective: -sum log(V_i) (minimize negative = maximize)
            f = sum(-np.log(v) for v in V)
            
            # Gradient
            Df = matrix(0.0, (1, n_attacker_vars))
            for local_i in range(n_attackers):
                for j in range(n_resources):
                    idx = local_i * n_resources + j
                    Df[idx] = -utilities_attacker[local_i, j] / V[local_i]
            
            if z is None:
                return (f, Df)
            
            # Hessian
            H = matrix(0.0, (n_attacker_vars, n_attacker_vars))
            for local_i in range(n_attackers):
                for j1 in range(n_resources):
                    for j2 in range(n_resources):
                        idx1 = local_i * n_resources + j1
                        idx2 = local_i * n_resources + j2
                        H[idx1, idx2] = z[0] * utilities_attacker[local_i, j1] * utilities_attacker[local_i, j2] / (V[local_i] ** 2)
            
            return (f, Df, H)
        
        sol_attacker = solvers.cp(F_attacker, G=matrix(G_attacker), h=matrix(h_attacker))
        
        if sol_attacker['status'] not in ['optimal', 'unknown']:
            if verbose:
                print(f"  [TwoPhase] Phase 1 failed: {sol_attacker['status']}")
            # Return equal allocation as fallback
            x_full = np.zeros(n_full_vars)
            for i in range(n_agents):
                x_full[i * n_resources:(i + 1) * n_resources] = supply / n_agents
            return x_full, sol_attacker['status']
        
        x_attacker = np.maximum(np.array(sol_attacker['x']).flatten(), 0)
        
        if verbose:
            print(f"  [TwoPhase] Phase 1 complete, status={sol_attacker['status']}")
        
    except Exception as e:
        if verbose:
            print(f"  [TwoPhase] Phase 1 exception: {e}")
        x_full = np.zeros(n_full_vars)
        for i in range(n_agents):
            x_full[i * n_resources:(i + 1) * n_resources] = supply / n_agents
        return x_full, f"phase1_error: {e}"
    
    # ========== PHASE 2: Non-attackers only ==========
    
    # Compute remaining supply after attackers
    attacker_allocation = np.zeros(n_resources)
    for local_i, a in enumerate(attackers):
        for j in range(n_resources):
            attacker_allocation[j] += x_attacker[local_i * n_resources + j]
    
    remaining_supply = supply - attacker_allocation
    remaining_supply = np.maximum(remaining_supply, 0)  # Ensure non-negative
    
    if verbose:
        print(f"  [TwoPhase] Attacker allocation per resource: {attacker_allocation.round(6)}")
        print(f"  [TwoPhase] Remaining supply for non-attackers: {remaining_supply.round(6)}")
    
    if n_non_attackers == 0:
        # No non-attackers, just return attacker allocation
        x_full = np.zeros(n_full_vars)
        for local_i, a in enumerate(attackers):
            for j in range(n_resources):
                x_full[a * n_resources + j] = x_attacker[local_i * n_resources + j]
        return x_full, 'optimal'
    
    # Check if there's any supply left for non-attackers
    if np.all(remaining_supply < 1e-10):
        # No supply left, give non-attackers zeros
        x_full = np.zeros(n_full_vars)
        for local_i, a in enumerate(attackers):
            for j in range(n_resources):
                x_full[a * n_resources + j] = x_attacker[local_i * n_resources + j]
        # Non-attackers get zeros (already initialized)
        return x_full, 'optimal'
    
    n_non_attacker_vars = n_non_attackers * n_resources
    
    # Initial point for non-attackers
    x0_non_attacker = np.zeros(n_non_attacker_vars)
    for local_i in range(n_non_attackers):
        for j in range(n_resources):
            if remaining_supply[j] > 1e-10:
                x0_non_attacker[local_i * n_resources + j] = remaining_supply[j] / n_non_attackers
    
    try:
        utilities_non_attacker = utilities[non_attackers, :]
        
        def F_non_attacker(x=None, z=None):
            if x is None:
                return (0, matrix(x0_non_attacker))
            
            x_np = np.array(x).flatten()
            
            # Compute utilities for non-attackers
            V = []
            for local_i in range(n_non_attackers):
                v_i = 0.0
                for j in range(n_resources):
                    v_i += utilities_non_attacker[local_i, j] * x_np[local_i * n_resources + j]
                V.append(max(v_i, 1e-12))
            
            # Objective: -sum log(V_i)
            f = sum(-np.log(v) for v in V)
            
            # Gradient
            Df = matrix(0.0, (1, n_non_attacker_vars))
            for local_i in range(n_non_attackers):
                for j in range(n_resources):
                    idx = local_i * n_resources + j
                    Df[idx] = -utilities_non_attacker[local_i, j] / V[local_i]
            
            if z is None:
                return (f, Df)
            
            # Hessian
            H = matrix(0.0, (n_non_attacker_vars, n_non_attacker_vars))
            for local_i in range(n_non_attackers):
                for j1 in range(n_resources):
                    for j2 in range(n_resources):
                        idx1 = local_i * n_resources + j1
                        idx2 = local_i * n_resources + j2
                        H[idx1, idx2] = z[0] * utilities_non_attacker[local_i, j1] * utilities_non_attacker[local_i, j2] / (V[local_i] ** 2)
            
            return (f, Df, H)
        
        # Inequality constraints: x >= 0
        G_non_attacker = -np.eye(n_non_attacker_vars)
        h_non_attacker = np.zeros(n_non_attacker_vars)
        
        # Equality constraints: sum_i x_ij = remaining_supply_j
        A_non_attacker = np.zeros((n_resources, n_non_attacker_vars))
        for j in range(n_resources):
            for local_i in range(n_non_attackers):
                A_non_attacker[j, local_i * n_resources + j] = 1.0
        b_non_attacker = remaining_supply
        
        sol_non_attacker = solvers.cp(
            F_non_attacker, 
            G=matrix(G_non_attacker), 
            h=matrix(h_non_attacker),
            A=matrix(A_non_attacker),
            b=matrix(b_non_attacker)
        )
        
        if sol_non_attacker['status'] not in ['optimal', 'unknown']:
            if verbose:
                print(f"  [TwoPhase] Phase 2 failed: {sol_non_attacker['status']}")
            # Use equal allocation for non-attackers as fallback
            x_non_attacker = x0_non_attacker
        else:
            x_non_attacker = np.maximum(np.array(sol_non_attacker['x']).flatten(), 0)
            if verbose:
                print(f"  [TwoPhase] Phase 2 complete, status={sol_non_attacker['status']}")
        
    except Exception as e:
        if verbose:
            print(f"  [TwoPhase] Phase 2 exception: {e}")
        x_non_attacker = x0_non_attacker
    
    # ========== Combine results ==========
    x_full = np.zeros(n_full_vars)
    
    # Fill in attacker allocations
    for local_i, a in enumerate(attackers):
        for j in range(n_resources):
            x_full[a * n_resources + j] = x_attacker[local_i * n_resources + j]
    
    # Fill in non-attacker allocations
    for local_i, na in enumerate(non_attackers):
        for j in range(n_resources):
            x_full[na * n_resources + j] = x_non_attacker[local_i * n_resources + j]
    
    return x_full, 'optimal'


def _solve_inner_optimization(
    utilities: np.ndarray,
    n_agents: int,
    n_resources: int,
    attacking_group: Set[int],
    target_group: Set[int],
    G: np.ndarray,
    h: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    supply: np.ndarray,
    maximize_harm: bool,
    verbose: bool,
    solver: Optional[str] = None,
    use_two_phase: bool = True
) -> Tuple[np.ndarray, str]:
    """
    Solve the inner optimization problem for the given constraints.
    
    The objective is: sum_{i in target} log(V_i)
    
    This maximizes/minimizes the Nash welfare of the target group.
    
    All agents are free variables with equality constraints sum_i x_ij = supply_j.
    
    Args:
        utilities: Utility matrix
        n_agents: Number of agents
        n_resources: Number of resources
        attacking_group: Set of attacker indices
        target_group: Set of target indices
        G: Inequality constraint matrix
        h: Inequality constraint RHS
        A_eq: Equality constraint matrix
        b_eq: Equality constraint RHS
        supply: Resource supply vector
        maximize_harm: If True, minimize target utility
        verbose: Print progress
        solver: 'gurobi', 'cvxopt', or None (use default)
        use_two_phase: If True, use two-phase optimization (attackers then non-attackers)
    
    Returns:
        (x_optimal, status): Optimal allocation (flattened) and status
    """
    if solver is None:
        solver = _DEFAULT_SOLVER
    
    # Use two-phase optimization by default (more efficient, no swap constraints needed)
    if use_two_phase and not maximize_harm:
        return _solve_inner_optimization_two_phase(
            utilities, n_agents, n_resources, attacking_group, target_group,
            G, h, A_eq, b_eq, supply, maximize_harm, verbose, solver
        )
    
    # Fall back to single-phase optimization for maximize_harm or if two-phase disabled
    # Try Gurobi first if requested
    if solver == 'gurobi':
        if _check_gurobi_available():
            return _solve_inner_optimization_gurobi(
                utilities, n_agents, n_resources, attacking_group, target_group,
                G, h, A_eq, b_eq, supply, maximize_harm, verbose
            )
        else:
            if verbose:
                print("  [InnerOpt] Gurobi not available, falling back to CVXOPT")
    
    # Use CVXOPT
    return _solve_inner_optimization_cvxopt(
        utilities, n_agents, n_resources, attacking_group, target_group,
        G, h, A_eq, b_eq, supply, maximize_harm, verbose
    )


def compute_optimal_self_benefit_constraint(
    utilities: np.ndarray,
    attacking_group: Set[int],
    victim_group: Optional[Set[int]] = None,
    supply: Optional[np.ndarray] = None,
    initial_constraints: Optional[Dict[int, AgentDiversityConstraint]] = None,
    verbose: bool = False,
    timing: bool = False,
    debug: bool = False,
    use_projection: bool = False,
    use_integral_method: bool = False
) -> Tuple[Optional[Dict[int, ProportionalityConstraint]], Dict]:
    """
    Compute the diversity constraints that maximize benefit for the attacking group.
    
    This finds the constraints that, when issued by the attacking group, minimize
    their p-MON score (i.e., maximize their utility gain from issuing the constraint).
    
    The optimization solves:
        min p-MON = (∏_{i ∈ K} u_i^before / u_i^after)^(1/|K|)
    
    which is equivalent to:
        max (∏_{i ∈ K} u_i^after)^(1/|K|) / (∏_{i ∈ K} u_i^before)^(1/|K|)
    
    Since u_i^before is fixed (from the initial allocation), this reduces to
    finding the constraints that maximize the geometric mean of the attacking
    group's utilities in the resulting allocation.
    
    Args:
        utilities: Preference matrix of shape (n_agents, n_resources) where
                   utilities[i, j] is agent i's value for resource j.
                   Each row should sum to 1.
        attacking_group: Set of agent indices who will issue the constraint.
                        Each agent in this group receives their own LinearConstraint.
        victim_group: Optional set of agent indices to compute q-NEE for.
                     If None, q-NEE is computed for all non-attackers.
                     Must be disjoint from attacking_group if specified.
        supply: Optional resource supply vector of shape (n_resources,).
                Defaults to all 1s.
        initial_constraints: Optional dict mapping agent_id -> AgentDiversityConstraint
                            for any pre-existing constraints.
        verbose: If True, print optimization progress.
        timing: If True, include timing information in the returned info dict.
        debug: If True, include additional debug information and enable verbose.
        use_projection: If True, use projection-based optimization (default True).
        use_integral_method: If True, use integral-based boundary finding instead of binary search (default False).
    
    Returns:
        Tuple of:
        - constraints: Dict mapping agent_id -> ProportionalityConstraint for each 
                      agent in attacking_group, or None if no beneficial constraint 
                      exists. Each ProportionalityConstraint specifies the ratios
                      in which the agent must receive resources.
        - info: Dict containing:
            - 'initial_result': AllocationResult before constraints
            - 'final_result': AllocationResult after optimal constraints
            - 'initial_utilities': Utilities before the constraints
            - 'final_utilities': Utilities after the optimal constraints
            - 'p_mon_group': The group p-MON score achieved
            - 'p_mon_individual': Dict of individual p-MON scores for attackers
            - 'q_nee_group': The group q-NEE score for victim group
            - 'q_nee_individual': Dict of individual q-NEE scores for victims
            - 'welfare_loss_ratio': Ratio of final/initial Nash welfare
            - 'initial_allocation': Allocation before constraints
            - 'final_allocation': Allocation after constraints
            - 'attacking_group': The attacking group (for reference)
            - 'victim_group': The victim group used (for reference)
            - 'status': Optimization status ('optimal', 'no_benefit', or error)
    
    Example:
        >>> utilities = generate_normalized_utilities(4, 6)
        >>> attacking_group = {0, 1}  # Agents 0 and 1 will issue constraints
        >>> victim_group = {2, 3}     # Compute harm to agents 2 and 3
        >>> constraints, info = compute_optimal_self_benefit_constraint(
        ...     utilities, attacking_group, victim_group
        ... )
        >>> print(f"Achieved p-MON: {info['p_mon_group']}")
        >>> print(f"Collateral q-NEE: {info['q_nee_group']}")
        >>> # p-MON < 1 means the attacking group benefited
        >>> # Each constraint specifies ratios, e.g. [1, 2, 3] means
        >>> # for every 1 unit of item 0, must have 2 of item 1, 3 of item 2
        >>> for agent_id, constraint in constraints.items():
        ...     print(f"Agent {agent_id} ratios: {constraint.get_ratios()}")
        ...     optimizer.set_agent_constraint(agent_id, constraint)
    
    Notes:
        - Each agent in attacking_group receives their own ProportionalityConstraint.
        - All returned constraints are guaranteed to contain the origin 
          (receiving nothing is always feasible).
        - If the attacking group cannot benefit from any constraint, returns None.
    
    Raises:
        ValueError: If attacking_group and victim_group overlap.
    """
    # Validate inputs
    if victim_group is not None and (attacking_group & victim_group):
        raise ValueError(
            f"attacking_group and victim_group must be disjoint. "
            f"Overlap: {attacking_group & victim_group}"
        )
    
    import time
    timing_info = {}
    debug_info = {}
    
    # Debug implies verbose
    if debug:
        verbose = True
    
    n_agents, n_resources = utilities.shape
    supply = supply if supply is not None else np.ones(n_resources)
    initial_constraints = initial_constraints or {}
    
    # Default victim group: all non-attackers
    if victim_group is None:
        victim_group = set(range(n_agents)) - attacking_group
    
    if debug:
        print(f"\n[DEBUG] compute_optimal_self_benefit_constraint")
        print(f"[DEBUG] n_agents={n_agents}, n_resources={n_resources}")
        print(f"[DEBUG] attacking_group={attacking_group}")
        print(f"[DEBUG] victim_group={victim_group}")
    
    # Step 1: Compute initial allocation (before any new constraint)
    t_start = time.time()
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    for agent_id, constraint in initial_constraints.items():
        optimizer.set_agent_constraint(agent_id, constraint)
    
    initial_result = optimizer.solve(verbose=verbose)
    initial_utilities = initial_result.agent_utilities
    initial_allocation = initial_result.allocation
    if timing:
        timing_info['initial_solve_time'] = time.time() - t_start
    
    # Create the same oracles used in _solve_optimal_constraint_convex_program for consistency
    swap_oracle = SwapOptimalityOracle(n_agents, n_resources, utilities, attacking_group, supply, verbose=debug)
    dir_deriv_oracle = DirectionalDerivativeOracle(n_agents, n_resources, utilities, attacking_group, supply, verbose=debug, use_integral_method=use_integral_method)
    
    # Convert initial allocation to free variables format (all agents now)
    initial_x_free = initial_allocation.flatten()
    
    # Check both oracles on initial allocation
    initial_swap_violated, _, _ = swap_oracle.is_violated(initial_x_free)
    initial_dir_violated, _, _ = dir_deriv_oracle.is_violated(initial_x_free)
    initial_on_boundary = not initial_swap_violated and not initial_dir_violated
    
    # Compute utilities and directional derivative
    x_full = initial_x_free.reshape((n_agents, n_resources))
    V = np.array([np.dot(utilities[i], x_full[i]) for i in range(n_agents)])
    
    # Compute directional derivative using dir_deriv_oracle's method
    forward_direction, forward_valid = dir_deriv_oracle._compute_direction(x_full, V, backward=False)
    if forward_valid:
        initial_directional_deriv = dir_deriv_oracle._compute_directional_derivative_with_direction(x_full, V, forward_direction)
    else:
        backward_direction, _ = dir_deriv_oracle._compute_direction(x_full, V, backward=True)
        backward_deriv = dir_deriv_oracle._compute_directional_derivative_with_direction(x_full, V, backward_direction)
        initial_directional_deriv = -backward_deriv
    
    if debug:
        print(f"[DEBUG] Initial allocation:\n{initial_allocation.round(4)}")
        print(f"[DEBUG] Initial utilities: {initial_utilities.round(6)}")
        print(f"[DEBUG] Reconstructed utilities: {V.round(6)}")
        
        # Show v_{i,j}/V_i for each agent to check KKT conditions
        print(f"[DEBUG] Checking KKT conditions (v_ij/V_i for each agent):")
        for i in range(n_agents):
            ratios = utilities[i] / V[i]
            print(f"[DEBUG]   Agent {i}: {ratios.round(6)}")
        
        print(f"[DEBUG] Initial swap violated: {initial_swap_violated}")
        print(f"[DEBUG] Initial dir_deriv violated: {initial_dir_violated}")
        print(f"[DEBUG] Initial on optimal boundary: {initial_on_boundary}")
        print(f"[DEBUG] Initial directional derivative: {initial_directional_deriv:.6e}")
    
    # Step 2: Find optimal constraints for attacking group using convex program
    t_start = time.time()
    optimal_directions, convex_program_allocation, opt_status, cp_debug_info = _solve_optimal_constraint_convex_program(
        utilities=utilities,
        attacking_group=attacking_group,
        target_group=attacking_group,  # Maximize benefit to attackers
        supply=supply,
        initial_constraints=initial_constraints,
        maximize_harm=False,  # Maximize utility (minimize p-MON)
        verbose=verbose,
        debug=debug,
        use_projection=use_projection,
        use_integral_method=use_integral_method
    )
    if timing:
        timing_info['optimization_time'] = time.time() - t_start
    
    # Compute utilities from convex program allocation
    convex_program_utilities = np.array([
        np.dot(utilities[i], convex_program_allocation[i]) for i in range(n_agents)
    ])
    
    if debug:
        print(f"[DEBUG] Optimization status: {opt_status}")
        print(f"[DEBUG] CP iterations: {cp_debug_info['iterations']}, cuts added: {cp_debug_info['total_cuts']} (swap: {cp_debug_info['swap_cuts']}, dir_deriv: {cp_debug_info['dir_deriv_cuts']}), converged: {cp_debug_info['converged']}")
        print(f"[DEBUG] Optimal directions: {optimal_directions}")
        print(f"[DEBUG] Convex program allocation:\n{convex_program_allocation.round(6)}")
        print(f"[DEBUG] Convex program utilities: {convex_program_utilities.round(6)}")
        # Show what p-MON would be if we used convex program allocation directly
        for agent_id in attacking_group:
            cp_pmon = initial_utilities[agent_id] / convex_program_utilities[agent_id] if convex_program_utilities[agent_id] > 1e-10 else float('inf')
            print(f"[DEBUG] Agent {agent_id} p-MON from convex program: {cp_pmon:.6f}")
    
    # Step 3: Create ProportionalityConstraints from the directions
    optimal_constraints: Optional[Dict[int, ProportionalityConstraint]] = None
    
    if optimal_directions is not None and opt_status == 'optimal':
        optimal_constraints = {}
        for agent_id, direction in optimal_directions.items():
            if direction is not None:
                optimal_constraints[agent_id] = ProportionalityConstraint(n_resources, direction)
                if debug:
                    print(f"[DEBUG] Created ProportionalityConstraint for agent {agent_id}")
                    print(f"[DEBUG]   ratios: {optimal_constraints[agent_id].get_ratios().round(6)}")
    
    # Check oracles on convex program allocation
    cp_x_free = convex_program_allocation.flatten()
    
    cp_x_full = cp_x_free.reshape((n_agents, n_resources))
    cp_V = np.array([np.dot(utilities[i], cp_x_full[i]) for i in range(n_agents)])
    
    # Compute directional derivative
    cp_forward_direction, cp_forward_valid = dir_deriv_oracle._compute_direction(cp_x_full, cp_V, backward=False)
    if cp_forward_valid:
        cp_directional_deriv = dir_deriv_oracle._compute_directional_derivative_with_direction(cp_x_full, cp_V, cp_forward_direction)
    else:
        cp_backward_direction, _ = dir_deriv_oracle._compute_direction(cp_x_full, cp_V, backward=True)
        cp_backward_deriv = dir_deriv_oracle._compute_directional_derivative_with_direction(cp_x_full, cp_V, cp_backward_direction)
        cp_directional_deriv = -cp_backward_deriv
    
    # Check both oracles
    cp_swap_violated, _, _ = swap_oracle.is_violated(cp_x_free)
    cp_dir_violated, _, _ = dir_deriv_oracle.is_violated(cp_x_free)
    cp_on_boundary = not cp_swap_violated and not cp_dir_violated
    
    if debug:
        print(f"[DEBUG] Convex program swap violated: {cp_swap_violated}")
        print(f"[DEBUG] Convex program dir_deriv violated: {cp_dir_violated}")
        print(f"[DEBUG] Convex program directional derivative: {cp_directional_deriv:.6e}")
        print(f"[DEBUG] Convex program on optimal boundary: {cp_on_boundary}")
    
    # Step 4: Compute final result by solving with the constraints
    t_start = time.time()
    if optimal_constraints:
        if debug:
            print(f"[DEBUG] Solving final Nash welfare with {len(optimal_constraints)} constraints")
            print(f"[DEBUG] Optimal directions from convex program:")
            for agent_id, direction in optimal_directions.items():
                print(f"[DEBUG]   Agent {agent_id}: {direction}")
            print(f"[DEBUG] Convex program allocation for attackers:")
            for agent_id in attacking_group:
                print(f"[DEBUG]   Agent {agent_id}: {convex_program_allocation[agent_id]}")
        
        final_optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
        for agent_id, constraint in initial_constraints.items():
            final_optimizer.set_agent_constraint(agent_id, constraint)
        for agent_id, constraint in optimal_constraints.items():
            final_optimizer.add_agent_constraint(agent_id, constraint)
            if debug:
                print(f"[DEBUG] Added constraint for agent {agent_id}, ratios: {constraint.get_ratios()}")
        
        final_result = final_optimizer.solve(verbose=verbose)
        final_utilities = final_result.agent_utilities
        final_allocation = final_result.allocation
        
        if debug:
            print(f"[DEBUG] Final allocation for attackers:")
            for agent_id in attacking_group:
                print(f"[DEBUG]   Agent {agent_id}: {final_allocation[agent_id]}")
                # Check if final allocation satisfies the constraint
                constraint = optimal_constraints[agent_id]
                violated, _, _ = constraint.is_violated(final_allocation[agent_id])
                print(f"[DEBUG]   Constraint violated: {violated}")
        
        # Check oracles on final allocation
        final_x_free = final_allocation.flatten()
        
        final_x_full = final_x_free.reshape((n_agents, n_resources))
        final_V = np.array([np.dot(utilities[i], final_x_full[i]) for i in range(n_agents)])
        
        # Compute directional derivative
        final_forward_direction, final_forward_valid = dir_deriv_oracle._compute_direction(final_x_full, final_V, backward=False)
        if final_forward_valid:
            final_directional_deriv = dir_deriv_oracle._compute_directional_derivative_with_direction(final_x_full, final_V, final_forward_direction)
        else:
            final_backward_direction, _ = dir_deriv_oracle._compute_direction(final_x_full, final_V, backward=True)
            final_backward_deriv = dir_deriv_oracle._compute_directional_derivative_with_direction(final_x_full, final_V, final_backward_direction)
            final_directional_deriv = -final_backward_deriv
        
        # Check both oracles
        final_swap_violated, _, _ = swap_oracle.is_violated(final_x_free)
        final_dir_violated, _, _ = dir_deriv_oracle.is_violated(final_x_free)
        final_on_boundary = not final_swap_violated and not final_dir_violated
        
        # Check if final allocation matches convex program allocation
        allocation_diff_norm = np.linalg.norm(final_allocation - convex_program_allocation)
        utility_diff_norm = np.linalg.norm(final_utilities - convex_program_utilities)
        allocations_match = allocation_diff_norm < 1e-4
        
        if not allocations_match:
            print(f"[WARNING] Final allocation differs from convex program allocation!")
            print(f"[WARNING]   Allocation difference norm: {allocation_diff_norm:.6e}")
            print(f"[WARNING]   Utility difference norm: {utility_diff_norm:.6e}")
            print(f"[WARNING]   CP utilities: {convex_program_utilities.round(6)}")
            print(f"[WARNING]   Final utilities: {final_utilities.round(6)}")
            print(f"[WARNING]   CP directional derivative: {cp_directional_deriv:.6e} (should be ~0)")
            print(f"[WARNING]   Final directional derivative: {final_directional_deriv:.6e} (should be ~0)")
            print(f"[WARNING]   CP swap violated: {cp_swap_violated}, CP dir_deriv violated: {cp_dir_violated}")
            print(f"[WARNING]   Final swap violated: {final_swap_violated}, Final dir_deriv violated: {final_dir_violated}")
            print(f"[WARNING]   Attacker allocations:")
            for agent_id in attacking_group:
                cp_alloc = convex_program_allocation[agent_id]
                final_alloc = final_allocation[agent_id]
                print(f"[WARNING]     Agent {agent_id} CP:    {cp_alloc.round(6)}")
                print(f"[WARNING]     Agent {agent_id} Final: {final_alloc.round(6)}")
                
                # Check if they are scalar multiples of each other
                cp_norm = np.linalg.norm(cp_alloc)
                final_norm = np.linalg.norm(final_alloc)
                if cp_norm > 1e-10 and final_norm > 1e-10:
                    # Normalize both vectors
                    cp_normalized = cp_alloc / cp_norm
                    final_normalized = final_alloc / final_norm
                    # Check if normalized vectors are the same (or opposite)
                    dot_product = np.dot(cp_normalized, final_normalized)
                    if abs(abs(dot_product) - 1.0) < 1e-6:
                        # They are scalar multiples
                        scalar_ratio = final_norm / cp_norm
                        print(f"[WARNING]     Agent {agent_id} ARE scalar multiples! Ratio (Final/CP): {scalar_ratio:.6f}")
                    else:
                        # They are not scalar multiples - compute angle
                        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
                        angle_deg = np.degrees(angle_rad)
                        print(f"[WARNING]     Agent {agent_id} NOT scalar multiples. Angle between: {angle_deg:.2f} degrees")
                elif cp_norm < 1e-10 and final_norm < 1e-10:
                    print(f"[WARNING]     Agent {agent_id} Both are zero vectors")
                else:
                    print(f"[WARNING]     Agent {agent_id} One is zero, other is not (CP norm: {cp_norm:.6e}, Final norm: {final_norm:.6e})")
        
        if debug:
            print(f"[DEBUG] Final allocation (after re-solving with constraints):\n{final_allocation.round(6)}")
            print(f"[DEBUG] Final utilities: {final_utilities.round(6)}")
            print(f"[DEBUG] Final Nash welfare: {final_result.nash_welfare:.6f}")
            print(f"[DEBUG] Final swap violated: {final_swap_violated}")
            print(f"[DEBUG] Final dir_deriv violated: {final_dir_violated}")
            print(f"[DEBUG] Final directional derivative: {final_directional_deriv:.6e}")
            print(f"[DEBUG] Final on optimal boundary: {final_on_boundary}")
            # Compare convex program vs final allocation
            print(f"[DEBUG] === COMPARISON: Convex Program vs Final ===")
            for agent_id in attacking_group:
                cp_util = convex_program_utilities[agent_id]
                final_util = final_utilities[agent_id]
                print(f"[DEBUG] Agent {agent_id}: CP utility={cp_util:.6f}, Final utility={final_util:.6f}, diff={final_util - cp_util:.6f}")
            print(f"[DEBUG] Allocation difference norm: {allocation_diff_norm:.6e}")
            print(f"[DEBUG] Allocations match (within 1e-4): {allocations_match}")
            print(f"[DEBUG] Directional deriv: CP={cp_directional_deriv:.6e}, Final={final_directional_deriv:.6e}")
    else:
        if debug:
            print(f"[DEBUG] No constraints found, using initial allocation")
        final_result = initial_result
        final_utilities = initial_utilities.copy()
        final_allocation = initial_allocation.copy()
        final_directional_deriv = initial_directional_deriv
        final_on_boundary = initial_on_boundary
    if timing:
        timing_info['final_solve_time'] = time.time() - t_start
    
    # Step 5: Compute metrics
    t_start = time.time()
    
    # p-MON for ALL agents (attacking group will have meaningful values)
    p_mon_individual_all = {}
    for i in range(n_agents):
        if final_utilities[i] > 1e-10:
            p_mon_individual_all[i] = initial_utilities[i] / final_utilities[i]
        else:
            p_mon_individual_all[i] = float('inf') if initial_utilities[i] > 1e-10 else 1.0
    
    # p-MON group (for attacking group only)
    if attacking_group:
        k = len(attacking_group)
        log_sum = sum(np.log(max(p_mon_individual_all[i], 1e-10)) for i in attacking_group)
        p_mon_group = np.exp(log_sum / k)
    else:
        p_mon_group = 1.0
    
    # q-NEE for ALL agents (victim group will have meaningful values)
    q_nee_individual_all = {}
    for i in range(n_agents):
        if initial_utilities[i] > 1e-10:
            q_nee_individual_all[i] = final_utilities[i] / initial_utilities[i]
        else:
            q_nee_individual_all[i] = float('inf') if final_utilities[i] > 1e-10 else 1.0
    
    # q-NEE group (for victim group only)
    if victim_group:
        k = len(victim_group)
        log_sum = sum(np.log(max(q_nee_individual_all[i], 1e-10)) for i in victim_group)
        q_nee_group = np.exp(log_sum / k)
    else:
        q_nee_group = 1.0
    
    # Welfare loss ratio
    welfare_loss_ratio = final_result.nash_welfare / initial_result.nash_welfare
    
    if timing:
        timing_info['metrics_time'] = time.time() - t_start
    
    # Extract constraint ratios for debug output
    constraint_ratios = {}
    if optimal_constraints:
        for agent_id, constraint in optimal_constraints.items():
            constraint_ratios[agent_id] = constraint.get_ratios()
    
    info = {
        'initial_result': initial_result,
        'final_result': final_result,
        'initial_utilities': initial_utilities,
        'final_utilities': final_utilities,
        'p_mon_group': p_mon_group,
        'p_mon_individual': {i: p_mon_individual_all[i] for i in attacking_group},  # Attacking group only
        'p_mon_individual_all': p_mon_individual_all,  # All agents
        'q_nee_group': q_nee_group,
        'q_nee_individual': {i: q_nee_individual_all[i] for i in victim_group},  # Victim group only
        'q_nee_individual_all': q_nee_individual_all,  # All agents
        'welfare_loss_ratio': welfare_loss_ratio,
        'initial_allocation': initial_allocation,
        'final_allocation': final_allocation,
        'convex_program_allocation': convex_program_allocation,  # Allocation from convex program (before re-solving)
        'convex_program_utilities': convex_program_utilities,  # Utilities from convex program allocation
        'constraint_ratios': constraint_ratios,  # The ratios used in ProportionalityConstraint
        'attacking_group': attacking_group,
        'victim_group': victim_group,
        'cp_optimization_status': opt_status,  # Status from convex program optimization
        'cp_iterations': cp_debug_info['iterations'],
        'cp_swap_cuts': cp_debug_info['swap_cuts'],
        'cp_dir_deriv_cuts': cp_debug_info['dir_deriv_cuts'],
        'cp_total_cuts': cp_debug_info['total_cuts'],
        'cp_backward_checks': cp_debug_info.get('backward_checks', 0),
        'cp_backward_violations': cp_debug_info.get('backward_violations', 0),
        'cp_converged': cp_debug_info['converged'],
        'cp_cut_details': cp_debug_info.get('cut_details', []),  # Details of each cut
        'status': opt_status if optimal_constraints else 'no_benefit',
        'timing': timing_info if timing else None,
        # Boundary check info
        'initial_on_boundary': initial_on_boundary,
        'initial_directional_deriv': initial_directional_deriv,
        'cp_on_boundary': cp_on_boundary,
        'cp_directional_deriv': cp_directional_deriv,
        'final_on_boundary': final_on_boundary,
        'final_directional_deriv': final_directional_deriv,
    }
    
    return optimal_constraints, info


def compute_optimal_harm_constraint(
    utilities: np.ndarray,
    attacking_group: Set[int],
    victim_group: Set[int],
    supply: Optional[np.ndarray] = None,
    initial_constraints: Optional[Dict[int, AgentDiversityConstraint]] = None,
    verbose: bool = False,
    timing: bool = False,
    debug: bool = False
) -> Tuple[Optional[Dict[int, ProportionalityConstraint]], Dict]:
    """
    Compute the diversity constraints that maximize harm to a victim group.
    
    This finds the constraints that, when issued by the attacking group, minimize
    the q-NEE score for the victim group (i.e., maximize the utility loss
    experienced by the victims).
    
    The optimization solves:
        min q-NEE = (∏_{i ∈ V} u_i^after / u_i^before)^(1/|V|)
    
    where V is the victim group. Since u_i^before is fixed, this is equivalent to:
        min (∏_{i ∈ V} u_i^after)^(1/|V|)
    
    i.e., minimizing the geometric mean of victim utilities after the constraints.
    
    Args:
        utilities: Preference matrix of shape (n_agents, n_resources) where
                   utilities[i, j] is agent i's value for resource j.
                   Each row should sum to 1.
        attacking_group: Set of agent indices who will issue the constraints.
                        Each agent in this group receives their own LinearConstraint.
        victim_group: Set of agent indices whose harm we want to maximize.
                     Must be disjoint from attacking_group.
        supply: Optional resource supply vector of shape (n_resources,).
                Defaults to all 1s.
        initial_constraints: Optional dict mapping agent_id -> AgentDiversityConstraint
                            for any pre-existing constraints.
        verbose: If True, print optimization progress.
        timing: If True, include timing information in the returned info dict.
        debug: If True, include additional debug information and enable verbose.
    
    Returns:
        Tuple of:
        - constraints: Dict mapping agent_id -> ProportionalityConstraint for each
                      agent in attacking_group, or None if no harmful constraint 
                      exists. Each ProportionalityConstraint specifies the ratios
                      in which the agent must receive resources.
        - info: Dict containing:
            - 'initial_result': AllocationResult before constraints
            - 'final_result': AllocationResult after optimal constraints
            - 'initial_utilities': Utilities before the constraints
            - 'final_utilities': Utilities after the optimal constraints
            - 'q_nee_group': The group q-NEE score achieved for victim group
            - 'q_nee_individual': Dict of individual q-NEE scores for victims
            - 'p_mon_group': The group p-MON score for the attacking group
            - 'p_mon_individual': Dict of individual p-MON scores for attackers
            - 'welfare_loss_ratio': Ratio of final/initial Nash welfare
            - 'initial_allocation': Allocation before constraints
            - 'final_allocation': Allocation after constraints
            - 'attacking_group': The attacking group (for reference)
            - 'victim_group': The victim group (for reference)
            - 'status': Optimization status ('optimal', 'no_harm', or error)
    
    Example:
        >>> utilities = generate_normalized_utilities(4, 6)
        >>> attacking_group = {0}      # Agent 0 attacks
        >>> victim_group = {2, 3}      # Agents 2, 3 are victims
        >>> constraints, info = compute_optimal_harm_constraint(
        ...     utilities, attacking_group, victim_group
        ... )
        >>> print(f"Achieved q-NEE: {info['q_nee_group']}")
        >>> print(f"Attacker p-MON: {info['p_mon_group']}")
        >>> # q-NEE < 1 means the victim group was harmed
        >>> # Each constraint specifies ratios, e.g. [1, 2, 3] means
        >>> # for every 1 unit of item 0, must have 2 of item 1, 3 of item 2
        >>> for agent_id, constraint in constraints.items():
        ...     print(f"Agent {agent_id} ratios: {constraint.get_ratios()}")
        ...     optimizer.set_agent_constraint(agent_id, constraint)
    
    Notes:
        - attacking_group and victim_group must be disjoint.
        - Each agent in attacking_group receives their own ProportionalityConstraint.
        - All returned constraints are guaranteed to contain the origin.
        - This finds the "worst case" constraint from the victim's perspective,
          useful for analyzing the vulnerability of allocation mechanisms.
    
    Raises:
        ValueError: If attacking_group and victim_group overlap.
    """
    # Validate inputs
    if attacking_group & victim_group:
        raise ValueError(
            f"attacking_group and victim_group must be disjoint. "
            f"Overlap: {attacking_group & victim_group}"
        )
    
    import time
    timing_info = {}
    
    # Debug implies verbose
    if debug:
        verbose = True
    
    n_agents, n_resources = utilities.shape
    supply = supply if supply is not None else np.ones(n_resources)
    initial_constraints = initial_constraints or {}
    
    if debug:
        print(f"\n[DEBUG] compute_optimal_harm_constraint")
        print(f"[DEBUG] n_agents={n_agents}, n_resources={n_resources}")
        print(f"[DEBUG] attacking_group={attacking_group}")
        print(f"[DEBUG] victim_group={victim_group}")
    
    # Step 1: Compute initial allocation (before any new constraint)
    t_start = time.time()
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    for agent_id, constraint in initial_constraints.items():
        optimizer.set_agent_constraint(agent_id, constraint)
    
    initial_result = optimizer.solve(verbose=verbose)
    initial_utilities = initial_result.agent_utilities
    initial_allocation = initial_result.allocation
    if timing:
        timing_info['initial_solve_time'] = time.time() - t_start
    
    # Check if initial allocation is on optimal boundary
    boundary_oracle = OptimalBoundaryOracle(n_agents, n_resources, utilities, attacking_group, supply, verbose=debug)
    
    # Convert initial allocation to free variables format (all agents now)
    initial_x_free = initial_allocation.flatten()
    
    initial_boundary_violated, _, _ = boundary_oracle.is_violated(initial_x_free)
    initial_on_boundary = not initial_boundary_violated
    
    # Compute directional derivative
    x_full = boundary_oracle._get_full_allocation(initial_x_free)
    V = boundary_oracle._compute_utilities_from_allocation(x_full)
    initial_directional_deriv = boundary_oracle._compute_directional_derivative(x_full, V)
    
    if debug:
        print(f"[DEBUG] Initial allocation:\n{initial_allocation.round(4)}")
        print(f"[DEBUG] Initial utilities: {initial_utilities.round(6)}")
        print(f"[DEBUG] Initial on optimal boundary: {initial_on_boundary}")
        print(f"[DEBUG] Initial directional derivative: {initial_directional_deriv:.6e}")
    
    # Step 2: Find optimal constraints for maximum harm to victim group
    t_start = time.time()
    optimal_directions, final_allocation, opt_status, cp_debug_info = _solve_optimal_constraint_convex_program(
        utilities=utilities,
        attacking_group=attacking_group,
        target_group=victim_group,  # Minimize victim utility
        supply=supply,
        initial_constraints=initial_constraints,
        maximize_harm=True,  # Minimize utility (maximize harm)
        verbose=verbose,
        debug=debug
    )
    if timing:
        timing_info['optimization_time'] = time.time() - t_start
    
    if debug:
        print(f"[DEBUG] Optimization status: {opt_status}")
        print(f"[DEBUG] CP iterations: {cp_debug_info['iterations']}, cuts added: {cp_debug_info['total_cuts']} (swap: {cp_debug_info['swap_cuts']}, dir_deriv: {cp_debug_info['dir_deriv_cuts']}), converged: {cp_debug_info['converged']}")
        print(f"[DEBUG] Optimal directions: {optimal_directions}")
    
    # Step 3: Create ProportionalityConstraints from the directions
    optimal_constraints: Optional[Dict[int, ProportionalityConstraint]] = None
    
    if optimal_directions is not None and opt_status == 'optimal':
        optimal_constraints = {}
        for agent_id, direction in optimal_directions.items():
            if direction is not None:
                optimal_constraints[agent_id] = ProportionalityConstraint(n_resources, direction)
                if debug:
                    print(f"[DEBUG] Created ProportionalityConstraint for agent {agent_id}")
                    print(f"[DEBUG]   ratios: {optimal_constraints[agent_id].get_ratios().round(6)}")
    
    # Step 4: Compute final result by solving with the constraints
    t_start = time.time()
    if optimal_constraints:
        if debug:
            print(f"[DEBUG] Solving final Nash welfare with {len(optimal_constraints)} constraints")
        final_optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
        for agent_id, constraint in initial_constraints.items():
            final_optimizer.set_agent_constraint(agent_id, constraint)
        for agent_id, constraint in optimal_constraints.items():
            final_optimizer.add_agent_constraint(agent_id, constraint)
        final_result = final_optimizer.solve(verbose=verbose)
        final_utilities = final_result.agent_utilities
        final_allocation = final_result.allocation
        
        if debug:
            print(f"[DEBUG] Final allocation:\n{final_allocation.round(6)}")
            print(f"[DEBUG] Final utilities: {final_utilities.round(6)}")
            print(f"[DEBUG] Final Nash welfare: {final_result.nash_welfare:.6f}")
    else:
        if debug:
            print(f"[DEBUG] No constraints found, using initial allocation")
        final_result = initial_result
        final_utilities = initial_utilities.copy()
        final_allocation = initial_allocation.copy()
    if timing:
        timing_info['final_solve_time'] = time.time() - t_start
    
    # Step 5: Compute metrics
    t_start = time.time()
    
    # p-MON for ALL agents
    p_mon_individual_all = {}
    for i in range(n_agents):
        if final_utilities[i] > 1e-10:
            p_mon_individual_all[i] = initial_utilities[i] / final_utilities[i]
        else:
            p_mon_individual_all[i] = float('inf') if initial_utilities[i] > 1e-10 else 1.0
    
    # p-MON group (for attacking group only)
    if attacking_group:
        k = len(attacking_group)
        log_sum = sum(np.log(max(p_mon_individual_all[i], 1e-10)) for i in attacking_group)
        p_mon_group = np.exp(log_sum / k)
    else:
        p_mon_group = 1.0
    
    # q-NEE for ALL agents
    q_nee_individual_all = {}
    for i in range(n_agents):
        if initial_utilities[i] > 1e-10:
            q_nee_individual_all[i] = final_utilities[i] / initial_utilities[i]
        else:
            q_nee_individual_all[i] = float('inf') if final_utilities[i] > 1e-10 else 1.0
    
    # q-NEE group (for victim group only)
    if victim_group:
        k = len(victim_group)
        log_sum = sum(np.log(max(q_nee_individual_all[i], 1e-10)) for i in victim_group)
        q_nee_group = np.exp(log_sum / k)
    else:
        q_nee_group = 1.0
    
    # Welfare loss ratio
    welfare_loss_ratio = final_result.nash_welfare / initial_result.nash_welfare
    
    if timing:
        timing_info['metrics_time'] = time.time() - t_start
    
    info = {
        'initial_result': initial_result,
        'final_result': final_result,
        'initial_utilities': initial_utilities,
        'final_utilities': final_utilities,
        'q_nee_group': q_nee_group,
        'q_nee_individual': {i: q_nee_individual_all[i] for i in victim_group},  # Victim group only
        'q_nee_individual_all': q_nee_individual_all,  # All agents
        'p_mon_group': p_mon_group,
        'p_mon_individual': {i: p_mon_individual_all[i] for i in attacking_group},  # Attacking group only
        'p_mon_individual_all': p_mon_individual_all,  # All agents
        'welfare_loss_ratio': welfare_loss_ratio,
        'initial_allocation': initial_allocation,
        'final_allocation': final_allocation,
        'attacking_group': attacking_group,
        'victim_group': victim_group,
        'status': opt_status if optimal_constraints else 'no_harm',
        # Additional detailed info
        'all_initial_utilities': {i: initial_utilities[i] for i in range(n_agents)},
        'all_final_utilities': {i: final_utilities[i] for i in range(n_agents)},
        'constraints_found': optimal_constraints is not None,
        'constraint_ratios': {
            agent_id: constraint.get_ratios().tolist() 
            for agent_id, constraint in (optimal_constraints or {}).items()
        } if optimal_constraints else {},
        'timing': timing_info if timing else None,
        # Boundary check info
        'initial_on_boundary': initial_on_boundary,
        'initial_directional_deriv': initial_directional_deriv,
    }
    
    return optimal_constraints, info


def _solve_nash_welfare_for_subset(
    utilities: np.ndarray,
    agent_subset: Set[int],
    available_supply: np.ndarray,
    n_agents: int,
    n_resources: int
) -> Dict[int, np.ndarray]:
    """
    Solve Nash welfare maximization for a subset of agents given available supply.
    
    Args:
        utilities: Full utility matrix (n_agents, n_resources)
        agent_subset: Set of agent indices to optimize for
        available_supply: Supply available for this subset
        n_agents: Total number of agents
        n_resources: Number of resources
        
    Returns:
        Dict mapping agent_id -> allocation array for all agents
        (agents not in subset get zero allocation)
    """
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    
    subset_list = sorted(agent_subset)
    n_subset = len(subset_list)
    
    if n_subset == 0:
        return {i: np.zeros(n_resources) for i in range(n_agents)}
    
    if np.all(available_supply < 1e-10):
        return {i: np.zeros(n_resources) for i in range(n_agents)}
    
    n_vars = n_subset * n_resources
    
    def var_idx(local_i, j):
        return local_i * n_resources + j
    
    def F(x=None, z=None):
        if x is None:
            # Initial point: equal division of available supply
            x0 = matrix(0.0, (n_vars, 1))
            for local_i, agent in enumerate(subset_list):
                for j in range(n_resources):
                    x0[var_idx(local_i, j)] = available_supply[j] / n_subset
            return (0, x0)
        
        # Compute utilities
        V = []
        for local_i, agent in enumerate(subset_list):
            v_i = 0.0
            for j in range(n_resources):
                v_i += utilities[agent, j] * x[var_idx(local_i, j)]
            V.append(max(v_i, 1e-12))
        
        # Objective: -sum of log utilities (minimize negative = maximize)
        f = sum(-np.log(v) for v in V)
        
        # Gradient
        Df = matrix(0.0, (1, n_vars))
        for local_i, agent in enumerate(subset_list):
            for j in range(n_resources):
                Df[var_idx(local_i, j)] = -utilities[agent, j] / V[local_i]
        
        if z is None:
            return (f, Df)
        
        # Hessian
        H = matrix(0.0, (n_vars, n_vars))
        for local_i, agent in enumerate(subset_list):
            for j1 in range(n_resources):
                for j2 in range(n_resources):
                    idx1 = var_idx(local_i, j1)
                    idx2 = var_idx(local_i, j2)
                    H[idx1, idx2] = z[0] * utilities[agent, j1] * utilities[agent, j2] / (V[local_i] ** 2)
        
        return (f, Df, H)
    
    # Inequality constraints: x >= 0
    G = matrix(-np.eye(n_vars))
    h = matrix(np.zeros(n_vars))
    
    # Equality constraints: sum over agents for each resource = available_supply
    A = matrix(0.0, (n_resources, n_vars))
    b = matrix(available_supply)
    for j in range(n_resources):
        for local_i in range(n_subset):
            A[j, var_idx(local_i, j)] = 1.0
    
    try:
        sol = solvers.cp(F, G, h, A=A, b=b)
        
        if sol['status'] != 'optimal':
            # Return equal division as fallback
            result = {}
            for i in range(n_agents):
                if i in agent_subset:
                    result[i] = available_supply / n_subset
                else:
                    result[i] = np.zeros(n_resources)
            return result
        
        x_sol = np.array(sol['x']).flatten()
        
        result = {}
        for i in range(n_agents):
            if i in agent_subset:
                local_i = subset_list.index(i)
                alloc = np.zeros(n_resources)
                for j in range(n_resources):
                    alloc[j] = max(0, x_sol[var_idx(local_i, j)])
                result[i] = alloc
            else:
                result[i] = np.zeros(n_resources)
        
        return result
        
    except Exception as e:
        # Return equal division as fallback
        result = {}
        for i in range(n_agents):
            if i in agent_subset:
                result[i] = available_supply / n_subset
            else:
                result[i] = np.zeros(n_resources)
        return result


def solve_optimal_harm_constraint_pgd(
    utilities: np.ndarray,
    attacking_group: Set[int],
    defending_group: Set[int],
    supply: Optional[np.ndarray] = None,
    initial_constraints: Optional[Dict[int, AgentDiversityConstraint]] = None,
    max_iterations: int = 100,
    step_size: float = 0.1,
    convergence_tol: float = 1e-6,
    use_adam: bool = True,
    verbose: bool = False,
    debug: bool = False,
) -> Tuple[Optional[Dict[int, ProportionalityConstraint]], Dict]:
    """
    Find the optimal harm constraint using projected gradient descent.
    
    This finds the attacker allocation (and thus proportionality constraint) that
    minimizes the Nash welfare of the defending group (product of defender utilities)
    while staying on the optimal boundary.
    
    The algorithm:
    1. Start from the unconstrained Nash welfare solution
    2. Compute gradient of defender Nash welfare w.r.t. attacker allocation
    3. Take a step in the negative gradient direction (to minimize)
    4. Project back onto the optimal boundary using DirectionalDerivativeOracle
    5. Repeat until convergence
    
    Args:
        utilities: Preference matrix of shape (n_agents, n_resources)
        attacking_group: Set of agent indices who will issue the constraints
        defending_group: Set of agent indices whose harm we want to maximize
                        (must be a subset of non-attackers)
        supply: Optional resource supply vector. Defaults to n_agents per resource.
        initial_constraints: Optional dict of pre-existing constraints
        max_iterations: Maximum number of gradient descent iterations
        step_size: Initial step size for gradient descent
        convergence_tol: Stop when objective improvement is below this threshold
        use_adam: If True, use Adam optimizer; if False, use vanilla gradient descent
        verbose: If True, print progress
        debug: If True, print detailed debug information
    
    Returns:
        Tuple of:
        - constraints: Dict mapping agent_id -> ProportionalityConstraint, or None
        - info: Dict with optimization details including:
            - 'converged': True if converged by objective improvement
            - 'iterations': Number of iterations performed
            - 'initial_defender_welfare': Product of defender utilities at start
            - 'final_defender_welfare': Product of defender utilities at end
            - 'initial_allocation': Starting allocation
            - 'final_allocation': Final allocation on boundary
            - 'objective_history': List of objective values per iteration
    """
    import time
    
    if debug:
        verbose = True
    
    n_agents, n_resources = utilities.shape
    supply = supply if supply is not None else np.ones(n_resources)
    initial_constraints = initial_constraints or {}
    non_attacking_group = set(range(n_agents)) - attacking_group
    
    # Validate defending_group is subset of non-attackers
    if not defending_group.issubset(non_attacking_group):
        raise ValueError(
            f"defending_group must be subset of non-attackers. "
            f"defending_group={defending_group}, non_attackers={non_attacking_group}"
        )
    
    if verbose:
        print(f"[PGD] Starting projected gradient descent for optimal harm constraint")
        print(f"[PGD] n_agents={n_agents}, n_resources={n_resources}")
        print(f"[PGD] attacking_group={attacking_group}")
        print(f"[PGD] defending_group={defending_group}")
        print(f"[PGD] max_iterations={max_iterations}, step_size={step_size}")
    
    # Create oracles
    dir_deriv_oracle = DirectionalDerivativeOracle(
        n_agents, n_resources, utilities, attacking_group, supply, verbose=debug
    )
    swap_oracle = SwapOptimalityOracle(
        n_agents, n_resources, utilities, attacking_group, supply, verbose=debug
    )
    
    # Step 1: Get initial allocation from unconstrained Nash welfare
    optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
    for agent_id, constraint in initial_constraints.items():
        optimizer.set_agent_constraint(agent_id, constraint)
    
    initial_result = optimizer.solve(verbose=False)
    x_current = initial_result.allocation.copy()
    
    if verbose:
        print(f"[PGD] Initial Nash welfare: {initial_result.nash_welfare:.6f}")
    
    # Compute initial defender welfare
    def compute_defender_welfare(x: np.ndarray) -> float:
        """Compute product of defender utilities (Nash welfare of defenders)."""
        V = np.array([np.dot(utilities[i], x[i]) for i in range(n_agents)])
        welfare = 1.0
        for d in defending_group:
            welfare *= max(V[d], 1e-20)
        return welfare
    
    def compute_utilities_arr(x: np.ndarray) -> np.ndarray:
        """Compute utilities for all agents."""
        return np.array([np.dot(utilities[i], x[i]) for i in range(n_agents)])
    
    initial_defender_welfare = compute_defender_welfare(x_current)
    initial_allocation = x_current.copy()
    
    if verbose:
        print(f"[PGD] Initial defender welfare: {initial_defender_welfare:.6e}")
    
    # Adam optimizer state (initialized for attacker-sized arrays)
    if use_adam:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m_atk = np.zeros((len(attacking_group), n_resources))
        v_atk = np.zeros((len(attacking_group), n_resources))
    
    # Tracking
    objective_history = [initial_defender_welfare]
    converged = False
    final_iteration = 0
    
    # Non-defenders = everyone except defenders (includes attackers)
    non_defending_group = set(range(n_agents)) - defending_group
    attacker_list = sorted(attacking_group)
    
    for iteration in range(max_iterations):
        final_iteration = iteration + 1
        
        # Compute utilities at current point
        V = compute_utilities_arr(x_current)
        
        # Compute ratio matrix v_{i,j} / V_i for all agents
        ratios = np.zeros((n_agents, n_resources))
        for i in range(n_agents):
            for j in range(n_resources):
                ratios[i, j] = utilities[i, j] / max(V[i], 1e-10)
        
        # Compute gradient of defender Nash welfare w.r.t. attacker allocation
        # ∇_{a,j} = -|t_j ∩ D| / |t_j| * r_j
        # where:
        #   t_j = set of non-attackers tied at minimum ratio v_{i,j}/V_i for resource j
        #   D = defending_group
        #   r_j = the minimum ratio value
        
        attacker_gradient = np.zeros((len(attacking_group), n_resources))
        
        for j in range(n_resources):
            # Use the existing function to find tied agents at minimum ratio
            t_j_list, r_j = find_dependent_agents_for_resource(
                j, x_current, ratios, non_attacking_group, find_min=True
            )
            t_j = set(t_j_list)
            
            if debug and j < 3:  # Debug first few resources
                print(f"[PGD]   Resource {j}: t_j={t_j}, r_j={r_j:.6f}")
            
            if not t_j:
                # No non-attackers have meaningful allocation for this resource
                continue
            
            # Count |t_j ∩ D| = how many tied agents are defenders
            t_j_intersect_D = t_j & defending_group
            n_tied_defenders = len(t_j_intersect_D)
            n_tied_total = len(t_j)
            
            if debug and j < 3:
                print(f"[PGD]   Resource {j}: t_j∩D={t_j_intersect_D}, |t_j∩D|={n_tied_defenders}, |t_j|={n_tied_total}")
            
            if n_tied_defenders == 0:
                # No defenders are tied at minimum - gradient is 0 for this resource
                continue
            
            # Compute gradient: ∇_{a,j} = -|t_j ∩ D| / |t_j| * r_j
            # This is the same for ALL attackers (gradient doesn't depend on which attacker)
            grad_j = -(n_tied_defenders / n_tied_total) * r_j
            
            for idx, a in enumerate(attacker_list):
                attacker_gradient[idx, j] = grad_j
        
        if debug:
            print(f"[PGD] Iteration {iteration}:")
            print(f"[PGD]   Defender welfare: {objective_history[-1]:.6e}")
            print(f"[PGD]   Attacker gradient (∇_{a,j} = -|t_j∩D|/|t_j| * r_j):")
            for idx, a in enumerate(attacker_list):
                print(f"[PGD]     Agent {a}: {attacker_gradient[idx].round(6)}")
        
        # Check if any attacker has non-zero gradient
        attacker_has_nonzero_gradient = np.any(np.abs(attacker_gradient) > 1e-10)
        
        if not attacker_has_nonzero_gradient:
            if verbose:
                print(f"[PGD] Iteration {iteration}: Gradient is zero for all attackers - no defenders tied at minimum ratio")
        
        # Apply optimizer update to attacker allocations only
        if use_adam:
            m_atk = beta1 * m_atk + (1 - beta1) * attacker_gradient
            v_atk = beta2 * v_atk + (1 - beta2) * (attacker_gradient ** 2)
            m_hat = m_atk / (1 - beta1 ** (iteration + 1))
            v_hat = v_atk / (1 - beta2 ** (iteration + 1))
            update = step_size * m_hat / (np.sqrt(v_hat) + epsilon)
        else:
            update = step_size * attacker_gradient
        
        # Step 3: Update attacker allocations
        # Gradient descent: x_new = x_old - step * gradient (to minimize)
        # Since gradient is negative (e.g., -0.9), subtracting it increases attacker allocation
        x_attacker_new = np.zeros((len(attacking_group), n_resources))
        for idx, a in enumerate(attacker_list):
            x_attacker_new[idx] = x_current[a] - update[idx]  # SUBTRACT for gradient descent
            # Ensure non-negative
            x_attacker_new[idx] = np.maximum(x_attacker_new[idx], 0)
            # Ensure doesn't exceed supply
            x_attacker_new[idx] = np.minimum(x_attacker_new[idx], supply)
        
        if debug:
            print(f"[PGD]   New attacker allocations:")
            for idx, a in enumerate(attacker_list):
                print(f"[PGD]     Agent {a}: {x_attacker_new[idx].round(6)}")
        
        # Step 4: Recompute non-attacker allocations given new attacker allocations
        # Remaining supply after attackers
        attacker_total = np.sum(x_attacker_new, axis=0)
        remaining_supply = supply - attacker_total
        remaining_supply = np.maximum(remaining_supply, 0)
        
        if debug:
            print(f"[PGD]   Remaining supply for non-attackers: {remaining_supply.round(6)}")
        
        # Solve Nash welfare for non-attackers with remaining supply
        non_attacker_alloc = _solve_nash_welfare_for_subset(
            utilities, non_attacking_group, remaining_supply, n_agents, n_resources
        )
        
        # Build full allocation
        x_new = np.zeros((n_agents, n_resources))
        for idx, a in enumerate(attacker_list):
            x_new[a] = x_attacker_new[idx]
        for i in non_attacking_group:
            x_new[i] = non_attacker_alloc[i]
        
        # Step 5: Project back to boundary using DirectionalDerivativeOracle
        x_new_flat = x_new.flatten()
        dir_violated, _, _ = dir_deriv_oracle.is_violated(x_new_flat)
        
        if dir_violated:
            # Oracle found boundary point
            if dir_deriv_oracle.last_boundary_point is not None:
                x_projected = dir_deriv_oracle.last_boundary_point.reshape((n_agents, n_resources))
                if debug:
                    print(f"[PGD]   Projected back to boundary")
            else:
                x_projected = x_new
                if debug:
                    print(f"[PGD]   WARNING: Oracle violated but no boundary point returned")
        else:
            # Already on boundary
            x_projected = x_new
            if debug:
                print(f"[PGD]   Already on boundary (no projection needed)")
        
        # Sanity check: verify swap optimality
        swap_violated, _, _ = swap_oracle.is_violated(x_projected.flatten())
        if swap_violated:
            print(f"[PGD WARNING] Iteration {iteration}: Swap optimality violated after projection!")
        
        # Print allocations before computing new defender welfare (only in debug mode)
        if debug:
            print(f"[PGD] Iteration {iteration} - Allocations after projection:")
            for i in range(n_agents):
                if i in attacking_group:
                    role = "ATK"
                elif i in defending_group:
                    role = "DEF"
                else:
                    role = "OTH"
                print(f"[PGD]   Agent {i} ({role}): {x_projected[i].round(6)}")
        
        # Compute new objective (for tracking/debugging)
        new_defender_welfare = compute_defender_welfare(x_projected)
        objective_history.append(new_defender_welfare)
        
        # Warning if we moved in the wrong direction (welfare increased instead of decreased)
        # Only warn if the increase is significant (> 0.1% relative increase)
        improvement = objective_history[-2] - objective_history[-1]
        relative_increase = -improvement / max(abs(objective_history[-2]), 1e-10)
        if relative_increase > 1e-3:  # More than 0.1% increase
            print(f"[PGD WARNING] Iteration {iteration}: Moved in WRONG direction! Defender welfare INCREASED.")
            print(f"[PGD WARNING]   Previous welfare: {objective_history[-2]:.6e}")
            print(f"[PGD WARNING]   New welfare: {objective_history[-1]:.6e}")
            print(f"[PGD WARNING]   Relative increase: {relative_increase:.4f} ({relative_increase*100:.2f}%)")
        
        # Check convergence based on defender welfare change (the objective we're minimizing)
        if len(objective_history) >= 2:
            prev_welfare = objective_history[-2]
            curr_welfare = objective_history[-1]
            welfare_change = abs(curr_welfare - prev_welfare)
            relative_welfare_change = welfare_change / max(abs(prev_welfare), 1e-10)
        else:
            relative_welfare_change = float('inf')
        
        if debug:
            print(f"[PGD]   New defender welfare: {new_defender_welfare:.6e}")
            print(f"[PGD]   Welfare change: {welfare_change:.6e} (relative: {relative_welfare_change:.6e})")
        
        if relative_welfare_change < convergence_tol:
            converged = True
            if verbose:
                print(f"[PGD] Converged at iteration {iteration} (relative welfare change {relative_welfare_change:.6e} < {convergence_tol})")
            x_current = x_projected
            break
        
        x_current = x_projected
    
    if not converged and verbose:
        print(f"[PGD] Did not converge after {max_iterations} iterations")
    
    # Cleanup step: redistribute items that non-attackers don't value
    # For each non-attacker, if they have allocation of an item they don't value (v_{i,j} = 0),
    # free that allocation and redistribute via Nash welfare to non-defenders (attackers + neutrals)
    
    # Step 1: Identify freed supply from non-attackers holding items they don't value
    freed_supply = np.zeros(n_resources)
    locked_utility = {}  # c_i for each non-attacker (utility from items they DO value)
    
    # First compute locked utility for all non-attackers
    for i in non_attacking_group:
        locked_utility[i] = 0.0
        for j in range(n_resources):
            locked_utility[i] += utilities[i, j] * x_current[i, j]
    
    # Free allocations from all non-attackers (defenders + neutrals) for items they don't value
    for i in non_attacking_group:
        for j in range(n_resources):
            if utilities[i, j] < 1e-10 and x_current[i, j] > 1e-10:
                # Non-attacker i has allocation of item j but doesn't value it
                freed_supply[j] += x_current[i, j]
                # Update locked utility to exclude this freed allocation
                locked_utility[i] -= utilities[i, j] * x_current[i, j]  # This is 0 anyway since v_{i,j} = 0
                x_current[i, j] = 0.0
    
    if np.sum(freed_supply) > 1e-10:
        if verbose:
            print(f"[PGD] Cleanup: freed supply from unvalued items: {freed_supply.round(6)}")
        
        # Solve Nash welfare for non-defenders (attackers + neutrals) with freed supply
        # Objective: maximize sum_i log(c_i + v_i · x_i) where c_i is locked utility
        # This is equivalent to Nash welfare with a constant offset
        
        # non_defending_group = attackers + neutrals
        non_defenders = list(non_defending_group)
        
        # Compute locked utility for non-defenders
        for i in non_defending_group:
            if i not in locked_utility:
                locked_utility[i] = 0.0
                for j in range(n_resources):
                    locked_utility[i] += utilities[i, j] * x_current[i, j]
        
        try:
            from cvxopt import matrix, solvers
            solvers.options['show_progress'] = False
            
            n_non_defenders = len(non_defenders)
            n_vars = n_non_defenders * n_resources
            
            def var_idx(local_i, j):
                return local_i * n_resources + j
            
            def F(x=None, z=None):
                if x is None:
                    # Initial point: split freed supply equally
                    x0 = matrix(0.0, (n_vars, 1))
                    for local_i, agent in enumerate(non_defenders):
                        for j in range(n_resources):
                            if freed_supply[j] > 1e-10:
                                x0[var_idx(local_i, j)] = freed_supply[j] / n_non_defenders
                    return (0, x0)
                
                # Compute utilities: V_i = c_i + sum_j v_{i,j} * x_{i,j}
                V = []
                for local_i, agent in enumerate(non_defenders):
                    v_i = locked_utility[agent]
                    for j in range(n_resources):
                        v_i += utilities[agent, j] * x[var_idx(local_i, j)]
                    V.append(max(v_i, 1e-12))
                
                # Objective: -sum_i log(V_i)
                f = sum(-np.log(v) for v in V)
                
                # Gradient
                Df = matrix(0.0, (1, n_vars))
                for local_i, agent in enumerate(non_defenders):
                    for j in range(n_resources):
                        Df[var_idx(local_i, j)] = -utilities[agent, j] / V[local_i]
                
                if z is None:
                    return (f, Df)
                
                # Hessian
                H = matrix(0.0, (n_vars, n_vars))
                for local_i, agent in enumerate(non_defenders):
                    for j1 in range(n_resources):
                        for j2 in range(n_resources):
                            idx1 = var_idx(local_i, j1)
                            idx2 = var_idx(local_i, j2)
                            H[idx1, idx2] = z[0] * utilities[agent, j1] * utilities[agent, j2] / (V[local_i] ** 2)
                
                return (f, Df, H)
            
            # Constraints: x >= 0
            G = matrix(-np.eye(n_vars))
            h = matrix(np.zeros(n_vars))
            
            # Constraints: sum_i x_{i,j} = freed_supply[j] (equality - use all freed supply)
            A = matrix(0.0, (n_resources, n_vars))
            b = matrix(freed_supply)
            for j in range(n_resources):
                for local_i in range(n_non_defenders):
                    A[j, var_idx(local_i, j)] = 1.0
            
            sol = solvers.cp(F, G, h, A=A, b=b)
            
            if sol['status'] == 'optimal':
                x_sol = np.array(sol['x']).flatten()
                
                # Add the new allocations to x_current
                for local_i, agent in enumerate(non_defenders):
                    for j in range(n_resources):
                        x_current[agent, j] += max(0, x_sol[var_idx(local_i, j)])
                
                if verbose:
                    print(f"[PGD] Cleanup: redistributed freed supply via Nash welfare to non-defenders")
            else:
                if verbose:
                    print(f"[PGD] Cleanup: Nash welfare optimization failed, giving freed supply to highest-ratio non-defenders")
                # Fallback: give freed supply to non-defender with highest ratio for each resource
                V_current = compute_utilities_arr(x_current)
                for j in range(n_resources):
                    if freed_supply[j] > 1e-10:
                        best_agent = None
                        best_ratio = -1
                        for i in non_defending_group:
                            ratio = utilities[i, j] / max(V_current[i], 1e-10)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_agent = i
                        if best_agent is not None:
                            x_current[best_agent, j] += freed_supply[j]
        
        except Exception as e:
            if verbose:
                print(f"[PGD] Cleanup: optimization failed ({e}), giving freed supply to highest-ratio non-defenders")
            # Fallback: give freed supply to non-defender with highest ratio for each resource
            V_current = compute_utilities_arr(x_current)
            for j in range(n_resources):
                if freed_supply[j] > 1e-10:
                    best_agent = None
                    best_ratio = -1
                    for i in non_defending_group:
                        ratio = utilities[i, j] / max(V_current[i], 1e-10)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_agent = i
                    if best_agent is not None:
                        x_current[best_agent, j] += freed_supply[j]
    
    final_defender_welfare = compute_defender_welfare(x_current)
    final_allocation = x_current.copy()
    final_utilities = compute_utilities_arr(x_current)
    
    if verbose:
        print(f"[PGD] Final defender welfare: {final_defender_welfare:.6e}")
        print(f"[PGD] Welfare reduction: {initial_defender_welfare / max(final_defender_welfare, 1e-20):.4f}x")
    
    # Create ProportionalityConstraints from final attacker allocation
    optimal_constraints = {}
    for a in attacking_group:
        attacker_alloc = final_allocation[a]
        if np.linalg.norm(attacker_alloc) > 1e-10:
            optimal_constraints[a] = ProportionalityConstraint(n_resources, attacker_alloc)
            if debug:
                print(f"[PGD] Created constraint for agent {a}: {attacker_alloc.round(6)}")
    
    # Compute metrics
    initial_utilities_arr = compute_utilities_arr(initial_allocation)
    
    # q-NEE for defenders
    q_nee_individual = {}
    for d in defending_group:
        if initial_utilities_arr[d] > 1e-10:
            q_nee_individual[d] = final_utilities[d] / initial_utilities_arr[d]
        else:
            q_nee_individual[d] = float('inf') if final_utilities[d] > 1e-10 else 1.0
    
    if defending_group:
        k = len(defending_group)
        log_sum = sum(np.log(max(q_nee_individual[d], 1e-20)) for d in defending_group)
        q_nee_group = np.exp(log_sum / k)
    else:
        q_nee_group = 1.0
    
    info = {
        'converged': converged,
        'iterations': final_iteration,
        'initial_defender_welfare': initial_defender_welfare,
        'final_defender_welfare': final_defender_welfare,
        'initial_allocation': initial_allocation,
        'final_allocation': final_allocation,
        'initial_utilities': initial_utilities_arr,
        'final_utilities': final_utilities,
        'objective_history': objective_history,
        'q_nee_group': q_nee_group,
        'q_nee_individual': q_nee_individual,
        'attacking_group': attacking_group,
        'defending_group': defending_group,
    }
    
    return optimal_constraints if optimal_constraints else None, info


# =============================================================================
# Simulation Framework
# =============================================================================

ConstraintGenerator = Callable[[AllocationResult, int], Optional[AgentDiversityConstraint]]


@dataclass
class SimulationConfig:
    """Configuration for a simulation iteration."""
    n_agents: int
    n_resources: int
    utilities: np.ndarray
    supply: Optional[np.ndarray] = None
    initial_constraints: Dict[int, AgentDiversityConstraint] = field(default_factory=dict)
    additional_constraints_static: Dict[int, AgentDiversityConstraint] = field(default_factory=dict)
    additional_constraints_dynamic: Dict[int, ConstraintGenerator] = field(default_factory=dict)
    agent_names: Optional[List[str]] = None
    
    def get_issuer_ids(self) -> Set[int]:
        return set(self.additional_constraints_static.keys()) | set(self.additional_constraints_dynamic.keys())


@dataclass 
class SimulationResult:
    """Complete results from one simulation iteration."""
    config: SimulationConfig
    initial_result: AllocationResult
    final_result: AllocationResult
    metrics: SimulationMetrics
    generated_constraints: Dict[int, AgentDiversityConstraint]
    
    def summary(self) -> str:
        return self.metrics.summary(self.config.agent_names)


class AllocationSimulator:
    """Runs resource allocation simulations with metrics tracking."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def run(self, verbose: bool = False) -> SimulationResult:
        optimizer = NashWelfareOptimizer(
            n_agents=self.config.n_agents,
            n_resources=self.config.n_resources,
            utilities=self.config.utilities,
            supply=self.config.supply
        )
        
        for agent_id, constraint in self.config.initial_constraints.items():
            optimizer.set_agent_constraint(agent_id, constraint)
        
        if verbose:
            print("Solving initial problem...")
        initial_result = optimizer.solve(verbose=verbose)
        
        generated_constraints = {}
        for agent_id, generator in self.config.additional_constraints_dynamic.items():
            constraint = generator(initial_result, agent_id)
            if constraint is not None:
                generated_constraints[agent_id] = constraint
        
        for agent_id, constraint in self.config.additional_constraints_static.items():
            optimizer.add_agent_constraint(agent_id, constraint)
        
        for agent_id, constraint in generated_constraints.items():
            optimizer.add_agent_constraint(agent_id, constraint)
        
        if verbose:
            print("\nSolving with additional constraints...")
        final_result = optimizer.solve(verbose=verbose)
        
        issuer_ids = self.config.get_issuer_ids()
        metrics = compute_metrics(
            initial_result=initial_result,
            final_result=final_result,
            issuer_ids=issuer_ids,
            n_agents=self.config.n_agents
        )
        
        return SimulationResult(
            config=self.config,
            initial_result=initial_result,
            final_result=final_result,
            metrics=metrics,
            generated_constraints=generated_constraints
        )


# =============================================================================
# Batch Simulation & DataFrame Export
# =============================================================================

@dataclass
class BatchResults:
    """Results from batch simulation runs."""
    results: List[SimulationResult]
    n_agents: int
    
    @property
    def n_runs(self) -> int:
        return len(self.results)
    
    @property
    def welfare_loss_ratios(self) -> np.ndarray:
        return np.array([r.metrics.welfare_loss_ratio for r in self.results])
    
    @property
    def p_mon_groups(self) -> np.ndarray:
        return np.array([r.metrics.p_mon_group for r in self.results])
    
    @property
    def q_nee_groups(self) -> np.ndarray:
        return np.array([r.metrics.q_nee_group for r in self.results])
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"BATCH SUMMARY ({self.n_runs} runs)",
            "=" * 60,
            "",
            "Welfare Loss Ratio (final/initial):",
            f"  Mean: {np.mean(self.welfare_loss_ratios):.6f}",
            f"  Std:  {np.std(self.welfare_loss_ratios):.6f}",
            f"  Min:  {np.min(self.welfare_loss_ratios):.6f}",
            f"  Max:  {np.max(self.welfare_loss_ratios):.6f}",
            "",
            "p-MON (<1 = issuers benefited):",
            f"  Mean: {np.mean(self.p_mon_groups):.6f}",
            f"  Std:  {np.std(self.p_mon_groups):.6f}",
            f"  Fraction < 1 (benefited): {np.mean(self.p_mon_groups < 1):.1%}",
            "",
            "q-NEE (<1 = non-issuers harmed):",
            f"  Mean: {np.mean(self.q_nee_groups):.6f}",
            f"  Std:  {np.std(self.q_nee_groups):.6f}",
            f"  Fraction < 1 (harmed): {np.mean(self.q_nee_groups < 1):.1%}",
        ]
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame (one row per run)."""
        rows = [result.metrics.to_dict(run_id=i) for i, result in enumerate(self.results)]
        df = pd.DataFrame(rows)
        
        # Reorder columns for readability
        priority_cols = [
            'run_id', 
            'initial_nash_welfare', 
            'final_nash_welfare', 
            'welfare_loss_ratio',
            'p_mon_group', 
            'q_nee_group',
            'n_issuers',
            'n_non_issuers',
            'issuer_ids',
            'non_issuer_ids'
        ]
        other_cols = [c for c in df.columns if c not in priority_cols]
        df = df[[c for c in priority_cols if c in df.columns] + other_cols]
        
        return df
    
    def to_detailed_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per (run, agent) pair."""
        rows = []
        for run_id, result in enumerate(self.results):
            metrics = result.metrics
            n_resources = result.config.n_resources
            
            for agent_id in range(self.n_agents):
                is_issuer = agent_id in metrics.issuer_ids
                
                # Get preference and allocation vectors
                preference_vector = result.config.utilities[agent_id]
                initial_allocation = result.initial_result.allocation[agent_id]
                final_allocation = result.final_result.allocation[agent_id]
                
                row = {
                    'run_id': run_id,
                    'agent_id': agent_id,
                    'is_issuer': is_issuer,
                    'initial_utility': metrics.initial_utilities[agent_id],
                    'final_utility': metrics.final_utilities[agent_id],
                    'utility_ratio': metrics.final_utilities[agent_id] / max(metrics.initial_utilities[agent_id], 1e-10),
                    'welfare_loss_ratio': metrics.welfare_loss_ratio,
                    'p_mon_group': metrics.p_mon_group,
                    'q_nee_group': metrics.q_nee_group,
                    'p_mon_individual': metrics.p_mon_individual.get(agent_id, None) if is_issuer else None,
                    'q_nee_individual': metrics.q_nee_individual.get(agent_id, None) if not is_issuer else None,
                }
                
                # Add preference vector columns
                for j in range(n_resources):
                    row[f'preference_{j}'] = preference_vector[j]
                
                # Add initial allocation columns
                for j in range(n_resources):
                    row[f'initial_allocation_{j}'] = initial_allocation[j]
                
                # Add final allocation columns
                for j in range(n_resources):
                    row[f'final_allocation_{j}'] = final_allocation[j]
                
                rows.append(row)
        
        return pd.DataFrame(rows)


def run_batch_simulations(
    configs: List[SimulationConfig],
    verbose: bool = False
) -> BatchResults:
    """Run multiple simulations and collect results."""
    results = []
    n_agents = configs[0].n_agents if configs else 0
    
    for i, config in enumerate(configs):
        if verbose:
            print(f"\n--- Simulation {i+1}/{len(configs)} ---")
        
        simulator = AllocationSimulator(config)
        result = simulator.run(verbose=verbose)
        results.append(result)
    
    return BatchResults(results=results, n_agents=n_agents)


# =============================================================================
# Utility Functions
# =============================================================================

def generate_normalized_utilities(n_agents: int, n_resources: int, 
                                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a random utility matrix where each agent's preferences sum to 1.
    
    Args:
        n_agents: Number of agents
        n_resources: Number of resources
        rng: NumPy random generator (optional, uses default if not provided)
    
    Returns:
        Utility matrix of shape (n_agents, n_resources) where each row sums to 1
    """
    if rng is None:
        rng = np.random.default_rng()
    
    utilities = rng.random((n_agents, n_resources)) + 0.1
    utilities = utilities / utilities.sum(axis=1, keepdims=True)
    return utilities


# =============================================================================
# Example Usage
# =============================================================================

def example_single_simulation():
    """Example of a single simulation run."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Single Simulation")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_agents = 4
    n_resources = 6
    type_a = np.array([True, True, True, False, False, False])
    type_b = ~type_a
    
    # Generate random utilities and normalize each row to sum to 1
    utilities = np.random.rand(n_agents, n_resources) + 0.1
    utilities = utilities / utilities.sum(axis=1, keepdims=True)
    
    config = SimulationConfig(
        n_agents=n_agents,
        n_resources=n_resources,
        utilities=utilities,
        agent_names=["Alice", "Bob", "Carol", "Dave"],
        initial_constraints={
            0: CategoryBalanceConstraint(n_resources, {'A': type_a}, {'A': 0.2}),
        },
        additional_constraints_static={
            1: MaxCategoryConstraint(n_resources, type_a, max_fraction=0.4),
            2: CategoryBalanceConstraint(n_resources, {'B': type_b}, {'B': 0.5}),
        }
    )
    
    simulator = AllocationSimulator(config)
    result = simulator.run(verbose=True)
    
    print("\n" + result.summary())
    
    return result


def example_dynamic_constraints():
    """Example with dynamically generated constraints."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Dynamic Constraints")
    print("=" * 60)
    
    np.random.seed(123)
    
    n_agents = 3
    n_resources = 4
    
    # Generate random utilities and normalize each row to sum to 1
    utilities = np.random.rand(n_agents, n_resources) + 0.1
    utilities = utilities / utilities.sum(axis=1, keepdims=True)
    
    def cap_highest_resource(initial_result: AllocationResult, agent_id: int) -> AgentDiversityConstraint:
        """Generate constraint that caps the resource agent got most of."""
        allocation = initial_result.allocation[agent_id]
        highest_resource = np.argmax(allocation)
        
        a = np.ones(n_resources) * (-0.3)
        a[highest_resource] += 1
        
        print(f"  Agent {agent_id} generating constraint: cap resource {highest_resource}")
        return LinearConstraint(n_resources, a, 0.0)
    
    config = SimulationConfig(
        n_agents=n_agents,
        n_resources=n_resources,
        utilities=utilities,
        agent_names=["Agent0", "Agent1", "Agent2"],
        additional_constraints_dynamic={
            1: cap_highest_resource,
        }
    )
    
    simulator = AllocationSimulator(config)
    result = simulator.run(verbose=True)
    
    print("\n" + result.summary())
    
    return result


def example_batch_with_csv():
    """Example batch simulation with CSV export."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Batch Simulation with CSV Export")
    print("=" * 60)
    
    n_agents = 4
    n_resources = 6
    type_a = np.array([True, True, True, False, False, False])
    
    configs = []
    for seed in range(20):
        np.random.seed(seed)
        
        # Generate random utilities and normalize each row to sum to 1
        utilities = np.random.rand(n_agents, n_resources) + 0.1
        utilities = utilities / utilities.sum(axis=1, keepdims=True)
        
        issuer = seed % n_agents
        
        config = SimulationConfig(
            n_agents=n_agents,
            n_resources=n_resources,
            utilities=utilities,
            additional_constraints_static={
                issuer: MaxCategoryConstraint(n_resources, type_a, max_fraction=0.5),
            }
        )
        configs.append(config)
    
    # Run batch
    batch_results = run_batch_simulations(configs, verbose=False)
    
    # Print summary
    print(batch_results.summary())
    
    # Get DataFrames
    df = batch_results.to_dataframe()
    df_detailed = batch_results.to_detailed_dataframe()
    
    # Export to CSV using pandas native method
    df.to_csv("simulation_results.csv", index=False)
    df_detailed.to_csv("simulation_results_detailed.csv", index=False)
    
    print(f"\nExported {len(df)} runs to simulation_results.csv")
    print(f"Exported {len(df_detailed)} agent-level rows to simulation_results_detailed.csv")
    
    # Show some pandas analysis examples
    print("\n--- Example Analysis with pandas ---")
    print("\nSummary statistics:")
    print(df[['welfare_loss_ratio', 'p_mon_group', 'q_nee_group']].describe())
    
    print("\nMean metrics by number of issuers:")
    print(df.groupby('n_issuers')[['welfare_loss_ratio', 'p_mon_group', 'q_nee_group']].mean())
    
    print("\nMean utility ratio by issuer status:")
    print(df_detailed.groupby('is_issuer')['utility_ratio'].mean())
    
    return batch_results


if __name__ == "__main__":
    example_single_simulation()
    example_dynamic_constraints()
    example_batch_with_csv()