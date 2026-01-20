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
ALLOCATION_THRESHOLD = 1e-6

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
    
    At the boundary, D(x) = 0. The gradient with respect to x_aj gives us the normal.
    
    For the simple case (ignoring second-order terms from V_d depending on x):
        ∂D/∂x_aj ≈ (v_aj/V_a - v_dj/V_d)
    
    The normal pointing toward the feasible region (where D >= 0) is this gradient.
    The normal pointing toward the excluded zone is the negative.
    
    When multiple agents are tied for dependent, we average their contributions.
    
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
    
    # Use shared helper to find dependent ratios (forward direction = find_min=True)
    dependent_ratios, _ = compute_dependent_ratios_for_all_resources(
        x_full_boundary, ratios, non_attacking_group, n_resources, find_min=True
    )
    
    # Compute attacker ratios (average over attackers if multiple)
    attacker_ratios = np.zeros(n_resources)
    for j in range(n_resources):
        total = 0.0
        count = 0
        for a in attacking_group:
            V_a = max(V_boundary[a], 1e-10)
            total += utilities[a, j] / V_a
            count += 1
        attacker_ratios[j] = total / count if count > 0 else 0.0
    
    # The gradient of D with respect to attacker allocation x_aj is approximately:
    # ∂D/∂x_aj ≈ (v_aj/V_a - v_dj/V_d)
    # This points toward increasing D (toward feasible region)
    # Normal toward excluded zone is the negative
    gradient = attacker_ratios - dependent_ratios
    
    # Normal points toward excluded zone (where D < 0, i.e., where attackers want more)
    normal_attacker = gradient  # Positive gradient means attacker wants more of this resource
    
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
        
        G_list.append(-np.eye(self.n_vars))
        h_list.append(np.zeros(self.n_vars))
        
        G_list.append(np.eye(self.n_vars))
        h_list.append(np.ones(self.n_vars))
        
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
                 verbose: bool = False):
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.utilities = utilities
        self.attacking_group = attacking_group
        self.non_attacking_group = set(range(n_agents)) - attacking_group
        self.supply = supply if supply is not None else np.ones(n_resources)
        self.verbose = verbose
        self.n_free_vars = n_agents * n_resources
        
        # Counters for debugging
        self.n_backward_checks = 0
        self.n_backward_violations = 0
        
        # Store the last computed boundary point (set by _find_separating_hyperplane)
        self.last_boundary_point = None
    
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
        if self.verbose:
            print(f"  [DirDerivOracle.is_violated] Checking directional derivative...")
        
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
            if self.verbose:
                print(f"    Non-attacker has zero utility - this means we've definitely overshot")
                print(f"    (taking from them would be -∞ change, giving to them would be +∞ benefit)")
                print(f"    Proceeding directly to binary search...")
            
            # We've definitely overshot - go directly to binary search
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
                if self.verbose:
                    print(f"    -> NOT VIOLATED (forward dir_deriv >= 0)")
                return False, None, None
            
            # Forward derivative is negative - we've overshot
            if self.verbose:
                print(f"    -> VIOLATED (forward dir_deriv < 0), finding separating hyperplane...")
            
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
                if self.verbose:
                    print(f"    Backward direction also invalid, returning not violated")
                return False, None, None
            
            backward_deriv = self._compute_directional_derivative_with_direction(x_full, V, backward_direction)
            
            if self.verbose:
                print(f"    Backward directional derivative: {backward_deriv:.6e}")
            
            if backward_deriv <= 1e-8:
                # Backward derivative is non-positive, meaning going backward wouldn't help
                # We haven't overshot
                if self.verbose:
                    print(f"    -> NOT VIOLATED (backward dir_deriv <= 0, we haven't overshot)")
                return False, None, None
            
            # Backward derivative is positive - going backward would help
            # This means we've overshot in the forward direction
            # Use the same binary search - as we scale down, forward direction becomes valid
            self.n_backward_violations += 1
            
            if self.verbose:
                print(f"    -> VIOLATED (backward dir_deriv > 0, we've overshot)")
                print(f"    Finding separating hyperplane using binary search...")
            
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
            
            # Solve Nash welfare for non-attackers with remaining supply
            non_attacker_allocation = self._solve_nash_for_non_attackers(remaining_supply)
            
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
        
        non_attacker_allocation_boundary = self._solve_nash_for_non_attackers(remaining_supply_boundary)
        
        x_full_boundary = np.zeros((self.n_agents, self.n_resources))
        for i in self.attacking_group:
            x_full_boundary[i] = attacker_allocation_boundary[i]
        if non_attacker_allocation_boundary is not None:
            for i in self.non_attacking_group:
                x_full_boundary[i] = non_attacker_allocation_boundary[i]
        
        # Store the boundary point for later retrieval
        self.last_boundary_point = x_full_boundary.flatten().copy()
        
        V_boundary = self._compute_utilities_from_allocation(x_full_boundary)
        
        # Count how many non-attackers share each resource (have allocation)
        agents_per_resource = np.zeros(self.n_resources)
        for j in range(self.n_resources):
            for d in self.non_attacking_group:
                if x_full_boundary[d, j] > 1e-10:
                    agents_per_resource[j] += 1
        
        # Compute v_ij/V_i for all agents at x*
        ratios = np.zeros((self.n_agents, self.n_resources))
        for i in range(self.n_agents):
            V_i = max(V_boundary[i], 1e-10)
            for j in range(self.n_resources):
                ratios[i, j] = self.utilities[i, j] / V_i
        
        # For each resource j, find the dependent agent (non-attacker with smallest v_dj/V_d who has allocation)
        dependent_agent = [None] * self.n_resources
        for j in range(self.n_resources):
            min_ratio = float('inf')
            min_agent = None
            for d in self.non_attacking_group:
                if x_full_boundary[d, j] > 1e-10:  # Must have allocation
                    if ratios[d, j] < min_ratio:
                        min_ratio = ratios[d, j]
                        min_agent = d
            dependent_agent[j] = min_agent
        
        # Compute total attacker allocation for each resource: S_a(k)
        S_a = np.zeros(self.n_resources)
        for k in range(self.n_resources):
            for a in self.attacking_group:
                S_a[k] += x_full_boundary[a, k]
        
        # Compute the gradient for each resource j using corrected formula
        # ∂D/∂x_aj = -[(v_dj / V_d) - (v_dj / (V_d² * agents_per_resource[j])) * Σ_{k: d_k = d_j} S_a(k) * v_dk]
        # Normal points toward excluded zone, so we negate: normal = -gradient
        normal_full = np.zeros((self.n_agents, self.n_resources))
        
        for j in range(self.n_resources):
            d_j = dependent_agent[j]
            if d_j is None:
                continue
            
            V_d = V_boundary[d_j]
            v_dj = self.utilities[d_j, j]
            
            # Compute the sum: Σ_{k: d_k = d_j} S_a(k) * v_dk
            sum_term = 0.0
            for k in range(self.n_resources):
                if dependent_agent[k] == d_j:
                    v_dk = self.utilities[d_j, k]
                    sum_term += S_a[k] * v_dk
            
            # Compute gradient component
            if agents_per_resource[j] > 0:
                gradient_j = -((v_dj / V_d) + (v_dj / (V_d**2 * agents_per_resource[j])) * sum_term)
            else:
                gradient_j = -(v_dj / V_d)
            
            # Normal is negative of gradient (points toward excluded zone)
            # Set for all attackers (they share the same normal for resource j)
            for a in self.attacking_group:
                normal_full[a, j] = -gradient_j
        
        normal = normal_full.flatten()
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        else:
            # Fallback: use direction from x* to x_violating directly
            if saved_verbose:
                print(f"      WARNING: Normal is zero, falling back to direction vector")
            x_violating_full = self._get_full_allocation(x_free)
            direction = x_violating_full - x_full_boundary
            normal = direction.flatten()
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
        
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
            print(f"      Boundary point found:")
            print(f"        V at boundary: {V_boundary.round(6)}")
            print(f"        Dependent agents: {dependent_agent}")
            print(f"        Agents per resource: {agents_per_resource}")
            print(f"        S_a (attacker allocation): {S_a.round(6)}")
            print(f"        Normal (attacker components):")
            for a in self.attacking_group:
                print(f"          Agent {a}: {normal_full[a].round(6)}")
            print(f"        rhs = normal · x* = {rhs:.6f}")
            print(f"        Violating point: normal · x_free = {violating_dot:.6f}")
            violation = violating_dot - rhs
            print(f"        Violation (lhs - rhs): {violation:.6f} (should be > 0)")
            print(f"    ╔══════════════════════════════════════════════════════════════════╗")
            print(f"    ║          BINARY SEARCH FOR BOUNDARY - END                        ║")
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
                    # Require improvement >= 0.001 and rounded ratios to differ
                    if ratio_j_rounded > ratio_i_rounded and improvement >= 0.001:
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
    use_projection: bool = True,
    use_two_phase: bool = True
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
    
    Returns:
        Tuple of:
        - optimal_directions: Dict mapping attacker agent_id -> ratio vector, or None
        - final_allocation: The resulting allocation matrix
        - status: 'optimal', 'infeasible', or error message
        - cp_debug_info: Dict with debug info about the convex program
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
    dir_deriv_oracle = DirectionalDerivativeOracle(n_agents, n_resources, utilities, attacking_group, supply, verbose=debug)
    
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
        'inner_solves': n_inner_solves
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
    debug: bool = False
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
        print(f"[DEBUG] Reconstructed allocation matches: {np.allclose(x_full, initial_allocation)}")
        print(f"[DEBUG] Reconstructed utilities: {V.round(6)}")
        
        # Show v_{i,j}/V_i for each agent to check KKT conditions
        print(f"[DEBUG] Checking KKT conditions (v_ij/V_i for each agent):")
        for i in range(n_agents):
            ratios = utilities[i] / V[i]
            print(f"[DEBUG]   Agent {i}: {ratios.round(6)}")
        
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
        debug=debug
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
    
    # Compute directional derivative for convex program allocation
    cp_x_free = convex_program_allocation.flatten()
    
    cp_x_full = boundary_oracle._get_full_allocation(cp_x_free)
    cp_V = boundary_oracle._compute_utilities_from_allocation(cp_x_full)
    cp_directional_deriv = boundary_oracle._compute_directional_derivative(cp_x_full, cp_V)
    cp_on_boundary, _, _ = boundary_oracle.is_violated(cp_x_free)
    cp_on_boundary = not cp_on_boundary
    
    if debug:
        print(f"[DEBUG] Convex program directional derivative: {cp_directional_deriv:.6e}")
        print(f"[DEBUG] Convex program on optimal boundary: {cp_on_boundary}")
    
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
        
        # Compute directional derivative for final allocation
        final_x_free = final_allocation.flatten()
        
        final_x_full = boundary_oracle._get_full_allocation(final_x_free)
        final_V = boundary_oracle._compute_utilities_from_allocation(final_x_full)
        final_directional_deriv = boundary_oracle._compute_directional_derivative(final_x_full, final_V)
        final_on_boundary, _, _ = boundary_oracle.is_violated(final_x_free)
        final_on_boundary = not final_on_boundary
        
        if debug:
            print(f"[DEBUG] Final allocation (after re-solving with constraints):\n{final_allocation.round(6)}")
            print(f"[DEBUG] Final utilities: {final_utilities.round(6)}")
            print(f"[DEBUG] Final Nash welfare: {final_result.nash_welfare:.6f}")
            print(f"[DEBUG] Final directional derivative: {final_directional_deriv:.6e}")
            print(f"[DEBUG] Final on optimal boundary: {final_on_boundary}")
            # Compare convex program vs final allocation
            print(f"[DEBUG] === COMPARISON: Convex Program vs Final ===")
            for agent_id in attacking_group:
                cp_util = convex_program_utilities[agent_id]
                final_util = final_utilities[agent_id]
                print(f"[DEBUG] Agent {agent_id}: CP utility={cp_util:.6f}, Final utility={final_util:.6f}, diff={final_util - cp_util:.6f}")
            print(f"[DEBUG] Allocation difference norm: {np.linalg.norm(final_allocation - convex_program_allocation):.6e}")
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