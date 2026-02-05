"""
Folktables-based simulation for p-MON and q-NEE experiments.

This simulation uses real US Census ACS data to create utility matrices where:
- Agents = Job categories (occupations), sampled with replacement weighted by popularity
- Resources = Demographic buckets (unique combinations of age, sex, race/ethnicity)

Job Sampling:
- Jobs are sampled with replacement, weighted by their popularity (total workers)
- A fixed number of jobs (default: 20) are sampled as agents
- The same job may appear multiple times, resulting in agents with identical utility vectors

Utility calculation:
- For each job-demographic pair, compute the representation ratio:
  ratio = (% of job that is demographic) / (% of total workforce that is demographic)
- Combine ratios across dimensions (multiplicative or average mode)
- Multiply by demographic bucket count
- Normalize each job's utility row to sum to 1

Demographic dimensions:
- Age: 6 buckets (16-24, 25-34, 35-44, 45-54, 55-64, 65+)
- Sex: 2 categories (Male, Female)
- Race/Ethnicity: 10 categories (5 race × 2 hispanic)

Total possible demographic buckets: 6 × 2 × 10 = 120
"""

import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import from nash_welfare_optimizer
from nash_welfare_optimizer import (
    NashWelfareOptimizer,
    compute_optimal_self_benefit_constraint,
    solve_optimal_harm_constraint_pgd,
)


# =============================================================================
# Constants for demographic bucketing
# =============================================================================

# Age buckets: (min_age, max_age, label)
AGE_BUCKETS = [
    (16, 24, '16-24'),
    (25, 34, '25-34'),
    (35, 44, '35-44'),
    (45, 54, '45-54'),
    (55, 64, '55-64'),
    (65, 99, '65+'),
]

# Sex categories (from ACS SEX variable)
SEX_CATEGORIES = {
    1: 'Male',
    2: 'Female',
}

# Race categories (from ACS RAC1P variable)
# We consolidate into 5 categories
RACE_MAPPING = {
    1: 'White',
    2: 'Black',
    3: 'AIAN',  # American Indian alone
    4: 'AIAN',  # Alaska Native alone
    5: 'AIAN',  # American Indian and Alaska Native tribes specified
    6: 'Asian',
    7: 'Other',  # Native Hawaiian and Other Pacific Islander
    8: 'Other',  # Some Other Race
    9: 'Other',  # Two or More Races
}

RACE_CATEGORIES = ['White', 'Black', 'Asian', 'AIAN', 'Other']

# Hispanic categories (from ACS HISP variable)
# HISP = 01 means Not Hispanic, anything else is Hispanic
HISPANIC_CATEGORIES = ['Non-Hispanic', 'Hispanic']


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_folktables_data(
    years: List[int] = None,
    states: List[str] = None,
    data_folder: str = 'data',
    download: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load ACS PUMS data using folktables, with caching to CSV.
    
    Loads data state-by-state to manage memory, filters immediately,
    and saves to filtered_cache folder. Incrementally updates cache if some states are missing.
    
    Args:
        years: List of years to load (default: most recent year)
        states: List of state codes to load (default: all US states + DC + PR)
        data_folder: Folder for data storage (default: 'data')
        download: Whether to download data if not cached
        verbose: Print progress information
    
    Returns:
        Tuple of (DataFrame, List[str]) where:
        - DataFrame has columns: OCCP, AGEP, SEX, RAC1P, HISP, PWGTP, year, state
        - List[str] contains any states that failed to load (for logging)
    """
    import gc
    
    # All 52 US state/territory codes
    ALL_STATES = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC', 'PR'  # Include DC and Puerto Rico
    ]
    
    # Default to most recent year
    if years is None:
        years = [2022]
    
    # Use all states if not specified
    if states is None:
        states = ALL_STATES.copy()
    
    # Create folders if they don't exist
    os.makedirs(data_folder, exist_ok=True)
    filtered_cache_folder = os.path.join(data_folder, 'filtered_cache')
    os.makedirs(filtered_cache_folder, exist_ok=True)
    
    # Generate cache filename based on years (always "all_states" for the cache)
    years_str = "_".join(str(y) for y in sorted(years))
    cache_filename = os.path.join(filtered_cache_folder, f"acs_filtered_{years_str}_all_states.csv")
    
    columns_needed = ['OCCP', 'AGEP', 'SEX', 'RAC1P', 'HISP', 'PWGTP']
    failed_states = []
    
    # Check if cache exists and what states are in it
    cached_df = None
    cached_states = set()
    
    if os.path.exists(cache_filename):
        if verbose:
            print(f"Loading existing cache from {cache_filename}...")
        cached_df = pd.read_csv(cache_filename)
        
        # Get states already in cache (for each year)
        for year in years:
            year_data = cached_df[cached_df['year'] == year]
            cached_states.update(year_data['state'].unique())
        
        if verbose:
            print(f"  Found {len(cached_df):,} records")
            print(f"  States in cache: {len(cached_states)}/52")
    
    # Determine which states need to be loaded
    states_to_load = set(ALL_STATES) - cached_states
    
    if not states_to_load:
        if verbose:
            print("  All 52 states already in cache.")
        # Filter to requested states if needed
        if set(states) != set(ALL_STATES):
            cached_df = cached_df[cached_df['state'].isin(states)]
        return cached_df, failed_states
    
    if verbose:
        print(f"  Missing states: {len(states_to_load)} - {sorted(states_to_load)}")
        print("Loading missing states from folktables...")
    
    try:
        from folktables import ACSDataSource
    except ImportError:
        raise ImportError(
            "folktables package not installed. "
            "Install with: pip install folktables"
        )
    
    new_data = []
    total_to_load = len(states_to_load) * len(years)
    processed = 0
    
    for year in years:
        if verbose:
            print(f"\nProcessing year {year}...")
        
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person',
            root_dir=data_folder
        )
        
        for state in sorted(states_to_load):
            processed += 1
            if verbose:
                print(f"  [{processed}/{total_to_load}] Loading {state} {year}...", end=" ")
            
            try:
                # Load single state - this is the key memory optimization
                raw_data = data_source.get_data(states=[state], download=download)
                
                # Check columns exist
                available_columns = [c for c in columns_needed if c in raw_data.columns]
                if len(available_columns) < len(columns_needed):
                    if verbose:
                        missing = set(columns_needed) - set(available_columns)
                        print(f"SKIP (missing {missing})")
                    failed_states.append(state)
                    del raw_data
                    gc.collect()
                    continue
                
                # Filter to needed columns immediately
                filtered = raw_data[columns_needed].copy()
                
                # Delete raw data immediately to free memory
                del raw_data
                gc.collect()
                
                # Add metadata
                filtered['year'] = year
                filtered['state'] = state
                
                # Filter to employed workers 16+
                filtered = filtered[filtered['OCCP'].notna() & (filtered['OCCP'] > 0)]
                filtered = filtered[filtered['AGEP'] >= 16]
                
                if verbose:
                    print(f"{len(filtered):,} records")
                
                new_data.append(filtered)
                
                # Periodic memory cleanup
                if processed % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
                failed_states.append(state)
                continue
    
    # Combine new data with cached data
    if new_data:
        if verbose:
            print(f"\nCombining {len(new_data)} newly loaded state/year datasets...")
        new_df = pd.concat(new_data, ignore_index=True)
        del new_data
        gc.collect()
        
        if cached_df is not None:
            if verbose:
                print(f"Merging with existing cache ({len(cached_df):,} records)...")
            combined_data = pd.concat([cached_df, new_df], ignore_index=True)
            del cached_df
            del new_df
            gc.collect()
        else:
            combined_data = new_df
        
        if verbose:
            print(f"Total records: {len(combined_data):,}")
        
        # Save updated cache
        if verbose:
            print(f"Saving updated cache to: {cache_filename}...")
        combined_data.to_csv(cache_filename, index=False)
        if verbose:
            print(f"  Cache saved ({os.path.getsize(cache_filename) / 1024 / 1024:.1f} MB)")
            
            # Report final state count
            final_states = combined_data['state'].unique()
            print(f"  States in updated cache: {len(final_states)}/52")
    else:
        combined_data = cached_df
        if verbose:
            print("No new data loaded (all attempted states failed)")
    
    if combined_data is None or len(combined_data) == 0:
        raise ValueError("No data could be loaded for any state/year")
    
    # Log failed states
    if failed_states and verbose:
        print(f"\nWarning: Failed to load {len(failed_states)} states: {sorted(failed_states)}")
    
    # Filter to requested states if needed
    if set(states) != set(ALL_STATES):
        combined_data = combined_data[combined_data['state'].isin(states)]
    
    return combined_data, failed_states


def get_age_bucket(age: int) -> str:
    """Convert age to bucket label."""
    for min_age, max_age, label in AGE_BUCKETS:
        if min_age <= age <= max_age:
            return label
    return '65+'  # Default for ages > 99


def get_race_category(rac1p: int) -> str:
    """Convert RAC1P code to race category."""
    return RACE_MAPPING.get(int(rac1p), 'Other')


def get_hispanic_category(hisp: int) -> str:
    """Convert HISP code to Hispanic category."""
    # HISP = 1 means "Not Spanish/Hispanic/Latino"
    return 'Non-Hispanic' if int(hisp) == 1 else 'Hispanic'


def get_race_ethnicity(rac1p: int, hisp: int) -> str:
    """Get combined race/ethnicity category."""
    race = get_race_category(rac1p)
    hispanic = get_hispanic_category(hisp)
    return f"{race}_{hispanic}"


def create_demographic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add demographic bucket columns to dataframe."""
    df = df.copy()
    df['age_bucket'] = df['AGEP'].apply(get_age_bucket)
    df['sex'] = df['SEX'].map(SEX_CATEGORIES)
    df['race_ethnicity'] = df.apply(
        lambda row: get_race_ethnicity(row['RAC1P'], row['HISP']), 
        axis=1
    )
    return df


# =============================================================================
# Utility Matrix Construction
# =============================================================================

def compute_workforce_demographics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute the demographic distribution of the total workforce.
    
    Returns:
        Dictionary with keys 'age', 'sex', 'race_ethnicity', each containing
        a dict mapping category -> proportion of workforce
    """
    total_weight = df['PWGTP'].sum()
    
    demographics = {}
    
    # Age distribution
    age_dist = df.groupby('age_bucket')['PWGTP'].sum() / total_weight
    demographics['age'] = age_dist.to_dict()
    
    # Sex distribution
    sex_dist = df.groupby('sex')['PWGTP'].sum() / total_weight
    demographics['sex'] = sex_dist.to_dict()
    
    # Race/Ethnicity distribution
    race_eth_dist = df.groupby('race_ethnicity')['PWGTP'].sum() / total_weight
    demographics['race_ethnicity'] = race_eth_dist.to_dict()
    
    return demographics


def compute_job_demographics(
    df: pd.DataFrame, 
    min_job_size: int = 100
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[int, float]]:
    """
    Compute demographic distribution for each job.
    
    Args:
        df: DataFrame with demographic columns
        min_job_size: Minimum weighted count for a job to be included
    
    Returns:
        (job_demographics, job_weights)
        - job_demographics: Dict mapping job_code -> {dimension -> {category -> proportion}}
        - job_weights: Dict mapping job_code -> total weighted count (popularity)
    """
    job_demographics = {}
    job_weights = {}
    
    # Get unique occupations
    job_codes = df['OCCP'].unique()
    
    for job_code in job_codes:
        job_df = df[df['OCCP'] == job_code]
        job_weight = job_df['PWGTP'].sum()
        
        if job_weight < min_job_size:
            continue
        
        job_weights[job_code] = job_weight
        
        job_demo = {}
        
        # Age distribution within job
        age_dist = job_df.groupby('age_bucket')['PWGTP'].sum() / job_weight
        job_demo['age'] = age_dist.to_dict()
        
        # Sex distribution within job
        sex_dist = job_df.groupby('sex')['PWGTP'].sum() / job_weight
        job_demo['sex'] = sex_dist.to_dict()
        
        # Race/Ethnicity distribution within job
        race_eth_dist = job_df.groupby('race_ethnicity')['PWGTP'].sum() / job_weight
        job_demo['race_ethnicity'] = race_eth_dist.to_dict()
        
        job_demographics[job_code] = job_demo
    
    return job_demographics, job_weights


def sample_jobs(
    job_weights: Dict[int, float],
    n_jobs: int = 20,
    seed: int = 42
) -> List[int]:
    """
    Sample jobs with replacement, weighted by popularity.
    
    Args:
        job_weights: Dict mapping job_code -> total weighted count (popularity)
        n_jobs: Number of jobs to sample (default: 20)
        seed: Random seed for sampling
    
    Returns:
        List of job codes (may contain duplicates)
    """
    np.random.seed(seed + 20000)  # Different seed offset for job sampling
    
    job_codes = list(job_weights.keys())
    weights = np.array([job_weights[j] for j in job_codes])
    weights = weights / weights.sum()  # Normalize to probabilities
    
    # Sample with replacement
    sampled_indices = np.random.choice(
        len(job_codes),
        size=n_jobs,
        replace=True,
        p=weights
    )
    
    sampled_jobs = [job_codes[i] for i in sampled_indices]
    
    return sampled_jobs


def compute_representation_ratios(
    job_demographics: Dict[str, Dict[str, float]],
    workforce_demographics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute representation ratios for a single job.
    
    ratio = (% of job that is category) / (% of workforce that is category)
    
    Returns:
        Dict mapping dimension -> {category -> ratio}
    """
    ratios = {}
    
    for dimension in ['age', 'sex', 'race_ethnicity']:
        ratios[dimension] = {}
        workforce_dist = workforce_demographics[dimension]
        job_dist = job_demographics.get(dimension, {})
        
        for category, workforce_prop in workforce_dist.items():
            job_prop = job_dist.get(category, 0.0)
            
            # Avoid division by zero
            if workforce_prop > 1e-10:
                ratio = job_prop / workforce_prop
            else:
                ratio = 1.0  # Default to neutral if category not in workforce
            
            ratios[dimension][category] = ratio
    
    return ratios


def create_demographic_buckets(
    df: pd.DataFrame, 
    sample_size: int = 10000, 
    seed: int = 42,
    demographic_factors: List[str] = None
) -> pd.DataFrame:
    """
    Sample individuals and create demographic buckets based on selected factors.
    
    The number of buckets depends on which factors are enabled:
    - ['sex']: 2 buckets (Male, Female)
    - ['age']: 6 buckets (age groups)
    - ['race']: up to 20 buckets (race × ethnicity)
    - ['age', 'sex']: 12 buckets
    - ['age', 'sex', 'race']: up to 120 buckets
    
    Args:
        df: DataFrame with demographic columns
        sample_size: Number of individuals to sample
        seed: Random seed for sampling
        demographic_factors: List of factors to use: 'age', 'sex', 'race'
                            Default None means all three ['age', 'sex', 'race']
    
    Returns:
        DataFrame with demographic buckets that have count > 0
        Columns depend on factors: always has 'count', plus columns for each factor
    """
    # Default to all factors
    if demographic_factors is None:
        demographic_factors = ['age', 'sex', 'race']
    
    np.random.seed(seed)
    
    # Sample individuals (weighted sampling)
    weights = df['PWGTP'] / df['PWGTP'].sum()
    sample_indices = np.random.choice(
        len(df), 
        size=min(sample_size, len(df)), 
        replace=False,
        p=weights
    )
    sampled = df.iloc[sample_indices].copy()
    
    # Determine which columns to group by based on enabled factors
    groupby_cols = []
    if 'age' in demographic_factors:
        groupby_cols.append('age_bucket')
    if 'sex' in demographic_factors:
        groupby_cols.append('sex')
    if 'race' in demographic_factors:
        groupby_cols.append('race_ethnicity')
    
    if not groupby_cols:
        # Edge case: no factors selected, create single bucket with all individuals
        return pd.DataFrame({'count': [len(sampled)]})
    
    # Count sampled individuals in each bucket
    bucket_counts = sampled.groupby(groupby_cols).size().reset_index(name='count')
    
    # Filter to only buckets with count > 0 (remove empty demographics)
    bucket_counts = bucket_counts[bucket_counts['count'] > 0].reset_index(drop=True)
    
    return bucket_counts


def compute_utility_matrix(
    sampled_job_codes: List[int],
    job_demographics: Dict[int, Dict[str, Dict[str, float]]],
    workforce_demographics: Dict[str, Dict[str, float]],
    demographic_buckets: pd.DataFrame,
    combination_mode: str = 'multiplicative'
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Compute the utility matrix and supply vector.
    
    For each job and demographic bucket:
        utility = I_counts[demographic] * age_ratio * sex_ratio * race_ratio
    Then normalize each job's utility row to sum to 1.
    
    The I_counts multiplication effectively bakes the demographic counts into
    the utility, so supply is simply n_agents for all items.
    
    Args:
        sampled_job_codes: List of sampled job codes (agents) - may contain duplicates
        job_demographics: Demographics for each unique job
        workforce_demographics: Overall workforce demographics
        demographic_buckets: DataFrame with demographic buckets and counts (I_counts)
        combination_mode: 'multiplicative' or 'average'
    
    Returns:
        (utilities, supply, sampled_job_codes, bucket_labels)
        - utilities: Matrix of shape (n_jobs, n_buckets), each row normalized to sum to 1
        - supply: Vector of shape (n_buckets,), all values = n_jobs (number of agents)
        - sampled_job_codes: List of job codes (preserves order and duplicates)
        - bucket_labels: List of bucket labels
    
    Note: If the same job code appears multiple times in sampled_job_codes,
    those agents will have identical utility vectors.
    """
    n_jobs = len(sampled_job_codes)
    n_buckets = len(demographic_buckets)
    
    # Create bucket labels and extract I_counts
    bucket_labels = []
    I_counts = np.zeros(n_buckets)
    for j, (_, row) in enumerate(demographic_buckets.iterrows()):
        label = f"{row['age_bucket']}_{row['sex']}_{row['race_ethnicity']}"
        bucket_labels.append(label)
        I_counts[j] = row['count']
    
    # Supply is n_agents for all items (I_counts is baked into utility)
    supply = np.ones(n_buckets) * n_jobs
    
    # Pre-compute ratios for each unique job to avoid redundant computation
    unique_jobs = set(sampled_job_codes)
    job_ratios_cache = {}
    for job_code in unique_jobs:
        job_demo = job_demographics[job_code]
        job_ratios_cache[job_code] = compute_representation_ratios(job_demo, workforce_demographics)
    
    # Initialize utility matrix
    utilities = np.zeros((n_jobs, n_buckets))
    
    for i, job_code in enumerate(sampled_job_codes):
        ratios = job_ratios_cache[job_code]
        
        for j, (_, bucket_row) in enumerate(demographic_buckets.iterrows()):
            age_ratio = ratios['age'].get(bucket_row['age_bucket'], 1.0)
            sex_ratio = ratios['sex'].get(bucket_row['sex'], 1.0)
            race_eth_ratio = ratios['race_ethnicity'].get(bucket_row['race_ethnicity'], 1.0)
            
            # Get the count for this demographic bucket
            bucket_count = I_counts[j]
            
            # Compute utility: I_counts * age_ratio * sex_ratio * race_ratio
            if combination_mode == 'multiplicative':
                utilities[i, j] = bucket_count * age_ratio * sex_ratio * race_eth_ratio
            elif combination_mode == 'average':
                # For average mode, still multiply by count but average the ratios
                avg_ratio = (age_ratio + sex_ratio + race_eth_ratio) / 3
                utilities[i, j] = bucket_count * avg_ratio
            else:
                raise ValueError(f"Unknown combination mode: {combination_mode}")
    
    # Normalize each row to sum to 1
    row_sums = utilities.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
    utilities = utilities / row_sums
    
    return utilities, supply, sampled_job_codes, bucket_labels


# =============================================================================
# Simulation Data Class
# =============================================================================

@dataclass
class FolktablesSimulationResult:
    """Result from a single folktables-based simulation."""
    # Configuration
    n_jobs: int
    n_buckets: int
    attacker_size: int
    defender_size: int
    seed: int
    combination_mode: str
    sample_size: int
    min_job_size: int
    
    # Groups
    attacking_group: Set[int]
    defending_group: Set[int]
    
    # p-MON optimization results
    pmon_converged: bool
    pmon_time_seconds: float
    pmon_n_iterations: int
    p_mon_attacking_group: float
    q_nee_defending_group: float
    q_nee_non_attackers: float
    p_mon_individual: Dict[int, float]
    q_nee_individual: Dict[int, float]
    V_initial: np.ndarray
    V_pmon_final: np.ndarray
    nash_welfare_initial: float
    nash_welfare_pmon_final: float
    
    # Worst-case q-NEE optimization results
    qnee_worst_converged: bool
    qnee_worst_time_seconds: float
    qnee_worst_iterations: int
    q_nee_worst_defending_group: float
    q_nee_worst_individual: Dict[int, float]
    p_mon_under_qnee_worst: float
    p_mon_under_qnee_worst_individual: Dict[int, float]
    V_qnee_worst_final: np.ndarray
    nash_welfare_qnee_worst_final: float
    defender_welfare_initial: float
    defender_welfare_qnee_worst_final: float
    
    # Warning tracking
    pmon_had_warnings: bool = False
    qnee_had_warnings: bool = False
    pmon_warning_types: List[str] = field(default_factory=list)
    qnee_warning_types: List[str] = field(default_factory=list)
    
    # Job info (optional)
    job_codes: List[int] = field(default_factory=list)
    attacker_job_codes: List[int] = field(default_factory=list)
    defender_job_codes: List[int] = field(default_factory=list)
    
    # HHI and Gini metrics
    hhi_individual: Dict[int, float] = field(default_factory=dict)
    gini_individual: Dict[int, float] = field(default_factory=dict)
    attacker_mean_hhi: float = 0.0
    attacker_mean_gini: float = 0.0
    defender_mean_hhi: float = 0.0
    defender_mean_gini: float = 0.0
    
    # Similarity metrics
    att_def_sim_mean: float = 0.0
    att_def_sim_min: float = 0.0
    att_def_sim_max: float = 0.0
    att_att_sim_mean: float = 0.0
    att_att_sim_min: float = 0.0
    att_att_sim_max: float = 0.0
    def_def_sim_mean: float = 0.0
    def_def_sim_min: float = 0.0
    def_def_sim_max: float = 0.0
    
    # Sampling info
    n_unique_jobs: int = 0
    n_buckets_found: int = 0
    
    # States that failed to load
    failed_states: List[str] = field(default_factory=list)


@dataclass 
class PrecomputedSimulation:
    """Lightweight precomputed data for a single simulation (memory efficient)."""
    # Job ratios: list of dicts, one per agent (job)
    # Each dict: {'age': {bucket: ratio}, 'sex': {cat: ratio}, 'race_ethnicity': {cat: ratio}}
    job_ratios: List[Dict[str, Dict[str, float]]]
    job_codes: List[int]
    
    # Demographic bucket info
    I_counts: np.ndarray  # Count per demographic bucket
    bucket_labels: List[str]  # Labels for each bucket
    bucket_info: List[Dict[str, str]]  # [{age_bucket, sex, race_ethnicity}, ...]
    
    # Group assignments
    attacking_group: Set[int]
    defending_group: Set[int]
    
    # Config info
    attacker_size: int
    defender_size: int
    seed: int
    n_unique_jobs: int
    n_buckets_found: int
    
    # States that failed to load (passed through from CensusData)
    failed_states: List[str] = field(default_factory=list)


# =============================================================================
# Simulation Functions
# =============================================================================

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


def select_groups(
    n_agents: int, 
    attacker_size: int, 
    defender_size: int, 
    seed: int
) -> Tuple[Set[int], Set[int]]:
    """Randomly select attacking and defending groups."""
    np.random.seed(seed + 10000)
    all_agents = list(range(n_agents))
    np.random.shuffle(all_agents)
    
    attacking_group = set(all_agents[:attacker_size])
    defending_group = set(all_agents[attacker_size:attacker_size + defender_size])
    
    return attacking_group, defending_group


def run_folktables_simulation(
    precomputed: PrecomputedSimulation,
    combination_mode: str,
    sample_size: int,
    min_job_size: int,
    demographic_factors: List[str] = None,
    verbose: bool = False,
    debug: bool = False,
    pgd_max_iterations: int = 100,
    pgd_step_size: float = 0.1,
) -> FolktablesSimulationResult:
    """
    Run a single simulation with precomputed data.
    
    Args:
        precomputed: PrecomputedSimulation containing ratios, I_counts, and groups
        combination_mode: How demographics were combined
        sample_size: Sample size used for buckets
        min_job_size: Minimum job size filter used
        demographic_factors: List of factors to use: 'age', 'sex', 'race'
                            Default None means all three ['age', 'sex', 'race']
        verbose: Print verbose output
        debug: Print debug output
        pgd_max_iterations: Max iterations for PGD
        pgd_step_size: Step size for PGD
    
    Returns:
        FolktablesSimulationResult
    """
    start_time = time.perf_counter()
    
    # Compute utility matrix from precomputed ratios
    utilities = compute_utility_matrix_from_ratios(
        job_ratios=precomputed.job_ratios,
        I_counts=precomputed.I_counts,
        bucket_info=precomputed.bucket_info,
        combination_mode=combination_mode,
        demographic_factors=demographic_factors,
    )
    
    n_agents, n_resources = utilities.shape
    supply = np.ones(n_resources) * n_agents  # Supply = n_agents for all items
    
    job_codes = precomputed.job_codes
    attacking_group = precomputed.attacking_group
    defending_group = precomputed.defending_group
    non_attacking_group = set(range(n_agents)) - attacking_group
    
    if verbose:
        print(f"  Attackers: {attacking_group}")
        print(f"  Defenders: {defending_group}")
    
    # Compute HHI and Gini on-the-fly
    hhi_values, gini_values = compute_hhi_gini_from_ratios(
        job_ratios=precomputed.job_ratios,
        utilities=utilities,
        I_counts=precomputed.I_counts,
        bucket_info=precomputed.bucket_info,
        combination_mode=combination_mode,
        demographic_factors=demographic_factors,
    )
    
    hhi_individual = {i: hhi_values[i] for i in range(n_agents)}
    gini_individual = {i: gini_values[i] for i in range(n_agents)}
    
    attacker_mean_hhi = np.mean([hhi_values[i] for i in attacking_group])
    attacker_mean_gini = np.mean([gini_values[i] for i in attacking_group])
    defender_mean_hhi = np.mean([hhi_values[i] for i in defending_group]) if defending_group else 0.0
    defender_mean_gini = np.mean([gini_values[i] for i in defending_group]) if defending_group else 0.0
    
    # Compute similarity metrics
    similarity_metrics = compute_similarity_metrics(utilities, attacking_group, defending_group)
    
    if verbose:
        print(f"  HHI: att={attacker_mean_hhi:.4f}, def={defender_mean_hhi:.4f}")
        print(f"  Gini: att={attacker_mean_gini:.4f}, def={defender_mean_gini:.4f}")
        print(f"  Similarity att-def: mean={similarity_metrics['att_def_sim_mean']:.4f}")
    
    # Warning tracking
    pmon_warnings = []
    qnee_warnings = []
    
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
        )
        
        pmon_warnings = info.get('warnings', [])
        V_initial = info['initial_utilities']
        V_pmon_final = info['final_utilities']
        pmon_converged = info.get('status') == 'optimal' or info.get('status') == 'no_benefit'
        pmon_n_iterations = info.get('cp_iterations', 0)
        
    except Exception as e:
        if verbose:
            print(f"  [p-MON] ERROR: {e}")
            import traceback
            traceback.print_exc()
        optimizer = NashWelfareOptimizer(n_agents, n_resources, utilities, supply)
        initial_result = optimizer.solve(verbose=False)
        V_initial = initial_result.agent_utilities
        V_pmon_final = V_initial.copy()
        pmon_converged = False
        pmon_n_iterations = 0
        pmon_warnings.append(f"Exception:{str(e)}")
    
    pmon_time = time.perf_counter() - pmon_start
    
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
    defender_welfare_initial = float(np.prod([V_initial[d] for d in defending_group]))
    
    if verbose:
        print(f"  [p-MON] p-MON attacking group: {p_mon_attacking_group:.6f}")
        print(f"  [p-MON] q-NEE defending group: {q_nee_defending_group:.6f}")
        print(f"  [p-MON] Time: {pmon_time:.2f}s")
    
    # =========================================================================
    # Part 2: Worst-case q-NEE optimization
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
        
        qnee_warnings = qnee_info.get('warnings', [])
        V_qnee_worst_final = qnee_info['final_utilities']
        qnee_worst_converged = qnee_info['converged']
        qnee_worst_iterations = qnee_info['iterations']
        defender_welfare_qnee_worst_final = qnee_info['final_defender_welfare']
        
    except Exception as e:
        if verbose:
            print(f"  [q-NEE worst] ERROR: {e}")
            import traceback
            traceback.print_exc()
        V_qnee_worst_final = V_initial.copy()
        qnee_worst_converged = False
        qnee_worst_iterations = 0
        defender_welfare_qnee_worst_final = defender_welfare_initial
        qnee_warnings.append(f"Exception:{str(e)}")
    
    qnee_time = time.perf_counter() - qnee_start
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
    
    # p-MON for attackers under worst-case constraint
    p_mon_under_qnee_worst_individual = {}
    for i in attacking_group:
        if V_qnee_worst_final[i] > 1e-10:
            p_mon_under_qnee_worst_individual[i] = V_initial[i] / V_qnee_worst_final[i]
        else:
            p_mon_under_qnee_worst_individual[i] = float('inf')
    
    p_mon_under_qnee_worst = geometric_mean(list(p_mon_under_qnee_worst_individual.values()))
    
    if verbose:
        print(f"  [q-NEE worst] q-NEE defending group: {q_nee_worst_defending_group:.6f}")
        print(f"  [q-NEE worst] Time: {qnee_time:.2f}s")
    
    # Get job codes for attackers/defenders
    attacker_job_codes = [job_codes[i] for i in sorted(attacking_group)]
    defender_job_codes = [job_codes[i] for i in sorted(defending_group)]
    
    return FolktablesSimulationResult(
        n_jobs=n_agents,
        n_buckets=n_resources,
        attacker_size=precomputed.attacker_size,
        defender_size=precomputed.defender_size,
        seed=precomputed.seed,
        combination_mode=combination_mode,
        sample_size=sample_size,
        min_job_size=min_job_size,
        attacking_group=attacking_group,
        defending_group=defending_group,
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
        pmon_had_warnings=len(pmon_warnings) > 0,
        qnee_had_warnings=len(qnee_warnings) > 0,
        pmon_warning_types=pmon_warnings,
        qnee_warning_types=qnee_warnings,
        job_codes=job_codes,
        attacker_job_codes=attacker_job_codes,
        defender_job_codes=defender_job_codes,
        hhi_individual=hhi_individual,
        gini_individual=gini_individual,
        attacker_mean_hhi=attacker_mean_hhi,
        attacker_mean_gini=attacker_mean_gini,
        defender_mean_hhi=defender_mean_hhi,
        defender_mean_gini=defender_mean_gini,
        att_def_sim_mean=similarity_metrics['att_def_sim_mean'],
        att_def_sim_min=similarity_metrics['att_def_sim_min'],
        att_def_sim_max=similarity_metrics['att_def_sim_max'],
        att_att_sim_mean=similarity_metrics['att_att_sim_mean'],
        att_att_sim_min=similarity_metrics['att_att_sim_min'],
        att_att_sim_max=similarity_metrics['att_att_sim_max'],
        def_def_sim_mean=similarity_metrics['def_def_sim_mean'],
        def_def_sim_min=similarity_metrics['def_def_sim_min'],
        def_def_sim_max=similarity_metrics['def_def_sim_max'],
        n_unique_jobs=precomputed.n_unique_jobs,
        n_buckets_found=precomputed.n_buckets_found,
        failed_states=precomputed.failed_states,
    )


# =============================================================================
# Results Export Functions
# =============================================================================

def results_to_rows(results: List[FolktablesSimulationResult]) -> List[Dict]:
    """Convert simulation results to spreadsheet rows."""
    rows = []
    
    for r in results:
        attacking_group_str = ";".join(str(x) for x in sorted(r.attacking_group))
        defending_group_str = ";".join(str(x) for x in sorted(r.defending_group))
        pmon_warning_types_str = ";".join(r.pmon_warning_types) if r.pmon_warning_types else ""
        qnee_warning_types_str = ";".join(r.qnee_warning_types) if r.qnee_warning_types else ""
        
        # Format individual HHI/Gini values (semicolon-separated, in agent order)
        hhi_individual_str = ";".join(f"{r.hhi_individual.get(i, 0.0):.6f}" for i in range(r.n_jobs))
        gini_individual_str = ";".join(f"{r.gini_individual.get(i, 0.0):.6f}" for i in range(r.n_jobs))
        
        row = {
            'n_jobs': r.n_jobs,
            'n_buckets': r.n_buckets,
            'n_unique_jobs': r.n_unique_jobs,
            'n_buckets_found': r.n_buckets_found,
            'attacker_size': r.attacker_size,
            'defender_size': r.defender_size,
            'seed': r.seed,
            'combination_mode': r.combination_mode,
            'sample_size': r.sample_size,
            'min_job_size': r.min_job_size,
            'attacking_group': attacking_group_str,
            'defending_group': defending_group_str,
            
            # HHI and Gini metrics
            'attacker_mean_hhi': r.attacker_mean_hhi,
            'attacker_mean_gini': r.attacker_mean_gini,
            'defender_mean_hhi': r.defender_mean_hhi,
            'defender_mean_gini': r.defender_mean_gini,
            'hhi_individual': hhi_individual_str,
            'gini_individual': gini_individual_str,
            
            # Similarity metrics
            'att_def_sim_mean': r.att_def_sim_mean,
            'att_def_sim_min': r.att_def_sim_min,
            'att_def_sim_max': r.att_def_sim_max,
            'att_att_sim_mean': r.att_att_sim_mean,
            'att_att_sim_min': r.att_att_sim_min,
            'att_att_sim_max': r.att_att_sim_max,
            'def_def_sim_mean': r.def_def_sim_mean,
            'def_def_sim_min': r.def_def_sim_min,
            'def_def_sim_max': r.def_def_sim_max,
            
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
            
            # Warning tracking
            'pmon_had_warnings': r.pmon_had_warnings,
            'qnee_had_warnings': r.qnee_had_warnings,
            'pmon_warning_types': pmon_warning_types_str,
            'qnee_warning_types': qnee_warning_types_str,
            
            # Failed states (semicolon-separated)
            'failed_states': ";".join(r.failed_states) if r.failed_states else "",
        }
        
        rows.append(row)
    
    return rows


def save_to_csv(rows: List[Dict], output_folder: str, file_prefix: str, checkpoint_num: int) -> str:
    """Save results to CSV file."""
    if not rows:
        return ""
    
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_folder, f"{file_prefix}_checkpoint{checkpoint_num:03d}_{timestamp}.csv")
    
    columns = list(rows[0].keys())
    
    with open(filename, 'w') as f:
        f.write(','.join(columns) + '\n')
        
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


# =============================================================================
# Main Entry Point
# =============================================================================

def compute_gini(values: np.ndarray, counts: np.ndarray) -> float:
    """
    Compute Gini coefficient for a population where each group has a count and value.
    
    Args:
        values: Array of values (wealth) for each group
        counts: Array of counts (number of individuals) in each group
    
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Expand to full population
    # Sort by value
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Total population and total value
    n = sorted_counts.sum()
    total_value = (sorted_values * sorted_counts).sum()
    
    if n == 0 or total_value == 0:
        return 0.0
    
    # Compute Gini using the formula based on cumulative shares
    # For weighted data: G = 1 - (2/total_value) * sum(cumulative_value * count) / n + 1/n
    cumulative_value = np.cumsum(sorted_values * sorted_counts)
    
    # Gini = 1 - 2 * (area under Lorenz curve)
    # Area under Lorenz curve = sum of (cumulative proportion of population) * (proportion of value in each group)
    cumulative_count = np.cumsum(sorted_counts)
    
    # Using the trapezoid method for Gini
    # G = 1 - sum((cum_count[i] + cum_count[i-1]) * (cum_value[i] - cum_value[i-1])) / (n * total_value)
    cum_count_with_zero = np.concatenate([[0], cumulative_count])
    cum_value_with_zero = np.concatenate([[0], cumulative_value])
    
    # Area under Lorenz curve
    area = 0.0
    for i in range(1, len(cum_count_with_zero)):
        width = (cum_count_with_zero[i] - cum_count_with_zero[i-1]) / n
        height = (cum_value_with_zero[i] + cum_value_with_zero[i-1]) / (2 * total_value)
        area += width * height
    
    gini = 1 - 2 * area
    return max(0.0, min(1.0, gini))  # Clamp to [0, 1]


def compute_hhi(normalized_utility: np.ndarray, counts: np.ndarray) -> float:
    """
    Compute HHI (Herfindahl-Hirschman Index) for an agent's preferences.
    
    HHI = sum over demographics of: I_counts[demographic] * (normalized_utility[demographic])^2
    
    Args:
        normalized_utility: Normalized utility vector (sums to 1)
        counts: I_counts for each demographic
    
    Returns:
        HHI value (higher = more concentrated preferences)
    """
    return (counts * (normalized_utility ** 2)).sum()


def compute_agent_metrics(
    utilities: np.ndarray,
    I_counts: np.ndarray,
    job_demographics: Dict[int, Dict[str, Dict[str, float]]],
    workforce_demographics: Dict[str, Dict[str, float]],
    sampled_job_codes: List[int],
    demographic_buckets: pd.DataFrame,
    combination_mode: str = 'multiplicative'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HHI and Gini coefficient for each agent.
    
    Args:
        utilities: Normalized utility matrix (n_agents x n_demographics)
        I_counts: Count of individuals in each demographic bucket
        job_demographics: Demographics for each unique job
        workforce_demographics: Overall workforce demographics
        sampled_job_codes: List of job codes for each agent
        demographic_buckets: DataFrame with demographic info
        combination_mode: 'multiplicative' or 'average'
    
    Returns:
        (hhi_values, gini_values) - arrays of length n_agents
    """
    n_agents = len(sampled_job_codes)
    hhi_values = np.zeros(n_agents)
    gini_values = np.zeros(n_agents)
    
    # Pre-compute ratios for each unique job
    unique_jobs = set(sampled_job_codes)
    job_ratios_cache = {}
    for job_code in unique_jobs:
        job_demo = job_demographics[job_code]
        job_ratios_cache[job_code] = compute_representation_ratios(job_demo, workforce_demographics)
    
    for i, job_code in enumerate(sampled_job_codes):
        ratios = job_ratios_cache[job_code]
        
        # Compute raw valuations for this agent
        valuations = np.zeros(len(demographic_buckets))
        for j, (_, bucket_row) in enumerate(demographic_buckets.iterrows()):
            age_ratio = ratios['age'].get(bucket_row['age_bucket'], 1.0)
            sex_ratio = ratios['sex'].get(bucket_row['sex'], 1.0)
            race_eth_ratio = ratios['race_ethnicity'].get(bucket_row['race_ethnicity'], 1.0)
            
            if combination_mode == 'multiplicative':
                valuations[j] = age_ratio * sex_ratio * race_eth_ratio
            else:
                valuations[j] = (age_ratio + sex_ratio + race_eth_ratio) / 3
        
        # HHI uses the normalized utility (which already includes I_counts)
        hhi_values[i] = compute_hhi(utilities[i], I_counts)
        
        # Gini uses raw valuations as "wealth" with I_counts as population counts
        gini_values[i] = compute_gini(valuations, I_counts)
    
    return hhi_values, gini_values


@dataclass
class CensusData:
    """Cached census data that can be reused across simulations."""
    df: pd.DataFrame
    workforce_demographics: Dict[str, Dict[str, float]]
    job_demographics: Dict[int, Dict[str, Dict[str, float]]]
    job_weights: Dict[int, float]
    years: List[int]
    states: List[str]
    min_job_size: int
    failed_states: List[str] = field(default_factory=list)


def load_census_data(
    years: List[int] = None,
    states: List[str] = None,
    min_job_size: int = 100,
    data_folder: str = 'data',
    verbose: bool = False,
    download: bool = True,
) -> CensusData:
    """
    Load and prepare census data (cached/reusable across simulations).
    
    This computes the workforce demographics and job demographics from the
    full census data. These don't change between simulations.
    
    Args:
        years: List of years to load
        states: List of states to load
        min_job_size: Minimum workers per job to be eligible for sampling
        data_folder: Folder to cache downloaded data
        verbose: Print progress
        download: Whether to download data
    
    Returns:
        CensusData object with cached data (includes failed_states list)
    """
    # Load data
    if verbose:
        print("Loading folktables data...")
    df, failed_states = load_folktables_data(
        years=years, 
        states=states, 
        data_folder=data_folder,
        download=download, 
        verbose=verbose
    )
    
    # Add demographic columns
    if verbose:
        print("Creating demographic columns...")
    df = create_demographic_columns(df)
    
    # Compute workforce demographics
    if verbose:
        print("Computing workforce demographics...")
    workforce_demographics = compute_workforce_demographics(df)
    
    # Compute job demographics and weights
    if verbose:
        print("Computing job demographics...")
    job_demographics, job_weights = compute_job_demographics(df, min_job_size=min_job_size)
    
    if verbose:
        print(f"Found {len(job_weights)} jobs with >= {min_job_size} workers")
        if failed_states:
            print(f"Warning: {len(failed_states)} states failed to load: {failed_states}")
    
    return CensusData(
        df=df,
        workforce_demographics=workforce_demographics,
        job_demographics=job_demographics,
        job_weights=job_weights,
        years=years,
        states=states,
        min_job_size=min_job_size,
        failed_states=failed_states,
    )


def precompute_simulation_data(
    census_data: CensusData,
    n_jobs: int,
    sample_size: int,
    attacker_size: int,
    defender_size: int,
    seed: int,
    demographic_factors: List[str] = None,
    verbose: bool = False,
) -> PrecomputedSimulation:
    """
    Precompute lightweight data for a simulation.
    
    Extracts only the ratios and counts needed, not the full utility matrix.
    This allows us to free the census data from memory after precomputation.
    
    Args:
        census_data: Full census data
        n_jobs: Number of jobs to sample
        sample_size: Number of individuals to sample
        attacker_size: Number of attackers
        defender_size: Number of defenders
        seed: Random seed
        demographic_factors: List of factors to use: 'age', 'sex', 'race'
                            Default None means all three ['age', 'sex', 'race']
        verbose: Print progress
    
    Returns:
        PrecomputedSimulation with lightweight data
    """
    from collections import Counter
    
    # Default to all factors
    if demographic_factors is None:
        demographic_factors = ['age', 'sex', 'race']
    
    # Sample jobs (with replacement, weighted by popularity)
    sampled_job_codes = sample_jobs(census_data.job_weights, n_jobs=n_jobs, seed=seed)
    
    # Count duplicates for reporting
    job_counts = Counter(sampled_job_codes)
    n_unique = len(job_counts)
    
    # Extract ratios for each sampled job
    job_ratios = []
    for job_code in sampled_job_codes:
        job_demo = census_data.job_demographics[job_code]
        ratios = compute_representation_ratios(job_demo, census_data.workforce_demographics)
        job_ratios.append(ratios)
    
    # Create demographic buckets based on selected factors
    demographic_buckets = create_demographic_buckets(
        census_data.df, 
        sample_size=sample_size, 
        seed=seed,
        demographic_factors=demographic_factors
    )
    
    n_found = len(demographic_buckets)
    
    # Calculate max possible buckets based on factors
    max_possible = 1
    if 'age' in demographic_factors:
        max_possible *= 6
    if 'sex' in demographic_factors:
        max_possible *= 2
    if 'race' in demographic_factors:
        max_possible *= 20  # 10 races × 2 hispanic categories
    
    # Extract bucket info based on which factors are enabled
    bucket_labels = []
    bucket_info = []
    I_counts = demographic_buckets['count'].values.astype(float)
    
    for _, row in demographic_buckets.iterrows():
        # Build label and info based on enabled factors
        label_parts = []
        info = {}
        
        if 'age' in demographic_factors:
            label_parts.append(row['age_bucket'])
            info['age_bucket'] = row['age_bucket']
        if 'sex' in demographic_factors:
            label_parts.append(row['sex'])
            info['sex'] = row['sex']
        if 'race' in demographic_factors:
            label_parts.append(row['race_ethnicity'])
            info['race_ethnicity'] = row['race_ethnicity']
        
        label = "_".join(label_parts) if label_parts else "all"
        bucket_labels.append(label)
        bucket_info.append(info)
    
    # Select attacking and defending groups
    attacking_group, defending_group = select_groups(
        n_jobs, attacker_size, defender_size, seed
    )
    
    if verbose:
        print(f"  Seed {seed}: {n_unique} unique jobs, {n_found}/{max_possible} demographics, "
              f"att={len(attacking_group)}, def={len(defending_group)}")
    
    return PrecomputedSimulation(
        job_ratios=job_ratios,
        job_codes=sampled_job_codes,
        I_counts=I_counts,
        bucket_labels=bucket_labels,
        bucket_info=bucket_info,
        attacking_group=attacking_group,
        defending_group=defending_group,
        attacker_size=attacker_size,
        defender_size=defender_size,
        seed=seed,
        n_unique_jobs=n_unique,
        n_buckets_found=n_found,
        failed_states=census_data.failed_states,
    )


def compute_utility_matrix_from_ratios(
    job_ratios: List[Dict[str, Dict[str, float]]],
    I_counts: np.ndarray,
    bucket_info: List[Dict[str, str]],
    combination_mode: str = 'multiplicative',
    demographic_factors: List[str] = None,
) -> np.ndarray:
    """
    Compute utility matrix from precomputed ratios and I_counts.
    
    Args:
        job_ratios: List of ratio dicts per job
        I_counts: Count per demographic bucket
        bucket_info: List of dicts with keys only for enabled factors
        combination_mode: 'multiplicative' or 'average'
        demographic_factors: List of factors to use: 'age', 'sex', 'race'
                            Default None means all three ['age', 'sex', 'race']
    
    Returns:
        Normalized utility matrix (n_jobs x n_buckets)
    """
    # Default to all factors
    if demographic_factors is None:
        demographic_factors = ['age', 'sex', 'race']
    
    use_age = 'age' in demographic_factors
    use_sex = 'sex' in demographic_factors
    use_race = 'race' in demographic_factors
    
    n_jobs = len(job_ratios)
    n_buckets = len(bucket_info)
    
    utilities = np.zeros((n_jobs, n_buckets))
    
    for i, ratios in enumerate(job_ratios):
        for j, bucket in enumerate(bucket_info):
            # Get ratios for enabled factors (bucket_info only has keys for enabled factors)
            age_ratio = ratios['age'].get(bucket.get('age_bucket', ''), 1.0) if use_age else 1.0
            sex_ratio = ratios['sex'].get(bucket.get('sex', ''), 1.0) if use_sex else 1.0
            race_eth_ratio = ratios['race_ethnicity'].get(bucket.get('race_ethnicity', ''), 1.0) if use_race else 1.0
            
            # Compute utility: I_counts * combined_ratio
            if combination_mode == 'multiplicative':
                utilities[i, j] = I_counts[j] * age_ratio * sex_ratio * race_eth_ratio
            else:  # average
                # Only average over enabled factors
                enabled_ratios = []
                if use_age:
                    enabled_ratios.append(age_ratio)
                if use_sex:
                    enabled_ratios.append(sex_ratio)
                if use_race:
                    enabled_ratios.append(race_eth_ratio)
                avg_ratio = sum(enabled_ratios) / len(enabled_ratios) if enabled_ratios else 1.0
                utilities[i, j] = I_counts[j] * avg_ratio
    
    # Normalize each row to sum to 1
    row_sums = utilities.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    utilities = utilities / row_sums
    
    return utilities


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors using L2 norm."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def compute_similarity_metrics(
    utilities: np.ndarray,
    attacking_group: Set[int],
    defending_group: Set[int],
) -> Dict[str, float]:
    """
    Compute cosine similarity metrics between and within groups.
    
    Args:
        utilities: Utility matrix (n_agents x n_buckets)
        attacking_group: Set of attacker indices
        defending_group: Set of defender indices
    
    Returns:
        Dict with similarity metrics
    """
    attackers = sorted(attacking_group)
    defenders = sorted(defending_group)
    
    metrics = {}
    
    # Attacker-Defender similarities (all pairwise)
    att_def_sims = []
    for att_idx in attackers:
        for def_idx in defenders:
            sim = compute_cosine_similarity(utilities[att_idx], utilities[def_idx])
            att_def_sims.append(sim)
    
    if att_def_sims:
        metrics['att_def_sim_mean'] = np.mean(att_def_sims)
        metrics['att_def_sim_min'] = np.min(att_def_sims)
        metrics['att_def_sim_max'] = np.max(att_def_sims)
    else:
        metrics['att_def_sim_mean'] = float('nan')
        metrics['att_def_sim_min'] = float('nan')
        metrics['att_def_sim_max'] = float('nan')
    
    # Attacker-Attacker similarities (distinct pairs only, skip if only 1 attacker)
    if len(attackers) > 1:
        att_att_sims = []
        for i, att_i in enumerate(attackers):
            for att_j in attackers[i+1:]:
                sim = compute_cosine_similarity(utilities[att_i], utilities[att_j])
                att_att_sims.append(sim)
        
        metrics['att_att_sim_mean'] = np.mean(att_att_sims)
        metrics['att_att_sim_min'] = np.min(att_att_sims)
        metrics['att_att_sim_max'] = np.max(att_att_sims)
    else:
        metrics['att_att_sim_mean'] = float('nan')
        metrics['att_att_sim_min'] = float('nan')
        metrics['att_att_sim_max'] = float('nan')
    
    # Defender-Defender similarities (distinct pairs only, skip if only 1 defender)
    if len(defenders) > 1:
        def_def_sims = []
        for i, def_i in enumerate(defenders):
            for def_j in defenders[i+1:]:
                sim = compute_cosine_similarity(utilities[def_i], utilities[def_j])
                def_def_sims.append(sim)
        
        metrics['def_def_sim_mean'] = np.mean(def_def_sims)
        metrics['def_def_sim_min'] = np.min(def_def_sims)
        metrics['def_def_sim_max'] = np.max(def_def_sims)
    else:
        metrics['def_def_sim_mean'] = float('nan')
        metrics['def_def_sim_min'] = float('nan')
        metrics['def_def_sim_max'] = float('nan')
    
    return metrics


def compute_hhi_gini_from_ratios(
    job_ratios: List[Dict[str, Dict[str, float]]],
    utilities: np.ndarray,
    I_counts: np.ndarray,
    bucket_info: List[Dict[str, str]],
    combination_mode: str = 'multiplicative',
    demographic_factors: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HHI and Gini for each agent from ratios.
    
    Args:
        job_ratios: List of ratio dicts per job
        utilities: Normalized utility matrix
        I_counts: Count per demographic bucket
        bucket_info: List of bucket info dicts
        combination_mode: 'multiplicative' or 'average'
        demographic_factors: List of factors to use: 'age', 'sex', 'race'
                            Default None means all three ['age', 'sex', 'race']
    
    Returns:
        (hhi_values, gini_values) arrays
    """
    # Default to all factors
    if demographic_factors is None:
        demographic_factors = ['age', 'sex', 'race']
    
    use_age = 'age' in demographic_factors
    use_sex = 'sex' in demographic_factors
    use_race = 'race' in demographic_factors
    
    n_jobs = len(job_ratios)
    hhi_values = np.zeros(n_jobs)
    gini_values = np.zeros(n_jobs)
    
    for i, ratios in enumerate(job_ratios):
        # Compute raw valuations for Gini
        valuations = np.zeros(len(bucket_info))
        for j, bucket in enumerate(bucket_info):
            # bucket_info only has keys for enabled factors
            age_ratio = ratios['age'].get(bucket.get('age_bucket', ''), 1.0) if use_age else 1.0
            sex_ratio = ratios['sex'].get(bucket.get('sex', ''), 1.0) if use_sex else 1.0
            race_eth_ratio = ratios['race_ethnicity'].get(bucket.get('race_ethnicity', ''), 1.0) if use_race else 1.0
            
            if combination_mode == 'multiplicative':
                valuations[j] = age_ratio * sex_ratio * race_eth_ratio
            else:
                # Only average over enabled factors
                enabled_ratios = []
                if use_age:
                    enabled_ratios.append(age_ratio)
                if use_sex:
                    enabled_ratios.append(sex_ratio)
                if use_race:
                    enabled_ratios.append(race_eth_ratio)
                valuations[j] = sum(enabled_ratios) / len(enabled_ratios) if enabled_ratios else 1.0
        
        # HHI uses normalized utility
        hhi_values[i] = compute_hhi(utilities[i], I_counts)
        
        # Gini uses raw valuations as wealth with I_counts as population
        gini_values[i] = compute_gini(valuations, I_counts)
    
    return hhi_values, gini_values


def generate_configurations(
    n_jobs: int = 20,
    attacker_sizes: List[int] = None,
    defender_sizes: List[int] = None,
    seeds_per_config: int = 2,
) -> List[Tuple[int, int, int]]:
    """
    Generate all (attacker_size, defender_size, seed) configurations.
    
    Args:
        n_jobs: Total number of jobs (agents) - default 20
        attacker_sizes: List of attacker sizes to test
        defender_sizes: List of defender sizes to test
        seeds_per_config: Number of seeds per configuration
    
    Returns:
        List of (attacker_size, defender_size, seed) tuples
    """
    if attacker_sizes is None:
        # Default attacker sizes for 20 jobs
        attacker_sizes = [1, 3, 5, 10, 19]
    
    if defender_sizes is None:
        defender_sizes = attacker_sizes.copy()
    
    configs = []
    
    for att_size in attacker_sizes:
        for def_size in defender_sizes:
            if att_size + def_size <= n_jobs:
                for seed_offset in range(seeds_per_config):
                    seed = att_size * 10000 + def_size * 1000 + seed_offset
                    configs.append((att_size, def_size, seed))
    
    return configs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run p-MON/q-NEE simulations using folktables ACS data"
    )
    
    # Data loading arguments
    parser.add_argument("--years", type=str, default="2022",
                        help="Comma-separated list of years to load (default: 2022)")
    parser.add_argument("--states", type=str, default=None,
                        help="Comma-separated list of state codes (default: all states)")
    parser.add_argument("--n-jobs", type=int, default=20, dest="n_jobs",
                        help="Number of jobs to sample as agents (default: 20)")
    parser.add_argument("--sample-size", type=int, default=10000, dest="sample_size",
                        help="Number of individuals to sample for demographic buckets (default: 10000)")
    parser.add_argument("--min-job-size", type=int, default=100, dest="min_job_size",
                        help="Minimum workers per job to be eligible for sampling (default: 100)")
    parser.add_argument("--combination-mode", type=str, choices=['multiplicative', 'average'],
                        default='multiplicative', dest="combination_mode",
                        help="How to combine demographic ratios (default: multiplicative)")
    parser.add_argument("--data-folder", type=str, default="data", dest="data_folder",
                        help="Folder to cache downloaded ACS data (default: data)")
    
    # Simulation configuration
    parser.add_argument("--attacker-sizes", type=str, default=None, dest="attacker_sizes",
                        help="Comma-separated list of attacker sizes (default: 1,3,5,10,19)")
    parser.add_argument("--defender-sizes", type=str, default=None, dest="defender_sizes",
                        help="Comma-separated list of defender sizes (default: same as attacker)")
    parser.add_argument("--seeds-per-config", type=int, default=2, dest="seeds_per_config",
                        help="Number of random seeds per configuration (default: 2)")
    
    # Output arguments
    parser.add_argument("-o", "--output-folder", type=str, default="results",
                        dest="output_folder",
                        help="Output folder for CSV results (default: results)")
    parser.add_argument("-p", "--prefix", type=str, default="folktables_sim",
                        help="Filename prefix (default: folktables_sim)")
    parser.add_argument("--save-interval", type=int, default=6, dest="save_interval",
                        help="Save checkpoint every N simulations (default: 6)")
    
    # Runtime arguments
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print verbose output")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug output")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Just print configurations without running")
    parser.add_argument("--no-download", action="store_true", dest="no_download",
                        help="Don't download data (use cached only)")
    
    # Demographic factors
    parser.add_argument("--demographic-factors", type=str, default="age,sex,race",
                        dest="demographic_factors",
                        help="Comma-separated list of demographic factors to use: age,sex,race (default: all three)")
    
    # PGD parameters
    parser.add_argument("--pgd-max-iter", type=int, default=100, dest="pgd_max_iterations",
                        help="Max iterations for PGD (default: 100)")
    parser.add_argument("--pgd-step-size", type=float, default=0.1, dest="pgd_step_size",
                        help="Step size for PGD (default: 0.1)")
    
    args = parser.parse_args()
    
    # Parse arguments
    years = [int(y.strip()) for y in args.years.split(',')]
    states = args.states.split(',') if args.states else None
    attacker_sizes = [int(x) for x in args.attacker_sizes.split(',')] if args.attacker_sizes else None
    defender_sizes = [int(x) for x in args.defender_sizes.split(',')] if args.defender_sizes else None
    
    # Parse demographic factors
    demographic_factors = [f.strip().lower() for f in args.demographic_factors.split(',')]
    valid_factors = {'age', 'sex', 'race'}
    invalid_factors = set(demographic_factors) - valid_factors
    if invalid_factors:
        print(f"Warning: Invalid demographic factors ignored: {invalid_factors}")
        demographic_factors = [f for f in demographic_factors if f in valid_factors]
    if not demographic_factors:
        print("Error: No valid demographic factors specified. Using all three.")
        demographic_factors = ['age', 'sex', 'race']
    
    print("=" * 70)
    print("Folktables p-MON / q-NEE Simulation")
    print("=" * 70)
    print(f"Years: {years}")
    print(f"States: {states or 'All US states'}")
    print(f"Number of jobs (agents): {args.n_jobs}")
    print(f"Sample size (individuals): {args.sample_size}")
    print(f"Min job size: {args.min_job_size}")
    print(f"Combination mode: {args.combination_mode}")
    print(f"Demographic factors: {demographic_factors}")
    print(f"Data folder: {args.data_folder}")
    print(f"Output folder: {args.output_folder}")
    print()
    
    # Step 1: Load census data (cached, reusable)
    print("Step 1: Loading census data...")
    census_data = load_census_data(
        years=years,
        states=states,
        min_job_size=args.min_job_size,
        data_folder=args.data_folder,
        verbose=True,
        download=not args.no_download,
    )
    
    print()
    print(f"Census data loaded:")
    print(f"  Total jobs available: {len(census_data.job_weights)}")
    print()
    
    # Generate configurations
    configs = generate_configurations(
        n_jobs=args.n_jobs,
        attacker_sizes=attacker_sizes,
        defender_sizes=defender_sizes,
        seeds_per_config=args.seeds_per_config,
    )
    
    print(f"Total configurations: {len(configs)}")
    print(f"Output folder: {args.output_folder}")
    print(f"File prefix: {args.prefix}")
    print(f"Save interval: every {args.save_interval} simulations")
    print()
    
    if args.dry_run:
        print("Configurations:")
        for i, (att, def_, seed) in enumerate(configs[:20]):
            print(f"  {i+1}. attackers={att}, defenders={def_}, seed={seed}")
        if len(configs) > 20:
            print(f"  ... and {len(configs) - 20} more")
        return
    
    # Step 2: Precompute all simulations (extract ratios and counts, then free census data)
    print("Step 2: Precomputing simulation data (extracting ratios and counts)...")
    precomputed_sims = []
    for i, (att_size, def_size, seed) in enumerate(configs):
        precomputed = precompute_simulation_data(
            census_data=census_data,
            n_jobs=args.n_jobs,
            sample_size=args.sample_size,
            attacker_size=att_size,
            defender_size=def_size,
            seed=seed,
            demographic_factors=demographic_factors,
            verbose=args.verbose,
        )
        precomputed_sims.append(precomputed)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(configs):
            print(f"  Precomputed {i+1}/{len(configs)} simulations...")
    
    # Free census data to save memory
    print()
    print("Freeing census data from memory...")
    del census_data
    import gc
    gc.collect()
    print("  Memory freed.")
    print()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Step 3: Run simulations using precomputed data
    print("Step 3: Running simulations...")
    results = []
    checkpoint_num = 0
    total_start_time = time.perf_counter()
    sim_times = []
    
    for i, precomputed in enumerate(precomputed_sims):
        completed = i
        total = len(precomputed_sims)
        percent = (completed / total) * 100
        
        elapsed_total = time.perf_counter() - total_start_time
        if completed > 0 and sim_times:
            avg_time = sum(sim_times) / len(sim_times)
            eta_seconds = avg_time * (total - completed)
            eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds >= 60 else f"{eta_seconds:.0f}s"
        else:
            eta_str = "calculating..."
        
        elapsed_str = f"{elapsed_total/60:.1f}min" if elapsed_total >= 60 else f"{elapsed_total:.0f}s"
        
        print(f"[{i+1}/{total}] ({percent:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}")
        print(f"  Config: attackers={precomputed.attacker_size}, defenders={precomputed.defender_size}, seed={precomputed.seed}")
        
        # Run the simulation with precomputed data
        result = run_folktables_simulation(
            precomputed=precomputed,
            combination_mode=args.combination_mode,
            sample_size=args.sample_size,
            min_job_size=args.min_job_size,
            demographic_factors=demographic_factors,
            verbose=args.verbose or args.debug,
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
        print(f"  HHI: att={result.attacker_mean_hhi:.4f}, def={result.defender_mean_hhi:.4f} | "
              f"Gini: att={result.attacker_mean_gini:.4f}, def={result.defender_mean_gini:.4f}")
        print(f"  Sim att-def: mean={result.att_def_sim_mean:.4f}")
        
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
    
    # HHI/Gini summary
    all_att_hhi = [r.attacker_mean_hhi for r in results]
    all_def_hhi = [r.defender_mean_hhi for r in results]
    all_att_gini = [r.attacker_mean_gini for r in results]
    all_def_gini = [r.defender_mean_gini for r in results]
    all_att_def_sim = [r.att_def_sim_mean for r in results]
    print()
    print(f"HHI (mean across sims): attackers={np.mean(all_att_hhi):.4f}, defenders={np.mean(all_def_hhi):.4f}")
    print(f"Gini (mean across sims): attackers={np.mean(all_att_gini):.4f}, defenders={np.mean(all_def_gini):.4f}")
    print(f"Att-Def Similarity (mean across sims): {np.mean(all_att_def_sim):.4f}")
    
    print()
    print(f"Results saved to: {final_file}")


if __name__ == "__main__":
    main()