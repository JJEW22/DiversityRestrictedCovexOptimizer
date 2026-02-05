#!/usr/bin/env python3
"""
Plotting module for folktables simulation results.

Load CSV output files and generate visualizations.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from scipy import stats


def load_results(csv_path: str, cap_values: bool = False) -> pd.DataFrame:
    """
    Load simulation results from CSV file.
    
    Args:
        csv_path: Path to CSV file
        cap_values: If True, cap values at 1.0 using min(1, value) to handle rounding errors.
                   This ensures 1 - q-NEE and 1 - p-MON are always non-negative.
    
    Returns:
        DataFrame with simulation results
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    if cap_values:
        # Cap values at 1.0: min(1, value)
        # This ensures 1 - value is always >= 0
        if 'q_nee_worst_defending_group' in df.columns:
            orig_above_1 = (df['q_nee_worst_defending_group'] > 1.0).sum()
            df['q_nee_worst_defending_group'] = df['q_nee_worst_defending_group'].clip(upper=1.0)
            if orig_above_1 > 0:
                print(f"  q_nee_worst_defending_group: {orig_above_1} values > 1.0 capped to 1.0")
        
        if 'q_nee_defending_group' in df.columns:
            orig_above_1 = (df['q_nee_defending_group'] > 1.0).sum()
            df['q_nee_defending_group'] = df['q_nee_defending_group'].clip(upper=1.0)
            if orig_above_1 > 0:
                print(f"  q_nee_defending_group: {orig_above_1} values > 1.0 capped to 1.0")
        
        if 'p_mon_attacking_group' in df.columns:
            orig_above_1 = (df['p_mon_attacking_group'] > 1.0).sum()
            df['p_mon_attacking_group'] = df['p_mon_attacking_group'].clip(upper=1.0)
            if orig_above_1 > 0:
                print(f"  p_mon_attacking_group: {orig_above_1} values > 1.0 capped to 1.0")
        
        print("  Applied value capping (values capped at 1.0)")
    
    return df


def compute_theoretical_qnee(attacker_size: int, defender_size: int) -> float:
    """
    Compute theoretical worst-case q-NEE.
    
    Theory: q-NEE = g / (k + g)
    where k = attacker size, g = defender size
    
    Returns:
        Theoretical q-NEE value
    """
    k = attacker_size
    g = defender_size
    if k + g == 0:
        return 1.0
    return g / (k + g)


def compute_theoretical_pmon(attacker_size: int, n_agents: int) -> float:
    """
    Compute theoretical worst-case p-MON.
    
    Theory: p-MON = 1 / (2 - k/n)
    where k = attacker size, n = total agents
    
    Returns:
        Theoretical p-MON value
    """
    k = attacker_size
    n = n_agents
    if n == 0:
        return 1.0
    denominator = 2 - (k / n)
    if denominator <= 0:
        return float('inf')
    return 1 / denominator


def plot_qnee_vs_attacker_size(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    use_worst_case: bool = True,
    show_individual_points: bool = True,
    show_theoretical: bool = False,
    figsize: tuple = (10, 6),
):
    """
    Plot (1 - q-NEE) vs attacker group size.
    
    Shows how the envy experienced by defenders changes as the attacking
    coalition size increases.
    
    Args:
        df: DataFrame with simulation results
        output_path: Path to save figure (if None, displays interactively)
        title: Plot title (auto-generated if None)
        use_worst_case: If True, use q_nee_worst_defending_group; else use q_nee_defending_group
        show_individual_points: If True, show scatter of individual simulation results
        show_theoretical: If True, show theoretical worst-case bounds by defender size
        figsize: Figure size tuple
    """
    # Determine which q-NEE column to use
    if use_worst_case:
        qnee_col = 'q_nee_worst_defending_group'
        qnee_label = 'Worst-case q-NEE'
    else:
        qnee_col = 'q_nee_defending_group'
        qnee_label = 'q-NEE (under p-MON optimal)'
    
    if qnee_col not in df.columns:
        raise ValueError(f"Column '{qnee_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Compute 1 - q-NEE (the "envy" or deviation from fairness)
    df = df.copy()
    df['one_minus_qnee'] = 1 - df[qnee_col]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_theoretical:
        # Group by both attacker and defender size
        defender_sizes = sorted(df['defender_size'].unique())
        attacker_sizes = sorted(df['attacker_size'].unique())
        
        # Color maps for empirical (solid) and theoretical (dashed)
        colors = plt.cm.tab10(np.linspace(0, 1, len(defender_sizes)))
        
        for i, def_size in enumerate(defender_sizes):
            subset = df[df['defender_size'] == def_size]
            grouped = subset.groupby('attacker_size')['one_minus_qnee'].agg(['mean', 'std'])
            grouped = grouped.reset_index()
            
            # Plot empirical data
            ax.errorbar(
                grouped['attacker_size'],
                grouped['mean'],
                yerr=grouped['std'],
                fmt='o-',
                color=colors[i],
                linewidth=2,
                markersize=6,
                capsize=3,
                label=f'Empirical (def={def_size})'
            )
            
            # Plot theoretical line
            # Theory: q-NEE = g/(k+g), so 1 - q-NEE = k/(k+g)
            theoretical_x = np.array(attacker_sizes)
            theoretical_y = theoretical_x / (theoretical_x + def_size)  # 1 - g/(k+g) = k/(k+g)
            
            ax.plot(
                theoretical_x,
                theoretical_y,
                '--',
                color=colors[i],
                linewidth=2,
                alpha=0.7,
                label=f'Theory (def={def_size})'
            )
    else:
        # Original behavior: aggregate all defender sizes
        grouped = df.groupby('attacker_size')['one_minus_qnee'].agg(['mean', 'std', 'min', 'max', 'count'])
        grouped = grouped.reset_index()
        
        # Plot individual points if requested
        if show_individual_points:
            jitter = np.random.uniform(-0.15, 0.15, len(df))
            ax.scatter(
                df['attacker_size'] + jitter, 
                df['one_minus_qnee'],
                alpha=0.3, 
                s=20, 
                c='steelblue',
                label='Individual simulations'
            )
        
        # Plot mean with error bars
        ax.errorbar(
            grouped['attacker_size'],
            grouped['mean'],
            yerr=grouped['std'],
            fmt='o-',
            color='darkred',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2,
            label=f'Mean ± Std Dev (n={grouped["count"].iloc[0]} per size)'
        )
    
    # Labels and title
    ax.set_xlabel('Attacker Group Size (k)', fontsize=12)
    ax.set_ylabel('1 - q-NEE (Defender Envy)', fontsize=12)
    
    if title is None:
        if show_theoretical:
            title = f'Defender Envy vs Attacker Size: Empirical vs Theoretical\n({qnee_label}, Theory: 1 - g/(k+g) = k/(k+g))'
        else:
            title = f'Defender Envy vs Attacker Coalition Size\n({qnee_label})'
    ax.set_title(title, fontsize=14)
    
    # Set x-axis to show integer ticks
    ax.set_xticks(grouped['attacker_size'].values)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add annotation with summary stats
    total_sims = len(df)
    unique_configs = len(df.groupby(['attacker_size', 'defender_size']))
    annotation = f'Total simulations: {total_sims}\nUnique configs: {unique_configs}'
    ax.annotate(
        annotation,
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        verticalalignment='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return grouped


def plot_qnee_vs_attacker_size_by_defender(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    use_worst_case: bool = True,
    figsize: tuple = (12, 6),
):
    """
    Plot (1 - q-NEE) vs attacker size, with separate lines for each defender size.
    
    Args:
        df: DataFrame with simulation results
        output_path: Path to save figure (if None, displays interactively)
        title: Plot title (auto-generated if None)
        use_worst_case: If True, use q_nee_worst_defending_group
        figsize: Figure size tuple
    """
    # Determine which q-NEE column to use
    if use_worst_case:
        qnee_col = 'q_nee_worst_defending_group'
        qnee_label = 'Worst-case q-NEE'
    else:
        qnee_col = 'q_nee_defending_group'
        qnee_label = 'q-NEE (under p-MON optimal)'
    
    df = df.copy()
    df['one_minus_qnee'] = 1 - df[qnee_col]
    
    # Get unique defender sizes
    defender_sizes = sorted(df['defender_size'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(defender_sizes)))
    
    for i, def_size in enumerate(defender_sizes):
        subset = df[df['defender_size'] == def_size]
        grouped = subset.groupby('attacker_size')['one_minus_qnee'].agg(['mean', 'std'])
        grouped = grouped.reset_index()
        
        ax.errorbar(
            grouped['attacker_size'],
            grouped['mean'],
            yerr=grouped['std'],
            fmt='o-',
            color=colors[i],
            linewidth=2,
            markersize=6,
            capsize=3,
            label=f'Defender size = {def_size}'
        )
    
    # Labels and title
    ax.set_xlabel('Attacker Group Size', fontsize=12)
    ax.set_ylabel('1 - q-NEE (Defender Envy)', fontsize=12)
    
    if title is None:
        title = f'Defender Envy vs Attacker Size by Defender Group Size\n({qnee_label})'
    ax.set_title(title, fontsize=14)
    
    # Set x-axis to show integer ticks
    attacker_sizes = sorted(df['attacker_size'].unique())
    ax.set_xticks(attacker_sizes)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', title='Defender Size')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pmon_vs_attacker_size(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_individual_points: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot (1 - p-MON) vs attacker group size.
    
    Shows how much attackers can benefit themselves as coalition size increases.
    1 - p-MON represents the "loss" from not being in the attacking coalition.
    
    Args:
        df: DataFrame with simulation results
        output_path: Path to save figure (if None, displays interactively)
        title: Plot title (auto-generated if None)
        show_individual_points: If True, show scatter of individual results
        figsize: Figure size tuple
    """
    df = df.copy()
    df['one_minus_pmon'] = 1 - df['p_mon_attacking_group']
    
    # Group by attacker size
    grouped = df.groupby('attacker_size')['one_minus_pmon'].agg(['mean', 'std', 'min', 'max', 'count'])
    grouped = grouped.reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_individual_points:
        jitter = np.random.uniform(-0.15, 0.15, len(df))
        ax.scatter(
            df['attacker_size'] + jitter,
            df['one_minus_pmon'],
            alpha=0.3,
            s=20,
            c='steelblue',
            label='Individual simulations'
        )
    
    ax.errorbar(
        grouped['attacker_size'],
        grouped['mean'],
        yerr=grouped['std'],
        fmt='o-',
        color='darkgreen',
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label=f'Mean ± Std Dev'
    )
    
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.7, label='1 - p-MON = 0 (no benefit)')
    
    ax.set_xlabel('Attacker Group Size (k)', fontsize=12)
    ax.set_ylabel('1 - p-MON', fontsize=12)
    
    if title is None:
        title = 'Attacker Self-Benefit vs Coalition Size'
    ax.set_title(title, fontsize=14)
    
    ax.set_xticks(grouped['attacker_size'].values)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return grouped


def plot_pmon_vs_attacker_size_theoretical(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_individual_points: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot (1 - p-MON) vs attacker group size with theoretical worst-case bound.
    
    Theory: p-MON = 1 / (2 - k/n) where k = attacker size, n = total agents
    So: 1 - p-MON = 1 - 1/(2 - k/n)
    
    Args:
        df: DataFrame with simulation results
        output_path: Path to save figure (if None, displays interactively)
        title: Plot title (auto-generated if None)
        show_individual_points: If True, show scatter of individual results
        figsize: Figure size tuple
    """
    df = df.copy()
    df['one_minus_pmon'] = 1 - df['p_mon_attacking_group']
    
    # Get n_agents (total agents)
    if 'n_agents' in df.columns:
        n_agents = df['n_agents'].iloc[0]
    elif 'n_jobs' in df.columns:
        # Fallback for older CSV files
        n_agents = df['n_jobs'].iloc[0]
    else:
        raise ValueError("n_agents or n_jobs column required for theoretical p-MON calculation")
    
    # Group by attacker size
    grouped = df.groupby('attacker_size')['one_minus_pmon'].agg(['mean', 'std', 'min', 'max', 'count'])
    grouped = grouped.reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_individual_points:
        jitter = np.random.uniform(-0.15, 0.15, len(df))
        ax.scatter(
            df['attacker_size'] + jitter,
            df['one_minus_pmon'],
            alpha=0.3,
            s=20,
            c='steelblue',
            label='Individual simulations'
        )
    
    # Empirical mean with error bars
    ax.errorbar(
        grouped['attacker_size'],
        grouped['mean'],
        yerr=grouped['std'],
        fmt='o-',
        color='darkgreen',
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=2,
        label=f'Empirical Mean ± Std Dev'
    )
    
    # Theoretical line: 1 - p-MON = 1 - 1/(2 - k/n)
    attacker_sizes = sorted(df['attacker_size'].unique())
    theoretical_x = np.array(attacker_sizes)
    theoretical_pmon = 1 / (2 - theoretical_x / n_agents)
    theoretical_y = 1 - theoretical_pmon
    
    ax.plot(
        theoretical_x,
        theoretical_y,
        '--',
        color='darkred',
        linewidth=2,
        label=f'Theory: 1 - 1/(2 - k/n), n={n_agents}'
    )
    
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.7, label='1 - p-MON = 0 (no benefit)')
    
    ax.set_xlabel('Attacker Group Size (k)', fontsize=12)
    ax.set_ylabel('1 - p-MON', fontsize=12)
    
    if title is None:
        title = f'Attacker Self-Benefit: Empirical vs Theoretical\n(Theory: 1 - p-MON = 1 - 1/(2 - k/n), n={n_agents})'
    ax.set_title(title, fontsize=14)
    
    ax.set_xticks(grouped['attacker_size'].values)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return grouped


def plot_scatter_qnee_vs_pmon(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 8),
    color_by: str = 'attacker_size',
):
    """
    Plot scatter of (1 - worst q-NEE) vs (1 - resulting p-MON).
    
    X-axis: 1 - q_nee_worst_defending_group (capped at 1)
    Y-axis: 1 - p_mon_under_qnee_worst (not capped - shows actual resulting value)
    
    Shows: when optimizing for worst-case q-NEE, what's the resulting p-MON?
    
    Args:
        df: DataFrame with simulation results
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        color_by: Column to color points by ('attacker_size' or 'defender_size')
    """
    df = df.copy()
    
    # Check required columns
    if 'q_nee_worst_defending_group' not in df.columns:
        raise ValueError("q_nee_worst_defending_group column required")
    if 'p_mon_under_qnee_worst' not in df.columns:
        raise ValueError("p_mon_under_qnee_worst column required")
    
    # Cap worst q-NEE at 1.0 (the optimized metric), don't cap resulting p-MON
    df['x'] = 1 - df['q_nee_worst_defending_group'].clip(upper=1.0)
    df['y'] = 1 - df['p_mon_under_qnee_worst']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by specified column
    if color_by in df.columns:
        unique_vals = sorted(df[color_by].unique())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_vals)))
        
        for i, val in enumerate(unique_vals):
            subset = df[df[color_by] == val]
            ax.scatter(subset['x'], subset['y'], c=[colors[i]], label=f'{color_by}={val}', alpha=0.6, s=40)
    else:
        ax.scatter(df['x'], df['y'], alpha=0.6, s=40, c='steelblue')
    
    # Line of best fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
    x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\n(R² = {r_value**2:.3f})')
    
    ax.legend(loc='best')
    
    ax.set_xlabel('1 - q-NEE (worst-case, capped)', fontsize=12)
    ax.set_ylabel('1 - p-MON (under q-NEE optimization)', fontsize=12)
    
    if title is None:
        title = 'Worst q-NEE vs Resulting p-MON\n(When optimizing for defender protection)'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scatter_pmon_vs_qnee(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 8),
    color_by: str = 'attacker_size',
):
    """
    Plot scatter of (1 - worst p-MON) vs (1 - resulting q-NEE).
    
    X-axis: 1 - p_mon_attacking_group (capped at 1)
    Y-axis: 1 - q_nee_defending_group (not capped - shows actual resulting value)
    
    Shows: when optimizing for p-MON, what's the resulting q-NEE?
    
    Args:
        df: DataFrame with simulation results
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        color_by: Column to color points by
    """
    df = df.copy()
    
    # Check required columns
    if 'p_mon_attacking_group' not in df.columns:
        raise ValueError("p_mon_attacking_group column required")
    if 'q_nee_defending_group' not in df.columns:
        raise ValueError("q_nee_defending_group column required")
    
    # Cap p-MON at 1.0 (the optimized metric), don't cap resulting q-NEE
    df['x'] = 1 - df['p_mon_attacking_group'].clip(upper=1.0)
    df['y'] = 1 - df['q_nee_defending_group']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by specified column
    if color_by in df.columns:
        unique_vals = sorted(df[color_by].unique())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_vals)))
        
        for i, val in enumerate(unique_vals):
            subset = df[df[color_by] == val]
            ax.scatter(subset['x'], subset['y'], c=[colors[i]], label=f'{color_by}={val}', alpha=0.6, s=40)
    else:
        ax.scatter(df['x'], df['y'], alpha=0.6, s=40, c='steelblue')
    
    # Line of best fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
    x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\n(R² = {r_value**2:.3f})')
    
    ax.legend(loc='best')
    
    ax.set_xlabel('1 - p-MON (optimized, capped)', fontsize=12)
    ax.set_ylabel('1 - q-NEE (under p-MON optimization)', fontsize=12)
    
    if title is None:
        title = 'Optimal p-MON vs Resulting q-NEE\n(When optimizing for attacker benefit)'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scatter_worst_pmon_vs_worst_qnee(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 8),
    color_by: str = 'attacker_size',
):
    """
    Plot scatter of (1 - worst p-MON) vs (1 - worst q-NEE).
    
    X-axis: 1 - p_mon_attacking_group (capped at 1)
    Y-axis: 1 - q_nee_worst_defending_group (capped at 1)
    
    Shows: comparison of the two worst-case metrics.
    
    Args:
        df: DataFrame with simulation results
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        color_by: Column to color points by
    """
    df = df.copy()
    
    # Check required columns
    if 'p_mon_attacking_group' not in df.columns:
        raise ValueError("p_mon_attacking_group column required")
    if 'q_nee_worst_defending_group' not in df.columns:
        raise ValueError("q_nee_worst_defending_group column required")
    
    # Cap both at 1.0 (both are optimized/worst-case metrics)
    df['x'] = 1 - df['p_mon_attacking_group'].clip(upper=1.0)
    df['y'] = 1 - df['q_nee_worst_defending_group'].clip(upper=1.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by specified column
    if color_by in df.columns:
        unique_vals = sorted(df[color_by].unique())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_vals)))
        
        for i, val in enumerate(unique_vals):
            subset = df[df[color_by] == val]
            ax.scatter(subset['x'], subset['y'], c=[colors[i]], label=f'{color_by}={val}', alpha=0.6, s=40)
    else:
        ax.scatter(df['x'], df['y'], alpha=0.6, s=40, c='steelblue')
    
    # Line of best fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
    x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\n(R² = {r_value**2:.3f})')
    
    ax.legend(loc='best')
    
    ax.set_xlabel('1 - p-MON (optimized, capped)', fontsize=12)
    ax.set_ylabel('1 - q-NEE (worst-case, capped)', fontsize=12)
    
    if title is None:
        title = 'Worst p-MON vs Worst q-NEE\n(Comparing worst-case metrics)'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_vs_n_items(
    csv_files: List[str],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    cap_values: bool = True,
):
    """
    Plot average worst-case p-MON and q-NEE vs number of items (buckets).
    
    Takes multiple CSV files (each with different n_buckets) and plots how
    the average metrics change.
    
    Args:
        csv_files: List of paths to CSV files with different n_buckets
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        cap_values: If True, cap values at 1.0
    """
    results = []
    
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        
        # Get n_buckets (number of items)
        if 'n_buckets' in df.columns:
            n_items = df['n_buckets'].iloc[0]
        elif 'n_buckets_found' in df.columns:
            n_items = df['n_buckets_found'].iloc[0]
        else:
            print(f"Warning: No n_buckets column in {csv_path}, skipping")
            continue
        
        # Get metrics
        if 'p_mon_attacking_group' in df.columns:
            pmon = df['p_mon_attacking_group']
            if cap_values:
                pmon = pmon.clip(upper=1.0)
            avg_one_minus_pmon = (1 - pmon).mean()
            std_one_minus_pmon = (1 - pmon).std()
        else:
            avg_one_minus_pmon = np.nan
            std_one_minus_pmon = np.nan
        
        if 'q_nee_worst_defending_group' in df.columns:
            qnee = df['q_nee_worst_defending_group']
            if cap_values:
                qnee = qnee.clip(upper=1.0)
            avg_one_minus_qnee = (1 - qnee).mean()
            std_one_minus_qnee = (1 - qnee).std()
        else:
            avg_one_minus_qnee = np.nan
            std_one_minus_qnee = np.nan
        
        results.append({
            'n_items': n_items,
            'avg_one_minus_pmon': avg_one_minus_pmon,
            'std_one_minus_pmon': std_one_minus_pmon,
            'avg_one_minus_qnee': avg_one_minus_qnee,
            'std_one_minus_qnee': std_one_minus_qnee,
            'n_simulations': len(df),
        })
        
        print(f"Loaded {csv_path}: n_items={n_items}, n_sims={len(df)}")
    
    if not results:
        print("No valid data to plot")
        return
    
    # Sort by n_items
    results = sorted(results, key=lambda x: x['n_items'])
    
    n_items = [r['n_items'] for r in results]
    avg_pmon = [r['avg_one_minus_pmon'] for r in results]
    std_pmon = [r['std_one_minus_pmon'] for r in results]
    avg_qnee = [r['avg_one_minus_qnee'] for r in results]
    std_qnee = [r['std_one_minus_qnee'] for r in results]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot p-MON
    ax.errorbar(n_items, avg_pmon, yerr=std_pmon, fmt='o-', color='blue',
                linewidth=2, markersize=8, capsize=5, capthick=2,
                label='1 - p-MON (attacker benefit)')
    
    # Plot q-NEE
    ax.errorbar(n_items, avg_qnee, yerr=std_qnee, fmt='s-', color='red',
                linewidth=2, markersize=8, capsize=5, capthick=2,
                label='1 - q-NEE (defender envy)')
    
    ax.set_xlabel('Number of Items (Demographic Buckets)', fontsize=12)
    ax.set_ylabel('Average (1 - metric)', fontsize=12)
    
    if title is None:
        title = 'Worst-Case Metrics vs Number of Items'
    ax.set_title(title, fontsize=14)
    
    ax.set_xticks(n_items)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary
    print("\n=== Summary ===")
    print(f"{'n_items':>10} {'1-pMON':>12} {'1-qNEE':>12} {'n_sims':>10}")
    for r in results:
        print(f"{r['n_items']:>10} {r['avg_one_minus_pmon']:>12.4f} {r['avg_one_minus_qnee']:>12.4f} {r['n_simulations']:>10}")


def plot_metrics_vs_items_single_file(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    cap_values: bool = True,
    items_column: str = None,
):
    """
    Plot average worst-case p-MON and q-NEE vs number of items from a single DataFrame.
    
    Groups by number of items and computes average metrics for each group.
    
    Args:
        df: DataFrame with simulation results (must have varying n_buckets or n_resources)
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        cap_values: If True, cap values at 1.0
        items_column: Column name for number of items (auto-detected if None)
    """
    df = df.copy()
    
    # Auto-detect items column if not specified
    if items_column is None:
        if 'n_resources' in df.columns:
            items_column = 'n_resources'
        elif 'n_buckets' in df.columns:
            items_column = 'n_buckets'
        elif 'n_buckets_found' in df.columns:
            items_column = 'n_buckets_found'
        elif 'n_items' in df.columns:
            items_column = 'n_items'
        else:
            raise ValueError(f"No items column found. Available columns: {list(df.columns)}")
    elif items_column not in df.columns:
        raise ValueError(f"Column '{items_column}' not found. Available columns: {list(df.columns)}")
    
    print(f"Using '{items_column}' as items column")
    print(f"Unique item counts: {sorted(df[items_column].unique())}")
    
    # Cap values if requested
    if cap_values:
        if 'p_mon_attacking_group' in df.columns:
            df['p_mon_attacking_group'] = df['p_mon_attacking_group'].clip(upper=1.0)
        if 'q_nee_worst_defending_group' in df.columns:
            df['q_nee_worst_defending_group'] = df['q_nee_worst_defending_group'].clip(upper=1.0)
    
    # Compute 1 - metrics
    if 'p_mon_attacking_group' in df.columns:
        df['one_minus_pmon'] = 1 - df['p_mon_attacking_group']
    if 'q_nee_worst_defending_group' in df.columns:
        df['one_minus_qnee'] = 1 - df['q_nee_worst_defending_group']
    
    # Group by number of items
    grouped = df.groupby(items_column).agg({
        'one_minus_pmon': ['mean', 'std', 'count'] if 'one_minus_pmon' in df.columns else ['count'],
        'one_minus_qnee': ['mean', 'std'] if 'one_minus_qnee' in df.columns else ['count'],
    })
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    n_items = grouped[items_column].values
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot p-MON
    if 'one_minus_pmon_mean' in grouped.columns:
        ax.errorbar(n_items, grouped['one_minus_pmon_mean'], 
                    yerr=grouped['one_minus_pmon_std'], 
                    fmt='o-', color='blue',
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    label='1 - p-MON (attacker benefit)')
    
    # Plot q-NEE
    if 'one_minus_qnee_mean' in grouped.columns:
        ax.errorbar(n_items, grouped['one_minus_qnee_mean'], 
                    yerr=grouped['one_minus_qnee_std'], 
                    fmt='s-', color='red',
                    linewidth=2, markersize=8, capsize=5, capthick=2,
                    label='1 - q-NEE (defender envy)')
    
    ax.set_xlabel('Number of Items', fontsize=12)
    ax.set_ylabel('Average (1 - metric)', fontsize=12)
    
    if title is None:
        title = 'Worst-Case Metrics vs Number of Items'
    ax.set_title(title, fontsize=14)
    
    ax.set_xticks(n_items)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary
    print("\n=== Summary: Metrics by Number of Items ===")
    count_col = [c for c in grouped.columns if 'count' in c][0]
    print(f"{'n_items':>10} {'1-pMON':>12} {'std':>10} {'1-qNEE':>12} {'std':>10} {'n_sims':>8}")
    for _, row in grouped.iterrows():
        pmon_mean = row.get('one_minus_pmon_mean', float('nan'))
        pmon_std = row.get('one_minus_pmon_std', float('nan'))
        qnee_mean = row.get('one_minus_qnee_mean', float('nan'))
        qnee_std = row.get('one_minus_qnee_std', float('nan'))
        print(f"{row[items_column]:>10} {pmon_mean:>12.4f} {pmon_std:>10.4f} {qnee_mean:>12.4f} {qnee_std:>10.4f} {int(row[count_col]):>8}")


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics from the results."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal simulations: {len(df)}")
    print(f"Attacker sizes: {sorted(df['attacker_size'].unique())}")
    print(f"Defender sizes: {sorted(df['defender_size'].unique())}")
    print(f"Seeds per config: {df.groupby(['attacker_size', 'defender_size']).size().iloc[0]}")
    
    if 'n_agents' in df.columns:
        print(f"Number of agents: {df['n_agents'].iloc[0]}")
    elif 'n_jobs' in df.columns:
        print(f"Number of agents (n_jobs): {df['n_jobs'].iloc[0]}")
    if 'n_buckets' in df.columns:
        print(f"Number of buckets (resources): {df['n_buckets'].iloc[0]}")
    
    print("\n--- q-NEE Statistics (Defender Envy = 1 - q-NEE) ---")
    if 'q_nee_worst_defending_group' in df.columns:
        qnee_worst = df['q_nee_worst_defending_group']
        print(f"Worst-case q-NEE: mean={qnee_worst.mean():.4f}, std={qnee_worst.std():.4f}, "
              f"min={qnee_worst.min():.4f}, max={qnee_worst.max():.4f}")
        print(f"Worst-case 1-q-NEE: mean={1-qnee_worst.mean():.4f}")
    
    if 'q_nee_defending_group' in df.columns:
        qnee = df['q_nee_defending_group']
        print(f"q-NEE (p-MON opt): mean={qnee.mean():.4f}, std={qnee.std():.4f}")
    
    print("\n--- p-MON Statistics (Attacker Benefit) ---")
    if 'p_mon_attacking_group' in df.columns:
        pmon = df['p_mon_attacking_group']
        print(f"p-MON: mean={pmon.mean():.4f}, std={pmon.std():.4f}, "
              f"min={pmon.min():.4f}, max={pmon.max():.4f}")
    
    print("\n--- Convergence ---")
    if 'pmon_converged' in df.columns:
        pmon_conv = df['pmon_converged'].sum()
        print(f"p-MON converged: {pmon_conv}/{len(df)} ({100*pmon_conv/len(df):.1f}%)")
    if 'qnee_worst_converged' in df.columns:
        qnee_conv = df['qnee_worst_converged'].sum()
        print(f"q-NEE worst converged: {qnee_conv}/{len(df)} ({100*qnee_conv/len(df):.1f}%)")
    
    print("\n--- By Attacker Size ---")
    by_att = df.groupby('attacker_size').agg({
        'p_mon_attacking_group': 'mean',
        'q_nee_worst_defending_group': 'mean' if 'q_nee_worst_defending_group' in df.columns else 'first',
    })
    by_att['one_minus_qnee'] = 1 - by_att.get('q_nee_worst_defending_group', 0)
    print(by_att.to_string())


def main():
    parser = argparse.ArgumentParser(
        description='Plot folktables simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic plot of 1-qNEE vs attacker size
  python plot_results.py results.csv
  
  # Save plot to file
  python plot_results.py results.csv -o qnee_plot.png
  
  # Plot with defender size breakdown
  python plot_results.py results.csv --by-defender -o qnee_by_defender.png
  
  # Plot p-MON instead
  python plot_results.py results.csv --plot-type pmon
  
  # Plot with theoretical bounds (groups by defender size)
  python plot_results.py results.csv --theoretical -o theoretical.png
  
  # Cap values at 1.0 (fixes rounding errors)
  python plot_results.py results.csv --capped
  
  # Scatter plots comparing metrics
  python plot_results.py results.csv --scatter qnee-pmon -o scatter1.png
  python plot_results.py results.csv --scatter pmon-qnee -o scatter2.png
  python plot_results.py results.csv --scatter worst -o scatter3.png
  
  # Plot metrics vs number of items (multiple CSV files with different n_buckets)
  python plot_results.py results_sex.csv results_age_sex.csv results_all.csv --items-comparison -o items.png
  
  # Plot metrics vs number of items (single CSV where n_buckets varies across rows)
  python plot_results.py random_sims.csv --items-vary -o items_vary.png
  
  # Just print statistics
  python plot_results.py results.csv --stats-only
        """
    )
    
    parser.add_argument('csv_file', type=str, nargs='+', 
                        help='Path to CSV results file(s). Multiple files for --items-comparison')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for plot (displays interactively if not specified)')
    parser.add_argument('--plot-type', type=str, default='qnee',
                        choices=['qnee', 'pmon', 'both'],
                        help='Type of plot to generate (default: qnee)')
    parser.add_argument('--by-defender', action='store_true',
                        help='Show separate lines for each defender size')
    parser.add_argument('--use-pmon-optimal', action='store_true',
                        help='Use q-NEE under p-MON optimal instead of worst-case')
    parser.add_argument('--no-points', action='store_true',
                        help='Hide individual simulation points')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, no plots')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    parser.add_argument('--figsize', type=str, default='10,6',
                        help='Figure size as width,height (default: 10,6)')
    parser.add_argument('--capped', action='store_true',
                        help='Cap q-NEE and p-MON values at 1.0 to handle rounding errors')
    parser.add_argument('--theoretical', action='store_true',
                        help='Show theoretical worst-case bounds (groups by defender size)')
    parser.add_argument('--scatter', type=str, default=None,
                        choices=['qnee-pmon', 'pmon-qnee', 'worst', 'all'],
                        help='Generate scatter plot: qnee-pmon (worst qNEE vs resulting pMON), '
                             'pmon-qnee (optimal pMON vs resulting qNEE), '
                             'worst (optimal pMON vs worst qNEE), '
                             'all (generate all three)')
    parser.add_argument('--color-by', type=str, default='attacker_size',
                        dest='color_by',
                        help='Column to color scatter plot points by (default: attacker_size)')
    parser.add_argument('--items-comparison', action='store_true',
                        help='Plot metrics vs number of items. Requires multiple CSV files with different n_buckets.')
    parser.add_argument('--items-vary', action='store_true',
                        help='Plot metrics vs number of items from single CSV where n_buckets varies across rows.')
    
    args = parser.parse_args()
    
    # Parse figsize
    figsize = tuple(float(x) for x in args.figsize.split(','))
    
    # Handle items comparison (multiple files)
    if args.items_comparison:
        if len(args.csv_file) < 2:
            print("Warning: --items-comparison works best with multiple CSV files with different n_buckets")
        plot_metrics_vs_n_items(
            csv_files=args.csv_file,
            output_path=args.output,
            title=args.title,
            figsize=figsize,
            cap_values=True,  # Always cap for this plot
        )
        return
    
    # For single file operations, use first file
    csv_file = args.csv_file[0]
    
    # Load data (don't apply global capping here, scatter plots handle it internally)
    df = load_results(csv_file, cap_values=args.capped)
    
    # Print statistics
    print_summary_stats(df)
    
    if args.stats_only:
        return
    
    # Handle items-vary plot (single file with varying n_buckets)
    if args.items_vary:
        plot_metrics_vs_items_single_file(
            df,
            output_path=args.output,
            title=args.title,
            figsize=figsize,
            cap_values=True,  # Always cap for this plot
        )
        return
    
    # Handle scatter plots
    if args.scatter:
        if args.scatter in ['qnee-pmon', 'all']:
            output = args.output
            if args.scatter == 'all' and output:
                output = output.replace('.png', '_qnee_pmon.png')
            plot_scatter_qnee_vs_pmon(
                df,
                output_path=output,
                title=args.title,
                figsize=(float(args.figsize.split(',')[0]), float(args.figsize.split(',')[0])),  # Square
                color_by=args.color_by,
            )
        
        if args.scatter in ['pmon-qnee', 'all']:
            output = args.output
            if args.scatter == 'all' and output:
                output = output.replace('.png', '_pmon_qnee.png')
            plot_scatter_pmon_vs_qnee(
                df,
                output_path=output,
                title=args.title if args.scatter != 'all' else None,
                figsize=(float(args.figsize.split(',')[0]), float(args.figsize.split(',')[0])),
                color_by=args.color_by,
            )
        
        if args.scatter in ['worst', 'all']:
            output = args.output
            if args.scatter == 'all' and output:
                output = output.replace('.png', '_worst.png')
            plot_scatter_worst_pmon_vs_worst_qnee(
                df,
                output_path=output,
                title=args.title if args.scatter != 'all' else None,
                figsize=(float(args.figsize.split(',')[0]), float(args.figsize.split(',')[0])),
                color_by=args.color_by,
            )
        
        return  # Exit after scatter plots
    
    # Generate standard plots
    use_worst_case = not args.use_pmon_optimal
    show_points = not args.no_points
    
    if args.plot_type in ['qnee', 'both']:
        if args.theoretical:
            # Theoretical mode: show empirical vs theoretical by defender size
            output = args.output
            if args.plot_type == 'both' and output:
                output = output.replace('.png', '_qnee_theoretical.png')
            plot_qnee_vs_attacker_size(
                df,
                output_path=output,
                title=args.title,
                use_worst_case=use_worst_case,
                show_individual_points=False,  # Don't show points in theoretical mode
                show_theoretical=True,
                figsize=figsize,
            )
        elif args.by_defender:
            output = args.output
            if args.plot_type == 'both' and output:
                output = output.replace('.png', '_qnee_by_def.png')
            plot_qnee_vs_attacker_size_by_defender(
                df,
                output_path=output,
                title=args.title,
                use_worst_case=use_worst_case,
                figsize=figsize,
            )
        else:
            output = args.output
            if args.plot_type == 'both' and output:
                output = output.replace('.png', '_qnee.png')
            plot_qnee_vs_attacker_size(
                df,
                output_path=output,
                title=args.title,
                use_worst_case=use_worst_case,
                show_individual_points=show_points,
                show_theoretical=False,
                figsize=figsize,
            )
    
    if args.plot_type in ['pmon', 'both']:
        output = args.output
        if args.plot_type == 'both' and output:
            output = output.replace('.png', '_pmon.png')
        
        if args.theoretical:
            plot_pmon_vs_attacker_size_theoretical(
                df,
                output_path=output,
                title=args.title if args.plot_type == 'pmon' else None,
                show_individual_points=show_points,
                figsize=figsize,
            )
        else:
            plot_pmon_vs_attacker_size(
                df,
                output_path=output,
                title=args.title if args.plot_type == 'pmon' else None,
                show_individual_points=show_points,
                figsize=figsize,
            )


if __name__ == '__main__':
    main()