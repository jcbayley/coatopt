#!/usr/bin/env python3
"""
Utility for tracking and plotting multi-objective training progress over time.
Saves a running log in CSV and updates a 4-panel premium visualizer.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode for server compatibility
import matplotlib.pyplot as plt

def update_training_progress_plot(save_dir: Path, episode: int, values_df: pd.DataFrame):
    """
    Appends the best values found in the current Pareto front to training_history.csv
    and updates a 4-panel dashboard of historical bests.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history_csv = save_dir / "training_history.csv"
    
    # Calculate current bests on each target specification
    best_r = values_df['reflectivity'].max() if 'reflectivity' in values_df.columns and len(values_df) > 0 else np.nan
    best_abs = values_df['absorption'].min() if 'absorption' in values_df.columns and len(values_df) > 0 else np.nan
    best_tn = values_df['thermal_noise'].min() if 'thermal_noise' in values_df.columns and len(values_df) > 0 else np.nan
    pareto_sz = len(values_df)
    
    new_row = {
        "episode": episode,
        "best_reflectivity": best_r,
        "best_absorption": best_abs,
        "best_thermal_noise": best_tn,
        "pareto_size": pareto_sz
    }
    
    # Append/Update the running CSV history
    if history_csv.exists():
        try:
            history_df = pd.read_csv(history_csv)
            # Prevent duplicate records if training is resumed or logged at the same checkpoint
            history_df = history_df[history_df['episode'] != episode]
            history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            history_df = pd.DataFrame([new_row])
    else:
        history_df = pd.DataFrame([new_row])
        
    history_df = history_df.sort_values('episode').reset_index(drop=True)
    history_df.to_csv(history_csv, index=False)
    
    if len(history_df) < 2:
        return  # Need at least two points to plot trends
        
    # Render the premium 4-panel progress dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Multi-Objective Training Progress (Episode {episode})", fontsize=16, fontweight='bold', y=0.96)
    
    episodes = history_df['episode'].values
    
    # 1. Top-Left: Reflectivity Loss (1 - R) - Log Scale
    ax = axes[0, 0]
    r_vals = history_df['best_reflectivity'].values
    r_loss = 1.0 - r_vals
    ax.plot(episodes, r_loss, color='#1f77b4', marker='o', ms=4, lw=1.8, label="1 - Best R")
    ax.set_yscale('log')
    ax.set_ylabel("1 - Reflectivity (Loss)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Best Reflectivity Loss (1 - R)", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 2. Top-Right: Minimum Absorption (ppm) - Log Scale
    ax = axes[0, 1]
    abs_ppm = history_df['best_absorption'].values
    ax.plot(episodes, abs_ppm, color='#d62728', marker='o', ms=4, lw=1.8, label="Min Absorption")
    ax.set_yscale('log')
    ax.set_ylabel("Absorption [ppm]", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Minimum Absorption", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 3. Bottom-Left: Minimum Thermal Noise - Log Scale
    ax = axes[1, 0]
    tn_vals = history_df['best_thermal_noise'].values
    ax.plot(episodes, tn_vals, color='#2ca02c', marker='o', ms=4, lw=1.8, label="Min CTN")
    ax.set_yscale('log')
    ax.set_ylabel("Brownian Noise [m/√Hz]", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Minimum Coating Thermal Noise (100Hz)", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 4. Bottom-Right: Pareto Front Size (Proliferation)
    ax = axes[1, 1]
    sizes = history_df['pareto_size'].values
    ax.plot(episodes, sizes, color='#9467bd', marker='s', ms=4, lw=1.8, label="Pareto Size")
    ax.set_ylabel("Pareto Front Size", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Pareto-Optimal Design Count", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plot_path = save_dir / "training_progress.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

def update_training_diagnostics_plot(save_dir: Path, episode: int, diagnostics_data: dict):
    """
    Appends the current diagnostic statistics to training_diagnostics.csv
    and updates a 4-panel diagnostic plot (training_diagnostics.png).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    history_csv = save_dir / "training_diagnostics.csv"
    
    # Create the row dictionary
    new_row = {"episode": episode}
    new_row.update(diagnostics_data)
    
    # Append/Update the running CSV history
    if history_csv.exists():
        try:
            history_df = pd.read_csv(history_csv)
            # Prevent duplicate records if training is resumed or logged at the same checkpoint
            history_df = history_df[history_df['episode'] != episode]
            history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            history_df = pd.DataFrame([new_row])
    else:
        history_df = pd.DataFrame([new_row])
        
    history_df = history_df.sort_values('episode').reset_index(drop=True)
    history_df.to_csv(history_csv, index=False)
    
    if len(history_df) < 2:
        return  # Need at least two points to plot trends
        
    # Render the premium 6-panel diagnostics dashboard
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f"HPPO Training Diagnostics (Episode {episode})", fontsize=16, fontweight='bold', y=0.96)
    
    episodes = history_df['episode'].values
    
    # 1. Top-Left: Policy Entropies
    ax = axes[0, 0]
    if "entropy_discrete" in history_df.columns:
        ax.plot(episodes, history_df['entropy_discrete'].values, color='#1f77b4', lw=1.8, label="Discrete (Material)")
    if "entropy_continuous" in history_df.columns:
        ax.plot(episodes, history_df['entropy_continuous'].values, color='#aec7e8', lw=1.8, label="Continuous (Thickness)")
    ax.set_ylabel("Entropy", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Policy Entropies (Exploration)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 2. Top-Right: Material Diversity
    ax = axes[0, 1]
    if "unique_materials_rollout" in history_df.columns:
        ax.plot(episodes, history_df['unique_materials_rollout'].values, color='#2ca02c', lw=1.8, label="Mean Unique Materials")
    if "three_mat_ratio_rollout" in history_df.columns:
        ax.plot(episodes, history_df['three_mat_ratio_rollout'].values, color='#9467bd', ls="--", lw=1.8, label="3-Mat Ratio (Rollout)")
    if "three_mat_ratio_pareto" in history_df.columns:
        ax.plot(episodes, history_df['three_mat_ratio_pareto'].values, color='#ff7f0e', marker='o', ms=4, lw=1.8, label="3-Mat Ratio (Pareto)")
    ax.set_ylabel("Diversity Metric", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Material Diversity & Pareto Composition", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 3. Middle-Left: Volume Fractions of Materials in Rollout
    ax = axes[1, 0]
    frac_cols = sorted([c for c in history_df.columns if c.startswith("fraction_mat_")])
    colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
    for idx, col in enumerate(frac_cols):
        mat_num = col.split("_")[-1]
        mat_label = f"Mat {mat_num}"
        if mat_num == "1":
            mat_label = "SiO2 (Mat 1)"
        elif mat_num == "2":
            mat_label = "Ti:Ta2O5 (Mat 2)"
        elif mat_num == "3":
            mat_label = "aSi (Mat 3)"
        ax.plot(episodes, history_df[col].values * 100.0, lw=1.8, color=colors[idx % len(colors)], label=mat_label)
    ax.set_ylabel("Thickness Fraction (%)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Rollout Material Composition (Volume %)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 4. Middle-Right: EFI Spatial Alignment
    ax = axes[1, 1]
    if "max_efi_high_loss" in history_df.columns:
        ax.plot(episodes, history_df['max_efi_high_loss'].values, color='#d62728', lw=1.8, label="Peak EFI in High-Loss Mat")
        ax.set_ylabel("Peak Electric Field Intensity (EFI)", color='#d62728', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='#d62728')
        
    if "mean_depth_high_loss" in history_df.columns:
        ax2 = ax.twinx()
        ax2.plot(episodes, history_df['mean_depth_high_loss'].values, color='#9467bd', ls=":", lw=1.8, label="Mean Depth of High-Loss Mat")
        ax2.set_ylabel("Normalized Mean Depth (0=top, 1=substrate)", color='#9467bd', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#9467bd')
        # combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("EFI Shielding & Depth in High-Loss Layer", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 5. Bottom-Left: Mean Total Reward
    ax = axes[2, 0]
    if "reward_mean" in history_df.columns:
        ax.plot(episodes, history_df['reward_mean'].values, color='#8c564b', marker='o', ms=4, lw=1.8, label="Mean Reward (Rollout)")
    ax.set_ylabel("Reward Value", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Mean Rollout Episode Reward", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # 6. Bottom-Right: Pareto Front Hypervolume
    ax = axes[2, 1]
    if "hypervolume" in history_df.columns:
        ax.plot(episodes, history_df['hypervolume'].values, color='#e377c2', marker='d', ms=4, lw=1.8, label="Hypervolume")
    ax.set_ylabel("Hypervolume (Reward Space)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_title("Pareto Front Hypervolume", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    
    plot_path = save_dir / "training_diagnostics.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

