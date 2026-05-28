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
