"""
Plotting utilities for HPPO training.
Extracted from pc_hppo_oml.py for better organization.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Any
from coatopt.algorithms.config import HPPOConstants

def pad_lists(list_of_lists: List[List[Any]], padding_value: Any = 0, max_length: int = None) -> np.ndarray:
    """
    Pad lists to same length with specified padding value.
    
    Args:
        list_of_lists: List of lists to pad
        padding_value: Value to use for padding
        max_length: Maximum length to pad to
        
    Returns:
        Numpy array with padded lists
    """
    if max_length is None:
        max_length = max(len(lst) for lst in list_of_lists)
    padded_lists = []
    for lst in list_of_lists:
        t_lst = []
        for l in lst:
            t_lst.append(l)

        if len(t_lst) < max_length:
            diff = int(max_length - len(t_lst))
            for i in range(diff):
                t_lst.append(padding_value)

        
        padded_lists.append(t_lst)

    #padded_lists = [lst + [padding_value] * (max_length - len(lst)) for lst in list_of_lists]
    return np.array(padded_lists)


def make_reward_plot(metrics: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive reward plots showing training progress.
    
    Args:
        metrics: DataFrame containing training metrics
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(
        nrows=HPPOConstants.PLOT_NROWS_REWARD, 
        figsize=HPPOConstants.PLOT_FIGSIZE_REWARD
    )
    
    window_size = HPPOConstants.WINDOW_SIZE
    
    # Total reward plot
    _plot_metric_with_smoothing(axes[0], metrics, "reward", "Episode number", "Reward", window_size)
    
    # Individual reward components
    _plot_metric_with_smoothing(axes[1], metrics, "reflectivity_reward", "Episode number", "Reflectivity reward", window_size)
    _plot_metric_with_smoothing(axes[2], metrics, "thermal_noise_reward", "Episode number", "Thermal noise reward", window_size)
    _plot_metric_with_smoothing(axes[3], metrics, "thickness_reward", "Episode number", "Thickness reward", window_size)
    _plot_metric_with_smoothing(axes[4], metrics, "absorption_reward", "Episode number", "Absorption reward", window_size)
    
    # Training parameters
    axes[5].plot(metrics["episode"], metrics["beta"])
    axes[5].set_xlabel("Episode number")
    axes[5].set_ylabel("Entropy weight")
    
    # Learning rates
    axes[6].plot(metrics["episode"], metrics["lr_discrete"], label="discrete")
    axes[6].plot(metrics["episode"], metrics["lr_continuous"], label="continuous")
    axes[6].plot(metrics["episode"], metrics["lr_value"], label="value")
    axes[6].set_xlabel("Episode number")
    axes[6].set_ylabel("Learning Rate")
    axes[6].legend()
    
    # Objective weights
    axes[7].plot(metrics["episode"], metrics["reflectivity_reward_weights"], label="reflectivity")
    axes[7].plot(metrics["episode"], metrics["absorption_reward_weights"], label="absorption")
    axes[7].plot(metrics["episode"], metrics["thermalnoise_reward_weights"], label="thermal noise")
    axes[7].set_xlabel("Episode number")
    axes[7].set_ylabel("Objective weighting")
    axes[7].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, HPPOConstants.RUNNING_REWARDS_PLOT))
    plt.close(fig)


def make_val_plot(metrics: pd.DataFrame, output_dir: str) -> None:
    """
    Create value plots showing actual physical values during training.
    
    Args:
        metrics: DataFrame containing training metrics
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(
        nrows=HPPOConstants.PLOT_NROWS_VAL, 
        figsize=HPPOConstants.PLOT_FIGSIZE_VAL
    )
    
    window_size = HPPOConstants.WINDOW_SIZE
    
    # Total reward
    _plot_metric_with_smoothing(axes[0], metrics, "reward", "Episode number", "Reward", window_size)
    
    # Physical values
    _plot_metric_with_smoothing(axes[1], metrics, "reflectivity", "Episode number", "Reflectivity", window_size)
    _plot_metric_with_smoothing(axes[2], metrics, "thermal_noise", "Episode number", "Thermal noise", window_size, log_scale=True)
    _plot_metric_with_smoothing(axes[3], metrics, "thickness", "Episode number", "Thickness", window_size)
    _plot_metric_with_smoothing(axes[4], metrics, "absorption", "Episode number", "Absorption", window_size, log_scale=True)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, HPPOConstants.RUNNING_VALUES_PLOT))
    plt.close(fig)


def make_loss_plot(metrics: pd.DataFrame, output_dir: str) -> None:
    """
    Create loss plots showing training losses.
    
    Args:
        metrics: DataFrame containing training metrics
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(nrows=HPPOConstants.PLOT_NROWS_LOSS)
    
    axes[0].plot(metrics["episode"], metrics["loss_policy_discrete"])
    axes[0].set_ylabel("Policy discrete loss")
    
    axes[1].plot(metrics["episode"], metrics["loss_policy_continuous"])
    axes[1].set_ylabel("Policy continuous loss")
    
    axes[2].plot(metrics["episode"], metrics["loss_value"])
    axes[2].set_ylabel("Value loss")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Episode number")
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, HPPOConstants.RUNNING_LOSSES_PLOT))
    plt.close(fig)


def make_materials_plot(all_materials: List[List[List[float]]], n_layers: int, output_dir: str, n_materials: int) -> None:
    """
    Create materials selection plot showing how material choices evolve.
    
    Args:
        all_materials: List of material probability distributions over episodes
        n_layers: Number of layers in coating
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(nrows=n_layers)
    
    # Pad materials data
    padding_value = [0.0] * n_materials
    all_materials_padded = pad_lists(all_materials, padding_value, n_materials)

    # Ensure itis of correct number of layers
    if all_materials_padded.shape[0] < n_layers:
        all_materials_padded = np.pad(
            all_materials_padded, 
            ((0, n_layers - all_materials_padded.shape[0]), (0, 0)), 
            mode='constant', 
            constant_values=0.0
        )
    
    # Plot material probabilities for each layer
    for layer_idx in range(n_layers):
        for material_idx in range(len(all_materials_padded[layer_idx])):
            axes[layer_idx].scatter(
                np.arange(len(all_materials_padded)), 
                np.ones(len(all_materials_padded)) * material_idx,
                s=100 * all_materials_padded[layer_idx, material_idx], 
                color=HPPOConstants.MATERIAL_COLOR_MAP[material_idx]
            )
        axes[layer_idx].set_ylabel(f"Layer {layer_idx}")
    
    axes[-1].set_xlabel("Episode number")
    #plt.tight_layout()
    fig.savefig(os.path.join(output_dir, HPPOConstants.RUNNING_MATERIALS_PLOT))
    plt.close(fig)


def _plot_metric_with_smoothing(
    ax: plt.Axes, 
    metrics: pd.DataFrame, 
    column: str, 
    xlabel: str, 
    ylabel: str, 
    window_size: int,
    log_scale: bool = False
) -> None:
    """
    Plot metric with smoothed overlay.
    
    Args:
        ax: Matplotlib axis to plot on
        metrics: DataFrame containing metrics
        column: Column name to plot
        xlabel: X-axis label
        ylabel: Y-axis label
        window_size: Window size for smoothing
        log_scale: Whether to use log scale for y-axis
    """
    # Plot raw data
    ax.plot(metrics["episode"], metrics[column], alpha=0.3)
    
    # Plot smoothed data
    smoothed_data = metrics[column].rolling(window=window_size, center=False).median()
    smoothed_episodes = metrics['episode'].rolling(window=window_size, center=False).median()
    ax.plot(smoothed_episodes, smoothed_data, linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.grid(True, alpha=0.3)