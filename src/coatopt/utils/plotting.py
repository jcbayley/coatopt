"""Shared plotting utilities for CoatOpt experiments."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pareto_front(
    df: pd.DataFrame,
    objectives: list,
    save_dir: Union[str, Path],
    plot_type: str = "vals",
    algorithm_name: Optional[str] = None,
):
    """Plot Pareto front in either vals or rewards space.

    Args:
        df: DataFrame with objective columns. Expects either:
            - Simple columns: objective names (e.g., 'reflectivity', 'absorption')
            - Full columns: {param}_val and {param}_reward columns
        objectives: List of objective names (e.g., ['reflectivity', 'absorption'])
        save_dir: Directory to save plot
        plot_type: Either "vals" (physical values) or "rewards" (log-transformed rewards)
        algorithm_name: Optional algorithm name for title
    """
    save_dir = Path(save_dir)
    fig, ax = plt.subplots(figsize=(10, 8))

    obj1, obj2 = objectives[0], objectives[1]

    # Handle both simple and full column naming
    if plot_type == "vals":
        # Try full naming first, fall back to simple
        if f"{obj1}_val" in df.columns:
            x = df[f"{obj1}_val"].values
            y = df[f"{obj2}_val"].values
        else:
            x = df[obj1].values
            y = df[obj2].values

        # Handle reflectivity specially
        if obj1 == "reflectivity":
            x = 1 - x  # Plot as loss
            xlabel = "1 - Reflectivity (Loss)"
        else:
            xlabel = obj1.replace("_", " ").title()

        if obj2 == "absorption":
            ylabel = "Absorption (ppm)"
        elif obj2 == "thermal_noise":
            ylabel = "Thermal Noise (m/√Hz)"
        else:
            ylabel = obj2.replace("_", " ").title()

        color = "red"
        filename = "pareto_front_vals.png"
        title_suffix = "Physical Values"

    elif plot_type == "rewards":
        # Try full naming first, fall back to simple
        if f"{obj1}_reward" in df.columns:
            x = df[f"{obj1}_reward"].values
            y = df[f"{obj2}_reward"].values
        else:
            # Assume columns are rewards if plot_type is rewards
            x = df[obj1].values
            y = df[obj2].values

        xlabel = f"{obj1.replace('_', ' ').title()} Reward"
        ylabel = f"{obj2.replace('_', ' ').title()} Reward"

        color = "blue"
        filename = "pareto_front_rewards.png"
        title_suffix = "Rewards"
    else:
        raise ValueError(f"plot_type must be 'vals' or 'rewards', got {plot_type}")

    ax.scatter(x, y, c=color, s=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Build title
    title_parts = []
    if algorithm_name:
        title_parts.append(algorithm_name.upper())
    title_parts.extend(["Pareto Front", f"- {title_suffix}", f"({len(df)} solutions)"])
    ax.set_title(" ".join(title_parts))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()

    return save_dir / filename


def plot_coating_stack(
    thicknesses: np.ndarray,
    material_indices: np.ndarray,
    materials: dict,
    save_path: Union[str, Path] = None,
    title: str = "Coating Stack Design",
    ax: Optional[plt.Axes] = None,
    convert_to_nm: bool = True,
):
    """Plot coating stack design as a vertical bar chart.

    Args:
        thicknesses: Array of layer thicknesses (in meters if convert_to_nm=True, else as-is)
        material_indices: Array of material indices for each layer
        materials: Dict mapping material index to material properties (must have 'name' field)
        save_path: Path to save the plot (if None and ax is None, returns figure)
        title: Plot title
        ax: Optional matplotlib axis to plot on (for subplots)
        convert_to_nm: Whether to convert thicknesses from meters to nm
    """
    # Create figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
        own_fig = True
    else:
        own_fig = False

    # Handle thickness conversion
    if convert_to_nm:
        # Check if optical thickness (values ~0.1-1.0) or physical (values ~1e-8 to 1e-6)
        if len(thicknesses) > 0 and thicknesses[thicknesses > 0].max() < 10:
            # Optical thickness, keep as-is
            thicknesses_nm = thicknesses
            ylabel = "Optical Thickness"
        else:
            # Physical thickness, convert to nm
            thicknesses_nm = thicknesses * 1e9
            ylabel = "Thickness (nm)"
    else:
        thicknesses_nm = thicknesses
        ylabel = "Thickness (nm)"

    # Material colors (extend as needed)
    colors = {
        0: "lightgray",  # Air
        1: "steelblue",  # Substrate
        2: "coral",
        3: "mediumseagreen",
        4: "gold",
        5: "mediumpurple",
        6: "salmon",
        7: "lightskyblue",
    }

    # Plot stack from bottom to top
    y_pos = 0
    plotted_materials = set()

    for i, (thickness, mat_idx) in enumerate(zip(thicknesses_nm, material_indices)):
        if thickness < 1e-3:  # Skip very thin layers (likely air)
            continue

        color = colors.get(mat_idx, "gray")
        mat_name = materials.get(mat_idx, {}).get("name", f"Material {mat_idx}")

        # Only add label for first occurrence of each material
        label = mat_name if mat_idx not in plotted_materials else ""
        if label:
            plotted_materials.add(mat_idx)

        ax.bar(
            0,
            thickness,
            bottom=y_pos,
            width=0.6,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        y_pos += thickness

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)

    # Legend with unique materials only
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc="upper right")

    # Save if we created our own figure and save_path provided
    if own_fig:
        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            return ax.figure

    return ax


def plot_coating_stack_from_state_array(
    state_array: np.ndarray,
    materials: dict,
    save_path: Union[str, Path] = None,
    title: str = "Coating Stack Design",
    ax: Optional[plt.Axes] = None,
):
    """Plot coating stack from state array (one-hot encoded format).

    Args:
        state_array: State array with shape (n_layers, n_materials + 1)
                     Column 0: thickness, Columns 1+: one-hot material encoding
        materials: Dict mapping material index to material properties
        save_path: Path to save the plot
        title: Plot title
        ax: Optional matplotlib axis to plot on
    """
    # Extract thicknesses and filter active layers
    thicknesses = state_array[:, 0]
    active_mask = thicknesses > 1e-12
    active_thicknesses = thicknesses[active_mask]

    # Get material indices from one-hot encoding
    material_onehot = state_array[active_mask, 1:]
    material_indices = np.argmax(material_onehot, axis=1)

    return plot_coating_stack(
        thicknesses=active_thicknesses,
        material_indices=material_indices,
        materials=materials,
        save_path=save_path,
        title=title,
        ax=ax,
        convert_to_nm=True,
    )
