#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coatopt.environments.state import CoatingState
from coatopt.utils.metrics import compute_hypervolume


def compute_hypervolume_from_df(df: pd.DataFrame, space: str = "reward") -> float:
    """Compute hypervolume from a Pareto front DataFrame.

    Args:
        df: DataFrame with either value space (reflectivity, absorption) or
            reward space (reflectivity_reward, absorption_reward) columns
        space: "reward" or "value"

    Returns:
        Hypervolume value (float)
    """
    if space == "reward":
        if (
            "reflectivity_reward" not in df.columns
            or "absorption_reward" not in df.columns
        ):
            return 0.0
        points = df[["absorption_reward", "reflectivity_reward"]].values
        ref_point = np.array([0.0, 0.0])  # Worst case in reward space
        return compute_hypervolume(points, ref_point, maximize=True)
    else:
        if "reflectivity" not in df.columns or "absorption" not in df.columns:
            return 0.0
        points = df[["absorption", "reflectivity"]].values
        # For value space with mixed objectives: absorption minimize, reflectivity maximize
        # Use a reference point that's worse than all points
        ref_point = np.array(
            [np.max(points[:, 0]) * 1.1, 0.0]
        )  # Worst absorption, worst reflectivity
        # Transform for mixed objectives
        from coatopt.utils.metrics import compute_hypervolume_mixed

        objective_directions = [
            False,
            True,
        ]  # absorption: minimize, reflectivity: maximize
        return compute_hypervolume_mixed(points, ref_point, objective_directions)


def load_pareto_front(
    directory: Path, label: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """Load Pareto fronts (value and reward space) from a directory.

    Args:
        directory: Path to directory containing pareto_front_values.csv and pareto_front_rewards.csv
        label: Optional custom label

    Returns:
        Tuple of (values_df, rewards_df, label)
    """
    values_path = directory / "pareto_front_values.csv"
    rewards_path = directory / "pareto_front_rewards.csv"
    legacy_path = directory / "pareto_front.csv"

    if label is None:
        label = directory.name

    values_df = None
    rewards_df = None

    # Try loading value space Pareto front
    if values_path.exists():
        values_df = pd.read_csv(values_path)
    elif legacy_path.exists():
        # Fallback to legacy pareto_front.csv (assume it's value space)
        values_df = pd.read_csv(legacy_path)

    # Try loading reward space Pareto front
    if rewards_path.exists():
        rewards_df = pd.read_csv(rewards_path)

    if values_df is None and rewards_df is None:
        raise FileNotFoundError(f"No pareto_front CSV files found in {directory}")

    return values_df, rewards_df, label


def create_reference_data(
    env, materials: dict, n_layers: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create reference designs transitioning from SiO2/aSi to SiO2/Ti:Ta2O5.

    Creates two sets of designs:
    - Starting with SiO2 (layer 0 = SiO2)
    - Starting with high-index material (layer 0 = aSi/Ti:Ta2O5)

    Args:
        env: CoatingEnvironment to compute values/rewards
        materials: Materials dictionary
        n_layers: Number of layers (must be even)

    Returns:
        Tuple of (values_df, rewards_df)
    """

    n_high_index = n_layers // 2
    values_list = []
    rewards_list = []

    # Material IDs: SiO2=1, Ti:Ta2O5=2, aSi=3
    avg_thickness = 0.25
    start_with_sio2 = True

    for n_ti in range(n_high_index + 1):
        # Create two state arrays with different starting materials
        state_array = np.zeros((n_layers, 2))

        for i in range(n_layers):
            # Determine if this layer should be SiO2 or high-index
            is_sio2_layer = (i % 2 == 0) if start_with_sio2 else (i % 2 == 1)

            state_array[i, 0] = avg_thickness

            if is_sio2_layer:
                # SiO2 layer
                state_array[i, 1] = 1  # SiO2
            else:
                # High-index layer: transition from aSi to Ti:Ta2O5
                layer_idx = i // 2
                if layer_idx < n_ti:
                    state_array[i, 1] = 3  # aSi
                else:
                    state_array[i, 1] = 2  # Ti:Ta2O5

        # Compute values and rewards for this design
        coating_state = CoatingState.from_array(
            state_array,
            len(materials),
            env.air_material_index,
            env.substrate_material_index,
            materials,
        )

        # Use compute_reward to get both values and rewards
        reflectivity, thermal_noise, absorption, thickness = env.compute_state_value(
            coating_state
        )
        rewards, vals = env.compute_reward(coating_state, normalised=True)
        print(vals["reflectivity"], vals["absorption"])

        # Extract thicknesses and materials for plotting
        thicknesses = state_array[:, 0]
        material_indices = state_array[:, 1].astype(int)

        values_list.append(
            {
                "reflectivity": reflectivity,
                "absorption": absorption,
                "thermal_noise": thermal_noise if thermal_noise is not None else 0.0,
                "n_layers": n_layers,
                "thicknesses": ",".join(f"{t:.6f}" for t in thicknesses),
                "materials": ",".join(map(str, material_indices)),
            }
        )
        rewards_list.append({f"{k}_reward": v for k, v in rewards.items()})

    values_df = pd.DataFrame(values_list) if values_list else None
    rewards_df = pd.DataFrame(rewards_list) if rewards_list else None

    print(rewards_df)

    return values_df, rewards_df


def plot_both_spaces_comparison(
    pareto_fronts: List[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]],
    save_path: Optional[Path] = None,
    title: str = "Pareto Front Comparison",
    figsize: Tuple[int, int] = (18, 7),
    reference_values: Optional[pd.DataFrame] = None,
    reference_rewards: Optional[pd.DataFrame] = None,
):
    """Plot both VALUE and REWARD space Pareto fronts side by side.

    Args:
        pareto_fronts: List of (values_df, rewards_df, label) tuples
        save_path: Path to save plot
        title: Overall plot title
        figsize: Figure size
    """
    fig, (ax_values, ax_rewards) = plt.subplots(1, 2, figsize=figsize)
    # Use matplotlib's default color cycle for better color separation
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        default_colors[i % len(default_colors)] for i in range(len(pareto_fronts))
    ]

    # Add reference points to VALUE space if provided
    if (
        reference_values is not None
        and "reflectivity" in reference_values.columns
        and "absorption" in reference_values.columns
    ):
        x_ref = reference_values["absorption"].values
        y_ref = reference_values["reflectivity"].values
        y_ref_loss = 1 - y_ref
        valid_ref = ~(
            np.isnan(x_ref)
            | np.isnan(y_ref_loss)
            | np.isinf(x_ref)
            | np.isinf(y_ref_loss)
            | (y_ref_loss <= 0)
        )
        ax_values.scatter(
            x_ref[valid_ref],
            y_ref_loss[valid_ref],
            marker="x",
            s=400,
            color="black",
            edgecolor="black",
            linewidth=3.0,
            label="Reference",
            zorder=100,
            alpha=0.7,
        )

    for i, (values_df, rewards_df, label) in enumerate(pareto_fronts):
        if (
            values_df is None
            or "reflectivity" not in values_df.columns
            or "absorption" not in values_df.columns
        ):
            print(f"Warning: {label} missing value space data, skipping value plot")
            continue

        # Compute hypervolume for value space
        hv_value = compute_hypervolume_from_df(values_df, space="value")
        label_with_hv = f"{label} (HV: {hv_value:.4f})" if hv_value > 0 else label

        x = values_df["absorption"].values
        y = values_df["reflectivity"].values

        # Convert to 1-reflectivity (loss)
        y_loss = 1 - y

        # Remove invalid values
        valid = ~(
            np.isnan(x)
            | np.isnan(y_loss)
            | np.isinf(x)
            | np.isinf(y_loss)
            | (y_loss <= 0)
        )
        x, y_loss = x[valid], y_loss[valid]

        if len(x) == 0:
            continue

        # Sort by x
        sorted_idx = np.argsort(x)
        x_sorted, y_loss_sorted = x[sorted_idx], y_loss[sorted_idx]

        ax_values.scatter(
            x,
            y_loss,
            color=colors[i],
            s=80,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label=label_with_hv,
            zorder=10 + i,
        )
        ax_values.plot(
            x_sorted,
            y_loss_sorted,
            color=colors[i],
            alpha=0.4,
            linewidth=2,
            linestyle="--",
            zorder=5 + i,
        )

    ax_values.set_xlabel("Absorption (ppm)", fontsize=12)
    ax_values.set_ylabel("1 - Reflectivity", fontsize=12)
    ax_values.set_title("VALUE Space", fontsize=13, fontweight="bold")
    ax_values.set_xscale("log")
    ax_values.set_yscale("log")  # Log scale for 1-reflectivity
    ax_values.grid(True, alpha=0.3, linestyle="--")
    ax_values.legend(loc="best", fontsize=9, framealpha=0.9)

    # reward plots

    # Add reference points to REWARD space if provided
    if (
        reference_rewards is not None
        and "reflectivity_reward" in reference_rewards.columns
        and "absorption_reward" in reference_rewards.columns
    ):
        x_ref = reference_rewards["absorption_reward"].values
        y_ref = reference_rewards["reflectivity_reward"].values
        valid_ref = ~(
            np.isnan(x_ref) | np.isnan(y_ref) | np.isinf(x_ref) | np.isinf(y_ref)
        )
        ax_rewards.scatter(
            x_ref[valid_ref],
            y_ref[valid_ref],
            marker="x",
            s=400,
            color="black",
            edgecolor="black",
            linewidth=3.0,
            label="Reference",
            zorder=100,
            alpha=0.7,
        )

    for i, (values_df, rewards_df, label) in enumerate(pareto_fronts):
        if (
            rewards_df is None
            or "reflectivity_reward" not in rewards_df.columns
            or "absorption_reward" not in rewards_df.columns
        ):
            print(f"Warning: {label} missing reward space data, skipping reward plot")
            continue

        # Compute hypervolume for reward space
        hv_reward = compute_hypervolume_from_df(rewards_df, space="reward")
        label_with_hv = f"{label} (HV: {hv_reward:.4f})" if hv_reward > 0 else label

        x = rewards_df["absorption_reward"].values
        y = rewards_df["reflectivity_reward"].values

        # Remove invalid values
        valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x, y = x[valid], y[valid]

        if len(x) == 0:
            continue

        # Sort by x
        sorted_idx = np.argsort(x)
        x_sorted, y_sorted = x[sorted_idx], y[sorted_idx]

        ax_rewards.scatter(
            x,
            y,
            color=colors[i],
            s=80,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label=label_with_hv,
            zorder=10 + i,
        )
        ax_rewards.plot(
            x_sorted,
            y_sorted,
            color=colors[i],
            alpha=0.4,
            linewidth=2,
            linestyle="--",
            zorder=5 + i,
        )

    ax_rewards.set_xlabel("Absorption Reward (normalized)", fontsize=12)
    ax_rewards.set_ylabel("Reflectivity Reward (normalized)", fontsize=12)
    ax_rewards.set_title("REWARD Space", fontsize=13, fontweight="bold")
    ax_rewards.grid(True, alpha=0.3, linestyle="--")
    ax_rewards.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved combined comparison plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_pareto_comparison(
    pareto_fronts: List[Tuple[pd.DataFrame, str]],
    save_path: Optional[Path] = None,
    obj_x: str = "absorption",
    obj_y: str = "reflectivity",
    title: str = "Pareto Front Comparison",
    xscale: str = "log",
    yscale: str = "linear",
    figsize: Tuple[int, int] = (12, 8),
):
    """Plot multiple Pareto fronts on the same axes.

    Args:
        pareto_fronts: List of (DataFrame, label) tuples
        save_path: Path to save plot (if None, displays plot)
        obj_x: Objective to plot on x-axis
        obj_y: Objective to plot on y-axis
        title: Plot title
        xscale: X-axis scale ('log' or 'linear')
        yscale: Y-axis scale ('log' or 'linear')
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use matplotlib's default color cycle for better color separation
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        default_colors[i % len(default_colors)] for i in range(len(pareto_fronts))
    ]

    for i, (df, label) in enumerate(pareto_fronts):
        # Extract objectives
        if obj_x not in df.columns or obj_y not in df.columns:
            print(f"Warning: {label} missing {obj_x} or {obj_y} columns, skipping")
            continue

        x = df[obj_x].values
        y = df[obj_y].values

        # Remove NaN or invalid values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) == 0:
            print(f"Warning: {label} has no valid data points")
            continue

        # Sort by x for line plot
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Plot points and connecting line
        ax.scatter(
            x,
            y,
            color=colors[i],
            s=80,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=10 + i,
        )
        ax.plot(
            x_sorted,
            y_sorted,
            color=colors[i],
            alpha=0.4,
            linewidth=2,
            linestyle="--",
            zorder=5 + i,
        )

    # Axis labels
    xlabel = obj_x.replace("_", " ").title()
    ylabel = obj_y.replace("_", " ").title()

    # Special formatting for common objectives
    if obj_x == "absorption":
        xlabel = "Absorption (ppm)"
    if obj_y == "reflectivity":
        ylabel = "Reflectivity"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set scales
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_normalized_comparison(
    pareto_fronts: List[Tuple[pd.DataFrame, str]],
    save_path: Optional[Path] = None,
    obj_x: str = "absorption",
    obj_y: str = "reflectivity",
    title: str = "Pareto Front Comparison (Normalized)",
    figsize: Tuple[int, int] = (12, 8),
):
    """Plot Pareto fronts with normalized objectives [0, 1].

    Normalizes each objective to [0, 1] based on min/max across ALL fronts.

    Args:
        pareto_fronts: List of (DataFrame, label) tuples
        save_path: Path to save plot
        obj_x: Objective for x-axis
        obj_y: Objective for y-axis
        title: Plot title
        figsize: Figure size
    """
    # Collect all values to determine global min/max
    all_x = []
    all_y = []
    for df, _ in pareto_fronts:
        if obj_x in df.columns and obj_y in df.columns:
            x = df[obj_x].values
            y = df[obj_y].values
            valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            all_x.extend(x[valid])
            all_y.extend(y[valid])

    if not all_x or not all_y:
        print("No valid data for normalized plot")
        return

    # Determine normalization bounds
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    # For reflectivity (maximize), invert normalization to keep high values high
    # For absorption (minimize), keep low values high
    def normalize_x(val):
        if x_max == x_min:
            return 0.5
        # Absorption: lower is better, so invert
        if obj_x == "absorption":
            return 1.0 - (val - x_min) / (x_max - x_min)
        else:
            return (val - x_min) / (x_max - x_min)

    def normalize_y(val):
        if y_max == y_min:
            return 0.5
        # Reflectivity: higher is better
        if obj_y == "reflectivity":
            return (val - y_min) / (y_max - y_min)
        else:
            return 1.0 - (val - y_min) / (y_max - y_min)

    fig, ax = plt.subplots(figsize=figsize)
    # Use matplotlib's default color cycle for better color separation
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [
        default_colors[i % len(default_colors)] for i in range(len(pareto_fronts))
    ]

    for i, (df, label) in enumerate(pareto_fronts):
        if obj_x not in df.columns or obj_y not in df.columns:
            continue

        x = df[obj_x].values
        y = df[obj_y].values
        valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))

        x_norm = np.array([normalize_x(v) for v in x[valid]])
        y_norm = np.array([normalize_y(v) for v in y[valid]])

        # Sort for line plot
        sorted_indices = np.argsort(x_norm)
        x_sorted = x_norm[sorted_indices]
        y_sorted = y_norm[sorted_indices]

        ax.scatter(
            x_norm,
            y_norm,
            color=colors[i],
            s=80,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=10 + i,
        )
        ax.plot(
            x_sorted,
            y_sorted,
            color=colors[i],
            alpha=0.4,
            linewidth=2,
            linestyle="--",
            zorder=5 + i,
        )

    # Labels
    xlabel_norm = f"{obj_x.replace('_', ' ').title()} (normalized)"
    ylabel_norm = f"{obj_y.replace('_', ' ').title()} (normalized)"

    ax.set_xlabel(xlabel_norm, fontsize=12)
    ax.set_ylabel(ylabel_norm, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved normalized comparison plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def print_statistics(pareto_fronts: List[Tuple[pd.DataFrame, str]]):
    """Print statistics for each Pareto front (legacy function for value space only).

    Args:
        pareto_fronts: List of (DataFrame, label) tuples
    """
    print("\n" + "=" * 80)
    print("PARETO FRONT STATISTICS")
    print("=" * 80)

    for df, label in pareto_fronts:
        print(f"\n{label}:")
        print(f"  Number of points: {len(df)}")

        # Statistics for each objective
        for col in df.columns:
            if col in ["reflectivity", "absorption", "thermal_noise"]:
                values = df[col].dropna()
                if len(values) > 0:
                    print(f"  {col}:")
                    print(f"    Min:  {values.min():.6e}")
                    print(f"    Max:  {values.max():.6e}")
                    print(f"    Mean: {values.mean():.6e}")
                    print(f"    Std:  {values.std():.6e}")


def print_statistics_both_spaces(
    pareto_fronts: List[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]]
):
    """Print statistics for both VALUE and REWARD space Pareto fronts.

    Args:
        pareto_fronts: List of (values_df, rewards_df, label) tuples
    """
    print("\n" + "=" * 80)
    print("PARETO FRONT STATISTICS (VALUE + REWARD SPACE)")
    print("=" * 80)

    for values_df, rewards_df, label in pareto_fronts:
        print(f"\n{label}:")

        # VALUE SPACE statistics
        if values_df is not None:
            hv_value = compute_hypervolume_from_df(values_df, space="value")
            print(f"  VALUE SPACE:")
            print(f"    Number of points: {len(values_df)}")
            print(f"    Hypervolume: {hv_value:.6f}")
            for col in ["reflectivity", "absorption", "thermal_noise"]:
                if col in values_df.columns:
                    values = values_df[col].dropna()
                    if len(values) > 0:
                        print(f"    {col}:")
                        print(f"      Min:  {values.min():.6e}")
                        print(f"      Max:  {values.max():.6e}")
                        print(f"      Mean: {values.mean():.6e}")
        else:
            print(f"  VALUE SPACE: No data")

        # REWARD SPACE statistics
        if rewards_df is not None:
            hv_reward = compute_hypervolume_from_df(rewards_df, space="reward")
            print(f"  REWARD SPACE:")
            print(f"    Number of points: {len(rewards_df)}")
            print(f"    Hypervolume: {hv_reward:.6f}")
            for col in ["reflectivity_reward", "absorption_reward"]:
                if col in rewards_df.columns:
                    values = rewards_df[col].dropna()
                    if len(values) > 0:
                        obj_name = col.replace("_reward", "")
                        print(f"    {obj_name} reward:")
                        print(f"      Min:  {values.min():.6f}")
                        print(f"      Max:  {values.max():.6f}")
                        print(f"      Mean: {values.mean():.6f}")
        else:
            print(f"  REWARD SPACE: No data")


def plot_coating_designs(
    pareto_fronts: List[Tuple[pd.DataFrame, pd.DataFrame, str]],
    materials: dict,
    max_designs: int = 10,
    sort_by: str = "reflectivity",
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
):
    """Plot coating designs from Pareto fronts, sorted by objective value.

    Args:
        pareto_fronts: List of (values_df, rewards_df, label) tuples
        materials: Dictionary mapping material indices to material info
        max_designs: Maximum number of designs to plot per front
        sort_by: Objective to sort by ('reflectivity' or 'absorption')
        figsize: Figure size (width, height)
        save_path: Path to save figure
    """
    from matplotlib.patches import Rectangle

    # Material color map
    color_map = {
        "SiO2": "#1f77b4",  # Blue
        "Ti:Ta2O5": "#ff7f0e",  # Orange
        "aSi": "#2ca02c",  # Green
        "air": "#d3d3d3",  # Light gray
        "AlGaAs": "#9467bd",  # Purple
        "GaAs": "#8c564b",  # Brown
    }

    n_fronts = len(pareto_fronts)
    if figsize is None:
        figsize = (max_designs * 1.5, n_fronts * 6)

    fig, axes = plt.subplots(n_fronts, max_designs, figsize=figsize, squeeze=False)

    for front_idx, (values_df, rewards_df, label) in enumerate(pareto_fronts):
        if values_df is None or "thicknesses" not in values_df.columns:
            print(f"Warning: {label} missing design data, skipping")
            continue

        # Sort by specified objective
        ascending = sort_by == "absorption"  # Lower is better for absorption
        df_sorted = values_df.sort_values(sort_by, ascending=ascending)

        # Plot up to max_designs
        for design_idx in range(min(max_designs, len(df_sorted))):
            ax = axes[front_idx, design_idx]

            if design_idx >= len(df_sorted):
                ax.axis("off")
                continue

            row = df_sorted.iloc[design_idx]

            # Parse thicknesses and materials
            thicknesses = np.array([float(x) for x in row["thicknesses"].split(",")])
            material_indices = np.array([int(x) for x in row["materials"].split(",")])

            # Convert to nm
            thicknesses_nm = thicknesses * 1e9

            # Plot stack from bottom to top
            cumulative_height = 0
            x_position = 0
            width = 1.0

            for thickness_nm, mat_idx in zip(thicknesses_nm, material_indices):
                # Get material name and color
                mat_name = materials.get(mat_idx, {}).get("name", f"M{mat_idx}")
                color = color_map.get(mat_name, "#cccccc")

                # Draw rectangle
                rect = Rectangle(
                    (x_position, cumulative_height),
                    width,
                    thickness_nm,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.85,
                )
                ax.add_patch(rect)
                cumulative_height += thickness_nm

            # Set axis properties
            ax.set_xlim(-0.1, width + 0.1)
            ax.set_ylim(0, cumulative_height * 1.05)
            ax.set_xticks([])

            # Add title with reflectivity and absorption
            refl = row["reflectivity"]
            absorp = row["absorption"]
            loss = 1 - refl
            title = f"R={refl:.9f}\nL={loss:.6e}\nA={absorp:.6e}"
            ax.set_title(title, fontsize=9, fontweight="bold")

            # Only show y-label on leftmost plots
            if design_idx == 0:
                ax.set_ylabel(f"{label}\nDepth (nm)", fontweight="bold")
            else:
                ax.set_yticklabels([])

            ax.grid(True, alpha=0.2, linestyle="--", axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved coating designs to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare Pareto fronts from multiple optimization runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs
  python compare_pareto_fronts.py --dirs nsga2_output discrete_output

  # Auto-discover and plot all subdirectories
  python compare_pareto_fronts.py --alldirs runs/

  # Custom labels
  python compare_pareto_fronts.py --dirs run1 run2 --labels "NSGA-II" "Discrete PPO"

  # Save to specific location
  python compare_pareto_fronts.py --dirs run1 run2 --output comparison.png

  # Compare different objectives
  python compare_pareto_fronts.py --dirs run1 run2 --obj-x thermal_noise --obj-y reflectivity
        """,
    )

    parser.add_argument(
        "--dirs",
        nargs="+",
        help="Directories containing pareto_front.csv files",
    )
    parser.add_argument(
        "--alldirs",
        type=str,
        default=None,
        help="Parent directory - automatically plots all subdirectories containing Pareto fronts",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Custom labels for each directory (default: use directory names)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison plot (default: pareto_comparison.png)",
    )
    parser.add_argument(
        "--obj-x",
        type=str,
        default="absorption",
        help="Objective for x-axis (default: absorption)",
    )
    parser.add_argument(
        "--obj-y",
        type=str,
        default="reflectivity",
        help="Objective for y-axis (default: reflectivity)",
    )
    parser.add_argument(
        "--xscale",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="X-axis scale (default: log)",
    )
    parser.add_argument(
        "--yscale",
        type=str,
        default="linear",
        choices=["log", "linear"],
        help="Y-axis scale (default: linear)",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Also plot normalized comparison (objectives scaled to [0,1])",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics for each Pareto front",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pareto Front Comparison",
        help="Plot title",
    )
    parser.add_argument(
        "--add-reference",
        action="store_true",
        help="Add reference designs (SiO2/aSi -> SiO2/Ti:Ta2O5 transition)",
    )
    parser.add_argument(
        "--reference-layers",
        type=int,
        default=20,
        help="Number of layers for reference designs (default: 20)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file to initialize environment for reference designs",
    )
    parser.add_argument(
        "--plot-designs",
        action="store_true",
        help="Plot coating layer stack designs",
    )
    parser.add_argument(
        "--max-designs",
        type=int,
        default=10,
        help="Maximum number of designs to plot per front (default: 10)",
    )
    parser.add_argument(
        "--sort-designs-by",
        type=str,
        default="reflectivity",
        choices=["reflectivity", "absorption"],
        help="Objective to sort designs by (default: reflectivity)",
    )

    args = parser.parse_args()

    # Determine which directories to process
    if args.alldirs:
        # Auto-discover subdirectories
        parent_dir = Path(args.alldirs)
        if not parent_dir.exists():
            parser.error(f"Parent directory {parent_dir} does not exist")

        # Find all subdirectories that contain Pareto front files
        discovered_dirs = []
        for subdir in sorted(parent_dir.iterdir()):
            if subdir.is_dir():
                # Check if directory contains any pareto front files
                has_pareto = (
                    (subdir / "pareto_front_values.csv").exists()
                    or (subdir / "pareto_front_rewards.csv").exists()
                    or (subdir / "pareto_front.csv").exists()
                )
                if has_pareto:
                    discovered_dirs.append(str(subdir))

        if not discovered_dirs:
            parser.error(f"No subdirectories with Pareto fronts found in {parent_dir}")

        directories = discovered_dirs
        print(f"Auto-discovered {len(directories)} directories with Pareto fronts:")
        for d in directories:
            print(f"  - {Path(d).name}")
    elif args.dirs:
        directories = args.dirs
    else:
        parser.error("Either --dirs or --alldirs must be specified")

    # Validate inputs
    if args.labels and len(args.labels) != len(directories):
        parser.error(
            f"Number of labels ({len(args.labels)}) must match number of directories ({len(directories)})"
        )

    # Load Pareto fronts (both value and reward space)
    pareto_fronts = []
    pareto_fronts_values_only = []  # For legacy single-space plots

    for i, dir_path in enumerate(directories):
        directory = Path(dir_path)
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist, skipping")
            continue

        label = args.labels[i] if args.labels else None
        try:
            values_df, rewards_df, lbl = load_pareto_front(directory, label)
            pareto_fronts.append((values_df, rewards_df, lbl))

            # Keep values_df for legacy plots if available
            if values_df is not None:
                pareto_fronts_values_only.append((values_df, lbl))
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    if not pareto_fronts:
        print("Error: No valid Pareto fronts found")
        return 1

    # Create reference designs if requested
    reference_values = None
    reference_rewards = None
    if args.add_reference:
        print(f"\nCreating reference designs with {args.reference_layers} layers...")

        import configparser

        from coatopt.environments.environment import CoatingEnvironment
        from coatopt.utils.configs import load_config
        from coatopt.utils.utils import load_materials

        # Try to find config file
        if args.config:
            config_path = Path(args.config)
        else:
            # Try to find in first directory
            config_path = Path(args.dirs[0]) / "config.ini" if args.dirs else None

        if config_path and config_path.exists():
            parser = configparser.ConfigParser()
            parser.read(config_path)
            materials_path = parser.get("general", "materials_path")
            materials = load_materials(materials_path)
            config = load_config(str(config_path))
            env = CoatingEnvironment(config, materials)
            reference_values, reference_rewards = create_reference_data(
                env, materials, args.reference_layers
            )
            print(f"Created {len(reference_values)} reference designs")
            print(reference_values)
        else:
            print(f"Warning: Config file not found. Skipping reference designs.")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif args.alldirs:
        # Save to the alldirs directory
        parent_dir = Path(args.alldirs)
        output_path = parent_dir / "pareto_comparison.png"
    else:
        output_path = Path("pareto_comparison.png")

    # Plot BOTH VALUE and REWARD space comparison (new default)
    print("\nGenerating combined VALUE + REWARD space comparison plot...")
    combined_path = output_path.parent / (
        output_path.stem + "_both_spaces" + output_path.suffix
    )
    plot_both_spaces_comparison(
        pareto_fronts,
        save_path=combined_path,
        title=args.title,
        reference_values=reference_values,
        reference_rewards=reference_rewards,
    )

    # Plot VALUE space only (legacy)
    if pareto_fronts_values_only:
        print("\nGenerating VALUE space comparison plot...")
        plot_pareto_comparison(
            pareto_fronts_values_only,
            save_path=output_path,
            obj_x=args.obj_x,
            obj_y=args.obj_y,
            title=args.title + " (Value Space)",
            xscale=args.xscale,
            yscale=args.yscale,
        )

    # Plot normalized comparison if requested
    if args.normalized and pareto_fronts_values_only:
        norm_path = output_path.parent / (
            output_path.stem + "_normalized" + output_path.suffix
        )
        print("\nGenerating normalized comparison plot...")
        plot_normalized_comparison(
            pareto_fronts_values_only,
            save_path=norm_path,
            obj_x=args.obj_x,
            obj_y=args.obj_y,
            title=args.title + " (Normalized)",
        )

    # Print statistics if requested
    if args.stats:
        print_statistics_both_spaces(pareto_fronts)

    # Plot coating designs if requested
    if args.plot_designs:
        print("\nGenerating coating design plots...")

        # Load materials from config
        materials = {}
        if args.config:
            config_path = Path(args.config)
        elif args.dirs:
            config_path = Path(args.dirs[0]) / "config.ini"
        else:
            print(
                "Warning: No config file found, cannot plot designs. Use --config to specify."
            )
            print("\nDone!")
            return 0

        if config_path.exists():
            import configparser

            from coatopt.utils.utils import load_materials

            parser_cfg = configparser.ConfigParser()
            parser_cfg.read(config_path)
            materials_path = parser_cfg.get("general", "materials_path")
            materials = load_materials(materials_path)

            designs_path = output_path.parent / (
                output_path.stem + "_designs" + output_path.suffix
            )

            # Include reference designs if they exist
            fronts_to_plot = pareto_fronts.copy()
            if reference_values is not None:
                fronts_to_plot.append(
                    (reference_values, reference_rewards, "Reference")
                )

            plot_coating_designs(
                fronts_to_plot,
                materials,
                max_designs=args.max_designs,
                sort_by=args.sort_designs_by,
                save_path=designs_path,
            )
        else:
            print(
                f"Warning: Config file not found at {config_path}, cannot plot designs."
            )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
