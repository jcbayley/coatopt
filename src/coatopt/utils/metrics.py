"""Utility functions for multi-objective optimization metrics."""

from typing import Dict, List, Optional, Tuple

import numpy as np


def dominates(obj1: np.ndarray, obj2: np.ndarray, maximize: bool = True) -> bool:
    """Check if obj1 Pareto dominates obj2.

    Args:
        obj1: First objective vector
        obj2: Second objective vector
        maximize: If True, higher is better for all objectives.
                 If False, lower is better for all objectives.

    Returns:
        True if obj1 dominates obj2
    """
    if len(obj1) != len(obj2):
        return False

    better_or_equal = True
    strictly_better = False

    for i in range(len(obj1)):
        if maximize:
            # Higher is better
            if obj1[i] < obj2[i]:
                better_or_equal = False
                break
            elif obj1[i] > obj2[i]:
                strictly_better = True
        else:
            # Lower is better
            if obj1[i] > obj2[i]:
                better_or_equal = False
                break
            elif obj1[i] < obj2[i]:
                strictly_better = True

    return better_or_equal and strictly_better


def dominates_mixed(
    obj1: np.ndarray, obj2: np.ndarray, objective_directions: List[bool]
) -> bool:
    """Check if obj1 Pareto dominates obj2 with mixed objectives.

    Args:
        obj1: First objective vector
        obj2: Second objective vector
        objective_directions: List where True = maximize, False = minimize for each objective

    Returns:
        True if obj1 dominates obj2
    """
    if len(obj1) != len(obj2) or len(obj1) != len(objective_directions):
        return False

    better_or_equal = True
    strictly_better = False

    for i, maximize in enumerate(objective_directions):
        if maximize:
            # Higher is better
            if obj1[i] < obj2[i]:
                better_or_equal = False
                break
            elif obj1[i] > obj2[i]:
                strictly_better = True
        else:
            # Lower is better
            if obj1[i] > obj2[i]:
                better_or_equal = False
                break
            elif obj1[i] < obj2[i]:
                strictly_better = True

    return better_or_equal and strictly_better


def update_pareto_front(
    pareto_front: List[np.ndarray], new_point: np.ndarray, maximize: bool = True
) -> List[np.ndarray]:
    """Update Pareto front with a new point.

    Args:
        pareto_front: Current Pareto front (list of objective vectors)
        new_point: New objective vector to consider
        maximize: If True, all objectives are maximized. If False, all minimized.

    Returns:
        Updated Pareto front (list of objective vectors)
    """
    # Check if new point is dominated by any existing point
    for existing_point in pareto_front:
        if dominates(existing_point, new_point, maximize=maximize):
            # New point is dominated, don't add it
            return pareto_front

    # New point is not dominated, remove points it dominates
    updated_front = []
    for existing_point in pareto_front:
        if not dominates(new_point, existing_point, maximize=maximize):
            # Existing point is not dominated by new point, keep it
            updated_front.append(existing_point)

    # Add new point
    updated_front.append(new_point)
    return updated_front


def update_pareto_front_mixed(
    pareto_front: List[np.ndarray],
    new_point: np.ndarray,
    objective_directions: List[bool],
) -> List[np.ndarray]:
    """Update Pareto front with mixed objectives.

    Args:
        pareto_front: Current Pareto front (list of objective vectors)
        new_point: New objective vector to consider
        objective_directions: List where True = maximize, False = minimize for each objective

    Returns:
        Updated Pareto front (list of objective vectors)
    """
    # Check if new point is dominated by any existing point
    for existing_point in pareto_front:
        if dominates_mixed(existing_point, new_point, objective_directions):
            return pareto_front

    # New point is not dominated, remove points it dominates
    updated_front = []
    for existing_point in pareto_front:
        if not dominates_mixed(new_point, existing_point, objective_directions):
            updated_front.append(existing_point)

    # Add new point
    updated_front.append(new_point)
    return updated_front


def compute_hypervolume(
    points: np.ndarray, ref_point: np.ndarray, maximize: bool = True
) -> float:
    """Compute hypervolume indicator for a set of points.

    Uses pymoo's hypervolume indicator. Pymoo expects all objectives to be minimized,
    so we transform the points if maximize=True.

    Args:
        points: Array of shape (n_points, n_objectives)
        ref_point: Reference point for hypervolume (shape: n_objectives)
        maximize: If True, negate points for maximization. If False, use as-is for minimization.

    Returns:
        Hypervolume value (float). Returns 0.0 if no points or pymoo unavailable.
    """
    if len(points) == 0:
        return 0.0

    try:
        from pymoo.indicators.hv import HV
    except ImportError:
        return 0.0

    points = np.array(points)
    ref_point = np.array(ref_point)

    # Pymoo expects minimization, so negate if maximizing
    if maximize:
        points = -points
        ref_point = -ref_point

    ind = HV(ref_point=ref_point)
    hv = ind(points)

    return float(hv)


def compute_hypervolume_mixed(
    points: np.ndarray, ref_point: np.ndarray, objective_directions: List[bool]
) -> float:
    """Compute hypervolume with mixed objectives (some maximize, some minimize).

    Args:
        points: Array of shape (n_points, n_objectives)
        ref_point: Reference point for hypervolume
        objective_directions: List where True = maximize, False = minimize for each objective

    Returns:
        Hypervolume value (float). Returns 0.0 if no points or pymoo unavailable.
    """
    if len(points) == 0:
        return 0.0

    try:
        from pymoo.indicators.hv import HV
    except ImportError:
        return 0.0

    points = np.array(points)
    ref_point = np.array(ref_point)

    # Transform points and ref_point: negate objectives that are maximized
    transformed_points = []
    for point in points:
        transformed = []
        for i, maximize in enumerate(objective_directions):
            if maximize:
                transformed.append(-point[i])
            else:
                transformed.append(point[i])
        transformed_points.append(transformed)

    transformed_ref = []
    for i, maximize in enumerate(objective_directions):
        if maximize:
            transformed_ref.append(-ref_point[i])
        else:
            transformed_ref.append(ref_point[i])

    points = np.array(transformed_points)
    ref_point = np.array(transformed_ref)

    ind = HV(ref_point=ref_point)
    hv = ind(points)

    return float(hv)


def save_pareto_to_csv(env, filename: str = "pareto_front.csv", save_dir=None):
    """Save Pareto front from environment to CSV files.

    Args:
        env: Environment with get_pareto_front method
        filename: Name of CSV file to save (can be absolute, relative to cwd, or just a filename)
        save_dir: Directory to save to. If None and filename is relative, saves to cwd.
    """
    from pathlib import Path

    import pandas as pd

    # Build filepath
    filepath = Path(filename)
    # Only prepend save_dir if filename is just a filename (no directory components) and save_dir is provided
    if save_dir and not filepath.is_absolute() and filepath.parent == Path("."):
        filepath = Path(save_dir) / filename

    # Unwrap environment if needed (e.g., VecEnv wrappers)
    if hasattr(env, "env"):
        env = env.env

    # Get both value and reward space Pareto fronts
    pareto_front_values = (
        env.get_pareto_front(space="value") if hasattr(env, "get_pareto_front") else []
    )
    pareto_front_rewards = (
        env.get_pareto_front(space="reward") if hasattr(env, "get_pareto_front") else []
    )

    if not pareto_front_values:
        print("No Pareto front data to save")
        return

    # Convert to CSV format (value space)
    data = []
    for obj_vector, state in pareto_front_values:
        row = {}

        # Add objective values
        if hasattr(env, "optimise_parameters"):
            for i, param_name in enumerate(env.optimise_parameters):
                if i < len(obj_vector):
                    row[param_name] = obj_vector[i]

        # Add state information
        state_array = state.get_array()
        thicknesses = state_array[:, 0]
        material_indices = state_array[:, 1].astype(int)

        # Filter active layers
        active_mask = thicknesses > 1e-12
        active_thicknesses = thicknesses[active_mask]
        active_materials = material_indices[active_mask]

        row["n_layers"] = len(active_thicknesses)
        row["thicknesses"] = ",".join(f"{t:.6f}" for t in active_thicknesses)
        row["materials"] = ",".join(map(str, active_materials))

        data.append(row)

    df = pd.DataFrame(data)

    # Sort by absorption if available (lower is better)
    if "absorption" in df.columns:
        df = df.sort_values("absorption", ascending=True)
    elif "reflectivity" in df.columns:
        df = df.sort_values("reflectivity", ascending=False)

    # Save value space
    values_path = filepath.parent / "pareto_front_values.csv"
    df.to_csv(values_path, index=False)

    # Save reward space
    data_rewards = []
    for obj_vector, state in pareto_front_rewards:
        row = {}
        if hasattr(env, "optimise_parameters"):
            for i, param_name in enumerate(env.optimise_parameters):
                if i < len(obj_vector):
                    row[f"{param_name}_reward"] = obj_vector[i]
        data_rewards.append(row)

    if data_rewards:
        df_rewards = pd.DataFrame(data_rewards)
        rewards_path = filepath.parent / "pareto_front_rewards.csv"
        df_rewards.to_csv(rewards_path, index=False)
        print(
            f"Saved Pareto front ({len(df)} designs) to {values_path} and {rewards_path}"
        )
    else:
        print(f"Saved Pareto front ({len(df)} designs) to {values_path}")
