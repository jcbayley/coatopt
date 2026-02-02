"""Utility functions for multi-objective optimization metrics."""

from typing import List, Tuple, Optional, Dict
import numpy as np


def dominates(
    obj1: np.ndarray,
    obj2: np.ndarray,
    maximize: bool = True
) -> bool:
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
    obj1: np.ndarray,
    obj2: np.ndarray,
    objective_directions: List[bool]
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
    pareto_front: List[np.ndarray],
    new_point: np.ndarray,
    maximize: bool = True
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
    objective_directions: List[bool]
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
    points: np.ndarray,
    ref_point: np.ndarray,
    maximize: bool = True
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
    points: np.ndarray,
    ref_point: np.ndarray,
    objective_directions: List[bool]
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
