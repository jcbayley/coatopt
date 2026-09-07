"""Utility functions for multi-objective optimization metrics."""

from typing import Any, Dict, List, Optional, Tuple, Union

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


def compute_target_yield(
    df: Any,
    targets: Dict[str, float],
    tolerances: Optional[List[float]] = None,
    primary_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the Target Region Yield Y(alpha) across multiple tolerance levels.

    Tolerances are relative margin relaxation multipliers:
      - Reflectivity (maximize): R >= target_R - alpha * (1 - target_R)
        (Equivalently: loss <= target_loss * (1 + alpha))
      - Transmission (minimize): T <= target_T * (1 + alpha)
      - Absorption (minimize): Abs <= target_Abs * (1 + alpha)
      - Thermal Noise (minimize): TN <= target_TN * (1 + alpha)
      - Thickness (minimize, optional): Thick <= target_Thick * (1 + alpha)

    Args:
        df: pandas DataFrame or dict containing objective columns ('reflectivity',
            'absorption', 'thermal_noise', and optionally 'total_thickness').
        targets: Dictionary with keys 'reflectivity', 'absorption', 'thermal_noise',
            and optionally 'total_thickness'.
        tolerances: List of tolerance multipliers (defaults to [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]).
        primary_metric: 'transmission' or 'reflectivity' (optional).

    Returns:
        Dict containing yield at alpha=0, total designs, yield curve data, and passing indices.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if len(df) == 0:
        return {
            "yield_zero": 0.0,
            "count_zero": 0,
            "total_designs": 0,
            "yield_curve": [],
            "passing_indices_zero": [],
        }

    if tolerances is None:
        tolerances = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

    use_trans = (primary_metric == "transmission") or (
        targets.get("primary_metric") == "transmission"
    ) or (
        "transmission" in df.columns and "transmission" in targets and (
            "reflectivity" not in targets or "reflectivity" not in df.columns or primary_metric != "reflectivity"
        )
    )

    t_abs = float(targets.get("absorption", 0.30))
    t_tn = float(targets.get("thermal_noise", 4.0e-21))

    abs_vals = df["absorption"].values if "absorption" in df.columns else np.zeros(len(df))
    tn_vals = df["thermal_noise"].values if "thermal_noise" in df.columns else np.zeros(len(df))

    yield_curve = []
    passing_indices_zero = []

    if use_trans:
        t_trans = float(targets.get("transmission", 10.0))
        trans_vals = df["transmission"].values
        for alpha in tolerances:
            # Transmission constraint (minimize): T <= target_trans * (1 + alpha)
            trans_thresh = t_trans * (1.0 + alpha)
            abs_thresh = t_abs * (1.0 + alpha)
            tn_thresh = t_tn * (1.0 + alpha)

            mask = (trans_vals <= trans_thresh) & (abs_vals <= abs_thresh) & (tn_vals <= tn_thresh)

            pass_count = int(np.sum(mask))
            pass_pct = float(pass_count / len(df) * 100.0)

            if alpha == 0.0:
                passing_indices_zero = np.where(mask)[0].tolist()

            yield_curve.append({
                "tolerance": float(alpha),
                "tolerance_pct": float(alpha * 100.0),
                "yield_pct": pass_pct,
                "count": pass_count,
            })
    else:
        t_refl = float(targets.get("reflectivity", 0.99999))
        refl_loss_target = max(1e-9, 1.0 - t_refl)
        refl_vals = df["reflectivity"].values if "reflectivity" in df.columns else np.ones(len(df))

        for alpha in tolerances:
            # Reflectivity constraint (maximize): loss <= target_loss * (1 + alpha)
            r_thresh = 1.0 - refl_loss_target * (1.0 + alpha)
            abs_thresh = t_abs * (1.0 + alpha)
            tn_thresh = t_tn * (1.0 + alpha)

            mask = (refl_vals >= r_thresh) & (abs_vals <= abs_thresh) & (tn_vals <= tn_thresh)

            pass_count = int(np.sum(mask))
            pass_pct = float(pass_count / len(df) * 100.0)

            if alpha == 0.0:
                passing_indices_zero = np.where(mask)[0].tolist()

            yield_curve.append({
                "tolerance": float(alpha),
                "tolerance_pct": float(alpha * 100.0),
                "yield_pct": pass_pct,
                "count": pass_count,
            })

    return {
        "yield_zero": yield_curve[0]["yield_pct"] if yield_curve else 0.0,
        "count_zero": yield_curve[0]["count"] if yield_curve else 0,
        "total_designs": len(df),
        "yield_curve": yield_curve,
        "passing_indices_zero": passing_indices_zero,
    }


def compute_objective_breakdown(
    df: Any,
    targets: Dict[str, float],
    primary_metric: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Calculate per-objective pass rates, best values, and margin to target.

    Args:
        df: pandas DataFrame or dict containing objective columns.
        targets: Dictionary with target values for objectives.

    Returns:
        List of dicts, each describing an objective's performance breakdown.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if len(df) == 0:
        return []

    breakdown = []

    # 1. Transmission or Reflectivity
    show_trans = (primary_metric == "transmission") or (
        targets.get("primary_metric") == "transmission"
    ) or (
        "transmission" in targets and "transmission" in df.columns and (
            "reflectivity" not in targets or "reflectivity" not in df.columns or primary_metric != "reflectivity"
        )
    )
    if show_trans:
        t_val = float(targets["transmission"])
        vals = df["transmission"].values
        pass_mask = vals <= t_val
        pass_cnt = int(np.sum(pass_mask))
        best_val = float(np.min(vals))
        worst_val = float(np.max(vals))
        margin_ppm = t_val - best_val
        margin_rel = float((t_val - best_val) / t_val * 100.0) if t_val > 0 else 0.0

        breakdown.append({
            "objective": "transmission",
            "display_name": "Transmission",
            "direction": "minimize",
            "unit": "ppm",
            "target": t_val,
            "pass_count": pass_cnt,
            "pass_pct": float(pass_cnt / len(df) * 100.0),
            "best_value": best_val,
            "worst_value": worst_val,
            "margin_note": f"Best: {best_val:.2f} ppm ({margin_rel:+.1f}% margin vs target {t_val:.2f} ppm, Δ: {margin_ppm:+.2f} ppm)",
            "is_bottleneck": False,
        })

    if "reflectivity" in df.columns and "reflectivity" in targets and not show_trans:
        t_val = float(targets["reflectivity"])
        vals = df["reflectivity"].values
        pass_mask = vals >= t_val
        pass_cnt = int(np.sum(pass_mask))
        best_val = float(np.max(vals))
        worst_val = float(np.min(vals))
        
        # Margin in ppm loss: target_loss - best_loss
        t_loss = (1.0 - t_val) * 1e6
        best_loss = (1.0 - best_val) * 1e6
        margin_ppm = t_loss - best_loss

        breakdown.append({
            "objective": "reflectivity",
            "display_name": "Reflectivity",
            "direction": "maximize",
            "unit": "",
            "target": t_val,
            "pass_count": pass_cnt,
            "pass_pct": float(pass_cnt / len(df) * 100.0),
            "best_value": best_val,
            "worst_value": worst_val,
            "margin_note": f"Best loss: {best_loss:.2f} ppm (Target loss: {t_loss:.2f} ppm, Δ: {margin_ppm:+.2f} ppm)",
            "is_bottleneck": False,
        })

    # 2. Absorption
    if "absorption" in df.columns and "absorption" in targets:
        t_val = float(targets["absorption"])
        vals = df["absorption"].values
        pass_mask = vals <= t_val
        pass_cnt = int(np.sum(pass_mask))
        best_val = float(np.min(vals))
        worst_val = float(np.max(vals))
        margin_rel = float((t_val - best_val) / t_val * 100.0) if t_val > 0 else 0.0

        breakdown.append({
            "objective": "absorption",
            "display_name": "Absorption",
            "direction": "minimize",
            "unit": "ppm",
            "target": t_val,
            "pass_count": pass_cnt,
            "pass_pct": float(pass_cnt / len(df) * 100.0),
            "best_value": best_val,
            "worst_value": worst_val,
            "margin_note": f"Best: {best_val:.3f} ppm ({margin_rel:+.1f}% margin vs target {t_val:.2f} ppm)",
            "is_bottleneck": False,
        })

    # 3. Thermal Noise
    if "thermal_noise" in df.columns and "thermal_noise" in targets:
        t_val = float(targets["thermal_noise"])
        vals = df["thermal_noise"].values
        pass_mask = vals <= t_val
        pass_cnt = int(np.sum(pass_mask))
        best_val = float(np.min(vals))
        worst_val = float(np.max(vals))
        margin_rel = float((t_val - best_val) / t_val * 100.0) if t_val > 0 else 0.0

        breakdown.append({
            "objective": "thermal_noise",
            "display_name": "Thermal Noise (CTN)",
            "direction": "minimize",
            "unit": "m/√Hz",
            "target": t_val,
            "pass_count": pass_cnt,
            "pass_pct": float(pass_cnt / len(df) * 100.0),
            "best_value": best_val,
            "worst_value": worst_val,
            "margin_note": f"Best: {best_val:.3e} ({margin_rel:+.1f}% margin vs target {t_val:.2e})",
            "is_bottleneck": False,
        })

    # Mark lowest pass percentage as bottleneck
    if breakdown:
        min_pass_pct = min(item["pass_pct"] for item in breakdown)
        for item in breakdown:
            if item["pass_pct"] == min_pass_pct and min_pass_pct < 100.0:
                item["is_bottleneck"] = True

    return breakdown


def compute_spacing_metric(
    df: Any,
    obj_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute Schott's spacing metric (S) to quantify the uniformity of Pareto solutions.

    Formula:
        d_i = min_{j != i} sum_{k=1}^M |f_k(x_i) - f_k(x_j)| (in [0, 1] normalized space)
        d_mean = mean(d_i)
        S = sqrt( 1 / (N - 1) * sum_{i=1}^N (d_i - d_mean)^2 )

    Interpretation:
        S = 0 represents a perfectly equidistant, uniform distribution across the front.
        Lower values indicate superior diversity and uniformity without clustering or gaps.

    Args:
        df: pandas DataFrame containing objective columns.
        obj_cols: List of objective column names. If None, auto-detects ['reflectivity',
            'absorption', 'thermal_noise'] (or 'transmission').

    Returns:
        Dict with 'spacing' (float), 'mean_distance' (float), 'min_distance' (float),
        'max_distance' (float), and 'n_points' (int).
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if len(df) < 2:
        return {
            "spacing": 0.0,
            "mean_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "n_points": len(df),
        }

    if obj_cols is None:
        candidates = ["transmission", "reflectivity", "absorption", "thermal_noise"]
        obj_cols = [c for c in candidates if c in df.columns]
        if "transmission" in obj_cols and "reflectivity" in obj_cols:
            obj_cols.remove("reflectivity")

    if not obj_cols:
        return {
            "spacing": 0.0,
            "mean_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "n_points": len(df),
        }

    # Extract objective values
    raw_data = df[obj_cols].values.astype(float)
    n_points, n_objs = raw_data.shape

    # For reflectivity, convert to optical loss (1 - R) so all objectives are minimization
    proc_data = raw_data.copy()
    for col_idx, col_name in enumerate(obj_cols):
        if col_name == "reflectivity":
            proc_data[:, col_idx] = 1.0 - proc_data[:, col_idx]

    # Normalize each objective to [0, 1]
    mins = np.min(proc_data, axis=0)
    maxs = np.max(proc_data, axis=0)
    ranges = np.where(maxs - mins > 1e-12, maxs - mins, 1.0)
    norm_data = (proc_data - mins) / ranges

    # Compute Manhattan / L1 nearest neighbour distance for each point
    if n_points < 2:
        return {
            "spacing": 0.0,
            "mean_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "n_points": n_points,
        }

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(norm_data)
        dists, _ = tree.query(norm_data, k=2, p=1)
        distances = dists[:, 1]
    except Exception:
        # Fallback to loop if scipy is not available or errors
        distances = np.zeros(n_points)
        for i in range(n_points):
            diffs = np.abs(norm_data - norm_data[i])
            manhattan_dists = np.sum(diffs, axis=1)
            manhattan_dists[i] = np.inf
            distances[i] = np.min(manhattan_dists)

    d_mean = float(np.mean(distances))
    if n_points > 1:
        s_metric = float(np.sqrt(np.sum((distances - d_mean) ** 2) / (n_points - 1)))
    else:
        s_metric = 0.0

    return {
        "spacing": s_metric,
        "mean_distance": d_mean,
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "n_points": n_points,
    }


def compute_asf_scores(
    df: Any,
    targets: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    rho: float = 1e-4,
    primary_metric: Optional[str] = None,
) -> np.ndarray:
    """Compute Wierzbicki's Augmented Achievement Scalarizing Function (ASF) Chebyshev distance.

    For target vector z* and solution x:
      Normalized deviation for each objective:
        - Reflectivity (max): dev_R = ( (1-R) - (1-R*) ) / (1-R*) = (R* - R) / (1 - R*)
        - Transmission (min): dev_T = (T - T*) / T*
        - Absorption (min): dev_A = (A - A*) / A*
        - Thermal Noise (min): dev_TN = (TN - TN*) / TN*

      ASF(x) = max_k [ w_k * dev_k(x) ] + rho * sum_k [ w_k * dev_k(x) ]

    Properties:
      - ASF <= 0: The solution meets or exceeds ALL targets simultaneously.
      - ASF == 0: The solution is exactly on the target boundary in its worst objective.
      - ASF > 0: The solution violates at least one target; score is governed by the largest deficit.
      - argmin(ASF): Identifies the single closest Pareto-optimal design to the target.

    Args:
        df: pandas DataFrame containing objective columns.
        targets: Target values dict ('reflectivity', 'transmission', 'absorption', 'thermal_noise').
        weights: Relative objective weights (defaults to equal weighting).
        rho: Augmentation factor for strict Pareto optimality (default 1e-4).
        primary_metric: 'transmission' or 'reflectivity' (optional).

    Returns:
        1D numpy array of ASF scores for each row in df.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if len(df) == 0:
        return np.array([])

    use_trans = (primary_metric == "transmission") or (
        targets.get("primary_metric") == "transmission"
    ) or (
        "transmission" in df.columns and "transmission" in targets and (
            "reflectivity" not in targets or "reflectivity" not in df.columns or primary_metric != "reflectivity"
        )
    )

    t_abs = float(targets.get("absorption", 0.30))
    t_tn = float(targets.get("thermal_noise", 4.0e-21))

    if weights is None:
        weights = {}

    opt1_key = "transmission" if use_trans else "reflectivity"
    w_opt1 = float(weights.get(opt1_key, weights.get("reflectivity", weights.get("transmission", 0.3333))))
    w_abs = float(weights.get("absorption", 0.3333))
    w_tn = float(weights.get("thermal_noise", 0.3334))

    # Normalize weights
    tot_w = w_opt1 + w_abs + w_tn
    if tot_w > 0:
        w_opt1 /= tot_w
        w_abs /= tot_w
        w_tn /= tot_w

    if use_trans:
        t_trans = float(targets.get("transmission", 10.0))
        trans_scale = max(0.1, t_trans)
        trans_vals = df["transmission"].values if "transmission" in df.columns else np.full(len(df), t_trans)
        dev_opt1 = (trans_vals - t_trans) / trans_scale
    else:
        t_refl = float(targets.get("reflectivity", 0.99999))
        refl_scale = max(1e-9, 1.0 - t_refl)
        refl_vals = df["reflectivity"].values if "reflectivity" in df.columns else np.full(len(df), t_refl)
        dev_opt1 = (t_refl - refl_vals) / refl_scale

    abs_scale = max(1e-9, t_abs)
    abs_vals = df["absorption"].values if "absorption" in df.columns else np.full(len(df), t_abs)
    dev_abs = (abs_vals - t_abs) / abs_scale

    tn_scale = max(1e-25, t_tn)
    tn_vals = df["thermal_noise"].values if "thermal_noise" in df.columns else np.full(len(df), t_tn)
    dev_tn = (tn_vals - t_tn) / tn_scale

    weighted_devs = [w_opt1 * dev_opt1, w_abs * dev_abs, w_tn * dev_tn]

    weighted_devs_matrix = np.column_stack(weighted_devs)  # Shape (N, 3)
    max_dev = np.max(weighted_devs_matrix, axis=1)
    sum_dev = np.sum(weighted_devs_matrix, axis=1)

    asf_scores = max_dev + rho * sum_dev
    return asf_scores


def compute_roi_hypervolume(
    df: Any,
    targets: Dict[str, float],
    roi_factor: float = 1.5,
    primary_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Region-of-Interest Hypervolume (R-HV) bounded around design targets.

    Evaluates the hypervolume of Pareto solutions residing inside the target neighborhood,
    referenced to an anti-ideal ROI boundary point (z* * roi_factor).

    Args:
        df: pandas DataFrame containing objective columns.
        targets: Target values dictionary.
        roi_factor: Expansion factor for ROI boundary (default 1.5, meaning +50% margin).
        primary_metric: 'transmission' or 'reflectivity' (optional).

    Returns:
        Dict with 'roi_hv' (float), 'roi_points_count' (int), 'roi_fraction' (float).
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if len(df) == 0:
        return {"roi_hv": 0.0, "roi_points_count": 0, "roi_fraction": 0.0}

    use_trans = (primary_metric == "transmission") or (
        targets.get("primary_metric") == "transmission"
    ) or (
        "transmission" in df.columns and "transmission" in targets and (
            "reflectivity" not in targets or "reflectivity" not in df.columns or primary_metric != "reflectivity"
        )
    )

    t_abs = float(targets.get("absorption", 0.30))
    t_tn = float(targets.get("thermal_noise", 4.0e-21))

    # Boundary thresholds
    abs_boundary = t_abs * roi_factor
    tn_boundary = t_tn * roi_factor

    abs_vals = df["absorption"].values if "absorption" in df.columns else np.zeros(len(df))
    tn_vals = df["thermal_noise"].values if "thermal_noise" in df.columns else np.zeros(len(df))

    if use_trans:
        t_trans = float(targets.get("transmission", 10.0))
        trans_boundary = t_trans * roi_factor
        trans_vals = df["transmission"].values if "transmission" in df.columns else np.zeros(len(df))

        mask = (trans_vals <= trans_boundary) & (abs_vals <= abs_boundary) & (tn_vals <= tn_boundary)
        roi_count = int(np.sum(mask))

        if roi_count == 0:
            return {"roi_hv": 0.0, "roi_points_count": 0, "roi_fraction": 0.0}

        roi_trans = trans_vals[mask]
        roi_abs = abs_vals[mask]
        roi_tn = tn_vals[mask]

        norm_trans = np.clip(roi_trans / trans_boundary, 0.0, 1.0)
        norm_abs = np.clip(roi_abs / abs_boundary, 0.0, 1.0)
        norm_tn = np.clip(roi_tn / tn_boundary, 0.0, 1.0)

        roi_norm_points = np.column_stack([norm_trans, norm_abs, norm_tn])
    else:
        t_refl = float(targets.get("reflectivity", 0.99999))
        refl_loss_target = max(1e-9, 1.0 - t_refl)
        r_boundary = 1.0 - refl_loss_target * roi_factor
        refl_vals = df["reflectivity"].values if "reflectivity" in df.columns else np.ones(len(df))

        mask = (refl_vals >= r_boundary) & (abs_vals <= abs_boundary) & (tn_vals <= tn_boundary)
        roi_count = int(np.sum(mask))

        if roi_count == 0:
            return {"roi_hv": 0.0, "roi_points_count": 0, "roi_fraction": 0.0}

        roi_refl = refl_vals[mask]
        roi_abs = abs_vals[mask]
        roi_tn = tn_vals[mask]

        # Normalize within ROI space [0, 1] where 0 is ideal and 1 is boundary
        loss_vals = 1.0 - roi_refl
        max_loss = max(1e-9, 1.0 - r_boundary)
        norm_loss = np.clip(loss_vals / max_loss, 0.0, 1.0)
        norm_abs = np.clip(roi_abs / abs_boundary, 0.0, 1.0)
        norm_tn = np.clip(roi_tn / tn_boundary, 0.0, 1.0)

        roi_norm_points = np.column_stack([norm_loss, norm_abs, norm_tn])

    ref_point = np.array([1.0, 1.0, 1.0])

    # All normalized objectives are minimized in [0, 1] with reference point at [1, 1, 1]
    hv_val = compute_hypervolume(roi_norm_points, ref_point, maximize=False)

    return {
        "roi_hv": float(hv_val),
        "roi_points_count": roi_count,
        "roi_fraction": float(roi_count / len(df)),
    }


def evaluate_dataset_proximity_metrics(
    df: Any,
    targets: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    primary_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprehensive multi-objective evaluation against target requirements.

    Calculates:
      1. Target Region Yield Y(0) & Yield Curve Y(alpha)
      2. Per-Objective Achievement Breakdown (pass rates, bests, bottlenecks)
      3. Schott's Spacing Metric (uniformity of solution distribution)
      4. Augmented Chebyshev Achievement Scalarizing Function (ASF) scores
      5. Region-of-Interest Hypervolume (R-HV)

    Args:
        df: pandas DataFrame containing Pareto front solutions.
        targets: Target specifications for objectives.
        weights: Weights for objectives.
        primary_metric: 'transmission' or 'reflectivity' (optional).

    Returns:
        Master dictionary containing all computed metrics.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if primary_metric is None:
        primary_metric = targets.get("primary_metric")

    yield_res = compute_target_yield(df, targets, primary_metric=primary_metric)
    obj_breakdown = compute_objective_breakdown(df, targets, primary_metric=primary_metric)
    spacing_res = compute_spacing_metric(df)
    asf_scores = compute_asf_scores(df, targets, weights, primary_metric=primary_metric)
    roi_hv_res = compute_roi_hypervolume(df, targets, primary_metric=primary_metric)

    best_asf_idx = int(np.argmin(asf_scores)) if len(asf_scores) > 0 else None
    best_asf_val = float(asf_scores[best_asf_idx]) if best_asf_idx is not None else 0.0

    return {
        "total_designs": len(df),
        "yield": yield_res,
        "objective_breakdown": obj_breakdown,
        "spacing": spacing_res,
        "asf": {
            "best_index": best_asf_idx,
            "best_score": best_asf_val,
            "exceeds_all_targets": bool(best_asf_val <= 0.0),
            "asf_scores": asf_scores.tolist(),
        },
        "roi_hypervolume": roi_hv_res,
        "targets": targets,
    }
    """Comprehensive multi-objective evaluation against target requirements.

    Calculates:
      1. Target Region Yield Y(0) & Yield Curve Y(alpha)
      2. Per-Objective Achievement Breakdown (pass rates, bests, bottlenecks)
      3. Schott's Spacing Metric (uniformity of solution distribution)
      4. Augmented Chebyshev Achievement Scalarizing Function (ASF) scores
      5. Region-of-Interest Hypervolume (R-HV)

    Args:
        df: pandas DataFrame containing Pareto front solutions.
        targets: Target specifications for objectives.
        weights: Weights for objectives.

    Returns:
        Master dictionary containing all computed metrics.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    yield_res = compute_target_yield(df, targets)
    obj_breakdown = compute_objective_breakdown(df, targets)
    spacing_res = compute_spacing_metric(df)
    asf_scores = compute_asf_scores(df, targets, weights)
    roi_hv_res = compute_roi_hypervolume(df, targets)

    best_asf_idx = int(np.argmin(asf_scores)) if len(asf_scores) > 0 else None
    best_asf_val = float(asf_scores[best_asf_idx]) if best_asf_idx is not None else 0.0

    return {
        "total_designs": len(df),
        "yield": yield_res,
        "objective_breakdown": obj_breakdown,
        "spacing": spacing_res,
        "asf": {
            "best_index": best_asf_idx,
            "best_score": best_asf_val,
            "exceeds_all_targets": bool(best_asf_val <= 0.0),
            "asf_scores": asf_scores.tolist(),
        },
        "roi_hypervolume": roi_hv_res,
        "targets": targets,
    }
