#!/usr/bin/env python3
"""
Shared interactive plotting utilities for Pareto front visualisation.

Provides:
    Objective helpers:
        _OBJ_CONFIG, _obj_transform, _obj_label, _obj_scale, _detect_objectives

    Run-grouping colour helpers:
        _BASE_COLORS, _parse_run_name, _color_gradient, _build_color_map

    Plot functions:
        plot_pairwise_comparison_interactive  – VALUE vs REWARD side-by-side
        plot_pareto_3d_interactive            – 3-D scatter of the first 3 objectives
        plot_pareto_parallel_coords_interactive – parallel-coordinates with brushing
"""

import itertools
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Objective display config ──────────────────────────────────────────────────
_OBJ_CONFIG = {
    "reflectivity": {
        "transform": lambda x: 1.0 - x,
        "label": "1 − Reflectivity",
        "scale": "log",
    },
    "absorption": {
        "transform": lambda x: x,
        "label": "Absorption (ppm)",
        "scale": "log",
    },
    "thermal_noise": {
        "transform": lambda x: x,
        "label": "Thermal Noise",
        "scale": "log",
    },
}


def _obj_transform(obj: str, values: np.ndarray) -> np.ndarray:
    if obj in _OBJ_CONFIG:
        return _OBJ_CONFIG[obj]["transform"](values)
    return values


def _obj_label(obj: str) -> str:
    if obj in _OBJ_CONFIG:
        return _OBJ_CONFIG[obj]["label"]
    return obj.replace("_", " ").title()


def _obj_scale(obj: str) -> str:
    return _OBJ_CONFIG.get(obj, {}).get("scale", "linear")


def _detect_objectives(values_df: pd.DataFrame) -> List[str]:
    """Return objective columns (exclude design vars and reward cols)."""
    exclude = {
        c
        for c in values_df.columns
        if c.startswith(("thickness_", "material_")) or c.endswith("_reward")
    }
    return [c for c in values_df.columns if c not in exclude]


# ── Run-grouping helpers ──────────────────────────────────────────────────────
_BASE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _parse_run_name(label: str) -> Tuple[str, Optional[int]]:
    match = re.search(r"[_-]run(\d+)$", label, re.IGNORECASE)
    if match:
        return label[: match.start()], int(match.group(1))
    return label, None


def _color_gradient(base: str, n: int) -> List[str]:
    base = base.lstrip("#")
    br, bg, bb = int(base[0:2], 16), int(base[2:4], 16), int(base[4:6], 16)
    if n == 1:
        return [f"#{base}"]
    out = []
    for i in range(n):
        f = 0.6 + 0.7 * i / (n - 1)
        out.append(
            "#{:02x}{:02x}{:02x}".format(
                min(255, max(0, int(br * f))),
                min(255, max(0, int(bg * f))),
                min(255, max(0, int(bb * f))),
            )
        )
    return out


def _build_color_map(pareto_fronts, group_runs: bool) -> dict:
    if group_runs:
        groups: dict = {}
        for _, _, label in pareto_fronts:
            base, num = _parse_run_name(label)
            groups.setdefault(base, []).append((label, num))
        for base in groups:
            groups[base].sort(key=lambda x: x[1] if x[1] is not None else -1)
        color_map = {}
        for g_idx, (base, runs) in enumerate(groups.items()):
            grads = _color_gradient(_BASE_COLORS[g_idx % len(_BASE_COLORS)], len(runs))
            for i, (label, _) in enumerate(runs):
                color_map[label] = grads[i]
        return color_map
    return {
        label: _BASE_COLORS[i % len(_BASE_COLORS)]
        for i, (_, _, label) in enumerate(pareto_fronts)
    }


def _discrete_colorscale(n: int) -> list:
    """Stepped Plotly colorscale with n equally-spaced bands (tab10 palette)."""
    cs = []
    for i in range(n):
        color = _BASE_COLORS[i % len(_BASE_COLORS)]
        lo, hi = i / n, (i + 1) / n
        cs.append([lo, color])
        cs.append([hi, color])
    return cs


# ── Pairwise VALUE vs REWARD comparison ──────────────────────────────────────


def plot_pairwise_comparison_interactive(
    pareto_fronts: List[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]],
    save_path: Optional[Path] = None,
    title: str = "Pareto Front Comparison",
    reference_values: Optional[pd.DataFrame] = None,
    reference_rewards: Optional[pd.DataFrame] = None,
    group_runs: bool = True,
    compute_hv_fn: Optional[Callable] = None,
) -> go.Figure:
    """Plot VALUE and REWARD space Pareto fronts for all objective pairs.

    Auto-detects N objectives and creates one row per pairwise combination,
    with VALUE space (left) and REWARD space (right) in each row.

    Args:
        pareto_fronts: List of (values_df, rewards_df, label) tuples.
        save_path: If given, saves HTML to <save_path.stem>_interactive.html.
        title: Figure title.
        reference_values: Optional reference Pareto front in value space.
        reference_rewards: Optional reference Pareto front in reward space.
        group_runs: Group runs with similar names using colour gradients.
        compute_hv_fn: Optional callable(df, space) -> float for hypervolume.
            If None, HV annotations are omitted.

    Returns:
        Plotly Figure.
    """
    # ── Detect objectives & build pairs ──────────────────────────────────────
    objectives = []
    for vdf, _, _ in pareto_fronts:
        objectives = _detect_objectives(vdf) if vdf is not None else []
        if objectives:
            break
    if not objectives and reference_values is not None:
        objectives = _detect_objectives(reference_values)
    if len(objectives) < 2:
        objectives = ["absorption", "reflectivity"]

    pairs = list(itertools.combinations(objectives, 2))
    n_pairs = len(pairs)

    # ── Colors ────────────────────────────────────────────────────────────────
    color_map = _build_color_map(pareto_fronts, group_runs)

    # ── HV cache ─────────────────────────────────────────────────────────────
    if compute_hv_fn is not None:
        hv_val = {
            lbl: compute_hv_fn(vdf, "value") if vdf is not None else 0.0
            for vdf, _, lbl in pareto_fronts
        }
        hv_rew = {
            lbl: compute_hv_fn(rdf, "reward") if rdf is not None else 0.0
            for _, rdf, lbl in pareto_fronts
        }
    else:
        hv_val = {lbl: 0.0 for _, _, lbl in pareto_fronts}
        hv_rew = {lbl: 0.0 for _, _, lbl in pareto_fronts}

    # ── Best solutions (per objective, across all runs) ────────────────────────
    # Value space: minimum of each transformed objective
    # Reward space: maximum of each reward
    best_val: dict = {}
    best_rew: dict = {}
    for obj in objectives:
        v_all, r_all = [], []
        for vdf, rdf, _ in pareto_fronts:
            if vdf is not None and obj in vdf.columns:
                v = _obj_transform(obj, vdf[obj].values)
                v_all.extend(v[np.isfinite(v)].tolist())
            if rdf is not None and obj in rdf.columns:
                r = rdf[obj].values
                r_all.extend(r[np.isfinite(r)].tolist())
        best_val[obj] = float(np.min(v_all)) if v_all else None
        best_rew[obj] = float(np.max(r_all)) if r_all else None

    # ── Subplot grid: n_pairs rows × 2 cols (VALUE | REWARD) ─────────────────
    subplot_titles = []
    for obj_x, obj_y in pairs:
        subplot_titles += [
            f"VALUE · {obj_x} vs {obj_y}",
            f"REWARD · {obj_x} vs {obj_y}",
        ]

    fig = make_subplots(
        rows=n_pairs,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )

    # ── Plot each pair row ────────────────────────────────────────────────────
    for pair_idx, (obj_x, obj_y) in enumerate(pairs):
        row = pair_idx + 1
        is_primary = {"absorption", "reflectivity"} == {obj_x, obj_y}
        first_row = pair_idx == 0

        # Reference markers – VALUE
        if (
            reference_values is not None
            and obj_x in reference_values.columns
            and obj_y in reference_values.columns
        ):
            rx = _obj_transform(obj_x, reference_values[obj_x].values)
            ry = _obj_transform(obj_y, reference_values[obj_y].values)
            valid = np.isfinite(rx) & np.isfinite(ry) & (ry > 0)
            hv_ref = (
                compute_hv_fn(reference_values, "value")
                if (compute_hv_fn and is_primary)
                else 0.0
            )
            ref_name = f"Reference (HV: {hv_ref:.4f})" if hv_ref > 0 else "Reference"
            fig.add_trace(
                go.Scatter(
                    x=rx[valid],
                    y=ry[valid],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=12,
                        color="black",
                        line=dict(width=2, color="black"),
                    ),
                    name=ref_name,
                    legendgroup="reference",
                    showlegend=first_row,
                ),
                row=row,
                col=1,
            )

        # Reference markers – REWARD
        if (
            reference_rewards is not None
            and obj_x in reference_rewards.columns
            and obj_y in reference_rewards.columns
        ):
            rx = reference_rewards[obj_x].values
            ry = reference_rewards[obj_y].values
            valid = np.isfinite(rx) & np.isfinite(ry)
            fig.add_trace(
                go.Scatter(
                    x=rx[valid],
                    y=ry[valid],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=12,
                        color="black",
                        line=dict(width=2, color="black"),
                    ),
                    name="Reference",
                    legendgroup="reference",
                    showlegend=False,
                ),
                row=row,
                col=2,
            )

        # Each run
        for vdf, rdf, label in pareto_fronts:
            color = color_map.get(label, "#808080")
            base_name, run_num = _parse_run_name(label)
            legend_group = base_name if group_runs else label

            if group_runs and run_num is not None:
                short = f"run{run_num:03d}"
            else:
                short = label[-40:] if len(label) > 40 else label

            leg_v = (
                f"{short} (HV: {hv_val[label]:.4f})"
                if (hv_val[label] > 0 and is_primary)
                else short
            )
            leg_r = (
                f"{short} (HV: {hv_rew[label]:.4f})"
                if (hv_rew[label] > 0 and is_primary)
                else short
            )

            # VALUE panel
            if vdf is not None and obj_x in vdf.columns and obj_y in vdf.columns:
                x = _obj_transform(obj_x, vdf[obj_x].values)
                y = _obj_transform(obj_y, vdf[obj_y].values)
                valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
                x, y = x[valid], y[valid]
                if len(x):
                    idx = np.argsort(x)
                    fig.add_trace(
                        go.Scatter(
                            x=x[idx],
                            y=y[idx],
                            mode="lines",
                            line=dict(color=color, width=2, dash="dash"),
                            opacity=0.4,
                            name=leg_v,
                            legendgroup=legend_group,
                            legendgrouptitle_text=legend_group if group_runs else None,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            marker=dict(
                                color=color, size=8, line=dict(width=1, color="black")
                            ),
                            name=leg_v,
                            legendgroup=legend_group,
                            legendgrouptitle_text=legend_group if group_runs else None,
                            showlegend=first_row,
                            hovertemplate=(
                                f"<b>%{{fullData.name}}</b><br>"
                                f"{_obj_label(obj_x)}: %{{x:.3e}}<br>"
                                f"{_obj_label(obj_y)}: %{{y:.3e}}<br><extra></extra>"
                            ),
                        ),
                        row=row,
                        col=1,
                    )

            # REWARD panel
            if rdf is not None and obj_x in rdf.columns and obj_y in rdf.columns:
                x = rdf[obj_x].values
                y = rdf[obj_y].values
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                if len(x):
                    idx = np.argsort(x)
                    fig.add_trace(
                        go.Scatter(
                            x=x[idx],
                            y=y[idx],
                            mode="lines",
                            line=dict(color=color, width=2, dash="dash"),
                            opacity=0.4,
                            name=leg_r,
                            legendgroup=legend_group,
                            legendgrouptitle_text=legend_group if group_runs else None,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=2,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            marker=dict(
                                color=color, size=8, line=dict(width=1, color="black")
                            ),
                            name=leg_r,
                            legendgroup=legend_group,
                            legendgrouptitle_text=legend_group if group_runs else None,
                            showlegend=False,
                            hovertemplate=(
                                f"<b>%{{fullData.name}}</b><br>"
                                f"{obj_x.replace('_', ' ').title()} reward: %{{x:.4f}}<br>"
                                f"{obj_y.replace('_', ' ').title()} reward: %{{y:.4f}}<br>"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=2,
                    )

        # Best-solution markers
        ux_v, uy_v = best_val.get(obj_x), best_val.get(obj_y)
        if ux_v is not None and uy_v is not None:
            fig.add_trace(
                go.Scatter(
                    x=[ux_v],
                    y=[uy_v],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color="#FFD700",
                        line=dict(width=1.5, color="black"),
                    ),
                    name="Best solution",
                    legendgroup="best_solution",
                    showlegend=(first_row),
                    hovertemplate=(
                        "<b>Best solution</b><br>"
                        f"{_obj_label(obj_x)}: %{{x:.3e}}<br>"
                        f"{_obj_label(obj_y)}: %{{y:.3e}}<br><extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        ux_r, uy_r = best_rew.get(obj_x), best_rew.get(obj_y)
        if ux_r is not None and uy_r is not None:
            fig.add_trace(
                go.Scatter(
                    x=[ux_r],
                    y=[uy_r],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color="#FFD700",
                        line=dict(width=1.5, color="black"),
                    ),
                    name="Best solution",
                    legendgroup="best_solution",
                    showlegend=False,
                    hovertemplate=(
                        "<b>Best solution</b><br>"
                        f"{obj_x.replace('_', ' ').title()} reward: %{{x:.4f}}<br>"
                        f"{obj_y.replace('_', ' ').title()} reward: %{{y:.4f}}<br><extra></extra>"
                    ),
                ),
                row=row,
                col=2,
            )

        # Axes
        fig.update_xaxes(
            title_text=_obj_label(obj_x),
            type=_obj_scale(obj_x),
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text=_obj_label(obj_y),
            type=_obj_scale(obj_y),
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=row,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"{obj_x.replace('_', ' ').title()} reward",
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=row,
            col=2,
        )
        fig.update_yaxes(
            title_text=f"{obj_y.replace('_', ' ').title()} reward",
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=row,
            col=2,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        height=max(500, 500 * n_pairs),
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            tracegroupgap=10,
            itemwidth=30,
            itemsizing="constant",
        ),
        margin=dict(b=180),  # extra bottom margin for legend
        hovermode="closest",
        template="plotly_white",
    )

    if save_path:
        html_path = Path(save_path).parent / (
            Path(save_path).stem + "_interactive.html"
        )
        fig.write_html(str(html_path))
        print(f"Saved interactive comparison plot to {html_path}")

    return fig


# ── 3D scatter ───────────────────────────────────────────────────────────────


def plot_pareto_3d_interactive(
    pareto_fronts: List[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]],
    save_path: Optional[Path] = None,
    title: str = "Pareto Front — 3D",
    group_runs: bool = True,
    objectives_override: Optional[List[str]] = None,
) -> go.Figure:
    """3-D scatter of first three objectives — value space (left) and reward space (right).

    A gold star marks the best possible solution corner on each panel.

    Args:
        pareto_fronts: List of (values_df, rewards_df, label) tuples.
        save_path: If given, saves HTML alongside (stem + "_3d.html").
        title: Figure title.
        group_runs: Use colour-gradient grouping.
        objectives_override: Explicit list of 3 objective names.

    Returns:
        Plotly Figure.
    """
    from plotly.subplots import make_subplots as _make_subplots

    # ── Detect objectives ─────────────────────────────────────────────────────
    if objectives_override:
        objectives = objectives_override
    else:
        objectives = []
        for vdf, _, _ in pareto_fronts:
            if vdf is not None:
                objectives = _detect_objectives(vdf)
                if len(objectives) >= 3:
                    break
        if len(objectives) < 3:
            print("Warning: fewer than 3 objectives; padding with last objective.")
            while len(objectives) < 3:
                objectives.append(objectives[-1] if objectives else "obj")

    obj_x, obj_y, obj_z = objectives[0], objectives[1], objectives[2]
    color_map = _build_color_map(pareto_fronts, group_runs)

    fig = _make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Value Space", "Reward Space"],
        horizontal_spacing=0.05,
    )

    # ── Collect all data to compute best-solution point ──────────────────────
    val_pool = {o: [] for o in (obj_x, obj_y, obj_z)}
    rew_pool = {o: [] for o in (obj_x, obj_y, obj_z)}
    for vdf, rdf, _ in pareto_fronts:
        if vdf is not None:
            for o in (obj_x, obj_y, obj_z):
                if o in vdf.columns:
                    v = _obj_transform(o, vdf[o].values)
                    val_pool[o].extend(v[np.isfinite(v)].tolist())
        if rdf is not None:
            for o in (obj_x, obj_y, obj_z):
                if o in rdf.columns:
                    v = rdf[o].values
                    rew_pool[o].extend(v[np.isfinite(v)].tolist())

    # Value best solution: minimum of each transformed objective (all are "minimise")
    best_val = {
        o: float(np.min(val_pool[o])) if val_pool[o] else 0.0
        for o in (obj_x, obj_y, obj_z)
    }
    # Reward best solution: maximum of each reward (higher = better)
    best_rew = {
        o: float(np.max(rew_pool[o])) if rew_pool[o] else 1.0
        for o in (obj_x, obj_y, obj_z)
    }

    # ── Add run traces ────────────────────────────────────────────────────────
    for vdf, rdf, label in pareto_fronts:
        color = color_map.get(label, "#808080")
        base_name, run_num = _parse_run_name(label)
        legend_group = base_name if group_runs else label
        short = (
            f"run{run_num:03d}"
            if (group_runs and run_num is not None)
            else (label[-40:] if len(label) > 40 else label)
        )
        show_leg = True  # shown once; subsequent panels suppress

        # Value space
        if vdf is not None and all(o in vdf.columns for o in (obj_x, obj_y, obj_z)):
            x = _obj_transform(obj_x, vdf[obj_x].values)
            y = _obj_transform(obj_y, vdf[obj_y].values)
            z = _obj_transform(obj_z, vdf[obj_z].values)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            fig.add_trace(
                go.Scatter3d(
                    x=x[valid],
                    y=y[valid],
                    z=z[valid],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=color,
                        opacity=0.8,
                        line=dict(width=0.5, color="black"),
                    ),
                    name=short,
                    legendgroup=legend_group,
                    legendgrouptitle_text=legend_group if group_runs else None,
                    showlegend=show_leg,
                    hovertemplate=(
                        f"<b>%{{fullData.name}}</b><br>"
                        f"{_obj_label(obj_x)}: %{{x:.3e}}<br>"
                        f"{_obj_label(obj_y)}: %{{y:.3e}}<br>"
                        f"{_obj_label(obj_z)}: %{{z:.3e}}<br><extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            show_leg = False

        # Reward space
        if rdf is not None and all(o in rdf.columns for o in (obj_x, obj_y, obj_z)):
            x = rdf[obj_x].values
            y = rdf[obj_y].values
            z = rdf[obj_z].values
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            fig.add_trace(
                go.Scatter3d(
                    x=x[valid],
                    y=y[valid],
                    z=z[valid],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=color,
                        opacity=0.8,
                        line=dict(width=0.5, color="black"),
                    ),
                    name=short,
                    legendgroup=legend_group,
                    legendgrouptitle_text=legend_group if group_runs else None,
                    showlegend=show_leg,
                    hovertemplate=(
                        f"<b>%{{fullData.name}}</b><br>"
                        f"{obj_x.replace('_',' ').title()} reward: %{{x:.4f}}<br>"
                        f"{obj_y.replace('_',' ').title()} reward: %{{y:.4f}}<br>"
                        f"{obj_z.replace('_',' ').title()} reward: %{{z:.4f}}<br><extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )

    # ── Best-solution markers ─────────────────────────────────────────────────
    if val_pool[obj_x]:
        fig.add_trace(
            go.Scatter3d(
                x=[best_val[obj_x]],
                y=[best_val[obj_y]],
                z=[best_val[obj_z]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=8,
                    color="#FFD700",
                    line=dict(width=1.5, color="black"),
                ),
                name="Best solution",
                legendgroup="best_solution",
                showlegend=True,
                hovertemplate=(
                    "<b>Best solution</b><br>"
                    f"{_obj_label(obj_x)}: %{{x:.3e}}<br>"
                    f"{_obj_label(obj_y)}: %{{y:.3e}}<br>"
                    f"{_obj_label(obj_z)}: %{{z:.3e}}<br><extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if rew_pool[obj_x]:
        fig.add_trace(
            go.Scatter3d(
                x=[best_rew[obj_x]],
                y=[best_rew[obj_y]],
                z=[best_rew[obj_z]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=8,
                    color="#FFD700",
                    line=dict(width=1.5, color="black"),
                ),
                name="Best solution",
                legendgroup="best_solution",
                showlegend=False,
                hovertemplate=(
                    "<b>Best solution</b><br>"
                    f"{obj_x.replace('_',' ').title()} reward: %{{x:.4f}}<br>"
                    f"{obj_y.replace('_',' ').title()} reward: %{{y:.4f}}<br>"
                    f"{obj_z.replace('_',' ').title()} reward: %{{z:.4f}}<br><extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    scene_style = dict(
        xaxis=dict(
            backgroundcolor="rgb(240,240,250)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
        yaxis=dict(
            backgroundcolor="rgb(240,250,240)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
        zaxis=dict(
            backgroundcolor="rgb(250,240,240)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white",
        ),
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        scene=dict(
            xaxis_title=_obj_label(obj_x),
            yaxis_title=_obj_label(obj_y),
            zaxis_title=_obj_label(obj_z),
            **scene_style,
        ),
        scene2=dict(
            xaxis_title=f"{obj_x.replace('_',' ').title()} reward",
            yaxis_title=f"{obj_y.replace('_',' ').title()} reward",
            zaxis_title=f"{obj_z.replace('_',' ').title()} reward",
            **scene_style,
        ),
        height=700,
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            tracegroupgap=10,
        ),
        margin=dict(b=160),
        template="plotly_white",
    )

    if save_path:
        html_path = Path(save_path).parent / (Path(save_path).stem + "_3d.html")
        fig.write_html(str(html_path))
        print(f"Saved 3D Pareto plot to {html_path}")

    return fig


# ── Parallel coordinates ──────────────────────────────────────────────────────


def plot_pareto_parallel_coords_interactive(
    pareto_fronts: List[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]],
    save_path: Optional[Path] = None,
    title: str = "Pareto Front — Parallel Coordinates",
    group_runs: bool = True,
    objectives_override: Optional[List[str]] = None,
) -> go.Figure:
    """Parallel-coordinates for value space and reward space with toggle buttons.

    A gold line marks the best possible solution on each view.
    Discrete colours encode run identity; dummy scatter traces power the legend.

    Args:
        pareto_fronts: List of (values_df, rewards_df, label) tuples.
        save_path: If given, saves HTML alongside (stem + "_parcoords.html").
        title: Figure title.
        group_runs: Use colour-gradient grouping in legend.
        objectives_override: Explicit list of objective names.

    Returns:
        Plotly Figure.
    """
    # ── Detect objectives ─────────────────────────────────────────────────────
    if objectives_override:
        objectives = objectives_override
    else:
        objectives = []
        for vdf, _, _ in pareto_fronts:
            if vdf is not None:
                objectives = _detect_objectives(vdf)
                if objectives:
                    break

    if not objectives:
        raise ValueError("No objectives detected from pareto_fronts.")

    n_runs = len(pareto_fronts)
    # n_runs+1 colour bands: 0..n_runs-1 for runs, n_runs for best solution (gold)
    n_bands = n_runs + 1
    best_idx = float(n_runs) + 0.5  # centre of the best-solution band

    def _make_colorscale(n: int) -> list:
        cs = []
        for i in range(n - 1):
            color = _BASE_COLORS[i % len(_BASE_COLORS)]
            lo, hi = i / n, (i + 1) / n
            cs += [[lo, color], [hi, color]]
        lo, hi = (n - 1) / n, 1.0
        cs += [[lo, "#FFD700"], [hi, "#FFD700"]]
        return cs

    cs = _make_colorscale(n_bands)

    def _build_rows(df_key: str) -> List[pd.DataFrame]:
        rows = []
        for run_idx, pf in enumerate(pareto_fronts):
            df = pf[0] if df_key == "val" else pf[1]
            if df is None:
                continue
            cols = [c for c in objectives if c in df.columns]
            if not cols:
                continue
            sub = df[cols].copy()
            if df_key == "val":
                for obj in cols:
                    sub[obj] = _obj_transform(obj, sub[obj].values)
            sub["_run_idx"] = float(run_idx) + 0.5
            rows.append(sub)
        return rows

    def _build_dims(df: pd.DataFrame, objs: list) -> list:
        dims = []
        for obj in objs:
            if obj not in df.columns:
                continue
            vals = df[obj].values
            finite = vals[np.isfinite(vals)]
            dims.append(
                dict(
                    label=_obj_label(obj),
                    values=vals,
                    range=(
                        [float(finite.min()), float(finite.max())]
                        if len(finite)
                        else [0, 1]
                    ),
                )
            )
        return dims

    def _add_best(df: pd.DataFrame, objs: list, maximize: bool) -> pd.DataFrame:
        row = {"_run_idx": best_idx}
        for obj in objs:
            if obj not in df.columns:
                continue
            finite = df[obj].values[np.isfinite(df[obj].values)]
            if len(finite):
                row[obj] = float(np.max(finite)) if maximize else float(np.min(finite))
            else:
                row[obj] = 1.0 if maximize else 0.0
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    val_rows = _build_rows("val")
    rew_rows = _build_rows("rew")

    if not val_rows and not rew_rows:
        raise ValueError("No data found in pareto_fronts.")

    fig = go.Figure()

    # ── Trace 0: value space parcoords ────────────────────────────────────────
    has_val = bool(val_rows)
    if has_val:
        df_val = _add_best(
            pd.concat(val_rows, ignore_index=True), objectives, maximize=False
        )
        val_dims = _build_dims(df_val, objectives)
        print(
            f"[parcoords] value dims: {[d['label'] for d in val_dims]}, "
            f"rows: {len(df_val)}"
        )
        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=df_val["_run_idx"].values,
                    colorscale=cs,
                    cmin=0,
                    cmax=n_bands,
                    showscale=False,
                ),
                dimensions=val_dims,
                visible=True,
            )
        )
    else:
        fig.add_trace(go.Parcoords(dimensions=[], visible=False))  # placeholder

    # ── Trace 1: reward space parcoords ───────────────────────────────────────
    has_rew = bool(rew_rows)
    if has_rew:
        df_rew = _add_best(
            pd.concat(rew_rows, ignore_index=True), objectives, maximize=True
        )
        rew_dims = _build_dims(df_rew, objectives)
        # Rename labels to make clear these are reward values
        for d in rew_dims:
            d["label"] = d["label"] + " (reward)"
        print(
            f"[parcoords] reward dims: {[d['label'] for d in rew_dims]}, "
            f"rows: {len(df_rew)}"
        )
        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=df_rew["_run_idx"].values,
                    colorscale=cs,
                    cmin=0,
                    cmax=n_bands,
                    showscale=False,
                ),
                dimensions=rew_dims,
                visible=False,
            )
        )
    else:
        fig.add_trace(go.Parcoords(dimensions=[], visible=False))  # placeholder

    # ── Traces 2..: dummy scatter for legend (always visible) ─────────────────
    n_legend = 0
    for run_idx, (_, _, label) in enumerate(pareto_fronts):
        base_name, run_num = _parse_run_name(label)
        legend_group = base_name if group_runs else label
        short = (
            f"run{run_num:03d}"
            if (group_runs and run_num is not None)
            else (label[-40:] if len(label) > 40 else label)
        )
        color = _BASE_COLORS[run_idx % len(_BASE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=short,
                legendgroup=legend_group,
                legendgrouptitle_text=legend_group if group_runs else None,
                showlegend=True,
            )
        )
        n_legend += 1

    # Best-solution legend entry
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=10,
                color="#FFD700",
                line=dict(width=1, color="black"),
            ),
            name="Best solution",
            legendgroup="best_solution",
            showlegend=True,
        )
    )
    n_legend += 1

    # ── Toggle buttons (val / rew) ────────────────────────────────────────────
    # Traces: [0=val parcoords, 1=rew parcoords, 2..n_legend+1=scatter dummies]
    always_vis = [True] * n_legend
    buttons = []
    if has_val:
        buttons.append(
            dict(
                label="Value Space",
                method="update",
                args=[
                    {"visible": [True, False] + always_vis},
                    {
                        "title": {
                            "text": f"{title} — Value Space",
                            "x": 0.5,
                            "xanchor": "center",
                            "font": {"size": 16},
                        }
                    },
                ],
            )
        )
    if has_rew:
        buttons.append(
            dict(
                label="Reward Space",
                method="update",
                args=[
                    {"visible": [False, True] + always_vis},
                    {
                        "title": {
                            "text": f"{title} — Reward Space",
                            "x": 0.5,
                            "xanchor": "center",
                            "font": {"size": 16},
                        }
                    },
                ],
            )
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    initial_title = f"{title} — {'Value' if has_val else 'Reward'} Space"
    fig.update_layout(
        title=dict(text=initial_title, x=0.5, xanchor="center", font=dict(size=16)),
        height=600,
        autosize=True,
        updatemenus=(
            [
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.12,
                    yanchor="top",
                )
            ]
            if len(buttons) > 1
            else []
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            tracegroupgap=10,
        ),
        margin=dict(t=100, b=160),
        template="plotly_white",
    )

    if save_path:
        html_path = Path(save_path).parent / (Path(save_path).stem + "_parcoords.html")
        fig.write_html(str(html_path))
        print(f"Saved parallel-coordinates plot to {html_path}")

    return fig
