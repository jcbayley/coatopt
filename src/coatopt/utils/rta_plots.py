#!/usr/bin/env python3
"""RTA + thermal-noise interactive comparison plots.

Recomputes reflectivity, transmission and absorption from the saved Pareto
designs with a single complex-index transfer-matrix evaluation (so energy
balance holds: R + T + A = 1), plus coating thermal noise — meaning every run
shows the full R/T/A/CTN picture regardless of which objectives were optimised.

Provides:
    evaluate_designs_rta            - designs_df -> RTA/CTN DataFrame
    evaluate_reference_rta          - reference values_df -> RTA/CTN DataFrame
    plot_rta_comparison_interactive - physical space, all pairs of 1-R/T/A/CTN
    plot_reward_comparison_interactive - reward space, one panel per pair
"""

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tmm
from plotly.subplots import make_subplots

from coatopt.environments.utils.EFI_tmm import optical_to_physical
from coatopt.environments.utils.YAM_CoatingBrownian import getCoatingThermalNoise
from coatopt.utils.interactive_plots import (
    _build_color_map,
    _pareto_2d,
    _parse_run_name,
)

# ── Evaluation ────────────────────────────────────────────────────────────────

_RTA_COLUMNS = [
    "reflectivity",
    "transmission_ppm",
    "absorption_ppm",
    "total_loss",
    "thermal_noise",
]

_RTA_LABELS = {
    "transmission_ppm": "Transmission (ppm)",
    "absorption_ppm": "Absorption 1−R−T (ppm)",
    "total_loss": "1 − R",
    "thermal_noise": "Thermal noise",
}


def _layers_from_arrays(
    thicknesses: np.ndarray, material_inds: np.ndarray, air_index: int = 0
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Trim at the first air layer and reverse, matching compute_state_value.

    Returns (d_opt, material_inds) in physics order, or None if no real layers.
    """
    keep_t, keep_m = [], []
    for i, (t, m) in enumerate(zip(thicknesses, material_inds)):
        if int(m) == air_index:
            break
        if t <= 0:
            break
        keep_t.append(float(t))
        keep_m.append(int(m))
    if not keep_t:
        return None
    # Reverse to match environment.compute_state_value (state_trim[::-1])
    return np.array(keep_t[::-1]), np.array(keep_m[::-1])


def _evaluate_layers(
    d_opt: np.ndarray,
    material_inds: np.ndarray,
    materials: Dict,
    wavelength_m: float,
    frequency: float = 100.0,
    wBeam: float = 0.062,
    temperature: float = 293.0,
    air_index: int = 0,
    substrate_index: int = 1,
) -> Dict[str, float]:
    """Evaluate one stack: lossy R/T/A via complex-index TMM + thermal noise."""
    wavelength_nm = wavelength_m * 1e9

    # Lossy transfer matrix: n + ik per layer, physical thickness in nm
    n_air = materials[air_index]["n"]
    n_sub = materials[substrate_index]["n"]
    n_list = [complex(n_air, 0)]
    t_list = [np.inf]
    for dopt, m in zip(d_opt, material_inds):
        n = materials[m]["n"]
        k = materials[m]["k"]
        n_list.append(complex(n, k))
        t_list.append(optical_to_physical(dopt, wavelength_nm, n))
    n_list.append(complex(n_sub, 0))
    t_list.append(np.inf)

    coh = tmm.coh_tmm("p", n_list, t_list, 0.0, wavelength_nm)
    reflectivity = float(coh["R"])
    transmission = float(coh["T"])
    absorption = max(0.0, 1.0 - reflectivity - transmission)

    # Thermal noise (same call and frequency handling as merit_function)
    noise_summary, _, _, _, _, _ = getCoatingThermalNoise(
        dOpt=d_opt,
        materialLayer=material_inds,
        materialParams=materials,
        materialSub=substrate_index,
        lambda_=wavelength_m,
        f=frequency,
        wBeam=wBeam,
        Temp=temperature,
        plots=False,
    )
    if isinstance(noise_summary["Frequency"], (float, np.floating)):
        thermal_noise = float(noise_summary["BrownianNoise"])
    else:
        idx = np.absolute(noise_summary["Frequency"] - 100).argmin()
        thermal_noise = float(noise_summary["BrownianNoise"][idx])

    return {
        "reflectivity": reflectivity,
        "transmission_ppm": transmission * 1e6,
        "absorption_ppm": absorption * 1e6,
        "total_loss": 1.0 - reflectivity,
        "thermal_noise": thermal_noise,
    }


def _nan_row() -> Dict[str, float]:
    return {c: np.nan for c in _RTA_COLUMNS}


def evaluate_designs_rta(
    designs_df: pd.DataFrame,
    materials: Dict,
    wavelength_m: float,
    frequency: float = 100.0,
    wBeam: float = 0.062,
    temperature: float = 293.0,
    use_optical_thickness: bool = True,
    air_index: int = 0,
    substrate_index: int = 1,
) -> pd.DataFrame:
    """Recompute R/T/A/CTN for every design in a Pareto designs DataFrame."""
    thickness_cols = sorted(
        [c for c in designs_df.columns if c.startswith("thickness_")],
        key=lambda c: int(c.split("_")[1]),
    )
    material_cols = sorted(
        [c for c in designs_df.columns if c.startswith("material_")],
        key=lambda c: int(c.split("_")[1]),
    )

    rows = []
    wavelength_nm = wavelength_m * 1e9
    for _, row in designs_df.iterrows():
        try:
            thicknesses = np.array([row[c] for c in thickness_cols], dtype=float)
            mats = np.array([int(row[c]) for c in material_cols])
            layers = _layers_from_arrays(thicknesses, mats, air_index)
            if layers is None:
                rows.append(_nan_row())
                continue
            d, m = layers
            if not use_optical_thickness:
                # Physical thickness (m) -> optical
                d = np.array(
                    [
                        di * materials[mi]["n"] / (wavelength_nm * 1e-9)
                        for di, mi in zip(d, m)
                    ]
                )
            rows.append(
                _evaluate_layers(
                    d,
                    m,
                    materials,
                    wavelength_m,
                    frequency,
                    wBeam,
                    temperature,
                    air_index,
                    substrate_index,
                )
            )
        except Exception:
            rows.append(_nan_row())
    return pd.DataFrame(rows, columns=_RTA_COLUMNS)


def evaluate_reference_rta(
    reference_values: pd.DataFrame,
    materials: Dict,
    wavelength_m: float,
    frequency: float = 100.0,
    wBeam: float = 0.062,
    temperature: float = 293.0,
    air_index: int = 0,
    substrate_index: int = 1,
) -> Optional[pd.DataFrame]:
    """Recompute R/T/A/CTN for the quarter-wave reference designs.

    Uses the 'thicknesses' and 'materials' comma-joined columns written by
    compare_outputs.create_reference_data.
    """
    if reference_values is None or "thicknesses" not in reference_values.columns:
        return None
    rows = []
    for _, ref in reference_values.iterrows():
        try:
            thicknesses = np.array(
                [float(t) for t in str(ref["thicknesses"]).split(",")]
            )
            mats = np.array([int(m) for m in str(ref["materials"]).split(",")])
            layers = _layers_from_arrays(thicknesses, mats, air_index)
            if layers is None:
                rows.append(_nan_row())
                continue
            d, m = layers
            rows.append(
                _evaluate_layers(
                    d,
                    m,
                    materials,
                    wavelength_m,
                    frequency,
                    wBeam,
                    temperature,
                    air_index,
                    substrate_index,
                )
            )
        except Exception:
            rows.append(_nan_row())
    return pd.DataFrame(rows, columns=_RTA_COLUMNS) if rows else None


# ── Physical-space plot ───────────────────────────────────────────────────────

# All pairwise combinations of {1-R, T, A, CTN}: (x, y) per panel, 3x2 grid
_RTA_PANELS = [
    ("absorption_ppm", "total_loss"),  # the classic absorption vs 1-R view
    ("total_loss", "thermal_noise"),
    ("transmission_ppm", "total_loss"),
    ("transmission_ppm", "absorption_ppm"),  # RTA budget plane (iso-1-R guides)
    ("absorption_ppm", "thermal_noise"),
    ("transmission_ppm", "thermal_noise"),
]

_RTA_PANEL_TITLES = [
    "<b>Absorption vs 1−R</b>",
    "<b>1−R vs thermal noise</b>",
    "<b>Transmission vs 1−R</b>",
    "<b>RTA budget: T vs A</b> (diagonals: constant 1−R)",
    "<b>Absorption vs thermal noise</b>",
    "<b>Transmission vs thermal noise</b>",
]


def _hover_customdata(df: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            df["total_loss"].values,
            df["transmission_ppm"].values,
            df["absorption_ppm"].values,
            df["thermal_noise"].values,
        ],
        axis=1,
    )


_HOVER_SUFFIX = (
    "1−R: %{customdata[0]:.3g}<br>"
    "T: %{customdata[1]:.3g} ppm<br>"
    "A: %{customdata[2]:.3g} ppm<br>"
    "CTN: %{customdata[3]:.3e}<extra></extra>"
)


def _add_iso_loss_guides(fig, runs, reference_rta, row, col):
    """Faint diagonal T + A = const guides on the T-A panel (iso-reflectivity).

    Guide coordinates are in ppm (the panel's axes); labels quote 1-R as a
    fraction.
    """
    losses = []
    for df, _ in runs:
        v = df["total_loss"].values
        losses.extend(v[np.isfinite(v) & (v > 0)].tolist())
    if reference_rta is not None:
        v = reference_rta["total_loss"].values
        losses.extend(v[np.isfinite(v) & (v > 0)].tolist())
    if not losses:
        return
    lo, hi = np.log10(min(losses)), np.log10(max(losses))
    decades = [10.0**e for e in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)]
    for c_frac in decades[:7]:
        c = c_frac * 1e6  # panel axes are in ppm
        x = np.logspace(np.log10(c) - 4, np.log10(c) - 1e-3, 120)
        y = c - x
        keep = y > 0
        fig.add_trace(
            go.Scatter(
                x=x[keep],
                y=y[keep],
                mode="lines",
                line=dict(color="lightgray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=np.log10(c) - 4,
            y=np.log10(c * 0.9999),
            text=f"1−R={c_frac:g}",
            showarrow=False,
            font=dict(size=8, color="gray"),
            xanchor="left",
            row=row,
            col=col,
        )


def plot_rta_comparison_interactive(
    runs: List[Tuple[pd.DataFrame, str]],
    reference_rta: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
    title: str = "R/T/A + thermal noise comparison",
    group_runs: bool = True,
    pareto_only: bool = True,
) -> go.Figure:
    """Physical-space comparison: all pairwise combinations of 1−R, T, A and
    thermal noise (6 log-log panels), recomputed values.

    Args:
        runs: List of (rta_df, label); rta_df from evaluate_designs_rta.
        reference_rta: Optional quarter-wave reference RTA DataFrame.
        save_path: If given, saves <stem>_rta_2d.html next to it.
        pareto_only: Show only per-panel non-dominated points as a step front.
    """
    color_map = _build_color_map([(df, None, lbl) for df, lbl in runs], group_runs)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=_RTA_PANEL_TITLES,
        horizontal_spacing=0.09,
        vertical_spacing=0.09,
    )

    panel_pos = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]

    # Iso-1-R diagonals on the T vs A budget panel
    ta_panel = _RTA_PANELS.index(("transmission_ppm", "absorption_ppm"))
    _add_iso_loss_guides(
        fig, runs, reference_rta, row=panel_pos[ta_panel][0], col=panel_pos[ta_panel][1]
    )

    for panel_idx, (obj_x, obj_y) in enumerate(_RTA_PANELS):
        r, c = panel_pos[panel_idx]
        first_panel = panel_idx == 0

        # Quarter-wave reference
        if reference_rta is not None:
            rx = reference_rta[obj_x].values
            ry = reference_rta[obj_y].values
            valid = np.isfinite(rx) & np.isfinite(ry) & (rx > 0) & (ry > 0)
            if valid.any():
                order = np.argsort(rx[valid])
                cd_ref = _hover_customdata(reference_rta)[valid][order]
                fig.add_trace(
                    go.Scatter(
                        x=rx[valid][order],
                        y=ry[valid][order],
                        mode="lines+markers",
                        line=dict(color="gray", width=1),
                        marker=dict(
                            symbol="x",
                            size=11,
                            color="black",
                            line=dict(width=2, color="black"),
                        ),
                        name="Quarter-wave reference",
                        legendgroup="reference",
                        showlegend=first_panel,
                        customdata=cd_ref,
                        hovertemplate="<b>QW reference</b><br>" + _HOVER_SUFFIX,
                    ),
                    row=r,
                    col=c,
                )

        for df, label in runs:
            color = color_map.get(label, "#808080")
            base_name, run_num = _parse_run_name(label)
            legend_group = base_name if group_runs else label
            short = (
                f"run{run_num:03d}"
                if (group_runs and run_num is not None)
                else (label[-40:] if len(label) > 40 else label)
            )

            x = df[obj_x].values
            y = df[obj_y].values
            cd = _hover_customdata(df)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            x, y, cd = x[valid], y[valid], cd[valid]
            if not len(x):
                continue
            if pareto_only:
                m = _pareto_2d(x, y)
                x, y, cd = x[m], y[m], cd[m]
            order = np.argsort(x)
            fig.add_trace(
                go.Scatter(
                    x=x[order],
                    y=y[order],
                    mode="lines+markers" if pareto_only else "markers",
                    line=dict(color=color, width=2, shape="hv"),
                    marker=dict(color=color, size=6, line=dict(width=1, color="black")),
                    name=short,
                    legendgroup=legend_group,
                    legendgrouptitle_text=(legend_group if group_runs else None),
                    showlegend=first_panel,
                    customdata=cd[order],
                    hovertemplate=f"<b>{short}</b><br>" + _HOVER_SUFFIX,
                ),
                row=r,
                col=c,
            )

        # Gold star at the best corner (everything minimised -> bottom left)
        bx, by = [], []
        for df, _ in runs:
            vx, vy = df[obj_x].values, df[obj_y].values
            ok = np.isfinite(vx) & np.isfinite(vy) & (vx > 0) & (vy > 0)
            bx.extend(vx[ok].tolist())
            by.extend(vy[ok].tolist())
        if bx and by:
            fig.add_trace(
                go.Scatter(
                    x=[min(bx)],
                    y=[min(by)],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color="#FFD700",
                        line=dict(width=1.5, color="black"),
                    ),
                    name="Best solution",
                    legendgroup="best_solution",
                    showlegend=first_panel,
                    hovertemplate=(
                        "<b>Best solution corner</b><br>"
                        f"{_RTA_LABELS[obj_x]}: %{{x:.3g}}<br>"
                        f"{_RTA_LABELS[obj_y]}: %{{y:.3g}}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(
            title_text=_RTA_LABELS[obj_x],
            type="log",
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=r,
            col=c,
        )
        fig.update_yaxes(
            title_text=_RTA_LABELS[obj_y],
            type="log",
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=r,
            col=c,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        height=1450,
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            tracegroupgap=10,
        ),
        margin=dict(b=170),
        hovermode="closest",
        template="plotly_white",
    )

    if save_path:
        html_path = Path(save_path).parent / (Path(save_path).stem + "_rta_2d.html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"Saved RTA comparison plot to {html_path}")
    return fig


# ── Reward-space plot ─────────────────────────────────────────────────────────

_NON_OBJECTIVE_REWARDS = {
    "constraint_penalty",
    "pareto_bonus",
    "bounds_penalty",
    "air_penalty",
    "total",
}


def plot_reward_comparison_interactive(
    pareto_fronts: List[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]],
    reference_rewards: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
    title: str = "Reward-space comparison",
    group_runs: bool = True,
    pareto_only: bool = True,
) -> Optional[go.Figure]:
    """Reward-space Pareto comparison: one panel per objective-reward pair.

    Args:
        pareto_fronts: List of (values_df, rewards_df, label) tuples (values
            unused; kept for signature compatibility with compare_outputs).
        reference_rewards: Optional reference rewards DataFrame.
        save_path: If given, saves <stem>_rewards_2d.html next to it.
    """
    # Detect reward objectives from the first run that has them
    objectives: List[str] = []
    for _, rdf, _ in pareto_fronts:
        if rdf is not None:
            objectives = [
                c
                for c in rdf.columns
                if c not in _NON_OBJECTIVE_REWARDS and rdf[c].dtype.kind == "f"
            ]
            if objectives:
                break
    if len(objectives) < 2:
        print("Reward plot skipped: fewer than 2 reward objectives found.")
        return None

    pairs = list(itertools.combinations(objectives, 2))
    n_pairs = len(pairs)
    n_cols = 2 if n_pairs > 1 else 1
    n_rows = int(np.ceil(n_pairs / n_cols))

    color_map = _build_color_map(pareto_fronts, group_runs)
    subplot_titles = [
        f"<b>{ox.replace('_', ' ').title()} vs {oy.replace('_', ' ').title()}</b>"
        for ox, oy in pairs
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    for pair_idx, (obj_x, obj_y) in enumerate(pairs):
        r = pair_idx // n_cols + 1
        c = pair_idx % n_cols + 1
        first_panel = pair_idx == 0

        if (
            reference_rewards is not None
            and obj_x in reference_rewards.columns
            and obj_y in reference_rewards.columns
        ):
            rx = reference_rewards[obj_x].values
            ry = reference_rewards[obj_y].values
            valid = np.isfinite(rx) & np.isfinite(ry)
            if valid.any():
                fig.add_trace(
                    go.Scatter(
                        x=rx[valid],
                        y=ry[valid],
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=11,
                            color="black",
                            line=dict(width=2, color="black"),
                        ),
                        name="Quarter-wave reference",
                        legendgroup="reference",
                        showlegend=first_panel,
                        hovertemplate=(
                            "<b>QW reference</b><br>"
                            f"{obj_x}: %{{x:.4f}}<br>{obj_y}: %{{y:.4f}}<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )

        for _, rdf, label in pareto_fronts:
            if rdf is None or obj_x not in rdf.columns or obj_y not in rdf.columns:
                continue
            color = color_map.get(label, "#808080")
            base_name, run_num = _parse_run_name(label)
            legend_group = base_name if group_runs else label
            short = (
                f"run{run_num:03d}"
                if (group_runs and run_num is not None)
                else (label[-40:] if len(label) > 40 else label)
            )
            x = rdf[obj_x].values
            y = rdf[obj_y].values
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if not len(x):
                continue
            if pareto_only:
                m = _pareto_2d(x, y, maximize=True)
                x, y = x[m], y[m]
            order = np.argsort(x)
            fig.add_trace(
                go.Scatter(
                    x=x[order],
                    y=y[order],
                    mode="lines+markers" if pareto_only else "markers",
                    line=dict(color=color, width=2, shape="hv"),
                    marker=dict(color=color, size=6, line=dict(width=1, color="black")),
                    name=short,
                    legendgroup=legend_group,
                    legendgrouptitle_text=(legend_group if group_runs else None),
                    showlegend=first_panel,
                    hovertemplate=(
                        f"<b>{short}</b><br>"
                        f"{obj_x} reward: %{{x:.4f}}<br>"
                        f"{obj_y} reward: %{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

        # Gold star at the best corner (rewards maximised -> top right)
        bx, by = [], []
        for _, rdf, _ in pareto_fronts:
            if rdf is None or obj_x not in rdf.columns or obj_y not in rdf.columns:
                continue
            vx, vy = rdf[obj_x].values, rdf[obj_y].values
            ok = np.isfinite(vx) & np.isfinite(vy)
            bx.extend(vx[ok].tolist())
            by.extend(vy[ok].tolist())
        if bx and by:
            fig.add_trace(
                go.Scatter(
                    x=[max(bx)],
                    y=[max(by)],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color="#FFD700",
                        line=dict(width=1.5, color="black"),
                    ),
                    name="Best solution",
                    legendgroup="best_solution",
                    showlegend=first_panel,
                    hovertemplate=(
                        "<b>Best solution corner</b><br>"
                        f"{obj_x} reward: %{{x:.4f}}<br>"
                        f"{obj_y} reward: %{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(
            title_text=f"{obj_x.replace('_', ' ').title()} reward",
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=r,
            col=c,
        )
        fig.update_yaxes(
            title_text=f"{obj_y.replace('_', ' ').title()} reward",
            gridcolor="lightgray",
            gridwidth=0.5,
            griddash="dash",
            row=r,
            col=c,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        height=max(520, 460 * n_rows),
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            tracegroupgap=10,
        ),
        margin=dict(b=160),
        hovermode="closest",
        template="plotly_white",
    )

    if save_path:
        html_path = Path(save_path).parent / (Path(save_path).stem + "_rewards_2d.html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"Saved reward-space comparison plot to {html_path}")
    return fig
