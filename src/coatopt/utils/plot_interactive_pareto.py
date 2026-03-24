#!/usr/bin/env python3
"""
Interactive Pareto front and coating design visualization using Plotly.

For 2 objectives: single scatter panel + coating design side by side.
For N > 2 objectives: lower-triangle pairwise grid (matching plot_pareto_projections.py)
with points coloured by a third objective, plus coating design in the top-right cell.
"""

import argparse
import configparser
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from coatopt.utils.interactive_plots import (
    _OBJ_CONFIG,
    _detect_objectives,
    _obj_label,
    _obj_scale,
    _obj_transform,
)
from coatopt.utils.utils import load_pareto_front


def load_materials(materials_path: str) -> Dict:
    """Load material properties from JSON file."""
    with open(materials_path, "r") as f:
        materials = json.load(f)
    return {int(k): v for k, v in materials.items()}


def parse_design(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract thicknesses and materials from a Pareto front row.

    Args:
        row: DataFrame row containing thickness_i and material_i columns

    Returns:
        Tuple of (thicknesses, material_indices) arrays
    """
    # Find all thickness and material columns
    thickness_cols = [col for col in row.index if col.startswith("thickness_")]
    material_cols = [col for col in row.index if col.startswith("material_")]

    # Sort by index number
    thickness_cols = sorted(thickness_cols, key=lambda x: int(x.split("_")[1]))
    material_cols = sorted(material_cols, key=lambda x: int(x.split("_")[1]))

    # Extract values
    thicknesses = np.array([row[col] for col in thickness_cols])
    materials = np.array([int(row[col]) for col in material_cols])

    return thicknesses, materials


def create_coating_trace(
    thicknesses: np.ndarray,
    material_indices: np.ndarray,
    materials: Dict,
    name: str = "Coating",
    visible: bool = True,
) -> List[go.Bar]:
    """Create bar chart traces for coating design visualization.

    Args:
        thicknesses: Array of layer thicknesses
        material_indices: Array of material indices
        materials: Material properties dictionary
        name: Name for the trace
        visible: Whether trace is initially visible

    Returns:
        Single bar trace with all layers stacked
    """
    # Material colors
    color_map = {
        "air": "#F0F0F0",
        "SiO2": "#1f77b4",
        "Ti:Ta2O5": "#ff7f0e",
        "aSi": "#2ca02c",
    }

    # Convert to nanometers
    thicknesses_nm = thicknesses * 1e9

    # Get colors for each layer based on material
    colors = [
        color_map.get(materials[idx]["name"], "#cccccc") for idx in material_indices
    ]

    # Create hover text for each layer
    hover_text = [
        f"Layer {i}<br>{materials[material_indices[i]]['name']}<br>Thickness: {thicknesses_nm[i]:.2f} nm"
        for i in range(len(thicknesses))
    ]

    # Single trace with all layers
    trace = go.Bar(
        name=name,
        x=[0] * len(thicknesses_nm),  # Single column
        y=thicknesses_nm,
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=1,
        showlegend=False,
        visible=visible,
        hovertemplate="%{text}<extra></extra>",
        text=hover_text,
    )

    return [trace]


def create_interactive_plot(
    designs_df: pd.DataFrame,
    values_df: pd.DataFrame,
    materials: Dict,
    max_designs: int = None,
) -> Tuple[go.Figure, int]:
    """Create interactive Plotly figure with pairwise Pareto projections and coating designs.

    For 2 objectives: single Pareto scatter + coating design (1×2 grid).
    For N > 2 objectives: lower-triangle pairwise grid + coating design in top-right.
    Points are coloured by a third objective (viridis) when available.

    Args:
        designs_df: DataFrame with design variables
        values_df: DataFrame with objective values
        materials: Material properties dictionary
        max_designs: Maximum number of designs to include (None = all)

    Returns:
        (figure, n_pairs) – the Plotly figure and the number of Pareto scatter panels,
        needed to correctly route click events in the HTML post_script.
    """
    # ── sort & limit ──────────────────────────────────────────────────────────
    combined_df = pd.concat([designs_df, values_df], axis=1)
    sort_col = (
        "reflectivity"
        if "reflectivity" in combined_df.columns
        else combined_df.columns[0]
    )
    combined_df = combined_df.sort_values(sort_col, ascending=False).reset_index(
        drop=True
    )
    if max_designs is not None:
        combined_df = combined_df.head(max_designs)
    n_designs = len(combined_df)

    # ── objectives ────────────────────────────────────────────────────────────
    objectives = _detect_objectives(values_df)
    n_obj = len(objectives)
    pairs = list(itertools.combinations(range(n_obj), 2))
    n_pairs = len(pairs)

    # ── layout ────────────────────────────────────────────────────────────────
    # Grid: n_rows = n_obj-1, n_cols = n_obj
    # Lower triangle (col <= row): pairwise Pareto panels
    # Top-right corner (row=0, col=n_obj-1): coating design
    n_rows = max(1, n_obj - 1)
    n_cols = n_obj  # last col reserved for coating

    # Build specs (None = hidden/empty cell)
    specs = []
    for r in range(n_rows):
        row_specs = []
        for c in range(n_cols):
            if r == 0 and c == n_cols - 1:
                row_specs.append({"type": "bar"})
            elif c < n_cols - 1 and c <= r:
                row_specs.append({"type": "scatter"})
            else:
                row_specs.append(None)
        specs.append(row_specs)

    # Equal column widths
    col_widths = [1.0 / n_cols] * n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        column_widths=col_widths,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # Map pair index → (plotly_row, plotly_col) – both 1-indexed
    # Pair (ci, cj) with ci < cj maps to grid cell (row=cj-1, col=ci) 0-indexed
    # → Plotly (row=cj, col=ci+1)
    panel_pos = [(cj, ci + 1) for (ci, cj) in pairs]

    # ── colour range for third-objective coloring ─────────────────────────────
    color_ranges: Dict[int, Tuple[float, float]] = {}
    for ci, cj in pairs:
        ck_list = [k for k in range(n_obj) if k not in (ci, cj)]
        if ck_list:
            ck = ck_list[0]
            if ck not in color_ranges:
                vals = _obj_transform(
                    objectives[ck], combined_df[objectives[ck]].values
                )
                fin = vals[np.isfinite(vals)]
                if len(fin):
                    color_ranges[ck] = (float(fin.min()), float(fin.max()))

    # ── Trace layout ──────────────────────────────────────────────────────────
    # Indices 0 … n_pairs-1                : main Pareto scatter (always visible)
    # Indices n_pairs … n_pairs*(1+n_designs)-1: highlighted points (pair-major order)
    # Indices n_pairs*(1+n_designs) … +n_designs-1: coating bars

    # Add main Pareto scatter traces (one per pair panel)
    for pair_idx, (ci, cj) in enumerate(pairs):
        r1, c1 = panel_pos[pair_idx]
        obj_x, obj_y = objectives[ci], objectives[cj]
        x_vals = _obj_transform(obj_x, combined_df[obj_x].values)
        y_vals = _obj_transform(obj_y, combined_df[obj_y].values)

        ck_list = [k for k in range(n_obj) if k not in (ci, cj)]
        ck = ck_list[0] if ck_list else None

        if ck is not None and ck in color_ranges:
            color_vals = _obj_transform(
                objectives[ck], combined_df[objectives[ck]].values
            )
            vmin, vmax = color_ranges[ck]
            # Only show colorbar on the first panel to avoid duplicates
            show_cbar = pair_idx == 0
            marker = dict(
                size=10,
                color=color_vals,
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                line=dict(width=0.8, color="black"),
                colorbar=(
                    dict(
                        title=dict(text=_obj_label(objectives[ck]), side="right"),
                        len=0.5,
                        x=1.02,
                    )
                    if show_cbar
                    else None
                ),
                showscale=show_cbar,
            )
        else:
            marker = dict(
                size=10, color="steelblue", line=dict(width=0.8, color="black")
            )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=marker,
                name=f"{_obj_label(obj_x)} vs {_obj_label(obj_y)}",
                customdata=list(range(n_designs)),
                showlegend=False,
                hovertemplate=(
                    "Design %{customdata}<br>"
                    + f"{_obj_label(obj_x)}: %{{x:.3e}}<br>"
                    + f"{_obj_label(obj_y)}: %{{y:.3e}}<br>"
                    + "<i>Click to view design</i><extra></extra>"
                ),
            ),
            row=r1,
            col=c1,
        )

    # Add highlighted point traces (n_pairs × n_designs, pair-major order)
    for pair_idx, (ci, cj) in enumerate(pairs):
        r1, c1 = panel_pos[pair_idx]
        obj_x, obj_y = objectives[ci], objectives[cj]

        for idx in range(n_designs):
            x_val = float(
                _obj_transform(obj_x, np.array([combined_df.iloc[idx][obj_x]]))[0]
            )
            y_val = float(
                _obj_transform(obj_y, np.array([combined_df.iloc[idx][obj_y]]))[0]
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_val],
                    y=[y_val],
                    mode="markers",
                    marker=dict(
                        size=15, color="red", line=dict(width=2, color="darkred")
                    ),
                    name="Selected",
                    showlegend=False,
                    visible=(idx == 0),
                ),
                row=r1,
                col=c1,
            )

    # Add coating design traces (top-right corner, row=1, col=n_cols)
    coating_row, coating_col = 1, n_cols
    for idx in range(n_designs):
        row_data = combined_df.iloc[idx]
        thicknesses, material_indices = parse_design(row_data)
        traces = create_coating_trace(
            thicknesses,
            material_indices,
            materials,
            f"Design {idx}",
            visible=(idx == 0),
        )
        for trace in traces:
            fig.add_trace(trace, row=coating_row, col=coating_col)

    # ── Dropdown buttons ──────────────────────────────────────────────────────
    # Visibility layout:
    #   [0 .. n_pairs-1]                        always True
    #   [n_pairs + pair*n_designs + d]           True only when d == idx
    #   [n_pairs*(1+n_designs) + d]              True only when d == idx
    buttons = []
    for idx in range(n_designs):
        row_data = combined_df.iloc[idx]
        title_parts = [
            f"{_obj_label(obj)}={row_data[obj]:.4e}"
            for obj in objectives
            if obj in row_data
        ]
        title_str = f"Design {idx + 1}<br>" + " | ".join(title_parts)

        vis: List[bool] = []
        for _ in range(n_pairs):  # main pareto panels: always on
            vis.append(True)
        for _ in range(n_pairs):  # highlighted points per panel
            for d in range(n_designs):
                vis.append(d == idx)
        for d in range(n_designs):  # coating traces
            vis.append(d == idx)

        buttons.append(
            dict(
                label=f"Design {idx + 1}",
                method="update",
                args=[{"visible": vis}, {"title": title_str}],
            )
        )

    # ── Axis configuration ────────────────────────────────────────────────────
    for pair_idx, (ci, cj) in enumerate(pairs):
        r1, c1 = panel_pos[pair_idx]
        obj_x, obj_y = objectives[ci], objectives[cj]
        fig.update_xaxes(
            title_text=_obj_label(obj_x),
            type=_obj_scale(obj_x),
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            row=r1,
            col=c1,
        )
        fig.update_yaxes(
            title_text=_obj_label(obj_y),
            type=_obj_scale(obj_y),
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            row=r1,
            col=c1,
        )

    fig.update_xaxes(showticklabels=False, row=coating_row, col=coating_col)
    fig.update_yaxes(
        title_text="Thickness (nm)",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        row=coating_row,
        col=coating_col,
    )

    # Add panel title annotations manually (subplot_titles not used to avoid
    # ambiguity with None specs)
    annotations = list(fig.layout.annotations)
    for pair_idx, (ci, cj) in enumerate(pairs):
        r1, c1 = panel_pos[pair_idx]
        obj_x, obj_y = objectives[ci], objectives[cj]
        # Get the subplot domain to position the title
        xref = f"x{'' if c1 == 1 else c1} domain"
        yref = f"y{'' if r1 == 1 else r1} domain"
        annotations.append(
            dict(
                text=f"<b>{_obj_label(obj_x)} vs {_obj_label(obj_y)}</b>",
                xref=xref,
                yref=yref,
                x=0.5,
                y=1.05,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11),
            )
        )
    annotations.append(
        dict(
            text="<b>Coating Design</b>",
            xref=f"x{n_cols} domain",
            yref="y domain",
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=11),
        )
    )
    fig.update_layout(annotations=annotations)

    # ── Overall layout ────────────────────────────────────────────────────────
    height = max(600, 320 * n_rows)
    width = max(1200, 340 * n_cols)

    fig.update_layout(
        title_text=(
            f"Interactive Pareto Front — {n_designs} designs, {n_obj} objectives<br>"
            f"<sub>Click any point to view its coating design</sub>"
        ),
        title_x=0.5,
        title_font_size=16,
        showlegend=False,
        height=height,
        width=width,
        hovermode="closest",
        template="plotly_white",
        barmode="stack",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.02,
                xanchor="left",
                y=1.0,
                yanchor="top",
            )
        ],
    )

    return fig, n_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive Pareto front and coating design visualization",
        epilog="""
Example:
  python plot_interactive_pareto.py experiments/outputs/20_layer/genetic
        """,
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing config.ini and pareto_front_values.csv",
    )
    parser.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Maximum number of designs to include (default: all designs)",
    )

    args = parser.parse_args()

    # Convert to Path object, resolving relative to current working directory
    directory = Path(args.directory)
    if not directory.is_absolute():
        # Resolve relative to the actual shell's current directory
        directory = Path(os.getcwd()) / directory

    directory = directory.resolve()

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return 1

    # Find config.ini
    config_path = directory / "config.ini"
    if not config_path.exists():
        print(f"Error: config.ini not found in {directory}")
        return 1

    # Load config to get materials path
    config = configparser.ConfigParser()
    config.read(config_path)

    try:
        materials_path = config.get("general", "materials_path")
        # Resolve relative to config directory or one level above
        if not Path(materials_path).is_absolute():
            candidate1 = (config_path.parent / materials_path).resolve()
            candidate2 = (config_path.parent.parent / materials_path).resolve()
            candidate3 = (config_path.parent.parent.parent / materials_path).resolve()
            candidate4 = (
                config_path.parent.parent.parent.parent / materials_path
            ).resolve()

            if candidate1.exists():
                materials_path = candidate1
            elif candidate2.exists():
                materials_path = candidate2
            elif candidate3.exists():
                materials_path = candidate3
            elif candidate4.exists():
                materials_path = candidate4
            else:
                print(f"Error: Could not find materials file at:")
                print(f"  {candidate1}")
                print(f"  {candidate2}")
                print(f"  {candidate3}")
                return 1
        else:
            materials_path = Path(materials_path)
    except (configparser.NoSectionError, configparser.NoOptionError):
        print(f"Error: Could not find 'materials_path' in config.ini")
        return 1

    # Output path
    output_path = directory / "pareto_interactive.html"

    print(f"Directory: {directory}")
    print(f"Loading Pareto front...")
    try:
        designs_df, values_df, rewards_df = load_pareto_front(directory)
        print(f"  Found {len(designs_df)} designs")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loading materials from {materials_path}...")
    materials = load_materials(str(materials_path))
    print(f"  Found {len(materials)} materials")

    print(f"Creating interactive visualization...")
    fig, n_pairs = create_interactive_plot(
        designs_df, values_df, materials, max_designs=args.max_designs
    )

    print(f"Saving to {output_path}...")
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    # Click on any main Pareto scatter panel (curveNumber 0 .. n_pairs-1)
    # triggers the dropdown to show that design's coating
    post_script = f"""
        var N_PAIRS = {n_pairs};
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {{
            plotDiv.on('plotly_click', function(data) {{
                var pt = data.points[0];
                if (pt.curveNumber < N_PAIRS) {{
                    var designIdx = pt.customdata;
                    var buttons = document.querySelectorAll('[data-title^="Design"]');
                    if (buttons[designIdx]) {{ buttons[designIdx].click(); }}
                }}
            }});
        }}
    """
    fig.write_html(
        str(output_path),
        config=plotly_config,
        include_plotlyjs="cdn",
        post_script=post_script,
    )

    print(f"\nDone! Open {output_path} in your browser to view the interactive plot.")
    print(f"Click on any point in the Pareto front to view its coating design.")
    print(
        f"You can also use the dropdown menu on the right to select specific designs."
    )

    return 0


if __name__ == "__main__":
    exit(main())
