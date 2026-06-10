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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from coatopt.utils.interactive_plots import (
    _detect_objectives,
    _obj_label,
    _obj_scale,
    _obj_transform,
)
from coatopt.utils.utils import load_pareto_front

# Setup library path for CoatingAnalysis once at module level
lib_path = "/Users/simon/Library/CloudStorage/GoogleDrive-simon.tait@ligo.org/My Drive/BackupFromDropbox/Python/CoatingAnalysis/src"
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

try:
    from coating_analysis.Coatings_development import thin_film_stack
except ImportError:
    thin_film_stack = None


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
    thicknesses_nm: np.ndarray,
    material_indices: np.ndarray,
    materials: Dict,
    name: str = "Coating",
    visible: bool = True,
    shown_in_legend: set = None,
) -> List[go.Bar]:
    """Create bar chart traces for coating design visualization.

    Each layer is added as a separate go.Bar trace so Plotly stacks them correctly.
    """
    # Material colors mapping
    color_map = {
        "air": "#F0F0F0",
        "SiO2": "#1f77b4",
        "Ti:Ta2O5": "#c837ab",
        "TiGermania": "#c837ab",
        "substrate": "#7f7f7f",
    }

    traces = []
    accumulated_thickness = 0.0
    
    # We add one trace per layer in order from substrate to air (or bottom to top)
    for i in range(len(thicknesses_nm)):
        mat_idx = material_indices[i]
        mat_name = materials.get(mat_idx, {}).get("name", "Unknown")
        color = color_map.get(mat_name, "#7f7f7f")
        thick_val = thicknesses_nm[i]
        
        # Build legend visibility (only show each material name once in legend)
        show_leg = False
        if shown_in_legend is not None and mat_name not in shown_in_legend:
            show_leg = True
            shown_in_legend.add(mat_name)

        # Build hover info
        hover_text = (
            f"Layer {i+1}<br>"
            f"Material: {mat_name}<br>"
            f"Thickness: {thick_val:.2f} nm<br>"
            f"Cumulative: {accumulated_thickness + thick_val:.2f} nm"
        )
        
        # Create a single bar for this layer
        trace = go.Bar(
            name=mat_name,
            x=["Coating Stack"],
            y=[thick_val],
            marker=dict(
                color=color,
                line=dict(color="black", width=1.5)
            ),
            showlegend=show_leg,
            legendgroup=mat_name,
            visible=visible,
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
        )
        traces.append(trace)
        accumulated_thickness += thick_val

    return traces


def get_physical_thicknesses(
    dOpt: np.ndarray,
    materialLayer: np.ndarray,
    materials: dict,
    lambda_nm: float = 1064.0,
) -> np.ndarray:
    """Calculate physical thicknesses in nanometers using thin_film_stack or fallback."""
    # Filter out inactive layers (where material is 0/Air or thickness is 0)
    active_mask = (materialLayer != 0) & (dOpt > 0)
    if not np.any(active_mask):
        return np.array([])
        
    active_dOpt = dOpt[active_mask]
    active_materialLayer = materialLayer[active_mask]
    
    # Use preloaded library
    try:
        if thin_film_stack is None:
            raise ImportError("thin_film_stack is not imported")
        
        # Build materialParams structure
        materialParams = {}
        for k, v in materials.items():
            mat_key = int(k)
            mat_data = v.copy()
            if mat_data.get("n") is None:
                mat_data["n"] = 1.0
            if mat_data.get("k") is None:
                mat_data["k"] = 0.0
                
            if mat_key == 0:
                materialParams[999] = mat_data
                materialParams[0] = mat_data
            else:
                materialParams[mat_key] = mat_data
        
        if 999 not in materialParams:
            materialParams[999] = {'name': 'air', 'n': 1.0, 'k': 0.0}
            materialParams[0] = {'name': 'air', 'n': 1.0, 'k': 0.0}
            
        # Map materialLayer: 0 becomes 999
        mapped_layer = np.array([999 if m == 0 else m for m in active_materialLayer])
        n_input = np.array([materialParams[m]['n'] for m in mapped_layer])
        
        # Call thin_film_stack
        _, _, d_physical_m = thin_film_stack(
            dOpt=active_dOpt,
            n_input=n_input,
            materialLayer=mapped_layer,
            materialParams=materialParams,
            lambda_=lambda_nm,
            plots=False,
            verbose=False,
        )
        # Convert meters to nanometers if returned in meters
        if np.any(d_physical_m > 1e-3):
            return d_physical_m
        else:
            return d_physical_m * 1e9
        
    except Exception as e:
        # Fallback to direct Python calculation
        d_physical_nm = []
        for i in range(len(active_dOpt)):
            mat_idx = active_materialLayer[i]
            refractive_index = materials.get(mat_idx, {}).get('n', 1.45)
            # physical = optical * wavelength / n
            t_nm = active_dOpt[i] * lambda_nm / refractive_index
            d_physical_nm.append(t_nm)
        return np.array(d_physical_nm)


def create_interactive_plot(
    designs_df: pd.DataFrame,
    values_df: pd.DataFrame,
    materials: Dict,
    max_designs: int = None,
) -> Tuple[go.Figure, int]:
    """Create interactive Plotly figure with a premium Coating-Inspection two-column layout.

    Left Column: 2x2 grid representing Pareto projections (R vs Abs, R vs TN, Abs vs TN).
    Right Column: 3 vertically stacked axes representing physical diagnostics of the selected design:
      - Coating stack diagram (stacked bars with legend)
      - Electric field profile (blue lines with dashed interface markers)
      - Simulated spectral response (transmission spectrum line)
    """
    # Import physics functions from CoatingAnalysis
    lib_path = "/Users/simon/Library/CloudStorage/GoogleDrive-simon.tait@ligo.org/My Drive/BackupFromDropbox/Python/CoatingAnalysis/src"
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    from coating_analysis.EFI_tmm import CalculateEFI_tmm, CalculateTransmission_tmm

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

    # Calculate active layer counts and template max_layers
    active_counts = []
    max_layers = 50
    for idx, row in combined_df.iterrows():
        dOpt, mat_idx = parse_design(row)
        active_mask = (mat_idx != 0) & (dOpt > 0)
        active_counts.append(int(np.sum(active_mask)))
        max_layers = len(dOpt)
    combined_df['active_layer_count'] = active_counts
    combined_df['max_layers'] = max_layers

    # ── objectives ────────────────────────────────────────────────────────────
    objectives = _detect_objectives(values_df)
    n_obj = len(objectives)
    pairs = list(itertools.combinations(range(n_obj), 2))
    n_pairs = len(pairs)

    # ── layout ────────────────────────────────────────────────────────────────
    # Grid: 6 rows, 3 columns
    # Left column spans 3 rows for each of the 2 Pareto rows (Col 1 & Col 2)
    # Right column spans 2 rows for each of the 3 diagnostic subplots (Col 3)
    n_rows = 6
    n_cols = 3

    specs = [
        # Row 1
        [
            {"type": "scatter", "rowspan": 3},   # Col 1: R vs Absorption
            None,                                 # Col 2: Empty
            {"type": "bar", "rowspan": 2},       # Col 3: Coating stack
        ],
        # Row 2
        [
            None,                                 # Col 1: Spanned
            None,                                 # Col 2: Empty
            None,                                 # Col 3: Spanned
        ],
        # Row 3
        [
            None,                                 # Col 1: Spanned
            None,                                 # Col 2: Empty
            {"type": "scatter", "rowspan": 2},   # Col 3: Electric field
        ],
        # Row 4
        [
            {"type": "scatter", "rowspan": 3},   # Col 1: R vs TN
            {"type": "scatter", "rowspan": 3},   # Col 2: Abs vs TN
            None,                                 # Col 3: Spanned
        ],
        # Row 5
        [
            None,                                 # Col 1: Spanned
            None,                                 # Col 2: Spanned
            {"type": "scatter", "rowspan": 2},   # Col 3: Spectral response
        ],
        # Row 6
        [
            None,                                 # Col 1: Spanned
            None,                                 # Col 2: Spanned
            None,                                 # Col 3: Spanned
        ],
    ]

    # Width ratios: Left column is slightly wider to hold the 2x2 grid beautifully
    coating_width = 0.44
    scatter_cols = 2
    col_widths = []
    for c in range(scatter_cols):
        col_widths.append((1.0 - coating_width) / scatter_cols)
    col_widths.append(coating_width)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        column_widths=col_widths,
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    # Subplot placements matching left-hand and right-hand targets
    panel_pos = [
        (1, 1),  # 1 - Reflectivity vs Absorption (row=1, col=1)
        (4, 1),  # 1 - Reflectivity vs Thermal Noise (row=4, col=1)
        (4, 2),  # Absorption vs Thermal Noise (row=4, col=2)
    ]

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
    # Track the exact trace counts for visibility dropdown configuration
    diagnostic_traces_by_design = {d: [] for d in range(n_designs)}
    total_traces = 0

    # 1. Add main Pareto scatter traces
    for pair_idx, (ci, cj) in enumerate(pairs):
        r1, c1 = panel_pos[pair_idx]
        obj_x, obj_y = objectives[ci], objectives[cj]
        x_vals = _obj_transform(obj_x, combined_df[obj_x].values)
        y_vals = _obj_transform(obj_y, combined_df[obj_y].values)

        ck_list = [k for k in range(n_obj) if k not in (ci, cj)]
        ck = ck_list[0] if ck_list else None

        active_c = combined_df['active_layer_count'].values
        max_l = int(combined_df['max_layers'].values[0])
        thresh = int(0.8 * max_l)
        
        sizes = [10 if count >= thresh else 6 for count in active_c]
        symbols = ['circle' if count >= thresh else 'x' for count in active_c]

        if ck is not None and ck in color_ranges:
            color_vals = _obj_transform(
                objectives[ck], combined_df[objectives[ck]].values
            )
            vmin, vmax = color_ranges[ck]
            show_cbar = pair_idx == 0
            
            # Position colorbar in the gap between scatter plots and the coating stack
            cbar_x = (1.0 - coating_width) - 0.02
            
            marker = dict(
                size=sizes,
                symbol=symbols,
                color=color_vals,
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                line=dict(width=0.8, color="black"),
                colorbar=(
                    dict(
                        title=dict(text=_obj_label(objectives[ck]), side="right"),
                        len=0.5,
                        x=cbar_x,
                    )
                    if show_cbar
                    else None
                ),
                showscale=show_cbar,
            )
        else:
            marker = dict(
                size=sizes,
                symbol=symbols,
                color="steelblue",
                line=dict(width=0.8, color="black")
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
        total_traces += 1

    # 2. Add highlighted point traces
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
                    name=f"highlight_{idx}",
                    showlegend=False,
                    visible=(idx == 0),
                ),
                row=r1,
                col=c1,
            )
            total_traces += 1

    # Build materialParams structure dynamically
    materialParams = {}
    for k, v in materials.items():
        mat_key = int(k)
        mat_data = v.copy()
        if mat_data.get("n") is None:
            mat_data["n"] = 1.0
        if mat_data.get("k") is None:
            mat_data["k"] = 0.0
            
        if mat_key == 0:
            materialParams[999] = mat_data
            materialParams[0] = mat_data
        else:
            materialParams[mat_key] = mat_data
            
    if 999 not in materialParams:
        materialParams[999] = {'name': 'air', 'n': 1.0, 'k': 0.0}
        materialParams[0] = {'name': 'air', 'n': 1.0, 'k': 0.0}
    if 1 not in materialParams:
        materialParams[1] = {'name': 'SiO2', 'n': 1.45, 'k': 0.0}

    # 3. Add coating stack diagram, electric field, and spectral response traces
    shown_in_legend = set()
    
    try:
        from tqdm import tqdm
        design_iterator = tqdm(range(n_designs), desc="  Simulating coating designs")
    except ImportError:
        design_iterator = range(n_designs)
        
    for idx in design_iterator:
        if not hasattr(design_iterator, "container") and (idx % 50 == 0 or idx == n_designs - 1):
            print(f"  Simulated {idx + 1}/{n_designs} designs...")
            
        row_data = combined_df.iloc[idx]
        dOpt, material_indices = parse_design(row_data)
        
        # Filter active layers
        active_mask = (material_indices != 0) & (dOpt > 0)
        active_dOpt = dOpt[active_mask]
        active_materialLayer = material_indices[active_mask]

        # A. Physical thicknesses
        thicknesses_nm = get_physical_thicknesses(dOpt, material_indices, materials)
        
        # Coating Stack Bars
        traces = create_coating_trace(
            thicknesses_nm,
            active_materialLayer,
            materials,
            f"coating_{idx}",
            visible=(idx == 0),
            shown_in_legend=shown_in_legend if idx == 0 else None,
        )
        for trace in traces:
            fig.add_trace(trace, row=1, col=3)
            diagnostic_traces_by_design[idx].append(total_traces)
            total_traces += 1
            
        # Add substrate bar
        if len(thicknesses_nm) > 0:
            sub_trace = go.Bar(
                name="Substrate",
                x=["Coating Stack"],
                y=[100.0],  # Mock substrate thickness
                marker=dict(color="#7f7f7f", line=dict(color="black", width=1.5)),
                showlegend=(idx == 0 and "Substrate" not in shown_in_legend),
                legendgroup="Substrate",
                visible=(idx == 0),
                hovertemplate="Substrate (SiO2)<extra></extra>",
            )
            if idx == 0:
                shown_in_legend.add("Substrate")
            fig.add_trace(sub_trace, row=1, col=3)
            diagnostic_traces_by_design[idx].append(total_traces)
            total_traces += 1

        # B. Electric Field Profile using CalculateEFI_tmm
        try:
            # We pass active layers with t_air=500nm
            _, _, ds, E, _, _, _ = CalculateEFI_tmm(
                dOpt=active_dOpt,
                materialLayer=active_materialLayer,
                materialParams=materialParams,
                lambda_=1064.0,  # Pass in nanometers
                plots=False,
            )
            
            # Add blue E-field line trace
            field_trace = go.Scatter(
                x=ds,
                y=E,
                mode="lines",
                line=dict(color="blue", width=2),
                name=f"field_{idx}",
                showlegend=False,
                visible=(idx == 0),
                hovertemplate="Depth: %{x:.1f} nm<br>Electric Field: %{y:.3f}<extra></extra>",
            )
            fig.add_trace(field_trace, row=3, col=3)
            diagnostic_traces_by_design[idx].append(total_traces)
            total_traces += 1
            
            # Add vertical dashed lines representing interfaces
            accumulated = 0.0
            # Air to Layer 1 interface is at 0
            interfaces = [0.0]
            for t_val in thicknesses_nm:
                accumulated += t_val
                interfaces.append(accumulated)
                
            for x_val in interfaces:
                line_trace = go.Scatter(
                     x=[x_val, x_val],
                     y=[0, max(E) * 1.1 if len(E) > 0 else 4.0],
                     mode="lines",
                     line=dict(color="gray", width=1, dash="dash"),
                     name=f"field_{idx}",
                     showlegend=False,
                     visible=(idx == 0),
                     hoverinfo="skip",
                )
                fig.add_trace(line_trace, row=3, col=3)
                diagnostic_traces_by_design[idx].append(total_traces)
                total_traces += 1
                
        except Exception as e:
            print(f"Warning: Could not calculate electric field profile: {e}")
 
        # C. Simulated Spectral Response using CalculateTransmission_tmm
        try:
            # Wavelength list from 400nm to 1400nm (in nanometers)
            lambda_list = np.linspace(400.0, 1400.0, 200)
            wavelengths, transmission, _ = CalculateTransmission_tmm(
                dOpt=active_dOpt,
                materialLayer=active_materialLayer,
                materialParams=materialParams,
                lambda_list=lambda_list,
                lambda_0=1064.0,  # Pass in nanometers
                plots=False,
            )
            
            # Add orange transmission trace
            spectrum_trace = go.Scatter(
                x=wavelengths,
                y=transmission * 100,  # Convert fraction to percentage
                mode="lines",
                line=dict(color="#ff7f0e", width=2),
                name=f"spectrum_{idx}",
                showlegend=False,
                visible=(idx == 0),
                hovertemplate="Wavelength: %{x:.1f} nm<br>Transmission: %{y:.3f}%<extra></extra>",
            )
            fig.add_trace(spectrum_trace, row=5, col=3)
            diagnostic_traces_by_design[idx].append(total_traces)
            total_traces += 1
            
        except Exception as e:
            print(f"Warning: Could not calculate spectral response: {e}")

    # ── Dropdown buttons ──────────────────────────────────────────────────────
    buttons = []
    for idx in range(n_designs):
        row_data = combined_df.iloc[idx]
        title_parts = [
            f"{_obj_label(obj)}={row_data[obj]:.4e}"
            for obj in objectives
            if obj in row_data
        ]
        title_str = f"Design {idx + 1}<br>" + " | ".join(title_parts)

        # Build precise visibility array
        vis = [True] * n_pairs  # main scatter always True
        
        # Highlighted traces
        for pair_idx in range(n_pairs):
            for d in range(n_designs):
                vis.append(d == idx)
                
        # Coating, field, and spectrum diagnostic traces
        for d in range(n_designs):
            n_diag = len(diagnostic_traces_by_design[d])
            for _ in range(n_diag):
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

    # Coating Stack formatting
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(
        title_text="Physical Thickness (nm)",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        row=1,
        col=3,
    )

    # Electric Field formatting
    fig.update_xaxes(title_text="Depth (nm)", showgrid=True, row=3, col=3)
    fig.update_yaxes(
        title_text="Electric Field Intensity",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        row=3,
        col=3,
    )

    # Simulated Spectral Response formatting
    fig.update_xaxes(title_text="Wavelength (nm)", showgrid=True, row=5, col=3)
    fig.update_yaxes(
        title_text="Transmission (%)",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        row=5,
        col=3,
    )

    # Add annotations
    annotations = list(fig.layout.annotations)
    for pair_idx, (ci, cj) in enumerate(pairs):
        r1, c1 = panel_pos[pair_idx]
        obj_x, obj_y = objectives[ci], objectives[cj]
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
            text="<b>Coating Stack Diagram</b>",
            xref="x3 domain",
            yref="y domain",
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
            text="<b>Electric Field Profile</b>",
            xref="x3 domain",
            yref="y3 domain",
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
            text="<b>Simulated Spectral Response</b>",
            xref="x3 domain",
            yref="y5 domain",
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
    height = 950
    width = 1600

    fig.update_layout(
        title_text=(
            f"Interactive Pareto Front & Coating Inspection Dashboard<br>"
            f"<sub>Click any point in the left panels to inspect its physical diagnostics on the right</sub>"
        ),
        title_x=0.5,
        title_font_size=16,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
        ),
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
                y=0.5,
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
                print("Error: Could not find materials file at:")
                print(f"  {candidate1}")
                print(f"  {candidate2}")
                print(f"  {candidate3}")
                return 1
        else:
            materials_path = Path(materials_path)
    except (configparser.NoSectionError, configparser.NoOptionError):
        print("Error: Could not find 'materials_path' in config.ini")
        return 1

    # Output path
    output_path = directory / "pareto_interactive.html"

    print(f"Directory: {directory}")
    print("Loading Pareto front...")
    try:
        designs_df, values_df, rewards_df = load_pareto_front(directory)
        print(f"  Found {len(designs_df)} designs")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loading materials from {materials_path}...")
    materials = load_materials(str(materials_path))
    print(f"  Found {len(materials)} materials")

    print("Creating interactive visualization...")
    fig, n_pairs = create_interactive_plot(
        designs_df, values_df, materials, max_designs=args.max_designs
    )

    print(f"Saving to {output_path}...")
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    
    # Restyle interactive javascript click event handler for Coating Inspection Dashboard
    post_script = f"""
        var N_PAIRS = {n_pairs};
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (plotDiv) {{
            plotDiv.on('plotly_click', function(data) {{
                var pt = data.points[0];
                if (pt.curveNumber < N_PAIRS) {{
                    var designIdx = pt.customdata;
                    
                    var vis = [];
                    for (var i = 0; i < plotDiv.data.length; i++) {{
                        var trace = plotDiv.data[i];
                        if (i < N_PAIRS) {{
                            vis.push(true);
                        }} else if (trace.name === "highlight_" + designIdx || 
                                   trace.name === "coating_" + designIdx ||
                                   trace.name === "field_" + designIdx ||
                                   trace.name === "spectrum_" + designIdx) {{
                            vis.push(true);
                        }} else {{
                            vis.push(false);
                        }}
                    }}
                    Plotly.restyle(plotDiv, {{ 'visible': vis }});
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
    print("Click on any point in the Pareto front to view its coating design.")
    print("You can also use the dropdown menu on the right to select specific designs.")

    return 0


if __name__ == "__main__":
    exit(main())
