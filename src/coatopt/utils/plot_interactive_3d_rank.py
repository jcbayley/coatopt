#!/usr/bin/env python3
"""
Interactive 3D Rank Plot for Coating Optimization.

This script loads Pareto front designs and plots:
- X-axis: Absorption (ppm)
- Y-axis: Coating Thermal Noise / CTN (m/sqrt(Hz))
- Z-axis: Rank (sorted by reflectivity descending, Rank 1 = highest reflectivity)
- Color: Reflectivity (or reflectivity loss)

Saves as an interactive HTML file and opens it in the default web browser.
"""

import argparse
import configparser
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Set up local paths so it can find coatopt packages when run directly
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Try importing plotly
try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: plotly is required to run this script. Please install it using 'pip install plotly'.")
    sys.exit(1)

# Import local helpers
from coatopt.utils.utils import load_pareto_front


def parse_design(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract thicknesses and materials from a Pareto front row."""
    thickness_cols = [col for col in row.index if col.startswith("thickness_")]
    material_cols = [col for col in row.index if col.startswith("material_")]

    thickness_cols = sorted(thickness_cols, key=lambda x: int(x.split("_")[1]))
    material_cols = sorted(material_cols, key=lambda x: int(x.split("_")[1]))

    thicknesses = np.array([row[col] for col in thickness_cols])
    materials = np.array([int(row[col]) for col in material_cols])

    return thicknesses, materials


def calculate_physical_thickness(row: pd.Series, materials_dict: dict, lambda_nm: float = 1064.0) -> float:
    """Calculate the total physical thickness of a design in nm."""
    try:
        thicknesses, material_indices = parse_design(row)
        total_thick = 0.0
        for tOpt, mat_idx in zip(thicknesses, material_indices):
            if mat_idx == 0 or tOpt <= 1e-12:
                continue
            n = materials_dict.get(mat_idx, {}).get("n", 1.0)
            total_thick += (tOpt * lambda_nm) / n
        return total_thick
    except Exception:
        return 0.0


def create_3d_rank_plot(
    designs_df: pd.DataFrame,
    values_df: pd.DataFrame,
    title: str = "Coating Optimization: 3D Rank Plot",
    dark_mode: bool = True,
    color_by_loss: bool = False,
    compare_refl: Optional[float] = None,
    compare_abs: Optional[float] = None,
    compare_tn: Optional[float] = None,
    compare_label: str = "Reference Design",
    min_refl: Optional[float] = None,
    max_abs: Optional[float] = None,
    max_tn: Optional[float] = None,
    materials: Optional[dict] = None,
    rank_by_utility: bool = False,
    weight_refl: float = 0.10,
    weight_abs: float = 0.35,
    weight_tn: float = 0.45,
    weight_thick: float = 0.10,
    compare_thick: Optional[float] = None,
    target_refl: float = 0.9999,
    target_abs: float = 0.30,
    target_tn: float = 4.0e-21,
    target_thick: float = 6000.0,
    top_n: Optional[int] = None,
) -> go.Figure:
    """Create interactive 3D scatter plot of Absorption, TN, and Rank.

    Args:
        designs_df: DataFrame containing layer thicknesses/materials.
        values_df: DataFrame containing reflectivity, absorption, thermal_noise.
        title: Title of the plot.
        dark_mode: If True, uses dark mode theme.
        color_by_loss: If True, colors by reflectivity loss instead of raw reflectivity.
        compare_refl: Reflectivity of comparison reference design.
        compare_abs: Absorption (ppm) of reference design.
        compare_tn: Thermal noise (m/sqrt(Hz)) of reference design.
        compare_label: Label for the reference design.
        min_refl: Minimum reflectivity threshold to filter Pareto designs before ranking.
        max_abs: Maximum absorption threshold to filter Pareto designs before ranking.
        max_tn: Maximum thermal noise threshold to filter Pareto designs before ranking.
        materials: Loaded materials dict from JSON to compute physical thickness.
        rank_by_utility: Rank by utility score on the Z-axis instead of raw reflectivity.
        weight_refl: Weight for reflectivity in utility score (default 0.10).
        weight_abs: Weight for absorption in utility score (default 0.35).
        weight_tn: Weight for thermal noise in utility score (default 0.45).
        weight_thick: Weight for total thickness penalty in utility score (default 0.10).
        compare_thick: Physical thickness (nm) of comparison design.

    Returns:
        Plotly Figure.
    """
    combined_df = pd.concat([designs_df, values_df], axis=1)

    # Filter by minimum reflectivity if specified
    if min_refl is not None:
        combined_df = combined_df[combined_df["reflectivity"] >= min_refl].reset_index(drop=True)

    # Filter by maximum absorption if specified
    if max_abs is not None:
        combined_df = combined_df[combined_df["absorption"] <= max_abs].reset_index(drop=True)

    # Filter by maximum thermal noise if specified
    if max_tn is not None:
        combined_df = combined_df[combined_df["thermal_noise"] <= max_tn].reset_index(drop=True)

    # Calculate physical thicknesses if materials are loaded, otherwise fall back to sum of dOpt
    thickness_vals = []
    for _, row in combined_df.iterrows():
        if materials is not None:
            thick = calculate_physical_thickness(row, materials)
        else:
            try:
                dOpt, _ = parse_design(row)
                thick = float(np.sum(dOpt))
            except Exception:
                thick = 0.0
        thickness_vals.append(thick)
    combined_df["total_thickness"] = thickness_vals

    # Calculate target-based scores with 0.90 target baseline and exceeding bonus
    # Maximize (Reflectivity)
    refl_loss_scale = max(1e-6, 1.0 - target_refl)
    r_score = np.where(
        combined_df["reflectivity"] >= target_refl,
        0.9 + 0.1 * (combined_df["reflectivity"] - target_refl) / refl_loss_scale,
        0.9 * np.exp(-(target_refl - combined_df["reflectivity"]) / refl_loss_scale)
    )

    # Minimize (Absorption)
    abs_score = np.where(
        combined_df["absorption"] <= target_abs,
        0.9 + 0.1 * (target_abs - combined_df["absorption"]) / target_abs,
        0.9 * np.exp(-(combined_df["absorption"] - target_abs) / target_abs)
    )

    # Minimize (Thermal Noise)
    tn_score = np.where(
        combined_df["thermal_noise"] <= target_tn,
        0.9 + 0.1 * (target_tn - combined_df["thermal_noise"]) / target_tn,
        0.9 * np.exp(-(combined_df["thermal_noise"] - target_tn) / target_tn)
    )

    # Minimize (Thickness)
    thick_score = np.where(
        combined_df["total_thickness"] <= target_thick,
        0.9 + 0.1 * (target_thick - combined_df["total_thickness"]) / target_thick,
        0.9 * np.exp(-(combined_df["total_thickness"] - target_thick) / target_thick)
    )

    # Normalize weights so they sum to 1.0
    total_w = weight_refl + weight_abs + weight_tn + weight_thick
    w_refl = weight_refl / total_w if total_w > 0 else 0.10
    w_abs = weight_abs / total_w if total_w > 0 else 0.35
    w_tn = weight_tn / total_w if total_w > 0 else 0.45
    w_thick = weight_thick / total_w if total_w > 0 else 0.10

    combined_df["utility_score"] = (
        w_refl * r_score +
        w_abs * abs_score +
        w_tn * tn_score +
        w_thick * thick_score
    )

    # Determine sorting column based on rank_by_utility
    if rank_by_utility:
        sort_col = "utility_score"
        ascending = False
        title_suffix = "Ranked by Multi-Objective Utility Score"
    else:
        sort_col = "reflectivity" if "reflectivity" in combined_df.columns else combined_df.columns[0]
        ascending = False
        title_suffix = "Ranked by Reflectivity"

    # Sort descending
    combined_df = combined_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    combined_df["rank"] = combined_df.index + 1

    # Preserve full arrays for virtual ranking of reference design before slicing for display
    full_utility_vals = combined_df["utility_score"].values
    full_refl_vals = combined_df["reflectivity"].values
    total_designs = len(combined_df)

    if top_n is not None and top_n > 0:
        combined_df = combined_df.head(top_n)

    # Extract active layer counts
    active_layer_counts = []
    for _, row in combined_df.iterrows():
        try:
            dOpt, mat_idx = parse_design(row)
            active_mask = (mat_idx != 0) & (dOpt > 1e-12)
            active_layer_counts.append(int(np.sum(active_mask)))
        except Exception:
            active_layer_counts.append(0)
    combined_df["active_layer_count"] = active_layer_counts

    # Compute customdata for hovers
    customdata = np.stack(
        (
            combined_df["rank"].values,
            combined_df["reflectivity"].values,
            1.0 - combined_df["reflectivity"].values,
            combined_df["active_layer_count"].values,
            combined_df["total_thickness"].values,
            combined_df["utility_score"].values,
        ),
        axis=-1,
    )

    # X, Y, Z data
    x_data = combined_df["absorption"].values
    y_data = combined_df["thermal_noise"].values
    z_data = combined_df["rank"].values

    # Determine marker colorscale and values
    if color_by_loss:
        color_values = 1.0 - combined_df["reflectivity"].values
        colorbar_title = "Reflectivity Loss (1-R)"
        colorscale = "Magma" if dark_mode else "Reds"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
    else:
        color_values = combined_df["reflectivity"].values
        colorbar_title = "Reflectivity"
        colorscale = "Plasma" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))

    fig = go.Figure()

    # Add the primary 3D scatter trace
    fig.add_trace(
        go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode="markers",
            marker=dict(
                size=8,
                color=color_values,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right"),
                    thickness=18,
                    len=0.7,
                ),
                showscale=True,
                opacity=0.9,
                line=dict(width=0.5, color="black" if not dark_mode else "white"),
            ),
            customdata=customdata,
            name="Pareto Front Designs",
            showlegend=True,
            hovertemplate=(
                "<b>Design Rank #%{customdata[0]:d}</b><br><br>"
                "Reflectivity: %{customdata[1]:.6f}<br>"
                "Reflectivity Loss: %{customdata[2]:.3e}<br>"
                "Absorption: %{x:.4f} ppm<br>"
                "Thermal Noise: %{y:.4e} m/√Hz<br>"
                "Active Layers: %{customdata[3]:d}<br>"
                + ("Total Thickness: %{customdata[4]:.2f} nm<br>" if materials is not None else "Total dOpt: %{customdata[4]:.2f}<br>")
                + "Utility Score: %{customdata[5]:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add optional reference point for design comparison
    if compare_abs is not None and compare_tn is not None:
        if rank_by_utility:
            # Calculate utility score for the reference design using 0.90 target baseline with bonus
            # using the exact same weights and targets as other designs
            r_comp_val = compare_refl if compare_refl is not None else target_refl
            if r_comp_val >= target_refl:
                r_comp_score = 0.9 + 0.1 * (r_comp_val - target_refl) / refl_loss_scale
            else:
                r_comp_score = 0.9 * np.exp(-(target_refl - r_comp_val) / refl_loss_scale)

            if compare_abs <= target_abs:
                abs_comp_score = 0.9 + 0.1 * (target_abs - compare_abs) / target_abs
            else:
                abs_comp_score = 0.9 * np.exp(-(compare_abs - target_abs) / target_abs)

            if compare_tn <= target_tn:
                tn_comp_score = 0.9 + 0.1 * (target_tn - compare_tn) / target_tn
            else:
                tn_comp_score = 0.9 * np.exp(-(compare_tn - target_tn) / target_tn)
            
            if compare_thick is not None:
                if compare_thick <= target_thick:
                    thick_comp_score = 0.9 + 0.1 * (target_thick - compare_thick) / target_thick
                else:
                    thick_comp_score = 0.9 * np.exp(-(compare_thick - target_thick) / target_thick)
            else:
                # If thickness of comparison design is not specified, assume it meets target exactly (0.90)
                thick_comp_score = 0.90

            compare_utility = (
                w_refl * r_comp_score +
                w_abs * abs_comp_score +
                w_tn * tn_comp_score +
                w_thick * thick_comp_score
            )
            
            virtual_rank = float(np.searchsorted(-full_utility_vals, -compare_utility)) + 1.0
            if virtual_rank > total_designs:
                virtual_rank = float(total_designs + 0.5)
            rank_str = f"#{int(virtual_rank)}" if virtual_rank.is_integer() else f"#{virtual_rank:.1f}"
            legend_name = f"{compare_label} (Virtual Utility Rank: {rank_str} of {total_designs})"
            
            hover_comp_str = (
                f"<b>{compare_label} (Reference)</b><br><br>"
                + (f"Reflectivity: {compare_refl:.6f}<br>" if compare_refl is not None else "")
                + (f"Reflectivity Loss: {1.0 - compare_refl:.3e}<br>" if compare_refl is not None else "")
                + "Absorption: %{x:.4f} ppm<br>"
                "Thermal Noise: %{y:.4e} m/sqrt(Hz)<br>"
                + (f"Total Thickness: {compare_thick:.2f} nm<br>" if compare_thick is not None else "")
                + f"Virtual Utility Rank: {rank_str}<br>"
                f"Reference Utility: {compare_utility:.4f}<br>"
                "<extra></extra>"
            )
        else:
            # Determine virtual rank based on reflectivity
            if compare_refl is not None:
                if len(full_refl_vals) > 0:
                    virtual_rank = float(np.searchsorted(-full_refl_vals, -compare_refl)) + 1.0
                    if virtual_rank > total_designs:
                        virtual_rank = float(total_designs + 0.5)
                else:
                    virtual_rank = 1.0
            else:
                virtual_rank = 1.0
                
            rank_str = f"#{int(virtual_rank)}" if virtual_rank.is_integer() else f"#{virtual_rank:.1f}"
            legend_name = f"{compare_label} (Virtual Rank: {rank_str} of {total_designs})"
            hover_comp_str = (
                f"<b>{compare_label} (Reference)</b><br><br>"
                + (f"Reflectivity: {compare_refl:.6f}<br>" if compare_refl is not None else "")
                + (f"Reflectivity Loss: {1.0 - compare_refl:.3e}<br>" if compare_refl is not None else "")
                + "Absorption: %{x:.4f} ppm<br>"
                "Thermal Noise: %{y:.4e} m/sqrt(Hz)<br>"
                + (f"Total Thickness: {compare_thick:.2f} nm<br>" if compare_thick is not None else "")
                + f"Virtual Rank: {rank_str}<br>"
                "<extra></extra>"
            )

        print(f"Calculated virtual rank for reference design '{compare_label}': {rank_str} (out of {total_designs} designs)")

        fig.add_trace(
            go.Scatter3d(
                x=[compare_abs],
                y=[compare_tn],
                z=[virtual_rank],
                mode="markers",
                marker=dict(
                    size=14,
                    color="#ff007f",
                    symbol="diamond",
                    line=dict(width=1.5, color="black" if not dark_mode else "white"),
                ),
                name=legend_name,
                hovertemplate=hover_comp_str,
                showlegend=True,
            )
        )

    max_rank = float(combined_df["rank"].max()) if len(combined_df) > 0 else 100.0

    # Style layout
    template = "plotly_dark" if dark_mode else "plotly_white"
    grid_color = "rgba(100, 100, 100, 0.3)" if dark_mode else "rgba(200, 200, 200, 0.7)"
    bg_color = "#121212" if dark_mode else "#ffffff"

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>Interactive Pareto front - {title_suffix}</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(
                title="Absorption (ppm)",
                type="log",
                gridcolor=grid_color,
                showbackground=True,
                backgroundcolor=bg_color,
            ),
            yaxis=dict(
                title="Thermal Noise (m/√Hz)",
                type="log",
                gridcolor=grid_color,
                showbackground=True,
                backgroundcolor=bg_color,
            ),
            zaxis=dict(
                title="Design Rank (Utility)" if rank_by_utility else "Design Rank (Reflectivity)",
                range=[max_rank + 2.0, 0.5],  # Rank 1 at the top, worst rank at the bottom
                gridcolor=grid_color,
                showbackground=True,
                backgroundcolor=bg_color,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        height=750,
        autosize=True,
        template=template,
        margin=dict(l=0, r=0, b=50, t=80),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Run interactive 3D Rank Pareto front visualizer",
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing pareto_front.csv",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use light theme layout instead of default premium dark mode",
    )
    parser.add_argument(
        "--color-by-loss",
        action="store_true",
        help="Color map points by Reflectivity Loss (1-R) instead of raw Reflectivity",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated HTML file in the default web browser",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path for the generated HTML. Defaults to run_directory/pareto_3d_rank.html",
    )
    parser.add_argument(
        "--compare-refl",
        type=float,
        default=None,
        help="Reflectivity of custom reference design to plot as comparison point",
    )
    parser.add_argument(
        "--compare-abs",
        type=float,
        default=None,
        help="Absorption in ppm of custom reference design",
    )
    parser.add_argument(
        "--compare-tn",
        type=float,
        default=None,
        help="Thermal noise (CTN) of custom reference design",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default="Reference Design",
        help="Custom label for the comparison point",
    )
    parser.add_argument(
        "--min-refl",
        type=float,
        default=None,
        help="Minimum reflectivity threshold to filter Pareto designs before ranking",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help="Maximum absorption threshold (ppm) to filter Pareto designs before ranking",
    )
    parser.add_argument(
        "--max-tn",
        type=float,
        default=None,
        help="Maximum thermal noise (CTN) threshold to filter Pareto designs before ranking",
    )
    parser.add_argument(
        "--rank-by-utility",
        action="store_true",
        help="Rank designs on the Z-axis by multi-objective utility score instead of reflectivity",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only display the top N ranked designs on the plot (default: show all)",
    )
    parser.add_argument(
        "--weight-refl",
        type=float,
        default=0.10,
        help="Weight for reflectivity in utility score (default: 0.10)",
    )
    parser.add_argument(
        "--weight-abs",
        type=float,
        default=0.35,
        help="Weight for absorption in utility score (default: 0.35)",
    )
    parser.add_argument(
        "--weight-tn",
        type=float,
        default=0.45,
        help="Weight for thermal noise (CTN) in utility score (default: 0.45)",
    )
    parser.add_argument(
        "--weight-thick",
        type=float,
        default=0.10,
        help="Weight for physical thickness in utility score (default: 0.10)",
    )
    parser.add_argument(
        "--compare-thick",
        type=float,
        default=None,
        help="Physical thickness in nm of custom reference design",
    )
    parser.add_argument(
        "--target-refl",
        type=float,
        default=None,
        help="Target reflectivity for utility scoring (defaults to --compare-refl if set, else 0.9999)",
    )
    parser.add_argument(
        "--target-abs",
        type=float,
        default=None,
        help="Target absorption in ppm for utility scoring (defaults to --compare-abs if set, else 0.30)",
    )
    parser.add_argument(
        "--target-tn",
        type=float,
        default=None,
        help="Target thermal noise (CTN) for utility scoring (defaults to --compare-tn if set, else 4.0e-21)",
    )
    parser.add_argument(
        "--target-thick",
        type=float,
        default=None,
        help="Target physical thickness in nm for utility scoring (defaults to --compare-thick if set, else 6000.0)",
    )
    args = parser.parse_args()

    # Resolve target values, defaulting to comparison design values if they are provided,
    # and falling back to default values otherwise.
    target_refl = args.target_refl if args.target_refl is not None else (args.compare_refl if args.compare_refl is not None else 0.9999)
    target_abs = args.target_abs if args.target_abs is not None else (args.compare_abs if args.compare_abs is not None else 0.30)
    target_tn = args.target_tn if args.target_tn is not None else (args.compare_tn if args.compare_tn is not None else 4.0e-21)
    target_thick = args.target_thick if args.target_thick is not None else (args.compare_thick if args.compare_thick is not None else 6000.0)

    # Convert to Path object
    directory = Path(args.directory)
    if not directory.is_absolute():
        directory = Path(os.getcwd()) / directory
    directory = directory.resolve()

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return 1

    print(f"Loading Pareto front from {directory}...")
    try:
        designs_df, values_df, _ = load_pareto_front(directory)
        print(f"  Loaded {len(designs_df)} designs successfully.")
    except Exception as e:
        print(f"Error: Failed to load Pareto front from {directory}: {e}")
        return 1

    # Load materials from config.ini if possible
    materials = None
    config_path = directory / "config.ini"
    if config_path.exists():
        config = configparser.ConfigParser()
        config.read(config_path)
        try:
            materials_path_str = config.get("general", "materials_path")
            materials_path = Path(materials_path_str)
            if not materials_path.is_absolute():
                candidate1 = (config_path.parent / materials_path).resolve()
                candidate2 = (config_path.parent.parent / materials_path).resolve()
                if candidate1.exists():
                    materials_path = candidate1
                elif candidate2.exists():
                    materials_path = candidate2
            else:
                # If absolute path does not exist, search local candidates based on filename
                if not materials_path.exists():
                    filename = materials_path.name
                    project_root = Path(__file__).parent.parent.parent.parent
                    local_candidate1 = (project_root / "experiments" / filename).resolve()
                    local_candidate2 = (config_path.parent / filename).resolve()
                    local_candidate3 = (config_path.parent.parent / "experiments" / filename).resolve()
                    if local_candidate1.exists():
                        materials_path = local_candidate1
                    elif local_candidate2.exists():
                        materials_path = local_candidate2
                    elif local_candidate3.exists():
                        materials_path = local_candidate3
                
            if Path(materials_path).exists():
                from coatopt.utils.utils import load_materials
                materials = load_materials(str(materials_path))
                print(f"  Loaded materials library from: {materials_path}")
            else:
                print(f"  Warning: Materials file not found at: {materials_path_str}")
        except Exception as e:
            print(f"  Warning: Could not resolve materials library path: {e}")

    title = f"Pareto Front 3D Rank Plot: {directory.name}"
    fig = create_3d_rank_plot(
        designs_df=designs_df,
        values_df=values_df,
        title=title,
        dark_mode=not args.light,
        color_by_loss=args.color_by_loss,
        compare_refl=args.compare_refl,
        compare_abs=args.compare_abs,
        compare_tn=args.compare_tn,
        compare_label=args.compare_label,
        min_refl=args.min_refl,
        max_abs=args.max_abs,
        max_tn=args.max_tn,
        materials=materials,
        rank_by_utility=args.rank_by_utility,
        weight_refl=args.weight_refl,
        weight_abs=args.weight_abs,
        weight_tn=args.weight_tn,
        weight_thick=args.weight_thick,
        compare_thick=args.compare_thick,
        target_refl=target_refl,
        target_abs=target_abs,
        target_tn=target_tn,
        target_thick=target_thick,
        top_n=args.top,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = directory / "pareto_3d_rank.html"

    print(f"Saving interactive visualization to {output_path}...")
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
    }
    fig.write_html(
        str(output_path),
        config=plotly_config,
        include_plotlyjs="cdn",
    )

    print(f"✓ Visualization successfully saved to {output_path}")

    if not args.no_open:
        print(f"Opening plot in browser: file://{output_path}")
        webbrowser.open(f"file://{output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
