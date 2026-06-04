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

# Global placeholders for lazy-loaded physics dependencies
getCoatingThermalNoise = None
optical_to_physical = None
CalculateEFI_tmm = None
CalculateTransmission_tmm = None
thin_film_stack = None

def load_physics_dependencies() -> bool:
    """Dynamically load physics and TMM libraries from CoatingAnalysis."""
    global getCoatingThermalNoise, optical_to_physical, CalculateEFI_tmm, CalculateTransmission_tmm, thin_film_stack
    if getCoatingThermalNoise is not None:
        return True
    try:
        lib_path = "/Users/simon/Library/CloudStorage/GoogleDrive-simon.tait@ligo.org/My Drive/BackupFromDropbox/Python/CoatingAnalysis/src"
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
            
        from coating_analysis.YAM_CoatingBrownian import getCoatingThermalNoise as temp_gctn
        from coating_analysis.EFI_tmm import (
            optical_to_physical as temp_otp,
            CalculateEFI_tmm as temp_cefi,
            CalculateTransmission_tmm as temp_ctrans
        )
        from coating_analysis.Coatings_development import thin_film_stack as temp_tfs
        
        getCoatingThermalNoise = temp_gctn
        optical_to_physical = temp_otp
        CalculateEFI_tmm = temp_cefi
        CalculateTransmission_tmm = temp_ctrans
        thin_film_stack = temp_tfs
        return True
    except Exception as e:
        print(f"Warning: Could not load physical coating solvers: {e}")
        return False


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


def precompute_tmm_details(combined_df: pd.DataFrame, materials_dict: dict, max_count: int = 50, lambda_nm: float = 1064.0) -> dict:
    """Precompute EFI profile and spectral transmission response for the top N designs."""
    load_physics_dependencies()
    
    tmm_data = {}
    total_designs = len(combined_df)
    
    for idx, row in combined_df.iterrows():
        design_idx = int(idx)
        rank = int(row["rank"])
        
        if (design_idx + 1) % 50 == 0 or design_idx == 0 or design_idx == total_designs - 1:
            print(f"  Precomputing design {design_idx + 1}/{total_designs} (Rank {rank})...")
            
        # Build layer variables
        dOpt, material_indices = parse_design(row)
        active_mask = (material_indices != 0) & (dOpt > 1e-12)
        active_dOpt = dOpt[active_mask]
        active_materialLayer = material_indices[active_mask]
        
        # Reverse layers so they are in air-to-substrate order for the physical solvers
        active_dOpt = active_dOpt[::-1]
        active_materialLayer = active_materialLayer[::-1]
        
        mapped_layer = np.array([999 if m == 0 else m for m in active_materialLayer])
        
        # Build materialParams structure
        materialParams = {}
        for k, v in materials_dict.items():
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
            
        n_input = np.array([materialParams[m]['n'] for m in mapped_layer])
        
        # Physical thickness calculation
        d_physical_nm = []
        if thin_film_stack is not None:
            try:
                _, _, d_physical_m = thin_film_stack(
                    dOpt=active_dOpt,
                    n_input=n_input,
                    materialLayer=mapped_layer,
                    materialParams=materialParams,
                    lambda_=lambda_nm,
                    plots=False,
                    verbose=False
                )
                d_physical_nm = list(d_physical_m * 1e9)
            except Exception:
                pass
                
        if len(d_physical_nm) == 0:
            for i in range(len(active_dOpt)):
                mat_idx = mapped_layer[i]
                n_layer = materialParams.get(mat_idx, {}).get("n", 1.45)
                d_physical_nm.append(float(active_dOpt[i] * lambda_nm / n_layer))
                
        # Generate diagnostic text (verbose printout format)
        info_lines = []
        info_lines.append(f"  SELECTED DESIGN SUMMARY")
        info_lines.append(f"  -------------------------")
        info_lines.append(f"  Design Rank: #{rank} / {total_designs}")
        info_lines.append(f"  Reflectivity: {row['reflectivity']:.6f}")
        info_lines.append(f"  Loss (1 - R): {1.0 - row['reflectivity']:.4e}")
        info_lines.append(f"  Absorption: {row['absorption']:.3f} ppm")
        info_lines.append(f"  Thermal Noise: {row['thermal_noise']:.4e} m/sqrt(Hz)")
        if "utility_score" in row:
            info_lines.append(f"  Utility Score: {row['utility_score']:.4f}")
        info_lines.append(f"  Active Layers: {int(row['active_layer_count'])}")
        info_lines.append(f"  Total Physical Thickness: {sum(d_physical_nm):.2f} nm")
        info_text = "\\n".join(info_lines)
        
        # Structure design's base properties
        design_data = {
            "rank": rank,
            "reflectivity": float(row["reflectivity"]),
            "absorption": float(row["absorption"]),
            "thermal_noise": float(row["thermal_noise"]),
            "utility_score": float(row.get("utility_score", 0.0)),
            "active_layer_count": int(row["active_layer_count"]),
            "total_thickness": float(row["total_thickness"]),
            "dOpt": [float(x) for x in active_dOpt],
            "materialLayer": [int(x) for x in mapped_layer],
            "d_physical_nm": [float(x) for x in d_physical_nm],
            "material_names": [materialParams.get(int(m), {}).get("name", f"Material {m}") for m in mapped_layer],
            "material_indices": [int(m) for m in mapped_layer],
            "info_text": info_text,
            "precomputed": False
        }
        
        # Precompute TMM details only for the top N designs
        if design_idx < max_count:
            design_data["precomputed"] = True
            
            # Calculate EFI
            if CalculateEFI_tmm is not None:
                try:
                    _, _, ds, E, _, _, _ = CalculateEFI_tmm(
                        dOpt=active_dOpt,
                        materialLayer=mapped_layer,
                        materialParams=materialParams,
                        lambda_=lambda_nm,
                        plots=False,
                    )
                    design_data["efi_depths"] = [float(x) for x in ds]
                    design_data["efi_intensity"] = [float(x) for x in E]
                except Exception as e:
                    print(f"Warning: Could not precompute EFI for design {rank}: {e}")
                    
            # Calculate Transmission Spectrum
            if CalculateTransmission_tmm is not None:
                try:
                    lambda_list = np.linspace(400.0, 1400.0, 200)
                    wavelengths, transmission, _ = CalculateTransmission_tmm(
                        dOpt=active_dOpt,
                        materialLayer=mapped_layer,
                        materialParams=materialParams,
                        lambda_list=lambda_list,
                        lambda_0=lambda_nm,
                        plots=False,
                    )
                    design_data["spec_wavelengths"] = [float(x) for x in wavelengths]
                    design_data["spec_transmission"] = [float(x * 100) for x in transmission]
                except Exception as e:
                    print(f"Warning: Could not precompute spectrum for design {rank}: {e}")
                    
        tmm_data[design_idx] = design_data
        
    return tmm_data


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
) -> Tuple[go.Figure, pd.DataFrame]:
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

    # Compute customdata for hovers (including 0-based design index for HTML interactivity)
    customdata = np.stack(
        (
            combined_df["rank"].values,
            combined_df["reflectivity"].values,
            1.0 - combined_df["reflectivity"].values,
            combined_df["active_layer_count"].values,
            combined_df["total_thickness"].values,
            combined_df["utility_score"].values,
            combined_df.index.values,
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
                    title=dict(text=colorbar_title, side="right", font=dict(color="#e0e0e0" if dark_mode else "#333333")),
                    tickfont=dict(color="#e0e0e0" if dark_mode else "#333333"),
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

    # Always add a trace for the reference point for design comparison (initialized empty if not set)
    has_ref = (compare_abs is not None and compare_tn is not None)
    ref_x = [compare_abs] if has_ref else []
    ref_y = [compare_tn] if has_ref else []
    ref_z = []
    legend_name = compare_label
    hover_comp_str = ""

    if has_ref:
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
            ref_z = [virtual_rank]
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
                
            ref_z = [virtual_rank]
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
            x=ref_x,
            y=ref_y,
            z=ref_z,
            mode="markers",
            marker=dict(
                size=14,
                color="#ff007f",
                symbol="diamond",
                line=dict(width=1.5, color="black" if not dark_mode else "white"),
            ),
            name=legend_name,
            hovertemplate=hover_comp_str,
            showlegend=has_ref,
            visible=True if has_ref else "legendonly",
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
            font=dict(size=16, color="#e0e0e0" if dark_mode else "#333333"),
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="Absorption (ppm)", font=dict(color="#e0e0e0" if dark_mode else "#333333")),
                tickfont=dict(color="#e0e0e0" if dark_mode else "#333333"),
                type="log",
                gridcolor=grid_color,
                showbackground=True,
                backgroundcolor=bg_color,
            ),
            yaxis=dict(
                title=dict(text="Thermal Noise (m/√Hz)", font=dict(color="#e0e0e0" if dark_mode else "#333333")),
                tickfont=dict(color="#e0e0e0" if dark_mode else "#333333"),
                type="log",
                gridcolor=grid_color,
                showbackground=True,
                backgroundcolor=bg_color,
            ),
            zaxis=dict(
                title=dict(text="Design Rank (Utility)" if rank_by_utility else "Design Rank (Reflectivity)", font=dict(color="#e0e0e0" if dark_mode else "#333333")),
                tickfont=dict(color="#e0e0e0" if dark_mode else "#333333"),
                range=[max_rank + 2.0, 0.5],  # Rank 1 at the top, worst rank at the bottom
                gridcolor=grid_color,
                showbackground=True,
                backgroundcolor=bg_color,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        legend=dict(
            font=dict(color="#e0e0e0" if dark_mode else "#333333"),
        ),
        autosize=True,
        template=template,
        margin=dict(l=0, r=0, b=50, t=30),
    )

    return fig, combined_df


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
    parser.add_argument(
        "--precompute-tmm-count",
        type=int,
        default=-1,
        help="Number of top designs to precompute full TMM details (EFI and spectrum) for (default: -1, meaning all)",
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
    fig, combined_df = create_3d_rank_plot(
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
        top_n=None,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = directory / "pareto_3d_rank.html"

    # Precompute TMM details for top designs
    precompute_count = args.precompute_tmm_count
    if precompute_count is None or precompute_count < 0:
        precompute_count = len(combined_df)
    
    print(f"Precomputing TMM physics data for {precompute_count} of {len(combined_df)} designs...")
    tmm_data = precompute_tmm_details(
        combined_df=combined_df,
        materials_dict=materials if materials is not None else {},
        max_count=precompute_count,
        lambda_nm=1064.0
    )
    tmm_data_json = json.dumps(tmm_data)

    # Build materials mappings dict for client-side exporter
    materials_params_dict = {}
    if materials is not None:
        for k, v in materials.items():
            mat_key = int(k)
            mat_data = v.copy()
            if mat_data.get("n") is None:
                mat_data["n"] = 1.0
            if mat_data.get("k") is None:
                mat_data["k"] = 0.0
            if mat_key == 0:
                materials_params_dict[999] = mat_data
                materials_params_dict[0] = mat_data
            else:
                materials_params_dict[mat_key] = mat_data
    if 999 not in materials_params_dict:
        materials_params_dict[999] = {'name': 'air', 'n': 1.0, 'k': 0.0}
        materials_params_dict[0] = {'name': 'air', 'n': 1.0, 'k': 0.0}
    if 1 not in materials_params_dict:
        materials_params_dict[1] = {'name': 'SiO2', 'n': 1.45, 'k': 0.0}
    if 2 not in materials_params_dict:
        materials_params_dict[2] = {'name': 'TiGermania', 'n': 2.1, 'k': 0.0}
        
    materials_params_json = json.dumps(materials_params_dict)

    # Compile the HTML page using replacements
    import plotly.utils
    plotly_data_json = json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)
    plotly_layout_json = json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)
    compare_refl_val = args.compare_refl if args.compare_refl is not None else 0.9999
    compare_abs_val = args.compare_abs if args.compare_abs is not None else 0.3
    compare_tn_val = args.compare_tn if args.compare_tn is not None else 4e-21
    compare_thick_val = args.compare_thick if args.compare_thick is not None else 0.0
    compare_label_str = args.compare_label if args.compare_label is not None else "Reference Design"

    # HTML dynamic template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>__TITLE__</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .header {
            text-align: center;
            padding: 15px 10px 5px 10px;
            background-color: #1a1a1a;
            border-bottom: 1px solid #2d2d2d;
        }
        .header h1 {
            margin: 0 0 5px 0;
            font-size: 20px;
            color: #00bcd4;
        }
        .header p {
            margin: 0;
            font-size: 13px;
            color: #888;
        }
        .container {
            display: flex;
            height: calc(100vh - 65px);
            box-sizing: border-box;
            padding: 15px;
            gap: 15px;
        }
        .left-col {
            width: 58%;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #2d2d2d;
            box-sizing: border-box;
            background-color: #121212;
            height: 100%;
            overflow: hidden;
        }
        .plot-container-3d {
            flex-grow: 1;
            width: 100%;
            min-height: 350px;
        }
        .controls-toolbar {
            display: flex;
            gap: 10px;
            padding: 10px 15px;
            background-color: #1e1e1e;
            border: 1px solid #2d2d2d;
            border-radius: 6px;
            align-items: center;
        }
        .btn {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #444;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: background-color 0.2s, border-color 0.2s;
        }
        .btn:hover {
            background-color: #3d3d3d;
            border-color: #666;
        }
        .btn:active {
            background-color: #1f1f1f;
        }
        .btn-primary {
            background-color: #005a70;
            border-color: #00bcd4;
            color: #ffffff;
        }
        .btn-primary:hover {
            background-color: #007c99;
            border-color: #00e5ff;
        }
        .btn:disabled {
            background-color: #1f1f1f;
            border-color: #2d2d2d;
            color: #555;
            cursor: not-allowed;
        }
        .right-col {
            width: 42%;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            box-sizing: border-box;
            background-color: #161616;
            padding: 15px;
            gap: 15px;
        }
        .card {
            background-color: #1e1e1e;
            border: 1px solid #2d2d2d;
            border-radius: 6px;
            padding: 12px;
            box-sizing: border-box;
        }
        .card-title {
            font-size: 13px;
            font-weight: 600;
            color: #00bcd4;
            margin: 0 0 10px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid #2d2d2d;
            padding-bottom: 5px;
        }
        .info-card {
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 11px;
            background-color: #0d0d0d;
            color: #a5d6a7;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #222;
        }
        .plot-2d {
            height: 180px;
            width: 100%;
        }
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-top: 5px;
            flex-wrap: wrap;
        }
        .error-message {
            color: #ff5252;
            text-align: center;
            padding: 20px;
            font-size: 12px;
            font-style: italic;
        }
        .targets-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px 12px;
            margin-bottom: 12px;
            font-size: 11px;
        }
        .targets-grid label {
            display: block;
            color: #888;
            margin-bottom: 3px;
            font-weight: 500;
        }
        .targets-grid input {
            width: 100%;
            background: #121212;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 11px;
            box-sizing: border-box;
            transition: border-color 0.2s;
        }
        .targets-grid input:focus {
            border-color: #00bcd4;
            outline: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>__TITLE__</h1>
        <p>Interactive Pareto Front & Diagnostics Dashboard — Left click any point in 3D to inspect</p>
    </div>
    <div class="container">
        <div class="left-col">
            <div id="plot-3d" class="plot-container-3d"></div>
            
            <div class="card">
                <div class="card-title">Selected Design Information</div>
                <div style="display: flex; gap: 15px; align-items: stretch;">
                    <div id="info-content" class="info-card" style="flex-grow: 1; margin: 0; min-height: 80px;">Click a point in the 3D plot to inspect design details.</div>
                    <div class="action-buttons" style="flex-direction: column; justify-content: center; gap: 8px; margin: 0; min-width: 200px;">
                        <button class="btn btn-primary" id="btn-export-py" disabled style="width: 100%;">Export Python Design Script</button>
                        <button class="btn btn-primary" id="btn-export-csv" disabled style="width: 100%;">Export CSV Layers</button>
                        <button class="btn" id="btn-set-baseline" disabled style="width: 100%; background-color: #4e342e; border-color: #5d4037; color: #ffab91;">[+] Set as Baseline Target</button>
                        <button class="btn" id="btn-set-comparison-stack" disabled style="width: 100%; background-color: #1a237e; border-color: #283593; color: #c5cae9;">[+] Set as Comparison Stack</button>
                        <button class="btn" id="btn-clear-comparison-stack" style="width: 100%; display: none; background-color: #37474f; border-color: #455a64; color: #cfd8dc;">Clear Comparison Stack</button>
                    </div>
                </div>
            </div>

            <div class="controls-toolbar">
                <span style="font-size: 11px; color: #888; font-weight: bold; margin-right: 5px; text-transform: uppercase; letter-spacing: 0.5px;">3D VIEW OPTIONS:</span>
                <button class="btn" id="btn-reverse-z">Invert Z-Axis View</button>
                <button class="btn" id="btn-toggle-x-scale">Toggle X-Scale (Log/Linear)</button>
                <button class="btn" id="btn-toggle-y-scale">Toggle Y-Scale (Log/Linear)</button>
                
                <span style="font-size: 11px; color: #888; font-weight: bold; margin-left: 15px; margin-right: 5px; text-transform: uppercase; letter-spacing: 0.5px;">SHOW TOP:</span>
                <input type="text" id="input-top-x" placeholder="All" style="width: 50px; background: #2b2b2b; border: 1px solid #444; color: #e0e0e0; padding: 5px 8px; border-radius: 4px; font-size: 12px; text-align: center; box-sizing: border-box;">
                <button class="btn" id="btn-apply-top">Apply</button>
            </div>
        </div>
        <div class="right-col">
            <div class="card">
                <div class="card-title">Comparison Target Benchmarks</div>
                <div class="targets-grid">
                    <div>
                        <label for="input-target-refl">Reflectivity Target (R)</label>
                        <input type="number" id="input-target-refl" step="any">
                    </div>
                    <div>
                        <label for="input-target-abs">Absorption Target (ppm)</label>
                        <input type="number" id="input-target-abs" step="any">
                    </div>
                    <div>
                        <label for="input-target-tn">Thermal Noise Target (m/√Hz)</label>
                        <input type="text" id="input-target-tn">
                    </div>
                    <div>
                        <label for="input-target-thick">Thickness Target (nm)</label>
                        <input type="number" id="input-target-thick" step="any">
                    </div>
                </div>
                <button class="btn btn-primary" id="btn-apply-targets" style="width: 100%;">Apply Comparison Targets</button>
            </div>

            <div class="card">
                <div class="card-title">Custom 3D Plot Comparison Point</div>
                <div style="margin-bottom: 8px;">
                    <label for="input-comp-label" style="display: block; color: #888; margin-bottom: 3px; font-weight: 500; font-size: 11px;">Point Label</label>
                    <input type="text" id="input-comp-label" placeholder="Reference Design" style="width: 100%; background: #121212; border: 1px solid #444; color: #e0e0e0; padding: 5px 8px; border-radius: 4px; font-size: 11px; box-sizing: border-box;">
                </div>
                <div class="targets-grid">
                    <div>
                        <label for="input-comp-refl">Reflectivity (R)</label>
                        <input type="number" id="input-comp-refl" step="any" placeholder="e.g. 0.9999">
                    </div>
                    <div>
                        <label for="input-comp-abs">Absorption (ppm)</label>
                        <input type="number" id="input-comp-abs" step="any" placeholder="e.g. 0.5">
                    </div>
                    <div>
                        <label for="input-comp-tn">Thermal Noise (m/√Hz)</label>
                        <input type="text" id="input-comp-tn" placeholder="e.g. 4.0e-21">
                    </div>
                    <div>
                        <label for="input-comp-thick">Thickness (nm)</label>
                        <input type="number" id="input-comp-thick" step="any" placeholder="e.g. 6000">
                    </div>
                </div>
                <div style="display: flex; gap: 8px; margin-top: 8px;">
                    <button class="btn btn-primary" id="btn-apply-comp-point" style="flex-grow: 1; padding: 5px 10px; font-size: 11px;">Apply Point</button>
                    <button class="btn" id="btn-set-selected-comp-point" disabled style="flex-grow: 1; padding: 5px 10px; font-size: 11px; background-color: #3e2723; border-color: #4e342e; color: #d7ccc8;">[+] Set Selected</button>
                    <button class="btn" id="btn-clear-comp-point" style="flex-grow: 1; padding: 5px 10px; font-size: 11px; background-color: #37474f; border-color: #455a64; color: #cfd8dc;">Clear Point</button>
                </div>
            </div>


            <div class="card">
                <div class="card-title">Coating Stack Diagram</div>
                <div id="plot-stack" class="plot-2d"></div>
            </div>
            <div class="card">
                <div class="card-title">Electric Field Intensity Profile</div>
                <div id="plot-field" class="plot-2d"></div>
            </div>
            <div class="card">
                <div class="card-title">Simulated Spectral Response</div>
                <div id="plot-spectrum" class="plot-2d"></div>
            </div>
        </div>
    </div>

    <script>
        // Embedded Data
        var data3d = __PLOTLY_DATA_3D__;
        var layout3d = __PLOTLY_LAYOUT_3D__;
        var tmmData = __TMM_DATA__;
        var materialsParamsDict = __MATERIALS_PARAMS__;
        
        // Reference design details
        var hasReference = __HAS_REFERENCE__;
        var referenceLabel = "__REFERENCE_LABEL__";
        var compareRefl = __COMPARE_REFL__;
        var compareAbs = __COMPARE_ABS__;
        var compareTN = __COMPARE_TN__;
        var compareThick = __COMPARE_THICK__;

        var weightRefl = __WEIGHT_REFL__;
        var weightAbs = __WEIGHT_ABS__;
        var weightTN = __WEIGHT_TN__;
        var weightThick = __WEIGHT_THICK__;

        // Convert tmmData to designs array for easier sorting
        var designsList = [];
        for (var key in tmmData) {
            var d = tmmData[key];
            d.originalIdx = parseInt(key);
            designsList.push(d);
        }

        // Initialize target fields
        document.getElementById('input-target-refl').value = __TARGET_REFL__;
        document.getElementById('input-target-abs').value = __TARGET_ABS__;
        document.getElementById('input-target-tn').value = __TARGET_TN__.toExponential(4);
        document.getElementById('input-target-thick').value = __TARGET_THICK__;

        // Initialize custom comparison point fields
        if (hasReference) {
            document.getElementById('input-comp-label').value = referenceLabel;
            document.getElementById('input-comp-refl').value = compareRefl !== null ? compareRefl : "";
            document.getElementById('input-comp-abs').value = compareAbs !== null ? compareAbs : "";
            document.getElementById('input-comp-tn').value = compareTN !== null ? compareTN.toExponential(4) : "";
            document.getElementById('input-comp-thick').value = (compareThick !== null && compareThick > 0) ? compareThick : "";
        } else {
            document.getElementById('input-comp-label').value = "Reference Design";
            document.getElementById('input-comp-refl').value = "";
            document.getElementById('input-comp-abs').value = "";
            document.getElementById('input-comp-tn').value = "";
            document.getElementById('input-comp-thick').value = "";
        }


        // Initialize top-X field
        var initialTopX = __INITIAL_TOP_X__;
        if (initialTopX !== null) {
            document.getElementById('input-top-x').value = initialTopX;
        } else {
            document.getElementById('input-top-x').value = "";
        }

        // Update layout colors globally & remove chart title
        layout3d.title = undefined;
        layout3d.height = undefined;
        layout3d.paper_bgcolor = '#121212';
        layout3d.plot_bgcolor = '#121212';
        layout3d.scene.xaxis.color = '#e0e0e0';
        layout3d.scene.yaxis.color = '#e0e0e0';
        layout3d.scene.zaxis.color = '#e0e0e0';
        layout3d.scene.xaxis.gridcolor = '#2d2d2d';
        layout3d.scene.yaxis.gridcolor = '#2d2d2d';
        layout3d.scene.zaxis.gridcolor = '#2d2d2d';

        // Update colorbar title
        if(data3d[0].marker && data3d[0].marker.colorbar) {
            data3d[0].marker.colorbar.title = { text: 'Reflectivity', font: { color: '#e0e0e0' } };
            data3d[0].marker.colorbar.tickfont = { color: '#e0e0e0' };
        }

        // Initial Plotly setup
        Plotly.newPlot('plot-3d', data3d, layout3d, {responsive: true, displaylogo: false});

        // Run initial recalculation and filtering to sync UI state
        recalculateUtilityAndRerank();

        var selectedDesignIdx = null;
        var comparisonDesignIdx = null;

        function showPlotMessage(divId, message) {
            var div = document.getElementById(divId);
            div.innerHTML = '<div style="display: flex; height: 100%; justify-content: center; align-items: center;" class="error-message">' + message + '</div>';
        }

        function getStackTraces(design, xaxis, yaxis, legendShown, materialColors) {
            var d_phys = design.d_physical_nm;
            var material_names = design.material_names;
            var traces = [];
            
            var depth_so_far = 0.0;
            for (var i = 0; i < d_phys.length; i++) {
                var matName = material_names[i];
                var thick = d_phys[i];
                
                var showLegend = false;
                if (!legendShown[matName]) {
                    showLegend = true;
                    legendShown[matName] = true;
                }
                
                traces.push({
                    x: [depth_so_far + thick / 2.0],
                    y: [thick],
                    width: [thick],
                    xaxis: xaxis,
                    yaxis: yaxis,
                    name: matName,
                    type: 'bar',
                    marker: {
                        color: materialColors[matName] || '#555555',
                        line: { width: 0.5, color: '#000000' }
                    },
                    showlegend: showLegend,
                    legendgroup: matName,
                    hovertemplate: "Layer " + (i+1) + ": " + matName + "<br>Thickness: " + thick.toFixed(2) + " nm<extra></extra>"
                });
                
                depth_so_far += thick;
            }
            
            var subWidth = 150.0;
            var showSubLegend = false;
            if (!legendShown["Substrate"]) {
                showSubLegend = true;
                legendShown["Substrate"] = true;
            }
            traces.push({
                x: [depth_so_far + subWidth / 2.0],
                y: [120.0],
                width: [subWidth],
                xaxis: xaxis,
                yaxis: yaxis,
                name: "Substrate",
                type: 'bar',
                marker: {
                    color: '#7f7f7f',
                    line: { width: 0.5, color: '#000000' }
                },
                showlegend: showSubLegend,
                legendgroup: "Substrate",
                hovertemplate: "Substrate<br>Thickness: 150 nm<extra></extra>"
            });
            
            return traces;
        }

        function drawStackPlot(design) {
            var legendShown = {};
            var materialColors = {
                "air": "#333333",
                "SiO2": "#1f77b4",
                "TiTa": "#c837ab",
                "Ti:Ta2O5": "#c837ab",
                "TiGermania": "#e377c2",
                "Substrate": "#7f7f7f"
            };
            
            var allMats = [];
            if (design && design.material_names) {
                allMats = allMats.concat(design.material_names);
            }
            var hasComp = (comparisonDesignIdx !== null && comparisonDesignIdx !== -1);
            var compDesign = hasComp ? tmmData[comparisonDesignIdx] : null;
            if (compDesign && compDesign.material_names) {
                allMats = allMats.concat(compDesign.material_names);
            }
            var uniqueMaterials = [...new Set(allMats)];
            var palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#bcbd22", "#17becf"];
            uniqueMaterials.forEach(function(mat, i) {
                if (!materialColors[mat]) {
                    materialColors[mat] = palette[i % palette.length];
                }
            });
            
            var traces = [];
            if (compDesign) {
                var compTraces = getStackTraces(compDesign, 'x2', 'y2', legendShown, materialColors);
                traces = traces.concat(compTraces);
            }
            
            var selTraces = getStackTraces(design, 'x', 'y', legendShown, materialColors);
            traces = traces.concat(selTraces);
            
            var layout = {
                paper_bgcolor: '#1e1e1e',
                plot_bgcolor: '#1e1e1e',
                margin: { l: 45, r: 20, t: 15, b: 35 },
                height: 180,
                legend: {
                    font: { size: 9, color: '#e0e0e0' },
                    orientation: 'h',
                    y: -0.4
                },
                hovermode: 'closest'
            };
            
            if (compDesign) {
                layout.yaxis = {
                    domain: [0.55, 1.0],
                    title: { text: "Sel [nm]", font: { size: 8, color: '#e0e0e0' } },
                    tickfont: { size: 8, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    range: [0, Math.max(...design.d_physical_nm) * 1.15]
                };
                layout.yaxis2 = {
                    domain: [0.0, 0.45],
                    title: { text: "Comp [nm]", font: { size: 8, color: '#e0e0e0' } },
                    tickfont: { size: 8, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    range: [0, Math.max(...compDesign.d_physical_nm) * 1.15]
                };
                layout.xaxis = {
                    anchor: 'y',
                    tickfont: { size: 8, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    showticklabels: false
                };
                layout.xaxis2 = {
                    anchor: 'y2',
                    title: { text: "Coating Depth Position [nm]", font: { size: 9, color: '#e0e0e0' } },
                    tickfont: { size: 8, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true
                };
            } else {
                layout.yaxis = {
                    title: { text: "Physical Thickness [nm]", font: { size: 10, color: '#e0e0e0' } },
                    tickfont: { size: 9, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    range: [0, Math.max(...design.d_physical_nm) * 1.15]
                };
                layout.xaxis = {
                    title: { text: "Coating Depth Position [nm]", font: { size: 10, color: '#e0e0e0' } },
                    tickfont: { size: 9, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true
                };
            }
            
            Plotly.newPlot('plot-stack', traces, layout, {responsive: true, displayModeBar: false});
        }

        function drawEFIPlot(design) {
            if (!design.precomputed || !design.efi_depths) {
                showPlotMessage('plot-field', 'EFI solver details not precomputed for this design');
                return;
            }
            
            var traces = [];
            
            // Selected design trace
            traces.push({
                x: design.efi_depths,
                y: design.efi_intensity,
                mode: 'lines',
                line: { color: '#00bcd4', width: 2 },
                name: 'Selected (Rank ' + design.rank + ')',
                hovertemplate: "Selected (Rank " + design.rank + ")<br>Depth: %{x:.1f} nm<br>EFI: %{y:.3f}<extra></extra>"
            });
            
            var hasComp = (comparisonDesignIdx !== null && comparisonDesignIdx !== -1);
            var compDesign = hasComp ? tmmData[comparisonDesignIdx] : null;
            
            if (compDesign && compDesign.efi_depths) {
                traces.push({
                    x: compDesign.efi_depths,
                    y: compDesign.efi_intensity,
                    mode: 'lines',
                    line: { color: '#ff4081', width: 1.5, dash: 'dash' },
                    name: 'Comparison (Rank ' + compDesign.rank + ')',
                    hovertemplate: "Comparison (Rank " + compDesign.rank + ")<br>Depth: %{x:.1f} nm<br>EFI: %{y:.3f}<extra></extra>"
                });
            }
            
            var layout = {
                paper_bgcolor: '#1e1e1e',
                plot_bgcolor: '#1e1e1e',
                margin: { l: 45, r: 20, t: 15, b: 35 },
                height: 180,
                xaxis: {
                    title: { text: "Depth (nm)", font: { size: 10, color: '#e0e0e0' } },
                    tickfont: { size: 9, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true
                },
                yaxis: {
                    title: { text: "Intensity", font: { size: 10, color: '#e0e0e0' } },
                    tickfont: { size: 9, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true
                },
                legend: {
                    font: { size: 9, color: '#e0e0e0' },
                    orientation: 'h',
                    y: -0.4
                },
                shapes: []
            };
            
            var accumulated = 0.0;
            var d_phys = design.d_physical_nm;
            
            layout.shapes.push({
                type: 'line',
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 1,
                yref: 'paper',
                line: { color: 'rgba(150, 150, 150, 0.4)', width: 1, dash: 'dash' }
            });
            
            for (var i = 0; i < d_phys.length; i++) {
                accumulated += d_phys[i];
                layout.shapes.push({
                    type: 'line',
                    x0: accumulated,
                    y0: 0,
                    x1: accumulated,
                    y1: 1,
                    yref: 'paper',
                    line: { color: 'rgba(150, 150, 150, 0.4)', width: 1, dash: 'dash' }
                });
            }
            
            Plotly.newPlot('plot-field', traces, layout, {responsive: true, displayModeBar: false});
        }

        function drawSpectrumPlot(design) {
            if (!design.precomputed || !design.spec_wavelengths) {
                showPlotMessage('plot-spectrum', 'Transmission spectrum details not precomputed for this design');
                return;
            }
            
            var traces = [];
            
            // Selected design trace
            traces.push({
                x: design.spec_wavelengths,
                y: design.spec_transmission,
                mode: 'lines',
                line: { color: '#ff9800', width: 2 },
                name: 'Selected (Rank ' + design.rank + ')',
                hovertemplate: "Selected (Rank " + design.rank + ")<br>Wavelength: %{x:.1f} nm<br>Transmission: %{y:.4f}%<extra></extra>"
            });
            
            var hasComp = (comparisonDesignIdx !== null && comparisonDesignIdx !== -1);
            var compDesign = hasComp ? tmmData[comparisonDesignIdx] : null;
            
            if (compDesign && compDesign.spec_wavelengths) {
                traces.push({
                    x: compDesign.spec_wavelengths,
                    y: compDesign.spec_transmission,
                    mode: 'lines',
                    line: { color: '#ff4081', width: 1.5, dash: 'dash' },
                    name: 'Comparison (Rank ' + compDesign.rank + ')',
                    hovertemplate: "Comparison (Rank " + compDesign.rank + ")<br>Wavelength: %{x:.1f} nm<br>Transmission: %{y:.4f}%<extra></extra>"
                });
            }
            
            var layout = {
                paper_bgcolor: '#1e1e1e',
                plot_bgcolor: '#1e1e1e',
                margin: { l: 45, r: 20, t: 15, b: 35 },
                height: 180,
                xaxis: {
                    title: { text: "Wavelength (nm)", font: { size: 10, color: '#e0e0e0' } },
                    tickfont: { size: 9, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    range: [400, 1400]
                },
                yaxis: {
                    title: { text: "Transmission (%)", font: { size: 10, color: '#e0e0e0' } },
                    tickfont: { size: 9, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    range: [0, 105]
                },
                legend: {
                    font: { size: 9, color: '#e0e0e0' },
                    orientation: 'h',
                    y: -0.4
                },
                shapes: [
                    {
                        type: 'line',
                        x0: 1064.0,
                        y0: 0,
                        x1: 1064.0,
                        y1: 1,
                        yref: 'paper',
                        line: { color: '#e53935', width: 1.5, dash: 'dot' }
                    }
                ]
            };
            
            Plotly.newPlot('plot-spectrum', traces, layout, {responsive: true, displayModeBar: false});
        }

        function updateSelectedDesign(idx) {
            selectedDesignIdx = idx;
            
            if (idx === -1) {
                var loss = compareRefl !== null ? 1.0 - compareRefl : 0.0;
                var text = "  REFERENCE DESIGN SUMMARY\n";
                text += "  -------------------------\n";
                text += "  Label: " + referenceLabel + "\n";
                if (compareRefl !== null) {
                    text += "  Reflectivity: " + compareRefl.toFixed(6) + "\n";
                    text += "  Loss (1 - R): " + loss.toExponential(4) + "\n";
                }
                if (compareAbs !== null) {
                    text += "  Absorption: " + compareAbs.toFixed(3) + " ppm\n";
                }
                if (compareTN !== null) {
                    text += "  Thermal Noise: " + compareTN.toExponential(4) + " m/sqrt(Hz)\n";
                }
                if (compareThick !== null && compareThick > 0) {
                    text += "  Physical Thickness: " + compareThick.toFixed(2) + " nm\n";
                }
                document.getElementById('info-content').innerText = text;
                document.getElementById('btn-export-py').disabled = true;
                document.getElementById('btn-export-csv').disabled = true;
                document.getElementById('btn-set-baseline').disabled = true;
                document.getElementById('btn-set-comparison-stack').disabled = true;
                document.getElementById('btn-set-selected-comp-point').disabled = true;
                
                showPlotMessage('plot-stack', 'Detailed layout not available for scalar reference point');
                showPlotMessage('plot-field', 'EFI solver details not available for reference');
                showPlotMessage('plot-spectrum', 'Spectrum details not available for reference');
                return;
            }
            
            var design = tmmData[idx];
            if (!design) return;
            
            document.getElementById('info-content').innerText = design.info_text;
            document.getElementById('btn-export-py').disabled = false;
            document.getElementById('btn-export-csv').disabled = false;
            document.getElementById('btn-set-baseline').disabled = false;
            document.getElementById('btn-set-comparison-stack').disabled = false;
            document.getElementById('btn-set-selected-comp-point').disabled = false;
            
            drawStackPlot(design);
            drawEFIPlot(design);
            drawSpectrumPlot(design);
        }

        // Bind Plotly click handler
        var plot3dDiv = document.getElementById('plot-3d');
        plot3dDiv.on('plotly_click', function(data) {
            if (data.points && data.points.length > 0) {
                var pt = data.points[0];
                var customdata = pt.customdata;
                
                if (pt.curveNumber === 1) {
                    updateSelectedDesign(-1);
                } else if (customdata && customdata.length > 6) {
                    var designIdx = parseInt(customdata[6]);
                    updateSelectedDesign(designIdx);
                }
            }
        });

        // Export scripts
        function getPythonExportString(design) {
            var dOpt_lines = design.dOpt.map(v => "    " + v.toFixed(6)).join(",\\n");
            var materialLayer_lines = design.materialLayer.map(v => "    " + v).join(",\\n");
            var d_phys_lines = design.d_physical_nm.map(v => "    " + v.toFixed(6)).join(",\\n");
            
            var matParamsLines = [];
            for (var k in materialsParamsDict) {
                var v = materialsParamsDict[k];
                matParamsLines.push("    " + k + ": " + JSON.stringify(v));
            }
            var materialParamsStr = matParamsLines.join(",\\n");

            var py = `# ==============================================================================\\n` +
                     `# Rank ${design.rank} Coating Design - Exported from coatopt\\n` +
                     `# Reflectivity: ${design.reflectivity.toFixed(6)}\\n` +
                     `# Absorption: ${design.absorption.toFixed(3)} ppm\\n` +
                     `# Thermal Noise: ${design.thermal_noise.toExponential(4)} m/sqrt(Hz)\\n` +
                     `# ==============================================================================\\n\\n` +
                     `import numpy as np\\n\\n` +
                     `# --- Design Parameters ---\\n` +
                     `# Number of layers: ${design.dOpt.length}\\n` +
                     `# Total physical thickness: ${design.d_physical_nm.reduce((a,b)=>a+b, 0).toFixed(2)} nm\\n\\n` +
                     `# Optical Thicknesses (dOpt)\\n` +
                     `dOpt = np.array([\\n${dOpt_lines}\\n])\\n\\n` +
                     `# Material Layer Indices (materialLayer)\\n` +
                     `# 999/0 = Air, 1 = SiO2, 2 = TiGermania\\n` +
                     `materialLayer = np.array([\\n${materialLayer_lines}\\n])\\n\\n` +
                     `# Physical Thicknesses (nm)\\n` +
                     `physical_thickness = np.array([\\n${d_phys_lines}\\n])\\n\\n` +
                     `# Material Definitions\\n` +
                     `materialParams = {\\n${materialParamsStr}\\n}\\n\\n` +
                     `# --- aLIGO Params Structure ---\\n` +
                     `aLIGO_params = {}\\n\\n` +
                     `## INPUTS \\n` +
                     `aLIGO_params['StackName']      = 'Rank ${design.rank} Design'               # Label for run \\n` +
                     `aLIGO_params["dOpt"]           = dOpt                               # optical thickness array \\n` +
                     `aLIGO_params["materialLayer"]  = materialLayer                      # material array containing keys which index materialParams\\n` +
                     `aLIGO_params["materialParams"] = materialParams                     # dictionary of material properties \\n` +
                     `aLIGO_params["materialSub"]    = 1                                  # substrate type - Silica \\n` +
                     `lambda_ = 1064.0\\n` +
                     `aLIGO_params["lambda_"]        = lambda_                            # IFO wavelength (nm)\\n` +
                     `aLIGO_params["f"]              = np.logspace(1, 3, 100)             # Frequency range to evaluate CTN \\n` +
                     `aLIGO_params["wBeam"]          = 0.062                              # laser beam size on ETM (m) \\n` +
                     `aLIGO_params["Temp"]           = 293.0                              # detector temperature (K) \\n` +
                     `aLIGO_params["plots "]         = False                              # boolean for activating plots \\n` +
                     `aLIGO_params["t_air"]          = 500                                # thickness of air in EFI calculations for optical absorption : Default is 500nm\\n` +
                     `aLIGO_params["polarisation"]   = 'p'                                # light polarisation for EFI calculations \\n` +
                     `aLIGO_params["lambda_list"]    = np.linspace(0, lambda_*1.5, 10000)\\n\\n` +
                     `# --- Design Table ---\\n` +
                     `# Layer | Material Name | Refractive Index | dOpt | Physical Thickness (nm)\\n`;
                     
            for (var i = 0; i < design.dOpt.length; i++) {
                var thick = design.d_physical_nm[i];
                var matIdx = design.materialLayer[i];
                var matName = design.material_names[i];
                var nVal = materialsParamsDict[matIdx] ? materialsParamsDict[matIdx].n : 1.0;
                py += `# ${(i+1).toString().padEnd(5)} | ${matName.padEnd(13)} | ${nVal.toString().padEnd(16)} | ${design.dOpt[i].toFixed(6)} | ${thick.toFixed(2)} nm\\n`;
            }
            py += `\\nprint("Rank ${design.rank} design variables loaded successfully.")\\n`;
            return py;
        }

        function getCSVExportString(design) {
            var csv = "Layer,Material_Index,Material_Name,Refractive_Index,dOpt,Physical_Thickness_nm\\n";
            for (var i = 0; i < design.dOpt.length; i++) {
                var thick = design.d_physical_nm[i];
                var matIdx = design.materialLayer[i];
                var matName = design.material_names[i];
                var nVal = materialsParamsDict[matIdx] ? materialsParamsDict[matIdx].n : 1.0;
                csv += (i+1) + "," + matIdx + "," + matName + "," + nVal + "," + design.dOpt[i].toFixed(6) + "," + thick.toFixed(6) + "\\n";
            }
            return csv;
        }

        function triggerDownload(content, filename, contentType) {
            var blob = new Blob([content], {type: contentType});
            var a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
        
        document.getElementById('btn-export-py').addEventListener('click', function() {
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                var design = tmmData[selectedDesignIdx];
                var content = getPythonExportString(design);
                triggerDownload(content, "rank_" + design.rank + "_design.py", "text/plain");
            }
        });
        
        document.getElementById('btn-export-csv').addEventListener('click', function() {
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                var design = tmmData[selectedDesignIdx];
                var content = getCSVExportString(design);
                triggerDownload(content, "rank_" + design.rank + "_design.csv", "text/csv");
            }
        });

        // Click handler for Comparison Targets Card Apply button
        document.getElementById('btn-apply-targets').addEventListener('click', function() {
            recalculateUtilityAndRerank();
        });

        // Click handler for Custom Comparison Point Card Apply button
        document.getElementById('btn-apply-comp-point').addEventListener('click', function() {
            recalculateUtilityAndRerank();
        });

        // Set Selected Design to Custom Plot Comparison Point inputs
        document.getElementById('btn-set-selected-comp-point').addEventListener('click', function() {
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                var design = tmmData[selectedDesignIdx];
                if (design) {
                    document.getElementById('input-comp-label').value = "Design Rank #" + design.rank;
                    document.getElementById('input-comp-refl').value = design.reflectivity.toFixed(6);
                    document.getElementById('input-comp-abs').value = design.absorption.toFixed(4);
                    document.getElementById('input-comp-tn').value = design.thermal_noise.toExponential(4);
                    document.getElementById('input-comp-thick').value = design.total_thickness.toFixed(2);
                    recalculateUtilityAndRerank();
                }
            }
        });

        // Clear Custom Plot Comparison Point
        document.getElementById('btn-clear-comp-point').addEventListener('click', function() {
            document.getElementById('input-comp-label').value = "";
            document.getElementById('input-comp-refl').value = "";
            document.getElementById('input-comp-abs').value = "";
            document.getElementById('input-comp-tn').value = "";
            document.getElementById('input-comp-thick').value = "";
            recalculateUtilityAndRerank();
        });


        // Set as Baseline Target click handler
        document.getElementById('btn-set-baseline').addEventListener('click', function() {
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                var design = tmmData[selectedDesignIdx];
                if (design) {
                    document.getElementById('input-target-refl').value = design.reflectivity.toFixed(6);
                    document.getElementById('input-target-abs').value = design.absorption.toFixed(4);
                    document.getElementById('input-target-tn').value = design.thermal_noise.toExponential(4);
                    document.getElementById('input-target-thick').value = design.total_thickness.toFixed(2);
                    recalculateUtilityAndRerank();
                }
            }
        });

        // Set as Comparison Stack click handler
        document.getElementById('btn-set-comparison-stack').addEventListener('click', function() {
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                comparisonDesignIdx = selectedDesignIdx;
                var compDesign = tmmData[comparisonDesignIdx];
                document.getElementById('btn-clear-comparison-stack').innerText = "Clear Comp Stack (Rank " + compDesign.rank + ")";
                document.getElementById('btn-clear-comparison-stack').style.display = 'block';
                
                // Redraw plots with the new comparison stack
                var design = tmmData[selectedDesignIdx];
                if (design) {
                    drawStackPlot(design);
                    drawEFIPlot(design);
                    drawSpectrumPlot(design);
                }
            }
        });

        // Clear Comparison Stack click handler
        document.getElementById('btn-clear-comparison-stack').addEventListener('click', function() {
            comparisonDesignIdx = null;
            document.getElementById('btn-clear-comparison-stack').style.display = 'none';
            
            // Redraw plots for the selected design without comparison stack
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                var design = tmmData[selectedDesignIdx];
                if (design) {
                    drawStackPlot(design);
                    drawEFIPlot(design);
                    drawSpectrumPlot(design);
                }
            }
        });

        function recalculateUtilityAndRerank() {
            var target_refl = parseFloat(document.getElementById('input-target-refl').value);
            var target_abs = parseFloat(document.getElementById('input-target-abs').value);
            var target_tn = parseFloat(document.getElementById('input-target-tn').value);
            var target_thick = parseFloat(document.getElementById('input-target-thick').value);

            if (isNaN(target_refl) || isNaN(target_abs) || isNaN(target_tn) || isNaN(target_thick)) {
                alert("Please enter valid numeric values for all targets.");
                return;
            }

            var total_w = weightRefl + weightAbs + weightTN + weightThick;
            var w_refl = total_w > 0 ? weightRefl / total_w : 0.10;
            var w_abs = total_w > 0 ? weightAbs / total_w : 0.35;
            var w_tn = total_w > 0 ? weightTN / total_w : 0.45;
            var w_thick = total_w > 0 ? weightThick / total_w : 0.10;

            var refl_loss_scale = Math.max(1e-6, 1.0 - target_refl);

            designsList.forEach(function(d) {
                // Maximize Reflectivity
                var r_score = d.reflectivity >= target_refl ? 
                    (0.9 + 0.1 * (d.reflectivity - target_refl) / refl_loss_scale) :
                    (0.9 * Math.exp(-(target_refl - d.reflectivity) / refl_loss_scale));

                // Minimize Absorption
                var abs_score = d.absorption <= target_abs ?
                    (0.9 + 0.1 * (target_abs - d.absorption) / target_abs) :
                    (0.9 * Math.exp(-(d.absorption - target_abs) / target_abs));

                // Minimize Thermal Noise
                var tn_score = d.thermal_noise <= target_tn ?
                    (0.9 + 0.1 * (target_tn - d.thermal_noise) / target_tn) :
                    (0.9 * Math.exp(-(d.thermal_noise - target_tn) / target_tn));

                // Minimize Thickness
                var thick_score = d.total_thickness <= target_thick ?
                    (0.9 + 0.1 * (target_thick - d.total_thickness) / target_thick) :
                    (0.9 * Math.exp(-(d.total_thickness - target_thick) / target_thick));

                d.utility_score = w_refl * r_score + w_abs * abs_score + w_tn * tn_score + w_thick * thick_score;
            });

            // Sort descending by utility
            designsList.sort((a, b) => b.utility_score - a.utility_score);

            // Re-assign ranks 1 to M and update info_text
            designsList.forEach(function(d, index) {
                d.rank = index + 1;
                var loss = 1.0 - d.reflectivity;
                var info_lines = [];
                info_lines.push("  SELECTED DESIGN SUMMARY");
                info_lines.push("  -------------------------");
                info_lines.push("  Design Rank: #" + d.rank + " / " + designsList.length);
                info_lines.push("  Reflectivity: " + d.reflectivity.toFixed(6));
                info_lines.push("  Loss (1 - R): " + loss.toExponential(4));
                info_lines.push("  Absorption: " + d.absorption.toFixed(3) + " ppm");
                info_lines.push("  Thermal Noise: " + d.thermal_noise.toExponential(4) + " m/sqrt(Hz)");
                info_lines.push("  Utility Score: " + d.utility_score.toFixed(4));
                info_lines.push("  Active Layers: " + d.active_layer_count);
                info_lines.push("  Total Physical Thickness: " + d.d_physical_nm.reduce((a, b) => a + b, 0).toFixed(2) + " nm");
                d.info_text = info_lines.join("\\n");
            });

            // Update currently selected design card text
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                updateSelectedDesign(selectedDesignIdx);
            }

            // Update comparison design button label
            if (comparisonDesignIdx !== null && comparisonDesignIdx !== -1) {
                var compDesign = tmmData[comparisonDesignIdx];
                if (compDesign) {
                    document.getElementById('btn-clear-comparison-stack').innerText = "Clear Comp Stack (Rank " + compDesign.rank + ")";
                }
            }

            // Top X filtering
            var topXVal = document.getElementById('input-top-x').value.trim();
            var topX = topXVal === "" ? null : parseInt(topXVal);

            var displayList = designsList;
            if (topX !== null && !isNaN(topX) && topX > 0) {
                displayList = designsList.slice(0, topX);
            }

            var x_data = displayList.map(d => d.absorption);
            var y_data = displayList.map(d => d.thermal_noise);
            var z_data = displayList.map(d => d.rank);

            var customdata = displayList.map(d => [
                d.rank,
                d.reflectivity,
                1.0 - d.reflectivity,
                d.active_layer_count,
                d.total_thickness,
                d.utility_score,
                d.originalIdx
            ]);

            var color_values = displayList.map(d => d.reflectivity);
            var cmin = Math.min(...color_values);
            var cmax = Math.max(...color_values);

            data3d[0].x = x_data;
            data3d[0].y = y_data;
            data3d[0].z = z_data;
            data3d[0].customdata = customdata;
            data3d[0].marker.color = color_values;
            data3d[0].marker.cmin = cmin;
            data3d[0].marker.cmax = cmax;

            // Recalculate custom comparison plot point virtual rank and update coordinates
            var comp_label = document.getElementById('input-comp-label').value.trim() || "Reference Design";
            var comp_refl = parseFloat(document.getElementById('input-comp-refl').value);
            var comp_abs = parseFloat(document.getElementById('input-comp-abs').value);
            var comp_tn = parseFloat(document.getElementById('input-comp-tn').value);
            var comp_thick = parseFloat(document.getElementById('input-comp-thick').value);

            var show_comp_point = !isNaN(comp_abs) && !isNaN(comp_tn);

            if (show_comp_point) {
                // If reflectivity is not specified, assume it meets target exactly (0.90 score)
                var r_val = isNaN(comp_refl) ? target_refl : comp_refl;
                var r_comp_score = r_val >= target_refl ? 
                    (0.9 + 0.1 * (r_val - target_refl) / refl_loss_scale) :
                    (0.9 * Math.exp(-(target_refl - r_val) / refl_loss_scale));

                var abs_comp_score = comp_abs <= target_abs ?
                    (0.9 + 0.1 * (target_abs - comp_abs) / target_abs) :
                    (0.9 * Math.exp(-(comp_abs - target_abs) / target_abs));

                var tn_comp_score = comp_tn <= target_tn ?
                    (0.9 + 0.1 * (target_tn - comp_tn) / target_tn) :
                    (0.9 * Math.exp(-(comp_tn - target_tn) / target_tn));

                var thick_comp_score = 0.90;
                if (!isNaN(comp_thick) && comp_thick > 0) {
                    thick_comp_score = comp_thick <= target_thick ?
                        (0.9 + 0.1 * (target_thick - comp_thick) / target_thick) :
                        (0.9 * Math.exp(-(comp_thick - target_thick) / target_thick));
                }

                var compare_utility = w_refl * r_comp_score + w_abs * abs_comp_score + w_tn * tn_comp_score + w_thick * thick_comp_score;

                var virtual_rank = 1;
                for (var i = 0; i < designsList.length; i++) {
                    if (compare_utility >= designsList[i].utility_score) {
                        break;
                    }
                    virtual_rank++;
                }
                if (virtual_rank > designsList.length) {
                    virtual_rank = designsList.length + 0.5;
                }

                var rank_str = Number.isInteger(virtual_rank) ? "#" + virtual_rank : "#" + virtual_rank.toFixed(1);
                var legend_name = comp_label + " (Virtual Rank: " + rank_str + " of " + designsList.length + ")";

                var loss_comp = 1.0 - r_val;
                var hover_comp_str = "<b>" + comp_label + " (Reference)</b><br><br>" +
                                     (!isNaN(comp_refl) ? "Reflectivity: " + comp_refl.toFixed(6) + "<br>" : "") +
                                     (!isNaN(comp_refl) ? "Reflectivity Loss: " + loss_comp.toExponential(3) + "<br>" : "") +
                                     "Absorption: " + comp_abs.toFixed(4) + " ppm<br>" +
                                     "Thermal Noise: " + comp_tn.toExponential(4) + " m/sqrt(Hz)<br>" +
                                     (!isNaN(comp_thick) && comp_thick > 0 ? "Total Thickness: " + comp_thick.toFixed(2) + " nm<br>" : "") +
                                     "Virtual Utility Rank: " + rank_str + "<br>" +
                                     "Reference Utility: " + compare_utility.toFixed(4) + "<br>" +
                                     "<extra></extra>";

                data3d[1].x = [comp_abs];
                data3d[1].y = [comp_tn];
                data3d[1].z = [virtual_rank];
                data3d[1].name = legend_name;
                data3d[1].hovertemplate = hover_comp_str;
                data3d[1].visible = true;
                data3d[1].showlegend = true;

                // Update global reference values so clicking the reference point shows the new parameters
                referenceLabel = comp_label;
                compareRefl = isNaN(comp_refl) ? null : comp_refl;
                compareAbs = comp_abs;
                compareTN = comp_tn;
                compareThick = isNaN(comp_thick) ? null : comp_thick;
            } else {
                data3d[1].x = [];
                data3d[1].y = [];
                data3d[1].z = [];
                data3d[1].visible = false;
                data3d[1].showlegend = false;
            }


            var maxRank = Math.max(...data3d[0].z) || 100;
            if (reversedZ) {
                layout3d.scene.zaxis.range = [maxRank + 2.0, 0.5];
            } else {
                layout3d.scene.zaxis.range = [0.5, maxRank + 2.0];
            }

            Plotly.react('plot-3d', data3d, layout3d);
        }

        document.getElementById('btn-apply-top').addEventListener('click', recalculateUtilityAndRerank);
        document.getElementById('input-top-x').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                recalculateUtilityAndRerank();
            }
        });

        // Z-axis view controls
        var reversedZ = true;
        document.getElementById('btn-reverse-z').addEventListener('click', function() {
            reversedZ = !reversedZ;
            var maxRank = Math.max(...data3d[0].z) || 100;
            if (reversedZ) {
                Plotly.relayout('plot-3d', {'scene.zaxis.range': [maxRank + 2.0, 0.5]});
            } else {
                Plotly.relayout('plot-3d', {'scene.zaxis.range': [0.5, maxRank + 2.0]});
            }
        });

        // X & Y scale controls
        var xLog = true;
        document.getElementById('btn-toggle-x-scale').addEventListener('click', function() {
            xLog = !xLog;
            Plotly.relayout('plot-3d', {'scene.xaxis.type': xLog ? 'log' : 'linear'});
        });

        var yLog = true;
        document.getElementById('btn-toggle-y-scale').addEventListener('click', function() {
            yLog = !yLog;
            Plotly.relayout('plot-3d', {'scene.yaxis.type': yLog ? 'log' : 'linear'});
        });
    </script>
</body>
</html>"""

    initial_top_x_str = str(args.top) if args.top is not None else "null"

    # Populate the placeholders using standard replace method (fully robust to f-string brackets)
    compiled_html = html_template.replace("__TITLE__", title)
    compiled_html = compiled_html.replace("__INITIAL_TOP_X__", initial_top_x_str)
    compiled_html = compiled_html.replace("__PLOTLY_DATA_3D__", plotly_data_json)
    compiled_html = compiled_html.replace("__PLOTLY_LAYOUT_3D__", plotly_layout_json)
    compiled_html = compiled_html.replace("__TMM_DATA__", tmm_data_json)
    compiled_html = compiled_html.replace("__MATERIALS_PARAMS__", materials_params_json)
    compiled_html = compiled_html.replace("__HAS_REFERENCE__", "true" if args.compare_abs is not None else "false")
    compiled_html = compiled_html.replace("__REFERENCE_LABEL__", compare_label_str)
    compiled_html = compiled_html.replace("__COMPARE_REFL__", str(compare_refl_val))
    compiled_html = compiled_html.replace("__COMPARE_ABS__", str(compare_abs_val))
    compiled_html = compiled_html.replace("__COMPARE_TN__", str(compare_tn_val))
    compiled_html = compiled_html.replace("__COMPARE_THICK__", str(compare_thick_val))
    compiled_html = compiled_html.replace("__TARGET_REFL__", f"{target_refl:.6f}")
    compiled_html = compiled_html.replace("__TARGET_ABS__", f"{target_abs:.4f}")
    compiled_html = compiled_html.replace("__TARGET_TN__", f"{target_tn:.4e}")
    compiled_html = compiled_html.replace("__TARGET_THICK__", f"{target_thick:.2f}")
    compiled_html = compiled_html.replace("__WEIGHT_REFL__", f"{args.weight_refl:.4f}")
    compiled_html = compiled_html.replace("__WEIGHT_ABS__", f"{args.weight_abs:.4f}")
    compiled_html = compiled_html.replace("__WEIGHT_TN__", f"{args.weight_tn:.4e}")
    compiled_html = compiled_html.replace("__WEIGHT_THICK__", f"{args.weight_thick:.4f}")

    print(f"Saving interactive dashboard to {output_path}...")
    with open(output_path, "w") as f:
        f.write(compiled_html)

    print(f"✓ Dashboard successfully saved to {output_path}")

    if not args.no_open:
        print(f"Opening dashboard in browser: file://{output_path}")
        webbrowser.open(f"file://{output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
