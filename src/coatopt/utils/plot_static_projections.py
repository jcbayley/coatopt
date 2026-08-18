#!/usr/bin/env python3
"""
Static 2D Projection Plots and Coating Design Exporter.
"""

import argparse
import configparser
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add coatopt packages to system path if needed
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from coatopt.utils.utils import load_pareto_front
from coatopt.utils.plot_interactive_3d_rank import parse_design, calculate_physical_thickness

def load_data(directory: Path, aggregate: bool) -> Tuple[pd.DataFrame, dict]:
    """Aggregate Pareto fronts and load materials dictionary."""
    subdirs = []
    direct_file = directory / "pareto_front.csv"
    if direct_file.exists() and not aggregate:
        subdirs = [directory]
    else:
        for root, dirs, files in os.walk(directory):
            if "pareto_front.csv" in files:
                subdirs.append(Path(root))
        subdirs.sort()
        if subdirs and not aggregate:
            print(f"[INFO] Automatically aggregating runs under {directory} because pareto_front.csv was not found directly in the root.")

    if not subdirs:
        raise ValueError(f"No pareto_front.csv files found in {directory}")

    all_designs = []
    all_values = []
    materials = {}

    for subdir in subdirs:
        try:
            designs_df, values_df, _ = load_pareto_front(subdir)
            
            # Layer filtering (exactly as in main plot script)
            temp_counts = []
            max_active_in_run = 0
            for idx, row in designs_df.iterrows():
                dOpt, mat_idx = parse_design(row)
                active_mask = (mat_idx != 0) & (dOpt > 1e-12)
                active_layer_count = int(np.sum(active_mask))
                temp_counts.append(active_layer_count)
                max_active_in_run = max(max_active_in_run, active_layer_count)
                
            min_required_layers = min(10, max_active_in_run) if max_active_in_run > 0 else 0
            valid_indices = [i for i, c in enumerate(temp_counts) if c >= min_required_layers]
            
            designs_df = designs_df.iloc[valid_indices].reset_index(drop=True)
            values_df = values_df.iloc[valid_indices].reset_index(drop=True)
            
            run_name = str(subdir.relative_to(directory)) if subdir != directory else subdir.name
            values_df["run_name"] = run_name
            all_designs.append(designs_df)
            all_values.append(values_df)

            # Try to load materials: first check if materials.json exists directly in run directory
            run_materials_path = subdir / "materials.json"
            if run_materials_path.exists():
                try:
                    from coatopt.utils.utils import load_materials
                    sub_materials = load_materials(str(run_materials_path))
                    if sub_materials:
                        for k, v in sub_materials.items():
                            if isinstance(k, int):
                                materials[k] = v
                        print(f"  Loaded materials library from run directory: {run_materials_path.name}")
                except Exception:
                    pass

            # Otherwise try loading config-specified materials
            if not materials:
                config_path = subdir / "config.ini"
                if config_path.exists():
                    config = configparser.ConfigParser()
                    config.read(config_path)
                    try:
                        section = "General" if config.has_section("General") else ("general" if config.has_section("general") else None)
                        if section:
                            materials_path_str = config.get(section, "materials_path", fallback=None)
                            if materials_path_str:
                                materials_path = Path(materials_path_str)
                                if not materials_path.is_absolute():
                                    candidate1 = (config_path.parent / materials_path).resolve()
                                    candidate2 = (config_path.parent.parent / materials_path).resolve()
                                    if candidate1.exists():
                                        materials_path = candidate1
                                    elif candidate2.exists():
                                        materials_path = candidate2
                                else:
                                    if not materials_path.exists():
                                        filename = materials_path.name
                                        project_root = Path(__file__).parent.parent.parent.parent
                                        local_candidate1 = (config_path.parent / filename).resolve()
                                        local_candidate2 = (project_root / "experiments" / filename).resolve()
                                        local_candidate3 = (config_path.parent.parent / "experiments" / filename).resolve()
                                        if local_candidate1.exists():
                                            materials_path = local_candidate1
                                        elif local_candidate2.exists():
                                            materials_path = local_candidate2
                                        elif local_candidate3.exists():
                                            materials_path = local_candidate3
                                
                                if Path(materials_path).exists():
                                    from coatopt.utils.utils import load_materials
                                    sub_materials = load_materials(str(materials_path))
                                    if sub_materials:
                                        for k, v in sub_materials.items():
                                            if isinstance(k, int):
                                                materials[k] = v
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: Failed to load Pareto front from {subdir}: {e}")

    # Fallback to load materials from parent directory if still empty
    if not materials:
        parent_materials_path = directory / "materials.json"
        if parent_materials_path.exists():
            try:
                from coatopt.utils.utils import load_materials
                materials = load_materials(str(parent_materials_path))
            except Exception:
                pass

    # Fallback to load default materials
    if not materials:
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            default_mats_path = project_root / "experiments" / "materials.json"
            if default_mats_path.exists():
                from coatopt.utils.utils import load_materials
                materials = load_materials(str(default_mats_path))
                print(f"Loaded default materials library from: {default_mats_path}")
        except Exception:
            pass

    # Detect laser wavelength from config.ini files
    wavelength_nm = None
    wavelength_src = None
    for subdir in subdirs:
        config_path = subdir / "config.ini"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in ["General", "general", "Data", "data"]:
                if cfg.has_section(section) and cfg.has_option(section, "wavelength"):
                    try:
                        w_val = float(cfg.get(section, "wavelength"))
                        if w_val <= 1e-3:
                            w_val *= 1e9
                        wavelength_nm = w_val
                        wavelength_src = f"{subdir.name}/config.ini [{section}]"
                        break
                    except ValueError:
                        pass
        if wavelength_nm is not None:
            break

    if wavelength_nm is None:
        config_path = directory / "config.ini"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in ["General", "general", "Data", "data"]:
                if cfg.has_section(section) and cfg.has_option(section, "wavelength"):
                    try:
                        w_val = float(cfg.get(section, "wavelength"))
                        if w_val <= 1e-3:
                            w_val *= 1e9
                        wavelength_nm = w_val
                        wavelength_src = f"{directory.name}/config.ini [{section}]"
                        break
                    except ValueError:
                        pass

    if wavelength_nm is not None:
        print(f"  Loaded laser wavelength: {wavelength_nm:.1f} nm (from {wavelength_src})")
    else:
        wavelength_nm = 1064.0
        print("  No 'wavelength' key found in config.ini. Defaulting laser wavelength to 1064.0 nm.")

    if materials:
        materials = {int(k): v for k, v in materials.items()}
    else:
        materials = {1: {"name": "SiO2", "n": 1.45}, 2: {"name": "TiGermania", "n": 2.1}}

    designs_df = pd.concat(all_designs, axis=0, ignore_index=True)
    values_df = pd.concat(all_values, axis=0, ignore_index=True)
    combined_df = pd.concat([designs_df, values_df], axis=1)

    return combined_df, materials, wavelength_nm

def main():
    parser = argparse.ArgumentParser(description="Generate static 2D projections and stack layouts")
    parser.add_argument("directory", type=str, help="Directory containing Pareto front runs")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate Pareto fronts recursively")
    parser.add_argument("--selected-rank", type=int, default=1494, help="Selected design rank to highlight")
    parser.add_argument("--compare-label", type=str, default="Chirp N3", help="Comparison design label")
    parser.add_argument("--compare-refl", type=float, default=0.9999, help="Comparison design reflectivity")
    parser.add_argument("--compare-abs", type=float, default=0.3, help="Comparison design absorption (ppm)")
    parser.add_argument("--compare-tn", type=float, default=4.0e-21, help="Comparison design thermal noise")
    parser.add_argument("--compare-thick", type=float, default=9003.0, help="Comparison design thickness (nm)")
    parser.add_argument("--output-projections", type=str, default=None, help="Output path for 2D projections PNG")
    parser.add_argument("--output-stack", type=str, default=None, help="Output path for stack design PNG")
    
    # Matching interactive script parameters
    parser.add_argument("--min-refl", type=float, default=None, help="Minimum reflectivity threshold to filter Pareto designs before ranking")
    parser.add_argument("--max-abs", type=float, default=None, help="Maximum absorption threshold (ppm) to filter Pareto designs before ranking")
    parser.add_argument("--max-tn", type=float, default=None, help="Maximum thermal noise (CTN) threshold to filter Pareto designs before ranking")
    parser.add_argument("--weight-refl", type=float, default=0.10, help="Weight for reflectivity in utility score (default: 0.10)")
    parser.add_argument("--weight-abs", type=float, default=0.35, help="Weight for absorption in utility score (default: 0.35)")
    parser.add_argument("--weight-tn", type=float, default=0.45, help="Weight for thermal noise (CTN) in utility score (default: 0.45)")
    parser.add_argument("--weight-thick", type=float, default=0.10, help="Weight for physical thickness in utility score (default: 0.10)")
    parser.add_argument("--target-refl", type=float, default=None, help="Target reflectivity for utility scoring (defaults to compare-refl if set, else 0.9999)")
    parser.add_argument("--target-abs", type=float, default=None, help="Target absorption in ppm for utility scoring (defaults to compare-abs if set, else 0.30)")
    parser.add_argument("--target-tn", type=float, default=None, help="Target thermal noise (CTN) for utility scoring (defaults to compare-tn if set, else 4.0e-21)")
    parser.add_argument("--target-thick", type=float, default=None, help="Target thickness for utility scoring (defaults to compare-thick if set, else 6000.0)")
    
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

    print(f"Loading data from {directory}...")
    try:
        combined_df, materials_dict, wavelength_nm = load_data(directory, args.aggregate)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Loaded {len(combined_df)} designs. Performing utility ranking...")

    # Filter by minimum reflectivity if specified
    if args.min_refl is not None:
        combined_df = combined_df[combined_df["reflectivity"] >= args.min_refl].reset_index(drop=True)

    # Filter by maximum absorption if specified
    if args.max_abs is not None:
        combined_df = combined_df[combined_df["absorption"] <= args.max_abs].reset_index(drop=True)

    # Filter by maximum thermal noise if specified
    if args.max_tn is not None:
        combined_df = combined_df[combined_df["thermal_noise"] <= args.max_tn].reset_index(drop=True)

    # Calculate physical thicknesses
    thickness_vals = []
    for _, row in combined_df.iterrows():
        thick = calculate_physical_thickness(row, materials_dict)
        thickness_vals.append(thick)
    combined_df["total_thickness"] = thickness_vals

    # Scoring parameters (default to standard dashboard targets, independent of comparison points unless explicitly set)
    target_refl = args.target_refl if args.target_refl is not None else 0.9999
    target_abs = args.target_abs if args.target_abs is not None else 0.30
    target_tn = args.target_tn if args.target_tn is not None else 4.0e-21
    target_thick = args.target_thick if args.target_thick is not None else 6000.0

    # Normalize weights so they sum to 1.0
    total_w = args.weight_refl + args.weight_abs + args.weight_tn + args.weight_thick
    w_refl = args.weight_refl / total_w if total_w > 0 else 0.10
    w_abs = args.weight_abs / total_w if total_w > 0 else 0.35
    w_tn = args.weight_tn / total_w if total_w > 0 else 0.45
    w_thick = args.weight_thick / total_w if total_w > 0 else 0.10

    refl_loss_scale = max(1e-6, 1.0 - target_refl)
    r_score = np.where(
        combined_df["reflectivity"] >= target_refl,
        0.9 + 0.1 * (combined_df["reflectivity"] - target_refl) / refl_loss_scale,
        0.9 * np.exp(-(target_refl - combined_df["reflectivity"]) / refl_loss_scale)
    )
    abs_score = np.where(
        combined_df["absorption"] <= target_abs,
        0.9 + 0.1 * (target_abs - combined_df["absorption"]) / target_abs,
        0.9 * np.exp(-(combined_df["absorption"] - target_abs) / target_abs)
    )
    tn_score = np.where(
        combined_df["thermal_noise"] <= target_tn,
        0.9 + 0.1 * (target_tn - combined_df["thermal_noise"]) / target_tn,
        0.9 * np.exp(-(combined_df["thermal_noise"] - target_tn) / target_tn)
    )
    thick_score = np.where(
        combined_df["total_thickness"] <= target_thick,
        0.9 + 0.1 * (target_thick - combined_df["total_thickness"]) / target_thick,
        0.9 * np.exp(-(combined_df["total_thickness"] - target_thick) / target_thick)
    )

    combined_df["utility_score"] = (
        w_refl * r_score +
        w_abs * abs_score +
        w_tn * tn_score +
        w_thick * thick_score
    )

    # Sort and rank
    combined_df = combined_df.sort_values("utility_score", ascending=False).reset_index(drop=True)
    combined_df["rank"] = combined_df.index + 1

    # Extract selected design
    sel_rank = args.selected_rank
    if sel_rank < 1 or sel_rank > len(combined_df):
        print(f"Error: Selected rank {sel_rank} is out of bounds (1 to {len(combined_df)})")
        sys.exit(1)
    sel_row = combined_df[combined_df["rank"] == sel_rank].iloc[0]

    print(f"\nTarget Design Selected at Rank #{sel_rank}:")
    print(f"  Run Directory: {sel_row['run_name']}")
    print(f"  Reflectivity: {sel_row['reflectivity']:.6f}")
    print(f"  Loss (1-R): {1.0 - sel_row['reflectivity']:.4e}")
    print(f"  Absorption: {sel_row['absorption']:.3f} ppm")
    print(f"  Thermal Noise: {sel_row['thermal_noise']:.4e} m/sqrt(Hz)")
    print(f"  Total Thickness: {sel_row['total_thickness']:.2f} nm")

    # Export design files for the UI and standalone simulation
    import csv
    from datetime import datetime

    # Parse thicknesses and materials
    thicknesses, material_indices = parse_design(sel_row)
    active_mask = (material_indices != 0) & (thicknesses > 1e-12)
    active_dOpt = thicknesses[active_mask][::-1]
    active_material_indices = material_indices[active_mask][::-1]

    d_physical_nm = []
    for i in range(len(active_dOpt)):
        mat_idx = active_material_indices[i]
        n_layer = materials_dict.get(mat_idx, {}).get("n", 1.45)
        t_nm = active_dOpt[i] * wavelength_nm / n_layer
        d_physical_nm.append(t_nm)

    # 1. Export JSON file for UI import
    layers_json = []
    for t, m in zip(d_physical_nm, active_material_indices):
        layers_json.append({
            "thickness": float(t),
            "material": int(m)
        })

    materialParams = {}
    for k, v in materials_dict.items():
        mat_key = int(k)
        mat_data = v.copy()
        for field in ["a", "alpha", "beta", "kappa", "C", "Y", "prat", "phiM", "k"]:
            if mat_data.get(field) is None:
                if field == "C":
                    mat_data[field] = 1.64e6 if "SiO2" in mat_data.get("name", "") else 2.51e6
                elif field == "Y":
                    mat_data[field] = 70e9 if "SiO2" in mat_data.get("name", "") else 92e9
                elif field == "prat":
                    mat_data[field] = 0.19 if "SiO2" in mat_data.get("name", "") else 0.29
                elif field == "phiM":
                    mat_data[field] = 2.3e-5 if "SiO2" in mat_data.get("name", "") else 9.013672e-5
                else:
                    mat_data[field] = 0.0
        materialParams[str(mat_key)] = mat_data

    for key in ["0", "999"]:
        if key not in materialParams:
            materialParams[key] = {
                "name": "air",
                "desc": "Air",
                "n": 1.0,
                "k": 0.0,
                "a": None,
                "alpha": None,
                "beta": None,
                "kappa": None,
                "C": None,
                "Y": None,
                "prat": None,
                "phiM": None
            }

    ui_json_payload = {
        "stack_name": f"Rank {sel_rank} Design",
        "layers": layers_json,
        "materialParams": materialParams,
        "globals": {
            "lambda": wavelength_nm,
            "targetLambda": 532.0,
            "wBeam": 0.062,
            "temp": 293.0,
            "polarisation": "p",
            "angle": 0.0
        }
    }

    json_out_path = directory / f"rank_{sel_rank}_design.json"
    with open(json_out_path, "w") as f:
        json.dump(ui_json_payload, f, indent=4)
    print(f"✓ Saved UI JSON design to: {json_out_path}")

    # 2. Export CSV file for UI import
    csv_out_path = directory / f"rank_{sel_rank}_design.csv"
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer_Number", "Material_Index", "Physical_Thickness_nm"])
        for i, (thick, mat) in enumerate(zip(d_physical_nm, active_material_indices)):
            writer.writerow([i + 1, int(mat), float(thick)])
    print(f"✓ Saved UI CSV layers to: {csv_out_path}")

    # 3. Export standalone Python design script
    dOpt_list_str = ",\n".join([f"    {val:.6f}" for val in active_dOpt])
    materialLayer_list_str = ",\n".join([f"    {val}" for val in active_material_indices])
    d_physical_list_str = ",\n".join([f"    {val:.6f}" for val in d_physical_nm])
    params_lines = [f"    {k}: {repr(v)}" for k, v in sorted(materialParams.items())]
    materialParams_dict_str = ",\n".join(params_lines)
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py_out_content = f'''# ==============================================================================
# Rank {sel_rank} Coating Design - Exported from plot_static_projections
# Date: {current_date}
# Reflectivity: {sel_row['reflectivity']:.6f}
# Absorption: {sel_row['absorption']:.3f} ppm
# Thermal Noise: {sel_row['thermal_noise']:.4e} m/sqrt(Hz)
# ==============================================================================

import numpy as np

# --- Design Parameters ---
# Number of layers: {len(active_dOpt)}
# Total physical thickness: {sum(d_physical_nm):.2f} nm

# Optical Thicknesses (dOpt)
dOpt = np.array([
{dOpt_list_str}
])

# Material Layer Indices (materialLayer)
# 999/0 = Air, 1 = SiO2, 2 = TiGermania
materialLayer = np.array([
{materialLayer_list_str}
])

# Physical Thicknesses (nm)
physical_thickness = np.array([
{d_physical_list_str}
])

# Material Definitions
materialParams = {{
{materialParams_dict_str}
}}
'''
    py_out_path = directory / f"rank_{sel_rank}_design.py"
    with open(py_out_path, "w") as f:
        f.write(py_out_content)
    print(f"✓ Saved standalone Python design to: {py_out_path}\n")


    # Set up matplotlib style (sleek premium light style)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["grid.color"] = "#e0e0e0"
    plt.rcParams["grid.linestyle"] = ":"

    # --- PART 1: 2D PROJECTIONS PLOT ---
    # Prepare plot data
    x_abs = combined_df["absorption"].values
    y_tn = combined_df["thermal_noise"].values
    z_loss = np.maximum(1e-10, 1.0 - combined_df["reflectivity"].values)
    z_nines = -np.log10(z_loss)
    
    # Colors determined by CTN Log10 (the user's colorbar rules)
    colors = np.log10(y_tn)
    norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max())
    cmap = plt.cm.viridis_r  # Reversed viridis so lower thermal noise (better) is brighter

    # Markers for highlight
    highlight_sel = {
        "x": sel_row["absorption"],
        "y": sel_row["thermal_noise"],
        "z": -np.log10(1.0 - sel_row["reflectivity"]),
        "label": f"Rank #{sel_rank} (Selected)"
    }
    
    highlight_comp = {
        "x": args.compare_abs,
        "y": args.compare_tn,
        "z": -np.log10(1.0 - args.compare_refl),
        "label": args.compare_label
    }

    # Bounding calculations to ensure all points are fully confined within the visible axes
    all_abs = np.append(x_abs, [highlight_comp["x"], highlight_sel["x"]])
    all_tn = np.append(y_tn, [highlight_comp["y"], highlight_sel["y"]])
    all_nines = np.append(z_nines, [highlight_comp["z"], highlight_sel["z"]])

    log_abs_min = np.log10(all_abs.min())
    log_abs_max = np.log10(all_abs.max())
    abs_pad = 0.08 * (log_abs_max - log_abs_min)
    abs_lim = [10 ** (log_abs_min - abs_pad), 10 ** (log_abs_max + abs_pad)]

    log_tn_min = np.log10(all_tn.min())
    log_tn_max = np.log10(all_tn.max())
    tn_pad = 0.08 * (log_tn_max - log_tn_min)
    tn_lim = [10 ** (log_tn_min - tn_pad), 10 ** (log_tn_max + tn_pad)]

    nines_min = all_nines.min()
    nines_max = all_nines.max()
    nines_pad = 0.08 * (nines_max - nines_min)
    nines_lim = [nines_min - nines_pad, nines_max + nines_pad]
    
    fig = plt.figure(figsize=(22, 7.5), dpi=200)
    # Define a GridSpec with 2 rows and 4 columns:
    # Row 0 contains the subplots and the narrow colorbar axis
    # Row 1 is a thin empty spacer row at the bottom to leave dedicated space for the bottom legend
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.04], height_ratios=[1, 0.08])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    
    axes = [ax1, ax2, ax3]
    for ax in axes:
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=18, colors="black")
           # fig.suptitle(f"Coating Optimization: 2D Pareto Front Projections\nAggregated across {directory.name}", fontsize=20, fontweight="bold", color="#00838f", y=0.98)

    # Plot 1: Thermal Noise vs Absorption (Log-Log)
    ax = ax1
    sc = ax.scatter(x_abs, y_tn, c=colors, cmap=cmap, norm=norm, s=12, alpha=0.6, edgecolors="none")
    ax.scatter(highlight_comp["x"], highlight_comp["y"], marker="D", c="#ff007f", s=350, edgecolors="black", linewidth=2.5, zorder=10, label=highlight_comp["label"])
    ax.scatter(highlight_sel["x"], highlight_sel["y"], marker="*", c="#00e5ff", s=500, edgecolors="black", linewidth=2.5, zorder=11, label=highlight_sel["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(abs_lim)
    ax.set_ylim(tn_lim)
    ax.set_xlabel("Absorption (ppm)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Thermal Noise (m/$\sqrt{\mathrm{Hz}}$)", fontsize=18, fontweight="bold")
    ax.set_title("CTN vs Absorption", fontsize=18, color="#333333", fontweight="bold")
    ax.grid(True, which="both", color="#e0e0e0")

    # Plot 2: Reflectivity Nines vs Absorption (Log-Linear)
    ax = ax2
    ax.scatter(x_abs, z_nines, c=colors, cmap=cmap, norm=norm, s=12, alpha=0.6, edgecolors="none")
    ax.scatter(highlight_comp["x"], highlight_comp["z"], marker="D", c="#ff007f", s=350, edgecolors="black", linewidth=2.5, zorder=10, label=highlight_comp["label"])
    ax.scatter(highlight_sel["x"], highlight_sel["z"], marker="*", c="#00e5ff", s=500, edgecolors="black", linewidth=2.5, zorder=11, label=highlight_sel["label"])
    ax.set_xscale("log")
    ax.set_xlim(abs_lim)
    ax.set_ylim(nines_lim)
    ax.set_xlabel("Absorption (ppm)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Reflectivity (Nines: $-\log_{10}(1-R)$)", fontsize=18, fontweight="bold")
    ax.set_title("Reflectivity vs Absorption", fontsize=18, color="#333333", fontweight="bold")
    ax.grid(True, which="both", color="#e0e0e0")

    # Plot 3: Reflectivity Nines vs Thermal Noise (Log-Linear)
    ax = ax3
    ax.scatter(y_tn, z_nines, c=colors, cmap=cmap, norm=norm, s=12, alpha=0.6, edgecolors="none")
    ax.scatter(highlight_comp["y"], highlight_comp["z"], marker="D", c="#ff007f", s=350, edgecolors="black", linewidth=2.5, zorder=10, label=highlight_comp["label"])
    ax.scatter(highlight_sel["y"], highlight_sel["z"], marker="*", c="#00e5ff", s=500, edgecolors="black", linewidth=2.5, zorder=11, label=highlight_sel["label"])
    ax.set_xscale("log")
    ax.set_xlim(tn_lim)
    ax.set_ylim(nines_lim)
    ax.set_xlabel("Thermal Noise (m/$\sqrt{\mathrm{Hz}}$)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Reflectivity (Nines: $-\log_{10}(1-R)$)", fontsize=18, fontweight="bold")
    ax.set_title("Reflectivity vs Thermal Noise", fontsize=18, color="#333333", fontweight="bold")
    ax.grid(True, which="both", color="#e0e0e0")

    # Find the range of thermal noise to generate clean physical labels for colorbar
    tn_min, tn_max = y_tn.min(), y_tn.max()
    candidates = np.array([5e-22, 8e-22, 1e-21, 1.5e-21, 2e-21, 2.5e-21, 3e-21, 3.5e-21, 4e-21, 4.5e-21, 5e-21])
    cb_tick_vals = candidates[(candidates >= tn_min * 0.95) & (candidates <= tn_max * 1.05)]
    cb_ticks_log = np.log10(cb_tick_vals)
    cb_labels = []
    for val in cb_tick_vals:
        exponent = int(np.floor(np.log10(val)))
        coeff = val / (10 ** exponent)
        if coeff == 1.0:
            cb_labels.append(f"$10^{{{exponent}}}$")
        else:
            if coeff.is_integer():
                cb_labels.append(f"${int(coeff)}\\times 10^{{{exponent}}}$")
            else:
                cb_labels.append(f"${coeff:.1f}\\times 10^{{{exponent}}}$")

    # Add Colorbar (placed cleanly in the dedicated cax column, completely outside the subplots)
    cbar = fig.colorbar(sc, cax=cax, orientation="vertical", ticks=cb_ticks_log)
    cbar.ax.set_yticklabels(cb_labels, fontsize=18)
    cbar.set_label("Thermal Noise (m/$\sqrt{\mathrm{Hz}}$)", fontsize=18, fontweight="bold", color="black")
    cbar.ax.tick_params(labelsize=18, colors="black")

    # Single Unified Legend at the bottom, outside all subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, framealpha=0.9, facecolor="white", edgecolor="#cccccc", fontsize=16, bbox_to_anchor=(0.46, 0.01))

    # Adjust layout
    plt.tight_layout()
    # Leave space at the bottom for the unified legend and at the top for the title
    plt.subplots_adjust(top=0.92, bottom=0.18, left=0.08, right=0.92, wspace=0.28)
    
    out_proj = Path(args.output_projections) if args.output_projections else directory / "pareto_2d_projections.png"
    plt.savefig(out_proj, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"✓ Saved 2D projections to: {out_proj}")

    # Run Coating Stack Analysis using the imported run_stack_analysis
    print("Running Coating Stack Analysis via CoatingAnalysis workflow...")
    try:
        run_stack_analysis = import_coating_analysis()
        
        # Patch materials_dict to inject defaults and air index 999 for CoatingAnalysis
        materials_dict = patch_materials_dict(materials_dict)

        # Prepare active layers (reversed to air-to-substrate order)
        thicknesses, material_indices = parse_design(sel_row)
        active_mask = (material_indices != 0) & (thicknesses > 1e-12)
        active_dOpt = thicknesses[active_mask][::-1]
        active_material_indices = material_indices[active_mask][::-1]
        
        active_d_physical = []
        for d, m in zip(active_dOpt, active_material_indices):
            n_val = materials_dict[m]["n"]
            active_d_physical.append(d * wavelength_nm / n_val)
        active_d_physical = np.array(active_d_physical)
        
        # Run workflow
        design = run_stack_analysis(
            stack_name=f"rank_{sel_rank}_design",
            dOpt=active_dOpt,
            d_physical_layers=active_d_physical,
            materialLayer=active_material_indices,
            materialParams=materials_dict,
            lambda_=wavelength_nm,
            f=np.logspace(0, 3, 100),
            wBeam=0.062,
            Temp=293.15,
            plot_range=[min(380.0, wavelength_nm * 0.3), max(1564.0, wavelength_nm * 1.5)],
            plots=False
        )
        
        # 1. Coating Stack Plot
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        colors_dict = {"SiO2": "#1f77b4", "TiGermania": "#e0115f", "Ta2O5": "#F58518", "aSi": "#54A24B"}
        depth = 0.0
        for thickness, mat_idx in zip(active_d_physical, active_material_indices):
            mat_info = materials_dict.get(mat_idx, {})
            mat_name = mat_info.get("name", f"Material {mat_idx}")
            color_val = colors_dict.get(mat_name, "#bcbd22")
            ax.bar(
                depth + thickness / 2,
                thickness,
                width=thickness,
                align="center",
                color=color_val,
                edgecolor="black",
                linewidth=0.5,
            )
            depth += thickness
            
        ax.set_xlabel("Depth from incident surface (nm)", fontsize=18, fontweight="bold", color="black")
        ax.set_ylabel("Layer thickness (nm)", fontsize=18, fontweight="bold", color="black")
        ax.tick_params(labelsize=18, colors="black")
        ax.set_xlim(0, active_d_physical.sum())
        ax.set_ylim(0, active_d_physical.max() * 1.25)
        ax.grid(True, axis="y", color="#e0e0e0")
        
        from matplotlib.patches import Patch
        legend_elements = []
        unique_materials = sorted(list(set(active_material_indices)))
        for mat_idx in unique_materials:
            mat_info = materials_dict.get(mat_idx, {})
            mat_name = mat_info.get("name", f"Material {mat_idx}")
            color_val = colors_dict.get(mat_name, "#bcbd22")
            legend_elements.append(Patch(facecolor=color_val, edgecolor="black", label=mat_name))
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, facecolor="white", edgecolor="#cccccc", fontsize=16)
        
        plt.tight_layout()
        out_stack = Path(args.output_stack) if args.output_stack else directory / f"rank_{sel_rank}_coating_design.png"
        plt.savefig(out_stack, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close()
        print(f"✓ Saved stack design to: {out_stack}")
        
        # 2. EFI Plot
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        ds = np.array(design["ds"])
        EFI = np.array(design["EFI"])
        coating_end_nm = active_d_physical.sum()
        coating_mask = (ds >= 0) & (ds <= coating_end_nm)
        
        ax.plot(ds[coating_mask], EFI[coating_mask], color="#222222", lw=2)
        ax.set_xlabel("Depth from incident surface (nm)", fontsize=18, fontweight="bold", color="black")
        ax.set_ylabel("Normalized EFI", fontsize=18, fontweight="bold", color="black")
        ax.tick_params(labelsize=18, colors="black")
        ax.set_xlim(0, coating_end_nm)
        ax.set_ylim(0, np.max(EFI[coating_mask]) * 1.1)
        ax.grid(True, color="#e0e0e0")
        
        plt.tight_layout()
        out_efi = directory / f"rank_{sel_rank}_efi_profile.png"
        plt.savefig(out_efi, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close()
        print(f"✓ Saved EFI profile to: {out_efi}")
        
        # 3. Transmission Spectrum Plot
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        lambda_list = np.array(design["lambda_list"])
        transmission_spec = np.array(design["transmission_spec"])
        
        ax.plot(lambda_list, transmission_spec * 1e6, color="#1f77b4", lw=2)
        ax.set_xlabel("Wavelength (nm)", fontsize=18, fontweight="bold", color="black")
        ax.set_ylabel("Transmission (ppm)", fontsize=18, fontweight="bold", color="black")
        ax.tick_params(labelsize=18, colors="black")
        ax.set_xlim(lambda_list.min(), lambda_list.max())
        ax.grid(True, color="#e0e0e0")
        
        plt.tight_layout()
        out_trans = directory / f"rank_{sel_rank}_transmission_spectrum.png"
        plt.savefig(out_trans, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close()
        print(f"✓ Saved Transmission spectrum to: {out_trans}")
        
    except Exception as e:
        print(f"Warning: Bypassing CoatingAnalysis stack plotting due to error: {e}")
        # Fallback simple bar plot
        print("Falling back to simple stack design plot...")
        # Fallback simple bar plot (reversed to air-to-substrate order)
        thicknesses, material_indices = parse_design(sel_row)
        active_mask = (material_indices != 0) & (thicknesses > 1e-12)
        active_dOpt = thicknesses[active_mask][::-1]
        active_material_indices = material_indices[active_mask][::-1]
        
        active_d_physical = []
        for d, m in zip(active_dOpt, active_material_indices):
            n_val = materials_dict[m]["n"]
            active_d_physical.append(d * wavelength_nm / n_val)
        active_d_physical = np.array(active_d_physical)
        
        fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        colors_dict = {"SiO2": "#1f77b4", "TiGermania": "#e0115f", "Ta2O5": "#F58518", "aSi": "#54A24B"}
        depth = 0.0
        for thickness, mat_idx in zip(active_d_physical, active_material_indices):
            mat_info = materials_dict.get(mat_idx, {})
            mat_name = mat_info.get("name", f"Material {mat_idx}")
            color_val = colors_dict.get(mat_name, "#bcbd22")
            ax.bar(
                depth + thickness / 2,
                thickness,
                width=thickness,
                align="center",
                color=color_val,
                edgecolor="black",
                linewidth=0.5,
            )
            depth += thickness
            
        ax.set_xlabel("Depth from incident surface (nm)", fontsize=18, fontweight="bold", color="black")
        ax.set_ylabel("Layer thickness (nm)", fontsize=18, fontweight="bold", color="black")
        ax.tick_params(labelsize=18, colors="black")
        ax.set_xlim(0, active_d_physical.sum())
        ax.set_ylim(0, active_d_physical.max() * 1.25)
        ax.grid(True, axis="y", color="#e0e0e0")
        
        from matplotlib.patches import Patch
        legend_elements = []
        unique_materials = sorted(list(set(active_material_indices)))
        for mat_idx in unique_materials:
            mat_info = materials_dict.get(mat_idx, {})
            mat_name = mat_info.get("name", f"Material {mat_idx}")
            color_val = colors_dict.get(mat_name, "#bcbd22")
            legend_elements.append(Patch(facecolor=color_val, edgecolor="black", label=mat_name))
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, facecolor="white", edgecolor="#cccccc", fontsize=16)
        
        plt.tight_layout()
        out_stack = Path(args.output_stack) if args.output_stack else directory / f"rank_{sel_rank}_coating_design.png"
        plt.savefig(out_stack, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close()
        print(f"✓ Saved fallback stack design to: {out_stack}")

def import_coating_analysis():
    candidate_paths = [
        "/Users/simon/Developer/Python/CoatingAnalysis/src",
        "/Users/simon/Library/CloudStorage/GoogleDrive-simon.tait@ligo.org/My Drive/BackupFromDropbox/Python/CoatingAnalysis/src"
    ]
    for path_str in candidate_paths:
        p = Path(path_str)
        if p.exists():
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            try:
                from coating_analysis.Coatings_development import run_stack_analysis
                return run_stack_analysis
            except ImportError:
                pass
    raise ImportError("Could not find or import CoatingAnalysis library.")

def patch_materials_dict(materials_dict):
    # Ensure all required keys exist and default if not present
    for k, v in list(materials_dict.items()):
        if "rho" not in v:
            if "SiO2" in v.get("name", ""):
                v["rho"] = 2202.0
            elif "Ti" in v.get("name", "") or "Ge" in v.get("name", ""):
                v["rho"] = 6850.0  # fallback to Ta2O5/TiGermania density
            else:
                v["rho"] = 2202.0
        # Populate null values with standard physical defaults for CTN calculations
        for field in ["alpha", "beta", "kappa", "C", "Y", "prat", "phiM", "k"]:
            if v.get(field) is None:
                if field == "C":
                    v[field] = 1.64e6 if "SiO2" in v.get("name", "") else 2.1e6
                elif field == "Y":
                    v[field] = 70e9 if "SiO2" in v.get("name", "") else 120e9
                elif field == "prat":
                    v[field] = 0.17 if "SiO2" in v.get("name", "") else 0.23
                elif field == "phiM":
                    v[field] = 2.3e-5 if "SiO2" in v.get("name", "") else 2.3e-4
                else:
                    v[field] = 0.0

    # Ensure 999 (Air) exists for CoatingAnalysis library compatibility
    if 999 not in materials_dict:
        materials_dict[999] = {
            "name": "Air",
            "desc": "Incident medium",
            "n": 1.0,
            "a": 0,
            "alpha": 0.0,
            "beta": 0.0,
            "kappa": 0.0,
            "C": 0.0,
            "Y": 0.0,
            "prat": 0.0,
            "rho": 0.0,
            "phiM": 0.0,
            "k": 0.0,
        }
    return materials_dict

if __name__ == "__main__":
    main()
