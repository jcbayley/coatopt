#!/usr/bin/env python3
"""
Interactive Pareto front and Coating-Inspection Layout using Matplotlib.

This script implements a premium two-column dashboard:
- Left Column: 2x2 grid representing the Pareto front projections
  (1 - Reflectivity vs Absorption, 1 - Reflectivity vs Thermal Noise, Absorption vs Thermal Noise).
- Right Column: 3 vertically stacked axes representing physical diagnostics of the selected design:
  - Coating stack diagram (vertical bar representation using the thin_film_stack machinery).
  - Electric field profile (EFI vs depth with vertical interface lines).
  - Simulated spectral response (transmission spectra from 400nm to 1400nm).

Clicking any point in the left Pareto plots instantly updates all right-hand subplots.
"""

import argparse
import configparser
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Try to import rich for the beautiful startup loading bar
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Global placeholders for lazy-loaded dependencies
plt = None
np = None
pd = None
Patch = None
getCoatingThermalNoise = None
optical_to_physical = None
CalculateEFI_tmm = None
CalculateTransmission_tmm = None
thin_film_stack = None
load_materials = None
parse_design = None
load_pareto_front = None

def load_dependencies() -> bool:
    """Load the heavy scientific and physics libraries inside a rich progress bar."""
    global plt, np, pd, Patch, getCoatingThermalNoise, optical_to_physical, CalculateEFI_tmm, CalculateTransmission_tmm, thin_film_stack, load_materials, parse_design, load_pareto_front
    
    if HAS_RICH:
        console = Console()
        console.print("[bold blue]Starting Coating Inspection Visualizer...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Task 1: Graphics backend selection
            t1 = progress.add_task("[yellow]Configuring GUI Graphics engine...", total=100)
            import matplotlib
            import platform
            if platform.system() == 'Darwin':
                for backend in ['MacOSX', 'TkAgg', 'Qt5Agg', 'QtAgg']:
                    try:
                        # Force active import validation to prevent lazy-load crashes
                        if backend == 'MacOSX':
                            import matplotlib.backends.backend_macosx
                        elif backend == 'TkAgg':
                            import tkinter
                            import matplotlib.backends._backend_tk
                        elif backend == 'Qt5Agg':
                            import matplotlib.backends.backend_qt5agg
                        elif backend == 'QtAgg':
                            import matplotlib.backends.backend_qtagg
                        matplotlib.use(backend)
                        break
                    except Exception:
                        pass
            progress.update(t1, completed=100)
            
            # Task 2: Core scientific libraries
            t2 = progress.add_task("[yellow]Loading core scientific libraries (numpy, pandas, pyplot)...", total=100)
            import matplotlib.pyplot as temp_plt
            import numpy as temp_np
            import pandas as temp_pd
            from matplotlib.patches import Patch as temp_Patch
            plt = temp_plt
            np = temp_np
            pd = temp_pd
            Patch = temp_Patch
            progress.update(t2, completed=100)
            
            # Task 3: Physics engines (CoatingAnalysis)
            t3 = progress.add_task("[yellow]Loading physics engines (CoatingAnalysis, TMM solvers)...", total=100)
            
            # Setup path to import from local directories
            src_path = str(Path(__file__).parent.parent.parent)
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

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
            progress.update(t3, completed=100)
            
            # Task 4: Local helpers
            t4 = progress.add_task("[yellow]Loading experiment result helpers...", total=100)
            from coatopt.utils.plot_interactive_pareto import load_materials as temp_lm, parse_design as temp_pd_func
            from coatopt.utils.utils import load_pareto_front as temp_lpf
            
            load_materials = temp_lm
            parse_design = temp_pd_func
            load_pareto_front = temp_lpf
            progress.update(t4, completed=100)
            
        console.print("[bold green]✓ All physics and plotting engines loaded successfully![/bold green]\n")
    else:
        # Standard CLI progress printing if rich is missing
        print("Configuring GUI Graphics engine...")
        import matplotlib
        import platform
        if platform.system() == 'Darwin':
            for backend in ['MacOSX', 'TkAgg', 'Qt5Agg', 'QtAgg']:
                try:
                    if backend == 'MacOSX':
                        import matplotlib.backends.backend_macosx
                    elif backend == 'TkAgg':
                        import tkinter
                        import matplotlib.backends._backend_tk
                    elif backend == 'Qt5Agg':
                        import matplotlib.backends.backend_qt5agg
                    elif backend == 'QtAgg':
                        import matplotlib.backends.backend_qtagg
                    matplotlib.use(backend)
                    break
                except Exception:
                    pass
                    
        print("Loading core scientific libraries...")
        import matplotlib.pyplot as temp_plt
        import numpy as temp_np
        import pandas as temp_pd
        from matplotlib.patches import Patch as temp_Patch
        plt = temp_plt
        np = temp_np
        pd = temp_pd
        Patch = temp_Patch
        
        print("Loading physics engines...")
        src_path = str(Path(__file__).parent.parent.parent)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

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
        
        print("Loading experiment result helpers...")
        from coatopt.utils.plot_interactive_pareto import load_materials as temp_lm, parse_design as temp_pd_func
        from coatopt.utils.utils import load_pareto_front as temp_lpf
        load_materials = temp_lm
        parse_design = temp_pd_func
        load_pareto_front = temp_lpf
        print("All engines loaded successfully!\n")
        
    return True


def design_to_thin_film_stack_inputs(row: "pd.Series", materials: Dict) -> Dict:
    """Convert one Pareto front row/design into inputs for thin_film_stack."""
    dOpt, material_indices = parse_design(row)
    
    # Filter active layers
    active_mask = (material_indices != 0) & (dOpt > 0)
    active_dOpt = dOpt[active_mask]
    active_materialLayer = material_indices[active_mask]
    
    # Reverse layers so they are in air-to-substrate order (which physical solvers expect)
    active_dOpt = active_dOpt[::-1]
    active_materialLayer = active_materialLayer[::-1]
    
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
    if 1 not in materialParams:
        materialParams[1] = {'name': 'SiO2', 'n': 1.45, 'k': 0.0}
        
    mapped_layer = np.array([999 if m == 0 else m for m in active_materialLayer])
    n_input = np.array([materialParams[m]['n'] for m in mapped_layer])
    
    return {
        "dOpt": active_dOpt,
        "n_input": n_input,
        "materialLayer": mapped_layer,
        "materialParams": materialParams,
        "lambda_nm": 1064.0,
    }


def export_design(row, rank: int, materials: dict, directory: Path, export_base_path: str = None) -> Tuple[Path, Path]:
    """Export the given design row to a Python script and a CSV file.

    Args:
        row: pd.Series representing the design row
        rank: 1-based rank of the design
        materials: dictionary of material properties
        directory: experiment directory Path to save files into by default
        export_base_path: optional custom file path/prefix

    Returns:
        Tuple of (py_path, csv_path)
    """
    import csv
    from datetime import datetime
    import numpy as np

    # 1. Parse thicknesses and materials
    dOpt, material_indices = parse_design(row)

    # Filter active layers (non-zero optical thickness and non-air material)
    active_mask = (material_indices != 0) & (dOpt > 0)
    active_dOpt = dOpt[active_mask]
    active_materialLayer = material_indices[active_mask]

    # Reverse layers so they are exported in air-to-substrate order (which standard TMM models expect)
    active_dOpt = active_dOpt[::-1]
    active_materialLayer = active_materialLayer[::-1]

    # Map layers to Air (999) if 0
    mapped_layer = np.array([999 if m == 0 else m for m in active_materialLayer])

    # 2. Build materialParams structure
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

    # 3. Calculate physical thicknesses
    lambda_nm = 1064.0
    d_physical_nm = []

    # Use thin_film_stack if available, or fallback
    n_input = np.array([materialParams[m]['n'] for m in mapped_layer])
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
            d_physical_nm = d_physical_m * 1e9
        except Exception:
            pass

    if len(d_physical_nm) == 0:
        # Fallback to direct Python calculation
        for i in range(len(active_dOpt)):
            mat_idx = mapped_layer[i]
            n_layer = materialParams.get(mat_idx, {}).get("n", 1.45)
            t_nm = active_dOpt[i] * lambda_nm / n_layer
            d_physical_nm.append(t_nm)
        d_physical_nm = np.array(d_physical_nm)

    # 4. Format files and paths
    if export_base_path:
        base_path = Path(export_base_path)
    else:
        base_path = directory / f"rank_{rank}_design"

    py_path = base_path.with_suffix(".py")
    csv_path = base_path.with_suffix(".csv")

    # Format list strings
    dOpt_list_str = ",\n".join([f"    {val:.6f}" for val in active_dOpt])
    materialLayer_list_str = ",\n".join([f"    {val}" for val in mapped_layer])
    d_physical_list_str = ",\n".join([f"    {val:.6f}" for val in d_physical_nm])

    params_lines = []
    for k, v in sorted(materialParams.items()):
        params_lines.append(f"    {k}: {repr(v)}")
    materialParams_dict_str = ",\n".join(params_lines)

    # Get diagnostics values
    reflectivity = row.get('reflectivity', 0.0)
    absorption_ppm = row.get('absorption', 0.0)
    thermal_noise = row.get('thermal_noise', 0.0)

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    py_content = f'''# ==============================================================================
# Rank {rank} Coating Design - Exported from coatopt
# Date: {current_date}
# Reflectivity: {reflectivity:.6f}
# Absorption: {absorption_ppm:.3f} ppm
# Thermal Noise: {thermal_noise:.4e} m/sqrt(Hz)
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

# --- aLIGO Params Structure ---
aLIGO_params = {{}}

## INPUTS 
aLIGO_params['StackName']      = 'Rank {rank} Design'               # Label for run 
aLIGO_params["dOpt"]           = dOpt                               # optical thickness array 
aLIGO_params["materialLayer"]  = materialLayer                      # material array containing keys which index materialParams
aLIGO_params["materialParams"] = materialParams                     # dictionary of material properties 
aLIGO_params["materialSub"]    = 1                                  # substrate type - Silica 
aLIGO_params["lambda_"]        = {lambda_nm:.1f}                    # IFO wavelength (nm)
aLIGO_params["f"]              = np.logspace(1, 3, 100)             # Frequency range to evaluate CTN 
aLIGO_params["wBeam"]          = 0.062                              # laser beam size on ETM (m)
aLIGO_params["Temp"]           = 293.0                              # detector temperature (K)
aLIGO_params["plots "]         = False                              # boolean for activating plots 
aLIGO_params["t_air"]          = 500                                # thickness of air in EFI calculations for optical absorption : Default is 500nm
aLIGO_params["polarisation"]   = 'p'                                # light polarisation for EFI calculations 
aLIGO_params["lambda_list"]    = np.linspace(0, aLIGO_params["lambda_"]*1.5, 10000)

# --- Design Table ---
# Layer | Material Name | Refractive Index | dOpt | Physical Thickness (nm)
'''
    # Add design table as comments
    for i, (thick, mat) in enumerate(zip(d_physical_nm, mapped_layer)):
        name = materialParams.get(mat, {}).get("name", f"Material {mat}")
        n_val = materialParams.get(mat, {}).get("n", 1.0)
        py_content += f"# {i+1:<5} | {name:<13} | {n_val:<16} | {active_dOpt[i]:.6f} | {thick:.2f} nm\n"

    py_content += f'\nprint("Rank {rank} design variables loaded successfully.")\n'

    # 5. Write Python file
    with open(py_path, "w") as f:
        f.write(py_content)

    # 6. Write CSV file
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Material_Index", "Material_Name", "Refractive_Index", "dOpt", "Physical_Thickness_nm"])
        for i, (thick, mat) in enumerate(zip(d_physical_nm, mapped_layer)):
            name = materialParams.get(mat, {}).get("name", f"Material {mat}")
            n_val = materialParams.get(mat, {}).get("n", 1.0)
            writer.writerow([i + 1, mat, name, n_val, active_dOpt[i], thick])

    return py_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Run interactive Matplotlib Pareto and Coating Inspection Dashboard",
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing config.ini and pareto_front_values.csv",
    )
    parser.add_argument(
        "--export-rank",
        type=int,
        default=None,
        help="Export a specific design by rank (1-based) to Python/CSV files and exit.",
    )
    parser.add_argument(
        "--export-file",
        type=str,
        default=None,
        help="Custom base filename/path for export (without extension).",
    )
    parser.add_argument(
        "--compare-refl",
        type=float,
        default=None,
        help="Reflectivity of your custom design (e.g. 0.95533) to plot as a reference comparison.",
    )
    parser.add_argument(
        "--compare-abs",
        type=float,
        default=None,
        help="Absorption in ppm of your custom design (e.g. 0.59) to plot as a reference comparison.",
    )
    parser.add_argument(
        "--compare-tn",
        type=float,
        default=None,
        help="Thermal noise (CTN) at 100Hz of your custom design (e.g. 3.4e-21) to plot as a reference comparison.",
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default="Your Design",
        help="Custom label for your reference point on the plots.",
    )
    args = parser.parse_args()

    # Convert to Path object
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
            if candidate1.exists():
                materials_path = candidate1
            elif candidate2.exists():
                materials_path = candidate2
            else:
                print(f"Error: Could not find materials file at {candidate1}")
                return 1
        else:
            materials_path = Path(materials_path)
    except (configparser.NoSectionError, configparser.NoOptionError):
        print("Error: Could not find 'materials_path' in config.ini")
        return 1

    # Dynamic dynamic import loading bar!
    if not load_dependencies():
        return 1

    print("Loading Pareto front...")
    try:
        designs_df, values_df, _ = load_pareto_front(directory)
        print(f"  Found {len(designs_df)} designs")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loading materials from {materials_path}...")
    materials = load_materials(str(materials_path))
    print("  Loaded materials successfully.")

    # Combine dataframes and sort by reflectivity descending
    combined_df = pd.concat([designs_df, values_df], axis=1)
    sort_col = "reflectivity" if "reflectivity" in combined_df.columns else combined_df.columns[0]
    combined_df = combined_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    n_designs = len(combined_df)

    # Headless CLI exporter path
    if args.export_rank is not None:
        rank = args.export_rank
        if rank < 1 or rank > n_designs:
            print(f"Error: Requested rank {rank} is out of bounds (1 to {n_designs})")
            return 1

        row = combined_df.iloc[rank - 1]

        # Load dependencies so thin_film_stack works if available
        load_dependencies()

        try:
            py_path, csv_path = export_design(row, rank, materials, directory, args.export_file)
            print(f"\n[Export] ✓ Successfully exported Rank {rank} Design!")
            print(f"[Export] Python script: file://{py_path}")
            print(f"[Export] CSV columns:   file://{csv_path}")
            return 0
        except Exception as e:
            print(f"Error: Failed to export design: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Calculate active layer counts and template max_layers (highly optimized: takes 0.07s)
    active_counts = []
    max_layers = 50
    for idx, row in combined_df.iterrows():
        dOpt, mat_idx = parse_design(row)
        active_mask = (mat_idx != 0) & (dOpt > 0)
        active_counts.append(int(np.sum(active_mask)))
        max_layers = len(dOpt)
    combined_df['active_layer_count'] = active_counts
    combined_df['max_layers'] = max_layers
    print(f"  Processed {n_designs} Pareto front designs.")

    # Setup plotting style for a premium, clean look
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # ── 1. Create a Two-Column Figure Layout ──────────────────────────────────
    fig = plt.figure(figsize=(18, 9.5))
    fig.canvas.manager.set_window_title('Coating Inspection Dashboard')

    outer = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.15, 1.0],
        wspace=0.18
    )
    print("  Created dashboard layout structure.")

    # ── 2. Build the Left-Hand Pareto Plot Region ─────────────────────────────
    left = outer[0].subgridspec(
        nrows=2,
        ncols=2,
        hspace=0.32,
        wspace=0.32
    )

    ax_r_abs = fig.add_subplot(left[0, 0])
    ax_r_tn = fig.add_subplot(left[1, 0])
    ax_abs_tn = fig.add_subplot(left[1, 1])
    ax_info = fig.add_subplot(left[0, 1])  # Use top-right of left side for design info/summary card

    # Make the design info axes look like a clean text card
    ax_info.axis('off')
    info_bg = ax_info.patch
    info_bg.set_facecolor('#fafafa')
    info_bg.set_edgecolor('lightgray')

    # ── 3. Build the Right-Hand Coating Diagnostic Region ────────────────────
    right = outer[1].subgridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.0, 1.0, 1.3],
        hspace=0.36
    )

    ax_stack = fig.add_subplot(right[0])
    ax_field = fig.add_subplot(right[1])
    ax_spectrum = fig.add_subplot(right[2])

    # Store handles to highlight markers so they can be updated on selection
    highlight_markers = []
    current_state = {"selected_index": 0}

    # ── 4. Pareto Plotting Function ──────────────────────────────────────────
    def plot_pareto_front_panels():
        # X and Y values
        r_loss_val = 1.0 - combined_df['reflectivity'].values
        abs_ppm = combined_df['absorption'].values
        tn = combined_df['thermal_noise'].values
        active_c = combined_df['active_layer_count'].values
        
        # Calculate dynamic threshold based on 80% of max layers
        max_l = int(combined_df['max_layers'].values[0])
        thresh = int(0.8 * max_l)
        
        mask_optimal = active_c >= thresh
        mask_under = active_c < thresh

        # Axis 1: 1 - Reflectivity vs Absorption
        # Plot optimal designs (solid circles)
        sc1 = ax_r_abs.scatter(
            r_loss_val[mask_optimal], abs_ppm[mask_optimal], 
            c=tn[mask_optimal], cmap='viridis', vmin=min(tn), vmax=max(tn),
            s=55, edgecolor='black', marker='o', alpha=0.85, zorder=3, label=f'Optimal (>= {thresh} layers)'
        )
        # Plot under-layered designs (faded crosses)
        ax_r_abs.scatter(
            r_loss_val[mask_under], abs_ppm[mask_under], 
            c=tn[mask_under], cmap='viridis', vmin=min(tn), vmax=max(tn),
            s=45, marker='x', alpha=0.3, zorder=2, label=f'Under-layered (< {thresh} layers)'
        )
        ax_r_abs.set_xlabel('1 - Reflectivity (Loss)')
        ax_r_abs.set_ylabel('Absorption (ppm)')
        ax_r_abs.set_title('Reflectivity vs Absorption', fontweight='bold', fontsize=11)
        ax_r_abs.set_xscale('log')
        ax_r_abs.set_yscale('log')
        
        # Plot custom reference comparison point if provided
        if args.compare_refl is not None and args.compare_abs is not None:
            ax_r_abs.scatter(
                1.0 - args.compare_refl, args.compare_abs,
                color='#ff007f', edgecolor='black', s=200, marker='*', zorder=10,
                label=args.compare_label
            )
            
        ax_r_abs.legend(loc='lower left', frameon=True, fontsize=8)

        # Add colorbar for thermal noise
        cbar = fig.colorbar(sc1, ax=ax_r_abs, pad=0.02, shrink=0.8)
        cbar.set_label('Thermal Noise', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # Axis 2: 1 - Reflectivity vs Thermal Noise
        sc2 = ax_r_tn.scatter(
            r_loss_val[mask_optimal], tn[mask_optimal], 
            c=abs_ppm[mask_optimal], cmap='magma', vmin=min(abs_ppm), vmax=max(abs_ppm),
            s=55, edgecolor='black', marker='o', alpha=0.85, zorder=3, label=f'Optimal (>= {thresh} layers)'
        )
        ax_r_tn.scatter(
            r_loss_val[mask_under], tn[mask_under], 
            c=abs_ppm[mask_under], cmap='magma', vmin=min(abs_ppm), vmax=max(abs_ppm),
            s=45, marker='x', alpha=0.3, zorder=2, label=f'Under-layered (< {thresh} layers)'
        )
        ax_r_tn.set_xlabel('1 - Reflectivity (Loss)')
        ax_r_tn.set_ylabel('Thermal Noise (m/√Hz)')
        ax_r_tn.set_title('Reflectivity vs Thermal Noise', fontweight='bold', fontsize=11)
        ax_r_tn.set_xscale('log')
        
        # Plot custom reference comparison point if provided
        if args.compare_refl is not None and args.compare_tn is not None:
            ax_r_tn.scatter(
                1.0 - args.compare_refl, args.compare_tn,
                color='#ff007f', edgecolor='black', s=200, marker='*', zorder=10,
                label=args.compare_label
            )
            
        ax_r_tn.legend(loc='lower left', frameon=True, fontsize=8)

        # Add colorbar for absorption
        cbar2 = fig.colorbar(sc2, ax=ax_r_tn, pad=0.02, shrink=0.8)
        cbar2.set_label('Absorption (ppm)', fontsize=9)
        cbar2.ax.tick_params(labelsize=8)

        # Axis 3: Absorption vs Thermal Noise
        sc3 = ax_abs_tn.scatter(
            abs_ppm[mask_optimal], tn[mask_optimal], 
            c=r_loss_val[mask_optimal], cmap='plasma', vmin=min(r_loss_val), vmax=max(r_loss_val),
            s=55, edgecolor='black', marker='o', alpha=0.85, zorder=3, label=f'Optimal (>= {thresh} layers)'
        )
        ax_abs_tn.scatter(
            abs_ppm[mask_under], tn[mask_under], 
            c=r_loss_val[mask_under], cmap='plasma', vmin=min(r_loss_val), vmax=max(r_loss_val),
            s=45, marker='x', alpha=0.3, zorder=2, label=f'Under-layered (< {thresh} layers)'
        )
        ax_abs_tn.set_xlabel('Absorption (ppm)')
        ax_abs_tn.set_ylabel('Thermal Noise (m/√Hz)')
        ax_abs_tn.set_title('Absorption vs Thermal Noise', fontweight='bold', fontsize=11)
        ax_abs_tn.set_xscale('log')
        
        # Plot custom reference comparison point if provided
        if args.compare_abs is not None and args.compare_tn is not None:
            ax_abs_tn.scatter(
                args.compare_abs, args.compare_tn,
                color='#ff007f', edgecolor='black', s=200, marker='*', zorder=10,
                label=args.compare_label
            )
            
        ax_abs_tn.legend(loc='upper right', frameon=True, fontsize=8)

        # Add colorbar for reflectivity loss
        cbar3 = fig.colorbar(sc3, ax=ax_abs_tn, pad=0.02, shrink=0.8)
        cbar3.set_label('1 - Reflectivity', fontsize=9)
        cbar3.ax.tick_params(labelsize=8)

        # Create red highlight markers on each axes (initially at design 0)
        h1, = ax_r_abs.plot([r_loss_val[0]], [abs_ppm[0]], 'ro', ms=11, mec='black', mew=1.5, label='Selected', zorder=5)
        h2, = ax_r_tn.plot([r_loss_val[0]], [tn[0]], 'ro', ms=11, mec='black', mew=1.5, zorder=5)
        h3, = ax_abs_tn.plot([abs_ppm[0]], [tn[0]], 'ro', ms=11, mew=1.5, zorder=5)
        
        highlight_markers.extend([h1, h2, h3])

    # ── 5. Coating Plotting Functions ────────────────────────────────────────
    def update_selected_design(design_index: int):
        """Update the coating diagnostic plots for the selected Pareto design."""
        current_state["selected_index"] = design_index
        row = combined_df.iloc[design_index]
        
        # 1. Update Highlight Markers
        r_loss = 1.0 - row['reflectivity']
        abs_ppm = row['absorption']
        tn = row['thermal_noise']

        highlight_markers[0].set_data([r_loss], [abs_ppm])
        highlight_markers[1].set_data([r_loss], [tn])
        highlight_markers[2].set_data([abs_ppm], [tn])

        # 3. Clear Coating Diagnostic axes
        ax_stack.clear()
        ax_field.clear()
        ax_spectrum.clear()

        # 4. Generate Thin Film inputs
        stack_inputs = design_to_thin_film_stack_inputs(row, materials)
        active_dOpt = stack_inputs["dOpt"]
        n_input = stack_inputs["n_input"]
        mapped_layer = stack_inputs["materialLayer"]
        materialParams = stack_inputs["materialParams"]
        lambda_nm = stack_inputs["lambda_nm"]

        d_physical_nm = []
        info_text = ""

        # Call thin_film_stack with verbose=True to capture the custom formatted library printout
        if thin_film_stack is not None:
            try:
                import io
                import contextlib

                absor_val = np.round(row['absorption'], decimals=2)
                reflectivity_lambda_0 = f"{row['reflectivity']:.5f}"
                transmission_lambda_0 = 1.0 - row['reflectivity'] - row['absorption'] * 1e-6
                transmission_1064 = f"{transmission_lambda_0 * 1e6:.5f} ppm"
                stack_name = f"Rank {design_index + 1} Design"

                f_output = io.StringIO()
                with contextlib.redirect_stdout(f_output):
                    _, _, d_physical_m = thin_film_stack(
                        dOpt=active_dOpt,
                        n_input=n_input,
                        materialLayer=mapped_layer,
                        materialParams=materialParams,
                        lambda_=lambda_nm,
                        base_path=str(directory),
                        plots=False,
                        verbose=True,
                        absorption=f"{absor_val:.2f} ppm",
                        CTN_at_100Hz=tn,
                        Reflectivity_1064=reflectivity_lambda_0,
                        Transmission_1064=transmission_1064,
                        stack_name=stack_name
                    )
                d_physical_nm = d_physical_m * 1e9
                info_text = f_output.getvalue().strip().replace('\t', '    ')
            except Exception as e:
                info_text = f"Warning: thin_film_stack failed: {e}\n"

        # Fallback thickness calculation if library fails or is absent
        if len(d_physical_nm) == 0:
            for i in range(len(active_dOpt)):
                mat_idx = mapped_layer[i]
                n_layer = materialParams.get(mat_idx, {}).get("n", 1.45)
                t_nm = active_dOpt[i] * lambda_nm / n_layer
                d_physical_nm.append(t_nm)
            d_physical_nm = np.array(d_physical_nm)
            
            info_text += (
                f"  SELECTED DESIGN SUMMARY (FALLBACK)\n"
                f"  -------------------------\n"
                f"  Design Rank: #{design_index + 1} / {n_designs}\n"
                f"  Reflectivity: {row['reflectivity']:.6f}\n"
                f"  Loss (1 - R): {r_loss:.4e}\n"
                f"  Absorption: {abs_ppm:.3f} ppm\n"
                f"  Thermal Noise: {tn:.4e} m/√Hz\n"
                f"  Active Layers: {int(row['active_layer_count'])} / {int(row['max_layers'])}"
            )

        # 2. Update Design Info Summary Card
        ax_info.clear()
        ax_info.axis('off')
        ax_info.text(0.02, 0.98, info_text, transform=ax_info.transAxes, fontsize=8.0, 
                     family='monospace', va='top', ha='left',
                     bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="lightgray"))


        # Plot Stack Diagram
        unique_mats = sorted(list(set(mapped_layer)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_mats) + 1))
        mat_colors = {mat: colors[i] for i, mat in enumerate(unique_mats)}
        
        # Override default colors with nice customized ones
        custom_colors = {
            "air": "#F0F0F0",
            "SiO2": "#1f77b4",
            "Ti:Ta2O5": "#c837ab",
            "TiGermania": "#c837ab",
            "substrate": "#7f7f7f",
        }
        for mat in unique_mats:
            name = materialParams.get(mat, {}).get("name", "Unknown")
            if name in custom_colors:
                mat_colors[mat] = custom_colors[name]

        # Draw vertical stacked bar representation of the coating
        depth_so_far = 0.0
        legend_patches = []
        shown_names = set()
        
        for i, (thick, mat) in enumerate(zip(d_physical_nm, mapped_layer)):
            name = materialParams.get(mat, {}).get("name", f"Material {mat}")
            color = mat_colors[mat]
            ax_stack.bar(depth_so_far + thick / 2.0, thick, width=thick, color=color, edgecolor='black', linewidth=0.5)
            depth_so_far += thick
            
            if name not in shown_names:
                legend_patches.append(Patch(facecolor=color, edgecolor='black', label=name))
                shown_names.add(name)

        # Draw substrate block
        sub_width = 150.0
        ax_stack.bar(depth_so_far + sub_width / 2.0, 120, width=sub_width, color='#7f7f7f', edgecolor='black', linewidth=0.5)
        if "Substrate" not in shown_names:
            legend_patches.append(Patch(facecolor='#7f7f7f', edgecolor='black', label='Substrate'))

        ax_stack.set_xlim([0, depth_so_far + sub_width])
        ax_stack.set_ylim([0, max(d_physical_nm) * 1.15 if len(d_physical_nm) > 0 else 250])
        ax_stack.set_ylabel("Physical Thickness [nm]", fontsize=9)
        ax_stack.set_xlabel("Coating Depth Position [nm]", fontsize=9)
        ax_stack.set_title("Coating Stack Diagram", fontweight='bold', fontsize=11)
        ax_stack.legend(handles=legend_patches, loc="upper right", frameon=True, fontsize=8)
        ax_stack.grid(False)

        # Plot Electric Field Profile
        try:
            _, _, ds, E, _, _, _ = CalculateEFI_tmm(
                dOpt=active_dOpt,
                materialLayer=mapped_layer,
                materialParams=materialParams,
                lambda_=lambda_nm,  # Pass in nanometers
                plots=False,
            )
            ax_field.plot(ds, E, color='blue', linewidth=1.8, label='Electric Field Intensity')
            ax_field.set_xlabel("Depth (nm)", fontsize=9)
            ax_field.set_ylabel("Electric Field Intensity", fontsize=9)
            ax_field.set_title("Electric Field Profile", fontweight='bold', fontsize=11)
            ax_field.set_ylim([0, max(E) * 1.15 if len(E) > 0 else 4.0])

            # Draw vertical dashed interface lines
            accumulated = 0.0
            ax_field.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)
            for thick in d_physical_nm:
                accumulated += thick
                ax_field.axvline(x=accumulated, color='gray', linestyle='--', alpha=0.5)

            # Arrow indicating light propagation inside air boundary
            ax_field.annotate("Light Propagation", xy=(-50, max(E)*0.8 if len(E)>0 else 2), 
                              xytext=(-400, max(E)*0.8 if len(E)>0 else 2),
                              arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                              fontsize=8, va='center')
        except Exception as e:
            print(f"Warning: Could not calculate electric field profile: {e}")
            ax_field.text(0.5, 0.5, "Error calculating EFI profile", ha='center', va='center')

        # Plot Simulated Spectral Response
        try:
            lambda_list = np.linspace(400.0, 1400.0, 200)  # Sweep wavelengths in nanometers
            wavelengths, transmission, _ = CalculateTransmission_tmm(
                dOpt=active_dOpt,
                materialLayer=mapped_layer,
                materialParams=materialParams,
                lambda_list=lambda_list,
                lambda_0=lambda_nm,  # Pass in nanometers
                plots=False,
            )
            ax_spectrum.plot(wavelengths, transmission * 100, color='#ff7f0e', linewidth=2.0)
            ax_spectrum.set_xlabel("Wavelength (nm)", fontsize=9)
            ax_spectrum.set_ylabel("Transmission (%)", fontsize=9)
            ax_spectrum.set_title("Simulated Spectral Response", fontweight='bold', fontsize=11)
            ax_spectrum.set_xlim([400, 1400])
            ax_spectrum.set_ylim([0, 105])
            ax_spectrum.axvline(x=lambda_nm, color='red', linestyle=':', alpha=0.8, label=f'{int(lambda_nm)}nm reference')
            ax_spectrum.legend(loc="upper right", frameon=True, fontsize=8)
        except Exception as e:
            print(f"Warning: Could not calculate spectral response: {e}")
            ax_spectrum.text(0.5, 0.5, "Error simulating spectrum", ha='center', va='center')

        fig.canvas.draw_idle()

    # ── 6. Connect Mouse Click Event ─────────────────────────────────────────
    def on_click(event):
        if event.inaxes not in [ax_r_abs, ax_r_tn, ax_abs_tn]:
            return

        x_click = event.xdata
        y_click = event.ydata
        ax = event.inaxes

        # Normalized coordinates distance matching
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        
        # Account for log scales
        is_log_x = ax.get_xscale() == 'log'
        is_log_y = ax.get_yscale() == 'log'

        x_click_val = np.log10(x_click) if is_log_x else x_click
        y_click_val = np.log10(y_click) if is_log_y else y_click
        
        x_range = np.log10(x_lim[1] / x_lim[0]) if is_log_x else (x_lim[1] - x_lim[0])
        y_range = np.log10(y_lim[1] / y_lim[0]) if is_log_y else (y_lim[1] - y_lim[0])

        best_idx = None
        min_dist = float('inf')

        for idx, row in combined_df.iterrows():
            r_loss = 1.0 - row['reflectivity']
            abs_ppm = row['absorption']
            tn = row['thermal_noise']

            # Find coordinates based on clicked axes
            if ax == ax_r_abs:
                x_val = r_loss
                y_val = abs_ppm
            elif ax == ax_r_tn:
                x_val = r_loss
                y_val = tn
            elif ax == ax_abs_tn:
                x_val = abs_ppm
                y_val = tn
            else:
                continue

            x_pt = np.log10(x_val) if is_log_x else x_val
            y_pt = np.log10(y_val) if is_log_y else y_val

            dx = (x_pt - x_click_val) / x_range
            dy = (y_pt - y_click_val) / y_range
            dist = dx*dx + dy*dy

            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        # Click matching threshold
        if best_idx is not None and min_dist < 0.08:
            update_selected_design(best_idx)

    def on_key(event):
        if event.key == 'e':
            idx = current_state["selected_index"]
            rank = idx + 1
            row = combined_df.iloc[idx]

            try:
                py_path, csv_path = export_design(row, rank, materials, directory)
                print(f"\n[Export] ✓ Successfully exported Rank {rank} Design!")
                print(f"[Export] Python script: file://{py_path}")
                print(f"[Export] CSV columns:   file://{csv_path}")

                if ax_info.texts:
                    current_text = ax_info.texts[0].get_text()
                    banner = (
                        f"================================================\n"
                        f"  ✓ EXPORTED RANK {rank} DESIGN TO FILES!\n"
                        f"  - rank_{rank}_design.py\n"
                        f"  - rank_{rank}_design.csv\n"
                        f"================================================\n\n"
                    )
                    if "EXPORTED RANK" not in current_text:
                        ax_info.clear()
                        ax_info.axis('off')
                        ax_info.text(0.02, 0.98, banner + current_text, transform=ax_info.transAxes, fontsize=8.0, 
                                     family='monospace', va='top', ha='left',
                                     bbox=dict(boxstyle="round,pad=0.4", fc="#e6f4ea", ec="#137333"))
                        fig.canvas.draw_idle()
            except Exception as ex:
                print(f"Error: Failed to export design: {ex}")

    # Plot initial states
    print("  Plotting initial Pareto front panels...")
    plot_pareto_front_panels()
    print("  Initial Pareto front panels plotted successfully.")
    
    # ── 7. Render Initial Rank 1 Design at startup ──
    # If the initial calculation fails, we exit immediately to prevent opening a blank viewer.
    print("  Pre-rendering Rank 1 coating design using physical solvers...")
    try:
        update_selected_design(0)
        print("  Rank 1 coating design successfully pre-rendered!")
    except Exception as e:
        print(f"Error: Initial Rank 1 design calculation failed: {e}")
        return 1

    # Bind the click callback event
    cid = fig.canvas.mpl_connect("button_press_event", on_click)

    # Bind the key press event for exporting
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)
    print("\n  [Interactive] ✓ Press the 'e' key on your keyboard with the dashboard active to export the selected design!")

    # Title of the interactive dashboard
    fig.suptitle('Interactive Pareto Front & Coating Inspection Dashboard', fontsize=14, fontweight='bold', y=0.97)
    # Using specific subplots adjustments to prevent compatibility warnings with subgridspec
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.96)
    print("  Opening Coating Inspection Dashboard window (blocking call)...")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
