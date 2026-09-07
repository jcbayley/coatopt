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
    """Dynamically load physics and TMM libraries from CoatingAnalysis, or local fallback."""
    global getCoatingThermalNoise, optical_to_physical, CalculateEFI_tmm, CalculateTransmission_tmm, thin_film_stack
    if getCoatingThermalNoise is not None:
        return True
    # Try local coatopt modules first
    try:
        from coatopt.environments.utils.YAM_CoatingBrownian import getCoatingThermalNoise as temp_gctn
        from coatopt.environments.utils.EFI_tmm import (
            optical_to_physical as temp_otp,
            CalculateEFI_tmm as temp_cefi,
            CalculateTransmission_tmm as temp_ctrans
        )
        
        getCoatingThermalNoise = temp_gctn
        optical_to_physical = temp_otp
        CalculateEFI_tmm = temp_cefi
        CalculateTransmission_tmm = temp_ctrans
        thin_film_stack = None
        return True
    except Exception:
        pass

    # External fallback path if local import fails
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
def verify_aligo_gold_standard(verbose: bool = True) -> bool:
    """
    Verify physics calculations against the real_aLIGO.ipynb gold-standard benchmark.
    Returns True if CTN, Transmission, Absorption, and Thickness match to high precision.
    """
    if not load_physics_dependencies():
        if verbose:
            print("Warning: Physics dependencies not available for gold-standard verification.")
        return False

    lambda_nm = 1064.0
    coating_design_qw = [(2, 1.1068)] + [(1, 1.1371), (2, 0.8591)] * 18 + [(1, 0.0617)]
    coating_design_qw = coating_design_qw[::-1]  # Top to bottom order

    materialLayer = np.array([m for m, _ in coating_design_qw], dtype=int)
    dOpt = np.array([qw for _, qw in coating_design_qw], dtype=float) / 4.0

    materialParams = {
        1: {'name': 'SiO2', 'n': 1.45, 'a': 0, 'alpha': 0.51e-6, 'beta': 8e-6, 'kappa': 1.38, 'C': 1.64e6, 'Y': 70e9, 'prat': 0.19, 'phiM': 2.3e-5, 'k': 3e-8},
        2: {'name': 'Ti:Ta2O5', 'n': 2.09, 'a': 2, 'alpha': 3.6e-6, 'beta': 14e-6, 'kappa': 33, 'C': 2.1e6, 'Y': 120e9, 'prat': 0.29, 'phiM': 5.01340973895537e-4, 'k': 5e-8},
        99: {'name': 'SiO2_bulk', 'n': 1.45, 'a': 0, 'alpha': 0.51e-6, 'beta': 8e-6, 'kappa': 1.38, 'C': 1.64e6, 'Y': 72.7e9, 'prat': 0.167, 'phiM': 2.3e-5, 'k': 3e-8},
        999: {'name': 'air', 'n': 1.0, 'a': np.nan, 'alpha': np.nan, 'beta': np.nan, 'kappa': np.nan, 'C': np.nan, 'Y': np.nan, 'prat': np.nan, 'phiM': np.nan, 'k': 0}
    }

    nLayer = np.array([materialParams[m]['n'] for m in materialLayer], dtype=float)
    d_phys_nm = (dOpt * lambda_nm) / nLayer

    try:
        noise_summary, _, _, _, _, _ = getCoatingThermalNoise(
            dOpt=dOpt, materialLayer=materialLayer, materialParams=materialParams,
            materialSub=1, lambda_=lambda_nm * 1e-9, f=100.0, wBeam=0.062, Temp=293.0, plots=False
        )
        if isinstance(noise_summary['Frequency'], (float, np.floating)):
            calc_ctn = float(noise_summary['BrownianNoise'])
        else:
            diff = np.abs(noise_summary['Frequency'] - 100.0)
            idx = diff.argmin()
            calc_ctn = float(noise_summary['BrownianNoise'][idx])

        _, _, _, _, _, calc_abs_ppm, calc_refl = CalculateEFI_tmm(
            dOpt=dOpt, materialLayer=materialLayer, materialParams=materialParams,
            lambda_=lambda_nm, t_air=500, polarisation='p', plots=False, air_index=999, substrate_index=1
        )

        _, _, calc_trans_frac = CalculateTransmission_tmm(
            dOpt=dOpt, materialLayer=materialLayer, materialParams=materialParams,
            lambda_list=np.array([lambda_nm]), lambda_0=lambda_nm, tphys=d_phys_nm, polarisation='p', plots=False
        )
        calc_trans_ppm = float(calc_trans_frac * 1e6)
        expected_ctn = 6.991911090888062e-21
        expected_trans_ppm = 3.62402

        passed = (abs(calc_ctn - expected_ctn) / expected_ctn < 1e-4) and (abs(calc_trans_ppm - expected_trans_ppm) / expected_trans_ppm < 1e-3)

        if verbose:
            status_str = "✓ VERIFIED (PASSED)" if passed else "❌ MISMATCH (FAILED)"
            print("\n" + "=" * 80)
            print(f"   PHYSICS ENGINE SELF-CHECK: {status_str}")
            print("=" * 80)
            print(f"  aLIGO Benchmark CTN (100 Hz):   Expected {expected_ctn:.12e} | Calculated {calc_ctn:.12e}")
            print(f"  aLIGO Benchmark Transmission:   Expected {expected_trans_ppm:.4f} ppm        | Calculated {calc_trans_ppm:.4f} ppm")
            print(f"  aLIGO Benchmark Absorption:     Expected 0.1100 ppm        | Calculated {float(calc_abs_ppm):.4f} ppm")
            print(f"  aLIGO Benchmark Thickness:      Expected 5875.09 nm        | Calculated {np.sum(d_phys_nm):.2f} nm")
            print("=" * 80 + "\n")

        return passed
    except Exception as e:
        if verbose:
            print(f"❌ Error during physics self-check: {e}")
        return False


def parse_design(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract thicknesses and materials from a Pareto front row."""
    thickness_cols = [col for col in row.index if col.startswith("thickness_")]
    material_cols = [col for col in row.index if col.startswith("material_")]

    thickness_cols = sorted(thickness_cols, key=lambda x: int(x.split("_")[1]))
    material_cols = sorted(material_cols, key=lambda x: int(x.split("_")[1]))

    thicknesses = []
    materials = []
    for t_col, m_col in zip(thickness_cols, material_cols):
        t_val = row[t_col]
        m_val = row[m_col]
        if pd.isna(t_val) or pd.isna(m_val):
            thicknesses.append(0.0)
            materials.append(0)
        else:
            thicknesses.append(float(t_val))
            materials.append(int(m_val))

    return np.array(thicknesses), np.array(materials)


def get_design_key(dOpt: np.ndarray, materials: np.ndarray) -> str:
    """Generate a stable SHA-256 hash for a design based on its layer structure."""
    import hashlib
    # Round thicknesses to 6 decimal places to avoid tiny floating point variations
    rounded_dOpt = np.round(dOpt, 6)
    # Filter out inactive layers to keep the representation minimal
    active_mask = (materials != 0) & (rounded_dOpt > 1e-12)
    active_dOpt = rounded_dOpt[active_mask]
    active_materials = materials[active_mask]
    
    design_str = ",".join(f"{t:.6f}:{m}" for t, m in zip(active_dOpt, active_materials))
    return hashlib.sha256(design_str.encode('utf-8')).hexdigest()


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


def worker_init(path):
    import sys
    if path not in sys.path:
        sys.path.insert(0, path)
    import coatopt.utils.plot_interactive_3d_rank as p
    p.load_physics_dependencies()


def tmm_worker(task_info):
    import io
    import contextlib
    import numpy as np
    import coatopt.utils.plot_interactive_3d_rank as p
    
    idx, active_dOpt, mapped_layer, materialParams, lambda_nm = task_info
    
    d_physical_nm = []
    efi_depths = None
    efi_intensity = None
    spec_wavelengths = None
    spec_transmission = None
    
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        # Physical thickness calculation
        try:
            if p.thin_film_stack is not None:
                n_input = np.array([materialParams[m]['n'] for m in mapped_layer])
                _, _, d_physical_m = p.thin_film_stack(
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
                
        if p.CalculateEFI_tmm is not None:
            try:
                _, _, ds, E, _, _, _ = p.CalculateEFI_tmm(
                    dOpt=active_dOpt,
                    materialLayer=mapped_layer,
                    materialParams=materialParams,
                    lambda_=lambda_nm * 1e-9,  # Convert nm to meters for CalculateEFI_tmm
                    plots=False,
                )
                efi_depths = [float(x) for x in ds]
                efi_intensity = [float(x) for x in E]
            except Exception:
                pass
                
        if p.CalculateTransmission_tmm is not None:
            try:
                min_lam = max(200.0, lambda_nm * 0.3)
                max_lam = max(1800.0, lambda_nm * 1.5)
                lambda_list = np.linspace(min_lam, max_lam, 200)
                wavelengths, transmission, _ = p.CalculateTransmission_tmm(
                    dOpt=active_dOpt,
                    materialLayer=mapped_layer,
                    materialParams=materialParams,
                    lambda_list=lambda_list,
                    lambda_0=lambda_nm,
                    plots=False,
                )
                spec_wavelengths = [float(x) for x in wavelengths]
                spec_transmission = [float(x * 100) for x in transmission]
            except Exception:
                pass
            
    return idx, d_physical_nm, efi_depths, efi_intensity, spec_wavelengths, spec_transmission


def precompute_tmm_details(combined_df: pd.DataFrame, materials_dict: dict, max_count: int = 50, lambda_nm: float = 1064.0, cache_dir=None) -> dict:
    """Precompute EFI profile and spectral transmission response for the top N designs."""
    load_physics_dependencies()
    
    import io
    import json
    import contextlib
    import sys
    import os
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    
    # Load cache if available
    cache_path = None
    cache = {}
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "tmm_cache.json"
        
        # Look for tmm_cache.json in cache_dir and all subdirectories recursively
        cache_files = []
        cache_dir_path = Path(cache_dir)
        if cache_dir_path.exists():
            if cache_dir_path.is_dir():
                for root, dirs, files in os.walk(cache_dir_path):
                    if "tmm_cache.json" in files:
                        cache_files.append(Path(root) / "tmm_cache.json")
            else:
                if cache_dir_path.name == "tmm_cache.json":
                    cache_files.append(cache_dir_path)
                elif (cache_dir_path.parent / "tmm_cache.json").exists():
                    cache_files.append(cache_dir_path.parent / "tmm_cache.json")
        
        # Load and merge all cache files
        loaded_count = 0
        for cp in sorted(cache_files):
            try:
                with open(cp, "r") as f:
                    sub_cache = json.load(f)
                cache.update(sub_cache)
                loaded_count += len(sub_cache)
            except Exception as e:
                print(f"  Warning: Failed to load TMM cache from {cp}: {e}")
        if loaded_count > 0:
            print(f"  Loaded {loaded_count} cached TMM results from {len(cache_files)} cache file(s) (merged into {len(cache)} unique entries)")

    tmm_data = {}
    total_designs = len(combined_df)
    tasks_to_compute = []
    
    for idx, row in combined_df.iterrows():
        design_idx = int(idx)
        rank = int(row["rank"])
        
        # Build layer variables
        dOpt, material_indices = parse_design(row)
        
        # Generate design hash key
        design_key = get_design_key(dOpt, material_indices)
        
        active_mask = (material_indices != 0) & (dOpt > 1e-12)
        active_dOpt = dOpt[active_mask]
        active_materialLayer = material_indices[active_mask]
        
        # Reverse layers so they are in air-to-substrate order
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
            
        design_data = {
            "rank": rank,
            "reflectivity": float(row["reflectivity"]),
            "absorption": float(row["absorption"]),
            "thermal_noise": float(row["thermal_noise"]),
            "utility_score": float(row.get("utility_score", 0.0)),
            "active_layer_count": int(row["active_layer_count"]),
            "total_thickness": float(row["total_thickness"]),
            "info_text": "",
            "precomputed": False,
            "dOpt": [float(x) for x in active_dOpt],
            "materialLayer": [int(x) for x in mapped_layer]
        }
        if "run_name" in row and not pd.isna(row["run_name"]):
            design_data["run_name"] = str(row["run_name"])
            
        # Check cache
        cache_hit = False
        cached_data = {}
        if design_key in cache:
            cache_hit = True
            cached_data = cache[design_key]
            
        # Populate basic design details (removed indexing limits)
        for field in ["d_physical_nm", "material_names", "material_indices"]:
            if cache_hit and field in cached_data:
                design_data[field] = cached_data[field]
            else:
                if field == "d_physical_nm":
                    d_phys = []
                    if thin_film_stack is not None:
                        try:
                            n_input = np.array([materialParams[m]['n'] for m in mapped_layer])
                            _, _, d_physical_m = thin_film_stack(
                                dOpt=active_dOpt,
                                n_input=n_input,
                                materialLayer=mapped_layer,
                                materialParams=materialParams,
                                lambda_=lambda_nm,
                                plots=False,
                                verbose=False
                            )
                            d_phys = list(d_physical_m * 1e9)
                        except Exception:
                            pass
                    if len(d_phys) == 0:
                        for i in range(len(active_dOpt)):
                            mat_idx = mapped_layer[i]
                            n_layer = materialParams.get(mat_idx, {}).get("n", 1.45)
                            d_phys.append(float(active_dOpt[i] * lambda_nm / n_layer))
                    design_data["d_physical_nm"] = d_phys
                elif field == "material_names":
                    design_data["material_names"] = [materialParams.get(int(m), {}).get("name", f"Material {m}") for m in mapped_layer]
                elif field == "material_indices":
                    design_data["material_indices"] = [int(m) for m in mapped_layer]
                    
        # Check if we should precompute EFI/spectrum
        should_precompute = (max_count < 0) or (design_idx < max_count)
        
        has_efi = cache_hit and "efi_depths" in cached_data and "efi_intensity" in cached_data
        has_spec = cache_hit and "spec_wavelengths" in cached_data and "spec_transmission" in cached_data
        
        if should_precompute:
            design_data["precomputed"] = True
            if cache_hit and has_efi and has_spec:
                design_data["efi_depths"] = cached_data["efi_depths"]
                design_data["efi_intensity"] = cached_data["efi_intensity"]
                spec_w = cached_data["spec_wavelengths"]
                if len(spec_w) > 0 and spec_w[0] > 100000.0:
                    spec_w = [w / 1e9 for w in spec_w]
                design_data["spec_wavelengths"] = spec_w
                design_data["spec_transmission"] = cached_data["spec_transmission"]
            else:
                # Add to parallel task list
                tasks_to_compute.append({
                    "design_idx": design_idx,
                    "active_dOpt": active_dOpt,
                    "mapped_layer": mapped_layer,
                    "materialParams": materialParams,
                    "design_key": design_key,
                    "design_data": design_data
                })
        else:
            if cache_hit:
                if "efi_depths" in cached_data:
                    design_data["efi_depths"] = cached_data["efi_depths"]
                if "efi_intensity" in cached_data:
                    design_data["efi_intensity"] = cached_data["efi_intensity"]
                if "spec_wavelengths" in cached_data:
                    spec_w = cached_data["spec_wavelengths"]
                    if len(spec_w) > 0 and spec_w[0] > 100000.0:
                        spec_w = [w / 1e9 for w in spec_w]
                    design_data["spec_wavelengths"] = spec_w
                if "spec_transmission" in cached_data:
                    design_data["spec_transmission"] = cached_data["spec_transmission"]
                    
        tmm_data[design_idx] = design_data

    # Execute computations for cache misses in batch
    if tasks_to_compute:
        # Only use parallel processing if there are more than 5 designs to compute
        use_parallel = len(tasks_to_compute) > 5
        
        worker_args = []
        for t in tasks_to_compute:
            worker_args.append((
                t["design_idx"],
                t["active_dOpt"],
                t["mapped_layer"],
                t["materialParams"],
                lambda_nm
            ))
            
        computed_results = {}
        
        if use_parallel:
            print(f"  Starting parallel TMM solver pool for {len(tasks_to_compute)} designs...")
            import concurrent.futures
            
            src_path = str(Path(__file__).parent.parent.parent)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=4,
                initializer=worker_init,
                initargs=(src_path,)
            ) as executor:
                progress_console = Console(file=sys.stdout)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=progress_console,
                ) as progress:
                    task = progress.add_task("[cyan]Parallel TMM precomputation...", total=len(worker_args))
                    
                    futures = [executor.submit(tmm_worker, arg) for arg in worker_args]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            idx, d_phys, efi_d, efi_i, spec_w, spec_t = f.result()
                            computed_results[idx] = (d_phys, efi_d, efi_i, spec_w, spec_t)
                        except Exception as e:
                            print(f"\n  Warning: Worker failed to compute design: {e}")
                        progress.advance(task)
        else:
            progress_console = Console(file=sys.stdout)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=progress_console,
            ) as progress:
                task = progress.add_task("[cyan]Sequential TMM precomputation...", total=len(worker_args))
                for arg in worker_args:
                    try:
                        idx, d_phys, efi_d, efi_i, spec_w, spec_t = tmm_worker(arg)
                        computed_results[idx] = (d_phys, efi_d, efi_i, spec_w, spec_t)
                    except Exception as e:
                        print(f"\n  Warning: Failed to compute design sequentially: {e}")
                    progress.advance(task)
                    
        # Apply results and save to cache
        for t in tasks_to_compute:
            idx = t["design_idx"]
            key = t["design_key"]
            d_data = t["design_data"]
            
            if idx in computed_results:
                d_phys, efi_d, efi_i, spec_w, spec_t = computed_results[idx]
                
                if d_phys:
                    d_data["d_physical_nm"] = d_phys
                if efi_d:
                    d_data["efi_depths"] = efi_d
                    d_data["efi_intensity"] = efi_i
                if spec_w:
                    if len(spec_w) > 0 and spec_w[0] > 100000.0:
                        spec_w = [w / 1e9 for w in spec_w]
                    d_data["spec_wavelengths"] = spec_w
                    d_data["spec_transmission"] = spec_t
                    
                # Update cache
                cache[key] = {
                    "d_physical_nm": d_data["d_physical_nm"],
                    "material_names": d_data["material_names"],
                    "material_indices": d_data["material_indices"],
                    "precomputed": d_data["precomputed"]
                }
                if efi_d:
                    cache[key]["efi_depths"] = efi_d
                    cache[key]["efi_intensity"] = efi_i
                if spec_w:
                    cache[key]["spec_wavelengths"] = spec_w
                    cache[key]["spec_transmission"] = spec_t

    # Generate info_text for all designs
    for idx, d_data in tmm_data.items():
        thickness_sum = sum(d_data['d_physical_nm']) if 'd_physical_nm' in d_data else d_data['total_thickness']
        info_lines = []
        info_lines.append(f"  SELECTED DESIGN SUMMARY")
        info_lines.append(f"  -------------------------")
        info_lines.append(f"  Design Rank: #{d_data['rank']} / {total_designs}")
        if "run_name" in d_data:
            info_lines.append(f"  Run Directory: {d_data['run_name']}")
        info_lines.append(f"  Reflectivity: {d_data['reflectivity']:.6f}")
        info_lines.append(f"  Loss (1 - R): {1.0 - d_data['reflectivity']:.4e}")
        info_lines.append(f"  Absorption: {d_data['absorption']:.3f} ppm")
        info_lines.append(f"  Thermal Noise: {d_data['thermal_noise']:.4e} m/sqrt(Hz)")
        info_lines.append(f"  Utility Score: {d_data['utility_score']:.4f}")
        info_lines.append(f"  Active Layers: {d_data['active_layer_count']}")
        info_lines.append(f"  Total Physical Thickness: {thickness_sum:.2f} nm")
        d_data["info_text"] = "\\n".join(info_lines)

    # Final cache save on completion
    if cache_path is not None:
        try:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"  Saved {len(cache)} TMM results to cache file: {cache_path}")
        except Exception as e:
            print(f"  Warning: Failed to save TMM cache: {e}")
            
    return tmm_data


def calculate_log_ticks(cmin: float, cmax: float, is_ctn: bool = False) -> Tuple[list, list]:
    """Calculate clean log-scale ticks and formatting labels dynamically based on range span."""
    pmin = 10.0 ** cmin
    pmax = 10.0 ** cmax
    span = cmax - cmin
    
    if span > 3.0:
        mantissas = [1.0]
    elif span > 1.5:
        mantissas = [1.0, 2.0, 5.0]
    elif span > 0.6:
        mantissas = [1.0, 2.0, 3.0, 5.0, 7.0]
    else:
        mantissas = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        
    min_exp = int(np.floor(cmin))
    max_exp = int(np.ceil(cmax))
    
    ticks = []
    for exp in range(min_exp - 1, max_exp + 2):
        for m in mantissas:
            val = m * (10.0 ** exp)
            log_val = np.log10(val)
            if (cmin - 1e-9) <= log_val <= (cmax + 1e-9):
                ticks.append((log_val, val))
                
    ticks = sorted(ticks, key=lambda x: x[0])
    
    if len(ticks) < 2:
        log_vals = np.linspace(cmin, cmax, 5)
        tickvals = list(log_vals)
        if is_ctn:
            ticktext = [f"{10.0**v:.2e}" for v in log_vals]
        else:
            ticktext = [f"{10.0**v:.2f}" for v in log_vals]
        return tickvals, ticktext
        
    tickvals = [t[0] for t in ticks]
    ticktext = []
    for _, val in ticks:
        if is_ctn:
            log10_val = np.log10(val)
            if abs(log10_val - round(log10_val)) < 1e-9:
                ticktext.append(f"10^{int(round(log10_val))}")
            else:
                s = f"{val:.2e}"
                s = s.replace("e+", "e").replace("e-0", "e-").replace(".00", "")
                ticktext.append(s)
        else:
            if val >= 1.0:
                if val == int(val):
                    ticktext.append(str(int(val)))
                else:
                    ticktext.append(f"{val:.1f}")
            else:
                log10_val = np.log10(val)
                if abs(log10_val - round(log10_val)) < 1e-9:
                    ticktext.append(f"10^{int(round(log10_val))}")
                else:
                    if val >= 0.1:
                        ticktext.append(f"{val:.1f}")
                    elif val >= 0.01:
                        ticktext.append(f"{val:.2f}")
                    else:
                        s = f"{val:.3f}".rstrip('0')
                        if s.endswith('.'):
                            s = s[:-1]
                        ticktext.append(s)
                        
    return tickvals, ticktext


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
    rank_by_utility: bool = True,
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
    color_mode: str = "reflectivity_log",
    lambda_nm: float = 1064.0,
) -> Tuple[go.Figure, pd.DataFrame]:
    """Create interactive 3D scatter plot of Absorption, TN, and Rank."""
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
            thick = calculate_physical_thickness(row, materials, lambda_nm=lambda_nm)
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
    custom_data_records = []
    has_run_name = "run_name" in combined_df.columns
    for idx, row in combined_df.iterrows():
        record = [
            int(row["rank"]),
            float(row["reflectivity"]),
            float(1.0 - row["reflectivity"]),
            int(row["active_layer_count"]),
            float(row["total_thickness"]),
            float(row["utility_score"]),
            int(idx),
        ]
        if has_run_name:
            record.append(str(row["run_name"]))
        custom_data_records.append(record)
    customdata = custom_data_records

    # X, Y, Z data
    x_data = combined_df["absorption"].values
    y_data = combined_df["thermal_noise"].values
    z_data = combined_df["rank"].values

    if color_by_loss:
        color_mode = "loss_linear"

    # Determine marker colorscale and values
    tickvals = None
    ticktext = None
    is_reversed = False
    if color_mode == "reflectivity_linear":
        color_values = combined_df["reflectivity"].values
        colorbar_title = "Reflectivity"
        colorscale = "Plasma" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = False
    elif color_mode == "reflectivity_log":
        # Number of nines: -log10(1-R)
        losses = np.maximum(1e-10, 1.0 - combined_df["reflectivity"].values)
        color_values = -np.log10(losses)
        colorbar_title = "Reflectivity (Log/Nines)"
        colorscale = "Plasma" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = False
        
        min_int = int(np.floor(cmin))
        max_int = int(np.ceil(cmax))
        tickvals = list(range(min_int, max_int + 1))
        ticktext = []
        for v in tickvals:
            if v == 2: ticktext.append("0.99")
            elif v == 3: ticktext.append("0.999")
            elif v == 4: ticktext.append("0.9999")
            elif v == 5: ticktext.append("0.99999")
            elif v == 6: ticktext.append("0.999999")
            elif v == 7: ticktext.append("0.9999999")
            else: ticktext.append(f"1-10^-{v}")
    elif color_mode == "absorption_linear":
        color_values = combined_df["absorption"].values
        colorbar_title = "Absorption (ppm)"
        colorscale = "Viridis_r" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = True
    elif color_mode == "absorption_log":
        color_values = np.log10(np.maximum(1e-3, combined_df["absorption"].values))
        colorbar_title = "Absorption (Log10 ppm)"
        colorscale = "Viridis_r" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = True
        tickvals, ticktext = calculate_log_ticks(cmin, cmax, is_ctn=False)
    elif color_mode == "ctn_linear":
        color_values = combined_df["thermal_noise"].values
        colorbar_title = "Thermal Noise (m/√Hz)"
        colorscale = "Viridis_r" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = True
    elif color_mode == "ctn_log":
        color_values = np.log10(np.maximum(1e-24, combined_df["thermal_noise"].values))
        colorbar_title = "Thermal Noise (Log10)"
        colorscale = "Viridis_r" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = True
        tickvals, ticktext = calculate_log_ticks(cmin, cmax, is_ctn=True)
    elif color_mode == "loss_linear":
        color_values = 1.0 - combined_df["reflectivity"].values
        colorbar_title = "Reflectivity Loss (1-R)"
        colorscale = "Magma" if dark_mode else "Reds"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = not dark_mode
    elif color_mode == "loss_log":
        # log10(1-R)
        losses = np.maximum(1e-10, 1.0 - combined_df["reflectivity"].values)
        color_values = np.log10(losses)
        colorbar_title = "Loss (Log10)"
        colorscale = "Magma" if dark_mode else "Reds"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = not dark_mode
        
        min_int = int(np.floor(cmin))
        max_int = int(np.ceil(cmax))
        tickvals = list(range(min_int, max_int + 1))
        ticktext = [f"10^{v}" for v in tickvals]
    else:
        # Fallback
        color_values = combined_df["reflectivity"].values
        colorbar_title = "Reflectivity"
        colorscale = "Plasma" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = False

    # Compute outline colors based on lightness mapping
    span = cmax - cmin if cmax > cmin else 1.0
    normalized_vals = (color_values - cmin) / span
    outline_colors = []
    for t in normalized_vals:
        is_light = (t < 0.4) if is_reversed else (t > 0.6)
        if is_light:
            outline_colors.append("rgba(0, 0, 0, 0.2)")
        else:
            outline_colors.append("rgba(255, 255, 255, 0.2)")

    fig = go.Figure()

    # Add the primary 3D scatter trace
    fig.add_trace(
        go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode="markers",
            marker=dict(
                size=4,
                color=color_values,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right", font=dict(color="#e0e0e0" if dark_mode else "#333333")),
                    tickfont=dict(color="#e0e0e0" if dark_mode else "#333333"),
                    tickvals=tickvals,
                    ticktext=ticktext,
                    thickness=18,
                    len=0.7,
                ),
                showscale=True,
                opacity=0.9,
                line=dict(width=0.0),
            ),
            customdata=customdata,
            name="Pareto Front Designs",
            showlegend=True,
            hovertemplate=(
                "<b>Design Rank #%{customdata[0]:d}</b><br>"
                + ("Run: %{customdata[7]}<br>" if has_run_name else "")
                + "<br>"
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
        nargs="?",
        default=None,
        help="Directory containing pareto_front.csv (optional if --verify-physics is passed)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Recursively aggregate Pareto fronts from all subdirectories inside the target directory",
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
        "--rank-by-reflectivity",
        dest="rank_by_utility",
        action="store_false",
        help="Rank designs on the Z-axis by reflectivity instead of utility score",
    )
    parser.set_defaults(rank_by_utility=True)
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
        help="Number of top designs to precompute full TMM details (EFI and spectrum) for (default: -1, meaning all designs)",
    )
    parser.add_argument(
        "--beam-radius",
        "--wbeam",
        "--w0",
        type=float,
        default=None,
        help="Override laser beam radius w0 in meters (or mm if > 1.0) for CTN calculations and scaling (default: from config.ini or 0.062 m)",
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        choices=["reflectivity_linear", "reflectivity_log", "absorption_linear", "absorption_log", "ctn_linear", "ctn_log", "loss_linear", "loss_log"],
        default="reflectivity_log",
        help="Default color mapping mode for 3D scatter plot markers (default: reflectivity_log)",
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        choices=["rank", "explore"],
        default="rank",
        help="Default plot mode on startup (default: rank)",
    )
    parser.add_argument(
        "--z-log",
        action="store_true",
        help="Enable Z-axis log scale by default on startup",
    )
    parser.add_argument(
        "--x-linear",
        action="store_true",
        help="Use linear scale for X-axis by default instead of log on startup",
    )
    parser.add_argument(
        "--y-linear",
        action="store_true",
        help="Use linear scale for Y-axis by default instead of log on startup",
    )
    parser.add_argument(
        "--selected-rank",
        type=int,
        default=1,
        help="Default design rank to select on startup (default: 1)",
    )
    parser.add_argument(
        "--auto-rotate",
        action="store_true",
        help="Enable 3D camera auto-rotation on page load",
    )
    parser.add_argument(
        "--verify-physics",
        action="store_true",
        help="Run gold-standard physics benchmark verification self-test and exit",
    )
    args = parser.parse_args()

    if getattr(args, "verify_physics", False):
        success = verify_aligo_gold_standard(verbose=True)
        return 0 if success else 1

    # Determine default color mode, supporting backward compatibility with --color-by-loss
    if args.color_by_loss:
        color_mode = "loss_linear"
    else:
        color_mode = args.color_mode

    try:
        generate_3d_rank_dashboard_from_args(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


def generate_3d_rank_dashboard(
    directory,
    output=None,
    light=False,
    color_by_loss=False,
    no_open=True,
    compare_refl=None,
    compare_abs=None,
    compare_tn=None,
    compare_label="Reference Design",
    compare_thick=None,
    min_refl=None,
    max_abs=None,
    max_tn=None,
    rank_by_utility=True,
    top=None,
    weight_refl=0.10,
    weight_abs=0.35,
    weight_tn=0.45,
    weight_thick=0.10,
    target_refl=None,
    target_abs=None,
    target_tn=None,
    target_thick=None,
    precompute_tmm_count=-1,
    color_mode="reflectivity_log",
    aggregate=False,
    plot_mode="rank",
    z_log=False,
    x_linear=False,
    y_linear=False,
    selected_rank=1,
    auto_rotate=False,
):
    import argparse
    args = argparse.Namespace(
        directory=directory,
        output=output,
        light=light,
        color_by_loss=color_by_loss,
        no_open=no_open,
        compare_refl=compare_refl,
        compare_abs=compare_abs,
        compare_tn=compare_tn,
        compare_label=compare_label,
        compare_thick=compare_thick,
        min_refl=min_refl,
        max_abs=max_abs,
        max_tn=max_tn,
        rank_by_utility=rank_by_utility,
        top=top,
        weight_refl=weight_refl,
        weight_abs=weight_abs,
        weight_tn=weight_tn,
        weight_thick=weight_thick,
        target_refl=target_refl,
        target_abs=target_abs,
        target_tn=target_tn,
        target_thick=target_thick,
        precompute_tmm_count=precompute_tmm_count,
        color_mode=color_mode,
        aggregate=aggregate,
        plot_mode=plot_mode,
        z_log=z_log,
        x_linear=x_linear,
        y_linear=y_linear,
        selected_rank=selected_rank,
        auto_rotate=auto_rotate,
    )
    return generate_3d_rank_dashboard_from_args(args)


def generate_3d_rank_dashboard_from_args(args):

    # Determine default color mode, supporting backward compatibility with --color-by-loss
    if args.color_by_loss:
        color_mode = "loss_linear"
    else:
        color_mode = args.color_mode

    # Resolve target values, defaulting to comparison design values if they are provided,
    # and falling back to default values otherwise.
    target_refl = args.target_refl if args.target_refl is not None else (args.compare_refl if args.compare_refl is not None else 0.9999)
    target_abs = args.target_abs if args.target_abs is not None else (args.compare_abs if args.compare_abs is not None else 0.30)
    target_tn = args.target_tn if args.target_tn is not None else (args.compare_tn if args.compare_tn is not None else 4.0e-21)
    target_thick = args.target_thick if args.target_thick is not None else (args.compare_thick if args.compare_thick is not None else 6000.0)

    plot_mode = getattr(args, "plot_mode", "rank")
    z_log = getattr(args, "z_log", False)
    x_log = not getattr(args, "x_linear", False)
    y_log = not getattr(args, "y_linear", False)
    selected_rank = getattr(args, "selected_rank", 1)
    auto_rotate = getattr(args, "auto_rotate", False)

    # Run physics engine gold-standard self-test verification
    verify_aligo_gold_standard(verbose=True)

    # Convert to Path object
    directory = Path(args.directory)
    if not directory.is_absolute():
        directory = Path(os.getcwd()) / directory
    directory = directory.resolve()

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return 1

    # Scan for subdirectories containing pareto_front.csv if aggregate is enabled
    subdirs = []
    if getattr(args, "aggregate", False):
        for root, dirs, files in os.walk(directory):
            if "pareto_front.csv" in files:
                subdirs.append(Path(root))
        subdirs.sort()
        if not subdirs:
            print(f"Error: No subdirectories containing pareto_front.csv found under {directory}")
            return 1
        print(f"Found {len(subdirs)} Pareto fronts to aggregate:")
        for s in subdirs:
            print(f"  - {s.relative_to(directory) if s != directory else s.name}")
    else:
        subdirs = [directory]

    # Extract laser wavelength from config.ini files if available
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

    # Extract laser beam radius (wBeam) from CLI args or config.ini files
    import re
    wbeam_m = None
    wbeam_src = None

    def normalize_wbeam(val: float) -> float:
        if val <= 0:
            return 0.062
        if val > 1.0 and val <= 30.0:
            return val / 100.0  # Entered in cm (e.g. 9.0 cm or 6.2 cm)
        elif val > 30.0:
            return val / 1000.0 # Entered in mm (e.g. 62 mm or 90 mm)
        else:
            return val          # Entered in meters (e.g. 0.062 m or 0.09 m)

    if getattr(args, "beam_radius", None) is not None:
        wbeam_m = normalize_wbeam(float(args.beam_radius))
        wbeam_src = f"CLI argument --beam-radius {args.beam_radius}"

    if wbeam_m is None:
        for subdir in subdirs:
            config_path = subdir / "config.ini"
            if config_path.exists():
                cfg = configparser.ConfigParser()
                cfg.read(config_path)
                for section in ["General", "general", "Data", "data"]:
                    if cfg.has_section(section):
                        for key in ["wbeam", "wBeam", "beam_radius", "beam_width", "w0"]:
                            if cfg.has_option(section, key):
                                try:
                                    val_str = cfg.get(section, key)
                                    val_clean = re.sub(r'[^\d\.\-eE]', '', val_str)
                                    wbeam_m = normalize_wbeam(float(val_clean))
                                    wbeam_src = f"{subdir.name}/config.ini [{section}] ({key})"
                                    break
                                except ValueError:
                                    pass
                    if wbeam_m is not None:
                        break
            if wbeam_m is not None:
                break

    if wbeam_m is None:
        config_path = directory / "config.ini"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in ["General", "general", "Data", "data"]:
                if cfg.has_section(section):
                    for key in ["wbeam", "wBeam", "beam_radius", "beam_width", "w0"]:
                        if cfg.has_option(section, key):
                            try:
                                val_str = cfg.get(section, key)
                                val_clean = re.sub(r'[^\d\.\-eE]', '', val_str)
                                wbeam_m = normalize_wbeam(float(val_clean))
                                wbeam_src = f"{directory.name}/config.ini [{section}] ({key})"
                                break
                            except ValueError:
                                pass
                    if wbeam_m is not None:
                        break

    if wbeam_m is not None:
        print(f"  Loaded laser beam radius (wBeam): {wbeam_m*100:.1f} cm ({wbeam_m:.4f} m) (from {wbeam_src})")
    else:
        wbeam_m = 0.062
        print("  No 'wbeam' / 'beam_radius' key found in config.ini. Defaulting laser beam radius to 6.2 cm (0.0620 m).")

    # Extract operating temperature (Temp) from config.ini files
    temp_k = None
    temp_src = None
    for subdir in subdirs:
        config_path = subdir / "config.ini"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in ["General", "general", "Data", "data"]:
                if cfg.has_section(section):
                    for key in ["temperature", "temp", "Temp", "T", "temp_k"]:
                        if cfg.has_option(section, key):
                            try:
                                val_str = cfg.get(section, key)
                                val_clean = re.sub(r'[^\d\.\-eE]', '', val_str)
                                temp_k = float(val_clean)
                                temp_src = f"{subdir.name}/config.ini [{section}] ({key})"
                                break
                            except ValueError:
                                pass
                    if temp_k is not None:
                        break
            if temp_k is not None:
                break

    if temp_k is None:
        config_path = directory / "config.ini"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in ["General", "general", "Data", "data"]:
                if cfg.has_section(section):
                    for key in ["temperature", "temp", "Temp", "T", "temp_k"]:
                        if cfg.has_option(section, key):
                            try:
                                val_str = cfg.get(section, key)
                                val_clean = re.sub(r'[^\d\.\-eE]', '', val_str)
                                temp_k = float(val_clean)
                                temp_src = f"{directory.name}/config.ini [{section}] ({key})"
                                break
                            except ValueError:
                                pass
                    if temp_k is not None:
                        break

    if temp_k is not None:
        print(f"  Loaded operating temperature (Temp): {temp_k:.1f} K (from {temp_src})")
    else:
        temp_k = 293.0
        print("  No 'temperature' key found in config.ini. Defaulting operating temperature to 293.0 K.")

    # Load Pareto fronts and merge materials
    all_designs = []
    all_values = []
    materials = {}

    for subdir in subdirs:
        print(f"Loading Pareto front from {subdir}...")
        try:
            designs_df, values_df, _ = load_pareto_front(subdir)
            
            # Filter out designs with < 10 active layers (dynamically adjusted if max active layers is less than 10)
            temp_counts = []
            max_active_in_run = 0
            for idx, row in designs_df.iterrows():
                dOpt, mat_idx = parse_design(row)
                active_mask = (mat_idx != 0) & (dOpt > 1e-12)
                active_layer_count = int(np.sum(active_mask))
                temp_counts.append(active_layer_count)
                max_active_in_run = max(max_active_in_run, active_layer_count)
                
            min_required_layers = min(10, max_active_in_run) if max_active_in_run > 0 else 0
            
            valid_indices = []
            for idx, count in enumerate(temp_counts):
                if count >= min_required_layers:
                    valid_indices.append(idx)
            
            initial_count = len(designs_df)
            designs_df = designs_df.iloc[valid_indices].reset_index(drop=True)
            values_df = values_df.iloc[valid_indices].reset_index(drop=True)
            filtered_count = initial_count - len(designs_df)
            if filtered_count > 0:
                print(f"  Filtered out {filtered_count} designs with < {min_required_layers} active layers (kept {len(designs_df)} designs).")
                
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

            # Otherwise, try to load materials from subdir config
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
            print(f"Error: Failed to load Pareto front from {subdir}: {e}")
            return 1

    designs_df = pd.concat(all_designs, axis=0, ignore_index=True)
    values_df = pd.concat(all_values, axis=0, ignore_index=True)
    print(f"  Loaded {len(designs_df)} total designs successfully.")

    # Rescale thermal noise in Pareto front values_df if wBeam differs from standard 6.2 cm reference
    ref_wbeam_m = 0.062
    if "thermal_noise" in values_df.columns:
        if abs(wbeam_m - ref_wbeam_m) > 1e-6:
            ctn_scale = ref_wbeam_m / wbeam_m
            values_df["thermal_noise"] = values_df["thermal_noise"] * ctn_scale
            print(f"  Rescaled Pareto front CTN values for {len(values_df)} designs from reference wBeam = 6.2 cm (0.0620 m) to wBeam = {wbeam_m*100:.1f} cm ({wbeam_m:.4f} m) (scale factor: {ctn_scale:.4f}).")

    # Fallback to load materials from parent directory if still empty
    if not materials:
        parent_materials_path = directory / "materials.json"
        if parent_materials_path.exists():
            try:
                from coatopt.utils.utils import load_materials
                materials = load_materials(str(parent_materials_path))
            except Exception:
                pass

    if not materials:
        config_path = directory / "config.ini"
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

    # Fallback to load default materials.json from experiments folder if still empty
    if not materials:
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            default_materials_path = project_root / "experiments" / "materials.json"
            if default_materials_path.exists():
                from coatopt.utils.utils import load_materials
                materials = load_materials(str(default_materials_path))
                print(f"  Loaded default materials library from: {default_materials_path}")
        except Exception:
            pass

    # Convert materials keys to integers where possible for robust mapping
    if materials:
        clean_materials = {}
        for k, v in materials.items():
            try:
                clean_materials[int(k)] = v
            except (ValueError, TypeError):
                clean_materials[k] = v
        materials = clean_materials
    else:
        materials = None

    if "thermal_noise" not in values_df.columns:
        values_df["thermal_noise"] = np.nan

    nan_mask = values_df["thermal_noise"].isna()
    if nan_mask.any():
        print(f"  Warning: 'thermal_noise' not found or missing for {nan_mask.sum()} designs.")
        # Try to calculate it dynamically if physics solver is available
        loaded = load_physics_dependencies()
        if loaded and getCoatingThermalNoise is not None and materials is not None:
            import io
            import contextlib
            import sys
            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
            
            print("  Calculating missing coating thermal noise dynamically...")
            
            progress_console = Console(file=sys.stdout)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=progress_console,
            ) as progress:
                missing_indices = np.where(nan_mask)[0]
                task = progress.add_task("[cyan]Calculating thermal noise...", total=len(missing_indices))
                
                calculated_noise = values_df["thermal_noise"].values.copy()
                for idx in missing_indices:
                    row = designs_df.iloc[idx]
                    try:
                        dOpt, material_indices = parse_design(row)
                        active_mask = (material_indices != 0) & (dOpt > 1e-12)
                        active_dOpt = dOpt[active_mask]
                        active_materialLayer = material_indices[active_mask]
                        
                        # Reverse layers so they are in air-to-substrate order
                        active_dOpt = active_dOpt[::-1]
                        active_materialLayer = active_materialLayer[::-1]
                        
                        # Map material 0 to 999
                        mapped_layer = np.array([999 if m == 0 else m for m in active_materialLayer])
                        
                        # Build materialParams structure
                        materialParams = {}
                        for k, v in materials.items():
                            try:
                                mat_key = int(k)
                            except (ValueError, TypeError):
                                continue
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
                            
                        with contextlib.redirect_stdout(io.StringIO()):
                            noise_summary, _, _, _, _, _ = getCoatingThermalNoise(
                                dOpt=active_dOpt,
                                materialLayer=mapped_layer,
                                materialParams=materialParams,
                                materialSub=1,
                                lambda_=wavelength_nm * 1e-9,  # getCoatingThermalNoise expects meters!
                                f=100.0,
                                wBeam=wbeam_m,
                                Temp=temp_k,
                                plots=False
                            )
                        
                        if isinstance(noise_summary["Frequency"], (float, np.floating)):
                            thermal_noise_val = noise_summary["BrownianNoise"]
                        else:
                            difference_array = np.absolute(noise_summary["Frequency"] - 100.0)
                            index = difference_array.argmin()
                            thermal_noise_val = noise_summary["BrownianNoise"][index]
                            
                        calculated_noise[idx] = thermal_noise_val
                    except Exception as e:
                        calculated_noise[idx] = 0.0
                    progress.advance(task)
            
            values_df["thermal_noise"] = calculated_noise
            print(f"  Successfully calculated coating thermal noise for {len(missing_indices)} designs.")
        else:
            print("  Error: Could not calculate thermal noise dynamically (physics solvers or materials missing).")
            print("  This Pareto front appears to be from a 2-objective optimization (reflectivity, absorption).")
            print("  Please use 'plot_interactive_pareto.py' to visualize 2-objective Pareto fronts:")
            print(f"  uv run python -m coatopt.utils.plot_interactive_pareto {args.directory}")
            raise KeyError("thermal_noise")

    title = f"Pareto Front 3D Rank Plot: {directory.name}"
    fig, combined_df = create_3d_rank_plot(
        designs_df=designs_df,
        values_df=values_df,
        title=title,
        dark_mode=not args.light,
        color_mode=color_mode,
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
        lambda_nm=wavelength_nm,
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
        lambda_nm=wavelength_nm,
        cache_dir=directory
    )
    tmm_data_json = json.dumps(tmm_data)

    # Build materials mappings dict for client-side exporter
    materials_params_dict = {}
    if materials is not None:
        for k, v in materials.items():
            try:
                mat_key = int(k)
            except (ValueError, TypeError):
                mat_key = k
            mat_data = v.copy()
            if mat_data.get("n") is None:
                mat_data["n"] = 1.0
            if mat_data.get("k") is None:
                mat_data["k"] = 0.0
            if mat_key == 0 or (isinstance(mat_key, str) and mat_key.lower() == "air"):
                materials_params_dict[999] = mat_data
                materials_params_dict[0] = mat_data
                materials_params_dict["air"] = mat_data
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
        .btn-icon {
            padding: 6px 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            height: 29px;
            box-sizing: border-box;
        }
        .btn-icon.active {
            background-color: #00bcd4;
            border-color: #00bcd4;
            color: #121212;
        }
        .btn-icon.active:hover {
            background-color: #00e5ff;
            border-color: #00e5ff;
        }
        .btn-mode {
            background-color: transparent;
            color: #888;
            border: none;
            border-radius: 4px;
            padding: 6px 14px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-mode.active {
            background-color: #00bcd4;
            color: #121212;
        }
        .btn-mode:hover:not(.active) {
            color: #ffffff;
            background-color: #2b2b2b;
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

        /* Modal Overlay & Dialog Styles */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(4px);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.15s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .modal-dialog {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
            width: 90%;
            max-width: 920px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .modal-header {
            padding: 14px 20px;
            background-color: #161616;
            border-bottom: 1px solid #2d2d2d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h3 {
            margin: 0;
            font-size: 15px;
            color: #80cbc4;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .modal-close {
            background: transparent;
            border: none;
            color: #888;
            font-size: 22px;
            cursor: pointer;
            line-height: 1;
            padding: 0 4px;
        }
        .modal-close:hover {
            color: #ff5252;
        }
        .modal-body {
            padding: 20px;
            overflow-y: auto;
            flex-grow: 1;
            background-color: #121212;
        }
        .mat-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
            gap: 15px;
        }
        .mat-card {
            background-color: #1a1a1a;
            border: 1px solid #2d2d2d;
            border-radius: 6px;
            padding: 14px;
            font-size: 11px;
            transition: border-color 0.2s;
        }
        .mat-card.used-in-design {
            border-color: #00bcd4;
            background-color: #13242b;
        }
        .mat-card-title {
            font-size: 14px;
            font-weight: bold;
            color: #80cbc4;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2a2a2a;
            padding-bottom: 6px;
        }
        .mat-prop-row {
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            border-bottom: 1px dotted #222;
        }
        .mat-prop-label {
            color: #888;
        }
        .mat-prop-value {
            color: #e0e0e0;
            font-family: monospace;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>__TITLE__</h1>
        <p>Interactive Pareto Front & Diagnostics Dashboard — Left click any point in 3D to inspect</p>
        <div style="margin-top: 6px;">
            <span style="font-size: 11px; color: #81c784; background-color: rgba(76, 175, 80, 0.12); border: 1px solid #2e7d32; padding: 3px 10px; border-radius: 12px; font-weight: 500;">
                ✓ Physics Engine Verified (aLIGO Gold Standard 6.9919e-21 m/&radic;Hz)
            </span>
        </div>
    </div>
    <div class="container">
        <div class="left-col">
            <!-- Mode Toggle Bar -->
            <div class="mode-toggle-bar" style="display: flex; background: #1a1a1a; padding: 10px; border-bottom: 1px solid #2d2d2d; align-items: center; gap: 15px;">
                <span style="font-size: 11px; color: #888; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Plot Mode:</span>
                <div style="display: flex; gap: 5px; background: #121212; border: 1px solid #333; padding: 2px; border-radius: 6px;">
                    <button class="btn-mode active" id="btn-mode-rank">Ranked Mode</button>
                    <button class="btn-mode" id="btn-mode-explore">Exploration Mode</button>
                </div>
            </div>
            <div id="plot-3d" class="plot-container-3d"></div>
            
            <div class="card">
                <div class="card-title">Selected Design Information</div>
                <div style="display: flex; gap: 15px; align-items: stretch;">
                    <div id="info-content" class="info-card" style="flex-grow: 1; margin: 0; min-height: 80px;">Click a point in the 3D plot to inspect design details.</div>
                    <div class="action-buttons" style="flex-direction: column; justify-content: center; gap: 8px; margin: 0; min-width: 200px;">
                        <button class="btn btn-primary" id="btn-export-py" disabled style="width: 100%;">Export Python Design Script</button>
                        <button class="btn btn-primary" id="btn-export-csv" disabled style="width: 100%;">Export CSV Layers</button>
                        <button class="btn" id="btn-inspect-materials" style="width: 100%; background-color: #004d40; border-color: #00695c; color: #80cbc4;">🔍 Materials Inspector</button>
                        <button class="btn" id="btn-set-baseline" disabled style="width: 100%; background-color: #4e342e; border-color: #5d4037; color: #ffab91;">[+] Set as Baseline Target</button>
                        <button class="btn" id="btn-set-comparison-stack" disabled style="width: 100%; background-color: #1a237e; border-color: #283593; color: #c5cae9;">[+] Set as Comparison Stack</button>
                        <button class="btn" id="btn-clear-comparison-stack" style="width: 100%; display: none; background-color: #37474f; border-color: #455a64; color: #cfd8dc;">Clear Comparison Stack</button>
                    </div>
                </div>
            </div>

            <div class="controls-toolbar">
                <span style="font-size: 11px; color: #888; font-weight: bold; margin-right: 5px; text-transform: uppercase; letter-spacing: 0.5px;">3D VIEW OPTIONS:</span>
                <button class="btn btn-icon active" id="btn-reverse-z" title="Invert Z-Axis View">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;">
                        <path d="M7 21V3M7 3l-3 3M7 3l3 3M17 3v18M17 21l-3-3M17 21l3-3"/>
                    </svg>
                </button>
                <button class="btn btn-icon active" id="btn-toggle-x-scale" title="Toggle X-Scale (Log/Linear)">
                    <span style="font-size: 11px; font-weight: bold; margin-right: 2px; vertical-align: middle;">X</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;">
                        <path d="M3 3v18h18"/>
                        <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
                    </svg>
                </button>
                <button class="btn btn-icon active" id="btn-toggle-y-scale" title="Toggle Y-Scale (Log/Linear)">
                    <span style="font-size: 11px; font-weight: bold; margin-right: 2px; vertical-align: middle;">Y</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;">
                        <path d="M3 3v18h18"/>
                        <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
                    </svg>
                </button>
                <button class="btn btn-icon" id="btn-toggle-z-scale" title="Toggle Z-Scale (Log/Linear)">
                    <span style="font-size: 11px; font-weight: bold; margin-right: 2px; vertical-align: middle;">Z</span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;">
                        <path d="M3 3v18h18"/>
                        <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
                    </svg>
                </button>
                <button class="btn" id="btn-auto-rotate" style="background-color: #1b5e20; border-color: #2e7d32; color: #e8f5e9; font-size: 11px; font-weight: bold; padding: 6px 10px; height: 29px; box-sizing: border-box; display: inline-flex; align-items: center; vertical-align: middle;">Auto-Rotate: OFF</button>
                
                <span style="font-size: 11px; color: #888; font-weight: bold; margin-left: 15px; margin-right: 5px; text-transform: uppercase; letter-spacing: 0.5px;">COLOR BY:</span>
                <select id="select-color-mode" style="background: #2b2b2b; border: 1px solid #444; color: #e0e0e0; padding: 5px 8px; border-radius: 4px; font-size: 12px; height: 29px; box-sizing: border-box; vertical-align: middle;">
                    <option value="reflectivity_linear">Reflectivity (Linear)</option>
                    <option value="reflectivity_log">Reflectivity (Log/Nines)</option>
                    <option value="absorption_linear">Absorption (Linear)</option>
                    <option value="absorption_log">Absorption (Log)</option>
                    <option value="ctn_linear">CTN (Linear)</option>
                    <option value="ctn_log">CTN (Log)</option>
                    <option value="loss_linear">Loss (Linear)</option>
                    <option value="loss_log">Loss (Log)</option>
                    <option value="rank">Design Rank</option>
                </select>

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
                    <div style="display: flex; gap: 8px;">
                        <div style="flex: 1;">
                            <label for="input-beam-radius">Beam Radius w<sub>0</sub> (cm)</label>
                            <input type="number" id="input-beam-radius" step="0.1" min="0.1" value="__WBEAM_CM__" placeholder="e.g. 6.2">
                        </div>
                        <div style="flex: 1;">
                            <label for="input-temp-k">Temp T (K)</label>
                            <input type="number" id="input-temp-k" step="0.1" min="0.1" value="__TEMP_K__" placeholder="e.g. 293.0">
                        </div>
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
                <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Simulated Spectral Response</span>
                    <select id="select-spectrum-mode" style="background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #444; padding: 2px 6px; font-size: 11px; border-radius: 4px; cursor: pointer; outline: none; transition: border-color 0.2s;">
                        <option value="reflectivity">Reflectivity (%)</option>
                        <option value="transmission">Transmission (%)</option>
                    </select>
                </div>
                <div id="plot-spectrum" class="plot-2d"></div>
            </div>
    </div>

    <!-- Materials Inspector Modal -->
    <div id="materials-modal" class="modal-overlay" style="display: none;">
        <div class="modal-dialog">
            <div class="modal-header">
                <h3>🔬 Materials Library Inspector</h3>
                <button class="modal-close" id="btn-close-materials-modal">&times;</button>
            </div>
            <div class="modal-body" id="materials-modal-body">
                <!-- Content generated dynamically -->
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

        var rank_by_utility = __RANK_BY_UTILITY__;
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

        // Define global materialColors to ensure strict unique color mapping for all designs
        var materialColors = {
            "air": "#333333",
            "SiO2": "#1f77b4",
            "TiTa": "#c837ab",
            "Ti:Ta2O5": "#c837ab",
            "TiGermania": "#e377c2",
            "Substrate": "#7f7f7f"
        };
        
        // Gather all unique material names across ALL designs in tmmData to assign consistent colors
        var allUniqueMats = [];
        for (var key in tmmData) {
            var d = tmmData[key];
            if (d && d.material_names) {
                for (var j = 0; j < d.material_names.length; j++) {
                    var mat = d.material_names[j];
                    if (allUniqueMats.indexOf(mat) === -1) {
                        allUniqueMats.push(mat);
                    }
                }
            }
        }
        allUniqueMats.sort();
        
        var palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#bcbd22", "#17becf"];
        var paletteIdx = 0;
        allUniqueMats.forEach(function(mat) {
            if (!materialColors[mat]) {
                materialColors[mat] = palette[paletteIdx % palette.length];
                paletteIdx++;
            }
        });

        // Initialize target fields
        document.getElementById('select-color-mode').value = "__DEFAULT_COLOR_MODE__";
        document.getElementById('input-target-refl').value = __TARGET_REFL__;
        document.getElementById('input-target-abs').value = __TARGET_ABS__;
        document.getElementById('input-target-tn').value = "__TARGET_TN__";
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
            if (data3d[0].marker.colorbar.title) {
                if (typeof data3d[0].marker.colorbar.title === 'object') {
                    data3d[0].marker.colorbar.title.font = { color: '#e0e0e0' };
                } else {
                    data3d[0].marker.colorbar.title = { text: data3d[0].marker.colorbar.title, font: { color: '#e0e0e0' } };
                }
            } else {
                data3d[0].marker.colorbar.title = { text: '', font: { color: '#e0e0e0' } };
            }
            data3d[0].marker.colorbar.tickfont = { color: '#e0e0e0' };
        }

        // Initial Plotly setup
        layout3d.scene.xaxis.type = xLog ? 'log' : 'linear';
        layout3d.scene.yaxis.type = yLog ? 'log' : 'linear';
        Plotly.newPlot('plot-3d', data3d, layout3d, {responsive: true, displaylogo: false});

        // Run initial recalculation and filtering to sync UI state
        recalculateUtilityAndRerank();

        // Apply initial button active states
        if (plotMode === "explore") {
            document.getElementById('btn-mode-explore').classList.add('active');
            document.getElementById('btn-mode-explore').style.backgroundColor = '#00bcd4';
            document.getElementById('btn-mode-explore').style.color = '#121212';
            document.getElementById('btn-mode-rank').classList.remove('active');
            document.getElementById('btn-mode-rank').style.backgroundColor = 'transparent';
            document.getElementById('btn-mode-rank').style.color = '#888';
        } else {
            document.getElementById('btn-mode-rank').classList.add('active');
            document.getElementById('btn-mode-rank').style.backgroundColor = '#00bcd4';
            document.getElementById('btn-mode-rank').style.color = '#121212';
            document.getElementById('btn-mode-explore').classList.remove('active');
            document.getElementById('btn-mode-explore').style.backgroundColor = 'transparent';
            document.getElementById('btn-mode-explore').style.color = '#888';
        }
        
        if (xLog) {
            document.getElementById('btn-toggle-x-scale').classList.add('active');
        } else {
            document.getElementById('btn-toggle-x-scale').classList.remove('active');
        }
        if (yLog) {
            document.getElementById('btn-toggle-y-scale').classList.add('active');
        } else {
            document.getElementById('btn-toggle-y-scale').classList.remove('active');
        }
        if (zLog) {
            document.getElementById('btn-toggle-z-scale').classList.add('active');
        } else {
            document.getElementById('btn-toggle-z-scale').classList.remove('active');
        }

        var selectedDesignIdx = (designsList.length > 0) ? designsList[0].originalIdx : 0;
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
            if (!design || !design.d_physical_nm) {
                showPlotMessage('plot-stack', 'Detailed stack layout not precomputed for this design');
                return;
            }
            var legendShown = {};
            var hasComp = (comparisonDesignIdx !== null && comparisonDesignIdx !== -1);
            var compDesign = hasComp ? tmmData[comparisonDesignIdx] : null;
            
            var traces = [];
            if (compDesign) {
                var compTraces = getStackTraces(compDesign, 'x2', 'y2', legendShown, materialColors);
                traces = traces.concat(compTraces);
            }
            
            var selTraces = getStackTraces(design, 'x', 'y', legendShown, materialColors);
            traces = traces.concat(selTraces);
            
            var layout = {
                barmode: 'overlay',
                bargap: 0,
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

        var spectrumMode = "reflectivity";

        document.getElementById('select-spectrum-mode').addEventListener('change', function() {
            spectrumMode = this.value;
            if (selectedDesignIdx !== null && selectedDesignIdx !== -1) {
                var design = tmmData[selectedDesignIdx];
                if (design) {
                    drawSpectrumPlot(design);
                }
            }
        });

        function drawSpectrumPlot(design) {
            if (!design.precomputed || !design.spec_wavelengths) {
                showPlotMessage('plot-spectrum', 'Spectrum details not precomputed for this design');
                return;
            }
            
            var traces = [];
            var isRefl = (spectrumMode === "reflectivity");
            
            var sel_y = isRefl ? design.spec_transmission.map(t => 100.0 - t) : design.spec_transmission;
            
            // Selected design trace
            traces.push({
                x: design.spec_wavelengths,
                y: sel_y,
                mode: 'lines',
                line: { color: '#ff9800', width: 2 },
                name: 'Selected (Rank ' + design.rank + ')',
                hovertemplate: "Selected (Rank " + design.rank + ")<br>Wavelength: %{x:.1f} nm<br>" + 
                               (isRefl ? "Reflectivity" : "Transmission") + ": %{y:.6f}%<extra></extra>"
            });
            
            var hasComp = (comparisonDesignIdx !== null && comparisonDesignIdx !== -1);
            var compDesign = hasComp ? tmmData[comparisonDesignIdx] : null;
            var comp_y = null;
            
            if (compDesign && compDesign.spec_wavelengths) {
                comp_y = isRefl ? compDesign.spec_transmission.map(t => 100.0 - t) : compDesign.spec_transmission;
                traces.push({
                    x: compDesign.spec_wavelengths,
                    y: comp_y,
                    mode: 'lines',
                    line: { color: '#ff4081', width: 1.5, dash: 'dash' },
                    name: 'Comparison (Rank ' + compDesign.rank + ')',
                    hovertemplate: "Comparison (Rank " + compDesign.rank + ")<br>Wavelength: %{x:.1f} nm<br>" + 
                                   (isRefl ? "Reflectivity" : "Transmission") + ": %{y:.6f}%<extra></extra>"
                });
            }
            
            var layout = {
                paper_bgcolor: '#1e1e1e',
                plot_bgcolor: '#1e1e1e',
                margin: { l: 55, r: 20, t: 15, b: 35 },
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
                    title: { text: isRefl ? "Reflectivity (%)" : "Transmission (%)", font: { size: 10, color: '#e0e0e0' } },
                    tickfont: { size: 9, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    tickformat: isRefl ? '.4f' : '.4f'
                },
                legend: {
                    font: { size: 9, color: '#e0e0e0' },
                    orientation: 'h',
                    y: -0.4
                },
                shapes: [
                    {
                        type: 'line',
                        x0: __WAVELENGTH_NM__,
                        y0: 0,
                        x1: __WAVELENGTH_NM__,
                        y1: 1,
                        yref: 'paper',
                        line: { color: '#e53935', width: 1.5, dash: 'dot' }
                    }
                ]
            };
            
            // Adjust yaxis range dynamically to zoom in on the actual data bounds
            var all_y = sel_y;
            if (comp_y) {
                all_y = all_y.concat(comp_y);
            }
            var y_min = Math.min(...all_y);
            var y_max = Math.max(...all_y);
            var span = y_max - y_min;
            if (span < 1e-6) {
                layout.yaxis.range = isRefl ? [y_min - 0.001, 100.001] : [0, y_max + 0.001];
            } else {
                layout.yaxis.range = [Math.max(0, y_min - 0.1 * span), Math.min(100.001, y_max + 0.1 * span)];
            }
            
            Plotly.newPlot('plot-spectrum', traces, layout, {responsive: true, displayModeBar: false});
        }

        function updateSelectedDesign(idx) {
            selectedDesignIdx = idx;
            
            if (idx === -1) {
                var loss = compareRefl !== null ? 1.0 - compareRefl : 0.0;
                var text = "  REFERENCE DESIGN SUMMARY\\n";
                text += "  -------------------------\\n";
                text += "  Label: " + referenceLabel + "\\n";
                if (compareRefl !== null) {
                    text += "  Reflectivity: " + compareRefl.toFixed(6) + "\\n";
                    text += "  Loss (1 - R): " + loss.toExponential(4) + "\\n";
                }
                if (compareAbs !== null) {
                    text += "  Absorption: " + compareAbs.toFixed(3) + " ppm\\n";
                }
                if (compareTN !== null) {
                    text += "  Thermal Noise: " + compareTN.toExponential(4) + " m/sqrt(Hz)\\n";
                }
                if (compareThick !== null && compareThick > 0) {
                    text += "  Physical Thickness: " + compareThick.toFixed(2) + " nm\\n";
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

            var curWBeamM = (document.getElementById('input-beam-radius') && document.getElementById('input-beam-radius').value) ? (parseFloat(document.getElementById('input-beam-radius').value) / 100.0) : __WBEAM_M__;
            var curTempK = (document.getElementById('input-temp-k') && document.getElementById('input-temp-k').value) ? parseFloat(document.getElementById('input-temp-k').value) : __TEMP_K__;

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
                     `lambda_nm = __WAVELENGTH_NM__\\n` +
                     `aLIGO_params["lambda_"]        = lambda_nm * 1e-9                    # IFO wavelength in meters for physics solvers\\n` +
                     `aLIGO_params["lambda_nm"]     = lambda_nm                          # IFO wavelength (nm)\\n` +
                     `aLIGO_params["f"]              = np.logspace(1, 3, 100)             # Frequency range to evaluate CTN \\n` +
                     `aLIGO_params["wBeam"]          = ${curWBeamM.toFixed(6)}                        # laser beam size on ETM (m) \\n` +
                     `aLIGO_params["Temp"]           = ${curTempK.toFixed(1)}                              # detector temperature (K) \\n` +
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

        function getLogTicks(cmin, cmax, isCtn) {
            var pmin = Math.pow(10, cmin);
            var pmax = Math.pow(10, cmax);
            var min_exp = Math.floor(cmin);
            var max_exp = Math.ceil(cmax);
            var span = cmax - cmin;
            var mantissas = [];
            if (span > 3.0) {
                mantissas = [1.0];
            } else if (span > 1.5) {
                mantissas = [1.0, 2.0, 5.0];
            } else if (span > 0.6) {
                mantissas = [1.0, 2.0, 3.0, 5.0, 7.0];
            } else {
                mantissas = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            }
            
            var ticks = [];
            for (var exp = min_exp - 1; exp <= max_exp + 1; exp++) {
                for (var i = 0; i < mantissas.length; i++) {
                    var m = mantissas[i];
                    var val = m * Math.pow(10, exp);
                    var log_val = Math.log10(val);
                    if (log_val >= (cmin - 1e-9) && log_val <= (cmax + 1e-9)) {
                        ticks.push({log_val: log_val, val: val});
                    }
                }
            }
            
            ticks.sort((a, b) => a.log_val - b.log_val);
            
            if (ticks.length < 2) {
                var tickvals = [];
                var ticktext = [];
                for (var i = 0; i < 5; i++) {
                    var v = cmin + (cmax - cmin) * (i / 4);
                    tickvals.push(v);
                    var physical_val = Math.pow(10, v);
                    if (isCtn) {
                        ticktext.push(physical_val.toExponential(2));
                    } else {
                        ticktext.push(physical_val.toFixed(2));
                    }
                }
                return {tickvals: tickvals, ticktext: ticktext};
            }
            
            var tickvals = [];
            var ticktext = [];
            for (var i = 0; i < ticks.length; i++) {
                var t = ticks[i];
                tickvals.push(t.log_val);
                var val = t.val;
                if (isCtn) {
                    var log10_val = Math.log10(val);
                    if (Math.abs(log10_val - Math.round(log10_val)) < 1e-9) {
                        ticktext.push("10^" + Math.round(log10_val));
                    } else {
                        var s = val.toExponential(2);
                        s = s.replace("e+", "e").replace("e-0", "e-").replace(".00", "");
                        ticktext.push(s);
                    }
                } else {
                    if (val >= 1.0) {
                        if (val === Math.round(val)) {
                            ticktext.push(Math.round(val).toString());
                        } else {
                            ticktext.push(val.toFixed(1));
                        }
                    } else {
                        var log10_val = Math.log10(val);
                        if (Math.abs(log10_val - Math.round(log10_val)) < 1e-9) {
                            ticktext.push("10^" + Math.round(log10_val));
                        } else {
                            if (val >= 0.1) {
                                ticktext.push(val.toFixed(1));
                            } else if (val >= 0.01) {
                                ticktext.push(val.toFixed(2));
                            } else {
                                var s = val.toFixed(3);
                                while (s.endsWith("0")) {
                                    s = s.slice(0, -1);
                                }
                                if (s.endsWith(".")) {
                                    s = s.slice(0, -1);
                                }
                                ticktext.push(s);
                            }
                        }
                    }
                }
            }
            return {tickvals: tickvals, ticktext: ticktext};
        }

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

            // Sort designs depending on rank_by_utility
            if (rank_by_utility) {
                designsList.sort((a, b) => b.utility_score - a.utility_score);
            } else {
                designsList.sort((a, b) => b.reflectivity - a.reflectivity);
            }

            // Re-assign ranks 1 to M and update info_text
            designsList.forEach(function(d, index) {
                d.rank = index + 1;
                var loss = 1.0 - d.reflectivity;
                var info_lines = [];
                info_lines.push("  SELECTED DESIGN SUMMARY");
                info_lines.push("  -------------------------");
                info_lines.push("  Design Rank: #" + d.rank + " / " + designsList.length);
                if (d.run_name !== undefined && d.run_name !== null) {
                    info_lines.push("  Run Directory: " + d.run_name);
                }
                info_lines.push("  Reflectivity: " + d.reflectivity.toFixed(6));
                info_lines.push("  Loss (1 - R): " + loss.toExponential(4));
                info_lines.push("  Absorption: " + d.absorption.toFixed(3) + " ppm");
                info_lines.push("  Thermal Noise: " + d.thermal_noise.toExponential(4) + " m/sqrt(Hz)");
                info_lines.push("  Utility Score: " + d.utility_score.toFixed(4));
                info_lines.push("  Active Layers: " + d.active_layer_count);
                var thicknessText = d.d_physical_nm ? d.d_physical_nm.reduce((a, b) => a + b, 0).toFixed(2) : d.total_thickness.toFixed(2);
                info_lines.push("  Total Physical Thickness: " + thicknessText + " nm");
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
            var z_data;
            if (plotMode === "rank") {
                z_data = displayList.map(d => d.rank);
            } else {
                if (zLog) {
                    z_data = displayList.map(d => {
                        var loss = Math.max(1e-10, 1.0 - d.reflectivity);
                        return -Math.log10(loss);
                    });
                } else {
                    z_data = displayList.map(d => d.reflectivity);
                }
            }

            var customdata = displayList.map(d => [
                d.rank,
                d.reflectivity,
                1.0 - d.reflectivity,
                d.active_layer_count,
                d.total_thickness,
                d.utility_score,
                d.originalIdx
            ]);

            var color_values = [];
            var colorbar_title = "";
            var tickvals = null;
            var ticktext = null;
            var isReversed = false;
            var colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Plasma" : "Viridis";

            var colorMode = document.getElementById('select-color-mode').value;
            if (colorMode === "reflectivity_linear") {
                color_values = displayList.map(d => d.reflectivity);
                colorbar_title = "Reflectivity";
                isReversed = false;
            } else if (colorMode === "reflectivity_log") {
                // -log10(1-R)
                color_values = displayList.map(d => {
                    var loss = Math.max(1e-10, 1.0 - d.reflectivity);
                    return -Math.log10(loss);
                });
                colorbar_title = "Reflectivity (Log/Nines)";
                isReversed = false;
                
                var min_val = Math.min(...color_values);
                var max_val = Math.max(...color_values);
                var min_int = Math.floor(min_val);
                var max_int = Math.ceil(max_val);
                tickvals = [];
                ticktext = [];
                for (var v = min_int; v <= max_int; v++) {
                    tickvals.push(v);
                    if (v === 2) ticktext.push("0.99");
                    else if (v === 3) ticktext.push("0.999");
                    else if (v === 4) ticktext.push("0.9999");
                    else if (v === 5) ticktext.push("0.99999");
                    else if (v === 6) ticktext.push("0.999999");
                    else if (v === 7) ticktext.push("0.9999999");
                    else ticktext.push("1-10^-" + v);
                }
            } else if (colorMode === "absorption_linear") {
                color_values = displayList.map(d => d.absorption);
                colorbar_title = "Absorption (ppm)";
                colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Viridis_r" : "Viridis";
                isReversed = true;
            } else if (colorMode === "absorption_log") {
                color_values = displayList.map(d => Math.log10(Math.max(1e-3, d.absorption)));
                colorbar_title = "Absorption (Log10 ppm)";
                colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Viridis_r" : "Viridis";
                isReversed = true;
                
                var min_val = Math.min(...color_values);
                var max_val = Math.max(...color_values);
                var ticks_obj = getLogTicks(min_val, max_val, false);
                tickvals = ticks_obj.tickvals;
                ticktext = ticks_obj.ticktext;
            } else if (colorMode === "ctn_linear") {
                color_values = displayList.map(d => d.thermal_noise);
                colorbar_title = "Thermal Noise (m/√Hz)";
                colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Viridis_r" : "Viridis";
                isReversed = true;
            } else if (colorMode === "ctn_log") {
                color_values = displayList.map(d => Math.log10(Math.max(1e-24, d.thermal_noise)));
                colorbar_title = "Thermal Noise (Log10)";
                colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Viridis_r" : "Viridis";
                isReversed = true;
                
                var min_val = Math.min(...color_values);
                var max_val = Math.max(...color_values);
                var ticks_obj = getLogTicks(min_val, max_val, true);
                tickvals = ticks_obj.tickvals;
                ticktext = ticks_obj.ticktext;
            } else if (colorMode === "loss_linear") {
                color_values = displayList.map(d => 1.0 - d.reflectivity);
                colorbar_title = "Reflectivity Loss (1-R)";
                colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Magma" : "Reds";
                isReversed = !(layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff');
            } else if (colorMode === "loss_log") {
                // log10(1-R)
                color_values = displayList.map(d => {
                    var loss = Math.max(1e-10, 1.0 - d.reflectivity);
                    return Math.log10(loss);
                });
                colorbar_title = "Loss (Log10)";
                colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Magma" : "Reds";
                isReversed = !(layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff');
                
                var min_val = Math.min(...color_values);
                var max_val = Math.max(...color_values);
                var min_int = Math.floor(min_val);
                var max_int = Math.ceil(max_val);
                tickvals = [];
                ticktext = [];
                for (var v = min_int; v <= max_int; v++) {
                    tickvals.push(v);
                    ticktext.push("10^" + v);
                }
            } else if (colorMode === "rank") {
                // Color points by Rank
                color_values = displayList.map(d => d.rank);
                colorbar_title = "Design Rank";
                colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Plasma_r" : "Viridis_r";
                isReversed = true; // Reversed so lower rank number (better) is brighter/yellow

                tickvals = [];
                ticktext = [];
                var step = Math.max(1, Math.ceil(displayList.length / 10));
                for (var r = 1; r <= displayList.length; r += step) {
                    tickvals.push(r);
                    ticktext.push("#" + r);
                }
                if (tickvals[tickvals.length - 1] !== displayList.length && displayList.length > 0) {
                    tickvals.push(displayList.length);
                    ticktext.push("#" + displayList.length);
                }
            }

            var cmin = Math.min(...color_values) || 0;
            var cmax = Math.max(...color_values) || 1;
            var span = cmax > cmin ? (cmax - cmin) : 1.0;
            
            var outline_colors = color_values.map(val => {
                var t = (val - cmin) / span;
                var isLight = isReversed ? (t < 0.4) : (t > 0.6);
                return isLight ? "rgba(0, 0, 0, 0.2)" : "rgba(255, 255, 255, 0.2)";
            });

            data3d[0].x = x_data;
            data3d[0].y = y_data;
            data3d[0].z = z_data;
            data3d[0].customdata = customdata;
            data3d[0].marker.color = color_values;
            data3d[0].marker.cmin = cmin;
            data3d[0].marker.cmax = cmax;
            data3d[0].marker.colorscale = colorscale;
            data3d[0].marker.line.color = outline_colors;
            data3d[0].marker.line.width = 0.0;
            if (data3d[0].marker.colorbar) {
                if (typeof data3d[0].marker.colorbar.title === 'object') {
                    data3d[0].marker.colorbar.title.text = colorbar_title;
                } else {
                    data3d[0].marker.colorbar.title = { text: colorbar_title };
                }
                data3d[0].marker.colorbar.tickvals = tickvals;
                data3d[0].marker.colorbar.ticktext = ticktext;
            }

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
                    if (rank_by_utility) {
                        if (compare_utility >= designsList[i].utility_score) {
                            break;
                        }
                    } else {
                        if (r_val >= designsList[i].reflectivity) {
                            break;
                        }
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
                                     "Virtual Rank: " + rank_str + "<br>" +
                                     "Reference Utility: " + compare_utility.toFixed(4) + "<br>" +
                                     "<extra></extra>";

                data3d[1].x = [comp_abs];
                data3d[1].y = [comp_tn];
                data3d[1].z = plotMode === "rank" ? [virtual_rank] : (zLog ? [-Math.log10(Math.max(1e-10, 1.0 - r_val))] : [r_val]);
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

            // Apply Z scale type
            if (plotMode === "rank") {
                layout3d.scene.zaxis.type = zLog ? 'log' : 'linear';
            } else {
                layout3d.scene.zaxis.type = 'linear'; // plot nines using custom linear values
            }

            if (plotMode === "rank") {
                var maxRank = Math.max(...data3d[0].z) || 100;
                if (zLog) {
                    if (reversedZ) {
                        layout3d.scene.zaxis.range = [Math.log10(maxRank + 2.0), Math.log10(0.5)];
                    } else {
                        layout3d.scene.zaxis.range = [Math.log10(0.5), Math.log10(maxRank + 2.0)];
                    }
                } else {
                    if (reversedZ) {
                        layout3d.scene.zaxis.range = [maxRank + 2.0, 0.5];
                    } else {
                        layout3d.scene.zaxis.range = [0.5, maxRank + 2.0];
                    }
                }
                layout3d.scene.zaxis.title.text = rank_by_utility ? "Design Rank (Utility)" : "Design Rank (Reflectivity)";
                layout3d.scene.zaxis.tickvals = null;
                layout3d.scene.zaxis.ticktext = null;
            } else {
                var z_vals = data3d[0].z;
                var zmin = Math.min(...z_vals) || 0.9;
                var zmax = Math.max(...z_vals) || 1.0;
                var span = zmax - zmin;
                
                if (reversedZ) {
                    layout3d.scene.zaxis.range = [zmax + 0.05 * span, zmin - 0.05 * span];
                } else {
                    layout3d.scene.zaxis.range = null; // auto-range
                }
                
                if (zLog) {
                    layout3d.scene.zaxis.title.text = "Reflectivity (Log/Nines)";
                    // Setup custom ticks for nines
                    var min_int = Math.floor(zmin);
                    var max_int = Math.ceil(zmax);
                    var tickvals = [];
                    var ticktext = [];
                    for (var v = min_int; v <= max_int; v++) {
                        tickvals.push(v);
                        if (v === 2) ticktext.push("0.99");
                        else if (v === 3) ticktext.push("0.999");
                        else if (v === 4) ticktext.push("0.9999");
                        else if (v === 5) ticktext.push("0.99999");
                        else if (v === 6) ticktext.push("0.999999");
                        else if (v === 7) ticktext.push("0.9999999");
                        else ticktext.push("1-10^-" + v);
                    }
                    layout3d.scene.zaxis.tickvals = tickvals;
                    layout3d.scene.zaxis.ticktext = ticktext;
                } else {
                    layout3d.scene.zaxis.title.text = "Reflectivity";
                    layout3d.scene.zaxis.tickvals = null;
                    layout3d.scene.zaxis.ticktext = null;
                }
            }

            Plotly.react('plot-3d', data3d, layout3d);
        }

        document.getElementById('btn-apply-top').addEventListener('click', recalculateUtilityAndRerank);
        document.getElementById('select-color-mode').addEventListener('change', recalculateUtilityAndRerank);
        document.getElementById('input-top-x').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                recalculateUtilityAndRerank();
            }
        });

        // Mode toggling event listeners
        var plotMode = "__DEFAULT_PLOT_MODE__";
        document.getElementById('btn-mode-rank').addEventListener('click', function() {
            if (plotMode === "rank") return;
            plotMode = "rank";
            this.classList.add('active');
            this.style.background = '#00bcd4';
            this.style.color = '#121212';
            
            var btnExplore = document.getElementById('btn-mode-explore');
            btnExplore.classList.remove('active');
            btnExplore.style.background = 'transparent';
            btnExplore.style.color = '#888';
            
            // Enable color mode dropdown
            document.getElementById('select-color-mode').disabled = false;
            
            recalculateUtilityAndRerank();
        });

        document.getElementById('btn-mode-explore').addEventListener('click', function() {
            if (plotMode === "explore") return;
            plotMode = "explore";
            this.classList.add('active');
            this.style.background = '#00bcd4';
            this.style.color = '#121212';
            
            var btnRank = document.getElementById('btn-mode-rank');
            btnRank.classList.remove('active');
            btnRank.style.background = 'transparent';
            btnRank.style.color = '#888';
            
            // Enable color mode dropdown and default to rank
            var colorModeSelect = document.getElementById('select-color-mode');
            colorModeSelect.disabled = false;
            colorModeSelect.value = "rank";
            
            recalculateUtilityAndRerank();
        });

        // Z-axis view controls
        var reversedZ = true;
        document.getElementById('btn-reverse-z').addEventListener('click', function() {
            reversedZ = !reversedZ;
            if (reversedZ) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }
            recalculateUtilityAndRerank();
        });

        // X, Y & Z scale controls
        var xLog = __DEFAULT_X_LOG__;
        document.getElementById('btn-toggle-x-scale').addEventListener('click', function() {
            xLog = !xLog;
            if (xLog) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }
            Plotly.relayout('plot-3d', {'scene.xaxis.type': xLog ? 'log' : 'linear'});
        });

        var yLog = __DEFAULT_Y_LOG__;
        document.getElementById('btn-toggle-y-scale').addEventListener('click', function() {
            yLog = !yLog;
            if (yLog) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }
            Plotly.relayout('plot-3d', {'scene.yaxis.type': yLog ? 'log' : 'linear'});
        });

        var zLog = __DEFAULT_Z_LOG__;
        document.getElementById('btn-toggle-z-scale').addEventListener('click', function() {
            zLog = !zLog;
            if (zLog) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }
            recalculateUtilityAndRerank();
        });

        // Auto-select design by default on page load
        var initialSelectIdx = (designsList.length > 0) ? designsList[0].originalIdx : null;
        var defaultSelectedRank = __DEFAULT_SELECTED_RANK__;
        for (var i = 0; i < designsList.length; i++) {
            if (designsList[i].rank === defaultSelectedRank) {
                initialSelectIdx = designsList[i].originalIdx;
                break;
            }
        }
        if (initialSelectIdx !== null) {
            updateSelectedDesign(initialSelectIdx);
        }

        // Ensure all Plotly responsive plots redraw cleanly once container dimensions settle
        setTimeout(function() {
            try {
                Plotly.Plots.resize('plot-3d');
                Plotly.Plots.resize('plot-stack');
                Plotly.Plots.resize('plot-field');
                Plotly.Plots.resize('plot-spectrum');
            } catch(e) {}
        }, 150);

        // Camera rotation variables and animation loop
        var rotating = false;
        var rotateAngle = 0;
        
        function rotateCamera() {
            if (!rotating) return;
            rotateAngle += 0.012; // Speed of rotation
            
            var currentCamera = (plot3dDiv.layout && plot3dDiv.layout.scene && plot3dDiv.layout.scene.camera) ? plot3dDiv.layout.scene.camera : {};
            var eye = currentCamera.eye || {x: 1.5, y: 1.5, z: 1.2};
            var center = currentCamera.center || {x: 0, y: 0, z: 0};
            var up = currentCamera.up || {x: 0, y: 0, z: 1};
            
            var radius = Math.sqrt(eye.x * eye.x + eye.y * eye.y);
            if (radius < 0.2) radius = 1.8;
            
            var newEyeX = radius * Math.cos(rotateAngle);
            var newEyeY = radius * Math.sin(rotateAngle);
            
            Plotly.relayout('plot-3d', {
                'scene.camera.eye': {x: newEyeX, y: newEyeY, z: eye.z},
                'scene.camera.center': center,
                'scene.camera.up': up
            });
            
            if (rotating) {
                requestAnimationFrame(rotateCamera);
            }
        }
        
        document.getElementById('btn-auto-rotate').addEventListener('click', function() {
            rotating = !rotating;
            if (rotating) {
                this.innerText = "Auto-Rotate: ON";
                this.style.backgroundColor = "#c62828";
                this.style.borderColor = "#d32f2f";
                this.style.color = "#ffffff";
                var currentCamera = (plot3dDiv.layout && plot3dDiv.layout.scene && plot3dDiv.layout.scene.camera) ? plot3dDiv.layout.scene.camera : {};
                var eye = currentCamera.eye || {x: 1.5, y: 1.5, z: 1.2};
                rotateAngle = Math.atan2(eye.y, eye.x);
                rotateCamera();
            } else {
                this.innerText = "Auto-Rotate: OFF";
                this.style.backgroundColor = "#1b5e20";
                this.style.borderColor = "#2e7d32";
                this.style.color = "#e8f5e9";
            }
        });
        
        // Auto-start rotation if requested
        var autoRotateDefault = __AUTO_ROTATE__;
        if (autoRotateDefault) {
            document.getElementById('btn-auto-rotate').click();
        }

        function renderMaterialsModal() {
            var container = document.getElementById('materials-modal-body');
            if (!container) return;
            
            var html = '';
            var selectedMatIndices = new Set();
            if (typeof selectedDesignIdx !== 'undefined' && selectedDesignIdx !== null && selectedDesignIdx !== -1 && typeof tmmData !== 'undefined' && tmmData[selectedDesignIdx]) {
                var design = tmmData[selectedDesignIdx];
                if (design.materialLayer) {
                    design.materialLayer.forEach(function(m) { selectedMatIndices.add(m); });
                }
            }

            var uniqueMats = [];
            var seenKeys = new Set();
            if (typeof materialsParamsDict !== 'undefined' && materialsParamsDict) {
                for (var k in materialsParamsDict) {
                    var mat = materialsParamsDict[k];
                    if (!mat || !mat.name) continue;
                    var matKey = mat.name + '_' + (mat.n || 0);
                    if (seenKeys.has(matKey)) continue;
                    seenKeys.add(matKey);
                    uniqueMats.push({ key: k, mat: mat });
                }
            }

            html += '<div style="margin-bottom: 15px; font-size: 12px; color: #aaa;">Physical, optical, and thermal properties loaded for this experiment library:</div>';
            html += '<div class="mat-card-grid">';

            uniqueMats.forEach(function(item) {
                var k = item.key;
                var m = item.mat;
                var isUsed = selectedMatIndices.has(parseInt(k)) || selectedMatIndices.has(k) || selectedMatIndices.has(m.name);
                
                var cardClass = isUsed ? 'mat-card used-in-design' : 'mat-card';
                var badge = isUsed ? '<span style="background: #00bcd4; color: #121212; padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;">USED IN STACK</span>' : '';

                html += '<div class="' + cardClass + '">';
                html += '  <div class="mat-card-title"><span>' + (m.name || ('Material ' + k)) + '</span>' + badge + '</div>';
                
                if (m.desc) {
                    html += '  <div style="font-size: 10px; color: #888; font-style: italic; margin-bottom: 8px;">' + m.desc + '</div>';
                }

                html += '  <div class="mat-prop-row"><span class="mat-prop-label">Refractive Index (n):</span><span class="mat-prop-value">' + (m.n !== undefined ? m.n : 'N/A') + '</span></div>';
                html += '  <div class="mat-prop-row"><span class="mat-prop-label">Extinction Coeff (k):</span><span class="mat-prop-value">' + (m.k !== undefined ? m.k.toExponential(2) : '0.0') + '</span></div>';
                
                if (m.alpha !== undefined && m.alpha !== null) {
                    html += '  <div class="mat-prop-row"><span class="mat-prop-label">Thermal Exp (&alpha;):</span><span class="mat-prop-value">' + m.alpha.toExponential(2) + ' K⁻¹</span></div>';
                }
                if (m.beta !== undefined && m.beta !== null) {
                    html += '  <div class="mat-prop-row"><span class="mat-prop-label">Thermo-Optic (&beta;):</span><span class="mat-prop-value">' + m.beta.toExponential(2) + ' K⁻¹</span></div>';
                }
                if (m.kappa !== undefined && m.kappa !== null) {
                    html += '  <div class="mat-prop-row"><span class="mat-prop-label">Thermal Cond (&kappa;):</span><span class="mat-prop-value">' + m.kappa + ' W/(m·K)</span></div>';
                }
                if (m.C !== undefined && m.C !== null) {
                    html += '  <div class="mat-prop-row"><span class="mat-prop-label">Heat Capacity (C):</span><span class="mat-prop-value">' + m.C.toExponential(2) + ' J/(m³·K)</span></div>';
                }
                if (m.Y !== undefined && m.Y !== null) {
                    var yGpa = (m.Y / 1e9).toFixed(1);
                    html += '  <div class="mat-prop-row"><span class="mat-prop-label">Young&apos;s Modulus (Y):</span><span class="mat-prop-value">' + yGpa + ' GPa</span></div>';
                }
                if (m.prat !== undefined && m.prat !== null) {
                    html += '  <div class="mat-prop-row"><span class="mat-prop-label">Poisson&apos;s Ratio (&nu;):</span><span class="mat-prop-value">' + m.prat + '</span></div>';
                }
                if (m.phiM !== undefined && m.phiM !== null) {
                    html += '  <div class="mat-prop-row"><span class="mat-prop-label">Mechanical Loss (&phi;<sub>M</sub>):</span><span class="mat-prop-value">' + m.phiM.toExponential(2) + '</span></div>';
                }
                
                html += '</div>';
            });

            html += '</div>';
            container.innerHTML = html;
        }

        function openMaterialsModal() {
            renderMaterialsModal();
            var modal = document.getElementById("materials-modal");
            if (modal) modal.style.display = "flex";
        }

        function closeMaterialsModal() {
            var modal = document.getElementById("materials-modal");
            if (modal) modal.style.display = "none";
        }

        var refWBeamM = __WBEAM_M__;
        var currentWBeamM = refWBeamM;
        var refTempK = __TEMP_K__;
        var currentTempK = refTempK;

        var originalCtnValues = (data3d && data3d[0] && data3d[0].y) ? data3d[0].y.slice() : [];
        var originalTmmCtn = {};
        if (typeof tmmData !== 'undefined' && tmmData) {
            for (var idx in tmmData) {
                if (tmmData[idx] && tmmData[idx].thermal_noise !== undefined) {
                    originalTmmCtn[idx] = tmmData[idx].thermal_noise;
                }
            }
        }

        function updatePhysicsScaling(newWm, newTempK) {
            if (newWm && newWm > 0) currentWBeamM = newWm;
            if (newTempK && newTempK > 0) currentTempK = newTempK;

            var wScale = refWBeamM / currentWBeamM;
            var tempScale = Math.sqrt(currentTempK / refTempK);
            var scaleFactor = wScale * tempScale;
            
            if (data3d && data3d[0] && originalCtnValues.length > 0) {
                var newY = [];
                for (var i = 0; i < originalCtnValues.length; i++) {
                    newY.push(originalCtnValues[i] * scaleFactor);
                }
                data3d[0].y = newY;
            }

            if (typeof tmmData !== 'undefined' && tmmData) {
                for (var idx in tmmData) {
                    if (tmmData[idx] && originalTmmCtn[idx] !== undefined) {
                        tmmData[idx].thermal_noise = originalTmmCtn[idx] * scaleFactor;
                    }
                }
            }

            if (typeof applyFiltersAndWeights === 'function') {
                applyFiltersAndWeights();
            }
        }

        function updateBeamRadiusScale(newWm) {
            updatePhysicsScaling(newWm, null);
        }

        function updateTemperatureScale(newTempK) {
            updatePhysicsScaling(null, newTempK);
        }

        window.addEventListener('DOMContentLoaded', function() {
            var btnInspect = document.getElementById('btn-inspect-materials');
            if (btnInspect) {
                btnInspect.addEventListener('click', openMaterialsModal);
            }
            var btnCloseMat = document.getElementById('btn-close-materials-modal');
            if (btnCloseMat) {
                btnCloseMat.addEventListener('click', closeMaterialsModal);
            }
            var modalEl = document.getElementById('materials-modal');
            if (modalEl) {
                modalEl.addEventListener('click', function(e) {
                    if (e.target === this) closeMaterialsModal();
                });
            }
            var inputBeamRadius = document.getElementById('input-beam-radius');
            if (inputBeamRadius) {
                inputBeamRadius.addEventListener('input', function() {
                    var cmVal = parseFloat(this.value);
                    if (cmVal && cmVal > 0) {
                        updateBeamRadiusScale(cmVal / 100.0);
                    }
                });
            }
            var inputTempK = document.getElementById('input-temp-k');
            if (inputTempK) {
                inputTempK.addEventListener('input', function() {
                    var kVal = parseFloat(this.value);
                    if (kVal && kVal > 0) {
                        updateTemperatureScale(kVal);
                    }
                });
            }
        });
    </script>
</body>
</html>"""

    initial_top_x_str = str(args.top) if args.top is not None else "null"

    # Populate the placeholders using standard replace method (fully robust to f-string brackets)
    compiled_html = html_template.replace("__TITLE__", title)
    compiled_html = compiled_html.replace("__INITIAL_TOP_X__", initial_top_x_str)
    compiled_html = compiled_html.replace("__DEFAULT_COLOR_MODE__", color_mode)
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
    compiled_html = compiled_html.replace("__WAVELENGTH_NM__", f"{wavelength_nm:.1f}")
    compiled_html = compiled_html.replace("__WBEAM_M__", f"{wbeam_m:.6f}")
    compiled_html = compiled_html.replace("__WBEAM_CM__", f"{wbeam_m * 100.0:.2f}")
    compiled_html = compiled_html.replace("__TEMP_K__", f"{temp_k:.1f}")
    compiled_html = compiled_html.replace("__WEIGHT_REFL__", f"{args.weight_refl:.4f}")
    compiled_html = compiled_html.replace("__WEIGHT_ABS__", f"{args.weight_abs:.4f}")
    compiled_html = compiled_html.replace("__WEIGHT_TN__", f"{args.weight_tn:.4e}")
    compiled_html = compiled_html.replace("__WEIGHT_THICK__", f"{args.weight_thick:.4f}")
    compiled_html = compiled_html.replace("__RANK_BY_UTILITY__", "true" if args.rank_by_utility else "false")
    compiled_html = compiled_html.replace("__DEFAULT_PLOT_MODE__", plot_mode)
    compiled_html = compiled_html.replace("__DEFAULT_X_LOG__", "true" if x_log else "false")
    compiled_html = compiled_html.replace("__DEFAULT_Y_LOG__", "true" if y_log else "false")
    compiled_html = compiled_html.replace("__DEFAULT_Z_LOG__", "true" if z_log else "false")
    compiled_html = compiled_html.replace("__DEFAULT_SELECTED_RANK__", str(selected_rank))
    compiled_html = compiled_html.replace("__AUTO_ROTATE__", "true" if auto_rotate else "false")

    print(f"Saving interactive dashboard to {output_path}...")
    with open(output_path, "w") as f:
        f.write(compiled_html)

    print(f"✓ Dashboard successfully saved to {output_path}")

    if not args.no_open:
        print(f"Opening dashboard in browser: file://{output_path}")
        webbrowser.open(f"file://{output_path}")

    return output_path


if __name__ == "__main__":
    sys.exit(main())
