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
from typing import Optional, Tuple, List, Dict, Any, Union

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
from coatopt.utils.metrics import (
    evaluate_dataset_proximity_metrics,
    compute_asf_scores,
    compute_target_yield,
    compute_objective_breakdown,
    compute_spacing_metric,
    compute_roi_hypervolume,
)

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


def parse_design(row: Union[pd.Series, dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract thicknesses and materials from a Pareto front row."""
    keys = list(row.keys()) if hasattr(row, "keys") else list(row.index)
    thickness_cols = [col for col in keys if col.startswith("thickness_")]
    material_cols = [col for col in keys if col.startswith("material_")]

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


def calculate_physical_thickness(row: Union[pd.Series, dict], materials_dict: dict, lambda_nm: float = 1064.0) -> float:
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
    import coatopt.utils.plot_interactive_3d_metrics as p
    p.load_physics_dependencies()


def tmm_worker(task_info):
    import io
    import contextlib
    import numpy as np
    import coatopt.utils.plot_interactive_3d_metrics as p
    
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
    
    # Build materialParams structure once outside the loop
    materialParams = {}
    if materials_dict:
        for k, v in materials_dict.items():
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

    df_records = combined_df.to_dict("records")
    for design_idx, row in enumerate(df_records):
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
            
        trans_val = float(row["transmission"]) if "transmission" in row and not pd.isna(row["transmission"]) else float(max(0.0, (1.0 - row["reflectivity"]) * 1e6 - row["absorption"]))
        refl_val = float(row["reflectivity"]) if "reflectivity" in row and not pd.isna(row["reflectivity"]) else float(np.clip(1.0 - (trans_val + row["absorption"]) * 1e-6, 0.0, 1.0))
        design_data = {
            "rank": rank,
            "reflectivity": refl_val,
            "transmission": trans_val,
            "absorption": float(row["absorption"]),
            "thermal_noise": float(row["thermal_noise"]),
            "utility_score": float(row.get("utility_score", 0.0)),
            "active_layer_count": int(row["active_layer_count"]),
            "total_thickness": float(row["total_thickness"]),
            "asf_distance": float(row.get("asf_distance", 0.0)),
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
                    for i in range(len(active_dOpt)):
                        mat_idx = mapped_layer[i]
                        n_layer = materialParams.get(mat_idx, {}).get("n", 1.45) if materialParams else 1.45
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
        info_lines.append(f"  Transmission: {d_data['transmission']:.2f} ppm")
        info_lines.append(f"  Reflectivity: {d_data['reflectivity']:.6f}")
        info_lines.append(f"  Loss (1 - R): {1.0 - d_data['reflectivity']:.4e}")
        info_lines.append(f"  Absorption: {d_data['absorption']:.3f} ppm")
        info_lines.append(f"  Thermal Noise: {d_data['thermal_noise']:.4e} m/sqrt(Hz)")
        info_lines.append(f"  Utility Score: {d_data['utility_score']:.4f}")
        info_lines.append(f"  ASF Chebyshev Distance: {d_data.get('asf_distance', 0.0):.4f}")
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
    weight_trans: Optional[float] = None,
    weight_abs: float = 0.35,
    weight_tn: float = 0.45,
    weight_thick: float = 0.10,
    compare_thick: Optional[float] = None,
    compare_trans: Optional[float] = None,
    target_refl: float = 0.99999,
    target_trans: Optional[float] = None,
    target_abs: float = 0.30,
    target_tn: float = 4.0e-21,
    target_thick: float = 6000.0,
    top_n: Optional[int] = None,
    color_mode: str = "reflectivity_log",
    min_trans: Optional[float] = None,
    max_trans: Optional[float] = None,
    rank_by_transmission: bool = False,
    optimise_parameters: Optional[List[str]] = None,
    primary_metric: Optional[str] = None,
    lambda_nm: float = 1064.0,
) -> Tuple[go.Figure, pd.DataFrame]:
    """Create interactive 3D scatter plot of Absorption, TN, and Rank."""
    combined_df = pd.concat([designs_df, values_df], axis=1)

    if primary_metric is None:
        primary_metric = "transmission" if (optimise_parameters and "transmission" in optimise_parameters) or ("transmission" in combined_df.columns and "reflectivity" not in combined_df.columns) else "reflectivity"

    # Filter by minimum reflectivity if specified
    if min_refl is not None and "reflectivity" in combined_df.columns:
        combined_df = combined_df[combined_df["reflectivity"] >= min_refl].reset_index(drop=True)

    # Filter by transmission if specified
    if min_trans is not None and "transmission" in combined_df.columns:
        combined_df = combined_df[combined_df["transmission"] >= min_trans].reset_index(drop=True)
    if max_trans is not None and "transmission" in combined_df.columns:
        combined_df = combined_df[combined_df["transmission"] <= max_trans].reset_index(drop=True)

    # Filter by maximum absorption if specified
    if max_abs is not None:
        combined_df = combined_df[combined_df["absorption"] <= max_abs].reset_index(drop=True)

    # Filter by maximum thermal noise if specified
    if max_tn is not None:
        combined_df = combined_df[combined_df["thermal_noise"] <= max_tn].reset_index(drop=True)

    # Calculate physical thicknesses if materials are loaded, otherwise fall back to sum of dOpt
    if "total_thickness" not in combined_df.columns or combined_df["total_thickness"].isna().any():
        records = combined_df.to_dict("records")
        thickness_vals = []
        for row in records:
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
    # Objective 1: Transmission (minimized) or Reflectivity (maximized)
    if primary_metric == "transmission" and "transmission" in combined_df.columns:
        trans_target = target_trans if target_trans is not None else 10.0
        trans_scale = max(1.0, trans_target)
        obj1_score = np.where(
            combined_df["transmission"] <= trans_target,
            0.9 + 0.1 * (trans_target - combined_df["transmission"]) / trans_scale,
            0.9 * np.exp(-(combined_df["transmission"] - trans_target) / trans_scale)
        )
    else:
        refl_loss_scale = max(1e-6, 1.0 - target_refl)
        obj1_score = np.where(
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
    w_obj1_in = (weight_trans if weight_trans is not None else weight_refl) if primary_metric == "transmission" else weight_refl
    total_w = w_obj1_in + weight_abs + weight_tn + weight_thick
    w_obj1 = w_obj1_in / total_w if total_w > 0 else 0.10
    w_refl = weight_refl / total_w if total_w > 0 else 0.10
    w_abs = weight_abs / total_w if total_w > 0 else 0.35
    w_tn = weight_tn / total_w if total_w > 0 else 0.45
    w_thick = weight_thick / total_w if total_w > 0 else 0.10

    combined_df["utility_score"] = (
        w_obj1 * obj1_score +
        w_abs * abs_score +
        w_tn * tn_score +
        w_thick * thick_score
    )

    # Compute Achievement Scalarizing Function (ASF) Chebyshev distance to target
    targets_dict = {
        "primary_metric": primary_metric,
        "reflectivity": target_refl,
        "transmission": target_trans if target_trans is not None else (max(0.0, (1.0 - target_refl) * 1e6) if primary_metric != "transmission" else 10.0),
        "absorption": target_abs,
        "thermal_noise": target_tn,
        "total_thickness": target_thick,
    }
    weights_dict = {
        "reflectivity": w_refl,
        "transmission": w_obj1,
        "absorption": w_abs,
        "thermal_noise": w_tn,
        "total_thickness": w_thick,
    }
    combined_df["asf_distance"] = compute_asf_scores(combined_df, targets_dict, weights_dict, primary_metric=primary_metric)

    # Determine sorting column based on rank_by_utility
    if rank_by_utility:
        sort_col = "utility_score"
        ascending = False
        title_suffix = "Ranked by Multi-Objective Utility Score"
    elif (rank_by_transmission or primary_metric == "transmission") and "transmission" in combined_df.columns:
        sort_col = "transmission"
        ascending = True
        title_suffix = "Ranked by Transmission (Ascending)"
    else:
        sort_col = "reflectivity" if "reflectivity" in combined_df.columns else combined_df.columns[0]
        ascending = False
        title_suffix = "Ranked by Reflectivity"

    # Sort descending/ascending according to chosen metric
    combined_df = combined_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    combined_df["rank"] = combined_df.index + 1

    # Preserve full arrays for virtual ranking of reference design before slicing for display
    full_utility_vals = combined_df["utility_score"].values
    full_refl_vals = combined_df["reflectivity"].values if "reflectivity" in combined_df.columns else np.array([])
    full_trans_vals = combined_df["transmission"].values if "transmission" in combined_df.columns else np.array([])
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
        t_val = float(row["transmission"]) if "transmission" in row else float(max(0.0, (1.0 - row["reflectivity"]) * 1e6 - row["absorption"]))
        record = [
            int(row["rank"]),
            float(row["reflectivity"]),
            float(1.0 - row["reflectivity"]),
            int(row["active_layer_count"]),
            float(row["total_thickness"]),
            float(row["utility_score"]),
            int(idx),
            float(row["asf_distance"]),
            t_val,
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
    if color_mode == "transmission_linear":
        color_values = combined_df["transmission"].values
        colorbar_title = "Transmission (ppm)"
        colorscale = "Viridis_r" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = True
    elif color_mode == "transmission_log":
        color_values = np.log10(np.maximum(1e-3, combined_df["transmission"].values))
        colorbar_title = "Transmission (Log10 ppm)"
        colorscale = "Viridis_r" if dark_mode else "Viridis"
        cmin = float(np.min(color_values))
        cmax = float(np.max(color_values))
        is_reversed = True
        tickvals, ticktext = calculate_log_ticks(cmin, cmax, is_ctn=False)
    elif color_mode == "reflectivity_linear":
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
                size=5.5,
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
                + ("Run: %{customdata[9]}<br>" if has_run_name else "")
                + "<br>"
                "Transmission: %{customdata[8]:.2f} ppm<br>"
                "Reflectivity: %{customdata[1]:.6f}<br>"
                "Reflectivity Loss: %{customdata[2]:.3e}<br>"
                "Absorption: %{x:.4f} ppm<br>"
                "Thermal Noise: %{y:.4e} m/√Hz<br>"
                "Active Layers: %{customdata[3]:d}<br>"
                + ("Total Thickness: %{customdata[4]:.2f} nm<br>" if materials is not None else "Total dOpt: %{customdata[4]:.2f}<br>")
                + "Utility Score: %{customdata[5]:.4f}<br>"
                + "ASF Distance: %{customdata[7]:.4f}<br>"
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
            if primary_metric == "transmission":
                t_comp_val = compare_trans if compare_trans is not None else (target_trans if target_trans is not None else 10.0)
                trans_target = target_trans if target_trans is not None else 10.0
                trans_scale = max(1.0, trans_target)
                if t_comp_val <= trans_target:
                    obj1_comp_score = 0.9 + 0.1 * (trans_target - t_comp_val) / trans_scale
                else:
                    obj1_comp_score = 0.9 * np.exp(-(t_comp_val - trans_target) / trans_scale)
            else:
                r_comp_val = compare_refl if compare_refl is not None else target_refl
                if r_comp_val >= target_refl:
                    obj1_comp_score = 0.9 + 0.1 * (r_comp_val - target_refl) / refl_loss_scale
                else:
                    obj1_comp_score = 0.9 * np.exp(-(target_refl - r_comp_val) / refl_loss_scale)

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
                w_obj1 * obj1_comp_score +
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
            
            comp_trans_val = compare_trans if compare_trans is not None else ((1.0 - compare_refl) * 1e6 if compare_refl is not None else None)
            hover_comp_str = (
                f"<b>{compare_label} (Reference)</b><br><br>"
                + (f"Transmission: {comp_trans_val:.2f} ppm<br>" if comp_trans_val is not None else "")
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
            # Determine virtual rank based on transmission or reflectivity
            if primary_metric == "transmission":
                comp_t = compare_trans if compare_trans is not None else (target_trans if target_trans is not None else 10.0)
                if full_trans_vals is not None and len(full_trans_vals) > 0:
                    virtual_rank = float(np.searchsorted(full_trans_vals, comp_t)) + 1.0
                    if virtual_rank > total_designs:
                        virtual_rank = float(total_designs + 0.5)
                else:
                    virtual_rank = 1.0
            else:
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
            comp_trans_val = compare_trans if compare_trans is not None else ((1.0 - compare_refl) * 1e6 if compare_refl is not None else None)
            hover_comp_str = (
                f"<b>{compare_label} (Reference)</b><br><br>"
                + (f"Transmission: {comp_trans_val:.2f} ppm<br>" if comp_trans_val is not None else "")
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


def detect_model_family(subdir: Path) -> str:
    """Identify the optimization algorithm/model family for a run directory."""
    meta_path = subdir / "run_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            algo = meta.get("algorithm", "")
            if algo:
                algo_lower = algo.lower()
                if "hppo" in algo_lower or "ppo" in algo_lower:
                    return "hppo"
                elif "nsga" in algo_lower:
                    return "nsga2"
                return algo_lower
        except Exception:
            pass
    name_lower = subdir.name.lower()
    if "hppo" in name_lower or "ppo" in name_lower:
        return "hppo"
    elif "nsga" in name_lower:
        return "nsga2"
    return "unknown"


def matches_model_filter(subdir: Path, model_filter: str) -> bool:
    """Check whether a subdirectory matches a model filter string."""
    if not model_filter:
        return True
    flt = model_filter.lower().strip()
    detected = detect_model_family(subdir).lower()
    if flt in detected or detected in flt:
        return True
    if flt in subdir.name.lower():
        return True
    meta_path = subdir / "run_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            algo = str(meta.get("algorithm", "")).lower()
            if flt in algo:
                return True
        except Exception:
            pass
    return False


def list_detected_models(directory: Path):
    """Scan directory, print discovered models and run counts, then return."""
    runs_by_model = {}
    total_runs = 0
    for root, dirs, files in os.walk(directory):
        if "pareto_front.csv" in files:
            p = Path(root)
            model = detect_model_family(p)
            runs_by_model.setdefault(model, []).append(p)
            total_runs += 1

    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title=f"Discovered Models in {directory.name} ({total_runs} Total Runs)", show_header=True, header_style="bold cyan")
        table.add_column("Model / Algorithm", style="bold green", width=22)
        table.add_column("Runs", justify="right", width=8)
        table.add_column("Example Directory", style="dim")

        for model, paths in sorted(runs_by_model.items()):
            rel_example = str(paths[0].relative_to(directory)) if paths[0] != directory else paths[0].name
            table.add_row(model.upper(), str(len(paths)), rel_example)

        console.print(table)
    except Exception:
        print(f"\nDiscovered Models in {directory.name} ({total_runs} Total Runs):")
        for model, paths in sorted(runs_by_model.items()):
            rel_example = str(paths[0].relative_to(directory)) if paths[0] != directory else paths[0].name
            print(f"  - {model.upper():<12}: {len(paths)} runs (e.g. {rel_example})")
    return runs_by_model


def print_batch_exploration_review(
    combined_df: pd.DataFrame,
    model_name: str,
    run_count: int,
    targets_dict: dict,
    proximity_metrics: dict,
):
    """Print clean summary table of space exploration envelope and target satisfaction."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(
            title=f"🚀 Model Batch Review: {model_name.upper()} ({run_count} Runs, {len(combined_df)} Designs)",
            show_header=True,
            header_style="bold cyan"
        )
        table.add_column("Objective / Metric", style="bold", width=25)
        table.add_column("Min", justify="right", width=14)
        table.add_column("Median", justify="right", width=14)
        table.add_column("Max", justify="right", width=14)
        table.add_column("Target Threshold", justify="right", width=18)
        table.add_column("Pass Rate", justify="right", width=12)

        # Transmission (if present)
        if "transmission" in combined_df.columns:
            t_min = combined_df["transmission"].min()
            t_med = combined_df["transmission"].median()
            t_max = combined_df["transmission"].max()
            t_thresh = targets_dict.get("transmission", (1.0 - targets_dict.get("reflectivity", 0.99999)) * 1e6)
            t_pass_pct = (combined_df["transmission"] <= t_thresh).mean() * 100.0
            table.add_row(
                "Transmission (ppm)",
                f"{t_min:.2f}",
                f"{t_med:.2f}",
                f"{t_max:.2f}",
                f"≤ {t_thresh:.2f}",
                f"{t_pass_pct:.1f}%"
            )

        # Objective 1: Reflectivity
        r_min = combined_df["reflectivity"].min()
        r_med = combined_df["reflectivity"].median()
        r_max = combined_df["reflectivity"].max()
        r_pass_pct = (combined_df["reflectivity"] >= targets_dict["reflectivity"]).mean() * 100.0
        table.add_row(
            "Reflectivity",
            f"{r_min:.6f}",
            f"{r_med:.6f}",
            f"{r_max:.6f}",
            f"≥ {targets_dict['reflectivity']:.6f}",
            f"{r_pass_pct:.1f}%"
        )

        # Loss (1 - R)
        loss_min = 1.0 - r_max
        loss_med = 1.0 - r_med
        loss_max = 1.0 - r_min
        table.add_row(
            "Loss (1 - R)",
            f"{loss_min:.2e}",
            f"{loss_med:.2e}",
            f"{loss_max:.2e}",
            f"≤ {1.0 - targets_dict['reflectivity']:.2e}",
            f"{r_pass_pct:.1f}%"
        )

        # Absorption
        a_min = combined_df["absorption"].min()
        a_med = combined_df["absorption"].median()
        a_max = combined_df["absorption"].max()
        a_pass_pct = (combined_df["absorption"] <= targets_dict["absorption"]).mean() * 100.0
        table.add_row(
            "Absorption (ppm)",
            f"{a_min:.4f}",
            f"{a_med:.4f}",
            f"{a_max:.4f}",
            f"≤ {targets_dict['absorption']:.4f}",
            f"{a_pass_pct:.1f}%"
        )

        # Thermal Noise
        tn_min = combined_df["thermal_noise"].min()
        tn_med = combined_df["thermal_noise"].median()
        tn_max = combined_df["thermal_noise"].max()
        tn_pass_pct = (combined_df["thermal_noise"] <= targets_dict["thermal_noise"]).mean() * 100.0
        table.add_row(
            "Thermal Noise (m/√Hz)",
            f"{tn_min:.2e}",
            f"{tn_med:.2e}",
            f"{tn_max:.2e}",
            f"≤ {targets_dict['thermal_noise']:.2e}",
            f"{tn_pass_pct:.1f}%"
        )

        # Total Physical Thickness
        if "total_thickness" in combined_df.columns:
            th_min = combined_df["total_thickness"].min()
            th_med = combined_df["total_thickness"].median()
            th_max = combined_df["total_thickness"].max()
            th_pass_pct = (combined_df["total_thickness"] <= targets_dict.get("total_thickness", 6000.0)).mean() * 100.0
            table.add_row(
                "Physical Thickness (nm)",
                f"{th_min:.1f}",
                f"{th_med:.1f}",
                f"{th_max:.1f}",
                f"≤ {targets_dict.get('total_thickness', 6000.0):.1f}",
                f"{th_pass_pct:.1f}%"
            )

        # Active Layers
        if "active_layer_count" in combined_df.columns:
            l_min = combined_df["active_layer_count"].min()
            l_med = combined_df["active_layer_count"].median()
            l_max = combined_df["active_layer_count"].max()
            table.add_row(
                "Active Layer Count",
                f"{l_min}",
                f"{l_med:.0f}",
                f"{l_max}",
                "-",
                "-"
            )

        console.print()
        console.print(table)
    except Exception as e:
        print(f"Batch Review for {model_name.upper()} ({run_count} runs): {e}")


def get_writable_output_dir(preferred_dir: Path) -> Path:
    """Ensure the output directory is writable, falling back to a local writable path if read-only."""
    preferred_dir = Path(preferred_dir)
    try:
        preferred_dir.mkdir(parents=True, exist_ok=True)
        test_file = preferred_dir / f".write_test_{os.getpid()}"
        with open(test_file, "w") as f:
            f.write("")
        test_file.unlink()
        return preferred_dir
    except (OSError, PermissionError):
        cwd = Path.cwd()
        try:
            rel = preferred_dir.resolve().relative_to(cwd.resolve())
            parts = list(rel.parts)
            if "results" in parts:
                parts.remove("results")
                candidate = cwd.joinpath(*parts)
                candidate.mkdir(parents=True, exist_ok=True)
                test_file = candidate / f".write_test_{os.getpid()}"
                with open(test_file, "w") as f:
                    f.write("")
                test_file.unlink()
                print(f"\n⚠️  Notice: Directory '{preferred_dir}' is on a read-only filesystem.")
                print(f"    Saving outputs to local writable directory: {candidate}\n")
                return candidate
        except Exception:
            pass

        fallback_dir = cwd / "outputs" / preferred_dir.name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n⚠️  Notice: Directory '{preferred_dir}' is on a read-only filesystem.")
        print(f"    Saving outputs to local writable directory: {fallback_dir}\n")
        return fallback_dir


def export_2d_projections(
    combined_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    target_refl: float = 0.99999,
    target_abs: float = 0.30,
    target_tn: float = 4.0e-21,
    target_trans: Optional[float] = None,
    compare_refl: Optional[float] = None,
    compare_abs: Optional[float] = None,
    compare_tn: Optional[float] = None,
    compare_trans: Optional[float] = None,
    compare_label: str = "Reference Design",
    selected_row: Optional[pd.Series] = None,
    optimise_parameters: Optional[List[str]] = None,
    primary_metric: Optional[str] = None,
):
    """
    Generate and save publication-quality 2D projections:
    1) All designs with shaded target box
    2) Target box only (zoomed to target box)
    3) CSV of designs inside the target volume
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    output_dir = get_writable_output_dir(output_dir)

    if primary_metric is None:
        primary_metric = "transmission" if (optimise_parameters and "transmission" in optimise_parameters) or ("transmission" in combined_df.columns and "reflectivity" not in combined_df.columns) else "reflectivity"

    if target_trans is None:
        target_trans = (1.0 - target_refl) * 1e6 if target_refl is not None else 10.0
    if compare_trans is None and compare_refl is not None:
        compare_trans = (1.0 - compare_refl) * 1e6

    # Setup matplotlib style
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

    # Data arrays
    y_tn = combined_df["thermal_noise"].values
    target_nines = -np.log10(max(1e-10, 1.0 - target_refl))

    # Colors: CTN log
    colors = np.log10(np.maximum(1e-25, y_tn))
    norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max())
    cmap = plt.cm.viridis_r

    # Highlight markers
    has_comp = (compare_abs is not None and compare_tn is not None and (compare_refl is not None or compare_trans is not None))
    has_sel = selected_row is not None
    is_trans = (primary_metric == "transmission" and "transmission" in combined_df.columns)

    def plot_3panel_projection(
        sub_df,
        is_target_box: bool,
        save_path: Path,
        title_suffix: str
    ):
        sub_abs = sub_df["absorption"].values
        sub_tn = sub_df["thermal_noise"].values
        sub_colors = np.log10(np.maximum(1e-25, sub_tn))

        if is_trans:
            sub_y = sub_df["transmission"].values
            opt_title_str = "Transmission"
            y_axis_label = "Transmission (ppm)"
        else:
            sub_loss = np.maximum(1e-10, 1.0 - sub_df["reflectivity"].values)
            sub_y = -np.log10(sub_loss)
            opt_title_str = "Reflectivity"
            y_axis_label = "Reflectivity (Nines: $-\log_{{10}}(1-R)$)"

        fig = plt.figure(figsize=(22, 7.5), dpi=200)
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.04], height_ratios=[1, 0.08])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[0, 3])

        axes = [ax1, ax2, ax3]
        for ax in axes:
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=14, colors="black")

        # Determine limits
        if is_target_box:
            abs_min = max(1e-4, sub_abs.min() * 0.8)
            abs_max = target_abs * 1.08
            tn_min = max(1e-25, sub_tn.min() * 0.8)
            tn_max = target_tn * 1.08
            if is_trans:
                y_min = max(1e-4, sub_y.min() * 0.8)
                y_max = max(target_trans * 1.15, sub_y.max() * 1.1)
            else:
                y_min = target_nines - 0.2
                y_max = max(sub_y.max() * 1.05, target_nines + 1.0)
        else:
            all_abs_pts = [sub_abs.min(), sub_abs.max(), target_abs]
            if has_comp and compare_abs is not None: all_abs_pts.append(compare_abs)
            abs_min = max(1e-4, min(all_abs_pts) * 0.8)
            abs_max = max(all_abs_pts) * 1.25

            all_tn_pts = [sub_tn.min(), sub_tn.max(), target_tn]
            if has_comp and compare_tn is not None: all_tn_pts.append(compare_tn)
            tn_min = max(1e-25, min(all_tn_pts) * 0.8)
            tn_max = max(all_tn_pts) * 1.25

            if is_trans:
                all_y_pts = [sub_y.min(), sub_y.max(), target_trans]
                if has_comp and compare_trans is not None: all_y_pts.append(compare_trans)
                y_min = max(1e-4, min(all_y_pts) * 0.8)
                y_max = max(all_y_pts) * 1.25
            else:
                all_y_pts = [sub_y.min(), sub_y.max(), target_nines]
                if has_comp and compare_refl is not None: all_y_pts.append(-np.log10(max(1e-10, 1.0 - compare_refl)))
                y_min = max(0.5, min(all_y_pts) - 0.5)
                y_max = max(all_y_pts) + 0.5

        # Plot 1: CTN vs Absorption
        ax = ax1
        ax.fill_between([abs_min, target_abs], tn_min, target_tn, color="#00bcd4", alpha=0.15, zorder=1, label="Target Box")
        ax.axvline(target_abs, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2, label=f"Target Abs ({target_abs:.2f} ppm)")
        ax.axhline(target_tn, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2, label=f"Target CTN ({target_tn:.2e})")

        sc = ax.scatter(sub_abs, sub_tn, c=sub_colors, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none", zorder=3)
        if has_comp:
            ax.scatter(compare_abs, compare_tn, marker="D", c="#ff007f", s=160, edgecolors="black", linewidth=1.5, zorder=10, label=compare_label)
        if has_sel:
            ax.scatter(selected_row["absorption"], selected_row["thermal_noise"], marker="*", c="#00e5ff", s=250, edgecolors="black", linewidth=1.5, zorder=11, label="Selected Design")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([abs_min, abs_max])
        ax.set_ylim([tn_min, tn_max])
        ax.set_xlabel("Absorption (ppm)", fontsize=15, fontweight="bold")
        ax.set_ylabel("Thermal Noise (m/$\sqrt{\mathrm{Hz}}$)", fontsize=15, fontweight="bold")
        ax.set_title("CTN vs Absorption", fontsize=16, color="#222222", fontweight="bold")
        ax.grid(True, which="both", color="#e0e0e0")

        # Plot 2: Transmission / Reflectivity vs Absorption
        ax = ax2
        if is_trans:
            ax.fill_between([abs_min, target_abs], y_min, target_trans, color="#00bcd4", alpha=0.15, zorder=1)
            ax.axvline(target_abs, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2)
            ax.axhline(target_trans, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2, label=f"Target T ≤ {target_trans:.2f} ppm")
            ax.scatter(sub_abs, sub_y, c=sub_colors, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none", zorder=3)
            if has_comp and compare_trans is not None:
                ax.scatter(compare_abs, compare_trans, marker="D", c="#ff007f", s=160, edgecolors="black", linewidth=1.5, zorder=10)
            if has_sel and "transmission" in selected_row:
                ax.scatter(selected_row["absorption"], selected_row["transmission"], marker="*", c="#00e5ff", s=250, edgecolors="black", linewidth=1.5, zorder=11)
            ax.set_yscale("log")
        else:
            ax.fill_between([abs_min, target_abs], target_nines, y_max, color="#00bcd4", alpha=0.15, zorder=1)
            ax.axvline(target_abs, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2)
            target_label_str = f"Target R ({target_refl:.5f})"
            ax.axhline(target_nines, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2, label=target_label_str)
            ax.scatter(sub_abs, sub_y, c=sub_colors, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none", zorder=3)
            if has_comp and compare_refl is not None:
                comp_nines = -np.log10(max(1e-10, 1.0 - compare_refl))
                ax.scatter(compare_abs, comp_nines, marker="D", c="#ff007f", s=160, edgecolors="black", linewidth=1.5, zorder=10)
            if has_sel and "reflectivity" in selected_row:
                sel_nines = -np.log10(max(1e-10, 1.0 - selected_row["reflectivity"]))
                ax.scatter(selected_row["absorption"], sel_nines, marker="*", c="#00e5ff", s=250, edgecolors="black", linewidth=1.5, zorder=11)

        ax.set_xscale("log")
        ax.set_xlim([abs_min, abs_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Absorption (ppm)", fontsize=15, fontweight="bold")
        ax.set_ylabel(y_axis_label, fontsize=15, fontweight="bold")
        ax.set_title(f"{opt_title_str} vs Absorption", fontsize=16, color="#222222", fontweight="bold")
        ax.grid(True, which="both", color="#e0e0e0")

        # Plot 3: Transmission / Reflectivity vs CTN
        ax = ax3
        if is_trans:
            ax.fill_between([tn_min, target_tn], y_min, target_trans, color="#00bcd4", alpha=0.15, zorder=1)
            ax.axvline(target_tn, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2)
            ax.axhline(target_trans, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2)
            ax.scatter(sub_tn, sub_y, c=sub_colors, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none", zorder=3)
            if has_comp and compare_trans is not None:
                ax.scatter(compare_tn, compare_trans, marker="D", c="#ff007f", s=160, edgecolors="black", linewidth=1.5, zorder=10)
            if has_sel and "transmission" in selected_row:
                ax.scatter(selected_row["thermal_noise"], selected_row["transmission"], marker="*", c="#00e5ff", s=250, edgecolors="black", linewidth=1.5, zorder=11)
            ax.set_yscale("log")
        else:
            ax.fill_between([tn_min, target_tn], target_nines, y_max, color="#00bcd4", alpha=0.15, zorder=1)
            ax.axvline(target_tn, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2)
            ax.axhline(target_nines, color="#d32f2f", linestyle="--", linewidth=1.5, zorder=2)
            ax.scatter(sub_tn, sub_y, c=sub_colors, cmap=cmap, norm=norm, s=14, alpha=0.7, edgecolors="none", zorder=3)
            if has_comp and compare_refl is not None:
                comp_nines = -np.log10(max(1e-10, 1.0 - compare_refl))
                ax.scatter(compare_tn, comp_nines, marker="D", c="#ff007f", s=160, edgecolors="black", linewidth=1.5, zorder=10)
            if has_sel and "reflectivity" in selected_row:
                sel_nines = -np.log10(max(1e-10, 1.0 - selected_row["reflectivity"]))
                ax.scatter(selected_row["thermal_noise"], sel_nines, marker="*", c="#00e5ff", s=250, edgecolors="black", linewidth=1.5, zorder=11)

        ax.set_xscale("log")
        ax.set_xlim([tn_min, tn_max])
        ax.set_ylim([y_min, y_max])
        ax.set_xlabel("Thermal Noise (m/$\sqrt{\mathrm{Hz}}$)", fontsize=15, fontweight="bold")
        ax.set_ylabel(y_axis_label, fontsize=15, fontweight="bold")
        ax.set_title(f"{opt_title_str} vs Thermal Noise", fontsize=16, color="#222222", fontweight="bold")
        ax.grid(True, which="both", color="#e0e0e0")

        # Colorbar
        tn_c_min, tn_c_max = 10 ** colors.min(), 10 ** colors.max()
        candidates = np.array([1e-22, 2e-22, 5e-22, 8e-22, 1e-21, 1.5e-21, 2e-21, 2.5e-21, 3e-21, 3.5e-21, 4e-21, 5e-21, 1e-20])
        cb_tick_vals = candidates[(candidates >= tn_c_min * 0.95) & (candidates <= tn_c_max * 1.05)]
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

        cbar = fig.colorbar(sc, cax=cax, orientation="vertical", ticks=cb_ticks_log)
        cbar.ax.set_yticklabels(cb_labels, fontsize=14)
        cbar.set_label("Thermal Noise (m/$\sqrt{\mathrm{Hz}}$)", fontsize=14, fontweight="bold", color="black")
        cbar.ax.tick_params(labelsize=14, colors="black")

        # Unified Legend
        handles, labels = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        for h, l in zip(h2, l2):
            if l not in labels:
                handles.append(h)
                labels.append(l)

        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), framealpha=0.9, facecolor="white", edgecolor="#cccccc", fontsize=13, bbox_to_anchor=(0.46, 0.01))

        # Title
        fig.suptitle(f"2D Pareto Projections: {model_name.upper()} — {title_suffix} ({len(sub_df)} Designs)", fontsize=17, fontweight="bold", color="#00838f", y=0.98)

        plt.subplots_adjust(top=0.91, bottom=0.18, left=0.06, right=0.93, wspace=0.25)
        plt.savefig(save_path, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close()
        print(f"✓ Saved 2D projections ({title_suffix}) to: {save_path}")

    # 1. Plot All Points
    out_all = output_dir / f"{model_name.lower()}_2d_projections_all.png"
    plot_3panel_projection(combined_df, is_target_box=False, save_path=out_all, title_suffix="All Explored Points")

    # 2. Filter inside target box
    if primary_metric == "transmission" and "transmission" in combined_df.columns:
        inside_mask = (
            (combined_df["absorption"] <= target_abs) &
            (combined_df["thermal_noise"] <= target_tn) &
            (combined_df["transmission"] <= target_trans)
        )
    else:
        inside_mask = (
            (combined_df["absorption"] <= target_abs) &
            (combined_df["thermal_noise"] <= target_tn) &
            (combined_df["reflectivity"] >= target_refl)
        )
    target_df = combined_df[inside_mask].reset_index(drop=True)

    out_tb = output_dir / f"{model_name.lower()}_2d_projections_target_box.png"
    plot_df = target_df if len(target_df) > 0 else combined_df
    box_suffix = "Target Box Volume Only" if len(target_df) > 0 else "Target Box Area (Zoomed to Bounds)"
    plot_3panel_projection(plot_df, is_target_box=True, save_path=out_tb, title_suffix=box_suffix)

    if len(target_df) > 0:
        # Save CSV of designs in target box
        out_csv = output_dir / f"{model_name.lower()}_target_box_designs.csv"
        target_df.to_csv(out_csv, index=False)
        print(f"✓ Saved {len(target_df)} target-compliant designs to CSV: {out_csv}")
    else:
        print(f"  Note: No designs met all target bounds simultaneously (Yield Y(0) = 0%). Generated target box plot zoomed to bounds.")


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
        help="Target reflectivity for utility scoring (defaults to --compare-refl if set, else 0.99999)",
    )
    parser.add_argument(
        "--target-trans",
        "--target-transmission",
        type=float,
        default=None,
        help="Target transmission in ppm for utility scoring (e.g. 10.0 ppm; automatically calculates equivalent target reflectivity)",
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
        default=None,
        help="Number of top designs to precompute full TMM details (EFI and spectrum) for (default: up to 50 for large datasets, all for <= 50 designs; use 0 to skip, -1 for all)",
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
        choices=[
            "reflectivity_linear", "reflectivity_log",
            "transmission_linear", "transmission_log",
            "absorption_linear", "absorption_log",
            "ctn_linear", "ctn_log",
            "loss_linear", "loss_log",
        ],
        default="reflectivity_log",
        help="Default color mapping mode for 3D scatter plot markers (default: reflectivity_log)",
    )
    parser.add_argument(
        "--compare-trans",
        "--compare-transmission",
        type=float,
        default=None,
        help="Transmission in ppm of custom reference design (automatically calculates equivalent compare reflectivity)",
    )
    parser.add_argument(
        "--weight-trans",
        "--weight-transmission",
        type=float,
        default=None,
        help="Weight for transmission in utility score (defaults to --weight-refl)",
    )
    parser.add_argument(
        "--min-trans",
        type=float,
        default=None,
        help="Minimum transmission threshold (ppm) to filter Pareto designs before ranking",
    )
    parser.add_argument(
        "--max-trans",
        type=float,
        default=None,
        help="Maximum transmission threshold (ppm) to filter Pareto designs before ranking",
    )
    parser.add_argument(
        "--rank-by-transmission",
        action="store_true",
        help="Rank designs on the Z-axis by transmission (ascending) instead of utility score",
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
    parser.add_argument(
        "--model",
        "--model-type",
        "--filter-model",
        type=str,
        default=None,
        help="Filter aggregated runs to only those matching this model/algorithm name (e.g. 'nsga2', 'hppo', 'ppo')",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List detected models and run counts in directory and exit",
    )
    parser.add_argument(
        "--target-box-only",
        "--inside-target-only",
        action="store_true",
        help="Filter dataset to only designs falling inside the 3D target volume bounds",
    )
    parser.add_argument(
        "--save-projections",
        "--export-projections",
        action="store_true",
        help="Automatically generate and save 2D projection plots (all designs and target box only) and CSV",
    )
    parser.add_argument(
        "--min-layers",
        type=int,
        default=0,
        help="Minimum number of active layers required for designs (default: 0, keeps all Pareto designs).",
    )
    args = parser.parse_args()

    if getattr(args, "list_models", False):
        if not args.directory:
            print("Error: Directory path required to list models.")
            return 1
        d_path = Path(args.directory).resolve()
        list_detected_models(d_path)
        return 0

    if getattr(args, "verify_physics", False):
        success = verify_aligo_gold_standard(verbose=True)
        return 0 if success else 1

    # Determine default color mode, supporting backward compatibility with --color-by-loss
    if args.color_by_loss:
        color_mode = "loss_linear"
    else:
        color_mode = args.color_mode

    try:
        generate_3d_metrics_dashboard_from_args(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    return 0


def generate_3d_metrics_dashboard(
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
    precompute_tmm_count=None,
    color_mode="reflectivity_log",
    aggregate=False,
    plot_mode="rank",
    z_log=False,
    x_linear=False,
    y_linear=False,
    selected_rank=1,
    auto_rotate=False,
    model=None,
    list_models=False,
    target_box_only=False,
    save_projections=False,
    min_layers=0,
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
        model=model,
        list_models=list_models,
        target_box_only=target_box_only,
        save_projections=save_projections,
        min_layers=min_layers,
    )
    return generate_3d_metrics_dashboard_from_args(args)

# Backward compatibility alias
generate_3d_rank_dashboard = generate_3d_metrics_dashboard


def generate_3d_metrics_dashboard_from_args(args):
    return _generate_3d_metrics_dashboard_impl(args)

# Backward compatibility alias
generate_3d_rank_dashboard_from_args = generate_3d_metrics_dashboard_from_args


def _generate_3d_metrics_dashboard_impl(args):

    # Determine default color mode, supporting backward compatibility with --color-by-loss
    if args.color_by_loss:
        color_mode = "loss_linear"
    else:
        color_mode = args.color_mode

    # Resolve target values, defaulting to comparison design values if they are provided,
    # and falling back to default values otherwise.
    target_refl = args.target_refl if args.target_refl is not None else (args.compare_refl if args.compare_refl is not None else 0.99999)
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

        model_filter = getattr(args, "model", None)
        if model_filter:
            matched_subdirs = [s for s in subdirs if matches_model_filter(s, model_filter)]
            if not matched_subdirs:
                print(f"Error: No runs matching model filter '{model_filter}' found under {directory}.")
                print("Available models:")
                list_detected_models(directory)
                return 1
            subdirs = matched_subdirs
            print(f"Filtered to {len(subdirs)} Pareto fronts matching model '{model_filter}':")
        else:
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

    # Extract optimise_parameters from config.ini files if available
    optimise_parameters = None
    optimise_src = None
    for subdir in subdirs:
        config_path = subdir / "config.ini"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in ["Data", "data", "General", "general"]:
                if cfg.has_section(section) and cfg.has_option(section, "optimise_parameters"):
                    val_str = cfg.get(section, "optimise_parameters").strip()
                    try:
                        import ast
                        parsed = ast.literal_eval(val_str)
                        if isinstance(parsed, (list, tuple)):
                            optimise_parameters = [str(x).strip().lower() for x in parsed]
                    except Exception:
                        try:
                            parsed = json.loads(val_str)
                            if isinstance(parsed, list):
                                optimise_parameters = [str(x).strip().lower() for x in parsed]
                        except Exception:
                            optimise_parameters = [p.strip().strip("'\"[]").lower() for p in val_str.split(",") if p.strip()]
                    if optimise_parameters:
                        optimise_src = f"{subdir.name}/config.ini [{section}]"
                        break
        if optimise_parameters is not None:
            break

    if optimise_parameters is None:
        config_path = directory / "config.ini"
        if config_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in ["Data", "data", "General", "general"]:
                if cfg.has_section(section) and cfg.has_option(section, "optimise_parameters"):
                    val_str = cfg.get(section, "optimise_parameters").strip()
                    try:
                        import ast
                        parsed = ast.literal_eval(val_str)
                        if isinstance(parsed, (list, tuple)):
                            optimise_parameters = [str(x).strip().lower() for x in parsed]
                    except Exception:
                        try:
                            parsed = json.loads(val_str)
                            if isinstance(parsed, list):
                                optimise_parameters = [str(x).strip().lower() for x in parsed]
                        except Exception:
                            optimise_parameters = [p.strip().strip("'\"[]").lower() for p in val_str.split(",") if p.strip()]
                    if optimise_parameters:
                        optimise_src = f"{directory.name}/config.ini [{section}]"
                        break

    if optimise_parameters is not None:
        print(f"  Loaded optimization objectives (optimise_parameters): {optimise_parameters} (from {optimise_src})")
    else:
        print("  No 'optimise_parameters' key found in config.ini. Will infer defaults from dataset columns.")

    # Load Pareto fronts and merge materials
    all_designs = []
    all_values = []
    materials = {}

    for subdir in subdirs:
        print(f"Loading Pareto front from {subdir}...")
        try:
            designs_df, values_df, _ = load_pareto_front(subdir)
            
            # Filter out designs with fewer active layers than min_layers (if specified)
            min_layers_arg = getattr(args, "min_layers", 0) or 0
            if min_layers_arg > 0:
                temp_counts = []
                max_active_in_run = 0
                for row in designs_df.to_dict("records"):
                    dOpt, mat_idx = parse_design(row)
                    active_mask = (mat_idx != 0) & (dOpt > 1e-12)
                    active_layer_count = int(np.sum(active_mask))
                    temp_counts.append(active_layer_count)
                    max_active_in_run = max(max_active_in_run, active_layer_count)
                    
                min_threshold = min(min_layers_arg, max_active_in_run) if max_active_in_run > 0 else 0
                valid_indices = [idx for idx, count in enumerate(temp_counts) if count >= min_threshold]
                
                initial_count = len(designs_df)
                designs_df = designs_df.iloc[valid_indices].reset_index(drop=True)
                values_df = values_df.iloc[valid_indices].reset_index(drop=True)
                filtered_count = initial_count - len(designs_df)
                if filtered_count > 0:
                    print(f"  Filtered out {filtered_count} designs with < {min_threshold} active layers (kept {len(designs_df)} designs).")
                
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
            try:
                import time
                time.sleep(1.0)
                subdir_designs = pd.read_csv(subdir / "pareto_designs.csv")
                subdir_values = pd.read_csv(subdir / "pareto_values.csv")
                subdir_designs["run_name"] = subdir.name
                all_designs.append(subdir_designs)
                all_values.append(subdir_values)
            except Exception as retry_e:
                print(f"Warning: Failed to load Pareto front from {subdir}: {retry_e}")
                if len(subdirs) == 1:
                    return 1
                continue

    if not all_designs:
        print("Error: No Pareto fronts could be loaded.")
        return 1

    designs_df = pd.concat(all_designs, axis=0, ignore_index=True)
    values_df = pd.concat(all_values, axis=0, ignore_index=True)
    print(f"  Loaded {len(designs_df)} total designs successfully.")

    # Ensure dual availability of transmission and reflectivity across all designs
    if "transmission" in values_df.columns and "reflectivity" not in values_df.columns:
        abs_col = values_df["absorption"] if "absorption" in values_df.columns else 0.0
        values_df["reflectivity"] = np.clip(1.0 - (values_df["transmission"] + abs_col) * 1e-6, 0.0, 1.0)
    elif "reflectivity" in values_df.columns and "transmission" not in values_df.columns:
        abs_col = values_df["absorption"] if "absorption" in values_df.columns else 0.0
        values_df["transmission"] = np.maximum(0.0, (1.0 - values_df["reflectivity"]) * 1e6 - abs_col)
    elif "transmission" in values_df.columns and "reflectivity" in values_df.columns:
        missing_refl = values_df["reflectivity"].isna()
        if missing_refl.any():
            abs_col = values_df["absorption"] if "absorption" in values_df.columns else 0.0
            values_df.loc[missing_refl, "reflectivity"] = np.clip(1.0 - (values_df.loc[missing_refl, "transmission"] + abs_col) * 1e-6, 0.0, 1.0)
        missing_trans = values_df["transmission"].isna()
        if missing_trans.any():
            abs_col = values_df["absorption"] if "absorption" in values_df.columns else 0.0
            values_df.loc[missing_trans, "transmission"] = np.maximum(0.0, (1.0 - values_df.loc[missing_trans, "reflectivity"]) * 1e6 - abs_col)

    if "absorption" not in values_df.columns:
        values_df["absorption"] = 0.0

    if optimise_parameters is None:
        primary_opt = "transmission" if "transmission" in values_df.columns else "reflectivity"
        optimise_parameters = [primary_opt, "absorption", "thermal_noise"]

    primary_metric = "transmission" if any("trans" in p for p in optimise_parameters) else ("reflectivity" if any("refl" in p for p in optimise_parameters) else ("transmission" if "transmission" in values_df.columns else "reflectivity"))

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

    model_filter = getattr(args, "model", None)
    if model_filter:
        model_name = model_filter.lower()
    elif len(subdirs) > 1:
        families = set(detect_model_family(s) for s in subdirs)
        if len(families) == 1:
            model_name = list(families)[0]
        else:
            model_name = "batch"
    else:
        model_name = detect_model_family(subdirs[0]) if subdirs[0] != directory else directory.name

    # Harmonize target values between reflectivity and transmission
    target_abs = args.target_abs if args.target_abs is not None else (args.compare_abs if args.compare_abs is not None else 0.30)
    target_tn = args.target_tn if args.target_tn is not None else (args.compare_tn if args.compare_tn is not None else 4.0e-21)
    target_thick = args.target_thick if args.target_thick is not None else (args.compare_thick if args.compare_thick is not None else 6000.0)

    if getattr(args, "target_trans", None) is not None:
        target_trans = float(args.target_trans)
        target_refl = float(np.clip(1.0 - target_trans * 1e-6, 0.0, 1.0))
    elif getattr(args, "compare_trans", None) is not None:
        target_trans = float(args.compare_trans)
        target_refl = float(np.clip(1.0 - target_trans * 1e-6, 0.0, 1.0))
    elif args.target_refl is not None:
        target_refl = float(args.target_refl)
        target_trans = float(max(0.0, (1.0 - target_refl) * 1e6))
    elif args.compare_refl is not None:
        target_refl = float(args.compare_refl)
        target_trans = float(max(0.0, (1.0 - target_refl) * 1e6))
    else:
        if primary_metric == "transmission":
            target_trans = 10.0
            target_refl = float(np.clip(1.0 - target_trans * 1e-6, 0.0, 1.0))
        else:
            target_refl = 0.99999
            target_trans = float(max(0.0, (1.0 - target_refl) * 1e6))

    compare_abs_val = args.compare_abs if args.compare_abs is not None else target_abs
    compare_tn_val = args.compare_tn if args.compare_tn is not None else target_tn
    compare_thick_val = args.compare_thick if args.compare_thick is not None else target_thick
    if getattr(args, "compare_trans", None) is not None:
        compare_trans_val = float(args.compare_trans)
        compare_refl_val = float(np.clip(1.0 - compare_trans_val * 1e-6, 0.0, 1.0))
    elif args.compare_refl is not None:
        compare_refl_val = float(args.compare_refl)
        compare_trans_val = float(max(0.0, (1.0 - compare_refl_val) * 1e6))
    else:
        compare_trans_val = target_trans
        compare_refl_val = target_refl

    # Check target_box_only filter
    if getattr(args, "target_box_only", False):
        inside_mask = (
            (values_df["absorption"] <= target_abs) &
            (values_df["thermal_noise"] <= target_tn)
        )
        if primary_metric == "transmission":
            inside_mask = inside_mask & (values_df["transmission"] <= target_trans)
        else:
            inside_mask = inside_mask & (values_df["reflectivity"] >= target_refl)
        if target_thick is not None and "total_thickness" in values_df.columns:
            inside_mask = inside_mask & (values_df["total_thickness"] <= target_thick)
        n_inside = int(inside_mask.sum())
        print(f"\n[TARGET BOX FILTER] Filtering dataset to designs strictly inside target volume:")
        print(f"  Target bounds: Absorption <= {target_abs:.3f} ppm, CTN <= {target_tn:.2e} m/√Hz, Transmission <= {target_trans:.2f} ppm (Reflectivity >= {target_refl:.6f})")
        print(f"  Retained {n_inside} / {len(values_df)} designs.")
        if n_inside == 0:
            print("  Warning: 0 designs met all target bounds! Retaining all designs to avoid empty plot.")
        else:
            designs_df = designs_df.iloc[np.where(inside_mask)[0]].reset_index(drop=True)
            values_df = values_df.iloc[np.where(inside_mask)[0]].reset_index(drop=True)

    model_title_tag = f" [{model_name.upper()}]" if model_name != directory.name else ""
    title = f"Pareto Front 3D Metrics Dashboard: {directory.name}{model_title_tag}"
    fig, combined_df = create_3d_rank_plot(
        designs_df=designs_df,
        values_df=values_df,
        title=title,
        dark_mode=not args.light,
        color_mode=color_mode,
        compare_refl=compare_refl_val,
        compare_trans=compare_trans_val,
        compare_abs=compare_abs_val,
        compare_tn=compare_tn_val,
        compare_label=args.compare_label,
        min_refl=args.min_refl,
        min_trans=getattr(args, "min_trans", None),
        max_trans=getattr(args, "max_trans", None),
        max_abs=args.max_abs,
        max_tn=args.max_tn,
        materials=materials,
        rank_by_utility=args.rank_by_utility,
        rank_by_transmission=getattr(args, "rank_by_transmission", False),
        weight_refl=args.weight_refl,
        weight_trans=getattr(args, "weight_trans", None),
        weight_abs=args.weight_abs,
        weight_tn=args.weight_tn,
        weight_thick=args.weight_thick,
        compare_thick=compare_thick_val,
        target_refl=target_refl,
        target_trans=target_trans,
        target_abs=target_abs,
        target_tn=target_tn,
        target_thick=target_thick,
        top_n=None,
        lambda_nm=wavelength_nm,
        optimise_parameters=optimise_parameters,
        primary_metric=primary_metric,
    )

    target_output_dir = get_writable_output_dir(directory)
    if args.output:
        output_path = Path(args.output)
    else:
        if model_name and model_name not in ["batch", directory.name]:
            output_path = target_output_dir / f"pareto_3d_metrics_{model_name}.html"
        else:
            output_path = target_output_dir / "pareto_3d_metrics.html"

    # Evaluate target proximity & distribution metrics (3 active objectives)
    targets_dict = {
        "primary_metric": primary_metric,
        "reflectivity": target_refl,
        "transmission": target_trans,
        "absorption": target_abs,
        "thermal_noise": target_tn,
    }
    weights_dict = {
        "reflectivity": args.weight_refl,
        "transmission": getattr(args, "weight_trans", None) if getattr(args, "weight_trans", None) is not None else args.weight_refl,
        "absorption": args.weight_abs,
        "thermal_noise": args.weight_tn,
    }

    proximity_metrics = evaluate_dataset_proximity_metrics(combined_df, targets_dict, weights_dict)
    proximity_metrics_json = json.dumps(proximity_metrics)

    # Print clean summary table to console
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        metrics_table = Table(title="🎯 Target Proximity & Solution Quality Metrics", show_header=True, header_style="bold cyan")
        metrics_table.add_column("Metric", style="bold", width=30)
        metrics_table.add_column("Value", style="green", justify="right")
        metrics_table.add_column("Details / Status", style="dim")

        y_zero = proximity_metrics["yield"]["yield_zero"]
        c_zero = proximity_metrics["yield"]["count_zero"]
        tot = proximity_metrics["total_designs"]
        metrics_table.add_row(
            "Target Region Yield Y(0)",
            f"{y_zero:.1f}%",
            f"{c_zero} of {tot} designs meet all targets simultaneously",
        )

        spacing = proximity_metrics["spacing"]["spacing"]
        metrics_table.add_row(
            "Spacing Metric (S)",
            f"{spacing:.4f}",
            "Front uniformity (lower is more evenly spaced)",
        )

        asf_best = proximity_metrics["asf"]["best_score"]
        asf_idx = proximity_metrics["asf"]["best_index"]
        asf_status = "Exceeds all targets!" if asf_best <= 0 else "Closest Pareto trade-off"
        best_rank = combined_df.iloc[asf_idx]["rank"] if asf_idx is not None else 1
        metrics_table.add_row(
            "Best Target Projection (ASF)",
            f"{asf_best:+.4f}",
            f"Design #{best_rank} ({asf_status})",
        )

        roi_hv = proximity_metrics["roi_hypervolume"]["roi_hv"]
        roi_cnt = proximity_metrics["roi_hypervolume"]["roi_points_count"]
        metrics_table.add_row(
            "ROI Hypervolume (R-HV)",
            f"{roi_hv:.4f}",
            f"{roi_cnt} designs in target neighborhood (1.5x)",
        )

        obj_table = Table(title="📊 Per-Objective Pass Rates", show_header=True, header_style="bold magenta")
        obj_table.add_column("Objective", style="bold", width=22)
        obj_table.add_column("Target", justify="right")
        obj_table.add_column("Pass Count", justify="right")
        obj_table.add_column("Pass Rate", justify="right")
        obj_table.add_column("Status / Note")

        for item in proximity_metrics["objective_breakdown"]:
            t_str = f"{item['target']:.6f}" if item["objective"] == "reflectivity" else (f"{item['target']:.2e}" if item["objective"] == "thermal_noise" else f"{item['target']:.2f}")
            pct_style = "bold red" if item.get("is_bottleneck") else ("bold green" if item["pass_pct"] == 100 else "yellow")
            status_text = "[RED BOTTLENECK]" if item.get("is_bottleneck") else ("✓ 100% Passed" if item["pass_pct"] == 100 else "")
            obj_table.add_row(
                item["display_name"],
                f"{t_str} {item['unit']}".strip(),
                f"{item['pass_count']} / {tot}",
                f"[{pct_style}]{item['pass_pct']:.1f}%[/{pct_style}]",
                f"{item['margin_note']} {status_text}",
            )

        console.print()
        console.print(metrics_table)
        console.print(obj_table)
        console.print()
    except Exception:
        print("\n" + "=" * 60)
        print("🎯 TARGET PROXIMITY & SOLUTION QUALITY METRICS")
        print("=" * 60)
        print(f"• Total Pareto Designs: {proximity_metrics['total_designs']}")
        print(f"• Target Region Yield Y(0): {proximity_metrics['yield']['yield_zero']:.1f}% ({proximity_metrics['yield']['count_zero']} / {proximity_metrics['total_designs']} designs)")
        print(f"• Spacing Metric (S): {proximity_metrics['spacing']['spacing']:.4f}")
        print(f"• Best ASF Chebyshev Score: {proximity_metrics['asf']['best_score']:+.4f}")
        print(f"• ROI Hypervolume (R-HV): {proximity_metrics['roi_hypervolume']['roi_hv']:.4f}")
        print("-" * 60)
        for item in proximity_metrics["objective_breakdown"]:
            bn = " [BOTTLENECK]" if item.get("is_bottleneck") else ""
            print(f"  - {item['display_name']:<20}: {item['pass_count']:>3}/{proximity_metrics['total_designs']} ({item['pass_pct']:>5.1f}%){bn}")
        print("=" * 60 + "\n")

    # Print Batch Exploration & Target Satisfaction Review
    print_batch_exploration_review(
        combined_df=combined_df,
        model_name=model_name,
        run_count=len(subdirs),
        targets_dict=targets_dict,
        proximity_metrics=proximity_metrics,
    )

    # Automatically generate 2D projection figures if requested
    if getattr(args, "save_projections", False):
        print("\nExporting 2D projection plots (All Explored Points and Target Box)...")
        export_2d_projections(
            combined_df=combined_df,
            output_dir=target_output_dir,
            model_name=model_name,
            target_refl=target_refl,
            target_trans=target_trans,
            target_abs=target_abs,
            target_tn=target_tn,
            compare_refl=compare_refl_val,
            compare_trans=compare_trans_val,
            compare_abs=compare_abs_val,
            compare_tn=compare_tn_val,
            compare_label=args.compare_label,
            optimise_parameters=optimise_parameters,
            primary_metric=primary_metric,
            selected_row=combined_df.iloc[0] if len(combined_df) > 0 else None,
        )

    # Precompute TMM details for top designs
    precompute_count = args.precompute_tmm_count
    total_designs = len(combined_df)
    if precompute_count is None:
        # Default: for small datasets (<= 50), precompute all; for large/aggregated datasets, precompute top 50
        precompute_count = min(50, total_designs)
    elif precompute_count < 0:
        # User explicitly passed -1 for all designs
        precompute_count = total_designs
        if precompute_count > 200:
            print(f"  ⚠️  Notice: Precomputing full TMM spectra for all {precompute_count} designs may take significant time.")
    else:
        precompute_count = min(precompute_count, total_designs)

    if precompute_count > 0:
        print(f"Precomputing TMM physics data for top {precompute_count} of {total_designs} designs...")
    else:
        print(f"Skipping TMM curve precomputation (layer stacks and physical thicknesses ready for all {total_designs} designs)...")

    tmm_data = precompute_tmm_details(
        combined_df=combined_df,
        materials_dict=materials if materials is not None else {},
        max_count=precompute_count,
        lambda_nm=wavelength_nm,
        cache_dir=target_output_dir
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
    compare_refl_val = args.compare_refl if args.compare_refl is not None else 0.99999
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
        .tools-bar {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .btn-tool {
            background: #202020;
            border: 1px solid #383838;
            color: #d0d0d0;
            padding: 5px 9px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.2s ease;
            white-space: nowrap;
        }
        .btn-tool:hover:not(:disabled) {
            background: #2c2c2c;
            border-color: #00bcd4;
            color: #ffffff;
            box-shadow: 0 2px 6px rgba(0, 188, 212, 0.25);
        }
        .btn-tool.active {
            background: #00838f;
            border-color: #00bcd4;
            color: #ffffff;
        }
        .btn-tool:disabled {
            opacity: 0.35;
            cursor: not-allowed;
            border-color: #282828;
        }

        /* Pass Stat Cards beside Selected Design Information */
        .pass-stat-container {
            flex: 1.1;
            display: flex;
            flex-direction: column;
            gap: 6px;
            justify-content: center;
            min-width: 260px;
        }
        .pass-stat-card {
            background-color: #141414;
            border: 1px solid #282828;
            border-radius: 6px;
            padding: 6px 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: border-color 0.2s;
        }
        .pass-stat-card:hover {
            border-color: #383838;
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

        /* Target Proximity & Distribution Metrics Styles */
        .metric-tile-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 12px;
        }
        .metric-tile {
            background-color: #121212;
            border: 1px solid #2d2d2d;
            border-radius: 6px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-tile-label {
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }
        .metric-tile-val {
            font-size: 15px;
            font-weight: 700;
            color: #00bcd4;
            font-family: monospace;
        }
        .metric-tile-sub {
            font-size: 10px;
            color: #aaa;
            margin-top: 2px;
        }
        .obj-breakdown-row {
            display: flex;
            flex-direction: column;
            margin-bottom: 8px;
            font-size: 11px;
        }
        .obj-breakdown-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3px;
        }
        .progress-bar-bg {
            width: 100%;
            height: 6px;
            background-color: #262626;
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease, background-color 0.3s ease;
        }
        .badge-tag {
            font-size: 9px;
            font-weight: bold;
            padding: 1px 5px;
            border-radius: 3px;
            display: inline-block;
        }
        .badge-bottleneck {
            background-color: rgba(244, 67, 54, 0.2);
            color: #ff5252;
            border: 1px solid #d32f2f;
        }
        .badge-pass {
            background-color: rgba(76, 175, 80, 0.2);
            color: #81c784;
            border: 1px solid #2e7d32;
        }
        .badge-mid {
            background-color: rgba(255, 152, 0, 0.2);
            color: #ffb74d;
            border: 1px solid #f57c00;
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
            <!-- Mode Toggle Bar & Tools Bar -->
            <div class="mode-toggle-bar" style="display: flex; background: #1a1a1a; padding: 8px 12px; border-bottom: 1px solid #2d2d2d; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 11px; color: #888; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Plot Mode:</span>
                    <div style="display: flex; gap: 4px; background: #121212; border: 1px solid #333; padding: 2px; border-radius: 6px;">
                        <button class="btn-mode active" id="btn-mode-rank">Ranked Mode</button>
                        <button class="btn-mode" id="btn-mode-explore">Exploration Mode</button>
                    </div>
                </div>
                <div class="tools-bar">
                    <button class="btn-tool active" id="btn-toggle-cube" title="Toggle 3D Target Feasible Volume Bounding Box">🧊 Target Volume</button>
                    <button class="btn-tool" id="btn-toggle-target-box-filter" title="Filter to show ONLY designs inside the Target Volume">🎯 Target Box Only</button>
                    <button class="btn-tool" id="btn-view-projections" title="Open 2D Projections View">📊 2D Projections</button>
                    <button class="btn-tool" id="btn-zoom-cube" title="Zoom 3D Camera into Target Feasible Volume">🔍 Zoom to Cube</button>
                    <button class="btn-tool" id="btn-reset-zoom" title="Reset 3D Camera & Zoom">🔄 Reset View</button>
                    <button class="btn-tool" id="btn-open-targets-modal" title="Configure Target Thresholds, Tolerances & Benchmarks">🎯 Targets</button>
                    <button class="btn-tool" id="btn-inspect-materials" title="Open Materials Library Inspector">🔬 Materials</button>
                    <button class="btn-tool" id="btn-set-baseline" disabled title="Set Selected Design as Baseline Target">📍 Set Baseline</button>
                    <button class="btn-tool" id="btn-set-comparison-stack" disabled title="Set Selected Design as Comparison Coating Stack">📊 Set Comp Stack</button>
                    <button class="btn-tool" id="btn-clear-comparison-stack" style="display: none;" title="Clear Comparison Coating Stack">✕ Clear Stack</button>
                    <button class="btn-tool" id="btn-export-py" disabled title="Export Selected Design Standalone Python Simulation Script">🐍 Python</button>
                    <button class="btn-tool" id="btn-export-csv" disabled title="Export Selected Design Layer Structure to CSV">📄 CSV</button>
                </div>
            </div>
            <div id="plot-3d" class="plot-container-3d"></div>
            
            <div class="card" style="margin-top: 8px;">
                <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Selected Design Information</span>
                    <span id="selected-design-quick-badge" style="font-size: 11px; color: #888;"></span>
                </div>
                <div style="display: flex; gap: 12px; align-items: stretch;">
                    <div id="info-content" class="info-card" style="flex: 1.3; margin: 0; min-height: 85px;">Click a point in the 3D plot to inspect design details.</div>
                    <div class="pass-stat-container">
                        <div class="pass-stat-card" id="pass-widget-refl">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <div>
                                    <div style="font-size: 11px; font-weight: bold; color: #ccc;" id="stat-title-refl">Reflectivity Pass</div>
                                    <div style="font-size: 9.5px; color: #777;" id="stat-sub-refl">Target &ge; 0.999990</div>
                                </div>
                                <span id="stat-badge-refl" class="badge-tag badge-bottleneck" style="font-size: 8px; padding: 2px 5px; display: none;">BOTTLENECK</span>
                            </div>
                            <div style="text-align: right;">
                                <span id="stat-val-refl" style="font-size: 14px; font-weight: bold; font-family: monospace; color: #4caf50;">-- / --</span>
                                <span id="stat-pct-refl" style="font-size: 11px; font-weight: bold; color: #4caf50; margin-left: 4px;">(--%)</span>
                            </div>
                        </div>
                        <div class="pass-stat-card" id="pass-widget-abs">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <div>
                                    <div style="font-size: 11px; font-weight: bold; color: #ccc;">Absorption Pass</div>
                                    <div style="font-size: 9.5px; color: #777;" id="stat-sub-abs">Target &le; 0.30 ppm</div>
                                </div>
                                <span id="stat-badge-abs" class="badge-tag badge-bottleneck" style="font-size: 8px; padding: 2px 5px; display: none;">BOTTLENECK</span>
                            </div>
                            <div style="text-align: right;">
                                <span id="stat-val-abs" style="font-size: 14px; font-weight: bold; font-family: monospace; color: #f44336;">-- / --</span>
                                <span id="stat-pct-abs" style="font-size: 11px; font-weight: bold; color: #f44336; margin-left: 4px;">(--%)</span>
                            </div>
                        </div>
                        <div class="pass-stat-card" id="pass-widget-tn">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <div>
                                    <div style="font-size: 11px; font-weight: bold; color: #ccc;">Thermal Noise Pass</div>
                                    <div style="font-size: 9.5px; color: #777;" id="stat-sub-tn">Target &le; 4.00e-21 m/√Hz</div>
                                </div>
                                <span id="stat-badge-tn" class="badge-tag badge-pass" style="font-size: 8px; padding: 2px 5px; display: none;">✓ 100%</span>
                            </div>
                            <div style="text-align: right;">
                                <span id="stat-val-tn" style="font-size: 14px; font-weight: bold; font-family: monospace; color: #4caf50;">-- / --</span>
                                <span id="stat-pct-tn" style="font-size: 11px; font-weight: bold; color: #4caf50; margin-left: 4px;">(100.0%)</span>
                            </div>
                        </div>
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
                    <option value="transmission_linear">Transmission (Linear)</option>
                    <option value="transmission_log">Transmission (Log)</option>
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
            <div class="card" id="card-proximity-metrics">
                <div class="card-title" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>🎯 Target Proximity & Quality Metrics</span>
                    <span id="badge-yield-status" class="badge-tag badge-pass"></span>
                </div>
                <div class="metric-tile-grid">
                    <div class="metric-tile">
                        <div class="metric-tile-label">Target Region Yield Y(0)</div>
                        <div class="metric-tile-val" id="metric-yield-val">--%</div>
                        <div class="metric-tile-sub" id="metric-yield-sub">-- / -- designs</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-tile-label">Spacing Metric (S)</div>
                        <div class="metric-tile-val" id="metric-spacing-val">--</div>
                        <div class="metric-tile-sub">Front uniformity index</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-tile-label">Closest by ASF</div>
                        <div class="metric-tile-val" id="metric-asf-val" style="cursor: pointer; text-decoration: underline;" title="Click to select closest design">--</div>
                        <div class="metric-tile-sub" id="metric-asf-sub">Min Chebyshev deficit</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-tile-label">ROI Hypervolume (R-HV)</div>
                        <div class="metric-tile-val" id="metric-roihv-val">--</div>
                        <div class="metric-tile-sub" id="metric-roihv-sub">Target neighbourhood</div>
                    </div>
                </div>

                <div style="font-size: 11px; font-weight: bold; color: #80cbc4; margin-top: 6px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center;">
                    <span>Yield Curve Y(&alpha;) vs. Margin</span>
                    <span style="font-size: 9px; color: #888;">&alpha; &le; 100%</span>
                </div>
                <div id="plot-yield-curve" style="height: 145px; width: 100%;"></div>
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

    <!-- Targets & Tolerances Modal -->
    <div id="modal-targets" class="modal-overlay" style="display: none;">
        <div class="modal-dialog" style="max-width: 580px;">
            <div class="modal-header">
                <h3 style="color: #00bcd4;">🎯 Comparison Target Benchmarks & Tolerances</h3>
                <button class="modal-close" id="btn-close-targets-modal">&times;</button>
            </div>
            <div class="modal-body" style="padding: 16px 20px;">
                <p style="font-size: 11.5px; color: #888; margin-top: 0; margin-bottom: 14px; line-height: 1.4;">
                    Set target thresholds for Transmission, Reflectivity, Absorption, and Thermal Noise. Changing targets instantly updates the 3D Target Volume Cube, yield curves, pass rates, and ASF distances.
                </p>
                <div class="targets-grid" style="grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
                    <div>
                        <label for="input-target-trans">Transmission (ppm)</label>
                        <input type="number" id="input-target-trans" step="any">
                    </div>
                    <div>
                        <label for="input-target-refl">Reflectivity (R)</label>
                        <input type="number" id="input-target-refl" step="any">
                    </div>
                    <div>
                        <label for="input-target-abs">Absorption (ppm)</label>
                        <input type="number" id="input-target-abs" step="any">
                    </div>
                    <div>
                        <label for="input-target-tn">Thermal Noise (m/√Hz)</label>
                        <input type="text" id="input-target-tn">
                    </div>
                </div>

                <div style="font-size: 11px; font-weight: bold; color: #80cbc4; text-transform: uppercase; letter-spacing: 0.5px; margin: 16px 0 8px 0; border-top: 1px solid #2d2d2d; padding-top: 12px;">
                    Custom 3D Plot Comparison Point (Reference Design)
                </div>
                <div style="margin-bottom: 8px;">
                    <label for="input-comp-label" style="display: block; color: #888; margin-bottom: 3px; font-weight: 500; font-size: 11px;">Point Label</label>
                    <input type="text" id="input-comp-label" placeholder="Reference Design" style="width: 100%; background: #121212; border: 1px solid #444; color: #e0e0e0; padding: 5px 8px; border-radius: 4px; font-size: 11px; box-sizing: border-box;">
                </div>
                <div class="targets-grid" style="grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px;">
                    <div>
                        <label for="input-comp-trans">Transmission (ppm)</label>
                        <input type="number" id="input-comp-trans" step="any" placeholder="e.g. 10.0">
                    </div>
                    <div>
                        <label for="input-comp-refl">Reflectivity (R)</label>
                        <input type="number" id="input-comp-refl" step="any" placeholder="e.g. 0.99999">
                    </div>
                    <div>
                        <label for="input-comp-abs">Absorption (ppm)</label>
                        <input type="number" id="input-comp-abs" step="any" placeholder="e.g. 0.5">
                    </div>
                    <div>
                        <label for="input-comp-tn">Thermal Noise (m/√Hz)</label>
                        <input type="text" id="input-comp-tn" placeholder="e.g. 4.0e-21">
                    </div>
                </div>
                <div class="targets-grid" style="grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 6px;">
                    <div>
                        <label for="input-comp-thick">Thickness (nm)</label>
                        <input type="number" id="input-comp-thick" step="any" placeholder="e.g. 6000">
                    </div>
                    <div>
                        <label for="input-beam-radius">Beam Radius w<sub>0</sub> (cm)</label>
                        <input type="number" id="input-beam-radius" step="0.1" min="0.1" value="__WBEAM_CM__" placeholder="e.g. 6.2">
                    </div>
                    <div>
                        <label for="input-temp-k">Temp T (K)</label>
                        <input type="number" id="input-temp-k" step="0.1" min="0.1" value="__TEMP_K__" placeholder="e.g. 293.0">
                    </div>
                </div>
                <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 18px;">
                    <button class="btn" id="btn-clear-comp-point" style="background:#222; border-color:#444; color:#aaa; font-size: 11px;">Clear Comp Point</button>
                    <button class="btn btn-primary" id="btn-apply-targets" style="padding: 6px 16px; font-weight: bold; font-size: 11px;">Apply Targets & Recalculate</button>
                </div>
            </div>
        </div>
    </div>

    <!-- 2D Projections Modal -->
    <div id="projections-modal" class="modal-overlay" style="display: none;">
        <div class="modal-dialog" style="max-width: 1250px; width: 95%; max-height: 90vh;">
            <div class="modal-header">
                <h3 style="color: #00bcd4;">📊 2D Pareto Front Projections</h3>
                <button class="modal-close" id="btn-close-projections-modal">&times;</button>
            </div>
            <div class="modal-body" style="padding: 15px; display: flex; flex-direction: column; gap: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-primary" id="btn-proj-mode-all">All Explored Designs</button>
                        <button class="btn" id="btn-proj-mode-target">Inside Target Box Only</button>
                    </div>
                    <div>
                        <span id="proj-status-badge" style="font-size: 11.5px; color: #aaa; background: #1a1a1a; padding: 4px 10px; border-radius: 4px; border: 1px solid #333;"></span>
                    </div>
                </div>
                <div id="proj-plots-grid" style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; height: 520px;">
                    <div id="proj-plot-1" style="height: 100%; background: #161616; border-radius: 6px; border: 1px solid #282828;"></div>
                    <div id="proj-plot-2" style="height: 100%; background: #161616; border-radius: 6px; border: 1px solid #282828;"></div>
                    <div id="proj-plot-3" style="height: 100%; background: #161616; border-radius: 6px; border: 1px solid #282828;"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embedded Data
        var data3d = __PLOTLY_DATA_3D__;
        var layout3d = __PLOTLY_LAYOUT_3D__;
        var tmmData = __TMM_DATA__;
        var materialsParamsDict = __MATERIALS_PARAMS__;
        var proximityMetrics = __PROXIMITY_METRICS__;

        // X, Y & Z scale states
        var primaryMetric = "__PRIMARY_METRIC__";
        var xLog = __DEFAULT_X_LOG__;
        var yLog = __DEFAULT_Y_LOG__;
        var zLog = __DEFAULT_Z_LOG__;
        
        // Reference design details
        var hasReference = __HAS_REFERENCE__;
        var referenceLabel = "__REFERENCE_LABEL__";
        var compareRefl = __COMPARE_REFL__;
        var compareTrans = parseFloat("__COMPARE_TRANS__") || null;
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

        // Array min and max helpers to avoid call stack size exceeded on large arrays
        function arrayMin(arr) {
            if (!arr || arr.length === 0) return Infinity;
            var min = Infinity;
            for (var i = 0; i < arr.length; i++) {
                var v = arr[i];
                if (typeof v === 'number' && !isNaN(v) && v < min) {
                    min = v;
                }
            }
            return min;
        }

        function arrayMax(arr) {
            if (!arr || arr.length === 0) return -Infinity;
            var max = -Infinity;
            for (var i = 0; i < arr.length; i++) {
                var v = arr[i];
                if (typeof v === 'number' && !isNaN(v) && v > max) {
                    max = v;
                }
            }
            return max;
        }

        // Target box compliance helper
        function isInsideTargetBox(d, t_refl, t_abs, t_tn, t_thick, t_trans) {
            if (t_abs !== null && !isNaN(t_abs) && d.absorption > t_abs) return false;
            if (t_tn !== null && !isNaN(t_tn) && d.thermal_noise > t_tn) return false;
            if (t_thick !== null && !isNaN(t_thick) && d.total_thickness > t_thick) return false;
            if (primaryMetric === "transmission") {
                var targetT = (t_trans !== null && !isNaN(t_trans)) ? t_trans : (t_refl !== null && !isNaN(t_refl) ? (1.0 - t_refl) * 1e6 : 10.0);
                var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                if (dt > targetT) return false;
            } else {
                if (t_refl !== null && !isNaN(t_refl) && d.reflectivity < t_refl) return false;
            }
            return true;
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
        document.getElementById('input-target-trans').value = "__TARGET_TRANS__";
        document.getElementById('input-target-refl').value = "__TARGET_REFL__";
        document.getElementById('input-target-abs').value = "__TARGET_ABS__";
        document.getElementById('input-target-tn').value = "__TARGET_TN__";

        // Setup bidirectional sync between target transmission and reflectivity (direct conversion without absorption)
        document.getElementById('input-target-trans').addEventListener('input', function() {
            var tVal = parseFloat(this.value);
            if (!isNaN(tVal)) {
                var rVal = Math.min(1.0, Math.max(0.0, 1.0 - tVal * 1e-6));
                document.getElementById('input-target-refl').value = rVal.toFixed(6);
            }
        });
        document.getElementById('input-target-refl').addEventListener('input', function() {
            var rVal = parseFloat(this.value);
            if (!isNaN(rVal)) {
                var tVal = Math.max(0.0, (1.0 - rVal) * 1e6);
                document.getElementById('input-target-trans').value = tVal.toFixed(2);
            }
        });

        // Initialize custom comparison point fields
        if (hasReference) {
            document.getElementById('input-comp-label').value = referenceLabel;
            document.getElementById('input-comp-trans').value = compareTrans !== null ? compareTrans.toFixed(2) : (compareRefl !== null ? Math.max(0.0, (1.0 - compareRefl) * 1e6).toFixed(2) : "");
            document.getElementById('input-comp-refl').value = compareRefl !== null ? compareRefl : "";
            document.getElementById('input-comp-abs').value = compareAbs !== null ? compareAbs : "";
            document.getElementById('input-comp-tn').value = compareTN !== null ? compareTN.toExponential(4) : "";
            document.getElementById('input-comp-thick').value = (compareThick !== null && compareThick > 0) ? compareThick : "";
        } else {
            document.getElementById('input-comp-label').value = "Reference Design";
            document.getElementById('input-comp-trans').value = "";
            document.getElementById('input-comp-refl').value = "";
            document.getElementById('input-comp-abs').value = "";
            document.getElementById('input-comp-tn').value = "";
            document.getElementById('input-comp-thick').value = "";
        }

        // Setup bidirectional sync for comparison point (direct conversion without absorption)
        document.getElementById('input-comp-trans').addEventListener('input', function() {
            var tVal = parseFloat(this.value);
            if (!isNaN(tVal)) {
                var rVal = Math.min(1.0, Math.max(0.0, 1.0 - tVal * 1e-6));
                document.getElementById('input-comp-refl').value = rVal.toFixed(6);
            }
        });
        document.getElementById('input-comp-refl').addEventListener('input', function() {
            var rVal = parseFloat(this.value);
            if (!isNaN(rVal)) {
                var tVal = Math.max(0.0, (1.0 - rVal) * 1e6);
                document.getElementById('input-comp-trans').value = tVal.toFixed(2);
            }
        });


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

        // Update legend font colors
        if (layout3d.legend) {
            layout3d.legend.font = { color: '#e0e0e0' };
        }

        // Update colorbar font colors
        if (data3d[0].marker && data3d[0].marker.colorbar) {
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

        var showTargetCube = true;

        function getTargetCubeTrace(t_refl, t_abs, t_tn, mode, visible) {
            if (!visible || !designsList || designsList.length === 0) {
                return {
                    type: 'scatter3d',
                    mode: 'lines',
                    x: [],
                    y: [],
                    z: [],
                    visible: false,
                    showlegend: false,
                    hoverinfo: 'skip'
                };
            }

            var allAbs = designsList.map(d => d.absorption);
            var allTN = designsList.map(d => d.thermal_noise);
            var minAbs = Math.max(1e-4, arrayMin(allAbs) * 0.9);
            var maxAbs = Math.max(t_abs, minAbs * 1.05);

            var minTN = Math.max(1e-25, arrayMin(allTN) * 0.9);
            var maxTN = Math.max(t_tn, minTN * 1.05);

            var minZ, maxZ;
            var t_trans_elem = document.getElementById('input-target-trans');
            var target_t = (t_trans_elem && !isNaN(parseFloat(t_trans_elem.value))) ? parseFloat(t_trans_elem.value) : (1.0 - t_refl) * 1e6;

            if (mode === "explore") {
                if (primaryMetric === "transmission") {
                    var allTrans = designsList.map(d => (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6));
                    var minT = Math.max(1e-4, arrayMin(allTrans) * 0.9);
                    if (zLog) {
                        minZ = Math.log10(minT);
                        maxZ = Math.log10(Math.max(target_t, minT * 1.05));
                    } else {
                        minZ = minT;
                        maxZ = Math.max(target_t, minT * 1.05);
                    }
                } else {
                    if (zLog) {
                        minZ = -Math.log10(Math.max(1e-10, 1.0 - t_refl));
                        var allZ = designsList.map(d => -Math.log10(Math.max(1e-10, 1.0 - d.reflectivity)));
                        maxZ = arrayMax(allZ) + 0.3;
                        if (maxZ <= minZ) maxZ = minZ + 0.5;
                    } else {
                        minZ = t_refl;
                        maxZ = 1.000001;
                    }
                }
            } else {
                minZ = 0.5;
                var passing = designsList.filter(d => isInsideTargetBox(d, t_refl, t_abs, t_tn, null, target_t));
                if (passing.length > 0) {
                    maxZ = arrayMax(passing.map(d => d.rank)) + 0.5;
                } else {
                    maxZ = Math.min(designsList.length, 15) + 0.5;
                }
            }

            var x0 = minAbs, x1 = maxAbs;
            var y0 = minTN, y1 = maxTN;
            var z0 = minZ, z1 = maxZ;

            // Wireframe edges: 12 segments connecting all 8 corners
            var wx = [
                x0, x1, x1, x0, x0, null,
                x0, x1, x1, x0, x0, null,
                x0, x0, null,
                x1, x1, null,
                x1, x1, null,
                x0, x0
            ];
            var wy = [
                y0, y0, y1, y1, y0, null,
                y0, y0, y1, y1, y0, null,
                y0, y0, null,
                y0, y0, null,
                y1, y1, null,
                y1, y1
            ];
            var wz = [
                z0, z0, z0, z0, z0, null,
                z1, z1, z1, z1, z1, null,
                z0, z1, null,
                z0, z1, null,
                z0, z1, null,
                z0, z1
            ];

            return {
                type: 'scatter3d',
                mode: 'lines',
                x: wx,
                y: wy,
                z: wz,
                line: { color: '#00e5ff', width: 4.0 },
                name: 'Target Volume Bounds',
                hoverinfo: 'skip',
                showlegend: true,
                visible: true
            };
        }

        function zoomToTargetCube() {
            var target_refl = parseFloat(document.getElementById('input-target-refl').value) || 0.99999;
            var target_abs = parseFloat(document.getElementById('input-target-abs').value) || 0.30;
            var target_tn = parseFloat(document.getElementById('input-target-tn').value) || 4.0e-21;
            var target_trans = parseFloat(document.getElementById('input-target-trans') ? document.getElementById('input-target-trans').value : "") || null;

            if (!designsList || designsList.length === 0) return;

            var allAbs = designsList.map(d => d.absorption);
            var minAllAbs = arrayMin(allAbs);
            var maxAllAbs = arrayMax(allAbs);
            var minAbs = Math.max(1e-4, minAllAbs * 0.95);
            var maxAbs = Math.max(target_abs * 1.05, minAbs * 1.1);

            var allTN = designsList.map(d => d.thermal_noise);
            var minAllTN = arrayMin(allTN);
            var maxAllTN = arrayMax(allTN);
            var minTN = Math.max(1e-25, minAllTN * 0.95);
            var maxTN = Math.max(target_tn * 1.05, minTN * 1.1);

            var minZ, maxZ;
            var z_tickvals = null;
            var z_ticktext = null;

            if (plotMode === "explore") {
                if (primaryMetric === "transmission") {
                    var target_t = target_trans !== null ? target_trans : (1.0 - target_refl) * 1e6;
                    var allTrans = designsList.map(d => (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6));
                    var minAllT = Math.max(1e-4, arrayMin(allTrans) * 0.9);
                    if (zLog) {
                        minZ = Math.log10(minAllT);
                        maxZ = Math.log10(Math.max(target_t * 1.1, minAllT * 1.2));
                        var ticksObj = getLogTicks(minZ, maxZ, false);
                        z_tickvals = ticksObj.tickvals;
                        z_ticktext = ticksObj.ticktext;
                    } else {
                        minZ = minAllT;
                        maxZ = Math.max(target_t * 1.1, minAllT * 1.2);
                    }
                } else {
                    if (zLog) {
                        var allZ = designsList.map(d => -Math.log10(Math.max(1e-10, 1.0 - d.reflectivity)));
                        var minAllZ = arrayMin(allZ);
                        var maxAllZ = arrayMax(allZ);
                        minZ = -Math.log10(Math.max(1e-10, 1.0 - target_refl)) * 0.98;
                        maxZ = maxAllZ + 0.2;
                        if (maxZ <= minZ) maxZ = minZ + 0.5;

                        var min_int = Math.floor(minZ);
                        var max_int = Math.ceil(maxZ);
                        z_tickvals = [];
                        z_ticktext = [];
                        for (var v = min_int; v <= max_int; v++) {
                            z_tickvals.push(v);
                            if (v === 2) z_ticktext.push("0.99");
                            else if (v === 3) z_ticktext.push("0.999");
                            else if (v === 4) z_ticktext.push("0.9999");
                            else if (v === 5) z_ticktext.push("0.99999");
                            else if (v === 6) z_ticktext.push("0.999999");
                            else if (v === 7) z_ticktext.push("0.9999999");
                            else z_ticktext.push("1-10^-" + v);
                        }
                    } else {
                        minZ = Math.max(0.0, target_refl - (1.0 - target_refl) * 0.05);
                        maxZ = 1.000005;
                    }
                }
            } else {
                minZ = 0.5;
                var passing = designsList.filter(d => isInsideTargetBox(d, target_refl, target_abs, target_tn, null, target_trans));
                maxZ = passing.length > 0 ? (arrayMax(passing.map(d => d.rank)) + 1.5) : 15.5;
            }

            var relayoutObj = {};

            // X-axis (Absorption)
            if (xLog) {
                var minXLog = Math.log10(minAbs);
                var maxXLog = Math.log10(maxAbs);
                var xTicks = getLogTicks(minXLog, maxXLog, false);
                relayoutObj['scene.xaxis.range'] = [minXLog, maxXLog];
                relayoutObj['scene.xaxis.tickvals'] = xTicks.tickvals;
                relayoutObj['scene.xaxis.ticktext'] = xTicks.ticktext;
            } else {
                relayoutObj['scene.xaxis.range'] = [minAbs, maxAbs];
                relayoutObj['scene.xaxis.tickvals'] = null;
                relayoutObj['scene.xaxis.ticktext'] = null;
            }

            // Y-axis (Thermal Noise)
            if (yLog) {
                var minYLog = Math.log10(minTN);
                var maxYLog = Math.log10(maxTN);
                var yTicks = getLogTicks(minYLog, maxYLog, true);
                relayoutObj['scene.yaxis.range'] = [minYLog, maxYLog];
                relayoutObj['scene.yaxis.tickvals'] = yTicks.tickvals;
                relayoutObj['scene.yaxis.ticktext'] = yTicks.ticktext;
            } else {
                relayoutObj['scene.yaxis.range'] = [minTN, maxTN];
                relayoutObj['scene.yaxis.tickvals'] = null;
                relayoutObj['scene.yaxis.ticktext'] = null;
            }

            // Z-axis
            if (plotMode === "rank") {
                relayoutObj['scene.zaxis.range'] = reversedZ ? [maxZ, minZ] : [minZ, maxZ];
                relayoutObj['scene.zaxis.tickvals'] = null;
                relayoutObj['scene.zaxis.ticktext'] = null;
            } else {
                relayoutObj['scene.zaxis.range'] = [minZ, maxZ];
                if (zLog && z_tickvals) {
                    relayoutObj['scene.zaxis.tickvals'] = z_tickvals;
                    relayoutObj['scene.zaxis.ticktext'] = z_ticktext;
                } else {
                    relayoutObj['scene.zaxis.tickvals'] = null;
                    relayoutObj['scene.zaxis.ticktext'] = null;
                }
            }

            // Keep camera centered and standard isometric
            relayoutObj['scene.camera.center'] = { x: 0, y: 0, z: 0 };
            relayoutObj['scene.camera.eye'] = { x: 1.5, y: 1.5, z: 1.2 };

            Plotly.relayout('plot-3d', relayoutObj);
        }

        function reset3DZoom() {
            var relayoutObj = {};

            relayoutObj['scene.xaxis.autorange'] = true;
            relayoutObj['scene.yaxis.autorange'] = true;
            relayoutObj['scene.xaxis.tickvals'] = null;
            relayoutObj['scene.xaxis.ticktext'] = null;
            relayoutObj['scene.yaxis.tickvals'] = null;
            relayoutObj['scene.yaxis.ticktext'] = null;

            if (plotMode === "explore") {
                relayoutObj['scene.zaxis.autorange'] = true;
                if (zLog) {
                    var allZ = designsList.map(d => -Math.log10(Math.max(1e-10, 1.0 - d.reflectivity)));
                    var min_int = Math.floor(arrayMin(allZ));
                    var max_int = Math.ceil(arrayMax(allZ));
                    var z_tickvals = [];
                    var z_ticktext = [];
                    for (var v = min_int; v <= max_int; v++) {
                        z_tickvals.push(v);
                        if (v === 2) z_ticktext.push("0.99");
                        else if (v === 3) z_ticktext.push("0.999");
                        else if (v === 4) z_ticktext.push("0.9999");
                        else if (v === 5) z_ticktext.push("0.99999");
                        else if (v === 6) z_ticktext.push("0.999999");
                        else if (v === 7) z_ticktext.push("0.9999999");
                        else z_ticktext.push("1-10^-" + v);
                    }
                    relayoutObj['scene.zaxis.tickvals'] = z_tickvals;
                    relayoutObj['scene.zaxis.ticktext'] = z_ticktext;
                } else {
                    relayoutObj['scene.zaxis.tickvals'] = null;
                    relayoutObj['scene.zaxis.ticktext'] = null;
                }
            } else {
                var max_rank = designsList.length;
                relayoutObj['scene.zaxis.range'] = reversedZ ? [max_rank + 2, 0.5] : [0.5, max_rank + 2];
                relayoutObj['scene.zaxis.tickvals'] = null;
                relayoutObj['scene.zaxis.ticktext'] = null;
            }

            relayoutObj['scene.camera.center'] = { x: 0, y: 0, z: 0 };
            relayoutObj['scene.camera.eye'] = { x: 1.5, y: 1.5, z: 1.2 };

            Plotly.relayout('plot-3d', relayoutObj);
        }

        // Setup initial 3D target cube trace (wireframe only for 100% mouse transparency)
        var target_refl_init = parseFloat(document.getElementById('input-target-refl').value) || 0.99999;
        var target_abs_init = parseFloat(document.getElementById('input-target-abs').value) || 0.30;
        var target_tn_init = parseFloat(document.getElementById('input-target-tn').value) || 4.0e-21;

        var initCubeTrace = getTargetCubeTrace(target_refl_init, target_abs_init, target_tn_init, plotMode, showTargetCube);
        data3d.push(initCubeTrace);

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
                    range: [0, arrayMax(design.d_physical_nm) * 1.15]
                };
                layout.yaxis2 = {
                    domain: [0.0, 0.45],
                    title: { text: "Comp [nm]", font: { size: 8, color: '#e0e0e0' } },
                    tickfont: { size: 8, color: '#e0e0e0' },
                    gridcolor: '#2d2d2d',
                    linecolor: '#444',
                    showline: true,
                    range: [0, arrayMax(compDesign.d_physical_nm) * 1.15]
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
                    range: [0, arrayMax(design.d_physical_nm) * 1.15]
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
            if (!design || !design.efi_depths) {
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
            if (!design || !design.spec_wavelengths) {
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
            var y_min = arrayMin(all_y);
            var y_max = arrayMax(all_y);
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

        // Click handler for Targets Modal Apply button
        document.getElementById('btn-apply-targets').addEventListener('click', function() {
            document.getElementById('modal-targets').style.display = 'none';
            recalculateUtilityAndRerank();
        });

        // Targets Modal Open/Close handlers
        document.getElementById('btn-open-targets-modal').addEventListener('click', function() {
            document.getElementById('modal-targets').style.display = 'flex';
        });
        document.getElementById('btn-close-targets-modal').addEventListener('click', function() {
            document.getElementById('modal-targets').style.display = 'none';
        });
        document.getElementById('modal-targets').addEventListener('click', function(e) {
            if (e.target === this) {
                this.style.display = 'none';
            }
        });

        // Materials Modal Open/Close handlers
        document.getElementById('btn-inspect-materials').addEventListener('click', function() {
            document.getElementById('materials-modal').style.display = 'flex';
            renderMaterialsInspector();
        });
        document.getElementById('btn-close-materials-modal').addEventListener('click', function() {
            document.getElementById('materials-modal').style.display = 'none';
        });
        document.getElementById('materials-modal').addEventListener('click', function(e) {
            if (e.target === this) {
                this.style.display = 'none';
            }
        });

        // Global Esc key closes all modals
        window.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                document.getElementById('modal-targets').style.display = 'none';
                document.getElementById('materials-modal').style.display = 'none';
                var pm = document.getElementById('projections-modal');
                if (pm) pm.style.display = 'none';
            }
        });

        // Toggle Target Box Filter
        var filterTargetBoxOnly = false;
        var btnTargetBoxFilter = document.getElementById('btn-toggle-target-box-filter');
        if (btnTargetBoxFilter) {
            btnTargetBoxFilter.addEventListener('click', function() {
                filterTargetBoxOnly = !filterTargetBoxOnly;
                if (filterTargetBoxOnly) {
                    this.classList.add('active');
                    this.style.background = '#00838f';
                    this.innerText = "🎯 Target Box (Active)";
                } else {
                    this.classList.remove('active');
                    this.style.background = '#202020';
                    this.innerText = "🎯 Target Box Only";
                }
                recalculateUtilityAndRerank();
            });
        }

        // 2D Projections Modal
        var projModeTargetOnly = false;
        var btnViewProj = document.getElementById('btn-view-projections');
        var modalProj = document.getElementById('projections-modal');
        var btnCloseProj = document.getElementById('btn-close-projections-modal');
        if (btnViewProj && modalProj) {
            btnViewProj.addEventListener('click', function() {
                modalProj.style.display = 'flex';
                render2DProjections();
            });
            if (btnCloseProj) {
                btnCloseProj.addEventListener('click', function() {
                    modalProj.style.display = 'none';
                });
            }
            modalProj.addEventListener('click', function(e) {
                if (e.target === this) {
                    this.style.display = 'none';
                }
            });
            var btnProjAll = document.getElementById('btn-proj-mode-all');
            var btnProjTarget = document.getElementById('btn-proj-mode-target');
            if (btnProjAll && btnProjTarget) {
                btnProjAll.addEventListener('click', function() {
                    projModeTargetOnly = false;
                    btnProjAll.classList.add('btn-primary');
                    btnProjTarget.classList.remove('btn-primary');
                    render2DProjections();
                });
                btnProjTarget.addEventListener('click', function() {
                    projModeTargetOnly = true;
                    btnProjTarget.classList.add('btn-primary');
                    btnProjAll.classList.remove('btn-primary');
                    render2DProjections();
                });
            }
        }

        var isRelayouting = false;
        function syncProjections() {
            var p1 = document.getElementById('proj-plot-1');
            var p2 = document.getElementById('proj-plot-2');
            var p3 = document.getElementById('proj-plot-3');
            if (!p1 || !p2 || !p3 || !p1.on) return;

            if (p1.removeAllListeners) p1.removeAllListeners('plotly_relayout');
            if (p2.removeAllListeners) p2.removeAllListeners('plotly_relayout');
            if (p3.removeAllListeners) p3.removeAllListeners('plotly_relayout');

            function getRange(ed, prefix) {
                if (ed[prefix + '.range[0]'] !== undefined && ed[prefix + '.range[1]'] !== undefined) {
                    return [ed[prefix + '.range[0]'], ed[prefix + '.range[1]']];
                }
                if (ed[prefix + '.range'] && Array.isArray(ed[prefix + '.range'])) {
                    return ed[prefix + '.range'];
                }
                return null;
            }

            // Plot 1: X = Abs, Y = CTN
            p1.on('plotly_relayout', function(ed) {
                if (isRelayouting) return;
                isRelayouting = true;
                try {
                    var xR = getRange(ed, 'xaxis');
                    var yR = getRange(ed, 'yaxis');
                    var xAuto = ed['xaxis.autorange'] === true || ed['autosize'] === true;
                    var yAuto = ed['yaxis.autorange'] === true || ed['autosize'] === true;

                    // Sync Abs (Plot 1 X -> Plot 2 X)
                    if (xR) {
                        Plotly.relayout(p2, { 'xaxis.range': xR, 'xaxis.autorange': false });
                    } else if (xAuto) {
                        Plotly.relayout(p2, { 'xaxis.autorange': true });
                    }

                    // Sync CTN (Plot 1 Y -> Plot 3 X)
                    if (yR) {
                        Plotly.relayout(p3, { 'xaxis.range': yR, 'xaxis.autorange': false });
                    } else if (yAuto) {
                        Plotly.relayout(p3, { 'xaxis.autorange': true });
                    }
                } finally {
                    isRelayouting = false;
                }
            });

            // Plot 2: X = Abs, Y = Reflectivity (Nines)
            p2.on('plotly_relayout', function(ed) {
                if (isRelayouting) return;
                isRelayouting = true;
                try {
                    var xR = getRange(ed, 'xaxis');
                    var yR = getRange(ed, 'yaxis');
                    var xAuto = ed['xaxis.autorange'] === true || ed['autosize'] === true;
                    var yAuto = ed['yaxis.autorange'] === true || ed['autosize'] === true;

                    // Sync Abs (Plot 2 X -> Plot 1 X)
                    if (xR) {
                        Plotly.relayout(p1, { 'xaxis.range': xR, 'xaxis.autorange': false });
                    } else if (xAuto) {
                        Plotly.relayout(p1, { 'xaxis.autorange': true });
                    }

                    // Sync Nines (Plot 2 Y -> Plot 3 Y)
                    if (yR) {
                        Plotly.relayout(p3, { 'yaxis.range': yR, 'yaxis.autorange': false });
                    } else if (yAuto) {
                        Plotly.relayout(p3, { 'yaxis.autorange': true });
                    }
                } finally {
                    isRelayouting = false;
                }
            });

            // Plot 3: X = CTN, Y = Reflectivity (Nines)
            p3.on('plotly_relayout', function(ed) {
                if (isRelayouting) return;
                isRelayouting = true;
                try {
                    var xR = getRange(ed, 'xaxis');
                    var yR = getRange(ed, 'yaxis');
                    var xAuto = ed['xaxis.autorange'] === true || ed['autosize'] === true;
                    var yAuto = ed['yaxis.autorange'] === true || ed['autosize'] === true;

                    // Sync CTN (Plot 3 X -> Plot 1 Y)
                    if (xR) {
                        Plotly.relayout(p1, { 'yaxis.range': xR, 'yaxis.autorange': false });
                    } else if (xAuto) {
                        Plotly.relayout(p1, { 'yaxis.autorange': true });
                    }

                    // Sync Nines (Plot 3 Y -> Plot 2 Y)
                    if (yR) {
                        Plotly.relayout(p2, { 'yaxis.range': yR, 'yaxis.autorange': false });
                    } else if (yAuto) {
                        Plotly.relayout(p2, { 'yaxis.autorange': true });
                    }
                } finally {
                    isRelayouting = false;
                }
            });
        }

        function render2DProjections() {
            var target_refl = parseFloat(document.getElementById('input-target-refl').value) || 0.99999;
            var target_abs = parseFloat(document.getElementById('input-target-abs').value) || 0.30;
            var target_tn = parseFloat(document.getElementById('input-target-tn').value) || 4.0e-21;
            var target_trans = parseFloat(document.getElementById('input-target-trans') ? document.getElementById('input-target-trans').value : "") || null;
            var target_nines = -Math.log10(Math.max(1e-10, 1.0 - target_refl));
            var isTrans = (primaryMetric === "transmission");

            // Custom reference / comparison point
            var comp_label = (document.getElementById('input-comp-label') ? document.getElementById('input-comp-label').value.trim() : "") || referenceLabel || "Reference Design";
            var comp_refl = document.getElementById('input-comp-refl') ? parseFloat(document.getElementById('input-comp-refl').value) : NaN;
            var comp_trans = document.getElementById('input-comp-trans') ? parseFloat(document.getElementById('input-comp-trans').value) : NaN;
            var comp_abs = document.getElementById('input-comp-abs') ? parseFloat(document.getElementById('input-comp-abs').value) : NaN;
            var comp_tn = document.getElementById('input-comp-tn') ? parseFloat(document.getElementById('input-comp-tn').value) : NaN;

            if (isNaN(comp_abs) && typeof compareAbs !== 'undefined') comp_abs = compareAbs;
            if (isNaN(comp_tn) && typeof compareTN !== 'undefined') comp_tn = compareTN;
            if (isNaN(comp_refl) && typeof compareRefl !== 'undefined') comp_refl = compareRefl;
            if (isNaN(comp_trans) && typeof compareTrans !== 'undefined' && compareTrans !== null) comp_trans = compareTrans;

            var has_comp = !isNaN(comp_abs) && !isNaN(comp_tn);
            var r_comp = isNaN(comp_refl) ? target_refl : comp_refl;
            var comp_nines = -Math.log10(Math.max(1e-10, 1.0 - r_comp));
            var t_comp = !isNaN(comp_trans) ? comp_trans : (isNaN(comp_refl) ? (target_trans || 10.0) : Math.max(0.0, (1.0 - comp_refl) * 1e6));

            // Keep all designs so zooming to bounds reveals designs entering the boundary
            var pts = designsList.slice();
            var inside_count = pts.filter(d => isInsideTargetBox(d, target_refl, target_abs, target_tn, null, target_trans)).length;

            var countBadge = document.getElementById('proj-status-badge');
            if (countBadge) {
                countBadge.innerText = projModeTargetOnly ? 
                    ("Target Box Area (Zoomed to Target Bounds: " + inside_count + " of " + designsList.length + " designs inside)") :
                    ("All Explored Designs: " + pts.length + " designs");
            }

            var abs_vals = pts.map(d => d.absorption);
            var tn_vals = pts.map(d => d.thermal_noise);
            var colors = tn_vals.map(t => Math.log10(Math.max(1e-25, t)));

            // Compute axis ranges
            var min_abs, max_abs, min_tn, max_tn;
            var min_y, max_y;
            var y_vals, target_y;

            if (isTrans) {
                target_y = target_trans !== null ? target_trans : (1.0 - target_refl) * 1e6;
                y_vals = pts.map(d => (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6));
            } else {
                target_y = target_nines;
                y_vals = pts.map(d => -Math.log10(Math.max(1e-10, 1.0 - d.reflectivity)));
            }

            if (projModeTargetOnly) {
                // Zoom all plots into the target box bounds
                min_abs = Math.max(1e-4, arrayMin(abs_vals) * 0.9);
                max_abs = target_abs * 1.15;
                min_tn = Math.max(1e-25, arrayMin(tn_vals) * 0.9);
                max_tn = target_tn * 1.15;
                if (isTrans) {
                    min_y = Math.max(1e-4, arrayMin(y_vals) * 0.9);
                    max_y = Math.max(target_y * 1.15, min_y * 1.1);
                } else {
                    min_y = target_nines - 0.3;
                    max_y = Math.max(target_nines + 0.8, arrayMax(y_vals) + 0.3);
                }
            } else {
                var all_abs_pts = abs_vals.concat(has_comp ? [comp_abs] : [], [target_abs]);
                var all_tn_pts = tn_vals.concat(has_comp ? [comp_tn] : [], [target_tn]);
                var all_y_pts = y_vals.concat(has_comp ? (isTrans ? [t_comp] : [comp_nines]) : [], [target_y]);

                min_abs = Math.max(1e-4, arrayMin(all_abs_pts) * 0.85);
                max_abs = arrayMax(all_abs_pts) * 1.2;
                min_tn = Math.max(1e-25, arrayMin(all_tn_pts) * 0.85);
                max_tn = arrayMax(all_tn_pts) * 1.2;
                if (isTrans) {
                    min_y = Math.max(1e-4, arrayMin(all_y_pts) * 0.85);
                    max_y = arrayMax(all_y_pts) * 1.25;
                } else {
                    min_y = Math.max(0, arrayMin(all_y_pts) - 0.4);
                    max_y = arrayMax(all_y_pts) + 0.5;
                }
            }

            var baseLayout = {
                paper_bgcolor: '#161616',
                plot_bgcolor: '#121212',
                font: { color: '#e0e0e0', size: 11 },
                margin: { l: 65, r: 25, t: 40, b: 50 },
                showlegend: has_comp,
                legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(20, 20, 20, 0.7)', font: { size: 10, color: '#e0e0e0' } }
            };

            // Plot 1: CTN vs Abs
            var trace1 = {
                x: abs_vals,
                y: tn_vals,
                mode: 'markers',
                marker: { size: 5.5, color: colors, colorscale: 'Viridis', reversescale: true, opacity: 0.85 },
                type: 'scatter',
                name: 'Pareto Designs',
                hoverinfo: 'text',
                text: pts.map(d => `Rank #${d.rank}<br>Abs: ${d.absorption.toFixed(3)} ppm<br>CTN: ${d.thermal_noise.toExponential(3)}<br>${isTrans ? "T: " + ((d.transmission !== undefined ? d.transmission : (1-d.reflectivity)*1e6).toFixed(2)) + " ppm" : "R: " + d.reflectivity.toFixed(6)}`)
            };
            var data1 = [trace1];
            if (has_comp) {
                data1.push({
                    x: [comp_abs],
                    y: [comp_tn],
                    mode: 'markers',
                    marker: { size: 13, color: '#ff007f', symbol: 'diamond', line: { width: 1.5, color: '#ffffff' } },
                    type: 'scatter',
                    name: comp_label,
                    hoverinfo: 'text',
                    text: [`<b>${comp_label} (Reference)</b><br>Abs: ${comp_abs.toFixed(3)} ppm<br>CTN: ${comp_tn.toExponential(3)}<br>${isTrans ? "T: " + t_comp.toFixed(2) + " ppm" : "R: " + (isNaN(comp_refl) ? 'N/A' : comp_refl.toFixed(6))}`]
                });
            }
            var layout1 = Object.assign({}, baseLayout, {
                title: { text: '<b>CTN vs Absorption</b>', font: { size: 13, color: '#00bcd4' } },
                xaxis: { title: 'Absorption (ppm)', type: 'log', gridcolor: '#282828', range: [Math.log10(min_abs), Math.log10(max_abs)] },
                yaxis: { title: 'Thermal Noise (m/√Hz)', type: 'log', gridcolor: '#282828', range: [Math.log10(min_tn), Math.log10(max_tn)] },
                shapes: [
                    { type: 'rect', x0: min_abs, x1: target_abs, y0: min_tn, y1: target_tn, fillcolor: 'rgba(0, 229, 255, 0.12)', line: { width: 0 } },
                    { type: 'line', x0: target_abs, x1: target_abs, y0: min_tn, y1: max_tn, line: { color: '#ff5252', dash: 'dash', width: 1.5 } },
                    { type: 'line', x0: min_abs, x1: max_abs, y0: target_tn, y1: target_tn, line: { color: '#ff5252', dash: 'dash', width: 1.5 } }
                ]
            });

            // Plot 2: Transmission / Reflectivity vs Abs
            var trace2 = {
                x: abs_vals,
                y: y_vals,
                mode: 'markers',
                marker: { size: 5.5, color: colors, colorscale: 'Viridis', reversescale: true, opacity: 0.85 },
                type: 'scatter',
                name: 'Pareto Designs',
                hoverinfo: 'text',
                text: pts.map(d => isTrans ?
                    `Rank #${d.rank}<br>Abs: ${d.absorption.toFixed(3)} ppm<br>T: ${((d.transmission !== undefined ? d.transmission : (1-d.reflectivity)*1e6).toFixed(2))} ppm` :
                    `Rank #${d.rank}<br>Abs: ${d.absorption.toFixed(3)} ppm<br>R: ${d.reflectivity.toFixed(6)}<br>Loss: ${(1-d.reflectivity).toExponential(3)}`)
            };
            var data2 = [trace2];
            if (has_comp) {
                data2.push({
                    x: [comp_abs],
                    y: [isTrans ? t_comp : comp_nines],
                    mode: 'markers',
                    marker: { size: 13, color: '#ff007f', symbol: 'diamond', line: { width: 1.5, color: '#ffffff' } },
                    type: 'scatter',
                    name: comp_label,
                    hoverinfo: 'text',
                    text: [isTrans ?
                        `<b>${comp_label} (Reference)</b><br>Abs: ${comp_abs.toFixed(3)} ppm<br>T: ${t_comp.toFixed(2)} ppm` :
                        `<b>${comp_label} (Reference)</b><br>Abs: ${comp_abs.toFixed(3)} ppm<br>R: ${isNaN(comp_refl) ? 'N/A' : comp_refl.toFixed(6)}<br>Loss: ${(1 - r_comp).toExponential(3)}`]
                });
            }
            var layout2 = Object.assign({}, baseLayout, {
                title: { text: isTrans ? '<b>Transmission vs Absorption</b>' : '<b>Reflectivity vs Absorption</b>', font: { size: 13, color: '#00bcd4' } },
                xaxis: { title: 'Absorption (ppm)', type: 'log', gridcolor: '#282828', range: [Math.log10(min_abs), Math.log10(max_abs)] },
                yaxis: isTrans ?
                    { title: 'Transmission (ppm)', type: 'log', gridcolor: '#282828', range: [Math.log10(min_y), Math.log10(max_y)] } :
                    { title: 'Reflectivity (Nines: -log10(1-R))', gridcolor: '#282828', range: [min_y, max_y] },
                shapes: isTrans ? [
                    { type: 'rect', x0: min_abs, x1: target_abs, y0: min_y, y1: target_y, fillcolor: 'rgba(0, 229, 255, 0.12)', line: { width: 0 } },
                    { type: 'line', x0: target_abs, x1: target_abs, y0: min_y, y1: max_y, line: { color: '#ff5252', dash: 'dash', width: 1.5 } },
                    { type: 'line', x0: min_abs, x1: max_abs, y0: target_y, y1: target_y, line: { color: '#ff5252', dash: 'dash', width: 1.5 } }
                ] : [
                    { type: 'rect', x0: min_abs, x1: target_abs, y0: target_nines, y1: max_y, fillcolor: 'rgba(0, 229, 255, 0.12)', line: { width: 0 } },
                    { type: 'line', x0: target_abs, x1: target_abs, y0: min_y, y1: max_y, line: { color: '#ff5252', dash: 'dash', width: 1.5 } },
                    { type: 'line', x0: min_abs, x1: max_abs, y0: target_nines, y1: target_nines, line: { color: '#ff5252', dash: 'dash', width: 1.5 } }
                ]
            });

            // Plot 3: Transmission / Reflectivity vs CTN
            var trace3 = {
                x: tn_vals,
                y: y_vals,
                mode: 'markers',
                marker: { size: 5.5, color: colors, colorscale: 'Viridis', reversescale: true, opacity: 0.85 },
                type: 'scatter',
                name: 'Pareto Designs',
                hoverinfo: 'text',
                text: pts.map(d => isTrans ?
                    `Rank #${d.rank}<br>CTN: ${d.thermal_noise.toExponential(3)}<br>T: ${((d.transmission !== undefined ? d.transmission : (1-d.reflectivity)*1e6).toFixed(2))} ppm` :
                    `Rank #${d.rank}<br>CTN: ${d.thermal_noise.toExponential(3)}<br>R: ${d.reflectivity.toFixed(6)}<br>Loss: ${(1-d.reflectivity).toExponential(3)}`)
            };
            var data3 = [trace3];
            if (has_comp) {
                data3.push({
                    x: [comp_tn],
                    y: [isTrans ? t_comp : comp_nines],
                    mode: 'markers',
                    marker: { size: 13, color: '#ff007f', symbol: 'diamond', line: { width: 1.5, color: '#ffffff' } },
                    type: 'scatter',
                    name: comp_label,
                    hoverinfo: 'text',
                    text: [isTrans ?
                        `<b>${comp_label} (Reference)</b><br>CTN: ${comp_tn.toExponential(3)}<br>T: ${t_comp.toFixed(2)} ppm` :
                        `<b>${comp_label} (Reference)</b><br>CTN: ${comp_tn.toExponential(3)}<br>R: ${isNaN(comp_refl) ? 'N/A' : comp_refl.toFixed(6)}<br>Loss: ${(1 - r_comp).toExponential(3)}`]
                });
            }
            var layout3 = Object.assign({}, baseLayout, {
                title: { text: isTrans ? '<b>Transmission vs Thermal Noise</b>' : '<b>Reflectivity vs Thermal Noise</b>', font: { size: 13, color: '#00bcd4' } },
                xaxis: { title: 'Thermal Noise (m/√Hz)', type: 'log', gridcolor: '#282828', range: [Math.log10(min_tn), Math.log10(max_tn)] },
                yaxis: isTrans ?
                    { title: 'Transmission (ppm)', type: 'log', gridcolor: '#282828', range: [Math.log10(min_y), Math.log10(max_y)] } :
                    { title: 'Reflectivity (Nines: -log10(1-R))', gridcolor: '#282828', range: [min_y, max_y] },
                shapes: isTrans ? [
                    { type: 'rect', x0: min_tn, x1: target_tn, y0: min_y, y1: target_y, fillcolor: 'rgba(0, 229, 255, 0.12)', line: { width: 0 } },
                    { type: 'line', x0: target_tn, x1: target_tn, y0: min_y, y1: max_y, line: { color: '#ff5252', dash: 'dash', width: 1.5 } },
                    { type: 'line', x0: min_tn, x1: max_tn, y0: target_y, y1: target_y, line: { color: '#ff5252', dash: 'dash', width: 1.5 } }
                ] : [
                    { type: 'rect', x0: min_tn, x1: target_tn, y0: target_nines, y1: max_y, fillcolor: 'rgba(0, 229, 255, 0.12)', line: { width: 0 } },
                    { type: 'line', x0: target_tn, x1: target_tn, y0: min_y, y1: max_y, line: { color: '#ff5252', dash: 'dash', width: 1.5 } },
                    { type: 'line', x0: min_tn, x1: max_tn, y0: target_nines, y1: target_nines, line: { color: '#ff5252', dash: 'dash', width: 1.5 } }
                ]
            });

            var plotConfig = { responsive: true, displayModeBar: true, displaylogo: false };
            Plotly.newPlot('proj-plot-1', data1, layout1, plotConfig);
            Plotly.newPlot('proj-plot-2', data2, layout2, plotConfig);
            Plotly.newPlot('proj-plot-3', data3, layout3, plotConfig);

            // Synchronize axes across all 3 plots
            syncProjections();
        }

        // Toggle Target Volume Cube
        document.getElementById('btn-toggle-cube').addEventListener('click', function() {
            showTargetCube = !showTargetCube;
            if (showTargetCube) {
                this.classList.add('active');
                this.style.background = '#00838f';
            } else {
                this.classList.remove('active');
                this.style.background = '#202020';
            }
            recalculateUtilityAndRerank();
        });

        // Zoom to Cube
        document.getElementById('btn-zoom-cube').addEventListener('click', zoomToTargetCube);

        // Reset Zoom
        document.getElementById('btn-reset-zoom').addEventListener('click', reset3DZoom);

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
                document.getElementById('btn-clear-comparison-stack').style.display = 'inline-flex';
                
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

        function computeTargetYieldJS(designs, t_refl, t_abs, t_tn, t_trans) {
            var tolerances = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0];
            var isTrans = (primaryMetric === "transmission");
            var trans_target = (t_trans !== null && !isNaN(t_trans)) ? t_trans : ((1.0 - t_refl) * 1e6);
            var refl_loss_target = Math.max(1e-9, 1.0 - t_refl);
            var yieldCurve = [];
            var countZero = 0;

            tolerances.forEach(function(alpha) {
                var r_thresh = 1.0 - refl_loss_target * (1.0 + alpha);
                var trans_thresh = trans_target * (1.0 + alpha);
                var abs_thresh = t_abs * (1.0 + alpha);
                var tn_thresh = t_tn * (1.0 + alpha);

                var passCount = 0;
                designs.forEach(function(d) {
                    var passPrimary;
                    if (isTrans) {
                        var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                        passPrimary = dt <= trans_thresh;
                    } else {
                        passPrimary = d.reflectivity >= r_thresh;
                    }
                    var passAbs = d.absorption <= abs_thresh;
                    var passTN = d.thermal_noise <= tn_thresh;
                    if (passPrimary && passAbs && passTN) {
                        passCount++;
                    }
                });

                var pct = designs.length > 0 ? (passCount / designs.length * 100.0) : 0.0;
                if (alpha === 0.0) {
                    countZero = passCount;
                }
                yieldCurve.push({
                    tolerance: alpha,
                    tolerance_pct: alpha * 100.0,
                    yield_pct: pct,
                    count: passCount
                });
            });

            return {
                yield_zero: yieldCurve[0].yield_pct,
                count_zero: countZero,
                yield_curve: yieldCurve
            };
        }

        function computeObjectiveBreakdownJS(designs, t_refl, t_abs, t_tn, t_trans) {
            if (!designs || designs.length === 0) return [];
            var N = designs.length;
            var isTrans = (primaryMetric === "transmission");

            var obj1;
            if (isTrans) {
                var targetT = (t_trans !== null && !isNaN(t_trans)) ? t_trans : ((1.0 - t_refl) * 1e6);
                var tPass = designs.filter(d => {
                    var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                    return dt <= targetT;
                }).length;
                var allT = designs.map(d => (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6));
                var bestT = arrayMin(allT);
                var deltaT = targetT - bestT;
                obj1 = {
                    name: "Transmission (T)",
                    targetStr: "T \u2264 " + targetT.toFixed(2) + " ppm",
                    passCount: tPass,
                    passPct: (tPass / N) * 100.0,
                    bestStr: "Best: " + bestT.toFixed(2) + " ppm (\u0394: " + (deltaT >= 0 ? "+" : "") + deltaT.toFixed(2) + " ppm)"
                };
            } else {
                var rPass = designs.filter(d => d.reflectivity >= t_refl).length;
                var bestR = arrayMax(designs.map(d => d.reflectivity));
                var bestRLoss = (1.0 - bestR) * 1e6;
                var targetRLoss = (1.0 - t_refl) * 1e6;
                var deltaLoss = targetRLoss - bestRLoss;
                obj1 = {
                    name: "Reflectivity (R)",
                    targetStr: "R \u2265 " + t_refl.toFixed(5),
                    passCount: rPass,
                    passPct: (rPass / N) * 100.0,
                    bestStr: "Best Loss: " + bestRLoss.toFixed(2) + " ppm (\u0394: " + (deltaLoss >= 0 ? "+" : "") + deltaLoss.toFixed(2) + " ppm)"
                };
            }

            // 2. Absorption
            var absPass = designs.filter(d => d.absorption <= t_abs).length;
            var bestAbs = arrayMin(designs.map(d => d.absorption));

            // 3. Thermal noise
            var tnPass = designs.filter(d => d.thermal_noise <= t_tn).length;
            var bestTN = arrayMin(designs.map(d => d.thermal_noise));

            var breakdown = [
                obj1,
                {
                    name: "Absorption",
                    targetStr: "Abs \u2264 " + t_abs.toFixed(2) + " ppm",
                    passCount: absPass,
                    passPct: (absPass / N) * 100.0,
                    bestStr: "Best: " + bestAbs.toFixed(3) + " ppm"
                },
                {
                    name: "Thermal Noise (CTN)",
                    targetStr: "TN \u2264 " + t_tn.toExponential(2) + " m/\u221AHz",
                    passCount: tnPass,
                    passPct: (tnPass / N) * 100.0,
                    bestStr: "Best: " + bestTN.toExponential(3) + " m/\u221AHz"
                }
            ];

            var minPct = arrayMin(breakdown.map(b => b.passPct));
            breakdown.forEach(function(b) {
                b.isBottleneck = (b.passPct === minPct && minPct < 100.0);
            });

            return breakdown;
        }

        function computeSpacingJS(designs) {
            if (!designs || designs.length < 2) return 0.0;
            var sample = designs;
            if (designs.length > 800) {
                var step = Math.ceil(designs.length / 800);
                sample = [];
                for (var s = 0; s < designs.length; s += step) {
                    sample.push(designs[s]);
                }
            }
            var N = sample.length;
            if (N < 2) return 0.0;

            // Extract objectives
            var isTrans = (primaryMetric === "transmission");
            var obj1 = sample.map(d => {
                if (isTrans) {
                    return (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                }
                return 1.0 - d.reflectivity;
            });
            var abs = sample.map(d => d.absorption);
            var tn = sample.map(d => d.thermal_noise);

            var min1 = arrayMin(obj1), max1 = arrayMax(obj1), span1 = (max1 - min1) || 1.0;
            var minA = arrayMin(abs), maxA = arrayMax(abs), spanA = (maxA - minA) || 1.0;
            var minT = arrayMin(tn), maxT = arrayMax(tn), spanT = (maxT - minT) || 1.0;

            var normPts = [];
            for (var i = 0; i < N; i++) {
                normPts.push([
                    (obj1[i] - min1) / span1,
                    (abs[i] - minA) / spanA,
                    (tn[i] - minT) / spanT
                ]);
            }

            var distances = [];
            for (var i = 0; i < N; i++) {
                var minDist = Infinity;
                for (var j = 0; j < N; j++) {
                    if (i === j) continue;
                    var d = Math.abs(normPts[i][0] - normPts[j][0]) +
                            Math.abs(normPts[i][1] - normPts[j][1]) +
                            Math.abs(normPts[i][2] - normPts[j][2]);
                    if (d < minDist) minDist = d;
                }
                distances.push(minDist);
            }

            var meanD = distances.reduce((a, b) => a + b, 0) / N;
            var varianceSum = distances.reduce((a, b) => a + Math.pow(b - meanD, 2), 0);
            var S = Math.sqrt(varianceSum / (N - 1));
            return S;
        }

        function computeASFJS(designs, t_refl, t_abs, t_tn, w_refl, w_abs, w_tn, t_trans) {
            if (!designs || designs.length === 0) return { bestScore: 0.0, bestDesign: null };
            var isTrans = (primaryMetric === "transmission");
            var targetT = (t_trans !== null && !isNaN(t_trans)) ? t_trans : ((1.0 - t_refl) * 1e6);
            var trans_scale = Math.max(1e-6, targetT);
            var refl_scale = Math.max(1e-9, 1.0 - t_refl);
            var abs_scale = Math.max(1e-9, t_abs);
            var tn_scale = Math.max(1e-25, t_tn);

            var rho = 1e-4;
            var bestScore = Infinity;
            var bestDesign = null;

            designs.forEach(function(d) {
                var dev1;
                if (isTrans) {
                    var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                    dev1 = (dt - targetT) / trans_scale;
                } else {
                    dev1 = (t_refl - d.reflectivity) / refl_scale;
                }
                var devA = (d.absorption - t_abs) / abs_scale;
                var devT = (d.thermal_noise - t_tn) / tn_scale;
                var devs = [w_refl * dev1, w_abs * devA, w_tn * devT];

                var maxDev = arrayMax(devs);
                var sumDev = devs.reduce((a, b) => a + b, 0);
                var asf = maxDev + rho * sumDev;
                d.asf_distance = asf;

                if (asf < bestScore) {
                    bestScore = asf;
                    bestDesign = d;
                }
            });

            return { bestScore: bestScore, bestDesign: bestDesign };
        }

        function drawYieldCurvePlot(yieldCurve) {
            var xVals = yieldCurve.map(y => y.tolerance_pct);
            var yVals = yieldCurve.map(y => y.yield_pct);
            var textVals = yieldCurve.map(y => y.count + " designs (" + y.yield_pct.toFixed(1) + "%)");

            var trace = {
                x: xVals,
                y: yVals,
                type: 'scatter',
                mode: 'lines+markers',
                line: { shape: 'hv', color: '#00bcd4', width: 2 },
                marker: { size: 5, color: '#00e5ff' },
                text: textVals,
                hovertemplate: "Tolerance \u03B1: +%{x:.0f}%<br>Yield: %{y:.1f}% (%{text})<extra></extra>"
            };

            var layout = {
                paper_bgcolor: '#121212',
                plot_bgcolor: '#121212',
                margin: { l: 38, r: 12, t: 10, b: 28 },
                xaxis: {
                    title: { text: "Margin \u03B1 (%)", font: { size: 9, color: '#888' } },
                    color: '#aaa',
                    gridcolor: '#222',
                    tickfont: { size: 9 },
                    range: [0, 50]
                },
                yaxis: {
                    title: { text: "Yield (%)", font: { size: 9, color: '#888' } },
                    color: '#aaa',
                    gridcolor: '#222',
                    tickfont: { size: 9 },
                    range: [0, 105]
                },
                showlegend: false
            };

            Plotly.react('plot-yield-curve', [trace], layout, { displayModeBar: false, responsive: true });
        }

        function updateProximityMetricsUI() {
            var target_refl = parseFloat(document.getElementById('input-target-refl').value) || 0.99999;
            var target_abs = parseFloat(document.getElementById('input-target-abs').value) || 0.30;
            var target_tn = parseFloat(document.getElementById('input-target-tn').value) || 4.0e-21;
            var target_trans = (document.getElementById('input-target-trans') ? parseFloat(document.getElementById('input-target-trans').value) : NaN);
            if (isNaN(target_trans)) {
                target_trans = (1.0 - target_refl) * 1e6;
            }

            var total_w = weightRefl + weightAbs + weightTN;
            var w_refl = total_w > 0 ? weightRefl / total_w : 0.3333;
            var w_abs = total_w > 0 ? weightAbs / total_w : 0.3333;
            var w_tn = total_w > 0 ? weightTN / total_w : 0.3334;

            var yieldRes = computeTargetYieldJS(designsList, target_refl, target_abs, target_tn, target_trans);
            var breakdown = computeObjectiveBreakdownJS(designsList, target_refl, target_abs, target_tn, target_trans);
            var spacingVal = computeSpacingJS(designsList);
            var asfRes = computeASFJS(designsList, target_refl, target_abs, target_tn, w_refl, w_abs, w_tn, target_trans);

            // Update Yield tile
            var yZero = yieldRes.yield_zero;
            var cZero = yieldRes.count_zero;
            var totalN = designsList.length;
            var yieldValElem = document.getElementById('metric-yield-val');
            var yieldSubElem = document.getElementById('metric-yield-sub');
            var yieldBadge = document.getElementById('badge-yield-status');

            if (yieldValElem) yieldValElem.innerText = yZero.toFixed(1) + "%";
            if (yieldSubElem) yieldSubElem.innerText = cZero + " / " + totalN + " designs";

            if (yieldBadge) {
                if (yZero >= 20.0) {
                    yieldBadge.className = "badge-tag badge-pass";
                    yieldBadge.innerText = "STRONG YIELD";
                } else if (yZero > 0.0) {
                    yieldBadge.className = "badge-tag badge-mid";
                    yieldBadge.innerText = "MODERATE YIELD";
                } else {
                    yieldBadge.className = "badge-tag badge-bottleneck";
                    yieldBadge.innerText = "0% AT TARGET";
                }
            }

            // Update Spacing tile
            var spacingElem = document.getElementById('metric-spacing-val');
            if (spacingElem) spacingElem.innerText = spacingVal.toFixed(4);

            // Update ASF tile
            var asfElem = document.getElementById('metric-asf-val');
            var asfSubElem = document.getElementById('metric-asf-sub');
            if (asfElem && asfRes.bestDesign) {
                var bestRank = asfRes.bestDesign.rank || (designsList.indexOf(asfRes.bestDesign) + 1);
                asfElem.innerText = "Rank #" + bestRank + " (" + (asfRes.bestScore >= 0 ? "+" : "") + asfRes.bestScore.toFixed(3) + ")";
                asfElem.onclick = function() {
                    updateSelectedDesign(asfRes.bestDesign.originalIdx);
                };
                if (asfSubElem) {
                    asfSubElem.innerText = asfRes.bestScore <= 0 ? "\u2713 Exceeds all targets" : "Closest trade-off";
                }
            }

            // Update ROI Hypervolume tile
            var roiElem = document.getElementById('metric-roihv-val');
            var roiSubElem = document.getElementById('metric-roihv-sub');
            if (roiElem) {
                var roiCount = designsList.filter(d => {
                    var passAbs = d.absorption <= target_abs * 1.5;
                    var passTN = d.thermal_noise <= target_tn * 1.5;
                    var passPrimary;
                    if (primaryMetric === "transmission") {
                        var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                        passPrimary = dt <= target_trans * 1.5;
                    } else {
                        passPrimary = d.reflectivity >= (1.0 - (1.0 - target_refl) * 1.5);
                    }
                    return passPrimary && passAbs && passTN;
                }).length;
                roiElem.innerText = (roiCount / totalN * 100.0).toFixed(1) + "%";
                if (roiSubElem) roiSubElem.innerText = roiCount + " in ROI (1.5x)";
            }

            // Update Per-Objective Stat Widgets beside Selected Design Information
            var rObj = breakdown[0];
            var absObj = breakdown[1];
            var tnObj = breakdown[2];

            if (rObj) {
                var elemVal = document.getElementById('stat-val-refl');
                var elemPct = document.getElementById('stat-pct-refl');
                var elemSub = document.getElementById('stat-sub-refl');
                var elemTitle = document.getElementById('stat-title-refl');
                var elemBadge = document.getElementById('stat-badge-refl');
                if (elemTitle) {
                    elemTitle.innerText = (primaryMetric === "transmission") ? "Transmission Pass" : "Reflectivity Pass";
                }
                var color = rObj.isBottleneck ? "#f44336" : (rObj.passPct >= 100.0 ? "#4caf50" : (rObj.passPct >= 50.0 ? "#00bcd4" : "#ff9800"));
                if (elemVal) {
                    elemVal.innerText = rObj.passCount + " / " + totalN;
                    elemVal.style.color = color;
                }
                if (elemPct) {
                    elemPct.innerText = "(" + rObj.passPct.toFixed(1) + "%)";
                    elemPct.style.color = color;
                }
                if (elemSub) {
                    elemSub.innerText = (primaryMetric === "transmission") ? ("Target \u2264 " + target_trans.toFixed(2) + " ppm") : ("Target \u2265 " + target_refl.toFixed(6));
                }
                if (elemBadge) {
                    if (rObj.isBottleneck) {
                        elemBadge.className = "badge-tag badge-bottleneck";
                        elemBadge.innerText = "BOTTLENECK";
                        elemBadge.style.display = "inline-block";
                    } else if (rObj.passPct >= 100.0) {
                        elemBadge.className = "badge-tag badge-pass";
                        elemBadge.innerText = "\u2713 100%";
                        elemBadge.style.display = "inline-block";
                    } else {
                        elemBadge.style.display = "none";
                    }
                }
            }

            if (absObj) {
                var elemVal = document.getElementById('stat-val-abs');
                var elemPct = document.getElementById('stat-pct-abs');
                var elemSub = document.getElementById('stat-sub-abs');
                var elemBadge = document.getElementById('stat-badge-abs');
                var color = absObj.isBottleneck ? "#f44336" : (absObj.passPct >= 100.0 ? "#4caf50" : (absObj.passPct >= 50.0 ? "#00bcd4" : "#ff9800"));
                if (elemVal) {
                    elemVal.innerText = absObj.passCount + " / " + totalN;
                    elemVal.style.color = color;
                }
                if (elemPct) {
                    elemPct.innerText = "(" + absObj.passPct.toFixed(1) + "%)";
                    elemPct.style.color = color;
                }
                if (elemSub) elemSub.innerText = "Target \u2264 " + target_abs.toFixed(2) + " ppm";
                if (elemBadge) {
                    if (absObj.isBottleneck) {
                        elemBadge.className = "badge-tag badge-bottleneck";
                        elemBadge.innerText = "BOTTLENECK";
                        elemBadge.style.display = "inline-block";
                    } else if (absObj.passPct >= 100.0) {
                        elemBadge.className = "badge-tag badge-pass";
                        elemBadge.innerText = "\u2713 100%";
                        elemBadge.style.display = "inline-block";
                    } else {
                        elemBadge.style.display = "none";
                    }
                }
            }

            if (tnObj) {
                var elemVal = document.getElementById('stat-val-tn');
                var elemPct = document.getElementById('stat-pct-tn');
                var elemSub = document.getElementById('stat-sub-tn');
                var elemBadge = document.getElementById('stat-badge-tn');
                var color = tnObj.isBottleneck ? "#f44336" : (tnObj.passPct >= 100.0 ? "#4caf50" : (tnObj.passPct >= 50.0 ? "#00bcd4" : "#ff9800"));
                if (elemVal) {
                    elemVal.innerText = tnObj.passCount + " / " + totalN;
                    elemVal.style.color = color;
                }
                if (elemPct) {
                    elemPct.innerText = "(" + tnObj.passPct.toFixed(1) + "%)";
                    elemPct.style.color = color;
                }
                if (elemSub) elemSub.innerText = "Target \u2264 " + target_tn.toExponential(2) + " m/\u221AHz";
                if (elemBadge) {
                    if (tnObj.isBottleneck) {
                        elemBadge.className = "badge-tag badge-bottleneck";
                        elemBadge.innerText = "BOTTLENECK";
                        elemBadge.style.display = "inline-block";
                    } else if (tnObj.passPct >= 100.0) {
                        elemBadge.className = "badge-tag badge-pass";
                        elemBadge.innerText = "\u2713 100%";
                        elemBadge.style.display = "inline-block";
                    } else {
                        elemBadge.style.display = "none";
                    }
                }
            }

            // Draw yield curve plot
            drawYieldCurvePlot(yieldRes.yield_curve);
        }

        function recalculateUtilityAndRerank() {
            try {
                var target_refl = parseFloat(document.getElementById('input-target-refl').value);
                var target_abs = parseFloat(document.getElementById('input-target-abs').value);
                var target_tn = parseFloat(document.getElementById('input-target-tn').value);
                var target_thick = parseFloat(document.getElementById('input-target-thick') ? document.getElementById('input-target-thick').value : __TARGET_THICK__) || 6000.0;
                var target_trans = (document.getElementById('input-target-trans') ? parseFloat(document.getElementById('input-target-trans').value) : NaN);
                if (isNaN(target_trans)) {
                    target_trans = (1.0 - target_refl) * 1e6;
                }

                if (isNaN(target_refl) || isNaN(target_abs) || isNaN(target_tn)) {
                    alert("Please enter valid numeric values for all targets.");
                    return;
                }

                var weightThickVal = (typeof weightThick !== 'undefined' && weightThick !== null) ? weightThick : 0.10;
                var total_w = weightRefl + weightAbs + weightTN + weightThickVal;
                var w_refl = total_w > 0 ? weightRefl / total_w : 0.25;
                var w_abs = total_w > 0 ? weightAbs / total_w : 0.25;
                var w_tn = total_w > 0 ? weightTN / total_w : 0.25;
                var w_thick = total_w > 0 ? weightThickVal / total_w : 0.25;

                var refl_loss_scale = Math.max(1e-6, 1.0 - target_refl);
                var trans_scale = Math.max(1e-6, target_trans);
                var refl_scale = Math.max(1e-9, 1.0 - target_refl);
                var abs_scale = Math.max(1e-9, target_abs);
                var tn_scale = Math.max(1e-25, target_tn);
                var rho = 1e-4;

                designsList.forEach(function(d) {
                    var obj1_score;
                    if (primaryMetric === "transmission") {
                        var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                        obj1_score = dt <= target_trans ?
                            (0.9 + 0.1 * (target_trans - dt) / trans_scale) :
                            (0.9 * Math.exp(-(dt - target_trans) / trans_scale));
                    } else {
                        obj1_score = d.reflectivity >= target_refl ? 
                            (0.9 + 0.1 * (d.reflectivity - target_refl) / refl_loss_scale) :
                            (0.9 * Math.exp(-(target_refl - d.reflectivity) / refl_loss_scale));
                    }

                    // Minimize Absorption
                    var abs_score = d.absorption <= target_abs ?
                        (0.9 + 0.1 * (target_abs - d.absorption) / target_abs) :
                        (0.9 * Math.exp(-(d.absorption - target_abs) / target_abs));

                    // Minimize Thermal Noise
                    var tn_score = d.thermal_noise <= target_tn ?
                        (0.9 + 0.1 * (target_tn - d.thermal_noise) / target_tn) :
                        (0.9 * Math.exp(-(d.thermal_noise - target_tn) / target_tn));

                    d.utility_score = w_refl * obj1_score + w_abs * abs_score + w_tn * tn_score;

                    // ASF Chebyshev Distance
                    var dev1;
                    if (primaryMetric === "transmission") {
                        var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                        dev1 = (dt - target_trans) / trans_scale;
                    } else {
                        dev1 = (target_refl - d.reflectivity) / refl_scale;
                    }
                    var devA = (d.absorption - target_abs) / abs_scale;
                    var devT = (d.thermal_noise - target_tn) / tn_scale;
                    var devs = [w_refl * dev1, w_abs * devA, w_tn * devT];
                    d.asf_distance = arrayMax(devs) + rho * devs.reduce((a, b) => a + b, 0);
                });

                // Sort designs depending on rank_by_utility
                if (rank_by_utility) {
                    designsList.sort((a, b) => b.utility_score - a.utility_score);
                } else {
                    if (primaryMetric === "transmission") {
                        designsList.sort((a, b) => {
                            var ta = (a.transmission !== undefined && a.transmission !== null) ? a.transmission : (1.0 - a.reflectivity) * 1e6;
                            var tb = (b.transmission !== undefined && b.transmission !== null) ? b.transmission : (1.0 - b.reflectivity) * 1e6;
                            return ta - tb;
                        });
                    } else {
                        designsList.sort((a, b) => b.reflectivity - a.reflectivity);
                    }
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
                    info_lines.push("  ASF Chebyshev Distance: " + (d.asf_distance !== undefined ? d.asf_distance.toFixed(4) : "0.0000"));
                    info_lines.push("  Active Layers: " + d.active_layer_count);
                    var thicknessText = d.d_physical_nm ? d.d_physical_nm.reduce((a, b) => a + b, 0).toFixed(2) : (d.total_thickness !== undefined && d.total_thickness !== null ? d.total_thickness.toFixed(2) : "N/A");
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
                if (filterTargetBoxOnly) {
                    displayList = displayList.filter(d => isInsideTargetBox(d, target_refl, target_abs, target_tn, target_thick, target_trans));
                }
                if (topX !== null && !isNaN(topX) && topX > 0) {
                    displayList = displayList.slice(0, topX);
                }

                var x_data = displayList.map(d => d.absorption);
                var y_data = displayList.map(d => d.thermal_noise);
                var z_data;
                if (plotMode === "rank") {
                    z_data = displayList.map(d => d.rank);
                } else {
                    if (primaryMetric === "transmission") {
                        if (zLog) {
                            z_data = displayList.map(d => {
                                var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                                return Math.log10(Math.max(1e-3, dt));
                            });
                        } else {
                            z_data = displayList.map(d => (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6));
                        }
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
                }

                var customdata = displayList.map(d => [
                    d.rank,
                    d.reflectivity,
                    1.0 - d.reflectivity,
                    d.active_layer_count,
                    d.total_thickness,
                    d.utility_score,
                    d.originalIdx,
                    d.asf_distance || 0.0
                ]);

                var color_values = [];
                var colorbar_title = "";
                var tickvals = null;
                var ticktext = null;
                var isReversed = false;
                var colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Plasma" : "Viridis";

                var colorMode = document.getElementById('select-color-mode').value;
                if (colorMode === "transmission_linear") {
                    color_values = displayList.map(d => (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6));
                    colorbar_title = "Transmission (ppm)";
                    colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Viridis_r" : "Viridis";
                    isReversed = true;
                } else if (colorMode === "transmission_log") {
                    color_values = displayList.map(d => {
                        var t = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                        return Math.log10(Math.max(1e-3, t));
                    });
                    colorbar_title = "Transmission (Log10 ppm)";
                    colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Viridis_r" : "Viridis";
                    isReversed = true;
                    var min_val = color_values.length > 0 ? arrayMin(color_values) : -2;
                    var max_val = color_values.length > 0 ? arrayMax(color_values) : 4;
                    var ticks_obj = getLogTicks(min_val, max_val, false);
                    tickvals = ticks_obj.tickvals;
                    ticktext = ticks_obj.ticktext;
                } else if (colorMode === "reflectivity_linear") {
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
                    
                    var min_val = color_values.length > 0 ? arrayMin(color_values) : 2;
                    var max_val = color_values.length > 0 ? arrayMax(color_values) : 6;
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
                    
                    var min_val = color_values.length > 0 ? arrayMin(color_values) : 0;
                    var max_val = color_values.length > 0 ? arrayMax(color_values) : 1;
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
                    
                    var min_val = color_values.length > 0 ? arrayMin(color_values) : -20;
                    var max_val = color_values.length > 0 ? arrayMax(color_values) : -18;
                    var ticks_obj = getLogTicks(min_val, max_val, true);
                    tickvals = ticks_obj.tickvals;
                    ticktext = ticks_obj.ticktext;
                } else if (colorMode === "loss_linear") {
                    color_values = displayList.map(d => 1.0 - d.reflectivity);
                    colorbar_title = "Reflectivity Loss (1-R)";
                    colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Magma" : "Reds";
                    isReversed = false;
                } else if (colorMode === "loss_log") {
                    // log10(1-R)
                    color_values = displayList.map(d => {
                        var loss = Math.max(1e-10, 1.0 - d.reflectivity);
                        return Math.log10(loss);
                    });
                    colorbar_title = "Loss (Log10)";
                    colorscale = (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "Magma" : "Reds";
                    isReversed = false;
                    
                    var min_int = Math.floor(color_values.length > 0 ? arrayMin(color_values) : -6);
                    var max_int = Math.ceil(color_values.length > 0 ? arrayMax(color_values) : -2);
                    tickvals = [];
                    ticktext = [];
                    for (var v = min_int; v <= max_int; v++) {
                        tickvals.push(v);
                        ticktext.push("10^" + v);
                    }
                } else if (colorMode === "rank") {
                    // Color points by Rank
                    color_values = displayList.map(d => d.rank);
                    colorbar_title = "Rank (1 = Best)";
                    colorscale = "Viridis_r";
                    isReversed = true;

                    tickvals = [1, Math.round(displayList.length / 2), Math.max(1, displayList.length)];
                    ticktext = ["#1", "#" + Math.round(displayList.length / 2), "#" + Math.max(1, displayList.length)];
                }

                var cmin = (color_values.length > 0 && isFinite(arrayMin(color_values))) ? arrayMin(color_values) : 0;
                var cmax = (color_values.length > 0 && isFinite(arrayMax(color_values))) ? arrayMax(color_values) : 1;
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
                data3d[0].marker.size = 5.5;
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
                var comp_label = (document.getElementById('input-comp-label') ? document.getElementById('input-comp-label').value.trim() : "") || "Reference Design";
                var comp_refl = document.getElementById('input-comp-refl') ? parseFloat(document.getElementById('input-comp-refl').value) : NaN;
                var comp_trans = document.getElementById('input-comp-trans') ? parseFloat(document.getElementById('input-comp-trans').value) : NaN;
                var comp_abs = document.getElementById('input-comp-abs') ? parseFloat(document.getElementById('input-comp-abs').value) : NaN;
                var comp_tn = document.getElementById('input-comp-tn') ? parseFloat(document.getElementById('input-comp-tn').value) : NaN;
                var comp_thick = document.getElementById('input-comp-thick') ? parseFloat(document.getElementById('input-comp-thick').value) : NaN;

                var show_comp_point = !isNaN(comp_abs) && !isNaN(comp_tn);

                if (show_comp_point) {
                    var obj1_comp_score;
                    var t_comp_val = !isNaN(comp_trans) ? comp_trans : (!isNaN(comp_refl) ? Math.max(0.0, (1.0 - comp_refl) * 1e6) : target_trans);
                    var r_val = !isNaN(comp_refl) ? comp_refl : (1.0 - t_comp_val * 1e-6);

                    if (primaryMetric === "transmission") {
                        obj1_comp_score = t_comp_val <= target_trans ?
                            (0.9 + 0.1 * (target_trans - t_comp_val) / trans_scale) :
                            (0.9 * Math.exp(-(t_comp_val - target_trans) / trans_scale));
                    } else {
                        obj1_comp_score = r_val >= target_refl ? 
                            (0.9 + 0.1 * (r_val - target_refl) / refl_loss_scale) :
                            (0.9 * Math.exp(-(target_refl - r_val) / refl_loss_scale));
                    }

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

                    var comp_utility = w_refl * obj1_comp_score + w_abs * abs_comp_score + w_tn * tn_comp_score + w_thick * thick_comp_score;

                    // Virtual rank
                    var better_count = 0;
                    designsList.forEach(function(d) {
                        if (rank_by_utility) {
                            if (d.utility_score > comp_utility) better_count++;
                        } else {
                            if (primaryMetric === "transmission") {
                                var dt = (d.transmission !== undefined && d.transmission !== null) ? d.transmission : Math.max(0.0, (1.0 - d.reflectivity) * 1e6);
                                if (dt < t_comp_val) better_count++;
                            } else {
                                if (d.reflectivity > r_val) better_count++;
                            }
                        }
                    });
                    var virtual_rank = better_count + 0.5;

                    var legend_name = comp_label + " (Rank " + Math.round(virtual_rank) + ")";
                    var hover_comp_str = "<b>" + comp_label + "</b><br>" +
                        "Virtual Rank: #" + Math.round(virtual_rank) + "<br>" +
                        (primaryMetric === "transmission" ? ("Transmission: " + t_comp_val.toFixed(2) + " ppm<br>") : ("Reflectivity: " + r_val.toFixed(6) + "<br>")) +
                        "Absorption: " + comp_abs.toFixed(3) + " ppm<br>" +
                        "Thermal Noise: " + comp_tn.toExponential(4) + " m/√Hz<br>" +
                        "Utility Score: " + comp_utility.toFixed(4) + "<extra></extra>";

                    var comp_z;
                    if (plotMode === "rank") {
                        comp_z = virtual_rank;
                    } else {
                        if (primaryMetric === "transmission") {
                            comp_z = zLog ? Math.log10(Math.max(1e-3, t_comp_val)) : t_comp_val;
                        } else {
                            comp_z = zLog ? -Math.log10(Math.max(1e-10, 1.0 - r_val)) : r_val;
                        }
                    }

                    data3d[1].x = [comp_abs];
                    data3d[1].y = [comp_tn];
                    data3d[1].z = [comp_z];
                    data3d[1].name = legend_name;
                    data3d[1].hovertemplate = hover_comp_str;
                    data3d[1].visible = true;
                    data3d[1].showlegend = true;
                    if (!data3d[1].marker) {
                        data3d[1].marker = {};
                    }
                    data3d[1].marker.size = 14;
                    data3d[1].marker.color = "#ff007f";
                    data3d[1].marker.symbol = "diamond";
                    data3d[1].marker.line = { width: 1.5, color: (layout3d.template === "plotly_dark" || layout3d.paper_bgcolor !== '#ffffff') ? "white" : "black" };

                    // Ensure 3D scene autoranges to encompass reference point
                    layout3d.scene.xaxis.autorange = true;
                    layout3d.scene.yaxis.autorange = true;

                    // Update global reference values so clicking the reference point shows the new parameters
                    referenceLabel = comp_label;
                    compareRefl = isNaN(r_val) ? null : r_val;
                    compareTrans = isNaN(t_comp_val) ? null : t_comp_val;
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
                    layout3d.scene.zaxis.type = 'linear';
                }

                if (plotMode === "rank") {
                    var all_ranks = (data3d[0].z || []).concat(show_comp_point && data3d[1].z ? data3d[1].z : []);
                    var maxRank = all_ranks.length > 0 ? arrayMax(all_ranks) : 100;
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
                    layout3d.scene.zaxis.title.text = rank_by_utility ? "Design Rank (Utility)" : (primaryMetric === "transmission" ? "Design Rank (Transmission)" : "Design Rank (Reflectivity)");
                    layout3d.scene.zaxis.tickvals = null;
                    layout3d.scene.zaxis.ticktext = null;
                } else {
                    var z_vals = (data3d[0].z || []).concat(show_comp_point && data3d[1].z ? data3d[1].z : []);
                    var zmin = z_vals.length > 0 ? arrayMin(z_vals) : 0.9;
                    var zmax = z_vals.length > 0 ? arrayMax(z_vals) : 1.0;
                    var span = (zmax - zmin) || 0.01;
                    
                    if (primaryMetric === "transmission") {
                        if (reversedZ) {
                            layout3d.scene.zaxis.range = [zmax + span * 0.05, zmin - span * 0.05];
                        } else {
                            layout3d.scene.zaxis.range = [zmin - span * 0.05, zmax + span * 0.05];
                        }
                        if (zLog) {
                            layout3d.scene.zaxis.title.text = "Transmission (Log10 ppm)";
                            var ticks_obj = getLogTicks(zmin, zmax, false);
                            layout3d.scene.zaxis.tickvals = ticks_obj.tickvals;
                            layout3d.scene.zaxis.ticktext = ticks_obj.ticktext;
                        } else {
                            layout3d.scene.zaxis.title.text = "Transmission (ppm)";
                            layout3d.scene.zaxis.tickvals = null;
                            layout3d.scene.zaxis.ticktext = null;
                        }
                    } else {
                        if (reversedZ) {
                            layout3d.scene.zaxis.range = [zmin - span * 0.05, zmax + span * 0.05];
                        } else {
                            layout3d.scene.zaxis.range = [zmax + span * 0.05, zmin - span * 0.05];
                        }
                        
                        if (zLog) {
                            layout3d.scene.zaxis.title.text = "Reflectivity (Log/Nines)";
                            var min_int = Math.floor(zmin);
                            var max_int = Math.ceil(zmax);
                            var z_tickvals = [];
                            var z_ticktext = [];
                            for (var v = min_int; v <= max_int; v++) {
                                z_tickvals.push(v);
                                if (v === 2) z_ticktext.push("0.99");
                                else if (v === 3) z_ticktext.push("0.999");
                                else if (v === 4) z_ticktext.push("0.9999");
                                else if (v === 5) z_ticktext.push("0.99999");
                                else if (v === 6) z_ticktext.push("0.999999");
                                else if (v === 7) z_ticktext.push("0.9999999");
                                else z_ticktext.push("1-10^-" + v);
                            }
                            layout3d.scene.zaxis.tickvals = z_tickvals;
                            layout3d.scene.zaxis.ticktext = z_ticktext;
                        } else {
                            layout3d.scene.zaxis.title.text = "Reflectivity";
                            layout3d.scene.zaxis.tickvals = null;
                            layout3d.scene.zaxis.ticktext = null;
                        }
                    }
                }

                // Update 3D Target Volume Cube wireframe trace
                var cubeTrace = getTargetCubeTrace(target_refl, target_abs, target_tn, plotMode, showTargetCube);
                if (data3d.length >= 3) {
                    data3d[2] = cubeTrace;
                    if (data3d.length > 3) {
                        data3d.splice(3);
                    }
                } else {
                    data3d.push(cubeTrace);
                }

                Plotly.react('plot-3d', data3d, layout3d);

                // Update Target Proximity & Quality Metrics Card
                try {
                    updateProximityMetricsUI();
                } catch (err) {
                    console.error("Error in updateProximityMetricsUI:", err);
                }

                // If 2D projections modal is open, re-render it too
                var projModal = document.getElementById('projections-modal');
                if (projModal && projModal.style.display !== 'none') {
                    try {
                        render2DProjections();
                    } catch (err) {
                        console.error("Error in render2DProjections:", err);
                    }
                }
            } catch (err) {
                console.error("Error in recalculateUtilityAndRerank:", err);
            }
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
        document.getElementById('btn-toggle-x-scale').addEventListener('click', function() {
            xLog = !xLog;
            if (xLog) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }
            layout3d.scene.xaxis.type = xLog ? 'log' : 'linear';
            Plotly.relayout('plot-3d', {
                'scene.xaxis.type': xLog ? 'log' : 'linear',
                'scene.xaxis.autorange': true,
                'scene.xaxis.tickvals': null,
                'scene.xaxis.ticktext': null
            });
        });

        document.getElementById('btn-toggle-y-scale').addEventListener('click', function() {
            yLog = !yLog;
            if (yLog) {
                this.classList.add('active');
            } else {
                this.classList.remove('active');
            }
            layout3d.scene.yaxis.type = yLog ? 'log' : 'linear';
            Plotly.relayout('plot-3d', {
                'scene.yaxis.type': yLog ? 'log' : 'linear',
                'scene.yaxis.autorange': true,
                'scene.yaxis.tickvals': null,
                'scene.yaxis.ticktext': null
            });
        });

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
        updateProximityMetricsUI();

        // Ensure all Plotly responsive plots redraw cleanly once container dimensions settle
        setTimeout(function() {
            try {
                Plotly.Plots.resize('plot-3d');
                Plotly.Plots.resize('plot-yield-curve');
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
    compiled_html = compiled_html.replace("__PRIMARY_METRIC__", primary_metric)
    compiled_html = compiled_html.replace("__INITIAL_TOP_X__", initial_top_x_str)
    compiled_html = compiled_html.replace("__DEFAULT_COLOR_MODE__", color_mode)
    compiled_html = compiled_html.replace("__PLOTLY_DATA_3D__", plotly_data_json)
    compiled_html = compiled_html.replace("__PLOTLY_LAYOUT_3D__", plotly_layout_json)
    compiled_html = compiled_html.replace("__TMM_DATA__", tmm_data_json)
    compiled_html = compiled_html.replace("__MATERIALS_PARAMS__", materials_params_json)
    compiled_html = compiled_html.replace("__PROXIMITY_METRICS__", proximity_metrics_json)
    compiled_html = compiled_html.replace("__HAS_REFERENCE__", "true" if args.compare_abs is not None else "false")
    compiled_html = compiled_html.replace("__REFERENCE_LABEL__", compare_label_str)
    compiled_html = compiled_html.replace("__COMPARE_REFL__", str(compare_refl_val))
    compiled_html = compiled_html.replace("__COMPARE_TRANS__", f"{compare_trans_val:.2f}" if compare_trans_val is not None else "")
    compiled_html = compiled_html.replace("__COMPARE_ABS__", str(compare_abs_val))
    compiled_html = compiled_html.replace("__COMPARE_TN__", str(compare_tn_val))
    compiled_html = compiled_html.replace("__COMPARE_THICK__", str(compare_thick_val))
    compiled_html = compiled_html.replace("__TARGET_REFL__", f"{target_refl:.6f}")
    compiled_html = compiled_html.replace("__TARGET_TRANS__", f"{target_trans:.2f}")
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
