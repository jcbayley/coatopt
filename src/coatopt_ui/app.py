import os
import sys
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

# Setup system path to locate CoatingAnalysis source
candidate_paths = [
    "/Users/simon/Developer/Python/CoatingAnalysis/src",
    "/Users/simon/Library/CloudStorage/GoogleDrive-simon.tait@ligo.org/My Drive/BackupFromDropbox/Python/CoatingAnalysis/src"
]
lib_path = None
for path_str in candidate_paths:
    if Path(path_str).exists():
        lib_path = path_str
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)
        break

# Try importing from coating_analysis with fallback to local modules
try:
    if lib_path is None:
        raise ImportError("No valid CoatingAnalysis source path found")
    from coating_analysis.YAM_CoatingBrownian import getCoatingThermalNoise
    from coating_analysis.EFI_tmm import CalculateEFI_tmm, CalculateTransmission_tmm
except ImportError as e:
    print(f"Error importing physics libraries from CoatingAnalysis: {e}. Falling back to local modules...")
    try:
        from coatopt.environments.utils.YAM_CoatingBrownian import getCoatingThermalNoise
        from coatopt.environments.utils.EFI_tmm import CalculateEFI_tmm, CalculateTransmission_tmm
    except ImportError as local_e:
        print(f"Error importing local fallback physics libraries: {local_e}")


app = FastAPI(title="Interactive Coating Designer")

# Mount static files
static_dir = Path(__file__).parent / "static"
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Helper to serialize float values that could be NaN
def clean_nans(val):
    if isinstance(val, float) and np.isnan(val):
        return None
    elif isinstance(val, dict):
        return {k: clean_nans(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [clean_nans(v) for v in val]
    elif isinstance(val, np.ndarray):
        return [clean_nans(x) for x in val.tolist()]
    return val

# Schema definitions
class MaterialProp(BaseModel):
    name: str
    desc: Optional[str] = ""
    n: float
    k: float = 0.0

    a: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    kappa: Optional[float] = None
    C: Optional[float] = None
    Y: Optional[float] = None
    prat: Optional[float] = None
    phiM: Optional[float] = None

class Layer(BaseModel):
    thickness: float  # physical thickness in nm
    material: int     # material key

class AnalyzeRequest(BaseModel):
    layers: List[Layer]
    materialParams: Dict[str, MaterialProp]
    lambda_: float = 1064.0
    wBeam: float = 0.062
    Temp: float = 293.0
    polarisation: str = 'p'
    angle: float = 0.0
    target_lambdas: List[float] = [1064.0, 532.0]

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return index_file.read_text()
    return HTMLResponse(content="<h1>Interactive Coating Designer</h1><p>Frontend assets not found yet.</p>", status_code=404)

@app.post("/api/analyze")
async def analyze_stack(req: AnalyzeRequest):
    try:
        # Convert layers to numpy arrays
        tphys = np.array([l.thickness for l in req.layers])
        materialLayer = np.array([l.material for l in req.layers])
        
        # Convert materialParams, mapping string keys to ints and handling None -> NaN
        materialParams = {}
        for k, v in req.materialParams.items():
            mat_dict = v.model_dump()
            for key in ["a", "alpha", "beta", "kappa", "C", "Y", "prat", "phiM"]:
                if mat_dict[key] is None:
                    mat_dict[key] = np.nan
            materialParams[int(k)] = mat_dict
            
        # Recompute dOpt from tphys: dopt_i = tphys_i * n_i / lambda_
        dOpt = np.zeros_like(tphys)
        for i, mat_idx in enumerate(materialLayer):
            n_i = materialParams[mat_idx]['n']
            dOpt[i] = (tphys[i] * n_i) / req.lambda_
            
        # 1. Run Coating Thermal Noise
        f_arr = np.logspace(0, 3, 100)  # frequencies for plotting
        import inspect
        sig = inspect.signature(getCoatingThermalNoise)
        ctn_kwargs = {
            "dOpt": dOpt,
            "materialLayer": materialLayer,
            "materialParams": materialParams,
            "materialSub": 1,
            "lambda_": req.lambda_ * 1e-9,  # Convert nm to meters
            "f": f_arr,
            "wBeam": req.wBeam,
            "Temp": req.Temp,
            "plots": False
        }
        if "tphys" in sig.parameters:
            ctn_kwargs["tphys"] = tphys

        noise_summary, rCoat, dcdp, rbar, r, debug_df = getCoatingThermalNoise(**ctn_kwargs)

        
        # Extract CTN summary at 100 Hz
        idx_100 = (np.abs(f_arr - 100.0)).argmin()
        brownian_100 = float(noise_summary['BrownianNoise'][idx_100])
        thermo_optic_100 = float(noise_summary['ThermoOptic'][idx_100])
        thermo_elastic_100 = float(noise_summary['ThermoElastic'][idx_100])
        thermo_refractive_100 = float(noise_summary['ThermoRefractive'][idx_100])
        # Total CTN amplitude spectral density is sqrt(S_br + S_to)
        total_noise_100 = float(np.sqrt(noise_summary['BrownianNoise'][idx_100]**2 + noise_summary['ThermoOptic'][idx_100]**2))
        
        # 2. Run EFI for optical absorption
        sig_efi = inspect.signature(CalculateEFI_tmm)
        efi_kwargs = {
            "dOpt": dOpt,
            "materialLayer": materialLayer,
            "materialParams": materialParams,
            "lambda_": req.lambda_ * 1e-9,  # Convert nm to meters for CalculateEFI_tmm
            "t_air": 500,
            "polarisation": req.polarisation,
            "plots": False
        }
        if "tphys" in sig_efi.parameters:
            efi_kwargs["tphys"] = tphys
        E_sub, layer_idx_efi, ds_efi, E_efi, poyn, absor, _ = CalculateEFI_tmm(**efi_kwargs)
        
        # 3. Calculate Transmission Spectra
        # Include target wavelengths exactly in the lambda list to avoid grid-rounding issues
        lambda_list = np.sort(np.unique(np.concatenate((
            np.linspace(300, 1600, 500), 
            req.target_lambdas
        ))))
        sig_trans = inspect.signature(CalculateTransmission_tmm)
        trans_kwargs = {
            "dOpt": dOpt,
            "materialLayer": materialLayer,
            "materialParams": materialParams,
            "lambda_list": lambda_list,
            "lambda_0": req.lambda_,
            "polarisation": req.polarisation,
            "plots": False
        }
        if "tphys" in sig_trans.parameters:
            trans_kwargs["tphys"] = tphys
        if "angle" in sig_trans.parameters:
            trans_kwargs["angle"] = req.angle
        wavelengths, transmission, _ = CalculateTransmission_tmm(**trans_kwargs)
        
        # Normalize wavelengths if they are scaled by 1e9 due to the local fallback bug
        if len(wavelengths) > 0 and np.max(wavelengths) > 1e6:
            wavelengths = wavelengths / 1e9

        # Calculate transmission and reflectivity at target wavelengths
        target_results = {}
        for wl in req.target_lambdas:
            idx = (np.abs(wavelengths - wl)).argmin()
            t_val = float(transmission[idx])
            # If main wavelength (1064nm), reflectivity is 1 - T - A
            if wl == req.lambda_:
                r_val = 1.0 - t_val - (absor * 1e-6)
            else:
                r_val = 1.0 - t_val
            target_results[str(wl)] = {
                "transmission": t_val,
                "reflectivity": max(0.0, r_val)
            }
            
        # Calculate material statistics
        material_summary = {}
        for mat_idx in np.unique(materialLayer):
            mat_mask = (materialLayer == mat_idx)
            num_layers = int(np.sum(mat_mask))
            total_thick = float(np.sum(tphys[mat_mask]))
            name = materialParams[mat_idx]['name']
            n_val = materialParams[mat_idx]['n']
            k_val = materialParams[mat_idx]['k']
            material_summary[int(mat_idx)] = {
                "name": name,
                "layers": num_layers,
                "thickness": total_thick,
                "n": n_val,
                "k": k_val
            }
            
        def to_list(x):
            if hasattr(x, "tolist"):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]

        # Assemble complete result payload
        response_data = {
            "total_thickness": float(np.sum(tphys)),
            "absorption_ppm": float(absor),
            "noise_100hz": {
                "brownian": brownian_100,
                "thermo_optic": thermo_optic_100,
                "thermo_elastic": thermo_elastic_100,
                "thermo_refractive": thermo_refractive_100,
                "total": total_noise_100
            },
            "targets": target_results,
            "materials": material_summary,
            "charts": {
                "spectrum": {
                    "wavelengths": to_list(wavelengths),
                    "transmission": to_list(transmission)
                },
                "efi": {
                    "depths": to_list(ds_efi),
                    "intensity": to_list(E_efi),
                    "layer_idx": to_list(layer_idx_efi)
                },
                "noise": {
                    "frequencies": to_list(f_arr),
                    "brownian": to_list(noise_summary['BrownianNoise']),
                    "thermo_optic": to_list(noise_summary['ThermoOptic']),
                    "thermo_elastic": to_list(noise_summary['ThermoElastic']),
                    "thermo_refractive": to_list(noise_summary['ThermoRefractive'])
                }
            }
        }
        
        # Convert any remaining NaN to None so it outputs clean JSON
        return JSONResponse(content=clean_nans(response_data))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
