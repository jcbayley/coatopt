# ==============================================================================
# Rank 1 Coating Design - Exported from coatopt
# Date: 2026-05-31 12:57:03
# Reflectivity: 0.970457
# Absorption: 0.434 ppm
# Thermal Noise: 3.2806e-21 m/sqrt(Hz)
# ==============================================================================

import numpy as np

# --- Design Parameters ---
# Number of layers: 45
# Total physical thickness: 5486.09 nm

# Optical Thicknesses (dOpt)
dOpt = np.array([
    0.238945,
    0.270729,
    0.272827,
    0.261455,
    0.236655,
    0.286060,
    0.269934,
    0.265367,
    0.257684,
    0.270793,
    0.252921,
    0.256498,
    0.243644,
    0.252912,
    0.266049,
    0.285529,
    0.223345,
    0.232320,
    0.188291,
    0.168923,
    0.135380,
    0.131565,
    0.118147,
    0.123122,
    0.113585,
    0.123274,
    0.138926,
    0.136999,
    0.124919,
    0.154677,
    0.152973,
    0.137757,
    0.119482,
    0.161755,
    0.116935,
    0.141653,
    0.158658,
    0.160248,
    0.131067,
    0.125986,
    0.136525,
    0.175818,
    0.125893,
    0.120043,
    0.137579
])

# Material Layer Indices (materialLayer)
# 999/0 = Air, 1 = SiO2, 2 = TiGermania
materialLayer = np.array([
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2
])

# Physical Thicknesses (nm)
physical_thickness = np.array([
    136.247401,
    198.659425,
    155.567104,
    191.854140,
    134.941600,
    209.908576,
    153.917539,
    194.724144,
    146.932242,
    198.705889,
    144.216436,
    188.216801,
    138.926526,
    185.584935,
    151.701904,
    209.519138,
    127.352315,
    170.474616,
    107.364186,
    123.954437,
    77.194078,
    96.541578,
    67.368004,
    90.345869,
    64.766706,
    90.457894,
    79.216214,
    100.528723,
    71.229081,
    113.501209,
    87.225705,
    101.084931,
    68.129087,
    118.694909,
    66.676575,
    103.943965,
    90.467290,
    117.588723,
    74.734653,
    92.447548,
    77.846859,
    129.013772,
    71.784562,
    88.087066,
    78.447962
])

# Material Definitions
materialParams = {
    0: {'name': 'air', 'desc': 'Air', 'n': 1, 'a': None, 'alpha': None, 'beta': None, 'kappa': None, 'C': None, 'Y': None, 'prat': None, 'phiM': None, 'k': 0},
    1: {'name': 'SiO2', 'desc': 'Silica - Thin film Room Temperature', 'n': 1.45, 'a': 0, 'alpha': 5.1e-07, 'beta': 8e-06, 'kappa': 1.38, 'C': 1641200.0, 'Y': 70000000000.0, 'prat': 0.19, 'phiM': 2.3e-05, 'k': 0},
    2: {'name': 'TiGermania', 'desc': 'Titania doped Germania - Room Temperature LMA', 'n': 1.866, 'a': 1, 'alpha': 1.282e-06, 'beta': 2.4e-05, 'kappa': 33, 'C': 2510000.0, 'Y': 92000000000.0, 'prat': 0.29, 'phiM': 9.013672e-05, 'k': 2e-07},
    999: {'name': 'air', 'desc': 'Air', 'n': 1, 'a': None, 'alpha': None, 'beta': None, 'kappa': None, 'C': None, 'Y': None, 'prat': None, 'phiM': None, 'k': 0}
}

# --- aLIGO Params Structure ---
aLIGO_params = {}

## INPUTS 
aLIGO_params['StackName']      = 'Rank 1 Design'               # Label for run 
aLIGO_params["dOpt"]           = dOpt                               # optical thickness array 
aLIGO_params["materialLayer"]  = materialLayer                      # material array containing keys which index materialParams
aLIGO_params["materialParams"] = materialParams                     # dictionary of material properties 
aLIGO_params["materialSub"]    = 1                                  # substrate type - Silica 
aLIGO_params["lambda_"]        = 1064.0                    # IFO wavelength (nm)
aLIGO_params["f"]              = np.logspace(1, 3, 100)             # Frequency range to evaluate CTN 
aLIGO_params["wBeam"]          = 0.062                              # laser beam size on ETM (m)
aLIGO_params["Temp"]           = 293.0                              # detector temperature (K)
aLIGO_params["plots "]         = False                              # boolean for activating plots 
aLIGO_params["t_air"]          = 500                                # thickness of air in EFI calculations for optical absorption : Default is 500nm
aLIGO_params["polarisation"]   = 'p'                                # light polarisation for EFI calculations 
aLIGO_params["lambda_list"]    = np.linspace(0, aLIGO_params["lambda_"]*1.5, 10000)

# --- Design Table ---
# Layer | Material Name | Refractive Index | dOpt | Physical Thickness (nm)
# 1     | TiGermania    | 1.866            | 0.238945 | 136.25 nm
# 2     | SiO2          | 1.45             | 0.270729 | 198.66 nm
# 3     | TiGermania    | 1.866            | 0.272827 | 155.57 nm
# 4     | SiO2          | 1.45             | 0.261455 | 191.85 nm
# 5     | TiGermania    | 1.866            | 0.236655 | 134.94 nm
# 6     | SiO2          | 1.45             | 0.286060 | 209.91 nm
# 7     | TiGermania    | 1.866            | 0.269934 | 153.92 nm
# 8     | SiO2          | 1.45             | 0.265367 | 194.72 nm
# 9     | TiGermania    | 1.866            | 0.257684 | 146.93 nm
# 10    | SiO2          | 1.45             | 0.270793 | 198.71 nm
# 11    | TiGermania    | 1.866            | 0.252921 | 144.22 nm
# 12    | SiO2          | 1.45             | 0.256498 | 188.22 nm
# 13    | TiGermania    | 1.866            | 0.243644 | 138.93 nm
# 14    | SiO2          | 1.45             | 0.252912 | 185.58 nm
# 15    | TiGermania    | 1.866            | 0.266049 | 151.70 nm
# 16    | SiO2          | 1.45             | 0.285529 | 209.52 nm
# 17    | TiGermania    | 1.866            | 0.223345 | 127.35 nm
# 18    | SiO2          | 1.45             | 0.232320 | 170.47 nm
# 19    | TiGermania    | 1.866            | 0.188291 | 107.36 nm
# 20    | SiO2          | 1.45             | 0.168923 | 123.95 nm
# 21    | TiGermania    | 1.866            | 0.135380 | 77.19 nm
# 22    | SiO2          | 1.45             | 0.131565 | 96.54 nm
# 23    | TiGermania    | 1.866            | 0.118147 | 67.37 nm
# 24    | SiO2          | 1.45             | 0.123122 | 90.35 nm
# 25    | TiGermania    | 1.866            | 0.113585 | 64.77 nm
# 26    | SiO2          | 1.45             | 0.123274 | 90.46 nm
# 27    | TiGermania    | 1.866            | 0.138926 | 79.22 nm
# 28    | SiO2          | 1.45             | 0.136999 | 100.53 nm
# 29    | TiGermania    | 1.866            | 0.124919 | 71.23 nm
# 30    | SiO2          | 1.45             | 0.154677 | 113.50 nm
# 31    | TiGermania    | 1.866            | 0.152973 | 87.23 nm
# 32    | SiO2          | 1.45             | 0.137757 | 101.08 nm
# 33    | TiGermania    | 1.866            | 0.119482 | 68.13 nm
# 34    | SiO2          | 1.45             | 0.161755 | 118.69 nm
# 35    | TiGermania    | 1.866            | 0.116935 | 66.68 nm
# 36    | SiO2          | 1.45             | 0.141653 | 103.94 nm
# 37    | TiGermania    | 1.866            | 0.158658 | 90.47 nm
# 38    | SiO2          | 1.45             | 0.160248 | 117.59 nm
# 39    | TiGermania    | 1.866            | 0.131067 | 74.73 nm
# 40    | SiO2          | 1.45             | 0.125986 | 92.45 nm
# 41    | TiGermania    | 1.866            | 0.136525 | 77.85 nm
# 42    | SiO2          | 1.45             | 0.175818 | 129.01 nm
# 43    | TiGermania    | 1.866            | 0.125893 | 71.78 nm
# 44    | SiO2          | 1.45             | 0.120043 | 88.09 nm
# 45    | TiGermania    | 1.866            | 0.137579 | 78.45 nm

print("Rank 1 design variables loaded successfully.")
