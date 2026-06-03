# ==============================================================================
# Rank 153 Coating Design - Exported from coatopt
# Date: 2026-05-31 12:43:24
# Reflectivity: 0.999985
# Absorption: 0.467 ppm
# Thermal Noise: 4.0013e-21 m/sqrt(Hz)
# ==============================================================================

import numpy as np

# --- Design Parameters ---
# Number of layers: 50
# Total physical thickness: 8349.84 nm

# Optical Thicknesses (dOpt)
dOpt = np.array([
    0.217060,
    0.278831,
    0.212400,
    0.275742,
    0.210022,
    0.274785,
    0.210217,
    0.275802,
    0.211086,
    0.275010,
    0.211618,
    0.276826,
    0.214088,
    0.278096,
    0.216607,
    0.282314,
    0.221230,
    0.282870,
    0.222173,
    0.285290,
    0.221923,
    0.285955,
    0.222957,
    0.285902,
    0.224178,
    0.288670,
    0.222535,
    0.286024,
    0.222121,
    0.285516,
    0.224016,
    0.285695,
    0.223996,
    0.284966,
    0.221241,
    0.282388,
    0.219034,
    0.279013,
    0.213443,
    0.273780,
    0.210629,
    0.273628,
    0.212173,
    0.273046,
    0.216221,
    0.276093,
    0.221462,
    0.277024,
    0.254274,
    0.400000
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
    2,
    1,
    2,
    1,
    2,
    1
])

# Physical Thicknesses (nm)
physical_thickness = np.array([
    123.768574,
    204.604507,
    121.111214,
    202.337747,
    119.755538,
    201.635213,
    119.866465,
    202.381334,
    120.362075,
    201.800074,
    120.665155,
    203.132657,
    122.073644,
    204.065250,
    123.509890,
    207.159956,
    126.146255,
    207.568400,
    126.683854,
    209.343592,
    126.541497,
    209.831616,
    127.130751,
    209.792608,
    127.827158,
    211.823762,
    126.890553,
    209.882365,
    126.653976,
    209.509364,
    127.734985,
    209.640947,
    127.723410,
    209.106034,
    126.152510,
    207.214323,
    124.893934,
    204.737815,
    121.705874,
    200.897949,
    120.101697,
    200.786559,
    120.981727,
    200.359565,
    123.289774,
    202.594919,
    126.278405,
    203.278477,
    144.988053,
    293.517183
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
aLIGO_params['StackName']      = 'Rank 153 Design'               # Label for run 
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
# 1     | TiGermania    | 1.866            | 0.217060 | 123.77 nm
# 2     | SiO2          | 1.45             | 0.278831 | 204.60 nm
# 3     | TiGermania    | 1.866            | 0.212400 | 121.11 nm
# 4     | SiO2          | 1.45             | 0.275742 | 202.34 nm
# 5     | TiGermania    | 1.866            | 0.210022 | 119.76 nm
# 6     | SiO2          | 1.45             | 0.274785 | 201.64 nm
# 7     | TiGermania    | 1.866            | 0.210217 | 119.87 nm
# 8     | SiO2          | 1.45             | 0.275802 | 202.38 nm
# 9     | TiGermania    | 1.866            | 0.211086 | 120.36 nm
# 10    | SiO2          | 1.45             | 0.275010 | 201.80 nm
# 11    | TiGermania    | 1.866            | 0.211618 | 120.67 nm
# 12    | SiO2          | 1.45             | 0.276826 | 203.13 nm
# 13    | TiGermania    | 1.866            | 0.214088 | 122.07 nm
# 14    | SiO2          | 1.45             | 0.278096 | 204.07 nm
# 15    | TiGermania    | 1.866            | 0.216607 | 123.51 nm
# 16    | SiO2          | 1.45             | 0.282314 | 207.16 nm
# 17    | TiGermania    | 1.866            | 0.221230 | 126.15 nm
# 18    | SiO2          | 1.45             | 0.282870 | 207.57 nm
# 19    | TiGermania    | 1.866            | 0.222173 | 126.68 nm
# 20    | SiO2          | 1.45             | 0.285290 | 209.34 nm
# 21    | TiGermania    | 1.866            | 0.221923 | 126.54 nm
# 22    | SiO2          | 1.45             | 0.285955 | 209.83 nm
# 23    | TiGermania    | 1.866            | 0.222957 | 127.13 nm
# 24    | SiO2          | 1.45             | 0.285902 | 209.79 nm
# 25    | TiGermania    | 1.866            | 0.224178 | 127.83 nm
# 26    | SiO2          | 1.45             | 0.288670 | 211.82 nm
# 27    | TiGermania    | 1.866            | 0.222535 | 126.89 nm
# 28    | SiO2          | 1.45             | 0.286024 | 209.88 nm
# 29    | TiGermania    | 1.866            | 0.222121 | 126.65 nm
# 30    | SiO2          | 1.45             | 0.285516 | 209.51 nm
# 31    | TiGermania    | 1.866            | 0.224016 | 127.73 nm
# 32    | SiO2          | 1.45             | 0.285695 | 209.64 nm
# 33    | TiGermania    | 1.866            | 0.223996 | 127.72 nm
# 34    | SiO2          | 1.45             | 0.284966 | 209.11 nm
# 35    | TiGermania    | 1.866            | 0.221241 | 126.15 nm
# 36    | SiO2          | 1.45             | 0.282388 | 207.21 nm
# 37    | TiGermania    | 1.866            | 0.219034 | 124.89 nm
# 38    | SiO2          | 1.45             | 0.279013 | 204.74 nm
# 39    | TiGermania    | 1.866            | 0.213443 | 121.71 nm
# 40    | SiO2          | 1.45             | 0.273780 | 200.90 nm
# 41    | TiGermania    | 1.866            | 0.210629 | 120.10 nm
# 42    | SiO2          | 1.45             | 0.273628 | 200.79 nm
# 43    | TiGermania    | 1.866            | 0.212173 | 120.98 nm
# 44    | SiO2          | 1.45             | 0.273046 | 200.36 nm
# 45    | TiGermania    | 1.866            | 0.216221 | 123.29 nm
# 46    | SiO2          | 1.45             | 0.276093 | 202.59 nm
# 47    | TiGermania    | 1.866            | 0.221462 | 126.28 nm
# 48    | SiO2          | 1.45             | 0.277024 | 203.28 nm
# 49    | TiGermania    | 1.866            | 0.254274 | 144.99 nm
# 50    | SiO2          | 1.45             | 0.400000 | 293.52 nm

print("Rank 153 design variables loaded successfully.")
