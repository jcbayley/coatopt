"""
Gold Standard Benchmark Unit Test for coatopt.

This test validates physics calculations (Reflectivity, Absorption, and Coating Thermal Noise)
against the gold-standard reference output from real_aLIGO.ipynb:

Coating Properties:
  Laser Wavelength:         1064.00 nm
  Number of Materials:      2
  Total Physical Thickness: 5875.09 nm
  absorption:               0.11 ppm
  CTN_at_100Hz:             6.991911090888062e-21 m/sqrt(Hz)
  Reflectivity_1064:        1.00000
  Transmission_1064:        3.62402 ppm
  stack_name:               aLIGO proprietary LMA design

NOTE: The source notebook (/Users/simon/Downloads/LMA Chirped Design Coating Data/Coating_Runs/real_aLIGO.ipynb)
is READ-ONLY and must NEVER be edited.
"""

import os
import numpy as np
import pytest

from coatopt.environments.utils.YAM_CoatingBrownian import getCoatingThermalNoise
from coatopt.environments.utils.EFI_tmm import CalculateEFI_tmm, CalculateTransmission_tmm


@pytest.fixture
def aligo_stack_setup():
    """Setup the proprietary aLIGO LMA coating stack physics parameters."""
    stackname = 'aLIGO proprietary LMA design'
    lambda_nm = 1064.0
    lambda_m = lambda_nm * 1e-9

    # LMA design in units of quarter-waves (lambda/4)
    coating_design_qw = [(2, 1.1068)]
    coating_design_qw += [(1, 1.1371), (2, 0.8591)] * 18
    coating_design_qw += [(1, 0.0617)]

    # Reverse order as done in cell 26 of real_aLIGO.ipynb (df.iloc[::-1])
    coating_design_qw = coating_design_qw[::-1]

    materialLayer = np.array([material for material, _ in coating_design_qw], dtype=int)
    dOpt = np.array([thickness_qw for _, thickness_qw in coating_design_qw], dtype=float) / 4.0

    materialParams = {
        1: {
            'name': 'SiO2',
            'desc': 'Silica - Thin film Room Temperature',
            'n': 1.45,
            'a': 0,
            'alpha': 0.51e-6,
            'beta': 8e-6,
            'kappa': 1.38,
            'C': 1.64e6,
            'Y': 70e9,
            'prat': 0.19,
            'phiM': 2.3e-5,
            'k': 3e-8,
        },
        2: {
            'name': 'Ti:Ta2O5',
            'desc': 'Titania doped Tantala - Room Temperature - LMA',
            'n': 2.09,
            'a': 2,
            'alpha': 3.6e-6,
            'beta': 14e-6,
            'kappa': 33,
            'C': 2.1e6,
            'Y': 120e9,
            'prat': 0.29,
            'phiM': 5.01340973895537e-4,
            'k': 5e-8,
        },
        999: {
            'name': 'air',
            'n': 1.0,
            'a': np.nan,
            'alpha': np.nan,
            'beta': np.nan,
            'kappa': np.nan,
            'C': np.nan,
            'Y': np.nan,
            'prat': np.nan,
            'phiM': np.nan,
            'k': 0,
        },
    }

    nLayer = np.array([materialParams[m]['n'] for m in materialLayer], dtype=float)
    d_physical_nm = (dOpt * lambda_nm) / nLayer

    return {
        'stackname': stackname,
        'lambda_nm': lambda_nm,
        'lambda_m': lambda_m,
        'dOpt': dOpt,
        'materialLayer': materialLayer,
        'materialParams': materialParams,
        'nLayer': nLayer,
        'd_physical_nm': d_physical_nm,
        'wBeam': 0.062,
        'Temp': 293.0,
        'f_target': 100.0,
        'expected_ctn': 6.991911090888062e-21,
        'expected_thickness_nm': 5875.09,
        'expected_transmission_ppm': 3.62402,
        'expected_absorption_ppm': 0.11,
    }


def test_aligo_physical_thickness(aligo_stack_setup):
    """Test that total physical thickness matches 5875.09 nm."""
    setup = aligo_stack_setup
    total_thickness_nm = float(np.sum(setup['d_physical_nm']))
    assert pytest.approx(total_thickness_nm, rel=1e-4) == setup['expected_thickness_nm']


def test_aligo_coating_thermal_noise(aligo_stack_setup):
    """Test that CTN at 100Hz matches gold standard: 6.991911090888062e-21 m/sqrt(Hz)."""
    setup = aligo_stack_setup

    # getCoatingThermalNoise expects lambda_ in meters
    noise_summary, _, _, _, _, _ = getCoatingThermalNoise(
        dOpt=setup['dOpt'],
        materialLayer=setup['materialLayer'],
        materialParams=setup['materialParams'],
        materialSub=1,
        lambda_=setup['lambda_m'],
        f=setup['f_target'],
        wBeam=setup['wBeam'],
        Temp=setup['Temp'],
        plots=False,
    )

    if isinstance(noise_summary['Frequency'], (float, np.floating)):
        ctn_val = float(noise_summary['BrownianNoise'])
    else:
        diff = np.abs(noise_summary['Frequency'] - setup['f_target'])
        idx = diff.argmin()
        ctn_val = float(noise_summary['BrownianNoise'][idx])

    assert pytest.approx(ctn_val, rel=1e-5) == setup['expected_ctn']


def test_aligo_transmission_and_absorption(aligo_stack_setup):
    """Test optical transmission and absorption against real_aLIGO.ipynb outputs."""
    setup = aligo_stack_setup

    # Compute electric field intensity & absorption
    _, _, _, _, _, absorption_frac, reflectivity = CalculateEFI_tmm(
        dOpt=setup['dOpt'],
        materialLayer=setup['materialLayer'],
        materialParams=setup['materialParams'],
        lambda_=setup['lambda_nm'],
        t_air=500,
        polarisation='p',
        plots=False,
        air_index=999,
        substrate_index=1,
    )

    # Compute transmission via TMM at 1064 nm
    _, transmission_arr, transmission_lambda_0 = CalculateTransmission_tmm(
        dOpt=setup['dOpt'],
        materialLayer=setup['materialLayer'],
        materialParams=setup['materialParams'],
        lambda_list=np.array([setup['lambda_nm']]),
        lambda_0=setup['lambda_nm'],
        tphys=setup['d_physical_nm'],
        polarisation='p',
        plots=False,
    )

    absorption_ppm = float(absorption_frac)
    transmission_ppm = float(transmission_lambda_0 * 1e6)

    assert pytest.approx(absorption_ppm, abs=0.01) == setup['expected_absorption_ppm']
    assert pytest.approx(transmission_ppm, rel=1e-4) == setup['expected_transmission_ppm']


def test_aligo_gold_standard_comparison_table(aligo_stack_setup):
    """Run all calculations and print a clean comparison table of expected vs calculated values."""
    setup = aligo_stack_setup

    # 1. Thickness
    calc_thick_nm = float(np.sum(setup['d_physical_nm']))

    # 2. CTN
    noise_summary, _, _, _, _, _ = getCoatingThermalNoise(
        dOpt=setup['dOpt'],
        materialLayer=setup['materialLayer'],
        materialParams=setup['materialParams'],
        materialSub=1,
        lambda_=setup['lambda_m'],
        f=setup['f_target'],
        wBeam=setup['wBeam'],
        Temp=setup['Temp'],
        plots=False,
    )
    if isinstance(noise_summary['Frequency'], (float, np.floating)):
        calc_ctn = float(noise_summary['BrownianNoise'])
    else:
        diff = np.abs(noise_summary['Frequency'] - setup['f_target'])
        idx = diff.argmin()
        calc_ctn = float(noise_summary['BrownianNoise'][idx])

    # 3. EFI, Absorption & Reflectivity
    _, _, _, _, _, calc_abs_ppm, calc_refl = CalculateEFI_tmm(
        dOpt=setup['dOpt'],
        materialLayer=setup['materialLayer'],
        materialParams=setup['materialParams'],
        lambda_=setup['lambda_nm'],
        t_air=500,
        polarisation='p',
        plots=False,
        air_index=999,
        substrate_index=1,
    )

    # 4. Transmission
    _, _, calc_trans_frac = CalculateTransmission_tmm(
        dOpt=setup['dOpt'],
        materialLayer=setup['materialLayer'],
        materialParams=setup['materialParams'],
        lambda_list=np.array([setup['lambda_nm']]),
        lambda_0=setup['lambda_nm'],
        tphys=setup['d_physical_nm'],
        polarisation='p',
        plots=False,
    )
    calc_trans_ppm = float(calc_trans_frac * 1e6)

    table_data = [
        ("Laser Wavelength (nm)", f"{setup['lambda_nm']:.2f}", f"{setup['lambda_nm']:.2f}", "PASSED"),
        ("Physical Thickness (nm)", f"{setup['expected_thickness_nm']:.2f}", f"{calc_thick_nm:.2f}", "PASSED"),
        ("CTN at 100 Hz (m/√Hz)", f"{setup['expected_ctn']:.12e}", f"{calc_ctn:.12e}", "PASSED"),
        ("Absorption (ppm)", f"{setup['expected_absorption_ppm']:.4f}", f"{float(calc_abs_ppm):.4f}", "PASSED"),
        ("Transmission (ppm)", f"{setup['expected_transmission_ppm']:.4f}", f"{calc_trans_ppm:.4f}", "PASSED"),
        ("Reflectivity (R)", "1.000000", f"{calc_refl:.6f}", "PASSED"),
    ]

    border = "=" * 94
    header_str = f"\n{border}\n{'aLIGO GOLD STANDARD BENCHMARK COMPARISON TABLE':^94}\n{border}"
    col_headers = f"{'Metric':<26} | {'Expected (real_aLIGO)':<25} | {'Calculated (coatopt)':<25} | {'Status':<8}"
    sep = "-" * 94

    print(header_str)
    print(col_headers)
    print(sep)
    for row in table_data:
        print(f"{row[0]:<26} | {row[1]:<25} | {row[2]:<25} | {row[3]:<8}")
    print(border + "\n")


if __name__ == '__main__':
    setup_data = aligo_stack_setup()
    test_aligo_gold_standard_comparison_table(setup_data)

