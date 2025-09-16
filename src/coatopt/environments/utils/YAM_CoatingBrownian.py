# from thermal_noise_hong import getCoatBrownian
import os
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numba

# from gwinc import *
import numpy as np
import pandas as pd


def generate_coating_stack(lambda_):
    """
    Randomly generates a coating stack with paired layers and writes to a text file.

    :return: n_input, dOpt
    """
    # Generate a random integer between 2 and 4 to determine the number of materials
    num_materials = random.randint(2, 4)

    # Generate the specified number of random refractive indices between 1 and 4
    n_input = sorted([random.uniform(1, 4) for _ in range(num_materials)])

    # Generate a random integer between 1 and 5 to determine the number of pairs
    num_pairs = random.randint(2, 50)

    # Generate the specified number of pairs with maximum contrast
    dOpt = []
    for _ in range(num_pairs):
        # Add a pair with maximum contrast
        # Using 1 and num_materials as material numbers
        dOpt.extend([1, num_materials])

    # normalise dOpt
    unique_dOpt = np.unique(dOpt)
    mapping = {val: i + 1 for i, val in enumerate(unique_dOpt)}
    dOpt = [mapping[val] for val in dOpt]

    # Filter n_input to only include values corresponding to materials in dOpt
    n_input = [n_input[i - 1] for i in unique_dOpt]
    # Calculate individual physical thickness for each material
    d_physical = lambda_ / (4 * np.array(n_input))

    # Arrays for each layer
    n_layers = np.array(n_input)[np.array(dOpt) - 1]
    material_kind = dOpt
    d_physical_layers = np.array(d_physical)[np.array(dOpt) - 1] * 1e6

    # Check if file exists and create a new filename with an increasing number
    counter = 1
    filename = "generated_coating_01.txt"
    while os.path.exists(filename):
        filename = f"generated_coating_{counter:02d}.txt"
        counter += 1

    # Write to text file
    with open(filename, "w") as file:
        for i in dOpt:
            file.write(f"material_{i}\t {d_physical_layers[i-1]:.2f}\n")

    return n_input, dOpt


def thin_film_stack(n_input, dOpt, lambda_):
    """
    Generates and plots a thin film coating stack.

    :param n_input: Array of refractive indices for each material.
    :param dOpt: Array specifying the material for each layer.
    :param lambda_: Wavelength.
    :return: n_layers, material_kind, d_physical_layers
    """
    if len(n_input) < max(dOpt):
        raise ValueError(
            "The number of refractive indices provided is less than the required materials in dOpt."
        )

    # Calculate individual physical thickness for each material
    d_physical = lambda_ / (4 * np.array(n_input))

    # Arrays for each layer
    n_layers = np.array(n_input)[np.array(dOpt) - 1]
    material_kind = dOpt
    d_physical_layers = np.array(d_physical)[np.array(dOpt) - 1]

    # Plotting the thin film stack
    unique_materials = list(set(dOpt))
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(unique_materials))
    )  # generate distinct colors for materials

    # Check if the file exists to write headers
    file_exists = os.path.exists("coating_metadata.txt")

    # Write metadata to text file
    with open("coating_metadata.txt", "a") as file:
        if not file_exists:
            file.write("Lambda (m)\tTotal Layers\t")
            for i in range(1, 7):
                file.write(
                    f"Material_{i}\tNo. Layers_{i}\tPhysical Thickness_{i} (m)\tRefractive Index_{i}\t"
                )
            file.write("\n")

        file.write(f"{lambda_:.2e}\t{len(dOpt)}\t")
        for i in range(1, 7):
            if i in dOpt:
                file.write(f"Material {i}\t")
                file.write(f"{len([x for x in dOpt if x == i])}\t")
                file.write(f"{sum(d_physical_layers[dOpt == i]):.2e}\t")
                file.write(
                    f"{n_input[i-1]:.2f}\t"
                )  # Directly access the refractive index from n_input
            else:
                file.write("NaN\tNaN\tNaN\tNaN\t")
        file.write("\n")

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.grid(True)
    depth_so_far = 0  # To keep track of where to plot the next bar
    for i in range(len(dOpt)):
        material_idx = dOpt[i]
        color_idx = unique_materials.index(material_idx)
        plt.bar(
            depth_so_far + d_physical_layers[i] / 2,
            d_physical_layers[i],
            color=colors[color_idx],
            width=d_physical_layers[i],
        )
        depth_so_far += d_physical_layers[i]

    plt.xlim([0, sum(d_physical_layers) * 1.01])
    plt.ylabel("Physical Thickness [nm]")
    plt.xlabel("Layer Position")
    plt.title("Generated Stack")
    legend_str = ["n = " + str(n) for n in n_input]
    plt.grid(False)
    plt.legend(legend_str)

    # Additional code for plotting the normalised electric field intensity squared can be added here

    plt.show()

    # Printing coating properties
    print("\nCoating Properties:\n")
    print(f"\nLaser Wavelength:\t\t{lambda_*1E9:.2f} nm")
    print(f"Number of Materials:\t\t{len(unique_materials):d}")
    print(f"Total Physical Thickness:\t{sum(d_physical_layers):.2e} m")

    for i in unique_materials:
        print(f"\n--------- Material {i} -------------\n")
        dOpt_array = np.array(dOpt)
        matching_indices = dOpt_array == i
        n_layers_matching = n_layers[matching_indices]
        d_physical_layers_matching = d_physical_layers[matching_indices]

        if len(n_layers_matching) > 0:
            print(f"No. Layers:\t\t\t{len(n_layers_matching)}")
            print(f"Total Physical Thickness:\t{sum(d_physical_layers_matching):.2e} m")
            print(f"Refractive Index:\t\t{np.unique(n_layers_matching)[0]:.2f}")
        else:
            print(f"No layers of material {i} found.")

    return n_layers, material_kind, d_physical_layers


# functions used to Calculate Coating Thermal Noise
# not to be used to calculate optical properties


def getCoatRefl2(nIn, nOut, nLayer, dOpt):
    # Vector of all refractive indices
    nAll = np.concatenate(([nIn], nLayer, [nOut]))

    # Reflectivity of each interface
    r = (nAll[:-1] - nAll[1:]) / (nAll[:-1] + nAll[1:])

    # Combine reflectivities
    rbar = np.zeros_like(r, dtype=complex)
    ephi = np.zeros_like(r, dtype=complex)

    ephi[-1] = np.exp(-4j * np.pi * dOpt[-1])
    rbar[-1] = ephi[-1] * r[-1]

    for n in range(len(dOpt) - 1, -1, -1):
        # Round-trip phase in this layer
        ephi[n] = np.exp(-4j * np.pi * dOpt[n - 1]) if n > 0 else 1

        # Accumulate reflectivity
        rbar[n] = ephi[n] * (r[n] + rbar[n + 1]) / (1 + r[n] * rbar[n + 1])

    # Reflectivity derivatives
    dr_dphi = np.zeros_like(dOpt, dtype=complex)

    for n in range(len(dOpt) - 1, -1, -1):
        dr_dphi[n] = -1j * rbar[n + 1]
        for m in range(n, -1, -1):
            dr_dphi[n] = (
                dr_dphi[n] * ephi[m] * (1 - r[m] ** 2) / (1 + r[m] * rbar[m + 1]) ** 2
            )

    # Shift rbar index
    rCoat = rbar[0]
    rbar = rbar[1:]

    # Phase derivatives
    dcdp = np.imag(dr_dphi / rCoat)

    return rCoat, dcdp, rbar, r


def getCoatAbsorption(lambda_, dOpt, aLayer, nLayer, rbar, r):
    """
    Returns coating absorption as a function of depth.

    Parameters:
    - lambda_ : wavelength
    - dOpt : optical thickness/lambda of each layer
             = geometrical thickness * refractive index/lambda
    - aLayer : absorption per unit length in each layer
    - nLayer : refractive index of each layer, ordered input to output (N x 1)
    - rbar : amplitude reflectivity of coating from this layer down
    - r : amplitude reflectivity of this interface (r[0] is nIn to nLayer[0])

    Returns:
    - rho : power ratio in each layer
    - absLayer : absorption contribution from each layer
    - absCoat : coating total absorption = sum(absLayer)
    """

    # Power in each layer
    powerLayer = np.cumprod(np.abs((1 - r[:-1] ** 2) / (1 + r[:-1] * rbar) ** 2))

    # One-way phases in each layer
    phi = 2 * np.pi * dOpt

    # Average E-field squared in each layer
    rho = (1 + np.abs(rbar) ** 2) + 2 * (np.sin(phi) / phi) * np.real(
        rbar * np.exp(1j * phi)
    )

    # Geometrical thickness of each layer
    dGeo = lambda_ * dOpt / nLayer

    # Compute power weighting for each layer
    absLayer = aLayer * rho * powerLayer * dGeo

    # Total coating absorption
    absCoat = np.sum(absLayer)

    return absCoat, absLayer, powerLayer, rho


# functions used to Calculate Coating Thermal Noise
# not to be used to calculate optical properties


def getCoatRefl2(nIn, nOut, nLayer, dOpt):
    # Vector of all refractive indices
    nAll = np.concatenate(([nIn], nLayer, [nOut]))

    # Reflectivity of each interface
    r = (nAll[:-1] - nAll[1:]) / (nAll[:-1] + nAll[1:])

    # Combine reflectivities
    rbar = np.zeros_like(r, dtype=complex)
    ephi = np.zeros_like(r, dtype=complex)

    ephi[-1] = np.exp(-4j * np.pi * dOpt[-1])
    rbar[-1] = ephi[-1] * r[-1]

    for n in range(len(dOpt) - 1, -1, -1):
        # Round-trip phase in this layer
        ephi[n] = np.exp(-4j * np.pi * dOpt[n - 1]) if n > 0 else 1

        # Accumulate reflectivity
        rbar[n] = ephi[n] * (r[n] + rbar[n + 1]) / (1 + r[n] * rbar[n + 1])

    # Reflectivity derivatives
    dr_dphi = np.zeros_like(dOpt, dtype=complex)

    for n in range(len(dOpt) - 1, -1, -1):
        dr_dphi[n] = -1j * rbar[n + 1]
        for m in range(n, -1, -1):
            dr_dphi[n] = (
                dr_dphi[n] * ephi[m] * (1 - r[m] ** 2) / (1 + r[m] * rbar[m + 1]) ** 2
            )

    # Shift rbar index
    rCoat = rbar[0]
    rbar = rbar[1:]

    # Phase derivatives
    dcdp = np.imag(dr_dphi / rCoat)

    return rCoat, dcdp, rbar, r


def getCoatAbsorption(lambda_, dOpt, aLayer, nLayer, rbar, r):
    """
    Returns coating absorption as a function of depth.

    Parameters:
    - lambda_ : wavelength
    - dOpt : optical thickness/lambda of each layer
             = geometrical thickness * refractive index/lambda
    - aLayer : absorption per unit length in each layer
    - nLayer : refractive index of each layer, ordered input to output (N x 1)
    - rbar : amplitude reflectivity of coating from this layer down
    - r : amplitude reflectivity of this interface (r[0] is nIn to nLayer[0])

    Returns:
    - rho : power ratio in each layer
    - absLayer : absorption contribution from each layer
    - absCoat : coating total absorption = sum(absLayer)
    """

    # Power in each layer
    powerLayer = np.cumprod(np.abs((1 - r[:-1] ** 2) / (1 + r[:-1] * rbar) ** 2))

    # print(dOpt)
    # raise
    # One-way phases in each layer
    phi = 2 * np.pi * dOpt

    # Average E-field squared in each layer
    rho = (1 + np.abs(rbar) ** 2) + 2 * (np.sin(phi) / phi) * np.real(
        rbar * np.exp(1j * phi)
    )

    # Geometrical thickness of each layer
    dGeo = lambda_ * dOpt / nLayer

    # Compute power weighting for each layer
    absLayer = aLayer * rho * powerLayer * dGeo

    # Total coating absorption
    absCoat = np.sum(absLayer)

    return absCoat, absLayer, powerLayer, rho


def getCoatNoise2(
    f, lambda_, wBeam, Temp, materialParams, materialSub, materialLayer, dOpt, dcdp
):
    """
    Returns coating noise as a function of depth.

    Parameters:
    - f : frequency
    - lambda_ : wavelength
    - wBeam : beam width
    - Temp : temperatur
    - materialParams : dictionary containing material properties
    - materialSub : substrate material
    - materialLayer : list of layer materials
    - dOpt : optical thickness / lambda of each layer
    - dcdp : phase derivatives

    Returns:
    - SbrZ, StoZ, SteZ, StrZ, brLayer
    """

    # Boltzmann constant and temperature
    kBT = 1.3807e-23 * Temp

    # Angular frequency
    w = 2 * np.pi * f

    # Substrate properties
    alphaSub = materialParams[materialSub]["alpha"]
    cSub = materialParams[materialSub]["C"]
    kappaSub = materialParams[materialSub]["kappa"]
    ySub = materialParams[materialSub]["Y"]
    pratSub = materialParams[materialSub]["prat"]

    # Initialize vectors of material properties
    nN = np.zeros_like(dOpt)
    aN = np.zeros_like(dOpt)
    alphaN = np.zeros_like(dOpt)
    betaN = np.zeros_like(dOpt)
    kappaN = np.zeros_like(dOpt)
    cN = np.zeros_like(dOpt)
    yN = np.zeros_like(dOpt)
    pratN = np.zeros_like(dOpt)
    phiN = np.zeros_like(dOpt)

    for n, mat in enumerate(materialLayer):
        nN[n] = materialParams[mat]["n"]
        aN[n] = materialParams[mat]["a"]
        alphaN[n] = materialParams[mat]["alpha"]
        betaN[n] = materialParams[mat]["beta"]
        kappaN[n] = materialParams[mat]["kappa"]
        cN[n] = materialParams[mat]["C"]
        yN[n] = materialParams[mat]["Y"]
        pratN[n] = materialParams[mat]["prat"]
        phiN[n] = materialParams[mat]["phiM"]

    # Geometrical thickness of each layer and total
    dGeo = lambda_ * dOpt / nN
    dCoat = np.sum(dGeo)

    return speedyNoise(
        nN,
        dcdp,
        ySub,
        yN,
        pratSub,
        pratN,
        kBT,
        wBeam,
        w,
        dGeo,
        phiN,
        alphaSub,
        dCoat,
        alphaN,
        betaN,
        Temp,
        kappaSub,
        cSub,
        cN,
        lambda_,
        dOpt,
    )


@numba.jit(nopython=True)
def speedyNoise(
    nN,
    dcdp,
    ySub,
    yN,
    pratSub,
    pratN,
    kBT,
    wBeam,
    w,
    dGeo,
    phiN,
    alphaSub,
    dCoat,
    alphaN,
    betaN,
    Temp,
    kappaSub,
    cSub,
    cN,
    lambda_,
    dOpt,
):

    brLayer = (
        (
            (1 + nN * dcdp) ** 2 * (ySub / yN)
            + (1 - pratSub - 2 * pratSub**2) ** 2
            * yN
            / ((1 + pratN) ** 2 * (1 - 2 * pratN) * ySub)
        )
        / (1 - pratN)
        * ((1 - pratN - 2 * pratN**2))
        / ((1 - pratSub - 2 * pratSub**2))
    )

    SbrZ = (4 * kBT / (np.pi * wBeam**2 * w)) * np.sum(
        dGeo * brLayer * phiN * (1 - pratSub - 2 * pratSub**2) / ySub
    )

    # Thermo-optic
    alphaBarSub = 2 * (1 + pratSub) * alphaSub

    alphaBar = (
        (dGeo / dCoat)
        * ((1 + pratSub) / (1 - pratN))
        * ((1 + pratN) / (1 + pratSub) + (1 - 2 * pratSub) * yN / ySub)
        * alphaN
    )

    betaBar = (-dcdp) * dOpt * (betaN / nN + alphaN * (1 + pratN) / (1 - pratN))

    # Thermo-elastic
    SteZ = (
        4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))
    ) * (np.sum(alphaBar * dCoat) - alphaBarSub * np.sum(dGeo * cN) / cSub) ** 2

    # Thermo-refractive
    StrZ = (
        4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))
    ) * np.sum(betaBar * lambda_) ** 2

    # Total thermo-optic
    StoZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * (
        np.sum(alphaBar * dCoat)
        - np.sum(betaBar * lambda_)
        - alphaBarSub * np.sum(dGeo * cN) / cSub
    ) ** 2

    return SbrZ, StoZ, SteZ, StrZ, brLayer


def getCoatingThermalNoise(
    dOpt,
    materialLayer,
    materialParams,
    materialSub=1,
    lambda_=1,
    f=1,
    wBeam=1,
    Temp=1,
    plots=True,
):
    # Set seaborn style and viridis color palette
    # sns.set_style("whitegrid")
    # sns.set_palette("tab10")

    # Extract substrate properties
    nSub = materialParams[materialSub]["n"]
    ySub = materialParams[materialSub]["Y"]
    pratSub = materialParams[materialSub]["prat"]

    # Initialize vectors of material properties
    # Initialize vectors of material properties
    nLayer = np.zeros(len(materialLayer))
    aLayer = np.zeros(len(materialLayer))

    for n, mat in enumerate(materialLayer):
        nLayer[n] = materialParams[mat]["n"]
        # aLayer[n] = materialParams[mat]['a']

    # Compute reflectivities
    rCoat, dcdp, rbar, r = getCoatRefl2(1, nSub, nLayer, dOpt)

    # Compute absorption
    # absCoat, absLayer, powerLayer, rho = getCoatAbsorption(lambda_, dOpt, aLayer, nLayer, rbar, r)

    # Compute brownian and thermo-optic noises
    SbrZ, StoZ, SteZ, StrZ, brLayer = getCoatNoise2(
        f, lambda_, wBeam, Temp, materialParams, materialSub, materialLayer, dOpt, dcdp
    )

    if plots == True:
        # Plotting
        # Absorption values
        plt.figure()
        plt.semilogy(rho, "o")
        plt.semilogy(powerLayer, "o")
        plt.semilogy(rho * powerLayer)
        plt.legend(["rho_j", "P_j / P_0", "rho_bar_j"])
        plt.xlabel("Layer number")

        # Noise weights for each layer
        plt.figure()
        materials = np.unique(materialLayer)
        # Get a list of colors from the Seaborn viridis palette
        colors = sns.color_palette(
            "viridis", n_colors=len(materials) + 2
        )  # +2 for the two additional plots

        for idx, i in enumerate(materials):
            matidx = np.where(materialLayer == i)[0]  # Extract the array from the tuple
            plt.bar(
                matidx,
                nLayer[matidx],
                color=colors[idx],
                label=materialParams[i]["name"],
            )
        # Use the next color in the palette for the following plots
        plt.plot(-dcdp, "o", color=colors[-2], markersize=10, label="-dphi_c / dphi_j")
        plt.plot(brLayer, "o", color=colors[-1], markersize=10, label="b_j")
        plt.xlabel("Layer number")
        plt.legend()

        # Noise plots
        plt.figure()
        plt.loglog(f, np.sqrt(SbrZ), "--")
        plt.loglog(f, np.sqrt(StoZ))
        plt.loglog(f, np.sqrt(SteZ), "-.")
        plt.loglog(f, np.sqrt(StrZ), "-.")
        plt.legend(
            ["Brownian Noise", "Thermo-optic Noise", "TE Component", "TR Component"]
        )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Thermal noise [m/sqrt(Hz)]")

        plt.show()

    # Return Noise Summary
    noise_summary = {
        "Frequency": f,
        "BrownianNoise": np.sqrt(SbrZ),
        "ThermoOptic": np.sqrt(StoZ),
        "ThermoElastic": np.sqrt(SteZ),
        "ThermoRefractive": np.sqrt(StrZ),
    }

    # --------------------------------------------------------------------------
    # CREATE DEBUGGING DATAFRAME
    # --------------------------------------------------------------------------
    # Gather layer-wise information for each layer
    Youngs = [materialParams[mat]["Y"] for mat in materialLayer]
    poisson = [materialParams[mat]["prat"] for mat in materialLayer]
    loss = [
        materialParams[mat]["phiM"] for mat in materialLayer
    ]  # Make sure 'phiM' is in your params

    debug_data = {
        "Youngs": Youngs,
        "Poisson": poisson,
        "Loss": loss,
        "bcoeff": brLayer,  # brLayer has shape (num_layers,)
        "dcdp": dcdp,  # dcdp also typically has shape (num_layers,)
        "rbar": rbar,  # rbar typically has shape (num_layers,) from your code
    }
    debug_df = pd.DataFrame(debug_data)
    # --------------------------------------------------------------------------

    # Return everything plus the debug dataframe
    return noise_summary, rCoat, dcdp, rbar, r, debug_df
