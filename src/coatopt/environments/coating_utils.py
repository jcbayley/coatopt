import numpy as np
from .EFI_tmm import CalculateEFI_tmm
from .YAM_CoatingBrownian_2 import getCoatingThermalNoise
import copy
#functions used to Calculate Coating Thermal Noise 
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
    
    for n in range(len(dOpt)-1, -1, -1):
        # Round-trip phase in this layer
        ephi[n] = np.exp(-4j * np.pi * dOpt[n - 1]) if n > 0 else 1
        
        # Accumulate reflectivity
        rbar[n] = ephi[n] * (r[n] + rbar[n + 1]) / (1 + r[n] * rbar[n + 1])
    
    # Reflectivity derivatives
    dr_dphi = np.zeros_like(dOpt, dtype=complex)
    
    for n in range(len(dOpt)-1, -1, -1):
        dr_dphi[n] = -1j * rbar[n + 1]
        for m in range(n, -1, -1):
            dr_dphi[n] = dr_dphi[n] * ephi[m] * (1 - r[m]**2) / (1 + r[m] * rbar[m + 1])**2
    
    # Shift rbar index
    rCoat = rbar[0]
    rbar = rbar[1:]
    
    # Phase derivatives
    dcdp = np.imag(dr_dphi / rCoat)
    
    return rCoat, dcdp, rbar, r


def getCoatAbsorption(light_wavelength, dOpt, aLayer, nLayer, rbar, r):
    """
    Returns coating absorption as a function of depth.

    Parameters:
    - light_wavelength : wavelength
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
    powerLayer = np.cumprod(np.abs((1 - r[:-1]**2) / (1 + r[:-1] * rbar)**2))
    
    # One-way phases in each layer
    phi = 2 * np.pi * dOpt
    
    # Average E-field squared in each layer
    rho = (1 + np.abs(rbar)**2) + 2 * (np.sin(phi) / phi) * np.real(rbar * np.exp(1j * phi))
    
    # Geometrical thickness of each layer
    dGeo = light_wavelength * dOpt / nLayer
    
    # Compute power weighting for each layer
    absLayer = aLayer * rho * powerLayer * dGeo
    
    # Total coating absorption
    absCoat = np.sum(absLayer)
    
    return absCoat, absLayer, powerLayer, rho


def getCoatNoise2(f, light_wavelength, wBeam, Temp, materialParams, materialSub, materialLayer, dOpt, dcdp):
    """
    Returns coating noise as a function of depth.

    Parameters:
    - f : frequency
    - light_wavelength : wavelength
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
    alphaSub = materialParams[materialSub]['alpha']
    cSub = materialParams[materialSub]['C']
    kappaSub = materialParams[materialSub]['kappa']
    ySub = materialParams[materialSub]['Y']
    pratSub = materialParams[materialSub]['prat']
    
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
        nN[n] = materialParams[mat]['n']
        aN[n] = materialParams[mat]['a']
        alphaN[n] = materialParams[mat]['alpha']
        betaN[n] = materialParams[mat]['beta']
        kappaN[n] = materialParams[mat]['kappa']
        cN[n] = materialParams[mat]['C']
        yN[n] = materialParams[mat]['Y']
        pratN[n] = materialParams[mat]['prat']
        phiN[n] = materialParams[mat]['phiM']
    
    # Geometrical thickness of each layer and total
    dGeo = light_wavelength * dOpt / nN
    dCoat = np.sum(dGeo)
    
    # Brownian
    brLayer = ((1 + nN * dcdp / 2)**2 * (ySub / yN) + 
               (1 - pratSub - 2 * pratSub**2)**2 * yN / 
               ((1 + pratN)**2 * (1 - 2 * pratN) * ySub)) / (1 - pratN) * ((1 - pratN - 2 * pratN**2)) / ((1 - pratSub - 2 * pratSub**2))
    
    SbrZ = (4 * kBT / (np.pi * wBeam**2 * w)) * np.sum(dGeo * brLayer * phiN * (1 - pratSub - 2 * pratSub**2) / ySub)
    
    # Thermo-optic
    alphaBarSub = 2 * (1 + pratSub) * alphaSub
    
    alphaBar = (dGeo / dCoat) * ((1 + pratSub) / (1 - pratN)) * ((1 + pratN) / (1 + pratSub) + (1 - 2 * pratSub) * yN / ySub) * alphaN
    
    betaBar = (-dcdp) * dOpt * (betaN / nN + alphaN * (1 + pratN) / (1 - pratN))
    
    # Thermo-elastic
    SteZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * (np.sum(alphaBar * dCoat) - alphaBarSub * np.sum(dGeo * cN) / cSub)**2
    
    # Thermo-refractive
    StrZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * np.sum(betaBar * light_wavelength)**2
    
    # Total thermo-optic
    StoZ = (4 * kBT * Temp / (np.pi * wBeam**2 * np.sqrt(2 * kappaSub * cSub * w))) * (np.sum(alphaBar * dCoat) - np.sum(betaBar * light_wavelength) - alphaBarSub * np.sum(dGeo * cN) / cSub)**2
    
    return SbrZ, StoZ, SteZ, StrZ, brLayer



def re_integrand(
    state,
    EFI,
    light_wavelength,
    num_points=30000,
    all_materials: dict = {}):
    ### set up a function to integrate over the total elecric feild intensity as a function of depth 
    ### t
    #materialLayer:         numpy.ndarray - An array of integers where each element represents the material type for each layer in the coating stack.
    #materialParams:        dict - A dictionary containing the refractive indices for each material type. The keys are material types (as referenced in materialLayer), and each key maps to another dictionary with a key 'n' for refractive index.
    #light_wavelength:               float - The wavelength of light in nanometers used for calculating the layer thicknesses.
    #num_points:            int - The total number of points to represent in the array, distributed across the entire stack.
    #Returns
    #EFI/refractiveindex    numpy.ndarray - Electric feild intensity normallised to the refractive index at each point in the coating stack 

    # Initialize variables
    depths = []
    ref_indices = []
    
    # Calculate layer thicknesses
    #if np.shape(materialLayer)[0] != 1:
        
    #layer_thicknesses = [light_wavelength / (4 * materialParams[mat]['n']) for mat in materialLayer]
    #cumulative_thickness = np.cumsum(layer_thicknesses)
        
    
    #else: 
    #    cumulative_thickness = layer_thicknesses
    #    layer_thicknesses = light_wavelength / (4 * materialParams[materialLayer[0]]['n'])

    # Generate depth points linearly spaced across each layer
    layer_thicknesses = np.zeros(len(state))
    cumulative_thickness = np.zeros(len(state))
    for i,layer in enumerate(state):
        thickness = layer[0]
        mat = np.argmax(layer[1:]) + 1
        material = all_materials[mat]
        layer_thicknesses[i] = light_wavelength / (4 * material['n'])
        cumulative_thickness[:i+1] += layer_thicknesses[i]
        start_depth = cumulative_thickness[i] - thickness
        end_depth = cumulative_thickness[i]
        num_points_layer = int(num_points * (thickness / cumulative_thickness[-1]))

        layer_depths = np.linspace(start_depth, end_depth, num_points_layer, endpoint=False)
        layer_ref_indices = np.full(layer_depths.shape, material['n'])

        depths.extend(layer_depths)
        ref_indices.extend(layer_ref_indices)

    # Adjust the total number of points to be exactly 30,000
    current_total_points = len(depths)
    if current_total_points != num_points:
        adjustment = num_points - current_total_points
        final_layer_depths = np.linspace(cumulative_thickness[-2], cumulative_thickness[-1], adjustment + len(depths[-adjustment:]), endpoint=False)
        final_layer_ref_indices = np.full(final_layer_depths.shape, material['n'])
        depths[-adjustment:] = final_layer_depths
        ref_indices[-adjustment:] = final_layer_ref_indices

    # Create final array
    stack_info_array = np.column_stack((depths, ref_indices))
    #stack_info_array = pd.DataFrame(stack_info_array)


    return EFI/stack_info_array[1]

def integrand(EFI,light_wavelength,materialLayer,materialParams,num_points=30000):
    ### set up a function to integrate over the total elecric feild intensity as a function of depth 
    ### t
    #materialLayer:         numpy.ndarray - An array of integers where each element represents the material type for each layer in the coating stack.
    #materialParams:        dict - A dictionary containing the refractive indices for each material type. The keys are material types (as referenced in materialLayer), and each key maps to another dictionary with a key 'n' for refractive index.
    #light_wavelength:               float - The wavelength of light in nanometers used for calculating the layer thicknesses.
    #num_points:            int - The total number of points to represent in the array, distributed across the entire stack.
    #Returns
    #EFI/refractiveindex    numpy.ndarray - Electric feild intensity normallised to the refractive index at each point in the coating stack 

    # Initialize variables
    depths = []
    ref_indices = []
    
    # Calculate layer thicknesses
    #if np.shape(materialLayer)[0] != 1:
        
    layer_thicknesses = [light_wavelength / (4 * materialParams[mat]['n']) for mat in materialLayer]
    cumulative_thickness = np.cumsum(layer_thicknesses)
        
    
    #else: 
    #    cumulative_thickness = layer_thicknesses
    #    layer_thicknesses = light_wavelength / (4 * materialParams[materialLayer[0]]['n'])
        

    # Generate depth points linearly spaced across each layer
    for i, thickness in enumerate(layer_thicknesses):
        start_depth = cumulative_thickness[i] - thickness
        end_depth = cumulative_thickness[i]
        num_points_layer = int(num_points * (thickness / cumulative_thickness[-1]))

        layer_depths = np.linspace(start_depth, end_depth, num_points_layer, endpoint=False)
        layer_ref_indices = np.full(layer_depths.shape, materialParams[materialLayer[i]]['n'])

        depths.extend(layer_depths)
        ref_indices.extend(layer_ref_indices)

    # Adjust the total number of points to be exactly 30,000
    current_total_points = len(depths)
    if current_total_points != num_points:
        adjustment = num_points - current_total_points
        final_layer_depths = np.linspace(cumulative_thickness[-2], cumulative_thickness[-1], adjustment + len(depths[-adjustment:]), endpoint=False)
        final_layer_ref_indices = np.full(final_layer_depths.shape, materialParams[materialLayer[-1]]['n'])
        depths[-adjustment:] = final_layer_depths
        ref_indices[-adjustment:] = final_layer_ref_indices

    # Create final array
    stack_info_array = np.column_stack((depths, ref_indices))
    #stack_info_array = pd.DataFrame(stack_info_array)

    #print(np.shape(EFI), np.shape(stack_info_array))

    return EFI/stack_info_array[:,1]

def merit_function(
        state,
        all_materials,
        w_R=1.0, 
        w_T=1.0, 
        w_E=1.0, 
        w_D=1.0,
        wBeam = 0.062,              # 6cm beam for aLIGO 
        light_wavelength = 1064e-9,          # laser wavelength
        Temp = 293,                 # temperature - Room temperature 
        frequency = 100.0,                     # frequencies for plotting
        substrate_index = 1,
        air_index = 0
    ):
    #set up with default inputs to match aLIGO for testing = this should be modified to allow for varying inputs. 
    
    """
    Calculate the merit function for a given coating configuration.
    """
    
    #n1 = materialParams[np.unique(materialLayer)[0]]['n']
    #n2 = materialParams[np.unique(materialLayer)[1]]['n']
        
    #n_indicies = [n1, n2] * num21 + [1, 2] * num34
    
    # convert current state to format for EFI functions

    layer_thicknesses = state[:,0]
    layer_material_inds = np.argmax(state[:,1:], axis=1) 
    
    #layer_materials = np.array(layer_material_inds, dtype=np.int32)
    #new_all_materials.update(air_material)
    #new_all_materials = copy.copy(all_materials)
    new_all_materials = all_materials
    #print(new_all_materials.keys())
    num_points = 200

    E_total, _, PhysicalThickness = CalculateEFI_tmm(
        layer_thicknesses = layer_thicknesses,
        layer_materials = layer_material_inds, 
        material_parameters = new_all_materials,
        light_wavelength=light_wavelength ,
        t_air=500,
        polarisation='p' ,
        plots=False,
        num_points=num_points,
        air_index = air_index,
        substrate_index=substrate_index)
    
    ThermalNoise= getCoatingThermalNoise(
        layer_thicknesses, 
        layer_material_inds, 
        new_all_materials, 
        substrate_index=1, 
        light_wavelength=light_wavelength, 
        f=frequency, 
        wBeam=wBeam, 
        Temp=Temp,
        plots =False)



    if isinstance(ThermalNoise[0]['Frequency'],float):
        difference_array = np.absolute(ThermalNoise[0]['Frequency']-100)
        
        # find the index of minimum element from the array
        index = difference_array.argmin()
        
        ThermalNoise_Total = ThermalNoise[0]['BrownianNoise'][index]
        #use only the thermal noise at the specified frequency = default : 100 Hz 
    else:
        
        ThermalNoise_Total = ThermalNoise[0]['BrownianNoise']


    # Total Thickness
    D = PhysicalThickness[-1]
    
    normallised_EFI = integrand(E_total,light_wavelength,layer_material_inds,all_materials,num_points=num_points)
    #normallised_EFI = integrand(state,E_total,laser_wavelength,num_points=30000)
    
    depths = np.linspace(0, D, len(normallised_EFI))

    # Perform the integration using the trapezoidal rule
    E_integrated = np.trapz(normallised_EFI, depths)
    
    n_layer = np.array([all_materials[mat]["n"] for mat in layer_material_inds])

    optical_thickness = []
    for i in range(len(n_layer)):
        optical_thickness.append(2*np.pi*layer_thicknesses[i]*n_layer[i]/light_wavelength)

    nSub = all_materials[1]["n"]
    nAir = all_materials[0]["n"]
    
    # Reflectivity
    R, dcdp, rbar, r = getCoatRefl2(nAir, nSub, n_layer, optical_thickness)
    
    R = np.real(R)

    """
    # Clear the previous output (the number of spaces should cover the previous line)
    print("\r" + " " * 50, end="\r")

    # Merit Function
    print(f"{'Parameter':<10}{'Value':<10}")
    print(f"{'R':<10}{R:<10.5f}")
    print(f"{'CTN':<10}{ThermalNoise_Total:<10.2e}")
    print(f"{'E':<10}{(1/100 * E_integrated):<10.2f}")
    print(f"{'D':<10}{(D):<10.2f}")
    """

    R_scaled =  w_R * (R)
    CTN_scaled = w_T * (ThermalNoise_Total/(5.92672659826259e-21))
    EFI_scaled =  w_E * (1/10 * E_integrated)   
    thick_scaled = w_D * (1/4 * np.log10(D))

    #print(R_scaled, CTN_scaled, EFI_scaled, thick_scaled)
    
    M = w_R * (1/R) + w_T * ThermalNoise_Total + w_E * (1/E_integrated) + w_D * D
    
    M_scaled = 1./(R_scaled + CTN_scaled + EFI_scaled + thick_scaled)
    
    #return M_scaled, R_scaled , CTN_scaled , EFI_scaled , thick_scaled
    return M, M_scaled, R, ThermalNoise_Total, E_integrated,D

def optical_to_physical(optical_thickness, vacuum_wavelength, refractive_index):
    physical_thickness = optical_thickness*vacuum_wavelength/ refractive_index
    return physical_thickness
def physical_to_optical(physical_thickness, vacuum_wavelength, refractive_index):
    optical_thickness = physical_thickness*refractive_index/vacuum_wavelength
    return optical_thickness

def merit_function_2(
        state,
        all_materials,
        w_R=1.0, 
        w_T=1.0, 
        w_E=1.0, 
        w_D=1.0,
        wBeam = 0.062,              # 6cm beam for aLIGO 
        light_wavelength = 1064e-9,          # laser wavelength
        Temp = 293,                 # temperature - Room temperature 
        frequency = 100.0,                     # frequencies for plotting
        substrate_index = 1,
        air_index = 0
    ):

    layer_thicknesses = state[:,0]
    layer_material_inds = np.argmax(state[:,1:], axis=1) 
    ns = np.array([all_materials[i]["n"] for i in layer_material_inds])
    layer_optical_thicknesses = np.array([physical_to_optical(layer_thicknesses[i], light_wavelength, ns[i]) for i in range(len(layer_thicknesses))])



    noise_summary, rCoat, dcdp, rbar, r = getCoatingThermalNoise(
        layer_optical_thicknesses, 
        layer_material_inds, 
        all_materials, 
        materialSub=1, 
        lambda_=light_wavelength, 
        f=frequency, 
        wBeam=wBeam, 
        Temp=Temp)
    
    if isinstance(noise_summary['Frequency'],float):
        difference_array = np.absolute(noise_summary['Frequency']-100)
        
        # find the index of minimum element from the array
        index = difference_array.argmin()
        
        thermal_noise = noise_summary['BrownianNoise'][index]
        #use only the thermal noise at the specified frequency = default : 100 Hz 
    else:
        
        thermal_noise = noise_summary['BrownianNoise']
    
    return np.real(rCoat), thermal_noise