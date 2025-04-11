import math as m
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, leastsq, least_squares
from scipy import integrate as TG
import datetime as dt
import pandas as pd
import tmm
import numpy as np  
import matplotlib.pyplot as plt 
import os 
import warnings
import logging
from tmm_fast import coh_tmm as coh_tmm_fast
import numpy as np



def physical_to_optical(physical_thickness, wavelength, refractive_index):
    optical_thickness = physical_thickness * refractive_index / wavelength
    return optical_thickness

def optical_to_physical(optical_thickness, wavelength, refractive_index):
    physical_thickness = optical_thickness*wavelength/ refractive_index
    return physical_thickness

def CalculateEFI_tmm(dOpt ,materialLayer, materialParams,lambda_ =1064 ,t_air = 500,polarisation='p' ,plots ='False',depBreak=None, air_index=0, substrate_index=1):
    """
    function calcualtes the normallised electric field intensity inside a thin film coating/coating stack usign tmm. This method 
    takes into acount of : 
    lambda_          the wavelength of light (nm)  incident on the coxating stack         - default =  1064 nm 
    t_air            the total thickness of air before the coating stack                 - default = 500 nm 
    materailParams : Dictionary variable containing the material propeties of each layer in the coating stack (including air) 
                     Requires refractive       index values 'n' and complex refractive index values a.k.a attenuation coeffiencts  'k'
    
    polarisation   : string variable specifying either 'p' or 's' polarised light       - default = 'p' polarisation 
    plots          : bloolean input specifiying if EFI plots are to be generated        - default = True
        Author - S.Tait 2023 
    """
    
    wavelength = lambda_ *1E9
    
    # paramaters of air layer before coating
    
    n_air     = materialParams[air_index]['n']
    
    # set up coating 
    


    n_coat = np.zeros(np.shape(dOpt))  # refractive index of each layer
    t_coat = np.zeros(np.shape(dOpt))  # physical thickness of each layer in nm
    k_coat = np.zeros(np.shape(dOpt))  # attenuation coefficients for each layer

    for layer_idx, layer_material in enumerate(materialLayer):
        n_coat[layer_idx] = materialParams[layer_material]['n']
        t_coat[layer_idx ] = optical_to_physical(dOpt[layer_idx], wavelength,materialParams[layer_material]['n']) 
        k_coat[layer_idx] = materialParams[layer_material]['k']        
       

    
    n_coat_complex = np.asarray([complex(n_i,k_i) for n_i, k_i in zip(n_coat,k_coat)])
    
    # substrate parameters 
    
    n_sub    = materialParams[substrate_index]['n']  # refractive index of silica at the laser wavelength of 1064 nm
    t_sub    = 100  # thickness of substrate in nm 
    
    total_thickness = t_air + sum(t_coat) + t_sub                             # total thickness of system in  nm 
    
    ##################################################
    # set up calculation of EFI 
    #polarisation = 'p'                                            #polarisation of light
    angle      = np.deg2rad(0)                                     #angle of incidence - assuming normal incidence
    
    n_list     =  np.append(complex(n_air,0),n_coat_complex) 
    n_list     =  np.append(n_list,complex(n_sub,0))                           # theres definetly a better way to do this 
    
    # EFI requires values are wrapped in inf for some reason 
    t_list     =  np.append(np.inf,t_coat)                         # theres definitely a better way to do this 
    t_list     =  np.append(t_list,np.inf)
    
    

    
    
    #t_listsub  = np.insert(t_listsub,[0,len(t_listsub)],np.inf )  # EFI requires values are wrapped in inf for some reason 
    coh_tmm_data = tmm.coh_tmm(polarisation,n_list,t_list,th_0=angle,lam_vac=wavelength) #theta set to 0 (this is for the pump remember)
    coh_tmm_data_sub = tmm.coh_tmm(polarisation,n_list,t_list,th_0=angle,lam_vac=wavelength)
    
    reflectivity = coh_tmm_data['R']
    #####
    
    num_points = 1000# , 5000, 10000, 30000]:
    ds = np.linspace(-t_air, sum(t_coat) + t_sub, num=num_points)

# ds = np.linspace(-t_air,sum(t_coat)+t_sub,num=30000) #position in structure

    poyn=[]
    absor=[]
    poyn_in = []
    absor_in = []
    E = []
    E_sub = []
    layer_idx = []
    
    
    for d in ds:
        layer, d_in_layer = tmm.find_in_structure_with_inf(t_list,d)
        data        = tmm.position_resolved(layer,d_in_layer,coh_tmm_data)
        data_sub    = tmm.position_resolved(layer,d_in_layer,coh_tmm_data_sub)

        poyn.append(data['poyn'])
        absor.append(data['absor'])
        E.append(np.abs(data['Ex'])**2) # Ex is for p-polarisation
        E_sub.append(np.abs(data_sub['Ex'])**2)
        layer_idx.append(layer) 
    
    E     = np.array(E)
    E_sub = np.array(E_sub)
    poyn  = np.array(poyn) 
    absor = np.array(absor)
    
    # # ... (rest of the code)
    total_absorption = np.trapz(absor, ds)
    
    # tmm calcualtes for forward and backward propogation of the light in the coating  x2 
    # to match with the calcutions of TFCalc : total_absorption*1E5/2} 
    total_absorption = total_absorption*1E6/2
    

    if plots: 
        import matplotlib as mpl  # Ensure you have this import
        
        # Plotting the thin film stack
        unique_materials = sorted(set(materialLayer))
        
        # Use gist_rainbow_r colormap, resampled(48)
        cmap = mpl.colormaps["viridis_r"].resampled(20)
        colors = cmap(np.linspace(0, 1, len(unique_materials)))
        
        # Map each unique material to its color index
        
        

        material_to_color_idx = {um: i for i, um in enumerate(unique_materials)}
        depth_so_far = 0
        fig, ax1 = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(right=0.7)
        labeled_materials = set()

        # Plot the coating layers
        for i in range(len(materialLayer)):
            material_idx = materialLayer[i]
            color_index = material_to_color_idx[material_idx]
            
            if materialParams[material_idx]['name'] not in labeled_materials:
                label = materialParams[material_idx]['name']
                labeled_materials.add(label)
            else:
                label = None

            ax1.bar(
                depth_so_far + t_coat[i] / 2,  # bar center position
                t_coat[i],                     # bar height (thickness)
                color=colors[color_index],
                width=t_coat[i],
                label=label
            )
            
            depth_so_far += t_coat[i]

            # Only add the vertical line and print details if depBreak is specified
            if depBreak is not None and i == depBreak - 1:
                ax1.axvline(x=depth_so_far, color='black', linestyle='--')
                # Add annotation for deposition boundary
                ax1.text(
                    x=depth_so_far + 50,  # Adjust x position as needed
                    y=700, 
                    s="Deposition Boundary", 
                    fontsize=10, 
                    ha='left', 
                    va='top'
                )
                
                # Print details for the layer before the break
                before_layer = materialParams[material_idx]
                before_thickness = t_coat[i]
                before_position = i + 1  # converting to 1-indexed position
                print(f"Vertical line inserted after layer {before_position}:")
                print(f"Layer before break - Position: {before_position}, Material: {before_layer['name']}, Thickness: {before_thickness}")
                
                # Print details for the layer immediately after the break, if it exists
                if i + 1 < len(materialLayer):
                    next_material_idx = materialLayer[i + 1]
                    after_layer = materialParams[next_material_idx]
                    after_thickness = t_coat[i + 1]
                    after_position = i + 2  # converting to 1-indexed position
                    print(f"Layer after break - Position: {after_position}, Material: {after_layer['name']}, Thickness: {after_thickness}")
                else:
                    print("No layer exists after the break.")
            
        # Extend the left limit a bit more for the annotation & arrow
   
        
        # ---------------------------
        # ADD FINAL LAYER (SUBSTRATE)
        # ---------------------------
        # depth_so_far is now sum of all coating thicknesses.
        # Plot the substrate bar to the right side (outside the main coating stack).
        # Adjust color as you wish; here 'gray' is used.
        substrate_color = 'gray'
        ax1.bar(
            depth_so_far + (t_sub*4) / 2,  # center of the substrate bar
            700,                     # height of the bar
            color=substrate_color,
            width=t_sub * 10,          # make the bar thicker
            label='substrate'
        )
        # Update x-limits so you can see the substrate bar
        # (Otherwise, it might appear outside your plotted range)
        ax1.set_xlim([-t_air, depth_so_far + (t_sub*10)])

        ax1.set_ylabel('Physical Thickness [nm]')
        ax1.set_xlabel('Layer Position')
        ax1.set_title('Generated Stack')
        ax1.grid(False)
        

        
        # Plotting the electric field on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.grid()
        ax2.plot(ds, E_sub, 'blue')
        ax2.set_xlabel('depth (nm)')
        ax2.set_ylabel('Normalised Electric Field Intensity')
        ax2.set_ylim([0, np.max(E_sub)*1.2])
        
        
        # ---------------------------
        # ADD AN ARROW ON THE LEFT
        # ---------------------------
        # We'll place an arrow pointing right at x = -t_air, labeled "Light Propagation."
        # Adjust the arrow start/end positions and text location to your preference.
        # Let's first compute the y-position for the arrow based on the maximum EFI
        # 2. Place multiline text at a chosen (x, y). 
        left_margin =500
        ax1.set_xlim([-t_air - left_margin, depth_so_far + t_sub * 1.1])
    #    The example below places it near y=700 on the primary axis.
        ax1.text(
            x=-t_air - 150, 
            y=700, 
            s="Light Propagation\n----------->",  # The \n creates a new line
            fontsize=10, 
            ha='left',  # horizontal alignment of the text
            va='top'    # vertical alignment relative to y=700
        )
 
        
        # Explicitly call legend on ax1
       # Place legend on the right side, outside the main area:
        ax1.legend(
            loc='center left',          # position legend center-left
            bbox_to_anchor=(1.2, 0.5),  # anchor it just outside the axes
            borderaxespad=0.7,          # padding between axes and legend box
            fancybox=True,              # optional styling
            shadow=True                 # optional styling
        )
        plt.show()

    return E_sub, layer_idx,  ds,E, poyn, total_absorption, reflectivity
    

def CalculateAbsorption_tmm(dOpt, materialLayer, materialParams, lambda_=1064, t_air=500, polarisation='p'):
    """
    Calculate the absorption at each position within the layers of a thin film stack using tmm.
    """

    wavelength = lambda_ * 1e-9  # Convert to meters

    # Air layer parameters
    n_air = materialParams[999]['n']

    # Set up the coating stack
    n_coat = [materialParams[layer]['n'] for layer in materialLayer]
    k_coat = [materialParams[layer]['k'] for layer in materialLayer]
    t_coat = [dOpt[layer] for layer in materialLayer]  # Assuming dOpt contains the thickness of each layer

    # Refractive index list for the stack including air and substrate
    n_list = [complex(n_air, 0)] + [complex(n, k) for n, k in zip(n_coat, k_coat)]
    
    # Thickness list for the stack including air and substrate
    d_list = [np.inf] + t_coat + [np.inf]

    # Angle of incidence
    angle = 0

    # Calculate coherent TMM data
    coh_tmm_data = tmm.coh_tmm(polarisation, n_list, d_list, angle, wavelength)

    # Create a function to calculate absorption at any given depth in a layer
    def absorption_fn(layer, depth_in_layer):
        n = n_list[layer]
        return tmm.absorp_analytic_fn(layer, depth_in_layer, coh_tmm_data)

    # Calculate absorption in each layer
    absorption = []
    for i, thickness in enumerate(t_coat, start=1):  # start=1 skips the air layer
        absorption_in_layer = [absorption_fn(i, d) for d in np.linspace(0, thickness, num=100)]
        absorption.append(absorption_in_layer)

    return absorption

def CalculateTransmission_tmm(dOpt, materialLayer, materialParams, lambda_list,lambda_0,tphys=None ,polarisation='p',plots=False,plot_range=None):
    """
    Calculate the transmission as a function of wavelength for a thin film stack using tmm.

    Inputs:
        dOpt           : list or array of optical thicknesses
        materialLayer  : list of material identifiers for each layer
        materialParams : dictionary of material properties, including 'n' and 'k' (constants or functions of wavelength)
        lambda_list    : list or array of wavelengths (in nm)
        lambda_0       : design wavelength of coating (in nm) i.e 1064
        tphys          : optional list or array of physical thicknesses (in nm) - Default: None
        polarisation   : 's' or 'p' polarization
        plot_range     : optional x range constraints on Transmission plotting - Default :[380,1500 ] to match TFCalc Plotting 

    Returns:
        wavelengths    : array of wavelengths (nm)
        transmission   : array of transmission values corresponding to wavelengths
    """
    
       # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Air layer parameters
    n_air = materialParams[999]['n']
    k_air = materialParams[999].get('k', 0)

    # Substrate parameters
    n_sub = materialParams[1]['n']
    k_sub = materialParams[1].get('k', 0)

    # Initialize lists to store results
    transmission = []
    wavelengths = []
    
    # convert dOpt to tphys 
    # Calculate physical thicknesses
    
    
    # if tphys is None:
    #     print('Calculating physical thickness... ')
    #     tphys = []
    #     for layer_material in materialLayer:
    #         n_i = materialParams[layer_material]['n']
    #         physical_thickness = dOpt[layer_material] * lambda_0 / n_i  # in nm
    #         # print(f"Layer {layer_material}: Optical Thickness: {dOpt[layer_material]}, Physical Thickness: {physical_thickness}, Refractive Index: {n_i}")
    #         tphys.append(physical_thickness)
    #     tphys = np.array(tphys)
    # else:
    #     print('Using Physical Thickness... ')
    
    if tphys is None:
        print('tmm - Calculating physical thickness... ')
        tphys = []
        for layer_material in materialLayer:
            n_i = materialParams[layer_material]['n']
            physical_thickness = optical_to_physical(dOpt[layer_material], lambda_0, n_i)
            # print(f"Layer {layer_material}: Optical Thickness: {dOpt[layer_material]}, Physical Thickness: {physical_thickness}, Refractive Index: {n_i}")
            tphys.append(physical_thickness)
        tphys = np.array(tphys)
    else:
        print('Using Physical Thickness... ')

    for lambda_ in lambda_list:
        wavelength = lambda_
        

        # Get refractive indices and k values
        n_coat = []
        k_coat = []

        for layer_idx, layer_material in enumerate(materialLayer):
            n_material = materialParams[layer_material]['n']
            k_material = materialParams[layer_material].get('k', 0)

            # If n and k are functions, evaluate them
            if callable(n_material):
                n_i = n_material(lambda_)
            else:
                n_i = n_material  # Constant

            if callable(k_material):
                k_i = k_material(lambda_)
            else:
                k_i = k_material  # Constant

            n_coat.append(n_i)
            k_coat.append(k_i)

        # Build n_list and d_list
        n_list = [complex(n_air, k_air)] + [complex(n_i, k_i) for n_i, k_i in zip(n_coat, k_coat)] + [complex(n_sub, k_sub)]
        d_list = [np.inf] + (tphys).tolist() + [np.inf]  # Convert nm to meters

        # Angle of incidence
        angle = 0  # Normal incidence in degrees
        
        # Suppress runtime warnings but capture them
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                # Use tmm to compute the data
                coh_tmm_data = tmm.coh_tmm(polarisation, n_list, d_list, angle * np.pi / 180, wavelength)
                T = coh_tmm_data['T']
            except Exception as e:
                logger.error(f"Error at wavelength {lambda_} nm: {e}")
                T = np.nan

            # Check if any warnings were caught
            if w:
                # Inform the user that an issue occurred
                logger.warning(f"Issue encountered at wavelength {lambda_} nm.")
                for warning in w:
                    logger.warning(f"RuntimeWarning: {warning.message}")

        
        

        # # Use tmm to compute the data
        # coh_tmm_data = tmm.coh_tmm(polarisation, n_list, d_list, , wavelength)

        # Get the transmittance
        T = coh_tmm_data['T']

        # Store the results
        wavelengths.append(lambda_)
        transmission.append(T)

    # Convert to arrays
    idx = np.abs(np.array(wavelengths) - lambda_0).argmin()
    wavelengths = np.array(wavelengths)*1E9
    
    transmission = np.array(transmission)
    # Find the index where wavelength is closest to lambda_0

    # Get the corresponding transmission value
    transmission_lambda_0 = transmission[idx]
    
    
    if plots: 
        # Default plot range if not specified
        if plot_range is None:
            plot_range = [380, 1500]

        # Assuming wavelengths and transmission are lists or arrays
        data = {
            'Wavelength (nm)': wavelengths,
            'Transmission (%)': transmission * 100
        }
        df = pd.DataFrame(data)

        # Create an interactive line plot with customized hover data
        fig = px.line(
            df,
            x='Wavelength (nm)',
            y='Transmission (%)',
            title='Transmission vs. Wavelength',
            hover_data={
                'Wavelength (nm)': ':.2f',     # Format to 2 significant figures
                'Transmission (%)': ':.2g',    # Format to 2 significant figures
            }
        )

        # Customize axes labels, grid, and range
        fig.update_layout(
            xaxis_title='Wavelength (nm)',
            yaxis_title='Transmission (%)',
            xaxis=dict(
                showgrid=True,
                range=plot_range  # Set the x-axis range
            ),
            yaxis=dict(showgrid=True),
        )

        # Show the plot
        fig.show()

    return wavelengths, transmission, transmission_lambda_0 




def CalculateTransmission_tmm2(dOpt, materialLayer, materialParams, lambda_list,lambda_0,tphys=None ,polarisation='p',plots=False,plot_range=None):
    """
    Calculate the transmission as a function of wavelength for a thin film stack using tmm.

    Inputs:
        dOpt           : list or array of optical thicknesses
        materialLayer  : list of material identifiers for each layer
        materialParams : dictionary of material properties, including 'n' and 'k' (constants or functions of wavelength)
        lambda_list    : list or array of wavelengths (in nm)
        lambda_0       : design wavelength of coating (in nm) i.e 1064
        tphys          : optional list or array of physical thicknesses (in nm) - Default: None
        polarisation   : 's' or 'p' polarization
        plot_range     : optional x range constraints on Transmission plotting - Default :[380,1500 ] to match TFCalc Plotting 

    Returns:
        wavelengths    : array of wavelengths (nm)
        transmission   : array of transmission values corresponding to wavelengths
    """
    
       # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if tphys is None:
        print('Calculating physical thickness... ')
        tphys = []
        for layer_material in materialLayer:
            n_i = materialParams[layer_material]['n']
            physical_thickness = dOpt[layer_material] * lambda_0 / n_i  # in nm
            # print(f"Layer {layer_material}: Optical Thickness: {dOpt[layer_material]}, Physical Thickness: {physical_thickness}, Refractive Index: {n_i}")
            tphys.append(physical_thickness)
        tphys = np.array(tphys)
    else:
        print('Using Physical Thickness... ')



    # Convert everything to arrays
    transmission = []
    wavelengths  = np.array(lambda_list)  # in nm

    # Grab substrate, air from your dictionary
    n_air = materialParams[999]['n']
    k_air = materialParams[999].get('k', 0)
    n_sub = materialParams[1]['n']
    k_sub = materialParams[1].get('k', 0)

    for wnm in wavelengths:
        # Build n_list for each layer at this wavelength wnm
        n_coat = []
        k_coat = []
        for i, layer_material in enumerate(materialLayer):
            n_material = materialParams[layer_material]['n']
            k_material = materialParams[layer_material].get('k', 0)
            if callable(n_material):
                n_i = n_material(wnm)
            else:
                n_i = n_material
            if callable(k_material):
                k_i = k_material(wnm)
            else:
                k_i = k_material
            n_coat.append(n_i)
            k_coat.append(k_i)

        # Build full n_list and d_list (still in nm)
        n_list = [complex(n_air, k_air)] + [complex(a, b) for a,b in zip(n_coat, k_coat)] + [complex(n_sub, k_sub)]
        d_list = [np.inf] + tphys.tolist() + [np.inf]

        # TMM at normal incidence => 0 degrees
        angle_radians = 0.0

        # If python-tmm uses the convention: "wavelength in same unit as thickness"
        # we are consistent: thickness in nm, wavelength in nm
        try:
            result = tmm.coh_tmm(polarisation, n_list, d_list, angle_radians, wnm)
            T      = result['T']
        except Exception as e:
            T = np.nan
            logger.error(f"Error at wavelength {wnm} nm: {e}")

        transmission.append(T)

    transmission = np.array(transmission)
    
    # Identify index closest to lambda_0 (in nm)
    idx = np.abs(wavelengths - lambda_0).argmin()
    transmission_lambda_0 = transmission[idx]
    print('CalculateTransmission_tmm2')
    print("Wavelengths (first 10):", wavelengths[:10])
    print("Transmission (first 10):", transmission[:10])

    # Optionally plot
    if plots:
        if plot_range is None:
            plot_range = [380, 1500]
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame({
            'Wavelength (nm)': wavelengths,
            'Transmission (%)': transmission * 100
        })
        fig = px.line(
            df,
            x='Wavelength (nm)',
            y='Transmission (%)',
            title='Transmission vs. Wavelength'
        )
        fig.update_layout(
            xaxis=dict(range=plot_range),
            xaxis_title='Wavelength (nm)',
            yaxis_title='Transmission (%)',
        )
        fig.show()

    return wavelengths, transmission, transmission_lambda_0


def build_n_list(materialLayer,materialParams):
    """
    
    Function takes in an array of integers which are used to specify the position of a given layer in a stack. 
    layers are specified where light is propogating from the 0th idx to the last, with last being deposited on 
    the substrate. 

    Author STait 2025 
    
    
    Construct array of complex refractive indicies to be passed to tmm-fast
    
    
    Args:
      materialLayer  (np.array) : array of integer values. Specifies the position of each layer in a given coating. Used as key's
                                for materialParams
      materialParams (dict):  dictionary containing the material properties of different thin film and bulk materials. 
     
    Returns a dictionary with keys:
      "n_list": list of  complex touples containing the real and imaginary refractive index for each layer in materialLayer
    
    """
    
    n_coat, k_coat = [], []
    for layer_material in materialLayer:
        n_material = materialParams[layer_material]['n']
        k_material = materialParams[layer_material].get('k', 0)
        # For now we assume constant indices (no wavelength dependence)
        n_i = n_material  
        k_i = k_material  
        n_coat.append(n_i)
        k_coat.append(k_i)
    # Build full ns list: incident medium, then layers, then substrate.
    # Here, incident = 1.00 and substrate = 1.44 (with negligible absorption)
    n_list = [complex(1.00, 0.0)] + [complex(n_i, k_i) for n_i, k_i in zip(n_coat, k_coat)] + [complex(1.44, 0)]
    return n_list


# Print out sample comparisons.
def nearest_index(arr, val):
    """
    
    helper funciton to find nearest value for tmm-fast
    Author STait 2025 
    """
    return np.argmin(np.abs(arr - val))


    # stack_RT_fast(dOpt, materialLayer, materialParams, lambda_list,lambda_0,tphys=None ,polarisation='p',plots=False,plot_range=None):
def stack_RT_fast(dOpt, materialLayer, materialParams, lambda_list,lambda_0 ,tphys=None, polarisation='unploarised',theta=0.0,  plots=False,plot_range=None,verbose=False):
    
    
    # stack, theta=0.0, pol='unpolarized'):
    """
    
    wrapper for tmm-fast to calcualte transmission spectra, reflectivity and transmission 
    Author STait 2025 
    
    origional tmm-fast from Alexander Luce et al  Vol. 39, Issue 6, pp. 1007-1013 (2022) •https://doi.org/10.1364/JOSAA.450928
    
    
    Calculate the reflectance and transmission spectra of a stack using fast-tmm.
    Returns a dictionary with keys:
      "refl": amplitude reflectivity (taken as sqrt(intensity), phase lost)
      "R": intensity reflectivity (power, i.e. |r|^2)
      "T": transmission.
    
    Args:
      wavelengths (float or array-like): Wavelength(s) in meters.
      stack (dict): Dictionary with keys:
          "ns": list of refractive indices [incident, layer1, ..., substrate]
          "Ls": list or array of physical thicknesses for the coating layers.
                  (Must satisfy len(ns) == len(Ls)+2.)
      theta (float, optional): Angle of incidence in radians (default 0.0).
      pol (str, optional): 's'/'te', 'p'/'tm', or 'unpolarized' (default).
    
    Returns:
      dict: {"refl": amplitude (sqrt of intensity), "R": intensity, "T": transmission, "pol":pol}
    """


    #build complex refractive index list including air and sub 
    n_list = build_n_list(materialLayer,materialParams)


    #calculate physical thicknesses if not provided 
    if tphys is None:
        
        print('tmm-fast : Calculating physical thickness... ')
        tphys = []
        for layer_material in materialLayer:
            n_i = materialParams[layer_material]['n']
            physical_thickness = dOpt[layer_material] * lambda_0 / n_i  # in nm
            if verbose:
                print(f"Layer {layer_material}: Optical Thickness: {dOpt[layer_material]}, Physical Thickness: {physical_thickness}, Refractive Index: {n_i}")
            tphys.append(physical_thickness)
        tphys = np.array(tphys)
    else:
        print('tmm-fast : Using Physical Thickness... ')




    ns = n_list
    Ls = tphys

    # Build the full thickness array by adding semi-infinite boundaries.
    # This ensures that the thickness array length equals len(ns).
    d_list = [np.inf] + list(Ls) + [np.inf]   # NEW: full array of length = len(Ls) + 2
    
    
    d_array_1d = np.array(d_list, dtype=float) # Now d_array_1d.shape is (len(ns),)
    # === Modified Lines End ===
    
    # Convert refractive indices.
    n_array_1d = np.array(ns, dtype=complex)    # shape: (len(ns),)

    # Ensure wavelengths is an array.
    lambda_array = np.atleast_1d(np.array(lambda_list, dtype=float))  # shape: (Nwl,)
    Nwl = lambda_array.size

    # Expand to fast-tmm expected shapes:
    # n_array_3d: shape (1, len(ns), Nwl)
    n_array_3d = n_array_1d[None, :, None]       # shape: (1, len(ns), 1)
    n_array_3d = np.tile(n_array_3d, (1, 1, Nwl))  # shape: (1, len(ns), Nwl)
    # d_array_2d: shape (1, len(ns))
    d_array_2d = d_array_1d[None, :]              # shape: (1, len(ns))
    # Angle array:
    theta_array = np.array([theta], dtype=float)  # shape: (1,)

    if polarisation.lower() in ['s', 'te']:
        result = coh_tmm_fast('s', n_array_3d, d_array_2d, theta_array, lambda_array)
    elif polarisation.lower() in ['p', 'tm']:
        result = coh_tmm_fast('p', n_array_3d, d_array_2d, theta_array, lambda_array)
    else:
        result_s = coh_tmm_fast('s', n_array_3d, d_array_2d, theta_array, lambda_array)
        result_p = coh_tmm_fast('p', n_array_3d, d_array_2d, theta_array, lambda_array)
        result = {'R': 0.5 * (result_s['R'] + result_p['R']),
                  'T': 0.5 * (result_s['T'] + result_p['T'])}

    # In fast-tmm, result['R'] is the intensity reflectivity.
    intensity = result['R'][0, 0, :]  # shape: (Nwl,)
    amplitude = np.sqrt(intensity)     # (phase information lost)
    T = result['T'][0, 0, :] if 'T' in result and result['T'] is not None else None


    wavelengths = lambda_list 
    
    transmission = T 
   
    lambda_0_idx=  nearest_index(lambda_array, lambda_0)
    
    transmission_lambda_0 = T[lambda_0_idx]

     # Optionally plot
    if plots is not None:
        transmission_spec_scaled = transmission*100 

        if plots == True :
            
            plt.figure(figsize=(10, 6))
            plt.plot(lambda_list, transmission_spec_scaled, label='Transmission Spectrum (Fast)')
            plt.xlabel('Wavelength (m)')
            plt.ylabel('Transmission')
            plt.title('Transmission Spectrum')
            plt.grid(True)
            plt.legend()
            plt.show()
    
        elif isinstance(plots, str) and 'plotly' in plots.lower():
            
            if plot_range is None:
                    plot_range = [380, 1500]
            import pandas as pd
            import plotly.express as px
            df = pd.DataFrame({
                'Wavelength (nm)': lambda_list*1E9,
                'Transmission (%)': transmission_spec_scaled ,
            })
            fig = px.line(
                df,
                x='Wavelength (nm)',
                y='Transmission (%)',
                title='Transmission vs. Wavelength',
                hover_data={
                'Wavelength (nm)': ':.2f',     # Format to 2 decimal places
                'Transmission (%)': ':.4f',    # Format to 2 decimal places
                }
            )
            fig.update_layout(
                xaxis=dict(range=plot_range),
                xaxis_title='Wavelength (nm)',
                yaxis_title='Transmission (%)',
            )
            fig.show()



    return wavelengths,transmission, transmission_lambda_0, {"refl": amplitude, "R": intensity, "T": T, "pol": polarisation}