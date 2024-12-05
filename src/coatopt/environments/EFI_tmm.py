import math as m
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, leastsq, least_squares
from scipy import integrate as TG
import datetime as dt
import pandas as pd
import openpyxl as xl
import tmm
import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt 
import os 

def CalculateEFI_tmm(dOpt ,materialLayer, materialParams,lambda_ =1064 ,t_air = 500,polarisation='p' ,plots ='False',air_index=0,substrate_index=1):
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
    
  
    
    def optical_to_physical(optical_thickness, wavelength, refractive_index):
        physical_thickness = optical_thickness*wavelength/ refractive_index
        return physical_thickness

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
    
    #####
    
    
    for num_points in [1000, 5000, 10000, 30000]:
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
            data = tmm.position_resolved(layer,d_in_layer,coh_tmm_data)
            data_sub = tmm.position_resolved(layer,d_in_layer,coh_tmm_data_sub)
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
        # Plotting the thin film stack
        unique_materials = list(set(materialLayer))
        

       
       
        
        colors = plt.cm.viridis(np.linspace(0, 1, np.max(unique_materials)+2))  # generate distinct colors for materials
      
        depth_so_far = 0  # To keep track of where to plot the next bar
        
      
        
        fig, ax1 = plt.subplots()
        for i in range(len(materialLayer)):
            material_idx = materialLayer[i]
            # try:
            ax1.bar(depth_so_far + t_coat[i] / 2, t_coat[i], color=colors[material_idx],
                    width=t_coat[i])
            depth_so_far += t_coat[i]


                
        ax1.set_xlim([-1 , sum(t_coat) * 1.01])
        ax1.set_ylabel('Physical Thickness [nm]')
        ax1.set_xlabel('Layer Position')
        ax1.set_xlim([-t_air,np.sum(t_coat)])
        ax1.set_title('Generated Stack')
        ax1.grid(False)
        
        
        
        ax2 = ax1.twinx()
        ax2.grid()
        ax2.plot(ds,E_sub,'blue') #,ds,Ey,'purple',ds,absor*200,
        ax2.set_xlabel('depth (nm')
        ax2.set_ylabel('Normallised Electric Feild Intensity')
        ax2.set_xlim([-t_air,np.sum(t_coat)])
        #plt.vlines([0,t_coat,total_thickness],0,2)
        ax2.set_ylim([0,np.max(E_sub)*1.2])
        
        plt.legend()
        plt.show()

    return E_sub, layer_idx,  ds,E, poyn, total_absorption
    

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

def CalculateTransmission_tmm(dOpt, materialLayer, materialParams, lambda_list,lambda_=1064E-9, t_air=500, polarisation='p'):
    """
    Calculate the transmission as a function of wavelength for a thin film stack using tmm.

    Inputs:
        dOpt           : list or array of optical thicknesses (in units of wavelength)
        materialLayer  : list of material identifiers for each layer
        materialParams : dictionary of material properties, including 'n' and 'k' (constants or functions of wavelength)
        lambda_list    : list or array of wavelengths (in nm)
        t_air          : thickness of air layer before the coating stack (in meters, if used)
        polarisation   : 's' or 'p' polarization

    Returns:
        wavelengths    : array of wavelengths (nm)
        transmission   : array of transmission values corresponding to wavelengths
    """

    # Air layer parameters
    n_air = materialParams[999]['n']
    k_air = materialParams[999].get('k', 0)

    # Substrate parameters
    n_sub = materialParams[1]['n']
    k_sub = materialParams[1].get('k', 0)

    # Initialize lists to store results
    transmission = []
    wavelengths = []

    for lambda_ in lambda_list:
        wavelength = lambda_ * 1e-9  # Convert nm to meters

        # Initialize lists for this wavelength
        n_coat = []
        k_coat = []
        t_coat = []

        # Compute refractive indices and physical thicknesses
        for layer_idx, layer_material in enumerate(materialLayer):
            # Get refractive index and k at this wavelength
            n_material = materialParams[layer_material]['n']
            k_material = materialParams[layer_material]['k']

            # If n and k are functions of wavelength, evaluate them
            if callable(n_material):
                n_i = n_material(lambda_)
            else:
                n_i = n_material  # Assume constant

            if callable(k_material):
                k_i = k_material(lambda_)
            else:
                k_i = k_material  # Assume constant

            n_coat.append(n_i)
            k_coat.append(k_i)

            # Compute physical thickness in meters
            optical_thickness = dOpt[layer_idx]
            physical_thickness = optical_thickness * lambda_ / n_i  # Now in meters
            t_coat.append(physical_thickness)

        # Build n_list and d_list with consistent units (meters)
        n_list = [complex(n_air, k_air)] + [complex(n_i, k_i) for n_i, k_i in zip(n_coat, k_coat)] + [complex(n_sub, k_sub)]
        d_list = [np.inf] + t_coat + [np.inf]  # Thicknesses in meters

        # Angle of incidence
        angle = 0  # Normal incidence in degrees

        # Use tmm to compute the data
        coh_tmm_data = tmm.coh_tmm(polarisation, n_list, d_list, angle * np.pi / 180, wavelength)

        # Get the transmittance
        T = coh_tmm_data['T']

        # Store the results
        wavelengths.append(lambda_)
        transmission.append(T)

    # Convert to arrays
    wavelengths = np.array(wavelengths)
    transmission = np.array(transmission)

    return wavelengths, transmission


