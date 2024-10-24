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

def CalculateEFI_tmm(
        layer_thicknesses ,
        layer_materials, 
        material_parameters,
        light_wavelength =1064 ,
        t_air = 500,
        polarisation='p' ,
        plots ='False',
        num_points=20000,
        air_index = 0,
        substrate_index = 1):
    
    
    # function calcualtes the normallised electric field intensity inside a thin film coating/coating stack usign tmm. This method 
    # takes into acount of : 
    # light_wavelength          the wavelength of light (nm)  incident on the coxating stack         - default =  1064 nm 
    # t_air            the total thickness of air before the coating stack                 - default = 500 nm 
    # materailParams : Dictionary variable containing the material propeties of each layer in the coating stack (including air) 
    #                  Requires refractive       index values 'n' and complex refractive index values a.k.a attenuation coeffiencts  'k'
    # 
    # polarisation   : string variable specifying either 'p' or 's' polarised light       - default = 'p' polarisation 
    # plots          : bloolean input specifiying if EFI plots are to be generated        - default = True
    #
    # Author - S.Tait 2023 
    
    
    # wavelength in m
    wavelength = light_wavelength *1E9
    
    # paramaters of air layer before coating
    
    n_air     = material_parameters[air_index]['n']
    
    n_coat_complex = np.asarray([complex(material_parameters[i]["n"],material_parameters[i]["k"]) for i in layer_materials])
    # substrate parameters 
    
    n_sub    = material_parameters[substrate_index]['n']  # refractive index of silica at the laser wavelength of 1064 nm
    t_sub    = 100  # thickness of substrate in nm 
    
    total_thickness = t_air + sum(layer_thicknesses) + t_sub                             # total thickness of system in  nm 
    
    ##################################################
    # set up calculation of EFI 
    #polarisation = 'p'                                            #polarisation of light
    angle      = np.deg2rad(0)                                     #angle of incidence - assuming normal incidence
    
    # appending complex refractive index and k values for air and substrate
    all_refractive_indices = np.concatenate([[complex(n_air, 0)], n_coat_complex, [complex(n_sub, 0)]])
    
    # EFI requires values are wrapped in inf for some reason 
    # Appending infs to start and end
    all_thicknesses = np.concatenate([[np.inf], layer_thicknesses, [np.inf]])
    
    # running 
    coh_tmm_data = tmm.coh_tmm(polarisation,all_refractive_indices,all_thicknesses,th_0=angle,lam_vac=wavelength) #theta set to 0 (this is for the pump remember)
    
    # setup array to compute the EFI over, linearly space points throughout structure
    efi_positions = np.linspace(-t_air,total_thickness-t_air,num=num_points)
    
    poyn=[]
    absor=[]
    poyn_in = []
    absor_in = []
    E = []
    E_sub = []
    layer_idx = []
    
    
    for d in efi_positions:
        layer, d_in_layer = tmm.find_in_structure_with_inf(all_thicknesses,d)
        data = tmm.position_resolved(layer,d_in_layer,coh_tmm_data)
        #data_sub = tmm.position_resolved(layer,d_in_layer,coh_tmm_data_sub)
        poyn.append(data['poyn'])
        absor.append(data['absor'])
        E.append(np.abs(data['Ex'])**2) # Ex is for p-polarisation
        #E_sub.append(np.abs(data_sub['Ex'])**2)
        layer_idx.append(layer) 
    
    E     = np.array(E)
    #E_sub = np.array(E_sub)
    poyn  = np.array(poyn) 
    absor = np.array(absor)
    

    if plots: 
        # Plotting the thin film stack
        unique_materials = list(set(all_refractive_indices))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_materials)))  # generate distinct colors for materials
        
        depth_so_far = 0  # To keep track of where to plot the next bar
        
        fig, ax1 = plt.subplots()
        for i in range(len(layer_materials)):
            material_idx = layer_materials[i]
            ax1.bar(depth_so_far + t_coat[i] / 2, t_coat[i], color=colors[material_idx],
                    width=t_coat[i])
            depth_so_far += t_coat[i]
        #ax1.set_xlim([0, sum(t_coat) * 1.01])
        ax1.set_ylabel('Physical Thickness [nm]')
        ax1.set_xlabel('Layer Position')
        #ax1.set_xlim([-t_air,np.sum(t_coat)])
        ax1.set_title('Generated Stack')
        ax1.grid(False)
        
        
        ax2 = ax1.twinx()
        ax2.grid()
        ax2.plot(ds,E_sub,'blue') #,ds,Ey,'purple',ds,absor*200,
        ax2.set_xlabel('depth (nm')
        ax2.set_ylabel('Normallised Electric Feild Intensity')
        #ax2.set_xlim([-t_air,np.sum(t_coat)])
        #plt.vlines([0,t_coat,total_thickness],0,2)
        #ax2.set_ylim([0,np.max(E_sub)*1.2])
        plt.show()

    return E, layer_idx, efi_positions
    

def CalculateAbsorption_tmm(dOpt, materialLayer, materialParams, light_wavelength=1064, t_air=500, polarisation='p'):
    """
    Calculate the absorption at each position within the layers of a thin film stack using tmm.
    """

    wavelength = light_wavelength * 1e-9  # Convert to meters

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
