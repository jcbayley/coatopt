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
import plotly.express as px
import warnings
import logging
        
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

def CalculateEFI_tmm(dOpt=None,materialLayer=None, materialParams=None,lambda_ =1064 ,t_air = 500,polarisation='p' ,plots ='False',depBreak=None,tphys=None,verbose=False):
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
    
    wavelength = lambda_ 
    
    # paramaters of air layer before coating
    
    n_air     = materialParams[999]['n']
    
    # set up coating 
    if tphys is not None:
        if verbose:
            print('[EFI-tmm] Using physical thicknesses ... ')
        
        print(tphys )
        if np.size(tphys)==1:
            n_coat  = materialParams[materialLayer]['n']
            t_coat  = tphys
            k_coat  = materialParams[materialLayer]['k']
            
            
        else:
            n_coat = np.zeros(np.shape(tphys))  # refractive index of each layer
            t_coat = np.zeros(np.shape(tphys))  # physical thickness of each layer in nm
            k_coat = np.zeros(np.shape(tphys))  # attenuation coefficients for each layer
        
            for layer_idx, layer_material in enumerate(materialLayer):
                n_coat[layer_idx]  = materialParams[layer_material]['n']
                t_coat[layer_idx ] = tphys[layer_idx]
                k_coat[layer_idx]  = materialParams[layer_material]['k']
    
    elif dOpt is not None and tphys is None: 
        
        n_coat = np.zeros(np.shape(dOpt))  # refractive index of each layer
        t_coat = np.zeros(np.shape(dOpt))  # physical thickness of each layer in nm
        k_coat = np.zeros(np.shape(dOpt))  # attenuation coefficients for each layer

        for layer_idx, layer_material in enumerate(materialLayer):
            n_coat[layer_idx]  = materialParams[layer_material]['n']
            t_coat[layer_idx ] = optical_to_physical(dOpt[layer_idx], wavelength,materialParams[layer_material]['n'])
            k_coat[layer_idx]  = materialParams[layer_material]['k']
    else:
        raise ValueError("dOpt cannot be empty if tphys is not supplied")
    
    

     # substrate parameters 
    
    n_sub    = materialParams[1]['n']  # refractive index of silica at the laser wavelength of 1064 nm
    t_sub    = 100  # thickness of substrate in nm    
       

    
    if np.isscalar(n_coat) and np.isscalar(k_coat):
        n_coat_complex = np.array([complex(n_coat, k_coat)])
        total_thickness = t_air + t_coat + t_sub                             # total thickness of system in  nm 
    
    else:
        n_coat_complex = np.asarray([complex(n_i, k_i) for n_i, k_i in zip(n_coat, k_coat)])
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
    
    num_points = 1000# , 5000, 10000, 30000]:
    

    
    if isinstance(t_coat, np.ndarray) and t_coat.size == 1:
        ds = np.linspace(-t_air, t_coat + t_sub, num=num_points)
    else:
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

        # --------------------------
        # Define Layer Colors Using Provided Hex Codes
        # --------------------------
        layer_colors = [
            "#66B3D6",
            "#C166D6",
            "#D68966",
            "#7BD666",
            "#E15BDE",
            "#E1A05B",
            "#5BE15E"
        ]
        # Map each unique material to one of these colors (cycling if needed)
        if np.size(materialLayer) == 1:
            unique_materials = [materialLayer]
        else:
            unique_materials = sorted(set(materialLayer))
        material_to_color = {um: layer_colors[i % len(layer_colors)] for i, um in enumerate(unique_materials)}

        # --------------------------
        # Compute Cumulative Depths (for x-axis positioning)
        # --------------------------
        cumulative_depth = [0]
        
        
        if isinstance(t_coat, np.ndarray) and t_coat.size == 1:
            cumulative_depth.append(cumulative_depth[-1] + t_coat.item())
        else:
            for thickness in t_coat:
                cumulative_depth.append(cumulative_depth[-1] + thickness)
        total_coating = cumulative_depth[-1]

        # --------------------------
        # Dummy Electric Field Profile
        # --------------------------
        depth_profile = np.linspace(0, total_coating + t_sub, 500)
        E_field = np.abs(np.sin(np.pi * depth_profile / (total_coating + t_sub) * 6))

        # --------------------------
        # Create Subplots with Shared X-Axis
        # --------------------------
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Coating Stack Diagram", "Electric Field Profile")
        )

        # --------------------------
        # TOP SUBPLOT: Coating Stack as Rectangles and Hover Scatter
        # --------------------------
        shapes_top = []
        hover_x = []  # to store centers of bars for hover
        hover_y = []  # to store vertical center of bars
        hover_texts = []  # custom hover text for each layer

        depth_so_far = 0
        if np.size(materialLayer) == 1:
            # Handle single layer case
            thickness = t_coat.item() if isinstance(t_coat, np.ndarray) else t_coat
            x0 = depth_so_far
            x1 = depth_so_far + thickness
            y0 = 0
            y1 = thickness
            fillcolor = material_to_color[materialLayer]

            # Create rectangle shape for the layer.
            shapes_top.append(dict(
            type='rect',
            xref='x', yref='y',   # for top subplot
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color='black', width=1),
            fillcolor=fillcolor
            ))

            # Save center for hover trace.
            center_x = depth_so_far + thickness / 2
            center_y = thickness / 2
            hover_x.append(center_x)
            hover_y.append(center_y)
            hover_texts.append(f"Material: {materialParams[materialLayer]['name']}<br>Thickness: {thickness:.2f} nm")

            depth_so_far += thickness
        else:
            # Handle multiple layers
            depth_so_far = 0
            for i, mat_idx in enumerate(materialLayer):
                thickness = t_coat[i]
                x0 = depth_so_far
                x1 = depth_so_far + thickness
                y0 = 0
                y1 = thickness
                fillcolor = material_to_color[mat_idx]
                
                # Create rectangle shape for the layer.
                shapes_top.append(dict(
                    type='rect',
                    xref='x', yref='y',   # for top subplot
                    x0=x0, x1=x1,
                    y0=y0, y1=y1,
                    line=dict(color='black', width=1),
                    fillcolor=fillcolor
                ))
                
                # Save center for hover trace.
                center_x = depth_so_far + thickness / 2
                center_y = thickness / 2
                hover_x.append(center_x)
                hover_y.append(center_y)
                hover_texts.append(f"Material: {materialParams[mat_idx]['name']}<br>Thickness: {thickness} nm")
                
                depth_so_far += thickness

        # Add substrate rectangle to the top subplot.
        shapes_top.append(dict(
            type='rect',
            xref='x', yref='y',
            x0=total_coating, x1=total_coating + t_sub,
            y0=0, y1=1000,  # arbitrary height for display
            line=dict(color='black', width=1),
            fillcolor='gray'
        ))

        # Add an invisible scatter trace to provide hover information.
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode='markers',
                marker=dict(size=20, opacity=0),  # Invisible markers
                hoverinfo='text',
                hovertext=hover_texts,
                showlegend=False
            ),
            row=1, col=1
        )

        # Add shapes (rectangles) to the top subplot.
        fig.update_layout(shapes=shapes_top)

        # --------------------------
        # BOTTOM SUBPLOT: Electric Field Profile with Boundary Lines
        # --------------------------
        fig.add_trace(
            go.Scatter(
                x=depth_profile,
                y=E_sub,
                mode='lines',
                line=dict(color='blue'),
                name='Electric Field'
            ),
            row=2, col=1
        )

        # Create vertical dashed boundary lines for each layer boundary (using bottom subplot's reference).
        shapes_bottom = []
        for boundary in cumulative_depth[1:-1]:
            shapes_bottom.append(dict(
                type='line',
                xref='x2',  
                yref='y2 domain',  # confined to bottom subplot vertical domain (0 to 1)
                x0=boundary, x1=boundary,
                y0=0, y1=1,
                line=dict(color='black', width=1, dash='dash')
            ))

        # Add lightly shaded substrate region in the bottom subplot.
        shapes_bottom.append(dict(
            type='rect',
            xref='x2', yref='y2 domain',
            x0=total_coating, x1=total_coating + t_sub,
            y0=0, y1=1,
            fillcolor='gray',
            opacity=0.2,
            line_width=0
        ))

        # --------------------------
        # Combine Shapes and Update Layout
        # --------------------------
        try: 
            y_range = [0, max(t_coat)+t_sub]
        except:
            y_range = [0, t_coat+t_sub]
        
        fig.update_layout(
            shapes=shapes_top + shapes_bottom,
            # Set axis styles to mimic a Seaborn "ticks" style
            xaxis=dict(
                range=[-t_air, depth_so_far + t_sub],

                showline=True,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title='Physical Thickness [nm]',
                
                range=y_range,
                showline=True,
                linecolor='black',
                mirror=True
            ),
            xaxis2=dict(
                title='Depth (nm)',
                showline=True,
                linecolor='black',
                mirror=True
            ),
            yaxis2=dict(
                title='Electric Field Intensity',
                showline=True,
                linecolor='black',
                mirror=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width = 800
        )

        try:
            y=0.8 * max(t_coat)
        except: 
            y = 0.8 * t_coat
        
        # Add annotation for light propagation in the top subplot.
        fig.add_annotation(
            x=-180,
            y=y,
            text="AIR",
            showarrow=False,
            font=dict(size=12),
            xref='x',
            yref='y'
        )

        # Add legend traces for materials and substrate.
        legend_traces = []
        for um in unique_materials:
            legend_traces.append(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=10, color=material_to_color[um]),
                    name=materialParams[um]['name']
                )
            )
        legend_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color='gray'),
                name='Substrate'
            )
        )
        for trace in legend_traces:
            fig.add_trace(trace, row=1, col=1)

        fig.show()
    #     import matplotlib as mpl  # Ensure you have this import
        
    #     # Plotting the thin film stack
    #     unique_materials = sorted(set(materialLayer))
        
    #     # Use gist_rainbow_r colormap, resampled(48)
    #     cmap = mpl.colormaps["viridis_r"].resampled(20)
    #     colors = cmap(np.linspace(0, 1, len(unique_materials)))
        
    #     # Map each unique material to its color index
        
        

    #     material_to_color_idx = {um: i for i, um in enumerate(unique_materials)}
    #     depth_so_far = 0
    #     fig, ax1 = plt.subplots(figsize=(8, 6))
    #     plt.subplots_adjust(right=0.7)
    #     labeled_materials = set()

    #     # Plot the coating layers
    #     for i in range(len(materialLayer)):
    #         # print(materialLayer[i])
    #         material_idx = materialLayer[i]
    #         color_index = material_to_color_idx[material_idx]
            
    #         if materialParams[material_idx]['name'] not in labeled_materials:
    #             label = materialParams[material_idx]['name']
    #             labeled_materials.add(label)
    #         else:
    #             label = None
    #         # print(label)
    #         ax1.bar(
    #             depth_so_far + t_coat[i] / 2,  # bar center position
    #             t_coat[i],                     # bar height (thickness)
    #             color=colors[color_index],
    #             width=t_coat[i],
    #             label=label
    #         )
            
    #         depth_so_far += t_coat[i]

    #         # Only add the vertical line and print details if depBreak is specified
    #         if depBreak is not None and i == depBreak - 1:
    #             ax1.axvline(x=depth_so_far, color='black', linestyle='--')
    #             # Add annotation for deposition boundary
    #             ax1.text(
    #                 x=depth_so_far + 50,  # Adjust x position as needed
    #                 y=700, 
    #                 s="Deposition Boundary", 
    #                 fontsize=10, 
    #                 ha='left', 
    #                 va='top'
    #             )
                
    #             # Print details for the layer before the break
    #             before_layer = materialParams[material_idx]
    #             before_thickness = t_coat[i]
    #             before_position = i + 1  # converting to 1-indexed position
    #             print(f"Vertical line inserted after layer {before_position}:")
    #             print(f"Layer before break - Position: {before_position}, Material: {before_layer['name']}, Thickness: {before_thickness}")
                
    #             # Print details for the layer immediately after the break, if it exists
    #             if i + 1 < len(materialLayer):
    #                 next_material_idx = materialLayer[i + 1]
    #                 after_layer = materialParams[next_material_idx]
    #                 after_thickness = t_coat[i + 1]
    #                 after_position = i + 2  # converting to 1-indexed position
    #                 print(f"Layer after break - Position: {after_position}, Material: {after_layer['name']}, Thickness: {after_thickness}")
    #             else:
    #                 print("No layer exists after the break.")
            
    #     # Extend the left limit a bit more for the annotation & arrow
   
        
    #     # ---------------------------
    #     # ADD FINAL LAYER (SUBSTRATE)
    #     # ---------------------------
    #     # depth_so_far is now sum of all coating thicknesses.
    #     # Plot the substrate bar to the right side (outside the main coating stack).
    #     # Adjust color as you wish; here 'gray' is used.
    #     substrate_color = 'gray'
    #     ax1.bar(
    #         depth_so_far + (t_sub / 2)+5,  # center of the substrate bar
    #         1000,  # height of the bar
    #         color=substrate_color,
    #         width=t_sub,               # width of the bar
    #         label='substrate'
    #     )
    #     # Update x-limits so you can see the substrate bar
    #     # (Otherwise, it might appear outside your plotted range)
    #     ax1.set_xlim([-t_air, depth_so_far + (t_sub*10)])

    #     ax1.set_ylabel('Physical Thickness [nm]')
    #     ax1.set_xlabel('Layer Position')
    #     ax1.set_title('Generated Stack')
    #     ax1.grid(False)
        

        
    #     # Plotting the electric field on a secondary y-axis
    #     ax2 = ax1.twinx()
    #     ax2.grid()
    #     ax2.plot(ds, E_sub, 'blue')
    #     ax2.set_xlabel('depth (nm)')
    #     ax2.set_ylabel('Normalised Electric Field Intensity')
    #     ax2.set_ylim([0, np.max(E_sub)*1.2])
        
        
    #     # ---------------------------
    #     # ADD AN ARROW ON THE LEFT
    #     # ---------------------------
    #     # We'll place an arrow pointing right at x = -t_air, labeled "Light Propagation."
    #     # Adjust the arrow start/end positions and text location to your preference.
    #     # Let's first compute the y-position for the arrow based on the maximum EFI
    #     # 2. Place multiline text at a chosen (x, y). 
    #     left_margin =500
    #     ax1.set_xlim([-t_air - left_margin, depth_so_far + t_sub * 1.1])
    # #    The example below places it near y=700 on the primary axis.
    #     ax1.text(
    #         x=-t_air - 150, 
    #         y=700, 
    #         s="Light Propagation\n----------->",  # The \n creates a new line
    #         fontsize=10, 
    #         ha='left',  # horizontal alignment of the text
    #         va='top'    # vertical alignment relative to y=700
    #     )
 
        
    #     # Explicitly call legend on ax1
    #    # Place legend on the right side, outside the main area:
    #     ax1.legend(
    #         loc='center left',          # position legend center-left
    #         bbox_to_anchor=(1.2, 0.5),  # anchor it just outside the axes
    #         borderaxespad=0.7,          # padding between axes and legend box
    #         fancybox=True,              # optional styling
    #         shadow=True                 # optional styling
    #     )
    #     plt.show()
        
    debug_df = pd.DataFrame({
        'E_sub': E_sub,
        'layer_idx': layer_idx,
        'ds': ds,
        'E': E,
        'poyn': poyn,
        'absor':absor,
})

    

    return E_sub, layer_idx,  ds,E, poyn, total_absorption, debug_df
    

def CalculateAbsorption_tmm(dOpt=None, materialLayer=None, materialParams=None, lambda_=1064, t_air=500, polarisation='p',tphys=None):
    """
    Calculate the absorption at each position within the layers of a thin film stack using tmm.
    """

        
    
    if materialLayer is None:
        raise ValueError("materialLayer cannot be None")
    if materialParams is None:
        raise ValueError("materialParams cannot be None")


    wavelength = lambda_

    # Air layer parameters
    n_air = materialParams[999]['n']


   # set up coating 
    if tphys is not None:
        print('Using physical thicknesses ... ')
        
        n_coat = np.zeros(np.shape(tphys))  # refractive index of each layer
        t_coat = np.zeros(np.shape(tphys))  # physical thickness of each layer in nm
        k_coat = np.zeros(np.shape(tphys))  # attenuation coefficients for each layer
        
        for layer_idx, layer_material in enumerate(materialLayer):
            n_coat[layer_idx]  = materialParams[layer_material]['n']
            t_coat[layer_idx ] = tphys[layer_idx]
            k_coat[layer_idx]  = materialParams[layer_material]['k']
    
    elif dOpt is not None and tphys is None: 
        print('Using optical thicknesses')
        n_coat = np.zeros(np.shape(dOpt))  # refractive index of each layer
        t_coat = np.zeros(np.shape(dOpt))  # physical thickness of each layer in nm
        k_coat = np.zeros(np.shape(dOpt))  # attenuation coefficients for each layer

        for layer_idx, layer_material in enumerate(materialLayer):
            n_coat[layer_idx]  = materialParams[layer_material]['n']
            t_coat[layer_idx ] = optical_to_physical(dOpt[layer_idx], wavelength,materialParams[layer_material]['n'])
            k_coat[layer_idx]  = materialParams[layer_material]['k']
    else:
        raise ValueError("dOpt cannot be empty if tphys is not supplied")
    
    

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

def CalculateTransmission_tmm(dOpt=None, materialLayer=None, materialParams=None, lambda_list=None,lambda_0=None,tphys=None ,polarisation='p',plots=False,plot_range=None,angle=0,verbose=False):
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
    
    # Check for required parameters
    required_params = {
        'materialLayer': materialLayer,
        'materialParams': materialParams,
        'lambda_list': lambda_list,
        'lambda_0': lambda_0
    }

    missing_params = [param for param, value in required_params.items() if value is None]

    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}. ")
    
    
    
    
        
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
    
    
    if tphys is None and dOpt is not None:
        # Compute tphys from dOpt
        tphys = []
        for layer_material in materialLayer:
            n_i = materialParams[layer_material]['n'] 
            thickness = optical_to_physical(dOpt[layer_material], lambda_0, n_i)

            tphys.append(thickness)
        tphys = np.array(tphys)

    elif tphys is None and dOpt is None:
        raise ValueError("Missing required parameters: ...")
    else:
        # tphys was provided directly, just acknowledge
        if verbose:
            print("[EFI-trans] Using physical thickness array from user...")

    # print(f'tphys {tphys}')

        # Get refractive indices and k values
    n_coat = []
    k_coat = []

    if np.size(materialLayer) == 1:
        # Handle single layer case
        n_material = materialParams[materialLayer]['n']
        k_material = materialParams[materialLayer].get('k', 0)

        # If n and k are functions, evaluate them
        if callable(n_material):
            n_i = n_material(lambda_)
        else:
            n_i = n_material  # Constant

        if callable(k_material):
            k_i = k_material(lambda_)
        else:
            k_i = k_material  # Constant

        n_coat = [n_i]
        k_coat = [k_i]
    else:
        # Handle multiple layers
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
    
    
    if isinstance(tphys, np.ndarray) and tphys.size == 1:
        d_list = [np.inf, tphys.item(), np.inf]  # Handle single value array
    else:
        d_list = [np.inf] + tphys.tolist() + [np.inf]  # Convert nm to meters

    # Angle of incidence
    # angle = angle  # Normal incidence in degrees Default = 0 , normal insidence 
    # print('trying')
    # print(n_list)
    # print(d_list)
    # print(np.shape(n_list))
    # print(np.shape(d_list))

   
    for lambda_ in lambda_list:
        wavelength = lambda_

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                coh_tmm_data = tmm.coh_tmm(
                    polarisation, n_list, d_list,
                    angle * np.pi / 180, wavelength
                )
                T = coh_tmm_data['T']
            except Exception as e:
                logger.error(f"Error at wavelength {lambda_} nm: {e}")
                T = np.nan
                raise 
                
            
            # Log any runtime warnings
            for warning in w:
                logger.warning(f"Issue encountered at {lambda_} nm: {warning.message}")

        # Now safe to use T since we always set it in try/except
        wavelengths.append(lambda_)
        transmission.append(T)

    

    # Convert to arrays
    wavelengths  = np.array(wavelengths)
    transmission = np.array(transmission)


    if np.isscalar(lambda_0):
        # If lambda_0 is a single value, find the closest match
        idx = np.abs(wavelengths - lambda_0).argmin()
        transmission_lambda_0 = transmission[idx]
    else:
        # If lambda_0 is an array, find the closest matches for each value
        idx = [np.abs(wavelengths - l).argmin() for l in lambda_0]
        transmission_lambda_0 = transmission[idx]
    
    
    if plots: 
        # Default plot range if not specified
        if plot_range is None:
            plot_range = [380, 1500]

        # Assuming wavelengths and transmission are lists or arrays
        data = {
            'Wavelength (nm)': wavelengths,
            'Transmission (%)': transmission * 100,
            'Reflectivity (%)': 100 - transmission * 100  # Added reflectivity calculation
        }
        
         ##Reshape for plotting 
        transmissions = transmission.T
        if transmissions.ndim > 1:
            valid_mask = ~np.isnan(transmissions).all(axis=1)  # keep rows that aren't all NaN
            transmissions = transmissions[valid_mask, :]
        else:
            valid_mask = ~np.isnan(transmissions)
            transmissions = transmissions[valid_mask]
        
        
        
        fig ,result_data= interactive_transmission_plot(
                        lambda_list,
                        transmissions,
                        plot_range=plot_range,
                        title="Simulated Spectral Responce"
                    )
        fig.show()
        fig.update_xaxes(title_text="Wavelength (nm)", range=plot_range)
        
      
    if isinstance(transmission_lambda_0, np.ndarray) and transmission_lambda_0.size > 1:
        transmission_lambda_0 = pd.DataFrame({
            'Wavelength': lambda_0,
            'Transmission(%)': transmission_lambda_0*100,
            'Reflectivity (%)': (1-transmission_lambda_0)*100,
        })
        
    return wavelengths, transmission, transmission_lambda_0 


def physical_to_optical(physical_thickness, wavelength, refractive_index):
    optical_thickness = physical_thickness * refractive_index / wavelength
    return optical_thickness


def optical_to_physical(optical_thickness, wavelength, refractive_index):
    physical_thickness = optical_thickness*wavelength/ refractive_index
    return physical_thickness


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
    # print('CalculateTransmission_tmm2')
    # print("Wavelengths (first 10):", wavelengths[:10])
    # print("Transmission (first 10):", transmission[:10])

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
def stack_RT_fast(dOpt=None, materialLayer=None, materialParams=None, lambda_list=None,lambda_=1064 ,tphys=None, polarisation='unploarised',theta=0.0,  plots=False,plot_range=None,verbose=False):
    
    
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
    from tmm_fast import coh_tmm as coh_tmm_fast


    #build complex refractive index list including air and sub 
    n_list = build_n_list(materialLayer,materialParams)
    
    
    
    

    if tphys is None and dOpt is not None:
        if verbose: 
            print('[tmm-fast] - Calculating physical thickness... ')
        tphys = []
        for layer_material in materialLayer:
            n_i = materialParams[layer_material]['n']
            physical_thickness = optical_to_physical(dOpt[layer_material], lambda_, n_i)
            # print(f"Layer {layer_material}: Optical Thickness: {dOpt[layer_material]}, Physical Thickness: {physical_thickness}, Refractive Index: {n_i}")
            tphys.append(physical_thickness)
        tphys = np.array(tphys)
    elif tphys is None and dOpt is  None:
        raise ValueError(f" [tmm-fast] Missing required parameters: dOpt and tphys cannot both be empty -please supply an array of thicknesses. ")
    else:
        if verbose:
            print('[tmm-fast] Using Physical Thickness... ')


        



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
   
    lambda__idx=  nearest_index(lambda_array, lambda_)
    
    transmission_lambda_0 = T[lambda__idx]

     # Optionally plot
    if plots is not None:

        if plots == True :
            
            
            plt.figure(figsize=(10, 6))
            plt.plot(lambda_list, transmission, label='Transmission Spectrum (Fast)')
            if plot_range is not None:
                plt.xlim(plot_range)
            plt.xlabel('Wavelength (m)')
            plt.ylabel('Transmission')
            plt.title('Transmission Spectrum [TMM_FAST]')
            plt.grid(True)
            plt.legend()
            plt.show()
    
        elif isinstance(plots, str) and 'plotly' in plots.lower():
            

            valid_mask = ~np.isnan(transmission).all(axis=1)  # keep rows that aren't all NaN
            transmission = transmission[valid_mask, :]
                
            fig,results_data = interactive_transmission_plot(
                            lambda_list,
                            transmission,
                            plot_range=plot_range,
                            title="Simulated Spectral Responce"
                        )
            fig.show()
            fig.update_xaxes(title_text="Wavelength (nm)", range=plot_range)




    return wavelengths,transmission, transmission_lambda_0, {"refl": amplitude, "R": intensity, "T": T, "pol": polarisation}


import numpy as np
import plotly.graph_objs as go

def interactive_transmission_plot(
    wavelengths,
    transmissions,
    plot_range=None,
    title="Transmission vs. Wavelength"
):
    """
    Creates an interactive Plotly chart showing Transmission vs. Wavelength
    with toggle buttons to switch to Reflectivity and to switch between Linear/Log scale.

    This function supports:
        1) A single dataset (1D or 2D):
           - 1D => simple line
           - 2D => mean line with shading from min to max
           For Transmission, uses an orange color scheme; for Reflectivity, uses a blue color scheme.
        2) A dictionary of datasets, keyed by e.g. {'p', 's', 'unpolarised'/'average'}:
           - Each key can be 1D or 2D. 
           - 'p' => blue color, 's' => red color, 'unpolarised'/'average' => black color
           - If 2D, we shade min-to-max for that key. 
           - No orange/blue shading is used in dictionary mode; instead, each key is drawn in its own color.
           
    Toggles:
        - "Transmission"/"Reflectivity"
        - "Linear Scale"/"Log Scale"

    Parameters
    ----------
    wavelengths : array-like
        1D array of wavelengths (e.g. in nm).
    transmissions : array-like or dict
        If array:
            Shape can be (N,) or (N, M).
            The function computes reflectivity = (1 - T)*100 in parallel.
        If dict:
            Keys should be among {'p', 's', 'unpolarised', 'average'} (or any).
            Each value can be 1D or 2D. 
        In dictionary mode, each key is plotted on the same figure with a color mapping.
    plot_range : list of two floats, optional
        Sets the x-axis range, e.g. [380, 1500].
    title : str, optional
        The chart title.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The constructed figure (call fig.show() to display).
    result_data : dict
        A nested dictionary containing the min/mean/max for Transmission and Reflectivity
        for each dataset. If `transmissions` was a single array, you'll get:
            {
              "Transmission": {
                  "min": ...,
                  "mean": ...,
                  "max": ...
              },
              "Reflectivity": {
                  "min": ...,
                  "mean": ...,
                  "max": ...
              }
            }
        If it was a dictionary, you'll get something like:
            {
              "p": {
                "Transmission": {"min":..., "mean":..., "max":...},
                "Reflectivity": {"min":..., "mean":..., "max":...}
              },
              "s": {
                "Transmission": ...,
                "Reflectivity": ...
              },
              ...
            }

    """

        # Convert wavelengths to array for safety
    wavelengths = np.asarray(wavelengths)

    # Helper to compute (min, mean, max) in either 1D or 2D
    def compute_min_mean_max(arr):
        arr = np.asarray(arr)
        
        result_data = {} 
        
        if arr.ndim == 1:
            # Single curve => mean = min = max
            # Multiply by 100 => interpret arr as fraction [0..1]
            m = arr * 100
            result_data['Transmission'] = {} 
            return m, m, m,m,  result_data
        elif arr.ndim == 2:
            # shape (N, M) => we do min/mean/max across axis=1
            # Using nan* versions for safety if partial NaNs exist
            mn = np.nanmin(arr, axis=1) * 100
            mean_ = np.nanmean(arr, axis=1) * 100
            mx = np.nanmax(arr, axis=1) * 100
            st = np.nanstd(arr,axis=1)*100 
            result_data = {} 
            return mn, mean_, mx ,st, result_data 
        else:
            raise ValueError("Data must be 1D or 2D.")


    # We will build up a list of go.Scatter traces
    # plus a corresponding "visible" mask for Transmission or Reflectivity modes.
    fig = go.Figure()

    # We will build up a list of go.Scatter traces
    # We'll keep track of which ones are Transmission vs. Reflectivity
    # so the toggles can hide/show them appropriately.
    fig = go.Figure()
    t_traces = []
    r_traces = []



    # Utility: create 3 traces (min, mean, max) with shading for one label+color.
    # returns [trace_min, trace_mean, trace_max]
    def create_shaded_traces(
        x, y_min, y_mean, y_max, label, color, fillcolor=None, showlegend=True
    ):
        """
        Creates 3 traces for a shaded region from y_min to y_max,
        and a line on top at y_mean.

        Order:
          1) min line
          2) max line (fill='tonexty')
          3) mean line on top

        This ensures the mean line is fully visible above the shading.

        Returns [trace_min, trace_max, trace_mean].
    """
        fill_dict = {
        "blue":  "rgba(0,0,255,0.3)",
        "red":   "rgba(255,0,0,0.3)",
        "black": "rgba(0,0,0,0.3)",
        "gray":  "rgba(128,128,128,0.3)",
        # add more if needed
        }
        
        # If user didn't specify fillcolor, pick from fill_dict if we can
        if fillcolor is None:
            if color in fill_dict:
                fillcolor = fill_dict[color]
            else:
                # fallback
                fillcolor = "rgba(255,165,0,0.3)"  # orange

        # 1) min line (no fill)
        trace_min = go.Scatter(
            x=x,
            y=y_min,
            mode='lines',
            line=dict(width=0, color=color),
            showlegend=False,
            name=f"{label}-min",
        )

        # 2) max line with fill='tonexty'
        trace_max = go.Scatter(
            x=x,
            y=y_max,
            mode='lines',
            fill='tonexty',
            fillcolor=fillcolor,
            line=dict(width=0, color=color),
            showlegend=False,
            name=f"{label}-max",
        )

        # 3) mean line
        trace_mean = go.Scatter(
            x=x,
            y=y_mean,
            mode='lines',
            line=dict(color=color),
            showlegend=showlegend,
            name=f"{label}",
        )

        return [trace_min, trace_max, trace_mean]

    # Check if user passed a dict for polarizations
    if isinstance(transmissions, dict):
        # We'll define color for each key
        color_map = {
            'p': 'blue',
            's': 'red',
            'unpolarised': 'black',
            'average': 'black',
        }
        for key, val in transmissions.items():
            t_min, t_mean, t_max,t_std,result_data = compute_min_mean_max(val)
            # reflectivity = 100 - T
            r_min  = 100.0 - t_min
            r_mean = 100.0 - t_mean
            r_max  = 100.0 - t_max
            r_std = t_std

                    # Store in result_data
            result_data[key] = {
                "Transmission": {"min": t_min, "mean": t_mean, "max": t_max,"std" : t_std},
                "Reflectivity": {"min": r_min, "mean": r_mean, "max": r_max,"std" : r_std},
            }
            color_ = color_map.get(key, 'gray')  # fallback color if unknown
            label_ = f"{key.upper()}-pol"

            # Transmission traces
            T_traces = create_shaded_traces(
                wavelengths, t_min, t_mean, t_max,
                label=label_, color=color_,
                fillcolor=None, 
                showlegend=True
            )
            # Transmission visible by default
            for tr in T_traces:
                tr.visible = True
                fig.add_trace(tr)
            t_traces.extend(T_traces)

            # Reflectivity traces (hide by default)
            R_traces = create_shaded_traces(
                wavelengths, r_min, r_mean, r_max,
                label=label_, color=color_,
                fillcolor=None,
                showlegend=True  # or True if you want them in the legend in Reflectivity mode
            )
            for tr in R_traces:
                tr.visible = False
                fig.add_trace(tr)
            r_traces.extend(R_traces)

    else:
        # Single array scenario (1D or 2D). 
        # We'll do orange shading for T, blue for R
        t_min, t_mean, t_max,t_std ,result_data= compute_min_mean_max(transmissions)
        r_min  = 100.0 - t_min
        r_mean = 100.0 - t_mean
        r_max  = 100.0 - t_max
          
        r_std = t_std

                # Store in result_data
        print('[INFO] No polarisation info supplied - assuming P Polarisation')
        result_data['p'] = {
            "Transmission": {"min": t_min, "mean": t_mean, "max": t_max,"std" : t_std},
            "Reflectivity": {"min": r_min, "mean": r_mean, "max": r_max,"std" : r_std},
        }


        # Transmission (orange)
        T_traces = create_shaded_traces(
            wavelengths, t_min, t_mean, t_max,
            label="Transmission",
            color="orange",
            fillcolor="rgba(255,165,0,0.3)",
            showlegend=True
        )
        for tr in T_traces:
            tr.visible = True  # show by default
            fig.add_trace(tr)
        t_traces.extend(T_traces)

        # Reflectivity (blue)
        R_traces = create_shaded_traces(
            wavelengths, r_min, r_mean, r_max,
            label="Reflectivity",
            color="blue",
            fillcolor="rgba(0,0,255,0.3)",
            showlegend=False
        )
        for tr in R_traces:
            tr.visible = False
            fig.add_trace(tr)
        r_traces.extend(R_traces)

    # Build toggles for Transmission vs. Reflectivity, also Linear vs. Log y-scale
    total_traces = len(fig.data)

    # Identify which trace indexes belong to T vs. R
    t_indexes = []
    r_indexes = []
    for i, trace_obj in enumerate(fig.data):
        if trace_obj in t_traces:
            t_indexes.append(i)
        elif trace_obj in r_traces:
            r_indexes.append(i)

    def make_visibility(show_t):
        # If show_t=True, show Transmission, hide Reflectivity
        # If show_t=False, show Reflectivity, hide Transmission
        vis = [False]*total_traces
        if show_t:
            for idx in t_indexes:
                vis[idx] = True
        else:
            for idx in r_indexes:
                vis[idx] = True
        return vis

    updatemenus = [
        # 1) Transmission / Reflectivity
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    label="Transmission",
                    method="update",
                    args=[
                        {"visible": make_visibility(show_t=True)},
                        {"yaxis": {"title": "Transmission (%)"}}
                    ],
                ),
                dict(
                    label="Reflectivity",
                    method="update",
                    args=[
                        {"visible": make_visibility(show_t=False)},
                        {"yaxis": {"title": "Reflectivity (%)"}}
                    ],
                ),
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.40,  # Adjust button positions
            xanchor="left",
            y=1.25,
            yanchor="top"
        ),
        # 2) Linear vs. Log scale
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    label="Linear Scale",
                    method="relayout",
                    args=[{"yaxis.type": "linear"}]
                ),
                dict(
                    label="Log Scale",
                    method="relayout",
                    args=[{"yaxis.type": "log"}]
                ),
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.70,
            xanchor="left",
            y=1.25,
            yanchor="top"
        )
    ]

    # Configure figure layout
    fig.update_layout(
        title=title,
        updatemenus=updatemenus,
        margin=dict(t=100)  # Leaves room at the top for toggles
    )

    
    fig.update_yaxes(tickformat=".2e", hoverformat=".2g",title_text="Transmission (%)", range=[0, 100])  # Adjust if needed
    fig.update_xaxes(tickformat=".2e", hoverformat=".2g",title_text="Wavelength (nm)", range=plot_range)
    return fig, result_data