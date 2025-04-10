import json
import numpy as np

import os
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import re

from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Patch

def materialLibrary(action, input_data=None, new_material=None):
    """# Example usage
    # materialLibrary('read', 'SiO2')  # Read a material by name
    # materialLibrary('write', new_material={'name': 'NewMaterial', 'n': 2.0})  # Add a new material
    # materialLibrary('--listMaterials')  # List all materials
    """
    
    # Path to your JSON file
    json_file_path = '/Users/simon/Dropbox/Python/Optics/CoatingDevelopment/coatingstack/TiGe_Optimisation/materialParams.json'
    
    # Read the existing materials from the file
    try:
        with open(json_file_path, 'r') as file:
            materials = json.load(file)
    except FileNotFoundError:
        materials = {}
    
    if action == 'read':
        # Return material by index or name
        if input_data is not None:
            for key, material in materials.items():
                if str(key) == str(input_data) or material['name'] == input_data:
                    return material
            print(f"Material {input_data} not found.")
            return None

    elif action == 'write':
        # Write new material, ensuring no duplicates by name
        if new_material and 'name' in new_material:
            for material in materials.values():
                if material['name'] == new_material['name']:
                    raise ValueError("Material with the same name already exists. Please choose a different name.")
            # Find the next index
            next_index = str(max([int(k) for k in materials.keys()] + [0]) + 1)
            materials[next_index] = new_material
            
            # Write the updated materials back to the file
            with open(json_file_path, 'w') as file:
                json.dump(materials, file, indent=4)
            print(f"Material {new_material['name']} added successfully.")
        else:
            print("Invalid material data.")

    elif action == '--listMaterials':
        # List all material names
        names = [material['name'] for material in materials.values()]
        print("Materials in library:", ", ".join(names))

def thin_film_stack(n_input, materialLayer, lambda_, base_path, plots=True, **kwargs):
    
    """
    Generates and plots a thin film coating stack.

    :param n_input: Array of refractive indices for each material.
    :param materialLayer: Array specifying the material for each layer.
    :param lambda_: Wavelength.
    :return: n_layers, material_kind, d_physical_layers
    """
    
    
    if len(n_input) < max(materialLayer):
        raise ValueError('The number of refractive indices provided is less than the required materials in materialLayer.')

    # Calculate individual physical thickness for each material
    d_physical = lambda_ / (4 * np.array(n_input))



    # Arrays for each layer
    n_layers = np.array(n_input)[np.array(materialLayer) - 1]
    material_kind = materialLayer
    d_physical_layers = np.array(d_physical)[np.array(materialLayer) - 1]

    # print(f"n_layers {n_layers}")
    # print(f"material_kind {material_kind}")
    # print(f"d_physical_layers {d_physical_layers}")

    # Plotting the thin film stack
    unique_materials = list(set(materialLayer))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_materials)+2))  # generate distinct colors for materials


    # Check if the file exists to write headers
    out_dir = os.path.join(base_path,'coating_metadata.txt')
    file_exists = os.path.exists('out_dir.txt')

    # Write metadata to text file
    with open(out_dir, 'a') as file:
        if not file_exists:
            file.write("Lambda (m)\tTotal Layers\t")
            for i in range(1, 7):
                file.write(f"Material_{i}\tNo. Layers_{i}\tPhysical Thickness_{i} (m)\tRefractive Index_{i}\t")
            file.write("\n")
        
        file.write(f"{lambda_:.2e}\t{len(materialLayer)}\t")
        for i in range(1, 7):
            if i in materialLayer:
                file.write(f"Material {i}\t")
                file.write(f"{len([x for x in materialLayer if x == i])}\t")
                file.write(f"{sum(d_physical_layers[materialLayer == i]):.2e}\t")
                file.write(f"{n_input[i-1]:.2f}\t")  # Directly access the refractive index from n_input
            else:
                file.write("NaN\tNaN\tNaN\tNaN\t")
        file.write("\n")
    
    
    if plots:
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.grid(True)
        depth_so_far = 0  # To keep track of where to plot the next bar
        for i in range(len(materialLayer)):
            material_idx = materialLayer[i]
            color_idx = unique_materials.index(material_idx)
            plt.bar(depth_so_far + d_physical_layers[i] / 2, d_physical_layers[i], color=colors[color_idx],
                    width=d_physical_layers[i])
            depth_so_far += d_physical_layers[i]

        plt.xlim([0, sum(d_physical_layers) * 1.01])
        plt.ylabel('Physical Thickness [nm]')
        plt.xlabel('Layer Position')
        plt.title('Generated Stack')
        legend_str = ['n = ' + str(n) for n in n_input]
        plt.grid(False)
        plt.legend(legend_str)
    
    # Additional code for plotting the normalized electric field intensity squared can be added here

    plt.show()

   # Printing coating properties
    print("\nCoating Properties:\n")

    # Predefined labels and their corresponding values
    properties = {
        "Laser Wavelength": f"{lambda_ * 1E9:.2f} nm",
        "Number of Materials": f"{len(unique_materials):d}",
        "Total Physical Thickness": f"{sum(d_physical_layers):.2e} m"
    }

    # Merge the predefined properties with kwargs
    properties.update(kwargs)
    # Find the longest key for alignment
    longest_key = max(len(key) for key in properties.keys())
    # Print each property, aligned
    for key, value in properties.items():
        print(f"{key}:".ljust(longest_key + 8) + f"{value}")
    for i in unique_materials:
        print(f"\n--------- Material {i} -------------\n")
        materialLayer_array = np.array(materialLayer)
        matching_indices = materialLayer_array == i
        n_layers_matching = n_layers[matching_indices]
        d_physical_layers_matching = d_physical_layers[matching_indices]
    
        if len(n_layers_matching) > 0:
            print(f"No. Layers:\t\t\t{len(n_layers_matching)}")
            print(f"Total Physical Thickness:\t{sum(d_physical_layers_matching):.2e} m")
            print(f"Refractive Index:\t\t{np.unique(n_layers_matching)[0]:.2f}")
        else:
            print(f"No layers of material {i} found.")

    return n_layers, material_kind, d_physical_layers


def optical_to_physical(optical_thickness, vacuum_wavelength, refractive_index):
    physical_thickness = optical_thickness*vacuum_wavelength/ refractive_index
    return physical_thickness


def physical_to_optical(physical_thickness, wavelength, refractive_index):
    return physical_thickness * refractive_index / wavelength

def create_bar_plot(dOpt, materialLayer, title):
    
    unique_values = np.unique(materialLayer)
    color_map = plt.cm.get_cmap('hsv', len(unique_values) + 1)  # +1 to avoid reuse of the last color
    colors = [color_map(val) for val in materialLayer]

    # plt.ion()  # Turn on interactive mode
    # clear_output(wait=True)  # Clear the output to ensure only the latest plot is displayed 
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(dOpt)), dOpt, color=colors)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    # plt.draw()  # Draw the current plot
    # plt.pause(0.5)  # Pause to allow the plot to be updated
    # plt.clf()  # Clear the current figure to allow for the next plot
    
## COMPOSITE MATERIALS
    

def calculate_density(num_layers, layer_density, method='thickness', refractive_indx=None, wavelength=None, physical_thicknesses=None):
    """
    Calculate the average density of a material based on its refractive index, number of layers, layer density, and wavelength.
    
    @param num_layers - The number of layers in the material
    @param layer_density - The density of each layer
    @param method - Method to determine layer thickness ('thickness' or 'qw'). Default is 'thickness'.
    @param refractive_indx - The refractive index of the material (required if method is 'qw')
    @param wavelength - The wavelength of the light (required if method is 'qw')
    @param physical_thicknesses - The physical thicknesses of each layer (required if method is 'thickness')
    
    @return density - The calculated density
    @return denominator - The denominator value used in the calculation
    @return thicknesses_by_layer - The thicknesses of each layer multiplied by the number of layers
    
    Author S.Tait 2024 
    """
    if method.lower() == 'qw':
        if refractive_indx is None or wavelength is None:
            raise ValueError("Refractive index and wavelength must be provided when method is 'qw'")
        layer_thicknesses = [wavelength / (4 * r_indx) for r_indx in refractive_indx]
        if len(layer_thicknesses) != len(layer_density):
            raise ValueError("Refractive index and layer density must have the same dimensions")
    elif method.lower() == 'thickness':
        if physical_thicknesses is None:
            raise ValueError("Physical thicknesses must be provided when method is 'thickness'")
        if len(physical_thicknesses) != len(layer_density):
            raise ValueError("Physical thicknesses and layer density must have the same dimensions")
        layer_thicknesses = physical_thicknesses
    else:
        raise ValueError("Invalid method. Use 'qw' for quarter-wave or 'thickness' for physical thicknesses")

    densities = []
    thicknesses_by_layer = []

    for i in range(len(layer_thicknesses)):
        density_i = (layer_thicknesses[i] * num_layers[i] * layer_density[i])
        densities.append(density_i)
        thicknesses_by_layer.append(layer_thicknesses[i] * num_layers[i])

    numerator = sum(densities)
    denominator = sum(thicknesses_by_layer)

    density = numerator / denominator

    return density, denominator, thicknesses_by_layer
    
    


def comploss(dataframes, ymat, tmat, min_samples=2,normalization='robust',plots=True,debugging=True):
    """
    Calculates the expected composite loss of multiple component thin film materials.
    
    Parameters:
    - dataframes: list of pandas DataFrames, each containing 'Frequency' and 'Loss' columns for different materials.
    - ymat: list of floats, Young's moduli of the materials.
    - tmat: list of floats, thicknesses of the materials.
    - eps: float, the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: int, the number of samples in a neighborhood for a point to be considered as a core point.
    - normalisation: str, method of frequency normalisation used for preprocessing before DBSCAN can be 'robust','zscore','minmax'
    
    Returns:
    - DataFrame with composite loss for matched frequencies.
    """
    
    # Calculate composite Young's Modulus
    ycomp_numerator = sum(ym * tm for ym, tm in zip(ymat, tmat))
    ycomp_denominator = sum(tmat)

    ycomp = ycomp_numerator / ycomp_denominator



    # Combine and label data from all dataframes
    all_data = pd.DataFrame(columns=['Frequency', 'Loss', 'Material'])

    for i, df in enumerate(dataframes):
        temp_df = df.copy()
        temp_df['Material'] = i
        temp_df = temp_df.sort_values(by=['Frequency'])
        all_data = pd.concat([all_data, temp_df], ignore_index=True)
        
       
    # Apply custom  DBSCAN clustering based on frequency
    
    # filtered_alldata = cluster_frequencies(all_data, normalisation=normalization, plots=plots, debugging=debugging)
    filtered_alldata = all_data
   
    

    # clustering = DBSCAN(eps=eps, min_samples=2).fit(all_data[['Frequency']])
    # all_data['Group'] = clustering.labels_
    
    # Filter out data not belonging to any cluster
    # filtered_alldata = all_data[all_data['Group'] >= 0]

    comp_loss = []
    comp_freq = [] 
    
    
    if debugging:
        
        print()
        print(f"Debugging Info:")
        print(f"Material properties")
        
        print(f"ymat Lenght: {len(ymat):<15}")
        for index, value in enumerate(ymat):
            print(f"Y{index}              {value:<15}")  # Left align within 15 characters

        # Correct use of format specifier for print statement
        print(f"tmat Length: {len(tmat):<15}")

        # Correct loop and print statement for tmat, assuming it should be tmat not ymat here
        for index, value in enumerate(tmat):
            print(f"T{index}              {value:<15}")  # Left align within 15 characters

        # Correct print statement for ycomp
        print(f"Y_Comp          {ycomp:<15}")
        print()

        # Correct print statement for stack thickness
        print(f"Stack Thickness {sum(tmat):<15}")
        print()

    
    for i, group in enumerate(filtered_alldata['Cluster'].unique()):
        group_data = filtered_alldata[filtered_alldata['Cluster'] == group]
      

        if len(group_data) == len(filtered_alldata['Material'].unique()):
        
            if np.shape(group_data)[0] <3:
                calculated = (ymat[0] * tmat[0] * (np.array(group_data['Loss'])[0]) +(ymat[1] * tmat[1] * (np.array(group_data['Loss'])[1])))/(ycomp * (sum(tmat)))
            if np.shape(group_data)[0] ==3 :            
                calculated = (ymat[0] * tmat[0] * (np.array(group_data['Loss'])[0]) +(ymat[1] * tmat[1] * (np.array(group_data['Loss'])[1])) +(ymat[2] * tmat[2] * (np.array(group_data['Loss'])[2]))) /(ycomp * (sum(tmat)))
                
            comp_loss.append(calculated)
            comp_freq.append(np.array(group_data['Frequency'])[1])
            
        
    out = {
        'Frequency': comp_freq,
        'Loss': comp_loss,
    }

    out_df = pd.DataFrame(out)
    
    if debugging: 
        print(out_df)
    
    
    if plots: 
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, df in enumerate(dataframes):
            plt.plot(df['Frequency'], df['Loss'], 'o', label=f'Material {i+1}')
        
        plt.plot(out_df['Frequency'], out_df['Loss'], 'p', label='Composite Loss')
        plt.grid(True)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        return out_df, fig, ax     
    else: 
        return out_df


def readMaterials_from_json(serial,file_path,temp,dur,verbose=False): 
    """
        - serial: the serial number as a string.
        - temp  : temperature of heat treatment requested - As Depostied = 30 
        - dur   : duration of the heat treatment requested in hourss -  As Deposited = 0 
        - OutData: defaultdict to update with the new results.


    """
    
    # Attempt to load existing data if the file exists and is not empty
    data_exists = False
    
    temp_dur_key = f"{temp}_{dur}"
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
                
            # Check if the file was empty or if the specific key exists
            temp_dur_key_str = str(temp_dur_key)
            if existing_data and temp_dur_key_str in existing_data:
                data_exists = True
                
                if verbose:
                    print(f"Key {temp_dur_key_str} found in existing data.")
               
                

                    print(f"Loaded data for {serial} {temp_dur_key} from existing file.")
            else:
                
                print(f"Key {temp_dur_key_str} not found in existing data.")
                print(f"Analysing {temp_dur_key_str}")
                print()

        except json.JSONDecodeError:
            print(f"Existing file for {serial} is empty or corrupted. Proceeding with analysis.")
        return existing_data[temp_dur_key]
    
import numpy as np


## Estimating Material Properties 

## Density from mass 
import math

def estimate_density(mass_coated, mass_uncoated, edge, t_coat, d_sub=1.00):
    """
    Estimate the density of a coating material given the masses of the uncoated and coated substrates.

    Parameters:
    - mass_coated (float): Mass of the coated substrate in grams.
    - mass_uncoated (float): Mass of the uncoated substrate in grams.
    - edge (float): The edge value in millimeters which is the outer diameter of the coating minus the diameter of the substrate.
    - coating_thickness_nm (float): Thickness of the coating in nanometers.
    - diameter_substrate (float, optional): Diameter of the substrate in millimeters. Default is 1.00 mm.

    Returns:
    - float: Estimated density of the coating material in kilograms per cubic meter (kg/m³).

    Example:
    >>> mass_coated = 12.0  # in grams
    >>> mass_uncoated = 10.0  # in grams
    >>> edge = 2.0  # in mm
    >>> coating_thickness_nm = 100  # in nm
    >>> diameter_substrate = 50.0  # in mm
    >>> density = estimate_density(mass_coated, mass_uncoated, edge, coating_thickness_nm, diameter_substrate)
    >>> print(f"Estimated density of the coating material: {density:.4f} kg/m³")
    """
    
    # Convert masses from grams to kilograms
    mass_coated_kg = mass_coated * 0.001
    mass_uncoated_kg = mass_uncoated * 0.001
    
    # Calculate the mass of the coating in kilograms
    mass_coating_kg = mass_coated_kg - mass_uncoated_kg

    # Convert dimensions from millimeters to meters
    d_sub_m = d_sub * 0.001
    
    
    edge_m = edge * 0.001
    
    # Convert coating thickness from nanometers to meters
    t_coat_m = t_coat * 1e-9

    # Calculate the outer and inner radii in meters
    outer_radius_m = (d_sub_m - edge_m) / 2
    
    inner_radius_m = outer_radius_m - t_coat_m

    # Calculate the volume of the coating in cubic meters
    volume_coating_m3 = math.pi * (outer_radius_m**2 - inner_radius_m**2) * d_sub_m
    

    # Estimate the density of the coating material in kg/m³
    density_coating_kg_per_m3 = mass_coating_kg / volume_coating_m3

    return density_coating_kg_per_m3















## Composite Refractive Index 
def comp_refractiveIndex(C1, C2, rho1, rho2, n1, n2):
    """
    Calculate the effective refractive index of a composite material based on 
    the concentration, density, and refractive index of its constituent materials.

    Parameters:
    - C1: float, concentration of the first material by thickness
    - C2: float, concentration of the second material by thickness
    - rho1: float, density of the first material (kg/m^3)
    - rho2: float, density of the second material (kg/m^3)
    - n1: float, refractive index of the first material
    - n2: float, refractive index of the second material

    Returns:
    - n: float, effective refractive index of the composite material
    """
    # Calculate the intermediate values a1 and a2
    a1 = (n1**2 + 2)**(-1)
    a2 = (n2**2 + 2)**(-1)

    # Calculate the effective refractive index n
    numerator = (a1 * n1**2 * C1 / rho1) + (a2 * n2**2 * C2 / rho2)
    denominator = (a1 * C1 / rho1) + (a2 * C2 / rho2)
    n = np.sqrt(numerator / denominator)

    return n



def normalize_data(data, column_name, method='robust'):
    """Applies normalization to the specified column of the dataframe based on the method."""
    if method == 'minmax':
        min_val = data[column_name].min()
        max_val = data[column_name].max()
        data[column_name] = (data[column_name] - min_val) / (max_val - min_val)
    elif method == 'robust':
        scaler = RobustScaler()
        data[column_name] = scaler.fit_transform(data[column_name].values.reshape(-1, 1))
    elif method == 'zscore':
        normalized = zscore(data[column_name])
        data[column_name] = normalized + abs(min(normalized))
    return data

def evaluate_dbscan(data, column='Frequency', eps_range=(0.01, 1.01, 0.01), min_samples=2):
    """Performs DBSCAN over a range of eps values and evaluates clustering quality."""
    results = []
    for eps in np.arange(*eps_range):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data[[column]])
        labels = db.labels_
        num_noise = np.sum(labels == -1)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_data = data.copy()
        cluster_data['Cluster'] = labels
        mixed_clusters = sum(cluster_data.groupby('Cluster')['Material'].nunique() > 1)
        results.append({'eps': eps, 'num_noise': num_noise, 'num_clusters': num_clusters, 'mixed_clusters': mixed_clusters})
        
        
    return pd.DataFrame(results)

def reassign_clusters(cluster_data, threshold=4):
    """Reassigns clusters for those having exactly 'threshold' members."""
    counts = cluster_data['Cluster'].value_counts()
    target_clusters = counts[counts == threshold].index
    for cluster in target_clusters:
        members = cluster_data[cluster_data['Cluster'] == cluster]
        members_sorted = members.sort_values(by='Frequency')
        indices = members_sorted.index.tolist()
        # Assuming new labels start from max label + 1 and increment every two members
        for i, idx in enumerate(indices):
            cluster_data.loc[idx, 'Cluster'] = cluster_data['Cluster'].max() + 1 + i // threshold/2
    return cluster_data

def plot_clusters(data):
    """Plots clusters with different markers for small and large clusters."""
    markers = np.tile(['o', '^', 'p'], int(np.round(128 / 2) + 1))
    counter = 0
    fig, ax = plt.subplots()
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        if cluster >= 0:
            marker = markers[counter]
            counter += 1
        else:
            marker = 'x'
        plt.scatter(cluster_data['Frequency'], 1 / cluster_data['Loss'], marker=marker)
    plt.grid(True)
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mechanical Loss')
    plt.title('Number of Clusters identified: ' + str(data['Cluster'].nunique()))
    plt.show()

def cluster_frequencies(all_data, normalisation='robust', plots=False, debugging=False):
    """Main function to cluster frequencies based on material property."""
    if 'Material' not in all_data or 'Frequency' not in all_data:
        raise ValueError("Required columns are missing from the DataFrame.")

    all_data_normalized = pd.DataFrame()
    # Normalize data per material type
    for material in all_data['Material'].unique():
        subset = all_data[all_data['Material'] == material].copy()
        
        subset = normalize_data(subset, 'Frequency', normalisation)
        all_data_normalized = pd.concat([all_data_normalized, subset], ignore_index=True)

    
    
    
    results_df = evaluate_dbscan(all_data_normalized)
    results_df['score'] = results_df['num_clusters'] - results_df['mixed_clusters']
    if len(all_data['Material'].unique())>2:
        best_result = results_df.loc[results_df['score'].idxmin()]
       
    else:
        best_result = results_df.loc[results_df['score'].idxmax()]
        
    # Apply the best clustering
    clustering = DBSCAN(eps=best_result['eps'], min_samples=2)
    all_data_normalized['Cluster'] = clustering.fit_predict(all_data_normalized[['Frequency']])
    
    

    all_data_normalized = reassign_clusters(all_data_normalized,threshold=len(all_data['Material'].unique()) * 2 )
    
    for material in all_data['Material'].unique():
            
        all_data_normalized.loc[all_data_normalized['Material'] == material, 'Frequency'] = all_data.loc[all_data['Material'] == material, 'Frequency'] 

    # Plot results if requested
    if plots:
        plot_clusters(all_data_normalized)

    if debugging:
        print()
        print('Clustering Info:')
        print("Best Result ")
        print(best_result)
        print()
        print(all_data_normalized)

    return all_data_normalized

import re 
def get_measurements(serial):
    
    def get_file_path(serial):
        return f'/Users/simon/Desktop/CRIME_MACHINE/results/{serial}/{serial}_SampleInformation.json'

    def load_data(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    
    sample_info = {}
    sample_info[serial] = load_data(get_file_path(serial))
            
    processed_data = {}
    latest_measurements = {}
    heat_treated_keys = []

    # Dynamically find and process keys containing "State"
    for key, value in sample_info[serial][serial].items():
        if isinstance(value, dict):
            state_keys = [k for k in value.keys() if 'State' in k]
            if state_keys and 'UNCOATED SUSPENSIONS' not in key:
                heat_treated_keys.append(key)

    # Ensure the keys are unique
    unique_heat_treated_keys = np.unique(heat_treated_keys)
    
    
    for key in unique_heat_treated_keys: 
        if "uncoated" in key.lower():
            uncoated_measurements_key = key 
    uncoated_measurements = sample_info[serial][serial].get(uncoated_measurements_key, {})
    

    heat_treatments = {key: sample_info[serial][serial][key] for key in unique_heat_treated_keys if 'uncoated' not in key.lower()}    

    for treatment_key, treatment_info in heat_treatments.items():
        sample_state = [value for key, value in treatment_info.items() if 'State' in key]

        # Extract and sort date keys
        date_keys = sorted([key for key in treatment_info if 'Date' in key])
        dates = [treatment_info[key] for key in date_keys]

        try:
            dates = [date.strip() for date in dates if re.match(r'^\d{4}_\d{2}_\d{2}$', date.strip())]
        except:
            dates = [str(date).strip() for date in dates if isinstance(date, (str, int, float)) and re.match(r'^\d{4}_\d{2}_\d{2}$', str(date).strip())]
        
        processed_data[treatment_key] = {
            'sample_state': sample_state,
            'dates': dates
        }

    # Add in uncoated measurements
    uncoated_dates = [
        str(value).strip()
        for key, value in uncoated_measurements.items()
        if 'Date' in key and re.match(r'^\d{4}_\d{2}_\d{2}$', str(value).strip())
    ]

    processed_data['Uncoated'] = {
        'sample_state': 'UNCOATED',
        'dates': uncoated_dates
    }

    latest_measurements = {}
    latest_date = None

    for key, value in processed_data.items():
        if key != uncoated_measurements_key:
            sample_state = value['sample_state']
            dates = value['dates']

            if sample_state and dates:
                most_recent_date = max(dates)
                
                if latest_date is None or most_recent_date > latest_date:
                    latest_date = most_recent_date
                    latest_measurements[key] = {
                        'sample_state': sample_state,
                        'dates': dates
                    }

    return processed_data, latest_measurements, sample_info

def parse_treatment_states(treatment_states):
    temp_list = []
    dur_list = []

    temp_pattern = re.compile(r'(\d{2,3})C')
    dur_pattern = re.compile(r'(\d{2,4})hrs')

    for treatment, info in treatment_states.items():
        sample_state = info['sample_state']

        if 'Uncoated' in sample_state:
            temp_list.append(0)
            dur_list.append(0)
        elif 'As Deposited' in sample_state:
            temp_list.append(30)
            dur_list.append(0)
        else:
            temp_match = temp_pattern.search(sample_state[0] if sample_state else '')
            dur_match = dur_pattern.search(sample_state[0] if sample_state else '')
            temp = int(temp_match.group(1)) if temp_match else 0
            dur = int(dur_match.group(1)) if dur_match else 0
            temp_list.append(temp)
            dur_list.append(dur)

    return temp_list, dur_list





#Saving excel Files 
def save_data_as_excel(frequency, loss_angle, loss_angle_err, serial, temp, dur, output_dir):
    """Saves the extracted data into an Excel file formatted according to temperature and duration."""
    # Create a DataFrame
    data = {
        'frequency': frequency,
        'phi': loss_angle,
        'phi_error': loss_angle_err
    }
    df = pd.DataFrame(data)
    
    # Determine the file name
    if temp == 30:
        filename = f"{serial}_CoatingLoss_AsDeposited.xlsx"
    else:
        filename = f"{serial}_CoatingLoss_{temp}C_{dur}hrs.xlsx"
    
    # Path to save the file
    file_path = output_dir + '/' + filename
    
    # Write DataFrame to Excel
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)




## Printing and formatting stuff 
def print_dict_human_readable(d, indent=0):
    """
    Prints the contents of a dictionary in a human-readable way.
    
    :param d: Dictionary to print.
    :param indent: Current indentation level (used for nested dictionaries).
    """
    for key, value in d.items():
        print('    ' * indent + str(key) + ': ', end='')
        if isinstance(value, dict):
            print()
            print_dict_human_readable(value, indent + 1)
        else:
            print(value)

def convert_tuple_keys(d):
    """
    Converts tuple keys to strings and tuple values to lists in a dictionary for JSON serialization.
    @param d - A dictionary possibly containing tuple keys and values.
    @return A JSON-serializable dictionary.
    """
    """
    Recursively converts tuple keys to strings and tuple values to lists in a dictionary, 
    making it JSON-serializable. Nested dictionaries are also processed. 
    Tuple keys are converted to strings by joining their string representations with underscores. 
    This approach ensures the dictionary can be serialized using JSON without losing the 
    structure of tuple keys or values.
    
    Parameters:
    - d (dict): A dictionary possibly containing tuple keys and values.
    
    Returns:
    - dict: A dictionary with all tuple keys and values converted to strings and lists, respectively.
    Author S.Tait 2024 
    """
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary.")
    
    serializable_dict = {}
    
    
    for key, value in d.items():
        # Convert tuple keys to string by joining with underscore to prevent key ambiguity.
        new_key = "_".join(str(item) for item in key) if isinstance(key, tuple) else key
        
        # Recursively convert nested dictionaries or convert tuple values to lists.
        if isinstance(value, dict):
            new_value = convert_tuple_keys(value)
        elif isinstance(value, tuple):
            new_value = list(value)
        else:
            new_value = value
        
        serializable_dict[new_key] = new_value
    
    return serializable_dict



def save_MaterialProperties(s, temp, dur, Y, nu, e, Yb, nub, eb, samples, out_dir,debugging=False):
    
    key = f"{temp}_{dur}"
    # Prepare the elastic properties dictionary
    elasticProps = {
        'temperature': temp,
        'duration': dur,
        'Y': Y,
        'nu': nu,
        'e': e,
        'Yb': [Yb[1], 0.5 * (Yb[2] - Yb[0])],
        'nub': [nub[1], 0.5 * (nub[2] - nub[0])],
        'eb': [eb[1], 0.5 * (eb[2] - eb[0])]
    }

    json_path = os.path.join(out_dir, s, f'{s}_MaterialProperties.json')

    # Initialize or load existing data
    try:
        with open(json_path, 'r') as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    # Convert and merge new data
    converted_data = convert_tuple_keys(elasticProps)
    existing_data[key] = converted_data

    # Serialize the updated dictionary to JSON
    json_str = json.dumps(existing_data, indent=4)

    # Optional debugging printout
    if debugging:
        print('Info Written to Materials JSON\n')
        print(json_str)

    # Write the updated JSON string to the file
    with open(json_path, 'w') as file:
        file.write(json_str)


### Saving file formats 

# Parquet 

def convert_nested_dict_to_df(nested_dict):
    rows = []
    for key, value in nested_dict.items():
        for subkey, subvalue in value.items():
            if isinstance(subvalue, dict):
                for subsubkey, subsubvalue in subvalue.items():
                    if subsubvalue is None:
                        subsubvalue = [None]
                    if isinstance(subsubvalue, np.ndarray) or isinstance(subsubvalue, list):
                        for item in subsubvalue:
                            rows.append([key, subkey, subsubkey, item])
                    else:
                        rows.append([key, subkey, subsubkey, subsubvalue])
            elif isinstance(subvalue, np.ndarray):
                for subsubvalue in subvalue:
                    rows.append([key, subkey, None, subsubvalue])
            else:
                subsubvalue = subvalue if subvalue is not None else [None]
                if isinstance(subsubvalue, np.ndarray) or isinstance(subsubvalue, list):
                    for item in subsubvalue:
                        rows.append([key, subkey, None, item])
                else:
                    rows.append([key, subkey, None, subsubvalue])
    df = pd.DataFrame(rows, columns=["Type", "Category", "Serial", "Values"])
    return df



## Filtering 


def linear_frequency_dependence(dataset, plots=False, filter=False, sigma=1):
    """
    Fits a linear model to the provided dataset and optionally plots the results.
    
    Args:
    - dataset (pd.DataFrame): The input dataset with 'Frequency', 'Loss', and 'Loss_err' columns.
    - plots (bool): Whether to plot the results. Defaults to False.
    - filter (bool): Whether to filter based on confidence intervals. Defaults to False.
    - sigma (int): The number of standard deviations for the confidence interval. Defaults to 2.
    
    Returns:
    - average_slope (float): The average slope of the fitted linear model.
    - filtered_dataset (pd.DataFrame): The filtered dataset.
    - fig (plt.Figure): The figure object (only if plots=True).
    - ax (plt.Axes): The axes object (only if plots=True).
    """
    # Fit linear model to dataset
    model = LinearRegression()
    
    
    
    
    model.fit(dataset[['Frequency']], dataset['Loss Angle'])
    predictions = model.predict(dataset[['Frequency']])
    
    # Calculate average slope
    average_slope = model.coef_[0]

    # Calculate the residual standard error and the confidence intervals
    mse = mean_squared_error(dataset['Loss Angle'], predictions)
    standard_error = np.sqrt(mse) / np.sqrt(len(dataset['Frequency']))
    confidence_upper = predictions + sigma * standard_error
    confidence_lower = predictions - sigma * standard_error
    

    # Filter the dataset based on the conditions
    if filter:
        filtered_dataset = dataset[
            (dataset["Loss Angle"] - dataset['Loss Angle Error'] <= confidence_upper) &
            (dataset["Loss Angle"] + dataset['Loss Angle Error'] >= confidence_lower)
        ]
        removed_values = dataset[~((dataset["Loss Angle"] - dataset['Loss Angle Error'] <= confidence_upper) &
                                (dataset["Loss Angle"] + dataset['Loss Angle Error'] >= confidence_lower))]
        label_add = "Filtered "
    else:
        filtered_dataset = dataset.copy()
        removed_values = pd.DataFrame(columns=dataset.columns)  # No values removed
        label_add = ""

    # Convert the filtered dataset to an array
    filtered_array = filtered_dataset.values

    fig, ax = None, None
    if plots:
        # Plotting to check linearity and filtering effect
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(filtered_dataset['Frequency'], filtered_dataset['Loss Angle'], yerr=filtered_dataset['Loss Angle Error'], fmt='o', color='blue', label=label_add + 'Loss', alpha=0.5)
        ax.errorbar(removed_values['Frequency'], removed_values['Loss Angle'], yerr=removed_values['Loss Angle Error'], fmt='x', color='red', label='Removed', alpha=1)


        ax.plot(dataset['Frequency'], predictions, color='darkblue', label='Fit: Loss')
        ax.fill_between(dataset['Frequency'], confidence_lower, confidence_upper, color='darkblue', alpha=0.2)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Loss')
        ax.set_title('Linear Fit of Loss vs Frequency ' + label_add)
        ax.legend()
        ax.grid(True)
        plt.show()
    
    return average_slope, filtered_dataset, fig, ax











##### 
# plotting stuff

## Colourblindness 
# Function to determine if a color is too bright and adjust it
def adjust_color_for_visibility(colormap='viridis', num_colors=10, brightness_threshold=0.5, darken_factor=0.2):
    """
    Adjusts the colors in a given colormap for better visibility, particularly for colorblindness.
    
    Parameters:
    - colormap (str): The name of the colormap to use. Default is 'viridis'.
    - num_colors (int): The number of colors to generate from the colormap. Default is 10.
    - brightness_threshold (float): The threshold for brightness to determine if a color is too bright. Default is 0.5.
    - darken_factor (float): The factor by which to darken the color if it is too bright. Default is 0.2.
    
    Returns:
    - base_colors (list): List of base colors from the colormap.
    - edge_colors (list): List of adjusted edge colors for better visibility.
    """
    cmap = plt.get_cmap(colormap)
    base_colors = cmap(np.linspace(0, 1, num_colors))
    edge_colors = []
    
    step = max(1, len(base_colors) //(num_colors))
    sampled_colors = [base_colors[i] for i in range(0, len(base_colors), step)]

    for color in sampled_colors:
        rgb = to_rgb(color)
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if brightness > brightness_threshold:
            darker_rgb = tuple(c * darken_factor for c in rgb)  # Darken the color
            outline_color = to_hex(darker_rgb)
        else:
            outline_color = to_hex(rgb)
        edge_colors.append(outline_color)
    
    base_colors = [to_hex(c) for c in base_colors]
    return base_colors, edge_colors


# Custom scientific formatter
def sci_format(x, pos):
    return f'{x:.1e}'


# Function to filter out overlapping ticks
def filter_overlapping_ticks(existing_ticks, custom_ticks, axis_range=[1E-6, 5E-4],threshold=0.05):
    # Calculate the threshold as a small fraction of the axis range
    
    print(axis_range)
    threshold = threshold * (axis_range[1] - axis_range[0])
    filtered_ticks = []

    # Combine and sort all ticks
    combined_ticks = sorted(set(existing_ticks) | set(custom_ticks))

    # Filter ticks by ensuring they're spaced out by at least the threshold
    for tick in combined_ticks:
        if not filtered_ticks or abs(tick - filtered_ticks[-1]) > threshold:
            filtered_ticks.append(tick)

    return filtered_ticks



def parseTFCalc(file_path):
    Environment = {}
    LayerStructure = {}
    dOpt = {}
    MaterialLayer = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        parts = line.strip().split('*')
        if parts[0] == 'ENVIRON':
            # Extract environment settings
            Environment['Wavelength_range'] = [float(parts[1]), float(parts[2])]
            Environment['Medium'] = parts[6]
            Environment['Substrate'] = parts[7]
            Environment['Target_lambda'] = float(parts[4])
        elif parts[0] == 'LAYER':
            # Extract layer information
            idx = int(parts[1])  # Corrected this line
            MaterialLayer[idx] = parts[2]
            dOpt[idx] = float(parts[3])
            # You can extract more variables here if needed
            # For example, physical thickness:
            # physical_thickness[idx] = float(parts[4])
    
    LayerStructure['MaterialLayer'] = MaterialLayer
    LayerStructure['dOpt'] = dOpt
    
    return Environment, LayerStructure


def run_sensitivity_analysis(
    test_stack,
    YAM_CoatingBrownian,
    EFI_tmm,
    find_brownian_noise_for_frequency,
    N=50,
    generate_plots=True,
    save_results=False,
    results_filename="sensitivity_results.csv"
):
    """
    Runs a Morris sensitivity analysis on the given test_stack data using 
    the provided coating noise and optical calculation functions. This function returns the sensitivty of 
    each layer in the stack to the overall optical absorption , reflectivity and CTN 
    - Author STait 2024 -  
    stait@caltech.edu 
    
    
    Dependancies 
    ----------
    
    YAM_CoatingBrownian : module or callable
        A module or callable that provides the function getCoatingThermalNoise(dOpt, materialLayer, materialParams, materialSub, lambda_, f, wBeam, Temp, plots=False).
        
    EFI_tmm : module or callable
        A module or callable that provides the functions CalculateEFI_tmm(...) and CalculateTransmission_tmm(...).
        
    find_brownian_noise_for_frequency : callable
        A function that can find the coating brownian noise at a specific frequency.
    
    
    Inputs 
    ----------
    N : int, optional
        Number of trajectories for the Morris sampling. Default is 50.
        
    generate_plots : bool, optional
        If True, generates and displays the stack and sensitivity plots. Default True.
        
    save_results : bool, optional
        If True, saves the resulting sensitivity dataframe to a CSV file. Default False.
        
    results_filename : str, optional
        The name of the file to save the sensitivity results if save_results is True. Default "sensitivity_results.csv".
        
    Returns
    -------
    sensitivity_df : pandas.DataFrame
        A DataFrame containing the normalized sensitivity metrics for each layer.
    """
    
    # Extract parameters from test_stack
    dOpt_values = test_stack['dOpt']
    materialLayer = test_stack['materialLayer']
    materialParams = test_stack['materialParams']
    materialSub = test_stack['materialSub']
    lambda_ = test_stack['lambda_']
    f = test_stack['f']
    wBeam = test_stack['wBeam']
    Temp = test_stack['Temp']
    t_air = test_stack['t_air']
    polarisation = test_stack['polarisation']
    lambda_list = test_stack['lambda_list']

    # Number of layers
    num_layers = len(dOpt_values)

    # Define the problem for Morris sampling
    problem = {
        'num_vars': num_layers,
        'names': [f'dOpt_{i}' for i in range(num_layers)],
        'bounds': [
            [dOpt_values[i] * 0.99, dOpt_values[i] * 1.01]
            for i in range(num_layers)
        ]
    }

    # Generate samples
    param_values = morris.sample(problem, N=N, num_levels=4)

    # Prepare lists to store outputs
    # We'll append inside run_model, then convert to arrays later
    Y_absorption_list = []
    Y_CTN_list = []
    Y_reflectivity_list = []

    def optical_to_physical(optical_thickness, wavelength, refractive_index):
        return (optical_thickness * wavelength) / refractive_index

    def run_model(params):
        # Create a copy of the test stack and update dOpt
        test_stack_run = test_stack.copy()
        test_stack_run['dOpt'] = params

        # Extract arrays from test_stack_run for convenience
        current_dOpt = test_stack_run['dOpt']
        current_materialLayer = test_stack_run['materialLayer']

        # Calculate refractive index, physical thickness, and attenuation for each layer
        n_coat = np.zeros(np.shape(current_dOpt))
        t_coat = np.zeros(np.shape(current_dOpt))
        k_coat = np.zeros(np.shape(current_dOpt))
        
        for layer_idx, layer_material in enumerate(current_materialLayer):
            n_coat[layer_idx] = materialParams[layer_material]['n']
            k_coat[layer_idx] = materialParams[layer_material]['k']
            t_coat[layer_idx] = optical_to_physical(current_dOpt[layer_idx], lambda_, n_coat[layer_idx]) 

        test_stack_run["d_physical_layers"] = t_coat * 1e9
        test_stack_run['nLayer'] = n_coat

        # Run coating thermal noise calculation
        Noises, _, _, _, _ = YAM_CoatingBrownian.getCoatingThermalNoise(
            test_stack_run["dOpt"],
            test_stack_run["materialLayer"],
            test_stack_run["materialParams"],
            test_stack_run["materialSub"],
            test_stack_run["lambda_"],
            test_stack_run["f"],
            test_stack_run["wBeam"],
            test_stack_run["Temp"],
            plots=False
        )

        # Extract CTN at 100 Hz
        BN_at_F = find_brownian_noise_for_frequency(Noises, frequency=100)
        CTN_at_100Hz = BN_at_F[0]

        # Calculate absorption using EFI_tmm
        _, _, _, _, _, absor = EFI_tmm.CalculateEFI_tmm(
            test_stack_run["dOpt"],
            test_stack_run["materialLayer"],
            test_stack_run["materialParams"],
            test_stack_run["lambda_"],
            test_stack_run["t_air"],
            test_stack_run["polarisation"],
            plots=False
        )

        # Calculate reflectivity at 1064 nm
        _, _, transmission_lambda_0 = EFI_tmm.CalculateTransmission_tmm(
            test_stack_run["dOpt"],
            test_stack_run["materialLayer"],
            test_stack_run["materialParams"],
            [test_stack_run["lambda_"]],
            test_stack_run["lambda_"],
            polarisation=test_stack_run["polarisation"],
            plots=False
        )

        Reflectivity_1064 = 1 - transmission_lambda_0 - absor * 1e-6

        # Append outputs to global lists
        Y_absorption_list.append(absor)
        Y_CTN_list.append(CTN_at_100Hz)
        Y_reflectivity_list.append(Reflectivity_1064)

        return absor, CTN_at_100Hz, Reflectivity_1064

    # Run the parallel computation
    with tqdm_joblib(tqdm(desc="Processing", total=len(param_values))):
        results = Parallel(n_jobs=-1)(
            delayed(run_model)(params) for params in param_values
        )

    # Convert results to arrays
    Y_absorption = np.array([r[0] for r in results])
    Y_CTN = np.array([r[1] for r in results])
    Y_reflectivity = np.array([r[2] for r in results])

    # Analyze sensitivities using Morris
    Si_absorption = morris_analyze.analyze(problem, param_values, Y_absorption, conf_level=0.95, print_to_console=False)
    Si_CTN = morris_analyze.analyze(problem, param_values, Y_CTN, conf_level=0.95, print_to_console=False)
    Si_reflectivity = morris_analyze.analyze(problem, param_values, Y_reflectivity, conf_level=0.95, print_to_console=False)

    # Extract mu_star and normalize
    mu_star_absorption = Si_absorption['mu_star']
    mu_star_CTN = Si_CTN['mu_star']
    mu_star_reflectivity = Si_reflectivity['mu_star']

    normalized_absorption = mu_star_absorption / np.max(mu_star_absorption)
    normalized_CTN = mu_star_CTN / np.max(mu_star_CTN)
    normalized_reflectivity = mu_star_reflectivity / np.max(mu_star_reflectivity)

    # Create DataFrame
    sensitivity_df = pd.DataFrame({
        'Layer': np.arange(num_layers),
        'Absorption_Sensitivity': normalized_absorption,
        'CTN_Sensitivity': normalized_CTN,
        'Reflectivity_Sensitivity': normalized_reflectivity
    })

    # Optionally save results
    if save_results:
        sensitivity_df.to_csv(results_filename, index=False)

    # Optionally generate plots
    if generate_plots:
        unique_n = np.unique(test_stack['nLayer'])

        # Colors for stack (first subplot)
        unique_materials = list(set(materialLayer))
        Stackcolors = plt.cm.viridis(np.linspace(0, 1, np.max(unique_materials)+2))
        stack_color = dict(zip(unique_n, Stackcolors))

        # Colors for sensitivity plots
        colors = plt.cm.PiYG(np.linspace(0, 1, len(unique_n)+5))
        n_to_color = dict(zip(unique_n, colors))

        # Create figure and subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # First Subplot: Generated Stack
        ax_stack = axes[0]
        for i in range(num_layers):
            n_value = test_stack['nLayer'][i]
            color = stack_color[n_value]
            thickness = test_stack['d_physical_layers'][i]
            ax_stack.bar(i, thickness, color=color, width=0.8)

        ax_stack.set_ylabel('Physical Thickness [nm]')
        ax_stack.set_title('Generated Stack')
        ax_stack.set_xticks(np.arange(num_layers))
        ax_stack.set_xticklabels([])
        ax_stack.grid(False)

        legend_elements = [Patch(facecolor=n_to_color[n], label=f"n = {n}") for n in unique_n]
        ax_stack.legend(handles=legend_elements, loc='upper right', title='Refractive Index (n)')

        # Metrics for the next three subplots
        metrics = [
            ('Absorption Sensitivity', 'Absorption_Sensitivity'),
            ('CTN Sensitivity', 'CTN_Sensitivity'),
            ('Reflectivity Sensitivity', 'Reflectivity_Sensitivity')
        ]

        for idx, (title, column) in enumerate(metrics, start=1):
            ax = axes[idx]
            for i in range(num_layers):
                n_value = test_stack['nLayer'][i]
                color = n_to_color[n_value]
                # Normalized sensitivity per thickness
                sensitivity = sensitivity_df.iloc[i][column] / test_stack['d_physical_layers'][i]
                ax.bar(i, sensitivity, color=color, width=0.8)
            ax.set_ylabel('Normalized Sensitivity')
            ax.set_title(title)
            ax.grid(True)
            if 'ref' in title.lower():
                ax.set_yscale('log')

        axes[-1].set_xlabel('Layer Number')
        axes[-1].set_xticks(np.arange(num_layers))
        axes[-1].set_xticklabels(np.arange(num_layers) + 1)

        plt.tight_layout()
        plt.show()

    return sensitivity_df
