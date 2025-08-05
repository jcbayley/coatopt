"""
Evaluation and visualization utilities for coating optimization results.
Extracted from train_pc_hppo_oml_pareto.py to improve code organization.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import Dict, List, Tuple, Any
import coatopt.environments.coating_utils as coating_utils


def convert_state_to_physical(state: np.ndarray, env) -> np.ndarray:
    """
    Convert optical thickness to physical thickness for coating states.
    
    Args:
        state: Coating state array with optical thicknesses
        env: Environment object containing materials and wavelength info
        
    Returns:
        State array with physical thicknesses
    """
    converted_state = state.copy()
    for l_ind in range(len(converted_state)):
        material_idx = np.argmax(converted_state[l_ind][1:])
        refractive_index = env.materials[material_idx]['n']
        converted_state[l_ind, 0] = coating_utils.optical_to_physical(
            converted_state[l_ind, 0], 
            env.light_wavelength, 
            refractive_index
        )
    return converted_state


def process_sampled_results(sampled_rewards: List[Dict], sampled_vals: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Process sampled rewards and values into arrays for analysis.
    
    Args:
        sampled_rewards: List of reward dictionaries from sampling
        sampled_vals: List of value dictionaries from sampling
        
    Returns:
        Dictionary containing processed reward and value arrays
    """
    results = {}
    
    # Process rewards
    for key in sampled_rewards[0].keys():
        if key in ["updated_pareto_front", "front_updated"]:
            continue
        results[f"{key}_rewards"] = np.array([
            sampled_rewards[i][key] for i in range(len(sampled_rewards))
        ])
    
    # Process values
    for key in sampled_vals[0].keys():
        results[f"{key}_vals"] = np.array([
            sampled_vals[i][key] for i in range(len(sampled_vals))
        ])
    
    return results


def create_pareto_plots(results: Dict[str, np.ndarray], output_dir: str) -> None:
    """
    Create Pareto front visualization plots.
    
    Args:
        results: Processed results dictionary
        output_dir: Directory to save plots
    """
    # Reflectivity vs Absorption plot
    if "reflectivity_vals" in results and "absorption_vals" in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(results["reflectivity_vals"], results["absorption_vals"], "o", alpha=0.6)
        ax.set_xlabel("Reflectivity")
        ax.set_ylabel("Absorption")
        ax.set_title("Pareto Front: Reflectivity vs Absorption")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reflectivity_absorption.png"), dpi=150)
        plt.close(fig)
    
    # Reflectivity vs Thermal Noise plot
    if "reflectivity_vals" in results and "thermal_noise_vals" in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(results["reflectivity_vals"], results["thermal_noise_vals"], "o", alpha=0.6)
        ax.set_xlabel("Reflectivity")
        ax.set_ylabel("Thermal Noise")
        ax.set_title("Pareto Front: Reflectivity vs Thermal Noise")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reflectivity_thermal_noise.png"), dpi=150)
        plt.close(fig)


def save_results_to_hdf5(
    sampled_states: np.ndarray,
    sampled_weights: np.ndarray,
    results: Dict[str, np.ndarray],
    filename: str = "sampled_outputs.h5"
) -> None:
    """
    Save optimization results to HDF5 file.
    
    Args:
        sampled_states: Array of sampled coating states
        sampled_weights: Array of objective weights used
        results: Processed results dictionary
        filename: Output filename
    """
    with h5py.File(filename, "w") as f:
        f.create_dataset("states", data=sampled_states)
        f.create_dataset("weights", data=sampled_weights)
        for key, data in results.items():
            f.create_dataset(key, data=data)


def generate_optimal_states_analysis(env, output_dir: str) -> None:
    """
    Generate analysis of optimal states with different material combinations.
    
    Args:
        env: Environment object
        output_dir: Directory to save plots
    """
    optimal_configurations = [
        ([1, 2], "opt_state.png", "Optimal State (Materials 1,2)"),
        ([2, 1], "opt_state_reverse.png", "Optimal State Reverse (Materials 2,1)"),
        ([1, 3], "opt_state_2.png", "Optimal State (Materials 1,3)"),
        ([3, 1], "opt_state_reverse_2.png", "Optimal State Reverse (Materials 3,1)")
    ]
    
    for material_indices, filename, title in optimal_configurations:
        try:
            optimal_state = env.get_optimal_state(inds_alternate=material_indices)
            optimal_value = env.compute_state_value(optimal_state, return_separate=True)
            
            if env.use_optical_thickness:
                optimal_state = convert_state_to_physical(optimal_state, env)
            
            fig, ax = env.plot_stack(optimal_state)
            ax.set_title(f"{title}\nOptimal Value: {optimal_value}")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not generate {filename}: {e}")
    
    # Two-material optimal states if method exists
    if hasattr(env, 'get_optimal_state_2mat'):
        two_mat_configs = [
            ([1, 3, 2], "opt_state_2mat.png", "Optimal State (2-mat: 1,3,2)"),
            ([1, 2, 3], "opt_state_2mat_reverse.png", "Optimal State (2-mat: 1,2,3)")
        ]
        
        for material_indices, filename, title in two_mat_configs:
            try:
                optimal_state = env.get_optimal_state_2mat(inds_alternate=material_indices)
                optimal_value = env.compute_state_value(optimal_state, return_separate=True)
                
                if env.use_optical_thickness:
                    optimal_state = convert_state_to_physical(optimal_state, env)
                
                fig, ax = env.plot_stack(optimal_state)
                ax.set_title(f"{title}\nOptimal Value: {optimal_value}")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, filename), dpi=150)
                plt.close(fig)
                
            except Exception as e:
                print(f"Warning: Could not generate {filename}: {e}")


def run_evaluation_pipeline(
    trainer, 
    env, 
    n_samples: int, 
    output_dir: str
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Run complete evaluation pipeline including sampling, processing, and visualization.
    
    Args:
        trainer: Trained model for sampling
        env: Environment object
        n_samples: Number of samples to generate
        output_dir: Directory for output files
        
    Returns:
        Tuple of (sampled_states, processed_results, sampled_weights)
    """
    print(f"Generating {n_samples} solution samples...")
    sampled_states, sampled_rewards, sampled_weights, sampled_vals = trainer.generate_solutions(
        n_samples, random_weights=True
    )
    
    print("Processing results...")
    results = process_sampled_results(sampled_rewards, sampled_vals)
    
    print("Creating Pareto front visualizations...")
    create_pareto_plots(results, output_dir)
    
    print("Saving results to HDF5...")
    save_results_to_hdf5(sampled_states, sampled_weights, results)
    
    print("Generating optimal states analysis...")
    generate_optimal_states_analysis(env, output_dir)
    
    print(f"Evaluation complete. Results saved to {output_dir}")
    
    return sampled_states, results, sampled_weights