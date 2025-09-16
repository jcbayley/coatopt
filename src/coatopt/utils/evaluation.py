"""
Evaluation and visualization utilities for coating optimization results.
Extracted from train_pc_hppo_oml_pareto.py to improve code organization.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import coatopt.environments.utils.coating_utils as coating_utils
from coatopt.utils.plotting.stack import plot_stack


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
        refractive_index = env.materials[material_idx]["n"]
        converted_state[l_ind, 0] = coating_utils.optical_to_physical(
            converted_state[l_ind, 0], env.light_wavelength, refractive_index
        )
    return converted_state


def process_sampled_results(
    sampled_rewards: List[Dict], sampled_vals: List[Dict]
) -> Dict[str, np.ndarray]:
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
        results[f"{key}_rewards"] = np.array(
            [sampled_rewards[i][key] for i in range(len(sampled_rewards))]
        )

    # Process values
    for key in sampled_vals[0].keys():
        results[f"{key}_vals"] = np.array(
            [sampled_vals[i][key] for i in range(len(sampled_vals))]
        )

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
        ax.plot(
            results["reflectivity_vals"], results["absorption_vals"], "o", alpha=0.6
        )
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
        ax.plot(
            results["reflectivity_vals"], results["thermal_noise_vals"], "o", alpha=0.6
        )
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
    filename: str = "evaluation_outputs.h5",
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
        ([3, 1], "opt_state_reverse_2.png", "Optimal State Reverse (Materials 3,1)"),
    ]

    for material_indices, filename, title in optimal_configurations:
        try:
            optimal_state = env.get_optimal_state(inds_alternate=material_indices)
            optimal_value = env.compute_state_value(optimal_state, return_separate=True)

            if env.use_optical_thickness:
                optimal_state = convert_state_to_physical(optimal_state, env)

            fig, ax = plot_stack(optimal_state, env.materials)
            ax.set_title(f"{title}\nOptimal Value: {optimal_value}")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"Warning: Could not generate {filename}: {e}")

    # Two-material optimal states if method exists
    if hasattr(env, "get_optimal_state_2mat"):
        two_mat_configs = [
            ([1, 3, 2], "opt_state_2mat.png", "Optimal State (2-mat: 1,3,2)"),
            ([1, 2, 3], "opt_state_2mat_reverse.png", "Optimal State (2-mat: 1,2,3)"),
        ]

        for material_indices, filename, title in two_mat_configs:
            try:
                optimal_state = env.get_optimal_state_2mat(
                    inds_alternate=material_indices
                )
                optimal_value = env.compute_state_value(
                    optimal_state, return_separate=True
                )

                if env.use_optical_thickness:
                    optimal_state = convert_state_to_physical(optimal_state, env)

                fig, ax = plot_stack(optimal_state, env.materials)
                ax.set_title(f"{title}\nOptimal Value: {optimal_value}")
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, filename), dpi=150)
                plt.close(fig)

            except Exception as e:
                print(f"Warning: Could not generate {filename}: {e}")


def run_evaluation_pipeline(
    trainer, env, n_samples: int, output_dir: str
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

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print(f"Generating {n_samples} solution samples...")
    sampled_states, sampled_rewards, sampled_weights, sampled_vals = (
        trainer.generate_solutions(n_samples, random_weights=True)
    )

    print("Processing results...")
    results = process_sampled_results(sampled_rewards, sampled_vals)

    print("Creating Pareto front visualizations...")
    create_pareto_plots(results, output_dir)

    print("Saving results to HDF5...")
    save_results_to_hdf5(sampled_states, sampled_weights, results)

    print("Creating enhanced Pareto plots...")
    create_enhanced_pareto_plots(
        trainer, env, results, sampled_states, sampled_weights, output_dir
    )

    # print("Generating optimal states analysis...")
    # generate_optimal_states_analysis(env, output_dir)

    print(f"Evaluation complete. Results saved to {output_dir}")

    return sampled_states, results, sampled_weights


def create_enhanced_pareto_plots(
    trainer,
    env,
    evaluation_results: Dict[str, np.ndarray],
    sampled_states: np.ndarray,
    sampled_weights: np.ndarray,
    output_dir: str,
) -> None:
    """
    Create enhanced Pareto plots that combine training samples with evaluation samples.

    Args:
        trainer: Trained model with historical data
        env: Environment object
        evaluation_results: Results from evaluation pipeline
        sampled_states: Evaluation sampled states
        sampled_weights: Evaluation sampled weights
        output_dir: Directory to save plots
    """
    try:
        # Get objective labels and parameters
        objectives = (
            env.optimise_parameters if hasattr(env, "optimise_parameters") else []
        )

        if len(objectives) < 2:
            print("Warning: Need at least 2 objectives for Pareto plots")
            return

        # Extract evaluation data
        eval_data = {}
        for obj in objectives:
            vals_key = f"{obj}_vals"
            if vals_key in evaluation_results:
                eval_data[obj] = evaluation_results[vals_key]
            else:
                print(f"Warning: {vals_key} not found in evaluation results")
                return

        # Try to get training data from trainer
        training_data = None
        if hasattr(trainer, "best_states") and trainer.best_states:
            training_data = extract_training_pareto_data(
                trainer.best_states, objectives
            )

        # Create enhanced plots
        create_combined_pareto_plots(
            evaluation_data=eval_data,
            training_data=training_data,
            objectives=objectives,
            output_dir=output_dir,
            n_eval_samples=len(sampled_states),
        )

        print(f"Enhanced Pareto plots saved to {output_dir}")

    except Exception as e:
        print(f"Error creating enhanced Pareto plots: {e}")
        # Fallback to regular plots
        create_pareto_plots(evaluation_results, output_dir)


def extract_training_pareto_data(
    best_states: List, objectives: List[str]
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract Pareto front data from training best states.

    Args:
        best_states: List of best states from training
        objectives: List of objective parameter names

    Returns:
        Dictionary containing training Pareto data
    """
    try:
        # Extract objective values from best states
        training_vals = []
        for tot_reward, epoch, state, rewards, vals in best_states:
            obj_values = []
            for obj in objectives:
                if obj in vals:
                    obj_values.append(vals[obj])
                else:
                    obj_values = None
                    break

            if obj_values is not None:
                training_vals.append(obj_values)

        if not training_vals:
            return None

        # Convert to array and find Pareto front
        vals_array = np.array(training_vals)

        # Convert to minimization objectives for all parameters
        minimization_objectives = np.zeros_like(vals_array)
        for i, obj in enumerate(objectives):
            if obj == "reflectivity":
                minimization_objectives[:, i] = 1 - vals_array[:, i]  # Minimize (1-R)
            else:
                minimization_objectives[:, i] = vals_array[:, i]  # Minimize directly

        # Find Pareto front using non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(minimization_objectives)

        if len(fronts) > 0 and len(fronts[0]) > 0:
            pareto_indices = fronts[0]
            pareto_vals = vals_array[pareto_indices]

            # Convert back to dictionary format
            result = {}
            for i, obj in enumerate(objectives):
                result[obj] = pareto_vals[:, i]

            return result

        return None

    except Exception as e:
        print(f"Error extracting training Pareto data: {e}")
        return None


def create_combined_pareto_plots(
    evaluation_data: Dict[str, np.ndarray],
    training_data: Optional[Dict[str, np.ndarray]],
    objectives: List[str],
    output_dir: str,
    n_eval_samples: int,
) -> None:
    """
    Create combined Pareto plots showing both training and evaluation samples.

    Args:
        evaluation_data: Dictionary of evaluation objective values
        training_data: Dictionary of training Pareto front values (optional)
        objectives: List of objective parameter names
        output_dir: Directory to save plots
        n_eval_samples: Number of evaluation samples
    """
    # Label mapping for better plot labels
    label_mapping = {
        "reflectivity": "1 - Reflectivity",
        "absorption": "Absorption [ppm]",
        "thermal_noise": "Thermal Noise [m/√Hz]",
        "thickness": "Total Thickness [nm]",
    }

    # Scale mapping for log/linear scales
    scale_mapping = {
        "reflectivity": "log",
        "absorption": "log",
        "thermal_noise": "log",
        "thickness": "linear",
    }

    # Create all pairwise combinations
    n_objectives = len(objectives)

    if n_objectives == 2:
        # Single plot for 2 objectives
        obj_i, obj_j = objectives[0], objectives[1]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot evaluation samples
        eval_x = evaluation_data[obj_i]
        eval_y = evaluation_data[obj_j]

        # Transform for plotting (minimize all)
        if obj_i == "reflectivity":
            eval_x = 1 - eval_x
        if obj_j == "reflectivity":
            eval_y = 1 - eval_y

        ax.scatter(
            eval_x,
            eval_y,
            c="blue",
            alpha=0.6,
            s=20,
            label=f"Evaluation Samples ({n_eval_samples})",
            marker="o",
        )

        # Plot training Pareto front if available
        if training_data is not None:
            train_x = training_data[obj_i]
            train_y = training_data[obj_j]

            if obj_i == "reflectivity":
                train_x = 1 - train_x
            if obj_j == "reflectivity":
                train_y = 1 - train_y

            ax.scatter(
                train_x,
                train_y,
                c="red",
                alpha=0.8,
                s=40,
                label=f"Training Pareto Front ({len(train_x)})",
                marker="s",
            )

        # Set labels and scales
        ax.set_xlabel(label_mapping.get(obj_i, obj_i.replace("_", " ").title()))
        ax.set_ylabel(label_mapping.get(obj_j, obj_j.replace("_", " ").title()))

        if scale_mapping.get(obj_i) == "log":
            ax.set_xscale("log")
        if scale_mapping.get(obj_j) == "log":
            ax.set_yscale("log")

        ax.set_title("Combined Pareto Front: Training vs Evaluation")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "combined_pareto_front.png"), dpi=150)
        plt.close(fig)

    else:
        # Multiple pairwise plots
        n_pairs = n_objectives * (n_objectives - 1) // 2
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        pair_idx = 0
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                if pair_idx < len(axes):
                    ax = axes[pair_idx]
                    obj_i, obj_j = objectives[i], objectives[j]

                    # Plot evaluation samples
                    eval_x = evaluation_data[obj_i]
                    eval_y = evaluation_data[obj_j]

                    # Transform for plotting
                    if obj_i == "reflectivity":
                        eval_x = 1 - eval_x
                    if obj_j == "reflectivity":
                        eval_y = 1 - eval_y

                    ax.scatter(
                        eval_x,
                        eval_y,
                        c="blue",
                        alpha=0.6,
                        s=15,
                        label="Evaluation",
                        marker="o",
                    )

                    # Plot training Pareto front if available
                    if training_data is not None:
                        train_x = training_data[obj_i]
                        train_y = training_data[obj_j]

                        if obj_i == "reflectivity":
                            train_x = 1 - train_x
                        if obj_j == "reflectivity":
                            train_y = 1 - train_y

                        ax.scatter(
                            train_x,
                            train_y,
                            c="red",
                            alpha=0.8,
                            s=25,
                            label="Training",
                            marker="s",
                        )

                    # Set labels and scales
                    ax.set_xlabel(
                        label_mapping.get(obj_i, obj_i.replace("_", " ").title())
                    )
                    ax.set_ylabel(
                        label_mapping.get(obj_j, obj_j.replace("_", " ").title())
                    )

                    if scale_mapping.get(obj_i) == "log":
                        ax.set_xscale("log")
                    if scale_mapping.get(obj_j) == "log":
                        ax.set_yscale("log")

                    ax.grid(True, alpha=0.3)
                    if pair_idx == 0:  # Only show legend on first plot
                        ax.legend()

                pair_idx += 1

        # Hide unused subplots
        for idx in range(pair_idx, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"Combined Pareto Fronts: Training vs Evaluation ({n_eval_samples} samples)",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "combined_pareto_front_multi.png"), dpi=150
        )
        plt.close(fig)
