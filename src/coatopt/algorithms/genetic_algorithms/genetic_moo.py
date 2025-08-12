"""
Multi-objective genetic algorithm implementation for coating optimisation using PyMOO.
Provides CoatingMOO problem class and GeneticTrainer for structured training.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
import h5py

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback

from coatopt.environments import coating_reward_function
from coatopt.config.structured_config import GeneticConfig
from coatopt.tools import plotting


class CoatingMOO(ElementwiseProblem):
    """Multi-objective optimisation problem for coating design using PyMOO."""
    
    def __init__(self, environment, **kwargs):
        """
        Initialize coating optimisation problem.
        
        Args:
            environment: Coating environment instance
            **kwargs: Additional PyMOO problem arguments
        """
        self.env = environment
        self.n_var = self.env.max_layers * 2  # thickness + material for each layer
        n_obj = len(self.env.optimise_parameters)  # Number of objectives
        
        # Define variable bounds
        thick_lower = np.repeat(self.env.min_thickness, self.env.max_layers)
        thick_upper = np.repeat(self.env.max_thickness, self.env.max_layers)
        material_lower = np.repeat(0, self.env.max_layers)
        material_upper = np.repeat(self.env.n_materials - 1, self.env.max_layers)
        
        xl = np.concatenate((thick_lower, material_lower))
        xu = np.concatenate((thick_upper, material_upper))
        
        super().__init__(n_var=self.n_var, n_obj=n_obj, xl=xl, xu=xu, **kwargs)

    def make_state_from_vars(self, vars: np.ndarray) -> np.ndarray:
        """
        Convert optimisation variables to coating state representation.
        
        Args:
            vars: optimisation variables [thicknesses..., materials...]
            
        Returns:
            State array with shape (max_layers, n_materials+1)
        """
        state = np.zeros((self.env.max_layers, self.env.n_materials + 1))
        layer_thickness = vars[:self.env.max_layers]
        materials_inds = np.floor(vars[self.env.max_layers:]).astype(int)
        
        for i in range(self.env.max_layers):
            state[i, 0] = layer_thickness[i]
            # Add 2 to account for air (0) and substrate (1) indices
            state[i, materials_inds[i] + 2] = 1
            
        return state

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        """
        Evaluate objectives for given design variables.
        
        Args:
            X: Design variables
            out: Output dictionary for objectives and constraints
        """
        state = self.make_state_from_vars(X)
        total_reward, vals, rewards = self.env.compute_reward(state)
        
        # Create objective vector (minimization problem, so negate rewards)
        objectives = []
        for param in self.env.optimise_parameters:
            if param in rewards:
                objectives.append(-rewards[param])
            else:
                objectives.append(0.0)  # Default if parameter not found
        
        out["F"] = np.array(objectives)
        out["VALS"] = vals


class CheckpointCallback(Callback):
    """Callback for saving checkpoints and creating plots during optimization."""
    
    def __init__(self, trainer, checkpoint_interval=50):
        """
        Initialize checkpoint callback.
        
        Args:
            trainer: GeneticTrainer instance
            checkpoint_interval: Save checkpoint every N generations
        """
        super().__init__()
        self.trainer = trainer
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = os.path.join(trainer.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def notify(self, algorithm):
        """Called after each generation."""
        generation = algorithm.n_gen
        
        # Save checkpoint every N generations
        if generation % self.checkpoint_interval == 0 and generation > 0:
            self._save_checkpoint(algorithm, generation)
    
    def _save_checkpoint(self, algorithm, generation):
        """Save checkpoint data and create plots using existing trainer methods."""
        print(f"\n--- Saving checkpoint at generation {generation} ---")
        
        try:
            # Create checkpoint subdirectory
            checkpoint_subdir = os.path.join(self.checkpoint_dir, f"gen_{generation}")
            os.makedirs(checkpoint_subdir, exist_ok=True)
            
            # Create a Population object from current algorithm population
            current_pop = algorithm.pop
            
            # Use existing trainer method to process population data
            states, results = self.trainer._process_population_data(current_pop)
            
            # Save checkpoint data using existing methods
            self._save_checkpoint_data(states, results, checkpoint_subdir, generation)
            
            # Create plots using existing methods
            self._create_checkpoint_plots(results, states, checkpoint_subdir, generation)
            
            print(f"Checkpoint saved to: {checkpoint_subdir}")
            
        except Exception as e:
            print(f"Warning: Failed to save checkpoint at generation {generation}: {e}")
    
    def _save_checkpoint_data(self, states, results, checkpoint_dir, generation):
        """Save checkpoint data using existing trainer methods."""
        # Use existing dataframe creation method
        df = self.trainer._create_results_dataframe(states, results)
        
        # Save to CSV with generation-specific name
        csv_path = os.path.join(checkpoint_dir, f"population_gen_{generation}.csv")
        df.to_csv(csv_path, index=False)
        
        # Create summary file
        self._create_summary_file(results, checkpoint_dir, generation)
    
    def _create_summary_file(self, results, checkpoint_dir, generation):
        """Create a summary file with statistics."""
        summary_path = os.path.join(checkpoint_dir, f"summary_gen_{generation}.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Checkpoint Summary - Generation {generation}\n")
            f.write(f"================================\n\n")
            f.write(f"Population Size: {len(list(results.values())[0])}\n")
            f.write(f"Objectives: {', '.join(self.trainer.env.optimise_parameters)}\n")
            
            # Statistics for each objective
            for param in self.trainer.env.optimise_parameters:
                vals_key = f"{param}_vals"
                if vals_key in results:
                    vals = results[vals_key]
                    f.write(f"\n{param.title()} Statistics:\n")
                    f.write(f"  Min: {np.min(vals):.6f}\n")
                    f.write(f"  Max: {np.max(vals):.6f}\n")
                    f.write(f"  Mean: {np.mean(vals):.6f}\n")
                    f.write(f"  Std: {np.std(vals):.6f}\n")
    
    def _create_checkpoint_plots(self, results, states, checkpoint_dir, generation):
        """Create plots using existing trainer methods."""
        # Create Pareto front plot using existing method with modified output path
        if len(self.trainer.env.optimise_parameters) >= 2:
            try:
                # Temporarily modify trainer's output_dir to save to checkpoint dir
                original_output_dir = self.trainer.output_dir
                self.trainer.output_dir = checkpoint_dir
                
                # Use existing Pareto plot method
                self.trainer._create_pareto_plot(results)
                
                # Rename the generated plot to include generation number
                original_plot_path = os.path.join(checkpoint_dir, "pareto_front.png")
                new_plot_path = os.path.join(checkpoint_dir, f"pareto_front_gen_{generation}.png")
                
                if os.path.exists(original_plot_path):
                    os.rename(original_plot_path, new_plot_path)
                
                # Restore original output directory
                self.trainer.output_dir = original_output_dir
                
            except Exception as e:
                print(f"Warning: Failed to create Pareto plot for gen {generation}: {e}")
        
        # Create sample coating plots using existing method
        self._create_sample_coating_plots(states, checkpoint_dir, generation)
    
    def _create_sample_coating_plots(self, states, checkpoint_dir, generation):
        """Create sample coating plots using existing plotting functionality."""
        # Create a few sample coating plots (similar to existing _create_coating_plots but for checkpoint)
        n_sample_plots = min(5, len(states))
        
        for i in range(n_sample_plots):
            try:
                state = states[i]
                
                # Calculate rewards and values for this state
                total_reward, vals, rewards = self.trainer.env.compute_reward(state)
                
                # Use existing plotting function
                fig, ax = plotting.plot_stack(state, self.trainer.env.materials, rewards=rewards, vals=vals)
                fig.suptitle(f"Generation {generation} - Sample {i+1}")
                
                plot_path = os.path.join(checkpoint_dir, f"coating_sample_{i+1}_gen_{generation}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
            except Exception as e:
                print(f"Warning: Failed to create coating plot {i+1} for gen {generation}: {e}")


class GeneticTrainer:
    """Genetic algorithm trainer for coating optimisation."""
    
    def __init__(
        self,
        environment,
        config: GeneticConfig,
        output_dir: str
    ):
        """
        Initialize genetic algorithm trainer.
        
        Args:
            environment: Coating environment instance
            config: Genetic algorithm configuration
            output_dir: Directory for saving outputs
        """
        self.env = environment
        self.config = config
        self.output_dir = output_dir
        self.problem = CoatingMOO(environment)
        self.algorithm = self._create_algorithm()
        self.result = None
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "states"), exist_ok=True)

    def _create_algorithm(self):
        """Create PyMOO algorithm based on configuration."""
        if self.config.algorithm == "NSGA2":
            return NSGA2(
                pop_size=self.config.population_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=self.config.crossover_probability, eta=self.config.crossover_eta),
                mutation=PM(prob=self.config.mutation_probability, eta=self.config.mutation_eta),
                eliminate_duplicates=self.config.eliminate_duplicates,
                survival=RankAndCrowdingSurvival(),
            )
        
        elif self.config.algorithm == "NSGA3":
            n_partitions = self.config.n_partitions or 3000
            ref_dirs = get_reference_directions(
                "uniform", 
                len(self.env.optimise_parameters), 
                n_partitions=n_partitions
            )
            return NSGA3(
                pop_size=self.config.population_size,
                sampling=FloatRandomSampling(),
                ref_dirs=ref_dirs,
                eliminate_duplicates=self.config.eliminate_duplicates
            )
        
        elif self.config.algorithm == "MOEAD":
            n_partitions = self.config.n_partitions or 5000
            n_neighbors = self.config.n_neighbors or 20000
            prob_neighbor_mating = self.config.prob_neighbor_mating or 0.6
            
            ref_dirs = get_reference_directions(
                "uniform",
                len(self.env.optimise_parameters),
                n_partitions=n_partitions
            )
            return MOEAD(
                ref_dirs,
                n_neighbors=n_neighbors,
                prob_neighbor_mating=prob_neighbor_mating,
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def train(self, checkpoint_interval=None):
        """Execute genetic algorithm optimisation with checkpointing.
        
        Args:
            checkpoint_interval: Save checkpoints every N generations (if None, uses config value)
        """
        if checkpoint_interval is None:
            checkpoint_interval = getattr(self.config, 'checkpoint_interval', 50)
            
        print(f"Starting genetic algorithm optimisation with {self.config.algorithm}")
        print(f"Population size: {self.config.population_size}")
        print(f"Generations: {self.config.n_generations}")
        print(f"Checkpointing every {checkpoint_interval} generations")
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(self, checkpoint_interval)
        
        self.result = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', self.config.n_generations),
            seed=self.config.seed,
            save_history=True,
            verbose=True,
            callback=checkpoint_callback
        )
        
        print("Optimisation completed")
        print("Saving final results and creating visualizations...")
        
        # Automatically save results after training
        try:
            self.evaluate()
            print("Results saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
            # Continue execution even if saving fails
        
        # Save optimizer state
        try:
            self.save_optimizer_state()
        except Exception as e:
            print(f"Warning: Failed to save optimizer state: {e}")

    def evaluate(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Evaluate results and create visualizations.
        
        Args:
            n_samples: Number of samples to process (if None, use all)
            
        Returns:
            Tuple of (sampled_states, results, sampled_weights)
        """
        if self.result is None:
            raise RuntimeError("Must run training before evaluation")
        
        # Collect all population data from history
        all_pop = Population()
        for algorithm in self.result.history:
            all_pop = Population.merge(all_pop, algorithm.off)
        
        # Process population data
        states_data, results_data = self._process_population_data(all_pop, n_samples)
        
        # Process final Pareto front
        pareto_states_data, pareto_results_data = self._process_pareto_front()
        
        # Save results
        self._save_results(states_data, results_data, pareto_states_data, pareto_results_data)
        
        # Create visualizations
        self._create_visualizations(pareto_results_data, pareto_states_data)
        
        # Return in format expected by evaluation utilities
        sampled_weights = np.ones((len(states_data), len(self.env.optimise_parameters))) / len(self.env.optimise_parameters)
        
        return states_data, results_data, sampled_weights

    def _process_population_data(self, population: Population, n_samples: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Process population data to extract states and computed values."""
        X = population.get("X")
        F = population.get("F")
        
        if n_samples is not None:
            indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
            X = X[indices]
            F = F[indices]
        
        # Convert to states and compute actual values
        states = []
        results = {f"{param}_vals": [] for param in self.env.optimise_parameters}
        results.update({f"{param}_rewards": [] for param in self.env.optimise_parameters})
        
        for i, row in enumerate(X):
            state = self.problem.make_state_from_vars(row)
            states.append(state)
            
            # Compute actual values
            vals = self.env.compute_state_value(state, return_separate=True)
            val_names = ["reflectivity", "thermal_noise", "absorption", "thickness"]
            
            for j, param in enumerate(self.env.optimise_parameters):
                if param in val_names:
                    param_idx = val_names.index(param)
                    results[f"{param}_vals"].append(vals[param_idx])
                else:
                    results[f"{param}_vals"].append(0.0)
                
                # Use objective values (negated rewards)
                if j < len(F[i]):
                    results[f"{param}_rewards"].append(-F[i, j])
                else:
                    results[f"{param}_rewards"].append(0.0)
        
        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return np.array(states), results

    def _process_pareto_front(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Process Pareto front solutions."""
        if len(self.result.X.shape) < 2:
            X = np.array([self.result.X])
            F = np.array([self.result.F])
        else:
            X = self.result.X
            F = self.result.F
        
        return self._process_population_data(Population.new("X", X, "F", F))

    def _save_results(
        self,
        all_states: np.ndarray,
        all_results: Dict[str, np.ndarray],
        pareto_states: np.ndarray,
        pareto_results: Dict[str, np.ndarray]
    ):
        """Save results to CSV and HDF5 files."""
        # Save all population data
        all_df = self._create_results_dataframe(all_states, all_results)
        all_df.to_csv(os.path.join(self.output_dir, "population_data.csv"), index=False)
        
        # Save Pareto front data
        pareto_df = self._create_results_dataframe(pareto_states, pareto_results)
        pareto_df.to_csv(os.path.join(self.output_dir, "optimized_data.csv"), index=False)
        
        # Save to HDF5
        self._save_to_hdf5(all_states, all_results, pareto_states, pareto_results)

    def _create_results_dataframe(self, states: np.ndarray, results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Create pandas DataFrame from states and results."""
        # Create DataFrame with design variables
        df = pd.DataFrame()
        
        for i in range(self.problem.n_var):
            var_data = []
            for state in states:
                if i < self.env.max_layers:  # Thickness variables
                    var_data.append(state[i, 0])
                else:  # Material variables
                    layer_idx = i - self.env.max_layers
                    material_idx = np.argmax(state[layer_idx, 1:]) - 1  # -1 to account for air offset
                    var_data.append(material_idx)
            df[f"X{i+1}"] = var_data
        
        # Add objective values and rewards
        for param in self.env.optimise_parameters:
            if f"{param}_vals" in results:
                df[param.title()] = results[f"{param}_vals"]
            if f"{param}_rewards" in results:
                df[f"{param.title()}_r"] = results[f"{param}_rewards"]
        
        return df

    def _save_to_hdf5(
        self,
        all_states: np.ndarray,
        all_results: Dict[str, np.ndarray],
        pareto_states: np.ndarray,
        pareto_results: Dict[str, np.ndarray]
    ):
        """Save results to HDF5 file."""
        hdf5_path = os.path.join(self.output_dir, "genetic_optimisation_results.h5")
        
        with h5py.File(hdf5_path, 'w') as f:
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.create_dataset('algorithm', data=self.config.algorithm)
            meta_group.create_dataset('population_size', data=self.config.population_size)
            meta_group.create_dataset('n_generations', data=self.config.n_generations)
            meta_group.create_dataset('objectives', data=[param.encode() for param in self.env.optimise_parameters])
            
            # Save all population data
            pop_group = f.create_group('population_data')
            pop_group.create_dataset('states', data=all_states, compression='gzip')
            for key, data in all_results.items():
                pop_group.create_dataset(key, data=data, compression='gzip')
            
            # Save Pareto front data
            pareto_group = f.create_group('pareto_front')
            pareto_group.create_dataset('states', data=pareto_states, compression='gzip')
            for key, data in pareto_results.items():
                pareto_group.create_dataset(key, data=data, compression='gzip')

    def save_optimizer_state(self):
        """Save the optimizer state including results and training history."""
        if self.result is None:
            print("No optimization results to save. Run training first.")
            return
        
        import pickle
        
        # Save the full result object using pickle
        result_path = os.path.join(self.output_dir, "optimizer_result.pkl")
        with open(result_path, 'wb') as f:
            pickle.dump(self.result, f)
        
        print(f"Optimizer state saved to: {result_path}")
        
        # Also save a summary of the optimization
        summary_path = os.path.join(self.output_dir, "optimization_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Genetic Algorithm Optimization Summary\n")
            f.write(f"=====================================\n\n")
            f.write(f"Algorithm: {self.config.algorithm}\n")
            f.write(f"Population Size: {self.config.population_size}\n")
            f.write(f"Generations: {self.config.n_generations}\n")
            f.write(f"Objectives: {', '.join(self.env.optimise_parameters)}\n")
            f.write(f"Total Evaluations: {len(self.result.history) * self.config.population_size}\n")
            
            # Summary statistics of Pareto front
            if hasattr(self.result, 'F') and self.result.F is not None:
                if len(self.result.F.shape) == 1:
                    f.write(f"Final Pareto Front Size: 1\n")
                else:
                    f.write(f"Final Pareto Front Size: {len(self.result.F)}\n")
            
            f.write(f"\nOutput Directory: {self.output_dir}\n")
            f.write(f"Files Generated:\n")
            f.write(f"  - population_data.csv: All evaluated solutions\n")
            f.write(f"  - optimized_data.csv: Pareto front solutions\n")
            f.write(f"  - genetic_optimisation_results.h5: Detailed results\n")
            f.write(f"  - optimizer_result.pkl: Full optimizer state\n")
            f.write(f"  - pareto_front.png: Pareto front visualization\n")
            f.write(f"  - states/: Individual coating visualizations\n")
        
        print(f"Optimization summary saved to: {summary_path}")

    def load_optimizer_state(self, result_path: str):
        """Load a previously saved optimizer state."""
        import pickle
        
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Optimizer state file not found: {result_path}")
        
        with open(result_path, 'rb') as f:
            self.result = pickle.load(f)
        
        print(f"Optimizer state loaded from: {result_path}")
        return self.result

    def _create_visualizations(self, results: Dict[str, np.ndarray], states: np.ndarray):
        """Create visualization plots."""
        # Pareto front plot
        if len(self.env.optimise_parameters) >= 2:
            self._create_pareto_plot(results)
        
        # Individual coating visualizations
        self._create_coating_plots(states, results)

    def _create_pareto_plot(self, results: Dict[str, np.ndarray]):
        """Create Pareto front visualization."""
        params = self.env.optimise_parameters
        if len(params) < 2:
            return
        
        param1, param2 = params[0], params[1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert objectives back to actual values for plotting
        x_vals = results[f"{param1}_vals"]
        y_vals = results[f"{param2}_vals"]
        
        # Handle reflectivity conversion
        if param1 == "reflectivity":
            x_vals = 1 - 10**(-results[f"{param1}_rewards"])
        if param2 == "absorption":
            y_vals = 10**(-results[f"{param2}_rewards"] - 10)
        
        ax.scatter(x_vals, y_vals, s=10, c="red", alpha=0.5)
        ax.set_xlabel(param1.title())
        ax.set_ylabel(param2.title())
        ax.set_title(f"Pareto Front: {param1.title()} vs {param2.title()}")
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "pareto_front.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_coating_plots(self, states: np.ndarray, results: Dict[str, np.ndarray]):
        """Create individual coating stack visualizations."""
        n_plots = min(len(states), 10)  # Limit to first 10 solutions
        
        for i in range(n_plots):
            state = states[i]
            
            # Calculate rewards and values for this state
            total_reward, vals, rewards = self.env.compute_reward(state)
            
            try:
                fig, ax = plotting.plot_stack(state, self.env.materials, rewards=rewards, vals=vals)
                fig.savefig(os.path.join(self.output_dir, "states", f"stack_{i}.png"), 
                          dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Error creating plot for stack {i}: {e}")
                continue

    def generate_solutions(self, n_samples: int, random_weights: bool = True) -> Tuple[np.ndarray, List[Dict], np.ndarray, List[Dict]]:
        """
        Generate solutions compatible with evaluation pipeline interface.
        
        Args:
            n_samples: Number of samples to generate
            random_weights: Ignored for genetic algorithms
            
        Returns:
            Tuple of (states, rewards, weights, vals)
        """
        if self.result is None:
            raise RuntimeError("Must run training before generating solutions")
        
        # Use all available solutions from history
        all_pop = Population()
        for algorithm in self.result.history:
            all_pop = Population.merge(all_pop, algorithm.off)
        
        X = all_pop.get("X")
        F = all_pop.get("F")
        
        # Sample if needed
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            F = F[indices]
        
        # Convert to expected format
        states = []
        rewards_list = []
        vals_list = []
        
        for i, (vars, objectives) in enumerate(zip(X, F)):
            state = self.problem.make_state_from_vars(vars)
            states.append(state)
            
            # Compute actual values
            total_reward, vals, rewards = self.env.compute_reward(state)
            rewards_list.append(rewards)
            vals_list.append(vals)
        
        # Create dummy weights (equal weighting)
        weights = np.ones((len(states), len(self.env.optimise_parameters))) / len(self.env.optimise_parameters)
        
        return np.array(states), rewards_list, weights, vals_list
