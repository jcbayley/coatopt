"""
MCTS Trainer for multi-objective Pareto optimization of coating design.
Uses Monte Carlo Tree Search for discrete material selection combined with
continuous policy networks for thickness optimization.
"""

import os
import numpy as np
import torch
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt

from coatopt.algorithms.mcts.mcts import HybridMCTSAgent
from coatopt.algorithms.pc_hppo_oml import PCHPPO
from coatopt.environments.thermal_noise_environment_pareto import ParetoCoatingStack
from coatopt.config.structured_config import CoatingOptimisationConfig
from coatopt.utils.plotting import TrainingPlotManager


class MCTSTrainer:
    """
    MCTS-based trainer for multi-objective coating optimization.
    
    This trainer uses MCTS for systematic exploration of material sequences
    while training continuous policy networks for thickness optimization.
    During training, it samples different objective weights to build up the
    Pareto front through directed search and policy learning.
    """
    
    def __init__(self, config: CoatingOptimisationConfig, env: ParetoCoatingStack,
                 pc_hppo_agent: PCHPPO):
        """
        Initialize MCTS trainer.
        
        Args:
            config: Configuration object
            env: Pareto coating environment 
            pc_hppo_agent: PC-HPPO agent (can be randomly initialized)
        """
        self.config = config
        self.env = env
        self.pc_hppo_agent = pc_hppo_agent
        
        # MCTS configuration
        self.mcts_config = {
            'n_simulations': config.mcts.n_simulations if config.mcts else 1000,
            'c_puct': config.mcts.c_puct if config.mcts else 1.4,
            'use_policy_priors': config.mcts.use_policy_priors if config.mcts else True,
            'max_layers': config.data.n_layers,
            'available_materials': None  # Use all materials except air
        }
        
        # Initialize hybrid agent
        self.hybrid_agent = HybridMCTSAgent(pc_hppo_agent, self.mcts_config)
        self.hybrid_agent.initialize_mcts(env)
        
        # Training parameters
        self.n_iterations = config.training.n_iterations
        self.n_samples_per_iteration = config.training.n_episodes_per_update // 10  # Fewer samples since MCTS is slower
        self.eval_interval = config.mcts.eval_interval if config.mcts else 50
        
        # Policy training parameters
        self.policy_update_interval = 10  # Update policies every N iterations
        self.experience_buffer = []  # Store MCTS experiences for policy training
        self.max_buffer_size = 10000
        
        # Pareto front storage
        self.pareto_solutions = []
        self.training_history = {
            'iteration': [],
            'pareto_size': [],
            'hypervolume': [],
            'best_reflectivity': [],
            'best_thermal': [],
            'best_absorption': [],
            'search_times': [],
            'policy_losses': []
        }
        
        # Output directory
        self.output_dir = config.general.root_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Plot manager for training visualization
        self.plot_manager = TrainingPlotManager(self.output_dir)
        
        print(f"MCTS Trainer initialized (training from scratch):")
        print(f"  Simulations per search: {self.mcts_config['n_simulations']}")
        print(f"  Samples per iteration: {self.n_samples_per_iteration}")
        print(f"  Total training iterations: {self.n_iterations}")
        print(f"  Policy update interval: {self.policy_update_interval}")
        print(f"  Evaluation interval: {self.eval_interval}")
    
    def train(self):
        """
        Main training loop for MCTS Pareto optimization.
        
        At each iteration:
        1. Sample random objective weights
        2. Run MCTS optimization for each weight vector
        3. Update Pareto front with new solutions
        4. Evaluate progress and save checkpoints
        """
        print("Starting MCTS Pareto training...")
        
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            iteration_start = time.time()
            
            # Generate diverse objective weights for this iteration
            objective_weights = self._generate_diverse_weights(self.n_samples_per_iteration)
            
            # Run MCTS searches with different objective weights
            new_solutions = []
            search_times = []
            
            for i, weights in enumerate(objective_weights):
                search_start = time.time()
                
                # Run MCTS optimization
                materials, thicknesses, search_info = self.hybrid_agent.generate_full_sequence(
                    objective_weights=np.array(weights)
                )
                
                search_time = time.time() - search_start
                search_times.append(search_time)
                
                if materials and thicknesses:
                    # Evaluate the solution
                    solution = self._evaluate_solution(materials, thicknesses, weights)
                    if solution:
                        new_solutions.append(solution)
                
                if (i + 1) % 10 == 0:
                    print(f"  Iteration {iteration+1}/{self.n_iterations}: "
                          f"Completed {i+1}/{len(objective_weights)} searches")
            
            # Update Pareto front
            self._update_pareto_front(new_solutions)
            
            # Record training statistics
            iteration_time = time.time() - iteration_start
            self._record_iteration_stats(iteration, search_times, iteration_time)
            
        # Periodic evaluation and saving
        if (iteration + 1) % self.eval_interval == 0:
            # Update policies from collected experience
            if len(self.experience_buffer) > 100:
                policy_loss = self._update_policies_from_experience()
                self.training_history['policy_losses'].append(policy_loss)
            else:
                self.training_history['policy_losses'].append(0.0)
            
            self._periodic_evaluation(iteration + 1)
            self._save_checkpoint(iteration + 1)
        
        # Collect experiences from MCTS searches for policy learning
        if hasattr(self.hybrid_agent, 'get_recent_experiences'):
            experiences = self.hybrid_agent.get_recent_experiences()
            self._add_experiences_to_buffer(experiences)            # Progress update
            if (iteration + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration+1}/{self.n_iterations} complete. "
                      f"Pareto size: {len(self.pareto_solutions)}, "
                      f"Elapsed: {elapsed/3600:.2f}h")
        
        # Final evaluation and saving
        print("Training complete. Running final evaluation...")
        self._final_evaluation()
        self._save_final_results()
        
        total_time = time.time() - start_time
        print(f"MCTS training completed in {total_time/3600:.2f} hours")
        print(f"Final Pareto front size: {len(self.pareto_solutions)}")
        
    def _generate_diverse_weights(self, n_samples: int) -> List[List[float]]:
        """Generate diverse objective weight vectors for training."""
        weights = []
        
        # Mix of different sampling strategies for diversity
        n_random = n_samples // 2
        n_systematic = n_samples - n_random
        
        # Random Dirichlet sampling (encourages diversity)
        alpha = [0.5, 0.5, 0.5]  # Low alpha for more diversity
        random_weights = np.random.dirichlet(alpha, n_random)
        weights.extend(random_weights.tolist())
        
        # Systematic sampling on simplex edges and corners
        systematic_weights = []
        
        # Corner points (single objective focus)
        systematic_weights.extend([
            [1.0, 0.0, 0.0],  # Reflectivity only
            [0.0, 1.0, 0.0],  # Thermal only
            [0.0, 0.0, 1.0],  # Absorption only
        ])
        
        # Edge points (pairwise combinations)
        edge_samples = min(n_systematic - 3, 6)
        for i in range(edge_samples):
            if i < 2:
                # Reflectivity + thermal
                w = 0.1 + 0.8 * np.random.random()
                systematic_weights.append([w, 1-w, 0.0])
            elif i < 4:
                # Reflectivity + absorption  
                w = 0.1 + 0.8 * np.random.random()
                systematic_weights.append([w, 0.0, 1-w])
            else:
                # Thermal + absorption
                w = 0.1 + 0.8 * np.random.random()
                systematic_weights.append([0.0, w, 1-w])
        
        # Random interior points
        remaining = n_systematic - len(systematic_weights)
        for _ in range(remaining):
            w = np.random.uniform(0.1, 0.9, 3)
            w = w / np.sum(w)
            systematic_weights.append(w.tolist())
        
        weights.extend(systematic_weights)
        
        return weights[:n_samples]  # Ensure exact count
    
    def _evaluate_solution(self, materials: List[int], thicknesses: List[float], 
                          objective_weights: List[float]) -> Optional[Dict]:
        """Evaluate a coating solution and return metrics."""
        try:
            # Build coating state
            coating_state = self._build_coating_state(materials, thicknesses)
            
            # Evaluate using environment
            reflectivity, thermal_noise, absorption, total_thickness = self.env.compute_state_value(
                coating_state, return_separate=True
            )
            
            return {
                'materials': materials,
                'thicknesses': thicknesses,
                'objective_weights': objective_weights,
                'metrics': {
                    'reflectivity': float(reflectivity),
                    'thermal_noise': float(thermal_noise),
                    'absorption': float(absorption),
                    'total_thickness': float(total_thickness)
                },
                'coating_state': coating_state
            }
            
        except Exception as e:
            print(f"Warning: Solution evaluation failed: {e}")
            return None
    
    def _build_coating_state(self, materials: List[int], thicknesses: List[float]):
        """Build coating state from materials and thicknesses."""
        max_layers = self.env.max_layers
        n_materials = self.env.n_materials
        
        # Initialize with air
        state = np.zeros((max_layers, n_materials + 1))
        state[:, self.env.air_material_index + 1] = 1
        state[:, 0] = self.env.min_thickness
        
        # Fill actual layers
        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            if i < max_layers:
                state[i, :] = 0
                state[i, 0] = thickness
                state[i, material + 1] = 1
        
        return state
    
    def _update_pareto_front(self, new_solutions: List[Dict]):
        """Update Pareto front with new solutions."""
        # Add new solutions
        all_solutions = self.pareto_solutions + new_solutions
        
        # Find Pareto optimal solutions
        pareto_optimal = self._find_pareto_optimal_solutions(all_solutions)
        
        # Update Pareto front
        old_size = len(self.pareto_solutions)
        self.pareto_solutions = pareto_optimal
        new_size = len(self.pareto_solutions)
        
        print(f"    Pareto front updated: {old_size} -> {new_size} solutions "
              f"(+{len(new_solutions)} evaluated, {new_size - old_size} net gain)")
    
    def _find_pareto_optimal_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """Find Pareto optimal solutions (non-dominated set)."""
        if not solutions:
            return []
        
        pareto_optimal = []
        
        for i, sol_i in enumerate(solutions):
            metrics_i = sol_i['metrics']
            objectives_i = np.array([
                -metrics_i['reflectivity'],     # Maximize (convert to minimize)
                metrics_i['thermal_noise'],     # Minimize
                metrics_i['absorption']         # Minimize
            ])
            
            is_pareto_optimal = True
            
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                
                metrics_j = sol_j['metrics']
                objectives_j = np.array([
                    -metrics_j['reflectivity'],   # Maximize (convert to minimize)
                    metrics_j['thermal_noise'],   # Minimize
                    metrics_j['absorption']       # Minimize
                ])
                
                # Check if j dominates i
                dominates = all(objectives_j <= objectives_i) and any(objectives_j < objectives_i)
                
                if dominates:
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_optimal.append(sol_i)
        
        return pareto_optimal
    
    def _record_iteration_stats(self, iteration: int, search_times: List[float], 
                               iteration_time: float):
        """Record training statistics for this iteration."""
        self.training_history['iteration'].append(iteration)
        self.training_history['pareto_size'].append(len(self.pareto_solutions))
        self.training_history['search_times'].append(np.mean(search_times))
        
        if self.pareto_solutions:
            # Best metrics across Pareto front
            reflectivities = [s['metrics']['reflectivity'] for s in self.pareto_solutions]
            thermal_noises = [s['metrics']['thermal_noise'] for s in self.pareto_solutions]
            absorptions = [s['metrics']['absorption'] for s in self.pareto_solutions]
            
            self.training_history['best_reflectivity'].append(max(reflectivities))
            self.training_history['best_thermal'].append(min(thermal_noises))
            self.training_history['best_absorption'].append(min(absorptions))
            
            # Compute hypervolume (simplified)
            hypervolume = self._compute_hypervolume()
            self.training_history['hypervolume'].append(hypervolume)
        else:
            self.training_history['best_reflectivity'].append(0.0)
            self.training_history['best_thermal'].append(float('inf'))
            self.training_history['best_absorption'].append(float('inf'))
            self.training_history['hypervolume'].append(0.0)
    
    def _compute_hypervolume(self) -> float:
        """Compute simplified hypervolume indicator."""
        if not self.pareto_solutions:
            return 0.0
        
        # Reference point (nadir point)
        ref_point = np.array([0.0, 1e-15, 1.0])  # [min_refl, max_thermal, max_absorption]
        
        # Extract objectives (convert reflectivity to minimization)
        objectives = []
        for sol in self.pareto_solutions:
            m = sol['metrics']
            objectives.append([
                -m['reflectivity'],     # Convert to minimization
                m['thermal_noise'],
                m['absorption']
            ])
        
        objectives = np.array(objectives)
        
        # Simple hypervolume approximation
        # In practice, would use proper hypervolume computation
        dominated_volume = 0.0
        for obj in objectives:
            volume = np.prod(np.maximum(ref_point - obj, 0))
            dominated_volume += volume
        
        return dominated_volume
    
    def _periodic_evaluation(self, iteration: int):
        """Run periodic evaluation and create plots."""
        print(f"  Running evaluation at iteration {iteration}...")
        
        # Create training progress plots
        self._create_training_plots()
        
        # Save current Pareto front
        pareto_file = os.path.join(self.output_dir, f"pareto_front_iter_{iteration}.json")
        self._save_pareto_front(pareto_file)
        
        # Create Pareto front visualization
        self._create_pareto_plots(suffix=f"_iter_{iteration}")
    
    def _create_training_plots(self):
        """Create training progress plots."""
        if not self.training_history['iteration']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MCTS Training Progress')
        
        iterations = self.training_history['iteration']
        
        # Pareto front size
        axes[0, 0].plot(iterations, self.training_history['pareto_size'])
        axes[0, 0].set_title('Pareto Front Size')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Number of Solutions')
        axes[0, 0].grid(True)
        
        # Hypervolume
        axes[0, 1].plot(iterations, self.training_history['hypervolume'])
        axes[0, 1].set_title('Hypervolume Indicator')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Hypervolume')
        axes[0, 1].grid(True)
        
        # Best objectives
        axes[1, 0].plot(iterations, self.training_history['best_reflectivity'], label='Reflectivity')
        axes[1, 0].set_title('Best Reflectivity')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Reflectivity')
        axes[1, 0].grid(True)
        
        axes[1, 1].semilogy(iterations, self.training_history['best_thermal'], label='Thermal', color='red')
        axes[1, 1].set_title('Best Thermal Noise')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Thermal Noise')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mcts_training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pareto_plots(self, suffix: str = ""):
        """Create Pareto front visualization plots."""
        if not self.pareto_solutions:
            return
        
        # Extract metrics
        reflectivities = [s['metrics']['reflectivity'] for s in self.pareto_solutions]
        thermal_noises = [s['metrics']['thermal_noise'] for s in self.pareto_solutions]
        absorptions = [s['metrics']['absorption'] for s in self.pareto_solutions]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'MCTS Pareto Front{suffix}')
        
        # Reflectivity vs Thermal
        axes[0].scatter(reflectivities, thermal_noises, alpha=0.7, s=50)
        axes[0].set_xlabel('Reflectivity')
        axes[0].set_ylabel('Thermal Noise')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Reflectivity vs Absorption
        axes[1].scatter(reflectivities, absorptions, alpha=0.7, s=50, color='orange')
        axes[1].set_xlabel('Reflectivity')
        axes[1].set_ylabel('Absorption')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        # Thermal vs Absorption
        axes[2].scatter(thermal_noises, absorptions, alpha=0.7, s=50, color='green')
        axes[2].set_xlabel('Thermal Noise')
        axes[2].set_ylabel('Absorption')
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'mcts_pareto_front{suffix}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'pareto_solutions': self.pareto_solutions,
            'training_history': self.training_history,
            'mcts_config': self.mcts_config,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
        
        checkpoint_file = os.path.join(self.output_dir, f'mcts_checkpoint_{iteration}.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def _save_pareto_front(self, filename: str):
        """Save current Pareto front to file."""
        with open(filename, 'w') as f:
            json.dump(self.pareto_solutions, f, indent=2, default=str)
    
    def _final_evaluation(self):
        """Run comprehensive final evaluation."""
        print("Running final MCTS evaluation with diverse weights...")
        
        # Generate comprehensive evaluation weights
        eval_weights = self._generate_evaluation_weights(100)
        
        final_solutions = []
        for i, weights in enumerate(eval_weights):
            if i % 20 == 0:
                print(f"  Final evaluation: {i+1}/{len(eval_weights)}")
            
            materials, thicknesses, _ = self.hybrid_agent.generate_full_sequence(
                objective_weights=np.array(weights)
            )
            
            if materials and thicknesses:
                solution = self._evaluate_solution(materials, thicknesses, weights)
                if solution:
                    final_solutions.append(solution)
        
        # Update final Pareto front
        all_solutions = self.pareto_solutions + final_solutions
        self.pareto_solutions = self._find_pareto_optimal_solutions(all_solutions)
        
        print(f"Final evaluation complete. Final Pareto size: {len(self.pareto_solutions)}")
    
    def _generate_evaluation_weights(self, n_samples: int) -> List[List[float]]:
        """Generate comprehensive evaluation weight vectors."""
        weights = []
        
        # High-density systematic sampling for final evaluation
        n_dirichlet = n_samples // 2
        n_grid = n_samples - n_dirichlet
        
        # Dense Dirichlet sampling
        alpha = [1.0, 1.0, 1.0]  # More balanced for evaluation
        dirichlet_weights = np.random.dirichlet(alpha, n_dirichlet)
        weights.extend(dirichlet_weights.tolist())
        
        # Grid-based systematic sampling
        n_per_dim = int(np.ceil(n_grid**(1/2)))  # 2D grid sampling
        
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                w1 = 0.05 + 0.9 * i / (n_per_dim - 1) if n_per_dim > 1 else 0.5
                w2 = 0.05 + 0.9 * j / (n_per_dim - 1) if n_per_dim > 1 else 0.5
                w3 = 1.0 - w1 - w2
                
                if w3 > 0.05:  # Valid weight vector
                    weights.append([w1, w2, w3])
                    if len(weights) >= n_samples:
                        break
            if len(weights) >= n_samples:
                break
        
        return weights[:n_samples]
    
    def _save_final_results(self):
        """Save final results and create comprehensive plots."""
        # Save final Pareto front
        final_pareto_file = os.path.join(self.output_dir, 'final_pareto_front.json')
        self._save_pareto_front(final_pareto_file)
        
        # Save training history
        history_file = os.path.join(self.output_dir, 'mcts_training_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Create final plots
        self._create_training_plots()
        self._create_pareto_plots(suffix="_final")
        
        # Summary statistics
        self._create_summary_report()
    
    def _create_summary_report(self):
        """Create summary report of MCTS training."""
        if not self.pareto_solutions:
            return
        
        report_file = os.path.join(self.output_dir, 'mcts_summary_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("MCTS Training Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Training parameters
            f.write("Training Configuration:\n")
            f.write(f"  Total iterations: {self.n_iterations}\n")
            f.write(f"  MCTS simulations per search: {self.mcts_config['n_simulations']}\n")
            f.write(f"  Samples per iteration: {self.n_samples_per_iteration}\n")
            f.write(f"  C_PUCT parameter: {self.mcts_config['c_puct']}\n")
            f.write(f"  Use policy priors: {self.mcts_config['use_policy_priors']}\n\n")
            
            # Final results
            reflectivities = [s['metrics']['reflectivity'] for s in self.pareto_solutions]
            thermal_noises = [s['metrics']['thermal_noise'] for s in self.pareto_solutions]
            absorptions = [s['metrics']['absorption'] for s in self.pareto_solutions]
            
            f.write("Final Pareto Front Statistics:\n")
            f.write(f"  Number of Pareto-optimal solutions: {len(self.pareto_solutions)}\n")
            f.write(f"  Reflectivity range: {min(reflectivities):.6f} - {max(reflectivities):.6f}\n")
            f.write(f"  Thermal noise range: {min(thermal_noises):.2e} - {max(thermal_noises):.2e}\n")
            f.write(f"  Absorption range: {min(absorptions):.2e} - {max(absorptions):.2e}\n\n")
            
            # Best solutions
            best_refl_idx = np.argmax(reflectivities)
            best_thermal_idx = np.argmin(thermal_noises)
            best_absorption_idx = np.argmin(absorptions)
            
            f.write("Best Solutions:\n")
            f.write(f"  Best Reflectivity: {reflectivities[best_refl_idx]:.6f}\n")
            f.write(f"    Materials: {self.pareto_solutions[best_refl_idx]['materials']}\n")
            f.write(f"  Best Thermal Noise: {thermal_noises[best_thermal_idx]:.2e}\n")
            f.write(f"    Materials: {self.pareto_solutions[best_thermal_idx]['materials']}\n")
            f.write(f"  Best Absorption: {absorptions[best_absorption_idx]:.2e}\n")
            f.write(f"    Materials: {self.pareto_solutions[best_absorption_idx]['materials']}\n")
        
        print(f"Summary report saved to {report_file}")
    
    def evaluate(self, n_samples: int = 1000):
        """
        Evaluate the trained MCTS with many different objective weights.
        
        Args:
            n_samples: Number of evaluation samples
            
        Returns:
            Tuple of (sampled_states, results, sampled_weights)
        """
        print(f"Evaluating MCTS with {n_samples} diverse objective weights...")
        
        # Generate evaluation weights
        eval_weights = self._generate_evaluation_weights(n_samples)
        
        sampled_states = []
        results = []
        sampled_weights = []
        
        for i, weights in enumerate(eval_weights):
            if i % 100 == 0:
                print(f"  Evaluation progress: {i+1}/{n_samples}")
            
            # Run MCTS with these weights
            materials, thicknesses, search_info = self.hybrid_agent.generate_full_sequence(
                objective_weights=np.array(weights)
            )
            
            if materials and thicknesses:
                # Build coating state
                coating_state = self._build_coating_state(materials, thicknesses)
                
                # Evaluate metrics
                try:
                    reflectivity, thermal_noise, absorption, _ = self.env.compute_state_value(
                        coating_state, return_separate=True
                    )
                    
                    sampled_states.append(coating_state)
                except Exception as e:
                    print(f"Warning: Failed to evaluate random coating: {e}")
                    
        return sampled_states
    
    def _add_experiences_to_buffer(self, experiences: List[Dict]) -> None:
        """Add MCTS experiences to training buffer."""
        self.experience_buffer.extend(experiences)
        
        # Keep buffer size manageable
        if len(self.experience_buffer) > self.max_buffer_size:
            # Remove oldest experiences
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
    
    def _update_policies_from_experience(self) -> float:
        """
        Update PC-HPPO policies using experiences from MCTS searches.
        
        Returns:
            Average policy loss
        """
        if len(self.experience_buffer) < 50:
            return 0.0
        
        # Sample batch of experiences for training
        batch_size = min(256, len(self.experience_buffer))
        sampled_experiences = np.random.choice(
            self.experience_buffer, size=batch_size, replace=False
        ).tolist()
        
        try:
            # Convert experiences to training data format
            states, actions, rewards, next_states, dones = self._process_experiences_for_training(
                sampled_experiences
            )
            
            # Update PC-HPPO networks using the experience data
            # This would typically involve calling the PC-HPPO update method
            if hasattr(self.pc_hppo_agent, 'update_from_batch'):
                loss_info = self.pc_hppo_agent.update_from_batch(
                    states, actions, rewards, next_states, dones
                )
                return loss_info.get('policy_loss', 0.0)
            else:
                # Fallback: create training episodes and use standard update
                episodes = self._create_episodes_from_experiences(sampled_experiences)
                if episodes:
                    loss_info = self.pc_hppo_agent.update(episodes)
                    return loss_info.get('policy_loss', 0.0)
                    
        except Exception as e:
            print(f"Warning: Policy update failed: {e}")
            
        return 0.0
    
    def _process_experiences_for_training(self, experiences: List[Dict]) -> Tuple:
        """Convert MCTS experiences to training format."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for exp in experiences:
            if all(key in exp for key in ['state', 'action', 'reward', 'next_state', 'done']):
                states.append(exp['state'])
                actions.append(exp['action'])
                rewards.append(exp['reward'])
                next_states.append(exp['next_state'])
                dones.append(exp['done'])
        
        return (
            np.array(states) if states else np.array([]),
            np.array(actions) if actions else np.array([]),
            np.array(rewards) if rewards else np.array([]),
            np.array(next_states) if next_states else np.array([]),
            np.array(dones) if dones else np.array([])
        )
    
    def _create_episodes_from_experiences(self, experiences: List[Dict]) -> List:
        """Create episode format for PC-HPPO training."""
        # This would group experiences into episodes
        # Implementation depends on PC-HPPO's expected format
        episodes = []
        
        # Group experiences by episode_id if available
        episode_groups = {}
        for exp in experiences:
            episode_id = exp.get('episode_id', 0)
            if episode_id not in episode_groups:
                episode_groups[episode_id] = []
            episode_groups[episode_id].append(exp)
        
        # Convert to episode format
        for episode_id, episode_exps in episode_groups.items():
            if len(episode_exps) > 1:  # Need at least 2 steps for an episode
                episode_data = {
                    'states': [exp['state'] for exp in episode_exps],
                    'actions': [exp['action'] for exp in episode_exps],
                    'rewards': [exp['reward'] for exp in episode_exps],
                    'done': episode_exps[-1]['done']
                }
                episodes.append(episode_data)
        
        return episodes
        
        return sampled_states, results, sampled_weights
