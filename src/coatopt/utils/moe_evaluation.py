"""
Enhanced evaluation utilities for Mixture of Experts multi-objective optimization.

This module provides evaluation methods that leverage the MoE architecture to:
1. Generate comprehensive Pareto front coverage
2. Analyze expert specialization during evaluation
3. Create visualizations showing expert behavior
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from coatopt.algorithms.hppo.training.weight_cycling import sample_reward_weights


def evaluate_moe_pareto_coverage(
    agent, 
    env, 
    n_solutions: int = 1000,
    evaluation_strategy: str = "comprehensive",
    expert_analysis: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of MoE agent's Pareto front coverage.
    
    Args:
        agent: Trained MoE HPPO agent
        env: Environment for evaluation
        n_solutions: Number of solutions to evaluate
        evaluation_strategy: Strategy for weight sampling ('comprehensive', 'expert_focused', 'uniform')
        expert_analysis: Whether to track expert usage during evaluation
        
    Returns:
        Dictionary containing evaluation results and expert analysis
    """
    print(f"Evaluating MoE agent with {n_solutions} solutions using '{evaluation_strategy}' strategy...")
    
    # Load best trained networks
    agent.eval()
    
    results = {
        'states': [],
        'rewards': [],
        'objective_values': [],
        'objective_weights': [],
        'expert_usage': [] if expert_analysis else None,
        'gating_weights': [] if expert_analysis else None
    }
    
    # Generate weight combinations based on strategy
    if evaluation_strategy == "comprehensive":
        weights = _generate_comprehensive_weights(env.optimise_parameters, n_solutions)
    elif evaluation_strategy == "expert_focused":
        weights = _generate_expert_focused_weights(agent, env.optimise_parameters, n_solutions)
    elif evaluation_strategy == "uniform":
        weights = _generate_uniform_weights(env.optimise_parameters, n_solutions)
    else:
        raise ValueError(f"Unknown evaluation strategy: {evaluation_strategy}")
    
    print(f"Generated {len(weights)} weight combinations for evaluation")
    
    # Evaluate solutions
    for i, objective_weights in enumerate(weights):
        if i % 100 == 0:
            print(f"Evaluating solution {i+1}/{len(weights)}")
            
        # Reset environment
        state = env.reset()
        done = False
        
        while not done:
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            obj_weights_tensor = torch.FloatTensor(objective_weights).unsqueeze(0)
            
            # Get action from agent (this will use MoE if enabled)
            with torch.no_grad():
                action_output = agent.select_action(
                    state_tensor, 
                    objective_weights=obj_weights_tensor
                )
                
                # Extract auxiliary information if MoE is being used
                if len(action_output) > 11:  # MoE returns 12 items, standard returns 11
                    (action, actiond, actionc, log_prob_discrete, log_prob_continuous,
                     discrete_probs, continuous_means, continuous_std, state_value,
                     entropy_discrete, entropy_continuous, moe_aux_losses) = action_output
                     
                    if expert_analysis and moe_aux_losses:
                        # Track expert usage
                        if 'discrete_gate_weights' in moe_aux_losses:
                            results['gating_weights'].append({
                                'discrete': moe_aux_losses['discrete_gate_weights'].cpu().numpy(),
                                'continuous': moe_aux_losses.get('continuous_gate_weights', None),
                                'weights': objective_weights
                            })
                else:
                    # Standard agent without MoE
                    (action, actiond, actionc, log_prob_discrete, log_prob_continuous,
                     discrete_probs, continuous_means, continuous_std, state_value,
                     entropy_discrete, entropy_continuous) = action_output
            
            # Take step in environment
            state, reward, done, info = env.step(action.cpu().numpy())
        
        # Store final results
        results['states'].append(state)
        results['rewards'].append(reward)
        results['objective_values'].append(info.get('objective_values', {}))
        results['objective_weights'].append(objective_weights)
    
    # Compute Pareto front from results
    if len(results['objective_values']) > 0:
        # Extract objective values for Pareto computation
        obj_matrix = np.array([
            [obj_vals.get(param, 0.0) for param in env.optimise_parameters]
            for obj_vals in results['objective_values']
        ])
        
        # Compute Pareto front
        pareto_front = env.compute_pareto_front(obj_matrix)
        results['pareto_front'] = pareto_front
        results['pareto_indices'] = _find_pareto_indices(obj_matrix, pareto_front)
        
        print(f"Found {len(pareto_front)} points on Pareto front out of {len(results['states'])} evaluated")
    
    # Analyze expert specialization if MoE was used
    if expert_analysis and results['gating_weights']:
        results['expert_analysis'] = _analyze_expert_specialization(
            results['gating_weights'], 
            results['objective_weights'],
            env.optimise_parameters
        )
    
    return results


def _generate_comprehensive_weights(optimise_parameters: List[str], n_solutions: int) -> List[np.ndarray]:
    """Generate comprehensive weight combinations covering the entire weight space."""
    n_objectives = len(optimise_parameters)
    weights = []
    
    # Include corner points (single-objective focus)
    for i in range(n_objectives):
        corner = np.zeros(n_objectives)
        corner[i] = 1.0
        weights.append(corner)
    
    # Include center point
    center = np.ones(n_objectives) / n_objectives
    weights.append(center)
    
    # Add systematic grid sampling
    if n_objectives == 2:
        # For 2D, create uniform grid
        n_per_side = int(np.sqrt(n_solutions - len(weights)))
        for i in range(n_per_side):
            alpha = i / max(1, n_per_side - 1)
            weights.append(np.array([alpha, 1.0 - alpha]))
    else:
        # For higher dimensions, use Sobol sequence if available
        remaining = n_solutions - len(weights)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_objectives - 1, scramble=True)
            samples = sampler.random(remaining)
            for sample in samples:
                # Convert to simplex using stick-breaking
                simplex_weights = _stick_breaking_to_simplex(sample)
                weights.append(simplex_weights)
        except ImportError:
            # Fallback to Dirichlet sampling
            for _ in range(remaining):
                dirichlet_weights = np.random.dirichlet(np.ones(n_objectives))
                weights.append(dirichlet_weights)
    
    return weights[:n_solutions]


def _generate_expert_focused_weights(agent, optimise_parameters: List[str], n_solutions: int) -> List[np.ndarray]:
    """Generate weights focused on the expert specialization regions."""
    if not agent.use_mixture_of_experts:
        # Fallback to comprehensive for non-MoE agents
        return _generate_comprehensive_weights(optimise_parameters, n_solutions)
    
    n_objectives = len(optimise_parameters)
    weights = []
    
    # Get expert regions from one of the MoE networks (they should all be the same)
    expert_regions = agent.policy_discrete.expert_regions
    
    # Sample around each expert region
    samples_per_expert = n_solutions // len(expert_regions)
    remaining_samples = n_solutions % len(expert_regions)
    
    for i, expert_region in enumerate(expert_regions):
        n_samples = samples_per_expert + (1 if i < remaining_samples else 0)
        
        # Add the exact expert region
        weights.append(expert_region.numpy())
        
        # Add perturbations around the expert region
        for _ in range(n_samples - 1):
            # Add Gaussian noise around the expert region
            perturbed = expert_region.numpy() + np.random.normal(0, 0.1, n_objectives)
            # Project back to simplex
            perturbed = np.maximum(perturbed, 0)  # Ensure non-negative
            perturbed = perturbed / perturbed.sum()  # Normalize
            weights.append(perturbed)
    
    return weights


def _generate_uniform_weights(optimise_parameters: List[str], n_solutions: int) -> List[np.ndarray]:
    """Generate uniform random weights using Dirichlet distribution."""
    n_objectives = len(optimise_parameters)
    weights = []
    
    for _ in range(n_solutions):
        weight = np.random.dirichlet(np.ones(n_objectives))
        weights.append(weight)
    
    return weights


def _stick_breaking_to_simplex(uniform_sample: np.ndarray) -> np.ndarray:
    """Convert uniform sample to simplex using stick-breaking construction."""
    n = len(uniform_sample) + 1
    beta_samples = np.zeros(n)
    
    remaining = 1.0
    for i in range(n - 1):
        beta_samples[i] = uniform_sample[i] * remaining
        remaining *= (1.0 - uniform_sample[i])
    beta_samples[n - 1] = remaining
    
    return beta_samples


def _find_pareto_indices(obj_matrix: np.ndarray, pareto_front: np.ndarray) -> List[int]:
    """Find indices of Pareto front points in the original objective matrix."""
    pareto_indices = []
    for pf_point in pareto_front:
        # Find matching point in original matrix
        matches = np.where(np.allclose(obj_matrix, pf_point, rtol=1e-10, atol=1e-10))[0]
        if len(matches) > 0:
            pareto_indices.append(matches[0])
    return pareto_indices


def _analyze_expert_specialization(gating_weights: List[Dict], objective_weights: List[np.ndarray], 
                                 optimise_parameters: List[str]) -> Dict[str, Any]:
    """Analyze how experts specialize based on gating patterns."""
    analysis = {
        'expert_weight_preferences': {},
        'expert_usage_frequency': {},
        'specialization_clarity': 0.0
    }
    
    n_experts = len(gating_weights[0]['discrete'][0]) if gating_weights else 0
    if n_experts == 0:
        return analysis
    
    # Analyze expert preferences for different weight regions
    for expert_id in range(n_experts):
        expert_activations = []
        corresponding_weights = []
        
        for gating_info in gating_weights:
            discrete_gates = gating_info['discrete'][0]  # Assuming batch size 1
            expert_activations.append(discrete_gates[expert_id])
            corresponding_weights.append(gating_info['weights'])
        
        # Find weight regions where this expert is most active
        expert_activations = np.array(expert_activations)
        corresponding_weights = np.array(corresponding_weights)
        
        # Get top 20% activations for this expert
        top_activations = np.percentile(expert_activations, 80)
        active_indices = expert_activations >= top_activations
        
        if np.sum(active_indices) > 0:
            preferred_weights = corresponding_weights[active_indices]
            mean_preferred_weights = np.mean(preferred_weights, axis=0)
            
            analysis['expert_weight_preferences'][f'expert_{expert_id}'] = {
                'mean_weights': mean_preferred_weights,
                'weight_std': np.std(preferred_weights, axis=0),
                'n_activations': np.sum(active_indices),
                'max_activation': np.max(expert_activations)
            }
        
        # Calculate usage frequency
        analysis['expert_usage_frequency'][f'expert_{expert_id}'] = np.mean(expert_activations)
    
    # Calculate overall specialization clarity (how distinct are expert preferences)
    if len(analysis['expert_weight_preferences']) > 1:
        mean_weights = np.array([
            pref['mean_weights'] 
            for pref in analysis['expert_weight_preferences'].values()
        ])
        # Measure pairwise distances between expert preferences
        pairwise_distances = []
        for i in range(len(mean_weights)):
            for j in range(i + 1, len(mean_weights)):
                dist = np.linalg.norm(mean_weights[i] - mean_weights[j])
                pairwise_distances.append(dist)
        
        analysis['specialization_clarity'] = np.mean(pairwise_distances) if pairwise_distances else 0.0
    
    return analysis


def plot_moe_evaluation_results(results: Dict[str, Any], save_path: str = None, show_expert_analysis: bool = True):
    """
    Create comprehensive plots of MoE evaluation results.
    
    Args:
        results: Results from evaluate_moe_pareto_coverage
        save_path: Path to save plots (if None, displays interactively)
        show_expert_analysis: Whether to include expert analysis plots
    """
    n_objectives = len(results['objective_weights'][0])
    
    if n_objectives == 2:
        fig = plt.figure(figsize=(15, 5) if show_expert_analysis else (10, 5))
        
        # Plot 1: Pareto front
        ax1 = plt.subplot(1, 3 if show_expert_analysis else 2, 1)
        _plot_2d_pareto_results(ax1, results)
        
        # Plot 2: Objective weights coverage  
        ax2 = plt.subplot(1, 3 if show_expert_analysis else 2, 2)
        _plot_weight_coverage(ax2, results)
        
        # Plot 3: Expert analysis (if available and requested)
        if show_expert_analysis and 'expert_analysis' in results:
            ax3 = plt.subplot(1, 3, 3)
            _plot_expert_specialization(ax3, results['expert_analysis'])
    
    elif n_objectives == 3:
        fig = plt.figure(figsize=(15, 10))
        
        # 3D Pareto front plot
        ax1 = plt.subplot(2, 2, 1, projection='3d')
        _plot_3d_pareto_results(ax1, results)
        
        # Weight space coverage
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        _plot_3d_weight_coverage(ax2, results)
        
        # Expert analysis plots (if available)
        if show_expert_analysis and 'expert_analysis' in results:
            ax3 = plt.subplot(2, 2, 3)
            _plot_expert_specialization(ax3, results['expert_analysis'])
            
            ax4 = plt.subplot(2, 2, 4)
            _plot_expert_usage_frequency(ax4, results['expert_analysis'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {save_path}")
    else:
        plt.show()


def _plot_2d_pareto_results(ax, results):
    """Plot 2D Pareto front results."""
    # Extract objective values
    obj_vals = results['objective_values']
    param_names = list(obj_vals[0].keys()) if obj_vals else ['Obj1', 'Obj2']
    
    obj1_vals = [obj.get(param_names[0], 0) for obj in obj_vals]
    obj2_vals = [obj.get(param_names[1], 0) for obj in obj_vals]
    
    # Plot all solutions
    ax.scatter(obj1_vals, obj2_vals, alpha=0.6, s=20, color='lightblue', label='All Solutions')
    
    # Highlight Pareto front
    if 'pareto_indices' in results:
        pareto_obj1 = [obj1_vals[i] for i in results['pareto_indices']]
        pareto_obj2 = [obj2_vals[i] for i in results['pareto_indices']]
        ax.scatter(pareto_obj1, pareto_obj2, s=50, color='red', label='Pareto Front', zorder=5)
    
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title('Pareto Front Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_weight_coverage(ax, results):
    """Plot objective weight coverage."""
    weights = np.array(results['objective_weights'])
    ax.scatter(weights[:, 0], weights[:, 1], alpha=0.6, s=20)
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_title('Objective Weight Coverage')
    ax.grid(True, alpha=0.3)


def _plot_3d_pareto_results(ax, results):
    """Plot 3D Pareto front results."""
    obj_vals = results['objective_values']
    param_names = list(obj_vals[0].keys()) if obj_vals else ['Obj1', 'Obj2', 'Obj3']
    
    obj1_vals = [obj.get(param_names[0], 0) for obj in obj_vals]
    obj2_vals = [obj.get(param_names[1], 0) for obj in obj_vals]
    obj3_vals = [obj.get(param_names[2], 0) for obj in obj_vals]
    
    ax.scatter(obj1_vals, obj2_vals, obj3_vals, alpha=0.6, s=20, color='lightblue')
    
    if 'pareto_indices' in results:
        pareto_obj1 = [obj1_vals[i] for i in results['pareto_indices']]
        pareto_obj2 = [obj2_vals[i] for i in results['pareto_indices']]
        pareto_obj3 = [obj3_vals[i] for i in results['pareto_indices']]
        ax.scatter(pareto_obj1, pareto_obj2, pareto_obj3, s=50, color='red', label='Pareto Front')
    
    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1]) 
    ax.set_zlabel(param_names[2])
    ax.set_title('3D Pareto Front')


def _plot_3d_weight_coverage(ax, results):
    """Plot 3D weight coverage."""
    weights = np.array(results['objective_weights'])
    ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], alpha=0.6, s=20)
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Weight 3')
    ax.set_title('Weight Space Coverage')


def _plot_expert_specialization(ax, expert_analysis):
    """Plot expert specialization patterns."""
    if 'expert_weight_preferences' not in expert_analysis:
        ax.text(0.5, 0.5, 'No expert data available', ha='center', va='center')
        return
        
    expert_ids = []
    mean_weights = []
    
    for expert_id, prefs in expert_analysis['expert_weight_preferences'].items():
        expert_ids.append(expert_id)
        mean_weights.append(prefs['mean_weights'])
    
    if len(mean_weights) > 0:
        mean_weights = np.array(mean_weights)
        n_objectives = mean_weights.shape[1]
        
        # Create stacked bar chart showing each expert's weight preferences
        x = np.arange(len(expert_ids))
        bottom = np.zeros(len(expert_ids))
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_objectives))
        
        for obj_idx in range(n_objectives):
            ax.bar(x, mean_weights[:, obj_idx], bottom=bottom, 
                  label=f'Objective {obj_idx+1}', color=colors[obj_idx])
            bottom += mean_weights[:, obj_idx]
        
        ax.set_xlabel('Expert ID')
        ax.set_ylabel('Mean Weight Preference')
        ax.set_title('Expert Specialization Patterns')
        ax.set_xticks(x)
        ax.set_xticklabels(expert_ids, rotation=45)
        ax.legend()


def _plot_expert_usage_frequency(ax, expert_analysis):
    """Plot expert usage frequency."""
    if 'expert_usage_frequency' not in expert_analysis:
        ax.text(0.5, 0.5, 'No expert usage data available', ha='center', va='center')
        return
    
    experts = list(expert_analysis['expert_usage_frequency'].keys())
    frequencies = list(expert_analysis['expert_usage_frequency'].values())
    
    ax.bar(experts, frequencies)
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Average Usage Frequency')
    ax.set_title('Expert Usage During Evaluation')
    ax.tick_params(axis='x', rotation=45)
