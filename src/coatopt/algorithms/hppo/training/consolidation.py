"""
Consolidation strategy for multi-objective training.
Provides solution storage and behavior cloning to prevent catastrophic forgetting.
"""
import random
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ConsolidationConfig:
    """Configuration for solution consolidation."""
    bc_weight: float = 0.05  # Weight for behavior cloning loss
    max_solutions_per_objective: int = 100  # Max stored solutions per objective
    consolidation_interval: int = 50  # Episodes between consolidation updates
    min_solutions_for_consolidation: int = 10  # Min solutions needed before consolidation
    samples_per_objective: int = 5  # Number of samples per objective for consolidation
    percentile_threshold: float = 75.0  # Percentile threshold for "good" solutions


class ConsolidationStrategy:
    """
    Strategy class for multi-objective solution consolidation.
    
    Provides functionality to:
    - Store good solutions for each objective
    - Perform periodic behavior cloning to consolidate knowledge
    - Prevent catastrophic forgetting when switching between objectives
    """
    
    def __init__(self, config: ConsolidationConfig, agent):
        """Initialize consolidation strategy."""
        self.config = config
        self.agent = agent
        self.good_solutions = {}
        self.consolidation_losses = []
        self._initialize_solution_storage()
    
    def _initialize_solution_storage(self) -> None:
        """Initialize storage for good solutions per objective."""
        # Default objectives for coating optimization
        objective_names = ['reflectivity', 'thermal_noise', 'absorption']
        self.good_solutions = {obj_name: [] for obj_name in objective_names}
    
    def store_good_solution(self, state, action, rewards: Dict[str, Any], objective_weights: np.ndarray):
        """
        Store solutions that are good for specific objectives.
        
        Args:
            state: Environment state
            action: Action taken (discrete, continuous)
            rewards: Dictionary of rewards by objective
            objective_weights: Current objective weights
        """
        if len(objective_weights) == 0:
            return
            
        # Determine dominant objective
        dominant_obj_idx = np.argmax(objective_weights)
        objective_names = list(self.good_solutions.keys())
        
        if dominant_obj_idx >= len(objective_names):
            return
            
        dominant_obj_name = objective_names[dominant_obj_idx]
        
        # Only store if it's actually good for that objective
        current_solutions = self.good_solutions[dominant_obj_name]
        
        if len(current_solutions) == 0:
            # First solution for this objective
            should_store = True
        else:
            # Check if reward is above percentile threshold
            current_rewards = [sol['reward'] for sol in current_solutions]
            threshold = np.percentile(current_rewards, self.config.percentile_threshold)
            should_store = rewards.get(dominant_obj_name, float('-inf')) > threshold
            
        if should_store:
            solution_data = {
                'state': state.copy() if hasattr(state, 'copy') else state,
                'action': action,  # Tuple of (discrete_action, continuous_action)
                'weights': objective_weights.copy(),
                'reward': rewards.get(dominant_obj_name, 0.0),
                'full_rewards': rewards.copy() if hasattr(rewards, 'copy') else rewards
            }
            
            self.good_solutions[dominant_obj_name].append(solution_data)
            
            # Keep only best N solutions per objective
            max_solutions = self.config.max_solutions_per_objective
            if len(self.good_solutions[dominant_obj_name]) > max_solutions:
                # Sort by reward (descending) and keep top N
                self.good_solutions[dominant_obj_name].sort(
                    key=lambda x: x['reward'], reverse=True
                )
                self.good_solutions[dominant_obj_name] = self.good_solutions[dominant_obj_name][:max_solutions]
    
    def store_good_solution(self, state: np.ndarray, action: Tuple, rewards: Dict[str, float], 
                          objective_weights: np.ndarray) -> None:
        """
        Store solutions that are good for specific objectives.
        
        Args:
            state: Environment state
            action: Action taken (discrete, continuous)
            rewards: Dictionary of rewards by objective
            objective_weights: Current objective weights
        """
        if len(objective_weights) == 0:
            return
            
        # Determine dominant objective
        dominant_obj_idx = np.argmax(objective_weights)
        objective_names = list(self.good_solutions.keys())
        
        if dominant_obj_idx >= len(objective_names):
            return
            
        dominant_obj_name = objective_names[dominant_obj_idx]
        
        # Only store if it's actually good for that objective
        current_solutions = self.good_solutions[dominant_obj_name]
        
        if len(current_solutions) == 0:
            # First solution for this objective
            should_store = True
        else:
            # Check if reward is above percentile threshold
            current_rewards = [sol['reward'] for sol in current_solutions]
            threshold = np.percentile(current_rewards, self.consolidation_config.percentile_threshold)
            should_store = rewards.get(dominant_obj_name, float('-inf')) > threshold
            
        if should_store:
            solution_data = {
                'state': state.copy(),
                'action': action,  # Tuple of (discrete_action, continuous_action)
                'weights': objective_weights.copy(),
                'reward': rewards.get(dominant_obj_name, 0.0),
                'full_rewards': rewards.copy()
            }
            
            self.good_solutions[dominant_obj_name].append(solution_data)
            
            # Keep only best N solutions per objective
            max_solutions = self.consolidation_config.max_solutions_per_objective
            if len(self.good_solutions[dominant_obj_name]) > max_solutions:
                # Sort by reward (descending) and keep top N
                self.good_solutions[dominant_obj_name].sort(
                    key=lambda x: x['reward'], reverse=True
                )
                self.good_solutions[dominant_obj_name] = self.good_solutions[dominant_obj_name][:max_solutions]

    def consolidation_update(self) -> float:
        """
        Perform behavior cloning update using stored good solutions.
        
        Returns:
            Behavior cloning loss value
        """
        # Sample from stored good solutions
        replay_batch = []
        for obj_name in self.good_solutions:
            solutions = self.good_solutions[obj_name]
            min_solutions = self.consolidation_config.min_solutions_for_consolidation
            
            if len(solutions) > min_solutions:
                # Sample solutions for this objective
                n_samples = min(
                    self.consolidation_config.samples_per_objective,
                    len(solutions)
                )
                samples = random.sample(solutions, n_samples)
                replay_batch.extend(samples)
                
        if len(replay_batch) == 0:
            return 0.0
            
        # Prepare batch data
        batch_states = []
        batch_weights = []
        batch_actions_discrete = []
        batch_actions_continuous = []
        
        for solution in replay_batch:
            batch_states.append(solution['state'])
            batch_weights.append(solution['weights'])
            
            # Unpack actions
            discrete_action, continuous_action = solution['action']
            batch_actions_discrete.append(discrete_action)
            batch_actions_continuous.append(continuous_action)
            
        # Convert to tensors
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float32)
        batch_weights = torch.tensor(np.array(batch_weights), dtype=torch.float32)
        batch_actions_discrete = torch.cat(batch_actions_discrete, dim=0) if batch_actions_discrete[0].dim() > 0 else torch.stack(batch_actions_discrete)
        batch_actions_continuous = torch.cat(batch_actions_continuous, dim=0) if batch_actions_continuous[0].dim() > 0 else torch.stack(batch_actions_continuous)
        
        # Get current policy predictions
        layer_numbers = False  # Adjust based on your agent configuration
        if hasattr(self.agent, 'include_layer_number') and self.agent.include_layer_number:
            # You may need to store layer numbers in solutions as well
            layer_numbers = torch.zeros(len(batch_states), 1, dtype=torch.float32)
            
        # Extract log probabilities for stored actions
        _, _, _, log_prob_discrete, log_prob_continuous, _, _, _, _, _, _ = self.agent.select_action(
            batch_states,
            layer_numbers, 
            batch_actions_continuous,
            batch_actions_discrete,
            packed=True,
            objective_weights=batch_weights
        )
        
        # Behavior cloning loss (negative log likelihood)
        bc_loss_discrete = -log_prob_discrete.mean()
        bc_loss_continuous = -log_prob_continuous.mean()
        bc_loss = bc_loss_discrete + bc_loss_continuous
        
        # Apply behavior cloning update (separate from PPO)
        # Update discrete policy
        self.agent.optimiser_discrete.zero_grad()
        (self.consolidation_config.bc_weight * bc_loss_discrete).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent.policy_discrete.parameters(), max_norm=0.5)
        self.agent.optimiser_discrete.step()
        
        # Update continuous policy  
        self.agent.optimiser_continuous.zero_grad()
        (self.consolidation_config.bc_weight * bc_loss_continuous).backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_continuous.parameters(), max_norm=0.5)
        self.agent.optimiser_continuous.step()
        
        return bc_loss.item()

    def get_consolidation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored solutions and consolidation.
        
        Returns:
            Dictionary with consolidation statistics
        """
        stats = {
            'solutions_per_objective': {
                obj_name: len(solutions) 
                for obj_name, solutions in self.good_solutions.items()
            },
            'total_solutions': sum(len(solutions) for solutions in self.good_solutions.values()),
            'consolidation_updates': len(self.consolidation_losses),
            'avg_consolidation_loss': np.mean(self.consolidation_losses) if self.consolidation_losses else 0.0
        }
        
        # Add reward statistics per objective
        for obj_name, solutions in self.good_solutions.items():
            if solutions:
                rewards = [sol['reward'] for sol in solutions]
                stats[f'{obj_name}_reward_stats'] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards)
                }
                
        return stats

    def print_consolidation_summary(self) -> None:
        """Print summary of consolidation statistics."""
        stats = self.get_consolidation_stats()
        
        print("\n" + "="*50)
        print("CONSOLIDATION SUMMARY")
        print("="*50)
        print(f"Total stored solutions: {stats['total_solutions']}")
        print(f"Consolidation updates performed: {stats['consolidation_updates']}")
        if stats['avg_consolidation_loss'] > 0:
            print(f"Average consolidation loss: {stats['avg_consolidation_loss']:.6f}")
            
        print("\nSolutions per objective:")
        for obj_name, count in stats['solutions_per_objective'].items():
            print(f"  {obj_name}: {count}")
            
            # Print reward stats if available
            reward_key = f'{obj_name}_reward_stats'
            if reward_key in stats:
                reward_stats = stats[reward_key]
                print(f"    Reward - Mean: {reward_stats['mean']:.4f}, "
                      f"Std: {reward_stats['std']:.4f}, "
                      f"Range: [{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]")
        print("="*50 + "\n")
    
    def process_episode(self, episode: int, episode_data: Dict[str, Any]):
        """
        Process episode data for consolidation.
        
        Args:
            episode: Episode number
            episode_data: Data from the episode
        """
        # Store good solutions if available
        if 'objectives' in episode_data and episode_data['objectives']:
            episode_trajectory = episode_data.get('episode_data', {})
            states = episode_trajectory.get('states', [])
            actions = episode_trajectory.get('actions', [])
            
            if states and actions:
                # Use the final state and objectives for storage
                final_state = states[-1] if states else None
                final_action = actions[-1] if actions else None
                objectives = episode_data['objectives']
                
                if final_state is not None and final_action is not None:
                    # Assume equal weights for now - this could be extracted from environment
                    num_objectives = len(objectives)
                    weights = np.ones(num_objectives) / num_objectives
                    
                    self.store_good_solution(final_state, final_action, objectives, weights)
    
    def update_consolidation(self, episode: int) -> Optional[float]:
        """
        Update consolidation if interval reached.
        
        Args:
            episode: Current episode number
            
        Returns:
            Consolidation loss if update performed, None otherwise
        """
        if episode % self.config.consolidation_interval == 0 and episode > 0:
            loss = self.consolidation_update()
            if loss > 0:
                self.consolidation_losses.append(loss)
            return loss
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """Get consolidation state for checkpointing."""
        return {
            'good_solutions': self.good_solutions,
            'consolidation_losses': self.consolidation_losses,
        }
        
    def load_state(self, state: Dict[str, Any]):
        """Load consolidation state from checkpoint."""
        self.good_solutions = state.get('good_solutions', {})
        self.consolidation_losses = state.get('consolidation_losses', [])


# Backward compatibility
ConsolidationMixin = ConsolidationStrategy
