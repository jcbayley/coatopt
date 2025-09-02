"""
Enhanced policy networks for layer editing environment.
Extends the existing policy networks to handle multi-component discrete actions.
"""
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .policy_networks import BaseNetwork, get_activation_function


class MultiComponentDiscretePolicy(BaseNetwork):
    """
    Multi-component discrete policy network for layer editing actions.
    
    Outputs separate categorical distributions for:
    - Material selection (one-hot)
    - Layer index selection 
    - Insert/replace token (binary)
    """
    
    def __init__(
        self,
        input_dim: int,
        action_components: Dict[str, int],  # {'material': n_materials, 'layer_index': max_layers, 'insert_replace': 2}
        hidden_dim: int,
        n_layers: int = 2,
        include_layer_number: bool = False,
        activation: str = "relu",
        n_objectives: int = 0,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        self.action_components = action_components
        
        # Define output structure - separate logits for each component
        output_dims = {f'{component}_logits': dim for component, dim in action_components.items()}
        
        super().__init__(
            input_dim, output_dims, hidden_dim, n_layers, include_layer_number, False,  # No material in input
            activation, n_objectives, use_hyper_networks, hyper_hidden_dim, hyper_n_layers
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        layer_number: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning separate probability distributions for each action component.
        
        Args:
            state: Input state tensor
            layer_number: Current layer number (optional)
            objective_weights: Multi-objective weights (optional)
            
        Returns:
            Dictionary with probability distributions for each action component
        """
        # Prepare input
        x = self._prepare_input(state, layer_number, None, objective_weights)
        
        # Forward pass
        if self.use_hyper_networks:
            raw_outputs = self._forward_hyper_network(x, objective_weights)
        else:
            raw_outputs = self._forward_standard_network(x)
        
        # Process outputs to probabilities
        probabilities = {}
        for component in self.action_components.keys():
            logits_key = f'{component}_logits'
            probabilities[component] = torch.softmax(raw_outputs[logits_key], dim=-1)
        
        return probabilities
    
    def sample_action(
        self, 
        state: torch.Tensor, 
        layer_number: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None,
        action_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Sample actions from the discrete policy distributions.
        
        Args:
            state: Input state tensor
            layer_number: Current layer number
            objective_weights: Multi-objective weights
            action_masks: Optional masks for invalid actions
            
        Returns:
            Tuple of (sampled_actions, log_probabilities)
        """
        # Get probability distributions
        probs = self.forward(state, layer_number, objective_weights)
        
        # Apply masks if provided
        if action_masks is not None:
            for component, mask in action_masks.items():
                if component in probs:
                    # Apply mask and renormalize
                    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=probs[component].device)
                    probs[component] = probs[component] * mask_tensor
                    probs[component] = probs[component] / (probs[component].sum(dim=-1, keepdim=True) + 1e-8)
        
        # Sample from each distribution
        sampled_actions = {}
        log_probs = {}
        
        for component, prob_dist in probs.items():
            # Create categorical distribution and sample
            categorical = torch.distributions.Categorical(prob_dist)
            action = categorical.sample()
            log_prob = categorical.log_prob(action)
            
            sampled_actions[component] = action
            log_probs[component] = log_prob
        
        return sampled_actions, log_probs
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        layer_number: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None,
        action_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Evaluate given actions under current policy.
        
        Args:
            state: Input state tensor
            actions: Dictionary of actions to evaluate
            layer_number: Current layer number
            objective_weights: Multi-objective weights
            action_masks: Optional masks for invalid actions
            
        Returns:
            Tuple of (log_probabilities, entropies)
        """
        # Get probability distributions
        probs = self.forward(state, layer_number, objective_weights)
        
        # Apply masks if provided
        if action_masks is not None:
            for component, mask in action_masks.items():
                if component in probs:
                    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=probs[component].device)
                    probs[component] = probs[component] * mask_tensor
                    probs[component] = probs[component] / (probs[component].sum(dim=-1, keepdim=True) + 1e-8)
        
        # Evaluate each action component
        log_probs = {}
        entropies = {}
        
        for component, prob_dist in probs.items():
            if component in actions:
                categorical = torch.distributions.Categorical(prob_dist)
                log_probs[component] = categorical.log_prob(actions[component])
                entropies[component] = categorical.entropy()
        
        return log_probs, entropies


class LayerEditingAgentWrapper:
    """
    Agent wrapper that adapts the PC-HPPO agent for layer editing tasks.
    
    Handles the interface between the multi-component action space and the 
    existing agent architecture.
    """
    
    def __init__(self, base_agent, environment):
        """
        Initialize layer editing agent wrapper.
        
        Args:
            base_agent: Existing PCHPPO agent instance
            environment: LayerEditingEnvironment instance
        """
        self.base_agent = base_agent
        self.environment = environment
        
        # Replace discrete policy with multi-component version
        self._setup_multi_component_discrete_policy()
    
    def _setup_multi_component_discrete_policy(self):
        """Replace the agent's discrete policy with multi-component version."""
        # Get original discrete policy parameters
        original_policy = self.base_agent.policy_discrete
        
        # Create new multi-component policy with same architecture
        self.multi_discrete_policy = MultiComponentDiscretePolicy(
            input_dim=original_policy.input_dim,
            action_components=self.environment.discrete_action_components,
            hidden_dim=original_policy.hidden_dim,
            n_layers=original_policy.n_layers,
            include_layer_number=original_policy.include_layer_number,
            activation="relu",  # Use same activation as original
            n_objectives=self.base_agent.num_objectives,
            use_hyper_networks=original_policy.use_hyper_networks if hasattr(original_policy, 'use_hyper_networks') else False
        )
        
        # Replace in base agent
        self.base_agent.policy_discrete = self.multi_discrete_policy
    
    def select_action(
        self, 
        state, 
        layer_number=None, 
        objective_weights=None
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """
        Select action using the enhanced action space.
        
        Args:
            state: Current environment state
            layer_number: Current layer number
            objective_weights: Multi-objective weights
            
        Returns:
            Tuple of (action_dict, continuous_log_prob, discrete_log_probs_dict)
        """
        # Get action masks from environment
        action_masks = self.environment.get_action_mask()
        
        # Sample continuous action (thickness only)
        continuous_action, continuous_log_prob = self._sample_continuous_action(
            state, layer_number, objective_weights
        )
        
        # Sample discrete actions (material, layer_index, insert_replace)
        discrete_actions, discrete_log_probs = self.multi_discrete_policy.sample_action(
            state, layer_number, objective_weights, action_masks
        )
        
        # Combine into action dictionary
        action_dict = {
            'thickness': continuous_action.item(),
            'material': discrete_actions['material'].item(),
            'layer_index': discrete_actions['layer_index'].item(),
            'insert_replace': discrete_actions['insert_replace'].item()
        }
        
        return action_dict, continuous_log_prob, discrete_log_probs
    
    def _sample_continuous_action(
        self, 
        state, 
        layer_number=None, 
        objective_weights=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample continuous action (thickness) using existing infrastructure."""
        # Use existing continuous policy from base agent
        # This is a simplified version - you may need to adapt based on actual agent interface
        continuous_probs = self.base_agent.policy_continuous(state, layer_number, objective_weights=objective_weights)
        
        # Sample from distribution (implementation depends on agent's continuous policy output format)
        if isinstance(continuous_probs, tuple):
            mean, log_std = continuous_probs
            std = torch.exp(log_std)
            normal_dist = torch.distributions.Normal(mean, std)
            action = normal_dist.sample()
            log_prob = normal_dist.log_prob(action)
        else:
            # Handle other output formats as needed
            action = continuous_probs
            log_prob = torch.zeros_like(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self, 
        state, 
        action_dict: Dict[str, Any], 
        layer_number=None, 
        objective_weights=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Evaluate given actions under current policy.
        
        Args:
            state: Input state
            action_dict: Actions to evaluate
            layer_number: Current layer number  
            objective_weights: Multi-objective weights
            
        Returns:
            Tuple of (continuous_log_prob, discrete_log_probs, entropies)
        """
        # Get action masks
        action_masks = self.environment.get_action_mask()
        
        # Evaluate continuous action
        continuous_log_prob = self._evaluate_continuous_action(
            state, action_dict['thickness'], layer_number, objective_weights
        )
        
        # Prepare discrete actions for evaluation
        discrete_actions = {
            'material': torch.tensor([action_dict['material']]),
            'layer_index': torch.tensor([action_dict['layer_index']]),
            'insert_replace': torch.tensor([action_dict['insert_replace']])
        }
        
        # Evaluate discrete actions
        discrete_log_probs, discrete_entropies = self.multi_discrete_policy.evaluate_actions(
            state, discrete_actions, layer_number, objective_weights, action_masks
        )
        
        return continuous_log_prob, discrete_log_probs, discrete_entropies
    
    def _evaluate_continuous_action(
        self, 
        state, 
        action_value, 
        layer_number=None, 
        objective_weights=None
    ) -> torch.Tensor:
        """Evaluate continuous action using existing infrastructure."""
        # This is a placeholder - implement based on actual continuous policy interface
        action_tensor = torch.tensor([action_value], dtype=torch.float32)
        
        # Use existing continuous policy evaluation
        # Implementation depends on base agent's continuous policy structure
        return torch.zeros_like(action_tensor)  # Placeholder
