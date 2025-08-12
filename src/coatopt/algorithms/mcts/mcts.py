"""
Material-based Monte Carlo Tree Search with Continuous Policy for Coating Optimization

This implementation uses MCTS for discrete material selection while leveraging 
the existing PC-HPPO continuous policy network for thickness optimization.
"""

import numpy as np
import torch
import math
import time
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import copy

from coatopt.algorithms.pc_hppo_oml import PCHPPO
from coatopt.utils.truncated_normal import TruncatedNormalDist


class MCTSNode:
    """
    Node in the MCTS tree representing a partial coating stack.
    Each node contains only the material sequence - thicknesses are optimized on-demand.
    """
    
    def __init__(self, material_sequence: List[int], parent=None, action_material: int = None):
        self.material_sequence = material_sequence.copy()  # List of material indices
        self.parent = parent
        self.action_material = action_material  # Material that led to this node
        self.children: Dict[int, 'MCTSNode'] = {}
        
        # MCTS statistics
        self.visits = 0
        self.value_sum = np.zeros(3)  # For multi-objective: [reflectivity, thermal_noise, absorption]
        self.prior_prob = 0.0  # If using policy priors
        
        # Cached evaluation results
        self._cached_thickness = None
        self._cached_value = None
        self._cached_state = None
        
    def is_fully_expanded(self, available_materials: List[int]) -> bool:
        """Check if all valid materials have been tried."""
        return len(self.children) == len(available_materials)
    
    def is_terminal(self, max_layers: int) -> bool:
        """Check if this node represents a complete coating stack."""
        return len(self.material_sequence) >= max_layers
    
    def get_ucb_score(self, material: int, c_puct: float = 1.0, objective_weights: np.ndarray = None) -> float:
        """Calculate UCB score for a child material."""
        if material not in self.children:
            return float('inf')  # Unvisited children have highest priority
        
        child = self.children[material]
        if child.visits == 0:
            return float('inf')
        
        # Multi-objective Q-value with weights
        if objective_weights is None:
            objective_weights = np.ones(3) / 3  # Equal weighting
        
        q_value = np.dot(child.value_sum / child.visits, objective_weights)
        
        # UCB exploration term
        exploration = c_puct * math.sqrt(math.log(self.visits) / child.visits)
        
        return q_value + exploration
    
    def select_best_child(self, objective_weights: np.ndarray = None) -> 'MCTSNode':
        """Select child with highest average value."""
        if not self.children:
            return None
        
        if objective_weights is None:
            objective_weights = np.ones(3) / 3
        
        best_child = None
        best_value = -float('inf')
        
        for child in self.children.values():
            if child.visits > 0:
                value = np.dot(child.value_sum / child.visits, objective_weights)
                if value > best_value:
                    best_value = value
                    best_child = child
        
        return best_child


class MaterialMCTS:
    """
    Monte Carlo Tree Search for material selection with continuous thickness optimization.
    
    Uses MCTS to explore material sequences while leveraging PC-HPPO's continuous 
    policy network to optimize layer thicknesses.
    """
    
    def __init__(self, 
                 env,
                 continuous_policy_net,
                 value_net=None,
                 n_simulations: int = 1000,
                 c_puct: float = 1.0,
                 max_layers: int = 8,
                 available_materials: List[int] = None,
                 use_policy_priors: bool = False,
                 discrete_policy_net = None):
        """
        Initialize Material MCTS.
        
        Args:
            env: Coating environment
            continuous_policy_net: Trained continuous policy from PC-HPPO
            value_net: Optional value network for faster evaluation
            n_simulations: Number of MCTS simulations
            c_puct: UCB exploration constant
            max_layers: Maximum number of layers
            available_materials: List of available material indices
            use_policy_priors: Whether to use discrete policy for priors
            discrete_policy_net: Discrete policy network for priors
        """
        self.env = env
        self.continuous_policy = continuous_policy_net
        self.value_net = value_net
        self.discrete_policy = discrete_policy_net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_layers = max_layers
        self.use_policy_priors = use_policy_priors
        
        # Set available materials (exclude air if specified)
        if available_materials is None:
            self.available_materials = list(range(1, env.n_materials))  # Exclude air (index 0)
            if hasattr(env, 'ignore_air_option') and env.ignore_air_option:
                self.available_materials = [i for i in self.available_materials if i != env.air_material_index]
        else:
            self.available_materials = available_materials
            
        # Statistics
        self.search_stats = {
            'simulations': 0,
            'evaluations': 0,
            'cache_hits': 0,
            'search_time': 0.0
        }
    
    def search(self, initial_state=None, objective_weights: np.ndarray = None) -> Tuple[List[int], List[float], Dict]:
        """
        Perform MCTS search to find optimal material sequence and thicknesses.
        
        Args:
            initial_state: Starting coating state (if any)
            objective_weights: Weights for multi-objective optimization
            
        Returns:
            Tuple of (material_sequence, thickness_sequence, search_info)
        """
        start_time = time.time()
        
        if objective_weights is None:
            objective_weights = np.ones(3) / 3  # Equal weighting by default
        
        # Initialize root node
        initial_materials = []
        if initial_state is not None:
            initial_materials = self._extract_materials_from_state(initial_state)
        
        root = MCTSNode(initial_materials)
        
        # MCTS main loop
        for simulation in range(self.n_simulations):
            # 1. Selection: Walk down tree using UCB
            node = self._select(root, objective_weights)
            
            # 2. Expansion: Add new material if not terminal
            if not node.is_terminal(self.max_layers) and not node.is_fully_expanded(self.available_materials):
                node = self._expand(node, objective_weights)
            
            # 3. Evaluation: Get value using continuous policy + environment
            value = self._evaluate(node, objective_weights)
            
            # 4. Backpropagation: Update statistics up the tree
            self._backpropagate(node, value)
            
            self.search_stats['simulations'] += 1
        
        # Select best path
        best_path = self._extract_best_path(root, objective_weights)
        material_sequence, thickness_sequence = best_path
        
        # Gather search statistics
        search_info = {
            'tree_size': self._count_nodes(root),
            'max_depth': self._max_depth(root),
            'simulations': self.n_simulations,
            'search_time': time.time() - start_time,
            'evaluations': self.search_stats['evaluations'],
            'cache_hits': self.search_stats['cache_hits']
        }
        
        return material_sequence, thickness_sequence, search_info
    
    def _select(self, node: MCTSNode, objective_weights: np.ndarray) -> MCTSNode:
        """Selection phase: walk down tree using UCB until leaf."""
        while not node.is_terminal(self.max_layers) and node.children:
            # Find child with highest UCB score
            best_material = None
            best_score = -float('inf')
            
            for material in self.available_materials:
                if material in node.children:
                    score = node.get_ucb_score(material, self.c_puct, objective_weights)
                    if score > best_score:
                        best_score = score
                        best_material = material
            
            if best_material is None:
                break
                
            node = node.children[best_material]
        
        return node
    
    def _expand(self, node: MCTSNode, objective_weights: np.ndarray) -> MCTSNode:
        """Expansion phase: add a new child node."""
        if node.is_terminal(self.max_layers):
            return node
        
        # Find untried materials
        untried_materials = [m for m in self.available_materials if m not in node.children]
        
        if not untried_materials:
            return node
        
        # Select material to try
        if self.use_policy_priors and self.discrete_policy is not None:
            # Use policy network to guide expansion
            material = self._select_material_with_policy(node, untried_materials)
        else:
            # Random selection
            material = np.random.choice(untried_materials)
        
        # Create new child node
        new_material_sequence = node.material_sequence + [material]
        child = MCTSNode(new_material_sequence, parent=node, action_material=material)
        node.children[material] = child
        
        return child
    
    def _evaluate(self, node: MCTSNode, objective_weights: np.ndarray) -> np.ndarray:
        """Evaluation phase: get value of the node."""
        self.search_stats['evaluations'] += 1
        
        # Check cache first
        if node._cached_value is not None:
            self.search_stats['cache_hits'] += 1
            return node._cached_value
        
        # If empty sequence, return default low value
        if not node.material_sequence:
            return np.array([0.0, 0.0, 0.0])
        
        # Optimize thicknesses for this material sequence
        thicknesses = self._optimize_thicknesses(node.material_sequence)
        
        # Build complete coating state
        coating_state = self._build_coating_state(node.material_sequence, thicknesses)
        
        # Evaluate using environment
        try:
            if hasattr(self.env, 'compute_state_value'):
                reflectivity, thermal_noise, absorption, _ = self.env.compute_state_value(coating_state, return_separate=True)
                value = np.array([reflectivity, thermal_noise, absorption])
            else:
                # Fallback evaluation method
                value = self._evaluate_coating_simple(coating_state)
        except Exception as e:
            print(f"Evaluation error: {e}")
            value = np.array([0.0, 0.0, 0.0])
        
        # Cache results
        node._cached_value = value
        node._cached_thickness = thicknesses
        node._cached_state = coating_state
        
        return value
    
    def _optimize_thicknesses(self, material_sequence: List[int]) -> List[float]:
        """Use continuous policy to optimize thicknesses for given materials."""
        thicknesses = []
        
        # Build partial state for each layer and get optimal thickness
        for i, material in enumerate(material_sequence):
            # Create partial coating state up to this point
            partial_state = self._build_partial_state(material_sequence[:i], thicknesses, material)
            
            # Get thickness from continuous policy
            try:
                thickness = self._get_thickness_from_policy(partial_state, material, i)
                thicknesses.append(thickness)
            except Exception as e:
                print(f"Thickness optimization error: {e}")
                # Fallback to random thickness
                thickness = np.random.uniform(self.env.min_thickness, self.env.max_thickness)
                thicknesses.append(thickness)
        
        return thicknesses
    
    def _get_thickness_from_policy(self, state, material: int, layer_index: int) -> float:
        """Get optimal thickness from continuous policy network."""
        # Convert to torch tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0)
        
        # Prepare layer number if needed
        layer_number = None
        if hasattr(self.continuous_policy, 'include_layer_number') and self.continuous_policy.include_layer_number:
            layer_number = torch.tensor([layer_index]).unsqueeze(0)
        
        # Get material as one-hot if needed
        material_tensor = None
        if hasattr(self.continuous_policy, 'include_material_in_policy') and self.continuous_policy.include_material_in_policy:
            material_tensor = torch.tensor([material])
        
        # Forward pass through continuous policy
        with torch.no_grad():
            if hasattr(self.continuous_policy, 'select_action'):
                # Use the select_action method if available
                thickness_tensor = self.continuous_policy.select_action(state_tensor, layer_number, actiond=material_tensor)[2]
            else:
                # Direct forward pass through policy network
                if hasattr(self.continuous_policy, 'pre_network'):
                    pre_output = self.continuous_policy.pre_network(state_tensor, layer_number)
                    mean, std = self.continuous_policy.policy_continuous(pre_output, layer_number, material_tensor)
                    
                    # Sample from truncated normal
                    dist = TruncatedNormalDist(mean, std, self.continuous_policy.lower_bound, self.continuous_policy.upper_bound)
                    thickness_tensor = mean  # Use mean for deterministic policy
                else:
                    # Fallback
                    thickness_tensor = torch.tensor([0.5])  # Middle value
        
        return float(thickness_tensor.item())
    
    def _build_partial_state(self, materials: List[int], thicknesses: List[float], next_material: int):
        """Build partial coating state for thickness optimization."""
        # Create state representation similar to environment format
        max_layers = self.env.max_layers
        n_materials = self.env.n_materials
        
        # Initialize state with air
        state = np.zeros((max_layers, n_materials + 1))
        state[:, self.env.air_material_index + 1] = 1  # Fill with air initially
        state[:, 0] = self.env.min_thickness  # Default thickness
        
        # Fill in the layers we have so far
        for i, (material, thickness) in enumerate(zip(materials, thicknesses)):
            if i < max_layers:
                state[i, :] = 0  # Clear
                state[i, 0] = thickness
                state[i, material + 1] = 1  # +1 because thickness is at index 0
        
        return state
    
    def _build_coating_state(self, materials: List[int], thicknesses: List[float]):
        """Build complete coating state from materials and thicknesses."""
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
    
    def _backpropagate(self, node: MCTSNode, value: np.ndarray):
        """Backpropagation phase: update statistics up the tree."""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent
    
    def _extract_best_path(self, root: MCTSNode, objective_weights: np.ndarray) -> Tuple[List[int], List[float]]:
        """Extract best path from root to leaf."""
        materials = []
        thicknesses = []
        
        node = root
        while node.children:
            # Select child with highest average value
            child = node.select_best_child(objective_weights)
            if child is None:
                break
            
            materials.append(child.action_material)
            node = child
        
        # Get optimized thicknesses for this path
        if materials:
            thicknesses = self._optimize_thicknesses(materials)
        
        return materials, thicknesses
    
    def _extract_materials_from_state(self, state) -> List[int]:
        """Extract material sequence from environment state."""
        materials = []
        for layer in state:
            material_idx = np.argmax(layer[1:])  # Skip thickness, find material
            if material_idx != self.env.air_material_index:  # Skip air layers
                materials.append(material_idx)
        return materials
    
    def _select_material_with_policy(self, node: MCTSNode, untried_materials: List[int]) -> int:
        """Use discrete policy to select which material to try next."""
        # Build state for policy
        state = self._build_partial_state(node.material_sequence, [], 0)  # Dummy next material
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        
        # Get material probabilities from discrete policy
        with torch.no_grad():
            if hasattr(self.discrete_policy, 'pre_network'):
                pre_output = self.discrete_policy.pre_network(state_tensor, None)
                material_probs = self.discrete_policy.policy_discrete(pre_output, None)
            else:
                material_probs = self.discrete_policy(state_tensor)
        
        # Select from untried materials based on probabilities
        untried_probs = material_probs[0, untried_materials].numpy()
        untried_probs = untried_probs / np.sum(untried_probs)  # Normalize
        
        return np.random.choice(untried_materials, p=untried_probs)
    
    def _evaluate_coating_simple(self, coating_state) -> np.ndarray:
        """Simple fallback evaluation if full environment evaluation fails."""
        # Very basic evaluation - replace with proper optics calculations if needed
        return np.array([0.5, 0.5, 0.5])
    
    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
    
    def _max_depth(self, node: MCTSNode, depth: int = 0) -> int:
        """Find maximum depth in tree."""
        if not node.children:
            return depth
        return max(self._max_depth(child, depth + 1) for child in node.children.values())


class HybridMCTSAgent:
    """
    Agent that combines MCTS material selection with PC-HPPO continuous policies.
    Provides a similar interface to the existing PC-HPPO agent.
    """
    
    def __init__(self, pc_hppo_agent: PCHPPO, mcts_config: Dict = None):
        """
        Initialize hybrid agent.
        
        Args:
            pc_hppo_agent: PC-HPPO agent (can be randomly initialized for training from scratch)
            mcts_config: Configuration for MCTS (simulations, c_puct, etc.)
        """
        self.pc_hppo = pc_hppo_agent
        
        # Default MCTS configuration
        default_config = {
            'n_simulations': 1000,
            'c_puct': 1.0,
            'max_layers': 8,
            'use_policy_priors': True
        }
        self.mcts_config = {**default_config, **(mcts_config or {})}
        
        self.mcts = None  # Will be initialized when environment is provided
        
        # Experience collection for policy learning
        self.recent_experiences = []
        self.episode_id = 0
    
    def initialize_mcts(self, env):
        """Initialize MCTS with environment and policies."""
        self.mcts = MaterialMCTS(
            env=env,
            continuous_policy_net=self.pc_hppo,  # Use PC-HPPO agent directly
            value_net=self.pc_hppo.value if hasattr(self.pc_hppo, 'value') else None,
            discrete_policy_net=self.pc_hppo if self.mcts_config['use_policy_priors'] else None,
            **self.mcts_config
        )
        
    def get_recent_experiences(self) -> List[Dict]:
        """Get recent experiences for policy training."""
        experiences = self.recent_experiences.copy()
        self.recent_experiences.clear()  # Clear after retrieval
        return experiences
    
    def _record_experience(self, state, action, reward, next_state, done):
        """Record experience for policy learning."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'episode_id': self.episode_id
        }
        self.recent_experiences.append(experience)
        
        if done:
            self.episode_id += 1
    
    def select_action(self, state, objective_weights: np.ndarray = None):
        """
        Select action using MCTS + continuous policy.
        
        Args:
            state: Current environment state
            objective_weights: Multi-objective weights
            
        Returns:
            Selected action (material, thickness)
        """
        if self.mcts is None:
            raise ValueError("MCTS not initialized. Call initialize_mcts(env) first.")
        
        # Perform MCTS search
        materials, thicknesses, search_info = self.mcts.search(state, objective_weights)
        
        if materials and thicknesses:
            # Return first action in sequence
            return materials[0], thicknesses[0], search_info
        else:
            # Fallback to random action
            return self._random_action()
    
    def generate_full_sequence(self, initial_state=None, objective_weights: np.ndarray = None):
        """
        Generate complete coating sequence using MCTS.
        
        Returns:
            Complete material and thickness sequences
        """
        if self.mcts is None:
            raise ValueError("MCTS not initialized. Call initialize_mcts(env) first.")
        
        return self.mcts.search(initial_state, objective_weights)
    
    def _random_action(self):
        """Fallback random action."""
        material = np.random.choice(self.mcts.available_materials)
        thickness = np.random.uniform(self.mcts.env.min_thickness, self.mcts.env.max_thickness)
        return material, thickness, {'type': 'random_fallback'}
    
    def load_networks(self, path):
        """Load PC-HPPO networks."""
        self.pc_hppo.load_networks(path)
    
    def save_networks(self, path):
        """Save PC-HPPO networks."""
        self.pc_hppo.save_networks(path)


# Example usage and testing functions
def test_material_mcts():
    """Test function to demonstrate usage."""
    print("Material MCTS test would go here")
    # This would require actual environment and trained networks
    pass


if __name__ == "__main__":
    test_material_mcts()