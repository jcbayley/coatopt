"""
Replay buffer for storing and managing training experiences.
Extracted from pc_hppo_oml.py for better organization.
"""
from typing import List, Optional, Any
import torch
from torch.nn.utils.rnn import pad_sequence


class ReplayBuffer:
    """
    Buffer for storing training experiences in PPO algorithm.
    
    Stores states, actions, rewards, and other training data needed
    for policy and value function updates.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize replay buffer with optional size limit.
        
        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        
        # Main experience storage - use deque for efficient memory management
        from collections import deque
        self.discrete_actions: deque = deque(maxlen=max_size)
        self.continuous_actions: deque = deque(maxlen=max_size)
        self.states: deque = deque(maxlen=max_size)
        self.observations: deque = deque(maxlen=max_size)  # Pre-computed observation tensors
        self.logprobs_discrete: deque = deque(maxlen=max_size)
        self.logprobs_continuous: deque = deque(maxlen=max_size)
        self.rewards: deque = deque(maxlen=max_size)
        self.state_values: deque = deque(maxlen=max_size)
        self.dones: deque = deque(maxlen=max_size)
        self.entropy_discrete: deque = deque(maxlen=max_size)
        self.entropy_continuous: deque = deque(maxlen=max_size)
        self.returns: deque = deque(maxlen=max_size)
        self.layer_number: deque = deque(maxlen=max_size)
        self.hidden_state: deque = deque(maxlen=max_size)
        self.objective_weights: deque = deque(maxlen=max_size)

        # Remove unused temporary storage for cleaner memory footprint
    
    def clear(self) -> None:
        """Clear all stored experiences efficiently."""
        # Efficient clearing - deque.clear() is O(1) vs del [:] which is O(n)
        self.discrete_actions.clear()
        self.continuous_actions.clear()
        self.states.clear()
        self.observations.clear()
        self.logprobs_discrete.clear()
        self.logprobs_continuous.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()
        self.entropy_discrete.clear()
        self.entropy_continuous.clear()
        self.returns.clear()
        self.layer_number.clear()
        self.hidden_state.clear()
        self.objective_weights.clear()

    def update(
        self, 
        discrete_action: torch.Tensor, 
        continuous_action: torch.Tensor, 
        state: torch.Tensor, 
        observation: torch.Tensor,  # Pre-computed observation tensor
        logprob_discrete: torch.Tensor,
        logprob_continuous: torch.Tensor, 
        reward: float, 
        state_value: torch.Tensor, 
        done: bool,
        entropy_discrete: torch.Tensor,
        entropy_continuous: torch.Tensor,
        layer_number: int = 0,
        hidden_state: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None
    ) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            discrete_action: Discrete action taken
            continuous_action: Continuous action taken
            state: Environment state (CoatingState object)
            observation: Pre-computed observation tensor ready for networks
            logprob_discrete: Log probability of discrete action
            logprob_continuous: Log probability of continuous action
            reward: Reward received
            state_value: Estimated state value
            done: Whether episode is done
            entropy_discrete: Entropy of discrete action distribution
            entropy_continuous: Entropy of continuous action distribution
            layer_number: Current layer number
            hidden_state: Hidden state (if using RNN)
            objective_weights: Multi-objective weights
        """
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.states.append(state)
        self.observations.append(observation)
        self.logprobs_discrete.append(logprob_discrete)
        self.logprobs_continuous.append(logprob_continuous)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
        self.entropy_discrete.append(entropy_discrete)
        self.entropy_continuous.append(entropy_continuous)
        self.layer_number.append(layer_number)
        self.hidden_state.append(hidden_state)
        self.objective_weights.append(objective_weights)

    def update_returns(self, returns: List[float]) -> None:
        """
        Update the returns for stored experiences.
        
        Args:
            returns: List of discounted returns
        """
        self.returns.extend(returns)

    def pad_states(self) -> None:
        """Pad states to same length for batch processing."""
        self.states = pad_sequence(list(self.states), batch_first=True)
    
    def __len__(self) -> int:
        """Return number of stored experiences."""
        return len(self.discrete_actions)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.discrete_actions) == 0