"""
PC-HPPO (Proximal Constrained Hierarchical Proximal Policy optimisation) Agent.
Refactored from pc_hppo_oml.py for improved readability and maintainability.
"""
from typing import Union, Optional, Tuple, List
import numpy as np
import torch
from torch.nn import functional as F
from collections import deque
import os

from coatopt.networks.truncated_normal import TruncatedNormalDist
from coatopt.algorithms.pre_networks import PreNetworkLinear, PreNetworkLSTM, PreNetworkAttention
from coatopt.algorithms.replay_buffer import ReplayBuffer
from coatopt.algorithms.config import HPPOConstants
from coatopt.algorithms.action_utils import (
    prepare_state_input, prepare_layer_number, create_material_mask,
    pack_state_sequence, validate_probabilities, format_action_output
)


class PCHPPO:
    """
    Proximal Constrained Hierarchical Proximal Policy optimisation agent.
    
    Implements a multi-objective reinforcement learning agent with separate
    discrete and continuous action spaces for coating optimisation.
    """

    def __init__(
        self, 
        state_dim: Union[int, Tuple[int, ...]], 
        num_discrete: int, 
        num_cont: int, 
        hidden_size: int, 
        num_objectives: int = 3,
        disc_lr_policy: float = 1e-4, 
        cont_lr_policy: float = 1e-4, 
        lr_value: float = 2e-4, 
        lr_step: Union[int, List[int]] = 10,
        lr_min: float = 1e-6,
        T_mult: Union[int, List[int]] = 1,
        lower_bound: float = 0.1, 
        upper_bound: float = 1.0, 
        n_updates: int = 1, 
        beta: float = 0.1,
        clip_ratio: float = 0.5,
        gamma: float = 0.99,
        include_layer_number: bool = False,
        pre_type: str = "linear",
        n_heads: int = 2,
        n_pre_layers: int = 2,
        optimiser: str = "adam",
        n_discrete_layers: int = 2,
        n_continuous_layers: int = 2,
        n_value_layers: int = 2,
        value_hidden_size: int = 32,
        discrete_hidden_size: int = 32,
        continuous_hidden_size: int = 32,
        activation_function: str = "relu",
        include_material_in_policy: bool = False,
        substrate_material_index: int = 1,
        air_material_index: int = 0,
        ignore_air_option: bool = False,
        ignore_substrate_option: bool = False,
        beta_start: float = 1.0,
        beta_end: float = 0.001,
        beta_decay_length: int = 500,
        hyper_networks: bool = False
    ):
        """
        Initialize PC-HPPO agent.
        
        Args:
            state_dim: Dimensions of state space
            num_discrete: Number of discrete actions
            num_cont: Number of continuous actions
            hidden_size: Hidden layer size for networks
            num_objectives: Number of optimisation objectives
            disc_lr_policy: Learning rate for discrete policy
            cont_lr_policy: Learning rate for continuous policy
            lr_value: Learning rate for value function
            lr_step: Learning rate scheduler step size
            lr_min: Minimum learning rate
            T_mult: Learning rate scheduler multiplier
            lower_bound: Lower bound for continuous actions
            upper_bound: Upper bound for continuous actions
            n_updates: Number of policy updates per iteration
            beta: Entropy regularization coefficient
            clip_ratio: PPO clipping ratio
            gamma: Discount factor
            include_layer_number: Whether to include layer number in input
            pre_type: Type of pre-network ('linear', 'lstm', 'attn')
            n_heads: Number of attention heads (for attention pre-network)
            n_pre_layers: Number of pre-network layers
            optimiser: Optimizer type ('adam' or 'sgd')
            n_discrete_layers: Number of discrete policy layers
            n_continuous_layers: Number of continuous policy layers
            n_value_layers: Number of value function layers
            value_hidden_size: Hidden size for value network
            discrete_hidden_size: Hidden size for discrete policy
            continuous_hidden_size: Hidden size for continuous policy
            activation_function: Activation function type
            include_material_in_policy: Include material info in policy
            substrate_material_index: Index of substrate material
            air_material_index: Index of air material
            ignore_air_option: Whether to ignore air material
            ignore_substrate_option: Whether to ignore substrate material
            beta_start: Initial entropy coefficient
            beta_end: Final entropy coefficient
            beta_decay_length: Entropy decay length
            hyper_networks: Whether to use hypernetworks
        """
        # Import network classes based on hyper_networks flag
        if hyper_networks:
            from coatopt.algorithms.hyper_policy_nets import DiscretePolicy, ContinuousPolicy, Value
        else:
            from coatopt.algorithms.policy_nets import DiscretePolicy, ContinuousPolicy, Value

        # Store configuration
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.include_layer_number = include_layer_number
        self.substrate_material_index = substrate_material_index
        self.air_material_index = air_material_index
        self.ignore_air_option = ignore_air_option
        self.ignore_substrate_option = ignore_substrate_option
        self.num_objectives = num_objectives
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay_length = beta_decay_length
        self.pre_type = pre_type
        self.n_updates = n_updates

        # Initialize pre-network
        self.pre_output_dim = hidden_size
        self.pre_network = self._create_pre_network(
            state_dim, hidden_size, pre_type, n_heads, n_pre_layers, num_objectives
        )

        # Initialize policy and value networks (current and old versions)
        self._create_networks(
            DiscretePolicy, ContinuousPolicy, Value,
            num_discrete, num_cont, discrete_hidden_size, continuous_hidden_size,
            value_hidden_size, n_discrete_layers, n_continuous_layers, n_value_layers,
            lower_bound, upper_bound, include_layer_number, activation_function,
            include_material_in_policy
        )

        # Initialize optimizers and schedulers
        self._setup_optimizers(optimiser, disc_lr_policy, cont_lr_policy, lr_value)
        self._setup_schedulers(lr_step, T_mult, lr_min)

        # Initialize training components
        self.mse_loss = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        
        # Cache for expensive computations
        self._material_mask_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _create_pre_network(self, state_dim, hidden_size, pre_type, n_heads, n_pre_layers, num_objectives):
        """Create pre-network based on specified type."""
        if pre_type == "attn":
            return PreNetworkAttention(
                state_dim[-1], self.pre_output_dim, hidden_size,
                num_heads=n_heads, num_layers=n_pre_layers
            )
        elif pre_type == "lstm":
            return PreNetworkLSTM(
                state_dim[-1], self.pre_output_dim, hidden_size,
                include_layer_number=self.include_layer_number,
                n_layers=n_pre_layers, weight_dim=num_objectives
            )
        elif pre_type == "linear":
            return PreNetworkLinear(
                np.prod(state_dim), self.pre_output_dim, hidden_size,
                n_layers=n_pre_layers, include_layer_number=self.include_layer_number
            )
        else:
            raise ValueError(f"Unknown pre-network type: {pre_type}")

    def _create_networks(self, DiscretePolicy, ContinuousPolicy, Value, num_discrete, num_cont,
                        discrete_hidden_size, continuous_hidden_size, value_hidden_size,
                        n_discrete_layers, n_continuous_layers, n_value_layers,
                        lower_bound, upper_bound, include_layer_number, activation_function,
                        include_material_in_policy):
        """Create policy and value networks (current and old versions)."""
        for i in range(2):
            suffix = "" if i == 0 else "_old"
            
            # Discrete policy network
            setattr(self, f"policy_discrete{suffix}", DiscretePolicy(
                self.pre_output_dim, num_discrete, discrete_hidden_size,
                n_layers=n_discrete_layers, lower_bound=lower_bound, upper_bound=upper_bound,
                include_layer_number=include_layer_number, activation=activation_function,
                n_objectives=self.num_objectives
            ))
            
            # Continuous policy network
            setattr(self, f"policy_continuous{suffix}", ContinuousPolicy(
                self.pre_output_dim, num_cont, continuous_hidden_size,
                n_layers=n_continuous_layers, lower_bound=lower_bound, upper_bound=upper_bound,
                include_layer_number=include_layer_number, include_material=include_material_in_policy,
                activation=activation_function, n_objectives=self.num_objectives
            ))
            
            # Value network
            setattr(self, f"value{suffix}", Value(
                self.pre_output_dim, value_hidden_size, n_layers=n_value_layers,
                lower_bound=lower_bound, upper_bound=upper_bound,
                include_layer_number=include_layer_number, activation=activation_function,
                n_objectives=self.num_objectives
            ))

        # Copy parameters to old networks
        self.policy_discrete_old.load_state_dict(self.policy_discrete.state_dict())
        self.policy_continuous_old.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def _setup_optimizers(self, optimiser, disc_lr_policy, cont_lr_policy, lr_value):
        """Setup optimizers for networks."""
        self.lr_value = lr_value
        self.disc_lr_policy = disc_lr_policy
        self.cont_lr_policy = cont_lr_policy

        if optimiser == "adam":
            self.optimiser_discrete = torch.optim.Adam(self.policy_discrete.parameters(), lr=disc_lr_policy)
            self.optimiser_continuous = torch.optim.Adam(self.policy_continuous.parameters(), lr=cont_lr_policy)
            self.optimiser_value = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        elif optimiser == "sgd":
            self.optimiser_discrete = torch.optim.SGD(self.policy_discrete.parameters(), lr=disc_lr_policy)
            self.optimiser_continuous = torch.optim.SGD(self.policy_continuous.parameters(), lr=cont_lr_policy)
            self.optimiser_value = torch.optim.SGD(self.value.parameters(), lr=lr_value)
        else:
            raise ValueError(f"Unsupported optimizer: {optimiser}")

    def _setup_schedulers(self, lr_step, T_mult, lr_min):
        """Setup learning rate schedulers."""
        self.lr_step = lr_step
        
        # Handle different lr_step formats
        if isinstance(lr_step, (list, tuple)):
            lr_step_discrete, lr_step_continuous, lr_step_value = lr_step
        else:
            lr_step_discrete = lr_step_continuous = lr_step_value = lr_step

        # Handle different T_mult formats  
        if isinstance(T_mult, (list, tuple)):
            T_mult_discrete, T_mult_continuous, T_mult_value = T_mult
        else:
            T_mult_discrete = T_mult_continuous = T_mult_value = T_mult

        # Create schedulers
        self.scheduler_discrete = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser_discrete, T_0=lr_step_discrete, T_mult=T_mult_discrete, eta_min=lr_min
        )
        self.scheduler_continuous = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser_continuous, T_0=lr_step_continuous, T_mult=T_mult_continuous, eta_min=lr_min
        )
        self.scheduler_value = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser_value, T_0=lr_step_value, T_mult=T_mult_value, eta_min=lr_min
        )

    def get_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calculate discounted returns from rewards.
        
        Args:
            rewards: List of rewards
            
        Returns:
            Array of discounted returns
        """
        temp_r = deque()
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            temp_r.appendleft(R)
        return np.array(temp_r)

    def scheduler_step(self, step: int = 0, make_step: bool = True) -> Tuple[List[float], List[float], List[float], float]:
        """
        Step learning rate schedulers and calculate entropy coefficient.
        
        Args:
            step: Current step number
            make_step: Whether to actually step the schedulers
            
        Returns:
            Tuple of (discrete_lr, continuous_lr, value_lr, entropy_coeff)
        """
        if make_step:
            self.scheduler_discrete.step(step)
            self.scheduler_continuous.step(step)
            self.scheduler_value.step(step)

        # Calculate entropy coefficient based on current learning rate
        entropy_val = self.beta_start * self.scheduler_value.get_last_lr()[0] / self.lr_value
        
        return (
            self.scheduler_discrete.get_last_lr(),
            self.scheduler_continuous.get_last_lr(),
            self.scheduler_value.get_last_lr(),
            entropy_val
        )

    def select_action(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        layer_number: Optional[Union[np.ndarray, torch.Tensor]] = None,
        actionc: Optional[torch.Tensor] = None,
        actiond: Optional[torch.Tensor] = None,
        packed: bool = False,
        objective_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action based on current state and layer number.
        
        Args:
            state: Current environment state
            layer_number: Current layer number
            actionc: Continuous action to evaluate (if None, sample new)
            actiond: Discrete action to evaluate (if None, sample new)  
            packed: Whether state is already packed
            objective_weights: Multi-objective optimisation weights
            
        Returns:
            Tuple containing action components, probabilities, and values
        """
        # Prepare inputs
        state_tensor = prepare_state_input(state, self.pre_type)
        layer_number, layer_numbers_validated = prepare_layer_number(layer_number)
        
        # Pack state sequence only for LSTM processing
        if self.pre_type == "lstm":
            state_input = pack_state_sequence(state_tensor, layer_numbers_validated)
            use_packed = True
        else:
            state_input = state_tensor
            use_packed = False
        
        # Get pre-network outputs (optimized based on gradient needs)
        is_training = actionc is not None or actiond is not None  # If actions provided, likely training
        pre_output_d, pre_output_c, pre_output_v = self._get_pre_network_outputs(
            state_input, layer_number, objective_weights, needs_gradients=is_training, packed=use_packed
        )
        
        # Get discrete action probabilities
        discrete_probs = self.policy_discrete(pre_output_d, layer_number, objective_weights=objective_weights)
        validate_probabilities(discrete_probs, "discrete_probs")
        
        # Apply material constraints
        masked_discrete_probs = self._apply_material_constraints(
            discrete_probs, state_tensor, layer_number
        )
        
        # Sample discrete action
        discrete_dist = torch.distributions.Categorical(masked_discrete_probs)
        if actiond is None:
            actiond = discrete_dist.sample()
        
        # Get continuous action parameters and sample
        continuous_means, continuous_std = self.policy_continuous(
            pre_output_c, layer_number, actiond.unsqueeze(1), objective_weights=objective_weights
        )
        
        continuous_dist = TruncatedNormalDist(
            continuous_means, continuous_std, self.lower_bound, self.upper_bound
        )
        
        if actionc is None:
            actionc = continuous_dist.sample()
        
        actionc = torch.clamp(actionc, self.lower_bound, self.upper_bound)
        
        # Calculate log probabilities and entropy
        log_prob_discrete = discrete_dist.log_prob(actiond)
        log_prob_continuous = torch.sum(continuous_dist.log_prob(actionc), dim=-1)
        entropy_discrete = discrete_dist.entropy()
        entropy_continuous = torch.sum(continuous_dist._entropy, dim=-1)
        
        # Get state value
        state_value = self.value(pre_output_v, layer_number, objective_weights=objective_weights)
        
        # Format output action
        action = format_action_output(actiond, actionc)
        
        return (
            action, actiond, actionc, log_prob_discrete, log_prob_continuous,
            discrete_probs, continuous_means, continuous_std, state_value,
            entropy_discrete, entropy_continuous
        )

    def _get_pre_network_outputs(self, state_input, layer_number, objective_weights, needs_gradients=True, packed=False):
        """
        Get pre-network outputs for different network heads.
        
        Args:
            state_input: State input (packed sequence for LSTM, regular tensor for others)
            layer_number: Layer number tensor
            objective_weights: Multi-objective weights
            needs_gradients: Whether gradients are needed (training vs inference)
            packed: Whether the input is a packed sequence
            
        Returns:
            Tuple of (discrete_output, continuous_output, value_output)
        """
        if needs_gradients:
            # During training: separate forward passes for gradient computation
            if self.pre_type == "lstm":
                pre_output_d = self.pre_network(state_input, layer_number, packed=packed, weights=objective_weights)
                pre_output_c = self.pre_network(state_input, layer_number, packed=packed, weights=objective_weights)
                pre_output_v = self.pre_network(state_input, layer_number, packed=packed, weights=objective_weights)
            else:
                # For non-LSTM networks, don't pass packed or weights parameters
                pre_output_d = self.pre_network(state_input, layer_number)
                pre_output_c = self.pre_network(state_input, layer_number)
                pre_output_v = self.pre_network(state_input, layer_number)
            return pre_output_d, pre_output_c, pre_output_v
        else:
            # During inference: single forward pass, detach for efficiency
            with torch.no_grad():
                if self.pre_type == "lstm":
                    pre_output = self.pre_network(state_input, layer_number, packed=packed, weights=objective_weights)
                else:
                    # For non-LSTM networks, don't pass packed or weights parameters
                    pre_output = self.pre_network(state_input, layer_number)
                return pre_output.detach(), pre_output.detach(), pre_output.detach()

    def _apply_material_constraints(self, discrete_probs, state_tensor, layer_number):
        """Apply material selection constraints with caching for better performance."""
        if layer_number is None:
            return discrete_probs
            
        # Try to use cached mask for common layer configurations
        cache_key = self._get_material_mask_cache_key(state_tensor, layer_number)
        
        if cache_key in self._material_mask_cache:
            cached_mask = self._material_mask_cache[cache_key]
            # Verify cached mask has same shape as current discrete_probs
            if cached_mask.shape == discrete_probs.shape:
                self._cache_hits += 1
                return discrete_probs * cached_mask
            else:
                # Remove invalid cached entry
                del self._material_mask_cache[cache_key]
        
        # If no valid cache entry, compute mask
        if cache_key not in self._material_mask_cache:
            self._cache_misses += 1
        masked_probs = create_material_mask(
            discrete_probs, state_tensor, layer_number,
            self.substrate_material_index, self.air_material_index,
            self.ignore_air_option, self.ignore_substrate_option
        )
        
        # Cache the mask for future use (limit cache size to prevent memory growth)
        if cache_key is not None and len(self._material_mask_cache) < 1000:
            mask = masked_probs / (discrete_probs + 1e-8)  # Extract mask
            self._material_mask_cache[cache_key] = mask.detach()
        
        return masked_probs

    def _get_material_mask_cache_key(self, state_tensor, layer_number):
        """Create cache key for material mask based on previous materials used."""
        if isinstance(layer_number, torch.Tensor):
            layer_num = layer_number.item() if layer_number.numel() == 1 else tuple(layer_number.cpu().numpy())
        else:
            layer_num = layer_number
            
        # Create key based on layer number and previous material (simplified for common cases)
        if hasattr(layer_num, '__iter__'):
            # Batch case - don't cache for now to keep it simple
            return None
        else:
            # Single layer case
            if layer_num > 0 and layer_num < state_tensor.size(1):
                prev_material = torch.argmax(state_tensor[0, layer_num - 1, 1:]).item()
                return (layer_num, prev_material)
            else:
                return (layer_num, self.substrate_material_index)

    def get_cache_stats(self):
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._material_mask_cache)
        }

    def update(self, update_policy: bool = True, update_value: bool = True) -> Tuple[float, float, float]:
        """
        Update policy and value networks using PPO algorithm.
        
        Args:
            update_policy: Whether to update policy networks
            update_value: Whether to update value network
            
        Returns:
            Tuple of (discrete_policy_loss, continuous_policy_loss, value_loss)
        """
        if self.replay_buffer.is_empty():
            return 0.0, 0.0, 0.0

        # Prepare training data
        returns = torch.from_numpy(np.array(self.replay_buffer.returns))
        returns = (returns - returns.mean()) / (returns.std() + HPPOConstants.EPSILON)

        state_vals = torch.cat(list(self.replay_buffer.state_values)).to(torch.float32)
        old_lprobs_discrete = torch.cat(list(self.replay_buffer.logprobs_discrete)).to(torch.float32).detach()
        old_lprobs_continuous = torch.cat(list(self.replay_buffer.logprobs_continuous)).to(torch.float32).detach()
        objective_weights = torch.cat(list(self.replay_buffer.objective_weights)).to(torch.float32).detach()
        
        actionsc = torch.cat(list(self.replay_buffer.continuous_actions), dim=0).detach()
        actionsd = torch.cat(list(self.replay_buffer.discrete_actions), dim=-1).detach()

        # Perform multiple update epochs
        for _ in range(self.n_updates):
            # Prepare states and layer numbers
            states = torch.tensor(np.array(list(self.replay_buffer.states))).to(torch.float32)
            layer_numbers = (
                torch.tensor(list(self.replay_buffer.layer_number)).to(torch.float32)
                if self.include_layer_number else False
            )

            # Get current policy outputs
            _, _, _, log_prob_discrete, log_prob_continuous, _, _, _, state_value, entropy_discrete, entropy_continuous = self.select_action(
                states, layer_numbers, actionsc, actionsd, packed=True, objective_weights=objective_weights
            )

            # Calculate advantages
            advantage = returns.detach() - state_value.detach()

            # Calculate policy losses
            discrete_policy_loss = self._calculate_discrete_policy_loss(
                log_prob_discrete, old_lprobs_discrete, advantage, entropy_discrete
            )
            continuous_policy_loss = self._calculate_continuous_policy_loss(
                log_prob_continuous, old_lprobs_continuous, advantage, entropy_continuous
            )
            value_loss = self.mse_loss(returns.to(torch.float32).squeeze(), state_value.squeeze())

            # Update networks
            if update_policy:
                self._update_discrete_policy(discrete_policy_loss)
            
            self._update_continuous_policy(continuous_policy_loss)
            
            if update_value:
                self._update_value_network(value_loss)

        # Update old networks
        self.policy_discrete_old.load_state_dict(self.policy_discrete.state_dict())
        self.policy_continuous_old.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        # Clear replay buffer
        self.replay_buffer.clear()

        return discrete_policy_loss.item(), continuous_policy_loss.item(), value_loss.item()

    def _calculate_discrete_policy_loss(self, log_prob, old_log_prob, advantage, entropy):
        """Calculate PPO clipped discrete policy loss."""
        ratios = torch.exp(log_prob - old_log_prob)
        surr1 = ratios.squeeze() * advantage.squeeze()
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage.squeeze()
        return -(torch.min(surr1, surr2) + self.beta * entropy.squeeze()).mean()

    def _calculate_continuous_policy_loss(self, log_prob, old_log_prob, advantage, entropy):
        """Calculate PPO clipped continuous policy loss."""
        ratios = torch.exp(log_prob - old_log_prob)
        surr1 = ratios.squeeze() * advantage.squeeze()
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage.squeeze()
        return -(torch.min(surr1, surr2) + self.beta * entropy.squeeze()).mean()

    def _update_discrete_policy(self, loss):
        """Update discrete policy network."""
        self.optimiser_discrete.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_discrete.parameters(), max_norm=HPPOConstants.MAX_GRAD_NORM)
        self.optimiser_discrete.step()

    def _update_continuous_policy(self, loss):
        """Update continuous policy network."""
        self.optimiser_continuous.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_continuous.parameters(), max_norm=HPPOConstants.MAX_GRAD_NORM)
        self.optimiser_continuous.step()

    def _update_value_network(self, loss):
        """Update value network."""
        self.optimiser_value.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=HPPOConstants.MAX_GRAD_NORM)
        self.optimiser_value.step()

    def save_networks(self, save_directory: str, episode: Optional[int] = None) -> None:
        """
        Save network parameters to files.
        
        Args:
            save_directory: Directory to save networks
            episode: Current episode number
        """
        os.makedirs(save_directory, exist_ok=True)
        
        torch.save({
            "episode": episode,
            "model_state_dict": self.policy_discrete.state_dict(),
            "optimiser_state_dict": self.optimiser_discrete.state_dict()
        }, os.path.join(save_directory, HPPOConstants.DISCRETE_POLICY_FILE))

        torch.save({
            "episode": episode,
            "model_state_dict": self.policy_continuous.state_dict(),
            "optimiser_state_dict": self.optimiser_continuous.state_dict()
        }, os.path.join(save_directory, HPPOConstants.CONTINUOUS_POLICY_FILE))

        torch.save({
            "episode": episode,
            "model_state_dict": self.value.state_dict(),
            "optimiser_state_dict": self.optimiser_value.state_dict()
        }, os.path.join(save_directory, HPPOConstants.VALUE_FILE))

    def load_networks(self, load_directory: str) -> None:
        """
        Load network parameters from files.
        
        Args:
            load_directory: Directory containing saved networks
        """
        # Load discrete policy
        discrete_path = os.path.join(load_directory, HPPOConstants.DISCRETE_POLICY_FILE)
        dp = torch.load(discrete_path)
        self.policy_discrete.load_state_dict(dp["model_state_dict"])
        self.optimiser_discrete.load_state_dict(dp["optimiser_state_dict"])

        # Load continuous policy
        continuous_path = os.path.join(load_directory, HPPOConstants.CONTINUOUS_POLICY_FILE)
        cp = torch.load(continuous_path)
        self.policy_continuous.load_state_dict(cp["model_state_dict"])
        self.optimiser_continuous.load_state_dict(cp["optimiser_state_dict"])

        # Load value network
        value_path = os.path.join(load_directory, HPPOConstants.VALUE_FILE)
        vp = torch.load(value_path)
        self.value.load_state_dict(vp["model_state_dict"])
        self.optimiser_value.load_state_dict(vp["optimiser_state_dict"])