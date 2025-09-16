import os
from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from coatopt.algorithms.action_utils import (
    create_material_mask_from_coating_state,
    format_action_output,
    pack_state_sequence,
    prepare_layer_number,
    validate_probabilities,
)
from coatopt.algorithms.config import HPPOConstants
from coatopt.algorithms.hppo.core.networks.pre_networks import (
    PreNetworkAttention,
    PreNetworkLinear,
    PreNetworkLSTM,
)
from coatopt.algorithms.hppo.core.replay_buffer import ReplayBuffer
from coatopt.environments.core.state import CoatingState
from coatopt.utils.math_utils import TruncatedNormalDist


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
        lr_discrete_policy: float = 1e-4,
        lr_continuous_policy: float = 1e-4,
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
        entropy_beta_start: float = 1.0,
        entropy_beta_end: float = 0.001,
        entropy_beta_decay_start: int = 0,
        entropy_beta_decay_length: int = 500,
        entropy_beta_discrete_start: Optional[float] = None,
        entropy_beta_discrete_end: Optional[float] = None,
        entropy_beta_continuous_start: Optional[float] = None,
        entropy_beta_continuous_end: Optional[float] = None,
        entropy_beta_use_restarts: bool = False,
        hyper_networks: bool = False,
        buffer_size: int = 10000,
        batch_size: int = 64,
        use_mixture_of_experts: bool = False,
        moe_n_experts: int = 5,
        moe_expert_specialization: str = "weight_regions",
        moe_gate_hidden_dim: int = 64,
        moe_gate_temperature: float = 0.5,
        moe_load_balancing_weight: float = 0.01,
        multi_value_rewards: bool = False,
    ):
        """
        Initialize PC-HPPO agent.

        Args:
            state_dim: Dimensions of state space
            num_discrete: Number of discrete actions
            num_cont: Number of continuous actions
            hidden_size: Hidden layer size for networks
            num_objectives: Number of optimisation objectives
            lr_discrete_policy: Learning rate for discrete policy
            lr_continuous_policy: Learning rate for continuous policy
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
            entropy_beta_start: Initial entropy coefficient (default for both networks)
            entropy_beta_end: Final entropy coefficient (default for both networks)
            entropy_beta_decay_start: Step at which entropy decay begins
            entropy_beta_decay_length: Entropy decay length
            entropy_beta_discrete_start: Initial entropy coefficient for discrete policy (optional)
            entropy_beta_discrete_end: Final entropy coefficient for discrete policy (optional)
            entropy_beta_continuous_start: Initial entropy coefficient for continuous policy (optional)
            entropy_beta_continuous_end: Final entropy coefficient for continuous policy (optional)
            entropy_beta_use_restarts: Whether to use warm restarts for entropy beta decay (like LR scheduler)
            hyper_networks: Whether to use hypernetworks
            buffer_size: Maximum size of replay buffer
            batch_size: Mini-batch size for policy updates
            use_mixture_of_experts: Whether to use Mixture of Experts networks
            moe_n_experts: Number of expert networks in MoE
            moe_expert_specialization: How experts specialize ('weight_regions' or 'random')
            moe_gate_hidden_dim: Hidden dimension for MoE gating network
            moe_gate_temperature: Temperature for MoE gating softmax
            moe_load_balancing_weight: Weight for MoE load balancing loss
        """
        # Import network classes - now using unified policy networks
        from coatopt.algorithms.hppo.core.networks.policy_networks import (
            ContinuousPolicy,
            DiscretePolicy,
            ValueNetwork,
        )

        # Import MoE networks if needed
        if use_mixture_of_experts:
            from coatopt.algorithms.hppo.core.networks.mixture_of_experts import (
                MoEContinuousPolicy,
                MoEDiscretePolicy,
                MoEValueNetwork,
            )

        # Store configuration
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.include_layer_number = include_layer_number
        self.substrate_material_index = substrate_material_index
        self.air_material_index = air_material_index
        self.ignore_air_option = ignore_air_option
        self.ignore_substrate_option = ignore_substrate_option
        self.num_objectives = num_objectives
        self.entropy_beta_start = entropy_beta_start
        self.entropy_beta_end = entropy_beta_end
        self.entropy_beta_decay_start = entropy_beta_decay_start
        self.entropy_beta_decay_length = entropy_beta_decay_length
        self.hyper_networks = hyper_networks
        self.multi_value_rewards = multi_value_rewards

        # MoE configuration
        self.use_mixture_of_experts = use_mixture_of_experts
        self.moe_n_experts = moe_n_experts
        self.moe_expert_specialization = moe_expert_specialization
        self.moe_gate_hidden_dim = moe_gate_hidden_dim
        self.moe_gate_temperature = moe_gate_temperature
        self.moe_load_balancing_weight = moe_load_balancing_weight

        # Set separate entropy coefficients for discrete and continuous policies
        self.entropy_beta_discrete_start = (
            entropy_beta_discrete_start
            if entropy_beta_discrete_start is not None
            else entropy_beta_start
        )
        self.entropy_beta_discrete_end = (
            entropy_beta_discrete_end
            if entropy_beta_discrete_end is not None
            else entropy_beta_end
        )
        self.entropy_beta_continuous_start = (
            entropy_beta_continuous_start
            if entropy_beta_continuous_start is not None
            else entropy_beta_start
        )
        self.entropy_beta_continuous_end = (
            entropy_beta_continuous_end
            if entropy_beta_continuous_end is not None
            else entropy_beta_end
        )
        self.entropy_beta_use_restarts = entropy_beta_use_restarts

        self.pre_type = pre_type
        self.n_updates = n_updates
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Initialize current entropy coefficients
        self.beta_discrete = self.entropy_beta_discrete_start
        self.beta_continuous = self.entropy_beta_continuous_start

        # Initialize pre-network
        self.pre_output_dim = hidden_size
        self.pre_network = self._create_pre_network(
            state_dim, hidden_size, pre_type, n_heads, n_pre_layers, num_objectives
        )

        # Initialize policy and value networks (current and old versions)
        if use_mixture_of_experts:
            self._create_moe_networks(
                MoEDiscretePolicy,
                MoEContinuousPolicy,
                MoEValueNetwork,
                num_discrete,
                num_cont,
                discrete_hidden_size,
                continuous_hidden_size,
                value_hidden_size,
                n_discrete_layers,
                n_continuous_layers,
                n_value_layers,
                lower_bound,
                upper_bound,
                include_layer_number,
                activation_function,
                include_material_in_policy,
            )
        else:
            self._create_networks(
                DiscretePolicy,
                ContinuousPolicy,
                ValueNetwork,
                num_discrete,
                num_cont,
                discrete_hidden_size,
                continuous_hidden_size,
                value_hidden_size,
                n_discrete_layers,
                n_continuous_layers,
                n_value_layers,
                lower_bound,
                upper_bound,
                include_layer_number,
                activation_function,
                include_material_in_policy,
            )

        # Initialize optimizers and schedulers
        self._setup_optimizers(
            optimiser, lr_discrete_policy, lr_continuous_policy, lr_value
        )
        self._setup_schedulers(lr_step, T_mult, lr_min)

        # Initialize training components
        self.mse_loss = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.gamma = gamma

    def _create_pre_network(
        self, state_dim, hidden_size, pre_type, n_heads, n_pre_layers, num_objectives
    ):
        """Create pre-network based on specified type."""
        if pre_type == "attn":
            return PreNetworkAttention(
                state_dim[-1],
                self.pre_output_dim,
                hidden_size,
                num_heads=n_heads,
                num_layers=n_pre_layers,
            )
        elif pre_type == "lstm":
            return PreNetworkLSTM(
                state_dim[-1],
                self.pre_output_dim,
                hidden_size,
                include_layer_number=self.include_layer_number,
                n_layers=n_pre_layers,
                weight_dim=num_objectives,
            )
        elif pre_type == "linear":
            return PreNetworkLinear(
                np.prod(state_dim),
                self.pre_output_dim,
                hidden_size,
                n_layers=n_pre_layers,
                include_layer_number=self.include_layer_number,
            )
        else:
            raise ValueError(f"Unknown pre-network type: {pre_type}")

    def _create_networks(
        self,
        DiscretePolicy,
        ContinuousPolicy,
        Value,
        num_discrete,
        num_cont,
        discrete_hidden_size,
        continuous_hidden_size,
        value_hidden_size,
        n_discrete_layers,
        n_continuous_layers,
        n_value_layers,
        lower_bound,
        upper_bound,
        include_layer_number,
        activation_function,
        include_material_in_policy,
    ):
        """Create policy and value networks (current and old versions)."""
        for i in range(2):
            suffix = "" if i == 0 else "_old"

            # Discrete policy network
            setattr(
                self,
                f"policy_discrete{suffix}",
                DiscretePolicy(
                    self.pre_output_dim,
                    num_discrete,
                    discrete_hidden_size,
                    n_layers=n_discrete_layers,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    include_layer_number=include_layer_number,
                    activation=activation_function,
                    n_objectives=self.num_objectives,
                    use_hyper_networks=self.hyper_networks,
                ),
            )

            # Continuous policy network
            setattr(
                self,
                f"policy_continuous{suffix}",
                ContinuousPolicy(
                    self.pre_output_dim,
                    num_cont,
                    continuous_hidden_size,
                    n_layers=n_continuous_layers,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    include_layer_number=include_layer_number,
                    activation=activation_function,
                    n_objectives=self.num_objectives,
                    use_hyper_networks=self.hyper_networks,
                ),
            )

            # Value network
            setattr(
                self,
                f"value{suffix}",
                Value(
                    self.pre_output_dim,
                    value_hidden_size,
                    n_layers=n_value_layers,
                    include_layer_number=include_layer_number,
                    activation=activation_function,
                    n_objectives=self.num_objectives,
                    use_hyper_networks=self.hyper_networks,
                ),
            )

        # Copy parameters to old networks
        self.policy_discrete_old.load_state_dict(self.policy_discrete.state_dict())
        self.policy_continuous_old.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def _create_moe_networks(
        self,
        MoEDiscretePolicy,
        MoEContinuousPolicy,
        MoEValueNetwork,
        num_discrete,
        num_cont,
        discrete_hidden_size,
        continuous_hidden_size,
        value_hidden_size,
        n_discrete_layers,
        n_continuous_layers,
        n_value_layers,
        lower_bound,
        upper_bound,
        include_layer_number,
        activation_function,
        include_material_in_policy,
    ):
        """Create Mixture of Experts policy networks with shared value network (current and old versions).

        Note: Uses a single shared ValueNetwork instead of MoEValueNetwork for simplified value estimation.
        The policy networks remain as MoE (multiple experts), but the value network is shared across all experts.
        """
        # Import standard ValueNetwork for shared value network
        from coatopt.algorithms.hppo.core.networks.policy_networks import ValueNetwork

        for i in range(2):
            suffix = "" if i == 0 else "_old"

            # MoE Discrete policy network
            setattr(
                self,
                f"policy_discrete{suffix}",
                MoEDiscretePolicy(
                    self.pre_output_dim,
                    num_discrete,
                    discrete_hidden_size,
                    n_layers=n_discrete_layers,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    include_layer_number=include_layer_number,
                    activation=activation_function,
                    n_objectives=self.num_objectives,
                    n_experts=self.moe_n_experts,
                    expert_specialization=self.moe_expert_specialization,
                    gate_hidden_dim=self.moe_gate_hidden_dim,
                    gate_temperature=self.moe_gate_temperature,
                    load_balancing_weight=self.moe_load_balancing_weight,
                    use_hyper_networks=self.hyper_networks,
                ),
            )

            # MoE Continuous policy network
            setattr(
                self,
                f"policy_continuous{suffix}",
                MoEContinuousPolicy(
                    self.pre_output_dim,
                    num_cont,
                    continuous_hidden_size,
                    n_layers=n_continuous_layers,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    include_layer_number=include_layer_number,
                    include_material=include_material_in_policy,
                    activation=activation_function,
                    n_objectives=self.num_objectives,
                    n_experts=self.moe_n_experts,
                    expert_specialization=self.moe_expert_specialization,
                    gate_hidden_dim=self.moe_gate_hidden_dim,
                    gate_temperature=self.moe_gate_temperature,
                    load_balancing_weight=self.moe_load_balancing_weight,
                    use_hyper_networks=self.hyper_networks,
                ),
            )

            # Shared Value network (standard ValueNetwork, not MoE)
            setattr(
                self,
                f"value{suffix}",
                ValueNetwork(
                    self.pre_output_dim,
                    value_hidden_size,
                    n_layers=n_value_layers,
                    include_layer_number=include_layer_number,
                    activation=activation_function,
                    n_objectives=self.num_objectives,
                    use_hyper_networks=self.hyper_networks,
                    multi_value_rewards=self.multi_value_rewards,
                ),
            )

        # Copy parameters to old networks
        self.policy_discrete_old.load_state_dict(self.policy_discrete.state_dict())
        self.policy_continuous_old.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def _setup_optimizers(
        self, optimiser, lr_discrete_policy, lr_continuous_policy, lr_value
    ):
        """Setup optimizers for networks."""
        self.lr_value = lr_value
        self.lr_discrete_policy = lr_discrete_policy
        self.lr_continuous_policy = lr_continuous_policy

        if optimiser == "adam":
            self.optimiser_discrete = torch.optim.Adam(
                self.policy_discrete.parameters(), lr=lr_discrete_policy
            )
            self.optimiser_continuous = torch.optim.Adam(
                self.policy_continuous.parameters(), lr=lr_continuous_policy
            )
            self.optimiser_value = torch.optim.Adam(
                self.value.parameters(), lr=lr_value
            )
        elif optimiser == "sgd":
            self.optimiser_discrete = torch.optim.SGD(
                self.policy_discrete.parameters(), lr=lr_discrete_policy
            )
            self.optimiser_continuous = torch.optim.SGD(
                self.policy_continuous.parameters(), lr=lr_continuous_policy
            )
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
            self.optimiser_discrete,
            T_0=lr_step_discrete,
            T_mult=T_mult_discrete,
            eta_min=lr_min,
        )
        self.scheduler_continuous = (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimiser_continuous,
                T_0=lr_step_continuous,
                T_mult=T_mult_continuous,
                eta_min=lr_min,
            )
        )
        self.scheduler_value = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimiser_value, T_0=lr_step_value, T_mult=T_mult_value, eta_min=lr_min
        )

    def get_returns(
        self, rewards: List[float], multiobjective_rewards: List[List[float]] = None
    ) -> np.ndarray:
        """
        Calculate discounted returns from rewards.

        Args:
            rewards: List of scalar rewards (for backward compatibility)
            multiobjective_rewards: List of multi-objective reward vectors

        Returns:
            Array of discounted returns (scalar or multi-objective)
        """
        if multiobjective_rewards is not None:
            # Multi-objective returns computation
            temp_r = deque()
            R = np.zeros(
                len(multiobjective_rewards[0])
            )  # Initialize with correct dimensions
            for r in multiobjective_rewards[::-1]:
                R = np.array(r) + self.gamma * R
                temp_r.appendleft(R.copy())
            return np.array(temp_r)
        else:
            # Original scalar returns computation
            temp_r = deque()
            R = 0
            for r in rewards[::-1]:
                R = r + self.gamma * R
                temp_r.appendleft(R)
            return np.array(temp_r)

    def scheduler_step(
        self, step: int = 0, make_step: bool = True, scheduler_active: bool = True
    ) -> Tuple[List[float], List[float], List[float], float, float]:
        """
        Step learning rate schedulers and calculate entropy coefficients.

        Args:
            step: Current step number
            make_step: Whether to actually step the schedulers
            scheduler_active: Whether scheduler should be active (respects scheduler_start/end bounds)

        Returns:
            Tuple of (discrete_lr, continuous_lr, value_lr, discrete_entropy_coeff, continuous_entropy_coeff)
        """
        if make_step:
            self.scheduler_discrete.step(step)
            self.scheduler_continuous.step(step)
            self.scheduler_value.step(step)

        # Calculate entropy coefficients using cosine annealing with optional warm restarts
        def calculate_entropy_coefficient(start_val, end_val):
            # If scheduler not active, use end value (stay at minimum)
            if not scheduler_active:
                return end_val
            # If before decay starts, use start value
            elif step < self.entropy_beta_decay_start:
                return start_val
            elif self.entropy_beta_use_restarts and hasattr(self, "lr_step"):
                # Use warm restarts similar to learning rate scheduler
                # Calculate progress within current cycle
                adjusted_step = step - self.entropy_beta_decay_start
                cycle_length = self.lr_step  # Use same cycle length as LR scheduler

                # Find which cycle we're in and position within that cycle
                cycle_position = adjusted_step % cycle_length
                progress = cycle_position / cycle_length

                # Cosine annealing within current cycle
                return (
                    end_val + (start_val - end_val) * (1 + np.cos(np.pi * progress)) / 2
                )
            else:
                # Original single-decay behavior
                if self.entropy_beta_decay_length is None:
                    # If no decay length specified, use end value
                    return end_val
                elif (
                    step
                    >= self.entropy_beta_decay_start + self.entropy_beta_decay_length
                ):
                    # After decay ends, use end value
                    return end_val
                else:
                    # During decay period, use cosine annealing
                    progress = (
                        step - self.entropy_beta_decay_start
                    ) / self.entropy_beta_decay_length
                    return (
                        end_val
                        + (start_val - end_val) * (1 + np.cos(np.pi * progress)) / 2
                    )

        discrete_entropy_val = calculate_entropy_coefficient(
            self.entropy_beta_discrete_start, self.entropy_beta_discrete_end
        )
        continuous_entropy_val = calculate_entropy_coefficient(
            self.entropy_beta_continuous_start, self.entropy_beta_continuous_end
        )

        # Update current entropy coefficients
        self.beta_discrete = discrete_entropy_val
        self.beta_continuous = continuous_entropy_val

        return (
            self.scheduler_discrete.get_last_lr(),
            self.scheduler_continuous.get_last_lr(),
            self.scheduler_value.get_last_lr(),
            discrete_entropy_val,
            continuous_entropy_val,
        )

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        layer_number: Optional[Union[np.ndarray, torch.Tensor]] = None,
        actionc: Optional[torch.Tensor] = None,
        actiond: Optional[torch.Tensor] = None,
        packed: bool = False,
        objective_weights: Optional[torch.Tensor] = None,
        original_states: Optional[
            List
        ] = None,  # Original CoatingState objects for constraints
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
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
            Tuple containing action components, probabilities, values, and MoE auxiliary losses
        """
        # Prepare inputs - keep original state for constraints, get observation for networks
        original_state = state  # Keep reference to original CoatingState

        # Get observation tensor for networks - handles all formatting including pre_type
        if hasattr(state, "get_observation_tensor"):
            # CoatingState - use its comprehensive observation method with pre_type formatting
            obs_tensor = state.get_observation_tensor(pre_type=self.pre_type)
            # Add batch dimension if missing
            if obs_tensor.dim() == 1:  # [features] -> [1, features]
                obs_tensor = obs_tensor.unsqueeze(0)
            elif (
                obs_tensor.dim() == 2 and self.pre_type != "linear"
            ):  # [layers, features] -> [1, layers, features]
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            # Fallback for raw tensors (shouldn't happen with current flow but just in case)
            obs_tensor = state
            if self.pre_type == "linear" and obs_tensor.dim() > 1:
                obs_tensor = obs_tensor.flatten(1)

        layer_number, layer_numbers_validated = prepare_layer_number(layer_number)

        # Ensure objective_weights is properly formatted as float tensor
        if objective_weights is not None:
            if not isinstance(objective_weights, torch.Tensor):
                objective_weights = torch.FloatTensor(objective_weights)
            else:
                objective_weights = objective_weights.float()  # Ensure float dtype

            # Ensure correct batch dimension
            if objective_weights.dim() == 1:
                objective_weights = objective_weights.unsqueeze(0)

        # Pack state sequence only for LSTM processing
        if self.pre_type == "lstm":
            state_input = pack_state_sequence(obs_tensor, layer_numbers_validated)
            use_packed = True
        else:
            state_input = obs_tensor
            use_packed = False

        # Get pre-network outputs (optimized based on gradient needs)
        is_training = (
            actionc is not None or actiond is not None
        )  # If actions provided, likely training
        pre_output_d, pre_output_c, pre_output_v = self._get_pre_network_outputs(
            state_input,
            layer_number,
            objective_weights,
            needs_gradients=is_training,
            packed=use_packed,
        )

        # Get discrete action probabilities
        if self.use_mixture_of_experts:
            discrete_probs, discrete_aux = self.policy_discrete(
                pre_output_d, objective_weights, layer_number
            )
        else:
            discrete_probs = self.policy_discrete(
                pre_output_d, layer_number, objective_weights=objective_weights
            )
            discrete_aux = None

        validate_probabilities(discrete_probs, "discrete_probs")

        # Apply material constraints
        if original_states is not None:
            # Batch case: we have a list of original CoatingState objects
            masked_discrete_probs = self._apply_material_constraints_batch(
                discrete_probs, original_states, layer_number
            )
        else:
            # Single state case: use original_state (which is a CoatingState)
            masked_discrete_probs = self._apply_material_constraints(
                discrete_probs, original_state, layer_number
            )

        # Sample discrete action
        discrete_dist = torch.distributions.Categorical(masked_discrete_probs)
        if actiond is None:
            actiond = discrete_dist.sample()

        # Get continuous action parameters and sample
        if self.use_mixture_of_experts:
            continuous_means, continuous_log_std, continuous_aux = (
                self.policy_continuous(
                    pre_output_c, objective_weights, layer_number, actiond.unsqueeze(1)
                )
            )
        else:
            continuous_means, continuous_log_std = self.policy_continuous(
                pre_output_c,
                layer_number,
                actiond.unsqueeze(1),
                objective_weights=objective_weights,
            )
            continuous_aux = None

        continuous_std = torch.exp(continuous_log_std)

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

        # Get state value (now always returns raw multi-objective values)
        if self.use_mixture_of_experts:
            # With shared value network, we get raw N-dimensional values (no weighting in network)
            state_value = self.value(
                pre_output_v, layer_number, objective_weights=objective_weights
            )
            value_aux = None  # No auxiliary info from shared value network
        else:
            state_value = self.value(
                pre_output_v, layer_number, objective_weights=objective_weights
            )
            value_aux = None

        # Format output action
        action = format_action_output(actiond, actionc)

        # Collect auxiliary losses from MoE networks
        moe_aux_losses = {}
        if self.use_mixture_of_experts:
            if discrete_aux is not None:
                moe_aux_losses["discrete_specialization_loss"] = discrete_aux[
                    "specialization_loss"
                ]
                moe_aux_losses["discrete_load_balance_loss"] = discrete_aux[
                    "load_balance_loss"
                ]
                moe_aux_losses["discrete_total_aux_loss"] = discrete_aux[
                    "total_aux_loss"
                ]

            if continuous_aux is not None:
                moe_aux_losses["continuous_specialization_loss"] = continuous_aux[
                    "specialization_loss"
                ]
                moe_aux_losses["continuous_load_balance_loss"] = continuous_aux[
                    "load_balance_loss"
                ]
                moe_aux_losses["continuous_total_aux_loss"] = continuous_aux[
                    "total_aux_loss"
                ]

            if value_aux is not None:
                moe_aux_losses["value_specialization_loss"] = value_aux[
                    "specialization_loss"
                ]
                moe_aux_losses["value_load_balance_loss"] = value_aux[
                    "load_balance_loss"
                ]
                moe_aux_losses["value_total_aux_loss"] = value_aux["total_aux_loss"]

        return (
            action,
            actiond,
            actionc,
            log_prob_discrete,
            log_prob_continuous,
            discrete_probs,
            continuous_means,
            continuous_std,
            state_value,
            entropy_discrete,
            entropy_continuous,
            moe_aux_losses,
        )

    def _get_pre_network_outputs(
        self,
        state_input,
        layer_number,
        objective_weights,
        needs_gradients=True,
        packed=False,
    ):
        """
        Get pre-network outputs for different network heads with optimized caching.

        Args:
            state_input: State input (packed sequence for LSTM, regular tensor for others)
            layer_number: Layer number tensor
            objective_weights: Multi-objective weights
            needs_gradients: Whether gradients are needed (training vs inference)
            packed: Whether the input is a packed sequence

        Returns:
            Tuple of (discrete_output, continuous_output, value_output)
        """
        # Optimization: Use single forward pass during inference
        if not needs_gradients:
            # During inference: single forward pass with no_grad for efficiency
            with torch.no_grad():
                if self.pre_type == "lstm":
                    pre_output = self.pre_network(
                        state_input,
                        layer_number,
                        packed=packed,
                        weights=objective_weights,
                    )
                else:
                    # For non-LSTM networks, don't pass packed or weights parameters
                    pre_output = self.pre_network(state_input, layer_number)

                # Return detached copies for each head
                pre_output_detached = pre_output.detach()
                return pre_output_detached, pre_output_detached, pre_output_detached
        else:
            # During training: separate forward passes required for gradient computation
            # Note: Cannot optimize this due to computational graph constraints
            if self.pre_type == "lstm":
                pre_output_d = self.pre_network(
                    state_input, layer_number, packed=packed, weights=objective_weights
                )
                pre_output_c = self.pre_network(
                    state_input, layer_number, packed=packed, weights=objective_weights
                )
                pre_output_v = self.pre_network(
                    state_input, layer_number, packed=packed, weights=objective_weights
                )
            else:
                # For non-LSTM networks, don't pass packed or weights parameters
                pre_output_d = self.pre_network(state_input, layer_number)
                pre_output_c = self.pre_network(state_input, layer_number)
                pre_output_v = self.pre_network(state_input, layer_number)
            return pre_output_d, pre_output_c, pre_output_v

    def _apply_material_constraints(self, discrete_probs, coating_state, layer_number):
        """Apply material selection constraints using CoatingState."""
        if layer_number is None:
            return discrete_probs

        # Use CoatingState-specific material mask function
        masked_probs = create_material_mask_from_coating_state(
            discrete_probs,
            coating_state,
            layer_number,
            self.ignore_air_option,
            self.ignore_substrate_option,
        )

        return masked_probs

    def _apply_material_constraints_batch(
        self, discrete_probs, coating_states, layer_numbers
    ):
        """Apply material selection constraints for batch of CoatingState objects."""
        if layer_numbers is None or len(coating_states) == 0:
            return discrete_probs

        batch_size = discrete_probs.shape[0]
        masked_probs_list = []

        for i in range(batch_size):
            # Get the corresponding state and layer number for this batch item
            state_i = (
                coating_states[i] if i < len(coating_states) else coating_states[0]
            )
            layer_num_i = (
                layer_numbers[i : i + 1]
                if isinstance(layer_numbers, torch.Tensor)
                else layer_numbers
            )
            discrete_prob_i = discrete_probs[i : i + 1]

            # Apply constraints for this individual item
            masked_prob_i = create_material_mask_from_coating_state(
                discrete_prob_i,
                state_i,
                layer_num_i,
                self.ignore_air_option,
                self.ignore_substrate_option,
            )
            masked_probs_list.append(masked_prob_i)

        # Concatenate all masked probabilities
        return torch.cat(masked_probs_list, dim=0)

    def _create_mini_batches(self, total_size: int) -> List[slice]:
        """
        Create mini-batch indices for processing replay buffer data.

        Args:
            total_size: Total number of samples in replay buffer

        Returns:
            List of slice objects for mini-batches
        """
        indices = torch.randperm(total_size)  # Shuffle indices for better training
        mini_batches = []

        for start_idx in range(0, total_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_size)
            batch_indices = indices[start_idx:end_idx]
            mini_batches.append(batch_indices)

        return mini_batches

    def update(
        self, update_policy: bool = True, update_value: bool = True
    ) -> Tuple[float, float, float]:
        """
        Update policy and value networks using PPO algorithm with mini-batches.

        Args:
            update_policy: Whether to update policy networks
            update_value: Whether to update value network

        Returns:
            Tuple of (discrete_policy_loss, continuous_policy_loss, value_loss)
        """
        if self.replay_buffer.is_empty():
            return 0.0, 0.0, 0.0

        # Prepare training data
        returns_array = np.array(self.replay_buffer.returns)
        returns = torch.from_numpy(returns_array)

        # Handle normalisation for both scalar and multi-objective returns
        if returns.dim() > 1 and self.multi_value_rewards:  # Multi-objective returns
            # normalise each objective separately
            mean_returns = returns.mean(dim=0, keepdim=True)
            std_returns = returns.std(dim=0, keepdim=True) + HPPOConstants.EPSILON
            returns = (returns - mean_returns) / std_returns
        else:  # Scalar returns
            returns = (returns - returns.mean()) / (
                returns.std() + HPPOConstants.EPSILON
            )

        state_vals = torch.cat(list(self.replay_buffer.state_values)).to(torch.float32)
        old_lprobs_discrete = (
            torch.cat(list(self.replay_buffer.logprobs_discrete))
            .to(torch.float32)
            .detach()
        )
        old_lprobs_continuous = (
            torch.cat(list(self.replay_buffer.logprobs_continuous))
            .to(torch.float32)
            .detach()
        )
        objective_weights = (
            torch.cat(list(self.replay_buffer.objective_weights))
            .to(torch.float32)
            .detach()
        )

        actionsc = torch.cat(
            list(self.replay_buffer.continuous_actions), dim=0
        ).detach()
        actionsd = torch.cat(list(self.replay_buffer.discrete_actions), dim=-1).detach()

        # Prepare states and layer numbers (full batch for creating mini-batches)
        raw_states = list(self.replay_buffer.states)

        # Use pre-computed observations from replay buffer - much more efficient
        if len(self.replay_buffer.observations) > 0:
            states = torch.cat(list(self.replay_buffer.observations), dim=0).to(
                torch.float32
            )
        else:
            states = torch.empty(0, 0, 0).to(torch.float32)

        layer_numbers = (
            torch.tensor(list(self.replay_buffer.layer_number)).to(torch.float32)
            if self.include_layer_number
            else False
        )

        # Get total buffer size and create mini-batches
        total_size = len(self.replay_buffer)
        mini_batch_indices = self._create_mini_batches(total_size)

        # Accumulate losses across mini-batches
        total_discrete_loss = 0.0
        total_continuous_loss = 0.0
        total_value_loss = 0.0
        num_batches = len(mini_batch_indices)

        # Perform multiple update epochs
        for epoch_idx in range(self.n_updates):
            epoch_discrete_loss = 0.0
            epoch_continuous_loss = 0.0
            epoch_value_loss = 0.0

            # Process each mini-batch
            for batch_idx, batch_indices in enumerate(mini_batch_indices):
                # Extract mini-batch data
                batch_returns = returns[batch_indices]
                batch_state_vals = state_vals[batch_indices]
                batch_old_lprobs_discrete = old_lprobs_discrete[batch_indices]
                batch_old_lprobs_continuous = old_lprobs_continuous[batch_indices]
                batch_objective_weights = objective_weights[batch_indices]
                batch_actionsc = actionsc[batch_indices]
                batch_actionsd = actionsd[batch_indices]
                batch_states = states[batch_indices]
                batch_raw_states = [raw_states[i] for i in batch_indices]

                batch_layer_numbers = (
                    layer_numbers[batch_indices] if self.include_layer_number else False
                )

                # Get current policy outputs for this mini-batch
                (
                    _,
                    _,
                    _,
                    log_prob_discrete,
                    log_prob_continuous,
                    _,
                    _,
                    _,
                    state_value,
                    entropy_discrete,
                    entropy_continuous,
                    moe_aux_losses,
                ) = self.select_action(
                    batch_states,
                    batch_layer_numbers,
                    batch_actionsc,
                    batch_actionsd,
                    packed=True,
                    objective_weights=batch_objective_weights,
                    original_states=batch_raw_states,
                )

                # Calculate advantages for this mini-batch
                if (
                    batch_returns.ndim > 1 and self.multi_value_rewards
                ):  # Multi-objective returns
                    # Compute multi-objective advantages
                    multiobjective_advantages = (
                        batch_returns.detach() - state_value.detach()
                    )

                    # Weight the advantages using objective weights
                    if batch_objective_weights is not None:
                        # Ensure objective_weights has the right shape for broadcasting
                        if batch_objective_weights.size(
                            -1
                        ) != multiobjective_advantages.size(-1):
                            raise ValueError(
                                f"Objective weights dimension ({batch_objective_weights.size(-1)}) "
                                f"must match advantage dimension ({multiobjective_advantages.size(-1)})"
                            )
                        # Compute weighted scalar advantage
                        advantage = torch.sum(
                            multiobjective_advantages * batch_objective_weights,
                            dim=-1,
                            keepdim=True,
                        )

                        # Also compute weighted returns for value loss
                        weighted_returns = torch.sum(
                            batch_returns * batch_objective_weights,
                            dim=-1,
                            keepdim=True,
                        )
                        weighted_state_value = torch.sum(
                            state_value * batch_objective_weights, dim=-1, keepdim=True
                        )
                    else:
                        # Fallback: use mean of advantages if no weights provided
                        print(
                            "Warning: No objective weights provided for multi-objective advantages. Using mean advantage."
                        )
                        advantage = torch.mean(
                            multiobjective_advantages, dim=-1, keepdim=True
                        )
                        weighted_returns = torch.mean(
                            batch_returns, dim=-1, keepdim=True
                        )
                        weighted_state_value = torch.mean(
                            state_value, dim=-1, keepdim=True
                        )
                else:
                    # Original scalar advantage computation
                    advantage = batch_returns.detach() - state_value.detach().squeeze(1)
                    weighted_returns = batch_returns
                    weighted_state_value = state_value

                # Calculate policy losses (including MoE auxiliary losses)
                discrete_policy_loss = self._calculate_discrete_policy_loss(
                    log_prob_discrete,
                    batch_old_lprobs_discrete,
                    advantage,
                    entropy_discrete,
                    self.beta_discrete,
                )

                # Add MoE auxiliary losses to discrete policy loss
                if (
                    self.use_mixture_of_experts
                    and "discrete_total_aux_loss" in moe_aux_losses
                ):
                    discrete_policy_loss += moe_aux_losses["discrete_total_aux_loss"]

                continuous_policy_loss = self._calculate_continuous_policy_loss(
                    log_prob_continuous,
                    batch_old_lprobs_continuous,
                    advantage,
                    entropy_continuous,
                    self.beta_continuous,
                )

                # Add MoE auxiliary losses to continuous policy loss
                if (
                    self.use_mixture_of_experts
                    and "continuous_total_aux_loss" in moe_aux_losses
                ):
                    continuous_policy_loss += moe_aux_losses[
                        "continuous_total_aux_loss"
                    ]

                if (
                    batch_returns.ndim > 1 and self.multi_value_rewards
                ):  # Multi-objective returns
                    # For multi-objective case, compute value loss using weighted returns and values
                    value_loss = self.mse_loss(
                        weighted_returns.to(torch.float32).squeeze(),
                        weighted_state_value.squeeze(),
                    )
                else:
                    # Original scalar value loss
                    value_loss = self.mse_loss(
                        weighted_returns.to(torch.float32).squeeze(),
                        weighted_state_value.squeeze(),
                    )

                # Add MoE auxiliary losses to value loss
                if (
                    self.use_mixture_of_experts
                    and "value_total_aux_loss" in moe_aux_losses
                ):
                    value_loss += moe_aux_losses["value_total_aux_loss"]

                # Update networks for this mini-batch
                if update_policy:
                    self._update_discrete_policy(discrete_policy_loss)

                self._update_continuous_policy(continuous_policy_loss)

                if update_value:
                    self._update_value_network(value_loss)

                # Accumulate losses for this epoch
                epoch_discrete_loss += discrete_policy_loss.item()
                epoch_continuous_loss += continuous_policy_loss.item()
                epoch_value_loss += value_loss.item()

            # Average losses across mini-batches for this epoch
            total_discrete_loss += epoch_discrete_loss / num_batches
            total_continuous_loss += epoch_continuous_loss / num_batches
            total_value_loss += epoch_value_loss / num_batches

        # Average losses across epochs
        avg_discrete_loss = total_discrete_loss / self.n_updates
        avg_continuous_loss = total_continuous_loss / self.n_updates
        avg_value_loss = total_value_loss / self.n_updates

        # Update old networks
        self.policy_discrete_old.load_state_dict(self.policy_discrete.state_dict())
        self.policy_continuous_old.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        # Clear replay buffer
        self.replay_buffer.clear()

        return avg_discrete_loss, avg_continuous_loss, avg_value_loss

    def _calculate_discrete_policy_loss(
        self, log_prob, old_log_prob, advantage, entropy, entropy_coeff
    ):
        """Calculate PPO clipped discrete policy loss."""
        ratios = torch.exp(log_prob - old_log_prob)
        surr1 = ratios.squeeze() * advantage.squeeze()
        surr2 = (
            torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            * advantage.squeeze()
        )
        return -(torch.min(surr1, surr2) + entropy_coeff * entropy.squeeze()).mean()

    def _calculate_continuous_policy_loss(
        self, log_prob, old_log_prob, advantage, entropy, entropy_coeff
    ):
        """Calculate PPO clipped continuous policy loss."""
        ratios = torch.exp(log_prob - old_log_prob)
        surr1 = ratios.squeeze() * advantage.squeeze()
        surr2 = (
            torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
            * advantage.squeeze()
        )
        return -(torch.min(surr1, surr2) + entropy_coeff * entropy.squeeze()).mean()

    def _update_discrete_policy(self, loss):
        """Update discrete policy network."""
        self.optimiser_discrete.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_discrete.parameters(), max_norm=HPPOConstants.MAX_GRAD_NORM
        )
        self.optimiser_discrete.step()

    def _update_continuous_policy(self, loss):
        """Update continuous policy network."""
        self.optimiser_continuous.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_continuous.parameters(), max_norm=HPPOConstants.MAX_GRAD_NORM
        )
        self.optimiser_continuous.step()

    def _update_value_network(self, loss):
        """Update value network."""
        self.optimiser_value.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), max_norm=HPPOConstants.MAX_GRAD_NORM
        )
        self.optimiser_value.step()

    def save_networks(self, save_directory: str, episode: Optional[int] = None) -> None:
        """
        Save network parameters to files.

        Args:
            save_directory: Directory to save networks
            episode: Current episode number
        """
        os.makedirs(save_directory, exist_ok=True)

        torch.save(
            {
                "episode": episode,
                "model_state_dict": self.policy_discrete.state_dict(),
                "optimiser_state_dict": self.optimiser_discrete.state_dict(),
            },
            os.path.join(save_directory, HPPOConstants.DISCRETE_POLICY_FILE),
        )

        torch.save(
            {
                "episode": episode,
                "model_state_dict": self.policy_continuous.state_dict(),
                "optimiser_state_dict": self.optimiser_continuous.state_dict(),
            },
            os.path.join(save_directory, HPPOConstants.CONTINUOUS_POLICY_FILE),
        )

        torch.save(
            {
                "episode": episode,
                "model_state_dict": self.value.state_dict(),
                "optimiser_state_dict": self.optimiser_value.state_dict(),
            },
            os.path.join(save_directory, HPPOConstants.VALUE_FILE),
        )

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
        continuous_path = os.path.join(
            load_directory, HPPOConstants.CONTINUOUS_POLICY_FILE
        )
        cp = torch.load(continuous_path)
        self.policy_continuous.load_state_dict(cp["model_state_dict"])
        self.optimiser_continuous.load_state_dict(cp["optimiser_state_dict"])

        # Load value network
        value_path = os.path.join(load_directory, HPPOConstants.VALUE_FILE)
        vp = torch.load(value_path)
        self.value.load_state_dict(vp["model_state_dict"])
        self.optimiser_value.load_state_dict(vp["optimiser_state_dict"])
