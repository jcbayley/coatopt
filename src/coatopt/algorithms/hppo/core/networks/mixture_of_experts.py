from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy_networks import ContinuousPolicy, DiscretePolicy, ValueNetwork


class MixtureOfExpertsGating(nn.Module):
    """
    Gating network that determines expert weights based on objective weights.
    Maps from objective space to expert selection probabilities.
    """

    def __init__(
        self,
        n_objectives: int,
        n_experts: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_objectives = n_objectives
        self.n_experts = n_experts
        self.temperature = temperature

        # Gating network - maps objective weights to expert probabilities
        self.gate_network = nn.Sequential(
            nn.Linear(n_objectives, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )

        # Initialize to encourage diverse expert usage initially
        nn.init.xavier_uniform_(self.gate_network[-1].weight)
        nn.init.zeros_(self.gate_network[-1].bias)

    def forward(
        self, objective_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through gating network.

        Args:
            objective_weights: (batch_size, n_objectives) objective weight vectors

        Returns:
            gate_weights: (batch_size, n_experts) softmax probabilities
            gate_logits: (batch_size, n_experts) raw logits (for entropy computation)
        """
        # Ensure objective_weights is float tensor
        if objective_weights.dtype != torch.float32:
            objective_weights = objective_weights.float()

        gate_logits = self.gate_network(objective_weights)
        gate_weights = F.softmax(gate_logits / self.temperature, dim=-1)
        return gate_weights, gate_logits


class MixtureOfExpertsBase(nn.Module):
    """
    Base class for Mixture of Experts networks.
    Contains common functionality for managing multiple expert networks.
    """

    def __init__(
        self,
        n_experts: int,
        n_objectives: int,
        expert_specialization: str = "sobol_sequence",
        gate_hidden_dim: int = 64,
        gate_temperature: float = 0.5,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_objectives = n_objectives
        self.expert_specialization = expert_specialization
        self.expert_dropout = expert_dropout
        self.load_balancing_weight = load_balancing_weight

        # Gating network
        self.gating_network = MixtureOfExpertsGating(
            n_objectives, n_experts, gate_hidden_dim, gate_temperature
        )

        # Define expert specialization regions
        self.expert_regions = self._define_expert_regions()

        # Initialize constraint targets (for adaptive_constraints mode)
        self.constraint_targets = {}
        self.constraint_penalties = {}  # Will store constraint penalty weights

    def _define_expert_regions(self) -> List[torch.Tensor]:
        """
        Define weight regions that each expert should specialize in.
        Returns list of weight vectors, one per expert.
        """
        if self.expert_specialization == "sobol_sequence":
            return self._generate_sobol_sequence()
        elif self.expert_specialization == "random":
            return self._generate_random_regions()
        elif self.expert_specialization == "adaptive_constraints":
            return self._generate_adaptive_constraint_regions()
        else:
            raise ValueError(
                f"Unknown expert specialization: {self.expert_specialization}. "
                f"Valid options are: 'sobol_sequence', 'random', 'adaptive_constraints'"
            )

    def _generate_sobol_sequence(self) -> List[torch.Tensor]:
        """
        Generate expert regions ensuring dedicated specialists for each objective.

        First n_objectives experts are dedicated to individual objectives,
        remaining experts use Sobol sequence for balanced regions.
        """
        regions = []

        # First, create dedicated experts for each objective
        for i in range(self.n_objectives):
            weights = np.zeros(self.n_objectives)
            weights[i] = 1.0  # Pure specialist for objective i
            regions.append(torch.tensor(weights, dtype=torch.float32))

        # If we have more experts than objectives, fill remaining with Sobol sequence
        remaining_experts = self.n_experts - self.n_objectives
        if remaining_experts > 0:
            try:
                from scipy.stats import qmc

                sampler = qmc.Sobol(d=self.n_objectives - 1, scramble=True)
                samples = sampler.random(remaining_experts)

                for j, sample in enumerate(samples):
                    weights = self._stick_breaking_to_simplex(sample)
                    regions.append(torch.tensor(weights, dtype=torch.float32))

            except ImportError:
                # Fallback to random if scipy not available
                print(
                    "Warning: scipy not available for Sobol sequence, falling back to random sampling for remaining experts"
                )
                for _ in range(remaining_experts):
                    weights = np.random.dirichlet(np.ones(self.n_objectives))
                    regions.append(torch.tensor(weights, dtype=torch.float32))

        return regions

    def _generate_random_regions(self) -> List[torch.Tensor]:
        """
        Generate expert regions ensuring dedicated specialists for each objective.

        First n_objectives experts are dedicated to individual objectives,
        remaining experts use random Dirichlet distribution.
        """
        regions = []

        # First, create dedicated experts for each objective
        for i in range(self.n_objectives):
            weights = np.zeros(self.n_objectives)
            weights[i] = 1.0  # Pure specialist for objective i
            regions.append(torch.tensor(weights, dtype=torch.float32))

        # Fill remaining experts with random regions
        remaining_experts = self.n_experts - self.n_objectives
        for _ in range(remaining_experts):
            weights = np.random.dirichlet(np.ones(self.n_objectives))
            regions.append(torch.tensor(weights, dtype=torch.float32))

        return regions

    def update_constraint_expert_regions(
        self,
        reward_histories: Dict[str, List[float]],
        n_constraint_experts_per_objective: int = 2,
    ):
        """
        Update expert regions for constraint-based specialization after Phase 1 training.

        Args:
            reward_histories: Dict mapping objective names to reward history lists
            n_constraint_experts_per_objective: Number of constraint experts per objective
        """
        if self.expert_specialization != "adaptive_constraints":
            return

        # Validate we have enough experts
        required_experts = (
            self.n_objectives
            + (self.n_objectives * n_constraint_experts_per_objective)
            + 1
        )
        if self.n_experts < required_experts:
            print(
                f"Warning: Need at least {required_experts} experts for constraint specialization, "
                f"but only have {self.n_experts}. Using available experts."
            )
            n_constraint_experts_per_objective = max(
                1, (self.n_experts - self.n_objectives - 1) // self.n_objectives
            )

        # Analyze reward histories to determine constraint targets
        constraint_targets = self._analyze_reward_histories(
            reward_histories, n_constraint_experts_per_objective
        )

        # Update expert regions
        regions = []
        expert_idx = 0

        # Pure objective experts (unchanged)
        for i in range(self.n_objectives):
            weights = np.zeros(self.n_objectives)
            weights[i] = 1.0
            regions.append(torch.tensor(weights, dtype=torch.float32))
            expert_idx += 1

        # Constraint experts
        objective_names = list(constraint_targets.keys())
        for obj_idx, obj_name in enumerate(objective_names):
            for target_reward in constraint_targets[obj_name]:
                if (
                    expert_idx >= self.n_experts - 1
                ):  # Save last slot for balanced expert
                    break

                # Create constraint expert: fix obj_name at target_reward, optimize others
                # For now, use equal weights (actual constraints handled in reward function)
                weights = np.ones(self.n_objectives) / self.n_objectives
                regions.append(torch.tensor(weights, dtype=torch.float32))
                expert_idx += 1

        # Last expert: Balanced multi-objective
        if expert_idx < self.n_experts:
            balanced_weights = np.ones(self.n_objectives) / self.n_objectives
            regions.append(torch.tensor(balanced_weights, dtype=torch.float32))

        # Update the expert regions
        self.expert_regions = regions

        # Store constraint targets for use in reward computation
        self.constraint_targets = constraint_targets

    def _analyze_reward_histories(
        self, reward_histories: Dict[str, List[float]], n_targets_per_objective: int
    ) -> Dict[str, List[float]]:
        """
        Analyze reward histories to determine constraint targets.

        Args:
            reward_histories: Dict mapping objective names to reward history lists
            n_targets_per_objective: Number of constraint targets per objective

        Returns:
            Dict mapping objective names to list of target reward values
        """
        constraint_targets = {}

        for obj_name, rewards in reward_histories.items():
            if len(rewards) == 0:
                continue

            # Get min and max from history
            min_reward = min(rewards)
            max_reward = max(rewards)

            # Create equally spaced targets between min and max
            if n_targets_per_objective == 1:
                targets = [(min_reward + max_reward) / 2]
            else:
                targets = []
                for i in range(n_targets_per_objective):
                    target = min_reward + (i + 1) * (max_reward - min_reward) / (
                        n_targets_per_objective + 1
                    )
                    targets.append(target)

            constraint_targets[obj_name] = targets

        return constraint_targets

    def _stick_breaking_to_simplex(self, uniform_sample: np.ndarray) -> np.ndarray:
        """Convert uniform sample to simplex using stick-breaking construction."""
        # Stick-breaking: convert (n-1) uniform variables to n simplex coordinates
        n = len(uniform_sample) + 1
        beta_samples = np.zeros(n)

        remaining = 1.0
        for i in range(n - 1):
            # Convert uniform to beta(1, n-i-1) using inverse CDF
            beta_samples[i] = uniform_sample[i] * remaining
            remaining *= 1.0 - uniform_sample[i]
        beta_samples[n - 1] = remaining

        return beta_samples

    def get_expert_specialization_loss(
        self, objective_weights: torch.Tensor, gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss to encourage experts to specialize in their assigned regions.

        Args:
            objective_weights: (batch_size, n_objectives) current objective weights
            gate_weights: (batch_size, n_experts) gate probabilities

        Returns:
            specialization_loss: scalar tensor
        """
        batch_size = objective_weights.size(0)
        device = objective_weights.device

        # Move expert regions to correct device
        expert_regions = torch.stack(
            [region.to(device) for region in self.expert_regions]
        )  # (n_experts, n_objectives)

        # Compute distances from current weights to each expert's specialization region
        distances = torch.cdist(
            objective_weights.unsqueeze(1),  # (batch_size, 1, n_objectives)
            expert_regions.unsqueeze(0).expand(
                batch_size, -1, -1
            ),  # (batch_size, n_experts, n_objectives)
        ).squeeze(
            1
        )  # (batch_size, n_experts)

        # Encourage gate to select experts closest to their specialization
        # Lower distance should lead to higher gate weight
        target_gates = F.softmax(-distances, dim=-1)

        # KL divergence between actual and target gate weights (with much stronger weight)
        specialization_loss = 50.0 * F.kl_div(
            F.log_softmax(gate_weights, dim=-1), target_gates, reduction="batchmean"
        )

        return specialization_loss

    def get_expert_constraints(self, expert_idx: int) -> Dict[str, float]:
        """
        Get constraint targets for a specific expert.

        Args:
            expert_idx: Index of the expert

        Returns:
            Dict mapping objective names to constraint target rewards (empty if no constraints)
        """
        if self.expert_specialization != "adaptive_constraints" or not hasattr(
            self, "constraint_targets"
        ):
            return {}

        # Pure objective experts (first n_objectives) have no constraints
        if expert_idx < self.n_objectives:
            return {}

        # Balanced expert (last one) has no constraints
        if expert_idx == self.n_experts - 1:
            return {}

        # Constraint experts
        constraint_expert_idx = expert_idx - self.n_objectives
        objective_names = list(self.constraint_targets.keys())

        # Map constraint expert index to objective and target
        targets_per_obj = (
            len(next(iter(self.constraint_targets.values())))
            if self.constraint_targets
            else 0
        )
        if targets_per_obj == 0:
            return {}

        obj_idx = constraint_expert_idx // targets_per_obj
        target_idx = constraint_expert_idx % targets_per_obj

        if obj_idx < len(objective_names):
            obj_name = objective_names[obj_idx]
            if target_idx < len(self.constraint_targets[obj_name]):
                return {obj_name: self.constraint_targets[obj_name][target_idx]}

        return {}

    def _get_expert_activation_mask(
        self, objective_weights: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Determine which experts should be active based on objective weights.

        Args:
            objective_weights: (batch_size, n_objectives) current objective weights
            threshold: distance threshold for expert activation (increased from 0.3)

        Returns:
            active_mask: (n_experts,) boolean mask of active experts
        """
        device = objective_weights.device

        # Move expert regions to correct device
        expert_regions = torch.stack(
            [region.to(device) for region in self.expert_regions]
        )  # (n_experts, n_objectives)

        # Compute distances from current weights to each expert's region (use first sample if batch)
        current_weights = (
            objective_weights[0] if objective_weights.size(0) > 0 else objective_weights
        )
        distances = torch.norm(expert_regions - current_weights, dim=1)  # (n_experts,)

        # Debug: Print expert regions and current weights occasionally
        if torch.rand(1).item() < 0.01:  # 1% chance to print debug info
            print(f"Expert regions: {expert_regions.cpu().numpy()}")
            print(f"Current weights: {current_weights.cpu().numpy()}")
            print(f"Distances: {distances.cpu().numpy()}")
            print(
                f"Active experts: {(distances <= threshold).sum().item()}/{len(distances)}"
            )

        # Use softer masking: instead of hard threshold, use top-k experts
        # Always activate the top 3 closest experts to ensure learning continues
        k = min(3, len(distances))
        _, top_k_indices = torch.topk(-distances, k)  # Get k closest experts

        active_mask = torch.zeros_like(distances, dtype=torch.bool)
        active_mask[top_k_indices] = True

        return active_mask

    def get_load_balancing_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage using all experts.

        Args:
            gate_weights: (batch_size, n_experts) gate probabilities

        Returns:
            load_balance_loss: scalar tensor
        """
        # Average gate weights across batch
        avg_gate_weights = gate_weights.mean(dim=0)  # (n_experts,)

        # Encourage uniform distribution across experts
        target = torch.ones_like(avg_gate_weights) / self.n_experts

        # L2 loss between average usage and uniform
        load_balance_loss = F.mse_loss(avg_gate_weights, target)

        return load_balance_loss

    def forward_experts(self, *args, **kwargs):
        """Forward pass through all experts. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward_experts")

    def forward(
        self,
        state: torch.Tensor,
        objective_weights: torch.Tensor,
        layer_number: Optional[torch.Tensor] = None,
        material: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through mixture of experts.

        Args:
            state: Input state
            objective_weights: Objective weight vector for gating
            layer_number: Optional layer number
            material: Optional material information

        Returns:
            Mixed expert outputs plus auxiliary losses
        """
        # Get gating weights
        gate_weights, gate_logits = self.gating_network(objective_weights)

        # Forward through all experts
        expert_outputs = self.forward_experts(
            state, layer_number, material, objective_weights
        )

        # Mix expert outputs based on gate weights
        mixed_output = self.mix_expert_outputs(expert_outputs, gate_weights)

        # Compute auxiliary losses
        specialization_loss = self.get_expert_specialization_loss(
            objective_weights, gate_weights
        )
        load_balance_loss = self.get_load_balancing_loss(gate_weights)

        # Store auxiliary information for training
        aux_info = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "specialization_loss": specialization_loss,
            "load_balance_loss": load_balance_loss,
            "total_aux_loss": specialization_loss
            + self.load_balancing_weight * load_balance_loss,
        }

        return mixed_output, aux_info

    def mix_expert_outputs(self, expert_outputs, gate_weights):
        """Mix expert outputs. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement mix_expert_outputs")


class MoEDiscretePolicy(MixtureOfExpertsBase):
    """
    Mixture of Experts Discrete Policy Network.
    Each expert specializes in different objective weight regions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim_discrete: int,
        hidden_dim: int,
        n_layers: int = 2,
        lower_bound: float = 0,
        upper_bound: float = 1,
        include_layer_number: bool = False,
        activation: str = "relu",
        n_objectives: int = 2,
        n_experts: int = 5,
        expert_specialization: str = "weight_regions",
        gate_hidden_dim: int = 64,
        gate_temperature: float = 1.0,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        super().__init__(
            n_experts,
            n_objectives,
            expert_specialization,
            gate_hidden_dim,
            gate_temperature,
            expert_dropout,
            load_balancing_weight,
        )

        # Create expert networks - each is a standard discrete policy
        self.experts = nn.ModuleList(
            [
                DiscretePolicy(
                    input_dim=input_dim,
                    output_dim_discrete=output_dim_discrete,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    include_layer_number=include_layer_number,
                    activation=activation,
                    n_objectives=0,  # Experts don't take objective weights as input
                    use_hyper_networks=False,  # Individual experts use standard networks
                    hyper_hidden_dim=hyper_hidden_dim,
                    hyper_n_layers=hyper_n_layers,
                )
                for _ in range(n_experts)
            ]
        )

    def forward_experts(
        self,
        state: torch.Tensor,
        layer_number: Optional[torch.Tensor] = None,
        material: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None,
    ):
        """Forward pass through expert networks with soft specialization weighting."""
        expert_outputs = []

        for expert in self.experts:
            # All experts forward - let the gating network and specialization loss handle routing
            output = expert.forward(state, layer_number, material, None)
            expert_outputs.append(output)

        return expert_outputs

    def mix_expert_outputs(
        self, expert_outputs: List[torch.Tensor], gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix discrete policy outputs (probability distributions).

        Args:
            expert_outputs: List of (batch_size, output_dim) probability tensors
            gate_weights: (batch_size, n_experts) gating weights

        Returns:
            mixed_probs: (batch_size, output_dim) mixed probability distribution
        """
        # Stack expert outputs: (batch_size, n_experts, output_dim)
        stacked_outputs = torch.stack(expert_outputs, dim=1)

        # Weighted mixture: (batch_size, output_dim)
        mixed_probs = torch.sum(stacked_outputs * gate_weights.unsqueeze(-1), dim=1)

        # Ensure probabilities sum to 1 (numerical stability)
        mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)

        return mixed_probs

    def forward(
        self,
        state: torch.Tensor,
        objective_weights: torch.Tensor,
        layer_number: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with signature matching how the agent calls it.

        Args:
            state: Input state
            objective_weights: Objective weights for gating
            layer_number: Optional layer number

        Returns:
            Tuple of (probabilities, aux_info) for MoE
        """
        if objective_weights is None:
            raise ValueError("MoEDiscretePolicy requires objective_weights for gating")

        # Call parent forward with correct argument order
        mixed_probs, aux_info = super().forward(
            state, objective_weights, layer_number, None
        )

        return mixed_probs, aux_info


class MoEContinuousPolicy(MixtureOfExpertsBase):
    """
    Mixture of Experts Continuous Policy Network.
    Each expert specializes in different objective weight regions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim_continuous: int,
        hidden_dim: int,
        n_layers: int = 2,
        lower_bound: float = 0.1,
        upper_bound: float = 1.0,
        include_layer_number: bool = False,
        include_material: bool = False,
        activation: str = "relu",
        n_objectives: int = 2,
        n_experts: int = 5,
        expert_specialization: str = "weight_regions",
        gate_hidden_dim: int = 64,
        gate_temperature: float = 1.0,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        super().__init__(
            n_experts,
            n_objectives,
            expert_specialization,
            gate_hidden_dim,
            gate_temperature,
            expert_dropout,
            load_balancing_weight,
        )

        # Create expert networks - each is a standard continuous policy
        self.experts = nn.ModuleList(
            [
                ContinuousPolicy(
                    input_dim=input_dim,
                    output_dim_continuous=output_dim_continuous,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    include_layer_number=include_layer_number,
                    include_material=include_material,
                    activation=activation,
                    n_objectives=0,  # Experts don't take objective weights as input
                    use_hyper_networks=False,  # Individual experts use standard networks
                    hyper_hidden_dim=hyper_hidden_dim,
                    hyper_n_layers=hyper_n_layers,
                )
                for _ in range(n_experts)
            ]
        )

    def forward_experts(
        self,
        state: torch.Tensor,
        layer_number: Optional[torch.Tensor] = None,
        material: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None,
    ):
        """Forward pass through expert networks with soft specialization weighting."""
        expert_outputs = []

        for expert in self.experts:
            # All experts forward - let the gating network and specialization loss handle routing
            mean, log_std = expert.forward(state, layer_number, material, None)
            expert_outputs.append((mean, log_std))

        return expert_outputs

    def mix_expert_outputs(
        self,
        expert_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
        gate_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix continuous policy outputs (Gaussian parameters).

        Args:
            expert_outputs: List of (mean, log_std) tuples from experts
            gate_weights: (batch_size, n_experts) gating weights

        Returns:
            mixed_mean: (batch_size, output_dim) mixed mean
            mixed_log_std: (batch_size, output_dim) mixed log standard deviation
        """
        # Separate means and log_stds
        expert_means = torch.stack(
            [output[0] for output in expert_outputs], dim=1
        )  # (batch_size, n_experts, output_dim)
        expert_log_stds = torch.stack(
            [output[1] for output in expert_outputs], dim=1
        )  # (batch_size, n_experts, output_dim)

        # Weighted mixture of means
        mixed_mean = torch.sum(expert_means * gate_weights.unsqueeze(-1), dim=1)

        # For log_std, we need to be more careful - mix the variances, not log_stds
        expert_vars = torch.exp(2 * expert_log_stds)  # Convert to variances
        mixed_var = torch.sum(expert_vars * gate_weights.unsqueeze(-1), dim=1)
        mixed_log_std = 0.5 * torch.log(mixed_var)  # Convert back to log_std

        return mixed_mean, mixed_log_std

    def forward(
        self,
        state: torch.Tensor,
        objective_weights: torch.Tensor,
        layer_number: Optional[torch.Tensor] = None,
        material: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with signature matching how the agent calls it.

        Args:
            state: Input state
            objective_weights: Objective weights for gating
            layer_number: Optional layer number
            material: Optional material information

        Returns:
            Tuple of (mean, log_std, aux_info) for MoE
        """
        if objective_weights is None:
            raise ValueError(
                "MoEContinuousPolicy requires objective_weights for gating"
            )

        # Call parent forward with correct argument order
        (mixed_mean, mixed_log_std), aux_info = super().forward(
            state, objective_weights, layer_number, material
        )

        return mixed_mean, mixed_log_std, aux_info


class MoEValueNetwork(MixtureOfExpertsBase):
    """
    Mixture of Experts Value Network.
    Each expert specializes in different objective weight regions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        include_layer_number: bool = False,
        activation: str = "relu",
        n_objectives: int = 2,
        n_experts: int = 5,
        expert_specialization: str = "weight_regions",
        gate_hidden_dim: int = 64,
        gate_temperature: float = 1.0,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
        use_hyper_networks: bool = False,
        hyper_hidden_dim: int = 128,
        hyper_n_layers: int = 2,
    ):
        super().__init__(
            n_experts,
            n_objectives,
            expert_specialization,
            gate_hidden_dim,
            gate_temperature,
            expert_dropout,
            load_balancing_weight,
        )

        # Create expert networks - each is a standard value network
        # Note: Create with n_objectives=0 to avoid objective weights as input,
        # but manually adjust output to produce multi-objective values
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            expert = ValueNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                include_layer_number=include_layer_number,
                activation=activation,
                n_objectives=0,  # Don't take objective weights as input
                use_hyper_networks=False,
                hyper_hidden_dim=hyper_hidden_dim,
                hyper_n_layers=hyper_n_layers,
            )

            # Manually replace the output layer to output multi-objective values
            if n_objectives > 1:
                expert.output_layers["value"] = nn.Linear(
                    expert.hidden_dim, n_objectives
                )

            self.experts.append(expert)

    def forward_experts(
        self,
        state: torch.Tensor,
        layer_number: Optional[torch.Tensor] = None,
        material: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None,
    ):
        """Forward pass through expert networks with soft specialization weighting."""
        expert_outputs = []

        for expert in self.experts:
            # All experts forward - let the gating network and specialization loss handle routing
            output = expert.forward(state, layer_number, None)
            expert_outputs.append(output)

        return expert_outputs

    def mix_expert_outputs(
        self, expert_outputs: List[torch.Tensor], gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Mix value network outputs.

        Args:
            expert_outputs: List of (batch_size, n_objectives) value tensors
            gate_weights: (batch_size, n_experts) gating weights

        Returns:
            mixed_values: (batch_size, n_objectives) mixed values
        """
        # Stack expert outputs: (batch_size, n_experts, n_objectives)
        stacked_outputs = torch.stack(expert_outputs, dim=1)

        # Weighted mixture: (batch_size, n_objectives)
        mixed_values = torch.sum(stacked_outputs * gate_weights.unsqueeze(-1), dim=1)

        return mixed_values

    def forward(
        self,
        state: torch.Tensor,
        layer_number: Optional[torch.Tensor] = None,
        objective_weights: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass matching ValueNetwork signature.

        Args:
            state: Input state
            layer_number: Optional layer number
            objective_weights: Objective weights for gating and weighting

        Returns:
            Tuple of (scalar_value, aux_info) for MoE
        """
        if objective_weights is None:
            raise ValueError("MoEValueNetwork requires objective_weights for gating")

        # Get mixed multi-objective values and auxiliary info
        mixed_values, aux_info = super().forward(
            state, objective_weights, layer_number, None
        )

        # Apply objective weighting to get scalar value
        if mixed_values.size(-1) > 1 and objective_weights is not None:
            scalar_value = torch.sum(
                mixed_values * objective_weights, dim=-1, keepdim=True
            )
        else:
            scalar_value = mixed_values

        return scalar_value, aux_info
