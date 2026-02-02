#!/usr/bin/env python3
from pathlib import Path
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
import mlflow
import shutil
import warnings
from datetime import datetime
import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, DataConfig, TrainingConfig, load_config
from coatopt.utils.callbacks import PlottingCallback, EntropyAnnealingCallback
from coatopt.utils.utils import load_materials, evaluate_model
from coatopt.environments.state import CoatingState

class CoatOptDiscreteGymWrapper(gym.Env):
    """Gymnasium wrapper for CoatingEnvironment with discrete actions and sction masking.

    Observation space: Flattened state tensor (max_layers * features_per_layer,)

    Action Masking:
        - Blocks consecutive same material selection
        - Optionally blocks air (material 0) until min_layers_before_air reached
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        materials: dict,
        n_thickness_bins: int = 20,
        constraint_penalty: float = 100.0,
        consecutive_material_penalty: float = 0.2,
        # Action masking settings
        mask_consecutive_materials: bool = True,
        mask_air_until_min_layers: bool = True,
        min_layers_before_air: int = 4,
        air_material_idx: int = 0,
        substrate_material_idx: int = 1,
        # Schedule settings
        epochs_per_step: int = 200,
        steps_per_objective: int = 10,
        constraint_schedule: str = "interleaved",  # "interleaved" or "sequential"
        # Pareto bonus settings
        pareto_dominance_bonus: float = 0.0,
    ):
        super().__init__()
        self.env = CoatingEnvironment(config, materials)
        self.config = config
        self.n_thickness_bins = n_thickness_bins
        self.constraint_schedule = constraint_schedule

        # Action masking settings
        self.mask_consecutive_materials = mask_consecutive_materials
        self.mask_air_until_min_layers = mask_air_until_min_layers
        self.min_layers_before_air = min_layers_before_air
        self.air_material_idx = air_material_idx
        self.substrate_material_idx = substrate_material_idx

        # Precompute thickness values for each bin
        self.thickness_bins = np.linspace(
            self.env.min_thickness,
            self.env.max_thickness,
            n_thickness_bins
        )

        # Multi-objective constraint settings
        self.objectives = list(config.data.optimise_parameters)
        self.constraint_penalty = constraint_penalty

        # Consecutive material penalty (kept as backup, but masking is primary)
        self.consecutive_material_penalty = consecutive_material_penalty
        self.previous_material_idx = None

        # Track current layer for min_layers_before_air masking
        self.current_layer = 0

        self.epochs_per_step = epochs_per_step
        self.warmup_episodes_per_objective = epochs_per_step   # Warmup per objective
        self.total_warmup_episodes = self.warmup_episodes_per_objective * len(self.objectives)

        self.steps_per_objective = steps_per_objective
        self.n_objectives = len(self.objectives)
        self.total_levels = self.steps_per_objective
        self.total_phases = self.total_levels * self.n_objectives
        self.n_anneal_episodes = self.total_phases * self.epochs_per_step

        # Episode counter for scheduling
        self.episode_count = 0

        # Phase tracking
        self.is_warmup = True
        self.warmup_objective_idx = 0

        # Enable constrained training in environment
        self.env.enable_constrained_training(
            warmup_episodes_per_objective=epochs_per_step,
            steps_per_objective=steps_per_objective,
            epochs_per_step=epochs_per_step,
            constraint_penalty=constraint_penalty,
        )

        # Enable pareto dominance bonus if specified
        if pareto_dominance_bonus > 0:
            self.env.enable_pareto_bonus(bonus=pareto_dominance_bonus)

        # Current episode's target and constraints (synced with environment)
        self.target_objective = self.objectives[0]
        self.env.target_objective = self.target_objective
        self.target_objective_idx = 0
        self.current_phase = -1  # Start at -1 to trigger first phase transition
        self.current_level = 1

        # No constraints during warmup (synced with environment)
        self.constraints = {}
        self.env.constraints = {}

        # Observation space
        n_features = 1 + self.env.n_materials + 2
        obs_size = self.env.max_layers * n_features + 1  # +1 for layer number
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # DISCRETE Action space: [material_idx, thickness_bin_idx]
        self.action_space = gym.spaces.MultiDiscrete([
            self.env.n_materials,  # material choices
            n_thickness_bins,      # thickness bin choices
        ])

    def get_thickness_from_bin(self, bin_idx: int) -> float:
        """Convert thickness bin index to actual thickness value."""
        return self.thickness_bins[bin_idx]

    def action_masks(self) -> np.ndarray:
        """Return action masks for MaskablePPO.

        For MultiDiscrete spaces, returns a single concatenated boolean array:
            [material_mask (n_materials), thickness_mask (n_thickness_bins)]

        Total shape: (n_materials + n_thickness_bins,)

        Masking rules:
            1. Block consecutive same material (if enabled)
            2. Block air material until min_layers_before_air reached (if enabled)
        """
        # Start with all actions valid
        material_mask = np.ones(self.env.n_materials, dtype=bool)
        thickness_mask = np.ones(self.n_thickness_bins, dtype=bool)

        # Rule 1: Block consecutive same material
        if self.mask_consecutive_materials and self.previous_material_idx is not None:
            material_mask[self.previous_material_idx] = False

        # Rule 2: Block air until minimum layers reached
        if self.mask_air_until_min_layers and self.current_layer < self.min_layers_before_air:
            material_mask[self.air_material_idx] = False

        # Safety: ensure at least one material is valid
        if not material_mask.any():
            material_mask[:] = True  # Fall back to all valid

        # Concatenate masks for MultiDiscrete space
        return np.concatenate([material_mask, thickness_mask])

    def _get_obs(self, state) -> np.ndarray:
        """Convert CoatingState to fixed-size numpy array with layer number."""
        tensor = state.get_observation_tensor(pre_type="lstm")
        obs = tensor.numpy().flatten().astype(np.float32)

        # Append normalized layer number so agent knows its position in episode
        layer_number_normalized = self.current_layer / self.env.max_layers
        obs_with_layer = np.append(obs, layer_number_normalized)

        return obs_with_layer

    def _get_annealing_progress(self) -> float:
        """Get annealing progress from 0.0 (start) to 1.0 (fully annealed)."""
        return min(1.0, self.episode_count / self.n_anneal_episodes)

    def reset(self, seed=None, options=None):
        """Reset with two-phase training: warmup then constrained cycling."""
        super().reset(seed=seed)
        state = self.env.reset()

        # Reset material and layer tracking for action masking
        self.previous_material_idx = None
        self.current_layer = 0

        # Increment episode counter
        self.episode_count += 1

        # === PHASE 1: WARMUP ===
        # Optimize each objective individually to discover achievable bounds
        if self.episode_count <= self.total_warmup_episodes:
            self.is_warmup = True
            # Determine which objective we're warming up
            self.warmup_objective_idx = (self.episode_count - 1) // self.warmup_episodes_per_objective
            self.warmup_objective_idx = min(self.warmup_objective_idx, self.n_objectives - 1)
            self.target_objective = self.objectives[self.warmup_objective_idx]
            self.env.target_objective = self.target_objective
            self.constraints = {}
            self.env.constraints = {}

            progress = self.episode_count / self.total_warmup_episodes
            return self._get_obs(state), {
                "target": self.target_objective,
                "constraints": {},
                "annealing_progress": 0.0,
                "episode": self.episode_count,
                "phase": f"warmup_{self.target_objective}",
                "level": 0,
                "is_warmup": True,
            }

        # === PHASE 2: CONSTRAINED CYCLING ===
        # Transition from warmup to constrained phase
        if self.is_warmup:
            self.is_warmup = False
            self.env.is_warmup = False
            print(f"\n=== WARMUP COMPLETE ===")
            print(f"Observed value bounds (phase 1): {self.env.observed_value_bounds}")
            print(f"Best normalised rewards during warmup (phase 1): {self.env.warmup_best_rewards}")
            for obj in self.objectives:
                print(f"  {obj}: [0.0, {self.env.warmup_best_rewards[obj]:.4f}]")
            print(f"=== STARTING CONSTRAINED PHASE ===\n")

        # Episode count relative to end of warmup
        constrained_episode = self.episode_count - self.total_warmup_episodes
        new_phase = (constrained_episode - 1) // self.epochs_per_step

        # Calculate objective indices and constraint level
        if self.constraint_schedule == "interleaved":
            # cycle between objectives every epochs_per_step
            target_idx = new_phase % self.n_objectives
            level_cycle = (new_phase // self.n_objectives) % self.total_levels
            current_level = level_cycle + 1  # Start at level 1 (first constraint), not 0
            constrained_idx = None  

        elif self.constraint_schedule == "sequential":
            # complete all levels for one objective before switching
            # Offset by 1 to alternate properly from warmup (which ends with absorption)
            cycle_length = self.total_levels * self.n_objectives
            cycle_phase = new_phase % cycle_length
            constrained_idx = (cycle_phase // self.total_levels + 1) % self.n_objectives
            current_level = (cycle_phase % self.total_levels) + 1
            target_idx = (constrained_idx + 1) % self.n_objectives

        else:
            raise ValueError(f"Unknown constraint_schedule: {self.constraint_schedule}")

        # Update target objective only when it changes
        new_target = self.objectives[target_idx]
        if new_target != self.target_objective:
            self.target_objective = new_target
            self.env.target_objective = self.target_objective

        # Update phase tracking and constraints when entering a new phase
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.current_level = current_level

            # Set constraints in normalised [0, 1] space with randomization
            self.constraints = {}

            for i, obj in enumerate(self.objectives):
                # In sequential mode: constrain the constrained_idx objective
                # In interleaved mode: constrain all objectives except target
                if self.constraint_schedule == "sequential":
                    should_constrain = (i == constrained_idx)
                else:  # interleaved
                    should_constrain = (i != target_idx)

                if should_constrain:
                    step_fraction = min(1.0, self.current_level / self.total_levels)
                    # Scale constraint by best achieved during warmup (phase 1)
                    max_achievable = self.env.warmup_best_rewards[obj]
                    max_constraint = step_fraction * max_achievable

                    # Randomize constraint between 0.0 and step threshold
                    self.constraints[obj] = max_constraint#np.random.uniform(0.0, max_constraint)
                    print(f"  Constraint {obj}: threshold={max_constraint:.4f} (level {self.current_level}/{self.total_levels}, warmup_best={max_achievable:.4f})")

            # Sync constraints with environment
            self.env.constraints = self.constraints

        progress = min(1.0, constrained_episode / (self.total_phases * self.epochs_per_step))
        return self._get_obs(state), {
            "target": self.target_objective,
            "constraints": self.constraints,
            "annealing_progress": progress,
            "episode": self.episode_count,
            "phase": self.current_phase,
            "level": self.current_level,
            "is_warmup": False,
        }

    def step(self, action):
        """Take discrete action, apply constraint penalties at episode end."""
        # Decode discrete action
        material_idx = int(action[0])
        thickness_bin = int(action[1])
        thickness = self.get_thickness_from_bin(thickness_bin)
        
        # DEBUG: Print first few actions
        if self.current_layer < 3 and self.episode_count < 5:
            print(f"Episode {self.episode_count}, Layer {self.current_layer}: "
                  f"action={action}, material={material_idx}, thickness_bin={thickness_bin}, thickness={thickness:.4f}")

        # Check for consecutive same material penalty (backup if masking fails)
        consecutive_penalty = 0.0
        if self.previous_material_idx is not None and material_idx == self.previous_material_idx:
            consecutive_penalty = self.consecutive_material_penalty

        # Update tracking for action masking
        self.previous_material_idx = material_idx
        self.current_layer += 1

        # Build CoatOpt action
        coatopt_action = np.zeros(self.env.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + material_idx] = 1.0

        # Step environment
        state, rewards, terminated, finished, env_reward, _, vals = self.env.step(
            coatopt_action
        )

        obs = self._get_obs(state)
        done = finished
        truncated = False

        # Default: no reward for intermediate steps
        total_reward = 0.0 - consecutive_penalty
        info = {}

        # Only populate info at episode end
        if done:
            total_reward = env_reward - consecutive_penalty

            # Build info only at terminal step
            info = {
                "rewards": rewards,
                "vals": vals,
                "finished": finished,
                "target": self.target_objective,
                "constraints": self.constraints,
                "consecutive_penalty": consecutive_penalty,
                "annealing_progress": self._get_annealing_progress(),
                "state_array": state.get_array(),
                "constrained_reward": total_reward,
                "episode": {'r': total_reward, 'l': self.current_layer, 't': 0},
                "is_warmup": self.is_warmup,
                # Discrete action info
                "material_idx": material_idx,
                "thickness_bin": thickness_bin,
                "thickness": thickness,
                "current_layer": self.current_layer,
            }

            # Debug: print episode results with rewards and constraints
            phase_str = "WARMUP" if self.is_warmup else f"P{self.current_phase}/L{self.current_level}"
            r_reward = rewards.get('reflectivity', 0.0)
            a_reward = rewards.get('absorption', 0.0)
            r_constr = f"{self.constraints['reflectivity']:.3f}" if 'reflectivity' in self.constraints else "none"
            a_constr = f"{self.constraints['absorption']:.3f}" if 'absorption' in self.constraints else "none"
            constr_str = f"C[R≥{r_constr}, A≥{a_constr}]"
            print(f"Ep{self.episode_count:4d} {phase_str:8s} target={self.target_objective[:4]:4s} | R_rew={r_reward:.4f} A_rew={a_reward:.4f} | {constr_str} | total={total_reward:.4f}")

        return obs, float(total_reward), done, truncated, info

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Custom LSTM feature extractor for sequential coating layer processing.

    This processes the flattened observation (max_layers * features_per_layer)
    by reshaping it back into a sequence and running it through an LSTM.

    Args:
        observation_space: Gym observation space
        lstm_hidden_size: Hidden size of LSTM
        lstm_num_layers: Number of LSTM layers
        features_dim: Output dimension 
        max_layers: Maximum number of coating layers
        features_per_layer: Number of features per layer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        features_dim: int = 128,
        max_layers: int = 20,
        features_per_layer: int = None,
    ):
        # Features dim is what we output to the policy/value networks
        super().__init__(observation_space, features_dim)

        self.max_layers = max_layers

        # Calculate features per layer from observation space
        obs_size = observation_space.shape[0]
        if features_per_layer is None:
            features_per_layer = obs_size // max_layers
        self.features_per_layer = features_per_layer

        # LSTM processes sequences of shape (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=features_per_layer,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Project LSTM output + layer number to desired feature dimension
        # +1 for layer number concatenated after LSTM
        self.linear = nn.Linear(lstm_hidden_size + 1, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Process observations through LSTM.

        Args:
            observations: Shape (batch_size, max_layers * features_per_layer)

        Returns:
            features: Shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Extract layer number (last element) and sequence (everything else)
        layer_number = observations[:, -1:]  # (batch, 1) - keep dimension
        sequence_obs = observations[:, :-1]  # (batch, max_layers * features_per_layer)

        # Reshape flat observations back to sequence
        # (batch, max_layers * features_per_layer) -> (batch, max_layers, features_per_layer)
        sequences = sequence_obs.view(batch_size, self.max_layers, self.features_per_layer)

        # Run through LSTM
        # lstm_out shape: (batch, max_layers, lstm_hidden_size)
        # h_n shape: (num_layers, batch, lstm_hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(sequences)

        # Use the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)

        # Concatenate layer number to LSTM output
        combined = th.cat([last_output, layer_number], dim=1)  # (batch, lstm_hidden_size + 1)

        # Project to features_dim
        features = th.relu(self.linear(combined))  # (batch, features_dim)

        return features

class DiscreteActionPlottingCallback(PlottingCallback):
    """Extended callback with alternating materials plotting for discrete actions.

    Inherits all plotting from PlottingCallback and adds discrete-specific plots.
    """

    def __init__(
        self,
        env,
        plot_freq: int = 5000,
        design_plot_freq: int = 100,
        save_dir: str = ".",
        n_best_designs: int = 5,
        materials: dict = None,
        verbose: int = 0,
        disable_mlflow: bool = True,
    ):
        super().__init__(
            env=env,
            plot_freq=plot_freq,
            design_plot_freq=design_plot_freq,
            save_dir=save_dir,
            n_best_designs=n_best_designs,
            materials=materials,
            verbose=verbose,
            disable_mlflow=disable_mlflow,
        )

    def _plot_alternating_materials(self):
        """Plot coating stack designs for alternating material combinations."""
        import matplotlib.pyplot as plt

        if self.env is None:
            return

        # Material combinations to test: (mat1, mat2, label)
        # Based on typical materials: 1=SiO2, 2=Ti:Ta2O5, 3=aSi
        combinations = [
            (1, 2, "SiO2 / Ti:Ta2O5"),
            (1, 3, "SiO2 / aSi"),
            (2, 3, "Ti:Ta2O5 / aSi"),
        ]

        # Create alternating designs
        designs = []
        n_layers = 20
        avg_thickness1 = 0.25
        avg_thickness2 = 0.1

        for avg_thickness in [avg_thickness1, avg_thickness2]:
            for mat1, mat2, label in combinations:
                # Check if materials exist
                if mat1 not in self.materials or mat2 not in self.materials:
                    continue

                # Create state array in one-hot format: [thickness, material_onehot, n, k]
                state_array = np.zeros((n_layers, 1 + len(self.materials) + 2))
                for i in range(n_layers):
                    material_idx = mat1 if i % 2 == 0 else mat2
                    state_array[i, 0] = avg_thickness
                    state_array[i, 1 + material_idx] = 1.0

                # Create CoatingState from the internal (n_layers, 2) format
                # Column 0: thickness, Column 1: material_index
                internal_state = np.zeros((n_layers, 2))
                for i in range(n_layers):
                    material_idx = mat1 if i % 2 == 0 else mat2
                    internal_state[i, 0] = avg_thickness
                    internal_state[i, 1] = material_idx

                coating_state = CoatingState.from_array(
                    internal_state,
                    len(self.materials),
                    self.env.env.air_material_index,
                    self.env.env.substrate_material_index,
                    self.materials,
                )

                reflectivity, thermal_noise, absorption, total_thickness = (
                    self.env.env.compute_state_value(coating_state)
                )

                # Compute normalised rewards using environment's compute_reward
                # Temporarily save warmup state to prevent test designs from corrupting warmup_best
                saved_warmup_best = self.env.env.warmup_best_rewards.copy()
                rewards, vals = self.env.env.compute_reward(coating_state, normalised=True)
                self.env.env.warmup_best_rewards = saved_warmup_best  # Restore
                ref_reward = rewards.get('reflectivity', 0.0)
                abs_reward = rewards.get('absorption', 0.0)
                total_reward = ref_reward + abs_reward

                vals = {
                    'reflectivity': reflectivity,
                    'absorption': absorption,
                    'total_reward': total_reward,
                }

                designs.append({
                    'state_array': state_array,  # Use one-hot format for plotting
                    'vals': vals,
                    'label': label,
                })

        if not designs:
            return

        # Plot like _plot_best_designs
        n_designs = len(designs)
        fig, axs = plt.subplots(1, n_designs, figsize=(4 * n_designs, 6), squeeze=False)

        for i, design in enumerate(designs):
            ax = axs[0, i]
            self._plot_single_stack(ax, design["state_array"], design["vals"], rank=design["label"])

        plt.tight_layout()
        save_path = self.save_dir / "alternating_materials_designs.png"
        plt.savefig(save_path)
        plt.close(fig)


def train(config_path: str, save_dir: str = None):
    """Train MaskablePPO with DISCRETE actions and ACTION MASKING on CoatOpt.

    Args:
        config_path: Path to config INI file
        save_dir: Directory to save results (if None, will create from config)

    Returns:
        Trained MaskablePPO model
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    materials_path = parser.get('General', 'materials_path')

    # If save_dir not provided, create it 
    if save_dir is None:
        base_save_dir = parser.get('General', 'save_dir')
        run_name = parser.get('General', 'run_name', fallback='')
        date_str = datetime.now().strftime("%Y%m%d")
        algorithm_name = "sb3_discrete"
        if run_name:
            run_dir_name = f"{date_str}-{algorithm_name}-{run_name}"
        else:
            run_dir_name = f"{date_str}-{algorithm_name}"
        save_dir = Path(base_save_dir) / run_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(save_dir)

    # [sb3_discrete] or [sb3_discrete_lstm] 
    section = 'sb3_discrete_lstm' if parser.has_section('sb3_discrete_lstm') else 'sb3_discrete'

    total_timesteps = parser.getint(section, 'total_timesteps')
    n_thickness_bins = parser.getint(section, 'n_thickness_bins')
    verbose = parser.getint(section, 'verbose')
    mask_consecutive_materials = parser.getboolean(section, 'mask_consecutive_materials')
    mask_air_until_min_layers = parser.getboolean(section, 'mask_air_until_min_layers')
    min_layers_before_air = parser.getint(section, 'min_layers_before_air')
    epochs_per_step = parser.getint(section, 'epochs_per_step')
    steps_per_objective = parser.getint(section, 'steps_per_objective')

    # Constraint and entropy settings (with fallbacks)
    constraint_penalty = parser.getfloat(section, 'constraint_penalty', fallback=10.0)
    max_entropy = parser.getfloat(section, 'max_entropy', fallback=0.2)
    min_entropy = parser.getfloat(section, 'min_entropy', fallback=0.01)
    pareto_dominance_bonus = parser.getfloat(section, 'pareto_dominance_bonus', fallback=0.0)
    adaptive_entropy_to_constraints = parser.getboolean(section, 'adaptive_entropy_to_constraints', fallback=False)

    # [Data] section
    n_layers = parser.getint('Data', 'n_layers')
    constraint_schedule = parser.get('Data', 'constraint_schedule', fallback='interleaved').strip('"').strip("'")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if materials_path is None:
        materials_path = Path(__file__).parent / "config" / "materials.json"

    materials = load_materials(str(materials_path))
    print(f"Loaded {len(materials)} materials from {materials_path}")

    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    config.data.n_layers = n_layers

    env = CoatOptDiscreteGymWrapper(
        config,
        materials,
        n_thickness_bins=n_thickness_bins,
        constraint_penalty=constraint_penalty,
        # Action masking
        mask_consecutive_materials=mask_consecutive_materials,
        mask_air_until_min_layers=mask_air_until_min_layers,
        min_layers_before_air=min_layers_before_air,
        # Schedule
        epochs_per_step=epochs_per_step,
        steps_per_objective=steps_per_objective,
        constraint_schedule=constraint_schedule,
        # Pareto bonus
        pareto_dominance_bonus=pareto_dominance_bonus,
    )

    tb_log = None

    # Use MaskablePPO for action masking support
    algo_config = config.algorithm
    if algo_config.pre_network == "mlp":
        policy_kwargs = dict(
            net_arch=dict(
                pi=algo_config.net_arch_pi,
                vf=algo_config.net_arch_vf,
            ),
            # activation_fn=th.nn.ReLU, 
        )
    elif algo_config.pre_network == "lstm":
        # LSTM-specific settings (with defaults)
        lstm_hidden_size = parser.getint(section, 'lstm_hidden_size', fallback=128)
        lstm_num_layers = parser.getint(section, 'lstm_num_layers', fallback=2)
        lstm_features_dim = parser.getint(section, 'lstm_features_dim', fallback=128)
        n_features_per_layer = 1 + env.env.n_materials + 2  # thickness + materials_onehot + n + k
        policy_kwargs = dict(
            features_extractor_class=LSTMFeatureExtractor,
            features_extractor_kwargs=dict(
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                features_dim=lstm_features_dim,
                max_layers=env.env.max_layers,
                features_per_layer=n_features_per_layer,
            ),
            net_arch=dict(
                pi=algo_config.net_arch_pi,
                vf=algo_config.net_arch_vf,
            ),
        )

    model = MaskablePPO(
        "MlpPolicy",
        policy_kwargs=policy_kwargs,
        learning_rate=algo_config.learning_rate,
        n_steps=algo_config.n_steps,
        batch_size=algo_config.batch_size,
        n_epochs=algo_config.n_epochs,
        gamma=algo_config.gamma,
        gae_lambda=algo_config.gae_lambda,
        clip_range=algo_config.clip_range,
        ent_coef=algo_config.ent_coef,  # Initial value (will be updated by callback if entropy annealing is used)
        vf_coef=algo_config.vf_coef,
        max_grad_norm=algo_config.max_grad_norm,
        verbose=0,
        tensorboard_log=tb_log,
        env = env,
    )

    entropy_callback = EntropyAnnealingCallback(
        max_ent=max_entropy,  # High exploration at start of each cycle
        min_ent=min_entropy,  # Low exploration at end of each cycle
        epochs_per_step=epochs_per_step,  # Reset annealing every N episodes
        verbose=0,
        adaptive_to_constraints=adaptive_entropy_to_constraints,
        constraint_window=50,  # Track last 50 episodes
    )

    plotting_callback = DiscreteActionPlottingCallback(
        env=env,
        plot_freq=500,  # Reduced for shorter episodes
        design_plot_freq=50,
        save_dir=str(save_dir),
        n_best_designs=5,
        materials=materials,
        verbose=verbose,
        disable_mlflow=config.general.disable_mlflow,
    )

    print(f"\nStarting training for {total_timesteps} timesteps...")
    # Combine callbacks
    callbacks = CallbackList([entropy_callback, plotting_callback])
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model_path = save_dir / "coatopt_ppo_discrete"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    print("\nRunning final evaluation..")
    evaluate_model(model, env, n_episodes=10, use_action_masks=True)

    # Save Pareto front to CSV
    print("\nSaving Pareto front...")
    pareto_csv = save_dir / "pareto_front.csv"
    plotting_callback.save_pareto_front_to_csv(str(pareto_csv))

    # Log FINAL Pareto front to MLflow (only once, not every epoch)
    if not config.general.disable_mlflow and mlflow.active_run():
        print("Logging Pareto front to MLflow.....")

        # Log the CSV as artifact
        mlflow.log_artifact(str(pareto_csv))

        # Log Pareto front as a table (queryable in MLflow)
        import pandas as pd
        pareto_df = pd.read_csv(pareto_csv)
        mlflow.log_table(pareto_df, artifact_file="pareto_front_table.json")

        # Log summary metrics (for easy comparison)
        mlflow.log_metric("final_pareto_size", len(pareto_df))
        if len(pareto_df) > 0:
            for obj in config.data.optimise_parameters:
                if obj in pareto_df.columns:
                    mlflow.log_metric(f"pareto_best_{obj}", pareto_df[obj].max())
                    mlflow.log_metric(f"pareto_worst_{obj}", pareto_df[obj].min())

        # Log Pareto plots
        for plot_file in save_dir.glob("pareto*.png"):
            mlflow.log_artifact(str(plot_file))

    return model



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SB3 PPO (Discrete) on CoatOpt")
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--layers", type=int, default=20, help="Number of coating layers"
    )
    parser.add_argument(
        "--thickness-bins", type=int, default=20, help="Number of discrete thickness bins"
    )
    parser.add_argument(
        "--materials", type=str, default=None, help="Path to materials JSON"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./sb3_discrete_output", help="Output directory"
    )

    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--target-reflectivity", type=float, default=0.99,
        help="Target reflectivity constraint (tightest, at end of annealing)"
    )
    parser.add_argument(
        "--target-absorption", type=float, default=1,
        help="Target absorption constraint in ppm (tightest, at end of annealing)"
    )

    # Action masking arguments
    parser.add_argument(
        "--no-mask-consecutive", action="store_true",
        help="Disable masking of consecutive same material selection"
    )
    parser.add_argument(
        "--no-mask-air", action="store_true",
        help="Disable masking of air material until min layers"
    )
    parser.add_argument(
        "--min-layers-before-air", type=int, default=4,
        help="Minimum layers before air can be selected (default: 4)"
    )
    parser.add_argument(
        "--epochs-per-step", type=int, default=200,
        help="Episodes per phase before switching target objective (default: 200)"
    )

    parser.add_argument(
        "--steps-per-objective", type=int, default=10,
        help="Number of objective cycles before increasing annealing level (default: 10)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config INI file (optional, uses defaults if not provided)"
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_layers=args.layers,
        n_thickness_bins=args.thickness_bins,
        materials_path=args.materials,
        save_dir=args.save_dir,
        verbose=args.verbose,
        target_absorption=args.target_absorption,
        # Action masking
        mask_consecutive_materials=not args.no_mask_consecutive,
        mask_air_until_min_layers=not args.no_mask_air,
        min_layers_before_air=args.min_layers_before_air,
        # Schedule
        epochs_per_step=args.epochs_per_step,
        steps_per_objective=args.steps_per_objective,
        # Config
        config_path=args.config,
    )
