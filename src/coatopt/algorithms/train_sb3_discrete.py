#!/usr/bin/env python3
from pathlib import Path
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, DataConfig, TrainingConfig, load_config
from coatopt.utils.callbacks import PlottingCallback
from coatopt.utils.utils import load_materials, evaluate_model, EntropyAnnealingCallback


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
        target_constraint_bounds: dict = None,
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
        self.target_constraint_bounds = target_constraint_bounds or {
            "reflectivity": 0.99999999,
            "absorption": 0.1,
        }

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

        # Current episode's target and constraints (synced with environment)
        self.target_objective = self.objectives[0]
        self.env.target_objective = self.target_objective
        self.target_objective_idx = 0
        self.current_phase = 0
        self.current_level = 1

        # No constraints during warmup (synced with environment)
        self.constraints = {}
        self.env.constraints = {}

        # Observation space
        n_features = 1 + self.env.n_materials + 2
        obs_size = self.env.max_layers * n_features
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
        """Convert CoatingState to fixed-size numpy array."""
        tensor = state.get_observation_tensor(pre_type="lstm")
        return tensor.numpy().flatten().astype(np.float32)

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
            print(f"\nConstraint ranges will be scaled by warmup best:")
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
            level_cycle = (new_phase // self.n_objectives) % (self.total_levels + 1)
            current_level = level_cycle  
            constrained_idx = None  

        elif self.constraint_schedule == "sequential":
            # complete all levels for one objective before switching
            cycle_length = self.total_levels * self.n_objectives  
            cycle_phase = new_phase % cycle_length  
            constrained_idx = cycle_phase // self.total_levels 
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
        if new_phase != self.current_phase or self.current_phase == 0:
            self.current_phase = new_phase
            self.current_level = current_level

            # Set constraints in normalised [0, 1] space with randomization
            self.constraints = {}

            # In interleaved mode with level 0, no constraints (restart cycle)
            if self.constraint_schedule == "interleaved" and self.current_level == 0:
                print(f"\n=== CONSTRAINT CYCLE COMPLETE - Restarting with NO constraints (Phase {new_phase}) ===\n")
            else:
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
        state, rewards, terminated, finished, total_reward, _, vals = self.env.step(
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
            # Environment handles all reward computation and tracking
            total_reward = self.env._compute_training_reward(vals) - consecutive_penalty

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

            # Debug: print episode results
            phase_str = f"WARMUP[{self.target_objective}]" if self.is_warmup else f"CONSTR[{self.target_objective}]"
            print(f"{phase_str} R={vals.get('reflectivity', 0):.4f}, A={vals.get('absorption', 0):.1f}, reward={total_reward:.3f}")

        return obs, float(total_reward), done, truncated, info


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
    ):
        super().__init__(
            env=env,
            plot_freq=plot_freq,
            design_plot_freq=design_plot_freq,
            save_dir=save_dir,
            n_best_designs=n_best_designs,
            materials=materials,
            verbose=verbose,
            track_action_distributions=True,  # Enable action distribution tracking
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

                from coatopt.environments.core.state_simple import CoatingState

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
                _, vals, rewards = self.env.env.compute_reward(coating_state)
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


def train(config_path: str):
    """Train MaskablePPO with DISCRETE actions and ACTION MASKING on CoatOpt.

    Args:
        config_path: Path to config INI file

    Returns:
        Trained MaskablePPO model
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    save_dir = parser.get('General', 'save_dir')
    materials_path = parser.get('General', 'materials_path')

    # [sb3_discrete] section
    total_timesteps = parser.getint('sb3_discrete', 'total_timesteps')
    n_thickness_bins = parser.getint('sb3_discrete', 'n_thickness_bins')
    verbose = parser.getint('sb3_discrete', 'verbose')
    target_reflectivity = parser.getfloat('sb3_discrete', 'target_reflectivity')
    target_absorption = parser.getfloat('sb3_discrete', 'target_absorption')
    mask_consecutive_materials = parser.getboolean('sb3_discrete', 'mask_consecutive_materials')
    mask_air_until_min_layers = parser.getboolean('sb3_discrete', 'mask_air_until_min_layers')
    min_layers_before_air = parser.getint('sb3_discrete', 'min_layers_before_air')
    epochs_per_step = parser.getint('sb3_discrete', 'epochs_per_step')
    steps_per_objective = parser.getint('sb3_discrete', 'steps_per_objective')
    tensorboard_log = parser.get('sb3_discrete', 'tensorboard_log')

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
        constraint_penalty=10.0,
        target_constraint_bounds={
            "reflectivity": target_reflectivity,
            "absorption": target_absorption,
        },
        # Action masking
        mask_consecutive_materials=mask_consecutive_materials,
        mask_air_until_min_layers=mask_air_until_min_layers,
        min_layers_before_air=min_layers_before_air,
        # Schedule
        epochs_per_step=epochs_per_step,
        steps_per_objective=steps_per_objective,
        constraint_schedule=constraint_schedule,
    )

    tb_log = None

    # Use MaskablePPO for action masking support
    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 64, 32],  # Actor: 3 layers
            vf=[128, 64, 32],  # Critic: 3 layers
        ),
        # activation_fn=th.nn.ReLU,  # Default, can change to Tanh, LeakyReLU, etc.
    )

    model = MaskablePPO(
        "MlpPolicy",
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.2,  # Initial value (will be updated by callback)
        verbose=0,
        tensorboard_log=tb_log,
        env = env,
    )

    entropy_callback = EntropyAnnealingCallback(
        max_ent=0.2,  # High exploration at start of each cycle
        min_ent=0.01,  # Low exploration at end of each cycle
        epochs_per_step=epochs_per_step,  # Reset annealing every N episodes
        verbose=0,
    )

    plotting_callback = DiscreteActionPlottingCallback(
        env=env,
        plot_freq=500,  # Reduced for shorter episodes
        design_plot_freq=50,
        save_dir=str(save_dir),
        n_best_designs=5,
        materials=materials,
        verbose=verbose,
    )

    print(f"\nStarting training for {total_timesteps} timesteps...")
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([entropy_callback, plotting_callback])
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model_path = save_dir / "coatopt_ppo_discrete"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    print("\nRunning final evaluation...")
    evaluate_model(model, env, n_episodes=10, use_action_masks=True)

    # Save Pareto front to CSV
    print("\nSaving Pareto front...")
    plotting_callback.save_pareto_front_to_csv("pareto_front.csv")

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
    parser.add_argument(
        "--tensorboard", type=str, default="./sb3_discrete_logs", help="Tensorboard log dir"
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
        tensorboard_log=args.tensorboard,
        verbose=args.verbose,
        target_reflectivity=args.target_reflectivity,
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
