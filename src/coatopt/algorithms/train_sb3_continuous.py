#!/usr/bin/env python3
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, DataConfig, TrainingConfig
from coatopt.utils.callbacks import PlottingCallback
from coatopt.utils.utils import load_materials, evaluate_model, EntropyAnnealingCallback


# ============================================================================
# GYMNASIUM WRAPPER
# ============================================================================
class CoatOptGymWrapper(gym.Env):
    """Gymnasium wrapper for CoatingEnvironment with constraint-based multi-objective.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        materials: dict,
        constraint_penalty: float = 100.0,
        # Target constraint bounds (tightest constraints at end of annealing)
        # For reflectivity: higher is better, so this is minimum required
        # For absorption: lower is better, so this is maximum allowed
        target_constraint_bounds: dict = None,
        # Consecutive material penalty
        consecutive_material_penalty: float = 0.2,
        # Annealing schedule

    ):
        super().__init__()
        self.env = CoatingEnvironment(config, materials)
        self.config = config
        # Multi-objective constraint settings
        self.objectives = list(config.data.optimise_parameters)
        self.constraint_penalty = constraint_penalty
        self.target_constraint_bounds = target_constraint_bounds or {
            "reflectivity": 0.99,   # Target: must achieve >= 99% reflectivity
            "absorption": 1,     # Target: must achieve <= 1 ppm absorption
        }

        # Consecutive material penalty
        self.consecutive_material_penalty = consecutive_material_penalty
        self.previous_material_idx = None

        # Stepped constraint schedule
        self.epochs_per_step = 1000
        self.steps_per_objective = 10
        self.n_objectives = len(self.objectives)
        self.total_levels = self.steps_per_objective
        self.total_phases = self.total_levels * self.n_objectives  # 10 steps * 2 objectives = 20 phases
        self.n_anneal_episodes = self.total_phases * self.epochs_per_step  # 20 * 100 = 2000 episodes

        # Episode counter for scheduling
        self.episode_count = 0

        # Current episode's target and constraints (set in reset)
        self.target_objective = None
        self.target_objective_idx = 0  # Index into self.objectives
        self.constraints = {}  # {objective: threshold}
        self.current_phase = 0
        self.current_level = 0

        # Track observed bounds during training (for annealing starting point)
        # These get updated as we see what the agent can actually achieve
        self.observed_bounds = {obj: {"min": np.inf, "max": -np.inf} for obj in self.objectives}

        # Compute min and max rewards for annealing from objective bounds
        self.min_rewards = {}
        self.max_rewards = {}
        for obj in self.objectives:
            # Min reward: worst case from objective_bounds[0]
            self.min_rewards[obj] = self.reward_function(obj, self.config.data.objective_bounds[obj][0])
            # Max reward: use target_constraint_bounds as the best achievable value for annealing
            # This represents the tightest constraint we want to reach
            self.max_rewards[obj] = self.reward_function(obj, self.target_constraint_bounds[obj])

        # Observation space
        n_features = 1 + self.env.n_materials + 2
        obs_size = self.env.max_layers * n_features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action space
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, self.env.min_thickness], dtype=np.float32),
            high=np.array(
                [float(self.env.n_materials - 1), self.env.max_thickness],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def _get_obs(self, state) -> np.ndarray:
        """Convert CoatingState to fixed-size numpy array."""
        tensor = state.get_observation_tensor(pre_type="lstm")
        return tensor.numpy().flatten().astype(np.float32)

    def _get_annealing_progress(self) -> float:
        """Get annealing progress from 0.0 (start) to 1.0 (fully annealed)."""
        return min(1.0, self.episode_count / self.n_anneal_episodes)

    def reward_function(self, obj: str, val: float) -> float:
        """Compute the reward for an objective value."""
        target = self.config.data.optimise_targets[obj]
        return -np.log(np.abs(val - target) + 1e-30)

    def reset(self, seed=None, options=None):
        """Reset with scheduled objective cycling and annealed constraints."""
        super().reset(seed=seed)
        state = self.env.reset()

        # Reset material tracking
        self.previous_material_idx = None

        # Increment episode counter
        self.episode_count += 1

        # Determine current phase and level
        new_phase = (self.episode_count - 1) // self.epochs_per_step
        
        # Only recompute constraints when entering a new phase
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            # Alternate between objectives every phase
            target_idx = self.current_phase % self.n_objectives
            self.target_objective = self.objectives[target_idx]
            
            # Level increases every n_objectives phases (every 2 phases = every 200 episodes)
            self.current_level = (self.current_phase // self.n_objectives) + 1  # 1 to total_levels

            # Set constraints on other objectives: random within [min, step_level]
            self.constraints = {}
            for i, obj in enumerate(self.objectives):
                if i != target_idx:
                    min_r = self.min_rewards[obj]
                    max_r = self.max_rewards[obj]
                    step_fraction = self.current_level / self.total_levels
                    constraint_max = min_r + (max_r - min_r) * step_fraction
                    self.constraints[obj] = np.random.uniform(min_r, constraint_max)

        progress = min(1.0, self.episode_count / (self.total_phases * self.epochs_per_step))
        return self._get_obs(state), {
            "target": self.target_objective,
            "constraints": self.constraints,
            "annealing_progress": progress,
            "episode": self.episode_count,
            "phase": self.current_phase,
            "level": self.current_level,
        }

    def step(self, action):
        """Take action, apply constraint penalties at episode end."""
        # Decode action
        material_idx = int(np.clip(np.round(action[0]), 0, self.env.n_materials - 1))
        thickness = float(action[1])

        # Check for consecutive same material penalty
        consecutive_penalty = 0.0
        if self.previous_material_idx is not None and material_idx == self.previous_material_idx:
            consecutive_penalty = self.consecutive_material_penalty

        # Update previous material tracking
        self.previous_material_idx = material_idx

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

        # Override environment reward to ensure normalised scale
        total_reward = 0.0

        # Apply consecutive material penalty
        total_reward = total_reward - consecutive_penalty

        # Build info
        info = {
            "rewards": rewards,
            "vals": vals,
            "finished": finished,
            "target": self.target_objective,
            "constraints": self.constraints,
            "consecutive_penalty": consecutive_penalty,
            "annealing_progress": self._get_annealing_progress(),
            "episode": {'r': 0, 'l': 20, 't': 0}, 
            'state_array': None, 
            'constrained_reward': -1.0114863423176935e+19, 
            'TimeLimit.truncated': False, 
            'terminal_observation': None
            }
        

        # At episode end: check constraints and compute final reward
        if done:
            info["state_array"] = state.get_array()

            # Update observed bounds for annealing
            for obj in self.objectives:
                if obj in vals and vals[obj] is not None:
                    val = float(vals[obj])
                    if not np.isnan(val):
                        self.observed_bounds[obj]["min"] = min(self.observed_bounds[obj]["min"], val)
                        self.observed_bounds[obj]["max"] = max(self.observed_bounds[obj]["max"], val)

            # Compute reward: target objective value + constraint penalties
            total_reward = self._compute_constrained_reward(vals) - consecutive_penalty
            info["constrained_reward"] = total_reward

        return obs, float(total_reward), done, truncated, info

    def _compute_constrained_reward(self, vals: dict) -> float:
        """Compute reward for target objective with constraint penalties."""
        # Get target objective value as base reward
        target_val = vals.get(self.target_objective)
        if target_val is None or np.isnan(target_val):
            return -self.constraint_penalty

        # Compute log-target reward (from old coating_reward_function)
        reward = self.reward_function(self.target_objective, target_val)
        
        # Handle inf/nan rewards
        if np.isnan(reward) or np.isinf(reward):
            reward = -self.constraint_penalty

        # Check constraints and apply penalties (constraints are on reward levels)
        penalty = 0.0
        for obj, threshold in self.constraints.items():
            val = vals.get(obj)
            if val is None or np.isnan(val):
                penalty += self.constraint_penalty
                continue

            # Compute reward for this objective
            reward_obj = self.reward_function(obj, val)
            
            # Handle inf/nan constraint rewards
            if np.isnan(reward_obj) or np.isinf(reward_obj):
                penalty += self.constraint_penalty
                continue

            # Check if reward meets the threshold
            if reward_obj < threshold:
                violation_amount = (threshold - reward_obj) / max(abs(threshold), 1.0)
                penalty += self.constraint_penalty * violation_amount

        final_reward = reward - penalty
        return np.clip(final_reward, -1000, 1000)  # Prevent extreme values




# ============================================================================
# TRAINING
# ============================================================================
def train(config_path: str):
    """Train PPO on CoatOpt environment.

    Args:
        config_path: Path to config INI file

    Returns:
        Trained PPO model
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    save_dir = parser.get('general', 'save_dir')
    materials_path = parser.get('general', 'materials_path')

    # [sb3_simple] section
    total_timesteps = parser.getint('sb3_simple', 'total_timesteps')
    verbose = parser.getint('sb3_simple', 'verbose')
    target_reflectivity = parser.getfloat('sb3_simple', 'target_reflectivity')
    target_absorption = parser.getfloat('sb3_simple', 'target_absorption')
    tensorboard_log = parser.get('sb3_simple', 'tensorboard_log')

    # [Data] section
    n_layers = parser.getint('data', 'n_layers')

    # Set up directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load materials
    if materials_path is None:
        # Default path relative to this file (src/coatopt/train_sb3_simple.py)
        materials_path = Path(__file__).parent / "config" / "materials.json"

    materials = load_materials(str(materials_path))
    print(f"Loaded {len(materials)} materials from {materials_path}")

    # Create config
    config = Config(
        data=DataConfig(
            n_layers=n_layers,
            min_thickness=10e-9,
            max_thickness=500e-9,
            optimise_parameters=["reflectivity", "absorption"],
            optimise_targets={"reflectivity": 0.99999, "absorption": 0.0},
            use_optical_thickness=False,
            ignore_air_option=False,
            ignore_substrate_option=False,
            use_intermediate_reward=False,
            combine="sum",
            # Reward normalization
            use_reward_normalisation=True,
            reward_normalisation_apply_clipping=True,
            objective_bounds={
                "reflectivity": [0.0, 0.99999],
                "absorption": [10000, 0],  # ppm
            },
            # Air penalty
            apply_air_penalty=True,
            air_penalty_weight=0.5,
            apply_preference_constraints=False,
        ),
        training=TrainingConfig(cycle_weights="random"),
    )

    # Create wrapped environment with constraint-based multi-objective
    env = CoatOptGymWrapper(
        config,
        materials,
        constraint_penalty=10.0,
        target_constraint_bounds={
            "reflectivity": target_reflectivity,
            "absorption": target_absorption,
        },
    )

    # Compute derived schedule info
    steps_per_obj = env.steps_per_objective
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max layers: {env.env.max_layers}")
    print(f"  N materials: {env.env.n_materials}")
    print(f"  Objectives: {env.objectives}")
    print(f"\nAnnealing schedule:")
    print(f"  Epochs per step: {env.epochs_per_step}")
    print(f"  Steps per objective: {steps_per_obj}")
    print(f"  Total levels: {env.total_levels}")
    print(f"  Total phases: {env.total_phases}")
    print(f"  Total annealing episodes: {env.n_anneal_episodes}")
    print(f"  Target constraints: R>={target_reflectivity}, A<={target_absorption}")

    # Check if tensorboard is available
    try:
        import tensorboard  # noqa: F401

        tb_log = tensorboard_log
    except ImportError:
        print("Tensorboard not installed, disabling tensorboard logging")
        tb_log = None

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,  # Initial value (will be updated by callback)
        verbose=verbose,
        tensorboard_log=tb_log,
    )

    # Create entropy annealing callback (single annealing over entire training)
    entropy_callback = EntropyAnnealingCallback(
        max_ent=0.1,  # High exploration early
        min_ent=0.001,  # Low exploration late
        epochs_per_step=None,  # Single annealing (no cycling)
        verbose=0,
    )

    # Set up callbacks
    plotting_callback = PlottingCallback(
        plot_freq=5000,
        design_plot_freq=100,  # Plot best designs every 100 episodes
        save_dir=str(save_dir),
        n_best_designs=5,
        materials=materials,
        verbose=verbose,
    )

    # Train
    print(f"\nStarting training for {total_timesteps} timesteps...")
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([entropy_callback, plotting_callback])
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save model
    model_path = save_dir / "coatopt_ppo"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Final evaluation
    print("\nRunning final evaluation...")
    evaluate_model(model, env, n_episodes=10)

    return model




# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SB3 PPO on CoatOpt")
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--layers", type=int, default=20, help="Number of coating layers"
    )
    parser.add_argument(
        "--materials", type=str, default=None, help="Path to materials JSON"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./sb3_output", help="Output directory"
    )
    parser.add_argument(
        "--tensorboard", type=str, default="./sb3_logs", help="Tensorboard log dir"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    # Annealing schedule arguments
    parser.add_argument(
        "--target-reflectivity", type=float, default=0.99,
        help="Target reflectivity constraint (tightest, at end of annealing)"
    )
    parser.add_argument(
        "--target-absorption", type=float, default=1,
        help="Target absorption constraint in ppm (tightest, at end of annealing)"
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_layers=args.layers,
        materials_path=args.materials,
        save_dir=args.save_dir,
        tensorboard_log=args.tensorboard,
        verbose=args.verbose,
        target_reflectivity=args.target_reflectivity,
        target_absorption=args.target_absorption,
    )
