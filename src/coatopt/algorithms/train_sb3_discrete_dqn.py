#!/usr/bin/env python3
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import load_config
from coatopt.utils.callbacks import PlottingCallback
from coatopt.utils.utils import load_materials, save_run_metadata
import time

class CoatOptDQNGymWrapper(gym.Env):
    """Gymnasium wrapper for CoatingEnvironment with flattened discrete actions for DQN.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config,
        materials: dict,
        n_thickness_bins: int = 20,
        constraint_penalty: float = 100.0,
        epochs_per_step: int = 200,
        steps_per_objective: int = 10,
        constraint_schedule: str = "interleaved",
        mask_consecutive_materials: bool = True,
        mask_air_until_min_layers: bool = True,
        min_layers_before_air: int = 4,
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

        # Precompute thickness bins
        self.thickness_bins = np.linspace(
            self.env.min_thickness,
            self.env.max_thickness,
            n_thickness_bins
        )

        # Multi-objective settings
        self.objectives = list(config.data.optimise_parameters)
        self.constraint_penalty = constraint_penalty

        # Training schedule
        self.epochs_per_step = epochs_per_step
        self.warmup_episodes_per_objective = epochs_per_step
        self.total_warmup_episodes = self.warmup_episodes_per_objective * len(self.objectives)
        self.steps_per_objective = steps_per_objective
        self.n_objectives = len(self.objectives)
        self.total_levels = self.steps_per_objective
        self.total_phases = self.total_levels * self.n_objectives
        self.n_anneal_episodes = self.total_phases * self.epochs_per_step

        # Episode tracking
        self.episode_count = 0
        self.is_warmup = True
        self.warmup_objective_idx = 0
        self.current_phase = 0
        self.current_level = 1
        self.current_layer = 0
        self.previous_material_idx = None

        # Find air material index (material with index 0)
        self.air_material_idx = 0

        # Enable constrained training in environment
        self.env.enable_constrained_training(
            warmup_episodes_per_objective=epochs_per_step,
            steps_per_objective=steps_per_objective,
            epochs_per_step=epochs_per_step,
            constraint_penalty=constraint_penalty,
        )

        # Enable Pareto dominance bonus if specified
        if pareto_dominance_bonus > 0:
            self.env.enable_pareto_bonus(bonus=pareto_dominance_bonus)

        # Current target and constraints
        self.target_objective = self.objectives[0]
        self.env.target_objective = self.target_objective
        self.constraints = {}
        self.env.constraints = {}

        # Observation space
        n_features = 1 + self.env.n_materials + 2
        obs_size = self.env.max_layers * n_features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # FLATTENED Discrete action space for DQN
        self.action_space = gym.spaces.Discrete(self.env.n_materials * n_thickness_bins)

    def _decode_action(self, action: int) -> tuple:
        """Decode flattened action into (material_idx, thickness_bin_idx)."""
        material_idx = action // self.n_thickness_bins
        thickness_bin = action % self.n_thickness_bins
        return material_idx, thickness_bin

    def _get_obs(self, state) -> np.ndarray:
        """Convert CoatingState to fixed-size numpy array."""
        tensor = state.get_observation_tensor(pre_type="lstm")
        return tensor.numpy().flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset with two-phase training: warmup then constrained cycling."""
        super().reset(seed=seed)
        state = self.env.reset()
        self.current_layer = 0
        self.previous_material_idx = None
        self.episode_count += 1

        # === PHASE 1: WARMUP ===
        if self.episode_count <= self.total_warmup_episodes:
            self.is_warmup = True
            self.warmup_objective_idx = (self.episode_count - 1) // self.warmup_episodes_per_objective
            self.warmup_objective_idx = min(self.warmup_objective_idx, self.n_objectives - 1)
            self.target_objective = self.objectives[self.warmup_objective_idx]
            self.env.target_objective = self.target_objective
            self.constraints = {}
            self.env.constraints = {}

            return self._get_obs(state), {
                "target": self.target_objective,
                "constraints": {},
                "episode": self.episode_count,
                "is_warmup": True,
            }

        # === PHASE 2: CONSTRAINED CYCLING ===
        if self.is_warmup:
            self.is_warmup = False
            self.env.is_warmup = False
            print(f"\n=== WARMUP COMPLETE (DQN) ===")
            print(f"Observed bounds: {self.env.observed_value_bounds}")
            print(f"Best warmup rewards: {self.env.warmup_best_rewards}")
            print(f"=== STARTING CONSTRAINED PHASE ===\n")

        constrained_episode = self.episode_count - self.total_warmup_episodes
        new_phase = (constrained_episode - 1) // self.epochs_per_step

        # Calculate objective indices and constraint level
        if self.constraint_schedule == "interleaved":
            # Interleaved: alternate objectives, both constraints tighten together
            # After all levels complete, restart with no constraints
            target_idx = new_phase % self.n_objectives
            level_cycle = (new_phase // self.n_objectives) % (self.total_levels + 1)  # Cycle through 0-10
            current_level = level_cycle  # 0 means no constraints, 1-10 are constraint levels
            constrained_idx = None  # Not used in interleaved mode

        elif self.constraint_schedule == "sequential":
            # Sequential: constrain one objective for all levels, optimize the other
            # Phases 0-9: Constrain R (levels 1-10), Opt A
            # Phases 10-19: Constrain A (levels 1-10), Opt R
            cycle_length = self.total_levels * self.n_objectives
            cycle_phase = new_phase % cycle_length
            constrained_idx = cycle_phase // self.total_levels  # Which objective is CONSTRAINED
            current_level = (cycle_phase % self.total_levels) + 1
            # Target to optimize is the OPPOSITE of what's constrained
            target_idx = (constrained_idx + 1) % self.n_objectives

        else:
            raise ValueError(f"Unknown constraint_schedule: {self.constraint_schedule}")

        # Update target objective only when it changes
        new_target = self.objectives[target_idx]
        if new_target != self.target_objective:
            self.target_objective = new_target
            self.env.target_objective = self.target_objective
            print(f"\n=== TARGET OBJECTIVE CHANGED TO: {self.target_objective} (Phase {new_phase}) ===\n")

        # Update phase tracking and constraints when entering a new phase
        if new_phase != self.current_phase or self.current_phase == 0:
            self.current_phase = new_phase
            self.current_level = current_level

            # Set constraints
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
                        max_achievable = self.env.warmup_best_rewards[obj]
                        max_constraint = step_fraction * max_achievable
                        self.constraints[obj] = np.random.uniform(0.0, max_constraint)

            self.env.constraints = self.constraints

        return self._get_obs(state), {
            "target": self.target_objective,
            "constraints": self.constraints,
            "episode": self.episode_count,
            "is_warmup": False,
        }

    def step(self, action):
        """Take flattened discrete action."""
        # Decode action
        material_idx, thickness_bin = self._decode_action(action)
        thickness = self.thickness_bins[thickness_bin]

        # Apply action masking rules (modify invalid actions)
        # Rule 1: Block consecutive same material
        if self.mask_consecutive_materials and material_idx == self.previous_material_idx:
            # Find a different material
            for alt_mat in range(self.env.n_materials):
                if alt_mat != self.previous_material_idx:
                    material_idx = alt_mat
                    break

        # Rule 2: Block air until minimum layers reached
        if self.mask_air_until_min_layers and self.current_layer < self.min_layers_before_air:
            if material_idx == self.air_material_idx:
                # Choose first non-air material
                material_idx = 1 if self.env.n_materials > 1 else 0

        # Build CoatOpt action
        coatopt_action = np.zeros(self.env.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + material_idx] = 1.0

        # Step environment
        state, rewards, terminated, finished, total_reward, _, vals = self.env.step(coatopt_action)

        # Track previous material for next step
        self.previous_material_idx = material_idx
        self.current_layer += 1

        obs = self._get_obs(state)
        done = finished
        truncated = False
        total_reward = 0.0
        info = {}

        # Only populate info at episode end
        if done:
            total_reward, _, _ = self.env.compute_training_reward(state)
            info = {
                "rewards": rewards,
                "vals": vals,
                "finished": finished,
                "target": self.target_objective,
                "constraints": self.constraints,
                "state_array": state.get_array(),
                "constrained_reward": total_reward,
                "episode": {'r': total_reward, 'l': self.current_layer, 't': 0},
                "is_warmup": self.is_warmup,
            }

            phase_str = f"WARMUP[{self.target_objective}]" if self.is_warmup else f"CONSTR[{self.target_objective}]"
            print(f"{phase_str} R={vals.get('reflectivity', 0):.4f}, A={vals.get('absorption', 0):.1f}, reward={total_reward:.3f}")

        return obs, float(total_reward), done, truncated, info


# ============================================================================
# TRAINING
# ============================================================================
def train(config_path: str, save_dir: str):
    """Train DQN with discrete actions on CoatOpt.

    Args:
        config_path: Path to config INI file
        save_dir: Directory to save results

    Returns:
        Trained DQN model
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    materials_path = parser.get('General', 'materials_path')

    # [sb3_dqn] section
    section = 'sb3_dqn'
    total_timesteps = parser.getint(section, 'total_timesteps')
    n_thickness_bins = parser.getint(section, 'n_thickness_bins')
    verbose = parser.getint(section, 'verbose')
    epochs_per_step = parser.getint(section, 'epochs_per_step')
    steps_per_objective = parser.getint(section, 'steps_per_objective')

    # Action masking settings
    mask_consecutive_materials = parser.getboolean(section, 'mask_consecutive_materials', fallback=True)
    mask_air_until_min_layers = parser.getboolean(section, 'mask_air_until_min_layers', fallback=True)
    min_layers_before_air = parser.getint(section, 'min_layers_before_air', fallback=4)

    # Constraint settings
    constraint_penalty = parser.getfloat(section, 'constraint_penalty', fallback=10.0)
    pareto_dominance_bonus = parser.getfloat(section, 'pareto_dominance_bonus', fallback=0.0)

    # DQN hyperparameters
    learning_rate = parser.getfloat(section, 'learning_rate', fallback=1e-4)
    buffer_size = parser.getint(section, 'buffer_size', fallback=50000)
    learning_starts = parser.getint(section, 'learning_starts', fallback=300)
    batch_size = parser.getint(section, 'batch_size', fallback=128)
    gamma = parser.getfloat(section, 'gamma', fallback=0.99)
    train_freq = parser.getint(section, 'train_freq', fallback=4)
    gradient_steps = parser.getint(section, 'gradient_steps', fallback=1)
    target_update_interval = parser.getint(section, 'target_update_interval', fallback=100)
    exploration_fraction = parser.getfloat(section, 'exploration_fraction', fallback=0.3)
    exploration_initial_eps = parser.getfloat(section, 'exploration_initial_eps', fallback=1.0)
    exploration_final_eps = parser.getfloat(section, 'exploration_final_eps', fallback=0.05)

    # Network architecture
    net_arch_str = parser.get(section, 'net_arch', fallback='[64, 64]')
    net_arch = eval(net_arch_str)  # Parse list from string

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
    config.data.n_layers = n_layers

    # Create environment
    env = CoatOptDQNGymWrapper(
        config,
        materials,
        n_thickness_bins=n_thickness_bins,
        constraint_penalty=constraint_penalty,
        epochs_per_step=epochs_per_step,
        steps_per_objective=steps_per_objective,
        constraint_schedule=constraint_schedule,
        mask_consecutive_materials=mask_consecutive_materials,
        mask_air_until_min_layers=mask_air_until_min_layers,
        min_layers_before_air=min_layers_before_air,
        pareto_dominance_bonus=pareto_dominance_bonus,
    )

    print(f"\nEnvironment created (DQN):")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: Discrete({env.action_space.n})")
    print(f"  = {env.env.n_materials} materials × {n_thickness_bins} thickness bins")

    print(f"\nConstraint schedule: {constraint_schedule}")
    if constraint_schedule == "interleaved":
        print(f"  Pattern: Alternate objectives every {epochs_per_step} episodes, both constraints tighten together")
    elif constraint_schedule == "sequential":
        print(f"  Pattern: Complete all {steps_per_objective} constraint levels for each objective before switching")

    tb_log = None

    # Configure policy network architecture
    policy_kwargs = dict(
        net_arch=net_arch,
    )

    print(f"\nDQN Network Architecture:")
    print(f"  Q-network: {net_arch}")

    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=tb_log,
    )

    # Set up callbacks
    plotting_callback = PlottingCallback(
        env=env,
        plot_freq=500,
        design_plot_freq=50,
        save_dir=str(save_dir),
        n_best_designs=5,
        materials=materials,
        verbose=verbose,
    )

    # Train
    print(f"\nStarting DQN training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=plotting_callback)
    end_time = time.time()

    # Save model
    model_path = save_dir / "coatopt_dqn"
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Save Pareto front
    print("\nSaving Pareto front...")
    plotting_callback.save_pareto_front_to_csv("pareto_front_dqn.csv")

    # Get Pareto front size
    import pandas as pd
    pareto_csv = save_dir / "pareto_front_values.csv"
    pareto_size = 0
    if pareto_csv.exists():
        pareto_df = pd.read_csv(pareto_csv)
        pareto_size = len(pareto_df)

    # Save run metadata
    save_run_metadata(
        save_dir=save_dir,
        algorithm_name="SB3_DQN",
        start_time=start_time,
        end_time=end_time,
        pareto_front_size=pareto_size,
        total_episodes=plotting_callback.episode_count,
        config_path=config_path,
        additional_info={
            "total_timesteps": total_timesteps,
            "n_thickness_bins": n_thickness_bins,
            "constraint_schedule": constraint_schedule,
            "constraint_penalty": constraint_penalty,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
        }
    )

    return model


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SB3 DQN on CoatOpt")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config INI file"
    )

    args = parser.parse_args()
    train(config_path=args.config)
