#!/usr/bin/env python3
"""
MORLD (MOSAC population) for multi-objective optimization.

Uses MOSAC agents with weight adaptation (PSA) to explore the Pareto front.
Trains a population of agents with different weight vectors that adapt toward
underexplored regions.

Note on the action space: MOSAC is continuous-only, so the material choice is
relaxed onto a continuous axis and rounded back to an index in step().

Config section: [morl] or [morld]
  method                   = morld          # read by run.py
  total_timesteps          = 500000
  seed                     = 42
  verbose                  = 1
  net_arch                 = [256, 256]
  pop_size                 = 8              # Population size
  scalarization_method     = ws             # Weighted sum (ws only, tch broken)
  weight_adaptation_method = PSA            # PSA or none
  weight_init_method       = uniform
  neighborhood_size        = 2
  shared_buffer            = true
  exchange_every           = 50000          # Weight exchange frequency
  gamma                    = 0.99
  # MOSAC hyperparameters
  learning_rate            = 3e-4           # Actor LR (MOSAC policy_lr)
  q_learning_rate          = 1e-3           # Critic LR (MOSAC q_lr)
  buffer_size              = 1000000
  batch_size               = 128
  tau                      = 0.005
  learning_starts          = 1000
  alpha                    = 0.2            # Entropy coefficient
  autotune                 = true           # Learn alpha
  # Front evaluation (each eval episode costs one merit-function evaluation)
  num_eval_episodes_for_front = 5
  num_eval_weights_for_eval   = 50
  save_checkpoints         = false          # Write agent weights into save_dir
  # Action correction
  min_layers_before_air    = 4              # Min layers before air allowed
  mask_consecutive_materials = true         # Correct consecutive material actions
  consecutive_penalty      = 0.2            # Penalty for corrected actions
"""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, load_config
from coatopt.utils.plotting import plot_pareto_front
from coatopt.utils.utils import load_materials_from_parser


# ============================================================================
# MO-GYMNASIUM WRAPPER
# ============================================================================
class CoatOptEnvSpec:
    """Minimal spec object for MORL-baselines compatibility."""

    def __init__(self, env_id: str = "CoatOpt-v0"):
        self.id = env_id
        self.name = env_id


class CoatOptMOGymWrapper(gym.Env):
    """MO-Gymnasium compatible wrapper for CoatingEnvironment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        materials: dict,
        consecutive_material_penalty: float = 0.2,
        mask_consecutive_materials: bool = True,
        min_layers_before_air: int = 4,
    ):
        super().__init__()
        self.env = CoatingEnvironment(config, materials)

        # Spec for MORL-baselines compatibility
        self.spec = CoatOptEnvSpec("CoatOpt-v0")

        # Action correction settings
        self.consecutive_material_penalty = consecutive_material_penalty
        self.mask_consecutive_materials = mask_consecutive_materials
        self.min_layers_before_air = min_layers_before_air
        self.previous_material_idx = None
        self.current_layer_count = 0
        # CoatingEnvironment resolves air by material name, so a reordered
        # materials file cannot silently change which index means "stop here".
        self.air_material_idx = self.env.air_material_index

        # Multi-objective settings. Read the parsed list off the environment
        # rather than config.data, which can hold an unparsed string.
        self.objectives = list(self.env.optimise_parameters)
        self.reward_dim = len(self.objectives)

        # Observation space
        n_features = 1 + self.env.n_materials + 2
        obs_size = self.env.max_layers * n_features
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action space
        self.action_space = Box(
            low=np.array([0.0, self.env.min_thickness], dtype=np.float32),
            high=np.array(
                [float(self.env.n_materials - 1), self.env.max_thickness],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # MO-Gymnasium required: reward_space.  Normalised rewards sit in
        # [0, 1] less the correction penalty, unbounded without clipping.
        if self.env.reward_normalisation_apply_clipping:
            reward_low = np.full(
                self.reward_dim, -consecutive_material_penalty, dtype=np.float32
            )
            reward_high = np.ones(self.reward_dim, dtype=np.float32)
        else:
            reward_low = np.full(self.reward_dim, -np.inf, dtype=np.float32)
            reward_high = np.full(self.reward_dim, np.inf, dtype=np.float32)
        self.reward_space = Box(low=reward_low, high=reward_high, dtype=np.float32)

    def _get_obs(self, state) -> np.ndarray:
        """Convert CoatingState to fixed-size numpy array.

        max_thickness puts the per-layer columns on a common scale.
        """
        tensor = state.get_observation_tensor(
            pre_type="lstm", max_thickness=self.env.max_thickness
        )
        return tensor.numpy().flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        state = self.env.reset()

        # Reset material tracking
        self.previous_material_idx = None
        self.current_layer_count = 0

        return self._get_obs(state), {}

    def step(self, action):
        """Take action and return vector reward."""
        # Decode action
        material_idx = int(np.clip(np.round(action[0]), 0, self.env.n_materials - 1))
        thickness = float(action[1])

        # Track if action was corrected
        action_corrected = False
        consecutive_penalty = 0.0

        # Correct consecutive material selection.  Skip air: substituting it
        # would end the episode rather than correct the layer.
        if (
            self.mask_consecutive_materials
            and self.previous_material_idx is not None
            and material_idx == self.previous_material_idx
            and material_idx != self.air_material_idx  # Air can repeat
        ):
            # Find a different coating material
            for alt_idx in range(self.env.n_materials):
                if alt_idx != material_idx and alt_idx != self.air_material_idx:
                    material_idx = alt_idx
                    action_corrected = True
                    consecutive_penalty = self.consecutive_material_penalty
                    break

        # Correct early air selection (before minimum layers)
        if (
            material_idx == self.air_material_idx
            and self.current_layer_count < self.min_layers_before_air
        ):
            # Choose first non-air material
            for alt_idx in range(self.env.n_materials):
                if (
                    alt_idx != self.air_material_idx
                    and alt_idx != self.previous_material_idx
                ):
                    material_idx = alt_idx
                    action_corrected = True
                    consecutive_penalty = self.consecutive_material_penalty
                    break

        # Update tracking
        self.previous_material_idx = material_idx
        if material_idx != self.air_material_idx:
            self.current_layer_count += 1

        # Build CoatOpt action
        coatopt_action = np.zeros(self.env.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + material_idx] = 1.0

        # Step environment
        state, rewards, terminated, finished, total_reward, _, vals = self.env.step(
            coatopt_action
        )

        # step() only stages the finished design; it enters the archive on
        # flush. MORL/D exposes no rollout boundary, so flush per episode.
        if finished:
            self.env.flush_pareto_candidates()

        obs = self._get_obs(state)
        done = finished
        truncated = False

        # Build info
        info = {
            "rewards": rewards,
            "vals": vals,
            "finished": finished,
            "consecutive_penalty": consecutive_penalty,
            "action_corrected": action_corrected,
        }

        # Vector reward (MO-Gymnasium API)
        if done:
            # Final episode reward based on actual objective values
            # Get normalised rewards for all objectives
            normalised_rewards = self.env.compute_objective_rewards(
                vals, normalised=True
            )
            vec_reward = np.array(
                [normalised_rewards.get(obj, 0.0) for obj in self.objectives],
                dtype=np.float32,
            )
            # Apply consecutive penalty to all objectives
            vec_reward = vec_reward - consecutive_penalty
            info["state_array"] = state.get_array()
        else:
            # Intermediate reward: apply penalty if consecutive material used
            vec_reward = np.full(
                self.reward_dim, -consecutive_penalty, dtype=np.float32
            )

        return obs, vec_reward, done, truncated, info


def setup_morl_training(config_path: str, algorithm: str = "morld"):
    """Shared setup for all MORL algorithms.

    Args:
        config_path: Path to config INI file
        algorithm: Algorithm name (currently only "morld")

    Returns:
        Dictionary with config, env, eval_env, materials, save_dir, and algorithm params
    """
    import configparser

    config = load_config(config_path)
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    parser.read(config_path)

    # Common parameters (try algorithm-specific section, fall back to 'morl')
    section = algorithm if parser.has_section(algorithm) else "morl"

    total_timesteps = parser.getint(section, "total_timesteps")
    seed = parser.getint(section, "seed", fallback=42)
    verbose = parser.getint(section, "verbose", fallback=1)
    net_arch = eval(parser.get(section, "net_arch", fallback="[256, 256]"))

    # MORL/D evaluates the whole population between weight exchanges, and each
    # eval episode costs a full merit-function evaluation.
    num_eval_episodes_for_front = parser.getint(
        section, "num_eval_episodes_for_front", fallback=5
    )
    num_eval_weights_for_eval = parser.getint(
        section, "num_eval_weights_for_eval", fallback=50
    )
    # morl-baselines' checkpointing writes into ./weights/ with no way to
    # redirect it, so it stays off and save_dir gets the final weights.
    save_checkpoints = parser.getboolean(section, "save_checkpoints", fallback=False)

    # Directories
    save_dir = Path(parser.get("general", "save_dir"))
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "plots").mkdir(exist_ok=True)

    # Materials
    materials = load_materials_from_parser(parser, config_path)

    # Action masking/correction parameters
    consecutive_penalty = parser.getfloat(section, "consecutive_penalty", fallback=0.2)
    mask_consecutive = parser.getboolean(
        section, "mask_consecutive_materials", fallback=True
    )
    min_layers_before_air = parser.getint(section, "min_layers_before_air", fallback=4)

    # Create environments
    env = CoatOptMOGymWrapper(
        config,
        materials,
        consecutive_material_penalty=consecutive_penalty,
        mask_consecutive_materials=mask_consecutive,
        min_layers_before_air=min_layers_before_air,
    )
    eval_env = CoatOptMOGymWrapper(
        config,
        materials,
        consecutive_material_penalty=consecutive_penalty,
        mask_consecutive_materials=mask_consecutive,
        min_layers_before_air=min_layers_before_air,
    )

    print(
        f"\nEnvironment: obs={env.observation_space.shape}, "
        f"action={env.action_space}, reward_dim={env.reward_dim}"
    )

    return {
        "config": config,
        "parser": parser,
        "section": section,
        "env": env,
        "eval_env": eval_env,
        "materials": materials,
        "save_dir": save_dir,
        "config_path": config_path,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "verbose": verbose,
        "net_arch": net_arch,
        "num_eval_episodes_for_front": num_eval_episodes_for_front,
        "num_eval_weights_for_eval": num_eval_weights_for_eval,
        "save_checkpoints": save_checkpoints,
    }


def create_morl_agent(algorithm: str, setup_dict: dict):
    """Factory to create MORL algorithm agent.

    Args:
        algorithm: "morld"
        setup_dict: Dictionary from setup_morl_training()

    Returns:
        Initialized agent
    """
    env = setup_dict["env"]
    parser = setup_dict["parser"]
    section = setup_dict["section"]
    seed = setup_dict["seed"]
    net_arch = setup_dict["net_arch"]

    if algorithm == "morld":
        from morl_baselines.multi_policy.morld.morld import MORLD

        pop_size = parser.getint(section, "pop_size", fallback=8)
        # NOTE: "tch" (Tchebycheff) is broken in morl-baselines for MOSAC — it
        # receives batched PyTorch tensors but expects numpy scalars in the
        # reference-point loop, causing "Boolean value of Tensor is ambiguous".
        # Use "ws" (weighted sum) only. Non-convex Pareto coverage is instead
        # achieved via weight_adaptation_method="PSA" + larger pop_size.
        scalarization = parser.get(section, "scalarization_method", fallback="ws")
        # "PSA" shifts weights toward underexplored Pareto regions during training.
        weight_adaptation = parser.get(
            section, "weight_adaptation_method", fallback="PSA"
        )
        weight_init = parser.get(section, "weight_init_method", fallback="uniform")
        neighborhood_size = parser.getint(section, "neighborhood_size", fallback=2)
        shared_buffer = parser.getboolean(section, "shared_buffer", fallback=True)
        exchange_every = parser.getint(
            section,
            "exchange_every",
            fallback=setup_dict["total_timesteps"] // 10,
        )
        gamma = parser.getfloat(section, "gamma", fallback=0.99)

        # MOSAC takes policy_lr/q_lr, not learning_rate. MORL/D supplies
        # id/env/weights/scalarization/gamma/log/seed/parent_rng itself.
        policy_args = {
            "net_arch": net_arch,
            "policy_lr": parser.getfloat(section, "learning_rate", fallback=3e-4),
            "q_lr": parser.getfloat(section, "q_learning_rate", fallback=1e-3),
            "buffer_size": parser.getint(section, "buffer_size", fallback=int(1e6)),
            "batch_size": parser.getint(section, "batch_size", fallback=128),
            "tau": parser.getfloat(section, "tau", fallback=0.005),
            "learning_starts": parser.getint(
                section, "learning_starts", fallback=int(1e3)
            ),
            "alpha": parser.getfloat(section, "alpha", fallback=0.2),
            "autotune": parser.getboolean(section, "autotune", fallback=True),
        }

        return MORLD(
            env=env,
            scalarization_method=scalarization,
            evaluation_mode="ser",
            policy_name="MOSAC",
            gamma=gamma,
            pop_size=pop_size,
            seed=seed,
            exchange_every=exchange_every,
            neighborhood_size=neighborhood_size,
            shared_buffer=shared_buffer,
            weight_init_method=weight_init,
            weight_adaptation_method=(
                weight_adaptation if weight_adaptation != "none" else None
            ),
            log=False,
            device="auto",
            policy_args=policy_args,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. Supported: morld\n"
            "Note: pgmorl and moppo are excluded — they require vectorised envs "
            "and bypass the CoatingEnvironment, so designs/values cannot be tracked."
        )


def merge_pareto_front(target_env, source_env):
    """Fold source_env's archived designs into target_env's Pareto front.

    MORL/D evaluates on a second environment, so its designs land in that
    environment's archive; restaging re-derives them with the target's bounds.
    """
    source_env.flush_pareto_candidates()
    for val_vector, state in source_env.pareto_front_values:
        vals = dict(zip(source_env.optimise_parameters, val_vector))
        # No public restage API; this is the same call CoatingEnvironment.step
        # makes when a design finishes.
        target_env._stage_pareto_candidate(vals, state, None)
    target_env.flush_pareto_candidates()


def run_morl_training_loop(agent, setup_dict: dict, algorithm: str):
    """Shared training loop for all MORL algorithms.

    Args:
        agent: Initialized MORL agent
        setup_dict: Dictionary from setup_morl_training()
        algorithm: Algorithm name for logging
    """

    env = setup_dict["env"]
    eval_env = setup_dict["eval_env"]
    save_dir = setup_dict["save_dir"]
    total_timesteps = setup_dict["total_timesteps"]
    objectives = list(env.objectives)
    reward_dim = env.reward_dim
    ref_point = np.zeros(reward_dim, dtype=np.float32)

    print(f"\n{'='*60}")
    print(f"  Algorithm  : {algorithm.upper()}")
    print(f"  Objectives : {objectives}")
    print(f"  Reward dim : {reward_dim}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Save dir   : {save_dir}")
    print(f"{'='*60}")
    print("Starting training — morl-baselines handles the inner loop.\n")

    start_time = time.time()

    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
        num_eval_episodes_for_front=setup_dict["num_eval_episodes_for_front"],
        num_eval_weights_for_eval=setup_dict["num_eval_weights_for_eval"],
        checkpoints=False,
    )

    end_time = time.time()
    duration_min = (end_time - start_time) / 60

    if setup_dict.get("save_checkpoints"):
        weights_dir = save_dir / "weights"
        agent.save(
            save_dir=str(weights_dir), filename="morld_final", save_replay_buffer=False
        )
        print(f"Saved agent weights -> {weights_dir / 'morld_final.tar'}")

    # Designs found while evaluating the population live in the eval env.
    merge_pareto_front(env.env, eval_env.env)
    n_evaluations = env.env.n_evaluations + eval_env.env.n_evaluations

    designs_df, values_df, rewards_df = env.env.export_pareto_dataframes()

    if not rewards_df.empty:
        print(f"\nPareto front: {len(rewards_df)} solutions")
        for obj in objectives:
            col = values_df[obj]
            print(
                f"  {obj}: min={col.min():.4g}  max={col.max():.4g}  mean={col.mean():.4g}"
            )
        try:
            plot_path = plot_pareto_front(
                df=values_df,
                objectives=objectives,
                save_dir=save_dir,
                plot_type="vals",
                algorithm_name=algorithm,
            )
            print(f"Saved plot → {plot_path}")
        except Exception as e:
            print(f"Warning: Failed to plot: {e}")
    else:
        print("\nWarning: No Pareto solutions found.")

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"  Duration       : {duration_min:.1f} min")
    print(f"  Evaluations    : {n_evaluations:,}")
    print(f"  Pareto front   : {len(rewards_df)} solutions")
    print(f"{'='*60}")

    # Saving is handled by run.py via save_training_results — do not duplicate here.
    return {
        "pareto_designs": designs_df,
        "pareto_values": values_df,
        "pareto_rewards": rewards_df,
        "model": None,
        "metadata": {
            "total_timesteps": total_timesteps,
            "n_evaluations": n_evaluations,
            "seed": setup_dict["seed"],
        },
    }


def train(config_path: str, algorithm: str = "morld", save_dir: str = None):
    """Unified training function for MORL algorithms that work with a single env.

    Args:
        config_path: Path to config INI file
        algorithm: Algorithm to use — currently only "morld" (MOSAC population).
            pgmorl and moppo are not supported: they require vectorised envs and
            bypass the CoatingEnvironment, so designs/values cannot be tracked.
        save_dir: Optional override for save directory

    Returns:
        Results dict with pareto_designs, pareto_values, pareto_rewards, model, metadata
    """
    setup = setup_morl_training(config_path, algorithm)
    if save_dir:
        setup["save_dir"] = Path(save_dir)
        setup["save_dir"].mkdir(parents=True, exist_ok=True)
        (setup["save_dir"] / "plots").mkdir(exist_ok=True)

    agent = create_morl_agent(algorithm, setup)
    return run_morl_training_loop(agent, setup, algorithm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MORL on CoatOpt")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config INI file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="morld",
        choices=["morld"],
        help="MORL algorithm to use",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override save directory from config",
    )

    args = parser.parse_args()

    # train() returns a results dict, not a (agent, front) pair.
    results = train(
        config_path=args.config,
        algorithm=args.algorithm,
        save_dir=args.save_dir,
    )
    print(f"Pareto front: {len(results['pareto_rewards'])} solutions")
