#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, DataConfig, TrainingConfig
from coatopt.utils import load_materials
from coatopt.algorithms.hppo.core.agent_simple import SimplePCHPPO
from coatopt.algorithms.hppo.config_simple import (
    HPPOAgentConfig,
    NetworkConfig,
    LearningConfig,
    PPOConfig,
    ActionSpaceConfig,
)

### IN PROGRESS
class SimpleHPPOTrainingLoop:
    """Minimal HPPO training loop with Pareto tracking.

    """

    def __init__(
        self,
        agent: SimplePCHPPO,
        env: CoatingEnvironment,
        save_dir: str = ".",
        materials: dict = None,
        verbose: int = 0,
        scheduler_start: int = 0,
        scheduler_end: int = -1,
    ):
        self.agent = agent
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.materials = materials or {}
        self.verbose = verbose
        self.scheduler_start = scheduler_start
        self.scheduler_end = scheduler_end

        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.pareto_front_values = []  # Value space (R, A)
        self.pareto_front_rewards = []  # Reward space (normalized)
        self.all_episode_rewards = []
        self.episode_count = 0

    def _update_environment_phase(self, episode: int):
        """Update environment phase and constraints based on episode (matches SB3 wrapper)."""
        env = self.env
        env.episode_count = episode + 1  

        # Phase 1: Warmup - optimize each objective individually
        if env.episode_count <= env.total_warmup_episodes:
            env.is_warmup = True
            warmup_idx = (env.episode_count - 1) // env.warmup_episodes_per_objective
            warmup_idx = min(warmup_idx, len(env.optimise_parameters) - 1)
            env.target_objective = env.optimise_parameters[warmup_idx]
            env.constraints = {}

            if self.verbose and (env.episode_count % 100 == 1 or env.episode_count <= 5):
                print(f"  [WARMUP] Episode {env.episode_count}/{env.total_warmup_episodes}: Target={env.target_objective}, constraints={env.constraints}")

        # Phase 2: Constrained - cycle through objectives with constraints
        else:
            if env.is_warmup and self.verbose:
                print(f"\n=== STARTING CONSTRAINED PHASE (Episode {env.episode_count}) ===")
                print(f"  Warmup best rewards: {env.warmup_best_rewards}\n")

            env.is_warmup = False
            constrained_episode = env.episode_count - env.total_warmup_episodes

            # Which constraint level (0 to steps_per_objective-1)
            level = ((constrained_episode - 1) // env.epochs_per_step) % env.steps_per_objective

            # Which objective is target
            obj_idx = ((constrained_episode - 1) // (env.epochs_per_step * env.steps_per_objective)) % len(env.optimise_parameters)
            env.target_objective = env.optimise_parameters[obj_idx]

            # Set constraints on other objectives (progressive tightening)
            env.constraints = {}
            for obj in env.optimise_parameters:
                if obj != env.target_objective:
                    best_value = env.warmup_best_rewards.get(obj, 0.0)
                    # Start at 50% of best, tighten to 90% of best
                    constraint_fraction = 0.5 + (0.4 * level / max(1, env.steps_per_objective - 1))
                    env.constraints[obj] = best_value * constraint_fraction


    def _is_dominated(self, vals1: dict, vals2: dict) -> bool:
        """Check if vals1 dominates vals2 (R higher is better, A lower is better)."""
        r1 = vals1.get('reflectivity', 0)
        r2 = vals2.get('reflectivity', 0)
        a1 = vals1.get('absorption', float('inf'))
        a2 = vals2.get('absorption', float('inf'))

        better_or_equal = (r1 >= r2) and (a1 <= a2)
        strictly_better = (r1 > r2) or (a1 < a2)
        return better_or_equal and strictly_better

    def _update_pareto_front(self, vals: dict, rewards: dict, state_array):
        """Update Pareto fronts in both value and reward space."""
        # Check for near-duplicates (tolerance for numerical precision)
        tol_r = 1e-6  # Reflectivity tolerance
        tol_a = 1e-2  # Absorption tolerance (smaller since values are ~1e-6)

        new_r = vals.get('reflectivity', 0)
        new_a = vals.get('absorption', float('inf'))

        # Skip if too similar to existing point
        for d in self.pareto_front_values:
            existing_r = d['vals'].get('reflectivity', 0)
            existing_a = d['vals'].get('absorption', float('inf'))
            if abs(new_r - existing_r) < tol_r and abs(new_a - existing_a) < tol_a:
                return  

        # Value space Pareto front
        design_info = {
            "vals": vals.copy(),
            "rewards": rewards.copy(),
            "episode": self.episode_count,
            "state_array": state_array.copy() if state_array is not None else None,
        }

        # Remove dominated by new point
        self.pareto_front_values = [
            d for d in self.pareto_front_values
            if not self._is_dominated(vals, d["vals"])
        ]

        # Add if not dominated
        if not any(self._is_dominated(d["vals"], vals) for d in self.pareto_front_values):
            self.pareto_front_values.append(design_info)

    def run_episode(self, objective_weights: dict = None):
        """Run single episode and collect experience in replay buffer."""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        # Default equal weights 
        if objective_weights is None:
            n_obj = len(self.env.optimise_parameters)
            objective_weights = {obj: 1.0 / n_obj for obj in self.env.optimise_parameters}

        # Convert objective weights dict to tensor with batch dimension [1, n_objectives]
        weights_list = [objective_weights[obj] for obj in self.env.optimise_parameters]
        weights_tensor = torch.FloatTensor(weights_list).unsqueeze(0)  # [1, n_objectives]

        while not done:
            # Get observation tensor for storing in buffer
            obs_tensor = state.get_observation_tensor(pre_type=self.agent.config.network.pre_type)

            # Add batch dimension if needed for LSTM [20, 7] -> [1, 20, 7]
            if self.agent.config.network.pre_type == 'lstm' and obs_tensor.dim() == 2:
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

            # Get action from agent - pass the CoatingState directly
            layer_num_array = np.array([episode_length], dtype=np.float32)

            (discrete_action, continuous_action, prob_a_d, prob_a_c, val, std,
             act_logp_d, act_logp_c, entropy_d, entropy_c, act_logp, moe_aux) = self.agent.select_action(
                state,  # Pass CoatingState directly
                layer_number=layer_num_array,
                objective_weights=weights_tensor,
            )

            # Convert to environment format and ensure correct types
            if isinstance(discrete_action, torch.Tensor):
                material_idx = int(discrete_action[0].item())
                discrete_action_tensor = discrete_action.long()  
            elif isinstance(discrete_action, (list, np.ndarray)):
                material_idx = int(discrete_action[0])
                discrete_action_tensor = torch.LongTensor([int(discrete_action[0])])  
            else:
                material_idx = int(discrete_action)
                discrete_action_tensor = torch.LongTensor([int(discrete_action)])  

            # Continuous action is float
            if isinstance(continuous_action, torch.Tensor):
                thickness = continuous_action[0].item()
                continuous_action_tensor = continuous_action.float()
            elif isinstance(continuous_action, (list, np.ndarray)):
                thickness = float(continuous_action[0])
                continuous_action_tensor = torch.FloatTensor([float(continuous_action[0])])
            else:
                thickness = float(continuous_action)
                continuous_action_tensor = torch.FloatTensor([float(continuous_action)])

            # Denormalize thickness for environment
            thickness_range = self.env.max_thickness - self.env.min_thickness
            thickness_denorm = self.env.min_thickness + thickness * thickness_range

            # Create action for environment
            action = np.zeros(self.env.n_materials + 1, dtype=np.float32)
            action[0] = thickness_denorm
            action[1 + material_idx] = 1.0

            # Environment step (environment handles constraints internally if enabled)
            next_state, rewards, terminated, finished, step_reward, _, vals = self.env.step(
                action, objective_weights=objective_weights
            )

            # Store experience in replay buffer
            self.agent.replay_buffer.update(
                discrete_action=discrete_action_tensor.detach() if isinstance(discrete_action_tensor, torch.Tensor) else discrete_action_tensor,
                continuous_action=continuous_action_tensor.detach() if isinstance(continuous_action_tensor, torch.Tensor) else continuous_action_tensor,
                state=state,  # CoatingState object
                observation=obs_tensor.detach() if isinstance(obs_tensor, torch.Tensor) else obs_tensor,
                logprob_discrete=act_logp_d.detach() if isinstance(act_logp_d, torch.Tensor) else act_logp_d,
                logprob_continuous=act_logp_c.detach() if isinstance(act_logp_c, torch.Tensor) else act_logp_c,
                reward=float(step_reward),  # Ensure scalar
                state_value=val.detach() if isinstance(val, torch.Tensor) else val,
                done=bool(finished),  # Ensure bool
                entropy_discrete=entropy_d.detach() if isinstance(entropy_d, torch.Tensor) else entropy_d,
                entropy_continuous=entropy_c.detach() if isinstance(entropy_c, torch.Tensor) else entropy_c,
                layer_number=int(episode_length),  # Ensure int
                objective_weights=weights_tensor.detach() if isinstance(weights_tensor, torch.Tensor) else weights_tensor,
            )

            state = next_state
            episode_reward += step_reward
            episode_length += 1
            done = finished

        return episode_reward, episode_length, vals, rewards, state

    def train(
        self,
        total_episodes: int = 1000,
        n_episodes_per_update: int = 10,
        n_updates_per_epoch: int = 10,
        plot_freq: int = 100,
    ):
        """Train agent with simple loop."""


        for episode in range(total_episodes):
            self.episode_count = episode + 1

            # Update environment training phase (for constrained training)
            if hasattr(self.env, 'use_constrained_training') and self.env.use_constrained_training:
                self._update_environment_phase(episode)

            # Sample random objective weights (multi-objective)
            n_obj = len(self.env.optimise_parameters)
            weights_array = np.random.dirichlet(np.ones(n_obj))
            objective_weights = {
                obj: weights_array[i]
                for i, obj in enumerate(self.env.optimise_parameters)
            }

            # Run episode
            ep_reward, ep_length, vals, rewards, final_state = self.run_episode(objective_weights)

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)

            # Track objectives
            if rewards:
                ref_reward = rewards.get('reflectivity', 0.0)
                abs_reward = rewards.get('absorption', 0.0)
                self.all_episode_rewards.append((ref_reward, abs_reward))

            # Update Pareto front
            if vals:
                state_array = final_state.get_array()
                self._update_pareto_front(vals, rewards, state_array)

            # Agent update every N episodes 
            if episode > 10 and (episode + 1) % n_episodes_per_update == 0 and len(self.agent.replay_buffer) > 0:
                buffer = self.agent.replay_buffer
                gamma = self.agent.config.ppo.gamma

                # Simple return computation 
                returns = []
                running_return = 0
                for i in reversed(range(len(buffer.rewards))):
                    if buffer.dones[i]:
                        running_return = 0
                    running_return = buffer.rewards[i] + gamma * running_return
                    returns.insert(0, running_return)

                buffer.update_returns(returns)

                # Update agent using collected experiences 
                # Perform multiple updates on the collected data
                total_policy_losses = []
                total_value_losses = []
                for update_idx in range(n_updates_per_epoch):
                    policy_loss, value_loss, total_loss = self.agent.update(
                        update_policy=True,
                        update_value=True
                    )
                    total_policy_losses.append(policy_loss)
                    total_value_losses.append(value_loss)

                # Average losses across updates
                policy_loss = np.mean(total_policy_losses) if total_policy_losses else 0.0
                value_loss = np.mean(total_value_losses) if total_value_losses else 0.0
                total_loss = policy_loss + value_loss

                # Step schedulers (learning rate and entropy coefficient) - once per epoch
                make_scheduler_step = self.scheduler_start <= episode <= (
                    self.scheduler_end if self.scheduler_end >= 0 else total_episodes
                )
                lr_discrete, lr_continuous, lr_value, beta_discrete, beta_continuous = self.agent.scheduler_step(
                    episode, make_step=make_scheduler_step, scheduler_active=make_scheduler_step
                )

                # Clear buffer after update (PPO is on-policy)
                self.agent.replay_buffer.clear()

            # Logging
            if self.verbose and (episode + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_best = max(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else max(self.episode_rewards)

                # Build phase info string
                phase_info = ""
                if hasattr(self.env, 'use_constrained_training') and self.env.use_constrained_training:
                    if self.env.is_warmup:
                        phase_info = f", phase=warmup_{self.env.target_objective}"
                    else:
                        phase_info = f", phase=constrained_{self.env.target_objective}"

                print(
                    f"Ep {episode + 1}/{total_episodes}: "
                    f"reward={ep_reward:.4f}, "
                    f"avg_10={np.mean(recent_rewards):.4f}, "
                    f"best_100={recent_best:.4f}, "
                    f"pareto={len(self.pareto_front_values)}, "
                    f"buffer={len(self.agent.replay_buffer)}"
                    f"{phase_info}"
                )

            # Periodic plotting
            if (episode + 1) % plot_freq == 0:
                self.plot_training_progress()
                self.plot_pareto_front()
                self.save_pareto_to_csv()

        print(f"\nTraining complete!")
        print(f"Final Pareto front size: {len(self.pareto_front_values)}")

        # Final saves
        self.plot_training_progress()
        self.plot_pareto_front()
        self.save_pareto_to_csv()

        return self.episode_rewards, self.pareto_front_values

    def plot_training_progress(self):
        """Plot training metrics with phase annotations."""
        if not self.episode_rewards:
            return

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Episode rewards
        axs[0, 0].plot(self.episode_rewards, alpha=0.6)
        axs[0, 0].set_title("Episode Rewards")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")

        # Add phase markers if using constrained training
        if hasattr(self.env, 'use_constrained_training') and self.env.use_constrained_training:
            # Warmup phases
            for i, obj in enumerate(self.env.optimise_parameters):
                warmup_start = i * self.env.warmup_episodes_per_objective
                warmup_end = (i + 1) * self.env.warmup_episodes_per_objective
                if warmup_start < len(self.episode_rewards):
                    axs[0, 0].axvline(warmup_start, color='gray', linestyle='--', alpha=0.3)
                    axs[0, 0].text(warmup_start, axs[0, 0].get_ylim()[1], f'W:{obj[:3]}',
                                  rotation=90, verticalalignment='top', fontsize=8)

            # Constrained phase start
            if self.env.total_warmup_episodes < len(self.episode_rewards):
                axs[0, 0].axvline(self.env.total_warmup_episodes, color='red', linestyle='-', alpha=0.5, linewidth=2)
                axs[0, 0].text(self.env.total_warmup_episodes, axs[0, 0].get_ylim()[1], 'Constrained',
                              rotation=90, verticalalignment='top', fontsize=10, color='red')
        if len(self.episode_rewards) > 50:
            window = min(50, len(self.episode_rewards) // 5)
            if window > 1:
                rolling = np.convolve(
                    self.episode_rewards, np.ones(window) / window, mode="valid"
                )
                axs[0, 0].plot(
                    range(window - 1, len(self.episode_rewards)),
                    rolling,
                    "r-",
                    linewidth=2,
                    label=f"Rolling avg ({window})",
                )
                axs[0, 0].legend()

        # Episode lengths
        axs[0, 1].plot(self.episode_lengths, alpha=0.6)
        axs[0, 1].set_title("Episode Lengths")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Layers")

        # Pareto front size over time
        axs[1, 0].axhline(len(self.pareto_front_values), color='r', linestyle='--')
        axs[1, 0].set_title("Pareto Front Size")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_ylabel("Size")
        axs[1, 0].text(
            0.5, len(self.pareto_front_values) + 1,
            f"Current: {len(self.pareto_front_values)}",
            ha='center'
        )

        # Objective rewards over time
        if self.all_episode_rewards:
            rewards = np.array(self.all_episode_rewards)
            axs[1, 1].scatter(rewards[:, 1], rewards[:, 0], alpha=0.3, s=10)
            axs[1, 1].set_xlabel("Absorption Reward")
            axs[1, 1].set_ylabel("Reflectivity Reward")
            axs[1, 1].set_title("Objective Space")
            axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "training_progress.png", dpi=150)
        plt.close(fig)

    def plot_pareto_front(self):
        """Plot Pareto front in value space."""
        if not self.pareto_front_values:
            return

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Value space
        reflectivities = [d['vals'].get('reflectivity', 0) for d in self.pareto_front_values]
        absorptions = [d['vals'].get('absorption', 1e-3) for d in self.pareto_front_values]

        axs[0].scatter(absorptions, reflectivities, c='red', s=100, marker='*', zorder=5)
        axs[0].set_xlabel("Absorption")
        axs[0].set_ylabel("Reflectivity")
        axs[0].set_xscale("log")
        axs[0].set_title(f"Pareto Front (Value Space) - {len(self.pareto_front_values)} points")
        axs[0].grid(True, alpha=0.3)

        # Plot 2: Reward space
        if self.all_episode_rewards:
            all_rewards = np.array(self.all_episode_rewards)
            axs[1].scatter(all_rewards[:, 1], all_rewards[:, 0], alpha=0.15, s=15, c='gray', label='All episodes')

        # Pareto front in reward space
        pareto_rewards = [(d['rewards'].get('reflectivity', 0), d['rewards'].get('absorption', 0))
                          for d in self.pareto_front_values if d.get('rewards')]
        if pareto_rewards:
            pareto_rewards = np.array(pareto_rewards)
            axs[1].scatter(pareto_rewards[:, 1], pareto_rewards[:, 0], c='red', s=60, marker='*',
                          label=f'Pareto front ({len(pareto_rewards)})', zorder=5, edgecolor='black')

        axs[1].set_xlabel("Absorption Reward (normalized)")
        axs[1].set_ylabel("Reflectivity Reward (normalized)")
        axs[1].set_title("Pareto Front (Reward Space)")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "pareto_front.png", dpi=150)
        plt.close(fig)

    def save_pareto_to_csv(self):
        """Save Pareto fronts to CSV files."""
        if not self.pareto_front_values:
            return

        import pandas as pd

        # Save VALUE space Pareto front
        values_data = []
        for design in self.pareto_front_values:
            row = {
                "episode": design.get("episode", -1),
                "reflectivity": design["vals"].get("reflectivity", 0),
                "absorption": design["vals"].get("absorption", 0),
            }
            # Add state info if available
            state_array = design.get("state_array")
            if state_array is not None:
                thicknesses = state_array[:, 0]
                active_mask = thicknesses > 1e-12
                active_thicknesses = thicknesses[active_mask]
                if len(active_thicknesses) > 0:
                    row["n_layers"] = len(active_thicknesses)
            values_data.append(row)

        df_values = pd.DataFrame(values_data)
        df_values = df_values.sort_values(by="absorption", ascending=True)
        values_path = self.save_dir / "pareto_front_values.csv"
        df_values.to_csv(values_path, index=False)
        print(f"Saved VALUE space Pareto front to {values_path} ({len(df_values)} points)")

        # Save REWARD space Pareto front
        rewards_data = []
        for design in self.pareto_front_values:
            rewards = design.get("rewards", {})
            if rewards:
                row = {
                    "episode": design.get("episode", -1),
                    "reflectivity_reward": rewards.get("reflectivity", 0),
                    "absorption_reward": rewards.get("absorption", 0),
                    "reflectivity": design["vals"].get("reflectivity", 0),
                    "absorption": design["vals"].get("absorption", 0),
                }
                # Add state info if available
                state_array = design.get("state_array")
                if state_array is not None:
                    thicknesses = state_array[:, 0]
                    active_mask = thicknesses > 1e-12
                    active_thicknesses = thicknesses[active_mask]
                    if len(active_thicknesses) > 0:
                        row["n_layers"] = len(active_thicknesses)
                rewards_data.append(row)

        if rewards_data:
            df_rewards = pd.DataFrame(rewards_data)
            df_rewards = df_rewards.sort_values(by="absorption_reward", ascending=False)
            rewards_path = self.save_dir / "pareto_front_rewards.csv"
            df_rewards.to_csv(rewards_path, index=False)
            print(f"Saved REWARD space Pareto front to {rewards_path} ({len(df_rewards)} points)")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train(config_path: str):
    """Train HPPO on CoatOpt.

    Args:
        config_path: Path to config INI file

    Returns:
        Trained agent
    """
    import configparser
    from coatopt.experiments.configs import load_config

    parser = configparser.ConfigParser()
    parser.read(config_path)

    # [General] section
    save_dir = parser.get('General', 'save_dir')
    materials_path = parser.get('General', 'materials_path')

    # [hppo] section
    total_episodes = parser.getint('hppo', 'total_episodes', fallback=1000)
    n_episodes_per_update = parser.getint('hppo', 'n_episodes_per_update', fallback=10)
    n_updates_per_epoch = parser.getint('hppo', 'n_updates_per_epoch', fallback=10)
    plot_freq = parser.getint('hppo', 'plot_freq', fallback=100)
    verbose = parser.getint('hppo', 'verbose', fallback=1)
    scheduler_start = parser.getint('hppo', 'scheduler_start', fallback=0)
    scheduler_end = parser.getint('hppo', 'scheduler_end', fallback=-1)

    # Constraint scheduling parameters (same as SB3)
    epochs_per_step = parser.getint('hppo', 'epochs_per_step', fallback=2000)
    steps_per_objective = parser.getint('hppo', 'steps_per_objective', fallback=10)
    warmup_episodes_per_objective = parser.getint('hppo', 'warmup_episodes_per_objective', fallback=2000)
    constraint_penalty = parser.getfloat('hppo', 'constraint_penalty', fallback=10.0)

    # Network parameters
    hidden_size = parser.getint('hppo', 'hidden_size', fallback=128)
    pre_type = parser.get('hppo', 'pre_type', fallback='lstm')

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    materials = load_materials(str(materials_path))
    print(f"Loaded {len(materials)} materials from {materials_path}")

    # Load config from file (includes constraint scheduling parameters)
    config = load_config(config_path)

    # Create environment
    env = CoatingEnvironment(config, materials)
    print(f"Environment created:")
    print(f"  Max layers: {env.max_layers}")
    print(f"  N materials: {env.n_materials}")
    print(f"  Objectives: {env.optimise_parameters}")

    # Enable constrained training if configured
    if config.training.cycle_weights == "constrained":
        env.enable_constrained_training(
            warmup_episodes_per_objective=warmup_episodes_per_objective,
            steps_per_objective=steps_per_objective,
            epochs_per_step=epochs_per_step,
            constraint_penalty=constraint_penalty,
        )
        print(f"  Constrained training enabled:")
        print(f"    Warmup: {warmup_episodes_per_objective} episodes/objective")
        print(f"    Steps per objective: {steps_per_objective}")
        print(f"    Epochs per step: {epochs_per_step}")
        print(f"    Total warmup: {warmup_episodes_per_objective * len(env.optimise_parameters)} episodes")

    # Create HPPO agent config
    # Get actual observation shape from environment
    test_state = env.reset()
    test_obs = test_state.get_observation_tensor(pre_type=pre_type)

    print(f"Test observation shape: {test_obs.shape}")

    # For LSTM: observation is [max_layers, n_features]
    # For linear: observation is flattened [max_layers * n_features]
    if pre_type == 'lstm':
        if test_obs.dim() == 2:
            state_dim = tuple(test_obs.shape)  # (max_layers, n_features)
        else:
            # If flattened, unflatten it
            n_features = 1 + env.n_materials + 2
            state_dim = (env.max_layers, n_features)
    else:  # linear
        state_dim = test_obs.shape[0] if test_obs.dim() == 1 else test_obs.numel()

    print(f"Agent state_dim: {state_dim}")
    print(f"Num materials: {env.n_materials}")
    print(f"Num objectives: {len(env.optimise_parameters)}")

    agent_config = HPPOAgentConfig(
        state_dim=state_dim,
        num_materials=env.n_materials,
        num_objectives=len(env.optimise_parameters),
        network=NetworkConfig(
            pre_type=pre_type,
            hidden_size=hidden_size,
            n_pre_layers=2,
            discrete_hidden_size=32,
            continuous_hidden_size=32,
            value_hidden_size=32,
            activation_function='relu',
        ),
        learning=LearningConfig(
            lr_discrete=1e-4,
            lr_continuous=1e-4,
            lr_value=2e-4,
            optimiser='adam',
        ),
        ppo=PPOConfig(
            clip_ratio=0.2,
            gamma=0.99,
            buffer_size=10000,
            batch_size=64,
            entropy_beta_start=0.1,
            entropy_beta_end=0.01,
            entropy_beta_decay_length=500,
        ),
        action=ActionSpaceConfig(
            lower_bound=0.0,  # Will be normalized to [0,1]
            upper_bound=1.0,
        ),
    )

    # Create agent
    agent = SimplePCHPPO(agent_config)
    print(f"\nAgent created:")
    print(f"  State dim: {state_dim}")
    print(f"  Pre-network: {pre_type}")
    print(f"  Hidden size: {hidden_size}")

    # Create training loop
    trainer = SimpleHPPOTrainingLoop(
        agent=agent,
        env=env,
        save_dir=str(save_dir),
        materials=materials,
        verbose=verbose,
        scheduler_start=scheduler_start,
        scheduler_end=scheduler_end,
    )

    # Train
    episode_rewards, pareto_front = trainer.train(
        total_episodes=total_episodes,
        n_episodes_per_update=n_episodes_per_update,
        n_updates_per_epoch=n_updates_per_epoch,
        plot_freq=plot_freq,
    )

    # Save agent
    agent_path = save_dir / "hppo_agent.pt"
    torch.save(agent.state_dict(), agent_path)
    print(f"\nAgent saved to {agent_path}")

    return agent


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HPPO on CoatOpt")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config INI file"
    )

    args = parser.parse_args()
    train(args.config)
