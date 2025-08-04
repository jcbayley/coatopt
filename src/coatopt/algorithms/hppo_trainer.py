"""
HPPO Trainer for managing training process.
Refactored from pc_hppo_oml.py for improved readability and maintainability.
"""
import os
import time
import pickle
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from coatopt.algorithms.config import HPPOConstants
from coatopt.algorithms.plotting_utils import make_reward_plot, make_val_plot, make_loss_plot, make_materials_plot
from coatopt.algorithms.pc_hppo_agent import PCHPPO


class HPPOTrainer:
    """
    Trainer class for managing HPPO training process.
    
    Handles training loop, metric tracking, plotting, and model persistence.
    """

    def __init__(
        self,
        agent: PCHPPO, 
        env, 
        n_iterations: int = 1000,  
        n_layers: int = 4, 
        root_dir: str = "./",
        beta_start: float = 1.0,
        beta_end: float = 0.001,
        beta_decay_length: Optional[int] = None,
        beta_decay_start: int = 0,
        n_training_epochs: int = 10,
        use_obs: bool = True,
        scheduler_start: int = 0,
        scheduler_end: int = np.inf,
        continue_training: bool = False,
        weight_network_save: bool = False
    ):
        """
        Initialize HPPO trainer.
        
        Args:
            agent: PC-HPPO agent to train
            env: Environment for training
            n_iterations: Number of training iterations
            n_layers: Number of layers in coating
            root_dir: Root directory for outputs
            beta_start: Initial entropy coefficient
            beta_end: Final entropy coefficient
            beta_decay_length: Entropy decay length
            beta_decay_start: Episode to start entropy decay
            n_training_epochs: Episodes per training iteration
            use_obs: Whether to use observations vs raw states
            scheduler_start: Episode to start learning rate scheduling
            scheduler_end: Episode to end learning rate scheduling
            continue_training: Whether to continue from checkpoint
            weight_network_save: Whether to save network weights periodically
        """
        self.agent = agent
        self.env = env
        self.n_iterations = n_iterations
        self.root_dir = root_dir
        self.n_layers = n_layers
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay_length = beta_decay_length
        self.beta_decay_start = beta_decay_start
        self.n_training_epochs = n_training_epochs
        self.use_obs = use_obs
        self.weight_network_save = weight_network_save

        # Setup directories
        self._setup_directories()

        # Setup scheduler
        self.scheduler_start = scheduler_start
        self.scheduler_end = n_iterations if scheduler_end == -1 else scheduler_end

        # Initialize or load training state
        if continue_training:
            self._load_training_state()
        else:
            self._initialize_training_state()

    def _setup_directories(self) -> None:
        """Create necessary directories for training outputs."""
        os.makedirs(self.root_dir, exist_ok=True)
        
        if self.weight_network_save:
            self.network_weights_dir = os.path.join(self.root_dir, HPPOConstants.NETWORK_WEIGHTS_DIR)
            os.makedirs(self.network_weights_dir, exist_ok=True)

    def _initialize_training_state(self) -> None:
        """Initialize training state for new training run."""
        self.metrics = pd.DataFrame(columns=[
            "episode", "loss_policy_continuous", "loss_policy_discrete", "beta", 
            "lr_discrete", "lr_continuous", "lr_value", "reward", "reflectivity", 
            "thermal_noise", "thickness", "absorption", "reflectivity_reward", 
            "thermal_reward", "thickness_reward", "absorption_reward",
            "reflectivity_weight", "thermal_noise_weight", "thickness_weight",
            "absorption_weight", "reflectivity_reward_weights", "absorption_reward_weights",
            "thermalnoise_reward_weights"
        ])
        self.start_episode = 0
        self.continue_training = False
        self.best_states = []

    def _load_training_state(self) -> None:
        """Load training state from checkpoint."""
        self.load_metrics_from_file()
        self.start_episode = self.metrics["episode"].max()
        
        best_states_path = os.path.join(self.root_dir, HPPOConstants.BEST_STATES_FILE)
        with open(best_states_path, "rb") as f:
            self.best_states = pickle.load(f)
        
        self.continue_training = True

    def write_metrics_to_file(self) -> None:
        """Save training metrics to CSV file."""
        metrics_path = os.path.join(self.root_dir, HPPOConstants.TRAINING_METRICS_FILE)
        self.metrics.to_csv(metrics_path, index=False)

    def load_metrics_from_file(self) -> None:
        """Load training metrics from CSV file."""
        metrics_path = os.path.join(self.root_dir, HPPOConstants.TRAINING_METRICS_FILE)
        self.metrics = pd.read_csv(metrics_path)

    def make_plots(self) -> None:
        """Create all training plots."""
        make_reward_plot(self.metrics, self.root_dir)
        make_val_plot(self.metrics, self.root_dir)
        make_loss_plot(self.metrics, self.root_dir)

    def update_best_states(self, state: np.ndarray, rewards: Dict[str, float]) -> None:
        """
        Update list of best states encountered during training.
        
        Args:
            state: Coating state
            rewards: Reward dictionary
        """
        self.best_states.append((state, rewards))

    def train(self) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Main training loop.
        
        Returns:
            Tuple of (final_rewards, final_state)
        """
        print(f"Starting training from episode {self.start_episode} to {self.n_iterations}")
        
        # Initialize tracking variables
        all_means, all_stds, all_materials = [], [], []
        max_reward = self._get_initial_max_reward()
        max_state = None

        # Create states directory
        states_dir = os.path.join(self.root_dir, HPPOConstants.STATES_DIR)
        os.makedirs(states_dir, exist_ok=True)

        start_time = time.time()
        
        # Main training loop
        for episode in range(self.start_episode, self.n_iterations):
            episode_start_time = time.time()
            
            # Determine if scheduler should step
            make_scheduler_step = self.scheduler_start <= episode <= self.scheduler_end
            
            # Run training episode
            episode_metrics, episode_data, episode_reward, final_state = self._run_training_episode(episode)
            
            # Update best states and reward tracking
            self._update_episode_tracking(episode_data, all_means, all_stds, all_materials)
            max_reward, max_state = self._update_max_reward(episode_reward, final_state, max_reward, max_state)
            
            # Perform network updates
            self._perform_network_updates(episode, episode_metrics, make_scheduler_step)
            
            # Save network weights if needed
            self._save_network_weights_if_needed(episode, episode_data['objective_weights'])
            
            # Record metrics
            self._record_episode_metrics(episode, episode_metrics, episode_reward, final_state, episode_data)
            
            # Periodic tasks (plotting, saving, etc.)
            if episode % HPPOConstants.EPISODE_PRINT_INTERVAL == 0 and episode != 0:
                self._perform_periodic_tasks(episode, all_materials[0], episode_start_time, start_time, final_state, episode_reward)

        print(f"Training complete. Max reward: {max_reward}")
        return self.metrics.iloc[-1].to_dict(), max_state

    def _get_initial_max_reward(self) -> float:
        """Get initial maximum reward from existing metrics."""
        return np.max(self.metrics["reward"]) if len(self.metrics) > 0 else -np.inf

    def _run_training_episode(self, episode: int) -> Tuple[Dict, Dict, float, np.ndarray]:
        """
        Run a single training episode with multiple environment rollouts.
        
        Args:
            episode: Current episode number
            
        Returns:
            Tuple of (metrics_dict, episode_data, best_reward, best_state)
        """
        episode_metrics = {}
        min_episode_score = (-np.inf, 0, None, None, None)
        means_list, stds_list, materials_list = [], [], []
        
        # Run multiple rollouts per episode
        for rollout in range(self.n_training_epochs):
            rollout_data = self._run_single_rollout(episode)
            
            # Track rollout data
            means_list.extend(rollout_data['means'])
            stds_list.extend(rollout_data['stds'])
            materials_list.extend(rollout_data['materials'])
            
            # Update best score for this episode
            if rollout_data['total_reward'] > min_episode_score[0]:
                min_episode_score = (
                    rollout_data['total_reward'], rollout, 
                    rollout_data['final_state'], rollout_data['rewards'], 
                    rollout_data['vals']
                )
        
        # Store best episode data
        self.best_states.append(min_episode_score)
        
        episode_data = {
            'means': means_list,
            'stds': stds_list,
            'materials': materials_list,
            'objective_weights': rollout_data['objective_weights']  # Use last rollout's weights
        }
        
        return episode_metrics, episode_data, min_episode_score[0], min_episode_score[2]

    def _run_single_rollout(self, episode: int) -> Dict[str, Any]:
        """
        Run a single environment rollout.
        
        Args:
            episode: Current episode number
            
        Returns:
            Dictionary containing rollout data
        """
        state = self.env.reset()
        total_reward = 0
        means, stds, materials = [], [], []
        rewards_list = []
        
        # Sample objective weights for this rollout
        objective_weights = self.env.sample_reward_weights(epoch=episode)
        
        # Run episode steps
        for step in range(HPPOConstants.MAX_EPISODE_STEPS):
            # Prepare inputs
            obs = self.env.get_observation_from_state(state) if self.use_obs else state
            step_tensor = np.array([step])
            objective_weights_tensor = torch.tensor(objective_weights).unsqueeze(0).to(torch.float32)
            
            # Select action
            action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_prob, c_means, c_std, value, entropy_discrete, entropy_continuous = self.agent.select_action(
                obs, step_tensor, objective_weights=objective_weights_tensor
            )
            
            # Scale continuous action to environment bounds
            action[1] = action[1] * (self.env.max_thickness - self.env.min_thickness) + self.env.min_thickness
            
            # Take environment step
            next_state, rewards, done, finished, _, full_action, vals = self.env.step(action, objective_weights=objective_weights)
            reward = rewards["total_reward"]
            
            # Store experience in replay buffer
            self.agent.replay_buffer.update(
                actiond.detach(), actionc.detach(), obs, log_prob_discrete.detach(),
                log_prob_continuous.detach(), reward, value, done,
                entropy_discrete.detach(), entropy_continuous.detach(), step_tensor,
                objective_weights=objective_weights_tensor
            )
            
            # Track rollout data
            means.append(c_means.detach().numpy())
            stds.append(c_std.detach().numpy())
            materials.append(d_prob.detach().numpy().tolist()[0])
            rewards_list.append(reward)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            if done or finished:
                break
        
        # Calculate returns and update replay buffer
        returns = self.agent.get_returns(rewards_list)
        self.agent.replay_buffer.update_returns(returns)
        
        return {
            'total_reward': total_reward,
            'final_state': state,
            'rewards': rewards,
            'vals': vals,
            'means': means,
            'stds': stds,
            'materials': materials,
            'objective_weights': objective_weights
        }

    def _update_episode_tracking(self, episode_data: Dict, all_means: List, all_stds: List, all_materials: List) -> None:
        """Update episode tracking data."""
        all_means.append(episode_data['means'])
        all_stds.append(episode_data['stds'])
        all_materials.append(episode_data['materials'])

    def _update_max_reward(self, episode_reward: float, final_state: np.ndarray, max_reward: float, max_state: Optional[np.ndarray]) -> Tuple[float, Optional[np.ndarray]]:
        """Update maximum reward and associated state."""
        if episode_reward > max_reward:
            return episode_reward, final_state
        return max_reward, max_state

    def _perform_network_updates(self, episode: int, episode_metrics: Dict, make_scheduler_step: bool) -> None:
        """Perform network parameter updates."""
        if episode > HPPOConstants.DEFAULT_UPDATE_FREQUENCY:
            loss1, loss2, loss3 = self.agent.update(update_policy=True, update_value=True)
            lr_outs = self.agent.scheduler_step(episode, make_scheduler_step)
            
            # Update agent beta and store metrics
            self.agent.beta = lr_outs[3]
            episode_metrics.update({
                "lr_discrete": lr_outs[0][0],
                "lr_continuous": lr_outs[1][0], 
                "lr_value": lr_outs[2][0],
                "beta": self.agent.beta,
                "loss_policy_discrete": loss1,
                "loss_policy_continuous": loss2,
                "loss_value": loss3
            })
        else:
            # No updates yet, use default values
            episode_metrics.update({
                "lr_discrete": self.agent.disc_lr_policy,
                "lr_continuous": self.agent.cont_lr_policy,
                "lr_value": self.agent.lr_value,
                "beta": self.agent.beta,
                "loss_policy_discrete": np.nan,
                "loss_policy_continuous": np.nan,
                "loss_value": np.nan
            })

    def _save_network_weights_if_needed(self, episode: int, objective_weights: np.ndarray) -> None:
        """Save network weights periodically if enabled."""
        if self.weight_network_save and (episode + 1) % self.agent.lr_step == 0:
            weights_str = "_".join([str(round(float(w), 3)) for w in objective_weights])
            fname = os.path.join(self.network_weights_dir, f"weights_{weights_str}.pt")
            torch.save({
                "discrete_policy": self.agent.policy_discrete.state_dict(),
                "continuous_policy": self.agent.policy_continuous.state_dict(),
                "value": self.agent.value.state_dict(),
                "objective_weights": objective_weights,
                "weights": objective_weights
            }, fname)

    def _record_episode_metrics(self, episode: int, episode_metrics: Dict, episode_reward: float, final_state: np.ndarray, episode_data: Dict) -> None:
        """Record metrics for the current episode."""
        episode_metrics["episode"] = episode
        episode_metrics["reward"] = episode_reward
        
        # Compute final state values and rewards
        _, vals, rewards = self.env.compute_reward(final_state)
        
        # Store physical values
        for key in ["reflectivity", "thermal_noise", "absorption", "thickness"]:
            episode_metrics[key] = vals.get(key, 0.0)
            episode_metrics[f"{key}_reward"] = rewards.get(key, 0.0)
        
        # Store objective weights (use last rollout's weights)
        episode_metrics["reflectivity_reward_weights"] = episode_data['objective_weights'][0]
        episode_metrics["thermalnoise_reward_weights"] = episode_data['objective_weights'][1]
        episode_metrics["absorption_reward_weights"] = episode_data['objective_weights'][2]

        
        # Add to metrics dataframe
        self.metrics = pd.concat([self.metrics, pd.DataFrame([episode_metrics])], ignore_index=True)

    def _perform_periodic_tasks(self, episode: int, all_materials: List, episode_start_time: float, start_time: float, final_state: np.ndarray, episode_reward: float) -> None:
        """Perform periodic tasks like plotting, saving, and logging."""
        # Save metrics and create plots
        self.write_metrics_to_file()
        self.make_plots()
        
        # Create materials plot
        make_materials_plot(all_materials, self.n_layers, self.root_dir, self.env.n_materials)
        
        # Save pareto front data
        self._save_pareto_front_data(episode)
        
        # Save best states
        best_states_path = os.path.join(self.root_dir, HPPOConstants.BEST_STATES_FILE)
        with open(best_states_path, "wb") as f:
            pickle.dump(self.best_states, f)
        
        # Print progress
        episode_time = time.time() - episode_start_time
        total_time = time.time() - start_time
        print(f"Episode {episode + 1}: Total Reward: {episode_reward:.4f}, "
              f"Episode time: {episode_time:.2f}s, Total time: {total_time:.2f}s")
        
        # Save episode state visualization
        if episode % HPPOConstants.SAVE_INTERVAL == 0:
            self._save_episode_visualization(episode, final_state, episode_reward)

    def _save_pareto_front_data(self, episode: int) -> None:
        """Save Pareto front data."""
        pareto_dir = os.path.join(self.root_dir, HPPOConstants.PARETO_FRONTS_DIR)
        os.makedirs(pareto_dir, exist_ok=True)
        
        # Save Pareto front
        pareto_file = os.path.join(pareto_dir, f"pareto_front_it{episode}.txt")
        np.savetxt(pareto_file, self.env.pareto_front, header="Reflectivity, Absorption")
        
        # Save all points
        all_points_file = os.path.join(self.root_dir, HPPOConstants.ALL_POINTS_FILE)
        np.savetxt(all_points_file, self.env.saved_points)
        
        all_data_file = os.path.join(self.root_dir, HPPOConstants.ALL_POINTS_DATA_FILE)
        np.savetxt(all_data_file, self.env.saved_data)

    def _save_episode_visualization(self, episode: int, state: np.ndarray, reward: float) -> None:
        """Save visualization of episode state."""
        states_dir = os.path.join(self.root_dir, HPPOConstants.STATES_DIR)
        
        fig, ax = self.env.plot_stack(state)
        opt_value = self.env.compute_state_value(state, return_separate=True)
        ax.set_title(f"Episode {episode}: Reward: {reward:.4f}, Value: {opt_value}")
        
        episode_file = os.path.join(states_dir, f"episode_{episode}.png")
        fig.savefig(episode_file)
        plt.close(fig)

    def init_pareto_front(self, n_solutions: int = 1000) -> None:
        """
        Initialize Pareto front with random solutions.
        
        Args:
            n_solutions: Number of random solutions to generate
        """
        print(f"Initializing Pareto front with {n_solutions} random solutions...")
        
        sol_vals = {"reflectivity": [], "thermal_noise": [], "absorption": [], "thickness": []}
        
        for _ in range(n_solutions):
            state = self.env.sample_state_space(random_material=True)
            reflectivity, thermal_noise, absorption, thickness = self.env.compute_state_value(state, return_separate=True)
            
            _, _, rewards = self.env.select_reward(reflectivity, thermal_noise, thickness, absorption, weights=None)
            
            for key in sol_vals.keys():
                sol_vals[key].append(rewards[key])

        # Create Pareto front from objective values
        vals = np.array([sol_vals[key] for key in self.env.optimise_parameters]).T
        
        # Perform non-dominated sorting
        nds = NonDominatedSorting()
        fronts = nds.do(vals)
        
        # Set Pareto front and reference point
        self.env.pareto_front = vals[fronts[0]]
        self.env.reference_point = np.max(self.env.pareto_front, axis=0) * 1.1
        
        print(f"Initialized Pareto front with {len(self.env.pareto_front)} solutions")

    def generate_solutions(self, n_solutions: int, random_weights: bool = True) -> Tuple[List[np.ndarray], List[Dict], List[np.ndarray], List[Dict]]:
        """
        Generate solutions using trained policy.
        
        Args:
            n_solutions: Number of solutions to generate
            random_weights: Whether to use random objective weights
            
        Returns:
            Tuple of (states, rewards, weights, vals)
        """
        print(f"Generating {n_solutions} solutions...")
        
        # Load trained networks
        self.agent.load_networks(self.root_dir)
        
        all_rewards, all_states, all_vals, weights = [], [], [], []
        
        # Generate objective weights if not random
        if not random_weights:
            num_points = int(n_solutions ** (1 / self.agent.num_objectives))
            objweights = np.array(np.meshgrid(*[np.linspace(0, 1, num_points) for _ in range(self.agent.num_objectives)]))
            objweights = objweights.reshape(self.agent.num_objectives, -1).T
            objweights = np.round(objweights / objweights.sum(axis=1, keepdims=True), 2)
            objweights = np.unique(objweights, axis=0)
            objweights = objweights[~np.isnan(objweights).any(axis=1)]

        # Generate solutions
        for n in range(n_solutions):
            state = self.env.reset()
            
            # Get objective weights
            if random_weights:
                objective_weights = self.env.sample_reward_weights(epoch=self.env.final_weight_epoch + 10)
            else:
                if n >= len(objweights):
                    break
                objective_weights = objweights[n]
            
            # Run episode
            for step in range(HPPOConstants.MAX_EPISODE_STEPS):
                obs = self.env.get_observation_from_state(state) if self.use_obs else state
                step_tensor = np.array([step])
                objective_weights_tensor = torch.tensor(objective_weights).unsqueeze(0).to(torch.float32)
                
                action, _, _, _, _, _, _, _, _, _, _ = self.agent.select_action(
                    obs, step_tensor, objective_weights=objective_weights_tensor
                )
                
                action[1] = action[1] * (self.env.max_thickness - self.env.min_thickness) + self.env.min_thickness
                
                next_state, rewards, done, finished, _, full_action, vals = self.env.step(action, objective_weights=objective_weights)
                state = next_state
                
                if done or finished:
                    break
            
            # Store results
            all_rewards.append(rewards)
            all_states.append(next_state)
            weights.append(objective_weights)
            all_vals.append(vals)

        print(f"Generated {len(all_states)} solutions")
        return all_states, all_rewards, weights, all_vals