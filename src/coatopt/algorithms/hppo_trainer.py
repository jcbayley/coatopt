"""
Unified HPPO Trainer for managing training process.
Handles all training scenarios: CLI, UI, and standalone usage.
Refactored from pc_hppo_oml.py for improved readability and maintainability.
"""
import os
import time
import pickle
import queue
from typing import List, Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from coatopt.algorithms.config import HPPOConstants
from coatopt.algorithms.plotting_utils import make_reward_plot, make_val_plot, make_loss_plot, make_materials_plot
from coatopt.algorithms.pc_hppo_agent import PCHPPO
from coatopt.algorithms.training_checkpoint_manager import TrainingCheckpointManager
import traceback


@dataclass
class TrainingCallbacks:
    """Callbacks for training progress reporting and control."""
    
    # Progress reporting
    on_episode_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None
    on_periodic_update: Optional[Callable[[int, Dict[str, Any]], None]] = None
    on_training_start: Optional[Callable[[Dict[str, Any]], None]] = None
    on_training_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    
    # Control
    should_stop: Optional[Callable[[], bool]] = None
    
    # Configuration
    save_plots: bool = False
    save_visualizations: bool = False
    progress_interval: int = 10  # Episodes between progress updates
    summary_interval: int = 100  # Episodes between detailed summaries
    
    # UI-specific
    ui_queue: Optional[queue.Queue] = None  # For sending data to UI

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
        entropy_beta_start: float = 1.0,
        entropy_beta_end: float = 0.001,
        entropy_beta_decay_length: Optional[int] = None,
        entropy_beta_decay_start: int = 0,
        n_epochs_per_update: int = 10,
        use_obs: bool = True,
        scheduler_start: int = 0,
        scheduler_end: int = np.inf,
        continue_training: bool = False,
        weight_network_save: bool = False,
        use_unified_checkpoints: bool = True,
        save_plots: bool = False,
        save_episode_visualizations: bool = False,
        callbacks: Optional[TrainingCallbacks] = None
    ):
        """
        Initialize HPPO trainer.
        
        Args:
            agent: PC-HPPO agent to train
            env: Environment for training
            n_iterations: Number of training iterations
            n_layers: Number of layers in coating
            root_dir: Root directory for outputs
            entropy_beta_start: Initial entropy coefficient
            entropy_beta_end: Final entropy coefficient
            entropy_beta_decay_length: Entropy decay length
            entropy_beta_decay_start: Episode to start entropy decay
            n_epochs_per_update: Episodes per training iteration
            use_obs: Whether to use observations vs raw states
            scheduler_start: Episode to start learning rate scheduling
            scheduler_end: Episode to end learning rate scheduling
            continue_training: Whether to continue from checkpoint
            weight_network_save: Whether to save network weights periodically
            use_unified_checkpoints: Whether to use unified HDF5 checkpoints (recommended)
            save_plots: Whether to save plot files to disk (False = plots only in checkpoint)
            save_episode_visualizations: Whether to save individual episode PNG files
            callbacks: Optional callbacks for progress reporting and control
        """
        self.agent = agent
        self.env = env
        self.n_iterations = n_iterations
        self.root_dir = root_dir
        self.n_layers = n_layers
        self.entropy_beta_start = entropy_beta_start
        self.entropy_beta_end = entropy_beta_end
        self.entropy_beta_decay_length = entropy_beta_decay_length
        self.entropy_beta_decay_start = entropy_beta_decay_start
        self.n_epochs_per_update = n_epochs_per_update
        self.use_obs = use_obs
        self.weight_network_save = weight_network_save
        self.use_unified_checkpoints = use_unified_checkpoints
        self.save_plots = False
        self.save_episode_visualizations = save_episode_visualizations
        
        # Callback system
        self.callbacks = callbacks or TrainingCallbacks()
        self.training_start_time = None
        self.last_progress_time = time.time()

        # Setup directories
        self._setup_directories()

        # Setup unified checkpoint manager
        if self.use_unified_checkpoints:
            self.checkpoint_manager = TrainingCheckpointManager(
                root_dir=self.root_dir,
                checkpoint_name="training_checkpoint.h5"
            )

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
        """Load training state from checkpoint (unified or legacy format)."""
        if self.use_unified_checkpoints and self._has_unified_checkpoint():
            self._load_unified_checkpoint()
        else:
            # Load from legacy format
            self._load_legacy_training_state()

    def _has_unified_checkpoint(self) -> bool:
        """Check if unified checkpoint exists."""
        return hasattr(self, 'checkpoint_manager') and os.path.exists(self.checkpoint_manager.checkpoint_path)

    def _load_unified_checkpoint(self) -> None:
        """Load complete training state from unified checkpoint."""
        try:
            print("Loading from unified checkpoint...")
            checkpoint_data = self.checkpoint_manager.load_complete_checkpoint()
            
            if not checkpoint_data:
                print("Warning: Empty checkpoint data, falling back to legacy or initializing new")
                self._load_legacy_training_state()
                return
            
            # Load training data
            training_data = checkpoint_data.get('training_data', {})
            self.metrics = training_data.get('metrics_df', pd.DataFrame())
            
            if not self.metrics.empty:
                self.start_episode = self.metrics["episode"].max()
            else:
                self.start_episode = 0
            
            # Load environment state
            env_state = checkpoint_data.get('environment_state', {})
            pareto_data = checkpoint_data.get('pareto_data', {})
            
            # Restore Pareto front
            if 'current_front' in pareto_data and len(pareto_data['current_front']) > 0:
                self.env.pareto_front = pareto_data['current_front']
                print(f"Loaded Pareto front with {len(self.env.pareto_front)} points")
            else:
                self.env.pareto_front = np.empty((0, len(getattr(self.env, 'optimise_parameters', ['reflectivity', 'absorption']))))
                print("No Pareto front in checkpoint, initializing empty")
            
            # Restore other environment data
            if 'reference_point' in pareto_data:
                self.env.reference_point = pareto_data['reference_point']
            elif len(self.env.pareto_front) > 0:
                self.env.reference_point = np.max(self.env.pareto_front, axis=0) * 1.1
            
            if 'all_points' in pareto_data:
                self.env.saved_points = pareto_data['all_points'].tolist() if len(pareto_data['all_points']) > 0 else []
            else:
                self.env.saved_points = []
                
            if 'all_values' in pareto_data:
                self.env.saved_data = pareto_data['all_values'].tolist() if len(pareto_data['all_values']) > 0 else []
            else:
                self.env.saved_data = []
            
            # Load best states
            self.best_states = checkpoint_data.get('best_states', [])
            
            self.continue_training = True
            print(f"Successfully loaded unified checkpoint from episode {self.start_episode}")
            
            # Send initial pareto update to UI if we have callbacks and best_states
            if hasattr(self, 'callbacks') and self.callbacks.ui_queue and self.best_states:
                try:
                    pareto_update = self._generate_pareto_update(self.start_episode)
                    if pareto_update:
                        self.callbacks.ui_queue.put(pareto_update, block=False)
                        print(f"Sent initial Pareto update to UI with {len(self.best_states)} points")
                except Exception as e:
                    print(f"Warning: Could not send initial pareto update to UI: {e}")
            
        except Exception as e:
            print(f"Error loading unified checkpoint: {e}: traceback: {traceback.format_exc()}")
            print("Falling back to legacy loading...")
            self._load_legacy_training_state()

    def _load_legacy_training_state(self) -> None:
        """Load training state from legacy scattered files."""
        try:
            self.load_metrics_from_file()
            self.start_episode = self.metrics["episode"].max() if not self.metrics.empty else 0
            
            best_states_path = os.path.join(self.root_dir, HPPOConstants.BEST_STATES_FILE)
            if os.path.exists(best_states_path):
                with open(best_states_path, "rb") as f:
                    self.best_states = pickle.load(f)
            else:
                self.best_states = []
            
            # Load Pareto front into environment
            self._load_pareto_front()
            
            self.continue_training = True
            
            # If using unified checkpoints, migrate data after loading
            if self.use_unified_checkpoints:
                print("Migrating legacy data to unified checkpoint...")
                self._save_unified_checkpoint(episode=self.start_episode, migration=True)
                
        except Exception as e:
            print(f"Error loading legacy training state: {e}")
            self._initialize_training_state()

    def _load_pareto_front(self) -> None:
        """Load Pareto front from saved file into environment."""
        pareto_file = os.path.join(self.root_dir, HPPOConstants.PARETO_FRONT_FILE)
        all_points_file = os.path.join(self.root_dir, HPPOConstants.ALL_POINTS_FILE)
        all_data_file = os.path.join(self.root_dir, HPPOConstants.ALL_POINTS_DATA_FILE)
        
        try:
            if os.path.exists(pareto_file):
                pareto_front = np.loadtxt(pareto_file)
                # Handle case where only one point exists (1D array)
                if pareto_front.ndim == 1:
                    pareto_front = pareto_front.reshape(1, -1)
                self.env.pareto_front = pareto_front
                
                # Update reference point based on loaded Pareto front
                if len(pareto_front) > 0:
                    self.env.reference_point = np.max(self.env.pareto_front, axis=0) * 1.1
                
                print(f"Loaded Pareto front with {len(pareto_front)} points")
            else:
                print("No Pareto front file found, initializing empty Pareto front")
                self.env.pareto_front = np.empty((0, len(self.env.optimise_parameters)))
            
            # Load saved points and data if they exist
            if os.path.exists(all_points_file):
                saved_points = np.loadtxt(all_points_file)
                if saved_points.ndim == 1 and len(saved_points) > 0:
                    saved_points = saved_points.reshape(1, -1)
                self.env.saved_points = saved_points.tolist() if saved_points.size > 0 else []
            else:
                self.env.saved_points = []
                
            if os.path.exists(all_data_file):
                saved_data = np.loadtxt(all_data_file)
                if saved_data.ndim == 1 and len(saved_data) > 0:
                    saved_data = saved_data.reshape(1, -1)
                self.env.saved_data = saved_data.tolist() if saved_data.size > 0 else []
            else:
                self.env.saved_data = []
                
        except Exception as e:
            print(f"Warning: Failed to load Pareto front data: {e}")
            print("Initializing empty Pareto front")
            self.env.pareto_front = np.empty((0, len(self.env.optimise_parameters)))
            self.env.saved_points = []
            self.env.saved_data = []

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
        # Only make plots if requested (via callbacks or legacy save_plots)
        if self.callbacks.save_plots or self.save_plots:
            make_reward_plot(self.metrics, self.root_dir)
            make_val_plot(self.metrics, self.root_dir)
            make_loss_plot(self.metrics, self.root_dir)


    def train(self) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Main training loop with callback support.
        
        Returns:
            Tuple of (final_rewards, final_state)
        """
        self.training_start_time = time.time()
        
        # Notify training start
        if self.callbacks.on_training_start:
            start_info = {
                'start_episode': self.start_episode,
                'total_episodes': self.n_iterations,
                'output_directory': self.root_dir,
                'continue_training': self.start_episode > 0
            }
            self.callbacks.on_training_start(start_info)
        
        print(f"Starting training from episode {self.start_episode} to {self.n_iterations}")
        
        try:
            final_metrics, best_state = self._run_training_loop()
            
            # Notify completion
            if self.callbacks.on_training_complete:
                completion_info = {
                    'success': True,
                    'final_metrics': final_metrics,
                    'best_state': best_state,
                    'total_time': time.time() - self.training_start_time,
                    'final_episode': getattr(self, 'current_episode', self.n_iterations - 1)
                }
                self.callbacks.on_training_complete(completion_info)
            
            return final_metrics, best_state
            
        except Exception as e:
            # Notify failure
            if self.callbacks.on_training_complete:
                completion_info = {
                    'success': False,
                    'error': str(e),
                    'total_time': time.time() - self.training_start_time,
                    'final_episode': getattr(self, 'current_episode', self.start_episode)
                }
                self.callbacks.on_training_complete(completion_info)
            raise

    def _run_training_loop(self) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Internal training loop implementation.
        
        Returns:
            Tuple of (final_rewards, final_state)
        """
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
            self.current_episode = episode  # Track for callbacks
            
            # Check for stop signal
            if self.callbacks.should_stop and self.callbacks.should_stop():
                print("Training stopped by user")
                break
            
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
            
            # Episode completion callback
            if self.callbacks.on_episode_complete:
                episode_info = {
                    'episode': episode,
                    'reward': episode_reward,
                    'max_reward': max_reward,
                    'metrics': episode_metrics.copy(),
                    'final_state': final_state,
                    'pareto_front_size': len(self.env.pareto_front) if hasattr(self.env, 'pareto_front') else 0,
                    'training_time': time.time() - self.training_start_time
                }
                self.callbacks.on_episode_complete(episode, episode_info)
            
            # Send data to UI queue if available
            self._send_ui_updates(episode, episode_metrics, episode_reward, final_state, episode_data)
            
            # Periodic progress updates
            current_time = time.time()
            if (episode % self.callbacks.progress_interval == 0 or 
                current_time - self.last_progress_time > 30):  # At least every 30 seconds
                
                if self.callbacks.on_periodic_update:
                    progress_info = {
                        'episode': episode,
                        'total_episodes': self.n_iterations,
                        'reward': episode_reward,
                        'max_reward': max_reward,
                        'training_time': current_time - self.training_start_time,
                        'episodes_completed': episode - self.start_episode + 1,
                        'pareto_front_size': len(self.env.pareto_front) if hasattr(self.env, 'pareto_front') else 0,
                        'pareto_ranges': self._get_pareto_ranges() if hasattr(self.env, 'pareto_front') else {},
                        'metrics': episode_metrics.copy()
                    }
                    
                    # Add Pareto update for plot manager
                    if episode % 20 == 0 and hasattr(self.env, 'pareto_front'):
                        pareto_update = self._generate_pareto_update(episode)
                        if pareto_update:
                            progress_info['pareto_update'] = pareto_update
                    
                    self.callbacks.on_periodic_update(episode, progress_info)
                
                self.last_progress_time = current_time
            
            # Periodic tasks (detailed summaries, saving, etc.)
            if episode % self.callbacks.summary_interval == 0 and episode != 0:
                self._perform_periodic_tasks(episode, all_materials[0] if all_materials else [], 
                                           episode_start_time, start_time, final_state, episode_reward)

        print(f"Training complete. Max reward: {max_reward}")
        return self.metrics.iloc[-1].to_dict() if len(self.metrics) > 0 else {}, max_state

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
        for rollout in range(self.n_epochs_per_update):
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
                "lr_discrete": self.agent.lr_discrete_policy,
                "lr_continuous": self.agent.lr_continuous_policy,
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
        objective_weights = episode_data['objective_weights']
        objective_names = self.env.optimise_parameters if hasattr(self.env, 'optimise_parameters') else ['reflectivity', 'thermal_noise', 'absorption']
        
        for i, obj_name in enumerate(objective_names):
            if i < len(objective_weights):
                episode_metrics[f"{obj_name}_reward_weights"] = objective_weights[i]
            else:
                episode_metrics[f"{obj_name}_reward_weights"] = 0.0

        
        # Add to metrics dataframe
        self.metrics = pd.concat([self.metrics, pd.DataFrame([episode_metrics])], ignore_index=True)

    def _perform_periodic_tasks(self, episode: int, all_materials: List, episode_start_time: float, start_time: float, final_state: np.ndarray, episode_reward: float) -> None:
        """Perform periodic tasks like plotting, saving, and logging."""
        
        # Save using unified checkpoint system or legacy system
        if self.use_unified_checkpoints:
            self._save_unified_checkpoint(episode)
        else:
            # Legacy scattered saves
            self.write_metrics_to_file()
            self._save_pareto_front_data(episode)
            best_states_path = os.path.join(self.root_dir, HPPOConstants.BEST_STATES_FILE)
            with open(best_states_path, "wb") as f:
                pickle.dump(self.best_states, f)
        
        # Create plots only if requested (or in legacy mode)
        if self.save_plots or not self.use_unified_checkpoints:
            self.make_plots()
            make_materials_plot(all_materials, self.n_layers, self.root_dir, self.env.n_materials)
        
        # Save episode state visualization only if requested
        if self.save_episode_visualizations and episode % HPPOConstants.SAVE_INTERVAL == 0:
            self._save_episode_visualization(episode, final_state, episode_reward)
        
        # Save model checkpoints periodically (always save network weights separately)
        if episode % HPPOConstants.SAVE_INTERVAL == 0:
            self._save_model_checkpoint(episode)
        
        # Print progress
        episode_time = time.time() - episode_start_time
        total_time = time.time() - start_time
        print(f"Episode {episode + 1}: Total Reward: {episode_reward:.4f}, "
              f"Episode time: {episode_time:.2f}s, Total time: {total_time:.2f}s")

    def _save_unified_checkpoint(self, episode: int, migration: bool = False) -> None:
        """
        Save complete training state using unified checkpoint system.
        
        Args:
            episode: Current episode number
            migration: Whether this is a migration from legacy format
        """
        try:
            # Prepare all data for unified saving
            trainer_data = {
                'metadata': {
                    'training_config': {
                        'n_iterations': self.n_iterations,
                        'n_layers': self.n_layers,
                        'entropy_beta_start': self.entropy_beta_start,
                        'entropy_beta_end': self.entropy_beta_end,
                        'entropy_beta_decay_length': self.entropy_beta_decay_length,
                        'entropy_beta_decay_start': self.entropy_beta_decay_start,
                        'n_epochs_per_update': self.n_epochs_per_update,
                        'use_obs': self.use_obs,
                        'scheduler_start': self.scheduler_start,
                        'scheduler_end': self.scheduler_end,
                        'weight_network_save': self.weight_network_save,
                    },
                    'environment_config': {
                        'optimise_parameters': getattr(self.env, 'optimise_parameters', []),
                        'reward_function': getattr(self.env, 'reward_function', 'unknown'),
                        'n_materials': getattr(self.env, 'n_materials', 0),
                        'max_thickness': getattr(self.env, 'max_thickness', 0),
                        'min_thickness': getattr(self.env, 'min_thickness', 0),
                    },
                    'current_episode': episode,
                    'migration_from_legacy': migration,
                },
                
                'training_data': {
                    'metrics_df': self.metrics,
                    'start_episode': self.start_episode,
                },
                
                'pareto_data': {},
                'environment_state': {},
                'best_states': getattr(self, 'best_states', [])
            }
            
            # Add Pareto front data if available
            if hasattr(self.env, 'pareto_front') and self.env.pareto_front is not None and len(self.env.pareto_front) > 0:
                # Ensure pareto_front is a numpy array with appropriate dtype
                pareto_array = np.array(self.env.pareto_front)
                if pareto_array.dtype == 'object':
                    # Try to convert to float
                    try:
                        pareto_array = pareto_array.astype(float)
                    except (ValueError, TypeError):
                        print("Warning: Could not convert pareto_front to numeric, skipping...")
                        pareto_array = None
                
                if pareto_array is not None:
                    trainer_data['pareto_data']['current_front'] = pareto_array
            
            if hasattr(self.env, 'reference_point') and self.env.reference_point is not None:
                ref_point = np.array(self.env.reference_point)
                if ref_point.dtype != 'object':
                    trainer_data['pareto_data']['reference_point'] = ref_point
                
            if hasattr(self.env, 'saved_points') and self.env.saved_points:
                saved_points = np.array(self.env.saved_points)
                if saved_points.dtype != 'object':
                    trainer_data['pareto_data']['all_points'] = saved_points
                
            if hasattr(self.env, 'saved_data') and self.env.saved_data:
                saved_data = np.array(self.env.saved_data)
                if saved_data.dtype != 'object':
                    trainer_data['pareto_data']['all_values'] = saved_data
            
            # Add environment state
            for attr in ['optimise_parameters', 'final_weight_epoch', 'reward_function', 'n_materials']:
                if hasattr(self.env, attr):
                    trainer_data['environment_state'][attr] = getattr(self.env, attr)
            
            # Save checkpoint (network weights saved separately as .pt files)
            start_time = time.time()
            self.checkpoint_manager.save_complete_checkpoint(trainer_data)
            
            save_time = time.time() - start_time
            checkpoint_info = self.checkpoint_manager.get_checkpoint_info()
            
            if episode % HPPOConstants.EPISODE_PRINT_INTERVAL == 0 or migration:
                status = "migration" if migration else "checkpoint"
                print(f"Unified {status} saved: Episode {episode}, "
                      f"{checkpoint_info['size_mb']}MB, {save_time:.2f}s")
                      
        except Exception as e:
            print(f"Warning: Failed to save unified checkpoint: {e}")
            # Fallback to legacy saves if unified fails
            if not migration:  # Don't fallback during migration
                print("Falling back to legacy save methods...")
                self._fallback_legacy_saves(episode)

    def _save_pareto_front_data(self, episode: int) -> None:
        """Save Pareto front data."""
        pareto_dir = os.path.join(self.root_dir, HPPOConstants.PARETO_FRONTS_DIR)
        os.makedirs(pareto_dir, exist_ok=True)
        
        # Save Pareto front with episode number
        pareto_file = os.path.join(pareto_dir, f"pareto_front_it{episode}.txt")
        np.savetxt(pareto_file, self.env.pareto_front, header="Reflectivity, Absorption")
        
        # Save latest Pareto front for continuation
        latest_pareto_file = os.path.join(self.root_dir, HPPOConstants.PARETO_FRONT_FILE)
        np.savetxt(latest_pareto_file, self.env.pareto_front, header="Reflectivity, Absorption")
        
        # Save all points
        all_points_file = os.path.join(self.root_dir, HPPOConstants.ALL_POINTS_FILE)
        np.savetxt(all_points_file, self.env.saved_points)
        
        all_data_file = os.path.join(self.root_dir, HPPOConstants.ALL_POINTS_DATA_FILE)
        np.savetxt(all_data_file, self.env.saved_data)

    def _save_episode_visualization(self, episode: int, state: np.ndarray, reward: float) -> None:
        """Save visualization of episode state."""
        # Only save visualizations if requested (via callbacks or legacy save_episode_visualizations)
        if not (self.callbacks.save_visualizations or self.save_episode_visualizations):
            return
            
        states_dir = os.path.join(self.root_dir, HPPOConstants.STATES_DIR)
        
        fig, ax = self.env.plot_stack(state)
        opt_value = self.env.compute_state_value(state, return_separate=True)
        ax.set_title(f"Episode {episode}: Reward: {reward:.4f}, Value: {opt_value}")
        
        episode_file = os.path.join(states_dir, f"episode_{episode}.png")
        fig.savefig(episode_file)
        plt.close(fig)

    def _save_model_checkpoint(self, episode: int) -> None:
        """Save model checkpoint to output directory."""
        print(f"Saving model checkpoint at episode {episode}")
        self.agent.save_networks(os.path.join(self.root_dir, HPPOConstants.NETWORK_WEIGHTS_DIR), episode=episode)

    def init_pareto_front(self, n_solutions: int = 5) -> None:
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
            
            if self.env.reward_function in ["area", "hypervolume"]:
                sol_vals["reflectivity"].append(reflectivity)
                sol_vals["thermal_noise"].append(thermal_noise)
                sol_vals["absorption"].append(absorption)
                sol_vals["thickness"].append(thickness)
            else:
                _, _, rewards = self.env.select_reward(reflectivity, thermal_noise, thickness, absorption, weights=None)
                
                for key in sol_vals.keys():
                    sol_vals[key].append(rewards[key])

        # Create Pareto front from objective values
        vals = np.array([sol_vals[key] for key in self.env.optimise_parameters]).T
        
        pareto_front = self.env.compute_pareto_front(vals)
        
        # Set Pareto front and reference point
        self.env.pareto_front = pareto_front
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
        self.agent.load_networks(os.path.join(self.root_dir, HPPOConstants.NETWORK_WEIGHTS_DIR))
        
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

    def _fallback_legacy_saves(self, episode: int) -> None:
        """Fallback to legacy save methods if unified checkpoint fails."""
        try:
            self.write_metrics_to_file()
            self._save_pareto_front_data(episode)
            
            best_states_path = os.path.join(self.root_dir, HPPOConstants.BEST_STATES_FILE)
            with open(best_states_path, "wb") as f:
                pickle.dump(self.best_states, f)
                
            print("Fallback legacy saves completed")
        except Exception as e:
            print(f"Warning: Fallback legacy saves also failed: {e}")
    
    def _send_ui_updates(self, episode: int, episode_metrics: Dict, episode_reward: float, 
                        final_state: np.ndarray, episode_data: Dict) -> None:
        """Send training updates to UI queue if available."""
        if not self.callbacks.ui_queue:
            return
        
        try:
            # Send training data update
            training_update = {
                'type': 'training_data',
                'episode': episode,
                'reward': episode_reward,
                'metrics': episode_metrics.copy()
            }
            self.callbacks.ui_queue.put(training_update, block=False)
            
            # Send Pareto front data every 20 episodes
            if episode % 20 == 0 and hasattr(self.env, 'pareto_front'):
                pareto_update = self._generate_pareto_update(episode)
                if pareto_update:
                    self.callbacks.ui_queue.put(pareto_update, block=False)
                    
        except queue.Full:
            pass  # Skip if queue is full
    
    def _generate_pareto_update(self, episode: int) -> Optional[Dict]:
        """Generate Pareto front update for UI."""
        if not hasattr(self, 'best_states') or not self.best_states:
            return None
            
        best_points = []
        best_state_data = []
        best_vals_list = []
        
        # Extract data from best_states
        for tot_reward, epoch, state, rewards, vals in self.best_states:
            if hasattr(self.env, 'optimise_parameters'):
                # Build point based on actual optimisation parameters
                point = []
                val_point = []
                
                for param in self.env.optimise_parameters:
                    if param in vals:
                        val = vals[param]
                        val_point.append(val)
                        
                        # Apply transformations for plotting/minimization
                        if param == 'reflectivity':
                            point.append(1 - val)  # Convert to 1-R for minimization
                        else:
                            point.append(val)  # Other objectives use as-is
                    else:
                        point.append(0.0)
                        val_point.append(0.0)
                
                best_points.append(point)
                best_vals_list.append(val_point)
            else:
                # Fallback to 2-objective case
                reflectivity = vals.get('reflectivity', 0)
                absorption = vals.get('absorption', 0)
                ref_loss = 1 - reflectivity
                best_points.append([ref_loss, absorption])
                best_vals_list.append([reflectivity, absorption])
            
            best_state_data.append(state)
        
        # Recompute Pareto front
        recomputed_pareto_front = []
        pareto_indices = []
        
        if len(best_vals_list) > 0:
            vals_array = np.array(best_vals_list)
            
            # Create minimization objectives array
            if hasattr(self.env, 'optimise_parameters') and len(self.env.optimise_parameters) > 1:
                minimization_objectives = np.zeros_like(vals_array)
                
                for i, param in enumerate(self.env.optimise_parameters):
                    if param == 'reflectivity':
                        minimization_objectives[:, i] = 1 - vals_array[:, i]  # Minimize 1-R
                    else:
                        minimization_objectives[:, i] = vals_array[:, i]  # Minimize as-is
            else:
                # Fallback to 2-objective case
                minimization_objectives = np.column_stack([1 - vals_array[:, 0], vals_array[:, 1]])
            
            nds = NonDominatedSorting()
            fronts = nds.do(minimization_objectives)
            
            if len(fronts) > 0 and len(fronts[0]) > 0:
                pareto_indices = fronts[0]
                recomputed_pareto_front = [best_points[i] for i in pareto_indices]
        
        return {
            'type': 'pareto_data',
            'episode': episode,
            'pareto_front': np.array(recomputed_pareto_front) if recomputed_pareto_front else np.array([]),
            'best_points': best_points,
            'best_state_data': best_state_data,
            'pareto_indices': pareto_indices,
            'pareto_states': [best_state_data[i] for i in pareto_indices],
        }
    
    def _get_pareto_ranges(self) -> Dict[str, tuple]:
        """Get ranges of objectives in current Pareto front."""
        if not hasattr(self.env, 'pareto_front') or len(self.env.pareto_front) == 0:
            return {}
        
        ranges = {}
        pf = self.env.pareto_front
        for i, param in enumerate(self.env.optimise_parameters):
            obj_values = [pt[i] for pt in pf]
            ranges[param] = (min(obj_values), max(obj_values))
        return ranges

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'checkpoint_type': 'unified' if self.use_unified_checkpoints else 'legacy',
            'current_episode': getattr(self, 'start_episode', 0),
            'target_episodes': self.n_iterations,
            'completion_percent': round((getattr(self, 'start_episode', 0) / self.n_iterations) * 100, 2) if self.n_iterations > 0 else 0,
            'metrics_count': len(self.metrics) if hasattr(self, 'metrics') else 0,
            'best_states_count': len(self.best_states) if hasattr(self, 'best_states') else 0,
        }
        
        # Add unified checkpoint info if available
        if self.use_unified_checkpoints and hasattr(self, 'checkpoint_manager'):
            checkpoint_info = self.checkpoint_manager.get_checkpoint_info()
            summary['checkpoint_info'] = checkpoint_info
            
        # Add Pareto front info
        if hasattr(self.env, 'pareto_front'):
            summary['pareto_front_size'] = len(self.env.pareto_front) if self.env.pareto_front is not None else 0
        if hasattr(self.env, 'saved_points'):
            summary['total_points_explored'] = len(self.env.saved_points)
        if hasattr(self.env, 'optimise_parameters'):
            summary['objectives_count'] = len(self.env.optimise_parameters)
            
        return summary

    def backup_checkpoint(self) -> str:
        """Create a backup of the current training state."""
        if self.use_unified_checkpoints and hasattr(self, 'checkpoint_manager'):
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(self.root_dir, f"training_backup_{timestamp}.h5")
            
            import shutil
            if os.path.exists(self.checkpoint_manager.checkpoint_path):
                shutil.copy2(self.checkpoint_manager.checkpoint_path, backup_path)
                print(f"Unified checkpoint backed up to: {backup_path}")
                return backup_path
            else:
                print("No unified checkpoint found to backup")
                return ""
        else:
            # For legacy format, backup the whole directory
            import shutil
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.join(os.path.dirname(self.root_dir), f"training_backup_{timestamp}")
            shutil.copytree(self.root_dir, backup_dir, ignore=shutil.ignore_patterns('backup_*'))
            print(f"Legacy training directory backed up to: {backup_dir}")
            return backup_dir

    def load_historical_data_to_plot_manager(self, plot_manager) -> bool:
        """
        Load historical training data into plot manager.
        
        This is a shared utility function used by both CLI and UI to load
        historical training data and pareto states from checkpoints.
        
        Args:
            plot_manager: TrainingPlotManager instance to load data into
            
        Returns:
            bool: True if data was successfully loaded, False otherwise
        """
        if not plot_manager:
            return False
            
        try:
            if self.use_unified_checkpoints and hasattr(self, 'checkpoint_manager'):
                # Load from unified checkpoint
                if os.path.exists(self.checkpoint_manager.checkpoint_path):
                    checkpoint_data = self.checkpoint_manager.load_complete_checkpoint()
                    
                    if not checkpoint_data:
                        return False
                        
                    # Load training metrics
                    training_data = checkpoint_data.get('training_data', {})
                    
                    # Check for metrics_df (the correct key from checkpoint manager)
                    metrics_df = None
                    if 'metrics_df' in training_data:
                        metrics_df = training_data['metrics_df']
                    elif 'metrics' in training_data:
                        metrics_df = training_data['metrics']
                    
                    if metrics_df is not None:
                        
                        # Check if it's a pandas DataFrame
                        if hasattr(metrics_df, 'iterrows'):
                            # Convert to list of episode data for plot manager
                            for _, row in metrics_df.iterrows():
                                episode_data = {
                                    'episode': int(row.get('episode', 0)),
                                    'reward': float(row.get('reward', 0.0)),
                                    'metrics': {key: float(val) for key, val in row.items() 
                                              if key not in ['episode', 'reward'] and pd.notna(val)}
                                }
                                plot_manager.add_training_data(episode_data)
                        else:
                            print(f"Debug: metrics_df is not a DataFrame, it's: {type(metrics_df)}")
                            # Try to convert to DataFrame if it's a numpy array or similar
                            if hasattr(metrics_df, '__len__') and len(metrics_df) > 0:
                                print(f"Debug: metrics_df has {len(metrics_df)} entries")
                    else:
                        print("Debug: No 'metrics' or 'metrics_df' key found in training_data")
                    
                    # Load pareto data with states
                    pareto_data = checkpoint_data.get('pareto_data', {})
                    best_states = checkpoint_data.get('best_states', [])
                    
                    if 'fronts_history' in pareto_data and best_states:
                        fronts_history = pareto_data['fronts_history']
                        
                        # Process each episode's front data
                        for episode_num, front_data in fronts_history.items():
                            if isinstance(episode_num, str) and episode_num.startswith('episode_'):
                                episode = int(episode_num.split('_')[1])
                            else:
                                episode = int(episode_num)
                            
                            # Generate pareto states for this episode
                            pareto_states = self._generate_pareto_states_from_checkpoint(
                                front_data, best_states, episode)
                            
                            if pareto_states:
                                pareto_update = {
                                    'episode': episode,
                                    'pareto_front': front_data,
                                    'pareto_states': pareto_states,
                                    'best_state_data': pareto_states,
                                    'pareto_indices': list(range(len(pareto_states)))
                                }
                                plot_manager.add_pareto_data(pareto_update)
                    
                    print(f"Loaded historical data from unified checkpoint: {len(plot_manager.training_data)} episodes")
                    return True
                    
            else:
                # Load from legacy format
                print("Loading historical data from legacy format...")
                return self._load_historical_data_legacy(plot_manager)
                
        except Exception as e:
            print(f"Warning: Failed to load historical data: {e}")
            return False
        
        return False
    
    def _generate_pareto_states_from_checkpoint(self, front_data, best_states, episode):
        """Generate pareto states from checkpoint data."""
        if not front_data.size or not best_states:
            return []
            
        pareto_states = []
        
        # Try to match front points with best states based on proximity
        for front_point in front_data:
            closest_state = None
            min_distance = float('inf')
            
            for state_data in best_states:
                if len(state_data) >= 5:  # tot_reward, epoch, state, rewards, vals
                    _, _, state, rewards, vals = state_data[:5]
                    
                    # Calculate expected front point from this state
                    if hasattr(self.env, 'optimise_parameters'):
                        expected_point = []
                        for param in self.env.optimise_parameters:
                            if param in vals:
                                val = vals[param]
                                if param == 'reflectivity':
                                    expected_point.append(1 - val)  # Convert to 1-R
                                else:
                                    expected_point.append(val)
                        
                        if len(expected_point) == len(front_point):
                            distance = np.linalg.norm(np.array(expected_point) - np.array(front_point))
                            if distance < min_distance:
                                min_distance = distance
                                closest_state = state
            
            if closest_state is not None:
                pareto_states.append(closest_state)
        
        return pareto_states
    
    def _load_historical_data_legacy(self, plot_manager) -> bool:
        """Load historical data from legacy format."""
        try:
            # Load metrics from legacy CSV/pickle files
            metrics_file = os.path.join(self.root_dir, HPPOConstants.METRICS_FILE)
            if os.path.exists(metrics_file):
                if metrics_file.endswith('.csv'):
                    metrics_df = pd.read_csv(metrics_file)
                else:
                    with open(metrics_file, 'rb') as f:
                        metrics_df = pickle.load(f)
                
                # Convert to plot manager format
                for _, row in metrics_df.iterrows():
                    episode_data = {
                        'episode': int(row.get('episode', 0)),
                        'reward': float(row.get('reward', 0.0)),
                        'metrics': {key: float(val) for key, val in row.items() 
                                  if key not in ['episode', 'reward'] and pd.notna(val)}
                    }
                    plot_manager.add_training_data(episode_data)
                
                print(f"Loaded historical data from legacy format: {len(metrics_df)} episodes")
                return True
            
        except Exception as e:
            print(f"Warning: Failed to load legacy historical data: {e}")
        
        return False

# Migration utility function
def migrate_legacy_training_data(legacy_root_dir: str, new_root_dir: str = None) -> str:
    """
    Migrate from legacy scattered file format to unified checkpoint.
    
    Args:
        legacy_root_dir: Directory containing old format files
        new_root_dir: Optional new directory for unified checkpoint
        
    Returns:
        Path to the created unified checkpoint
    """
    if new_root_dir is None:
        new_root_dir = legacy_root_dir
    
    print(f"Migrating training data from {legacy_root_dir} to unified format...")
    
    # Create checkpoint manager
    checkpoint_manager = TrainingCheckpointManager(
        root_dir=new_root_dir,
        checkpoint_name="training_checkpoint_migrated.h5"
    )
    
    # Load legacy data
    legacy_data = {}
    
    # Load metrics
    metrics_file = os.path.join(legacy_root_dir, HPPOConstants.TRAINING_METRICS_FILE)
    if os.path.exists(metrics_file):
        legacy_data['metrics_df'] = pd.read_csv(metrics_file)
        current_episode = legacy_data['metrics_df']['episode'].max() if not legacy_data['metrics_df'].empty else 0
    else:
        legacy_data['metrics_df'] = pd.DataFrame()
        current_episode = 0
    
    # Load best states
    best_states_file = os.path.join(legacy_root_dir, HPPOConstants.BEST_STATES_FILE)
    if os.path.exists(best_states_file):
        with open(best_states_file, 'rb') as f:
            legacy_data['best_states'] = pickle.load(f)
    else:
        legacy_data['best_states'] = []
    
    # Load Pareto data
    pareto_data = {}
    pareto_file = os.path.join(legacy_root_dir, HPPOConstants.PARETO_FRONT_FILE)
    if os.path.exists(pareto_file):
        pareto_data['current_front'] = np.loadtxt(pareto_file)
    
    all_points_file = os.path.join(legacy_root_dir, HPPOConstants.ALL_POINTS_FILE)
    if os.path.exists(all_points_file):
        pareto_data['all_points'] = np.loadtxt(all_points_file)
    
    all_data_file = os.path.join(legacy_root_dir, HPPOConstants.ALL_POINTS_DATA_FILE)
    if os.path.exists(all_data_file):
        pareto_data['all_values'] = np.loadtxt(all_data_file)
    
    # Prepare checkpoint data
    trainer_data = {
        'metadata': {
            'training_config': {'migrated_from_legacy': True},
            'environment_config': {},
            'current_episode': current_episode,
            'migration_source': legacy_root_dir,
            'migration_from_legacy': True,
        },
        'training_data': legacy_data,
        'pareto_data': pareto_data,
        'environment_state': {},
        'best_states': legacy_data.get('best_states', []),
    }
    
    # Save unified checkpoint
    checkpoint_manager.save_complete_checkpoint(trainer_data)
    
    checkpoint_info = checkpoint_manager.get_checkpoint_info()
    print(f"Migration completed successfully!")
    print(f"Created unified checkpoint: {checkpoint_manager.checkpoint_path}")
    print(f"Checkpoint size: {checkpoint_info['size_mb']}MB")
    print(f"Contains {current_episode} episodes of training data")
    
    return str(checkpoint_manager.checkpoint_path)


# Convenience functions for creating callbacks

def create_cli_callbacks(verbose: bool = True, plot_manager=None) -> TrainingCallbacks:
    """Create callbacks optimized for CLI usage."""
    
    def cli_progress_update(episode: int, info: Dict[str, Any]):
        if verbose:
            elapsed = info['training_time']
            eps_per_sec = info['episodes_completed'] / elapsed if elapsed > 0 else 0
            eta_seconds = (info['total_episodes'] - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
            
            pareto_info = ""
            if info['pareto_front_size'] > 0:
                pareto_info = f", Pareto: {info['pareto_front_size']} pts"
            
            print(f"Episode {episode:4d}/{info['total_episodes']} | "
                  f"Reward: {info['reward']:8.3f} | "
                  f"Max: {info['max_reward']:8.3f} | "
                  f"Speed: {eps_per_sec:.2f} eps/s | "
                  f"ETA: {eta_str}{pareto_info}")
        
        # Update plots if plot manager is available
        if plot_manager:
            # Add training data
            episode_data = {
                'episode': episode,
                'reward': info['reward'],
                'metrics': info.get('metrics', {})
            }
            plot_manager.add_training_data(episode_data)
            
            # Add Pareto data if available
            if 'pareto_update' in info:
                plot_manager.add_pareto_data(info['pareto_update'])
            
            # Update plots periodically
            if plot_manager.should_update_plots(episode):
                plot_manager.update_all_plots(episode)
    
    def cli_summary_update(episode: int, info: Dict[str, Any]):
        if verbose and episode % 100 == 0:
            print(f"\n--- Episode {episode} Summary ---")
            print(f"Current reward: {info['reward']:.6f}")
            print(f"Max reward so far: {info['max_reward']:.6f}")
            print(f"Training time: {info['training_time']/3600:.2f} hours")
            
            if info['pareto_front_size'] > 0:
                print(f"Pareto front size: {info['pareto_front_size']}")
                #for param, (min_val, max_val) in info['pareto_ranges'].items():
                #    print(f"  {param}: [{min_val:.4f}, {max_val:.4f}]")
            print("-" * 30 + "\n")
    
    return TrainingCallbacks(
        on_periodic_update=cli_progress_update,
        on_episode_complete=cli_summary_update,
        progress_interval=10,
        summary_interval=100
    )


def create_ui_callbacks(ui_queue, ui_stop_check) -> TrainingCallbacks:
    """Create callbacks optimized for UI usage."""
    
    def ui_episode_complete(episode: int, info: Dict[str, Any]):
        try:
            training_update = {
                'type': 'training_data',
                'episode': episode,
                'reward': info['reward'],
                'metrics': info['metrics']
            }
            ui_queue.put(training_update, block=False)

        except:
            pass  # Don't let queue issues stop training
    
    return TrainingCallbacks(
        on_episode_complete=ui_episode_complete,
        should_stop=ui_stop_check,
        save_plots=False,  # UI handles plots separately
        save_visualizations=False,
        progress_interval=1,  # More frequent for UI
        ui_queue=ui_queue
    )