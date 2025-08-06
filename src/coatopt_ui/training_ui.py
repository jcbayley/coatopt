"""
Real-time Training UI for PC-HPPO-OML Coating Optimization

A simple GUI for loading configuration files and monitoring training progress
with live plots of rewards and Pareto front evolution.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import sys
import numpy as np

# Set matplotlib backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from coatopt.config import read_config, read_materials
from coatopt.config.structured_config import CoatingOptimizationConfig
from coatopt.factories import setup_optimization_pipeline
from coatopt.algorithms.hppo_trainer import HPPOTrainer

import traceback
import logging


class TrainingMonitorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PC-HPPO-OML Training Monitor")
        self.root.geometry("1400x900")
        
        # Training components
        self.trainer = None
        self.env = None
        self.agent = None
        self.config = None
        self.materials = None
        
        # Threading components
        self.training_thread = None
        self.training_queue = queue.Queue()
        self.is_training = False
        
        # Data storage
        self.training_data = []
        self.pareto_data = []
        self.pareto_states = []  # Store coating states for each Pareto point
        self.saved_states = []   # Store all sampled coating states
        self.historical_pareto_data = {}  # Store Pareto data by episode for slider
        
        # Event handler tracking
        self.click_event_connection = None  # Store event connection to prevent multiple handlers
        
        # Plot update throttling
        self.last_reward_plot_update = 0
        self.last_pareto_plot_update = 0
        self.plot_update_interval = 5  # Update plots every N episodes
        
        self.setup_ui()
        self.setup_plots()
        
        # Start monitoring for training updates (less frequent to improve responsiveness)
        self.root.after(250, self.check_training_updates)
    
    def setup_ui(self):
        """Setup the user interface components."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Training Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Configuration file selection
        config_frame = ttk.Frame(control_frame)
        config_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(config_frame, text="Configuration File:").pack(side=tk.LEFT)
        self.config_var = tk.StringVar()
        self.config_entry = ttk.Entry(config_frame, textvariable=self.config_var, width=60)
        self.config_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        self.browse_button = ttk.Button(config_frame, text="Browse", command=self.browse_config)
        self.browse_button.pack(side=tk.RIGHT)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.load_button = ttk.Button(button_frame, text="Load Configuration", command=self.load_configuration)
        self.load_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Retrain checkbox
        self.retrain_var = tk.BooleanVar(value=False)  # Unchecked by default
        self.retrain_checkbox = ttk.Checkbutton(button_frame, text="Retrain (ignore existing data)", variable=self.retrain_var)
        self.retrain_checkbox.pack(side=tk.LEFT, padx=(10, 5))
        
        self.start_button = ttk.Button(button_frame, text="Start Training", command=self.start_training, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready - Please load a configuration file")
        self.status_label = ttk.Label(button_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(5, 0))
        
        # Plot frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for plot tabs
        self.notebook = ttk.Notebook(plot_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
    
    def setup_plots(self):
        """Setup matplotlib plots in the UI."""
        # Rewards plot tab
        self.rewards_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rewards_frame, text="Training Rewards")
        
        self.rewards_fig = Figure(figsize=(12, 10), dpi=100)  # Larger figure for subplots
        self.rewards_canvas = FigureCanvasTkAgg(self.rewards_fig, self.rewards_frame)
        self.rewards_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Physical values plot tab
        self.values_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.values_frame, text="Physical Values")
        
        self.values_fig = Figure(figsize=(12, 8), dpi=100)
        self.values_canvas = FigureCanvasTkAgg(self.values_fig, self.values_frame)
        self.values_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pareto front plot tab
        self.pareto_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pareto_frame, text="Pareto Front Evolution")
        
        self.pareto_fig = Figure(figsize=(10, 8), dpi=100)
        self.pareto_canvas = FigureCanvasTkAgg(self.pareto_fig, self.pareto_frame)
        self.pareto_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Epoch slider for Pareto plot
        self.pareto_controls_frame = ttk.Frame(self.pareto_frame)
        self.pareto_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.pareto_controls_frame, text="Training Epoch:").pack(side=tk.LEFT)
        
        self.epoch_var = tk.IntVar(value=0)
        self.epoch_slider = tk.Scale(self.pareto_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   variable=self.epoch_var, command=self.on_epoch_change, 
                                   length=400, resolution=20)
        self.epoch_slider.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.epoch_label = ttk.Label(self.pareto_controls_frame, text="Episode: 0")
        self.epoch_label.pack(side=tk.RIGHT)
        
        # Initialize empty plots
        self.init_plots()
    
    def on_epoch_change(self, value):
        """Handle epoch slider change."""
        try:
            episode = int(value)
            self.epoch_label.config(text=f"Episode: {episode}")
            
            # Update Pareto plot with data from selected episode
            if episode in self.historical_pareto_data:
                self.update_pareto_plot_for_episode(episode)
        except Exception as e:
            print(f"Error in epoch change: {e}")
    
    def init_plots(self):
        """Initialize empty plots."""
        # Rewards plot with multiple subplots (no physical values)
        self.rewards_fig.clear()
        self.rewards_axes = self.rewards_fig.subplots(3, 2)  # Reduced to 6 subplots
        self.rewards_axes = self.rewards_axes.flatten()
        
        # Initialize subplot titles (excluding physical values)
        rewards_subplot_titles = [
            "Total Reward", "Individual Rewards", "Entropy Weight (β)", 
            "Learning Rates", "Objective Weights", "Training Losses"
        ]
        
        for i, ax in enumerate(self.rewards_axes):
            ax.set_title(rewards_subplot_titles[i] if i < len(rewards_subplot_titles) else f"Plot {i+1}")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Episode")
        
        self.rewards_fig.tight_layout()
        self.rewards_canvas.draw_idle()
        
        # Physical values plot with separate subplots for each metric
        self.values_fig.clear()
        self.values_axes = self.values_fig.subplots(2, 2)  # 4 subplots for physical values
        self.values_axes = self.values_axes.flatten()
        
        values_subplot_titles = [
            "Reflectivity", "Thermal Noise", "Absorption", "Thickness"
        ]
        
        for i, ax in enumerate(self.values_axes):
            ax.set_title(values_subplot_titles[i] if i < len(values_subplot_titles) else f"Value {i+1}")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Episode")
        
        self.values_fig.tight_layout()
        self.values_canvas.draw_idle()
        
        # Pareto front plot with two subplots
        self.pareto_fig.clear()
        self.pareto_axes = self.pareto_fig.subplots(1, 2)
        
        # Left subplot: Pareto front
        self.pareto_ax = self.pareto_axes[0]
        self.pareto_ax.set_title("Pareto Front: Reflectivity vs Absorption")
        self.pareto_ax.set_xlabel("1 - Reflectivity")
        self.pareto_ax.set_ylabel("Absorption [ppm]")
        self.pareto_ax.set_xscale("log")
        self.pareto_ax.set_yscale("log")
        self.pareto_ax.grid(True, alpha=0.3)
        
        # Right subplot: Coating stack visualization
        self.coating_ax = self.pareto_axes[1]
        self.coating_ax.set_title("Coating Stack (Click point to view)")
        self.coating_ax.set_xlabel("Layer")
        self.coating_ax.set_ylabel("Thickness")
        self.coating_ax.grid(True, alpha=0.3)
        
        self.pareto_fig.tight_layout()
        self.pareto_canvas.draw_idle()
    
    def browse_config(self):
        """Browse for configuration file."""
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )
        if filename:
            self.config_var.set(filename)
    
    def load_configuration(self):
        """Load the configuration file and setup components."""
        config_path = self.config_var.get()
        if not config_path or not os.path.exists(config_path):
            messagebox.showerror("Error", "Please select a valid configuration file.")
            return
        
        try:
            self.status_var.set("Loading configuration...")
            # self.root.update()  # Remove direct GUI update from non-main thread
            
            # Load configuration and materials
            raw_config = read_config(os.path.abspath(config_path))
            self.config = CoatingOptimizationConfig.from_config_parser(raw_config)
            self.materials = read_materials(self.config.general.materials_file)
            
            # Setup optimization components
            print(f"Optimization parameters: {self.config.data.optimise_parameters}")
            print(f"Number of objectives: {len(self.config.data.optimise_parameters)}")
            
            # Determine whether to continue training based on checkbox
            continue_training = not self.retrain_var.get()  # If retrain is checked, don't continue
            
            self.env, self.agent, self.trainer = setup_optimization_pipeline(
                self.config, self.materials, continue_training=continue_training
            )

            print(f"Agent num_objectives: {self.agent.num_objectives}")
            print(f"Environment optimise_parameters: {self.env.optimise_parameters}")
            
            # Load historical training data if not retraining
            if continue_training:
                self.load_historical_data()
            
            self.status_var.set(f"Configuration loaded successfully. {len(self.materials)} materials loaded.")
            self.start_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
            self.status_var.set("Failed to load configuration.")
            logging.CRITICAL(f"Error loading configuration: {e}, traceback: {traceback.format_exc()}")
    
    def load_historical_data(self):
        """Load historical training data for the epoch slider."""
        try:
            if hasattr(self.trainer, 'best_states') and self.trainer.best_states:
                print(f"Loading {len(self.trainer.best_states)} historical best states...")
                
                # Process best_states to generate historical Pareto data
                accumulated_states = []
                for i, best_state_entry in enumerate(self.trainer.best_states):
                    tot_reward, epoch, state, rewards, vals = best_state_entry
                    accumulated_states.append(best_state_entry)
                    
                    # Generate Pareto data for every 20 episodes or final episode
                    if (i + 1) % 20 == 0 or i == len(self.trainer.best_states) - 1:
                        episode = (i + 1) * 20  # Approximate episode number
                        pareto_data = self._generate_pareto_data_from_states(accumulated_states, episode)
                        self.historical_pareto_data[episode] = pareto_data
                
                # Update slider range
                if self.historical_pareto_data:
                    max_episode = max(self.historical_pareto_data.keys())
                    self.epoch_slider.config(to=max_episode)
                    self.epoch_var.set(max_episode)  # Start at the latest episode
                    self.epoch_label.config(text=f"Episode: {max_episode}")
                    
                    print(f"Loaded historical data for {len(self.historical_pareto_data)} time points")
                    # Update plot with the latest data
                    self.update_pareto_plot_for_episode(max_episode)
                
        except Exception as e:
            print(f"Error loading historical data: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_pareto_data_from_states(self, best_states, episode):
        """Generate Pareto data from a list of best states."""
        best_points = []
        best_state_data = []
        best_vals_list = []
        
        for tot_reward, epoch, state, rewards, vals in best_states:
            # Extract relevant metrics for Pareto front (use vals, not rewards)
            reflectivity = vals.get('reflectivity', 0)
            absorption = vals.get('absorption', 0)
            # Convert reflectivity to 1-R for plotting
            ref_loss = 1 - reflectivity
            best_points.append([ref_loss, absorption])
            best_state_data.append(state)
            best_vals_list.append([reflectivity, absorption])  # Store original vals for Pareto computation
        
        # Recompute Pareto front from best points using vals
        recomputed_pareto_front = []
        pareto_indices = []
        if len(best_vals_list) > 0:
            # Use non-dominated sorting on the actual values
            vals_array = np.array(best_vals_list)
            # For minimization: we want to minimize (1-reflectivity) and absorption
            # So we need to flip reflectivity to make it a minimization problem
            minimization_objectives = np.column_stack([1 - vals_array[:, 0], vals_array[:, 1]])
            
            nds = NonDominatedSorting()
            fronts = nds.do(minimization_objectives)
            
            if len(fronts) > 0 and len(fronts[0]) > 0:
                pareto_indices = fronts[0]
                # Convert back to plotting format (1-R, absorption)
                recomputed_pareto_front = [best_points[i] for i in pareto_indices]
        
        return {
            'episode': episode,
            'pareto_front': np.array(recomputed_pareto_front) if recomputed_pareto_front else np.array([]),
            'best_points': best_points,
            'best_state_data': best_state_data,
            'pareto_indices': pareto_indices,
            'pareto_states': [best_state_data[i] for i in pareto_indices] ,
        }
    
    def update_pareto_plot_for_episode(self, episode):
        """Update Pareto plot with data from a specific episode."""
        if episode not in self.historical_pareto_data:
            return
        
        # Temporarily store current pareto data and states
        original_pareto_data = self.pareto_data.copy()
        original_pareto_states = self.pareto_states.copy()
        original_saved_states = self.saved_states.copy()
        
        try:
            # Set up data for the selected episode
            episode_data = self.historical_pareto_data[episode]
            self.pareto_data = [episode_data]  # Temporarily set to single episode data
            self.pareto_states = episode_data.get('pareto_states', [])
            self.saved_states = episode_data.get('best_state_data', [])
            
            # Update the plot
            self.update_pareto_plot()
            
        finally:
            # Restore original data
            self.pareto_data = original_pareto_data
            self.pareto_states = original_pareto_states
            self.saved_states = original_saved_states
    
    def start_training(self):
        """Start the training process in a separate thread."""
        if self.trainer is None:
            messagebox.showerror("Error", "Please load a configuration first.")
            return
        
        self.is_training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.load_button.config(state=tk.DISABLED)
        self.progress.start()

        self.continue_training = not self.retrain_var.get() 
        
        # Start training thread
        self.training_thread = threading.Thread(target=self.training_worker, daemon=True)
        self.training_thread.start()
        
        self.status_var.set("Training in progress...")
    
    def stop_training(self):
        """Stop the training process."""
        self.is_training = False
        
        # Terminate the training thread if it exists
        if self.training_thread and self.training_thread.is_alive():
            # The training loop checks self.is_training, so it should stop naturally
            self.training_thread.join(timeout=2.0)  # Wait up to 2 seconds for clean shutdown
        
        # Reset UI state
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.load_button.config(state=tk.NORMAL)
        self.progress.stop()
        self.status_var.set("Training stopped.")
    
    def training_worker(self):
        """Worker thread for training process with monitoring."""
        try:
            # Create a custom trainer that reports progress
            class MonitoredTrainer(HPPOTrainer):
                def __init__(self, *args, ui_queue=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.ui_queue = ui_queue
                    self.episode_count = 0
                    self.ui_stop_signal = None  # Will be set from outside
                
                def make_plots(self):
                    """Override to disable plotting in training thread."""
                    pass  # Disable automatic plotting to prevent threading issues
                
                def _save_episode_visualization(self, episode, state, reward):
                    """Override to disable episode visualization in training thread."""
                    pass  # Disable visualization to prevent threading issues
                
                def train(self):
                    """Override train method to check for stop signal."""
                    print(f"Starting training from episode {self.start_episode} to {self.n_iterations}")
                    
                    # Initialize tracking variables
                    all_means, all_stds, all_materials = [], [], []
                    max_reward = self._get_initial_max_reward()
                    max_state = None

                    # Create states directory
                    import os
                    states_dir = os.path.join(self.root_dir, "states")
                    os.makedirs(states_dir, exist_ok=True)

                    import time
                    start_time = time.time()
                    
                    # Main training loop with stop signal check
                    for episode in range(self.start_episode, self.n_iterations):
                        # Check for stop signal
                        if self.ui_stop_signal and self.ui_stop_signal():
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
                        
                        # Periodic tasks (plotting, saving, etc.)
                        if episode % 20 == 0 and episode != 0:  # HPPOConstants.EPISODE_PRINT_INTERVAL
                            self._perform_periodic_tasks(episode, all_materials[0] if all_materials else [], episode_start_time, start_time, final_state, episode_reward)

                    print(f"Training complete. Max reward: {max_reward}")
                    return self.metrics.iloc[-1].to_dict() if len(self.metrics) > 0 else {}, max_state
                
                def _record_episode_metrics(self, episode, episode_metrics, episode_reward, final_state, episode_data):
                    super()._record_episode_metrics(episode, episode_metrics, episode_reward, final_state, episode_data)
                    
                    # Send data to UI queue
                    if self.ui_queue:
                        try:
                            # Prepare training data
                            training_update = {
                                'type': 'training_data',
                                'episode': episode,
                                'reward': episode_reward,
                                'metrics': episode_metrics.copy()
                            }
                            self.ui_queue.put(training_update, block=False)
                            
                            # Send Pareto front data every 20 episodes (more frequent updates)
                            if episode % 20 == 0 and hasattr(self.env, 'pareto_front'):
                                # Get states corresponding to Pareto front points
                                pareto_states = []
                                if hasattr(self.env, 'pareto_states') and self.env.pareto_states is not None:
                                    pareto_states = self.env.pareto_states.copy()
                                elif hasattr(self.env, 'saved_states') and self.env.saved_states is not None:
                                    # If pareto_states not available, try to get from saved_states
                                    pareto_states = self.env.saved_states.copy()
                                
                                # Get best_states from the trainer, which contains rewards and states
                                best_states = []
                                if hasattr(self, 'best_states') and self.best_states is not None:
                                    best_states = self.best_states.copy()
                                
                                # Extract points and states for Pareto analysis
                                best_points = []
                                best_state_data = []
                                best_vals_list = []
                
                                for tot_reward, epoch, state, rewards, vals in best_states:
                                    # Extract relevant metrics for Pareto front (use vals, not rewards)
                                    reflectivity = vals.get('reflectivity', 0)
                                    absorption = vals.get('absorption', 0)
                                    # Convert reflectivity to 1-R for plotting
                                    ref_loss = 1 - reflectivity
                                    best_points.append([ref_loss, absorption])
                                    best_state_data.append(state)
                                    best_vals_list.append([reflectivity, absorption])  # Store original vals for Pareto computation
                                
                                # Recompute Pareto front from best points using vals
                                recomputed_pareto_front = []
                                pareto_indices = []
                                if len(best_vals_list) > 0:
                                    # Use non-dominated sorting on the actual values
                                    vals_array = np.array(best_vals_list)
                                    # For minimization: we want to minimize (1-reflectivity) and absorption
                                    # So we need to flip reflectivity to make it a minimization problem
                                    minimization_objectives = np.column_stack([1 - vals_array[:, 0], vals_array[:, 1]])
                                    
                                    nds = NonDominatedSorting()
                                    fronts = nds.do(minimization_objectives)
                                    
                                    if len(fronts) > 0 and len(fronts[0]) > 0:
                                        pareto_indices = fronts[0]
                                        # Convert back to plotting format (1-R, absorption)
                                        recomputed_pareto_front = [best_points[i] for i in pareto_indices]
                                
                                pareto_update = {
                                    'type': 'pareto_data',
                                    'episode': episode,
                                    'pareto_front': np.array(recomputed_pareto_front) if recomputed_pareto_front else np.array([]),
                                    'best_points': best_points,  # All best points for background plotting
                                    'best_state_data': best_state_data,  # States corresponding to best points
                                    'pareto_indices': pareto_indices,  # Indices of Pareto-optimal points
                                    'pareto_states': [best_state_data[i] for i in pareto_indices],  # States for Pareto points
                                }
                                self.ui_queue.put(pareto_update, block=False)
                        except queue.Full:
                            pass  # Skip if queue is full
            
            # Replace trainer with monitored version
            monitored_trainer = MonitoredTrainer(
                self.agent, self.env,
                n_iterations=self.config.training.n_iterations,
                n_layers=self.config.data.n_layers,
                root_dir=self.config.general.root_dir,
                entropy_beta_start=self.config.training.entropy_beta_start,
                entropy_beta_end=self.config.training.entropy_beta_end,
                entropy_beta_decay_length=self.config.training.entropy_beta_decay_length,
                entropy_beta_decay_start=self.config.training.entropy_beta_decay_start,
                n_epochs_per_update=self.config.training.n_epochs_per_update,
                use_obs=self.config.data.use_observation,
                scheduler_start=self.config.training.scheduler_start,
                scheduler_end=self.config.training.scheduler_end,
                weight_network_save=self.config.training.weight_network_save,
                ui_queue=self.training_queue,
                continue_training=self.continue_training
            )
            
            # Initialize Pareto front for monitored trainer
            monitored_trainer.init_pareto_front(n_solutions=1000)
            
            # Run training with periodic checks for stop signal
            try:
                # Add stop signal to trainer
                monitored_trainer.ui_stop_signal = lambda: not self.is_training
                final_metrics, final_state = monitored_trainer.train()
            except Exception as e:
                self.training_queue.put({'type': 'error', 'message': str(e) + "\n" + traceback.format_exc()})
                print(f"Training error: {str(e)} \n{traceback.format_exc()}")
                return
            
            if self.is_training:
                self.training_queue.put({'type': 'complete', 'message': 'Training completed successfully!'})
        
        except Exception as e:
            self.training_queue.put({'type': 'error', 'message': f'Training error: {str(e)}'})
    
    def check_training_updates(self):
        """Check for training updates and update plots."""
        try:
            # Process only a limited number of updates per cycle to maintain responsiveness
            updates_processed = 0
            max_updates_per_cycle = 5
            
            while updates_processed < max_updates_per_cycle:
                try:
                    update = self.training_queue.get_nowait()
                    self.process_training_update(update)
                    updates_processed += 1
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error checking updates: {e}")
        
        # Schedule next check (less frequent for better GUI responsiveness)
        self.root.after(250, self.check_training_updates)
    
    def process_training_update(self, update):
        """Process training updates and update plots."""
        if update['type'] == 'training_data':
            self.training_data.append(update)
            # Throttle plot updates to improve responsiveness
            episode = update['episode']
            if episode - self.last_reward_plot_update >= self.plot_update_interval:
                self.update_rewards_plot()
                self.update_values_plot()  # Update physical values plot too
                self.last_reward_plot_update = episode
        
        elif update['type'] == 'pareto_data':
            self.pareto_data.append(update)
            # Store pareto states (recomputed) and best state data if available
            if 'pareto_states' in update and update['pareto_states']:
                self.pareto_states = update['pareto_states']  # States corresponding to Pareto points
            if 'best_state_data' in update and update['best_state_data']:
                self.saved_states = update['best_state_data']  # All best states for fallback
            
            # Store in historical data for slider
            episode = update['episode']
            self.historical_pareto_data[episode] = update.copy()
            
            # Update slider range as training progresses
            max_episode = max(self.historical_pareto_data.keys())
            if max_episode > self.epoch_slider.cget('to'):
                self.epoch_slider.config(to=max_episode)
            
            # Update epoch slider to current episode if we're at the latest
            if self.epoch_var.get() == max_episode - 20 or self.epoch_var.get() == 0:
                self.epoch_var.set(episode)
                self.epoch_label.config(text=f"Episode: {episode}")
            
            # Update plot more frequently to show changes
            if episode - self.last_pareto_plot_update >= 20:  # More frequent updates for Pareto plots
                # Update the current plot if we're viewing the latest episode
                current_episode = self.epoch_var.get()
                if current_episode == episode or current_episode >= episode - 20:
                    self.update_pareto_plot()
                self.last_pareto_plot_update = episode
        
        elif update['type'] == 'complete':
            self.stop_training()
            # Final plot update
            self.update_rewards_plot()
            self.update_values_plot()
            self.update_pareto_plot()
            messagebox.showinfo("Training Complete", update['message'])
        
        elif update['type'] == 'error':
            self.stop_training()
            messagebox.showerror("Training Error", update['message'])
    
    def update_rewards_plot(self):
        """Update the comprehensive rewards plot with multiple subplots."""
        if not self.training_data:
            return
        
        try:
            # Convert training data to DataFrame for easier processing
            df_data = []
            for d in self.training_data:
                row = {'episode': d['episode'], 'reward': d['reward']}
                if 'metrics' in d:
                    row.update(d['metrics'])
                df_data.append(row)
            
            if not df_data:
                return
                
            df = pd.DataFrame(df_data)
            episodes = df['episode'].values
            window_size = min(20, len(episodes) // 4) if len(episodes) > 20 else 5
            
            # Clear all axes
            for ax in self.rewards_axes:
                ax.clear()
                ax.grid(True, alpha=0.3)
            
            # 1. Total Reward
            self._plot_metric_with_smoothing(self.rewards_axes[0], df, 'reward', 'Total Reward', window_size)
            
            # 2. Individual Reward Components
            reward_components = ['reflectivity_reward', 'thermal_noise_reward', 'absorption_reward', 'thickness_reward']
            for component in reward_components:
                if component in df.columns:
                    self.rewards_axes[1].plot(episodes, df[component], alpha=0.6, label=component.replace('_reward', '').replace('_', ' ').title())
            self.rewards_axes[1].set_title('Individual Reward Components')
            self.rewards_axes[1].set_ylabel('Reward')
            self.rewards_axes[1].legend(fontsize=8)
            
            # 3. Entropy Weight (Beta)
            if 'beta' in df.columns:
                self.rewards_axes[2].plot(episodes, df['beta'], 'purple', linewidth=2)
                self.rewards_axes[2].set_title('Entropy Weight (β)')
                self.rewards_axes[2].set_ylabel('Beta')
            
            # 4. Learning Rates
            lr_components = ['lr_discrete', 'lr_continuous', 'lr_value']
            for lr_comp in lr_components:
                if lr_comp in df.columns:
                    self.rewards_axes[3].plot(episodes, df[lr_comp], alpha=0.8, label=lr_comp.replace('lr_', '').title())
            self.rewards_axes[3].set_title('Learning Rates')
            self.rewards_axes[3].set_ylabel('Learning Rate')
            self.rewards_axes[3].legend(fontsize=8)
            self.rewards_axes[3].set_yscale('log')
            
            # 5. Objective Weights
            weight_components = ['reflectivity_reward_weights', 'absorption_reward_weights', 'thermalnoise_reward_weights']
            for weight_comp in weight_components:
                if weight_comp in df.columns:
                    label = weight_comp.replace('_reward_weights', '').replace('thermalnoise', 'thermal_noise').replace('_', ' ').title()
                    self.rewards_axes[4].plot(episodes, df[weight_comp], alpha=0.8, label=label)
            self.rewards_axes[4].set_title('Objective Weights')
            self.rewards_axes[4].set_ylabel('Weight')
            self.rewards_axes[4].legend(fontsize=8)
            
            # 6. Training Losses (moved from position 7)
            loss_components = ['loss_policy_discrete', 'loss_policy_continuous', 'loss_value']
            for loss_comp in loss_components:
                if loss_comp in df.columns:
                    valid_losses = df[loss_comp].dropna()
                    if not valid_losses.empty:
                        label = loss_comp.replace('loss_policy_', '').replace('loss_', '').replace('_', ' ').title()
                        self.rewards_axes[5].plot(episodes[:len(valid_losses)], valid_losses, alpha=0.8, label=label)
            self.rewards_axes[5].set_title('Training Losses')
            self.rewards_axes[5].set_ylabel('Loss')
            self.rewards_axes[5].legend(fontsize=8)
            #self.rewards_axes[5].set_yscale('log')
            
            # Set x-labels for all subplots
            for ax in self.rewards_axes:
                ax.set_xlabel('Episode')
            
            # Update status
            if self.training_data:
                latest = self.training_data[-1]
                self.status_var.set(f"Training... Episode {latest['episode']}, Reward: {latest['reward']:.4f}")
            
            self.rewards_fig.tight_layout()
            self.rewards_canvas.draw_idle()
            self.root.update_idletasks()
            
        except Exception as e:
            print(f"Error updating rewards plot: {e}")
    
    def _plot_metric_with_smoothing(self, ax, df, column, title, window_size):
        """Plot metric with smoothing like in HPPO trainer."""
        if column not in df.columns:
            ax.set_title(f'{title} (No Data)')
            return
            
        episodes = df['episode'].values
        values = df[column].values
        
        # Plot raw data
        ax.plot(episodes, values, alpha=0.3, color='blue')
        
        # Plot smoothed data if enough points
        if len(values) > window_size:
            smoothed = pd.Series(values).rolling(window=window_size, center=False).median()
            ax.plot(episodes, smoothed, linewidth=2, color='red', label='Smoothed')
            ax.legend(fontsize=8)
        
        ax.set_title(title)
        ax.set_ylabel(column.replace('_', ' ').title())
    
    def update_values_plot(self):
        """Update the physical values plot with separate subplots for each metric."""
        if not self.training_data:
            return
        
        try:
            # Convert training data to DataFrame for easier processing
            df_data = []
            for d in self.training_data:
                row = {'episode': d['episode'], 'reward': d['reward']}
                if 'metrics' in d:
                    row.update(d['metrics'])
                df_data.append(row)
            
            if not df_data:
                return
                
            df = pd.DataFrame(df_data)
            episodes = df['episode'].values
            window_size = min(20, len(episodes) // 4) if len(episodes) > 20 else 5
            
            # Clear all axes
            for ax in self.values_axes:
                ax.clear()
                ax.grid(True, alpha=0.3)
            
            # Physical values components with individual scales
            physical_components = [
                ('reflectivity', 'Reflectivity', False),
                ('thermal_noise', 'Thermal Noise', True),
                ('absorption', 'Absorption [ppm]', True), 
                ('thickness', 'Thickness', False)
            ]
            
            for i, (component, title, use_log) in enumerate(physical_components):
                if component in df.columns and i < len(self.values_axes):
                    values = df[component].values
                    
                    # Plot raw data
                    self.values_axes[i].plot(episodes, values, alpha=0.4, color='blue', linewidth=0.8)
                    
                    # Plot smoothed data if enough points
                    if len(values) > window_size:
                        smoothed = pd.Series(values).rolling(window=window_size, center=False).median()
                        self.values_axes[i].plot(episodes, smoothed, linewidth=2, color='red', label='Smoothed')
                        self.values_axes[i].legend(fontsize=8)
                    
                    self.values_axes[i].set_title(title)
                    self.values_axes[i].set_ylabel(title)
                    self.values_axes[i].set_xlabel('Episode')
                    
                    # Use log scale for thermal noise and absorption
                    if use_log:
                        # Replace zeros with NaN to avoid log(0) issues
                        non_zero_values = values[values > 0] if len(values[values > 0]) > 0 else [1e-10]
                        if len(non_zero_values) > 0:
                            self.values_axes[i].set_yscale('log')
            
            # Update status for values tab
            if self.training_data:
                latest = self.training_data[-1]
                if 'metrics' in latest:
                    metrics = latest['metrics']
                    # Show current values in the title or status
                    current_values = []
                    for component, title, _ in physical_components:
                        if component in metrics:
                            if component == 'thermal_noise':
                                current_values.append(f"{title}: {metrics[component]:.2e}")
                            else:
                                current_values.append(f"{title}: {metrics[component]:.4f}")
            
            self.values_fig.tight_layout()
            self.values_canvas.draw_idle()
            self.root.update_idletasks()
            
        except Exception as e:
            print(f"Error updating values plot: {e}")
    
    def update_pareto_plot(self):
        """Update the interactive Pareto front plot with coating visualization."""
        if not self.pareto_data:
            return
        
        try:
            # Clear both subplots
            self.pareto_ax.clear()
            self.coating_ax.clear()
            
            # Get the latest data
            if not self.pareto_data:
                print("No Pareto data available yet")
                return
                
            latest_data = self.pareto_data[-1]
            pareto_front = latest_data['pareto_front'] 
            best_points = latest_data.get('best_points', None)  # Use best_points instead of saved_points
            
            current_episode = latest_data.get('episode', 'unknown')
            #print(f"Updating Pareto plot - Episode: {current_episode}")
            #print(f"Recomputed Pareto front size: {len(pareto_front) if pareto_front is not None else 0}")
            #print(f"Total best points size: {len(best_points) if best_points is not None else 0}")
            pareto_indices = latest_data.get('pareto_indices', [])
            #print(f"Pareto efficiency: {len(pareto_indices)}/{len(best_points) if best_points else 0} = {len(pareto_indices)/len(best_points)*100:.1f}%" if best_points and len(best_points) > 0 else "No efficiency data")
            
            if pareto_front is None or len(pareto_front) == 0:
                self.pareto_ax.text(0.5, 0.5, 'No Pareto front data available\nyet', 
                                  transform=self.pareto_ax.transAxes, ha='center', va='center')
                self.pareto_canvas.draw_idle()
                return
            
            # === Left subplot: Pareto front with all points ===
            # Plot all best points in background with low alpha
            if best_points is not None and len(best_points) > 0:
                try:
                    best_points_array = np.array(best_points)
                    if len(best_points_array.shape) == 2 and best_points_array.shape[1] >= 2:
                        ref_values = best_points_array[:, 0]  # Already converted to 1-R
                        abs_values = best_points_array[:, 1]
                        
                        self.pareto_ax.scatter(
                            ref_values, abs_values,
                            c='lightblue', alpha=0.3, s=10, label='All Best Solutions'
                        )
                except Exception as e:
                    print(f"Error plotting best points: {e}")
            
            # Plot current Pareto front with clickable points
            pareto_scatter = None
            if len(pareto_front) > 0:
                pareto_scatter = self.pareto_ax.scatter(
                    pareto_front[:, 0], pareto_front[:, 1],
                    c='red', s=50, alpha=0.8, label='Recomputed Pareto Front',
                    edgecolors='black', linewidths=1, picker=True
                )
                
                # Connect Pareto points with line
                sorted_indices = np.argsort(pareto_front[:, 0])
                sorted_front = pareto_front[sorted_indices]
                self.pareto_ax.plot(
                    sorted_front[:, 0], sorted_front[:, 1],
                    'r-', alpha=0.5, linewidth=1, label='Pareto Front Line'
                )
            
            # Add target lines if available
            if self.env and hasattr(self.env, 'design_criteria'):
                targets = self.env.design_criteria
                if 'reflectivity' in targets and 'absorption' in targets:
                    ref_target = 1 - targets['reflectivity']
                    abs_target = targets['absorption']
                    
                    self.pareto_ax.axhline(abs_target, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Absorption Criteria')
                    self.pareto_ax.axvline(ref_target, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Reflectivity Criteria')
            else:
                logging.warning("Environment design criteria not available for target lines")
            # Setup Pareto plot
            self.pareto_ax.set_title(f"Recomputed Pareto Front - Episode {current_episode}\n(Click point to view coating)")
            self.pareto_ax.set_xlabel("1 - Reflectivity")
            self.pareto_ax.set_ylabel("Absorption [ppm]")
            self.pareto_ax.set_xscale("log")
            self.pareto_ax.set_yscale("log")
            self.pareto_ax.grid(True, alpha=0.3)
            self.pareto_ax.legend(fontsize=8)
            
            # === Right subplot: Coating stack placeholder ===
            self.coating_ax.text(0.5, 0.5, 'Click a Pareto point\nto view coating stack', 
                               transform=self.coating_ax.transAxes, ha='center', va='center',
                               fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            self.coating_ax.set_title("Coating Stack Visualization")
            self.coating_ax.set_xlabel("Position")
            self.coating_ax.set_ylabel("Thickness")
            self.coating_ax.grid(True, alpha=0.3)
            
            # Set up click event handler (only once to avoid multiple handlers)
            def on_click(event):
                if event.inaxes == self.pareto_ax and pareto_scatter is not None and len(pareto_front) > 0:
                    # Find closest Pareto point
                    if event.xdata is not None and event.ydata is not None:
                        # Get click coordinates
                        click_x, click_y = event.xdata, event.ydata
                        
                        # Find closest point (accounting for log scales)
                        # Transform to log space for distance calculation
                        log_click_x, log_click_y = np.log10(max(click_x, 1e-10)), np.log10(max(click_y, 1e-10))
                        log_front_x = np.log10(np.maximum(pareto_front[:, 0], 1e-10))
                        log_front_y = np.log10(np.maximum(pareto_front[:, 1], 1e-10))
                        
                        distances = np.sqrt((log_front_x - log_click_x)**2 + (log_front_y - log_click_y)**2)
                        closest_idx = np.argmin(distances)
                        
                        #print(f"Clicked on Pareto point {closest_idx}: {pareto_front[closest_idx]}")
                        
                        # Update coating visualization
                        self.plot_coating_stack(closest_idx, pareto_front[closest_idx])
            
            # Disconnect any existing click event handler to prevent multiple handlers
            if self.click_event_connection is not None:
                self.pareto_canvas.mpl_disconnect(self.click_event_connection)
            
            # Connect new click event handler
            self.click_event_connection = self.pareto_canvas.mpl_connect('button_press_event', on_click)
            
            # Add statistics text box
            if len(pareto_front) > 0:
                n_pareto_points = len(pareto_front)
                n_total_points = len(best_points) if best_points is not None else 0
                
                stats_text = f'Pareto Optimal: {n_pareto_points}'
                if n_total_points > 0:
                    stats_text += f'\nTotal Solutions: {n_total_points}'
                    stats_text += f'\nEfficiency: {n_pareto_points/n_total_points*100:.1f}%'
                
                self.pareto_ax.text(0.02, 0.98, stats_text, transform=self.pareto_ax.transAxes,
                                   verticalalignment='top', horizontalalignment='left',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                   fontsize=9)
            
            self.pareto_fig.tight_layout()
            self.pareto_canvas.draw_idle()
            self.root.update_idletasks()
            
        except Exception as e:
            print(f"Error updating pareto plot: {e}")
    
    def plot_coating_stack(self, pareto_idx, pareto_point):
        """Plot the coating stack for a selected Pareto point."""
        try:
            self.coating_ax.clear()
            
            #print(f"Plotting coating stack for Pareto index {pareto_idx}")
            #print(f"Available pareto_states: {len(self.pareto_states) if self.pareto_states else 0}")
            #print(f"Available saved_states: {len(self.saved_states) if self.saved_states else 0}")
            
            # Try to get the coating state for this Pareto point
            state = None
            if self.pareto_states and pareto_idx < len(self.pareto_states):
                state = self.pareto_states[pareto_idx]
                #print(f"Using pareto_states[{pareto_idx}]")  # Commented out to reduce console spam
            elif self.saved_states and pareto_idx < len(self.saved_states):
                state = self.saved_states[pareto_idx]
                #print(f"Using saved_states[{pareto_idx}] as fallback")  # Commented out to reduce console spam
            
            if state is not None:
                #print(f"State shape: {np.array(state).shape}")
                # Parse coating state to extract layers
                layers = self.parse_coating_state(state)
                
                if layers:
                    # Material colors (matching your existing scheme)
                    material_colors = {
                        0: 'lightgray',    # Air
                        1: 'blue',         # Substrate/SiO2  
                        2: 'green',        # Ta2O5
                        3: 'red',          # TiO2
                        4: 'purple',       # Additional materials
                        5: 'orange',
                        6: 'brown',
                    }
                    
                    # Plot each layer as a bar
                    x_positions = range(len(layers))
                    thicknesses = [layer[1] for layer in layers]
                    materials = [layer[0] for layer in layers]
                    
                    bars = self.coating_ax.bar(x_positions, thicknesses, 
                                             color=[material_colors.get(mat, 'gray') for mat in materials],
                                             alpha=0.8, edgecolor='black', linewidth=1)
                    
                    # Add material labels
                    material_names = {0: 'Air', 1: 'SiO₂', 2: 'Ta₂O₅', 3: 'TiO₂', 4: 'Mat4', 5: 'Mat5', 6: 'Mat6'}
                    for i, (material, thickness) in enumerate(layers):
                        if thickness > 0.01:  # Only label if thick enough
                            self.coating_ax.text(i, thickness/2, material_names.get(material, f'M{material}'), 
                                               ha='center', va='center', fontsize=8, fontweight='bold')
                    
                    self.coating_ax.set_title(f"Coating Stack\n1-R: {pareto_point[0]:.2e}, Abs: {pareto_point[1]:.2f}")
                    self.coating_ax.set_xlabel("Layer Index")
                    self.coating_ax.set_ylabel("Optical Thickness")
                    self.coating_ax.grid(True, alpha=0.3)
                else:
                    self.coating_ax.text(0.5, 0.5, 'Could not parse\ncoating state', 
                                       transform=self.coating_ax.transAxes, ha='center', va='center')
            else:
                self.coating_ax.text(0.5, 0.5, 'Coating state\nnot available', 
                                   transform=self.coating_ax.transAxes, ha='center', va='center')
            
            self.pareto_canvas.draw_idle()
            
        except Exception as e:
            print(f"Error plotting coating stack: {e}")
            self.coating_ax.text(0.5, 0.5, f'Error: {str(e)}', 
                               transform=self.coating_ax.transAxes, ha='center', va='center')
    
    def cleanup(self):
        """Clean up resources when closing the UI."""
        try:
            # Disconnect click event handler
            if self.click_event_connection is not None:
                self.pareto_canvas.mpl_disconnect(self.click_event_connection)
                self.click_event_connection = None
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def parse_coating_state(self, state):
        """Parse a coating state array into (material, thickness) pairs."""
        try:
            if state is None or len(state) == 0:
                print("State is None or empty")
                return []
            
            state_array = np.array(state)
            #print(f"State array shape: {state_array.shape}")
            #print(f"First few values: {state_array.flatten()[:10]}")
            
            # Assuming state format: each row represents a layer with [thickness, material_1, material_2, ...]
            # where material columns are one-hot encoded
            layers = []
            
            for layer_idx in range(len(state_array)):
                layer = state_array[layer_idx]
                #print(f"Layer {layer_idx}: {layer}")
                if len(layer) > 1:
                    thickness = layer[0]
                    if thickness > 1e-6:  # Only include layers with significant thickness
                        # Find which material (one-hot encoded)
                        material_probs = layer[1:]
                        material = np.argmax(material_probs) if len(material_probs) > 0 else 0
                        layers.append((material, thickness))
                        #print(f"  -> Material: {material}, Thickness: {thickness}")
            
            #print(f"Parsed {len(layers)} layers: {layers}")
            return layers
            
        except Exception as e:
            print(f"Error parsing coating state: {e}")
            import traceback
            traceback.print_exc()
            return []


def main():
    """Main function to run the training UI."""
    root = tk.Tk()
    
    # Improve GUI responsiveness
    root.option_add('*tearOff', False)  # Disable tear-off menus
    root.wm_attributes('-topmost', False)  # Don't force window on top
    
    # Set minimum window size
    root.minsize(800, 600)
    
    app = TrainingMonitorUI(root)
    
    try:
        # Process events regularly to keep GUI responsive
        while True:
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                # Window was closed
                break
            except KeyboardInterrupt:
                print("Application interrupted by user")
                break
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        try:
            app.cleanup()  # Clean up event handlers
            root.quit()
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main()