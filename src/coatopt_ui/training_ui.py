"""
Real-time Training UI for PC-HPPO-OML Coating optimisation

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
from coatopt.config.structured_config import CoatingoptimisationConfig
from coatopt.factories import setup_optimisation_pipeline
from coatopt.algorithms.hppo_trainer import HPPOTrainer, create_ui_callbacks

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
        
        # Store objective information for consistent labeling
        self.objective_labels = []
        self.objective_scales = []
        self.optimisation_parameters = []
        
        # Event handler tracking
        self.click_event_connection = None  # Store event connection to prevent multiple handlers
        
        # Plot update throttling
        self.last_reward_plot_update = 0
        self.last_pareto_plot_update = 0
        self.plot_update_interval = 40  # Update plots every N episodes
        
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
        
        # Epoch slider and controls for Pareto plot - placed at top
        self.pareto_controls_frame = ttk.Frame(self.pareto_frame)
        self.pareto_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.pareto_controls_frame, text="Training Epoch:").pack(side=tk.LEFT)
        
        self.epoch_var = tk.IntVar(value=0)
        self.epoch_slider = tk.Scale(self.pareto_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   variable=self.epoch_var, command=self.on_epoch_change, 
                                   length=400, resolution=self.plot_update_interval)
        self.epoch_slider.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        self.epoch_label = ttk.Label(self.pareto_controls_frame, text="Episode: 0")
        self.epoch_label.pack(side=tk.RIGHT)
        
        # Visualization mode selection
        self.viz_mode_frame = ttk.Frame(self.pareto_controls_frame)
        self.viz_mode_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Label(self.viz_mode_frame, text="View:").pack(side=tk.LEFT)
        self.viz_mode = tk.StringVar(value="2D_pairs")
        self.viz_mode_combo = ttk.Combobox(self.viz_mode_frame, textvariable=self.viz_mode, 
                                          values=["2D_pairs", "parallel_coords", "3D_scatter"], 
                                          state="readonly", width=12)
        self.viz_mode_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.viz_mode_combo.bind('<<ComboboxSelected>>', self.on_viz_mode_change)
        
        # Create dynamic figure that adapts to number of objectives
        self.pareto_fig = Figure(figsize=(14, 10), dpi=100)  # Larger for multiple subplots
        self.pareto_canvas = FigureCanvasTkAgg(self.pareto_fig, self.pareto_frame)
        self.pareto_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty plots
        self.init_plots()
    
    def on_viz_mode_change(self, event=None):
        """Handle visualization mode change."""
        try:
            if hasattr(self, 'pareto_data') and self.pareto_data:
                # Ensure objective info is current before switching viz mode
                if not hasattr(self, 'objective_labels') or not self.objective_labels:
                    self.update_objective_info()
                self.update_pareto_plot()
        except Exception as e:
            print(f"Error changing visualization mode: {e}")

    def update_objective_info(self):
        """Update stored objective information based on current environment."""
        if not self.env or not hasattr(self.env, 'optimise_parameters'):
            self.objective_labels = []
            self.objective_scales = []
            return
        
        self.objective_labels = self.get_objective_labels()
        self.objective_scales = self.get_objective_scales()

    def get_objective_labels(self):
        """Get human-readable labels for objectives."""
        # First try to use stored parameters
        if hasattr(self, 'optimisation_parameters') and self.optimisation_parameters:
            objectives = self.optimisation_parameters
        elif self.env and hasattr(self.env, 'optimise_parameters'):
            objectives = self.env.optimise_parameters
        else:
            return []
        
        label_mapping = {
            'reflectivity': '1 - Reflectivity',
            'absorption': 'Absorption [ppm]', 
            'thermal_noise': 'Thermal Noise [m/√Hz]',
            'thickness': 'Total Thickness [nm]'
        }
        
        return [label_mapping.get(param, param.replace('_', ' ').title()) for param in objectives]

    def get_objective_scales(self):
        """Get appropriate scales (linear/log) for each objective."""
        # First try to use stored parameters
        if hasattr(self, 'optimisation_parameters') and self.optimisation_parameters:
            objectives = self.optimisation_parameters
        elif self.env and hasattr(self.env, 'optimise_parameters'):
            objectives = self.env.optimise_parameters
        else:
            return []
        
        scale_mapping = {
            'reflectivity': 'log',
            'absorption': 'log', 
            'thermal_noise': 'log',
            'thickness': 'linear'
        }
        
        return [scale_mapping.get(param, 'linear') for param in objectives]

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
        
        # Pareto front plot - will be dynamically created based on objectives
        self.pareto_fig.clear()
        ax = self.pareto_fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'Load configuration to see Pareto front visualization', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title("Pareto Front Visualization")
        
        self.pareto_fig.tight_layout()
        self.pareto_canvas.draw_idle()
        
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
    
    
    def browse_config(self):
        """Browse for configuration file."""
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )
        if filename:
            self.config_var.set(filename)
    
    def load_configuration(self):
        """Load the configuration file and setup components using unified engine."""
        config_path = self.config_var.get()
        if not config_path or not os.path.exists(config_path):
            messagebox.showerror("Error", "Please select a valid configuration file.")
            return
        
        try:
            self.status_var.set("Loading configuration...")
            
            # Determine whether to continue training based on checkbox
            continue_training = not self.retrain_var.get()  # If retrain is checked, don't continue
            
            # Load configuration and materials
            raw_config = read_config(config_path)
            self.config = CoatingoptimisationConfig.from_config_parser(raw_config)
            self.materials = read_materials(self.config.general.materials_file)
            
            # Setup optimization components
            self.env, self.agent, temp_trainer = setup_optimisation_pipeline(
                self.config, 
                self.materials, 
                continue_training=continue_training,
                init_pareto_front=False
            )
            
            # The trainer will be created in training_worker with UI callbacks
            self.trainer = None
            self.continue_training = continue_training
            
            print(f"Optimization parameters: {self.config.data.optimise_parameters}")
            print(f"Number of objectives: {len(self.config.data.optimise_parameters)}")
            print(f"Agent num_objectives: {self.agent.num_objectives}")
            print(f"Environment optimize_parameters: {self.env.optimise_parameters}")
            
            # Store objective information for consistent UI labeling
            self.optimisation_parameters = self.env.optimise_parameters.copy()
            self.update_objective_info()
            
            # Load historical training data if not retraining
            if continue_training:
                # Create temporary trainer to load historical data
                temp_trainer_for_data = HPPOTrainer(
                    self.agent, self.env,
                    n_iterations=self.config.training.n_iterations,
                    n_layers=self.config.data.n_layers,
                    root_dir=self.config.general.root_dir,
                    continue_training=True
                )
                self.trainer = temp_trainer_for_data  # Temporarily store for load_historical_data
                self.load_historical_data()
                self.trainer = None  # Reset for training_worker
            
            # Update status with configuration info
            pareto_info = ""
            if hasattr(self.env, 'pareto_front') and len(self.env.pareto_front) > 0:
                pareto_info = f" Loaded Pareto front: {len(self.env.pareto_front)} points."
            
            self.status_var.set(f"Configuration loaded successfully. {len(self.materials)} materials loaded.{pareto_info}")
            self.start_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
            self.status_var.set("Failed to load configuration.")
            logging.CRITICAL(f"Error loading configuration: {e}, traceback: {traceback.format_exc()}")
    
    def load_historical_data(self):
        """Load historical training data for the epoch slider."""
        try:
            # Load historical Pareto data
            if hasattr(self.trainer, 'best_states') and self.trainer.best_states:
                print(f"Loading {len(self.trainer.best_states)} historical best states...")
                
                # Process best_states to generate historical Pareto data
                accumulated_states = []
                for i, best_state_entry in enumerate(self.trainer.best_states):
                    tot_reward, epoch, state, rewards, vals = best_state_entry
                    accumulated_states.append(best_state_entry)
                    
                    # Generate Pareto data for every 20 episodes or final episode
                    if (i + 1) % self.plot_update_interval == 0 or i == len(self.trainer.best_states) - 1:
                        episode = (i + 1) * self.plot_update_interval  # Approximate episode number
                        pareto_data = self._generate_pareto_data_from_states(accumulated_states, episode)
                        self.historical_pareto_data[episode] = pareto_data
                
                # Update slider range
                if self.historical_pareto_data:
                    max_episode = max(self.historical_pareto_data.keys())
                    self.epoch_slider.config(to=max_episode)
                    self.epoch_var.set(max_episode)  # Start at the latest episode
                    self.epoch_label.config(text=f"Episode: {max_episode}")
                    
                    print(f"Loaded historical Pareto data for {len(self.historical_pareto_data)} time points")
                    # Update plot with the latest data
                    self.update_pareto_plot_for_episode(max_episode)
            
            # Load historical training/rewards data
            if hasattr(self.trainer, 'metrics') and not self.trainer.metrics.empty:
                print(f"Loading historical training metrics with {len(self.trainer.metrics)} episodes...")
                
                # Convert metrics DataFrame to training_data format
                self.training_data = []
                for _, row in self.trainer.metrics.iterrows():
                    episode_data = {
                        'episode': int(row.get('episode', 0)),
                        'reward': float(row.get('reward', 0.0)),
                        'metrics': {}
                    }
                    
                    # Add all available metrics
                    for col in row.index:
                        if col != 'episode' and not pd.isna(row[col]):
                            try:
                                episode_data['metrics'][col] = float(row[col])
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                pass
                    
                    self.training_data.append(episode_data)
                
                print(f"Loaded {len(self.training_data)} training data points")
                
                # Update reward and values plots
                self.update_rewards_plot()
                self.update_values_plot()
                
                # Update status to show data was loaded
                total_episodes = len(self.training_data)
                latest_reward = self.training_data[-1]['reward'] if self.training_data else 0.0
                
                # Include Pareto front information if environment is available
                pareto_info = ""
                if hasattr(self, 'env') and hasattr(self.env, 'pareto_front') and len(self.env.pareto_front) > 0:
                    pareto_info = f", Pareto front: {len(self.env.pareto_front)} points"
                
                self.status_var.set(f"Historical data loaded: {total_episodes} episodes, latest reward: {latest_reward:.4f}{pareto_info}")
                
        except Exception as e:
            print(f"Error loading historical data: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_pareto_data_from_states(self, best_states, episode):
        """Generate Pareto data from a list of best states."""
        best_points = []
        best_state_data = []
        best_vals_list = []
        
        if not best_states:
            return {
                'episode': episode,
                'pareto_front': np.array([]),
                'best_points': [],
                'best_state_data': [],
                'pareto_indices': [],
                'pareto_states': [],
            }
        
        # Get objectives from stored parameters or environment
        if hasattr(self, 'optimisation_parameters') and self.optimisation_parameters:
            objectives = self.optimisation_parameters
        elif hasattr(self, 'env') and self.env and hasattr(self.env, 'optimise_parameters'):
            objectives = self.env.optimise_parameters
        else:
            # Fallback: inspect first state to determine available objectives
            first_vals = best_states[0][4] if len(best_states) > 0 else {}
            objectives = list(first_vals.keys())
        
        for tot_reward, epoch, state, rewards, vals in best_states:
            # Extract relevant metrics for Pareto front based on actual objectives
            objective_values = []
            plot_values = []
            
            for obj in objectives:
                if obj in vals:
                    val = vals[obj]
                    objective_values.append(val)
                    
                    # Apply transformations for plotting (minimize all objectives)
                    if obj == 'reflectivity':
                        plot_values.append(1 - val)  # Convert to loss (1-R)
                    else:
                        plot_values.append(val)  # Use as-is for absorption, thermal_noise, thickness
                else:
                    # Skip this state if missing required objective data
                    objective_values = None
                    break
            
            if objective_values is not None:
                best_points.append(plot_values)
                best_state_data.append(state)
                best_vals_list.append(objective_values)
        
        # Recompute Pareto front from best points using vals
        recomputed_pareto_front = []
        pareto_indices = []
        
        if len(best_vals_list) > 0:
            # Use non-dominated sorting on the actual values
            vals_array = np.array(best_vals_list)
            
            # Convert to minimization objectives for all parameters
            minimization_objectives = np.zeros_like(vals_array)
            for i, obj in enumerate(objectives):
                if obj == 'reflectivity':
                    minimization_objectives[:, i] = 1 - vals_array[:, i]  # Minimize (1-R)
                else:
                    minimization_objectives[:, i] = vals_array[:, i]  # Minimize directly
            
            nds = NonDominatedSorting()
            fronts = nds.do(minimization_objectives)
            
            if len(fronts) > 0 and len(fronts[0]) > 0:
                pareto_indices = fronts[0]
                # Convert to plotting format
                recomputed_pareto_front = [best_points[i] for i in pareto_indices]
        
        return {
            'episode': episode,
            'pareto_front': np.array(recomputed_pareto_front) if recomputed_pareto_front else np.array([]),
            'best_points': best_points,
            'best_state_data': best_state_data,
            'pareto_indices': pareto_indices,
            'pareto_states': [best_state_data[i] for i in pareto_indices]
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
        if self.agent is None or self.env is None or self.config is None:
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
            # Create UI callbacks
            callbacks = create_ui_callbacks(self.training_queue, lambda: not self.is_training)
            
            # Create unified trainer with callbacks
            unified_trainer = HPPOTrainer(
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
                use_unified_checkpoints=True,  # Use unified HDF5 checkpoint system
                save_plots=False,  # UI handles plots separately
                save_episode_visualizations=False,  # UI handles visualizations separately
                continue_training=self.continue_training,
                callbacks=callbacks
            )
            
            # Log checkpoint system info
            if hasattr(unified_trainer, 'checkpoint_manager'):
                checkpoint_info = unified_trainer.checkpoint_manager.get_checkpoint_info()
                if checkpoint_info.get('exists', False):
                    checkpoint_msg = f"Using unified checkpoint: {checkpoint_info['size_mb']:.1f}MB"
                    print(checkpoint_msg)
                    # Update UI status to show checkpoint info
                    self.training_queue.put({
                        'type': 'status_update', 
                        'message': f"Loaded unified checkpoint ({checkpoint_info['size_mb']:.1f}MB) - Training ready..."
                    })
                else:
                    print("Initializing new unified checkpoint system")
                    self.training_queue.put({
                        'type': 'status_update', 
                        'message': "Initialized unified checkpoint system - Training ready..."
                    })
            
            # Initialize Pareto front if not continuing
            if not self.continue_training:
                unified_trainer.init_pareto_front(n_solutions=1000)
            
            # Run training
            try:
                final_metrics, final_state = unified_trainer.train()
            except Exception as e:
                error_msg = f"Training error: {str(e)}"
                traceback_str = traceback.format_exc()
                full_error = f"{error_msg}\n\nFull traceback:\n{traceback_str}"
                
                # Print to terminal for debugging
                print("=" * 60)
                print("TRAINING ERROR")
                print("=" * 60)
                print(full_error)
                print("=" * 60)
                
                # Send to UI queue
                self.training_queue.put({'type': 'error', 'message': full_error})
                return
            
            if self.is_training:
                self.training_queue.put({'type': 'complete', 'message': 'Training completed successfully!'})
        
        except Exception as e:
            error_msg = f'Training worker error: {str(e)}'
            traceback_str = traceback.format_exc()
            full_error = f"{error_msg}\n\nFull traceback:\n{traceback_str}"
            
            # Print to terminal for debugging
            print("=" * 60)
            print("TRAINING WORKER ERROR")
            print("=" * 60)
            print(full_error)
            print("=" * 60)
            
            # Send to UI queue
            self.training_queue.put({'type': 'error', 'message': full_error})
    
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
                # Save plots after updating
                self.save_plots_to_disk()
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
            if self.epoch_var.get() == max_episode - self.plot_update_interval or self.epoch_var.get() == 0:
                self.epoch_var.set(episode)
                self.epoch_label.config(text=f"Episode: {episode}")
            
            # Update plot more frequently to show changes
            if episode - self.last_pareto_plot_update >= self.plot_update_interval:  # More frequent updates for Pareto plots
                # Ensure objective information is up to date before plotting
                if not hasattr(self, 'objective_labels') or not self.objective_labels:
                    self.update_objective_info()
                
                # Update the current plot if we're viewing the latest episode
                current_episode = self.epoch_var.get()
                if current_episode == episode or current_episode >= episode - self.plot_update_interval:
                    self.update_pareto_plot()
                self.last_pareto_plot_update = episode
        
        elif update['type'] == 'complete':
            self.stop_training()
            # Final plot update - ensure objective info is current
            if not hasattr(self, 'objective_labels') or not self.objective_labels:
                self.update_objective_info()
            self.update_rewards_plot()
            self.update_values_plot()
            self.update_pareto_plot()
            # Save final plots
            self.save_plots_to_disk()
            messagebox.showinfo("Training Complete", update['message'])
        
        elif update['type'] == 'status_update':
            # Update status label with checkpoint information
            self.status_var.set(update['message'])
        
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
            
            # 2. Individual Reward Components - Dynamic based on optimize_parameters
            if hasattr(self, 'optimisation_parameters') and self.optimisation_parameters:
                # Use stored optimisation parameters for consistency
                optimise_params = self.optimisation_parameters
            elif hasattr(self, 'env') and self.env and hasattr(self.env, 'optimise_parameters'):
                # Get dynamic reward component names based on actual objectives
                optimise_params = self.env.optimise_parameters
            else:
                # Final fallback
                optimise_params = ['reflectivity', 'thermal_noise', 'absorption', 'thickness']
            
            reward_components = []
            
            # Map parameter names to potential reward column names
            for param in optimise_params:
                possible_names = [
                    f'{param}_reward',
                    f'{param.replace("thermal_noise", "thermalnoise")}_reward'  # Handle specific naming
                ]
                
                for name in possible_names:
                    if name in df.columns:
                        reward_components.append(name)
                        break
            
            for component in reward_components:
                if component in df.columns:
                    label = component.replace('_reward', '').replace('thermalnoise', 'thermal_noise').replace('_', ' ').title()
                    self.rewards_axes[1].plot(episodes, df[component], alpha=0.6, label=label)
                    
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
            
            # 5. Objective Weights - Dynamic based on optimize_parameters
            if hasattr(self, 'optimisation_parameters') and self.optimisation_parameters:
                # Use stored optimisation parameters for consistency
                optimise_params = self.optimisation_parameters
            elif hasattr(self, 'env') and self.env and hasattr(self.env, 'optimise_parameters'):
                # Get dynamic weight component names based on actual objectives
                optimise_params = self.env.optimise_parameters
            else:
                # Final fallback
                optimise_params = ['reflectivity', 'thermal_noise', 'absorption', 'thickness']
            
            weight_components = []
            
            # Map parameter names to potential weight column names
            for param in optimise_params:
                # Try different possible naming conventions for weights
                possible_names = [
                    f'{param}_reward_weights',
                    f'{param}_weights', 
                    f'{param.replace("_", "")}_reward_weights',
                    f'{param.replace("thermal_noise", "thermalnoise")}_reward_weights'  # Handle specific case
                ]
                
                for name in possible_names:
                    if name in df.columns:
                        weight_components.append((name, param))
                        break
            
            # Plot the weights
            for weight_comp, param in weight_components:
                label = param.replace('_', ' ').title()
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
        """Update the Pareto front plot with dynamic visualization based on number of objectives."""
        if not self.pareto_data:
            return
        
        try:
            # Clear the figure
            self.pareto_fig.clear()
            
            # Get the latest data
            latest_data = self.pareto_data[-1]
            
            # Check if this update contains actual pareto front data
            if 'pareto_front' not in latest_data:
                # This is likely a simplified update with only pareto_front_size
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                size = latest_data.get('pareto_front_size', 0)
                ax.text(0.5, 0.5, f'Pareto front size: {size}\nDetailed data not yet available', 
                       transform=ax.transAxes, ha='center', va='center')
                self.pareto_canvas.draw_idle()
                return
                
            pareto_front = latest_data['pareto_front'] 
            best_points = latest_data.get('best_points', None)
            current_episode = latest_data.get('episode', 'unknown')
            
            if pareto_front is None or len(pareto_front) == 0:
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No Pareto front data available yet', 
                       transform=ax.transAxes, ha='center', va='center')
                self.pareto_canvas.draw_idle()
                return
            
            if (not hasattr(self, 'objective_labels') or 
                not self.objective_labels or 
                len(self.objective_labels) != pareto_front.shape[1]):
                
                # Force refresh from environment and stored parameters
                if self.env and hasattr(self.env, 'optimise_parameters'):
                    self.optimisation_parameters = self.env.optimise_parameters.copy()
                self.update_objective_info()
            
            # Use stored objective information
            obj_labels = self.objective_labels.copy() if hasattr(self, 'objective_labels') else []
            obj_scales = self.objective_scales.copy() if hasattr(self, 'objective_scales') else []
            
            # Final validation - ensure we have valid objective information
            if not obj_labels or len(obj_labels) != pareto_front.shape[1]:
                # Emergency fallback: create from stored parameters or generic
                if hasattr(self, 'optimisation_parameters') and self.optimisation_parameters:
                    if len(self.optimisation_parameters) == pareto_front.shape[1]:
                        label_mapping = {
                            'reflectivity': '1 - Reflectivity',
                            'absorption': 'Absorption [ppm]', 
                            'thermal_noise': 'Thermal Noise [m/√Hz]',
                            'thickness': 'Total Thickness [nm]'
                        }
                        obj_labels = [label_mapping.get(param, param.replace('_', ' ').title()) 
                                     for param in self.optimisation_parameters]
                        obj_scales = ['log' if param in ['reflectivity', 'absorption', 'thermal_noise'] 
                                     else 'linear' for param in self.optimisation_parameters]
                    else:
                        obj_labels = [f'Objective {i+1}' for i in range(pareto_front.shape[1])]
                        obj_scales = ['linear'] * pareto_front.shape[1]
                else:
                    obj_labels = [f'Objective {i+1}' for i in range(pareto_front.shape[1])]
                    obj_scales = ['linear'] * pareto_front.shape[1]
            
            # Ensure scales list matches labels
            if not obj_scales or len(obj_scales) != len(obj_labels):
                obj_scales = ['linear'] * len(obj_labels)
            
            n_objectives = len(obj_labels)
            
            if n_objectives < 2:
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'Need at least 2 objectives for Pareto visualization', 
                       transform=ax.transAxes, ha='center', va='center')
                self.pareto_canvas.draw_idle()
                return
            
            viz_mode = self.viz_mode.get()
            
            if viz_mode == "parallel_coords":
                self.plot_parallel_coordinates(pareto_front, best_points, obj_labels, current_episode)
            elif viz_mode == "3D_scatter" and n_objectives >= 3:
                self.plot_3d_scatter(pareto_front, best_points, obj_labels, obj_scales, current_episode)
            else:  # Default to 2D pairs - this should use the same logic as initialization
                self.plot_2d_pairs(pareto_front, best_points, obj_labels, obj_scales, current_episode)
            
            self.pareto_fig.tight_layout()
            self.pareto_canvas.draw_idle()
            self.root.update_idletasks()
            
            # Save plots to disk after updating
            self.save_plots_to_disk()
            
        except Exception as e:
            print(f"Error updating Pareto plot: {e}")
            import traceback
            traceback.print_exc()

    def plot_2d_pairs(self, pareto_front, best_points, obj_labels, obj_scales, episode):
        """Plot all pairs of objectives in 2D subplots."""
        n_objectives = len(obj_labels)
        
        if n_objectives == 2:
            # Single 2D plot with coating visualization
            gs = self.pareto_fig.add_gridspec(1, 2, width_ratios=[2, 1])
            self.pareto_ax = self.pareto_fig.add_subplot(gs[0])
            self.coating_ax = self.pareto_fig.add_subplot(gs[1])
            
            self.plot_single_2d_pair(self.pareto_ax, pareto_front, best_points, 
                                   0, 1, obj_labels, obj_scales, episode, enable_click=True)
            
            # Coating visualization placeholder
            self.coating_ax.text(0.5, 0.5, 'Click a Pareto point\nto view coating stack', 
                               transform=self.coating_ax.transAxes, ha='center', va='center',
                               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            self.coating_ax.set_title("Coating Stack")
            
        else:
            # Multiple 2D plots for all pairs + coating stack subplot
            n_pairs = n_objectives * (n_objectives - 1) // 2
            total_subplots = n_pairs + 1  # Add 1 for coating stack
            n_cols = min(3, total_subplots)  # Max 3 columns
            n_rows = (total_subplots + n_cols - 1) // n_cols
            
            # Store subplot info for click handling
            self.subplot_info = []
            
            # Create 2D pair plots
            pair_idx = 0
            for i in range(n_objectives):
                for j in range(i + 1, n_objectives):
                    if pair_idx < n_pairs:
                        ax = self.pareto_fig.add_subplot(n_rows, n_cols, pair_idx + 1)
                        self.plot_single_2d_pair(ax, pareto_front, best_points, 
                                               i, j, obj_labels, obj_scales, episode, enable_click=False)
                        # Store info for click handling
                        self.subplot_info.append((ax, i, j, obj_scales))
                        pair_idx += 1
            
            # Add coating stack subplot in the next available position
            self.coating_ax = self.pareto_fig.add_subplot(n_rows, n_cols, total_subplots)
            self.coating_ax.text(0.5, 0.5, 'Click a Pareto point\nto view coating stack', 
                               transform=self.coating_ax.transAxes, ha='center', va='center',
                               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            self.coating_ax.set_title("Coating Stack")
            
            # Set up global click handler for all subplots
            def on_click_multi(event):
                if hasattr(self, 'subplot_info') and len(pareto_front) > 0:
                    for ax, i, j, scales in self.subplot_info:
                        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                            # Find closest point
                            click_x, click_y = event.xdata, event.ydata
                            
                            # Handle log scale distances
                            if i < len(scales) and j < len(scales) and scales[i] == 'log' and scales[j] == 'log':
                                log_click_x = np.log10(max(click_x, 1e-10))
                                log_click_y = np.log10(max(click_y, 1e-10))
                                log_front_x = np.log10(np.maximum(pareto_front[:, i], 1e-10))
                                log_front_y = np.log10(np.maximum(pareto_front[:, j], 1e-10))
                                distances = np.sqrt((log_front_x - log_click_x)**2 + (log_front_y - log_click_y)**2)
                            else:
                                distances = np.sqrt((pareto_front[:, i] - click_x)**2 + (pareto_front[:, j] - click_y)**2)
                            
                            closest_idx = np.argmin(distances)
                            self.plot_coating_stack(closest_idx, pareto_front[closest_idx])
                            break
            
            # Disconnect any existing handler and connect new one
            if self.click_event_connection is not None:
                self.pareto_canvas.mpl_disconnect(self.click_event_connection)
            self.click_event_connection = self.pareto_canvas.mpl_connect('button_press_event', on_click_multi)

    def plot_single_2d_pair(self, ax, pareto_front, best_points, i, j, obj_labels, obj_scales, episode, enable_click=False):
        """Plot a single 2D pair of objectives."""
        # Plot all best points in background
        if best_points is not None and len(best_points) > 0:
            try:
                best_points_array = np.array(best_points)
                if len(best_points_array.shape) == 2 and best_points_array.shape[1] > max(i, j):
                    ax.scatter(best_points_array[:, i], best_points_array[:, j],
                             c='lightblue', alpha=0.3, s=10, label='All Solutions')
            except Exception as e:
                print(f"Error plotting background points: {e}")
        
        # Plot Pareto front
        pareto_scatter = ax.scatter(pareto_front[:, i], pareto_front[:, j],
                                  c='red', s=40, alpha=0.8, label='Pareto Front',
                                  edgecolors='black', linewidths=1)
        
        # Connect Pareto points if feasible
        if len(pareto_front) < 20:  # Only connect if not too many points
            try:
                # Sort by first objective for line connection
                sorted_indices = np.argsort(pareto_front[:, i])
                sorted_front_i = pareto_front[sorted_indices, i]
                sorted_front_j = pareto_front[sorted_indices, j]
                ax.plot(sorted_front_i, sorted_front_j, 'r-', alpha=0.5, linewidth=1)
            except:
                pass  # Skip line if it causes issues
        
        # Add target lines if available
        if self.env and hasattr(self.env, 'design_criteria'):
            targets = self.env.design_criteria
            param_i = self.env.optimise_parameters[i] if hasattr(self.env, 'optimise_parameters') else None
            param_j = self.env.optimise_parameters[j] if hasattr(self.env, 'optimise_parameters') else None
            
            if param_i in targets:
                target_val = targets[param_i]
                if param_i == 'reflectivity':
                    target_val = 1 - target_val  # Convert to 1-R
                ax.axvline(target_val, color='green', linestyle='--', alpha=0.7, linewidth=1)
            
            if param_j in targets:
                target_val = targets[param_j]
                if param_j == 'reflectivity':
                    target_val = 1 - target_val  # Convert to 1-R
                ax.axhline(target_val, color='green', linestyle='--', alpha=0.7, linewidth=1)
        
        # Set labels and scales
        ax.set_xlabel(obj_labels[i] if i < len(obj_labels) else f'Objective {i+1}')
        ax.set_ylabel(obj_labels[j] if j < len(obj_labels) else f'Objective {j+1}')
        
        if i < len(obj_scales) and obj_scales[i] == 'log':
            ax.set_xscale('log')
        if j < len(obj_scales) and obj_scales[j] == 'log':
            ax.set_yscale('log')
        
        ax.set_title(f'Episode {episode}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set up click handler for main plot only
        if enable_click:
            def on_click(event):
                if event.inaxes == ax and len(pareto_front) > 0:
                    if event.xdata is not None and event.ydata is not None:
                        # Find closest point
                        click_x, click_y = event.xdata, event.ydata
                        
                        # Handle log scale distances
                        if obj_scales[i] == 'log' and obj_scales[j] == 'log':
                            log_click_x = np.log10(max(click_x, 1e-10))
                            log_click_y = np.log10(max(click_y, 1e-10))
                            log_front_x = np.log10(np.maximum(pareto_front[:, i], 1e-10))
                            log_front_y = np.log10(np.maximum(pareto_front[:, j], 1e-10))
                            distances = np.sqrt((log_front_x - log_click_x)**2 + (log_front_y - log_click_y)**2)
                        else:
                            distances = np.sqrt((pareto_front[:, i] - click_x)**2 + (pareto_front[:, j] - click_y)**2)
                        
                        closest_idx = np.argmin(distances)
                        self.plot_coating_stack(closest_idx, pareto_front[closest_idx])
            
            # Disconnect any existing handler and connect new one
            if self.click_event_connection is not None:
                self.pareto_canvas.mpl_disconnect(self.click_event_connection)
            self.click_event_connection = self.pareto_canvas.mpl_connect('button_press_event', on_click)

    def plot_parallel_coordinates(self, pareto_front, best_points, obj_labels, episode):
        """Plot parallel coordinates visualization for high-dimensional data."""
        ax = self.pareto_fig.add_subplot(1, 1, 1)
        
        n_objectives = pareto_front.shape[1]
        
        # Normalize data for parallel coordinates
        pareto_normalized = np.zeros_like(pareto_front)
        for i in range(n_objectives):
            col_min, col_max = pareto_front[:, i].min(), pareto_front[:, i].max()
            if col_max > col_min:
                pareto_normalized[:, i] = (pareto_front[:, i] - col_min) / (col_max - col_min)
            else:
                pareto_normalized[:, i] = 0.5
        
        # Plot background points if available
        if best_points is not None and len(best_points) > 0:
            try:
                best_points_array = np.array(best_points)
                if best_points_array.shape[1] == n_objectives:
                    best_normalized = np.zeros_like(best_points_array)
                    for i in range(n_objectives):
                        col_min, col_max = best_points_array[:, i].min(), best_points_array[:, i].max()
                        if col_max > col_min:
                            best_normalized[:, i] = (best_points_array[:, i] - col_min) / (col_max - col_min)
                        else:
                            best_normalized[:, i] = 0.5
                    
                    # Plot background lines
                    for idx in range(len(best_normalized)):
                        ax.plot(range(n_objectives), best_normalized[idx], 'b-', alpha=0.1, linewidth=0.5)
            except Exception as e:
                print(f"Error plotting background parallel coords: {e}")
        
        # Plot Pareto front lines
        for idx in range(len(pareto_normalized)):
            ax.plot(range(n_objectives), pareto_normalized[idx], 'r-', alpha=0.7, linewidth=2)
        
        # Customize plot
        ax.set_xlim(-0.5, n_objectives - 0.5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks(range(n_objectives))
        ax.set_xticklabels([label.replace(' ', '\n') for label in obj_labels], rotation=0, fontsize=10)
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'Parallel Coordinates - Episode {episode}\n{len(pareto_front)} Pareto Points')
        ax.grid(True, alpha=0.3)

    def plot_3d_scatter(self, pareto_front, best_points, obj_labels, obj_scales, episode):
        """Plot 3D scatter plot for 3+ dimensional data (shows first 3 dimensions)."""
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Warning: mpl_toolkits.mplot3d not available, 3D plots will be disabled")
        
        ax = self.pareto_fig.add_subplot(1, 1, 1, projection='3d')
        
        # Use first 3 dimensions
        if pareto_front.shape[1] >= 3:
            # Plot background points
            if best_points is not None and len(best_points) > 0:
                try:
                    best_points_array = np.array(best_points)
                    if best_points_array.shape[1] >= 3:
                        ax.scatter(best_points_array[:, 0], best_points_array[:, 1], best_points_array[:, 2],
                                 c='lightblue', alpha=0.3, s=10, label='All Solutions')
                except Exception as e:
                    print(f"Error plotting 3D background: {e}")
            
            # Plot Pareto front
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                      c='red', s=40, alpha=0.8, label='Pareto Front', edgecolors='black')
            
            # Set labels and scales
            ax.set_xlabel(obj_labels[0] if len(obj_labels) > 0 else 'Objective 1')
            ax.set_ylabel(obj_labels[1] if len(obj_labels) > 1 else 'Objective 2') 
            ax.set_zlabel(obj_labels[2] if len(obj_labels) > 2 else 'Objective 3')
            
            # Handle log scales (3D log scale is tricky, so we note it in title)
            log_note = ""
            if len(obj_scales) >= 3:
                log_axes = [i for i, scale in enumerate(obj_scales[:3]) if scale == 'log']
                if log_axes:
                    log_note = f" (Log scale: {[obj_labels[i] for i in log_axes]})"
            
            ax.set_title(f'3D Pareto Front - Episode {episode}{log_note}')
            ax.legend()
        
        else:
            ax.text(0.5, 0.5, 0.5, 'Need at least 3 objectives for 3D visualization', 
                   transform=ax.transAxes, ha='center', va='center')

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

    def save_plots_to_disk(self):
        """Save all current plots to disk in the root directory."""
        try:
            if not hasattr(self, 'config') or not self.config:
                return
                
            root_dir = self.config.general.root_dir
            if not root_dir or not os.path.exists(root_dir):
                return
                
            # Save Pareto plot
            pareto_path = os.path.join(root_dir, "pareto_plot.png")
            self.pareto_fig.savefig(pareto_path, dpi=150, bbox_inches='tight')
            
            # Save rewards plot
            rewards_path = os.path.join(root_dir, "rewards_plot.png")
            self.rewards_fig.savefig(rewards_path, dpi=150, bbox_inches='tight')
            
            # Save values plot
            values_path = os.path.join(root_dir, "values_plot.png")
            self.values_fig.savefig(values_path, dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"Warning: Failed to save plots to disk: {e}")

    def show_training_summary(self):
        """Show comprehensive training summary including checkpoint information."""
        if not self.trainer:
            messagebox.showwarning("No Training Data", "No trainer instance available.")
            return
            
       
            
        try:
            # Get training summary from the trainer
            if hasattr(self.trainer, 'get_training_summary'):
                summary = self.trainer.get_training_summary()
                
                summary_text = "Training Summary\n" + "="*50 + "\n\n"
                
                # Basic training info
                summary_text += f"Checkpoint Type: {summary.get('checkpoint_type', 'unknown')}\n"
                summary_text += f"Current Episode: {summary.get('current_episode', 0)}\n"
                summary_text += f"Target Episodes: {summary.get('target_episodes', 0)}\n"
                summary_text += f"Completion: {summary.get('completion_percent', 0)}%\n"
                summary_text += f"Metrics Recorded: {summary.get('metrics_count', 0)}\n"
                summary_text += f"Best States: {summary.get('best_states_count', 0)}\n"
                
                # Pareto front info
                if 'pareto_front_size' in summary:
                    summary_text += f"\nPareto Front Size: {summary['pareto_front_size']}\n"
                if 'total_points_explored' in summary:
                    summary_text += f"Total Points Explored: {summary['total_points_explored']}\n"
                if 'objectives_count' in summary:
                    summary_text += f"Objectives Count: {summary['objectives_count']}\n"
                
                # Checkpoint info
                if 'checkpoint_info' in summary:
                    checkpoint_info = summary['checkpoint_info']
                    summary_text += f"\nCheckpoint Information:\n"
                    summary_text += f"  Exists: {checkpoint_info.get('exists', False)}\n"
                    if checkpoint_info.get('exists', False):
                        summary_text += f"  Size: {checkpoint_info.get('size_mb', 0):.2f} MB\n"
                        summary_text += f"  Groups: {checkpoint_info.get('groups', [])}\n"
                        if 'last_updated' in checkpoint_info:
                            summary_text += f"  Last Updated: {checkpoint_info['last_updated']}\n"
                
                # Show summary in dialog
                dialog = tk.Toplevel(self.root)
                dialog.title("Training Summary")
                dialog.geometry("500x400")
                
                text_widget = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
                text_widget.pack(fill=tk.BOTH, expand=True)
                text_widget.insert(tk.END, summary_text)
                text_widget.config(state=tk.DISABLED)  # Make read-only
                
                # Add close button
                close_button = ttk.Button(dialog, text="Close", command=dialog.destroy)
                close_button.pack(pady=10)
                
            else:
                messagebox.showinfo("Training Summary", "Training summary not available for this trainer version.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get training summary: {str(e)}")


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