"""
Shared plotting utilities for both CLI and UI training visualization.

This module provides consistent plotting functionality that can be used
by both the command-line interface and the GUI interface.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from typing import List, Dict, Any, Optional, Tuple, Union


class TrainingPlotManager:
    """Manager for training plots that can be used by both CLI and UI."""
    
    def __init__(self, save_plots: bool = True, output_dir: str = None, 
                 ui_mode: bool = False, figure_size: Tuple[int, int] = (12, 10)):
        """
        Initialize plot manager.
        
        Args:
            save_plots: Whether to save plots to disk
            output_dir: Directory to save plots (required if save_plots=True)
            ui_mode: Whether running in UI mode (affects backend and display)
            figure_size: Size of figures (width, height)
        """
        self.save_plots = save_plots
        self.output_dir = output_dir
        self.ui_mode = ui_mode
        self.figure_size = figure_size
        
        # Data storage
        self.training_data = []
        self.pareto_data = []
        self.eval_data = {}
        self.historical_pareto_data = {}
        
        # Objective information
        self.objective_labels = []
        self.objective_scales = []
        self.optimisation_parameters = []
        self.objective_targets = {}
        self.design_criteria = {}
        
        # Plot update throttling
        self.last_plot_update = 0
        self.plot_update_interval = 40
        
        # Initialize figures
        self._init_figures()
    
    def _init_figures(self):
        """Initialize matplotlib figures."""
        if not self.ui_mode:
            # For CLI, use Agg backend for file saving
            matplotlib.use('Agg')
        
        # Create figures
        self.rewards_fig = Figure(figsize=self.figure_size, dpi=100)
        self.values_fig = Figure(figsize=self.figure_size, dpi=100)
        self.pareto_fig = Figure(figsize=(14, 10), dpi=100)
        
        self._init_subplot_layout()
    
    def _init_subplot_layout(self):
        """Initialize subplot layouts."""
        # Rewards plot with multiple subplots
        self.rewards_axes = self.rewards_fig.subplots(3, 2)
        self.rewards_axes = self.rewards_axes.flatten()
        
        rewards_subplot_titles = [
            "Total Reward", "Individual Rewards", "Entropy Weight (β)", 
            "Learning Rates", "Objective Weights", "Training Losses"
        ]
        
        for i, ax in enumerate(self.rewards_axes):
            ax.set_title(rewards_subplot_titles[i] if i < len(rewards_subplot_titles) else f"Plot {i+1}")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Episode")
        
        self.rewards_fig.tight_layout()
        
        # Physical values plot
        self.values_axes = self.values_fig.subplots(2, 2)
        self.values_axes = self.values_axes.flatten()
        
        values_subplot_titles = [
            "Reflectivity", "Thermal Noise", "Absorption", "Thickness"
        ]
        
        for i, ax in enumerate(self.values_axes):
            ax.set_title(values_subplot_titles[i] if i < len(values_subplot_titles) else f"Value {i+1}")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Episode")
        
        self.values_fig.tight_layout()
        
        # Pareto front plot - will be dynamically created
        self._init_pareto_plot()
    
    def _init_pareto_plot(self):
        """Initialize Pareto front plot."""
        self.pareto_fig.clear()
        ax = self.pareto_fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'Load configuration to see Pareto front visualization', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title("Pareto Front Visualization")
        self.pareto_fig.tight_layout()
    
    def set_objective_info(self, optimisation_parameters: List[str], objective_targets: Optional[Dict] = None, design_criteria: Optional[Dict] = None):
        """Set objective information for consistent labeling."""
        self.optimisation_parameters = optimisation_parameters.copy()
        self.objective_targets = objective_targets if objective_targets else {}
        self.design_criteria = design_criteria if design_criteria else {}
        self.objective_labels = self._get_objective_labels()
        self.objective_scales = self._get_objective_scales()
    
    def _get_objective_labels(self):
        """Get human-readable labels for objectives."""
        if not self.optimisation_parameters:
            return []
        
        label_mapping = {
            'reflectivity': '1 - Reflectivity',
            'absorption': 'Absorption [ppm]', 
            'thermal_noise': 'Thermal Noise [m/√Hz]',
            'thickness': 'Total Thickness [nm]'
        }
        
        return [label_mapping.get(param, param.replace('_', ' ').title()) 
                for param in self.optimisation_parameters]
    
    def _get_objective_scales(self):
        """Get appropriate scales (linear/log) for each objective."""
        if not self.optimisation_parameters:
            return []
        
        scale_mapping = {
            'reflectivity': 'log',
            'absorption': 'log', 
            'thermal_noise': 'log',
            'thickness': 'linear'
        }
        
        return [scale_mapping.get(param, 'linear') for param in self.optimisation_parameters]
    
    def add_training_data(self, episode_data: Dict[str, Any]):
        """Add training data for plotting."""
        self.training_data.append(episode_data)
    
    def add_pareto_data(self, pareto_data: Dict[str, Any]):
        """Add Pareto front data for plotting."""
        self.pareto_data.append(pareto_data)
        
        # Store in historical data
        episode = pareto_data.get('episode', 0)
        self.historical_pareto_data[episode] = pareto_data.copy()
    
    def add_eval_pareto_data(self, eval_data: Dict[str, Any]):
        """Add evaluation Pareto data for plotting."""        
        # If running in UI mode, update plots immediately
        self.eval_data = eval_data
    
    def should_update_plots(self, current_episode: int) -> bool:
        """Check if plots should be updated based on throttling."""
        return current_episode - self.last_plot_update >= self.plot_update_interval
    
    def update_all_plots(self, current_episode: int = None):
        """Update all plots."""
        if current_episode and not self.should_update_plots(current_episode):
            return
        
        self.update_rewards_plot()
        self.update_values_plot()
        self.update_pareto_plot()
        
        if current_episode:
            self.last_plot_update = current_episode
        
        if self.save_plots and self.output_dir:
            self.save_plots_to_disk()
    
    def update_rewards_plot(self):
        """Update the comprehensive rewards plot."""
        if not self.training_data:
            return
        
        try:
            # Convert training data to DataFrame
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
            optimise_params = self.optimisation_parameters if self.optimisation_parameters else \
                              ['reflectivity', 'thermal_noise', 'absorption', 'thickness']
            
            reward_components = []
            for param in optimise_params:
                possible_names = [
                    f'{param}_reward',
                    f'{param.replace("thermal_noise", "thermalnoise")}_reward'
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
            
            # 5. Objective Weights
            weight_components = []
            for param in optimise_params:
                possible_names = [
                    f'{param}_reward_weights',
                    f'{param}_weights', 
                    f'{param.replace("_", "")}_reward_weights',
                    f'{param.replace("thermal_noise", "thermalnoise")}_reward_weights'
                ]
                
                for name in possible_names:
                    if name in df.columns:
                        weight_components.append((name, param))
                        break
            
            for weight_comp, param in weight_components:
                label = param.replace('_', ' ').title()
                self.rewards_axes[4].plot(episodes, df[weight_comp], alpha=0.8, label=label)
                        
            self.rewards_axes[4].set_title('Objective Weights')
            self.rewards_axes[4].set_ylabel('Weight')
            self.rewards_axes[4].legend(fontsize=8)
            
            # 6. Training Losses
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
            
            # Set x-labels for all subplots
            for ax in self.rewards_axes:
                ax.set_xlabel('Episode')
            
            self.rewards_fig.tight_layout()
            
        except Exception as e:
            print(f"Error updating rewards plot: {e}")
    
    def update_values_plot(self):
        """Update the physical values plot."""
        if not self.training_data:
            return
        
        try:
            # Convert training data to DataFrame
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
            
            # Physical values components
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
                    
                    # Plot smoothed data
                    if len(values) > window_size:
                        smoothed = pd.Series(values).rolling(window=window_size, center=False).median()
                        self.values_axes[i].plot(episodes, smoothed, linewidth=2, color='red', label='Smoothed')
                        self.values_axes[i].legend(fontsize=8)
                    
                    self.values_axes[i].set_title(title)
                    self.values_axes[i].set_ylabel(title)
                    self.values_axes[i].set_xlabel('Episode')
                    
                    # Use log scale for thermal noise and absorption
                    if use_log:
                        non_zero_values = values[values > 0] if len(values[values > 0]) > 0 else [1e-10]
                        if len(non_zero_values) > 0:
                            self.values_axes[i].set_yscale('log')
            
            self.values_fig.tight_layout()
            
        except Exception as e:
            print(f"Error updating values plot: {e}")
    
    def update_pareto_plot(self, episode: int = None):
        """Update the Pareto front plot."""
        if not self.pareto_data:
            return
        
        try:
            # Clear the figure
            self.pareto_fig.clear()
            
            # Get the latest data or specific episode data
            if episode and episode in self.historical_pareto_data:
                latest_data = self.historical_pareto_data[episode]
            else:
                latest_data = self.pareto_data[-1]


            # Check if data contains actual pareto front
            if 'pareto_front' not in latest_data:
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                size = latest_data.get('pareto_front_size', 0)
                ax.text(0.5, 0.5, f'Pareto front size: {size}\nDetailed data not yet available', 
                       transform=ax.transAxes, ha='center', va='center')
                return
                
            pareto_front = latest_data['pareto_front'] 
            best_points = latest_data.get('best_points', None)
            current_episode = latest_data.get('episode', 'unknown')
            eval_points = self.eval_data.get('eval_points', None)
            
            if pareto_front is None or len(pareto_front) == 0:
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No Pareto front data available yet', 
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Ensure objective labels match data
            if (not self.objective_labels or 
                len(self.objective_labels) != pareto_front.shape[1]):
                self._update_objective_labels_from_data(pareto_front.shape[1])
            
            n_objectives = len(self.objective_labels)
            
            if n_objectives < 2:
                ax = self.pareto_fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'Need at least 2 objectives for Pareto visualization', 
                       transform=ax.transAxes, ha='center', va='center')
                return
            
            # Plot based on number of objectives
            if n_objectives == 2:
                self._plot_2d_pareto(pareto_front, best_points, current_episode, eval_points)
            else:
                self._plot_multi_objective_pareto(pareto_front, best_points, current_episode)
            
            self.pareto_fig.tight_layout()
            
        except Exception as e:
            print(f"Error updating Pareto plot: {e}")
    
    def _update_objective_labels_from_data(self, n_objectives: int):
        """Update objective labels based on data dimensions."""
        if self.optimisation_parameters and len(self.optimisation_parameters) == n_objectives:
            self.objective_labels = self._get_objective_labels()
            self.objective_scales = self._get_objective_scales()
        else:
            # Fallback to generic labels
            self.objective_labels = [f'Objective {i+1}' for i in range(n_objectives)]
            self.objective_scales = ['linear'] * n_objectives
    
    def _plot_2d_pareto(self, pareto_front, best_points, episode, eval_points = None):
        """Plot 2D Pareto front."""
        ax = self.pareto_fig.add_subplot(1, 1, 1)
        
        # Plot background points
        if best_points is not None and len(best_points) > 0:
            try:
                best_points_array = np.array(best_points)
                if len(best_points_array.shape) == 2 and best_points_array.shape[1] >= 2:
                    ax.scatter(best_points_array[:, 0], best_points_array[:, 1],
                             c='lightblue', alpha=0.3, s=10, label='All Solutions')
            except Exception as e:
                print(f"Error plotting background points: {e}")
        
        # Plot Pareto front
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                  c='red', s=40, alpha=0.8, label='Pareto Front',
                  edgecolors='black', linewidths=1)
        
        # add evaluation points as well as training points
        if eval_points is not None and len(eval_points) > 0:
            eval_points_array = np.array(eval_points)
            if eval_points_array.shape[1] >= 2:
                ax.scatter(eval_points_array[:, 0], eval_points_array[:, 1],
                           c='orange', s=10, alpha=0.5, label='Evaluation Points')
        
        # Connect Pareto points if not too many
        if len(pareto_front) < 20:
            try:
                sorted_indices = np.argsort(pareto_front[:, 0])
                sorted_front_x = pareto_front[sorted_indices, 0]
                sorted_front_y = pareto_front[sorted_indices, 1]
                ax.plot(sorted_front_x, sorted_front_y, 'r-', alpha=0.5, linewidth=1)
            except:
                pass
        
        # Add objective target lines if available
        if hasattr(self, 'objective_targets') and self.objective_targets:
            try:
                # Get axis limits for drawing lines across the plot
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Draw vertical line for first objective target
                if len(self.objective_targets) > 0 and self.objective_targets[0] is not None:
                    target_x = self.objective_targets[0]
                    ax.axvline(x=target_x, color='green', linestyle='--', alpha=0.7, 
                              linewidth=2, label=f'Target {self.objective_labels[0] if len(self.objective_labels) > 0 else "Obj 1"}')
                
                # Draw horizontal line for second objective target
                if len(self.objective_targets) > 1 and self.objective_targets[1] is not None:
                    target_y = self.objective_targets[1]
                    ax.axhline(y=target_y, color='orange', linestyle='--', alpha=0.7,
                              linewidth=2, label=f'Target {self.objective_labels[1] if len(self.objective_labels) > 1 else "Obj 2"}')
            except Exception as e:
                print(f"Error plotting objective targets: {e}")
        else:
            print(f"No objective targets set for Pareto plot.")
        
        # Set labels and scales
        ax.set_xlabel(self.objective_labels[0] if len(self.objective_labels) > 0 else 'Objective 1')
        ax.set_ylabel(self.objective_labels[1] if len(self.objective_labels) > 1 else 'Objective 2')
        
        if len(self.objective_scales) > 0 and self.objective_scales[0] == 'log':
            ax.set_xscale('log')
        if len(self.objective_scales) > 1 and self.objective_scales[1] == 'log':
            ax.set_yscale('log')
        
        ax.set_title(f'Pareto Front - Episode {episode}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_multi_objective_pareto(self, pareto_front, best_points, episode):
        """Plot multi-objective Pareto front using 2D pairs."""
        n_objectives = len(self.objective_labels)
        n_pairs = min(4, n_objectives * (n_objectives - 1) // 2)  # Limit to 4 pairs for readability
        
        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        pair_idx = 0
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                if pair_idx >= n_pairs:
                    break
                    
                ax = self.pareto_fig.add_subplot(n_rows, n_cols, pair_idx + 1)
                
                # Plot background points
                if best_points is not None and len(best_points) > 0:
                    try:
                        best_points_array = np.array(best_points)
                        if best_points_array.shape[1] > max(i, j):
                            ax.scatter(best_points_array[:, i], best_points_array[:, j],
                                     c='lightblue', alpha=0.3, s=5, label='All Solutions')
                    except:
                        pass
                
                # Plot Pareto front
                ax.scatter(pareto_front[:, i], pareto_front[:, j],
                          c='red', s=20, alpha=0.8, label='Pareto Front',
                          edgecolors='black', linewidths=0.5)
                
                # Set labels and scales
                ax.set_xlabel(self.objective_labels[i] if i < len(self.objective_labels) else f'Obj {i+1}')
                ax.set_ylabel(self.objective_labels[j] if j < len(self.objective_labels) else f'Obj {j+1}')
                
                if i < len(self.objective_scales) and self.objective_scales[i] == 'log':
                    ax.set_xscale('log')
                if j < len(self.objective_scales) and self.objective_scales[j] == 'log':
                    ax.set_yscale('log')
                
                ax.grid(True, alpha=0.3)
                if pair_idx == 0:  # Only show legend on first subplot
                    ax.legend(fontsize=6)
                
                pair_idx += 1
        
        # Add overall title
        self.pareto_fig.suptitle(f'Multi-Objective Pareto Front - Episode {episode}')
    
    def _plot_metric_with_smoothing(self, ax, df, column, title, window_size):
        """Plot metric with smoothing."""
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
    
    def save_plots_to_disk(self):
        """Save all plots to disk."""
        if not self.save_plots or not self.output_dir:
            return
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save plots
            self.rewards_fig.savefig(os.path.join(self.output_dir, "rewards_plot.png"), 
                                   dpi=150, bbox_inches='tight')
            self.values_fig.savefig(os.path.join(self.output_dir, "values_plot.png"), 
                                  dpi=150, bbox_inches='tight')
            self.pareto_fig.savefig(os.path.join(self.output_dir, "pareto_plot.png"), 
                                  dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"Warning: Failed to save plots to disk: {e}")
    
    def get_figures(self):
        """Get matplotlib figures for UI display."""
        return self.rewards_fig, self.values_fig, self.pareto_fig
    
    def clear_data(self):
        """Clear all stored data."""
        self.training_data.clear()
        self.pareto_data.clear()
        self.historical_pareto_data.clear()
        self.last_plot_update = 0