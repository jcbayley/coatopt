"""
Training-specific configuration classes for HPPO.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration class for HPPO training parameters."""
    
    # Training parameters
    n_iterations: int = 10000
    n_epochs_per_update: int = 4
    n_layers: int = 20
    
    # Entropy parameters
    entropy_beta_start: float = 0.1
    entropy_beta_end: float = 0.01
    entropy_beta_decay_length: int = 1000
    entropy_beta_decay_start: int = 0
    
    # Learning rate scheduling
    scheduler_start: int = 1000
    scheduler_end: int = 10000
    
    # Saving parameters
    weight_network_save: int = 100
    
    # Observation/state settings
    use_obs: bool = False
    
    # Consolidation settings (optional)
    use_consolidation: bool = False
    consolidation_bc_weight: float = 0.05
    consolidation_max_solutions: int = 100
    consolidation_interval: int = 50
    consolidation_min_solutions: int = 10
    consolidation_samples_per_obj: int = 5
    consolidation_percentile_threshold: float = 75.0


@dataclass 
class CallbackConfig:
    """Configuration for training callbacks."""
    
    save_plots: bool = False
    save_visualizations: bool = False
    progress_interval: int = 10
    summary_interval: int = 100


@dataclass
class HPPOTrainingConfig:
    """Combined configuration for HPPO training."""
    
    training: TrainingConfig
    callbacks: CallbackConfig
    
    def __init__(self, **kwargs):
        # Extract training config parameters
        training_params = {}
        callback_params = {}
        
        for key, value in kwargs.items():
            if key.startswith('consolidation_') or key in ['n_iterations', 'n_epochs_per_update', 'n_layers',
                                                          'entropy_beta_start', 'entropy_beta_end', 
                                                          'entropy_beta_decay_length', 'entropy_beta_decay_start',
                                                          'scheduler_start', 'scheduler_end', 'weight_network_save',
                                                          'use_obs', 'use_consolidation']:
                training_params[key] = value
            elif key in ['save_plots', 'save_visualizations', 'progress_interval', 'summary_interval']:
                callback_params[key] = value
                
        self.training = TrainingConfig(**training_params)
        self.callbacks = CallbackConfig(**callback_params)
