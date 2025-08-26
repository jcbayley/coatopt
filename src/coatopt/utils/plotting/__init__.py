"""
Unified plotting interface for coating optimization.

This package provides a consistent plotting API that consolidates all
plotting functionality from across the codebase.
"""

# Main plotting manager (most commonly used)
from .core import TrainingPlotManager

# Specific plotting functions
from .training import (
    pad_lists,
    make_reward_plot, 
    make_val_plot,
    make_loss_plot,
    make_materials_plot
)

from .stack import plot_stack, plot_coating

# Analysis tools can be imported directly when needed:
# from coatopt.utils.plotting.analysis import analyze_plotting_functions

# Convenience imports for backward compatibility
__all__ = [
    # Main interface
    'TrainingPlotManager',
    
    # Training plots
    'pad_lists',
    'make_reward_plot',
    'make_val_plot', 
    'make_loss_plot',
    'make_materials_plot',
    
    # Stack visualization
    'plot_stack',
    'plot_coating',
]
