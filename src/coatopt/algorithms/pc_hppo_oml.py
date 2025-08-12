"""
PC-HPPO-OML: Proximal Constrained Hierarchical Proximal Policy optimisation with Online Meta-Learning
Refactored and cleaned up for improved readability and maintainability.

This module provides imports for backward compatibility while the actual implementations
are now in separate, well-organized modules.
"""

# Import the refactored classes for backward compatibility
from coatopt.algorithms.hppo.replay_buffer import ReplayBuffer
from coatopt.algorithms.hppo.pc_hppo_agent import PCHPPO
from coatopt.algorithms.hppo.hppo_trainer import HPPOTrainer
from coatopt.algorithms.config import HPPOConstants

# Import utility functions
from coatopt.algorithms.plotting_utils import pad_lists

# Re-export for backward compatibility
__all__ = [
    'ReplayBuffer',
    'PCHPPO', 
    'HPPOTrainer',
    'HPPOConstants',
    'pad_lists'
]