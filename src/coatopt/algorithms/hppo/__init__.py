"""
PC-HPPO-OML: Proximal Constrained Hierarchical Proximal Policy optimisation with Online Meta-Learning
Refactored and cleaned up for improved readability and maintainability.

This module provides imports for backward compatibility while the actual implementations
are now in separate, well-organized modules.
"""

from coatopt.algorithms.config import HPPOConstants
from coatopt.algorithms.hppo.core.agent import PCHPPO

# Import the refactored classes for backward compatibility
from coatopt.algorithms.hppo.core.replay_buffer import ReplayBuffer
from coatopt.algorithms.hppo.training.trainer import HPPOTrainer

# Import utility functions
from coatopt.utils.plotting.training import pad_lists

# Re-export for backward compatibility
__all__ = ["ReplayBuffer", "PCHPPO", "HPPOTrainer", "HPPOConstants", "pad_lists"]
