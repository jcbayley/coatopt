"""
Training components for HPPO algorithm.
"""

from .trainer import UnifiedHPPOTrainer, TrainingCallbacks, HPPOTrainer, EnhancedHPPOTrainer
from .consolidation import ConsolidationStrategy, ConsolidationConfig
from .checkpoint_manager import TrainingCheckpointManager

__all__ = [
    'UnifiedHPPOTrainer', 'TrainingCallbacks',
    'ConsolidationStrategy', 'ConsolidationConfig', 
    'TrainingCheckpointManager',
    'HPPOTrainer', 'EnhancedHPPOTrainer'  # Backward compatibility
]
