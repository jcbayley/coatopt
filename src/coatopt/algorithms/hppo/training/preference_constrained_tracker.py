"""
Preference-constrained training tracker for managing reward bounds and constraints.
Integrates with existing TrainingCheckpointManager for persistence.
"""
import numpy as np
from typing import Dict, List, Optional, Any


class PreferenceConstrainedTracker:
    """
    Tracks reward bounds during preference-constrained training.
    
    Phase 1: Individual objective exploration to discover min/max reward bounds
    Phase 2: Constrained optimization with progressive constraint tightening
    
    Integrates with existing checkpoint system - no separate save/load methods needed.
    """
    
    def __init__(self, 
                 optimise_parameters: List[str],
                 phase1_epochs_per_objective: int = 1000,
                 phase2_epochs_per_step: int = 300,
                 constraint_steps: int = 8,
                 constraint_penalty_weight: float = 50.0,
                 constraint_margin: float = 0.05):
        """
        Initialize preference-constrained tracker.
        
        Args:
            optimise_parameters: List of objective parameters to optimize
            phase1_epochs_per_objective: Epochs to spend on each objective in Phase 1
            phase2_epochs_per_step: Epochs per constraint step in Phase 2
            constraint_steps: Number of progressive constraint levels
            constraint_penalty_weight: Weight for constraint violation penalties
            constraint_margin: Safety margin above minimum values (fraction of range)
        """
        self.optimise_parameters = optimise_parameters
        self.n_objectives = len(optimise_parameters)
        
        # Phase configuration
        self.phase1_epochs_per_objective = phase1_epochs_per_objective
        self.phase2_epochs_per_step = phase2_epochs_per_step
        self.constraint_steps = constraint_steps
        self.constraint_penalty_weight = constraint_penalty_weight
        self.constraint_margin = constraint_margin
        
        # Phase 1 duration
        self.phase1_total_epochs = self.n_objectives * phase1_epochs_per_objective
        
        # Phase 2 duration per objective cycle
        self.phase2_cycle_epochs = self.n_objectives * phase2_epochs_per_step
        
        # Reward bounds tracking
        self.reward_bounds: Dict[str, Dict[str, float]] = {
            param: {"min": float('inf'), "max": float('-inf')} 
            for param in optimise_parameters
        }
        
        # Phase tracking
        self.phase1_completed = False
        self.current_epoch = 0
        self.current_phase = 1
    
    def update_reward_bounds(self, rewards: Dict[str, float]) -> None:
        """Update reward bounds with new reward observations."""
        for param in self.optimise_parameters:
            if param in rewards:
                reward_val = rewards[param]
                if reward_val < self.reward_bounds[param]["min"]:
                    self.reward_bounds[param]["min"] = reward_val
                if reward_val > self.reward_bounds[param]["max"]:
                    self.reward_bounds[param]["max"] = reward_val
    
    def get_training_phase_info(self, epoch: int) -> Dict[str, Any]:
        """Get current training phase information."""
        self.current_epoch = epoch
        
        if epoch < self.phase1_total_epochs:
            return self._get_phase1_info(epoch)
        else:
            if not self.phase1_completed:
                self._finalize_phase1()
            return self._get_phase2_info(epoch)
    
    def _get_phase1_info(self, epoch: int) -> Dict[str, Any]:
        """Get Phase 1 training information."""
        self.current_phase = 1
        
        # Determine which objective to focus on
        objective_index = (epoch // self.phase1_epochs_per_objective) % self.n_objectives
        target_objective = self.optimise_parameters[objective_index]
        
        # Create one-hot weights
        weights = {param: 0.0 for param in self.optimise_parameters}
        weights[target_objective] = 1.0
        
        return {
            "phase": 1,
            "target_objective": target_objective,
            "weights": weights,
            "constraints": {},
            "phase1_completed": False
        }
    
    def _get_phase2_info(self, epoch: int) -> Dict[str, Any]:
        """Get Phase 2 training information."""
        self.current_phase = 2
        
        # Adjust epoch for Phase 2
        phase2_epoch = epoch - self.phase1_total_epochs
        
        # Determine constraint step (increases over time)
        constraint_step = min(
            (phase2_epoch // self.phase2_cycle_epochs) % self.constraint_steps,
            self.constraint_steps - 1
        )
        
        # Which objective cycle within this constraint step?
        epoch_in_step = phase2_epoch % self.phase2_cycle_epochs
        objective_index = (epoch_in_step // self.phase2_epochs_per_step) % self.n_objectives
        target_objective = self.optimise_parameters[objective_index]
        
        # Create one-hot weights for target objective
        weights = {param: 0.0 for param in self.optimise_parameters}
        weights[target_objective] = 1.0
        
        # Generate constraints for other objectives
        constraints = self._get_progressive_constraints(constraint_step, target_objective)
        
        return {
            "phase": 2,
            "target_objective": target_objective,
            "weights": weights,
            "constraints": constraints,
            "constraint_step": constraint_step,
            "phase1_completed": True
        }
    
    def _finalize_phase1(self) -> None:
        """Finalize Phase 1 and prepare for Phase 2."""
        self.phase1_completed = True
        
        # Validate bounds
        for param in self.optimise_parameters:
            bounds = self.reward_bounds[param]
            if bounds["min"] == float('inf') or bounds["max"] == float('-inf'):
                print(f"Warning: No valid reward bounds for {param}, using defaults")
                bounds["min"] = 0.0
                bounds["max"] = 1.0
            elif bounds["min"] == bounds["max"]:
                print(f"Warning: Equal bounds for {param}: {bounds['min']}")
                margin = abs(bounds["min"]) * 0.1 if bounds["min"] != 0 else 0.1
                bounds["max"] = bounds["min"] + margin
        
        print(f"Phase 1 completed. Reward bounds: {self.reward_bounds}")
    
    def _get_progressive_constraints(self, constraint_step: int, target_objective: str) -> Dict[str, float]:
        """Generate progressive constraints for non-target objectives."""
        constraints = {}
        
        # Calculate constraint progress (0.0 = minimum, 1.0 = maximum)
        constraint_progress = constraint_step / max(1, self.constraint_steps - 1)
        
        for param in self.optimise_parameters:
            if param != target_objective:
                bounds = self.reward_bounds[param]
                min_val = bounds["min"]
                max_val = bounds["max"]
                
                # Add margin above minimum
                range_size = max_val - min_val
                min_with_margin = min_val + self.constraint_margin * range_size
                
                # Progressive constraint: start near minimum, move towards maximum
                constraint_threshold = min_with_margin + constraint_progress * (max_val - min_with_margin)
                constraints[param] = constraint_threshold
        
        return constraints
    
    def apply_constraint_penalties(self, rewards: Dict[str, float], 
                                  constraints: Dict[str, float]) -> float:
        """
        Apply constraint penalties to rewards.
        Returns total penalty (to be subtracted from reward).
        """
        total_penalty = 0.0
        
        for param, threshold in constraints.items():
            if param in rewards:
                current_reward = rewards[param]
                if current_reward < threshold:
                    violation = abs(current_reward - threshold)
                    penalty = self.constraint_penalty_weight #* violation
                    total_penalty += penalty
        
        return total_penalty
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data for saving to checkpoint (integrates with existing system)."""
        return {
            "optimise_parameters": self.optimise_parameters,
            "phase1_epochs_per_objective": self.phase1_epochs_per_objective,
            "phase2_epochs_per_step": self.phase2_epochs_per_step,
            "constraint_steps": self.constraint_steps,
            "constraint_penalty_weight": self.constraint_penalty_weight,
            "constraint_margin": self.constraint_margin,
            "reward_bounds": self.reward_bounds,
            "phase1_completed": self.phase1_completed,
            "current_epoch": self.current_epoch,
            "current_phase": self.current_phase
        }
    
    def load_from_checkpoint_data(self, data: Dict[str, Any]) -> bool:
        """Load from checkpoint data (integrates with existing system)."""
        try:
            self.optimise_parameters = data["optimise_parameters"]
            self.phase1_epochs_per_objective = data["phase1_epochs_per_objective"]
            self.phase2_epochs_per_step = data["phase2_epochs_per_step"]
            self.constraint_steps = data["constraint_steps"]
            self.constraint_penalty_weight = data["constraint_penalty_weight"]
            self.constraint_margin = data["constraint_margin"]
            self.reward_bounds = data["reward_bounds"]
            self.phase1_completed = data["phase1_completed"]
            self.current_epoch = data["current_epoch"]
            self.current_phase = data["current_phase"]
            
            # Recalculate derived values
            self.n_objectives = len(self.optimise_parameters)
            self.phase1_total_epochs = self.n_objectives * self.phase1_epochs_per_objective
            self.phase2_cycle_epochs = self.n_objectives * self.phase2_epochs_per_step
            
            return True
            
        except Exception as e:
            print(f"Error loading preference-constrained tracker: {e}")
            return False