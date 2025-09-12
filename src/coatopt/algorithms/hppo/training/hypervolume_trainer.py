"""
Hypervolume-based multi-objective training for HPPO.
Implements direct hypervolume optimization as an alternative to weighted scalarization.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import warnings

from .trainer import UnifiedHPPOTrainer
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def compute_hypervolume_2d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute 2D hypervolume using efficient rectangle-based method.
    
    Args:
        points: Array of points (n_points, 2)
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if len(points) == 0:
        return 0.0
    
    # Ensure points dominate reference point
    dominated_points = points[np.all(points <= reference_point, axis=1)]
    if len(dominated_points) == 0:
        return 0.0
    
    # Sort by first objective
    sorted_points = dominated_points[np.argsort(dominated_points[:, 0])]
    
    # Calculate hypervolume using sweep line algorithm
    hv = 0.0
    prev_x = reference_point[0]
    
    for point in sorted_points:
        if point[1] < prev_x:  # Only consider improving points
            width = prev_x - point[0]
            height = reference_point[1] - point[1]
            hv += width * height
            prev_x = point[0]
    
    return hv


def compute_hypervolume_3d(points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute 3D hypervolume using inclusion-exclusion principle.
    
    Args:
        points: Array of points (n_points, 3)
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if len(points) == 0:
        return 0.0
    
    # Ensure points dominate reference point
    dominated_points = points[np.all(points <= reference_point, axis=1)]
    if len(dominated_points) == 0:
        return 0.0
    
    # For simplicity, use Monte Carlo approximation for 3D+
    return monte_carlo_hypervolume(dominated_points, reference_point)


def monte_carlo_hypervolume(points: np.ndarray, reference_point: np.ndarray, 
                          n_samples: int = 10000) -> float:
    """
    Estimate hypervolume using Monte Carlo sampling.
    
    Args:
        points: Array of points (n_points, n_dims)
        reference_point: Reference point for hypervolume calculation
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Estimated hypervolume value
    """
    if len(points) == 0:
        return 0.0
    
    n_dims = points.shape[1]
    
    # Define bounding box
    min_bounds = np.min(points, axis=0)
    max_bounds = reference_point
    
    # Generate random samples in bounding box
    samples = np.random.uniform(min_bounds, max_bounds, (n_samples, n_dims))
    
    # Count samples dominated by at least one Pareto point
    dominated_count = 0
    for sample in samples:
        # Check if sample is dominated by any Pareto point
        dominated = np.any(np.all(points <= sample, axis=1))
        if dominated:
            dominated_count += 1
    
    # Estimate hypervolume
    box_volume = np.prod(max_bounds - min_bounds)
    hypervolume = box_volume * (dominated_count / n_samples)
    
    return hypervolume


def compute_hypervolume_contribution(points: np.ndarray, reference_point: np.ndarray,
                                   point_idx: int) -> float:
    """
    Compute the hypervolume contribution of a specific point.
    
    Args:
        points: Array of all Pareto points
        reference_point: Reference point
        point_idx: Index of point to compute contribution for
        
    Returns:
        Hypervolume contribution of the point
    """
    if len(points) <= 1:
        return compute_hypervolume_nd(points, reference_point)
    
    # Compute hypervolume with and without the point
    all_points_hv = compute_hypervolume_nd(points, reference_point)
    
    # Remove the point
    other_points = np.delete(points, point_idx, axis=0)
    other_points_hv = compute_hypervolume_nd(other_points, reference_point)
    
    return all_points_hv - other_points_hv


def compute_hypervolume_nd(points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume for arbitrary dimensions.
    
    Args:
        points: Array of points (n_points, n_dims)
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if len(points) == 0:
        return 0.0
    
    n_dims = points.shape[1]
    
    if n_dims == 2:
        return compute_hypervolume_2d(points, reference_point)
    elif n_dims == 3:
        return compute_hypervolume_3d(points, reference_point)
    else:
        # Use Monte Carlo for 4+ dimensions
        return monte_carlo_hypervolume(points, reference_point)


class HypervolumeLoss(torch.nn.Module):
    """
    Custom loss function for direct hypervolume optimization.
    """
    
    def __init__(self, reference_point: np.ndarray, contribution_weight: float = 1.0):
        super().__init__()
        self.reference_point = torch.tensor(reference_point, dtype=torch.float32)
        self.contribution_weight = contribution_weight
    
    def forward(self, predicted_objectives: torch.Tensor, 
                current_pareto_front: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute hypervolume-based loss.
        
        Args:
            predicted_objectives: Batch of predicted objective values (batch_size, n_objectives)
            current_pareto_front: Current Pareto front points (optional)
            
        Returns:
            Loss tensor
        """
        batch_size = predicted_objectives.shape[0]
        
        if current_pareto_front is None or len(current_pareto_front) == 0:
            # No existing front, maximize distance from reference point
            distances = torch.norm(predicted_objectives - self.reference_point, dim=1)
            return -torch.mean(distances)  # Negative for maximization
        
        # Compute approximate hypervolume contribution
        contributions = []
        
        for i in range(batch_size):
            point = predicted_objectives[i]
            
            # Simple approximation: distance to reference point weighted by novelty
            ref_distance = torch.norm(point - self.reference_point)
            
            # Novelty: minimum distance to existing Pareto points
            if len(current_pareto_front) > 0:
                pareto_distances = torch.norm(current_pareto_front - point.unsqueeze(0), dim=1)
                novelty = torch.min(pareto_distances)
            else:
                novelty = torch.tensor(1.0)
            
            # Combine distance and novelty
            contribution = ref_distance * (1.0 + novelty)
            contributions.append(contribution)
        
        contributions = torch.stack(contributions)
        
        # Return negative for maximization (since we minimize loss)
        return -torch.mean(contributions)


class HypervolumeTrainer(UnifiedHPPOTrainer):
    """
    Extended HPPO trainer with direct hypervolume optimization capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract hypervolume-specific parameters
        self.use_hypervolume_loss = kwargs.pop('use_hypervolume_loss', False)
        self.hv_loss_weight = kwargs.pop('hv_loss_weight', 0.5)
        self.hv_update_interval = kwargs.pop('hv_update_interval', 10)
        self.adaptive_reference_point = kwargs.pop('adaptive_reference_point', True)
        
        super().__init__(*args, **kwargs)
        
        # Initialize hypervolume tracking
        self.hypervolume_history = []
        self.reference_point_history = []
        self.hv_loss_fn = None
        self.steps_since_hv_update = 0
        
        # Initialize reference point
        self._initialize_reference_point()
    
    def _initialize_reference_point(self):
        """Initialize reference point for hypervolume calculation."""
        n_objectives = len(self.env.optimise_parameters)
        
        # Start with a conservative reference point
        if hasattr(self.env, 'reference_point') and len(self.env.reference_point) > 0:
            self.reference_point = self.env.reference_point.copy()
        else:
            # Default reference point (assuming minimization objectives)
            self.reference_point = np.ones(n_objectives) * 2.0
        
        # Initialize hypervolume loss function
        if self.use_hypervolume_loss:
            self.hv_loss_fn = HypervolumeLoss(self.reference_point, self.hv_loss_weight)
    
    def _update_reference_point(self):
        """Adaptively update reference point based on current Pareto front."""
        if not self.adaptive_reference_point:
            return
        
        if hasattr(self.env, 'pareto_front') and len(self.env.pareto_front) > 0:
            pareto_front = np.array(self.env.pareto_front)
            
            # Set reference point to be slightly worse than the worst point in each objective
            margins = np.std(pareto_front, axis=0) * 0.5  # Adaptive margin
            new_reference = np.max(pareto_front, axis=0) + margins
            
            # Only update if significantly different
            if np.linalg.norm(new_reference - self.reference_point) > 0.1:
                self.reference_point = new_reference
                
                # Update loss function
                if self.hv_loss_fn is not None:
                    self.hv_loss_fn.reference_point = torch.tensor(
                        self.reference_point, dtype=torch.float32
                    )
                
                self.reference_point_history.append(self.reference_point.copy())
    
    def _compute_current_hypervolume(self) -> float:
        """Compute hypervolume of current Pareto front."""
        if not hasattr(self.env, 'pareto_front') or len(self.env.pareto_front) == 0:
            return 0.0
        
        pareto_front = np.array(self.env.pareto_front)
        return compute_hypervolume_nd(pareto_front, self.reference_point)
    
    def _compute_hypervolume_rewards(self, objective_values: np.ndarray) -> np.ndarray:
        """
        Compute rewards based on hypervolume contribution.
        
        Args:
            objective_values: Array of objective values (batch_size, n_objectives)
            
        Returns:
            Array of hypervolume-based rewards
        """
        if len(objective_values) == 0:
            return np.array([])
        
        current_pareto = np.array(self.env.pareto_front) if hasattr(self.env, 'pareto_front') else np.empty((0, len(self.env.optimise_parameters)))
        
        hv_rewards = []
        
        for obj_vals in objective_values:
            # Compute hypervolume contribution if this point were added
            if len(current_pareto) == 0:
                # First point contributes the full dominated hypervolume
                hv_contrib = compute_hypervolume_nd(
                    obj_vals.reshape(1, -1), self.reference_point
                )
            else:
                # Add point and compute incremental hypervolume
                extended_front = np.vstack([current_pareto, obj_vals])
                
                # Get non-dominated subset
                nds = NonDominatedSorting()
                fronts = nds.do(extended_front)
                non_dominated_indices = fronts[0] if len(fronts) > 0 else []
                non_dominated_front = extended_front[non_dominated_indices]
                
                # Check if new point is in non-dominated set
                new_point_idx = len(current_pareto)
                if new_point_idx in non_dominated_indices:
                    # Point contributes to Pareto front
                    new_hv = compute_hypervolume_nd(non_dominated_front, self.reference_point)
                    old_hv = compute_hypervolume_nd(current_pareto, self.reference_point)
                    hv_contrib = max(0, new_hv - old_hv)
                else:
                    # Point is dominated
                    hv_contrib = 0.0
            
            hv_rewards.append(hv_contrib)
        
        return np.array(hv_rewards)
    
    def train_single_iteration(self, episode: int) -> Dict[str, Any]:
        """
        Train single iteration with hypervolume-based enhancements.
        
        Args:
            episode: Current episode number
            
        Returns:
            Training metrics dictionary
        """
        # Update reference point periodically
        self.steps_since_hv_update += 1
        if self.steps_since_hv_update >= self.hv_update_interval:
            self._update_reference_point()
            self.steps_since_hv_update = 0
        
        # Run standard training iteration
        metrics = super().train_single_iteration(episode)
        
        # Add hypervolume metrics
        current_hv = self._compute_current_hypervolume()
        self.hypervolume_history.append(current_hv)
        
        metrics.update({
            'hypervolume': current_hv,
            'reference_point': self.reference_point.copy(),
            'pareto_front_size': len(self.env.pareto_front) if hasattr(self.env, 'pareto_front') else 0
        })
        
        return metrics
    
    def compute_enhanced_rewards(self, states: List, actions: List, 
                               rewards: List, objective_values: List) -> List:
        """
        Compute enhanced rewards incorporating hypervolume contributions.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of original rewards
            objective_values: List of objective value arrays
            
        Returns:
            List of enhanced rewards
        """
        if not self.use_hypervolume_loss or len(objective_values) == 0:
            return rewards
        
        # Compute hypervolume-based rewards
        obj_vals_array = np.array(objective_values)
        hv_rewards = self._compute_hypervolume_rewards(obj_vals_array)
        
        # Combine with original rewards
        enhanced_rewards = []
        for i, (orig_reward, hv_reward) in enumerate(zip(rewards, hv_rewards)):
            # Weighted combination
            enhanced_reward = (1 - self.hv_loss_weight) * orig_reward + self.hv_loss_weight * hv_reward
            enhanced_rewards.append(enhanced_reward)
        
        return enhanced_rewards
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get extended training statistics including hypervolume metrics."""
        stats = super().get_training_stats() if hasattr(super(), 'get_training_stats') else {}
        
        if self.hypervolume_history:
            stats.update({
                'hypervolume_current': self.hypervolume_history[-1],
                'hypervolume_max': max(self.hypervolume_history),
                'hypervolume_improvement': (
                    self.hypervolume_history[-1] - self.hypervolume_history[0]
                    if len(self.hypervolume_history) > 1 else 0
                ),
                'reference_point_current': self.reference_point.tolist()
            })
        
        return stats
