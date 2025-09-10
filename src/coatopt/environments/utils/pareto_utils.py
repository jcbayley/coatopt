"""
Efficient Pareto front utilities for multi-objective optimization.
Contains optimized algorithms for non-dominated sorting and Pareto front maintenance.
"""
import numpy as np
from typing import List, Tuple, Optional
from numba import jit, types
from numba.typed import List as TypedList


@jit(nopython=True)
def _fast_dominates(point1: np.ndarray, point2: np.ndarray) -> bool:
    """
    Fast dominance check using Numba JIT compilation.
    
    Args:
        point1: First point (assumes maximization)
        point2: Second point
        
    Returns:
        True if point1 dominates point2
    """
    better_in_at_least_one = False
    
    for i in range(len(point1)):
        if point1[i] < point2[i]:  # point1 is worse in this objective
            return False
        elif point1[i] > point2[i]:  # point1 is better in this objective
            better_in_at_least_one = True
    
    return better_in_at_least_one


@jit(nopython=True)  
def _fast_is_dominated_by_any(point: np.ndarray, pareto_front: np.ndarray) -> bool:
    """
    Fast check if point is dominated by any point in the current front.
    
    Args:
        point: Point to check
        pareto_front: Current Pareto front points
        
    Returns:
        True if point is dominated by any front point
    """
    for i in range(pareto_front.shape[0]):
        if _fast_dominates(pareto_front[i], point):
            return True
    return False


def incremental_pareto_update(current_front: np.ndarray, new_points: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Efficiently update Pareto front with new points using incremental approach.
    
    This is much faster than full recomputation for small numbers of new points.
    
    Args:
        current_front: Current Pareto front (n_current, n_objectives)
        new_points: New points to add (n_new, n_objectives)
        
    Returns:
        Tuple of (updated_front, was_updated)
    """
    if current_front.size == 0:
        # Empty front case - just compute Pareto front of new points
        return fast_pareto_front(new_points), True
    
    if new_points.size == 0:
        return current_front, False
    
    # Ensure proper shapes
    if new_points.ndim == 1:
        new_points = new_points.reshape(1, -1)
    if current_front.ndim == 1:
        current_front = current_front.reshape(1, -1)
    
    updated_front = []
    front_changed = False
    
    # Start with current front (will remove dominated points later)
    candidate_front = current_front.copy()
    
    # Process each new point
    for new_point in new_points:
        # Check if new point is dominated by existing front
        if not _fast_is_dominated_by_any(new_point, candidate_front):
            # New point is not dominated - add it
            candidate_front = np.vstack([candidate_front, new_point.reshape(1, -1)])
            front_changed = True
            
            # Remove points from current front that are now dominated by new point
            non_dominated_mask = np.array([
                not _fast_dominates(new_point, existing_point) 
                for existing_point in candidate_front[:-1]  # Exclude the just-added point
            ])
            
            # Keep non-dominated points plus the new point
            if not np.all(non_dominated_mask):
                candidate_front = np.vstack([
                    candidate_front[:-1][non_dominated_mask],  # Non-dominated existing points
                    new_point.reshape(1, -1)  # The new point
                ])
    
    return candidate_front, front_changed


def fast_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Fast Pareto front computation using efficient dominance checking.
    
    Args:
        points: Points to find Pareto front for (n_points, n_objectives)
        
    Returns:
        Pareto front points
    """
    if points.size == 0:
        return np.array([])
    
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    n_points = points.shape[0]
    if n_points == 1:
        return points
    
    # Use efficient O(n^2) algorithm for small fronts, more sophisticated for large ones
    if n_points <= 100:
        return _simple_pareto_front(points)
    else:
        return _divide_conquer_pareto_front(points)


@jit(nopython=True)
def _simple_pareto_front(points: np.ndarray) -> np.ndarray:
    """Simple O(n^2) Pareto front algorithm, JIT compiled for speed."""
    n_points, n_obj = points.shape
    is_pareto = np.ones(n_points, dtype=types.boolean)
    
    for i in range(n_points):
        if not is_pareto[i]:
            continue
            
        for j in range(n_points):
            if i == j or not is_pareto[j]:
                continue
                
            # Check if point j dominates point i
            if _fast_dominates(points[j], points[i]):
                is_pareto[i] = False
                break
    
    # Return only non-dominated points
    pareto_indices = np.where(is_pareto)[0]
    return points[pareto_indices]


def _divide_conquer_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Divide-and-conquer Pareto front algorithm for larger datasets.
    
    Args:
        points: Points to process
        
    Returns:
        Pareto front points
    """
    n_points = points.shape[0]
    
    if n_points <= 100:
        return _simple_pareto_front(points)
    
    # Divide into two halves
    mid = n_points // 2
    left_front = _divide_conquer_pareto_front(points[:mid])
    right_front = _divide_conquer_pareto_front(points[mid:])
    
    # Combine the two fronts
    combined = np.vstack([left_front, right_front])
    return _simple_pareto_front(combined)


class EfficientParetoTracker:
    """
    Efficient Pareto front tracker with lazy updates and caching.
    Can track multiple data types (rewards, values, states) in sync.
    """
    
    def __init__(self, update_interval: int = 10, max_pending: int = 50):
        """
        Initialize the tracker.
        
        Args:
            update_interval: Number of points to buffer before updating
            max_pending: Maximum pending points before forced update
        """
        self.current_front = np.array([])
        self.current_values = np.array([])  # Secondary data (e.g., values when front is rewards)
        self.current_states = np.array([])  # Tertiary data (states)
        self.pending_points = []
        self.pending_values = []  # Corresponding values for pending points
        self.pending_states = []  # Corresponding states for pending points
        self.update_interval = update_interval
        self.max_pending = max_pending
        self.points_since_update = 0
        
    def add_point(self, point: np.ndarray, values: np.ndarray = None, state: np.ndarray = None, 
                  force_update: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Add a point to the tracker with lazy evaluation.
        
        Args:
            point: New point to add (primary data for Pareto computation)
            values: Corresponding values (secondary data, optional)
            state: Corresponding state (tertiary data, optional)
            force_update: Force immediate update
            
        Returns:
            Tuple of (current_front, was_updated_this_call)
        """
        # Add to pending buffer
        if point.ndim == 1:
            point = point.reshape(1, -1)
        self.pending_points.append(point)
        
        # Add corresponding values and states if provided
        if values is not None:
            if values.ndim == 1:
                values = values.reshape(1, -1)
            self.pending_values.append(values)
        else:
            self.pending_values.append(None)
            
        if state is not None:
            if state.ndim == 2:
                state = state[np.newaxis, ...]
            self.pending_states.append(state)
        else:
            self.pending_states.append(None)
            
        self.points_since_update += 1
        
        # Check if update is needed
        should_update = (
            force_update or
            self.points_since_update >= self.update_interval or
            len(self.pending_points) >= self.max_pending
        )
        
        if should_update:
            return self._perform_update()
        else:
            return self.current_front, False
    
    def _perform_update(self) -> Tuple[np.ndarray, bool]:
        """Perform batched Pareto front update."""
        if not self.pending_points:
            return self.current_front, False
        
        # Combine all pending points
        new_points = np.vstack(self.pending_points)
        
        # Combine pending values and states (filter out None entries)
        new_values = None
        new_states = None
        
        valid_values = [v for v in self.pending_values if v is not None]
        if valid_values:
            new_values = np.vstack(valid_values)
            
        valid_states = [s for s in self.pending_states if s is not None]
        if valid_states:
            new_states = np.vstack(valid_states)
        
        # Update front using the primary data (points)
        updated_front, was_updated = incremental_pareto_update(self.current_front, new_points)
        
        if was_updated:
            # Find which indices from the combined data are in the new front
            if self.current_front.size == 0:
                # First update - all new points considered
                combined_points = new_points
                combined_values = new_values
                combined_states = new_states
            else:
                # Combine current and new data
                combined_points = np.vstack([self.current_front, new_points])
                
                combined_values = None
                if self.current_values.size > 0 and new_values is not None:
                    combined_values = np.vstack([self.current_values, new_values])
                elif new_values is not None:
                    combined_values = new_values
                elif self.current_values.size > 0:
                    combined_values = self.current_values
                    
                combined_states = None
                if self.current_states.size > 0 and new_states is not None:
                    combined_states = np.vstack([self.current_states, new_states])
                elif new_states is not None:
                    combined_states = new_states
                elif self.current_states.size > 0:
                    combined_states = self.current_states
            
            # Find indices of the updated front in the combined data
            front_indices = []
            for front_point in updated_front:
                # Find matching indices (allowing for floating point tolerance)
                matches = np.all(np.abs(combined_points - front_point) < 1e-10, axis=1)
                idx = np.where(matches)[0]
                if len(idx) > 0:
                    front_indices.append(idx[0])  # Take first match
            
            # Update current data with new front
            self.current_front = updated_front
            if combined_values is not None and len(front_indices) > 0:
                self.current_values = combined_values[front_indices]
            if combined_states is not None and len(front_indices) > 0:
                self.current_states = combined_states[front_indices]
        
        # Clear pending data
        self.pending_points = []
        self.pending_values = []
        self.pending_states = []
        self.points_since_update = 0
        
        return updated_front, was_updated
    
    def get_front(self, force_update: bool = False) -> np.ndarray:
        """Get current Pareto front, optionally forcing an update."""
        if force_update and self.pending_points:
            front, _ = self._perform_update()
            return front
        return self.current_front
    
    def get_values(self, force_update: bool = False) -> np.ndarray:
        """Get values corresponding to current Pareto front."""
        if force_update and self.pending_points:
            self._perform_update()
        return self.current_values
    
    def get_states(self, force_update: bool = False) -> np.ndarray:
        """Get states corresponding to current Pareto front.""" 
        if force_update and self.pending_points:
            self._perform_update()
        return self.current_states
    
    def set_current_data(self, front: np.ndarray, values: np.ndarray = None, states: np.ndarray = None):
        """Set current front data (for loading from checkpoint)."""
        if states.ndim == 2 and states.size > 0:
            raise Exception("States must be 3D array (n_points, state_dim1, state_dim2)")
        self.current_front = front.copy() if front.size > 0 else np.array([])
        self.current_values = values.copy() if values is not None and values.size > 0 else np.array([])
        self.current_states = states.copy() if states is not None and states.size > 0 else np.array([])
        
        # Clear pending data
        self.pending_points = []
        self.pending_values = []
        self.pending_states = []
        self.points_since_update = 0
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            'front_size': len(self.current_front) if self.current_front.size > 0 else 0,
            'values_size': len(self.current_values) if self.current_values.size > 0 else 0,
            'states_size': len(self.current_states) if self.current_states.size > 0 else 0,
            'pending_points': len(self.pending_points),
            'points_since_update': self.points_since_update,
            'update_interval': self.update_interval
        }