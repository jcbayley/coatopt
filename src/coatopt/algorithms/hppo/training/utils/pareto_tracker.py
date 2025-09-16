"""
Efficient Pareto front utilities for multi-objective optimization.
Contains optimized algorithms for non-dominated sorting and Pareto front maintenance.
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import jit, types
from numba.typed import List as TypedList
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def pareto_front(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast Pareto front computation using pymoo's NonDominatedSorting.

    Args:
        points: Points to find Pareto front for (n_points, n_objectives)

    Returns:
        Pareto front points and their indices
    """
    if points.size == 0:
        return np.array([]), np.array([])

    if points.ndim == 1:
        points = points.reshape(1, -1)

    nds = NonDominatedSorting()
    front_indices = nds.do(-points, only_non_dominated_front=True)
    return points[front_indices], front_indices


class ParetoTracker:
    """
    Efficient Pareto front tracker with lazy updates and caching.
    Now supports tracking values and states alongside rewards.
    """

    def __init__(self, update_interval: int = 10, max_pending: int = 50):
        """
        Initialize the tracker.

        Args:
            update_interval: Number of points to buffer before updating
            max_pending: Maximum pending points before forced update
        """
        self.current_front = np.array([])
        self.current_values = np.array([])  # Physical values corresponding to front
        self.current_states = np.array([])  # States corresponding to front

        self.pending_points = []
        self.pending_values = []
        self.pending_states = []

        self.update_interval = update_interval
        self.max_pending = max_pending
        self.points_since_update = 0

    def add_point(
        self,
        point: np.ndarray,
        values: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        force_update: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Add a point to the tracker with lazy evaluation.

        Args:
            point: New reward point to add
            values: Optional physical values corresponding to the point
            state: Optional state corresponding to the point
            force_update: Force immediate update

        Returns:
            Tuple of (current_front, was_updated_this_call)
        """
        # Add to pending buffers
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
            self.pending_states.append(state.copy()[np.newaxis, ...])
        else:
            self.pending_states.append(None)

        self.points_since_update += 1

        # Check if update is needed
        should_update = (
            force_update
            or self.points_since_update >= self.update_interval
            or len(self.pending_points) >= self.max_pending
        )

        if should_update:
            return self._perform_update()
        else:
            return self.current_front, False

    def _perform_update(self) -> Tuple[np.ndarray, bool]:
        """Perform batched Pareto front update with associated data."""
        if not self.pending_points:
            return self.current_front, False

        # Combine all pending points
        new_points = np.vstack(self.pending_points)
        new_values = np.vstack(self.pending_values)
        new_states = np.vstack(self.pending_states)

        # Combine all current data (existing + pending)
        all_points = (
            np.vstack([self.current_front, new_points])
            if self.current_front.size > 0
            else new_points
        )
        all_values = (
            np.vstack([self.current_values, new_values])
            if self.current_values.size > 0
            else new_values
        )
        all_states = (
            np.vstack([self.current_states, new_states])
            if self.current_states.size > 0
            else new_states
        )

        # Compute new Pareto front
        updated_front, front_indices = pareto_front(all_points)

        updated_values = (
            all_values[front_indices] if all_values.size > 0 else np.array([])
        )
        updated_states = (
            all_states[front_indices] if all_states.size > 0 else np.array([])
        )

        # Update state
        was_updated = not np.array_equal(self.current_front, updated_front)
        self.current_front = updated_front
        self.current_values = updated_values
        self.current_states = updated_states

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
        """Get current Pareto front values, optionally forcing an update."""
        if force_update and self.pending_points:
            self._perform_update()
        return self.current_values

    def get_states(self, force_update: bool = False) -> np.ndarray:
        """Get current Pareto front states, optionally forcing an update."""
        if force_update and self.pending_points:
            self._perform_update()
        return self.current_states

    def set_current_data(
        self,
        front: np.ndarray,
        values: Optional[np.ndarray] = None,
        states: Optional[np.ndarray] = None,
    ):
        """Set current Pareto front data (for loading from checkpoint)."""
        self.current_front = front.copy() if front.size > 0 else np.array([])
        self.current_values = (
            values.copy() if values is not None and values.size > 0 else np.array([])
        )
        self.current_states = (
            states.copy() if states is not None and states.size > 0 else np.array([])
        )

        # Clear pending data
        self.pending_points = []
        self.pending_values = []
        self.pending_states = []
        self.points_since_update = 0

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "front_size": len(self.current_front) if self.current_front.size > 0 else 0,
            "pending_points": len(self.pending_points),
            "points_since_update": self.points_since_update,
            "update_interval": self.update_interval,
        }
