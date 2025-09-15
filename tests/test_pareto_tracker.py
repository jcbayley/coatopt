"""
Unit tests for the ParetoTracker class.
Tests Pareto front computation, tracking, and lazy update functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock

from coatopt.algorithms.hppo.training.utils.pareto_tracker import ParetoTracker, pareto_front


class TestParetoFront:
    """Test the pareto_front function."""
    
    def test_pareto_front_basic_2d(self):
        """Test basic 2D Pareto front computation."""
        # Points where (0,3) and (3,0) should be on the front
        points = np.array([
            [0, 3],
            [1, 2], 
            [2, 1],
            [3, 0],
            [1.5, 1.5]  # This should be dominated
        ])
        
        front, indices = pareto_front(points)
        
        # Should contain (0,3) and (3,0)
        expected_points = np.array([[0, 3], [3, 0]])
        assert front.shape[0] == 2  # Two points on the front
        
        # Check that the front points are correct (order may vary)
        for expected_point in expected_points:
            assert any(np.allclose(front_point, expected_point) for front_point in front)
    
    def test_pareto_front_empty_input(self):
        """Test pareto_front with empty input."""
        points = np.array([])
        front, indices = pareto_front(points)
        
        assert front.size == 0
        assert indices.size == 0
    
    def test_pareto_front_single_point(self):
        """Test pareto_front with single point."""
        points = np.array([[1, 2]])
        front, indices = pareto_front(points)
        
        assert np.array_equal(front, points)
        assert np.array_equal(indices, [0])
    
    def test_pareto_front_1d_input(self):
        """Test pareto_front with 1D input (single objective)."""
        points = np.array([3, 1, 4, 2])
        front, indices = pareto_front(points)
        
        # For single objective, only the maximum should be on the front
        assert front.shape == (1, 4)  # Reshaped to 2D
        assert np.max(front) == 4
    
    def test_pareto_front_3d(self):
        """Test 3D Pareto front computation."""
        points = np.array([
            [1, 1, 1],
            [2, 0, 0],  # On front
            [0, 2, 0],  # On front
            [0, 0, 2],  # On front
            [0.5, 0.5, 0.5]  # Dominated
        ])
        
        front, indices = pareto_front(points)
        
        # Should have 3 points on the front
        assert front.shape[0] == 3


class TestParetoTracker:
    """Test the ParetoTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Basic tracker for testing."""
        return ParetoTracker(update_interval=2, max_pending=5)
    
    def test_init(self, tracker):
        """Test tracker initialization."""
        assert tracker.current_front.size == 0
        assert tracker.current_values.size == 0
        assert tracker.current_states.size == 0
        assert len(tracker.pending_points) == 0
        assert tracker.update_interval == 2
        assert tracker.max_pending == 5
        assert tracker.points_since_update == 0
    
    def test_add_point_no_immediate_update(self, tracker):
        """Test adding a point without triggering update."""
        point = np.array([1, 2])
        values = np.array([0.5, 0.3])
        state = np.array([1, 0, 1])
        
        front, was_updated = tracker.add_point(point, values, state)
        
        assert not was_updated
        assert front.size == 0  # No update yet
        assert len(tracker.pending_points) == 1
        assert len(tracker.pending_values) == 1
        assert len(tracker.pending_states) == 1
        assert tracker.points_since_update == 1
    
    def test_add_point_triggers_update(self, tracker):
        """Test adding points that trigger an update."""
        # Add first point (no update)
        tracker.add_point(np.array([1, 2]))
        
        # Add second point (should trigger update due to interval=2)
        front, was_updated = tracker.add_point(np.array([3, 0]))
        
        assert was_updated
        assert front.shape[0] == 2  # Both points should be on the front
        assert len(tracker.pending_points) == 0  # Pending should be cleared
        assert tracker.points_since_update == 0
    
    def test_add_point_force_update(self, tracker):
        """Test forcing an immediate update."""
        point = np.array([1, 2])
        
        front, was_updated = tracker.add_point(point, force_update=True)
        
        assert was_updated
        assert front.shape[0] == 1
        assert np.array_equal(front[0], point)
        assert len(tracker.pending_points) == 0
    
    def test_add_point_max_pending_triggers_update(self):
        """Test that reaching max_pending triggers update."""
        tracker = ParetoTracker(update_interval=10, max_pending=2)
        
        # Add points up to max_pending
        tracker.add_point(np.array([1, 0]))
        front, was_updated = tracker.add_point(np.array([0, 1]))
        
        # Should trigger update when reaching max_pending
        assert was_updated
        assert front.shape[0] == 2
        assert len(tracker.pending_points) == 0
    
    def test_pareto_front_computation(self, tracker):
        """Test that Pareto front is correctly computed."""
        # Add dominated and non-dominated points
        points = [
            np.array([0, 3]),  # Non-dominated
            np.array([1, 2]),  # Dominated
            np.array([2, 1]),  # Dominated  
            np.array([3, 0])   # Non-dominated
        ]
        
        for point in points:
            tracker.add_point(point, force_update=False)
        
        # Force final update
        front, _ = tracker.add_point(np.array([1.5, 1.5]), force_update=True)
        
        # Should have 2 non-dominated points
        assert front.shape[0] == 2
        
        # Check that the correct points are on the front
        expected_points = np.array([[0, 3], [3, 0]])
        for expected_point in expected_points:
            assert any(np.allclose(front_point, expected_point) for front_point in front)
    
    def test_get_front(self, tracker):
        """Test get_front method."""
        # Add point without update
        tracker.add_point(np.array([1, 2]))
        
        # Get front without forcing update
        front_no_force = tracker.get_front(force_update=False)
        assert front_no_force.size == 0
        
        # Get front with forced update
        front_forced = tracker.get_front(force_update=True)
        assert front_forced.shape[0] == 1
        assert np.array_equal(front_forced[0], [1, 2])
    
    def test_get_values_and_states(self, tracker):
        """Test getting values and states associated with front."""
        point = np.array([1, 2])
        values = np.array([0.5, 0.3])
        state = np.array([1, 0, 1, 0])
        
        tracker.add_point(point, values, state, force_update=True)
        
        retrieved_values = tracker.get_values()
        retrieved_states = tracker.get_states()
        
        assert np.array_equal(retrieved_values[0], values)
        assert np.array_equal(retrieved_states[0], state)
    
    def test_set_current_data(self, tracker):
        """Test setting current data (for loading from checkpoint)."""
        front = np.array([[1, 2], [3, 0]])
        values = np.array([[0.5, 0.3], [0.8, 0.1]])
        states = np.array([[1, 0], [0, 1]])
        
        tracker.set_current_data(front, values, states)
        
        assert np.array_equal(tracker.current_front, front)
        assert np.array_equal(tracker.current_values, values)
        assert np.array_equal(tracker.current_states, states)
        assert len(tracker.pending_points) == 0
        assert tracker.points_since_update == 0
    
    def test_set_current_data_empty_arrays(self, tracker):
        """Test setting current data with empty arrays."""
        tracker.set_current_data(np.array([]), np.array([]), np.array([]))
        
        assert tracker.current_front.size == 0
        assert tracker.current_values.size == 0
        assert tracker.current_states.size == 0
    
    def test_get_stats(self, tracker):
        """Test getting tracker statistics."""
        # Add some points
        tracker.add_point(np.array([1, 2]), force_update=True)
        tracker.add_point(np.array([0, 1]))  # Pending
        
        stats = tracker.get_stats()
        
        expected_stats = {
            'front_size': 1,
            'pending_points': 1,
            'points_since_update': 1,
            'update_interval': 2
        }
        assert stats == expected_stats
    
    def test_values_states_with_none_inputs(self, tracker):
        """Test adding points with None values and states."""
        point = np.array([1, 2])
        
        tracker.add_point(point, values=None, state=None, force_update=True)
        
        # Should still work, but values and states should be empty or None
        values = tracker.get_values()
        states = tracker.get_states()
        
        # The exact behavior may depend on implementation details
        # Just verify no errors occur
        assert values is not None
        assert states is not None
    
    def test_1d_point_handling(self, tracker):
        """Test that 1D points are properly handled."""
        point = np.array([1, 2])  # Will be reshaped to (1, 2)
        values = np.array([0.5, 0.3])  # Will be reshaped to (1, 2)
        
        front, was_updated = tracker.add_point(point, values, force_update=True)
        
        assert was_updated
        assert front.ndim == 2
        assert front.shape == (1, 2)
    
    def test_batched_update_performance(self, tracker):
        """Test that batched updates work correctly."""
        # Add multiple points that should trigger a batched update
        points = [np.array([i, 10-i]) for i in range(5)]
        
        # Add all but the last without update
        for point in points[:-1]:
            tracker.add_point(point, force_update=False)
        
        # Final point should trigger batched update
        front, was_updated = tracker.add_point(points[-1], force_update=True)
        
        assert was_updated
        # All points should be non-dominated in this case
        assert front.shape[0] == 5


class TestParetoTrackerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_update_with_identical_points(self):
        """Test updating with identical points."""
        tracker = ParetoTracker(update_interval=1)
        
        # Add same point multiple times
        point = np.array([1, 2])
        tracker.add_point(point, force_update=True)
        front2, was_updated = tracker.add_point(point, force_update=True)
        
        # Should not be updated (identical point)
        assert not was_updated
        assert front2.shape[0] == 1
    
    def test_large_number_of_points(self):
        """Test tracker with a large number of points."""
        tracker = ParetoTracker(update_interval=50, max_pending=100)
        
        # Generate many random points
        np.random.seed(42)
        points = np.random.rand(200, 3)
        
        for i, point in enumerate(points):
            if i == len(points) - 1:  # Force update on last point
                tracker.add_point(point, force_update=True)
            else:
                tracker.add_point(point)
        
        front = tracker.get_front()
        assert front.shape[0] > 0  # Should have some front points
        assert front.shape[1] == 3  # Should maintain dimensionality
    
    def test_mixed_dimensions_error_handling(self):
        """Test behavior with inconsistent dimensions."""
        tracker = ParetoTracker()
        
        # Add 2D point first
        tracker.add_point(np.array([1, 2]), force_update=True)
        
        # Try adding 3D point - this should handle gracefully or raise appropriate error
        # The exact behavior depends on implementation, but it shouldn't crash
        try:
            tracker.add_point(np.array([1, 2, 3]), force_update=True)
            # If it succeeds, verify the result makes sense
            front = tracker.get_front()
            assert front is not None
        except (ValueError, AssertionError):
            # If it raises an error, that's also acceptable behavior
            pass
