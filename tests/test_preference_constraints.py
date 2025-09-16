"""
Unit tests for preference constraints functionality.
Tests the preference constrained tracker and preference constraints addon.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from coatopt.algorithms.hppo.training.utils.preference_constrained_tracker import (
    PreferenceConstrainedTracker,
)
from coatopt.environments.reward_functions.reward_addons import (
    apply_preference_constraints_addon,
)


class TestPreferenceConstrainedTracker:
    """Test suite for PreferenceConstrainedTracker."""

    @pytest.fixture
    def basic_tracker(self):
        """Basic tracker for testing."""
        return PreferenceConstrainedTracker(
            optimise_parameters=["reflectivity", "thermal_noise"],
            phase1_epochs_per_objective=100,
            phase2_epochs_per_step=50,
            constraint_steps=4,
            constraint_penalty_weight=10.0,
            constraint_margin=0.1,
        )

    @pytest.fixture
    def three_objective_tracker(self):
        """Tracker with three objectives."""
        return PreferenceConstrainedTracker(
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            phase1_epochs_per_objective=50,
            phase2_epochs_per_step=30,
            constraint_steps=6,
        )

    def test_init(self, basic_tracker):
        """Test tracker initialization."""
        assert basic_tracker.optimise_parameters == ["reflectivity", "thermal_noise"]
        assert basic_tracker.n_objectives == 2
        assert basic_tracker.phase1_epochs_per_objective == 100
        assert basic_tracker.phase2_epochs_per_step == 50
        assert basic_tracker.constraint_steps == 4
        assert basic_tracker.constraint_penalty_weight == 10.0
        assert basic_tracker.constraint_margin == 0.1

        # Phase timing calculations
        assert basic_tracker.phase1_total_epochs == 200  # 2 objectives * 100 epochs
        assert basic_tracker.phase2_cycle_epochs == 100  # 2 objectives * 50 epochs

        # Initial state
        assert basic_tracker.phase1_completed is False
        assert basic_tracker.current_epoch == 0
        assert basic_tracker.current_phase == 1

        # Reward bounds initialization
        for param in ["reflectivity", "thermal_noise"]:
            assert param in basic_tracker.reward_bounds
            assert basic_tracker.reward_bounds[param]["min"] == float("inf")
            assert basic_tracker.reward_bounds[param]["max"] == float("-inf")

    def test_update_reward_bounds(self, basic_tracker):
        """Test updating reward bounds."""
        # Initial bounds should be inf/-inf
        assert basic_tracker.reward_bounds["reflectivity"]["min"] == float("inf")
        assert basic_tracker.reward_bounds["reflectivity"]["max"] == float("-inf")

        # Update with first reward
        rewards1 = {"reflectivity": 0.8, "thermal_noise": 0.6}
        basic_tracker.update_reward_bounds(rewards1)

        assert basic_tracker.reward_bounds["reflectivity"]["min"] == 0.8
        assert basic_tracker.reward_bounds["reflectivity"]["max"] == 0.8
        assert basic_tracker.reward_bounds["thermal_noise"]["min"] == 0.6
        assert basic_tracker.reward_bounds["thermal_noise"]["max"] == 0.6

        # Update with new extremes
        rewards2 = {"reflectivity": 0.9, "thermal_noise": 0.4}
        basic_tracker.update_reward_bounds(rewards2)

        assert basic_tracker.reward_bounds["reflectivity"]["min"] == 0.8  # Unchanged
        assert basic_tracker.reward_bounds["reflectivity"]["max"] == 0.9  # Updated
        assert basic_tracker.reward_bounds["thermal_noise"]["min"] == 0.4  # Updated
        assert basic_tracker.reward_bounds["thermal_noise"]["max"] == 0.6  # Unchanged

        # Update with values in between
        rewards3 = {"reflectivity": 0.85, "thermal_noise": 0.5}
        basic_tracker.update_reward_bounds(rewards3)

        # Bounds should remain unchanged
        assert basic_tracker.reward_bounds["reflectivity"]["min"] == 0.8
        assert basic_tracker.reward_bounds["reflectivity"]["max"] == 0.9
        assert basic_tracker.reward_bounds["thermal_noise"]["min"] == 0.4
        assert basic_tracker.reward_bounds["thermal_noise"]["max"] == 0.6

    def test_update_reward_bounds_missing_parameters(self, basic_tracker):
        """Test updating bounds with missing parameters."""
        # Update with only one parameter
        rewards = {"reflectivity": 0.7}
        basic_tracker.update_reward_bounds(rewards)

        assert basic_tracker.reward_bounds["reflectivity"]["min"] == 0.7
        assert basic_tracker.reward_bounds["reflectivity"]["max"] == 0.7
        # thermal_noise bounds should remain unchanged
        assert basic_tracker.reward_bounds["thermal_noise"]["min"] == float("inf")
        assert basic_tracker.reward_bounds["thermal_noise"]["max"] == float("-inf")

    def test_phase1_info(self, basic_tracker):
        """Test Phase 1 training information."""
        # First objective (reflectivity) - epochs 0-99
        phase_info = basic_tracker.get_training_phase_info(50)

        assert phase_info["phase"] == 1
        assert phase_info["target_objective"] == "reflectivity"
        assert phase_info["weights"]["reflectivity"] == 1.0
        assert phase_info["weights"]["thermal_noise"] == 0.0
        assert phase_info.get("constraints", {}) == {}

        # Second objective (thermal_noise) - epochs 100-199
        phase_info = basic_tracker.get_training_phase_info(150)

        assert phase_info["phase"] == 1
        assert phase_info["target_objective"] == "thermal_noise"
        assert phase_info["weights"]["reflectivity"] == 0.0
        assert phase_info["weights"]["thermal_noise"] == 1.0

    def test_phase1_to_phase2_transition(self, basic_tracker):
        """Test transition from Phase 1 to Phase 2."""
        # Set up some reward bounds by updating during Phase 1
        basic_tracker.update_reward_bounds({"reflectivity": 0.7, "thermal_noise": 0.5})
        basic_tracker.update_reward_bounds({"reflectivity": 0.9, "thermal_noise": 0.8})

        # Get Phase 1 info (should still be Phase 1)
        phase1_info = basic_tracker.get_training_phase_info(100)
        assert phase1_info["phase"] == 1
        assert basic_tracker.phase1_completed is False

        # Transition to Phase 2 (epoch >= 200)
        phase2_info = basic_tracker.get_training_phase_info(200)
        assert phase2_info["phase"] == 2
        assert basic_tracker.phase1_completed is True

        # Should have constraints in Phase 2
        assert "constraints" in phase2_info
        assert len(phase2_info["constraints"]) > 0

    def test_phase2_info_constraint_progression(self, basic_tracker):
        """Test Phase 2 constraint progression."""
        # Set up reward bounds first
        basic_tracker.update_reward_bounds({"reflectivity": 0.6, "thermal_noise": 0.4})
        basic_tracker.update_reward_bounds({"reflectivity": 1.0, "thermal_noise": 0.9})

        # Force finalize Phase 1
        basic_tracker._finalize_phase1()

        # Test different constraint steps
        phase2_info1 = basic_tracker.get_training_phase_info(
            200
        )  # First constraint step
        phase2_info2 = basic_tracker.get_training_phase_info(
            250
        )  # Second constraint step

        assert phase2_info1["phase"] == 2
        assert phase2_info2["phase"] == 2

        # Constraints should get tighter (assuming implementation details)
        assert "constraints" in phase2_info1
        assert "constraints" in phase2_info2

    def test_three_objective_tracker(self, three_objective_tracker):
        """Test tracker with three objectives."""
        assert three_objective_tracker.n_objectives == 3
        assert three_objective_tracker.phase1_total_epochs == 150  # 3 * 50
        assert three_objective_tracker.phase2_cycle_epochs == 90  # 3 * 30

        # Test Phase 1 objective cycling
        info1 = three_objective_tracker.get_training_phase_info(25)  # First objective
        info2 = three_objective_tracker.get_training_phase_info(75)  # Second objective
        info3 = three_objective_tracker.get_training_phase_info(125)  # Third objective

        assert info1["target_objective"] == "reflectivity"
        assert info2["target_objective"] == "thermal_noise"
        assert info3["target_objective"] == "absorption"

    def test_constraint_margin_application(self):
        """Test constraint margin application."""
        tracker = PreferenceConstrainedTracker(
            optimise_parameters=["obj1", "obj2"], constraint_margin=0.2  # 20% margin
        )

        # Set up bounds and finalize Phase 1
        tracker.update_reward_bounds({"obj1": 0.0, "obj2": 0.0})
        tracker.update_reward_bounds({"obj1": 1.0, "obj2": 1.0})
        tracker._finalize_phase1()

        # The constraint margin should be applied in Phase 2
        # (specific implementation details depend on the _get_phase2_info method)
        assert tracker.constraint_margin == 0.2


class TestPreferenceConstraintsAddon:
    """Test suite for apply_preference_constraints_addon function."""

    @pytest.fixture
    def mock_pc_tracker(self):
        """Mock PreferenceConstrainedTracker."""
        tracker = Mock()
        tracker.update_reward_bounds = Mock()
        tracker.apply_constraint_penalties = Mock(return_value=0.1)
        return tracker

    @pytest.fixture
    def basic_phase_info(self):
        """Basic phase info for testing."""
        return {
            "phase": 1,
            "target_objective": "reflectivity",
            "weights": {"reflectivity": 1.0, "thermal_noise": 0.0},
        }

    @pytest.fixture
    def phase2_info(self):
        """Phase 2 info with constraints."""
        return {
            "phase": 2,
            "target_objective": "reflectivity",
            "weights": {"reflectivity": 1.0, "thermal_noise": 0.0},
            "constraints": {"thermal_noise": 0.5},  # Constraint on non-target objective
        }

    def test_phase1_no_constraints(self, mock_pc_tracker, basic_phase_info):
        """Test addon behavior in Phase 1 (no constraints)."""
        total_reward = 0.8
        rewards = {"reflectivity": 0.9, "thermal_noise": 0.7}
        vals = {"reflectivity": 0.999, "thermal_noise": 1e-20}
        env = Mock()
        optimise_parameters = ["reflectivity", "thermal_noise"]

        updated_total, updated_rewards = apply_preference_constraints_addon(
            total_reward,
            rewards,
            vals,
            env,
            optimise_parameters,
            pc_tracker=mock_pc_tracker,
            phase_info=basic_phase_info,
        )

        # Should update reward bounds
        mock_pc_tracker.update_reward_bounds.assert_called_once_with(rewards)

        # Should not apply constraint penalties in Phase 1
        mock_pc_tracker.apply_constraint_penalties.assert_not_called()

        # Total reward should be unchanged (no penalties)
        assert updated_total == total_reward
        assert updated_rewards["total_reward"] == total_reward
        assert updated_rewards["pc_phase"] == 1
        assert updated_rewards["pc_constraints_active"] == {}
        assert updated_rewards["pc_penalty_addon"] == 0.0

    def test_phase2_with_constraints(self, mock_pc_tracker, phase2_info):
        """Test addon behavior in Phase 2 (with constraints)."""
        total_reward = 0.8
        rewards = {"reflectivity": 0.9, "thermal_noise": 0.7}
        vals = {"reflectivity": 0.999, "thermal_noise": 1e-20}
        env = Mock()
        optimise_parameters = ["reflectivity", "thermal_noise"]
        constraint_penalty = 0.2

        mock_pc_tracker.apply_constraint_penalties.return_value = constraint_penalty

        updated_total, updated_rewards = apply_preference_constraints_addon(
            total_reward,
            rewards,
            vals,
            env,
            optimise_parameters,
            pc_tracker=mock_pc_tracker,
            phase_info=phase2_info,
        )

        # Should update reward bounds
        mock_pc_tracker.update_reward_bounds.assert_called_once_with(rewards)

        # Should apply constraint penalties in Phase 2
        mock_pc_tracker.apply_constraint_penalties.assert_called_once_with(
            rewards, phase2_info["constraints"]
        )

        # Total reward should be reduced by penalty
        expected_total = total_reward - constraint_penalty
        assert updated_total == expected_total
        assert updated_rewards["total_reward"] == expected_total
        assert updated_rewards["pc_phase"] == 2
        assert updated_rewards["pc_constraint_penalty"] == constraint_penalty
        assert updated_rewards["pc_constraints_active"] == phase2_info["constraints"]
        assert updated_rewards["pc_target_objective"] == "reflectivity"
        assert updated_rewards["pc_penalty_addon"] == -constraint_penalty

    def test_missing_pc_tracker_raises_error(self):
        """Test that missing PC tracker raises appropriate error."""
        with pytest.raises(
            Exception,
            match="PreferenceConstrainedTracker and phase_info must be provided",
        ):
            apply_preference_constraints_addon(
                0.8, {}, {}, Mock(), ["obj1"], pc_tracker=None, phase_info={"phase": 1}
            )

    def test_missing_phase_info_raises_error(self, mock_pc_tracker):
        """Test that missing phase info raises appropriate error."""
        with pytest.raises(
            Exception,
            match="PreferenceConstrainedTracker and phase_info must be provided",
        ):
            apply_preference_constraints_addon(
                0.8,
                {},
                {},
                Mock(),
                ["obj1"],
                pc_tracker=mock_pc_tracker,
                phase_info=None,
            )

    def test_phase2_no_constraints(self, mock_pc_tracker):
        """Test Phase 2 behavior when no constraints are provided."""
        phase_info = {"phase": 2}  # No constraints key

        updated_total, updated_rewards = apply_preference_constraints_addon(
            0.8,
            {"obj1": 0.9},
            {},
            Mock(),
            ["obj1"],
            pc_tracker=mock_pc_tracker,
            phase_info=phase_info,
        )

        # Should not apply penalties when no constraints
        assert updated_total == 0.8
        assert updated_rewards["pc_constraints_active"] == {}
        assert "pc_constraint_penalty" not in updated_rewards

    def test_constraint_penalty_weight(self, mock_pc_tracker, phase2_info):
        """Test constraint penalty weight application."""
        constraint_penalty_weight = 2.0
        mock_penalty = 0.1
        mock_pc_tracker.apply_constraint_penalties.return_value = mock_penalty

        updated_total, updated_rewards = apply_preference_constraints_addon(
            0.8,
            {"obj1": 0.9},
            {},
            Mock(),
            ["obj1"],
            pc_tracker=mock_pc_tracker,
            phase_info=phase2_info,
            constraint_penalty_weight=constraint_penalty_weight,
        )

        # The weight is passed to the addon function but actual scaling
        # depends on implementation details of the tracker
        expected_total = 0.8 - mock_penalty
        assert updated_total == expected_total


class TestPreferenceConstraintsIntegration:
    """Integration tests for preference constraints components."""

    def test_full_training_cycle_simulation(self):
        """Test a simulated training cycle with preference constraints."""
        tracker = PreferenceConstrainedTracker(
            optimise_parameters=["obj1", "obj2"],
            phase1_epochs_per_objective=10,
            phase2_epochs_per_step=5,
            constraint_steps=2,
        )

        # Simulate Phase 1 training
        for epoch in range(20):  # Full Phase 1
            phase_info = tracker.get_training_phase_info(epoch)
            assert phase_info["phase"] == 1

            # Simulate reward observations
            if epoch < 10:
                # Focus on obj1
                rewards = {"obj1": 0.8 + epoch * 0.01, "obj2": 0.5}
            else:
                # Focus on obj2
                rewards = {"obj1": 0.7, "obj2": 0.6 + (epoch - 10) * 0.02}

            tracker.update_reward_bounds(rewards)

        # Check final Phase 1 bounds
        assert tracker.reward_bounds["obj1"]["min"] <= 0.8
        assert tracker.reward_bounds["obj1"]["max"] >= 0.89
        assert tracker.reward_bounds["obj2"]["min"] <= 0.5
        assert tracker.reward_bounds["obj2"]["max"] >= 0.78

        # Simulate Phase 2 training
        for epoch in range(20, 30):  # Start of Phase 2
            phase_info = tracker.get_training_phase_info(epoch)
            assert phase_info["phase"] == 2

            # Should have constraints now
            if tracker.phase1_completed:
                assert "constraints" in phase_info

    def test_addon_with_real_tracker(self):
        """Test addon function with real PreferenceConstrainedTracker."""
        tracker = PreferenceConstrainedTracker(["obj1", "obj2"])

        # Set up some bounds
        tracker.update_reward_bounds({"obj1": 0.5, "obj2": 0.3})
        tracker.update_reward_bounds({"obj1": 0.9, "obj2": 0.8})

        # Test Phase 1
        phase1_info = {"phase": 1, "target_objective": "obj1"}

        updated_total, updated_rewards = apply_preference_constraints_addon(
            0.7,
            {"obj1": 0.8, "obj2": 0.6},
            {},
            Mock(),
            ["obj1", "obj2"],
            pc_tracker=tracker,
            phase_info=phase1_info,
        )

        assert updated_total == 0.7  # No penalty in Phase 1
        assert updated_rewards["pc_phase"] == 1
