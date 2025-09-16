"""
Unit tests for the TrainingPlotManager class.
Tests plot generation, data management, and UI/CLI compatibility.
"""

import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from coatopt.utils.plotting.core import TrainingPlotManager


class TestTrainingPlotManager:
    """Test suite for TrainingPlotManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def basic_plot_manager(self, temp_dir):
        """Basic plot manager for testing."""
        return TrainingPlotManager(
            save_plots=True, output_dir=temp_dir, ui_mode=False, figure_size=(10, 8)
        )

    @pytest.fixture
    def ui_plot_manager(self, temp_dir):
        """UI mode plot manager for testing."""
        return TrainingPlotManager(
            save_plots=False, output_dir=temp_dir, ui_mode=True, figure_size=(12, 10)
        )

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for testing."""
        return [
            {
                "episode": 100,
                "total_reward": 0.85,
                "individual_rewards": {"reflectivity": 0.9, "thermal_noise": 0.8},
                "values": {"reflectivity": 0.9999, "thermal_noise": 1e-20},
                "weights": [0.5, 0.5],
            },
            {
                "episode": 200,
                "total_reward": 0.90,
                "individual_rewards": {"reflectivity": 0.95, "thermal_noise": 0.85},
                "values": {"reflectivity": 0.99995, "thermal_noise": 8e-21},
                "weights": [0.6, 0.4],
            },
        ]

    @pytest.fixture
    def sample_pareto_data(self):
        """Sample Pareto front data for testing."""
        return [
            {
                "reflectivity": 0.9999,
                "thermal_noise": 1e-20,
                "reward_reflectivity": 0.9,
                "reward_thermal_noise": 0.8,
            },
            {
                "reflectivity": 0.99995,
                "thermal_noise": 8e-21,
                "reward_reflectivity": 0.95,
                "reward_thermal_noise": 0.85,
            },
        ]

    def test_init_basic(self, temp_dir):
        """Test basic initialization."""
        manager = TrainingPlotManager(
            save_plots=True, output_dir=temp_dir, ui_mode=False
        )

        assert manager.save_plots is True
        assert manager.output_dir == temp_dir
        assert manager.ui_mode is False
        assert manager.figure_size == (12, 10)  # Default
        assert len(manager.training_data) == 0
        assert len(manager.pareto_data) == 0
        assert len(manager.eval_data) == 0
        assert len(manager.coating_states) == 0

    def test_init_ui_mode(self, temp_dir):
        """Test initialization in UI mode."""
        manager = TrainingPlotManager(
            save_plots=False, output_dir=temp_dir, ui_mode=True, figure_size=(8, 6)
        )

        assert manager.ui_mode is True
        assert manager.save_plots is False
        assert manager.figure_size == (8, 6)

    def test_set_objective_info(self, basic_plot_manager):
        """Test setting objective information."""
        labels = ["Reflectivity", "Thermal Noise"]
        scales = ["linear", "log"]
        params = ["reflectivity", "thermal_noise"]
        targets = {"reflectivity": 0.9999, "thermal_noise": 1e-20}

        basic_plot_manager.objective_labels = labels
        basic_plot_manager.objective_scales = scales
        basic_plot_manager.optimisation_parameters = params
        basic_plot_manager.objective_targets = targets

        assert basic_plot_manager.objective_labels == labels
        assert basic_plot_manager.objective_scales == scales
        assert basic_plot_manager.optimisation_parameters == params
        assert basic_plot_manager.objective_targets == targets

    def test_add_training_data(self, basic_plot_manager, sample_training_data):
        """Test adding training data."""
        for data_point in sample_training_data:
            basic_plot_manager.training_data.append(data_point)

        assert len(basic_plot_manager.training_data) == 2
        assert basic_plot_manager.training_data[0]["episode"] == 100
        assert basic_plot_manager.training_data[1]["total_reward"] == 0.90

    def test_add_pareto_data(self, basic_plot_manager, sample_pareto_data):
        """Test adding Pareto front data."""
        for data_point in sample_pareto_data:
            basic_plot_manager.pareto_data.append(data_point)

        assert len(basic_plot_manager.pareto_data) == 2
        assert basic_plot_manager.pareto_data[0]["reflectivity"] == 0.9999

    def test_coating_states_storage(self, basic_plot_manager):
        """Test storing coating states."""
        # Test episode -> coating states mapping
        episode = 100
        states = [Mock(), Mock()]  # Mock coating states

        basic_plot_manager.coating_states[episode] = states

        assert episode in basic_plot_manager.coating_states
        assert len(basic_plot_manager.coating_states[episode]) == 2

        # Test evaluation coating states
        eval_states = [Mock(), Mock(), Mock()]
        basic_plot_manager.eval_coating_states = eval_states

        assert len(basic_plot_manager.eval_coating_states) == 3

    def test_historical_pareto_data_storage(self, basic_plot_manager):
        """Test storing historical Pareto data."""
        episode = 1000
        pareto_points = np.array([[0.99, 1e-20], [0.995, 8e-21]])

        basic_plot_manager.historical_pareto_data[episode] = pareto_points

        assert episode in basic_plot_manager.historical_pareto_data
        assert np.array_equal(
            basic_plot_manager.historical_pareto_data[episode], pareto_points
        )

    def test_eval_data_storage(self, basic_plot_manager):
        """Test storing evaluation data."""
        eval_key = "test_eval"
        eval_results = {
            "pareto_front": np.array([[0.99, 1e-20]]),
            "hypervolume": 0.85,
            "episode": 500,
        }

        basic_plot_manager.eval_data[eval_key] = eval_results

        assert eval_key in basic_plot_manager.eval_data
        assert basic_plot_manager.eval_data[eval_key]["hypervolume"] == 0.85

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_save_plots_enabled(self, mock_close, mock_savefig, basic_plot_manager):
        """Test plot saving when enabled."""
        # This is a basic test - actual plot generation would require more setup
        # Just verify that save_plots flag affects behavior correctly
        assert basic_plot_manager.save_plots is True
        assert basic_plot_manager.output_dir is not None

    def test_save_plots_disabled(self, ui_plot_manager):
        """Test behavior when plot saving is disabled."""
        assert ui_plot_manager.save_plots is False
        # In UI mode, plots typically aren't saved to disk

    def test_figure_size_setting(self):
        """Test custom figure size setting."""
        custom_size = (14, 8)
        manager = TrainingPlotManager(save_plots=False, figure_size=custom_size)

        assert manager.figure_size == custom_size

    def test_ui_mode_vs_cli_mode(self, temp_dir):
        """Test differences between UI and CLI modes."""
        cli_manager = TrainingPlotManager(
            save_plots=True, output_dir=temp_dir, ui_mode=False
        )
        ui_manager = TrainingPlotManager(save_plots=False, ui_mode=True)

        assert cli_manager.ui_mode is False
        assert cli_manager.save_plots is True
        assert cli_manager.output_dir is not None

        assert ui_manager.ui_mode is True
        assert ui_manager.save_plots is False


class TestTrainingPlotManagerDataHandling:
    """Test data handling and manipulation in TrainingPlotManager."""

    @pytest.fixture
    def plot_manager(self):
        """Plot manager for data handling tests."""
        return TrainingPlotManager(save_plots=False)

    def test_empty_data_handling(self, plot_manager):
        """Test behavior with empty data."""
        assert len(plot_manager.training_data) == 0
        assert len(plot_manager.pareto_data) == 0
        assert len(plot_manager.eval_data) == 0
        assert len(plot_manager.coating_states) == 0
        assert len(plot_manager.eval_coating_states) == 0
        assert len(plot_manager.historical_pareto_data) == 0

    def test_large_data_volumes(self, plot_manager):
        """Test handling of large data volumes."""
        # Add many training data points
        for i in range(1000):
            training_point = {
                "episode": i * 10,
                "total_reward": np.random.random(),
                "individual_rewards": {
                    "obj1": np.random.random(),
                    "obj2": np.random.random(),
                },
                "values": {"obj1": np.random.random(), "obj2": np.random.random()},
                "weights": [np.random.random(), np.random.random()],
            }
            plot_manager.training_data.append(training_point)

        assert len(plot_manager.training_data) == 1000
        assert plot_manager.training_data[-1]["episode"] == 9990

    def test_data_consistency(self, plot_manager):
        """Test data consistency across different storage types."""
        episode = 500

        # Add training data
        training_point = {
            "episode": episode,
            "total_reward": 0.85,
            "individual_rewards": {"reflectivity": 0.9},
            "values": {"reflectivity": 0.9999},
        }
        plot_manager.training_data.append(training_point)

        # Add corresponding Pareto data
        pareto_point = {"reflectivity": 0.9999, "reward_reflectivity": 0.9}
        plot_manager.pareto_data.append(pareto_point)

        # Add historical Pareto data for same episode
        historical_point = np.array([[0.9999]])
        plot_manager.historical_pareto_data[episode] = historical_point

        # Verify consistency
        training_refl = plot_manager.training_data[0]["values"]["reflectivity"]
        pareto_refl = plot_manager.pareto_data[0]["reflectivity"]
        historical_refl = plot_manager.historical_pareto_data[episode][0][0]

        assert training_refl == pareto_refl == historical_refl

    def test_objective_info_consistency(self, plot_manager):
        """Test that objective information is consistent across components."""
        # Set up objective information
        plot_manager.objective_labels = ["Reflectivity", "Thermal Noise"]
        plot_manager.objective_scales = ["linear", "log"]
        plot_manager.optimisation_parameters = ["reflectivity", "thermal_noise"]
        plot_manager.objective_targets = {
            "reflectivity": 0.9999,
            "thermal_noise": 1e-20,
        }

        # Verify all components have consistent information
        assert len(plot_manager.objective_labels) == len(plot_manager.objective_scales)
        assert len(plot_manager.objective_labels) == len(
            plot_manager.optimisation_parameters
        )
        assert len(plot_manager.objective_targets) == len(
            plot_manager.optimisation_parameters
        )

        # Verify parameter names match between components
        for param in plot_manager.optimisation_parameters:
            assert param in plot_manager.objective_targets


class TestTrainingPlotManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_output_dir_with_save_plots(self):
        """Test that None output_dir works when save_plots=False."""
        manager = TrainingPlotManager(save_plots=False, output_dir=None)
        assert manager.output_dir is None
        assert manager.save_plots is False

    def test_missing_data_fields(self):
        """Test behavior with incomplete data structures."""
        manager = TrainingPlotManager(save_plots=False)

        # Add training data with missing fields
        incomplete_data = {
            "episode": 100,
            "total_reward": 0.85,
            # Missing individual_rewards, values, weights
        }
        manager.training_data.append(incomplete_data)

        # Should still store the data (error handling would be in plot generation)
        assert len(manager.training_data) == 1
        assert manager.training_data[0]["episode"] == 100

    def test_inconsistent_data_types(self):
        """Test behavior with inconsistent data types."""
        manager = TrainingPlotManager(save_plots=False)

        # Add data with different types
        data_point1 = {
            "episode": 100,
            "total_reward": 0.85,
            "values": {"obj1": 0.99},  # float
        }
        data_point2 = {
            "episode": 200,
            "total_reward": "0.90",  # string instead of float
            "values": {"obj1": "0.995"},  # string instead of float
        }

        manager.training_data.append(data_point1)
        manager.training_data.append(data_point2)

        # Data is stored as-is (type checking would be in plot generation)
        assert len(manager.training_data) == 2
        assert isinstance(manager.training_data[0]["total_reward"], float)
        assert isinstance(manager.training_data[1]["total_reward"], str)
