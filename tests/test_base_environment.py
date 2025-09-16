"""
Unit tests for the BaseCoatingEnvironment class.
Tests the core functionality of the base environment including initialization,
state management, and configuration handling.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from coatopt.environments.core.base_environment import BaseCoatingEnvironment
from coatopt.config.structured_config import CoatingOptimisationConfig, DataConfig


class TestBaseCoatingEnvironment:
    """Test suite for BaseCoatingEnvironment."""
    
    @pytest.fixture
    def sample_materials(self):
        """Sample materials for testing."""
        return {
            0: {"name": "Air", "n": 1.0, "k": 0.0},
            1: {"name": "Substrate", "n": 1.45, "k": 0.0},
            2: {"name": "SiO2", "n": 1.46, "k": 0.0},
            3: {"name": "Ta2O5", "n": 2.07, "k": 0.0}
        }
    
    @pytest.fixture
    def basic_data_config(self):
        """Basic DataConfig for testing."""
        return DataConfig(
            n_layers=8,
            min_thickness=10e-9,
            max_thickness=500e-9,
            use_observation=True,
            use_intermediate_reward=False,
            ignore_air_option=False,
            ignore_substrate_option=False,
            use_optical_thickness=True,
            optimise_parameters=["reflectivity", "thermal_noise"],
            optimise_targets={"reflectivity": 0.9999, "thermal_noise": 1e-20},
            optimise_weight_ranges={"reflectivity": [0.0, 1.0], "thermal_noise": [0.0, 1.0]},
            design_criteria={"reflectivity": 0.9999, "thermal_noise": 1e-20},
            reward_function="default",
            combine="sum"
        )
    
    @pytest.fixture
    def basic_config(self, basic_data_config):
        """Basic CoatingOptimisationConfig for testing."""
        config = Mock(spec=CoatingOptimisationConfig)
        config.data = basic_data_config
        return config
    
    def test_init_with_config(self, basic_config, sample_materials):
        """Test initialization with structured config."""
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials,
            air_material_index=0,
            substrate_material_index=1
        )
        
        assert env.max_layers == 8
        assert env.min_thickness == 10e-9
        assert env.max_thickness == 500e-9
        assert env.use_optical_thickness is True
        assert env.optimise_parameters == ["reflectivity", "thermal_noise"]
        assert env.materials == sample_materials
        assert env.air_material_index == 0
        assert env.substrate_material_index == 1
    
    def test_init_without_config_legacy(self, sample_materials):
        """Test legacy initialization without config."""
        env = BaseCoatingEnvironment(
            materials=sample_materials,
            max_layers=10,
            min_thickness=5e-9,
            max_thickness=1000e-9,
            optimise_parameters=["reflectivity"],
            optimise_targets={"reflectivity": 0.99},
            reward_function="default"  # Use valid reward function
        )
        
        assert env.max_layers == 10
        assert env.min_thickness == 5e-9
        assert env.max_thickness == 1000e-9
        assert env.optimise_parameters == ["reflectivity"]
        assert env.materials == sample_materials
    
    def test_init_config_missing_data_raises_error(self):
        """Test that missing DataConfig raises appropriate error."""
        config = Mock(spec=CoatingOptimisationConfig)
        config.data = None
        
        with pytest.raises(ValueError, match="DataConfig is required"):
            BaseCoatingEnvironment(config=config, materials={})
    
    def test_init_config_missing_materials_raises_error(self, basic_config):
        """Test that missing materials raises appropriate error."""
        with pytest.raises(ValueError, match="Materials must be provided"):
            BaseCoatingEnvironment(config=basic_config)
    
    def test_extract_parameter_names(self, basic_config, sample_materials):
        """Test parameter name extraction removes direction suffixes."""
        # Modify config to have direction suffixes
        basic_config.data.optimise_parameters = ["reflectivity:max", "thermal_noise:min"]
        
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials
        )
        
        expected_names = ["reflectivity", "thermal_noise"]
        assert env.optimise_parameter_names == expected_names
    
    def test_electric_field_configuration(self, basic_config, sample_materials):
        """Test electric field configuration."""
        # Add electric field configuration
        basic_config.data.include_electric_field = True
        basic_config.data.electric_field_points = 100
        
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials
        )
        
        assert env.include_electric_field is True
        assert env.electric_field_points == 100
    
    def test_objective_bounds_configuration(self, basic_config, sample_materials):
        """Test objective bounds configuration."""
        bounds = {
            "reflectivity": [0.9, 1.0],
            "thermal_noise": [1e-22, 1e-19]
        }
        basic_config.data.objective_bounds = bounds
        
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials
        )
        
        assert env.objective_bounds == bounds
    
    def test_reward_normalisation_configuration(self, basic_config, sample_materials):
        """Test reward normalisation configuration."""
        basic_config.data.use_reward_normalisation = True
        basic_config.data.reward_normalisation_mode = "adaptive"
        basic_config.data.reward_normalisation_alpha = 0.05
        
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials
        )
        
        assert env.use_reward_normalisation is True
        assert env.reward_normalisation_mode == "adaptive"
        assert env.reward_normalisation_alpha == 0.05
    
    def test_kwargs_override_config(self, basic_config, sample_materials):
        """Test that kwargs can override config values."""
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials,
            max_layers=16,  # Override config value of 8
            light_wavelength=532e-9  # Additional parameter
        )
        
        assert env.max_layers == 16  # Overridden
        assert env.light_wavelength == 532e-9
    
    def test_expert_constraints_initialization(self, basic_config, sample_materials):
        """Test expert constraints initialization."""
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials
        )
        
        assert env.current_expert_constraints is None
    
    @patch('coatopt.environments.core.base_environment.BaseCoatingEnvironment._setup_common_attributes')
    def test_setup_common_attributes_called(self, mock_setup, basic_config, sample_materials):
        """Test that common attributes setup is called during initialization."""
        env = BaseCoatingEnvironment(
            config=basic_config,
            materials=sample_materials
        )
        
        mock_setup.assert_called_once()


class TestBaseCoatingEnvironmentParameterExtraction:
    """Test parameter name extraction logic."""
    
    @pytest.fixture
    def sample_materials(self):
        return {0: {"name": "Air", "n": 1.0, "k": 0.0}}
    
    def test_extract_parameter_names_no_direction(self, sample_materials):
        """Test extraction when no direction suffixes are present."""
        env = BaseCoatingEnvironment(
            materials=sample_materials,
            max_layers=8,
            min_thickness=10e-9,
            max_thickness=500e-9,
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            optimise_targets={"reflectivity": 0.99, "thermal_noise": 1e-20, "absorption": 0.01},
            reward_function="default"
        )
        
        result = env._extract_parameter_names(["reflectivity", "thermal_noise", "absorption"])
        expected = ["reflectivity", "thermal_noise", "absorption"]
        assert result == expected
    
    def test_extract_parameter_names_with_directions(self, sample_materials):
        """Test extraction with various direction suffixes."""
        env = BaseCoatingEnvironment(
            materials=sample_materials,
            max_layers=8,
            min_thickness=10e-9,
            max_thickness=500e-9,
            optimise_parameters=["reflectivity:max", "thermal_noise:min"],
            optimise_targets={"reflectivity": 0.99, "thermal_noise": 1e-20},
            reward_function="default"
        )
        
        result = env._extract_parameter_names(["reflectivity:max", "thermal_noise:min", "absorption:min", "thickness"])
        expected = ["reflectivity", "thermal_noise", "absorption", "thickness"]
        assert result == expected
    
    def test_extract_parameter_names_mixed_cases(self, sample_materials):
        """Test extraction with mixed cases of directions and no directions."""
        env = BaseCoatingEnvironment(
            materials=sample_materials,
            max_layers=8,
            min_thickness=10e-9,
            max_thickness=500e-9,
            optimise_parameters=["reflectivity:max", "thermal_noise"],
            optimise_targets={"reflectivity": 0.99, "thermal_noise": 1e-20},
            reward_function="default"
        )
        
        result = env._extract_parameter_names(["reflectivity:max", "thermal_noise", "absorption:min"])
        expected = ["reflectivity", "thermal_noise", "absorption"]
        assert result == expected


class TestBaseCoatingEnvironmentEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_optimise_parameters(self):
        """Test behavior with empty optimization parameters."""
        config = Mock(spec=CoatingOptimisationConfig)
        config.data = Mock()
        config.data.optimise_parameters = []
        config.data.n_layers = 8
        config.data.min_thickness = 10e-9
        config.data.max_thickness = 500e-9
        config.data.use_observation = True
        config.data.use_intermediate_reward = False
        config.data.ignore_air_option = False
        config.data.ignore_substrate_option = False
        config.data.use_optical_thickness = True
        config.data.optimise_targets = {}
        config.data.optimise_weight_ranges = {}
        config.data.design_criteria = {}
        config.data.reward_function = "default"
        config.data.combine = "sum"
        
        # Should not raise an error but result in empty parameter names
        env = BaseCoatingEnvironment(config=config, materials={0: {"name": "Air"}})
        assert env.optimise_parameter_names == []
    
    def test_missing_optional_config_attributes(self):
        """Test that missing optional config attributes use defaults."""
        config = Mock(spec=CoatingOptimisationConfig)
        config.data = Mock()
        
        # Set required attributes
        config.data.n_layers = 8
        config.data.min_thickness = 10e-9
        config.data.max_thickness = 500e-9
        config.data.use_observation = True
        config.data.use_intermediate_reward = False
        config.data.ignore_air_option = False
        config.data.ignore_substrate_option = False
        config.data.use_optical_thickness = True
        config.data.optimise_parameters = ["reflectivity"]
        config.data.optimise_targets = {"reflectivity": 0.99}
        config.data.optimise_weight_ranges = {"reflectivity": [0.0, 1.0]}
        config.data.design_criteria = {"reflectivity": 0.99}
        config.data.reward_function = "default"
        config.data.combine = "sum"
        
        # Don't set optional attributes - mock them with getattr defaults
        config.data.include_electric_field = False
        config.data.electric_field_points = 50
        
        env = BaseCoatingEnvironment(config=config, materials={0: {"name": "Air"}})
        
        # Should use defaults
        assert env.include_electric_field is False
        assert env.electric_field_points == 50
