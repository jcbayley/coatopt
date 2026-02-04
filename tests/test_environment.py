"""
Unit tests for the CoatingEnvironment class.

Tests the core functionality including initialization, state management,
reward computation, step function, and multi-objective optimization.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from coatopt.environments.environment import CoatingEnvironment
from coatopt.environments.state import CoatingState
from coatopt.utils.configs import Config, DataConfig, TrainingConfig


@pytest.fixture
def materials():
    """Load materials from the default materials.json file."""
    materials_path = Path(__file__).parent.parent / "experiments" / "materials.json"
    with open(materials_path) as f:
        materials_dict = json.load(f)

    # Convert string keys to integers
    return {int(k): v for k, v in materials_dict.items()}


@pytest.fixture
def basic_config():
    """Create a basic configuration for testing."""
    data = DataConfig(
        n_layers=10,
        min_thickness=10e-9,
        max_thickness=500e-9,
        optimise_parameters=["reflectivity", "absorption"],
        optimise_targets={"reflectivity": 0.99999, "absorption": 0.0},
        objective_bounds={
            "reflectivity": [0.0, 0.99999],
            "absorption": [10000, 0],
        },
        use_intermediate_reward=False,
        ignore_air_option=False,
        ignore_substrate_option=False,
        use_optical_thickness=False,
        combine="sum",
    )
    training = TrainingConfig(cycle_weights="random")
    return Config(data=data, training=training)


@pytest.fixture
def single_objective_config():
    """Create a single-objective configuration for testing."""
    data = DataConfig(
        n_layers=8,
        min_thickness=10e-9,
        max_thickness=500e-9,
        optimise_parameters=["reflectivity"],
        optimise_targets={"reflectivity": 0.99999},
        objective_bounds={"reflectivity": [0.0, 0.99999]},
        use_intermediate_reward=False,
        combine="sum",
    )
    return Config(data=data, training=TrainingConfig())


class TestCoatingEnvironmentInitialization:
    """Test environment initialization."""

    def test_init_with_materials(self, basic_config, materials):
        """Test basic initialization with materials."""
        env = CoatingEnvironment(basic_config, materials)

        assert env.max_layers == 10
        assert env.min_thickness == 10e-9
        assert env.max_thickness == 500e-9
        assert env.n_materials == len(materials)
        assert env.materials == materials
        assert env.air_material_index == 0
        assert env.substrate_material_index == 1

    def test_init_multi_objective(self, basic_config, materials):
        """Test multi-objective configuration is detected."""
        env = CoatingEnvironment(basic_config, materials)

        assert env.multi_objective is True
        assert len(env.optimise_parameters) == 2
        assert "reflectivity" in env.optimise_parameters
        assert "absorption" in env.optimise_parameters

    def test_init_single_objective(self, single_objective_config, materials):
        """Test single-objective configuration."""
        env = CoatingEnvironment(single_objective_config, materials)

        assert env.multi_objective is False
        assert len(env.optimise_parameters) == 1
        assert env.optimise_parameters[0] == "reflectivity"

    def test_objective_directions(self, basic_config, materials):
        """Test objective directions are set correctly."""
        env = CoatingEnvironment(basic_config, materials)

        assert env.objective_directions["reflectivity"] is True  # Maximize
        assert env.objective_directions["absorption"] is False  # Minimize
        assert env.objective_directions["thermal_noise"] is False  # Minimize

    def test_reward_bounds_initialization(self, basic_config, materials):
        """Test reward bounds are computed from objective bounds."""
        env = CoatingEnvironment(basic_config, materials)

        assert "reflectivity" in env.reward_bounds
        assert "absorption" in env.reward_bounds
        assert len(env.reward_bounds["reflectivity"]) == 2
        assert len(env.reward_bounds["absorption"]) == 2

    def test_observation_space_shape(self, basic_config, materials):
        """Test observation space shape is computed correctly."""
        env = CoatingEnvironment(basic_config, materials)

        n_materials = len(materials)
        features_per_layer = 1 + n_materials + 2  # thickness + materials + n + k
        expected_shape = (10, features_per_layer, 0)

        assert env.obs_space_shape == expected_shape


class TestCoatingEnvironmentReset:
    """Test environment reset functionality."""

    def test_reset_creates_clean_state(self, basic_config, materials):
        """Test reset creates a new clean state."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        assert isinstance(state, CoatingState)
        assert state.max_layers == env.max_layers
        assert state.n_materials == env.n_materials
        assert env.current_index == 0
        assert env.done is False

    def test_reset_after_episode(self, basic_config, materials):
        """Test reset works after completing an episode."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        # Simulate episode
        env.current_index = 5
        env.done = True

        # Reset
        state = env.reset()
        assert env.current_index == 0
        assert env.done is False

    def test_multiple_resets(self, basic_config, materials):
        """Test multiple consecutive resets."""
        env = CoatingEnvironment(basic_config, materials)

        state1 = env.reset()
        state2 = env.reset()

        # Both should be clean states
        assert state1.get_num_active_layers() == 0
        assert state2.get_num_active_layers() == 0


class TestCoatingEnvironmentStep:
    """Test step function."""

    def test_step_with_agent_format_action(self, basic_config, materials):
        """Test step with agent format action [thickness, material_probs...]."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        # Agent format: [thickness, material_0_prob, material_1_prob, ...]
        n_materials = len(materials)
        action = np.zeros(n_materials + 1)
        action[0] = 100e-9  # thickness
        action[2] = 1.0  # Select material index 1 (material_idx = argmax of action[1:])

        state, rewards, terminated, finished, total_reward, full_action, vals = (
            env.step(action)
        )

        assert isinstance(state, CoatingState)
        assert isinstance(rewards, dict)
        assert isinstance(vals, dict)
        assert finished is False  # Not done yet
        assert env.current_index == 1
        assert full_action == [1, 100e-9]  # [material, thickness]

    def test_step_with_base_format_action(self, basic_config, materials):
        """Test step with base environment format action [material, thickness]."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        action = np.array([2, 150e-9])  # material 2, 150nm

        state, rewards, terminated, finished, total_reward, full_action, vals = (
            env.step(action)
        )

        assert env.current_index == 1
        assert full_action == [2, 150e-9]

    def test_step_thickness_clamping(self, basic_config, materials):
        """Test that thickness is clamped to valid range."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        # Try thickness outside bounds
        action = np.array([2, 1000e-9])  # Too thick

        state, rewards, terminated, finished, total_reward, full_action, vals = (
            env.step(action)
        )

        # Should be clamped to max_thickness
        assert full_action[1] <= env.max_thickness

    def test_step_termination_on_air(self, basic_config, materials):
        """Test episode terminates when air layer is selected."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        # Select air material (index 0)
        action = np.array([0, 100e-9])

        state, rewards, terminated, finished, total_reward, full_action, vals = (
            env.step(action)
        )

        assert finished is True
        assert env.done is True

    def test_step_termination_on_max_layers(self, basic_config, materials):
        """Test episode terminates when max layers reached."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        # Add max_layers - 1 layers
        for i in range(env.max_layers - 1):
            action = np.array([2, 100e-9])
            state, rewards, terminated, finished, total_reward, full_action, vals = (
                env.step(action)
            )
            assert finished is False

        # Add final layer
        action = np.array([2, 100e-9])
        state, rewards, terminated, finished, total_reward, full_action, vals = (
            env.step(action)
        )

        assert finished is True
        assert env.done is True

    def test_step_raises_error_when_done(self, basic_config, materials):
        """Test step raises error when episode is already done."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()
        env.done = True

        with pytest.raises(RuntimeError, match="Episode is done"):
            env.step(np.array([2, 100e-9]))

    def test_step_without_intermediate_reward(self, basic_config, materials):
        """Test that reward is 0 for intermediate steps when use_intermediate_reward=False."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        action = np.array([2, 100e-9])
        state, rewards, terminated, finished, total_reward, full_action, vals = (
            env.step(action)
        )

        # Should have zero reward for intermediate step
        assert total_reward == 0.0
        assert vals == {}

    def test_step_with_intermediate_reward(self, basic_config, materials):
        """Test that reward is computed for intermediate steps when use_intermediate_reward=True."""
        basic_config.data.use_intermediate_reward = True
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        action = np.array([2, 100e-9])
        state, rewards, terminated, finished, total_reward, full_action, vals = (
            env.step(action)
        )

        # Should have non-zero values
        assert "reflectivity" in vals
        assert "absorption" in vals


class TestCoatingEnvironmentRewards:
    """Test reward computation."""

    def test_compute_reward_basic(self, single_objective_config, materials):
        """Test basic reward computation."""
        env = CoatingEnvironment(single_objective_config, materials)
        state = env.reset()

        # Add some layers
        state.set_layer(0, 100e-9, 2)  # SiO2
        state.set_layer(1, 100e-9, 3)  # aSi

        individual_rewards, vals = env.compute_reward(state, normalised=False)

        assert "reflectivity" in individual_rewards
        assert "reflectivity" in vals
        assert isinstance(individual_rewards["reflectivity"], float)
        assert isinstance(vals["reflectivity"], (float, np.floating))

    def test_compute_reward_normalised(self, basic_config, materials):
        """Test normalized reward computation."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        state.set_layer(0, 100e-9, 2)
        state.set_layer(1, 100e-9, 3)

        individual_rewards, vals = env.compute_reward(state, normalised=True)

        # Normalized rewards should be in a reasonable range
        for param, reward in individual_rewards.items():
            assert isinstance(reward, float)
            # May be outside [0, 1] if outside objective bounds, but should be reasonable
            assert -100 < reward < 100

    def test_compute_reward_with_numpy_state(self, basic_config, materials):
        """Test reward computation with numpy array state."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        state.set_layer(0, 100e-9, 2)

        # Convert to numpy array
        state_array = state.get_array()

        individual_rewards, vals = env.compute_reward(state_array, normalised=False)

        assert "reflectivity" in individual_rewards
        assert "absorption" in individual_rewards

    def test_compute_objective_rewards(self, basic_config, materials):
        """Test objective-specific reward computation."""
        env = CoatingEnvironment(basic_config, materials)

        # Mock objective values
        vals = {
            "reflectivity": 0.95,
            "absorption": 100.0,
            "thermal_noise": 1e-19,
        }

        rewards = env.compute_objective_rewards(vals, normalised=False)

        assert "reflectivity" in rewards
        assert "absorption" in rewards
        assert isinstance(rewards["reflectivity"], float)

    def test_compute_training_reward_standard_mode(self, basic_config, materials):
        """Test training reward computation in standard mode."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        state.set_layer(0, 100e-9, 2)

        objective_weights = {"reflectivity": 0.6, "absorption": 0.4}
        total_reward, vals, individual_rewards = env.compute_training_reward(
            state, objective_weights=objective_weights
        )

        assert isinstance(total_reward, (float, np.floating))
        assert "reflectivity" in vals
        assert "absorption" in vals
        assert "reflectivity" in individual_rewards


class TestCoatingEnvironmentStateValue:
    """Test state value computation (physics)."""

    def test_compute_state_value_basic(self, basic_config, materials):
        """Test basic state value computation."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        # Add a simple coating
        state.set_layer(0, 100e-9, 2)  # SiO2
        state.set_layer(1, 100e-9, 3)  # aSi

        reflectivity, thermal_noise, absorption, thickness = env.compute_state_value(
            state
        )

        assert isinstance(reflectivity, (float, np.floating))
        assert isinstance(absorption, (float, np.floating))
        assert isinstance(thickness, (float, np.floating))
        # thermal_noise can be None if not computed
        assert thermal_noise is None or isinstance(thermal_noise, (float, np.floating))

    def test_compute_state_value_empty_coating(self, basic_config, materials):
        """Test state value for empty coating (all air)."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        # Don't add any layers (all air)
        reflectivity, thermal_noise, absorption, thickness = env.compute_state_value(
            state
        )

        # Empty coating returns substrate properties
        assert isinstance(reflectivity, (float, np.floating))
        assert thermal_noise is None or np.isnan(thermal_noise)
        assert isinstance(absorption, (float, np.floating))
        assert isinstance(thickness, (float, np.floating))

    def test_compute_state_value_with_field_data(self, basic_config, materials):
        """Test state value computation with field data."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        state.set_layer(0, 100e-9, 2)

        result = env.compute_state_value(state, return_field_data=True)

        assert len(result) == 5  # (r, thermal, absorption, thickness, field_data)


class TestCoatingEnvironmentActionSpace:
    """Test action space sampling."""

    def test_sample_action_space_basic(self, basic_config, materials):
        """Test basic action space sampling."""
        env = CoatingEnvironment(basic_config, materials)

        action = env.sample_action_space()

        assert len(action) == env.n_materials + 1
        assert env.min_thickness <= action[0] <= env.max_thickness
        # One material should be selected (one-hot)
        assert np.sum(action[1:]) == 1.0

    def test_sample_action_space_ignores_air(self, basic_config, materials):
        """Test action sampling with ignore_air_option enabled."""
        basic_config.data.ignore_air_option = True
        env = CoatingEnvironment(basic_config, materials)

        # Sample multiple times to ensure air is never selected
        for _ in range(20):
            action = env.sample_action_space()
            material_idx = np.argmax(action[1:])
            assert material_idx != env.air_material_index

    def test_sample_action_space_ignores_substrate(self, basic_config, materials):
        """Test action sampling with ignore_substrate_option enabled."""
        basic_config.data.ignore_substrate_option = True
        env = CoatingEnvironment(basic_config, materials)

        for _ in range(20):
            action = env.sample_action_space()
            material_idx = np.argmax(action[1:])
            assert material_idx != env.substrate_material_index

    def test_sample_action_space_optical_thickness(self, basic_config, materials):
        """Test action sampling with optical thickness mode."""
        basic_config.data.use_optical_thickness = True
        env = CoatingEnvironment(basic_config, materials)

        action = env.sample_action_space()

        # Optical thickness should be in [0.01, 1.0]
        assert 0.01 <= action[0] <= 1.0


class TestCoatingEnvironmentParetoFront:
    """Test Pareto front tracking."""

    def test_update_pareto_front_single_objective(
        self, single_objective_config, materials
    ):
        """Test Pareto front with single objective (should not update)."""
        env = CoatingEnvironment(single_objective_config, materials)
        state = env.reset()

        state.set_layer(0, 100e-9, 2)

        objectives = {"reflectivity": 0.95}
        env.update_pareto_front(objectives, state)

        # Single objective should not maintain pareto front
        assert len(env.pareto_front_rewards) == 0

    def test_update_pareto_front_first_point(self, basic_config, materials):
        """Test adding first point to Pareto front."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        state.set_layer(0, 100e-9, 2)

        objectives = {"reflectivity": 0.95, "absorption": 100.0}
        env.update_pareto_front(objectives, state)

        assert len(env.pareto_front_rewards) == 1
        assert len(env.pareto_front_values) == 1

    def test_update_pareto_front_dominated_point(self, basic_config, materials):
        """Test that dominated points are not added."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        state.set_layer(0, 100e-9, 2)

        # Add a good point
        objectives1 = {"reflectivity": 0.99, "absorption": 10.0}
        env.update_pareto_front(objectives1, state)

        # Add a dominated point (worse in both objectives)
        objectives2 = {"reflectivity": 0.90, "absorption": 100.0}
        env.update_pareto_front(objectives2, state)

        # Should still have only 1 point
        assert len(env.pareto_front_rewards) == 1

    def test_update_pareto_front_removes_dominated(self, basic_config, materials):
        """Test that new point removes dominated points."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        # Add first point
        objectives1 = {"reflectivity": 0.90, "absorption": 100.0}
        env.update_pareto_front(objectives1, state)

        # Add dominating point
        objectives2 = {"reflectivity": 0.99, "absorption": 10.0}
        env.update_pareto_front(objectives2, state)

        # Should have only the new point
        assert len(env.pareto_front_rewards) == 1

    def test_get_pareto_front_reward_space(self, basic_config, materials):
        """Test getting Pareto front in reward space."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        objectives = {"reflectivity": 0.95, "absorption": 50.0}
        env.update_pareto_front(objectives, state)

        pareto_front = env.get_pareto_front(space="reward")

        assert len(pareto_front) == 1
        assert len(pareto_front[0]) == 2  # (reward_vector, state)

    def test_get_pareto_front_value_space(self, basic_config, materials):
        """Test getting Pareto front in value space."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        objectives = {"reflectivity": 0.95, "absorption": 50.0}
        env.update_pareto_front(objectives, state)

        pareto_front = env.get_pareto_front(space="value")

        assert len(pareto_front) == 1


class TestCoatingEnvironmentConstrainedTraining:
    """Test constrained training functionality."""

    def test_enable_constrained_training(self, basic_config, materials):
        """Test enabling constrained training mode."""
        env = CoatingEnvironment(basic_config, materials)

        env.enable_constrained_training(
            warmup_episodes_per_objective=100,
            steps_per_objective=5,
            epochs_per_step=200,
            constraint_penalty=15.0,
        )

        assert env.use_constrained_training is True
        assert env.warmup_episodes_per_objective == 100
        assert env.steps_per_objective == 5
        assert env.constraint_penalty == 15.0

    def test_compute_constraint_penalty(self, basic_config, materials):
        """Test constraint penalty computation."""
        env = CoatingEnvironment(basic_config, materials)
        env.constraint_penalty = 10.0
        env.constraints = {"reflectivity": 0.5}  # Require normalized reward >= 0.5

        vals = {"reflectivity": 0.90, "absorption": 50.0}
        base_rewards = {
            "reflectivity": 0.3,
            "absorption": 0.7,
        }  # reflectivity violates constraint

        penalty = env._compute_constraint_penalty(vals, base_rewards)

        # Should have penalty for reflectivity constraint violation
        assert penalty > 0

    def test_update_warmup_best(self, basic_config, materials, capsys):
        """Test warmup best reward tracking."""
        env = CoatingEnvironment(basic_config, materials)

        env.update_warmup_best("reflectivity", 0.8)
        captured = capsys.readouterr()

        assert env.warmup_best_rewards["reflectivity"] == 0.8

        # Update with better value
        env.update_warmup_best("reflectivity", 0.9)
        assert env.warmup_best_rewards["reflectivity"] == 0.9


class TestCoatingEnvironmentParetoBonus:
    """Test Pareto dominance bonus."""

    def test_enable_pareto_bonus(self, basic_config, materials):
        """Test enabling Pareto bonus."""
        env = CoatingEnvironment(basic_config, materials)

        env.enable_pareto_bonus(bonus=2.0)

        assert env.use_pareto_bonus is True
        assert env.pareto_dominance_bonus == 2.0

    def test_compute_pareto_dominance_bonus_no_front(self, basic_config, materials):
        """Test Pareto bonus with empty front."""
        env = CoatingEnvironment(basic_config, materials)
        env.enable_pareto_bonus(bonus=1.0)

        vals = {"reflectivity": 0.95, "absorption": 50.0}
        bonus = env._compute_pareto_dominance_bonus(vals)

        # With hypervolume-based bonus, first point creates hypervolume improvement from 0
        assert bonus > 0.0

    def test_compute_pareto_dominance_bonus_with_dominated_points(
        self, basic_config, materials
    ):
        """Test Pareto bonus when dominating existing points."""
        env = CoatingEnvironment(basic_config, materials)
        env.enable_pareto_bonus(bonus=1.0)

        state = env.reset()

        # Add some inferior points to pareto front
        objectives1 = {"reflectivity": 0.90, "absorption": 100.0}
        env.update_pareto_front(objectives1, state)

        # Compute bonus for dominating point
        vals = {"reflectivity": 0.99, "absorption": 10.0}
        bonus = env._compute_pareto_dominance_bonus(vals)

        # Bonus is based on hypervolume improvement, should be positive
        assert bonus > 0.0


class TestCoatingEnvironmentStateMethods:
    """Test state getter/setter methods."""

    def test_get_state(self, basic_config, materials):
        """Test getting current state."""
        env = CoatingEnvironment(basic_config, materials)
        state = env.reset()

        retrieved_state = env.get_state()

        assert retrieved_state is state

    def test_set_state(self, basic_config, materials):
        """Test setting state."""
        env = CoatingEnvironment(basic_config, materials)
        env.reset()

        # Create new state
        new_state = CoatingState(
            max_layers=10,
            n_materials=len(materials),
            air_material_index=0,
            substrate_material_index=1,
            materials=materials,
        )
        new_state.set_layer(0, 200e-9, 2)

        env.set_state(new_state)

        # Verify state was copied
        retrieved = env.get_state()
        thickness, material = retrieved.get_layer(0)
        assert np.isclose(thickness, 200e-9)
        assert material == 2

    def test_get_parameter_names(self, basic_config, materials):
        """Test getting parameter names."""
        env = CoatingEnvironment(basic_config, materials)

        params = env.get_parameter_names()

        assert params == ["reflectivity", "absorption"]


class TestCoatingEnvironmentDominance:
    """Test dominance checking."""

    def test_dominates_reward_space(self, basic_config, materials):
        """Test dominance check in reward space (always maximize)."""
        import numpy as np

        from coatopt.utils.metrics import dominates

        obj1 = np.array([0.8, 0.9])  # Better in both
        obj2 = np.array([0.5, 0.6])

        assert dominates(obj1, obj2, maximize=True) is True
        assert dominates(obj2, obj1, maximize=True) is False

    def test_dominates_value_space(self, basic_config, materials):
        """Test dominance check in value space (use objective_directions)."""
        import numpy as np

        from coatopt.utils.metrics import dominates_mixed

        # reflectivity: maximize, absorption: minimize
        obj1 = np.array([0.99, 10.0])  # Better reflectivity, better absorption
        obj2 = np.array([0.95, 50.0])

        objective_directions = [True, False]  # reflectivity: max, absorption: min
        assert dominates_mixed(obj1, obj2, objective_directions) is True

    def test_dominates_non_dominated(self, basic_config, materials):
        """Test that non-dominated points are correctly identified."""
        import numpy as np

        from coatopt.utils.metrics import dominates_mixed

        # One better in first objective, other better in second
        obj1 = np.array([0.99, 100.0])
        obj2 = np.array([0.90, 10.0])

        objective_directions = [True, False]  # reflectivity: max, absorption: min
        assert dominates_mixed(obj1, obj2, objective_directions) is False
        assert dominates_mixed(obj2, obj1, objective_directions) is False

    def test_dominates_equal_points(self, basic_config, materials):
        """Test that equal points don't dominate each other."""
        import numpy as np

        from coatopt.utils.metrics import dominates

        obj1 = np.array([0.95, 50.0])
        obj2 = np.array([0.95, 50.0])

        assert dominates(obj1, obj2, maximize=True) is False


class TestCoatingEnvironmentObservedBounds:
    """Test observed value bounds tracking."""

    def test_update_observed_bounds(self, basic_config, materials):
        """Test updating observed bounds during training."""
        env = CoatingEnvironment(basic_config, materials)

        vals1 = {"reflectivity": 0.90, "absorption": 100.0}
        env.update_observed_bounds(vals1)

        assert env.observed_value_bounds["reflectivity"]["min"] == 0.90
        assert env.observed_value_bounds["reflectivity"]["max"] == 0.90
        assert env.observed_value_bounds["absorption"]["min"] == 100.0

        # Update with different values
        vals2 = {"reflectivity": 0.95, "absorption": 50.0}
        env.update_observed_bounds(vals2)

        assert env.observed_value_bounds["reflectivity"]["min"] == 0.90
        assert env.observed_value_bounds["reflectivity"]["max"] == 0.95
        assert env.observed_value_bounds["absorption"]["min"] == 50.0

    def test_update_observed_bounds_handles_nan(self, basic_config, materials):
        """Test that NaN values are ignored when updating bounds."""
        env = CoatingEnvironment(basic_config, materials)

        vals = {"reflectivity": np.nan, "absorption": 100.0}
        env.update_observed_bounds(vals)

        # reflectivity bounds should remain at infinity
        assert env.observed_value_bounds["reflectivity"]["min"] == np.inf
        assert env.observed_value_bounds["absorption"]["min"] == 100.0
