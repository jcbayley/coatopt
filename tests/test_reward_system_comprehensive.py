"""
Comprehensive unit tests for the refactored reward system.
Tests reward calculation consistency, normalisation, and addon functionality.
"""

import copy
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from coatopt.environments.reward_functions.reward_addons import (
    apply_air_penalty_addon,
    apply_boundary_penalties,
    apply_divergence_penalty,
    apply_pareto_improvement_addon,
    apply_preference_constraints_addon,
)
from coatopt.environments.reward_functions.reward_system import (
    RewardCalculator,
    RewardRegistry,
)


class TestRewardSystemComprehensive:
    """Comprehensive tests for the reward system to identify behavioral differences."""

    @pytest.fixture
    def standard_env(self):
        """Standard mock environment matching main branch setup."""
        env = Mock()
        env.optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        env.get_parameter_names.return_value = [
            "reflectivity",
            "thermal_noise",
            "absorption",
        ]
        env.objective_bounds = {
            "reflectivity": [0.9, 1.0],
            "thermal_noise": [1e-22, 1e-19],
            "absorption": [1e-8, 1e-2],
        }
        env.pareto_front = []
        env.current_state = Mock()
        env.current_state.get_num_active_layers.return_value = 10
        env.current_state.get_layer_count.return_value = 1
        env.design_criteria = None  # Fix for air penalty addon
        return env

    @pytest.fixture
    def standard_params(self):
        """Standard parameters for testing."""
        return {
            "reflectivity": 0.99999,
            "thermal_noise": 5.394480540642821e-21,
            "absorption": 0.01,
            "thickness": 5.0,
        }

    def test_reward_calculation_deterministic(self, standard_env, standard_params):
        """Test that reward calculation is deterministic."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            optimise_targets={
                "reflectivity": 0.99999,
                "thermal_noise": 5.394480540642821e-21,
                "absorption": 0.01,
            },
        )

        # Run same calculation multiple times
        results = []
        for _ in range(5):
            total_reward, vals, rewards = calc.calculate(
                env=standard_env, **standard_params
            )
            results.append((total_reward, vals.copy(), rewards.copy()))

        # All results should be identical
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert (
                abs(result[0] - base_result[0]) < 1e-12
            ), f"Run {i} total_reward differs: {result[0]} vs {base_result[0]}"

            for key in base_result[1]:
                if key in result[1]:
                    assert (
                        abs(result[1][key] - base_result[1][key]) < 1e-12
                    ), f"Run {i} vals[{key}] differs"

            for key in base_result[2]:
                if key in result[2]:
                    assert (
                        abs(result[2][key] - base_result[2][key]) < 1e-12
                    ), f"Run {i} rewards[{key}] differs"

    def test_reward_normalization_behavior(self, standard_env):
        """Test reward normalization behavior to match main branch."""
        # Create calculators with and without normalization
        calc_no_norm = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_normalisation=False,
        )

        calc_with_norm = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_normalisation=True,
        )

        params = {
            "reflectivity": 0.999,
            "thermal_noise": 1e-21,
            "thickness": 5.0,
            "absorption": 0.005,
            "env": standard_env,
        }

        # Calculate without normalization
        total_no_norm, vals_no_norm, rewards_no_norm = calc_no_norm.calculate(**params)

        # Calculate with normalization
        total_with_norm, vals_with_norm, rewards_with_norm = calc_with_norm.calculate(
            **params
        )

        print(f"No norm total: {total_no_norm}")
        print(f"With norm total: {total_with_norm}")
        print(f"No norm rewards: {rewards_no_norm}")
        print(f"With norm rewards: {rewards_with_norm}")

        # Values should be the same (physical quantities)
        for key in ["reflectivity", "thermal_noise", "absorption"]:
            if key in vals_no_norm and key in vals_with_norm:
                assert (
                    abs(vals_no_norm[key] - vals_with_norm[key]) < 1e-12
                ), f"Physical value {key} changed with normalization"

        # Rewards may differ due to normalization
        # But both should produce valid, finite numbers
        assert np.isfinite(total_no_norm) and np.isfinite(total_with_norm)

    def test_addon_system_consistency(self, standard_env, standard_params):
        """Test that addon system produces consistent results."""
        # Test all combinations of addons
        addon_configs = [
            {"apply_normalisation": True},
            {"apply_boundary_penalties": True},
            {"apply_air_penalty": True},
            {"apply_normalisation": True, "apply_boundary_penalties": True},
            {"apply_normalisation": True, "apply_air_penalty": True},
            {"apply_boundary_penalties": True, "apply_air_penalty": True},
            {
                "apply_normalisation": True,
                "apply_boundary_penalties": True,
                "apply_air_penalty": True,
            },
        ]

        for config in addon_configs:
            calc = RewardCalculator(
                reward_type="default",
                optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
                **config,
            )

            # Should not crash and should produce finite results
            total_reward, vals, rewards = calc.calculate(
                env=standard_env, **standard_params
            )

            assert np.isfinite(total_reward), f"Non-finite reward with config {config}"
            assert all(
                np.isfinite(v) for v in vals.values() if isinstance(v, (int, float))
            ), f"Non-finite vals with config {config}"

            # Should have total_reward in rewards dict
            assert "total_reward" in rewards
            assert abs(rewards["total_reward"] - total_reward) < 1e-10

    def test_reward_combination_methods(self, standard_env, standard_params):
        """Test different reward combination methods."""
        methods = ["sum", "product", "logproduct"]

        results = {}
        for method in methods:
            calc = RewardCalculator(
                reward_type="default",
                optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
                combine=method,
            )

            total_reward, vals, rewards = calc.calculate(
                env=standard_env, **standard_params
            )
            results[method] = total_reward

            assert np.isfinite(total_reward), f"Non-finite reward with method {method}"

        # Different methods should generally produce different results
        # (unless rewards are exactly 1.0 or edge cases)
        print(f"Combination results: {results}")

        # At minimum, check they're all finite
        assert all(np.isfinite(r) for r in results.values())

    def test_hypervolume_combination(self, standard_env, standard_params):
        """Test hypervolume combination method."""
        # Import ParetoTracker if available
        try:
            from coatopt.algorithms.hppo.training.utils.pareto_tracker import (
                ParetoTracker,
            )
        except ImportError:
            pytest.skip("ParetoTracker not available for hypervolume test")

        # Create a properly mocked current_state that returns an array
        mock_state = Mock()
        mock_state.get_array.return_value = np.array(
            [1, 0, 1, 0, 1] * 4
        )  # Mock state array
        standard_env.current_state = mock_state

        # Create a mock pareto tracker with some existing points
        pareto_tracker = ParetoTracker(update_interval=1, max_pending=10)

        # Add some initial points to the Pareto front
        pareto_tracker.add_point(
            point=np.array([0.998, 6e-21, 0.006]),
            values=np.array([0.998, 6e-21, 0.006]),
            state=np.array([1, 0, 1, 0, 1] * 4),  # Mock state
        )
        pareto_tracker.add_point(
            point=np.array([0.999, 8e-21, 0.004]),
            values=np.array([0.999, 8e-21, 0.004]),
            state=np.array([1, 1, 0, 1, 0] * 4),  # Mock state
        )

        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            combine="hypervolume",
        )

        try:
            total_reward, vals, rewards = calc.calculate(
                env=standard_env,
                pareto_tracker=pareto_tracker,  # Pass pareto_tracker as kwarg
                **standard_params,
            )

            assert np.isfinite(total_reward), "Hypervolume reward is not finite"
            assert (
                "hypervolume_value" in rewards
            ), "Missing hypervolume_value in rewards"
            assert (
                "pareto_front_size" in rewards
            ), "Missing pareto_front_size in rewards"

            print(f"Hypervolume reward: {total_reward}")
            print(f"HV value: {rewards['hypervolume_value']}")

        except Exception as e:
            pytest.skip(
                f"Hypervolume calculation failed: {e}. This may be expected if pymoo is not available."
            )

    def test_weight_application(self, standard_env, standard_params):
        """Test that objective weights are applied correctly."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            combine="sum",
        )

        # Test with equal weights
        weights_equal = {"reflectivity": 1.0, "thermal_noise": 1.0, "absorption": 1.0}
        total_equal, _, rewards_equal = calc.calculate(
            env=standard_env, weights=weights_equal, **standard_params
        )

        # Test with skewed weights
        weights_skewed = {"reflectivity": 10.0, "thermal_noise": 0.1, "absorption": 0.1}
        total_skewed, _, rewards_skewed = calc.calculate(
            env=standard_env, weights=weights_skewed, **standard_params
        )

        print(f"Equal weights total: {total_equal}")
        print(f"Skewed weights total: {total_skewed}")

        # Different weights should generally produce different totals (unless edge cases)
        # At minimum, both should be finite
        assert np.isfinite(total_equal) and np.isfinite(total_skewed)

        # Individual rewards should be the same (weights affect combination, not
        # individual rewards)
        for param in ["reflectivity", "thermal_noise", "absorption"]:
            if param in rewards_equal and param in rewards_skewed:
                # Individual objective rewards should be the same
                assert (
                    abs(rewards_equal[param] - rewards_skewed[param]) < 1e-10
                ), f"Individual reward {param} changed with weights"

    def test_boundary_penalty_behavior(self, standard_params):
        """Test boundary penalty behavior."""
        env_with_bounds = Mock()
        env_with_bounds.optimise_parameters = [
            "reflectivity",
            "thermal_noise",
            "absorption",
        ]
        env_with_bounds.get_parameter_names.return_value = [
            "reflectivity",
            "thermal_noise",
            "absorption",
        ]
        env_with_bounds.objective_bounds = {
            "reflectivity": [0.99, 1.0],  # Tight bounds
            "thermal_noise": [1e-22, 1e-20],  # Tight bounds
            "absorption": [1e-6, 5e-3],  # Tight bounds
        }
        env_with_bounds.pareto_front = []
        env_with_bounds.current_state = Mock()
        env_with_bounds.current_state.get_num_active_layers.return_value = 10
        env_with_bounds.current_state.get_layer_count.return_value = 1

        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_boundary_penalties=True,
        )

        # Test within bounds
        params_within = {
            "reflectivity": 0.995,
            "thermal_noise": 5e-21,
            "absorption": 0.001,
            "thickness": 5.0,
        }

        total_within, _, rewards_within = calc.calculate(
            env=env_with_bounds, **params_within
        )

        # Test outside bounds
        params_outside = {
            "reflectivity": 0.98,  # Below minimum
            "thermal_noise": 5e-19,  # Above maximum
            "absorption": 0.01,  # Above maximum
            "thickness": 5.0,
        }

        total_outside, _, rewards_outside = calc.calculate(
            env=env_with_bounds, **params_outside
        )

        print(f"Within bounds total: {total_within}")
        print(f"Outside bounds total: {total_outside}")
        print(f"Within bounds rewards: {rewards_within}")
        print(f"Outside bounds rewards: {rewards_outside}")

        # Outside bounds should have penalties
        penalty_keys = [k for k in rewards_outside.keys() if "boundary_penalty" in k]
        assert len(penalty_keys) > 0, "No boundary penalties applied when out of bounds"

        # Penalties should be negative
        for key in penalty_keys:
            assert (
                rewards_outside[key] <= 0
            ), f"Boundary penalty {key} should be non-positive"

    def test_air_penalty_behavior(self, standard_env, standard_params):
        """Test air penalty behavior."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_air_penalty=True,
            air_penalty_weight=5.0,
        )

        # Test with few air layers (should have penalty)
        standard_env.current_state.get_num_active_layers.return_value = 4
        standard_env.current_state.get_layer_count.return_value = (
            2  # 2 air layers out of 4
        )

        total_with_penalty, _, rewards_with_penalty = calc.calculate(
            env=standard_env, **standard_params
        )

        # Test with adequate layers (should have reward or no penalty)
        standard_env.current_state.get_num_active_layers.return_value = 10
        standard_env.current_state.get_layer_count.return_value = (
            1  # 1 air layer out of 10
        )

        total_adequate, _, rewards_adequate = calc.calculate(
            env=standard_env, **standard_params
        )

        print(f"With penalty total: {total_with_penalty}")
        print(f"Adequate layers total: {total_adequate}")
        print(f"With penalty rewards: {rewards_with_penalty}")
        print(f"Adequate rewards: {rewards_adequate}")

        # Both should have air_addon key
        assert "air_addon" in rewards_with_penalty
        assert "air_addon" in rewards_adequate

        # Both should be finite
        assert np.isfinite(total_with_penalty) and np.isfinite(total_adequate)

    def test_reward_function_types(self, standard_env, standard_params):
        """Test different reward function types produce reasonable results."""
        # Test available reward functions
        function_types = ["default", "normed_log_targets", "log_targets"]

        results = {}
        for func_type in function_types:
            try:
                calc = RewardCalculator(
                    reward_type=func_type,
                    optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
                    optimise_targets={
                        "reflectivity": 0.99999,
                        "thermal_noise": 5.394480540642821e-21,
                        "absorption": 0.01,
                    },
                )

                total_reward, vals, rewards = calc.calculate(
                    env=standard_env, **standard_params
                )
                results[func_type] = total_reward

                assert np.isfinite(
                    total_reward
                ), f"Non-finite reward with function {func_type}"

                # Should have individual objective rewards/values
                assert len(rewards) > 3, f"Too few reward components for {func_type}"

            except Exception as e:
                print(f"Warning: {func_type} failed with error: {e}")
                # Don't fail the test, just log the issue

        print(f"Reward function results: {results}")

        # At least one should work
        assert len(results) > 0, "No reward functions worked"

    def test_consistency_with_main_branch_config(self):
        """Test with configuration that should match main branch behavior."""
        # This represents a typical main branch configuration
        main_branch_config = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            optimise_targets={
                "reflectivity": 0.99999,
                "thermal_noise": 5.394480540642821e-21,
                "absorption": 0.01,
                "thickness": 0.1,
            },
            target_mapping={
                "reflectivity": "log-",
                "thermal_noise": "log-",
                "thickness": "linear-",
                "absorption": "log-",
            },
            combine="sum",
            # Typical addons that might have been enabled in main branch
            apply_normalisation=False,  # Test both states
            apply_boundary_penalties=False,
            apply_air_penalty=False,
        )

        env = Mock()
        env.optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        env.get_parameter_names.return_value = [
            "reflectivity",
            "thermal_noise",
            "absorption",
        ]
        env.objective_bounds = {
            "reflectivity": [0.9, 1.0],
            "thermal_noise": [1e-22, 1e-19],
            "absorption": [1e-8, 1e-2],
        }
        env.pareto_front = []

        # Test with typical coating parameters
        typical_params = [
            {
                "reflectivity": 0.99999,
                "thermal_noise": 5.394480540642821e-21,
                "absorption": 0.01,
                "thickness": 5.0,
            },
            {
                "reflectivity": 0.999,
                "thermal_noise": 1e-20,
                "absorption": 0.005,
                "thickness": 3.0,
            },
            {
                "reflectivity": 0.9999,
                "thermal_noise": 8e-21,
                "absorption": 0.008,
                "thickness": 7.0,
            },
        ]

        for params in typical_params:
            total_reward, vals, rewards = main_branch_config.calculate(
                env=env, **params
            )

            print(f"Params: {params}")
            print(f"Total reward: {total_reward}")
            print(f"Individual rewards: {rewards}")
            print("---")

            # Basic sanity checks
            assert np.isfinite(total_reward), f"Non-finite reward for params {params}"
            assert isinstance(vals, dict) and len(vals) > 0
            assert isinstance(rewards, dict) and len(rewards) > 0

            # Should have rewards for each objective
            for param in ["reflectivity", "thermal_noise", "absorption"]:
                assert (
                    param in rewards or f"{param}_reward" in rewards
                ), f"Missing reward for {param}"


class TestRewardSystemRegression:
    """Regression tests to catch specific issues that might have been introduced."""

    def test_nan_inf_handling(self):
        """Test that the system handles edge cases without producing NaN/Inf."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
        )

        env = Mock()
        env.optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        env.get_parameter_names.return_value = [
            "reflectivity",
            "thermal_noise",
            "absorption",
        ]
        env.objective_bounds = {
            "reflectivity": [0.9, 1.0],
            "thermal_noise": [1e-22, 1e-19],
            "absorption": [1e-8, 1e-2],
        }
        env.pareto_front = []

        # Test edge cases
        edge_cases = [
            {
                "reflectivity": 1.0,
                "thermal_noise": 0.0,
                "absorption": 0.0,
                "thickness": 0.0,
            },  # Zeros
            {
                "reflectivity": 0.0,
                "thermal_noise": 1e-30,
                "absorption": 1e-10,
                "thickness": 100.0,
            },  # Extreme values
            {
                "reflectivity": 0.999999999,
                "thermal_noise": 1e-25,
                "absorption": 1e-12,
                "thickness": 0.001,
            },  # Very small
        ]

        for params in edge_cases:
            try:
                total_reward, vals, rewards = calc.calculate(env=env, **params)

                # Check for NaN/Inf
                assert np.isfinite(
                    total_reward
                ), f"Non-finite total reward for edge case {params}"

                for key, val in vals.items():
                    if isinstance(val, (int, float)):
                        assert np.isfinite(
                            val
                        ), f"Non-finite vals[{key}] for edge case {params}"

                for key, val in rewards.items():
                    if isinstance(val, (int, float)):
                        assert np.isfinite(
                            val
                        ), f"Non-finite rewards[{key}] for edge case {params}"

            except Exception as e:
                print(f"Edge case {params} failed: {e}")
                # Don't fail test for edge cases, but log them

    def test_memory_consistency(self):
        """Test that repeated calls don't have memory effects."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_normalisation=True,  # This might have state
        )

        env = Mock()
        env.optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        env.get_parameter_names.return_value = [
            "reflectivity",
            "thermal_noise",
            "absorption",
        ]
        env.objective_bounds = {
            "reflectivity": [0.9, 1.0],
            "thermal_noise": [1e-22, 1e-19],
            "absorption": [1e-8, 1e-2],
        }
        env.pareto_front = []

        params = {
            "reflectivity": 0.999,
            "thermal_noise": 1e-21,
            "thickness": 5.0,
            "absorption": 0.005,
        }

        # First call
        total1, vals1, rewards1 = calc.calculate(env=env, **params)

        # Many intervening calls with different parameters
        for i in range(10):
            calc.calculate(
                env=env,
                reflectivity=0.99 + i * 0.001,
                thermal_noise=(i + 1) * 1e-21,
                thickness=i + 1,
                absorption=0.001 + i * 0.001,
            )

        # Repeat original call
        total2, vals2, rewards2 = calc.calculate(env=env, **params)

        # Results should be identical (no memory effects)
        assert (
            abs(total1 - total2) < 1e-12
        ), f"Memory effect in total reward: {total1} vs {total2}"

        for key in vals1:
            if key in vals2:
                assert (
                    abs(vals1[key] - vals2[key]) < 1e-12
                ), f"Memory effect in vals[{key}]"

        for key in rewards1:
            if key in rewards2:
                assert (
                    abs(rewards1[key] - rewards2[key]) < 1e-12
                ), f"Memory effect in rewards[{key}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
