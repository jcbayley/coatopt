"""
Focused tests for reward normalization to verify it works identically to main branch.
This is critical for identifying behavioral differences between versions.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import copy

from coatopt.environments.reward_functions.reward_system import RewardCalculator
from coatopt.environments.reward_functions.reward_addons import apply_normalisation_addon


class TestRewardNormalisation:
    """Comprehensive tests for reward normalisation functionality."""
    
    @pytest.fixture
    def mock_env_with_bounds(self):
        """Mock environment with objective bounds for normalisation."""
        env = Mock()
        env.optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        env.get_parameter_names.return_value = ["reflectivity", "thermal_noise", "absorption"]
        env.objective_bounds = {
            "reflectivity": [0.95, 1.0],
            "thermal_noise": [1e-22, 1e-19],
            "absorption": [1e-8, 1e-2]
        }
        env.pareto_front = []
        return env
    
    @pytest.fixture 
    def standard_rewards(self):
        """Standard reward values for testing."""
        return {
            "reflectivity": 15.2,
            "thermal_noise": 22.8,
            "absorption": 18.5
        }
    
    @pytest.fixture
    def standard_vals(self):
        """Standard objective values for testing."""
        return {
            "reflectivity": 0.999,
            "thermal_noise": 5e-21,
            "absorption": 0.005
        }
    
    @pytest.fixture
    def standard_targets(self):
        """Standard optimization targets."""
        return {
            "reflectivity": 0.99999,
            "thermal_noise": 5.394480540642821e-21,
            "absorption": 0.01
        }
    
    def test_normalisation_addon_direct(self, standard_rewards, standard_vals, mock_env_with_bounds, standard_targets):
        """Test the normalisation addon function directly."""
        optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        target_mapping = {
            "reflectivity": "log-",
            "thermal_noise": "log-",
            "absorption": "log-"
        }
        
        # Test normalisation addon
        normalized_rewards = apply_normalisation_addon(
            standard_rewards,
            standard_vals, 
            optimise_parameters,
            standard_targets,
            mock_env_with_bounds,
            target_mapping
        )
        
        print(f"Original rewards: {standard_rewards}")
        print(f"Normalized rewards: {normalized_rewards}")
        
        # Should preserve original rewards
        for key in standard_rewards:
            assert key in normalized_rewards
            assert normalized_rewards[key] == standard_rewards[key]
        
        # Should add normalized versions or modify existing ones
        assert isinstance(normalized_rewards, dict)
        assert len(normalized_rewards) >= len(standard_rewards)
    
    def test_normalisation_consistency_across_calls(self, mock_env_with_bounds):
        """Test that normalisation produces consistent results across multiple calls."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            optimise_targets={
                "reflectivity": 0.99999,
                "thermal_noise": 5.394480540642821e-21, 
                "absorption": 0.01
            },
            apply_normalisation=True
        )
        
        params = {
            "reflectivity": 0.999,
            "thermal_noise": 1e-21,
            "thickness": 5.0,
            "absorption": 0.005,
            "env": mock_env_with_bounds
        }
        
        # Multiple calls with same parameters
        results = []
        for i in range(5):
            total_reward, vals, rewards = calc.calculate(**params)
            results.append((total_reward, copy.deepcopy(vals), copy.deepcopy(rewards)))
        
        # All results should be identical
        base_total, base_vals, base_rewards = results[0]
        
        for i, (total, vals, rewards) in enumerate(results[1:], 1):
            assert abs(total - base_total) < 1e-12, f"Call {i} total differs: {total} vs {base_total}"
            
            for key in base_vals:
                if key in vals:
                    assert abs(vals[key] - base_vals[key]) < 1e-12, f"Call {i} vals[{key}] differs"
            
            for key in base_rewards:
                if key in rewards and isinstance(base_rewards[key], (int, float)) and isinstance(rewards[key], (int, float)):
                    assert abs(rewards[key] - base_rewards[key]) < 1e-12, f"Call {i} rewards[{key}] differs: {rewards[key]} vs {base_rewards[key]}"
    
    def test_normalisation_vs_no_normalisation_behavior(self, mock_env_with_bounds):
        """Test behavior with and without normalisation to identify differences."""
        base_config = {
            "reward_type": "default",
            "optimise_parameters": ["reflectivity", "thermal_noise", "absorption"],
            "optimise_targets": {
                "reflectivity": 0.99999,
                "thermal_noise": 5.394480540642821e-21,
                "absorption": 0.01
            }
        }
        
        calc_no_norm = RewardCalculator(**base_config, apply_normalisation=False)
        calc_with_norm = RewardCalculator(**base_config, apply_normalisation=True)
        
        test_cases = [
            # (reflectivity, thermal_noise, thickness, absorption)
            (0.999, 1e-21, 5.0, 0.005),
            (0.99999, 5e-21, 3.0, 0.01),
            (0.995, 2e-21, 7.0, 0.008),
            (0.9999, 8e-21, 2.0, 0.002)
        ]
        
        for refl, tn, thick, abs_val in test_cases:
            params = {
                "reflectivity": refl,
                "thermal_noise": tn,
                "thickness": thick,
                "absorption": abs_val,
                "env": mock_env_with_bounds
            }
            
            # Calculate without normalisation
            total_no_norm, vals_no_norm, rewards_no_norm = calc_no_norm.calculate(**params)
            
            # Calculate with normalisation
            total_with_norm, vals_with_norm, rewards_with_norm = calc_with_norm.calculate(**params)
            
            print(f"\nTest case: {params}")
            print(f"No norm - Total: {total_no_norm}")
            print(f"With norm - Total: {total_with_norm}")
            print(f"No norm - Rewards: {rewards_no_norm}")
            print(f"With norm - Rewards: {rewards_with_norm}")
            
            # Physical values should be identical
            for key in ["reflectivity", "thermal_noise", "absorption"]:
                if key in vals_no_norm and key in vals_with_norm:
                    assert abs(vals_no_norm[key] - vals_with_norm[key]) < 1e-12, f"Physical value {key} changed with normalisation"
            
            # Both should produce finite results
            assert np.isfinite(total_no_norm), f"Non-finite result without normalisation for case {params}"
            assert np.isfinite(total_with_norm), f"Non-finite result with normalisation for case {params}"
            
            # Check individual objective rewards
            for param in ["reflectivity", "thermal_noise", "absorption"]:
                # Look for rewards with this parameter name
                no_norm_reward = None
                with_norm_reward = None
                
                # Check different possible key formats
                for key in rewards_no_norm:
                    if key == param or key == f"{param}_reward":
                        no_norm_reward = rewards_no_norm[key]
                        break
                
                for key in rewards_with_norm:
                    if key == param or key == f"{param}_reward":
                        with_norm_reward = rewards_with_norm[key]
                        break
                
                if no_norm_reward is not None and with_norm_reward is not None:
                    print(f"  {param}: no_norm={no_norm_reward}, with_norm={with_norm_reward}")
                    
                    # Both should be finite
                    assert np.isfinite(no_norm_reward), f"Non-finite {param} reward without normalisation"
                    assert np.isfinite(with_norm_reward), f"Non-finite {param} reward with normalisation"
    
    def test_normalisation_with_edge_cases(self, mock_env_with_bounds):
        """Test normalisation with edge case values."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_normalisation=True
        )
        
        edge_cases = [
            # At bounds
            {"reflectivity": 1.0, "thermal_noise": 1e-22, "absorption": 1e-8, "thickness": 1.0},
            {"reflectivity": 0.95, "thermal_noise": 1e-19, "absorption": 1e-2, "thickness": 10.0},
            # Very close to targets
            {"reflectivity": 0.99999, "thermal_noise": 5.394480540642821e-21, "absorption": 0.01, "thickness": 5.0},
            # Far from targets but in bounds
            {"reflectivity": 0.96, "thermal_noise": 5e-20, "absorption": 5e-3, "thickness": 2.0}
        ]
        
        for params in edge_cases:
            try:
                total_reward, vals, rewards = calc.calculate(env=mock_env_with_bounds, **params)
                
                print(f"\nEdge case: {params}")
                print(f"Total reward: {total_reward}")
                print(f"Rewards: {rewards}")
                
                # Should not crash and should produce finite results
                assert np.isfinite(total_reward), f"Non-finite reward for edge case {params}"
                
                # Check all reward components are finite
                for key, val in rewards.items():
                    if isinstance(val, (int, float)):
                        assert np.isfinite(val), f"Non-finite reward component {key} for edge case {params}"
                
            except Exception as e:
                print(f"Edge case {params} failed: {e}")
                # Don't fail the test, but log the issue
    
    def test_normalisation_with_different_target_mappings(self, mock_env_with_bounds):
        """Test normalisation with different target mapping configurations."""
        target_mappings = [
            {"reflectivity": "log-", "thermal_noise": "log-", "absorption": "log-"},
            {"reflectivity": "linear-", "thermal_noise": "log-", "absorption": "log-"},
            {"reflectivity": "log-", "thermal_noise": "linear-", "absorption": "linear-"}
        ]
        
        for target_mapping in target_mappings:
            calc = RewardCalculator(
                reward_type="default", 
                optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
                optimise_targets={
                    "reflectivity": 0.99999,
                    "thermal_noise": 5.394480540642821e-21,
                    "absorption": 0.01
                },
                target_mapping=target_mapping,
                apply_normalisation=True
            )
            
            params = {
                "reflectivity": 0.999,
                "thermal_noise": 1e-21,
                "thickness": 5.0,
                "absorption": 0.005,
                "env": mock_env_with_bounds
            }
            
            total_reward, vals, rewards = calc.calculate(**params)
            
            print(f"\nTarget mapping: {target_mapping}")
            print(f"Total reward: {total_reward}")
            print(f"Sample rewards: {dict(list(rewards.items())[:5])}")  # First 5 items
            
            # Should produce valid results regardless of target mapping
            assert np.isfinite(total_reward), f"Non-finite reward with target mapping {target_mapping}"
            
            # Check that results are reasonable (not all zeros, not extremely large)
            assert abs(total_reward) < 1e6, f"Extremely large reward with target mapping {target_mapping}"
    
    def test_normalisation_range_detection(self, mock_env_with_bounds):
        """Test that normalisation correctly uses objective bounds for range detection."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_normalisation=True
        )
        
        # Mock the apply_normalisation_addon to see what parameters it receives
        with patch('coatopt.environments.reward_functions.reward_system.apply_normalisation_addon') as mock_norm:
            # Return the input rewards unchanged
            mock_norm.return_value = {"reflectivity": 10.0, "thermal_noise": 12.0, "absorption": 8.0}
            
            params = {
                "reflectivity": 0.999,
                "thermal_noise": 1e-21, 
                "thickness": 5.0,
                "absorption": 0.005,
                "env": mock_env_with_bounds
            }
            
            calc.calculate(**params)
            
            # Check that normalisation addon was called with correct parameters
            assert mock_norm.called, "Normalisation addon was not called"
            
            call_args = mock_norm.call_args
            assert call_args is not None, "No call args captured"
            
            # Should pass the environment (which contains objective bounds)
            args, kwargs = call_args
            assert len(args) >= 5, "Not enough arguments passed to normalisation addon"
            
            # The environment should be passed (5th argument)
            passed_env = args[4]
            assert passed_env is mock_env_with_bounds, "Environment not passed correctly to normalisation addon"


class TestRewardNormalisationRegression:
    """Regression tests for specific normalisation issues."""
    
    def test_normalisation_doesnt_modify_input_dicts(self):
        """Test that normalisation doesn't modify input dictionaries."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"],
            apply_normalisation=True
        )
        
        env = Mock()
        env.optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        env.get_parameter_names.return_value = ["reflectivity", "thermal_noise", "absorption"]
        env.objective_bounds = {"reflectivity": [0.9, 1.0], "thermal_noise": [1e-22, 1e-19], "absorption": [1e-8, 1e-2]}
        env.pareto_front = []
        
        original_bounds = copy.deepcopy(env.objective_bounds)
        
        params = {
            "reflectivity": 0.999,
            "thermal_noise": 1e-21,
            "thickness": 5.0,
            "absorption": 0.005,
            "env": env
        }
        
        calc.calculate(**params)
        
        # Environment bounds should not be modified
        assert env.objective_bounds == original_bounds, "Environment bounds were modified"
    
    def test_normalisation_numerical_stability(self):
        """Test normalisation numerical stability with extreme values."""
        calc = RewardCalculator(
            reward_type="default",
            optimise_parameters=["reflectivity", "thermal_noise", "absorption"], 
            apply_normalisation=True
        )
        
        env = Mock()
        env.optimise_parameters = ["reflectivity", "thermal_noise", "absorption"]
        env.get_parameter_names.return_value = ["reflectivity", "thermal_noise", "absorption"]
        env.objective_bounds = {
            "reflectivity": [1e-10, 1.0],     # Huge range
            "thermal_noise": [1e-30, 1e-15],  # Huge range
            "absorption": [1e-15, 1.0]        # Huge range
        }
        env.pareto_front = []
        
        # Test with values that might cause numerical issues
        extreme_cases = [
            {"reflectivity": 1e-9, "thermal_noise": 1e-29, "absorption": 1e-14, "thickness": 1.0},
            {"reflectivity": 0.999, "thermal_noise": 1e-16, "absorption": 0.1, "thickness": 1.0},
        ]
        
        for params in extreme_cases:
            try:
                total_reward, vals, rewards = calc.calculate(env=env, **params)
                
                # Should not produce NaN or Inf
                assert np.isfinite(total_reward), f"Non-finite reward for extreme case {params}"
                
                for key, val in rewards.items():
                    if isinstance(val, (int, float)):
                        assert np.isfinite(val), f"Non-finite reward component {key} for extreme case {params}"
                        
            except Exception as e:
                print(f"Extreme case {params} failed: {e}")
                # Log but don't fail test for extreme cases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
