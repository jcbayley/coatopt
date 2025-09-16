import numpy as np
import pytest

from coatopt.environments.reward_functions.reward_addons import (
    calculate_air_penalty_reward_new,
    apply_air_penalty_addon,
    apply_boundary_penalties
)
from coatopt.environments.reward_functions.reward_system import RewardCalculator


class DummyState:
    def __init__(self, total_layers, air_layers):
        self._total_layers = total_layers
        self._air_layers = air_layers

    def get_num_active_layers(self):
        return self._total_layers

    def get_layer_count(self, material_index=0):
        return self._air_layers


class DummyEnv:
    def __init__(self, state, design_criteria=None):
        self.current_state = state
        self.design_criteria = design_criteria


# Test calculate_air_penalty_reward_new
@pytest.mark.parametrize("total_layers, air_layers, min_real_layers, expected_sign, design_criteria", [
    (10, 2, 5, 0, None),   # Enough real layers (8>=5), no penalty
    (10, 1, 5, 1, {"reflectivity": 0}),    # Enough real layers + criteria met, reward  
    (4, 0, 5, -1, None),    # Too few real layers (4<5), penalty
])
def test_calculate_air_penalty_reward_new(
    total_layers, air_layers, min_real_layers, expected_sign, design_criteria
):
    state = DummyState(total_layers, air_layers)
    result = calculate_air_penalty_reward_new(
        state,
        air_material_index=0,
        min_real_layers=min_real_layers,
        design_criteria=design_criteria,
        current_vals={"reflectivity": 1.0} if design_criteria else None,
        optimise_parameters=["reflectivity"] if design_criteria else []
    )
    if expected_sign == 0:
        assert result == 0
    else:
        assert np.sign(result) == expected_sign


# Test apply_air_penalty_addon


def test_apply_air_penalty_addon_reward():
    state = DummyState(10, 2)
    env = DummyEnv(state)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base": 0.5}
    new_rewards = apply_air_penalty_addon(
        total_reward,
        rewards,
        vals,
        env,
        air_penalty_weight=1.0,
        reward_strength=1.0,
        penalty_strength=1.0,
        optimise_parameters=["base"],
    )
    assert "air_addon" in new_rewards
    assert new_rewards["total_reward"] == total_reward  # No penalty expected with 8 real >= 5 min


def test_apply_air_penalty_addon_reward_allair():
    state = DummyState(10, 10)
    design_criteria = {"base": 0.5}
    env = DummyEnv(state, design_criteria=design_criteria)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base": 10}
    new_rewards = apply_air_penalty_addon(
        total_reward,
        rewards,
        vals,
        env,
        air_penalty_weight=1.0,
        reward_strength=1.0,
        penalty_strength=1.0,
        optimise_parameters=["base"],
    )
    assert "air_addon" in new_rewards  # Changed from "air_penalty"
    assert new_rewards["total_reward"] != total_reward


def test_apply_air_penalty_addon_penalty():
    state = DummyState(4, 0)
    env = DummyEnv(state)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base": 0.5}
    new_rewards = apply_air_penalty_addon(
        total_reward,
        rewards,
        vals,
        env,
        air_penalty_weight=1.0,
        penalty_strength=1.0,
        optimise_parameters=["base"],
    )
    assert "air_addon" in new_rewards
    assert new_rewards["total_reward"] == total_reward - 1.0


def test_apply_air_penalty_addon_env_none():
    state = DummyState(5, 2)
    env = None
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base": 0.5}
    new_rewards = apply_air_penalty_addon(
        total_reward,
        rewards,
        vals,
        env,
        air_penalty_weight=1.0,
        penalty_strength=1.0,
        optimise_parameters=["base"],
    )
    assert "air_addon" in new_rewards
    # Should have some effect on total reward
    assert "total_reward" in new_rewards


def test_apply_air_penalty_addon_reward_allair():
    state = DummyState(10, 10)
    design_criteria = {"base": 0.5}
    env = DummyEnv(state, design_criteria=design_criteria)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base": 10}
    new_rewards = apply_air_penalty_addon(
        total_reward,
        rewards,
        vals,
        env,
        air_penalty_weight=1.0,
        reward_strength=1.0,
        penalty_strength=1.0,
        optimise_parameters=["base"],
    )
    assert "air_addon" in new_rewards  # Changed from "air_penalty"
    assert new_rewards["total_reward"] != total_reward


def test_apply_air_penalty_addon_penalty():
    state = DummyState(4, 0)
    env = DummyEnv(state)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base": 0.5}
    new_rewards = apply_air_penalty_addon(
        total_reward, rewards, vals, env, air_penalty_weight=1.0, penalty_strength=1.0, optimise_parameters=["base"]
    )
    assert "air_addon" in new_rewards
    assert new_rewards["total_reward"] == total_reward - 1.0


def test_apply_air_penalty_addon_env_none():
    state = DummyState(5, 2)
    env = None
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base": 0.5}
    new_rewards = apply_air_penalty_addon(
        total_reward, rewards, vals, env, air_penalty_weight=1.0, penalty_strength=1.0, optimise_parameters=["base"]
    )
    # When env is None, no air penalty is calculated, so original rewards returned
    assert new_rewards == rewards  # Should return original rewards unchanged


# Test apply_boundary_penalties
def test_apply_boundary_penalties_penalty():
    rewards = {
        "reflectivity": 0.5,
        "thermal_noise": 0.5,
        "thickness": 0.5,
        "absorption": 0.5,
    }
    vals = {
        "reflectivity": 0.98,
        "thermal_noise": 1e-19,
        "thickness": 0.25,
        "absorption": 0.1,
    }
    optimise_parameters = ["reflectivity", "thermal_noise", "thickness", "absorption"]
    env = DummyEnv(DummyState(10, 1))
    env.objective_bounds = {
        "reflectivity": [0.99, 1.0],
        "thermal_noise": [0, 1e-20],
        "thickness": [0, 0.2],
        "absorption": [0, 0.05],
    }
    updated = apply_boundary_penalties(rewards, vals, optimise_parameters, env)
    # Should apply penalties for reflectivity, thickness, absorption
    assert updated["reflectivity_boundary_penalty"] < 0
    assert updated["thickness_boundary_penalty"] < 0
    assert updated["absorption_boundary_penalty"] < 0


def test_apply_boundary_penalties_no_penalty():
    rewards = {
        "reflectivity": 0.5,
        "thermal_noise": 0.5,
        "thickness": 0.5,
        "absorption": 0.5,
    }
    vals = {
        "reflectivity": 0.995,
        "thermal_noise": 1e-21,
        "thickness": 0.1,
        "absorption": 0.01,
    }
    optimise_parameters = ["reflectivity", "thermal_noise", "thickness", "absorption"]
    env = DummyEnv(DummyState(10, 1))
    env.objective_bounds = {
        "reflectivity": [0.99, 1.0],
        "thermal_noise": [0, 1e-20],
        "thickness": [0, 0.2],
        "absorption": [0, 0.05],
    }
    updated = apply_boundary_penalties(rewards, vals, optimise_parameters, env)
    # No penalties should be applied
    assert "reflectivity_boundary_penalty" not in updated
    assert "thickness_boundary_penalty" not in updated
    assert "absorption_boundary_penalty" not in updated


# Test RewardCalculator.calculate
def test_reward_calculator_calculate_runs():
    env = DummyEnv(DummyState(10, 1))
    calc = RewardCalculator(
        reward_type="default", 
        optimise_parameters=["reflectivity", "thermal_noise"],
        apply_boundary_penalties=True, 
        apply_air_penalty=True
    )
    reflectivity = 0.995
    thermal_noise = 1e-21
    thickness = 0.1
    absorption = 0.01
    total_reward, vals, rewards = calc.calculate(
        reflectivity, thermal_noise, thickness, absorption, env=env
    )
    assert isinstance(total_reward, float)
    assert isinstance(vals, dict)
    assert isinstance(rewards, dict)
