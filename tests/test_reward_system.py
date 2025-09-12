import pytest
import numpy as np
from coatopt.environments.reward_functions.reward_system import (
    calculate_air_penalty_reward_new,
    apply_air_penalty_addon,
    apply_boundary_penalties,
    RewardCalculator,
)

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
    (10, 2, 5, -1, None),   # Not enough real layers, penalty
    (10, 1, 5, 1, {"base": 0}),    # Enough real layers, reward
    (4, 0, 5, -1, None),    # Too few layers, penalty
])
def test_calculate_air_penalty_reward_new(total_layers, air_layers, min_real_layers, expected_sign, design_criteria):
    state = DummyState(total_layers, air_layers)
    result = calculate_air_penalty_reward_new(
        state,
        air_material_index=0,
        min_real_layers=min_real_layers,
        design_criteria=design_criteria
    )
    assert np.sign(result) == expected_sign

# Test apply_air_penalty_addon

def test_apply_air_penalty_addon_reward():
    state = DummyState(10, 1)
    env = DummyEnv(state)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {}
    new_total, new_rewards = apply_air_penalty_addon(
        total_reward, rewards, vals, env, air_penalty_weight=1.0
    )[:2]
    print(total_reward, new_total, new_rewards)
    assert "air_penalty" in new_rewards
    assert new_total != total_reward

def test_apply_air_penalty_addon_reward_allair():
    state = DummyState(10, 10)
    design_criteria = {"base": 0.5}
    env = DummyEnv(state, design_criteria=design_criteria)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {"base":10}
    new_total, new_rewards = apply_air_penalty_addon(
        total_reward, rewards, vals, env, air_penalty_weight=1.0, reward_strength=1.0, penalty_strength=1.0, optimise_parameters=["base"]
    )[:2]
    assert "air_penalty" in new_rewards
    assert new_total != total_reward
    assert new_total == total_reward - 1.0

def test_apply_air_penalty_addon_penalty():
    state = DummyState(4, 2)
    env = DummyEnv(state)
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {}
    new_total, new_rewards = apply_air_penalty_addon(
        total_reward, rewards, vals, env, air_penalty_weight=1.0
    )[:2]
    assert "air_penalty" in new_rewards
    assert new_total != total_reward


def test_apply_air_penalty_addon_env_none():
    total_reward = 1.0
    rewards = {"base": 1.0}
    vals = {}
    new_total, new_rewards = apply_air_penalty_addon(
        total_reward, rewards, vals, None, air_penalty_weight=1.0
    )[:2]
    assert new_total == total_reward
    assert new_rewards == rewards


# Test apply_boundary_penalties
def test_apply_boundary_penalties_penalty():
    rewards = {"reflectivity": 0.5, "thermal_noise": 0.5, "thickness": 0.5, "absorption": 0.5}
    vals = {"reflectivity": 0.98, "thermal_noise": 1e-19, "thickness": 0.25, "absorption": 0.1}
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
    rewards = {"reflectivity": 0.5, "thermal_noise": 0.5, "thickness": 0.5, "absorption": 0.5}
    vals = {"reflectivity": 0.995, "thermal_noise": 1e-21, "thickness": 0.1, "absorption": 0.01}
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
        apply_normalization=True,
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


# Test normalization addon
from coatopt.environments.reward_functions.reward_system import apply_normalization_addon
def test_apply_normalization_addon():
    rewards = {"reflectivity": 0.5, "thermal_noise": 0.5, "thickness": 0.5, "absorption": 0.5}
    vals = {"reflectivity": 0.995, "thermal_noise": 1e-21, "thickness": 0.1, "absorption": 0.01}
    optimise_parameters = ["reflectivity", "thermal_noise", "thickness", "absorption"]
    optimise_targets = {"reflectivity": 0.999, "thermal_noise": 1e-22, "thickness": 0.05, "absorption": 0.0}
    env = DummyEnv(DummyState(10, 1))
    env.objective_bounds = {
        "reflectivity": [0.99, 1.0],
        "thermal_noise": [0, 1e-20],
        "thickness": [0, 0.2],
        "absorption": [0, 0.05],
    }
    updated = apply_normalization_addon(rewards, vals, optimise_parameters, optimise_targets, env)
    # Should add normalized values for each parameter
    for key in optimise_parameters:
        assert f"{key}_normalized" in updated


# Test divergence penalty
from coatopt.environments.reward_functions.reward_system import apply_divergence_penalty
def test_apply_divergence_penalty():
    rewards = {"reflectivity": 0.8, "thermal_noise": 0.2}
    optimise_parameters = ["reflectivity", "thermal_noise"]
    weights = {"reflectivity": 1.0, "thermal_noise": 1.0}
    penalty, updated = apply_divergence_penalty(rewards, optimise_parameters, weights, divergence_penalty_weight=2.0)
    assert "divergence_penalty" in updated
    assert penalty < 0
