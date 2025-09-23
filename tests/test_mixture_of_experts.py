"""
Unit tests for Mixture of Experts functionality.

Tests the MoE implementation including:
- Expert specialization strategies
- Gating network behavior
- Integration with preference-constrained training
- Objective-based expert routing
"""

import numpy as np
import pytest
import torch

from coatopt.algorithms.hppo.core.networks.mixture_of_experts import (
    MixtureOfExpertsBase,
    MixtureOfExpertsGating,
    MoEContinuousPolicy,
    MoEDiscretePolicy,
    MoEValueNetwork,
)


class TestMixtureOfExpertsBase:
    """Test the base MoE functionality and expert specialization strategies."""

    def test_objectives_specialization_exact_match(self):
        """Test objectives specialization with exact number of experts."""
        n_experts = 3
        n_objectives = 3

        base = MixtureOfExpertsBase(
            n_experts=n_experts,
            n_objectives=n_objectives,
            expert_specialization="objectives",
        )

        # Should create exactly 3 experts with pure objective weights
        assert len(base.expert_regions) == 3

        expected_regions = [
            [1.0, 0.0, 0.0],  # Expert 0: reflectivity specialist
            [0.0, 1.0, 0.0],  # Expert 1: absorption specialist
            [0.0, 0.0, 1.0],  # Expert 2: thermal noise specialist
        ]

        for i, expected in enumerate(expected_regions):
            actual = base.expert_regions[i].tolist()
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_objectives_specialization_more_experts(self):
        """Test objectives specialization with more experts than objectives."""
        n_experts = 5
        n_objectives = 3

        base = MixtureOfExpertsBase(
            n_experts=n_experts,
            n_objectives=n_objectives,
            expert_specialization="objectives",
        )

        # Should create 3 specialist + 2 balanced experts
        assert len(base.expert_regions) == 5

        # First 3 should be pure specialists
        expected_specialists = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        for i, expected in enumerate(expected_specialists):
            actual = base.expert_regions[i].tolist()
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

        # Remaining experts should be balanced (equal weights)
        expected_balanced = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        for i in range(3, 5):
            actual = base.expert_regions[i].tolist()
            np.testing.assert_allclose(actual, expected_balanced, rtol=1e-6)

    def test_objectives_specialization_fewer_experts(self):
        """Test objectives specialization with fewer experts than objectives."""
        n_experts = 2
        n_objectives = 3

        base = MixtureOfExpertsBase(
            n_experts=n_experts,
            n_objectives=n_objectives,
            expert_specialization="objectives",
        )

        # Should create only 2 experts for first 2 objectives
        assert len(base.expert_regions) == 2

        expected_regions = [
            [1.0, 0.0, 0.0],  # Expert 0: reflectivity specialist
            [0.0, 1.0, 0.0],  # Expert 1: absorption specialist
        ]

        for i, expected in enumerate(expected_regions):
            actual = base.expert_regions[i].tolist()
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_sobol_specialization_comparison(self):
        """Test that sobol sequence creates same specialists for first n objectives."""
        n_experts = 3
        n_objectives = 3

        # Objectives specialization
        objectives_base = MixtureOfExpertsBase(
            n_experts=n_experts,
            n_objectives=n_objectives,
            expert_specialization="objectives",
        )

        # Sobol specialization
        sobol_base = MixtureOfExpertsBase(
            n_experts=n_experts,
            n_objectives=n_objectives,
            expert_specialization="sobol_sequence",
        )

        # First 3 experts should be identical
        for i in range(n_objectives):
            objectives_region = objectives_base.expert_regions[i].tolist()
            sobol_region = sobol_base.expert_regions[i].tolist()
            np.testing.assert_allclose(objectives_region, sobol_region, rtol=1e-6)

    def test_invalid_specialization_raises_error(self):
        """Test that invalid specialization strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown expert specialization"):
            MixtureOfExpertsBase(
                n_experts=3, n_objectives=3, expert_specialization="invalid_strategy"
            )


class TestMixtureOfExpertsGating:
    """Test the MoE gating network behavior."""

    def test_gating_network_initialization(self):
        """Test gating network initializes correctly."""
        n_objectives = 3
        n_experts = 3
        temperature = 0.5

        gating = MixtureOfExpertsGating(
            n_objectives=n_objectives, n_experts=n_experts, temperature=temperature
        )

        assert gating.n_objectives == n_objectives
        assert gating.n_experts == n_experts
        assert gating.temperature == temperature

    def test_gating_forward_pass(self):
        """Test gating network forward pass produces valid probabilities."""
        n_objectives = 3
        n_experts = 3
        batch_size = 4

        gating = MixtureOfExpertsGating(n_objectives, n_experts)

        # Create test objective weights
        objective_weights = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Pure reflectivity
                [0.0, 1.0, 0.0],  # Pure absorption
                [0.0, 0.0, 1.0],  # Pure thermal noise
                [0.5, 0.3, 0.2],  # Mixed objectives
            ],
            dtype=torch.float32,
        )

        gate_weights, gate_logits = gating(objective_weights)

        # Check shapes
        assert gate_weights.shape == (batch_size, n_experts)
        assert gate_logits.shape == (batch_size, n_experts)

        # Check probabilities sum to 1
        prob_sums = gate_weights.sum(dim=1)
        np.testing.assert_allclose(prob_sums.detach().numpy(), 1.0, rtol=1e-6)

        # Check all probabilities are positive
        assert torch.all(gate_weights >= 0)

    def test_gating_temperature_effect(self):
        """Test that lower temperature creates sharper probability distributions."""
        n_objectives = 3
        n_experts = 3

        # Pure objective weights
        objective_weights = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

        # High temperature (softer distribution)
        gating_soft = MixtureOfExpertsGating(n_objectives, n_experts, temperature=2.0)
        gate_weights_soft, _ = gating_soft(objective_weights)

        # Low temperature (sharper distribution)
        gating_sharp = MixtureOfExpertsGating(n_objectives, n_experts, temperature=0.1)
        gate_weights_sharp, _ = gating_sharp(objective_weights)

        # Sharp distribution should have higher max probability
        max_prob_soft = torch.max(gate_weights_soft).item()
        max_prob_sharp = torch.max(gate_weights_sharp).item()

        assert max_prob_sharp > max_prob_soft


class TestMoEIntegration:
    """Test integration of MoE with preference-constrained training scenarios."""

    def test_pure_objective_routing(self):
        """Test that pure objective weights route to correct experts."""
        # This would require more complex setup with actual MoE policy networks
        # For now, test the specialization loss calculation
        base = MixtureOfExpertsBase(
            n_experts=3, n_objectives=3, expert_specialization="objectives"
        )

        # Pure objective weights should have zero distance to corresponding expert
        objective_weights = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Should match Expert 0
                [0.0, 1.0, 0.0],  # Should match Expert 1
                [0.0, 0.0, 1.0],  # Should match Expert 2
            ],
            dtype=torch.float32,
        )

        # Mock gate weights (uniform for simplicity)
        gate_weights = torch.ones(3, 3) / 3

        # Calculate specialization loss
        loss = base.get_expert_specialization_loss(objective_weights, gate_weights)

        # Should be a scalar tensor
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_load_balancing_loss(self):
        """Test load balancing loss calculation."""
        base = MixtureOfExpertsBase(
            n_experts=3, n_objectives=3, expert_specialization="objectives"
        )

        # Unbalanced gate weights (Expert 0 always selected)
        unbalanced_gates = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        # Balanced gate weights
        balanced_gates = torch.ones(3, 3) / 3

        loss_unbalanced = base.get_load_balancing_loss(unbalanced_gates)
        loss_balanced = base.get_load_balancing_loss(balanced_gates)

        # Unbalanced should have higher loss
        assert loss_unbalanced > loss_balanced
        assert torch.isfinite(loss_unbalanced)
        assert torch.isfinite(loss_balanced)


class TestMoEConfigurationValidation:
    """Test configuration validation and edge cases."""

    def test_single_objective_single_expert(self):
        """Test edge case with single objective and expert."""
        base = MixtureOfExpertsBase(
            n_experts=1, n_objectives=1, expert_specialization="objectives"
        )

        assert len(base.expert_regions) == 1
        expected = [1.0]
        actual = base.expert_regions[0].tolist()
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_zero_experts_handled_gracefully(self):
        """Test that zero experts is handled gracefully."""
        base = MixtureOfExpertsBase(
            n_experts=0, n_objectives=3, expert_specialization="objectives"
        )

        # Should create empty list of expert regions
        assert len(base.expert_regions) == 0
        assert base.expert_regions == []

    def test_expert_regions_are_tensors(self):
        """Test that expert regions are properly formatted as tensors."""
        base = MixtureOfExpertsBase(
            n_experts=3, n_objectives=3, expert_specialization="objectives"
        )

        for region in base.expert_regions:
            assert isinstance(region, torch.Tensor)
            assert region.dtype == torch.float32
            assert region.shape == (3,)  # n_objectives


if __name__ == "__main__":
    # Run tests manually if executed directly
    pytest.main([__file__, "-v"])
