"""
Smoke tests for SiO2:HfO2 (silica-doped hafnia) 4-material configuration.

Validates that:
1. The 4-material JSON passes validation
2. The physics engine (reflectivity, Brownian noise, absorption) produces
   finite results for stacks that include SiO2:HfO2
3. The RL environment (CoatingEnvironment) can run a full episode using
   all 4 non-air materials without crashing
"""

import json
from pathlib import Path

import numpy as np
import pytest

from coatopt.environments.environment import CoatingEnvironment
from coatopt.environments.utils.coating_utils import merit_function
from coatopt.utils.configs import Config, DataConfig, TrainingConfig
from coatopt.utils.utils import load_materials, validate_materials


MATERIALS_4MAT_PATH = (
    Path(__file__).parent.parent / "experiments" / "Craig_10K_1550_4mat.json"
)


@pytest.fixture
def materials_4mat():
    """Load 4-material library (air + SiO2 + Ta2O5 + aSi + SiO2:HfO2)."""
    with open(MATERIALS_4MAT_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


@pytest.fixture
def config_4mat():
    """Config matching the Craig 10K 1550 4-material experiment."""
    data = DataConfig(
        n_layers=20,
        wavelength=1550e-9,
        min_thickness=0.1,
        max_thickness=0.4,
        use_optical_thickness=True,
        optimise_parameters=["reflectivity", "absorption", "thermal_noise"],
        optimise_targets={
            "reflectivity": 1.0,
            "absorption": 0.0,
            "thermal_noise": 1e-23,
        },
        objective_bounds={
            "reflectivity": [0.0, 0.999999],
            "absorption": [1000, 0.1],
            "thermal_noise": [1e-24, 1e-19],
        },
        compute_efi=True,
        temperature=10.0,
    )
    training = TrainingConfig(cycle_weights="random")
    return Config(data=data, training=training)


class TestMaterialsFileValidation:
    """Test the 4-material JSON file itself."""

    def test_file_exists(self):
        assert MATERIALS_4MAT_PATH.exists(), (
            f"Materials file not found: {MATERIALS_4MAT_PATH}"
        )

    def test_validate_materials_passes(self):
        """validate_materials() should accept the file (no nulls in non-air)."""
        validate_materials(str(MATERIALS_4MAT_PATH))

    def test_has_five_entries(self, materials_4mat):
        """Should have 5 entries: air(0) + 4 coating materials."""
        assert len(materials_4mat) == 5

    def test_sihfo2_properties(self, materials_4mat):
        """Spot-check SiO2:HfO2 key properties."""
        sihfo2 = materials_4mat[4]
        assert sihfo2["name"] == "SiO2:HfO2"
        assert sihfo2["n"] == 1.91
        assert sihfo2["Y"] == 180e9
        assert sihfo2["prat"] == 0.23
        assert sihfo2["phiM"] == 3.7e-4
        assert sihfo2["k"] == 4.0e-6

    def test_no_none_in_coating_materials(self, materials_4mat):
        """No None values in any non-air material."""
        for idx in [1, 2, 3, 4]:
            mat = materials_4mat[idx]
            for key, val in mat.items():
                assert val is not None, (
                    f"Material {idx} ({mat.get('name')}) has None for '{key}'"
                )


class TestPhysicsEngineSmokeTest:
    """Run the physics pipeline on stacks containing SiO2:HfO2."""

    def _build_state_array(self, layer_specs, n_materials=5):
        """Build a state array from (optical_thickness, material_index) pairs.

        Args:
            layer_specs: List of (optical_thickness, material_index) tuples
            n_materials: Total number of materials (including air)
        """
        rows = []
        for opt_thick, mat_idx in layer_specs:
            row = np.zeros(n_materials + 1)  # thickness + one-hot
            row[0] = opt_thick
            row[mat_idx + 1] = 1.0  # +1 because first col is thickness
            rows.append(row)
        return np.array(rows)

    def test_sihfo2_sio2_doublet(self, materials_4mat):
        """SiO2:HfO2 / SiO2 doublet stack — basic sanity."""
        # 6 doublets of SiO2:HfO2 (4) / SiO2 (1), all quarter-wave
        layers = []
        for _ in range(6):
            layers.append((0.25, 4))  # SiO2:HfO2
            layers.append((0.25, 1))  # SiO2
        state = self._build_state_array(layers, n_materials=5)

        R, thermal, absorption, thickness = merit_function(
            state,
            materials_4mat,
            light_wavelength=1550e-9,
            frequency=100.0,
            wBeam=0.09,
            Temp=10.0,
            substrate_index=1,
            air_index=0,
            use_optical_thickness=True,
            compute_efi=True,
        )

        assert np.isfinite(R), f"Reflectivity not finite: {R}"
        assert 0.0 <= R <= 1.0, f"Reflectivity out of range: {R}"
        assert np.isfinite(thermal), f"Brownian noise not finite: {thermal}"
        assert thermal > 0, f"Brownian noise should be positive: {thermal}"
        assert np.isfinite(absorption), f"Absorption not finite: {absorption}"
        assert np.isfinite(thickness), f"Thickness not finite: {thickness}"
        assert thickness > 0, f"Thickness should be positive: {thickness}"

    def test_all_four_materials_mixed(self, materials_4mat):
        """Stack using all 4 non-air materials: SiO2, Ta2O5, aSi, SiO2:HfO2."""
        layers = [
            (0.25, 4),  # SiO2:HfO2
            (0.25, 1),  # SiO2
            (0.25, 2),  # Ta2O5
            (0.25, 1),  # SiO2
            (0.25, 3),  # aSi
            (0.25, 1),  # SiO2
            (0.25, 4),  # SiO2:HfO2
            (0.25, 1),  # SiO2
            (0.25, 2),  # Ta2O5
            (0.25, 1),  # SiO2
        ]
        state = self._build_state_array(layers, n_materials=5)

        R, thermal, absorption, thickness = merit_function(
            state,
            materials_4mat,
            light_wavelength=1550e-9,
            frequency=100.0,
            wBeam=0.09,
            Temp=10.0,
            substrate_index=1,
            air_index=0,
            use_optical_thickness=True,
            compute_efi=True,
        )

        assert np.isfinite(R), f"Reflectivity not finite: {R}"
        assert np.isfinite(thermal), f"Brownian noise not finite: {thermal}"
        assert np.isfinite(absorption), f"Absorption not finite: {absorption}"
        assert np.isfinite(thickness), f"Thickness not finite: {thickness}"

    def test_analytic_absorption_path(self, materials_4mat):
        """Same mixed stack but with compute_efi=False (analytic absorption)."""
        layers = [
            (0.25, 4),  # SiO2:HfO2
            (0.25, 1),  # SiO2
            (0.25, 2),  # Ta2O5
            (0.25, 1),  # SiO2
        ]
        state = self._build_state_array(layers, n_materials=5)

        R, thermal, absorption, thickness = merit_function(
            state,
            materials_4mat,
            light_wavelength=1550e-9,
            frequency=100.0,
            wBeam=0.09,
            Temp=10.0,
            substrate_index=1,
            air_index=0,
            use_optical_thickness=True,
            compute_efi=False,
        )

        assert np.isfinite(R)
        assert np.isfinite(thermal)
        assert np.isfinite(absorption)
        assert np.isfinite(thickness)


class TestRLEnvironmentSmokeTest:
    """Run the RL environment with 4 materials to ensure the full loop works."""

    def test_environment_initialises(self, config_4mat, materials_4mat):
        """CoatingEnvironment initialises with 5 materials (air + 4)."""
        env = CoatingEnvironment(config_4mat, materials_4mat)
        assert env.n_materials == 5

    def test_random_episode_completes(self, config_4mat, materials_4mat):
        """Run a full random episode — no crashes, finite reward at end."""
        env = CoatingEnvironment(config_4mat, materials_4mat)
        env.reset()

        # Pre-plan a short episode: cycle through all non-air materials
        material_cycle = [1, 2, 3, 4]  # SiO2, Ta2O5, aSi, SiO2:HfO2
        finished = False

        for i in range(env.max_layers):
            mat_idx = material_cycle[i % len(material_cycle)]
            thickness = np.random.uniform(env.min_thickness, env.max_thickness)
            action = np.array([mat_idx, thickness])

            state, rewards, terminated, finished, total_reward, full_action, vals = (
                env.step(action)
            )

            if finished:
                break

        # Episode should have finished (we filled all layers)
        assert finished, "Episode should have terminated"

        # Final reward and values should be finite
        assert np.isfinite(total_reward), f"Total reward not finite: {total_reward}"
        assert "reflectivity" in vals
        assert "thermal_noise" in vals
        assert "absorption" in vals

        R = vals["reflectivity"]
        thermal = vals["thermal_noise"]

        assert np.isfinite(R), f"Reflectivity not finite: {R}"
        assert np.isfinite(thermal), f"Brownian noise not finite: {thermal}"

    def test_step_with_sihfo2_only(self, config_4mat, materials_4mat):
        """Build a pure SiO2:HfO2 / SiO2 stack via step()."""
        env = CoatingEnvironment(config_4mat, materials_4mat)
        env.reset()

        finished = False
        for i in range(env.max_layers):
            mat_idx = 4 if i % 2 == 0 else 1  # Alternate SiO2:HfO2 / SiO2
            action = np.array([mat_idx, 0.25])

            state, rewards, terminated, finished, total_reward, full_action, vals = (
                env.step(action)
            )
            if finished:
                break

        assert finished
        assert np.isfinite(total_reward)
        assert np.isfinite(vals["thermal_noise"])
