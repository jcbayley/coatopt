"""
Simplified unified coating environment.

Reads configuration directly from config object to avoid parameter passing errors.
Uses existing physics modules (coating_utils, EFI_tmm, YAM_CoatingBrownian)
without modification.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..environments.state import CoatingState
from ..environments.utils import coating_utils, state_utils
from ..utils.metrics import (
    compute_hypervolume,
    compute_hypervolume_mixed,
    update_pareto_front,
)


# Below this the front is too sparse for its spread to mean anything, so
# constraint_range falls back to the objective_bounds scale.
MIN_FRONT_FOR_CONSTRAINT_RANGE = 10


def _dominated_by(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Mask of points that some row of `others` dominates (maximisation).

    A point tied with another is not dominated by it, so passing the same
    array as both arguments filters a set against itself.
    """
    ge = (others[None, :, :] >= points[:, None, :]).all(-1)
    gt = (others[None, :, :] > points[:, None, :]).any(-1)
    return (ge & gt).any(1)


class CoatingEnvironment:
    """
    Unified coating optimization environment. Reads directly from config.
    """

    def __init__(self, config, materials: Dict[int, Dict]):
        """
        Initialize from config object.

        Args:
            config: CoatingOptimisationConfig object
            materials: Dict mapping material index to material properties
        """
        data = config.data
        training = config.training

        # Materials
        self.materials = materials or {}
        self.n_materials = len(self.materials)

        # Core parameters from config.data
        self.max_layers = data.n_layers
        self.min_thickness = data.min_thickness
        self.max_thickness = data.max_thickness

        # Special materials are found by name, so a reordered materials file
        # cannot silently swap which physical material plays which role.
        self.air_material_index = self.material_index_by_name("air")
        non_air_indices = [i for i in self.materials if i != self.air_material_index]
        if not non_air_indices:
            raise ValueError("Materials must include at least one non-air material")
        self.substrate_material_index = min(non_air_indices)

        # Physics parameters
        wavelength_val = getattr(data, "wavelength", 1064e-9)
        if wavelength_val > 1e-3:
            wavelength_val *= 1e-9
        self.light_wavelength = wavelength_val

        beam_val = getattr(
            data,
            "wBeam",
            getattr(
                data,
                "beam_radius",
                getattr(data, "beam_width", getattr(data, "w0", 0.062)),
            ),
        )
        if beam_val > 1.0:
            beam_val *= 1e-3
        self.wBeam = beam_val

        self.frequency = getattr(data, "frequency", 100.0)
        self.Temp = getattr(data, "Temp", getattr(data, "temperature", 293.0))

        self.use_optical_thickness = getattr(data, "use_optical_thickness", False)
        self.compute_efi = getattr(data, "compute_efi", True)
        print(
            f"[CoatingEnvironment] compute_efi = {self.compute_efi}, "
            f"light_wavelength = {self.light_wavelength:.4e} m, "
            f"wBeam = {self.wBeam:.4f} m ({self.wBeam * 1000:.1f} mm), "
            f"Temp = {self.Temp:.1f} K, frequency = {self.frequency:.1f} Hz"
        )

        # Optimization parameters - strip direction suffixes
        raw_params = data.optimise_parameters or ["reflectivity"]
        self.optimise_parameters = [
            p.split(":")[0].strip() if ":" in p else p for p in raw_params
        ]
        self.optimise_targets = data.optimise_targets or {}
        self.optimise_weight_ranges = getattr(data, "optimise_weight_ranges", {}) or {}
        self.design_criteria = getattr(data, "design_criteria", {}) or {}

        # Objective directions: True = maximize, False = minimize
        self.objective_directions = {
            "reflectivity": True,  # Higher is better
            "absorption": False,  # Lower is better
            "thermal_noise": False,  # Lower is better
        }

        # Action space constraints
        self.ignore_air_option = getattr(data, "ignore_air_option", False)
        self.ignore_substrate_option = getattr(data, "ignore_substrate_option", False)

        # Multi-objective settings
        self.multi_objective = len(self.optimise_parameters) > 1
        self.cycle_weights = getattr(training, "cycle_weights", "random")
        self.objective_weights = {param: 1.0 for param in self.optimise_parameters}

        # Reward configuration
        self.use_intermediate_reward = getattr(data, "use_intermediate_reward", False)
        self.combine = getattr(data, "combine", "sum")

        # Reward normalization bounds (for algorithms that need it)
        self.objective_bounds = getattr(data, "objective_bounds", {}) or {}
        self.reward_bounds = {}  # {objective: [min_reward, max_reward]}
        self._initialize_reward_bounds()
        self.use_reward_normalisation = getattr(data, "use_reward_normalisation", True)
        self.reward_normalisation_apply_clipping = getattr(
            data, "reward_normalisation_apply_clipping", True
        )

        # Warmup tracking (for constrained training)
        self.warmup_best_rewards = {obj: 0.0 for obj in self.optimise_parameters}
        self.observed_value_bounds = {
            obj: {"min": np.inf, "max": -np.inf} for obj in self.optimise_parameters
        }

        # Constrained training state
        self.use_constrained_training = False  # Set by wrapper if needed
        self.update_constraint_bounds = False  # Set by enable_constrained_training
        self.constraint_anchor_mode = "warmup"  # Set by enable_constrained_training
        self.constraint_extend_low = 0.2  # Set by enable_constrained_training
        self.constraint_extend_high = 0.2  # Set by enable_constrained_training
        self.episode_count = 0
        self.is_warmup = True
        self.target_objective = None
        self.constraints = {}
        self.constraint_penalty = 10.0

        # Pareto dominance reward bonus (based on hypervolume improvement)
        self.pareto_dominance_bonus = 0.0  # Bonus weight for hypervolume improvement
        self.use_pareto_bonus = False  # Enable pareto dominance bonus

        # Hard objective bounds penalty (user-defined, applied every step)
        self.enforce_objective_bounds = getattr(data, "enforce_objective_bounds", False)
        self.objective_bounds_penalty_weight = getattr(
            data, "objective_bounds_penalty_weight", 1.0
        )
        self.max_hypervolume = 0.0  # Track maximum hypervolume achieved

        # Environment state
        self.current_state = None
        self.current_index = 0
        self.done = False

        # Multi-objective tracking
        # IMPORTANT: Reward space Pareto front is used for all calculations
        # Value space Pareto front is only for visual diagnostics
        self.pareto_front_rewards = []  # List of (reward_vector, state) - used for calculations
        self.pareto_front_values = []  # List of (value_vector, state) - used for plotting
        self.pareto_front_episodes = []  # List of episode data dicts (for BC loss)
        self.all_points = []
        self.pending_pareto_candidates = []  # Staged per-episode, flushed per-rollout
        # Grid resolution for the archive, in normalised-reward units. See
        # flush_pareto_candidates. 0.0 falls back to storing every distinct point.
        self.pareto_epsilon = float(getattr(data, "pareto_epsilon", 0.0005))

        # Observation space shape
        features_per_layer = 1 + self.n_materials + 2
        n_constraints = (
            len(self.optimise_parameters) if self.use_constrained_training else 0
        )
        self.obs_space_shape = (self.max_layers, features_per_layer, n_constraints)

    def material_index_by_name(self, name: str) -> int:
        """Return the index of the material with the given name (case-insensitive)."""
        for index, props in self.materials.items():
            if str(props.get("name", "")).lower() == name.lower():
                return index
        available = [
            str(props.get("name", index)) for index, props in self.materials.items()
        ]
        raise ValueError(
            f"Material '{name}' not found in materials. Available: {available}"
        )

    def reset(self) -> CoatingState:
        """Reset environment to initial state."""
        self.current_state = CoatingState(
            max_layers=self.max_layers,
            n_materials=self.n_materials,
            air_material_index=self.air_material_index,
            substrate_material_index=self.substrate_material_index,
            materials=self.materials,
        )
        self.current_index = 0
        self.done = False
        return self.current_state

    def step(
        self,
        action: np.ndarray,
        objective_weights: Optional[Dict[str, float]] = None,
        pc_tracker=None,
        phase_info=None,
        **kwargs,
    ) -> Tuple:
        """Take an action in the environment.

        Returns:
            Tuple of (new_state, rewards, terminated, finished, reward, full_action, vals)
            matching base environment interface.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")

        if objective_weights is None:
            objective_weights = self.objective_weights

        # Extract action - base environment format is [material, thickness]
        # but agent outputs [thickness, material_probs...]
        if len(action) > 2:
            # Agent format: [thickness, material_probs...]
            thickness = float(action[0])
            material_idx = int(np.argmax(action[1:]))
        else:
            # Base environment format: [material, thickness]
            material_idx = int(action[0])
            thickness = float(action[1])

        # Clamp thickness
        thickness = np.clip(thickness, self.min_thickness, self.max_thickness)

        # Create full_action in base environment format
        full_action = [material_idx, thickness]

        # Update state
        self.current_state.set_layer(self.current_index, thickness, material_idx)
        self.current_index += 1

        # Check termination conditions
        terminated = False
        finished = False
        if (
            self.current_index >= self.max_layers
            or material_idx == self.air_material_index
        ):
            finished = True
            self.done = True

        # Calculate reward
        if finished or self.use_intermediate_reward:
            total_reward, vals, reward_components = self.compute_training_reward(
                self.current_state,
                objective_weights=objective_weights,
            )

            # Stage pareto candidate; front is flushed once per rollout via flush_pareto_candidates()
            if finished and self.multi_objective:
                episode_data = kwargs.get("episode_data", None)
                self._stage_pareto_candidate(vals, self.current_state, episode_data)
        else:
            total_reward = 0.0
            vals = {}
            reward_components = {}

        # Use reward_components as rewards dict (includes PC metadata)
        rewards = reward_components

        return (
            self.current_state,
            rewards,
            terminated,
            finished,
            total_reward,
            full_action,
            vals,
        )

    # Updates
    def _initialize_reward_bounds(self):
        """Initialize reward bounds from objective_bounds config."""
        for obj in self.optimise_parameters:
            if obj in self.objective_bounds:
                bounds = self.objective_bounds[obj]
                if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                    min_val, max_val = float(bounds[0]), float(bounds[1])
                    # Compute raw rewards at bounds
                    target = self.optimise_targets.get(obj, 0.0)
                    min_reward = -np.log(np.abs(min_val - target) + 1e-30)
                    max_reward = -np.log(np.abs(max_val - target) + 1e-30)
                    self.reward_bounds[obj] = [
                        min(min_reward, max_reward),
                        max(min_reward, max_reward),
                    ]

    def update_observed_bounds(self, vals: dict):
        """Update observed value bounds during training."""
        for obj in self.optimise_parameters:
            if obj in vals and vals[obj] is not None:
                val = float(vals[obj])
                if not np.isnan(val):
                    self.observed_value_bounds[obj]["min"] = min(
                        self.observed_value_bounds[obj]["min"], val
                    )
                    self.observed_value_bounds[obj]["max"] = max(
                        self.observed_value_bounds[obj]["max"], val
                    )

    def update_warmup_best(
        self, objective: str, normalised_reward: float, phase: str = "WARMUP"
    ):
        """Raise the best normalised reward on record for an objective.

        These bests anchor every constraint threshold afterwards
        (``constraints[obj] = frac * best``), so if they stay frozen at what
        warmup managed, a policy that later exceeds them is no longer being
        constrained by anything demanding.
        """
        old_best = self.warmup_best_rewards[objective]
        if normalised_reward > old_best:
            print(
                f"    {phase}: New best {objective}={normalised_reward:.4f} (was {old_best:.4f})"
            )
        self.warmup_best_rewards[objective] = max(old_best, normalised_reward)

    def constraint_anchor(self, objective: str) -> float:
        """Scale that this objective's constraint thresholds are a fraction of.

        With "warmup" the scale is whatever that run's warmup happened to
        reach. That is a max over a noisy process, so it differs between runs
        and two runs of the same config end up solving genuinely different
        constrained problems - a large part of why repeat runs disagree.

        With "absolute" the scale starts at 1.0, which is what the normalised
        reward equals at the best end of objective_bounds, so every run sweeps
        the same thresholds and a given level means the same thing everywhere.
        It still rises if a run beats that, since a threshold below what the
        policy can already reach has stopped constraining anything.
        """
        best = self.warmup_best_rewards.get(objective, 0.0)
        if self.constraint_anchor_mode == "absolute":
            return max(1.0, best)
        return best

    def constraint_range(self, objective: str) -> Tuple[float, float]:
        """Reward range to draw this objective's constraint threshold from.

        objective_bounds only fixes an arbitrary origin and unit for the
        reward, so a fixed [0, anchor] interval leaves it to that guess how
        many thresholds land where designs actually are. Taking the range
        from the front's own spread instead makes it invariant to the guess,
        since the spread carries the same units. Both ends are widened by a
        fraction of that spread; the upper one is what asks for thresholds no
        design has met yet, and the lower one keeps the opposite corner
        reachable so the front can still extend downwards.
        """
        idx = self.optimise_parameters.index(objective)
        rewards = [r[idx] for r, _ in self.pareto_front_rewards]
        if len(rewards) < MIN_FRONT_FOR_CONSTRAINT_RANGE:
            return 0.0, self.constraint_anchor(objective)

        lo, hi = min(rewards), max(rewards)
        width = hi - lo
        return (
            lo - self.constraint_extend_low * width,
            hi + self.constraint_extend_high * width,
        )

    def enable_constrained_training(
        self,
        warmup_episodes_per_objective: int = 200,
        steps_per_objective: int = 10,
        episodes_per_step: int = 200,
        constraint_penalty: float = 10.0,
        update_constraint_bounds: bool = False,
        constraint_anchor_mode: str = "warmup",
        constraint_extend_low: float = 0.2,
        constraint_extend_high: float = 0.2,
    ):
        """Enable two-phase constrained training.

        Phase 1 (Warmup): Optimize each objective individually
        Phase 2 (Constrained): Cycle through objectives with constraints

        update_constraint_bounds keeps the per-objective bests rising during
        phase 2 rather than freezing them at warmup, so the thresholds scale
        with what the policy can actually reach.
        """
        if constraint_anchor_mode not in ("warmup", "absolute"):
            raise ValueError(
                f"constraint_anchor must be 'warmup' or 'absolute', "
                f"got {constraint_anchor_mode!r}"
            )
        self.use_constrained_training = True
        self.update_constraint_bounds = update_constraint_bounds
        self.constraint_anchor_mode = constraint_anchor_mode
        self.constraint_extend_low = constraint_extend_low
        self.constraint_extend_high = constraint_extend_high
        self.warmup_episodes_per_objective = warmup_episodes_per_objective
        self.total_warmup_episodes = warmup_episodes_per_objective * len(
            self.optimise_parameters
        )
        self.steps_per_objective = steps_per_objective
        self.episodes_per_step = episodes_per_step
        self.constraint_penalty = constraint_penalty
        self.total_levels = steps_per_objective
        self.total_phases = self.total_levels * len(self.optimise_parameters)

    def enable_pareto_bonus(self, bonus: float = 1.0):
        """Enable pareto dominance bonus reward based on hypervolume improvement.

        Args:
            bonus: Weight for hypervolume improvement bonus.
                   Since hypervolume improvements are typically small (0-0.1),
                   you may need larger values than the old dominated-count method.
                   Suggested range: 1.0-100.0 depending on desired bonus magnitude.
        """
        self.use_pareto_bonus = True
        self.pareto_dominance_bonus = bonus

    def _stage_pareto_candidate(
        self,
        objectives: Dict[str, float],
        state: CoatingState,
        episode_data: Dict = None,
    ):
        """Stage a candidate for the Pareto front. Call flush_pareto_candidates() to commit."""
        if not self.multi_objective:
            return

        val_vector = np.array(
            [objectives.get(param, 0.0) for param in self.optimise_parameters]
        )
        normalised = self.use_constrained_training and self.use_reward_normalisation
        reward_dict = self.compute_objective_rewards(objectives, normalised=normalised)
        reward_vector = np.array(
            [reward_dict.get(param, 0.0) for param in self.optimise_parameters]
        )
        self.pending_pareto_candidates.append(
            (reward_vector, val_vector, state.copy(), episode_data)
        )

    def _exact_survivors(self, cand_rewards, front_rewards):
        """Merge candidates keeping every distinct non-dominated point.

        The original behaviour, used when ``pareto_epsilon`` is 0.

        Returns:
            (survivors, keep_front) - indices of candidates to add, and a mask
            of incumbents to retain.
        """
        # Duplicates, at the same 6 dp resolution as before; incumbents win
        seen = {tuple(row) for row in np.round(front_rewards, 6)}
        keep = np.zeros(len(cand_rewards), dtype=bool)
        for i, key in enumerate(map(tuple, np.round(cand_rewards, 6))):
            if key not in seen:
                seen.add(key)
                keep[i] = True

        # Candidates an incumbent already dominates
        if len(front_rewards) and keep.any():
            keep &= ~_dominated_by(cand_rewards, front_rewards)

        # Candidates dominated by a better candidate in the same batch
        idx = np.flatnonzero(keep)
        if len(idx) > 1:
            keep[idx] = ~_dominated_by(cand_rewards[idx], cand_rewards[idx])

        survivors = np.flatnonzero(keep)
        keep_front = np.ones(len(front_rewards), dtype=bool)
        if len(front_rewards) and len(survivors):
            # Incumbents the survivors push off the front
            keep_front = ~_dominated_by(front_rewards, cand_rewards[survivors])
        return survivors, keep_front

    def _eps_box_survivors(self, cand_rewards, front_rewards):
        """Merge candidates onto an epsilon-box grid in reward space.

        Rewards are binned onto a grid of side ``pareto_epsilon`` and dominance
        is decided between boxes rather than between points, with one
        representative kept per box. This is the standard epsilon-dominance
        archive (Laumanns et al. 2002): the stored front is bounded by the
        number of boxes it can occupy instead of growing with every episode,
        at a cost of at most ``pareto_epsilon`` per objective in front quality.

        Returns:
            (survivors, keep_front) - indices of candidates to add, and a mask
            of incumbents to retain.
        """
        eps = self.pareto_epsilon
        cand_box = np.floor(cand_rewards / eps)
        front_box = (
            np.floor(front_rewards / eps)
            if len(front_rewards)
            else np.empty((0, cand_rewards.shape[1]), dtype=float)
        )
        keep_front = np.ones(len(front_box), dtype=bool)

        # An archive built before this setting was enabled, or reloaded from an
        # older checkpoint, can hold several points per box; collapse those to
        # their best member so the one-per-box invariant holds from here on.
        owner = {}  # box key -> (score, index, True if the index is an incumbent)
        for j, key in enumerate(map(tuple, front_box)):
            score = float(front_rewards[j].sum())
            held = owner.get(key)
            if held is None:
                owner[key] = (score, j, True)
            elif score > held[0]:
                keep_front[held[1]] = False
                owner[key] = (score, j, True)
            else:
                keep_front[j] = False

        # Candidates whose box a surviving incumbent's box already dominates.
        # Filtering before claiming boxes matters: a candidate that lost here
        # must not have displaced an incumbent on its way out.
        keep = ~_dominated_by(cand_box, front_box[keep_front])

        # Candidates dominated by a better candidate in the same batch
        idx = np.flatnonzero(keep)
        if len(idx) > 1:
            keep[idx] = ~_dominated_by(cand_box[idx], cand_box[idx])

        # One survivor per box, strongest first, so each box ends up held by the
        # best point that landed in it. Only an incumbent is ever displaced:
        # candidates arrive in descending score order, so a later one cannot
        # outscore an earlier candidate already holding the box.
        cand_scores = cand_rewards.sum(1)
        idx = np.flatnonzero(keep)
        for i in idx[np.argsort(-cand_scores[idx])]:
            key = tuple(cand_box[i])
            held = owner.get(key)
            if held is None:
                owner[key] = (float(cand_scores[i]), i, False)
            elif held[2] and cand_scores[i] > held[0]:
                keep_front[held[1]] = False
                owner[key] = (float(cand_scores[i]), i, False)
            else:
                keep[i] = False

        survivors = np.flatnonzero(keep)
        if len(front_box) and len(survivors):
            # Incumbents the survivors' boxes push off the front
            keep_front &= ~_dominated_by(front_box, cand_box[survivors])
        return survivors, keep_front

    def flush_pareto_candidates(self):
        """Merge all staged candidates into the Pareto front.

        Call once per rollout, before the policy update.

        The stored front is already non-dominated, so a new batch only has to
        be compared against it and against itself: O(front x candidates)
        instead of re-sorting the whole pool every rollout. Measured on a
        3-objective front, that takes a flush from 23 ms to 1.5 ms at 8k
        archived points, and the gap widens as the archive grows.

        With ``pareto_epsilon`` above 0 the merge runs on an epsilon-box grid
        (see _eps_box_survivors), which bounds the archive instead of letting
        thickness resolution decide its size. On a 20-layer 3-objective run,
        dropping min_thickness from 0.1 to 0.01 took the archive from 3k
        points to 30k; of those an eps of 0.0005 keeps ~1.4k for 0.05% of the
        hypervolume, and 0.005 keeps ~200 for 0.6%. Hypervolume over 30k
        points took 285 s, over 1.4k it takes under a second. 0.0 restores the
        old keep-everything behaviour.
        """
        if not self.pending_pareto_candidates:
            return

        candidates = self.pending_pareto_candidates
        cand_rewards = np.array([c[0] for c in candidates], dtype=float)
        front_rewards = (
            np.array([r for r, _ in self.pareto_front_rewards], dtype=float)
            if self.pareto_front_rewards
            else np.empty((0, cand_rewards.shape[1]), dtype=float)
        )

        if self.pareto_epsilon > 0:
            survivors, keep_front = self._eps_box_survivors(cand_rewards, front_rewards)
        else:
            survivors, keep_front = self._exact_survivors(cand_rewards, front_rewards)

        # keep_front can still drop incumbents with no survivors at all, so only
        # skip the rebuild when nothing at either end changed.
        if len(survivors) == 0 and keep_front.all():
            self.pending_pareto_candidates = []
            return

        self.pareto_front_rewards = [
            p for p, k in zip(self.pareto_front_rewards, keep_front) if k
        ] + [(list(candidates[i][0]), candidates[i][2]) for i in survivors]
        self.pareto_front_values = [
            p for p, k in zip(self.pareto_front_values, keep_front) if k
        ] + [(list(candidates[i][1]), candidates[i][2]) for i in survivors]
        self.pareto_front_episodes = [
            e for e, k in zip(self.pareto_front_episodes, keep_front) if k
        ] + [candidates[i][3] for i in survivors]
        self.pending_pareto_candidates = []

    # Reward computation
    def compute_state_value(
        self, state: CoatingState, return_field_data: bool = False
    ) -> Tuple:
        """
        Compute physics values using coating_utils.merit_function.
        """
        # Get state array - use get_array() to match base_environment behavior
        state_array = state.get_array()

        # Trim out air layers and reverse order (as base_environment does)
        state_trim = state_utils.trim_state(state_array)
        state_trim = state_trim[::-1]

        # Check if state is empty (all air layers)
        if len(state_trim) == 0:
            # Return default values for empty coating (nothing reflects, all transmits)
            if return_field_data:
                return (0.0, None, 0.0, 0.0, 1.0, None)
            else:
                return (0.0, None, 0.0, 0.0, 1.0)

        # Call existing physics code
        result = coating_utils.merit_function(
            np.array(state_trim),
            self.materials,
            light_wavelength=self.light_wavelength,
            frequency=self.frequency,
            wBeam=self.wBeam,
            Temp=self.Temp,
            substrate_index=self.substrate_material_index,
            air_index=self.air_material_index,
            use_optical_thickness=self.use_optical_thickness,
            return_field_data=return_field_data,
            compute_efi=self.compute_efi,
        )

        if return_field_data:
            return (
                result  # (r, thermal, absorption, thickness, transmission, field_data)
            )
        else:
            return result  # (r, thermal, absorption, thickness, transmission)

    def compute_reward(
        self,
        state,  # Can be CoatingState or numpy array
        normalised: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute base rewards for all objectives.

        Args:
            state: CoatingState or numpy array
            normalised: If True, scale rewards to [0, 1]. If False, return raw log-based rewards.

        Returns:
            Tuple of (individual_rewards, vals)
            - individual_rewards: Dict mapping objective names to their rewards
            - vals: Dict of physics values (reflectivity, thermal_noise, etc.)
        """
        # Convert numpy array to CoatingState if needed
        if isinstance(state, np.ndarray):
            state = CoatingState.from_array(
                state,
                self.n_materials,
                self.air_material_index,
                self.substrate_material_index,
                self.materials,
            )

        # Get physics values
        reflectivity, thermal_noise, absorption, total_thickness, transmission = (
            self.compute_state_value(state)
        )

        vals = {
            "reflectivity": reflectivity,
            "thermal_noise": thermal_noise,
            "thickness": total_thickness,
            "absorption": absorption,
            # Transmitted power fraction converted to ppm (absorption is already ppm)
            "transmission": transmission * 1e6,
        }

        # Compute base rewards for all objectives
        individual_rewards = self.compute_objective_rewards(vals, normalised=normalised)

        return individual_rewards, vals

    def compute_objective_rewards(
        self, vals: dict, normalised: bool = True
    ) -> Dict[str, float]:
        """Compute base rewards for all objectives.

        Args:
            vals: Dictionary of objective values (reflectivity, thermal_noise, etc.)
            normalised: If True, scale rewards to [0, 1]. If False, return raw log-based rewards.

        Returns:
            Dictionary mapping objective names to their rewards
        """
        rewards = {}

        for objective in self.optimise_parameters:
            val = vals.get(objective)

            # Physics undefined for this state (e.g. empty coating): worst reward
            if val is None or not np.isfinite(val):
                bounds = self.reward_bounds.get(objective, [-100, 0])
                rewards[objective] = 0.0 if normalised else bounds[0]
                continue

            # Compute raw log-based reward
            target = self.optimise_targets.get(objective, 0.0)
            raw_reward = -np.log(np.abs(val - target) + 1e-30)

            if normalised:
                # Scale to [0, 1] using reward bounds
                bounds = self.reward_bounds.get(objective, [-100, 0])
                min_reward, max_reward = bounds[0], bounds[1]

                if max_reward <= min_reward:
                    rewards[objective] = 0.5
                else:
                    r = (raw_reward - min_reward) / (max_reward - min_reward)
                    if self.reward_normalisation_apply_clipping:
                        r = float(np.clip(r, 0.0, 1.0))
                    rewards[objective] = r
            else:
                rewards[objective] = raw_reward

        return rewards

    def compute_training_reward(
        self,
        state,  # Can be CoatingState or numpy array
        objective_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Compute reward for training (base rewards + modifiers).

        Args:
            state: CoatingState or numpy array
            objective_weights: Weights for each objective (used in standard mode)
            pc_tracker: Optional progress tracker
            pareto_tracker: Optional pareto front tracker
            phase_info: Optional phase information

        Returns:
            Tuple of (total_reward, vals, individual_rewards)
        """
        # Get base rewards (normalised if constrained training AND normalisation enabled)
        normalised = self.use_constrained_training and self.use_reward_normalisation
        individual_rewards, vals = self.compute_reward(state, normalised=normalised)

        # Update observed bounds (for constrained training)
        if self.use_constrained_training:
            self.update_observed_bounds(vals)

        # Compute total reward based on training mode
        if self.use_constrained_training:
            # Constrained training mode
            if self.is_warmup:
                # Phase 1 (Warmup): Optimize single objective
                total_reward = individual_rewards.get(self.target_objective, 0.0)
                self.update_warmup_best(self.target_objective, total_reward)
            else:
                # Phase 2 (Constrained): Optimize target objective with constraints
                total_reward = individual_rewards.get(self.target_objective, 0.0)

                # Keep the constraint anchors current. Every objective is scored
                # on every episode, so any of them can raise its own best, not
                # just the one being targeted.
                if self.update_constraint_bounds:
                    for obj in self.optimise_parameters:
                        reward = individual_rewards.get(obj)
                        if reward is not None and np.isfinite(reward):
                            self.update_warmup_best(obj, reward, phase="CONSTRAINED")

                # Add constraint penalty modifier
                penalty = self._compute_constraint_penalty(vals, individual_rewards)
                total_reward -= penalty
                individual_rewards["constraint_penalty"] = -penalty
        else:
            # Standard mode: weighted sum of base rewards
            if objective_weights is None:
                objective_weights = self.objective_weights

            total_reward = sum(
                individual_rewards.get(param, 0.0) * objective_weights.get(param, 1.0)
                for param in self.optimise_parameters
            )

        # Add pareto dominance bonus modifier (for both modes)
        if self.use_pareto_bonus:
            pareto_bonus = self._compute_pareto_dominance_bonus(vals)
            total_reward += pareto_bonus
            individual_rewards["pareto_bonus"] = pareto_bonus

        # Apply hard objective bounds penalty (all modes, all phases)
        if self.enforce_objective_bounds:
            bounds_penalty = self._compute_bounds_penalty(vals)
            total_reward -= bounds_penalty
            individual_rewards["bounds_penalty"] = -bounds_penalty

        return total_reward, vals, individual_rewards

    # Reward Addons

    def _compute_constraint_penalty(self, vals, rewards: Dict[str, float]) -> float:
        """Compute constraint violation penalty.
        Args:
            base_rewards: Dictionary of normalised rewards for each objective
        Returns:
            Penalty value (positive number to subtract from reward)
        """
        penalty = 0.0

        for obj, threshold in self.constraints.items():
            norm_reward = rewards.get(obj, 0.0)

            if norm_reward < threshold:
                violation = threshold - norm_reward
                penalty += violation * self.constraint_penalty

        return penalty

    def _compute_bounds_penalty(self, vals: dict) -> float:
        """Compute penalty for objective values outside user-defined objective_bounds.

        Violation is expressed as a fraction of the bound range, so the penalty
        is scale-independent across objectives.

        Args:
            vals: Dictionary of raw objective values

        Returns:
            Penalty value (positive number to subtract from reward)
        """
        penalty = 0.0
        for obj, bounds in self.objective_bounds.items():
            if not (isinstance(bounds, (list, tuple)) and len(bounds) >= 2):
                continue
            val = vals.get(obj)
            if val is None:
                continue
            min_val, max_val = float(bounds[0]), float(bounds[1])
            bound_range = max_val - min_val
            if bound_range <= 0:
                continue
            if val < min_val:
                violation = (min_val - val) / bound_range
            elif val > max_val:
                violation = (val - max_val) / bound_range
            else:
                continue
            penalty += violation * self.objective_bounds_penalty_weight
        return penalty

    def _compute_pareto_dominance_bonus(self, vals: dict) -> float:
        """Compute pareto dominance bonus based on hypervolume improvement.

        Args:
            vals: Dictionary of objective values

        Returns:
            Bonus reward (hypervolume improvement * bonus weight)
        """
        if not self.use_pareto_bonus or not self.multi_objective:
            return 0.0

        # Build reward vector for current point (normalised rewards)
        reward_dict = self.compute_objective_rewards(vals, normalised=True)
        current_reward = np.array(
            [reward_dict.get(param, 0.0) for param in self.optimise_parameters]
        )

        # Get current Pareto front reward vectors
        current_front = [np.array(r) for r, _ in self.pareto_front_rewards]

        # Compute current hypervolume
        if len(current_front) == 0:
            hv_old = 0.0
        else:
            ref_point = np.zeros(len(self.optimise_parameters))
            hv_old = compute_hypervolume(current_front, ref_point, maximize=True)

        # Compute what the front would be with the new point
        new_front = update_pareto_front(current_front, current_reward, maximize=True)

        # Compute new hypervolume
        ref_point = np.zeros(len(self.optimise_parameters))
        hv_new = compute_hypervolume(new_front, ref_point, maximize=True)

        # Compute hypervolume improvement
        hv_improvement = max(0.0, hv_new - hv_old)

        # Update max hypervolume tracking
        if hv_new > self.max_hypervolume:
            self.max_hypervolume = hv_new

        # Return bonus proportional to hypervolume improvement
        return hv_improvement * self.pareto_dominance_bonus

    # Utility functions
    def sample_action_space(self) -> np.ndarray:
        """Sample random action."""
        action = np.zeros(self.n_materials + 1)

        if self.use_optical_thickness:
            action[0] = np.random.uniform(0.01, 1.0)
        else:
            action[0] = np.random.uniform(self.min_thickness, self.max_thickness)

        valid_materials = list(range(self.n_materials))
        if self.ignore_air_option:
            valid_materials = [
                m for m in valid_materials if m != self.air_material_index
            ]
        if self.ignore_substrate_option:
            valid_materials = [
                m for m in valid_materials if m != self.substrate_material_index
            ]

        material_idx = np.random.choice(valid_materials)
        action[material_idx + 1] = 1.0
        return action

    def get_observation(
        self, state: Optional[CoatingState] = None, **kwargs
    ) -> np.ndarray:
        """Get observation tensor with constraints included.

        Args:
            state: State to get observation for (defaults to current_state)
            **kwargs: Additional arguments passed to get_observation_tensor

        Returns:
            Observation as numpy array with constraints appended
        """
        if state is None:
            state = self.current_state

        return state.get_observation_tensor(
            constraints=self.constraints,
            objective_names=self.optimise_parameters,
            **kwargs,
        ).numpy()

    def get_state(self) -> CoatingState:
        """Get current state."""
        return self.current_state

    def set_state(self, state: CoatingState):
        """Set current state."""
        self.current_state = state.copy()

    def get_pareto_front(
        self, space: str = "reward"
    ) -> List[Tuple[List[float], CoatingState]]:
        """Get Pareto front.

        Args:
            space: "reward" for reward space (used for calculations), "value" for value space (visual diagnostics)

        Returns:
            List of (vector, state) tuples
        """
        if space == "value":
            return self.pareto_front_values.copy()
        else:
            return self.pareto_front_rewards.copy()

    def export_pareto_dataframes(self):
        """Export Pareto front as standardized DataFrames.

        Returns:
            Tuple of (designs_df, values_df, rewards_df)
        """
        import pandas as pd

        if not self.pareto_front_values:
            # Return empty DataFrames
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        design_data = []
        value_data = []
        reward_data = []

        for (value_vec, state), (reward_vec, _) in zip(
            self.pareto_front_values, self.pareto_front_rewards
        ):
            # Extract design from state
            state_array = (
                state.get_array()
            )  # One-hot encoded: [thickness, mat_0, mat_1, ..., mat_n]
            thicknesses = state_array[:, 0]
            # Decode one-hot encoding: find which column (1+) has value 1.0
            material_indices = np.argmax(state_array[:, 1:], axis=1).astype(int)

            design_row = {}
            for j in range(len(thicknesses)):
                design_row[f"thickness_{j}"] = thicknesses[j]
                design_row[f"material_{j}"] = material_indices[j]

            # Extract objective values and rewards
            value_row = {}
            reward_row = {}
            for i, param_name in enumerate(self.optimise_parameters):
                if i < len(value_vec):
                    value_row[param_name] = value_vec[i]
                if i < len(reward_vec):
                    reward_row[param_name] = reward_vec[i]

            design_data.append(design_row)
            value_data.append(value_row)
            reward_data.append(reward_row)

        designs_df = pd.DataFrame(design_data)
        values_df = pd.DataFrame(value_data)
        rewards_df = pd.DataFrame(reward_data)

        return designs_df, values_df, rewards_df

    def compute_hypervolume(
        self, space: str = "reward", ref_point: List[float] = None
    ) -> float:
        """Compute hypervolume of the Pareto front.

        Args:
            space: "reward" for reward space (used for calculations), "value" for value space
            ref_point: Reference point for hypervolume computation. If None, uses [0, 0, ...] for reward space
                      or worst-case bounds for value space.

        Returns:
            Hypervolume value (float). Returns 0.0 if Pareto front is empty or if pymoo is not available.
        """
        pareto_front = self.get_pareto_front(space=space)
        if not pareto_front:
            return 0.0

        # Extract objective vectors
        objective_vectors = np.array([vec for vec, _ in pareto_front])

        # Set reference point
        if ref_point is None:
            if space == "reward":
                # Reference point should be worse than all points (e.g., [0, 0])
                ref_point = np.zeros(len(self.optimise_parameters))
            else:
                # In value space, use objective bounds
                ref_point = []
                for param in self.optimise_parameters:
                    bounds = self.objective_bounds.get(param, [0.0, 1.0])
                    # Use worst-case value as reference
                    ref_point.append(bounds[0])
                ref_point = np.array(ref_point)

        # Use utility function for hypervolume computation
        if space == "reward":
            # Reward space: all objectives maximized
            return compute_hypervolume(objective_vectors, ref_point, maximize=True)
        else:
            # Value space: mixed objectives (use objective_directions)
            objective_dirs = [
                self.objective_directions.get(param, True)
                for param in self.optimise_parameters
            ]
            return compute_hypervolume_mixed(
                objective_vectors, ref_point, objective_dirs
            )

    def get_parameter_names(self) -> List[str]:
        """Get list of optimization parameter names."""
        return self.optimise_parameters
