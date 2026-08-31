#!/usr/bin/env python3
"""Pareto Conditioned Networks (PCN) for CoatOpt.

Self-contained, like every other trainer in this directory: it imports the
environment and the config helpers and nothing else from the project, and
nothing outside this file needs to change to use it.

    python -m coatopt.algorithms.train_pcn --config path/to/config.ini

Why PCN for this problem
------------------------
Reference: Reymond, Bargiacchi & Nowe, "Pareto Conditioned Networks",
AAMAS 2022. The implementation is MORL-Baselines' -- this file is a wrapper,
not a reimplementation.

PCN is a *return-conditioned* method: it learns a supervised map
``(state, desired return, desired horizon) -> action``, trained by cloning its
own best episodes. That gives it three properties that fit a coating stack
better than the value-based alternatives, all of which were measured on the
MO-Gymnasium benchmark in ``benchmarks/``:

* **No value function, so no credit assignment.** The coating environment pays
  a single terminal reward on the finished stack (``use_intermediate_reward``
  is False in every recorded experiment). For a bootstrapping method that is
  the hardest case; for PCN the episode return *is* the training label, so a
  sparse terminal reward is its natural input rather than an obstacle.

* **No scalarisation, so no convexity ceiling.** Weighted-sum methods return
  points on the convex hull of the front and nothing else. On the benchmark's
  two provably non-convex environments PCN placed first on minecart
  (0.439 of analytic-optimal hypervolume) and had the highest front coverage of
  any method on deep-sea-treasure-concave (0.88), while weighted-sum MORL/D
  came last on both (0.315 and 0.452). If coating fronts are non-convex, that
  gap is structural and no amount of compute closes it.

* **Deterministic, fixed-length episodes are its easy case** -- the return is a
  clean label for the trajectory that produced it, with no evaluation noise.

The honest caveat: PCN was the *worst* method on the benchmark's long-horizon
stochastic locomotion task (hopper), by an order of magnitude. That is the
opposite regime to this one, but it is worth knowing the failure mode exists.
Its other weakness is a narrow policy front -- it finds good points but
reproduces a thin set on demand, so the archive matters more than the model.

Action space
------------
PCN handles ``Discrete`` or ``Box``, not the hybrid (material, thickness) pair
this problem actually has, so the pair has to be encoded. Two encodings, set by
``action_mode``:

``discrete`` (default)
    ``Discrete(n_materials * n_thickness_bins)``. PCN's discrete head is a
    categorical trained with cross-entropy, which is the right loss for a
    material choice, and exploration is a categorical sample. The cost is
    thickness resolution -- ``n_thickness_bins`` sets it, and 64 bins over the
    admissible range is finer than it sounds.

``continuous``
    ``Box(n_materials + 1)``: a score per material plus a thickness, with the
    material taken as the argmax. Keeps thickness exact, but PCN then trains
    both with MSE and explores only through its desired-return perturbation,
    so material exploration is weak. Offered for comparison.

Config section ``[pcn]``
------------------------
    total_timesteps      = 300000
    action_mode          = discrete        # or "continuous"
    n_thickness_bins     = 64
    lr                   = 1e-3
    gamma                = 1.0             # terminal reward, fixed horizon
    batch_size           = 256
    hidden_dim           = 64
    num_er_episodes      = 50              # random warm-up episodes
    num_step_episodes    = 10              # episodes collected per outer loop
    num_model_updates    = 50              # gradient steps per outer loop
    max_buffer_size      = 200             # episodes kept
    min_layers_before_air = 2
    seed                 = 42
    verbose              = 1              # 2 to see PCN's own per-update log
"""

from __future__ import annotations

import configparser
import contextlib
import io
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import load_config
from coatopt.utils.utils import load_materials_from_parser

SECTION = "pcn"


# ---------------------------------------------------------------------------
# Environment wrapper


class CoatOptPCNEnv(gym.Env):
    """The coating environment as MORL-Baselines expects to find it.

    Two things differ from the other wrappers in this directory. The reward is
    returned as a **vector** over ``optimise_parameters`` rather than a scalar,
    because PCN is a genuine multi-objective method and conditions on the
    return vector; and there is no target objective or constraint threshold,
    because PCN does not scalarise at all.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config,
        materials: dict,
        action_mode: str = "discrete",
        n_thickness_bins: int = 64,
        min_layers_before_air: int = 2,
    ):
        super().__init__()
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(
                f"action_mode must be 'discrete' or 'continuous', got {action_mode!r}"
            )
        self.base_env = CoatingEnvironment(config, materials)
        self.action_mode = action_mode
        self.n_bins = int(n_thickness_bins)
        self.min_layers_before_air = int(min_layers_before_air)

        # PCN neither scalarises nor constrains, so the environment's
        # constrained-training machinery stays off and step() hands back the
        # per-objective rewards untouched.
        self.base_env.use_constrained_training = False
        self.base_env.is_warmup = False

        self.objectives = list(self.base_env.optimise_parameters)
        self.n_objectives = len(self.objectives)
        self.n_materials = self.base_env.n_materials
        self.min_thickness = self.base_env.min_thickness
        self.max_thickness = self.base_env.max_thickness

        if self.action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(self.n_materials * self.n_bins)
        else:
            self.action_space = gym.spaces.Box(
                low=np.concatenate([np.zeros(self.n_materials), [self.min_thickness]]).astype(np.float32),
                high=np.concatenate([np.ones(self.n_materials), [self.max_thickness]]).astype(np.float32),
                dtype=np.float32,
            )

        n_features = 1 + self.n_materials + 2
        obs_size = self.base_env.max_layers * n_features + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        #: MORL-Baselines reads this to size its reward heads.
        self.reward_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_objectives,), dtype=np.float32
        )
        self.reward_dim = self.n_objectives

        self.current_layer = 0
        self.prev_material: Optional[int] = None

    # -- observation --------------------------------------------------------

    def _obs(self, state) -> np.ndarray:
        obs = (
            state.get_observation_tensor(pre_type="lstm")
            .numpy()
            .flatten()
            .astype(np.float32)
        )
        return np.append(obs, self.current_layer / self.base_env.max_layers).astype(
            np.float32
        )

    # -- action decoding ----------------------------------------------------

    def valid_materials(self) -> np.ndarray:
        """Boolean mask: no immediate repeat, and no air before the minimum."""
        mask = np.ones(self.n_materials, dtype=bool)
        if self.prev_material is not None:
            mask[self.prev_material] = False
        if self.current_layer < self.min_layers_before_air:
            mask[self.base_env.air_material_index] = False
        if not mask.any():
            mask[:] = True
        return mask

    def _decode(self, action) -> Tuple[int, float]:
        if self.action_mode == "discrete":
            a = int(action)
            material, bin_idx = divmod(a, self.n_bins)
            # Bin centres, so no thickness sits exactly on a boundary.
            frac = (bin_idx + 0.5) / self.n_bins
            thickness = self.min_thickness + frac * (self.max_thickness - self.min_thickness)
        else:
            a = np.asarray(action, dtype=np.float64).ravel()
            material = int(np.argmax(a[: self.n_materials]))
            thickness = float(
                np.clip(a[self.n_materials], self.min_thickness, self.max_thickness)
            )
        # PCN has no action masking, so an illegal material is remapped rather
        # than rejected: the nearest legal index, searching upward and wrapping.
        mask = self.valid_materials()
        if not mask[material]:
            order = [(material + k) % self.n_materials for k in range(1, self.n_materials)]
            material = next(m for m in order if mask[m])
        return material, float(thickness)

    # -- gym API ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.base_env.reset()
        self.current_layer = 0
        self.prev_material = None
        return self._obs(state), {}

    def step(self, action):
        material, thickness = self._decode(action)
        self.prev_material = material
        self.current_layer += 1

        coatopt_action = np.zeros(self.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + material] = 1.0

        state, rewards, _, finished, _scalar, _full, vals = self.base_env.step(
            coatopt_action
        )
        # The vector PCN conditions on. Zero on every non-terminal step, which
        # is what the environment already does -- the merit function scores a
        # finished stack.
        reward_vec = np.array(
            [float(rewards.get(o, 0.0)) for o in self.objectives], dtype=np.float32
        )
        info: Dict[str, object] = {}
        if finished:
            info["vals"] = vals
            info["rewards"] = {o: float(rewards.get(o, 0.0)) for o in self.objectives}
        return self._obs(state), reward_vec, bool(finished), False, info


# ---------------------------------------------------------------------------
# Training entry point


def train(config_path: str, save_dir: str = None) -> dict:
    """Train PCN on the coating problem.

    Returns the same dict shape every trainer here returns, so
    ``save_training_results`` consumes it unchanged.
    """
    from morl_baselines.multi_policy.pcn.pcn import PCN

    parser = configparser.ConfigParser()
    parser.read(config_path)

    def _get(key, fallback, cast=str):
        return cast(parser.get(SECTION, key, fallback=str(fallback)))

    materials = load_materials_from_parser(parser, config_path)
    config = load_config(config_path)
    config.data.n_layers = parser.getint(
        "data", "n_layers", fallback=config.data.n_layers
    )

    seed = _get("seed", 42, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_timesteps = _get("total_timesteps", 300_000, int)
    action_mode = _get("action_mode", "discrete")
    n_thickness_bins = _get("n_thickness_bins", 64, int)
    lr = _get("lr", 1e-3, float)
    # Terminal reward on a fixed-length stack: there is nothing to discount
    # between layers, and gamma < 1 would make an identical design worth less
    # for being deeper in the episode.
    gamma = _get("gamma", 1.0, float)
    batch_size = _get("batch_size", 256, int)
    hidden_dim = _get("hidden_dim", 64, int)
    num_er_episodes = _get("num_er_episodes", 50, int)
    num_step_episodes = _get("num_step_episodes", 10, int)
    num_model_updates = _get("num_model_updates", 50, int)
    max_buffer_size = _get("max_buffer_size", 200, int)
    min_layers_before_air = _get("min_layers_before_air", 2, int)
    verbose = _get("verbose", 1, int)

    env = CoatOptPCNEnv(
        config,
        materials,
        action_mode=action_mode,
        n_thickness_bins=n_thickness_bins,
        min_layers_before_air=min_layers_before_air,
    )
    eval_env = CoatOptPCNEnv(
        config,
        materials,
        action_mode=action_mode,
        n_thickness_bins=n_thickness_bins,
        min_layers_before_air=min_layers_before_air,
    )

    # PCN divides the desired return by scaling_factor before feeding it to the
    # network, so it wants the rough magnitude of a return per objective plus
    # one entry for the horizon. Rewards here are already normalised to O(1) by
    # the environment, so the objective entries are 1 and the horizon entry is
    # 1/n_layers.
    scaling = np.concatenate(
        [np.ones(env.n_objectives), [1.0 / max(env.base_env.max_layers, 1)]]
    ).astype(np.float32)
    max_return = np.full(env.n_objectives, _get("max_return", 2.0, float), dtype=np.float32)
    ref_point = np.full(env.n_objectives, _get("ref_point", -1.0, float), dtype=np.float64)

    if save_dir is None:
        base = parser.get("general", "save_dir", fallback="./runs")
        run_name = parser.get("general", "run_name", fallback="")
        date_str = datetime.now().strftime("%Y%m%d")
        suffix = f"-{run_name}" if run_name else ""
        save_dir = Path(base) / f"{date_str}-pcn{suffix}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 60}")
        print("  Pareto Conditioned Networks (MORL-Baselines)")
        print(f"  Objectives      : {env.objectives}")
        print(f"  Action encoding : {action_mode}", end="")
        print(
            f"  -> Discrete({env.n_materials} x {n_thickness_bins})"
            if action_mode == "discrete"
            else f"  -> Box({env.n_materials + 1})"
        )
        print(f"  Layers          : {env.base_env.max_layers}")
        print(f"  Timesteps       : {total_timesteps:,}")
        print(f"  Save dir        : {save_dir}")
        print(f"{'=' * 60}\n", flush=True)

    agent = PCN(
        env,
        scaling_factor=scaling,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        log=False,
        seed=seed,
    )

    # PCN prints a line per update unconditionally; log=False does not silence
    # it. Keep it behind verbose like every other trainer here.
    sink = contextlib.nullcontext() if verbose > 1 else contextlib.redirect_stdout(io.StringIO())
    start = time.time()
    with sink:
        agent.train(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            ref_point=ref_point,
            num_er_episodes=num_er_episodes,
            num_step_episodes=num_step_episodes,
            num_model_updates=num_model_updates,
            max_return=max_return,
            max_buffer_size=max_buffer_size,
        )
    elapsed = time.time() - start

    # The environment stages a Pareto candidate on every finished episode; the
    # front is not readable until those are flushed.
    env.base_env.flush_pareto_candidates()
    designs_df, values_df, rewards_df = env.base_env.export_pareto_dataframes()

    # PCN.save() prepends its own "weights/" to whatever it is given, so it
    # cannot take an absolute path. Write the checkpoint here and hand back
    # None, as the other trainers in this directory do.
    try:
        agent.save(save_dir=str(save_dir) + "/", filename="pcn_model")
    except Exception as exc:  # a failed checkpoint must not lose the front
        if verbose:
            print(f"  warning: could not save PCN weights ({type(exc).__name__}: {exc})")

    if verbose:
        print(f"\n  done in {elapsed:.0f}s — {len(rewards_df)} Pareto solutions", flush=True)

    return {
        "pareto_designs": designs_df,
        "pareto_values": values_df,
        "pareto_rewards": rewards_df,
        "model": None,
        "metadata": {
            "algorithm": "pcn",
            "action_mode": action_mode,
            "n_thickness_bins": n_thickness_bins if action_mode == "discrete" else None,
            "total_timesteps": total_timesteps,
            "gamma": gamma,
            "seed": seed,
            "runtime_s": round(elapsed, 1),
        },
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pareto Conditioned Networks for CoatOpt")
    ap.add_argument("--config", required=True, help="path to config.ini")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    from coatopt.utils.utils import save_training_results

    t0 = time.time()
    results = train(args.config, args.save_dir)
    t1 = time.time()
    out = args.save_dir
    if out is None:
        parser = configparser.ConfigParser()
        parser.read(args.config)
        base = parser.get("general", "save_dir", fallback="./runs")
        run_name = parser.get("general", "run_name", fallback="")
        suffix = f"-{run_name}" if run_name else ""
        out = Path(base) / f"{datetime.now().strftime('%Y%m%d')}-pcn{suffix}"
    save_training_results(results, Path(out), "pcn", t0, t1, args.config)
