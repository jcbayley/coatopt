#!/usr/bin/env python3
"""
Discrete-action MORL algorithms (GPI-PD, Envelope, PCN) with action masking.

Wrapper provides flat discrete actions (material × thickness bins) with consecutive
material and early-air masking. Algorithms use Q-value or logit masking during eval.

Config section: [morl_discrete]
  sub_algorithm            = gpipd          # Algorithm: gpipd, envelope, pcn
  total_timesteps          = 500000
  seed                     = 42
  verbose                  = 1
  eval_freq                = 10000
  n_thickness_bins         = 20
  mask_consecutive_materials = true
  mask_air_until_min_layers = true
  min_layers_before_air    = 4
  consecutive_penalty      = 0.1            # Reward penalty per constraint violation
  print_freq               = 200
  net_arch                 = [256, 256]
  gamma                    = 0.99
  learning_rate            = 3e-4
  batch_size               = 256
  buffer_size              = 100000
  target_net_update_freq   = 200
  tau                      = 1.0
  epsilon_decay_steps      = 50000
  per                      = true           # Prioritized experience replay
  # GPI-PD specific:
  alpha_per                = 0.6            # PER alpha
  min_priority             = 0.01
  use_gpi                  = true
  dyna                     = false          # Model-based rollouts (expensive)
  gpi_pd                   = true
  # Envelope specific:
  per_alpha                = 0.6
  envelope                 = true
  num_sample_w             = 4
  # PCN specific:
  hidden_dim               = 256
"""

import configparser
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

from coatopt.environments.environment import CoatingEnvironment
from coatopt.utils.configs import Config, load_config
from coatopt.utils.plotting import plot_pareto_front
from coatopt.utils.utils import load_materials, save_run_metadata

# ============================================================================
# MO-Gymnasium wrapper — Discrete action space
# ============================================================================


class CoatOptMODiscreteWrapper(gym.Env):
    """MO-Gymnasium wrapper with a flat Discrete action space.

    Action encoding
    ---------------
    flat_action = material_idx * n_thickness_bins + thickness_bin_idx

    The wrapper exposes ``current_mask``, a boolean array of shape
    ``(n_actions,)`` updated on every reset/step.  Algorithm subclasses read
    this to apply Q-value / logit masking during action selection.

    Parameters
    ----------
    config : Config
    materials : dict
    n_thickness_bins : int
        Number of discrete thickness levels (linearly spaced between
        min_thickness and max_thickness).
    mask_consecutive_materials : bool
        Block selecting the same material twice in a row.
    mask_air_until_min_layers : bool
        Block the air/termination material until at least
        ``min_layers_before_air`` layers have been placed.
    min_layers_before_air : int
    air_material_idx : int
    consecutive_penalty : float
        Negative reward added to all objectives when a consecutive-material
        violation occurs (applies even if the mask corrected the action).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Config,
        materials: dict,
        n_thickness_bins: int = 20,
        mask_consecutive_materials: bool = True,
        mask_air_until_min_layers: bool = True,
        min_layers_before_air: int = 4,
        air_material_idx: int = 0,
        consecutive_penalty: float = 0.1,
        print_freq: int = 200,
    ):
        super().__init__()
        self.env = CoatingEnvironment(config, materials)
        self.config = config
        self.n_thickness_bins = n_thickness_bins

        # Masking settings
        self.mask_consecutive_materials = mask_consecutive_materials
        self.mask_air_until_min_layers = mask_air_until_min_layers
        self.min_layers_before_air = min_layers_before_air
        self.air_material_idx = air_material_idx
        self.consecutive_penalty = consecutive_penalty

        # Precompute thickness values
        self.thickness_bins = np.linspace(
            self.env.min_thickness, self.env.max_thickness, n_thickness_bins
        )

        # Objectives
        self.objectives = list(config.data.optimise_parameters)
        self.reward_dim = len(self.objectives)

        # Episode tracking
        self.previous_material_idx = None
        self.current_layer = 0

        # Progress tracking
        self.print_freq = print_freq
        self._episode_count = 0
        self._total_steps = 0
        self._best_rewards = {obj: 0.0 for obj in self.objectives}
        self._recent_rewards = {obj: [] for obj in self.objectives}  # rolling window
        self._window = 50  # episodes for rolling average

        # ----------------------------------------------------------------
        # Spaces
        # ----------------------------------------------------------------
        n_materials = self.env.n_materials
        self.n_actions = n_materials * n_thickness_bins

        # Flat discrete action space
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # Observation: flattened state (all layers, padded) + layer counter
        n_features_per_layer = 1 + n_materials + 2  # thickness + one-hot + n, k
        obs_size = self.env.max_layers * n_features_per_layer + 1  # +1 layer counter
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # MO-Gymnasium required: reward_space
        # compute_objective_rewards returns exp((raw - min) / (max - min)).
        # For raw in [min, max] this maps to [1, e].  Values outside the
        # configured objective_bounds can fall below 1, so use 0 as the safe
        # lower bound.  The upper bound is e (np.e ≈ 2.718).
        self.reward_space = gym.spaces.Box(
            low=np.zeros(self.reward_dim, dtype=np.float32),
            high=np.full(self.reward_dim, np.e, dtype=np.float32),
            dtype=np.float32,
        )

        # Current action mask (updated on reset/step) — read by algorithm subclasses
        self.current_mask = np.ones(self.n_actions, dtype=bool)

        # Spec stub for morl-baselines compatibility
        self.spec = type(
            "Spec", (), {"id": "CoatOptDiscrete-v0", "name": "CoatOptDiscrete-v0"}
        )()

    # ----------------------------------------------------------------
    # Action encoding / decoding
    # ----------------------------------------------------------------

    def decode_action(self, flat_action: int) -> tuple:
        """Return (material_idx, thickness_bin_idx)."""
        material_idx = int(flat_action) // self.n_thickness_bins
        thickness_bin = int(flat_action) % self.n_thickness_bins
        return material_idx, thickness_bin

    def encode_action(self, material_idx: int, thickness_bin: int) -> int:
        """Return flat action index."""
        return material_idx * self.n_thickness_bins + thickness_bin

    # ----------------------------------------------------------------
    # Masking helpers
    # ----------------------------------------------------------------

    def _compute_action_mask(self) -> np.ndarray:
        """Compute boolean mask of shape (n_actions,). True = valid."""
        n_materials = self.env.n_materials
        material_valid = np.ones(n_materials, dtype=bool)

        if self.mask_consecutive_materials and self.previous_material_idx is not None:
            material_valid[self.previous_material_idx] = False

        if (
            self.mask_air_until_min_layers
            and self.current_layer < self.min_layers_before_air
        ):
            material_valid[self.air_material_idx] = False

        # Safety: at least one material must remain valid
        if not material_valid.any():
            material_valid[:] = True

        # Broadcast: each material controls n_thickness_bins consecutive actions
        mask = np.repeat(material_valid, self.n_thickness_bins)
        return mask

    def _correct_action(self, flat_action: int) -> int:
        """Remap a masked action to the nearest valid action index."""
        if self.current_mask[flat_action]:
            return flat_action
        valid_indices = np.where(self.current_mask)[0]
        if len(valid_indices) == 0:
            return flat_action
        return int(valid_indices[np.argmin(np.abs(valid_indices - flat_action))])

    # ----------------------------------------------------------------
    # Observation
    # ----------------------------------------------------------------

    def _get_obs(self, state) -> np.ndarray:
        obs = (
            state.get_observation_tensor(pre_type="lstm")
            .numpy()
            .flatten()
            .astype(np.float32)
        )
        obs = np.append(obs, self.current_layer / self.env.max_layers)
        return obs

    # ----------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        state = self.env.reset()
        self.previous_material_idx = None
        self.current_layer = 0
        self.current_mask = self._compute_action_mask()

        return self._get_obs(state), {}

    def step(self, flat_action: int):
        flat_action = int(flat_action)

        # Step-level correction (primary safeguard)
        violated_mask = not self.current_mask[flat_action]
        flat_action = self._correct_action(flat_action)

        material_idx, thickness_bin = self.decode_action(flat_action)
        thickness = self.thickness_bins[thickness_bin]

        # Update tracking before physics step
        self.previous_material_idx = material_idx
        self.current_layer += 1

        # Build CoatOpt action format: [thickness, *material_one_hot]
        coatopt_action = np.zeros(self.env.n_materials + 1, dtype=np.float32)
        coatopt_action[0] = thickness
        coatopt_action[1 + material_idx] = 1.0

        state, rewards, terminated, finished, env_reward, _, vals = self.env.step(
            coatopt_action
        )

        # Refresh mask for the next step
        self.current_mask = self._compute_action_mask()

        self._total_steps += 1
        obs = self._get_obs(state)
        done = finished
        truncated = False
        info = {}

        if done:
            normalised = self.env.compute_objective_rewards(vals, normalised=True)
            vec_reward = np.array(
                [normalised.get(obj, 0.0) for obj in self.objectives],
                dtype=np.float32,
            )
            if violated_mask:
                vec_reward -= self.consecutive_penalty
            info = {"rewards": rewards, "vals": vals, "state_array": state.get_array()}

            # Progress tracking
            self._episode_count += 1
            for i, obj in enumerate(self.objectives):
                r = float(vec_reward[i])
                if r > self._best_rewards[obj]:
                    self._best_rewards[obj] = r
                buf = self._recent_rewards[obj]
                buf.append(r)
                if len(buf) > self._window:
                    buf.pop(0)

            if self.print_freq > 0 and self._episode_count % self.print_freq == 0:
                self._print_progress()
        else:
            vec_reward = np.zeros(self.reward_dim, dtype=np.float32)
            if violated_mask:
                vec_reward -= self.consecutive_penalty

        return obs, vec_reward, done, truncated, info

    def _print_progress(self):
        """Print a one-line episode progress summary."""
        pareto_size = len(getattr(self.env, "pareto_front_values", []))
        parts = [
            f"Ep {self._episode_count:>6d} | Step {self._total_steps:>8d} | Pareto={pareto_size:>4d}"
        ]
        for obj in self.objectives:
            avg = (
                np.mean(self._recent_rewards[obj]) if self._recent_rewards[obj] else 0.0
            )
            best = self._best_rewards[obj]
            parts.append(f"{obj[:4]}: avg={avg:.3f} best={best:.3f}")
        print(" | ".join(parts))


# ============================================================================
# Masking utility
# ============================================================================


def _get_env_mask(env) -> np.ndarray:
    """Return current_mask from the env (handles wrapped envs)."""
    if hasattr(env, "current_mask"):
        return env.current_mask
    if hasattr(env, "env") and hasattr(env.env, "current_mask"):
        return env.env.current_mask
    return None


# ============================================================================
# Masked GPIPD
# ============================================================================


def _make_masked_gpipd_class():
    """Return a GPIPD subclass with Q-value masking and progress printing.

    Deferred so the import of morl-baselines only happens when the class is
    actually instantiated (keeps the module importable without morl-baselines).
    """
    from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD

    class MaskedGPIPDImpl(GPIPD):
        """GPIPD with per-iteration progress printing and Q-value masking."""

        def __init__(self, env: CoatOptMODiscreteWrapper, **kwargs):
            super().__init__(env=env, **kwargs)
            self._mask_env = env
            self._iter_count = 0
            self._total_iters = None

        def train(self, total_timesteps: int, **kwargs):
            timesteps_per_iter = kwargs.get("timesteps_per_iter", 10_000)
            self._total_iters = max(1, total_timesteps // timesteps_per_iter)
            self._iter_count = 0
            return super().train(total_timesteps=total_timesteps, **kwargs)

        def train_iteration(self, total_timesteps: int, weight: np.ndarray, **kwargs):
            """Print per-iteration summary, then delegate to parent."""
            self._iter_count += 1
            w_str = "[" + " ".join(f"{x:.2f}" for x in weight) + "]"
            total_str = f"/{self._total_iters}" if self._total_iters else ""
            pareto_size = len(getattr(self._mask_env.env, "pareto_front_values", []))
            eps = getattr(self, "epsilon", float("nan"))
            print(
                f"\n[Iter {self._iter_count:>3d}{total_str}] "
                f"Step {self.global_step:>8,d} | w={w_str} | "
                f"ε={eps:.3f} | Pareto={pareto_size}"
            )
            return super().train_iteration(
                total_timesteps=total_timesteps, weight=weight, **kwargs
            )

        @th.no_grad()
        def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
            """Masked eval: set Q-values of invalid actions to -inf before argmax."""
            mask = _get_env_mask(self._mask_env)
            device = self.device

            obs_t = th.as_tensor(obs).float().to(device)
            w_t = th.as_tensor(w).float().to(device)

            for net in self.q_nets:
                net.eval()

            try:
                psi = th.min(
                    th.stack(
                        [
                            net(obs_t.unsqueeze(0), w_t.unsqueeze(0))
                            for net in self.q_nets
                        ]
                    ),
                    dim=0,
                )[0]  # (1, n_actions, reward_dim)
                q = th.einsum("r,bar->ba", w_t, psi).squeeze(0)  # (n_actions,)

                if mask is not None:
                    mask_t = th.tensor(mask, dtype=th.bool, device=device)
                    q = q.masked_fill(~mask_t, float("-inf"))

                action = int(q.argmax().item())
            except Exception:
                action = super().eval(obs, w)
                if mask is not None and not mask[action]:
                    valid = np.where(mask)[0]
                    if len(valid) > 0:
                        action = int(valid[np.argmin(np.abs(valid - action))])

            for net in self.q_nets:
                net.train()

            return action

    return MaskedGPIPDImpl


class MaskedGPIPD:
    """Factory wrapper: instantiates MaskedGPIPDImpl (a true GPIPD subclass)
    on first use so that morl-baselines is only imported when needed.

    The returned object IS a GPIPD instance, so all internal method dispatch
    (including train() → train_iteration()) works correctly.
    """

    def __new__(cls, env: CoatOptMODiscreteWrapper, **kwargs):
        impl_class = _make_masked_gpipd_class()
        return impl_class(env=env, **kwargs)


# ============================================================================
# Masked Envelope Q-Learning  (composition — Envelope has no train_iteration)
# ============================================================================


class MaskedEnvelope:
    """Envelope Q-Learning with Q-value masking.

    Composition wrapper — Envelope has no ``train_iteration`` hook.
    Envelope uses ``self.q_net`` (single net):
        q_net(obs, w) -> (batch, n_actions, reward_dim)
    Passes ``verbose=True`` to Envelope.train() for built-in episode logging.
    """

    def __init__(self, env: CoatOptMODiscreteWrapper, **kwargs):
        from morl_baselines.multi_policy.envelope.envelope import Envelope

        self._agent = Envelope(env=env, **kwargs)
        self._mask_env = env

    def train(self, total_timesteps: int, **kwargs):
        # Envelope.train() accepts verbose=True for built-in episode printing
        kwargs.setdefault("verbose", True)
        return self._agent.train(total_timesteps=total_timesteps, **kwargs)

    @th.no_grad()
    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        """Masked eval: set Q-values of invalid actions to -inf before argmax."""
        mask = _get_env_mask(self._mask_env)
        device = self._agent.device

        obs_t = th.as_tensor(obs).float().to(device)
        w_t = th.as_tensor(w).float().to(device)

        # Envelope uses self.q_net (singular)
        q_net = self._agent.q_net
        q_net.eval()

        try:
            # q_net(obs, w) -> (batch, n_actions, reward_dim)
            q_vals = q_net(obs_t.unsqueeze(0), w_t.unsqueeze(0))
            q = th.einsum("r,bar->ba", w_t, q_vals).squeeze(0)  # (n_actions,)

            if mask is not None:
                mask_t = th.tensor(mask, dtype=th.bool, device=device)
                q = q.masked_fill(~mask_t, float("-inf"))

            action = int(q.argmax().item())
        except Exception:
            action = self._agent.eval(obs, w)
            if mask is not None and not mask[action]:
                valid = np.where(mask)[0]
                if len(valid) > 0:
                    action = int(valid[np.argmin(np.abs(valid - action))])

        q_net.train()
        return action

    def __getattr__(self, name):
        return getattr(self._agent, name)


# ============================================================================
# Masked PCN
# ============================================================================


class MaskedPCN:
    """PCN with action logit masking applied in eval().

    PCN's model outputs log-probabilities over actions conditioned on
    (obs, desired_return, desired_horizon).  We set masked action logits
    to -1e9 before argmax.

    Note: PCN's eval() signature is eval(obs, w=None) where w is unused
    (PCN conditions on desired_return set during training).  For compatibility
    with the training loop we expose the same eval(obs, w) signature and
    fall back to the parent's desired_return from the last episode.
    """

    def __init__(self, env: CoatOptMODiscreteWrapper, **kwargs):
        from morl_baselines.multi_policy.pcn.pcn import PCN

        self._agent = PCN(env=env, **kwargs)
        self._mask_env = env

    def train(self, **kwargs):
        return self._agent.train(**kwargs)

    @th.no_grad()
    def eval(self, obs: np.ndarray, w=None) -> int:
        """Masked eval using PCN's internal model."""
        mask = _get_env_mask(self._mask_env)
        device = self._agent.device

        try:
            obs_t = th.as_tensor(obs).float().to(device).unsqueeze(0)

            # PCN stores the current desired_return / horizon used at train time
            desired_return = getattr(self._agent, "desired_return", None)
            desired_horizon = getattr(self._agent, "desired_horizon", None)

            if desired_return is None or desired_horizon is None:
                raise RuntimeError("PCN desired_return/horizon not set")

            dr_t = th.as_tensor(desired_return).float().to(device).unsqueeze(0)
            dh_t = th.as_tensor([desired_horizon]).float().to(device).unsqueeze(0)

            log_probs = self._agent.model(obs_t, dr_t, dh_t).squeeze(0)  # (n_actions,)

            if mask is not None:
                mask_t = th.tensor(mask, dtype=th.bool, device=device)
                log_probs = log_probs.masked_fill(~mask_t, -1e9)

            action = int(log_probs.argmax().item())
        except Exception:
            action = self._agent.eval(obs)
            if mask is not None and not mask[action]:
                valid = np.where(mask)[0]
                if len(valid) > 0:
                    action = int(valid[np.argmin(np.abs(valid - action))])

        return action

    def __getattr__(self, name):
        return getattr(self._agent, name)


# ============================================================================
# Agent factory
# ============================================================================


def _build_agent(
    algorithm: str,
    env: CoatOptMODiscreteWrapper,
    parser: configparser.ConfigParser,
    section: str,
    seed: int,
):
    """Instantiate the requested masked MORL agent with correct parameters."""
    net_arch = eval(parser.get(section, "net_arch", fallback="[256, 256]"))
    gamma = parser.getfloat(section, "gamma", fallback=0.99)
    learning_rate = parser.getfloat(section, "learning_rate", fallback=3e-4)
    batch_size = parser.getint(section, "batch_size", fallback=256)
    buffer_size = parser.getint(section, "buffer_size", fallback=100_000)
    target_net_update_freq = parser.getint(
        section, "target_net_update_freq", fallback=200
    )
    tau = parser.getfloat(section, "tau", fallback=1.0)
    epsilon_decay_steps = parser.getint(section, "epsilon_decay_steps", fallback=50_000)
    per = parser.getboolean(section, "per", fallback=True)

    if algorithm == "gpipd":
        return MaskedGPIPD(
            env=env,
            gamma=gamma,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=epsilon_decay_steps,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_net_update_freq=target_net_update_freq,
            tau=tau,
            net_arch=net_arch,
            per=per,
            alpha_per=parser.getfloat(section, "alpha_per", fallback=0.6),
            min_priority=parser.getfloat(section, "min_priority", fallback=0.01),
            use_gpi=parser.getboolean(section, "use_gpi", fallback=True),
            dyna=parser.getboolean(
                section, "dyna", fallback=False
            ),  # off by default (expensive)
            gpi_pd=parser.getboolean(section, "gpi_pd", fallback=True),
            project_name="coatopt-gpipd",
            experiment_name="coatopt-gpipd",
            log=False,
            seed=seed,
        )

    elif algorithm == "envelope":
        return MaskedEnvelope(
            env=env,
            gamma=gamma,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=epsilon_decay_steps,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_net_update_freq=target_net_update_freq,
            tau=tau,
            net_arch=net_arch,
            per=per,
            per_alpha=parser.getfloat(section, "per_alpha", fallback=0.6),
            envelope=parser.getboolean(section, "envelope", fallback=True),
            num_sample_w=parser.getint(section, "num_sample_w", fallback=4),
            project_name="coatopt-envelope",
            experiment_name="coatopt-envelope",
            log=False,
            seed=seed,
        )

    elif algorithm == "pcn":
        scaling_factor = np.ones(env.reward_dim + 1, dtype=np.float32)
        raw = parser.get(section, "scaling_factor", fallback=None)
        if raw is not None:
            scaling_factor = np.array(eval(raw), dtype=np.float32)

        return MaskedPCN(
            env=env,
            scaling_factor=scaling_factor,
            learning_rate=parser.getfloat(section, "learning_rate", fallback=1e-3),
            batch_size=batch_size,
            hidden_dim=parser.getint(section, "hidden_dim", fallback=256),
            log=False,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: gpipd, envelope, pcn"
        )


# ============================================================================
# Training entry point
# ============================================================================


def train(config_path: str, algorithm: str = "gpipd", save_dir: str = None) -> tuple:
    """Train a masked discrete-action MORL agent.

    Parameters
    ----------
    config_path : str
        Path to the INI config file.  Reads ``[morl_discrete]`` section
        (falls back to ``[morl]`` if absent).
    algorithm : str
        One of ``"gpipd"``, ``"envelope"``, ``"pcn"``.
    save_dir : str, optional
        Override the save directory from the config.

    Returns
    -------
    agent : masked agent wrapper
    pareto_df : pd.DataFrame
        Pareto front in objective space.

    [morl_discrete] config keys
    ---------------------------
    sub_algorithm = gpipd
    total_timesteps = 500000
    seed = 42
    verbose = 1
    eval_freq = 10000
    n_thickness_bins = 20
    mask_consecutive_materials = true
    mask_air_until_min_layers = true
    min_layers_before_air = 4
    consecutive_penalty = 0.1
    net_arch = [256, 256]
    gamma = 0.99
    learning_rate = 3e-4
    batch_size = 256
    buffer_size = 100000
    target_net_update_freq = 200
    tau = 1.0
    per = true
    alpha_per = 0.6          ; (gpipd) PER alpha
    per_alpha = 0.6          ; (envelope) PER alpha
    min_priority = 0.01      ; (gpipd)
    use_gpi = true           ; (gpipd)
    dyna = false             ; (gpipd) model-based rollouts — expensive, off by default
    gpi_pd = true            ; (gpipd) envelope-style target
    envelope = true          ; (envelope)
    num_sample_w = 4         ; (envelope)
    epsilon_decay_steps = 50000
    hidden_dim = 256         ; (pcn)
    # scaling_factor = [1.0, 1.0]  ; (pcn) shape: (reward_dim + 1,)
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)

    section = "morl_discrete" if parser.has_section("morl_discrete") else "morl"

    config = load_config(config_path)
    materials_path = parser.get("general", "materials_path")
    materials = load_materials(str(materials_path))

    seed = parser.getint(section, "seed", fallback=42)
    total_timesteps = parser.getint(section, "total_timesteps", fallback=500_000)
    eval_freq = parser.getint(section, "eval_freq", fallback=10_000)
    n_thickness_bins = parser.getint(section, "n_thickness_bins", fallback=20)
    mask_consecutive = parser.getboolean(
        section, "mask_consecutive_materials", fallback=True
    )
    mask_air = parser.getboolean(section, "mask_air_until_min_layers", fallback=True)
    min_layers_before_air = parser.getint(section, "min_layers_before_air", fallback=4)
    consecutive_penalty = parser.getfloat(section, "consecutive_penalty", fallback=0.1)
    print_freq = parser.getint(section, "print_freq", fallback=200)

    if save_dir is None:
        base = parser.get("general", "save_dir", fallback="./runs")
        save_dir = Path(base) / f"morl_discrete_{algorithm}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    th.manual_seed(seed)

    def make_env(is_eval: bool = False):
        return CoatOptMODiscreteWrapper(
            config=config,
            materials=materials,
            n_thickness_bins=n_thickness_bins,
            mask_consecutive_materials=mask_consecutive,
            mask_air_until_min_layers=mask_air,
            min_layers_before_air=min_layers_before_air,
            consecutive_penalty=consecutive_penalty,
            print_freq=0 if is_eval else print_freq,  # silence the eval env
        )

    env = make_env(is_eval=False)
    eval_env = make_env(is_eval=True)

    print("\nCoatOptMODiscreteWrapper")
    print(f"  Objectives : {env.objectives}")
    print(
        f"  Actions    : {env.n_actions}  "
        f"({env.env.n_materials} materials × {n_thickness_bins} thickness bins)"
    )
    print(f"  Obs shape  : {env.observation_space.shape}")
    print(f"  Reward dim : {env.reward_dim}")
    print(f"\nAlgorithm  : {algorithm.upper()}")
    print(f"Timesteps  : {total_timesteps:,}")

    agent = _build_agent(algorithm, env, parser, section, seed)

    ref_point = np.zeros(env.reward_dim, dtype=np.float32)

    print("\nStarting training...")
    start_time = time.time()

    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=None,
        eval_freq=eval_freq,
    )

    end_time = time.time()
    duration_min = (end_time - start_time) / 60

    # Export Pareto front from the inner CoatingEnvironment, which has been
    # accumulating non-dominated designs throughout training via update_pareto_front().
    # This gives designs + raw physics values + normalised rewards — matching NSGA-II output.
    designs_df, values_df, rewards_df = env.env.export_pareto_dataframes()

    if not rewards_df.empty:
        print(f"\nPareto front: {len(rewards_df)} solutions")
        for obj in env.objectives:
            col = values_df[obj]
            print(
                f"  {obj}: min={col.min():.4g}  max={col.max():.4g}  mean={col.mean():.4g}"
            )
        try:
            plot_path = plot_pareto_front(
                df=values_df,
                objectives=env.objectives,
                save_dir=save_dir,
                plot_type="vals",
                algorithm_name=algorithm,
            )
            print(f"Saved plot → {plot_path}")
        except Exception as exc:
            print(f"Warning: could not save plot: {exc}")
    else:
        print("\nWarning: no Pareto solutions found.")

    print(
        f"\nDuration : {duration_min:.1f} min | Pareto front size : {len(rewards_df)}"
    )

    save_run_metadata(
        save_dir=save_dir,
        algorithm_name=f"MORL-DISCRETE-{algorithm.upper()}",
        start_time=start_time,
        end_time=end_time,
        pareto_front_size=len(rewards_df),
        total_episodes=None,
        config_path=config_path,
        additional_info={
            "total_timesteps": total_timesteps,
            "algorithm": algorithm,
            "n_thickness_bins": n_thickness_bins,
            "n_actions": env.n_actions,
            "seed": seed,
            "mask_consecutive_materials": mask_consecutive,
            "mask_air_until_min_layers": mask_air,
        },
    )

    return {
        "pareto_designs": designs_df,
        "pareto_values": values_df,
        "pareto_rewards": rewards_df,
        "model": None,
        "metadata": {
            "total_timesteps": total_timesteps,
            "algorithm": algorithm,
            "n_thickness_bins": n_thickness_bins,
            "seed": seed,
        },
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(
        description="Train a discrete-action MORL agent (GPI-PD / Envelope / PCN) on CoatOpt"
    )
    cli.add_argument(
        "--config", type=str, required=True, help="Path to config INI file"
    )
    cli.add_argument(
        "--algorithm",
        type=str,
        default="gpipd",
        choices=["gpipd", "envelope", "pcn"],
        help="MORL algorithm (default: gpipd)",
    )
    cli.add_argument(
        "--save-dir", type=str, default=None, help="Override save directory"
    )

    args = cli.parse_args()
    train(config_path=args.config, algorithm=args.algorithm, save_dir=args.save_dir)
