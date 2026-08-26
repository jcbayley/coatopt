"""Shared configuration classes for CoatOpt experiments."""

import ast
import configparser
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration fields that CoatingEnvironment reads from config.data."""

    wavelength: float = 1064e-9  # Target light wavelength (meters)
    wBeam: float = (
        0.062  # Laser beam radius/width w0 (meters), default 0.062 m (6.2 cm) for aLIGO
    )
    beam_radius: float = 0.062  # Alias for wBeam
    frequency: float = 100.0  # Frequency for thermal noise calculation (Hz)
    temperature: float = 293.0  # Temperature in Kelvin (K)
    n_layers: int = 20
    min_thickness: float = 10e-9
    max_thickness: float = 500e-9
    optimise_parameters: list = field(
        default_factory=lambda: ["reflectivity", "absorption"]
    )
    optimise_targets: dict = field(
        default_factory=lambda: {"reflectivity": 0.99999, "absorption": 0.0}
    )
    optimise_weight_ranges: dict = field(default_factory=dict)
    design_criteria: dict = field(default_factory=dict)
    use_optical_thickness: bool = False
    ignore_air_option: bool = False
    ignore_substrate_option: bool = False
    use_intermediate_reward: bool = False
    # True: 500-point EFI field loop, absorption by integration, lossless
    # reflectivity. False: exact R/T/A from one complex-index tmm call
    # (absorption = 1-R-T, reflectivity includes absorption) — faster.
    compute_efi: bool = True
    combine: str = "sum"

    # Reward normalization settings
    use_reward_normalisation: bool = True
    reward_normalisation_apply_clipping: bool = True
    # Hard objective bounds penalty
    enforce_objective_bounds: bool = False
    objective_bounds_penalty_weight: float = 1.0
    # Objective bounds for normalization: [worst_case, best_case]
    objective_bounds: dict = field(
        default_factory=lambda: {
            "reflectivity": [0.0, 0.99999],
            "absorption": [10000, 0],  # Measured in ppm: worst 10000 ppm, best 0 ppm
        }
    )

    # Pareto archive resolution, in normalised-reward units. Candidates landing
    # in the same eps-box count as the same trade-off and only the best is kept,
    # which bounds the archive instead of letting it grow with every episode.
    # 0.0005 costs ~0.05% of the hypervolume; raise it for a smaller front
    # (0.005 costs ~0.6%). 0.0 stores every distinct point.
    pareto_epsilon: float = 0.0005

    # Preference constraints (disabled by default for SB3)
    apply_preference_constraints: bool = False

    # Constraint scheduling for multi-objective training
    constraint_schedule: str = "interleaved"  # "interleaved" or "sequential"


@dataclass
class TrainingConfig:
    """Configuration fields that CoatingEnvironment reads from config.training."""

    cycle_weights: str = "random"

    # Constraint scheduling parameters (for preference-constrained training)
    warmup_episodes_per_objective: int = 2000  # Phase 1: warmup per objective
    episodes_per_step: int = 2000  # Episodes per constraint step in Phase 2
    steps_per_objective: int = 10  # Number of constraint levels per objective
    constraint_penalty: float = 10.0  # Penalty weight for constraint violations


@dataclass
class AlgorithmConfig:
    """Configuration for RL algorithm hyperparameters (e.g., PPO)."""

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 32
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network architecture
    pre_network: str = "mlp"  # "mlp" or "lstm"
    net_arch_pi: list = field(default_factory=lambda: [128, 64, 32])  # Policy network
    net_arch_vf: list = field(default_factory=lambda: [128, 64, 32])  # Value network

    # LSTM parameters (for LSTM policies)
    lstm_hidden_size: int = 128


@dataclass
class GeneralConfig:
    """General configuration settings."""

    disable_mlflow: bool = True  # Default: disable MLflow logging
    mlflow_log_freq: int = 10  # Log to MLflow every N episodes (reduces API calls)


@dataclass
class Config:
    """config for CoatingEnvironment (no full TrainingConfig needed)."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)


def load_config(config_path: str) -> Config:
    """Load Config from INI file.

    Args:
        config_path: Path to INI configuration file

    Returns:
        Config object
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Parse General section
    general_kwargs = {}
    if parser.has_section("general"):
        for key, value in parser["general"].items():
            # Parse boolean values
            if key == "disable_mlflow":
                general_kwargs[key] = value.lower() == "true"
            # Parse int values
            elif key == "mlflow_log_freq":
                general_kwargs[key] = int(value)

    # Parse Data section
    data_kwargs = {}
    if parser.has_section("data"):
        for key, value in parser["data"].items():
            # Parse boolean values
            if value.lower() in ("true", "false"):
                data_kwargs[key] = value.lower() == "true"
            # Parse int values
            elif key in ("n_layers",):
                data_kwargs[key] = int(value)
            # Parse float values
            elif key in (
                "wavelength",
                "wbeam",
                "beam_radius",
                "beam_width",
                "w0",
                "frequency",
                "temperature",
                "temp",
                "min_thickness",
                "max_thickness",
                "objective_bounds_penalty_weight",
                "pareto_epsilon",
            ):
                val = float(value)
                # Convert nm to meters if value > 1e-3 (e.g. 1550 -> 1550e-9)
                if key == "wavelength" and val > 1e-3:
                    val *= 1e-9
                # Convert mm to meters if beam radius > 1.0 (e.g. 62 mm -> 0.062 m)
                if key in ("wbeam", "beam_radius", "beam_width", "w0") and val > 1.0:
                    val *= 1e-3
                data_kwargs[key] = val
            # Parse lists and dicts using ast.literal_eval
            elif key in (
                "optimise_parameters",
                "optimise_targets",
                "objective_bounds",
                "optimise_weight_ranges",
                "design_criteria",
            ):
                try:
                    data_kwargs[key] = ast.literal_eval(value)
                except Exception:
                    data_kwargs[key] = value
            else:
                data_kwargs[key] = value

    # Keys that used to be DataConfig fields. apply_air_penalty and
    # air_penalty_weight were parsed into the config but never read by any
    # environment, so setting them did nothing; dropped here rather than
    # rejected so archived config.ini files still load.
    for retired in ("apply_air_penalty", "air_penalty_weight"):
        data_kwargs.pop(retired, None)

    # Map beam radius aliases (INI keys are lowercased by configparser) onto the
    # wBeam/beam_radius dataclass fields, removing the alias keys so
    # DataConfig(**data_kwargs) only receives real fields.
    beam_val = None
    for alias in ("wbeam", "beam_radius", "beam_width", "w0"):
        if alias in data_kwargs:
            if beam_val is None:
                beam_val = data_kwargs[alias]
            del data_kwargs[alias]
    if beam_val is not None:
        data_kwargs["wBeam"] = beam_val
        data_kwargs["beam_radius"] = beam_val

    # Map temp alias onto temperature
    if "temp" in data_kwargs:
        data_kwargs.setdefault("temperature", data_kwargs.pop("temp"))

    # Fallback: check [general] section if not present in [data]
    if parser.has_section("general"):
        if "wavelength" not in data_kwargs and "wavelength" in parser["general"]:
            try:
                val = float(parser["general"]["wavelength"])
                if val > 1e-3:
                    val *= 1e-9
                data_kwargs["wavelength"] = val
            except ValueError:
                pass

        for b_key in ("wbeam", "beam_radius", "beam_width", "w0"):
            if "wBeam" not in data_kwargs and b_key in parser["general"]:
                try:
                    val = float(parser["general"][b_key])
                    if val > 1.0:
                        val *= 1e-3
                    data_kwargs["wBeam"] = val
                    data_kwargs["beam_radius"] = val
                    break
                except ValueError:
                    pass

    # Parse Training section
    training_kwargs = {}
    if parser.has_section("training"):
        for key, value in parser["training"].items():
            # Parse int values
            if key in (
                "warmup_episodes_per_objective",
                "episodes_per_step",
                "steps_per_objective",
            ):
                training_kwargs[key] = int(value)
            # Parse float values
            elif key in ("constraint_penalty",):
                training_kwargs[key] = float(value)
            else:
                training_kwargs[key] = value

    # Parse Algorithm section (PPO hyperparameters)
    algorithm_kwargs = {}
    if parser.has_section("algorithm"):
        for key, value in parser["algorithm"].items():
            # Parse int values
            if key in ("n_steps", "batch_size", "n_epochs", "lstm_hidden_size"):
                algorithm_kwargs[key] = int(value)
            # Parse float values
            elif key in (
                "learning_rate",
                "gamma",
                "gae_lambda",
                "clip_range",
                "ent_coef",
                "vf_coef",
                "max_grad_norm",
            ):
                algorithm_kwargs[key] = float(value)
            # Parse lists (net_arch)
            elif key in ("net_arch_pi", "net_arch_vf"):
                try:
                    algorithm_kwargs[key] = ast.literal_eval(value)
                except Exception:
                    algorithm_kwargs[key] = value
            # Parse string values (pre_network)
            elif key in ("pre_network",):
                algorithm_kwargs[key] = value.strip('"').strip("'")
            else:
                algorithm_kwargs[key] = value

    return Config(
        data=DataConfig(**data_kwargs),
        training=TrainingConfig(**training_kwargs),
        algorithm=AlgorithmConfig(**algorithm_kwargs),
        general=GeneralConfig(**general_kwargs),
    )
