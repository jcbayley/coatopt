"""Shared configuration classes for CoatOpt experiments."""

from dataclasses import dataclass, field
import configparser
import ast
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration fields that CoatingEnvironment reads from config.data."""

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
    combine: str = "sum"

    # Reward normalization settings
    use_reward_normalisation: bool = True
    reward_normalisation_apply_clipping: bool = True
    # Objective bounds for normalization: [worst_case, best_case]
    objective_bounds: dict = field(
        default_factory=lambda: {
            "reflectivity": [0.0, 0.99999],
            "absorption": [10000, 0],  # Measured in ppm: worst 10000 ppm, best 0 ppm
        }
    )

    # Air penalty (penalize early termination)
    apply_air_penalty: bool = True
    air_penalty_weight: float = 0.5

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
    epochs_per_step: int = 2000  # Episodes per constraint step in Phase 2
    steps_per_objective: int = 10  # Number of constraint levels per objective
    constraint_penalty: float = 10.0  # Penalty weight for constraint violations


@dataclass
class Config:
    """config for CoatingEnvironment (no full TrainingConfig needed)."""

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: str) -> Config:
    """Load Config from INI file.

    Args:
        config_path: Path to INI configuration file

    Returns:
        Config object
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Parse Data section
    data_kwargs = {}
    if parser.has_section('Data'):
        for key, value in parser['Data'].items():
            # Parse boolean values
            if value.lower() in ('true', 'false'):
                data_kwargs[key] = value.lower() == 'true'
            # Parse int values
            elif key in ('n_layers',):
                data_kwargs[key] = int(value)
            # Parse float values
            elif key in ('min_thickness', 'max_thickness', 'air_penalty_weight'):
                data_kwargs[key] = float(value)
            # Parse lists and dicts using ast.literal_eval
            elif key in ('optimise_parameters', 'optimise_targets', 'objective_bounds',
                        'optimise_weight_ranges', 'design_criteria'):
                try:
                    data_kwargs[key] = ast.literal_eval(value)
                except:
                    data_kwargs[key] = value
            else:
                data_kwargs[key] = value

    # Parse Training section
    training_kwargs = {}
    if parser.has_section('Training'):
        for key, value in parser['Training'].items():
            # Parse int values
            if key in ('warmup_episodes_per_objective', 'epochs_per_step', 'steps_per_objective'):
                training_kwargs[key] = int(value)
            # Parse float values
            elif key in ('constraint_penalty',):
                training_kwargs[key] = float(value)
            else:
                training_kwargs[key] = value

    return Config(
        data=DataConfig(**data_kwargs),
        training=TrainingConfig(**training_kwargs)
    )
