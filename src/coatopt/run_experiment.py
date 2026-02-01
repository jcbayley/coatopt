#!/usr/bin/env python3
import argparse
import configparser
from pathlib import Path


def run_experiment(config_path: str):
    """Run experiment based on config file.

    Args:
        config_path: Path to INI configuration file
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read config
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Determine algorithm from section names
    algorithm_sections = {'sb3_discrete', 'sb3_discrete_lstm', 'sb3_dqn', 'sb3_simple', 'morl', 'nsga2', 'hppo'}
    algorithm = None
    for section in parser.sections():
        if section.lower() in algorithm_sections:
            algorithm = section.lower()
            break

    if algorithm is None:
        raise ValueError(f"Config must have one of these algorithm sections: {algorithm_sections}")

    print(f"Running algorithm: {algorithm}")

    # Simple dispatch - each training script reads its own config
    if algorithm == 'sb3_discrete':
        from coatopt.algorithms.train_sb3_discrete import train
        train(config_path=str(config_path))

    elif algorithm == 'sb3_discrete_lstm':
        from coatopt.algorithms.train_sb3_discrete_lstm import train
        train(config_path=str(config_path))

    elif algorithm == 'sb3_dqn':
        from coatopt.algorithms.train_sb3_discrete_dqn import train
        train(config_path=str(config_path))

    elif algorithm == 'sb3_simple':
        from coatopt.algorithms.train_sb3_continuous import train
        train(config_path=str(config_path))

    elif algorithm == 'morl':
        from coatopt.algorithms.train_morl_simple import train_morld as train
        train(config_path=str(config_path))

    elif algorithm == 'nsga2':
        from coatopt.algorithms.train_genetic_simple import train_genetic as train
        train(config_path=str(config_path))

    elif algorithm == 'hppo':
        from coatopt.algorithms.train_hppo_simple import train
        train(config_path=str(config_path))

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Must be one of: sb3_discrete, sb3_discrete_lstm, sb3_dqn, sb3_simple, morl, nsga2, hppo")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CoatOpt experiment from config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )

    args = parser.parse_args()
    run_experiment(args.config)
