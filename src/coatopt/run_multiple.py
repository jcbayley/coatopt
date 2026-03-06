#!/usr/bin/env python3
"""
Run an experiment multiple times with different seeds and run names.

Usage:
    python run_multiple.py --config experiments/3mat/config_ppo_seq.ini --num-runs 10
"""

import argparse
import configparser
import shutil
import sys
import tempfile
from pathlib import Path

from coatopt.run import run_experiment


def run_multiple_experiments(config_path: str, num_runs: int, start_seed: int = 42):
    """Run experiment multiple times with different seeds and run names."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read original config
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Get base run name
    base_run_name = parser.get("general", "run_name", fallback="run")

    print(f"Running {num_runs} experiments with base name: {base_run_name}")
    print(f"Starting seed: {start_seed}\n")

    successful_runs = 0
    failed_runs = 0

    for i in range(1, num_runs + 1):
        print("=" * 60)
        print(f"Starting run {i}/{num_runs}")
        print("=" * 60)

        # Create temporary config file
        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
        temp_config_path = temp_config.name

        try:
            # Modify config
            new_run_name = f"{base_run_name}_run{i}"
            new_seed = start_seed + i

            parser.set("general", "run_name", new_run_name)

            # Update seed in algorithm section (check multiple possible sections)
            for section in parser.sections():
                if parser.has_option(section, "seed"):
                    parser.set(section, "seed", str(new_seed))
                    print(f"  Run name: {new_run_name}")
                    print(f"  Seed: {new_seed}")
                    break

            # Write modified config
            with open(temp_config_path, "w") as f:
                parser.write(f)

            # Run experiment
            run_experiment(temp_config_path)

            successful_runs += 1
            print(f"\n✓ Completed run {i}/{num_runs}\n")

        except Exception as e:
            failed_runs += 1
            print(f"\n✗ Run {i}/{num_runs} failed with error: {e}\n", file=sys.stderr)

        finally:
            # Clean up temp file
            Path(temp_config_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print(f"All runs completed!")
    print(f"  Successful: {successful_runs}/{num_runs}")
    print(f"  Failed: {failed_runs}/{num_runs}")
    print("=" * 60)

    return successful_runs == num_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment multiple times with different seeds"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs (default: 10)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="Starting seed value (default: 42)",
    )

    args = parser.parse_args()

    success = run_multiple_experiments(
        args.config,
        args.num_runs,
        args.start_seed,
    )

    sys.exit(0 if success else 1)
