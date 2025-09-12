#!/usr/bin/env python3
"""
MLflow dashboard launcher for CoatOpt experiments.

This script helps users launch the MLflow UI to view their experiment results.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_mlruns_directory(search_path: str = ".") -> Path:
    """
    Find the mlruns directory in the current or parent directories.
    
    Args:
        search_path: Starting directory to search from
        
    Returns:
        Path to mlruns directory
        
    Raises:
        FileNotFoundError: If mlruns directory is not found
    """
    current_path = Path(search_path).resolve()
    
    # Search current and parent directories
    for path in [current_path] + list(current_path.parents):
        mlruns_path = path / "mlruns"
        if mlruns_path.exists() and mlruns_path.is_dir():
            return mlruns_path
    
    raise FileNotFoundError("No mlruns directory found. Run a training job with MLflow enabled first.")


def launch_mlflow_ui(mlruns_path: Path, host: str = "127.0.0.1", port: int = 5000):
    """
    Launch the MLflow UI.
    
    Args:
        mlruns_path: Path to the mlruns directory
        host: Host to bind the UI to
        port: Port to bind the UI to
    """
    try:
        print(f"Launching MLflow UI...")
        print(f"MLruns directory: {mlruns_path}")
        print(f"UI will be available at: http://{host}:{port}")
        print("\nPress Ctrl+C to stop the MLflow UI")
        
        # Change to the directory containing mlruns
        os.chdir(mlruns_path.parent)
        
        # Launch MLflow UI
        cmd = [
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", f"file://{mlruns_path.absolute()}",
            "--host", host,
            "--port", str(port)
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error launching MLflow UI: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMLflow UI stopped")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Launch MLflow UI for CoatOpt experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch MLflow UI with default settings
  python mlflow_ui.py
  
  # Launch on specific host and port
  python mlflow_ui.py --host 0.0.0.0 --port 8080
  
  # Specify mlruns directory
  python mlflow_ui.py --mlruns-path /path/to/mlruns
        """
    )
    
    parser.add_argument(
        "--mlruns-path", 
        type=str,
        help="Path to mlruns directory (will search automatically if not provided)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host to bind MLflow UI to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Port to bind MLflow UI to (default: 5000)"
    )
    
    args = parser.parse_args()
    
    try:
        # Find or validate mlruns directory
        if args.mlruns_path:
            mlruns_path = Path(args.mlruns_path)
            if not mlruns_path.exists():
                print(f"Error: MLruns directory not found: {mlruns_path}")
                sys.exit(1)
        else:
            mlruns_path = find_mlruns_directory()
        
        # Launch MLflow UI
        launch_mlflow_ui(mlruns_path, args.host, args.port)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo create MLruns data, run a training job with MLflow enabled:")
        print("  coatopt-train -c config.ini")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
