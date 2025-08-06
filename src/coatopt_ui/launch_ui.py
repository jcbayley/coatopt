#!/usr/bin/env python3
"""
Launch script for the PC-HPPO-OML Training UI.

Simple launcher that ensures proper imports and environment setup.
"""
import sys
import os


def main():
    # Add the current directory to Python path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Import and run the training UI
    try:
        from training_ui import main
    
        print("Launching PC-HPPO-OML Training Monitor...")
        print("GUI Interface for real-time training monitoring")
        print("Features:")
        print("  - Load configuration files")
        print("  - Real-time reward plotting")
        print("  - Pareto front evolution visualization")
        print("  - Training progress monitoring")
        print()
        
        main()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  - tkinter (usually comes with Python)")
        print("  - matplotlib")
        print("  - pandas")
        print("  - numpy")
        print("  - pymoo")
        sys.exit(1)

    except Exception as e:
        print(f"Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()