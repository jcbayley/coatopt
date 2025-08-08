"""
Configuration constants for PC-HPPO-OML algorithm.
Extracted from pc_hppo_oml.py to improve maintainability.
"""

class HPPOConstants:
    """Constants used throughout the HPPO algorithm."""
    
    # Training constants
    EPISODE_PRINT_INTERVAL = 20
    SAVE_INTERVAL = 100
    MAX_EPISODE_STEPS = 100
    WINDOW_SIZE = 20
    MAX_GRAD_NORM = 1.0
    
    # Numerical constants
    EPSILON = 1e-8
    SMALL_PROBABILITY = 1e-10
    THICKNESS_ADJUSTMENT = 1e-4
    
    # Default parameters
    DEFAULT_N_HEADS = 2
    DEFAULT_UPDATE_FREQUENCY = 10
    
    # Plotting constants
    PLOT_NROWS_REWARD = 8
    PLOT_NROWS_VAL = 5
    PLOT_NROWS_LOSS = 3
    PLOT_FIGSIZE_REWARD = (7, 12)
    PLOT_FIGSIZE_VAL = (7, 9)
    
    # Color mapping for materials
    MATERIAL_COLOR_MAP = {
        0: 'gray',    # air
        1: 'blue',    # m1 - substrate
        2: 'green',   # m2
        3: 'red',     # m3
        4: 'black',
        5: 'yellow',
        6: 'orange',
        7: 'purple',
        8: 'cyan',
    }
    
    # File names
    DISCRETE_POLICY_FILE = "discrete_policy.pt"
    CONTINUOUS_POLICY_FILE = "continuous_policy.pt"
    VALUE_FILE = "value.pt"
    TRAINING_METRICS_FILE = "training_metrics.csv"
    BEST_STATES_FILE = "best_states.pkl"
    
    # Directory names
    STATES_DIR = "states"
    NETWORK_WEIGHTS_DIR = "network_weights"
    PARETO_FRONTS_DIR = "pareto_fronts"
    EVALUATION_DIR = "evaluation_outputs"
    
    # Plot file names
    RUNNING_REWARDS_PLOT = "running_rewards.png"
    RUNNING_VALUES_PLOT = "running_values.png"
    RUNNING_LOSSES_PLOT = "running_losses.png"
    RUNNING_MATERIALS_PLOT = "running_mats.png"
    BEST_STATE_PLOT = "best_state.png"
    ALL_POINTS_FILE = "all_points.txt"
    ALL_POINTS_DATA_FILE = "all_points_data.txt"
    PARETO_FRONT_FILE = "pareto_front_latest.txt"