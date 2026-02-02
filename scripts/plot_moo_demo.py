#!/usr/bin/env python3
"""
Demonstration plot for multi-objective optimization showing feasible points and Pareto front.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Set up nice plotting style for presentations
rc('font', size=14)
rc('axes', titlesize=16)
rc('axes', labelsize=14)
rc('legend', fontsize=12)


def is_pareto_efficient(costs):
    """
    Find the Pareto efficient points (minimization problem).

    Args:
        costs: An (n_points, n_costs) array

    Returns:
        A boolean array of Pareto efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Remove dominated points
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def generate_feasible_points(n_points=200, seed=42):
    """
    Generate random feasible points in objective space.

    Args:
        n_points: Number of points to generate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_points, 2) with objective values
    """
    np.random.seed(seed)

    # Generate points with a trade-off relationship
    # Use a convex Pareto front shape (typical for many problems)
    x = np.random.uniform(0.5, 10, n_points)

    # Create a base trade-off curve: f2 = a/f1 + much larger noise
    # to spread points into the dominated region (upper right)
    base_curve = 20 / x
    noise = np.random.exponential(5, n_points)  # Increased noise scale

    # Add additional uniform noise to spread more to the right
    additional_noise_x = np.random.uniform(0, 3, n_points)
    additional_noise_y = np.random.uniform(0, 8, n_points)

    x = x + additional_noise_x
    y = base_curve + noise + additional_noise_y

    points = np.column_stack([x, y])
    return points


def main():
    # Generate feasible points
    n_points = 80
    points = generate_feasible_points(n_points)

    # Find Pareto front
    pareto_mask = is_pareto_efficient(points)
    pareto_points = points[pareto_mask]
    non_pareto_points = points[~pareto_mask]

    # Sort Pareto points for line plotting
    pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot non-Pareto (feasible) points
    ax.scatter(non_pareto_points[:, 0], non_pareto_points[:, 1],
               c='lightgray', s=50, alpha=0.6, label='Feasible solutions',
               edgecolors='gray', linewidth=0.5, zorder=1)

    # Plot Pareto front points
    ax.scatter(pareto_sorted[:, 0], pareto_sorted[:, 1],
               c='#e74c3c', s=100, alpha=0.9, label='Pareto front',
               edgecolors='darkred', linewidth=1.5, zorder=3)

    # Connect Pareto front with a line
    ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
            'r-', linewidth=2.5, alpha=0.7, zorder=2)

    # Labels and title
    ax.set_xlabel('1 - Reflectivity', fontweight='bold')
    ax.set_ylabel('Absorption', fontweight='bold')
    ax.set_title('Multi-Objective Optimization: Pareto Front',
                 fontweight='bold', pad=20)

    # Add legend
    ax.legend(loc='upper right', framealpha=0.95)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set limits with some padding
    x_range = pareto_sorted[:, 0].max() - pareto_sorted[:, 0].min()
    y_range = pareto_sorted[:, 1].max() - pareto_sorted[:, 1].min()
    ax.set_xlim(pareto_sorted[:, 0].min() - 0.1 * x_range,
                points[:, 0].max() + 0.1 * x_range)
    ax.set_ylim(pareto_sorted[:, 1].min() - 0.1 * y_range,
                points[:, 1].max() + 0.1 * y_range)

    plt.tight_layout()

    # Save the figure
    output_path = 'moo_demonstration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
