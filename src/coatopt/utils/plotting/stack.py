"""
Coating stack visualization utilities.
Originally from coatopt/tools/plotting.py

Contains both detailed stack visualization and simple RL-style plotting.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_stack(data, materials, rewards=None, vals=None, ax=None):
    """
    Plot coating stack visualization showing layer thickness and materials.

    Args:
        data: Array with coating layer data (thickness and material encoding)
        materials: List of material dictionaries with properties
        rewards: Optional reward dictionary for title display
        vals: Optional values dictionary for title display
        ax: Optional matplotlib axis to plot on

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    # Extract the layers and their properties
    L = data.shape[0]
    thickness = data[:, 0]
    colors = []
    nmats = data.shape[1] - 1

    # Define colors for m1, m2, and m3
    color_map = {
        0: "gray",  # air
        1: "blue",  # m1 - substrate
        2: "green",  # m2
        3: "red",  # m3
        4: "black",
        5: "yellow",
        6: "orange",
        7: "purple",
        8: "cyan",
    }

    labels = []
    for row in data:
        row = np.argmax(row[1:])
        if row == 0:
            colors.append(color_map[0])  # m1
            labels.append(f"{materials[0]['name']}")
        else:
            colors.append(color_map[row])  # m2
            labels.append(
                f"{materials[row]['name']} (1/4 wave{1064e-9 /(4*materials[row]['n'])})"
            )

    # Create a bar plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None
    # bars = ax.bar(range(L), thickness, color=colors)
    bars = [ax.bar(x, thickness[x], color=colors[x], label=labels[x]) for x in range(L)]

    # Add labels and title
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Thickness")
    if rewards is not None:
        ax.set_title(
            f'TR: {rewards["total_reward"]}, R: {vals["reflectivity"]:.8f}, T: {vals["thermal_noise"]:.8e}, A: {vals["absorption"]:.8e}'
        )
    ax.set_xticks(range(L), [f"Layer {i + 1}" for i in range(L)])  # X-axis labels

    #  Show thickness values on top of bars
    # for x, bar in enumerate(bars):
    #    yval = thickness[x]
    #    ax.text(bar[0].get_x() + bar[0].get_width() / 2, yval, f'{yval:.1f}', ha='center', va='bottom')

    ax.set_ylim(0, np.max(thickness) * (1.1))  # Set Y-axis limit
    ax.grid(axis="y", linestyle="--")

    unique_labels = dict(zip(labels, colors))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markersize=10,
            markerfacecolor=color,
        )
        for label, color in unique_labels.items()
    ]
    ax.legend(handles=handles, title="Materials")

    return fig, ax


def plot_coating(state, fname=None):
    """
    Simple coating state visualization as horizontal bar chart.
    Originally from coatingstack/RL_code/plotting.py

    Args:
        state: Array representing coating state (thickness + material encoding)
        fname: Optional filename to save plot

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    depth_so_far = 0  # To keep track of where to plot the next bar
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i in range(len(state)):
        material_idx = np.argmax(state[i][1:])
        thickness = state[i][0]
        ax.bar(
            depth_so_far + thickness / 2,
            thickness,
            width=thickness,
            color=colors[material_idx],
        )
        depth_so_far += thickness

    ax.set_xlim([0, depth_so_far * 1.01])
    ax.set_ylabel("Physical Thickness [nm]")
    ax.set_xlabel("Layer Position")
    ax.set_title("Generated Stack")

    if fname is not None:
        fig.savefig(fname)

    return fig, ax
