import numpy as np
import matplotlib.pyplot as plt

def plot_stack(data, materials, rewards=None, vals=None, ax=None):
    #data = self.current_state

    # Extract the layers and their properties
    L = data.shape[0]
    thickness = data[:, 0]
    colors = []
    nmats = data.shape[1] - 1

    # Define colors for m1, m2, and m3
    color_map = {
        0: 'gray',    # air
        1: 'blue',    # m1 - substrate
        2: 'green',   # m2
        3: 'red',      # m3
        4: 'black',
        5: 'yellow',
        6: 'orange',
        7: 'purple',
        8: 'cyan',
    }

    labels = []
    for row in data:
        row = np.argmax(row[1:])
        if row == 0:
            colors.append(color_map[0])  # m1
            labels.append(f"{materials[0]['name']}")
        else:
            colors.append(color_map[row])  # m2
            labels.append(f"{materials[row]['name']} (1/4 wave{1064e-9 /(4*materials[row]['n'])})")



    # Create a bar plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig=None
    #bars = ax.bar(range(L), thickness, color=colors)
    bars = [ax.bar(x, thickness[x], color=colors[x], label=labels[x]) for x in range(L)]


    # Add labels and title
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Thickness')
    if rewards is not None:
        ax.set_title(f'TR: {rewards["total_reward"]}, R: {vals["reflectivity"]:.8f}, T: {vals["thermal_noise"]:.8e}, A: {vals["absorption"]:.8e}')
    ax.set_xticks(range(L), [f'Layer {i + 1}' for i in range(L)])  # X-axis labels

    #  Show thickness values on top of bars
    #for x, bar in enumerate(bars):
    #    yval = thickness[x]
    #    ax.text(bar[0].get_x() + bar[0].get_width() / 2, yval, f'{yval:.1f}', ha='center', va='bottom')


    ax.set_ylim(0, np.max(thickness)*(1.1) )  # Set Y-axis limit
    ax.grid(axis='y', linestyle='--')

    unique_labels = dict(zip(labels, colors))
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
            for label, color in unique_labels.items()]
    ax.legend(handles=handles, title="Materials")

    return fig, ax  