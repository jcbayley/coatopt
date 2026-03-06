#!/usr/bin/env python3
"""
Interactive Pareto front and coating design visualization using Plotly.
Creates a single HTML file with two plots side by side.
"""

import argparse
import configparser
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from coatopt.utils.utils import load_pareto_front


def load_materials(materials_path: str) -> Dict:
    """Load material properties from JSON file."""
    with open(materials_path, "r") as f:
        materials = json.load(f)
    return {int(k): v for k, v in materials.items()}


def parse_design(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract thicknesses and materials from a Pareto front row.

    Args:
        row: DataFrame row containing thickness_i and material_i columns

    Returns:
        Tuple of (thicknesses, material_indices) arrays
    """
    # Find all thickness and material columns
    thickness_cols = [col for col in row.index if col.startswith("thickness_")]
    material_cols = [col for col in row.index if col.startswith("material_")]

    # Sort by index number
    thickness_cols = sorted(thickness_cols, key=lambda x: int(x.split("_")[1]))
    material_cols = sorted(material_cols, key=lambda x: int(x.split("_")[1]))

    # Extract values
    thicknesses = np.array([row[col] for col in thickness_cols])
    materials = np.array([int(row[col]) for col in material_cols])

    return thicknesses, materials


def create_coating_trace(
    thicknesses: np.ndarray,
    material_indices: np.ndarray,
    materials: Dict,
    name: str = "Coating",
    visible: bool = True,
) -> List[go.Bar]:
    """Create bar chart traces for coating design visualization.

    Args:
        thicknesses: Array of layer thicknesses
        material_indices: Array of material indices
        materials: Material properties dictionary
        name: Name for the trace
        visible: Whether trace is initially visible

    Returns:
        Single bar trace with all layers stacked
    """
    # Material colors
    color_map = {
        "air": "#F0F0F0",
        "SiO2": "#1f77b4",
        "Ti:Ta2O5": "#ff7f0e",
        "aSi": "#2ca02c",
    }

    # Convert to nanometers
    thicknesses_nm = thicknesses * 1e9

    # Get colors for each layer based on material
    colors = [
        color_map.get(materials[idx]["name"], "#cccccc") for idx in material_indices
    ]

    # Create hover text for each layer
    hover_text = [
        f"Layer {i}<br>{materials[material_indices[i]]['name']}<br>Thickness: {thicknesses_nm[i]:.2f} nm"
        for i in range(len(thicknesses))
    ]

    # Single trace with all layers
    trace = go.Bar(
        name=name,
        x=[0] * len(thicknesses_nm),  # Single column
        y=thicknesses_nm,
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=1,
        showlegend=False,
        visible=visible,
        hovertemplate="%{text}<extra></extra>",
        text=hover_text,
    )

    return [trace]


def create_interactive_plot(
    designs_df: pd.DataFrame,
    values_df: pd.DataFrame,
    materials: Dict,
    max_designs: int = None,
) -> go.Figure:
    """Create interactive Plotly figure with Pareto front and coating designs.

    Args:
        designs_df: DataFrame with design variables (thickness_0, material_0, thickness_1, material_1, ...)
        values_df: DataFrame with objective values (reflectivity, absorption, etc.)
        materials: Material properties dictionary
        max_designs: Maximum number of designs to include (None = all designs)

    Returns:
        Plotly Figure object
    """
    # Combine for easier handling
    combined_df = pd.concat([designs_df, values_df], axis=1)

    # Sort by reflectivity (best first)
    combined_df = combined_df.sort_values("reflectivity", ascending=False).reset_index(
        drop=True
    )

    # Limit to max_designs if specified
    if max_designs is not None:
        combined_df = combined_df.head(max_designs)

    # Create subplots: Pareto front (left) and coating design (right)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Pareto Front", "Coating Design"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
    )

    # Add Pareto front scatter plot (all points) - ALWAYS VISIBLE
    # Add customdata to track design index for click events
    fig.add_trace(
        go.Scatter(
            x=combined_df["absorption"],
            y=1 - combined_df["reflectivity"],  # Loss = 1 - reflectivity
            mode="markers+lines",
            marker=dict(size=10, color="lightblue", line=dict(width=1, color="black")),
            line=dict(color="lightblue", width=2, dash="dash"),
            name="Pareto Front",
            customdata=list(range(len(combined_df))),  # Store design index
            hovertemplate=(
                "Design %{customdata}<br>"
                "Absorption: %{x:.2e} ppm<br>"
                "Loss (1-R): %{y:.2e}<br>"
                "<i>Click to view design</i><extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Add highlighted points for each design (one per design, toggle visibility)
    for idx in range(len(combined_df)):
        fig.add_trace(
            go.Scatter(
                x=[combined_df.iloc[idx]["absorption"]],
                y=[1 - combined_df.iloc[idx]["reflectivity"]],
                mode="markers",
                marker=dict(size=15, color="red", line=dict(width=2, color="darkred")),
                name="Selected",
                showlegend=False,
                visible=(idx == 0),  # Only first one visible initially
            ),
            row=1,
            col=1,
        )

    # Create coating design traces for each Pareto point
    # We'll create all traces but only show the first one initially
    all_coating_traces = []
    for idx in range(len(combined_df)):
        row = combined_df.iloc[idx]
        thicknesses, material_indices = parse_design(row)

        visible = idx == 0  # Only first design visible initially
        traces = create_coating_trace(
            thicknesses, material_indices, materials, f"Design {idx}", visible=visible
        )
        all_coating_traces.extend(traces)

        # Add to figure
        for trace in traces:
            fig.add_trace(trace, row=1, col=2)

    # Create buttons to select which design to show
    buttons = []
    n_designs = len(combined_df)

    for idx in range(n_designs):
        row = combined_df.iloc[idx]
        refl = row["reflectivity"]
        loss = 1 - refl
        absorp = row["absorption"]

        # Build visibility array
        # Trace 0: Pareto front (always visible)
        # Traces 1 to n_designs: Highlighted points (only one visible)
        # Traces n_designs+1 onwards: Coating designs (1 trace per design)
        visible_array = [True]  # Pareto front always visible

        # Highlighted points - only show the one for this design
        for highlight_idx in range(n_designs):
            visible_array.append(highlight_idx == idx)

        # Coating design traces - one per design, show only the selected one
        for design_idx in range(n_designs):
            visible_array.append(design_idx == idx)

        button = dict(
            label=f"Design {idx+1}",
            method="update",
            args=[
                {"visible": visible_array},
                {
                    "title": f"Coating Design {idx+1}<br>R={refl:.9f}, L={loss:.2e}, A={absorp:.2e} ppm"
                },
            ],
        )
        buttons.append(button)

    # Update layout
    fig.update_layout(
        title_text=f"Interactive Pareto Front and Coating Design Viewer<br><sub>Showing all {n_designs} designs - Click any point on the Pareto front to view its design</sub>",
        title_x=0.5,
        title_font_size=18,
        showlegend=True,
        height=600,
        width=1400,
        hovermode="closest",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.15,
                xanchor="left",
                y=1.0,
                yanchor="top",
            )
        ],
    )

    # Update axes
    fig.update_xaxes(
        title_text="Absorption (ppm)",
        type="log",
        row=1,
        col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    fig.update_yaxes(
        title_text="Loss (1 - Reflectivity)",
        type="log",
        row=1,
        col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        exponentformat="power",
    )

    # Coating design axes
    fig.update_xaxes(
        showticklabels=False,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Thickness (nm)",
        row=1,
        col=2,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        exponentformat="power",
    )

    # Set barmode to stack
    fig.update_layout(barmode="stack")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive Pareto front and coating design visualization",
        epilog="""
Example:
  python plot_interactive_pareto.py experiments/outputs/20_layer/genetic
        """,
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing config.ini and pareto_front_values.csv",
    )
    parser.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Maximum number of designs to include (default: all designs)",
    )

    args = parser.parse_args()

    # Convert to Path object, resolving relative to current working directory
    directory = Path(args.directory)
    if not directory.is_absolute():
        # Resolve relative to the actual shell's current directory
        directory = Path(os.getcwd()) / directory

    directory = directory.resolve()

    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return 1

    # Find config.ini
    config_path = directory / "config.ini"
    if not config_path.exists():
        print(f"Error: config.ini not found in {directory}")
        return 1

    # Load config to get materials path
    config = configparser.ConfigParser()
    config.read(config_path)

    try:
        materials_path = config.get("general", "materials_path")
        # Resolve relative to config directory or one level above
        if not Path(materials_path).is_absolute():
            candidate1 = (config_path.parent / materials_path).resolve()
            candidate2 = (config_path.parent.parent / materials_path).resolve()
            candidate3 = (config_path.parent.parent.parent / materials_path).resolve()
            candidate4 = (
                config_path.parent.parent.parent.parent / materials_path
            ).resolve()

            if candidate1.exists():
                materials_path = candidate1
            elif candidate2.exists():
                materials_path = candidate2
            elif candidate3.exists():
                materials_path = candidate3
            elif candidate4.exists():
                materials_path = candidate4
            else:
                print(f"Error: Could not find materials file at:")
                print(f"  {candidate1}")
                print(f"  {candidate2}")
                print(f"  {candidate3}")
                return 1
        else:
            materials_path = Path(materials_path)
    except (configparser.NoSectionError, configparser.NoOptionError):
        print(f"Error: Could not find 'materials_path' in config.ini")
        return 1

    # Output path
    output_path = directory / "pareto_interactive.html"

    print(f"Directory: {directory}")
    print(f"Loading Pareto front...")
    try:
        designs_df, values_df, rewards_df = load_pareto_front(directory)
        print(f"  Found {len(designs_df)} designs")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loading materials from {materials_path}...")
    materials = load_materials(str(materials_path))
    print(f"  Found {len(materials)} materials")

    print(f"Creating interactive visualization...")
    fig = create_interactive_plot(
        designs_df, values_df, materials, max_designs=args.max_designs
    )

    print(f"Saving to {output_path}...")
    # Add click event handler via config
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    fig.write_html(
        str(output_path),
        config=config,
        include_plotlyjs="cdn",
        post_script="""
        var plotDiv = document.getElementById('plotly-div');
        if (!plotDiv) {
            plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        }
        if (plotDiv) {
            plotDiv.on('plotly_click', function(data) {
                if (data.points[0].curveNumber === 0) {
                    var designIdx = data.points[0].customdata;
                    var buttons = document.querySelectorAll('button[data-title^="Design"]');
                    if (buttons[designIdx]) {
                        buttons[designIdx].click();
                    }
                }
            });
        }
        """,
    )

    print(f"\nDone! Open {output_path} in your browser to view the interactive plot.")
    print(f"Click on any point in the Pareto front to view its coating design.")
    print(
        f"You can also use the dropdown menu on the right to select specific designs."
    )

    return 0


if __name__ == "__main__":
    exit(main())
